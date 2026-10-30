"""The journal: every piece of a dataset that was streamed or investigated, with its exact
coordinates, written down as it happens.

Opening a Stereo-seq chip and panning around it reads a few hundred windows at a few bin sizes.
A week later the question is always the same: *which* windows, at which level, from which file —
and the answer used to be in the explorer's in-memory telemetry, gone with the process. The
journal is the durable form of that trail. Every `View.load()`, `chunks()` stream and
`overview()`, every `GefFile.read_window()` and `rasterize()`, every `select()`, and the first
opening of a file (with its gene list) append one record:

    {"schema": "geminga.journal/1", "t": 1760000000.1, "op": "load",
     "dataset": {"path": "/data/chip.gef", "kind": "gef", "key": "9f2c…", "fingerprint": {...}},
     "unit": "spots", "window": [1600, 3000, 1400, 2100], "level": {"bin": 50},
     "genes": ["GFAP", "SOX2"], "units": 48213, "bytes": 578556, "seconds": 0.21,
     "via": "explorer", "selection": "", "note": "", "tags": [], "host": "...", "pid": 4242,
     "session": "a1b2c3d4"}

The window is exact and half-open in the dataset's own coordinates (chip units for a GEF, row
numbers for a table, record numbers for a FASTQ), so `replay(entry)` reopens the very same
piece. The dataset key carries the file's size (and mtime, or URL) so a record is never
silently reattached to a different file of the same name.

JSONL on purpose, like the catalog and the selection book: the file reads with `cat`, greps
with `grep` or `rg`, diffs, and copies between machines; nothing is lost if the index beside it
or DuckDB is missing. Three ways to search it, from cheapest to fastest:

* `rg`/`grep` on the file itself — `rg -F GFAP ~/.geminga/journal.jsonl`;
* `Journal.search(text=..., overlaps=..., since=..., dataset=...)` — text plus the structured
  predicates grep cannot express (window overlap, time, dataset, operation);
* `Journal.index()` — a trigram index in the Rust core (the shape of Microsoft's *tgrep*): the
  file is cut into blocks of lines, every block's trigrams go into posting lists, and a query
  touches only the blocks that hold all of its trigrams. Built once (a second per 100 MB),
  extended incrementally as the journal grows, and a rare term — a gene, a file name, a
  selection id — comes back in under a millisecond however long the journal is. A term that
  occurs in every block (`"op"`) degrades to a scan, which is what grep does anyway.

Set `GEMINGA_JOURNAL=/path/to/journal.jsonl` to move it, `GEMINGA_JOURNAL=0` to turn recording
off; `g.use_journal(path_or_None)` does the same from Python. Default `~/.geminga/journal.jsonl`.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Optional, Sequence

import numpy as np

from .catalog import _append_line
from .selections import _abs, _hash, dataset_key

__all__ = ["Journal", "Entry", "JournalEntry", "DatasetTrail", "journal", "use_journal", "active",
           "recording", "journal_recording", "quiet", "record", "key_of_gef", "count_units",
           "JOURNAL_SCHEMA", "OPS"]

#: Bumped only on an incompatible change; additive fields keep the number.
JOURNAL_SCHEMA = "geminga.journal/1"

#: The operations that write a record. Consumers can rely on the names.
OPS = ("open", "load", "chunks", "overview", "read_window", "rasterize", "select", "similar",
       "note")

DEFAULT_PATH = "~/.geminga/journal.jsonl"
_SESSION = uuid.uuid4().hex[:8]
_HOST = socket.gethostname()


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

@dataclass
class Entry:
    """One streamed or investigated piece."""
    t: float
    op: str
    dataset: dict                       # path, kind, key, fingerprint — see selections.dataset_key
    unit: str = ""                      # what the window counts: spots, rows, records, pixels…
    window: Optional[list] = None       # exact, half-open, in the dataset's own coordinates
    level: dict = field(default_factory=dict)   # {"bin": 50} | {"zoom": -1, "stride": 4}
    genes: Optional[list] = None        # gene filter in force, or the gene list on "open"
    units: int = 0                      # rows / points / records produced
    bytes: int = 0                      # resident bytes handed out (or streamed, for chunks)
    seconds: float = 0.0
    via: str = ""                       # explorer | describe | similar | extract | handoff | …
    selection: str = ""                 # selection id when the piece belongs to one
    note: str = ""
    tags: list = field(default_factory=list)
    host: str = ""
    pid: int = 0
    session: str = ""
    schema: str = JOURNAL_SCHEMA
    line: int = -1                      # position in the file; not written, filled on read

    #: Field -> (JSON type, meaning): the contract other tools read.
    SCHEMA = {
        "t":         ("number", "unix time the piece was produced"),
        "op":        ("string", f"one of {OPS}"),
        "dataset":   ("object", "path|url, kind, key (blake2b of path and size), fingerprint"),
        "unit":      ("string", "what the window counts: spots, rows, records, pixels"),
        "window":    ("array of number or null", "exact half-open window in dataset coordinates"),
        "level":     ("object", "{bin} for a GEF, {zoom, stride} for a decimating view"),
        "genes":     ("array of string or null", "gene filter in force; the gene list on open"),
        "units":     ("integer", "rows / points / records produced"),
        "bytes":     ("integer", "resident bytes handed out, or bytes streamed for chunks"),
        "seconds":   ("number", "wall time of the read"),
        "via":       ("string", "what asked: explorer, describe, similar, extract, handoff, …"),
        "selection": ("string", "selection id this piece belongs to, if any"),
        "note":      ("string", "free text"),
        "tags":      ("array of string", "free labels"),
        "host":      ("string", "machine"),
        "pid":       ("integer", "process"),
        "session":   ("string", "one id per process, to group a sitting"),
        "schema":    ("string", JOURNAL_SCHEMA),
    }

    @classmethod
    def from_json(cls, ev: dict, line: int = -1) -> "Entry":
        """Tolerant: unknown fields are ignored, missing ones default."""
        e = cls(**{f: ev[f] for f in cls.__dataclass_fields__ if f in ev and f != "line"})
        e.line = line
        return e

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("line", None)
        return json.dumps(d, default=_json, separators=(",", ":"))

    @property
    def path(self) -> str:
        return str(self.dataset.get("path", ""))

    @property
    def key(self) -> str:
        return str(self.dataset.get("key", ""))

    def overlaps(self, window: Sequence[float]) -> bool:
        """Does this piece's window overlap `window` (same arity, half-open)?"""
        return _overlap(self.window, window)

    def line_text(self) -> str:
        w = "" if self.window is None else " " + ",".join(_fmt(v) for v in self.window)
        lv = ",".join(f"{k}={v}" for k, v in self.level.items())
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.t))
        g = f" {len(self.genes)} genes" if self.genes else ""
        via = f" via {self.via}" if self.via else ""
        return (f"{when} {self.op:<11} {os.path.basename(self.path) or self.path:<28}"
                f"{w:<28} {lv:<10} {self.units:>9,} {self.unit:<7} {_human(self.bytes):>9}"
                f"{g}{via}")


def _fmt(v) -> str:
    return str(int(v)) if float(v).is_integer() else f"{float(v):.3g}"


def _human(n) -> str:
    n = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def _overlap(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> bool:
    if a is None or b is None or len(a) != len(b) or len(a) % 2:
        return False
    for i in range(0, len(a), 2):
        lo0, hi0, lo1, hi1 = float(a[i]), float(a[i + 1]), float(b[i]), float(b[i + 1])
        if hi0 <= lo1 or hi1 <= lo0:
            return False
    return True


# ---------------------------------------------------------------------------
# Context: who is asking, and should it be written at all
# ---------------------------------------------------------------------------

_CTX: contextvars.ContextVar[dict] = contextvars.ContextVar("geminga_journal_ctx", default={})


@contextlib.contextmanager
def recording(**tags):
    """Tag every record written inside the block — `recording(via="explorer")`,
    `recording(selection=sel.id, via="describe")`. Inner blocks add to outer ones."""
    merged = {**_CTX.get(), **{k: v for k, v in tags.items() if v is not None}}
    token = _CTX.set(merged)
    try:
        yield
    finally:
        _CTX.reset(token)


@contextlib.contextmanager
def quiet():
    """Write nothing inside the block — for internal scans whose many reads are one event
    to the analyst (candidate windows of `similar()`, say), which the caller records once."""
    token = _CTX.set({**_CTX.get(), "_quiet": True})
    try:
        yield
    finally:
        _CTX.reset(token)


# ---------------------------------------------------------------------------
# The journal
# ---------------------------------------------------------------------------

@dataclass
class DatasetTrail:
    """What the journal knows about one dataset: the fold of its records."""
    key: str
    path: str
    kind: str
    first: float = 0.0
    last: float = 0.0
    n: int = 0
    bytes: int = 0
    units: int = 0
    ops: dict = field(default_factory=dict)
    levels: list = field(default_factory=list)
    bbox: Optional[list] = None          # union of windows, same arity as the windows
    unit: str = ""
    genes: list = field(default_factory=list)      # gene filters seen (not the open list)
    n_genes: int = 0                     # size of the gene list recorded on open
    sessions: int = 0

    def line_text(self) -> str:
        span = time.strftime("%Y-%m-%d", time.localtime(self.first))
        if time.localtime(self.last)[:3] != time.localtime(self.first)[:3]:
            span += "…" + time.strftime("%Y-%m-%d", time.localtime(self.last))
        bb = "" if not self.bbox else " [" + ",".join(_fmt(v) for v in self.bbox) + "]"
        return (f"{self.key:<16} {os.path.basename(self.path) or self.path:<32} {self.kind:<10}"
                f"{self.n:>6} pieces {_human(self.bytes):>9}  {span}{bb}")


class Journal:
    """Append-only JSONL of streamed pieces, with search over it.

        j = g.journal()                       # the active one (default ~/.geminga/journal.jsonl)
        j.print_table()                       # the last twenty pieces
        j.search("GFAP", overlaps=(1000, 2000, 1000, 2000))
        j.datasets()                          # one trail per dataset: pieces, bytes, bounding box
        v = j.replay(j.tail(1)[0])            # the very same window, reopened
    """

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = os.path.abspath(os.path.expanduser(path))
        self._opened: set = set()            # dataset keys whose "open" record this process saw
        self._opened_scanned = False
        self._index = None

    # ---- writing ----
    def record(self, op: str, dataset, *, window=None, level: Optional[dict] = None,
               unit: str = "", genes=None, units: int = 0, nbytes: int = 0,
               seconds: float = 0.0, **tags) -> Optional[Entry]:
        """Append one record. `dataset` is a view, a reader, a path, or a `dataset_key()` dict.
        Tags from an enclosing `recording()` block are merged in; inside `quiet()` nothing is
        written and None is returned. Never raises: a journal that cannot be written warns
        once and stays silent — the analysis must not stop for its diary."""
        ctx = _CTX.get()
        if ctx.get("_quiet"):
            return None
        ds = dataset if isinstance(dataset, dict) and "key" in dataset else dataset_key(dataset)
        e = Entry(t=time.time(), op=op, dataset=ds, unit=unit,
                  window=None if window is None else [_num(v) for v in window],
                  level=dict(level or {}), genes=None if genes is None else [str(x) for x in genes],
                  units=int(units), bytes=int(nbytes), seconds=float(seconds),
                  via=str(tags.pop("via", ctx.get("via", ""))),
                  selection=str(tags.pop("selection", ctx.get("selection", ""))),
                  note=str(tags.pop("note", ctx.get("note", ""))),
                  tags=list(tags.pop("tags", ctx.get("tags", ()))),
                  host=_HOST, pid=os.getpid(), session=_SESSION)
        for k, v in ctx.items():                    # context extras (frame=12) become tags
            if k not in ("via", "selection", "note", "tags", "_quiet"):
                e.tags.append(f"{k}={v}")
        for k, v in tags.items():                   # anything else goes into tags as k=v
            e.tags.append(f"{k}={v}")
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            _append_line(self.path, e.to_json() + "\n")
        except OSError as err:
            _warn_once(f"journal {self.path} cannot be written ({err}); recording is off for it")
            return None
        if op == "open":
            self._opened.add(e.key)
        return e

    def note(self, dataset, text: str, **tags) -> Optional[Entry]:
        """A free-text record about a dataset — what you concluded, what to check."""
        return self.record("note", dataset, note=text, **tags)

    def opened(self, dataset) -> bool:
        """Has this dataset (by key) an `open` record already — in this process or in the file?"""
        key = dataset["key"] if isinstance(dataset, dict) else dataset_key(dataset)["key"]
        if key in self._opened:
            return True
        if not self._opened_scanned:
            self._opened_scanned = True
            for line in self._grep_lines('"op":"open"'):
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("op") == "open":
                    self._opened.add(str((ev.get("dataset") or {}).get("key", "")))
        return key in self._opened

    # ---- reading ----
    def entries(self, since: Optional[float] = None, until: Optional[float] = None) -> Iterator[Entry]:
        """Every record, oldest first. Torn lines are skipped."""
        if not os.path.exists(self.path):
            return
        with open(self.path, "rb") as fh:
            for i, raw in enumerate(fh):
                e = _parse(raw, i)
                if e is None:
                    continue
                if since is not None and e.t < since:
                    continue
                if until is not None and e.t > until:
                    continue
                yield e

    def tail(self, n: int = 20) -> list:
        """The last `n` records. A journal under 64 MB is read whole (line numbers filled in);
        a larger one is read from its end (line numbers -1)."""
        if not os.path.exists(self.path):
            return []
        size = os.path.getsize(self.path)
        if size <= (64 << 20):
            ents = list(self.entries())
            return ents[-n:]
        out: list = []
        with open(self.path, "rb") as fh:
            pos, buf = size, b""
            while pos > 0 and buf.count(b"\n") <= n + 1:
                step = min(1 << 16, pos)
                pos -= step
                fh.seek(pos)
                buf = fh.read(step) + buf
        lines = buf.split(b"\n")
        if pos > 0:
            lines = lines[1:]                             # the first piece may be a partial line
        for raw in lines:
            e = _parse(raw, -1)
            if e is not None:
                out.append(e)
        return out[-n:]

    def __len__(self) -> int:
        return sum(1 for _ in self.entries())

    def search(self, text: Optional[str] = None, *, dataset=None, op=None, overlaps=None,
               within=None, since: Optional[float] = None, until: Optional[float] = None,
               genes: Optional[Sequence[str]] = None, via: Optional[str] = None,
               selection: Optional[str] = None, regex: bool = False, ignore_case: bool = True,
               limit: Optional[int] = None) -> list:
        """Records matching every given predicate.

        `text` is a substring (or a regex with `regex=True`) matched against the raw line, so a
        gene, a path, a selection id, a tag or a note all hit; it goes through the trigram index
        when one is built, else `rg`, `grep`, or Python. `dataset` is a key, a path, a view or a
        reader; `op` one of OPS or a list; `overlaps` a window the piece's window must overlap;
        `within` a window the piece's window must lie inside; `genes` names that must all be in
        the record's gene list; `since`/`until` unix times.
        """
        ops = None if op is None else ({op} if isinstance(op, str) else set(op))
        key = path = None
        if dataset is not None:
            if isinstance(dataset, dict):
                key = dataset.get("key")
            elif isinstance(dataset, str) and re.fullmatch(r"[0-9a-f]{16}", dataset):
                key = dataset
            elif isinstance(dataset, str):
                path = _abs(dataset)             # a path: any record of that file, whatever its size
            else:
                key = dataset_key(dataset)["key"]
        want = None if genes is None else {str(x) for x in genes}
        if text is not None:
            cands = self._text_candidates(text, regex, ignore_case)
        else:
            cands = self.entries(since, until)
        out = []
        for e in cands:
            if since is not None and e.t < since:
                continue
            if until is not None and e.t > until:
                continue
            if ops is not None and e.op not in ops:
                continue
            if key is not None and e.key != key:
                continue
            if path is not None and e.path != path:
                continue
            if via is not None and e.via != via:
                continue
            if selection is not None and e.selection != selection:
                continue
            if overlaps is not None and not e.overlaps(overlaps):
                continue
            if within is not None and not _inside(e.window, within):
                continue
            if want is not None and not want.issubset(set(e.genes or ())):
                continue
            out.append(e)
            if limit is not None and len(out) >= limit:
                break
        return out

    def grep(self, pattern: str, *, regex: bool = False, ignore_case: bool = True) -> list:
        """Raw matching lines as `(line_number, text)`, through the fastest engine available
        (see `engine`). What `rg -n` would print, from Python."""
        return [(ln, s) for ln, s in self._grep_numbered(pattern, regex, ignore_case)]

    @property
    def engine(self) -> str:
        """Which text-search engine `search()`/`grep()` use: `trigram` when an index is built
        and current, else `rg`, `grep`, or `python`."""
        idx = self._fresh_index()
        if idx is not None:
            return "trigram"
        if shutil.which("rg"):
            return "rg"
        if shutil.which("grep"):
            return "grep"
        return "python"

    # ---- the trigram index ----
    @property
    def index_path(self) -> str:
        return self.path + ".tgx"

    def index(self, rebuild: bool = False, block_lines: int = 64):
        """The trigram index over this journal, built (or extended to the current end of the
        file) and saved beside it as `<journal>.tgx`. Needs the Rust core; returns None when the
        installed core predates it."""
        TrigramIndex = _trigram_class()
        if TrigramIndex is None:
            return None
        if not os.path.exists(self.path):
            return None
        idx = None
        if not rebuild and os.path.exists(self.index_path):
            try:
                idx = TrigramIndex.load(self.index_path)
                if idx.source_path != self.path:
                    idx = None
            except Exception:
                idx = None
        if idx is None:
            idx = TrigramIndex.build(self.path, block_lines)
        else:
            idx.update()
        try:
            idx.save(self.index_path)
        except OSError as err:
            _warn_once(f"trigram index {self.index_path} cannot be saved ({err})")
        self._index = idx
        return idx

    def _fresh_index(self):
        """The loaded index if it covers the whole file as it is now, else None."""
        idx = self._index
        if idx is None and os.path.exists(self.index_path):
            TrigramIndex = _trigram_class()
            if TrigramIndex is None:
                return None
            try:
                idx = TrigramIndex.load(self.index_path)
            except Exception:
                return None
            if idx.source_path != self.path:
                return None
            self._index = idx
        if idx is None:
            return None
        try:
            if idx.indexed_bytes > os.path.getsize(self.path):
                return None                      # the file was rewritten shorter: not the same file
        except OSError:
            return None
        return idx                               # an index that lags is fine: search scans the tail

    # ---- text search engines ----
    def _text_candidates(self, text: str, regex: bool, ignore_case: bool) -> Iterator[Entry]:
        for ln, line in self._grep_numbered(text, regex, ignore_case):
            e = _parse(line.encode("utf-8") if isinstance(line, str) else line, ln)
            if e is not None:
                yield e

    def _grep_numbered(self, pattern: str, regex: bool, ignore_case: bool):
        """(line_number, line) for lines matching `pattern`, through the best engine."""
        if not os.path.exists(self.path):
            return []
        idx = self._fresh_index()
        if idx is not None:
            if not regex:
                return [(ln, s) for ln, s in idx.search(pattern, ignore_case)]
            # a regex: its longest literal run picks the blocks, the regex decides the lines;
            # a regex with no literal of three characters falls through to a scan
            lit = _longest_literal(pattern)
            if len(lit) >= 3:
                rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
                return [(ln, s) for ln, s in idx.search(lit, ignore_case) if rx.search(s)]
        tool = shutil.which("rg") or shutil.which("grep")
        if tool:
            args = [tool, "-n"]
            if not regex:
                args.append("-F")
            if ignore_case:
                args.append("-i")
            if os.path.basename(tool) == "rg":
                args += ["--no-heading", "--color", "never"]
            args += ["--", pattern, self.path]
            try:
                p = subprocess.run(args, capture_output=True, text=True, check=False,
                                   encoding="utf-8", errors="replace")
                if p.returncode in (0, 1):            # 1: no match
                    out = []
                    for row in p.stdout.splitlines():
                        n, _, s = row.partition(":")
                        if n.isdigit():
                            out.append((int(n) - 1, s))
                    return out
            except OSError:
                pass
        # Python
        out = []
        if regex:
            rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
            test = rx.search
        else:
            pat = pattern.lower() if ignore_case else pattern

            def test(s, pat=pat):
                return pat in (s.lower() if ignore_case else s)
        with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
            for i, s in enumerate(fh):
                if test(s):
                    out.append((i, s.rstrip("\n")))
        return out

    def _grep_lines(self, pattern: str) -> list:
        return [s for _, s in self._grep_numbered(pattern, False, False)]

    # ---- folds ----
    def datasets(self) -> dict:
        """One `DatasetTrail` per dataset key: pieces, bytes, levels, the union of windows."""
        out: dict = {}
        sessions: dict = {}
        for e in self.entries():
            d = out.get(e.key)
            if d is None:
                d = out[e.key] = DatasetTrail(e.key, e.path, str(e.dataset.get("kind", "")),
                                             first=e.t, last=e.t)
                sessions[e.key] = set()
            d.last = max(d.last, e.t)
            d.first = min(d.first, e.t)
            d.n += 1
            d.bytes += e.bytes
            d.units += e.units
            d.ops[e.op] = d.ops.get(e.op, 0) + 1
            sessions[e.key].add(e.session)
            lv = tuple(sorted(e.level.items()))
            if lv and lv not in d.levels:
                d.levels.append(lv)
            if e.op == "open":
                d.n_genes = len(e.genes or ())
            elif e.genes:
                for gname in e.genes:
                    if gname not in d.genes and len(d.genes) < 200:
                        d.genes.append(gname)
            if e.window is not None and e.op != "open":
                if d.bbox is None or len(d.bbox) != len(e.window):
                    d.bbox = list(e.window)
                    d.unit = e.unit
                else:
                    for i in range(0, len(e.window), 2):
                        d.bbox[i] = min(d.bbox[i], e.window[i])
                        d.bbox[i + 1] = max(d.bbox[i + 1], e.window[i + 1])
        for k, d in out.items():
            d.sessions = len(sessions[k])
            d.levels = [dict(lv) for lv in d.levels]
        return out

    def visited(self, dataset, *, level: Optional[dict] = None, ops=("load", "chunks",
                "read_window", "rasterize", "overview")) -> list:
        """The windows actually streamed from one dataset, oldest first."""
        out = []
        for e in self.search(dataset=dataset, op=ops):
            if e.window is None:
                continue
            if level is not None and any(e.level.get(k) != v for k, v in level.items()):
                continue
            out.append(e)
        return out

    def coverage(self, dataset, shape: int = 256, *, extent: Optional[Sequence[float]] = None,
                 level: Optional[dict] = None) -> np.ndarray:
        """How many times each part of a 2-D dataset was streamed: a `shape × shape` count
        image over `extent` (default: the union of all windows seen). Row `i` is y, column `j`
        is x, as `GefFile.rasterize` lays it out. Where it is zero, nobody has looked."""
        ents = [e for e in self.visited(dataset, level=level) if e.window and len(e.window) == 4]
        img = np.zeros((shape, shape), np.int32)
        if not ents:
            return img
        if extent is None:
            x0 = min(e.window[0] for e in ents); x1 = max(e.window[1] for e in ents)
            y0 = min(e.window[2] for e in ents); y1 = max(e.window[3] for e in ents)
        else:
            x0, x1, y0, y1 = (float(v) for v in extent)
        sx = shape / max(x1 - x0, 1e-9)
        sy = shape / max(y1 - y0, 1e-9)
        for e in ents:
            a, b, c, d = e.window
            j0 = int(np.clip(np.floor((a - x0) * sx), 0, shape))
            j1 = int(np.clip(np.ceil((b - x0) * sx), 0, shape))
            i0 = int(np.clip(np.floor((c - y0) * sy), 0, shape))
            i1 = int(np.clip(np.ceil((d - y0) * sy), 0, shape))
            if j1 > j0 and i1 > i0:
                img[i0:i1, j0:j1] += 1
        return img

    # ---- going back ----
    def replay(self, entry: Entry, budget=None, *, check: bool = True):
        """Reopen the piece an entry describes: the same file, level and window, as a view
        ready to `load()`. With `check`, a file whose size no longer matches the recorded
        fingerprint raises `ValueError` instead of pretending it is the same data."""
        from . import open_gef, view as _view
        path = entry.path
        kind = str(entry.dataset.get("kind", ""))
        fp = entry.dataset.get("fingerprint") or {}
        if check and "://" not in path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path}: recorded in the journal, not on this machine")
            size = fp.get("size")
            if size is not None and os.path.getsize(path) != int(size):
                raise ValueError(f"{path} is {os.path.getsize(path):,} bytes now, "
                                 f"{int(size):,} when the piece was streamed — not the same file")
        if kind == "gef":
            v = open_gef(path, budget, genes=entry.genes or None,
                         bin_size=entry.level.get("bin"))
        else:
            v = _view(path, budget=budget)
        if entry.window is not None and entry.op != "open":
            v.select(*entry.window)
            if "zoom" in entry.level and hasattr(v, "set_zoom") and kind != "gef":
                try:
                    v.set_zoom(int(entry.level["zoom"]))
                except Exception:
                    pass
        return v

    # ---- optional surfaces ----
    def duckdb(self, con=None):
        """A DuckDB connection with the journal exposed as the view `journal`. Storage stays
        the JSONL: `SELECT op, count(*) FROM journal GROUP BY op`."""
        import duckdb
        con = con or duckdb.connect()
        con.execute(
            f"CREATE OR REPLACE VIEW journal AS SELECT * FROM read_json_auto('{self.path}', "
            f"union_by_name=true, ignore_errors=true)")
        return con

    def to_parquet(self, path: str) -> str:
        import pyarrow as pa
        import pyarrow.parquet as pq
        rows = []
        for e in self.entries():
            r = asdict(e)
            r.pop("line", None)
            r["dataset"] = json.dumps(r["dataset"], default=_json)
            r["level"] = json.dumps(r["level"], default=_json)
            r["window"] = json.dumps(r["window"]) if r["window"] is not None else None
            r["genes"] = ",".join(r["genes"]) if r["genes"] else None
            r["tags"] = ",".join(r["tags"])
            rows.append(r)
        pq.write_table(pa.Table.from_pylist(rows), path)
        return path

    def print_table(self, n: int = 20, file=None) -> None:
        for e in self.tail(n):
            print(e.line_text(), file=file or sys.stdout)

    def __repr__(self) -> str:
        size = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        return f"Journal({self.path}, {_human(size)}, engine={self.engine})"


def _parse(raw: bytes, line: int) -> Optional[Entry]:
    raw = raw.strip()
    if not raw:
        return None
    try:
        ev = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(ev, dict) or "op" not in ev or "dataset" not in ev:
        return None
    try:
        return Entry.from_json(ev, line)
    except TypeError:
        return None


def _inside(w: Optional[Sequence[float]], box: Sequence[float]) -> bool:
    if w is None or len(w) != len(box) or len(w) % 2:
        return False
    for i in range(0, len(w), 2):
        if float(w[i]) < float(box[i]) or float(w[i + 1]) > float(box[i + 1]):
            return False
    return True


def _num(v):
    f = float(v)
    return int(f) if f.is_integer() else f


def _longest_literal(pattern: str) -> str:
    """The longest run of plain characters in a regex: what the trigram index can filter on."""
    runs = re.split(r"\\.|[\[\]\(\)\{\}\*\+\?\|\^\$\.]", pattern)
    return max(runs, key=len) if runs else ""


_WARNED: set = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        import warnings
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


def count_units(data) -> int:
    """How many rows / points / records a loaded piece holds, whatever its container."""
    if data is None:
        return 0
    n = getattr(data, "num_rows", None)                     # pyarrow Table / RecordBatch
    if isinstance(n, (int, np.integer)):
        return int(n)
    if isinstance(data, dict):
        for k in ("x", "count", "rows", "data"):
            v = data.get(k)
            if hasattr(v, "__len__"):
                return len(v)
        for v in data.values():
            if hasattr(v, "shape"):
                return int(np.asarray(v.shape)[0]) if np.ndim(v) else 0
        return 0
    shape = getattr(data, "shape", None)
    if shape is not None and len(shape):
        return int(shape[0])
    try:
        return len(data)
    except TypeError:
        return 0


def key_of_gef(gef) -> dict:
    """The dataset key of a `GefFile` itself — the same key a view over it gets, so records
    written by `read_window()` and by `StereoSeqView.load()` fold onto one dataset."""
    fp = gef.fingerprint()
    path = _abs(gef.path)
    return {"path": path, "kind": "gef", "fingerprint": fp, "key": _hash(path, fp.get("size"))}


def _trigram_class():
    try:
        from ._core import TrigramIndex
        return TrigramIndex
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# The active journal
# ---------------------------------------------------------------------------

_ACTIVE: Any = ...                       # ... = "not decided yet": read the environment first
_OFF = ("0", "off", "none", "false", "no")


def active() -> Optional[Journal]:
    """The journal records go to, or None when recording is off. Decided once, from
    `GEMINGA_JOURNAL`: unset → `~/.geminga/journal.jsonl`; `0`/`off` → None; a path → it."""
    global _ACTIVE
    if _ACTIVE is ...:
        env = os.environ.get("GEMINGA_JOURNAL")
        if env is not None and env.strip().lower() in _OFF:
            _ACTIVE = None
        else:
            _ACTIVE = Journal(env.strip() if env and env.strip() else DEFAULT_PATH)
    return _ACTIVE


def use_journal(target=None):
    """Set the active journal: a `Journal`, a path, or None to turn recording off. Returns the
    previous setting — a `Journal`, None, or the Ellipsis when it had not been decided yet —
    which can be passed back to restore it. `use_journal(...)` re-reads the environment."""
    global _ACTIVE
    prev = _ACTIVE
    if target is ...:
        _ACTIVE = ...
    elif target is None or isinstance(target, Journal):
        _ACTIVE = target
    else:
        _ACTIVE = Journal(str(target))
    return prev


def journal(path: Optional[str] = None) -> Journal:
    """`g.journal()` is the active journal (created if recording is off, without turning it
    on); `g.journal(path)` opens another file."""
    if path is not None:
        return Journal(path)
    j = active()
    return j if j is not None else Journal(DEFAULT_PATH)


def record(op: str, dataset, **kw) -> Optional[Entry]:
    """Write to the active journal, if there is one. The hook every reader calls."""
    j = active()
    if j is None:
        return None
    try:
        return j.record(op, dataset, **kw)
    except Exception as err:                 # the diary never stops the analysis
        _warn_once(f"journal record failed: {err!r}")
        return None


JournalEntry = Entry                 # the public names at package level
journal_recording = recording


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m geminga.journal",
                                 description="The trail of streamed pieces.")
    ap.add_argument("--path", default=None, help="journal file (default: the active one)")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("tail", help="the last N pieces"); p.add_argument("-n", type=int, default=20)
    p = sub.add_parser("grep", help="lines matching a pattern")
    p.add_argument("pattern"); p.add_argument("-E", "--regex", action="store_true")
    p.add_argument("-s", "--case-sensitive", action="store_true")
    sub.add_parser("datasets", help="one line per dataset")
    p = sub.add_parser("where", help="windows streamed from one dataset (key or path)")
    p.add_argument("dataset")
    p = sub.add_parser("coverage", help="an ASCII map of how often each part was streamed")
    p.add_argument("dataset"); p.add_argument("--shape", type=int, default=48)
    sub.add_parser("index", help="build or extend the trigram index")
    sub.add_parser("engine", help="which search engine would be used")
    a = ap.parse_args(argv)
    j = journal(a.path)
    if a.cmd in (None, "tail"):
        j.print_table(getattr(a, "n", 20))
    elif a.cmd == "grep":
        for ln, s in j.grep(a.pattern, regex=a.regex, ignore_case=not a.case_sensitive):
            print(f"{ln + 1}:{s}")
    elif a.cmd == "datasets":
        for d in sorted(j.datasets().values(), key=lambda d: -d.last):
            print(d.line_text())
    elif a.cmd == "where":
        for e in j.visited(a.dataset):
            print(e.line_text())
    elif a.cmd == "coverage":
        img = j.coverage(a.dataset, shape=a.shape)
        ramp = " .:-=+*#%@"
        top = max(int(img.max()), 1)
        for row in img[::-1]:
            print("".join(ramp[min(int(v * (len(ramp) - 1) / top), len(ramp) - 1)] for v in row))
        print(f"max {top} visits; {float((img > 0).mean()):.0%} of the visited extent")
    elif a.cmd == "index":
        idx = j.index()
        if idx is None:
            print("no trigram index: the installed core predates it, or the journal is empty")
            return 1
        print(f"{idx.n_lines:,} lines, {idx.n_blocks:,} blocks, {idx.n_trigrams:,} trigrams, "
              f"{_human(idx.postings_bytes)} of postings -> {j.index_path}")
    elif a.cmd == "engine":
        print(j.engine)
    return 0


if __name__ == "__main__":                    # pragma: no cover
    sys.exit(_main())
