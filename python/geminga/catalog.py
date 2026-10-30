"""A lab notebook for streamed datasets.

Students lose track of what they streamed, from where, and what it turned out to be. The
catalog is a durable, append-only record of exactly that: one JSON object per event, written
to a file you can read with `cat`, grep with `grep`, and query with DuckDB if it is installed.

JSONL first, on purpose. A text log survives a corrupted index, a half-written database and a
version bump; it diffs, it greps, and it copies between machines. DuckDB is layered on top as
a query surface over the same file rather than as the storage itself, so nothing is lost if it
is missing.
"""
from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Optional

__all__ = ["Catalog", "CatalogEntry", "catalog", "CATALOG_SCHEMA", "FACETS"]


def _append_line(path: str, line: str, expect: Optional[tuple] = None) -> Optional[tuple]:
    """Append one line to a JSONL log, and say whether the caller's cache survived the write.

    Two things go wrong with a bare ``open(path, "a").write(line)``:

    * A writer that died mid-line leaves a tail with no newline. The next record lands on that
      same line, and a fresh reader then loses *both* — the torn one and the new one. So if the
      file is non-empty and does not end in a newline, one is written first.
    * Another process may append between the caller deciding its cache is current and this
      write. Stamping the cache with the file's *new* size and mtime would then vouch for an
      event the cache never saw. So the open handle is measured with ``fstat`` before and after:
      the cache is only advanced if the file really went from the signature the caller holds
      (``expect``, a ``(size, mtime_ns)`` pair) to that size plus exactly the bytes written here.

    Returns the file's ``(size, mtime_ns)`` after the write when the cache may be advanced to it,
    otherwise ``None`` — the caller must then drop its cache and re-read the log.
    """
    data = line.encode("utf-8")
    with open(path, "a+b") as fh:
        before = os.fstat(fh.fileno())
        if before.st_size:
            fh.seek(before.st_size - 1)
            if fh.read(1) != b"\n":                  # a torn tail: start this record on a fresh line
                data = b"\n" + data
        fh.write(data)                               # append mode: always lands at the end
        fh.flush()
        after = os.fstat(fh.fileno())
    if expect is None or (before.st_size, before.st_mtime_ns) != tuple(expect):
        return None                                  # somebody wrote between the check and the open
    if after.st_size != before.st_size + len(data):
        return None                                  # somebody wrote between our open and our close
    return (after.st_size, after.st_mtime_ns)


#: Bumped only on an incompatible change. Additive fields keep the number. Other tools —
#: HyVESTA pulls facets from this file, PRISM dispatches on it — read the catalog now, so the
#: record shape is a contract: renaming, retyping or removing a field is a schema bump.
CATALOG_SCHEMA = "geminga.catalog/1"

#: The facet keys downstream tools may rely on. Values are free strings — use ontology terms
#: where you have them (UBERON, CL, MONDO, EFO) — but the keys are fixed, so a consumer can ask
#: facets_for(key)["tissue"] and never guess a column name.
FACETS = ("organism", "tissue", "assay", "disease", "sample", "donor", "condition", "platform")


@dataclass
class CatalogEntry:
    """One dataset, as the analyst understands it."""
    key: str
    target: str
    url: str = ""
    kind: str = ""               # format
    molecule: str = ""
    assay: str = ""
    assay_confidence: float = 0.0
    modality: str = ""
    note: str = ""
    tags: list = field(default_factory=list)
    bytes_streamed: int = 0
    metadata: dict = field(default_factory=dict)
    protocol_summary: str = ""
    sources: list = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    host: str = ""
    facets: dict = field(default_factory=dict)      # controlled keys, see FACETS
    local_paths: list = field(default_factory=list)  # where copies and regions live

    #: Field -> (JSON type, meaning). This is the contract; keep it in step with the class.
    SCHEMA = {
        "key":              ("string",  "analyst's short name; unique within a catalog"),
        "target":           ("string",  "path or Hub id as opened"),
        "url":              ("string",  "where it was downloaded from, if known"),
        "kind":             ("string",  "container format, e.g. 'AnnData h5ad'"),
        "molecule":         ("string",  "DNA | RNA | protein | lipid | metabolite | image | mixed"),
        "assay":            ("string",  "profiler's leading assay hypothesis"),
        "assay_confidence": ("number",  "probability of that hypothesis, 0..1"),
        "modality":         ("string",  "single | multi-modal | unknown"),
        "note":             ("string",  "free text"),
        "tags":             ("array of string", "free labels"),
        "bytes_streamed":   ("integer", "cumulative bytes streamed through GEMINGA"),
        "metadata":         ("object",  "profiler metadata; keys not frozen"),
        "protocol_summary": ("string",  "assistant or analyst prose about the protocol"),
        "sources":          ("array of string", "links worth reading"),
        "first_seen":       ("number",  "unix time"),
        "last_seen":        ("number",  "unix time"),
        "host":             ("string",  "machine that wrote the record"),
        "facets":           ("object",  f"keys from FACETS {FACETS}; string values"),
        "local_paths":      ("array of string", "absolute paths of local copies or regions"),
    }

    @classmethod
    def from_event(cls, ev: dict) -> "CatalogEntry":
        """Tolerant constructor: unknown fields are ignored, missing ones default."""
        return cls(**{f: ev[f] for f in cls.__dataclass_fields__ if f in ev})

    def line(self) -> str:
        a = f"{self.assay} ({self.assay_confidence:.0%})" if self.assay else "assay unknown"
        return f"{self.key:<20} {self.kind or '?':<18} {a:<28} {self.target}"


class Catalog:
    """Append-only JSONL, with a folded current view.

        cat = g.catalog("~/geminga-catalog.jsonl")
        cat.record("chip1", "run1.gef", url=..., profile=prof, note="tumour section")
        cat.find("stereo")
        cat.print_table()
    """

    def __init__(self, path: str = "~/.geminga/catalog.jsonl"):
        # absolute, so a later os.chdir() cannot fork the log into a second file
        self.path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.host = socket.gethostname()
        self._fold: Optional[tuple] = None          # ((size, mtime_ns), folded dict)

    def _signature(self) -> Optional[tuple]:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return None
        return (st.st_size, st.st_mtime_ns)

    # ---- writing ----
    def _append(self, obj: dict) -> None:
        obj = dict(obj)
        obj.setdefault("ts", time.time())
        obj.setdefault("schema", CATALOG_SCHEMA)
        line = json.dumps(obj, default=str)
        fold = self._fold if self._fold is not None and self._fold[0] == self._signature() else None
        # The cache is advanced only if the file went from the signature it was built against to
        # exactly that plus this line; if another process appended in between, drop it and re-read.
        sig = _append_line(self.path, line + "\n", fold[0] if fold is not None else None)
        if fold is not None and sig is not None:
            self._apply(fold[1], json.loads(line))
            self._fold = (sig, fold[1])
        else:
            self._fold = None

    @staticmethod
    def _clean_facets(facets: Optional[dict]) -> dict:
        """Keep only the controlled keys; a typo in a key is a silent data loss, so say so."""
        if not facets:
            return {}
        bad = [k for k in facets if k not in FACETS]
        if bad:
            raise KeyError(f"unknown facet key(s) {bad}; allowed: {list(FACETS)}. "
                           f"Put anything else in metadata= or tags=.")
        return {k: str(v) for k, v in facets.items() if v is not None and str(v) != ""}

    def record(self, key: str, target: Any, url: str = "", profile=None,
               note: str = "", tags: Optional[list] = None,
               bytes_streamed: int = 0, protocol_summary: str = "",
               sources: Optional[list] = None, facets: Optional[dict] = None,
               local_paths: Optional[list] = None, **extra) -> CatalogEntry:
        """Record (or update) what a dataset is. Safe to call repeatedly.

        The profiler's assay is mirrored into ``facets["assay"]`` for as long as the analyst has
        not named one: ``metadata["assay_from_profiler"]`` is True while the facet is the
        profiler's (so a newer profile refreshes it) and False once a person set it, in
        ``facets=`` here or via :meth:`annotate` (after which no profile overwrites it).
        """
        now = time.time()
        prev = self.get(key)
        # who owns the assay facet: the profiler (True), the analyst (False), or nobody yet (None)
        auto_assay = prev.metadata.get("assay_from_profiler") if prev else None
        e = CatalogEntry(
            key=key, target=str(target), url=url or (prev.url if prev else ""),
            note=note or (prev.note if prev else ""),
            tags=list(tags or (prev.tags if prev else [])),
            bytes_streamed=int(bytes_streamed) + (prev.bytes_streamed if prev else 0),
            protocol_summary=protocol_summary or (prev.protocol_summary if prev else ""),
            sources=list(sources or (prev.sources if prev else [])),
            first_seen=prev.first_seen if prev else now, last_seen=now, host=self.host,
            facets=dict(prev.facets) if prev else {},
            local_paths=list(prev.local_paths) if prev else [],
        )
        if profile is not None:
            e.kind = profile.format or ""
            e.molecule = profile.molecule or ""
            e.modality = profile.modality
            if profile.best:
                e.assay = profile.best.label
                e.assay_confidence = float(profile.best.probability)
            e.metadata = dict(profile.metadata)
            e.metadata["confident"] = profile.confident
            if not profile.confident and profile.best:
                e.metadata["unsettled_because"] = profile.best.distinguish
        elif prev:
            e.kind, e.molecule, e.assay = prev.kind, prev.molecule, prev.assay
            e.assay_confidence, e.modality = prev.assay_confidence, prev.modality
            e.metadata = dict(prev.metadata)
        e.metadata.update(extra)
        given = self._clean_facets(facets)
        e.facets.update(given)
        # the profiler's assay is a facet too — until the analyst names one, and only then final
        if "assay" in given:
            auto_assay = False
        elif e.assay and ("assay" not in e.facets or auto_assay is True):
            e.facets["assay"] = e.assay
            auto_assay = True
        if auto_assay is not None:
            e.metadata["assay_from_profiler"] = bool(auto_assay)
        for p in (local_paths or []):
            p = os.path.abspath(os.path.expanduser(str(p)))
            if p not in e.local_paths:
                e.local_paths.append(p)
        self._append({"event": "record", **asdict(e)})
        return e

    def note_opened(self, key: str, target: Any, ram: str, note: str = "") -> None:
        """Called by Workspace when a cell is added — a breadcrumb, not a full record."""
        self._append({"event": "open", "key": key, "target": str(target),
                      "ram": ram, "note": note, "host": self.host})

    def note_streamed(self, key: str, nbytes: int, what: str = "") -> None:
        self._append({"event": "stream", "key": key, "bytes": int(nbytes), "what": what})

    def note_local(self, key: str, path: str, what: str = "") -> None:
        """A copy or an extracted region landed on local disk (staging, harvest)."""
        self._append({"event": "local", "key": key,
                      "path": os.path.abspath(os.path.expanduser(str(path))), "what": what})

    def annotate(self, key: str, facets: Optional[dict] = None, **fields) -> None:
        """Add the analyst's own judgement — the thing automated profiling cannot supply.

        ``fields`` are :class:`CatalogEntry` fields (``note=``, ``tags=``, ``molecule=`` …);
        controlled vocabulary goes in ``facets={"tissue": "cortex"}``. Anything else raises
        ``TypeError`` instead of being logged and then silently dropped when the log is folded.
        """
        settable = [f for f in CatalogEntry.__dataclass_fields__ if f != "key"]
        bad = [f for f in fields if f not in settable]
        if bad:
            as_facets = [f for f in bad if f in FACETS]
            hint = (f" {as_facets} are facet keys: pass facets={{"
                    + ", ".join(f"{f!r}: ..." for f in as_facets) + "}}." if as_facets else "")
            raise TypeError(f"annotate() got unexpected field(s) {bad}.{hint} "
                            f"Fields it can set: {settable}.")
        ev = {"event": "annotate", "key": key, **fields}
        if facets:
            ev["facets"] = self._clean_facets(facets)
        self._append(ev)

    # ---- reading ----
    def events(self) -> Iterator[dict]:
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue          # a partial last line survives; skip it

    @staticmethod
    def _apply(out: dict, ev: dict) -> None:
        """Fold one event into the current state. The one place the log's semantics live."""
        k = ev.get("key")
        if not k:
            return
        kind = ev.get("event")
        if kind == "record":
            out[k] = CatalogEntry.from_event(ev)
        elif kind == "local" and k in out:
            p = ev.get("path", "")
            if p and p not in out[k].local_paths:
                out[k].local_paths.append(p)
        elif kind == "stream" and k in out:
            out[k].bytes_streamed += int(ev.get("bytes", 0))
            out[k].last_seen = ev.get("ts", out[k].last_seen)
        elif kind == "annotate" and k in out:
            for f, v in ev.items():
                if f == "facets" and isinstance(v, dict):
                    out[k].facets.update({kk: str(vv) for kk, vv in v.items()})
                    if str(v.get("assay") or "") and isinstance(out[k].metadata, dict):
                        out[k].metadata["assay_from_profiler"] = False   # a person named the assay
                elif f in CatalogEntry.__dataclass_fields__ and f not in ("key",):
                    setattr(out[k], f, v)
        elif kind == "open" and k not in out:
            out[k] = CatalogEntry(key=k, target=ev.get("target", ""),
                                  note=ev.get("note", ""),
                                  first_seen=ev.get("ts", 0), last_seen=ev.get("ts", 0))

    def _folded(self) -> dict:
        """The log folded into current state, cached against the file's size and mtime and kept
        current across this object's own appends — so `record()` in a loop, which looks each
        key up first, never re-parses the log. Callers get copies; this dict is the cache."""
        sig = self._signature()
        if sig is None:
            return {}
        if self._fold is not None and self._fold[0] == sig:
            return self._fold[1]
        out: dict[str, CatalogEntry] = {}
        for ev in self.events():
            self._apply(out, ev)
        self._fold = (sig, out)
        return out

    def entries(self) -> dict[str, CatalogEntry]:
        """The current state of every key, as copies you may edit."""
        return {k: _copy_entry(e) for k, e in self._folded().items()}

    def get(self, key: str) -> Optional[CatalogEntry]:
        e = self._folded().get(key)
        return _copy_entry(e) if e is not None else None

    def find(self, needle: str) -> list[CatalogEntry]:
        n = needle.lower()
        return [_copy_entry(e) for e in self._folded().values()
                if n in " ".join([e.key, e.target, e.kind, e.assay, e.molecule,
                                  e.note, " ".join(e.tags),
                                  " ".join(e.facets.values())]).lower()]

    def unsettled(self) -> list[CatalogEntry]:
        """Datasets whose assay the profiler could not pin down — the analyst's queue."""
        return [e for e in self.entries().values()
                if not e.metadata.get("confident", False)]

    def print_table(self, needle: str = "") -> None:
        rows = self.find(needle) if needle else list(self.entries().values())
        if not rows:
            print("(catalog empty)")
            return
        print(f"{'key':<20} {'format':<18} {'assay':<28} target")
        for e in sorted(rows, key=lambda x: -x.last_seen):
            print(" " + e.line())
        un = [e for e in rows if not e.metadata.get("confident", False)]
        if un:
            print(f"\n{len(un)} of {len(rows)} still unsettled — "
                  f"`catalog.unsettled()` lists what would decide each.")

    # ---- the surface other tools read ----
    def facets_for(self, key: str) -> dict:
        """``{facet: value}`` for one dataset, with every FACETS key present ('' if unset)."""
        e = self.get(key)
        if e is None:
            raise KeyError(f"no catalog entry {key!r}")
        return {f: e.facets.get(f, "") for f in FACETS}

    def facet_table(self) -> list:
        """One row per dataset: key, target, then every FACETS column. Stable column order."""
        rows = []
        for e in sorted(self.entries().values(), key=lambda x: x.key):
            row = {"key": e.key, "target": e.target}
            row.update({f: e.facets.get(f, "") for f in FACETS})
            row["local_paths"] = list(e.local_paths)
            rows.append(row)
        return rows

    def validate(self) -> list:
        """Problems found against CatalogEntry.SCHEMA, as strings. Empty means conformant."""
        problems = []
        types = {"string": str, "number": (int, float), "integer": int, "object": dict,
                 "array of string": list}
        for i, ev in enumerate(self.events(), 1):
            sch = ev.get("schema", "")
            if sch and sch != CATALOG_SCHEMA:
                problems.append(f"line {i}: schema {sch!r} is not {CATALOG_SCHEMA!r}")
            if ev.get("event") != "record":
                continue
            for f, (jtype, _) in CatalogEntry.SCHEMA.items():
                if f not in ev:
                    continue
                if not isinstance(ev[f], types[jtype]) or (jtype == "integer"
                                                            and isinstance(ev[f], bool)):
                    problems.append(f"line {i}: {f} should be {jtype}, got "
                                    f"{type(ev[f]).__name__}")
            bad = [k for k in (ev.get("facets") or {}) if k not in FACETS]
            if bad:
                problems.append(f"line {i}: unknown facet key(s) {bad}")
        return problems

    # ---- optional query surface ----
    def duckdb(self):
        """A DuckDB connection with the log exposed as a view. Storage stays the JSONL."""
        import duckdb
        con = duckdb.connect()
        con.execute(
            f"CREATE VIEW events AS SELECT * FROM read_json_auto('{self.path}', "
            f"union_by_name=true, ignore_errors=true)")
        return con

    def to_parquet(self, path: str) -> str:
        """Snapshot the folded view for sharing or archiving."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        rows = [asdict(e) for e in self.entries().values()]
        for r in rows:
            r["tags"] = ",".join(r["tags"])
            r["sources"] = ",".join(r["sources"])
            r["local_paths"] = ",".join(r["local_paths"])
            r["metadata"] = json.dumps(r["metadata"], default=str)
            r["facets"] = json.dumps(r["facets"], default=str)
        pq.write_table(pa.Table.from_pylist(rows), path)
        return path

    def __repr__(self) -> str:
        try:
            n = len(self.entries())
        except Exception:
            n = 0
        return f"Catalog({self.path}, {n} datasets)"


def _copy_entry(e: CatalogEntry) -> CatalogEntry:
    """A per-call copy, so a caller editing an entry cannot corrupt the cached fold."""
    d = dict(e.__dict__)
    for k in ("tags", "sources", "local_paths"):
        d[k] = list(d[k])
    d["metadata"] = dict(d["metadata"])
    d["facets"] = dict(d["facets"])
    return CatalogEntry(**d)


def catalog(path: str = "~/.geminga/catalog.jsonl") -> Catalog:
    """Open (or create) the catalog of datasets you have streamed."""
    return Catalog(path)
