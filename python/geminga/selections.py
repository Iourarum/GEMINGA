"""Selections: the regions you highlighted, why, and what you found about them.

Navigating a dataset produces questions — *this corner of the rim looks different*, *I need
more cells like these*, *what happened in the culture around hour 40*. A selection is that
question pinned to coordinates, and this module keeps it: what was selected, on which dataset, at
what level of detail, with what intent, and what later searches turned up.

**Geometry.** Rectangles and polygons on spatial maps, intervals on 1-D extents (records, cells,
rows), and value ranges on a named column — which is how a time series is selected: `t` between
two instants, whatever the row order.

**Persistence.** An append-only JSON-lines log beside the catalogue (`~/.geminga/`), one object
per event. It survives crashes, diffs cleanly, and can be read line by line from any language —
PRISM included — without understanding GEMINGA. A crash can leave the last line half-written;
the next append starts a fresh line, so only the torn record is lost. Each selection records its
dataset by path or URL *and* fingerprint, so it reattaches to the same file on the next session,
local or remote; local paths are stored absolute, so that holds from any working directory.

**Three searches, from nearest to farthest.**

1. `describe()` — what is inside. For a Stereo-seq region: observations, total MID, density
   against the section, and genes enriched or depleted relative to the whole section. For a
   time range: each numeric column inside against the whole series.
2. `similar()` — where else in *this* dataset looks like it. The section is tiled at the
   selection's own size, each tile summarised as a gene profile from one coarse level, and
   ranked by cosine similarity. Time series are compared window by window with Polars'
   `group_by_dynamic`. Results come back as candidate selections, linked to their parent.
3. `leads()` — where the answer might be in data you *do not have yet*. Locally, which other
   datasets you already hold measured these genes at all. Externally, queries against GEO,
   PubMed and ENA built from the selection's enriched genes and your context terms; accessions
   they return go straight into `acquisition_plan()`, so "find more data" ends in a priced,
   dry-run download rather than a browser tab.

Nothing here decides what a region *means*. It records what you marked, measures it, and finds
related data; interpretation stays with the analyst.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import timezone
from typing import Iterable, Optional, Sequence

import numpy as np

from .catalog import _append_line


__all__ = ["Selection", "SelectionBook", "selections", "select", "select_polygon",
           "select_values", "describe", "similar", "leads", "local_leads", "external_queries",
           "dataset_key", "INTENTS"]

#: Why a region was marked. Drawn in different colours; filterable in the book.
INTENTS = {
    "interesting": (0.91, 0.25, 0.18, 1.0),     # red
    "double-check": (0.93, 0.55, 0.13, 1.0),    # orange — something may be wrong here
    "need-data":   (0.18, 0.47, 0.77, 1.0),     # blue
    "compare":     (0.55, 0.35, 0.75, 1.0),     # violet
    "candidate":   (0.96, 0.77, 0.09, 1.0),     # yellow — produced by similar()
    "exclude":     (0.45, 0.45, 0.45, 1.0),     # grey
}


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def _abs(path):
    """The absolute form of a local path, whether or not it exists yet. A URL (anything with
    `://`), the repr of an object, an empty string or a non-string comes back unchanged. A
    relative path recorded as given would name a different file from every other working
    directory — and a different key."""
    if isinstance(path, str) and path and "://" not in path and not path.startswith("<"):
        return os.path.abspath(os.path.expanduser(path))
    return path


def dataset_key(view_or_path) -> dict:
    """Where a selection belongs: path or URL, plus a fingerprint so it reattaches to the same
    bytes next session and is *not* silently applied to a different file of the same name.

    Local paths are stored absolute, so the selection can be described and found again from any
    working directory; URLs are kept as given."""
    gef = getattr(view_or_path, "gef", None)
    if gef is not None:
        fp = gef.fingerprint()
        gpath = _abs(gef.path)
        return {"path": gpath, "kind": "gef", "fingerprint": fp,
                "key": _hash(gpath, fp.get("size"))}
    path, n_files = None, None
    if isinstance(view_or_path, str):
        path = view_or_path
    else:
        path = getattr(view_or_path, "path", None)
        # a view's source: a reader with a path, or a Parquet Dataset with its file list
        for attr in ("src", "a", "m", "vol", "index", "gem", "reader"):
            if path is not None:
                break
            obj = getattr(view_or_path, attr, None)
            if obj is None:
                continue
            path = getattr(obj, "path", None)
            if path is None:
                files = getattr(obj, "files", None)
                if files:
                    path, n_files = files[0], len(files)
        if path is None:
            path = repr(view_or_path)
    path = _abs(path)
    size = None
    if isinstance(path, str) and os.path.exists(path):
        size = os.path.getsize(path)
    kind = type(view_or_path).__name__ if not isinstance(view_or_path, str) else "path"
    fp = {"size": size}
    if n_files is not None and n_files > 1:
        fp["files"] = n_files                    # keyed by the first file; the count says it is a set
    return {"path": str(path), "kind": kind, "fingerprint": fp, "key": _hash(str(path), size)}


def _hash(path, size) -> str:
    return hashlib.blake2b(f"{path}|{size}".encode(), digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

@dataclass
class Selection:
    id: str
    name: str
    dataset: dict
    kind: str                      # "rect" | "polygon" | "interval" | "column-range"
    geometry: dict
    level: dict = field(default_factory=dict)
    genes: Optional[list] = None
    intent: str = "interesting"
    status: str = "open"           # open -> described -> searched, then whatever review adds
                                   # ("in-review" | "confirmed" | "refuted" | "resolved")
    note: str = ""
    tags: list = field(default_factory=list)
    parent: Optional[str] = None
    score: Optional[float] = None
    created: float = 0.0
    updated: float = 0.0
    summary: dict = field(default_factory=dict)
    leads: list = field(default_factory=list)
    # Free-form, queryable metadata (build, tissue, sample, assignee, …), the status trail, and
    # typed links to other records. Defaults keep older log lines loadable unchanged.
    metadata: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    links: list = field(default_factory=list)

    # ---- geometry helpers ----
    @property
    def bounds(self) -> tuple:
        """Extent in the kind's own coordinates. Always numbers: a record that has no position
        yet (an rsID-only variant, a sequence given only as record ids) reports 0, never None —
        `placed` says whether that 0 means anything."""
        g = self.geometry
        if self.kind in ("variant", "locus"):
            return (g.get("start") or 0, g.get("end") or 0)
        if self.kind == "sequence":
            return (g.get("start") or 0, g.get("end") or g.get("length") or 0)
        if self.kind in ("cells", "genes"):
            return (0, g.get("n", len(g.get("genes", []))))
        if self.kind == "rect":
            return (g["x0"], g["x1"], g["y0"], g["y1"])
        if self.kind == "polygon":
            p = np.asarray(g["points"], float)
            return (float(p[:, 0].min()), float(p[:, 0].max()),
                    float(p[:, 1].min()), float(p[:, 1].max()))
        return (g["lo"], g["hi"])

    @property
    def color(self) -> tuple:
        return INTENTS.get(self.intent, INTENTS["interesting"])

    def contains(self, x, y) -> np.ndarray:
        x = np.asarray(x, float); y = np.asarray(y, float)
        if self.kind == "rect":
            g = self.geometry
            return (x >= g["x0"]) & (x < g["x1"]) & (y >= g["y0"]) & (y < g["y1"])
        if self.kind == "polygon":
            return _in_polygon(x, y, self.geometry["points"])
        raise TypeError(f"{self.kind} selections are not spatial")

    @property
    def area(self) -> float:
        if self.kind == "rect":
            x0, x1, y0, y1 = self.bounds
            return max(x1 - x0, 0) * max(y1 - y0, 0)
        if self.kind == "polygon":
            p = np.asarray(self.geometry["points"], float)
            x, y = p[:, 0], p[:, 1]
            return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)
        lo, hi = self.bounds
        return float(hi - lo)

    def overlaps(self, other: "Selection") -> float:
        """Intersection over the smaller extent — used to suppress near-duplicate candidates."""
        a, b = self.bounds, other.bounds
        if len(a) != len(b):
            return 0.0
        if len(a) == 2:
            inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
            return inter / max(min(a[1] - a[0], b[1] - b[0]), 1e-12)
        ix = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[2], b[2]))
        return ix * iy / max(min((a[1] - a[0]) * (a[3] - a[2]),
                                 (b[1] - b[0]) * (b[3] - b[2])), 1e-12)

    @property
    def spatial(self) -> bool:
        return self.kind in ("rect", "polygon")

    @property
    def placed(self) -> bool:
        """False for a record whose coordinates are not known (yet): an rsID-only variant, or a
        sequence given only as record ids. Its `bounds` are then a placeholder 0."""
        g = self.geometry
        if self.kind in ("variant", "locus"):
            return g.get("start") is not None
        if self.kind == "sequence":
            return g.get("start") is not None or bool(g.get("end") or g.get("length"))
        return True

    def to_feature(self) -> dict:
        """GeoJSON, in chip coordinates — what QuPath and the napari shapes layer exchange."""
        if self.kind == "rect":
            x0, x1, y0, y1 = self.bounds
            ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        elif self.kind == "polygon":
            ring = [list(map(float, p)) for p in self.geometry["points"]]
            ring = ring + [ring[0]] if ring[0] != ring[-1] else ring
        else:
            raise TypeError("only spatial selections have a GeoJSON form")
        return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {"id": self.id, "name": self.name, "intent": self.intent,
                               "status": self.status, "note": self.note,
                               "dataset": self.dataset.get("path"), "level": self.level}}

    def __repr__(self) -> str:
        b = ", ".join(f"{v:,.0f}" for v in self.bounds) if self.placed else "?"
        s = f" score {self.score:.3f}" if self.score is not None else ""
        return f"Selection({self.name!r}, {self.kind} [{b}], {self.intent}/{self.status}{s})"


def _in_polygon(x, y, pts) -> np.ndarray:
    """Even-odd ray casting, vectorised over points."""
    p = np.asarray(pts, float)
    inside = np.zeros(np.asarray(x).shape, bool)
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]; xj, yj = p[j]
        dy = (yj - yi) if (yj - yi) != 0 else 1e-12
        cross = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / dy + xi)
        inside ^= cross
        j = i
    return inside


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

#: The part of the workflow that measuring and searching move a record along by themselves.
#: Everything past it (in-review, confirmed, refuted, resolved) is a person's decision.
_AUTO_FLOW = ("open", "described", "searched")


def _forward(current: str, target: str) -> str:
    """`target` if it is further along open -> described -> searched than `current`, else
    `current`. A status off that path means a person took the record over; it is never replaced."""
    if current not in _AUTO_FLOW or target not in _AUTO_FLOW:
        return current
    return target if _AUTO_FLOW.index(target) > _AUTO_FLOW.index(current) else current


class SelectionBook:
    """Append-only log of selections. Read it with `cat`, grep it, or fold it with `.all()`."""

    def __init__(self, path: str = "~/.geminga/selections.jsonl"):
        # absolute, so a later os.chdir() cannot fork the log into a second file
        self.path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fold: Optional[tuple] = None          # ((size, mtime_ns), folded dict)

    def _append(self, ev: dict):
        ev = dict(ev); ev.setdefault("ts", time.time())
        line = json.dumps(ev, default=_json)
        fold = self._fold if self._fold is not None and self._fold[0] == self._signature() else None
        # Repairs a torn tail first, and advances the cache only if nobody else wrote meanwhile.
        sig = _append_line(self.path, line + "\n", fold[0] if fold is not None else None)
        if fold is not None and sig is not None:
            self._apply(fold[1], json.loads(line))
            self._fold = (sig, fold[1])
        else:
            self._fold = None

    def _signature(self) -> Optional[tuple]:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return None
        return (st.st_size, st.st_mtime_ns)

    def add(self, sel: Selection) -> Selection:
        now = time.time()
        sel.created = sel.created or now
        sel.updated = now
        self._append({"event": "add", **asdict(sel)})
        return sel

    def update(self, sel_or_id, **fields) -> Selection:
        sid = sel_or_id.id if isinstance(sel_or_id, Selection) else sel_or_id
        cur = self.get(sid)
        if cur is None:
            raise KeyError(f"no selection {sid!r}")
        for k, v in fields.items():
            setattr(cur, k, v)
        cur.updated = time.time()
        self._append({"event": "update", "id": sid, **{k: v for k, v in fields.items()},
                      "updated": cur.updated})
        return cur

    def _advance(self, sel_or_id, status: str, **fields) -> Selection:
        """Move a record's status *forward* along open -> described -> searched, together with
        any other `fields`, in one event. The one gate through which `describe()` and `leads()`
        (and `describe_any()` / `leads_any()`) change status.

        The stored record is re-read, not the caller's object: that one is often stale, since
        `lib.confirm(s)` updates the log and not `s`. A record that is already further along, or
        that has left this path for review (in-review, confirmed, refuted, resolved), keeps its
        status; only the fields are written. Returns the live record.
        """
        sid = sel_or_id.id if isinstance(sel_or_id, Selection) else sel_or_id
        cur = self.get(sid)
        if cur is None:
            raise KeyError(f"no selection {sid!r}")
        new = _forward(cur.status, status)
        if new != cur.status:
            fields["status"] = new
        return self.update(cur, **fields) if fields else cur

    def delete(self, sel_or_id):
        sid = sel_or_id.id if isinstance(sel_or_id, Selection) else sel_or_id
        self._append({"event": "delete", "id": sid})

    @staticmethod
    def _apply(out: dict, ev: dict) -> None:
        ev = dict(ev)
        kind = ev.pop("event", None)
        if kind == "add":
            ev.pop("ts", None)
            out[ev["id"]] = Selection(**{k: v for k, v in ev.items()
                                         if k in Selection.__dataclass_fields__})
        elif kind == "update" and ev.get("id") in out:
            s = out[ev["id"]]
            for k, v in ev.items():
                if k in Selection.__dataclass_fields__ and k != "id":
                    setattr(s, k, v)
        elif kind == "delete":
            out.pop(ev.get("id"), None)

    def _folded(self) -> dict:
        """The log folded into current state, cached against the file's size and mtime and
        advanced on this object's own appends — `get`/`query`/`related` in a loop cost a stat
        each, not a parse of every line each."""
        sig = self._signature()
        if sig is None:
            return {}
        if self._fold is not None and self._fold[0] == sig:
            return self._fold[1]
        out: dict[str, Selection] = {}
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue                                  # a torn last line survives
                self._apply(out, ev)
        self._fold = (sig, out)
        return out

    def all(self) -> dict:
        """Every selection by id — a fresh dict over the live records."""
        return dict(self._folded())

    def get(self, sid: str) -> Optional[Selection]:
        return self._folded().get(sid)

    def find(self, name: Optional[str] = None, dataset=None, intent: Optional[str] = None,
             status: Optional[str] = None, include_candidates: bool = True) -> list:
        key = dataset_key(dataset)["key"] if dataset is not None and not isinstance(
            dataset, dict) else (dataset or {}).get("key")
        out = []
        for s in self.all().values():
            if name and name.lower() not in s.name.lower():
                continue
            if key and s.dataset.get("key") != key:
                continue
            if intent and s.intent != intent:
                continue
            if status and s.status != status:
                continue
            if not include_candidates and s.intent == "candidate":
                continue
            out.append(s)
        return sorted(out, key=lambda s: s.created)

    def open_questions(self) -> list:
        """Everything marked as needing data and not yet resolved — the analyst's queue."""
        return [s for s in self.all().values()
                if s.intent == "need-data" and s.status != "resolved"]

    def to_geojson(self, dataset=None) -> dict:
        feats = [s.to_feature() for s in self.find(dataset=dataset)
                 if s.kind in ("rect", "polygon")]
        return {"type": "FeatureCollection", "features": feats}

    def write_dispatch(self, path: str) -> str:
        """A flat, dataset-keyed index for PRISM: selections, geometry and leads, per dataset.
        Any JSON module can act on it without knowing what a Selection is."""
        by: dict[str, dict] = {}
        for s in self.all().values():
            d = by.setdefault(s.dataset.get("key", "?"), {"dataset": s.dataset.get("path"),
                                                         "selections": []})
            d["selections"].append({
                "id": s.id, "name": s.name, "kind": s.kind, "geometry": s.geometry,
                "level": s.level, "intent": s.intent, "status": s.status, "note": s.note,
                "parent": s.parent, "score": s.score,
                "top_genes": [g["gene"] for g in s.summary.get("enriched", [])[:10]],
                "leads": [{k: l.get(k) for k in ("kind", "source", "title", "accession", "url")}
                          for l in s.leads]})
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"schema": "geminga.selections/1", "datasets": by}, fh, indent=2,
                      default=_json)
        return path

    def print_table(self, dataset=None) -> None:
        rows = self.find(dataset=dataset)
        if not rows:
            print("(no selections)"); return
        print(f"  {'name':<24} {'kind':<12} {'intent':<11} {'status':<10} {'leads':>5}  dataset")
        for s in rows:
            ds = os.path.basename(str(s.dataset.get("path", "")))[:34]
            print(f"  {s.name[:24]:<24} {s.kind:<12} {s.intent:<11} {s.status:<10} "
                  f"{len(s.leads):>5}  {ds}")
        q = self.open_questions()
        if q:
            print(f"\n  {len(q)} open question(s) marked need-data")

    def __len__(self) -> int:
        return len(self.all())

    def __repr__(self) -> str:
        return f"SelectionBook({self.path}, {len(self)} selections)"


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def selections(path: str = "~/.geminga/selections.jsonl") -> SelectionBook:
    """Open (or create) the selection log."""
    return SelectionBook(path)


# ---------------------------------------------------------------------------
# Making selections
# ---------------------------------------------------------------------------

def _level_of(view) -> dict:
    if hasattr(view, "gef"):
        return {"bin": int(view.bin_size)}
    return {"zoom": int(getattr(view, "zoom", 0)), "stride": int(getattr(view, "stride", 1))}


def select(view, name: str, bounds: Optional[tuple] = None, *, intent: str = "interesting",
           note: str = "", tags: Sequence[str] = (), book: Optional[SelectionBook] = None,
           describe_now: bool = False) -> Selection:
    """Pin a region: the view's current window, or explicit `bounds`.

    Two-number bounds make an interval (records, cells, rows); four make a rectangle
    (x0, x1, y0, y1). The level of detail and any gene filter in force are recorded with it.
    """
    if intent not in INTENTS:
        raise ValueError(f"intent must be one of {list(INTENTS)}")
    b = tuple(bounds) if bounds is not None else tuple(view.window)
    if len(b) == 4:
        kind, geom = "rect", {"x0": float(b[0]), "x1": float(b[1]),
                              "y0": float(b[2]), "y1": float(b[3])}
    elif len(b) == 2:
        kind, geom = "interval", {"lo": float(b[0]), "hi": float(b[1])}
    else:
        raise ValueError("bounds must be (lo, hi) or (x0, x1, y0, y1); use select_polygon() "
                         "for 3-D volumes' slices or arbitrary shapes")
    sel = Selection(id=uuid.uuid4().hex[:12], name=name, dataset=dataset_key(view),
                    kind=kind, geometry=geom, level=_level_of(view),
                    genes=list(getattr(view, "genes", None) or []) or None,
                    intent=intent, note=note, tags=list(tags))
    _journal_select(sel, b, view)
    if describe_now:
        describe(sel, view)
    if book is not None:
        book.add(sel)
    return sel


def _journal_select(sel: Selection, bounds, view=None) -> None:
    """One `select` record in the active journal: the pinned window, with the selection's id so
    the pieces read while describing or searching it can be found again by that id."""
    from .journal import _warn_once, active
    j = active()
    if j is None:
        return
    unit = getattr(getattr(view, "extent", None), "kind", "") if view is not None else ""
    try:
        j.record("select", sel.dataset, window=list(bounds) if bounds is not None else None,
                 level=dict(sel.level), unit=unit, genes=sel.genes, selection=sel.id,
                 note=sel.note, tags=[f"name={sel.name}", f"intent={sel.intent}", f"kind={sel.kind}"]
                 + list(sel.tags))
    except Exception as err:
        _warn_once(f"journal: {err!r}")


def select_polygon(view, name: str, points: Sequence[Sequence[float]], **kw) -> Selection:  # noqa: E302
    """A freehand region: a tumour margin, a vessel, a lesion outline."""
    book = kw.pop("book", None)
    describe_now = kw.pop("describe_now", False)
    sel = select(view, name, bounds=_poly_bounds(points), **kw)
    sel.kind = "polygon"
    sel.geometry = {"points": [[float(x), float(y)] for x, y in points]}
    if describe_now:
        describe(sel, view)
    if book is not None:
        book.add(sel)
    return sel


def _poly_bounds(points):
    p = np.asarray(points, float)
    return (p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max())


def select_values(source, column: str, lo, hi, name: str, *, intent: str = "interesting",
                  note: str = "", book: Optional[SelectionBook] = None) -> Selection:
    """A range of values on a named column — how a time window is selected in a time series,
    independent of how the rows happen to be ordered on disk.

    `lo` and `hi` are numbers, or datetimes for a temporal column; a datetime without a time
    zone is taken as UTC (as Polars does), so the window does not depend on the local `TZ`.
    """
    path = source if isinstance(source, str) else (
        getattr(source, "transcripts", None) or getattr(source, "path", None) or source.files[0])
    sel = Selection(id=uuid.uuid4().hex[:12], name=name, dataset=dataset_key(path),
                    kind="column-range",
                    geometry={"column": column, "lo": _num(lo), "hi": _num(hi)},
                    intent=intent, note=note)
    _journal_select(sel, (sel.geometry["lo"], sel.geometry["hi"]))
    if book is not None:
        book.add(sel)
    return sel


def _num(v):
    """A bound as a float: seconds since the epoch for datetimes, the number itself otherwise.

    A datetime without a time zone is read as **UTC**, the same assumption Polars makes for a
    naive Datetime column when it takes `dt.epoch`. `datetime.timestamp()` would read it in the
    machine's *local* zone instead, and the same window would then select different rows
    depending on the `TZ` of whoever ran it. Aware datetimes keep their own offset.
    """
    if hasattr(v, "timestamp"):
        if getattr(v, "tzinfo", 1) is None:
            v = v.replace(tzinfo=timezone.utc)
        return float(v.timestamp())
    return float(v)


# ---------------------------------------------------------------------------
# 1. describe — what is inside
# ---------------------------------------------------------------------------

def describe(sel: Selection, view=None, *, background_bin: Optional[int] = None,
             min_count: int = 5, top: int = 12, book: Optional[SelectionBook] = None) -> dict:
    """Measure the selection and store the result in `sel.summary`."""
    from .journal import recording
    with recording(via="describe", selection=sel.id):
        if sel.kind in ("rect", "polygon") and view is not None and hasattr(view, "gef"):
            summ = _describe_gef(sel, view, background_bin, min_count, top)
        elif sel.kind == "column-range":
            summ = _describe_column_range(sel)
        elif sel.kind == "interval" and view is not None:
            summ = _describe_interval(sel, view)
        else:
            raise TypeError(f"cannot describe a {sel.kind} selection without a matching view")
    sel.summary = summ
    if book is not None:                       # the stored record decides, not this (maybe stale) object
        sel.status = book._advance(sel, "described", summary=summ).status
    else:
        sel.status = _forward(sel.status, "described")
    return summ


def _background_mid(gef, b_bg: int) -> np.ndarray:
    """Per-gene total MID across the section. Totals are invariant to binning, so a coarse
    level gives the right answer for a fraction of the reading."""
    names, offs, cnts = gef.gene_index(b_bg)
    exp = gef.f[gef.levels[b_bg].path]["expression"]
    tot = np.zeros(len(names), np.float64)
    for i in range(len(names)):
        n = int(cnts[i])
        if n:
            lo = int(offs[i])
            tot[i] = float(np.asarray(exp[lo:lo + n]["count"], np.float64).sum())
    return tot


def _describe_gef(sel, view, background_bin, min_count, top) -> dict:
    gef = view.gef
    b = int(sel.level.get("bin") or view.bin_size)
    x0, x1, y0, y1 = sel.bounds
    w = (int(math.floor(x0)), int(math.ceil(x1)), int(math.floor(y0)), int(math.ceil(y1)))
    t0 = time.perf_counter()
    d = gef.read_window(w, b, with_gene_ids=True)
    if sel.kind == "polygon" and d["x"].size:
        m = sel.contains(d["x"], d["y"])
        d = {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    names = gef.gene_index(b)[0]
    region = np.zeros(len(names), np.float64)
    if "gene_id" in d and d["x"].size:
        np.add.at(region, d["gene_id"], d["count"].astype(np.float64))
    # background from a mid-coarse level: coarsest levels can saturate narrow count dtypes
    b_bg = background_bin or next((lv for lv in sorted(gef.bins) if lv >= 50), gef.bins[-1])
    bg = _background_mid(gef, b_bg)
    names_bg = gef.gene_index(b_bg)[0]
    if list(names_bg) != list(names):                       # align if orders differ
        pos = {n: i for i, n in enumerate(names_bg)}
        bg = np.array([bg[pos[n]] if n in pos else 0.0 for n in names])
    r_tot, b_tot = max(region.sum(), 1.0), max(bg.sum(), 1.0)
    # Compare each gene with the count *expected* in this region if it had the section's
    # composition. A pseudocount of one molecule on both sides bounds the fold change sensibly:
    # the earlier CPM-with-epsilon form reported -16.8 for genes simply absent from the region,
    # which was the pseudocount talking. `z` says how surprising the gap is for its size.
    expected = bg / b_tot * r_tot
    lfc = np.log2((region + 1.0) / (expected + 1.0))
    z = (region - expected) / np.sqrt(expected + 1.0)

    def row(i):
        return {"gene": str(names[i]), "log2fc": round(float(lfc[i]), 3),
                "observed": int(region[i]), "expected": round(float(expected[i]), 1),
                "z": round(float(z[i]), 2), "section_mid": int(bg[i])}

    keep = region >= min_count
    order = np.argsort(-np.where(keep, lfc, -np.inf))
    enriched = [row(i) for i in order[:top] if keep[i] and lfc[i] > 0]
    dep_ok = expected >= min_count
    dep = np.argsort(np.where(dep_ok, z, np.inf))
    depleted = [row(i) for i in dep[:top] if dep_ok[i] and lfc[i] < 0]
    ex = gef.extent
    frac = sel.area / max((ex[1] - ex[0]) * (ex[3] - ex[2]), 1)
    return {"observations": int(d["x"].size), "mid": int(region.sum()), "bin": b,
            "area_fraction": round(frac, 6),
            "mid_share": round(float(region.sum() / b_tot), 6),
            "density_vs_section": round(float((region.sum() / b_tot) / max(frac, 1e-12)), 3),
            "genes_detected": int((region > 0).sum()), "enriched": enriched,
            "depleted": depleted, "background_bin": b_bg,
            "read_path": gef.read_plan(w, b).get("path"),
            "seconds": round(time.perf_counter() - t0, 3)}


def _describe_interval(sel, view) -> dict:
    lo, hi = sel.bounds
    ov = view.select(*view._full_window()).overview(512)
    x = np.asarray(ov.x, float)
    inside = (x >= lo) & (x < hi)
    out = {"units": view.extent.kind, "span": [lo, hi],
           "fraction": round(float((hi - lo) / max(view.extent.n, 1)), 6), "series": {}}
    for k, v in ov.series.items():
        a = np.asarray(v, float)
        if a.ndim != 1 or a.size != x.size:
            continue
        ins, al = a[inside & np.isfinite(a)], a[np.isfinite(a)]
        if ins.size and al.size:
            out["series"][k] = {"inside": round(float(ins.mean()), 6),
                                "overall": round(float(al.mean()), 6),
                                "z": round(float((ins.mean() - al.mean()) / (al.std() or 1)), 3)}
    return out


def _describe_column_range(sel) -> dict:
    import polars as pl
    g = sel.geometry
    path = sel.dataset["path"]
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    col = g["column"]
    tcol = pl.col(col)
    if schema[col].is_temporal():
        tcol = pl.col(col).dt.epoch("s").cast(pl.Float64)
    nums = [c for c, t in schema.items() if t.is_numeric() and c != col]
    inside = lf.filter((tcol >= g["lo"]) & (tcol < g["hi"]))
    a = inside.select([pl.len().alias("_n")] + [pl.col(c).mean().alias(c) for c in nums]).collect()
    b = lf.select([pl.len().alias("_n")] + [pl.col(c).mean().alias(c) for c in nums]
                  + [pl.col(c).std().alias(c + "__sd") for c in nums]).collect()
    out = {"rows": int(a["_n"][0]), "rows_total": int(b["_n"][0]), "column": col, "series": {}}
    for c in nums:
        ins, al, sd = a[c][0], b[c][0], b[c + "__sd"][0] or 1.0
        if ins is not None and al is not None:
            out["series"][c] = {"inside": round(float(ins), 6), "overall": round(float(al), 6),
                                "z": round(float((ins - al) / sd), 3)}
    return out


# ---------------------------------------------------------------------------
# 2. similar — where else in this dataset looks like it
# ---------------------------------------------------------------------------

def similar(sel: Selection, view=None, *, k: int = 6, profile_bin: Optional[int] = None,
            max_genes: int = 2000, overlap: float = 0.3, stride: float = 0.5,
            book: Optional[SelectionBook] = None) -> list:
    """Regions resembling `sel` elsewhere in the same dataset, best first, as candidates.

    Candidates overlapping the selection or each other by more than `overlap` are suppressed,
    so the list is k distinct places rather than k shifts of one.
    """
    from .journal import _warn_once, active, quiet
    t0 = time.perf_counter()
    with quiet():                              # the candidate scan is one event, recorded below
        if sel.kind in ("rect", "polygon") and view is not None and hasattr(view, "gef"):
            found = _similar_gef(sel, view, k, profile_bin, max_genes, overlap, stride)
        elif sel.kind == "column-range":
            found = _similar_time(sel, k, overlap, stride)
        elif sel.kind == "interval" and view is not None:
            found = _similar_interval(sel, view, k, overlap, stride)
        else:
            raise TypeError(f"no similarity search for a {sel.kind} selection here")
    for c in found:
        c.parent = sel.id
        if book is not None:
            book.add(c)
    j = active()
    if j is not None:
        try:
            j.record("similar", sel.dataset, window=list(sel.bounds), level=dict(sel.level),
                     unit=getattr(getattr(view, "extent", None), "kind", "") if view is not None else "",
                     genes=sel.genes, units=len(found), seconds=time.perf_counter() - t0,
                     selection=sel.id, via="similar",
                     tags=[f"candidate={c.id}" for c in found])
        except Exception as err:
            _warn_once(f"journal: {err!r}")
    return found


def _candidate(sel, kind, geom, score, name, level) -> Selection:
    return Selection(id=uuid.uuid4().hex[:12], name=name, dataset=sel.dataset, kind=kind,
                     geometry=geom, level=level, intent="candidate", score=float(score),
                     parent=sel.id, genes=sel.genes)


def _nms(cands: list, sel: Selection, k: int, overlap: float) -> list:
    out = []
    for c in cands:
        if c.overlaps(sel) > overlap:
            continue
        if any(c.overlaps(o) > overlap for o in out):
            continue
        out.append(c)
        if len(out) >= k:
            break
    return out


def _similar_gef(sel, view, k, profile_bin, max_genes, overlap, stride) -> list:
    gef = view.gef
    ex = gef.extent
    x0, x1, y0, y1 = sel.bounds
    tw, th = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    # one coarse level carries every tile's profile; pick one at least ~4 bins per tile side
    pb = profile_bin or max([b for b in gef.bins if b <= min(tw, th) / 4] or [gef.bins[0]])
    d = gef.read_window(ex, pb, with_gene_ids=True)
    names = gef.gene_index(pb)[0]
    gid = d["gene_id"]; cnt = d["count"].astype(np.float64)
    # keep the most abundant genes so the matrix stays bounded on a 25k-gene chip
    tot = np.bincount(gid, weights=cnt, minlength=len(names))
    keep_genes = np.argsort(-tot)[:max_genes]
    remap = -np.ones(len(names), np.int64); remap[keep_genes] = np.arange(keep_genes.size)
    gk = remap[gid]; m = gk >= 0
    xs, ys, gk, cnt = d["x"][m], d["y"][m], gk[m], cnt[m]
    sx, sy = tw * stride, th * stride
    nx = max(int(math.ceil((ex[1] - ex[0] - tw) / sx)) + 1, 1)
    ny = max(int(math.ceil((ex[3] - ex[2] - th) / sy)) + 1, 1)
    # each observation falls in up to (1/stride)^2 overlapping tiles
    reps = int(round(1 / stride))
    G = keep_genes.size
    prof = np.zeros((nx * ny, G), np.float32)
    for ox in range(reps):
        for oy in range(reps):
            ix = np.floor((xs - ex[0]) / sx).astype(np.int64) - ox
            iy = np.floor((ys - ex[2]) / sy).astype(np.int64) - oy
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            np.add.at(prof, (iy[ok] * nx + ix[ok], gk[ok]), cnt[ok])
    # the selection's own profile, from the same level, same genes
    inside = sel.contains(xs, ys)
    q = np.bincount(gk[inside], weights=cnt[inside], minlength=G).astype(np.float32)
    def norm(a):
        a = np.log1p(a / np.maximum(a.sum(-1, keepdims=True), 1) * 1e4)
        return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
    live = prof.sum(1) > 0
    score = np.full(prof.shape[0], -1.0)
    score[live] = norm(prof[live]) @ norm(q[None, :])[0]
    order = np.argsort(-score)
    cands = []
    for t in order[: max(k * 20, 60)]:
        if score[t] <= 0:
            break
        iy, ix = divmod(int(t), nx)
        gx0 = ex[0] + ix * sx; gy0 = ex[2] + iy * sy
        cands.append(_candidate(sel, "rect", {"x0": gx0, "x1": gx0 + tw, "y0": gy0, "y1": gy0 + th},
                                score[t], f"{sel.name} ~{len(cands) + 1}", {"bin": pb}))
    out = _nms(cands, sel, k, overlap)
    for i, c in enumerate(out):
        c.name = f"{sel.name} ~{i + 1}"
    return out


def _similar_interval(sel, view, k, overlap, stride) -> list:
    lo, hi = sel.bounds
    ov = view.select(*view._full_window()).overview(1024)
    x = np.asarray(ov.x, float)
    series = [np.asarray(v, float) for v in ov.series.values()
              if np.asarray(v).ndim == 1 and np.asarray(v).size == x.size]
    if not series:
        return []
    F = np.stack([(s - np.nanmean(s)) / (np.nanstd(s) or 1) for s in series], 1)
    F = np.nan_to_num(F)
    w = max(int(((x >= lo) & (x < hi)).sum()), 1)
    qi = np.flatnonzero((x >= lo) & (x < hi))
    q = F[qi].mean(0) if qi.size else F[:w].mean(0)
    step = max(int(w * stride), 1)
    cands = []
    for s in range(0, max(len(x) - w, 1), step):
        f = F[s:s + w].mean(0)
        dist = float(np.linalg.norm(f - q))
        c0, c1 = float(x[s]), float(x[min(s + w, len(x) - 1)])
        cands.append(_candidate(sel, "interval", {"lo": c0, "hi": c1}, 1 / (1 + dist),
                                f"{sel.name} ~", sel.level))
    cands.sort(key=lambda c: -c.score)
    out = _nms(cands, sel, k, overlap)
    for i, c in enumerate(out):
        c.name = f"{sel.name} ~{i + 1}"
    return out


def _similar_time(sel, k, overlap, stride) -> list:
    """Windowed features of every numeric column via Polars' group_by_dynamic, compared with the
    selection's own window by cosine on z-scored features."""
    import polars as pl
    g = sel.geometry
    path, col = sel.dataset["path"], g["column"]
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    nums = [c for c, t in schema.items() if t.is_numeric() and c != col]
    span = g["hi"] - g["lo"]
    temporal = schema[col].is_temporal()
    idx = pl.col(col) if temporal else pl.col(col).cast(pl.Int64)
    every = max(int(span * stride), 1)
    feats_exprs = []
    for c in nums:
        feats_exprs += [pl.col(c).mean().alias(f"{c}__mean"), pl.col(c).std().alias(f"{c}__sd"),
                        pl.col(c).max().alias(f"{c}__max"), pl.col(c).min().alias(f"{c}__min")]
    base = lf.with_columns(idx.alias("_t")).sort("_t")
    if temporal:
        win = base.group_by_dynamic("_t", every=f"{every}s", period=f"{int(span)}s").agg(feats_exprs)
    else:
        win = base.group_by_dynamic("_t", every=f"{every}i", period=f"{int(span)}i").agg(feats_exprs)
    W = win.collect()
    qdf = base.filter((pl.col("_t").cast(pl.Float64) if not temporal else
                       pl.col("_t").dt.epoch("s").cast(pl.Float64)) >= g["lo"]) \
              .filter((pl.col("_t").cast(pl.Float64) if not temporal else
                       pl.col("_t").dt.epoch("s").cast(pl.Float64)) < g["hi"]) \
              .select(feats_exprs).collect()
    fcols = [c for c in W.columns if "__" in c]
    M = W.select(fcols).to_numpy().astype(float)
    q = qdf.select(fcols).to_numpy().astype(float)[0]
    mu, sd = np.nanmean(M, 0), np.nanstd(M, 0)
    sd[sd == 0] = 1
    Z = np.nan_to_num((M - mu) / sd); qz = np.nan_to_num((q - mu) / sd)
    sim = (Z @ qz) / (np.linalg.norm(Z, axis=1) * (np.linalg.norm(qz) or 1) + 1e-9)
    starts = W["_t"]
    starts = (starts.dt.epoch("s").cast(pl.Float64) if temporal else starts.cast(pl.Float64)).to_numpy()
    cands = [_candidate(sel, "column-range", {"column": col, "lo": float(s), "hi": float(s + span)},
                        sc, f"{sel.name} ~", {}) for s, sc in zip(starts, sim) if np.isfinite(sc)]
    cands.sort(key=lambda c: -c.score)
    out = _nms(cands, sel, k, overlap)
    for i, c in enumerate(out):
        c.name = f"{sel.name} ~{i + 1}"
    return out


# ---------------------------------------------------------------------------
# 3. leads — data you do not have yet
# ---------------------------------------------------------------------------

def _genes_of(path: str) -> Optional[list]:
    """Cheap gene lists for the formats GEMINGA reads: GEF index, 10x features, h5ad var."""
    low = path.lower()
    try:
        if low.endswith(".gef"):
            from .stereoseq import GefFile
            gf = GefFile(path); out = [str(x) for x in gf.genes(gf.bins[0])]; gf.close(); return out
        if low.endswith(".h5") and not low.endswith(".h5ad"):
            from .tenx import TenXMatrix
            m = TenXMatrix(path); out = [str(x) for x in m.feature_names]; m.close(); return out
        if low.endswith(".h5ad"):
            import h5py
            with h5py.File(path, "r") as f:
                var = f["var"]
                idx = var.attrs.get("_index", "_index")
                idx = idx.decode() if isinstance(idx, bytes) else idx
                node = var.get(idx)
                if node is not None and hasattr(node, "shape"):
                    return [x.decode() if isinstance(x, bytes) else str(x) for x in node[:]]
    except Exception:
        return None
    return None


def local_leads(genes: Sequence[str], candidates: Iterable[str], *, exclude: Iterable[str] = ()
                ) -> list:
    """Which datasets you already hold measured these genes, ranked by coverage."""
    want = {g.upper() for g in genes}
    skip = {os.path.abspath(p) for p in exclude if isinstance(p, str) and os.path.exists(p)}
    out = []
    for p in candidates:
        if not isinstance(p, str) or not os.path.exists(p) or os.path.abspath(p) in skip:
            continue
        gl = _genes_of(p)
        if not gl:
            continue
        have = want & {x.upper() for x in gl}
        if have:
            out.append({"kind": "local", "source": "your data", "title": os.path.basename(p),
                        "url": p, "accession": None, "coverage": round(len(have) / len(want), 3),
                        "genes_present": sorted(have)})
    return sorted(out, key=lambda r: -r["coverage"])


def external_queries(genes: Sequence[str], context: Sequence[str] = (), organism: Optional[str] = None,
                     assay: Optional[str] = None, max_genes: int = 5) -> list:
    """Search requests against public repositories, built from the selection — not run.

    GEO and PubMed use NCBI E-utilities; ENA uses its portal search API. Each is a plain GET that
    any HTTP client can issue, so PRISM or a browser can run them as well as GEMINGA can.
    """
    from urllib.parse import urlencode
    gs = [g for g in genes][:max_genes]
    gene_q = " OR ".join(f"{g}[All Fields]" for g in gs)
    parts = [f"({gene_q})"] if gs else []
    for c in context:
        parts.append(f'"{c}"[All Fields]')
    if organism:
        parts.append(f'"{organism}"[Organism]')
    if assay:
        parts.append(f'"{assay}"[All Fields]')
    term = " AND ".join(parts) or "spatial transcriptomics"
    eu = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
    q = [
        {"source": "GEO", "db": "gds",
         "url": eu + urlencode({"db": "gds", "term": term + ' AND "gse"[Entry Type]',
                                "retmode": "json", "retmax": 20}),
         "term": term},
        {"source": "PubMed", "db": "pubmed",
         "url": eu + urlencode({"db": "pubmed", "term": term, "retmode": "json", "retmax": 20}),
         "term": term},
    ]
    words = [c for c in context] + ([assay] if assay else [])
    if words:
        ena_q = " AND ".join(f'study_title="*{w}*"' for w in words[:3])
        q.append({"source": "ENA", "db": "read_study",
                  "url": "https://www.ebi.ac.uk/ena/portal/api/search?" + urlencode(
                      {"result": "read_study", "query": ena_q,
                       "fields": "study_accession,study_title", "format": "json", "limit": 20}),
                  "term": ena_q})
    return q


def run_query(q: dict, timeout: float = 20.0) -> list:
    """Execute one external query and normalise the answer into leads. Network required."""
    import urllib.request
    req = urllib.request.Request(q["url"], headers={"User-Agent": "geminga/0.13"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8", "replace"))
    return parse_query_result(q, payload, timeout)


def parse_query_result(q: dict, payload, timeout: float = 20.0, fetch_summaries: bool = True) -> list:
    """Turn a repository's JSON into leads. Split out so it is testable without a network."""
    src = q["source"]
    if src in ("GEO", "PubMed"):
        ids = payload.get("esearchresult", {}).get("idlist", [])
        leads = [{"kind": "external", "source": src, "id": i, "title": None,
                  "accession": None, "url": None} for i in ids]
        if fetch_summaries and ids:
            try:
                import urllib.request
                from urllib.parse import urlencode
                su = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
                      + urlencode({"db": q["db"], "id": ",".join(ids), "retmode": "json"}))
                with urllib.request.urlopen(su, timeout=timeout) as r:
                    summ = json.loads(r.read().decode("utf-8", "replace")).get("result", {})
                for l in leads:
                    doc = summ.get(l["id"], {})
                    l["title"] = doc.get("title")
                    acc = doc.get("accession") or (f"PMID:{l['id']}" if src == "PubMed" else None)
                    l["accession"] = acc
                    l["url"] = (f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}"
                                if src == "GEO" and acc else
                                f"https://pubmed.ncbi.nlm.nih.gov/{l['id']}/" if src == "PubMed"
                                else None)
            except Exception:
                pass
        return leads
    if src == "ENA":
        rows = payload if isinstance(payload, list) else []
        return [{"kind": "external", "source": "ENA", "accession": r.get("study_accession"),
                 "title": r.get("study_title"),
                 "url": f"https://www.ebi.ac.uk/ena/browser/view/{r.get('study_accession')}"}
                for r in rows]
    return []


def leads(sel: Selection, *, context: Sequence[str] = (), organism: Optional[str] = None,
          assay: Optional[str] = None, local: Iterable[str] = (), online: bool = False,
          book: Optional[SelectionBook] = None, max_genes: int = 5) -> list:
    """Where more data on this region might be: in files you have, and in public repositories.

    Uses the selection's enriched genes (run `describe()` first) plus your context terms. With
    `online=False` the external part is a set of ready-to-run queries; with `online=True` they are
    executed and any accessions found are attached, ready for `acquisition_plan()`.
    """
    genes = [g["gene"] for g in sel.summary.get("enriched", [])][:max_genes] or list(sel.genes or [])
    local = [p for p in local if isinstance(p, str)]
    found = local_leads(genes, local, exclude=[sel.dataset.get("path")]) if genes else []
    checked = [p for p in local if os.path.exists(p)
               and os.path.abspath(p) != os.path.abspath(str(sel.dataset.get("path")))]
    # An empty local result is an answer, not a silence: record what was looked at.
    sel.summary["leads_checked"] = {"genes": genes, "local_checked": len(checked),
                                    "local_matched": len(found),
                                    "local_paths": [os.path.basename(p) for p in checked]}
    for q in external_queries(genes, context, organism, assay, max_genes):
        entry = {"kind": "query", "source": q["source"], "title": q["term"], "url": q["url"],
                 "accession": None}
        if online:
            try:
                got = run_query(q)
                found.extend(got)
                entry["result_count"] = len(got)
            except Exception as e:
                entry["error"] = f"{type(e).__name__}: {e}"
        found.append(entry)
    sel.leads = found
    if book is not None:
        sel.status = book._advance(sel, "searched", leads=found, summary=sel.summary).status
    else:
        sel.status = _forward(sel.status, "searched")
    return found
