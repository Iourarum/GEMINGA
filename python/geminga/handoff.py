"""Getting regions out of GEMINGA and into the tools that do the analysis.

GEMINGA moves bytes. It does not fit models, render manifolds or run flux balance analysis, and
it should not start. What it owes the rest of a toolchain is a clean, boring handoff: regions
written in a form other languages can read without ceremony, plus a manifest describing what
each one is and where it came from.

The interchange is **Arrow IPC, Parquet or NDJSON plus a JSON manifest**, chosen because the
receiving end is polyglot. R reads Arrow and Parquet through the `arrow` package; Ruby through
`red-arrow`; Python natively. Perl has no first-class Arrow reader, so a Perl-side orchestrator
gets NDJSON or CSV, and — more usefully — gets the manifest, which is the part it actually needs
to dispatch work to the others.

A `Harvest` is a named set of regions with that manifest. The manifest is the contract: every
entry carries its source, its window, the bin or zoom level it was taken at, its row count, a
checksum, and a free-form `targets` list naming which downstream tools it is meant for. Nothing
here imports those tools or depends on them existing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

from ._core import human

__all__ = ["ROI", "Harvest", "harvest", "FORMATS", "KINDS", "ManifestError", "ColumnTypeError"]

#: Known entry kinds. Free strings are accepted; these are the ones consumers switch on.
#: `region` is GEMINGA's own cut; the rest are what other producers — HyVESTA first — write.
KINDS = ("region", "patch", "relation", "bundle", "ridge", "enrichment", "table", "piece", "piece-table")

#: What a receiving tool is likely to want, and what can read it.
FORMATS = {
    "arrow":   dict(suffix=".arrow",   readers="Python, R (arrow), Ruby (red-arrow), Julia"),
    "parquet": dict(suffix=".parquet", readers="Python, R (arrow), Ruby (red-arrow), DuckDB"),
    "ndjson":  dict(suffix=".ndjson",  readers="anything, including Perl, line by line"),
    "csv":     dict(suffix=".csv",     readers="anything"),
    "npy":     dict(suffix=".npy",     readers="Python, R (reticulate), Julia (NPZ)"),
}


class ManifestError(ValueError, RuntimeError):
    """A harvest's manifest.json exists but cannot be read as one. Both a bad-data error (`ValueError`,
    which is what a JSON syntax error already was) and a state error (`RuntimeError`): the harvest is
    not usable until someone looks at the file."""


class ColumnTypeError(TypeError, ValueError):
    """A value given as a table column that cannot be one (a scalar, an object Arrow cannot read)."""


_NAME_RX = re.compile(r"^[A-Za-z0-9._-]{1,200}$")


def _check_name(name) -> str:
    """A region's name becomes a file name inside the harvest, so it must be one: letters,
    digits, ``.``, ``_`` and ``-`` — no separators, no spaces, not made of dots alone."""
    if not isinstance(name, str) or not _NAME_RX.match(name) or not name.strip("."):
        raise ValueError(f"invalid region name {name!r}: it becomes a file inside the harvest root and must "
                         f"not reach outside it, so use letters, digits, '.', '_' and '-' only "
                         f"(no path separators, at most 200 characters)")
    return name


@dataclass
class ROI:
    """A region of interest: where it came from, and how it was cut."""
    name: str
    source: str
    window: Optional[tuple] = None
    zoom: int = 0
    bin_size: Optional[int] = None
    genes: Optional[list] = None
    columns: Optional[list] = None
    note: str = ""
    targets: list = field(default_factory=list)     # e.g. ["regulus", "amethrine"]
    kind: str = "region"                            # see KINDS
    stat: dict = field(default_factory=dict)        # a statistic, its null, p, q — free JSON
    # filled in when written
    path: str = ""
    format: str = ""
    rows: int = 0
    bytes: int = 0
    sha256: str = ""
    written: float = 0.0

    def __repr__(self) -> str:
        w = "x".join(str(v) for v in self.window) if self.window else "-"
        k = "" if self.kind == "region" else f" {self.kind}"
        return f"ROI({self.name}{k}, {w}, {self.rows:,} rows, {human(self.bytes)})"


class Harvest:
    """A named set of regions written for downstream tools.

        h = g.harvest("/scratch/handoff/chapter3", note="tumour margin survey")
        h.add(view, "margin-a", targets=["regulus"], fmt="parquet")
        h.add(view.select(...), "margin-b", targets=["amethrine", "prism"])
        h.write_manifest()
        h.print_table()
    """

    MANIFEST = "manifest.json"
    #: /2 adds `kind` and `stat` per entry and `kinds` at the top; a /1 reader ignores them.
    SCHEMA = "geminga.harvest/2"

    def __init__(self, path: str, note: str = "", producer: str = "geminga"):
        self.path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(self.path, exist_ok=True)
        self.note = note
        self.producer = producer
        self.rois: dict[str, ROI] = {}
        self.created = time.time()
        self._load()

    # ---- manifest ----
    @property
    def manifest_path(self) -> str:
        return os.path.join(self.path, self.MANIFEST)

    def _load(self):
        if not os.path.exists(self.manifest_path):
            return
        try:
            with open(self.manifest_path, encoding="utf-8") as fh:
                d = json.load(fh)
            if not isinstance(d, dict):
                raise ValueError("the top level is not a JSON object")
            entries = d.get("regions", [])
            if not isinstance(entries, list):
                raise ValueError("'regions' is not a list")
            rois = {}
            for r in entries:
                if not isinstance(r, dict) or "name" not in r:
                    raise ValueError(f"a region entry without a name: {str(r)[:80]}")
                r = dict(r)
                if r.get("window"):
                    r["window"] = tuple(r["window"])
                rois[r["name"]] = ROI(**{k: v for k, v in r.items() if k in ROI.__dataclass_fields__})
        except (ValueError, TypeError) as e:       # JSONDecodeError is a ValueError
            # Carrying on with an empty set would make the next write replace a damaged manifest
            # with one that has forgotten every region: stop and let the person look at it.
            raise ManifestError(f"{self.manifest_path}: damaged or corrupt harvest manifest "
                                f"({type(e).__name__}: {e}); fix or move it aside before writing to this harvest") from e
        self.note = self.note or d.get("note", "")
        self.created = d.get("created", self.created)
        self.rois.update(rois)

    def manifest(self) -> dict:
        return {
            "schema": self.SCHEMA,
            "producer": self.producer,
            "created": self.created,
            "updated": time.time(),
            "note": self.note,
            "root": self.path,
            "formats": FORMATS,
            "kinds": KINDS,
            "regions": [asdict(r) for r in self.rois.values()],
        }

    def write_manifest(self) -> str:
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.manifest(), fh, indent=2, default=str)
        os.replace(tmp, self.manifest_path)
        return self.manifest_path

    # ---- adding regions ----
    def _dest(self, name: str, suffix: str) -> str:
        """Where a region's file goes: `<root>/<name><suffix>`, and nowhere else. A name that is
        not a plain file name, or a file that is a symlink out of the root, is refused."""
        _check_name(name)
        dest = os.path.join(self.path, name + suffix)
        if os.path.dirname(os.path.realpath(dest)) != os.path.realpath(self.path):
            raise ValueError(f"region {name!r} would be written outside the harvest root "
                             f"({os.path.realpath(dest)}); {dest} is a link out of {self.path}")
        return dest

    def _precheck(self, name: str, fmt: str, overwrite: bool) -> None:
        _check_name(name)
        if fmt not in FORMATS:
            raise ValueError(f"unknown format {fmt!r}; pick from {list(FORMATS)}")
        if name in self.rois and not overwrite:
            raise ValueError(f"region {name!r} already exists; pass overwrite=True")
        self._dest(name, FORMATS[fmt]["suffix"])

    def add(self, view, name: str, fmt: str = "parquet", targets: Optional[Sequence[str]] = None,
            columns: Optional[Sequence[str]] = None, note: str = "",
            overwrite: bool = False, data=None, kind: str = "region",
            stat: Optional[dict] = None) -> ROI:
        """Write the view's current window out as a region.

        Pass `data=` to write something you already have in hand instead of re-loading. The
        scalar entries of a dict payload (a view's ``bin_size``) are metadata, not columns.
        """
        self._precheck(name, fmt, overwrite)

        roi = ROI(
            name=name,
            source=(getattr(getattr(view, "gef", None), "path", None)
                    or getattr(view, "path", None) or str(view)),
            window=tuple(view.window) if hasattr(view, "window") else None,
            zoom=int(getattr(view, "zoom", 0) or 0),
            bin_size=getattr(view, "bin_size", None) if hasattr(view, "gef") else None,
            genes=list(getattr(view, "genes", None) or []) or None,
            columns=list(columns) if columns else None,
            note=note, targets=list(targets or []), kind=kind, stat=dict(stat or {}),
        )
        if data is None:
            from .journal import recording as _recording
            with _recording(via="handoff", note=f"roi:{name}",
                            tags=[f"target={t}" for t in (targets or ())]):
                payload = view.load()
        else:
            payload = data
        return self._finish(roi, payload, fmt, columns)

    def add_table(self, name: str, table, fmt: str = "parquet", source: str = "",
                  targets: Optional[Sequence[str]] = None, kind: str = "table",
                  stat: Optional[dict] = None, note: str = "",
                  columns: Optional[Sequence[str]] = None, window: Optional[tuple] = None,
                  overwrite: bool = False) -> ROI:
        """Write a table you already hold — no GEMINGA view required.

        `table` is a pyarrow Table/RecordBatch, a dict of columns, or a scipy sparse matrix
        (written as COO, in the matrix's own precision, with its ``shape`` and ``dtype`` in the
        entry's `stat`). A dict column may be a numpy array, a list or tuple (nested and ragged
        lists become list columns), a pandas or polars Series, or a pyarrow Array/ChunkedArray;
        anything else — a scalar included — raises ``TypeError`` instead of vanishing. The
        entry gets the same checksum, manifest line and dispatch as a cut region, which is
        what lets another producer — HyVESTA — write through this class instead of imitating it.
        """
        self._precheck(name, fmt, overwrite)
        roi = ROI(name=name, source=source or self.producer, window=window,
                  columns=list(columns) if columns else None, note=note,
                  targets=list(targets or []), kind=kind, stat=dict(stat or {}))
        return self._finish(roi, table, fmt, columns, strict=True)

    def _finish(self, roi: ROI, payload, fmt: str, columns, strict: bool = False) -> ROI:
        dest = self._dest(roi.name, FORMATS[fmt]["suffix"])
        rows = _write(payload, dest, fmt, columns, strict=strict)
        if hasattr(payload, "tocoo"):                    # a COO table forgets its shape and dtype
            roi.stat["shape"] = [int(n) for n in payload.shape]
            roi.stat["dtype"] = str(payload.dtype)
        roi.path = os.path.relpath(dest, self.path)
        roi.format = fmt
        roi.rows = rows
        roi.bytes = os.path.getsize(dest)
        roi.sha256 = _sha256(dest)
        roi.written = time.time()
        self.rois[roi.name] = roi
        self.write_manifest()
        return roi

    def add_staged(self, staging, key: str, name: Optional[str] = None,
                   targets: Optional[Sequence[str]] = None, note: str = "",
                   overwrite: bool = False) -> ROI:
        """Promote something already extracted into staging, without re-reading the vault.

        The region is named `name`, or after the staging key with every character that is not
        a file-name character turned into ``-``. An existing region of that name is kept unless
        `overwrite=True`."""
        reg = staging.get(key)
        if reg is None:
            raise KeyError(f"no staged region {key!r}")
        if name is None:
            name = re.sub(r"[^A-Za-z0-9._-]", "-", key)[:200]
            if not name.strip("."):
                name = "staged-" + hashlib.blake2b(key.encode(), digest_size=4).hexdigest()
        _check_name(name)
        if name in self.rois and not overwrite:
            raise ValueError(f"region {name!r} already exists; pass overwrite=True")
        dest = self._dest(name, os.path.splitext(reg.path)[1])
        if os.path.abspath(dest) != os.path.abspath(reg.path):
            import shutil
            shutil.copy2(reg.path, dest)
        roi = ROI(name=name, source=reg.source, window=reg.window, note=note or reg.note,
                  targets=list(targets or []), path=os.path.relpath(dest, self.path),
                  format=reg.format, rows=reg.rows, bytes=os.path.getsize(dest),
                  sha256=_sha256(dest), written=time.time())
        self.rois[name] = roi
        self.write_manifest()
        return roi

    # ---- consuming ----
    def read(self, name: str):
        r = self.rois.get(name)
        if r is None:
            raise KeyError(f"no region {name!r}; have {list(self.rois)}")
        full = os.path.join(self.path, r.path)
        if r.format == "npy":
            import numpy as np
            return np.load(full)
        if r.format in ("parquet",):
            import pyarrow.parquet as pq
            return pq.read_table(full)
        if r.format == "arrow":
            import pyarrow as pa
            with pa.memory_map(full, "rb") as src:
                return pa.ipc.open_file(src).read_all()
        if r.format in ("ndjson", "csv"):
            import pyarrow.csv as pacsv
            import pyarrow.json as pajson
            return pajson.read_json(full) if r.format == "ndjson" else pacsv.read_csv(full)
        raise ValueError(f"cannot read format {r.format!r}")

    def for_target(self, target: str) -> list:
        return [r for r in self.rois.values() if target in r.targets]

    def for_kind(self, kind: str) -> list:
        """Entries of one kind — all the bundles, all the patches."""
        return [r for r in self.rois.values() if r.kind == kind]

    def significant(self, alpha: float = 0.05, key: str = "q") -> list:
        """Entries whose stat carries `key` (q by default, falling back to p) <= alpha."""
        out = []
        for r in self.rois.values():
            v = r.stat.get(key, r.stat.get("p"))
            if v is not None and float(v) <= alpha:
                out.append(r)
        return out

    def verify(self) -> dict:
        """Re-check every region's checksum. Cheap insurance before handing a set to someone."""
        out = {}
        for name, r in self.rois.items():
            full = os.path.join(self.path, r.path)
            out[name] = (os.path.exists(full) and _sha256(full) == r.sha256)
        return out

    # ---- Perl-side orchestration ----
    def write_dispatch(self, filename: str = "dispatch.json") -> str:
        """A flat, target-keyed view of the manifest for an orchestrator such as PRISM.

        Deliberately dumb: one object per target, each listing absolute paths, formats and row
        counts. A Perl driver can read this with any JSON module and shell out to the R, Ruby
        or Python tool that owns that target without understanding Arrow at all.
        """
        by_target: dict[str, list] = {}
        for r in self.rois.values():
            for t in (r.targets or ["unassigned"]):
                by_target.setdefault(t, []).append({
                    "name": r.name,
                    "path": os.path.join(self.path, r.path),
                    "format": r.format,
                    "readers": FORMATS.get(r.format, {}).get("readers", ""),
                    "rows": r.rows,
                    "bytes": r.bytes,
                    "sha256": r.sha256,
                    "source": r.source,
                    "window": list(r.window) if r.window else None,
                    "bin_size": r.bin_size,
                    "genes": r.genes,
                    "note": r.note,
                    "kind": r.kind,
                    "stat": r.stat,
                })
        dest = os.path.join(self.path, filename)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump({"schema": "geminga.dispatch/2", "producer": self.producer,
                       "root": self.path, "targets": by_target}, fh, indent=2, default=str)
        return dest

    # ---- reporting ----
    def print_table(self) -> None:
        print(f"harvest {self.path}" + (f"  —  {self.note}" if self.note else ""))
        if not self.rois:
            print("  (no regions)")
            return
        total = sum(r.bytes for r in self.rois.values())
        print(f"  {len(self.rois)} region(s), {human(total)}")
        print(f"  {'name':<22} {'kind':<10} {'format':<8} {'rows':>11} {'size':>9}  targets")
        for r in self.rois.values():
            print(f"  {r.name:<22} {r.kind:<10} {r.format:<8} {r.rows:>11,} "
                  f"{human(r.bytes):>9}  {', '.join(r.targets) or '-'}")

    def __len__(self) -> int:
        return len(self.rois)

    def __iter__(self):
        return iter(self.rois.values())

    def __repr__(self) -> str:
        return (f"Harvest({self.path}, {len(self.rois)} regions, "
                f"{human(sum(r.bytes for r in self.rois.values()))})")


# ---------------------------------------------------------------------------

_SCALARS = (str, bytes, int, float, bool, type(None))


def _cell(x):
    """One element of a list column: arrays become lists, everything else is left to Arrow."""
    import numpy as np
    return x.tolist() if isinstance(x, np.ndarray) else x


def _column(v):
    """One column -> pyarrow array. Nested lists (a centroid per row) become list columns, however
    ragged; a pandas/polars Series, a pyarrow Array or ChunkedArray and a numpy array are taken
    as they are."""
    import numpy as np
    import pyarrow as pa
    if isinstance(v, (pa.Array, pa.ChunkedArray)):
        return v
    if isinstance(v, (list, tuple)):
        return pa.array([_cell(x) for x in v])          # never through numpy: ragged is fine
    if isinstance(v, np.ndarray):
        if v.dtype == object or v.ndim > 1:
            return pa.array([_cell(x) for x in v])
        return pa.array(v)
    if hasattr(v, "to_arrow"):                          # polars Series
        return v.to_arrow()
    return pa.array(v)                                  # pandas Series/Index, anything Arrow can read


def _to_table(payload, columns=None, strict: bool = False):
    """A pyarrow Table out of a table, a record batch, a dict of columns or a sparse matrix.

    In a dict, scalars are not columns. With `strict` (a caller's own dict) a scalar is an error;
    without it (a view's payload, which carries metadata such as ``bin_size`` beside its arrays)
    scalars are left out. A value that is neither a scalar nor convertible is always an error
    (`ColumnTypeError`, a `TypeError` and a `ValueError`)."""
    import numpy as np
    import pyarrow as pa
    if hasattr(payload, "num_rows"):
        t = pa.Table.from_batches([payload]) if isinstance(payload, pa.RecordBatch) else payload
        return t.select(columns) if columns else t
    if isinstance(payload, dict):
        cols = {}
        for k, v in payload.items():
            if columns and k not in columns:
                continue
            if isinstance(v, _SCALARS) or isinstance(v, np.generic):
                if strict:
                    raise ColumnTypeError(f"unsupported column {k!r}: a scalar ({type(v).__name__}) is not a column; "
                                          f"give every column one value per row (repeat it, or put it in the "
                                          f"entry's stat)")
                continue
            try:
                cols[k] = _column(v)
            except Exception as e:
                raise ColumnTypeError(f"unsupported column {k!r}: cannot make an Arrow column out of a "
                                      f"{type(v).__name__} ({e})") from e
        return pa.table(cols)
    if hasattr(payload, "tocoo"):
        coo = payload.tocoo()
        idx = np.int32 if max(coo.shape) <= np.iinfo(np.int32).max else np.int64
        return pa.table({"row": coo.row.astype(idx, copy=False), "col": coo.col.astype(idx, copy=False),
                         "value": coo.data})
    raise TypeError(f"cannot tabulate a {type(payload).__name__}")


def _write(payload, dest: str, fmt: str, columns=None, strict: bool = False) -> int:
    import numpy as np
    if fmt == "npy":
        if hasattr(payload, "tocoo"):
            raise ValueError("a sparse matrix is written as a COO table: pick parquet, arrow, csv or ndjson, not 'npy'")
        arr = payload if isinstance(payload, np.ndarray) else np.asarray(payload)
        np.save(dest, arr)
        return int(arr.shape[0]) if arr.ndim else 1
    if isinstance(payload, np.ndarray) and payload.ndim > 1:
        raise ValueError(f"a {payload.ndim}-D array needs fmt='npy', not {fmt!r}")
    tbl = _to_table(payload, columns, strict)
    if fmt == "parquet":
        import pyarrow.parquet as pq
        pq.write_table(tbl, dest, compression="zstd")
    elif fmt == "arrow":
        import pyarrow as pa
        with pa.OSFile(dest, "wb") as sink:
            with pa.ipc.new_file(sink, tbl.schema) as w:
                for b in tbl.to_batches(max_chunksize=65536):
                    w.write_batch(b)
    elif fmt == "csv":
        import pyarrow.csv as pacsv
        pacsv.write_csv(tbl, dest)
    elif fmt == "ndjson":
        with open(dest, "w", encoding="utf-8") as fh:
            for batch in tbl.to_batches(max_chunksize=8192):
                for row in batch.to_pylist():
                    fh.write(json.dumps(row, default=str) + "\n")
    return tbl.num_rows


def _sha256(path: str, limit: int = 1 << 28) -> str:
    """Checksum the file, or its first 256 MB for very large ones (noted by the suffix)."""
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            n += len(chunk)
            if n >= limit:
                return h.hexdigest() + f"~first{limit >> 20}MB"
    return h.hexdigest()


def harvest(path: str, note: str = "", **kw) -> Harvest:
    """A named set of regions written for downstream tools, with a manifest."""
    return Harvest(path, note, **kw)
