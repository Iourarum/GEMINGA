"""Polars as GEMINGA's dataframe layer.

Arrow is already the currency here — every tabular reader hands back `pyarrow` — and Polars is
built on Arrow with a Rust engine, so the conversion is a pointer move rather than a copy. That
makes it the natural dataframe for this library: same memory model, same language underneath,
no serialisation boundary to pay for.

What Polars adds that GEMINGA does not have, and should not build:

* **A query engine.** Group-bys, joins, window functions and expressions over a staged region.
  Writing those again would be absurd.
* **`scan_parquet` with pushdown and streaming.** Polars will push a predicate into the Parquet
  reader and execute out of core.

That last point overlaps with GEMINGA's own Parquet path, and the honest division is worth
stating because it decides which one to reach for:

* **GEMINGA `prune()`** answers *what would this cost* from footers alone, before fetching, and
  works over HTTP range requests against a remote file. Use it to decide.
* **Polars `scan_parquet`** executes the query well once you have decided, locally or on a
  store it can reach. Use it to compute.

They compose: prune to learn the selectivity, then hand the surviving row groups to Polars.
`lazy_window()` below does exactly that for coordinate-windowed data.

Polars is an optional dependency. Nothing in the core imports it; every function here raises a
clear message if it is missing.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import numpy as np


__all__ = ["to_polars", "from_polars", "narrow_arrow", "scan", "lazy_window",
           "frame_bytes", "sink_parquet", "POLARS_AVAILABLE"]

try:                                    # probed once, never required
    import polars as pl
    POLARS_AVAILABLE = True
except Exception:                       # pragma: no cover
    pl = None
    POLARS_AVAILABLE = False


def _need():
    if not POLARS_AVAILABLE:
        raise ImportError(
            "this needs Polars: pip install polars. GEMINGA's core does not require it — "
            "every reader still returns pyarrow, and .load() works without it.")
    return pl


def to_polars(obj, *, rechunk: bool = False):
    """Anything a GEMINGA reader returns -> `polars.DataFrame`.

    Arrow tables and batches convert without copying buffers. Dicts of arrays (Stereo-seq
    windows, GEM rows) and scipy sparse matrices (h5ad, 10x blocks) are built in COO form,
    which is the shape a dataframe can actually hold — a sparse matrix is not a table until
    someone decides what its rows mean.
    """
    p = _need()
    if obj is None:
        return p.DataFrame()
    if isinstance(obj, p.DataFrame):
        return obj
    if isinstance(obj, p.LazyFrame):
        return obj.collect()
    if hasattr(obj, "num_rows") or hasattr(obj, "schema"):          # pyarrow Table / Batch
        import pyarrow as pa
        tbl = obj if isinstance(obj, pa.Table) else pa.Table.from_batches([obj])
        return p.from_arrow(tbl, rechunk=rechunk)
    if isinstance(obj, dict):
        cols = {k: np.asarray(v) for k, v in obj.items()
                if isinstance(v, (np.ndarray, list)) and np.asarray(v).ndim == 1}
        if not cols:
            raise TypeError("no 1-D columns in this mapping")
        n = len(next(iter(cols.values())))
        cols = {k: v for k, v in cols.items() if len(v) == n}
        return p.DataFrame(cols)
    if hasattr(obj, "tocoo"):                                        # scipy sparse
        coo = obj.tocoo()
        return p.DataFrame({"row": coo.row.astype(np.int32),
                            "col": coo.col.astype(np.int32),
                            "value": coo.data.astype(np.float32)})
    if isinstance(obj, np.ndarray):
        if obj.ndim == 1:
            return p.DataFrame({"value": obj})
        if obj.ndim == 2:
            return p.DataFrame(obj, schema=[f"c{i}" for i in range(obj.shape[1])])
        raise TypeError(f"a {obj.ndim}-D array is not a table; keep it as numpy")
    raise TypeError(f"cannot make a DataFrame from {type(obj).__name__}")


def from_polars(df, *, to: str = "arrow", compat: bool = True):
    """`polars.DataFrame` -> pyarrow (default) or a dict of numpy arrays.

    Polars represents strings, binary and lists in their 64-bit Arrow forms — `large_string`
    rather than `string`. That is valid Arrow and the data is identical, but a round trip
    through Polars would otherwise change the schema of anything GEMINGA had already written,
    and a `Harvest` region whose type widened between one run and the next is a real problem
    for the R and Ruby readers on the other end of the manifest. `compat=True` narrows those
    types back on the way out; pass `compat=False` to keep Polars' native widths.
    """
    _need()
    if to == "numpy":
        return {c: df[c].to_numpy() for c in df.columns}
    if to != "arrow":
        raise ValueError(f"unknown target {to!r}; use 'arrow' or 'numpy'")
    tbl = df.to_arrow()
    return narrow_arrow(tbl) if compat else tbl


def narrow_arrow(tbl):
    """Cast 64-bit Arrow variants back to their 32-bit forms where the data allows it."""
    import pyarrow as pa
    narrow = {pa.large_string(): pa.string(), pa.large_binary(): pa.binary()}
    fields = []
    changed = False
    for f in tbl.schema:
        t = narrow.get(f.type)
        if t is None and pa.types.is_large_list(f.type):
            t = pa.list_(f.type.value_type)
        if t is not None:
            fields.append(pa.field(f.name, t, f.nullable)); changed = True
        else:
            fields.append(f)
    if not changed:
        return tbl
    try:
        return tbl.cast(pa.schema(fields))
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return tbl              # a column genuinely needs 64-bit offsets; leave it alone


def frame_bytes(df) -> int:
    """Resident size of a frame, for budget accounting."""
    _need()
    try:
        return int(df.estimated_size())
    except Exception:
        return int(sum(s.estimated_size() for s in df.get_columns()))


def scan(source, *, columns: Optional[Sequence[str]] = None, **kw):
    """A `polars.LazyFrame` over Parquet — a path, a glob, a list, or a GEMINGA object.

    Nothing is read until `.collect()`. Pass `columns=` to project up front; Polars pushes
    both projection and any later `.filter()` down into the Parquet reader.
    """
    p = _need()
    paths = _paths_of(source)
    lf = p.scan_parquet(paths, **kw)
    return lf.select(columns) if columns else lf


def _paths_of(source) -> Any:
    if isinstance(source, (str, os.PathLike)):
        return str(source)
    if isinstance(source, (list, tuple)):
        return [str(s) for s in source]
    for attr in ("transcripts", "path"):                 # XeniumBundle, StagedRegion, ...
        v = getattr(source, attr, None)
        if isinstance(v, str) and v.endswith(".parquet"):
            return v
    files = getattr(source, "files", None)                # GEMINGA Dataset
    if files:
        return list(files)
    raise TypeError(f"no Parquet path behind {type(source).__name__}")


def lazy_window(source, x: str, x0: float, x1: float, y: Optional[str] = None,
                y0: Optional[float] = None, y1: Optional[float] = None,
                columns: Optional[Sequence[str]] = None, *, explain: bool = False):
    """Prune with GEMINGA, then execute with Polars — the two halves in their right order.

    GEMINGA's `prune()` reports, from footer statistics alone, which row groups could contain
    the window and what fetching them would cost. Polars then runs the exact predicate with its
    own pushdown. Returns a LazyFrame, or `(LazyFrame, plan)` with `explain=True`, where `plan`
    carries the selectivity so you know what you avoided before collecting.

    The window is applied first and `columns` selects afterwards, so the window may be on
    columns that are not in the projection (Polars still reads only what both need). With a
    `y` bound the plan covers x *and* y: `prune()` is asked about each, the row groups that
    survive both are kept (`row_groups_per_file`, `row_groups_kept`, `candidate_rows` are
    exact for that intersection), `bytes_to_fetch` is the x plan's scaled by the compressed
    size of the groups that remain (`bytes_to_fetch_estimated`), and the two single-column plans
    are under `"x"` and `"y"`.
    """
    p = _need()
    lf = scan(source)
    pred = (p.col(x) >= x0) & (p.col(x) < x1)
    if y is not None and y0 is not None and y1 is not None:
        pred = pred & (p.col(y) >= y0) & (p.col(y) < y1)
    lf = lf.filter(pred)                       # filter first: x and y need not be projected
    if columns:
        lf = lf.select(list(columns))
    if not explain:
        return lf
    plan = None
    try:
        import geminga as g
        ds = source if hasattr(source, "prune") else g.open_paths(
            _paths_of(source) if isinstance(_paths_of(source), list) else [_paths_of(source)])
        has_y = y is not None and y0 is not None and y1 is not None
        fetch = None
        if columns:                            # the predicate columns are fetched too
            fetch = list(columns) + [c for c in (x, y if has_y else None) if c and c not in columns]
        plan = ds.prune(x, x0, x1, fetch)
        if has_y:
            plan = _both_bounds(ds, plan, ds.prune(y, y0, y1, fetch))
    except Exception as e:
        plan = {"error": f"{type(e).__name__}: {e}"}
    return lf, plan


def _both_bounds(ds, px: dict, py: dict) -> dict:
    """One plan for 'x and y': the row groups both single-column plans keep."""
    from ._core import human
    files = ds.describe()["files"]
    per_file, rows, comp_kept, comp_x = [], 0, 0, 0
    for fi, (a, b) in enumerate(zip(px["row_groups_per_file"], py["row_groups_per_file"])):
        rgs = files[fi]["row_groups"]
        keep = sorted(set(a) & set(b))
        per_file.append(keep)
        rows += sum(rgs[i]["num_rows"] for i in keep)
        comp_kept += sum(rgs[i]["compressed_bytes"] for i in keep)
        comp_x += sum(rgs[i]["compressed_bytes"] for i in a)
    fetch = int(px["bytes_to_fetch"] * comp_kept / comp_x) if comp_x else 0
    out = dict(px)
    out.update(row_groups_per_file=per_file, row_groups_kept=sum(len(k) for k in per_file),
               candidate_rows=rows, bytes_to_fetch=fetch, bytes_to_fetch_human=human(fetch),
               selectivity=(fetch / px["bytes_total"]) if px["bytes_total"] else 1.0,
               bytes_to_fetch_estimated=True,
               kept_without_stats=max(px["kept_without_stats"], py["kept_without_stats"]),
               x=px, y=py)
    return out


def sink_parquet(lf, path: str, *, compression: str = "zstd", **kw) -> str:
    """Stream a LazyFrame to Parquet without materialising it.

    This is the write side of the same promise the readers make: a result larger than memory
    still lands, because Polars executes the plan in batches.
    """
    _need()
    try:
        lf.sink_parquet(path, compression=compression, **kw)
    except Exception:
        # Older Polars, or a plan the streaming engine refuses: fall back honestly.
        lf.collect().write_parquet(path, compression=compression)
    return path


# ---------------------------------------------------------------------------
# Methods grafted onto the existing classes, so `.polars()` is available where
# `.load()` already is. Done here rather than in each module so that Polars stays
# genuinely optional.
# ---------------------------------------------------------------------------

def _install():
    from .view import View
    from .handoff import Harvest
    from .catalog import Catalog

    def view_polars(self, budget_check: bool = True):
        """The current window as a `polars.DataFrame`, budget-enforced like `load()`."""
        data = self.load() if budget_check else self._load(self.window, self.stride)
        return to_polars(data)

    def view_lazy(self, **kw):
        """A LazyFrame over this view's *source*, for views backed by Parquet."""
        return scan(getattr(self, "src", self), **kw)

    View.polars = view_polars
    View.lazy = view_lazy

    def harvest_polars(self, name: str):
        return to_polars(self.read(name))

    Harvest.polars = harvest_polars

    def catalog_polars(self):
        """The append-only log as a DataFrame — Polars reads NDJSON natively."""
        p = _need()
        if not os.path.exists(self.path):
            return p.DataFrame()
        return p.read_ndjson(self.path, infer_schema_length=None)

    Catalog.polars = catalog_polars

    try:
        from .spatial import XeniumBundle

        def xenium_lazy(self, columns: Optional[Sequence[str]] = None):
            """A LazyFrame over transcripts.parquet — per-molecule, pushdown-capable."""
            return scan(self.transcripts, columns=columns)

        def xenium_lazy_window(self, x0, x1, y0, y1, columns=None, explain=False):
            return lazy_window(self.transcripts, "x_location", x0, x1,
                               "y_location", y0, y1, columns, explain=explain)

        XeniumBundle.lazy = xenium_lazy
        XeniumBundle.lazy_window = xenium_lazy_window
    except Exception:
        pass


if POLARS_AVAILABLE:
    try:
        _install()
    except Exception:                                   # pragma: no cover
        pass
