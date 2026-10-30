"""Format readers that live in Python but obey the Rust budget.

Parquet and FASTA/FASTQ are handled by the Rust core. The formats here (h5ad, Arrow IPC,
imaging) lean on mature Python libraries — h5py, pyarrow, tifffile, zarr — because their
value is in the *plan* and the *budget*, not in re-implementing a decoder. Each reader:

  1. produces a `Plan` first (how many chunks, how many bytes the largest one is) without
     reading data,
  2. acquires from the budget before materializing a chunk,
  3. releases when the consumer asks for the next one.

That is the same contract the Rust readers follow, so `for chunk in reader` behaves
identically whether the bytes came from a Parquet row group, a CSR row block or an image tile.

Two rules every reader here shares:

* **The chunk floor rule.** With a budget attached and no chunk size chosen by the caller, a
  chunk is ``max(min(1 MB, free), free // 4)`` — a quarter of what is free, floored at 1 MB
  but never above what is free. "Not chosen" means the value equals the reader's default, so
  ``256 << 20`` and ``"256 MB"`` count as the default too. A chunk size the caller *did*
  choose is honoured, and if it exceeds what the budget has free the reader raises
  ``MemoryError`` at construction naming both numbers, before anything is read.
* **`plan()` names the largest chunk that will actually be handed out.** Sparse readers cut
  chunk boundaries from ``indptr``, so a skewed matrix is planned and streamed from its real
  per-row sizes, not from an average. ``Plan.bytes_per_chunk`` is that largest chunk, which
  makes ``Plan.fits(budget)`` honest; ``detail["largest_chunk_bytes"]`` repeats it and
  ``detail["fits"]`` says whether it is within ``chunk_bytes`` (and the budget, when attached).
"""
from __future__ import annotations

import contextlib
import itertools
import math
import operator
import os
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

import numpy as np

from ._core import Budget, human, parse_size


@dataclass
class Plan:
    """What streaming will cost, computed before anything is read."""
    kind: str
    source: str
    n_chunks: int
    bytes_per_chunk: int          # the LARGEST chunk — what has to fit
    total_bytes: int
    unit: str
    detail: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"Plan({self.kind}, {self.n_chunks} chunks x ~{human(self.bytes_per_chunk)} "
                f"of {self.unit}, {human(self.total_bytes)} total)")

    def fits(self, budget: Budget, tier: str = "ram") -> bool:
        return self.bytes_per_chunk <= budget.available(tier)


def _to_bytes(v) -> int:
    """``"256MB"`` / ``"256 MB"`` / ``268435456`` / ``2.5e8`` -> bytes."""
    n = int(parse_size(v)) if isinstance(v, str) else int(v)
    if n <= 0:
        raise ValueError(f"chunk_bytes must be positive, got {v!r}")
    return n


def resolve_chunk_bytes(chunk_bytes, default, budget: Optional[Budget] = None,
                        tier: str = "ram") -> int:
    """The chunk size a reader will use. See the module docstring for the rule.

    The comparison is on the parsed value, so every spelling of the default gets the floor
    rule; a different value is the caller's choice and must fit what the tier has free.
    """
    want = _to_bytes(default if chunk_bytes is None else chunk_bytes)
    if budget is None:
        return want
    avail = int(budget.available(tier))
    if want == _to_bytes(default):
        return max(min(1 << 20, avail), avail // 4, 1)
    if want > avail:
        raise MemoryError(
            f"cannot fit chunk_bytes={human(want)} in the '{tier}' tier ({human(avail)} available). "
            "Lower chunk_bytes, or raise the budget for this tier.")
    return want


class ChunkTooSmallError(MemoryError, ValueError):
    """A chunk size too small for the smallest useful unit of the data (a 16 px tile of every
    channel). It is a budget failure (``MemoryError``) and a bad argument (``ValueError``)."""


_PTR_BYTES = 8       # an indptr entry as it sits in memory once a block is sliced out


def csr_bounds(indptr, itemsize: int, chunk_bytes: int, r0: int = 0, r1: Optional[int] = None):
    """Row (or CSC column) boundaries ``[r0, b1, ..., r1]`` for a compressed-sparse pointer array.

    Cut from the real per-row sizes with ``searchsorted`` over the cumulative byte count, so
    every block holds at most ``chunk_bytes`` of ``data`` + ``indices`` — except a single row
    that alone is larger, which becomes a block of its own (the plan reports it). Rows are also
    capped so the sliced ``indptr`` stays within ``chunk_bytes``: a stretch of empty rows is
    not free.
    """
    r1 = len(indptr) - 1 if r1 is None else int(r1)
    r0 = int(r0)
    max_nnz = max(int(chunk_bytes) // max(int(itemsize), 1), 1)
    max_rows = max(int(chunk_bytes) // _PTR_BYTES, 1)
    out = [r0]
    pos = r0
    while pos < r1:
        limit = int(indptr[pos]) + max_nnz
        end = int(np.searchsorted(indptr[pos:r1 + 1], limit, side="right")) - 1 + pos
        end = min(max(end, pos + 1), r1, pos + max_rows)
        out.append(end)
        pos = end
    return np.asarray(out, dtype=np.int64)


def block_bytes(indptr, bounds, itemsize: int):
    """Bytes of ``data`` + ``indices`` in each block of ``bounds`` (int64 array)."""
    return np.diff(np.asarray(indptr)[bounds].astype(np.int64)) * int(itemsize)


def _largest(sizes, bounds, chunk_bytes: int, budget: Optional[Budget], tier: str, lo_key: str,
             hi_key: str) -> dict:
    """The plan fields that name the largest chunk, shared by the row-blocked readers."""
    if len(sizes) == 0:
        return {"largest_chunk_bytes": 0, "largest_chunk": {}, "fits": True}
    i = int(np.argmax(sizes))
    big = int(sizes[i])
    fits = big <= chunk_bytes and (budget is None or big <= int(budget.available(tier)))
    return {"largest_chunk_bytes": big,
            "largest_chunk": {"index": i, lo_key: int(bounds[i]), hi_key: int(bounds[i + 1]),
                              "bytes": big},
            "fits": bool(fits)}


class _Governed:
    """Shared plumbing: hold one lease at a time, release it when the next chunk is asked for."""

    def __init__(self, budget: Optional[Budget], tier: str = "ram"):
        self._budget, self._tier, self._lease = budget, tier, None

    def _swap_lease(self, nbytes: int):
        if self._lease is not None:
            self._lease.release()
            self._lease = None
        if self._budget is not None and nbytes > 0:
            self._lease = self._budget.acquire(self._tier, str(int(nbytes)))

    @contextlib.contextmanager
    def _transient(self, nbytes: int, tier: Optional[str] = None):
        """Hold ``nbytes`` of the budget for a with-block, apart from the chunk lease — for a
        scan or a conversion that allocates while a chunk iteration may be in progress."""
        lease = None
        if self._budget is not None and nbytes > 0:
            lease = self._budget.acquire(tier or self._tier, str(int(nbytes)))
        try:
            yield
        finally:
            if lease is not None:
                lease.release()

    def close(self):
        if self._lease is not None:
            self._lease.release()
            self._lease = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------------------------------------------------------------------
# obs / var columns — one reader for every AnnData encoding
# ---------------------------------------------------------------------------

def _s(v) -> str:
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)


def _enc_of(node) -> str:
    return _s(node.attrs.get("encoding-type", ""))


def _read_strings(ds, start: int = 0, end: Optional[int] = None) -> np.ndarray:
    """A string dataset — variable-length or fixed-width, any declared encoding — as ``str``."""
    import h5py
    sl = slice(start, end)
    info = h5py.check_string_dtype(ds.dtype)
    if info is not None and info.length is None:
        return np.asarray(ds.asstr(errors="replace")[sl], dtype=str)
    raw = ds[sl]
    if raw.dtype.kind == "S":
        return np.char.decode(raw, "utf-8", "replace")
    if raw.dtype.kind == "O":
        return np.asarray([_s(x) for x in raw], dtype=str)
    return raw.astype(str)


def _is_text(ds) -> bool:
    import h5py
    return h5py.check_string_dtype(ds.dtype) is not None or ds.dtype.kind in ("S", "U")


def obs_group(f, group: str = "obs"):
    """The dataframe group ``obs`` / ``var`` (or ``raw/var``) of an open AnnData file. Handing
    over the dataframe group itself (anything that is already ``encoding-type`` dataframe) works too."""
    import h5py
    if group not in f:
        if isinstance(f, h5py.Group) and _enc_of(f) == "dataframe":
            return f
        raise KeyError(f"{getattr(f, 'filename', 'file')} has no {group!r} group")
    grp = f[group]
    if not isinstance(grp, h5py.Group):
        raise ValueError(f"{group!r} is a compound dataset (the pre-0.7 AnnData layout); "
                         "re-save the file with a current anndata to stream its columns")
    return grp


def obs_index_key(grp) -> str:
    """Name of the dataset that holds the row names: the ``_index`` attribute, else a default."""
    idx = grp.attrs.get("_index")
    if idx is not None:
        return _s(idx)
    for k in ("_index", "index"):
        if k in grp:
            return k
    return "_index"


def obs_column_names(grp) -> list:
    """Columns of a dataframe group, in the order AnnData wrote them."""
    order = grp.attrs.get("column-order")
    if order is not None:
        return [_s(x) for x in np.atleast_1d(order)]
    idx = obs_index_key(grp)
    return [k for k in grp.keys() if k != idx and not k.startswith("__")]


def _categories(node, cache: Optional[dict], key) -> np.ndarray:
    if cache is not None and key in cache:
        return cache[key]
    ds = node["categories"] if "categories" in node else node
    cats = _read_strings(ds).astype(object) if _is_text(ds) else np.asarray(ds[()]).astype(object)
    if cache is not None:
        cache[key] = cats
    return cats


def _decode_codes(codes, cats: np.ndarray, name: str) -> np.ndarray:
    codes = np.asarray(codes)
    out = np.empty(codes.shape, dtype=object)
    ok = codes >= 0
    if ok.any():
        if int(codes.max()) >= len(cats):
            raise ValueError(f"obs column {name!r}: category code {int(codes.max())} is out of "
                             f"range for {len(cats)} categories")
        out[ok] = cats[codes[ok].astype(np.int64)]
    out[~ok] = None                                   # -1 is missing, never the last category
    return out


def _read_nullable(node, start: int, end: Optional[int], name: str) -> np.ndarray:
    sl = slice(start, end)
    vals_ds = node["values"]
    mask = np.asarray(node["mask"][sl], dtype=bool)
    if _is_text(vals_ds):
        out = _read_strings(vals_ds, start, end).astype(object)
        out[mask] = None
        return out
    v = np.asarray(vals_ds[sl])
    if v.dtype.kind == "b":
        out = v.astype(object)
        out[mask] = None
        return out
    if v.dtype.kind in "iu":
        live = v[~mask]
        if live.size and np.abs(live.astype(np.float64)).max() >= 2.0 ** 53:
            out = v.astype(object)                     # not exactly representable as a float
            out[mask] = None
            return out
        out = v.astype(np.float64)
        out[mask] = np.nan
        return out
    out = v.copy()
    out[mask] = np.nan
    return out


def read_obs_column(f, name: str, start: int = 0, end: Optional[int] = None, group: str = "obs",
                    cache: Optional[dict] = None) -> np.ndarray:
    """Rows ``start:end`` of one dataframe column of an open AnnData file, decoded.

    Dispatches on the node's ``encoding-type``:

    * ``categorical`` — an object array of the category values; a code of -1 is missing and
      comes back as ``None`` (it used to index the last category);
    * ``nullable-integer`` — float64 with NaN where masked (a stable dtype whether or not this
      slice has a missing value), or an object array of ints and ``None`` if a value could not
      be held exactly by a float;
    * ``nullable-boolean`` — an object array of ``bool`` and ``None``;
    * ``nullable-string-array`` — an object array of ``str`` and ``None``;
    * ``string-array`` / any string dataset (variable-length or fixed-width) — a ``str`` array;
    * a plain numeric dataset — as stored (floats keep their NaN).

    The pre-0.7 layout (integer codes in the column, categories under ``__categories``) is
    understood too. A column that is not there raises ``KeyError`` naming the ones that are.
    ``cache`` (a dict) keeps each categorical's categories between calls.
    """
    import h5py
    grp = obs_group(f, group)
    if name not in grp:
        raise KeyError(f"{getattr(f, 'filename', 'file')} has no {group} column {name!r}; "
                       f"columns: {obs_column_names(grp)}")
    node = grp[name]
    start = int(start or 0)
    end = None if end is None else int(end)
    if isinstance(node, h5py.Group):
        enc = _enc_of(node)
        if enc == "categorical" or ("codes" in node and "categories" in node):
            return _decode_codes(node["codes"][start:end], _categories(node, cache, (group, name)), name)
        if enc.startswith("nullable") or ("values" in node and "mask" in node):
            return _read_nullable(node, start, end, name)
        raise ValueError(f"{group} column {name!r}: unsupported encoding {enc or '(none)'!r} "
                         f"(a group holding {sorted(node.keys())})")
    legacy = grp.get("__categories")
    if legacy is not None and name in legacy and node.dtype.kind in "iu":
        return _decode_codes(node[start:end], _categories(legacy[name], cache, (group, name)), name)
    if _is_text(node):
        return _read_strings(node, start, end)
    return node[start:end]


def check_obs_columns(f, columns: Sequence[str], group: str = "obs") -> list:
    """Fail now, naming the columns that exist, if any requested column is missing."""
    grp = obs_group(f, group)
    cols = [str(columns)] if isinstance(columns, str) else [str(c) for c in columns]
    for c in cols:
        if c not in grp:
            raise KeyError(f"{getattr(f, 'filename', 'file')} has no {group} column {c!r}; "
                           f"columns: {obs_column_names(grp)}")
    return cols


def read_index(f, group: str = "obs", start: int = 0, end: Optional[int] = None):
    """Row names of ``obs`` / ``var`` as ``str``, or None if the file stores no index."""
    grp = obs_group(f, group)
    key = obs_index_key(grp)
    node = grp.get(key)
    if node is None or not hasattr(node, "shape"):
        return None
    return _read_strings(node, start, end) if _is_text(node) else np.asarray(node[start:end]).astype(str)


# ---------------------------------------------------------------------------
# h5ad — AnnData on HDF5 (scRNA-seq, scATAC-seq)
# ---------------------------------------------------------------------------

class H5adReader(_Governed):
    """Stream an .h5ad X matrix in row blocks sized to a byte budget.

    Handles CSR sparse (the usual layout for counts) and dense; CSC is planned but refused
    when iterated. Row blocks are cells, so a chunk is always a set of whole cells — directly
    usable for per-cell work.

    Sparse blocks are cut from ``indptr`` (see :func:`csr_bounds`): each block's ``data`` +
    ``indices`` is at most ``chunk_bytes`` however skewed the rows are, and a single row bigger
    than that is a block of its own that ``plan()`` reports. ``layer`` picks ``layers/<name>``;
    ``layer="raw"`` picks AnnData's ``.raw`` (``raw/X`` and ``raw/var``) unless a layer
    literally called ``raw`` exists, and ``layer=".raw"`` always does.

    ``obs_columns`` ride along in every chunk, decoded by :func:`read_obs_column` (categoricals,
    nullable integers and booleans, strings, numbers); a column the file does not have raises
    ``KeyError`` here, not mid-stream. ``obs_names`` / ``var_names`` are the row and column names.
    """

    def __init__(self, path: str, chunk_bytes: str | int = "256MB",
                 budget: Optional[Budget] = None, layer: Optional[str] = None,
                 obs_columns: Optional[Sequence[str]] = None, obs_names: bool = False):
        super().__init__(budget)
        import h5py
        self.path = path
        self.f = h5py.File(path, "r")
        try:
            self._setup(layer, chunk_bytes, budget, obs_columns, obs_names)
        except BaseException:
            self.f.close()
            raise

    def _setup(self, layer, chunk_bytes, budget, obs_columns, want_names):
        import h5py
        f = self.f
        self.layer = layer
        self._var_group = "var"
        if layer in (None, ""):
            if "X" not in f:
                raise ValueError(f"{self.path} has no X matrix (obs/var only)")
            node = f["X"]
        elif layer == ".raw" or (layer == "raw" and "raw" in f and "X" in f["raw"]
                                 and "raw" not in f.get("layers", {})):
            node = f["raw/X"]
            self._var_group = "raw/var"
        else:
            have = list(f["layers"].keys()) if "layers" in f else []
            if layer not in have:
                extra = ["raw"] if "raw" in f and "X" in f["raw"] else []
                raise KeyError(f"{self.path} has no layer {layer!r}; layers: {have + extra}")
            node = f[f"layers/{layer}"]
        self.node = node
        self.obs_columns = check_obs_columns(f, obs_columns) if obs_columns else []
        self.want_names = bool(want_names)
        enc = _enc_of(node)
        self.encoding = enc or ("array" if isinstance(node, h5py.Dataset) else "csr_matrix")

        if self.encoding in ("csr_matrix", "csc_matrix"):
            self.shape = tuple(int(x) for x in node.attrs["shape"])
            self.indptr = node["indptr"][:]           # small: n_rows + 1 integers
            self.dtype = node["data"].dtype
            self.nnz = int(self.indptr[-1])
            self._itemsize = self.dtype.itemsize + node["indices"].dtype.itemsize
            self._bytes_total = self.nnz * self._itemsize
        else:
            self.shape = node.shape
            self.dtype = node.dtype
            self.nnz = int(np.prod(self.shape))
            self._itemsize = self.dtype.itemsize
            self._bytes_total = self.nnz * self._itemsize

        self.n_obs, self.n_vars = int(self.shape[0]), int(self.shape[1])
        self.chunk_bytes = resolve_chunk_bytes(chunk_bytes, "256MB", budget)
        self._bounds, self._bounds_for = None, None
        self._cats: dict = {}
        self._obs_names = self._var_names = None

    # ---- names ----
    @property
    def var_names(self) -> np.ndarray:
        if self._var_names is None:
            names = read_index(self.f, self._var_group)
            self._var_names = names if names is not None else np.array(
                [str(i) for i in range(self.n_vars)])
        return self._var_names

    @property
    def obs_names(self) -> np.ndarray:
        """Cell names (the ``obs`` index), read once."""
        if self._obs_names is None:
            names = read_index(self.f, "obs")
            self._obs_names = names if names is not None else np.array(
                [str(i) for i in range(self.n_obs)])
        return self._obs_names

    def _names_slice(self, start: int, end: int) -> np.ndarray:
        if self._obs_names is not None:
            return self._obs_names[start:end]
        names = read_index(self.f, "obs", start, end)
        return names if names is not None else np.array([str(i) for i in range(start, end)])

    # ---- chunking ----
    def _row_bounds(self) -> np.ndarray:
        """Row boundaries of every chunk. Cut from ``indptr`` for CSR, arithmetic for dense."""
        if self._bounds is None or self._bounds_for != self.chunk_bytes:
            if self.encoding == "csr_matrix":
                b = csr_bounds(self.indptr, self._itemsize, self.chunk_bytes)
            elif self.encoding == "csc_matrix":
                b = np.array([0, self.n_obs], dtype=np.int64)        # not streamable by row
            else:
                rows = max(1, self.chunk_bytes // max(self.n_vars * self._itemsize, 1))
                b = np.append(np.arange(0, self.n_obs, rows, dtype=np.int64), self.n_obs)
                b = b[:1] if self.n_obs == 0 else b
            self._bounds, self._bounds_for = b, self.chunk_bytes
        return self._bounds

    def _chunk_sizes(self, bounds: np.ndarray) -> np.ndarray:
        if self.encoding == "csr_matrix":
            return block_bytes(self.indptr, bounds, self._itemsize)
        return np.diff(bounds) * self.n_vars * self._itemsize

    def _rows_per_chunk(self) -> int:
        """Rows in the biggest chunk by rows (kept for callers that sized buffers from it)."""
        d = np.diff(self._row_bounds())
        return int(d.max()) if d.size else 0

    def plan(self) -> Plan:
        bounds = self._row_bounds()
        detail = {"n_obs": self.n_obs, "n_vars": self.n_vars, "nnz": self.nnz,
                  "dtype": str(self.dtype), "file_size": os.path.getsize(self.path),
                  "chunk_bytes": self.chunk_bytes, "layer": self.layer}
        if self.encoding == "csc_matrix":
            n = max(1, math.ceil(self._bytes_total / self.chunk_bytes))
            detail.update(rows_per_chunk=max(1, self.n_obs // n), streamable=False,
                          largest_chunk_bytes=min(self._bytes_total, self.chunk_bytes),
                          fits=False, advice="CSC is column-major; convert to CSR to stream by cell")
            return Plan(kind=f"h5ad/{self.encoding}", source=self.path, n_chunks=n,
                        bytes_per_chunk=min(self._bytes_total, self.chunk_bytes),
                        total_bytes=self._bytes_total, unit="cells", detail=detail)
        sizes = self._chunk_sizes(bounds)
        big = _largest(sizes, bounds, self.chunk_bytes, self._budget, self._tier, "obs_start", "obs_end")
        rows = np.diff(bounds)
        detail.update(big)
        detail.update(rows_per_chunk=int(rows[big["largest_chunk"]["index"]]) if sizes.size else 0,
                      max_rows_per_chunk=int(rows.max()) if rows.size else 0,
                      mean_chunk_bytes=int(sizes.mean()) if sizes.size else 0, streamable=True,
                      mean_bytes_per_row=self._bytes_total / max(self.n_obs, 1))
        return Plan(kind=f"h5ad/{self.encoding}", source=self.path, n_chunks=len(sizes),
                    bytes_per_chunk=detail["largest_chunk_bytes"], total_bytes=self._bytes_total,
                    unit="cells", detail=detail)

    def __iter__(self) -> Iterator[dict]:
        """Yield {'X': scipy.sparse or ndarray, 'obs_start': int, 'obs_end': int, ...}."""
        import scipy.sparse as sp
        if self.encoding == "csc_matrix":
            raise NotImplementedError(
                "CSC h5ad is column-major; row-block streaming would touch every column chunk. "
                "Convert to CSR first (anndata: adata.X = adata.X.tocsr()) or stream by var instead."
            )
        bounds = self._row_bounds()
        for i in range(len(bounds) - 1):
            start, end = int(bounds[i]), int(bounds[i + 1])
            if self.encoding == "csr_matrix":
                lo, hi = int(self.indptr[start]), int(self.indptr[end])
                nbytes = (hi - lo) * self._itemsize
                self._swap_lease(nbytes)                  # lease first, then allocate
                data = self.node["data"][lo:hi]
                indices = self.node["indices"][lo:hi]
                indptr = self.indptr[start:end + 1] - lo
                X = sp.csr_matrix((data, indices, indptr), shape=(end - start, self.n_vars))
                # scipy kept its own (int32) indices and indptr; the int64 originals are dropped
                # now, not when the consumer asks for the next chunk
                del data, indices, indptr
            else:
                nbytes = (end - start) * self.n_vars * self._itemsize
                self._swap_lease(nbytes)
                X = self.node[start:end]
            out = {"X": X, "obs_start": start, "obs_end": end, "n_vars": self.n_vars,
                   "resident_bytes": nbytes}
            if self.want_names:
                out["obs_names"] = self._names_slice(start, end)
            for c in self.obs_columns:
                out[c] = read_obs_column(self.f, c, start, end, cache=self._cats)
            yield out
            out = X = data = indices = indptr = None      # let go before the next block is read

    def close(self):
        super().close()
        try:
            self.f.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Arrow IPC (.arrow / .feather)
# ---------------------------------------------------------------------------

class ArrowIPCReader(_Governed):
    """Stream an Arrow IPC file batch by batch, memory-mapped, budget-accounted.

    A batch larger than ``chunk_bytes`` is handed out as row slices (``RecordBatch.slice``: no
    copy), each at most ``chunk_bytes`` — a 24 MB batch under a 4 MB budget streams as 1 MB
    pieces instead of failing to fit. A batch that already fits is yielded whole. A single row
    larger than the chunk is a slice of its own, and the budget decides whether it can be held.
    The chunk floor rule applies when the caller did not choose a size. For a file (random
    access) ``plan()`` replays the split over the batch metadata, so it names the exact
    largest slice; for a stream it can only estimate.
    """

    def __init__(self, path: str, chunk_bytes: str | int = "256MB",
                 budget: Optional[Budget] = None, columns: Optional[Sequence[str]] = None):
        super().__init__(budget)
        import pyarrow as pa
        self.path, self.columns = path, list(columns) if columns else None
        self._src = pa.memory_map(path, "rb")
        try:
            try:
                self.reader = pa.ipc.open_file(self._src)
                self.is_file = True
                self.n_batches = self.reader.num_record_batches
            except pa.ArrowInvalid:
                self._src.seek(0)
                self.reader = pa.ipc.open_stream(self._src)
                self.is_file = False
                self.n_batches = -1
            self.schema = self.reader.schema
            self.chunk_bytes = resolve_chunk_bytes(chunk_bytes, "256MB", budget)
        except BaseException:
            self._src.close()
            raise

    def _pieces(self, b) -> list:
        """``(offset, rows)`` slices of a batch, each at most ``chunk_bytes`` (or one row)."""
        n, total = b.num_rows, b.nbytes
        if n <= 1 or total <= self.chunk_bytes:
            return [(0, n)]
        rows = max(1, int(self.chunk_bytes * n / total))
        out, off = [], 0
        while off < n:
            m = min(rows, n - off)
            nb = b.slice(off, m).nbytes
            while m > 1 and nb > self.chunk_bytes:            # variable-width rows: shrink to fit
                m = max(1, int(m * self.chunk_bytes / nb))
                nb = b.slice(off, m).nbytes
            out.append((off, m))
            off += m
        return out

    def _batches(self):
        src = (self.reader.get_batch(i) for i in range(self.n_batches)) if self.is_file \
            else iter(self.reader)
        for b in src:
            yield b.select(self.columns) if self.columns else b

    def plan(self) -> Plan:
        size = os.path.getsize(self.path)
        detail = {"columns": self.schema.names, "random_access": self.is_file,
                  "chunk_bytes": self.chunk_bytes}
        if self.is_file:
            sizes = []
            for b in self._batches():
                n = b.num_rows
                sizes += [(b if m == n else b.slice(off, m)).nbytes for off, m in self._pieces(b)]
            big = max(sizes, default=0)
            fits = big <= self.chunk_bytes and (self._budget is None
                                                or big <= int(self._budget.available(self._tier)))
            detail.update(batches=self.n_batches, largest_chunk_bytes=big, fits=bool(fits),
                          buffer_bytes=int(sum(sizes)))
            return Plan(kind="arrow-ipc", source=self.path, n_chunks=len(sizes), bytes_per_chunk=big,
                        total_bytes=size, unit="rows", detail=detail)
        n = max(1, size // self.chunk_bytes)
        big = min(size, self.chunk_bytes)
        detail.update(estimated=True, largest_chunk_bytes=big, fits=True)
        return Plan(kind="arrow-ipc", source=self.path, n_chunks=n, bytes_per_chunk=big,
                    total_bytes=size, unit="rows", detail=detail)

    def __iter__(self):
        for b in self._batches():
            n = b.num_rows
            for off, m in self._pieces(b):
                piece = b if m == n else b.slice(off, m)
                self._swap_lease(piece.nbytes)
                yield piece
                piece = None

    def close(self):
        super().close()
        try:
            self._src.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Imaging — tiled reads from TIFF / OME-Zarr
# ---------------------------------------------------------------------------

_MIN_TILE = 16          # below this a window is not worth reading; say so instead


def _spatial_axes(array, shape: tuple, axes=None) -> tuple:
    """``(axes, y_axis, x_axis)`` for an array: named, else guessed.

    Names come from the caller, else from an ``axes`` attribute on the array (a
    :class:`TiffTileArray` has one). With no names: 2-D is ``YX``; with more dimensions a short
    last axis (<= 4) after two longer ones is interleaved samples (``...YXS``, an RGB or RGBA
    image) and anything else puts Y and X last (``...CYX``, a channel stack).
    """
    nd = len(shape)
    if nd < 2:
        raise ValueError(f"an image needs at least two dimensions; got shape {shape}")
    ax = axes if axes is not None else getattr(array, "axes", None)
    if ax is not None and not isinstance(ax, str):
        ax = "".join(str(a) for a in ax)
    if ax:
        ax = ax.upper()
        if len(ax) != nd or ax.count("Y") != 1 or ax.count("X") != 1:
            raise ValueError(f"axes {ax!r} do not describe an array of shape {shape}: "
                             f"need {nd} letters with exactly one Y and one X")
        return ax, ax.index("Y"), ax.index("X")
    if nd == 2:
        return "YX", 0, 1
    if shape[-1] <= 4 and shape[-2] > 4:
        return "Q" * (nd - 3) + "YXS", nd - 3, nd - 2
    return ("C" if nd == 3 else "Q" * (nd - 2)) + "YX", nd - 2, nd - 1


def _ome_axes(group, ndim: int) -> Optional[str]:
    """Axis letters from an OME-Zarr group's ``multiscales`` metadata, if it has any."""
    try:
        attrs = dict(group.attrs)
        ms = attrs.get("multiscales") or attrs.get("ome", {}).get("multiscales")
        names = [(a["name"] if isinstance(a, dict) else str(a))[0].upper() for a in ms[0]["axes"]]
        return "".join(names) if len(names) == ndim else None
    except Exception:
        return None


class TiledImageReader(_Governed):
    """Yield (tile, (y0, y1, x0, x1)) windows sized so each tile fits the budget.

    Works on anything exposing `.shape` and numpy slicing: a zarr array, an OME-Zarr
    pyramid level, a :class:`TiffTileArray`. Overlap is for models that need
    context at tile edges (Mask R-CNN, cellpose, stardist), so masks can be stitched.

    Tiles are windows over the **Y and X axes** — found from ``axes`` (or the array's own
    ``axes`` attribute, or a rule when there are none: see ``_spatial_axes``) — and every other
    axis rides along whole: a tile of a ``(6, Y, X)`` stack is ``(6, h, w)``, of an RGB image
    ``(h, w, 3)``. Their bytes count towards ``chunk_bytes``, so the tile side is
    ``isqrt(chunk_bytes / (itemsize * channels))``. There is no 64 px floor to overshoot the
    chunk with; if not even a 16 px tile fits, construction raises ``MemoryError`` saying why.
    """

    def __init__(self, array, chunk_bytes: str | int = "512MB",
                 budget: Optional[Budget] = None, overlap: int = 0, tier: str = "ram",
                 axes: Optional[str] = None):
        super().__init__(budget, tier)
        self.a = array
        self.shape = tuple(int(n) for n in array.shape)
        self.dtype = np.dtype(array.dtype)
        self.overlap = int(overlap)
        self.chunk_bytes = resolve_chunk_bytes(chunk_bytes, "512MB", budget, tier)
        self.axes, self.y_axis, self.x_axis = _spatial_axes(array, self.shape, axes)
        self.h, self.w = self.shape[self.y_axis], self.shape[self.x_axis]
        # every non-spatial axis (channels, samples, z, t) is read whole with each tile
        self.channels = math.prod(n for i, n in enumerate(self.shape)
                                  if i not in (self.y_axis, self.x_axis))
        self.trailing = self.channels                         # the name this attribute used to have
        px = self.dtype.itemsize * self.channels
        side = math.isqrt(self.chunk_bytes // max(px, 1))
        need = min(_MIN_TILE, max(self.h, self.w, 1))
        if side < need:
            raise ChunkTooSmallError(
                f"chunk_bytes={human(self.chunk_bytes)} cannot hold even a {need} px tile of this "
                f"image: {self.channels} channel{'s' if self.channels != 1 else ''} x "
                f"{self.dtype.itemsize} B is {px} B a pixel, so {need} x {need} px needs "
                f"{human(need * need * px)}. Raise chunk_bytes (or the budget), or read fewer channels.")
        self.tile = max(1, min(side, max(self.h, self.w)))

    @classmethod
    def from_tiff(cls, path: str, level: int = 0, **kw):
        """Tiled/pyramidal TIFF. Reads individual TIFF tiles directly, so it does not depend
        on tifffile's zarr bridge (which pins a zarr major version)."""
        arr = TiffTileArray(path, level=level)
        try:
            r = cls(arr, **kw)
        except BaseException:
            arr.close()
            raise
        r._store = arr
        return r

    @classmethod
    def from_zarr(cls, path: str, component: Optional[str] = None, **kw):
        import zarr
        arr = zarr.open(path, mode="r")
        if component:
            group = arr
            arr = group[component]
            if kw.get("axes") is None:
                kw["axes"] = _ome_axes(group, len(arr.shape))
        return cls(arr, **kw)

    def windows(self) -> list[tuple[int, int, int, int]]:
        step = max(self.tile - self.overlap, 1)
        out = []
        for y0 in range(0, self.h, step):
            for x0 in range(0, self.w, step):
                out.append((y0, min(y0 + self.tile, self.h), x0, min(x0 + self.tile, self.w)))
                if x0 + self.tile >= self.w:
                    break
            if y0 + self.tile >= self.h:
                break
        return out

    def _window_key(self, y0: int, y1: int, x0: int, x1: int) -> tuple:
        key = [slice(None)] * len(self.shape)
        key[self.y_axis], key[self.x_axis] = slice(y0, y1), slice(x0, x1)
        return tuple(key)

    def plan(self) -> Plan:
        w = self.windows()
        px = self.dtype.itemsize * self.channels
        biggest = max(((y1 - y0) * (x1 - x0) for y0, y1, x0, x1 in w), default=0)
        per = biggest * px
        fits = per <= self.chunk_bytes and (self._budget is None
                                            or per <= int(self._budget.available(self._tier)))
        return Plan(kind="image-tiles", source=str(getattr(self.a, "path", "array")),
                    n_chunks=len(w), bytes_per_chunk=per,
                    total_bytes=math.prod(self.shape) * self.dtype.itemsize, unit="pixels",
                    detail={"shape": self.shape, "axes": self.axes, "dtype": str(self.dtype),
                            "channels": self.channels, "tile": self.tile, "overlap": self.overlap,
                            "windows": len(w), "chunk_bytes": self.chunk_bytes,
                            "largest_chunk_bytes": per, "fits": bool(fits)})

    def __iter__(self):
        px = self.dtype.itemsize * self.channels
        for (y0, y1, x0, x1) in self.windows():
            self._swap_lease((y1 - y0) * (x1 - x0) * px)          # lease, then read
            tile = np.asarray(self.a[self._window_key(y0, y1, x0, x1)])
            yield tile, (y0, y1, x0, x1)
            tile = None

    def close(self):
        super().close()
        st = getattr(self, "_store", None)
        if st is not None:
            try:
                st.close()
            except Exception:
                pass


def _axis_plan(r: range, tile: int, n: int) -> list:
    """For an ascending ``range`` over an axis cut into ``tile``-sized blocks of length ``n``:
    ``[(block, slice inside the block, slice of the output)]`` for the blocks that hold any of it."""
    a, s, cnt = r.start, r.step, len(r)
    last = a + (cnt - 1) * s
    out = []
    for t in range(a // tile, last // tile + 1):
        t0, t1 = t * tile, min((t + 1) * tile, n)
        k0 = 0 if t0 <= a else -(-(t0 - a) // s)
        k1 = min(cnt, -(-(t1 - a) // s))
        if k1 <= k0:
            continue
        e0 = a + k0 * s
        out.append((t, slice(e0 - t0, e0 - t0 + (k1 - k0 - 1) * s + 1, s), slice(k0, k1)))
    return out


class TiffTileArray:
    """Numpy-style slicing over a TIFF series, decoding only the tiles a selection touches.

    A gigapixel ssDNA or H&E page never becomes resident: ``a[y0:y1, x0:x1]`` reads and decodes
    exactly the TIFF tiles that intersect the window. Striped (non-tiled) TIFFs are read the
    same way, strip by strip. Works with any zarr version because it uses none.

    **Shape convention.** ``shape`` is the whole series shape, every axis kept, and ``axes``
    names them with tifffile's letters:

    ====================================  ===============  ============
    file                                  ``axes``         ``shape``
    ====================================  ===============  ============
    one grayscale plane                   ``YX``           ``(H, W)``
    interleaved RGB(A)                    ``YXS``          ``(H, W, 3)``
    planar-separate samples               ``SYX``          ``(3, H, W)``
    one page per channel (OME, CODEX)     ``CYX``          ``(C, H, W)``
    hyperstack                            ``ZCYX`` etc.    ``(Z, C, H, W)``
    ====================================  ===============  ============

    Channels are never folded away: a 6-plex stack is ``(6, 1500, 1500)``, not one plane.
    ``y_axis``/``x_axis`` give the spatial positions (``s_axis`` the sample axis if any), so
    code that must work for any layout slices through :meth:`window` instead of assuming
    ``a[y0:y1, x0:x1]`` means the first two axes.

    Indexing is numpy's for integers, slices (any step, negative included) and ``Ellipsis``: an
    integer drops its axis; a step reads the covering tiles and keeps only the sampled pixels,
    so a ``[::8, ::8]`` thumbnail is never materialised at full size. Every page (channel, z, t)
    and every sample plane a selection needs is read, and no other. JPEG-compressed tiles are
    decoded with their page's ``JPEGTables``.
    """

    def __init__(self, path: str, level: int = 0, series: int = 0):
        import tifffile
        self.path = path
        self._tf = tifffile.TiffFile(path)
        try:
            self._open(level, series)
        except BaseException:
            self._tf.close()
            raise

    def _open(self, level: int, series: int):
        tf = self._tf
        if not 0 <= series < len(tf.series):
            raise IndexError(f"{self.path}: series {series} out of range ({len(tf.series)} in file)")
        s = tf.series[series]
        levels = getattr(s, "levels", None) or [s]
        if not 0 <= level < len(levels):
            raise IndexError(f"{self.path}: level {level} out of range ({len(levels)} in file)")
        s = levels[level]
        pages = s.pages
        first = pages[0]
        page = getattr(first, "keyframe", None) or first
        sep, depth, length, width, contig = (int(v) for v in page.shaped)
        if depth != 1:
            raise NotImplementedError(f"{self.path}: volumetric TIFF pages (depth {depth}) "
                                      "are not supported")
        page_shape = ((sep,) if sep > 1 else ()) + (length, width) + ((contig,) if contig > 1 else ())
        shape = tuple(int(n) for n in s.shape)
        n_lead = len(shape) - len(page_shape)
        lead = shape[:max(n_lead, 0)]
        if n_lead < 0 or shape[n_lead:] != page_shape or math.prod(lead) != len(pages):
            raise NotImplementedError(
                f"{self.path}: cannot map series {series} (shape {shape}, axes {s.axes!r}, "
                f"{len(pages)} pages) onto pages of shape {page_shape}")
        axes = str(getattr(s, "axes", ""))
        if len(axes) != len(shape):
            axes = "Q" * n_lead + ("S" if sep > 1 else "") + "YX" + ("S" if contig > 1 else "")
        self.shape, self.axes, self.ndim = shape, axes, len(shape)
        self.dtype = np.dtype(page.dtype)
        self.page, self._pages, self._series = page, pages, s
        self._lead, self._n_lead = lead, n_lead
        self._planar, self._contig = sep > 1, contig > 1
        self.y_axis = n_lead + (1 if self._planar else 0)
        self.x_axis = self.y_axis + 1
        self.s_axis = n_lead if self._planar else (self.x_axis + 1 if self._contig else None)
        self.tiled = bool(page.is_tiled)
        if self.tiled:
            self.th, self.tw = int(page.tilelength), int(page.tilewidth)
        else:
            self.th, self.tw = max(1, min(int(getattr(page, "rowsperstrip", 0) or length), length)), width
        self.ny, self.nx = -(-length // self.th), -(-width // self.tw)
        self._jpeg = int(getattr(page, "compression", 1)) in (6, 7, 34892, 33007)

    # ---- numpy-ish surface ----
    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize

    @property
    def spatial_shape(self) -> tuple:
        return (self.shape[self.y_axis], self.shape[self.x_axis])

    def window(self, y0: int, y1: int, x0: int, x1: int, step: int = 1) -> np.ndarray:
        """Rows ``y0:y1`` and columns ``x0:x1`` with every other axis (channels, samples) whole,
        whatever order the file keeps its axes in."""
        key = [slice(None)] * self.ndim
        key[self.y_axis], key[self.x_axis] = slice(y0, y1, step), slice(x0, x1, step)
        return self[tuple(key)]

    def _selectors(self, key) -> list:
        nd = self.ndim
        if not isinstance(key, tuple):
            key = (key,)
        n_ell = sum(k is Ellipsis for k in key)
        if n_ell > 1:
            raise IndexError("an index can have only a single ellipsis")
        if n_ell:
            i = next(i for i, k in enumerate(key) if k is Ellipsis)
            key = key[:i] + (slice(None),) * (nd - len(key) + 1) + key[i + 1:]
        if len(key) > nd:
            raise IndexError(f"too many indices: the array is {nd}-dimensional, {len(key)} were given")
        key = key + (slice(None),) * (nd - len(key))
        sel = []
        for ax, (k, n) in enumerate(zip(key, self.shape)):
            if isinstance(k, slice):
                r = range(*k.indices(n))
                if r.step < 0 and len(r):
                    sel.append((False, range(r[-1], r[0] + 1, -r.step), True))   # read ascending, flip after
                else:
                    sel.append((False, r, False))
            else:
                try:
                    i = operator.index(k)
                except TypeError:
                    raise IndexError("only integers, slices and Ellipsis are valid indices") from None
                if not -n <= i < n:
                    raise IndexError(f"index {i} is out of bounds for axis {ax} with size {n}")
                i %= n
                sel.append((True, range(i, i + 1), False))
        return sel

    def __getitem__(self, key):
        sel = self._selectors(key)
        out = np.zeros([len(r) for _, r, _ in sel], self.dtype)
        if out.size:
            nl = self._n_lead
            ry, rx = sel[self.y_axis][1], sel[self.x_axis][1]
            plan_y = _axis_plan(ry, self.th, self.shape[self.y_axis])
            plan_x = _axis_plan(rx, self.tw, self.shape[self.x_axis])
            rs = sel[self.s_axis][1] if self.s_axis is not None else range(1)
            lead = [r for _, r, _ in sel[:nl]]
            for pos in itertools.product(*[range(len(r)) for r in lead]):
                idx = [r[p] for r, p in zip(lead, pos)]
                page = self._pages[int(np.ravel_multi_index(idx, self._lead)) if nl else 0]
                self._fill(page, out[pos] if nl else out, plan_y, plan_x, rs)
        view = tuple(0 if is_int else (slice(None, None, -1) if flip else slice(None))
                     for is_int, _, flip in sel)
        return out[view]

    def _fill(self, page, pv: np.ndarray, plan_y: list, plan_x: list, rs: range):
        planes = list(rs) if self._planar else [0]
        pick = list(rs) if self._contig else None
        for k, s in enumerate(planes):
            for ty, sy, dy in plan_y:
                for tx, sx, dx in plan_x:
                    tile = self._tile(page, s, ty, tx)
                    if tile is None:
                        continue                                  # an empty segment stays zero
                    sub = tile[sy, sx]
                    if self._contig:
                        pv[dy, dx] = sub[..., pick]
                    elif self._planar:
                        pv[k, dy, dx] = sub[..., 0]
                    else:
                        pv[dy, dx] = sub[..., 0]

    def _tile(self, page, plane: int, ty: int, tx: int) -> Optional[np.ndarray]:
        """One decoded tile or strip as ``(length, width, samples)``; None if it was never written."""
        i = (plane * self.ny + ty) * self.nx + tx
        off, cnt = int(page.dataoffsets[i]), int(page.databytecounts[i])
        if cnt == 0:
            return None
        fh = self._tf.filehandle
        with getattr(fh, "lock", None) or contextlib.nullcontext():
            fh.seek(off)
            raw = fh.read(cnt)
        kw = {"jpegtables": getattr(page, "jpegtables", None)}
        if self._jpeg:
            kw["jpegheader"] = getattr(self.page, "jpegheader", None)
        data, _, shape = page.decode(raw, i, **kw)
        if data is None:
            return None
        a = np.asarray(data)
        if a.ndim != 4:
            a = a.reshape(shape)
        return a[0].astype(self.dtype, copy=False)                # (depth, length, width, samples)

    def close(self):
        try:
            self._tf.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Sinks — write results out as they are produced, never accumulate
# ---------------------------------------------------------------------------

class ParquetSink:
    """Append RecordBatches / tables to one Parquet file without holding them.

    A Parquet file has one schema, fixed when the first row group is written, so a column that
    is all-null in the first batch (type ``null``) used to make every later batch fail with
    "Table schema does not match". The rules now:

    * ``schema=`` (a ``pyarrow.Schema`` or a ``{name: type}`` mapping) declares the file's
      columns up front. Every batch is conformed to it — null columns and compatible types are
      cast, a missing column is filled with nulls — and a column that cannot be cast raises
      ``ValueError`` naming it.
    * Without ``schema=``, the type of a column is the first non-null type any batch gives it.
      While some column is still all-null the sink holds the batches back (at most
      ``lookahead_bytes`` of them, 8 MB by default; this is the only time it holds anything),
      opens the writer when every column has a type, and flushes them. A column that is
      still all-null when that limit is reached, or when the sink closes, is written as
      ``string``. Later batches are cast to the file's schema the same way.
    """

    def __init__(self, path: str, compression: str = "zstd", schema=None,
                 lookahead_bytes: int = 8 << 20):
        import pyarrow as pa
        self.path, self.compression, self._w = path, compression, None
        self.rows = 0
        if schema is not None and not isinstance(schema, pa.Schema):
            schema = pa.schema(list(dict(schema).items()))
        self.schema = schema                      # declared, or the one the file ended up with
        self.lookahead_bytes = int(lookahead_bytes)
        self._pending: list = []
        self._pending_bytes = 0

    # ---- schema handling ----
    @staticmethod
    def _table(batch):
        import pyarrow as pa
        if isinstance(batch, pa.RecordBatch):
            return pa.Table.from_batches([batch])
        if isinstance(batch, pa.Table):
            return batch
        return pa.table(batch)

    def _inferred(self, final: bool):
        """The schema the held batches imply, or None while a column is still all-null."""
        import pyarrow as pa
        first = self._pending[0].schema
        fields = []
        for i, f in enumerate(first):
            t = f.type
            if pa.types.is_null(t):
                t = next((tb.schema.field(f.name).type for tb in self._pending
                          if f.name in tb.schema.names
                          and not pa.types.is_null(tb.schema.field(f.name).type)), None)
                if t is None:
                    if not final:
                        return None
                    t = pa.string()
            fields.append(pa.field(f.name, t, f.nullable))
        return pa.schema(fields, metadata=first.metadata)

    def _conform(self, tbl):
        import pyarrow as pa
        schema = self.schema
        extra = [n for n in tbl.column_names if n not in schema.names]
        if extra:
            raise ValueError(f"ParquetSink: columns {extra} are not in the file schema "
                             f"{schema.names}; declare them up front with schema=")
        cols = []
        for f in schema:
            if f.name not in tbl.column_names:
                cols.append(pa.nulls(tbl.num_rows, f.type))
                continue
            col = tbl[f.name]
            if col.type != f.type:
                try:
                    col = col.cast(f.type)
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as e:
                    raise ValueError(
                        f"ParquetSink: column {f.name!r} arrives as {col.type} but the file's type "
                        f"is {f.type} ({e}); declare the column types up front with schema=") from None
            cols.append(col)
        return pa.Table.from_arrays(cols, schema=schema)

    def _open(self):
        import pyarrow.parquet as pq
        self._w = pq.ParquetWriter(self.path, self.schema, compression=self.compression)

    def _flush(self):
        pending, self._pending, self._pending_bytes = self._pending, [], 0
        for t in pending:
            self._w.write_table(self._conform(t))

    def write(self, batch):
        tbl = self._table(batch)
        if self._w is None and self.schema is None:
            self._pending.append(tbl)
            self._pending_bytes += tbl.nbytes
            self.rows += tbl.num_rows
            schema = self._inferred(final=self._pending_bytes > self.lookahead_bytes)
            if schema is not None:
                self.schema = schema
                self._open()
                self._flush()
            return
        if self._w is None:
            self._open()
        self._w.write_table(self._conform(tbl))
        self.rows += tbl.num_rows

    def close(self):
        if self._w is None and self._pending:
            self.schema = self._inferred(final=True)
            self._open()
            self._flush()
        if self._w is not None:
            self._w.close()
            self._w = None

    def __enter__(self):
        return self

    def __exit__(self, *e):
        self.close()
        return False


class MaskSink:
    """Write per-tile segmentation output (Mask R-CNN, cellpose, stardist) straight to a
    Zarr array at full image resolution, so nothing but the current tile is ever resident.

    Instance labels are offset per tile so ids stay unique across the whole image.
    """

    def __init__(self, path: str, shape: tuple[int, int], dtype="uint32",
                 chunk: int = 1024, overwrite: bool = True):
        import zarr
        self.z = zarr.open(path, mode="w" if overwrite else "a", shape=shape,
                           chunks=(chunk, chunk), dtype=dtype)
        self.path, self.next_label, self.tiles = path, 1, 0

    def write(self, labels: np.ndarray, window: tuple[int, int, int, int],
              relabel: bool = True) -> int:
        y0, y1, x0, x1 = window
        lab = np.asarray(labels)
        if relabel:
            m = lab > 0
            n = int(lab.max()) if m.any() else 0
            if n:
                lab = lab.astype(np.uint32, copy=True)
                lab[m] += np.uint32(self.next_label - 1)
                self.next_label += n
        self.z[y0:y1, x0:x1] = lab
        self.tiles += 1
        return self.next_label - 1

    @property
    def n_objects(self) -> int:
        return self.next_label - 1

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False
