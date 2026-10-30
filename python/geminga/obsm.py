"""Stream an AnnData ``obsm`` matrix — cell embeddings — in row blocks sized to a byte budget.

``H5adReader`` streams ``X``: counts, one row per cell, one column per gene. The embeddings a
foundation model writes back — Geneformer, scGPT, scVI, a PCA — live beside it under
``obsm/<key>`` with the same rows and far fewer columns, and until 0.9 GEMINGA could not stream
them at all. This reader gives them the same contract as every other format:

  1. ``plan()`` from the array's shape and dtype, before touching data,
  2. one lease per block, acquired before the slice is read,
  3. released when the consumer asks for the next block.

Chunks are the dict shape ``H5adReader`` yields, so a consumer written for counts works on
embeddings unchanged — the matrix is under ``"X"``, ``"obsm_key"`` says which one, and
``"n_vars"`` now means the embedding dimension. ``obs_names`` (cell barcodes) ride along,
because an embedding row is worthless unless it maps back to a cell.

Dense obsm arrays are the common case. AnnData can also store an obsm entry as a sparse group
(``encoding-type`` ``csr_matrix``); that is handled the way ``X`` is, with blocks cut from
``indptr``. Column-major (``csc``) is refused with the same fix ``H5adReader`` gives. A
DataFrame-valued entry (``encoding-type`` ``dataframe``) is not a matrix: ``obsm_keys`` lists it
with kind ``"dataframe"`` and ``ObsmReader`` refuses it, saying which keys it can stream.

``obs_columns`` are read by the same :func:`geminga.formats.read_obs_column` as ``H5adReader``'s,
so categoricals (missing codes included), nullable integers and booleans, and strings come back
decoded, and a column the file does not have is a ``KeyError`` at construction.
"""
from __future__ import annotations

import math
import os
from typing import Iterator, Optional, Sequence

import numpy as np

from ._core import Budget
from .formats import (Plan, _enc_of, _Governed, _largest, block_bytes, check_obs_columns, csr_bounds,
                      obs_column_names, obs_index_key, read_index, read_obs_column,
                      resolve_chunk_bytes)

__all__ = ["ObsmReader", "obsm_keys", "open_obsm"]


def obsm_keys(path: str) -> dict:
    """``{key: (n_obs, dim, dtype, kind)}`` for every obsm entry, from metadata only.

    ``kind`` is ``"array"``, ``"csr_matrix"``, ``"csc_matrix"`` or ``"dataframe"``; for a
    DataFrame ``dim`` is its number of columns and ``dtype`` is ``"mixed"``. Only the first
    three can be streamed by :class:`ObsmReader`.
    """
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        if "obsm" not in f:
            return out
        for k, node in f["obsm"].items():
            enc = _enc_of(node)
            if isinstance(node, h5py.Dataset):
                out[k] = (int(node.shape[0]), int(node.shape[1]) if node.ndim > 1 else 1,
                          str(node.dtype), "array")
            elif enc in ("csr_matrix", "csc_matrix"):
                shape = tuple(int(x) for x in node.attrs["shape"])
                out[k] = (shape[0], shape[1], str(node["data"].dtype), enc)
            elif enc == "dataframe":
                idx = node.get(obs_index_key(node))
                n = int(idx.shape[0]) if idx is not None and hasattr(idx, "shape") else 0
                out[k] = (n, len(obs_column_names(node)), "mixed", "dataframe")
    return out


class ObsmReader(_Governed):
    """Stream ``obsm/<key>`` from an .h5ad in cell blocks sized to a byte budget.

    Yields ``{"X": ndarray | csr_matrix, "obsm_key": str, "obs_start": int, "obs_end": int,
    "n_vars": dim, "resident_bytes": int, ["obs_names": ndarray], <one key per obs column>}``.
    """

    def __init__(self, path: str, key: str, chunk_bytes: str | int = "256MB",
                 budget: Optional[Budget] = None,
                 obs_columns: Optional[Sequence[str]] = None,
                 obs_names: bool = True):
        super().__init__(budget)
        import h5py
        self.path, self.key = path, key
        self.f = h5py.File(path, "r")
        try:
            self._setup(key, chunk_bytes, budget, obs_columns, obs_names)
        except BaseException:
            self.f.close()
            raise

    def _setup(self, key, chunk_bytes, budget, obs_columns, want_names):
        import h5py
        f, path = self.f, self.path
        if "obsm" not in f or key not in f["obsm"]:
            have = list(f["obsm"].keys()) if "obsm" in f else []
            raise KeyError(f"{path} has no obsm/{key!r}; available: {have}")
        node = f[f"obsm/{key}"]
        self.node = node
        enc = _enc_of(node)
        if isinstance(node, h5py.Group) and enc not in ("csr_matrix", "csc_matrix"):
            usable = [k for k, v in obsm_keys(path).items() if v[3] != "dataframe"]
            if enc == "dataframe":
                raise ValueError(
                    f"obsm/{key} is a DataFrame (columns: {', '.join(obs_column_names(node))}), "
                    f"not an array or a sparse matrix, so ObsmReader cannot stream it as one block. "
                    f"Array-valued keys it can stream: {usable}")
            raise ValueError(f"obsm/{key} has encoding {enc or '(none)'!r}, which ObsmReader does "
                             f"not read. Array-valued keys it can stream: {usable}")
        self.obs_columns = check_obs_columns(f, obs_columns) if obs_columns else []
        self.want_names = bool(want_names)
        self.encoding = enc if enc.endswith("_matrix") else "array"

        if self.encoding in ("csr_matrix", "csc_matrix"):
            self.shape = tuple(int(x) for x in node.attrs["shape"])
            self.indptr = node["indptr"][:]
            self.dtype = node["data"].dtype
            self.nnz = int(self.indptr[-1])
            self._itemsize = self.dtype.itemsize + node["indices"].dtype.itemsize
            self._bytes_total = self.nnz * self._itemsize
        else:
            self.shape = ((int(node.shape[0]), 1) if node.ndim == 1
                          else (int(node.shape[0]), int(node.shape[1])))
            self.dtype = node.dtype
            self.nnz = int(self.shape[0]) * int(self.shape[1])
            self._itemsize = self.dtype.itemsize
            self._bytes_total = self.nnz * self._itemsize

        self.n_obs, self.dim = self.shape
        self.n_vars = self.dim                      # the field consumers already read
        # A quarter of what is free, floored at 1 MB — but never above what is free (a floor
        # larger than the budget plans a chunk that can never fit); a chunk the caller chose
        # must fit what is free or construction fails.
        self.chunk_bytes = resolve_chunk_bytes(chunk_bytes, "256MB", budget)
        self._bounds, self._bounds_for = None, None
        self._cats: dict = {}
        self._names = None

    # ---- metadata ----
    @property
    def obs_names(self) -> np.ndarray:
        """Cell barcodes, read once (they are small)."""
        if self._names is None:
            names = read_index(self.f, "obs") if "obs" in self.f else None
            self._names = names if names is not None else np.array([str(i) for i in range(self.n_obs)])
        return self._names

    def _names_slice(self, start: int, end: int) -> np.ndarray:
        if self._names is not None:
            return self._names[start:end]
        names = read_index(self.f, "obs", start, end) if "obs" in self.f else None
        return names if names is not None else np.array([str(i) for i in range(start, end)])

    # ---- chunking ----
    def _row_bounds(self) -> np.ndarray:
        if self._bounds is None or self._bounds_for != self.chunk_bytes:
            if self.encoding == "csr_matrix":
                b = csr_bounds(self.indptr, self._itemsize, self.chunk_bytes)
            elif self.encoding == "csc_matrix":
                b = np.array([0, self.n_obs], dtype=np.int64)
            else:
                rows = max(1, self.chunk_bytes // max(self.dim * self._itemsize, 1))
                b = np.append(np.arange(0, self.n_obs, rows, dtype=np.int64), self.n_obs)
                b = b[:1] if self.n_obs == 0 else b
            self._bounds, self._bounds_for = b, self.chunk_bytes
        return self._bounds

    def _chunk_sizes(self, bounds: np.ndarray) -> np.ndarray:
        if self.encoding == "csr_matrix":
            return block_bytes(self.indptr, bounds, self._itemsize)
        return np.diff(bounds) * self.dim * self._itemsize

    def _rows_per_chunk(self) -> int:
        d = np.diff(self._row_bounds())
        return int(d.max()) if d.size else 0

    def plan(self) -> Plan:
        detail = {"n_obs": self.n_obs, "dim": self.dim, "dtype": str(self.dtype),
                  "obsm_key": self.key, "file_size": os.path.getsize(self.path),
                  "chunk_bytes": self.chunk_bytes}
        if self.encoding == "csc_matrix":
            n = max(1, math.ceil(self._bytes_total / self.chunk_bytes))
            detail.update(rows_per_chunk=max(1, self.n_obs // n), streamable=False, fits=False,
                          largest_chunk_bytes=min(self._bytes_total, self.chunk_bytes))
            return Plan(kind=f"h5ad-obsm/{self.encoding}", source=f"{self.path}::obsm/{self.key}",
                        n_chunks=n, bytes_per_chunk=min(self._bytes_total, self.chunk_bytes),
                        total_bytes=self._bytes_total, unit="cells", detail=detail)
        bounds = self._row_bounds()
        sizes = self._chunk_sizes(bounds)
        big = _largest(sizes, bounds, self.chunk_bytes, self._budget, self._tier, "obs_start", "obs_end")
        rows = np.diff(bounds)
        detail.update(big)
        detail.update(rows_per_chunk=int(rows[big["largest_chunk"]["index"]]) if sizes.size else 0,
                      max_rows_per_chunk=int(rows.max()) if rows.size else 0, streamable=True,
                      mean_bytes_per_row=self._bytes_total / max(self.n_obs, 1))
        return Plan(kind=f"h5ad-obsm/{self.encoding}", source=f"{self.path}::obsm/{self.key}",
                    n_chunks=len(sizes), bytes_per_chunk=detail["largest_chunk_bytes"],
                    total_bytes=self._bytes_total, unit="cells", detail=detail)

    # ---- streaming ----
    def __iter__(self) -> Iterator[dict]:
        import scipy.sparse as sp
        if self.encoding == "csc_matrix":
            raise NotImplementedError(
                f"obsm/{self.key} is CSC (column-major); row-block streaming would touch every "
                "column chunk. Convert to CSR or dense first (anndata: adata.obsm[k] = "
                "adata.obsm[k].tocsr()).")
        bounds = self._row_bounds()
        for i in range(len(bounds) - 1):
            start, end = int(bounds[i]), int(bounds[i + 1])
            if self.encoding == "csr_matrix":
                lo, hi = int(self.indptr[start]), int(self.indptr[end])
                nbytes = (hi - lo) * self._itemsize
                self._swap_lease(nbytes)                  # lease first, then allocate
                X = sp.csr_matrix((self.node["data"][lo:hi], self.node["indices"][lo:hi],
                                   self.indptr[start:end + 1] - lo),
                                  shape=(end - start, self.dim))
            else:
                nbytes = (end - start) * self.dim * self._itemsize
                self._swap_lease(nbytes)
                X = self.node[start:end]
                if X.ndim == 1:
                    X = X[:, None]
            out = {"X": X, "obsm_key": self.key, "obs_start": start, "obs_end": end,
                   "n_vars": self.dim, "resident_bytes": nbytes}
            if self.want_names:
                out["obs_names"] = self._names_slice(start, end)
            for c in self.obs_columns:
                out[c] = read_obs_column(self.f, c, start, end, cache=self._cats)
            yield out
            out = X = None                                # let go before the next block is read

    def close(self):
        super().close()
        try:
            self.f.close()
        except Exception:
            pass


def open_obsm(path: str, key: str, chunk_bytes="256MB", budget: Optional[Budget] = None,
              obs_columns=None, obs_names: bool = True) -> ObsmReader:
    """Cell embeddings (``obsm/<key>``) from an .h5ad, in budgeted cell blocks."""
    return ObsmReader(path, key, chunk_bytes, budget, obs_columns, obs_names)
