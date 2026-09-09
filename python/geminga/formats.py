"""Format readers that live in Python but obey the Rust budget.

Parquet and FASTA/FASTQ are handled by the Rust core. The formats here (h5ad, Arrow IPC,
imaging) lean on mature Python libraries — h5py, pyarrow, tifffile, zarr — because their
value is in the *plan* and the *budget*, not in re-implementing a decoder. Each reader:

  1. produces a `Plan` first (how many chunks, how many bytes each) without reading data,
  2. acquires from the budget before materializing a chunk,
  3. releases when the consumer asks for the next one.

That is the same contract the Rust readers follow, so `for chunk in reader` behaves
identically whether the bytes came from a Parquet row group, a CSR row block or an image tile.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

import numpy as np

from ._core import Budget, human, parse_size


@dataclass
class Plan:
    """What streaming will cost, computed before anything is read."""
    kind: str
    source: str
    n_chunks: int
    bytes_per_chunk: int
    total_bytes: int
    unit: str
    detail: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"Plan({self.kind}, {self.n_chunks} chunks x ~{human(self.bytes_per_chunk)} "
                f"of {self.unit}, {human(self.total_bytes)} total)")

    def fits(self, budget: Budget, tier: str = "ram") -> bool:
        return self.bytes_per_chunk <= budget.available(tier)


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
# h5ad — AnnData on HDF5 (scRNA-seq, scATAC-seq)
# ---------------------------------------------------------------------------

class H5adReader(_Governed):
    """Stream an .h5ad X matrix in row blocks sized to a byte budget.

    Handles CSR/CSC sparse (the usual layout for counts) and dense. Row blocks are cells,
    so a chunk is always a set of whole cells — directly usable for per-cell work.
    Uses h5py's own chunked reads: only the slice requested crosses into memory.
    """

    def __init__(self, path: str, chunk_bytes: str | int = "256MB",
                 budget: Optional[Budget] = None, layer: Optional[str] = None,
                 obs_columns: Optional[Sequence[str]] = None):
        super().__init__(budget)
        import h5py
        self.path = path
        self.f = h5py.File(path, "r")
        self.layer = layer
        node = self.f[f"layers/{layer}"] if layer else self.f["X"]
        self.node = node
        self.obs_columns = list(obs_columns) if obs_columns else []
        enc = node.attrs.get("encoding-type", "")
        enc = enc.decode() if isinstance(enc, bytes) else enc
        self.encoding = enc or ("array" if hasattr(node, "shape") else "csr_matrix")

        if self.encoding in ("csr_matrix", "csc_matrix"):
            self.shape = tuple(node.attrs["shape"])
            self.indptr = node["indptr"][:]           # small: n_rows + 1 int64s
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
        self.chunk_bytes = parse_size(str(chunk_bytes)) if isinstance(chunk_bytes, str) else int(chunk_bytes)
        if budget is not None and chunk_bytes == "256MB":
            self.chunk_bytes = max(1 << 20, budget.available("ram") // 4)

    @property
    def var_names(self) -> np.ndarray:
        for key in ("var/_index", "var/index"):
            if key in self.f:
                v = self.f[key][:]
                return np.array([x.decode() if isinstance(x, bytes) else x for x in v])
        idx = self.f["var"].attrs.get("_index")
        idx = idx.decode() if isinstance(idx, bytes) else idx
        v = self.f[f"var/{idx}"][:]
        return np.array([x.decode() if isinstance(x, bytes) else x for x in v])

    def _rows_per_chunk(self) -> int:
        if self.encoding == "csr_matrix":
            avg_nnz = self.nnz / max(self.n_obs, 1)
            per_row = max(avg_nnz * self._itemsize, 1.0)
        elif self.encoding == "csc_matrix":
            per_row = max(self.nnz / max(self.n_obs, 1) * self._itemsize, 1.0)
        else:
            per_row = max(self.n_vars * self._itemsize, 1.0)
        return max(1, min(self.n_obs, int(self.chunk_bytes / per_row)))

    def plan(self) -> Plan:
        rows = self._rows_per_chunk()
        return Plan(
            kind=f"h5ad/{self.encoding}",
            source=self.path,
            n_chunks=math.ceil(self.n_obs / rows),
            bytes_per_chunk=int(self._bytes_total / max(math.ceil(self.n_obs / rows), 1)),
            total_bytes=self._bytes_total,
            unit="cells",
            detail={"n_obs": self.n_obs, "n_vars": self.n_vars, "nnz": self.nnz,
                    "rows_per_chunk": rows, "dtype": str(self.dtype),
                    "file_size": os.path.getsize(self.path)},
        )

    def __iter__(self) -> Iterator[dict]:
        """Yield {'X': scipy.sparse or ndarray, 'obs_start': int, 'obs_end': int, ...}."""
        import scipy.sparse as sp
        rows = self._rows_per_chunk()
        if self.encoding == "csc_matrix":
            raise NotImplementedError(
                "CSC h5ad is column-major; row-block streaming would touch every column chunk. "
                "Convert to CSR first (anndata: adata.X = adata.X.tocsr()) or stream by var instead."
            )
        for start in range(0, self.n_obs, rows):
            end = min(start + rows, self.n_obs)
            if self.encoding == "csr_matrix":
                lo, hi = int(self.indptr[start]), int(self.indptr[end])
                nbytes = (hi - lo) * self._itemsize
                self._swap_lease(nbytes)
                data = self.node["data"][lo:hi]
                indices = self.node["indices"][lo:hi]
                indptr = self.indptr[start:end + 1] - lo
                X = sp.csr_matrix((data, indices, indptr), shape=(end - start, self.n_vars))
            else:
                nbytes = (end - start) * self.n_vars * self._itemsize
                self._swap_lease(nbytes)
                X = self.node[start:end]
            out = {"X": X, "obs_start": start, "obs_end": end, "n_vars": self.n_vars,
                   "resident_bytes": nbytes}
            for c in self.obs_columns:
                out[c] = self.f[f"obs/{c}"][start:end]
            yield out

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
    """Stream an Arrow IPC file batch by batch, memory-mapped, budget-accounted."""

    def __init__(self, path: str, chunk_bytes: str | int = "256MB",
                 budget: Optional[Budget] = None, columns: Optional[Sequence[str]] = None):
        super().__init__(budget)
        import pyarrow as pa
        self.path, self.columns = path, list(columns) if columns else None
        self._src = pa.memory_map(path, "rb")
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
        self.chunk_bytes = parse_size(str(chunk_bytes)) if isinstance(chunk_bytes, str) else int(chunk_bytes)

    def plan(self) -> Plan:
        size = os.path.getsize(self.path)
        n = self.n_batches if self.n_batches > 0 else max(1, size // self.chunk_bytes)
        return Plan(kind="arrow-ipc", source=self.path, n_chunks=n,
                    bytes_per_chunk=size // max(n, 1), total_bytes=size, unit="rows",
                    detail={"columns": self.schema.names, "random_access": self.is_file})

    def __iter__(self):
        if self.is_file:
            for i in range(self.n_batches):
                b = self.reader.get_batch(i)
                if self.columns:
                    b = b.select(self.columns)
                self._swap_lease(b.get_total_buffer_size())
                yield b
        else:
            for b in self.reader:
                if self.columns:
                    b = b.select(self.columns)
                self._swap_lease(b.get_total_buffer_size())
                yield b

    def close(self):
        super().close()
        try:
            self._src.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Imaging — tiled reads from TIFF / OME-Zarr
# ---------------------------------------------------------------------------

class TiledImageReader(_Governed):
    """Yield (tile, (y0, y1, x0, x1)) windows sized so each tile fits the budget.

    Works on anything exposing `.shape` and numpy slicing: a zarr array, an OME-Zarr
    pyramid level, or a tifffile `aszarr()` store. Overlap is for models that need
    context at tile edges (Mask R-CNN, cellpose, stardist), so masks can be stitched.
    """

    def __init__(self, array, chunk_bytes: str | int = "512MB",
                 budget: Optional[Budget] = None, overlap: int = 0, tier: str = "ram"):
        super().__init__(budget, tier)
        self.a = array
        self.shape = tuple(array.shape)
        self.dtype = np.dtype(array.dtype)
        self.overlap = int(overlap)
        self.chunk_bytes = parse_size(str(chunk_bytes)) if isinstance(chunk_bytes, str) else int(chunk_bytes)
        if budget is not None and chunk_bytes == "512MB":
            self.chunk_bytes = max(1 << 20, budget.available(tier) // 4)
        # Trailing dims (channels) ride along with every tile.
        self.h, self.w = self.shape[0], self.shape[1]
        self.trailing = int(np.prod(self.shape[2:])) if len(self.shape) > 2 else 1
        px = self.dtype.itemsize * self.trailing
        side = int(math.sqrt(max(self.chunk_bytes, 1) / max(px, 1)))
        self.tile = max(64, min(side, max(self.h, self.w)))

    @classmethod
    def from_tiff(cls, path: str, level: int = 0, **kw):
        """Tiled/pyramidal TIFF. Reads individual TIFF tiles directly, so it does not depend
        on tifffile's zarr bridge (which pins a zarr major version)."""
        arr = TiffTileArray(path, level=level)
        r = cls(arr, **kw)
        r._store = arr
        return r

    @classmethod
    def from_zarr(cls, path: str, component: Optional[str] = None, **kw):
        import zarr
        arr = zarr.open(path, mode="r")
        if component:
            arr = arr[component]
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

    def plan(self) -> Plan:
        w = self.windows()
        per = self.tile * self.tile * self.dtype.itemsize * self.trailing
        return Plan(kind="image-tiles", source=str(getattr(self.a, "path", "array")),
                    n_chunks=len(w), bytes_per_chunk=per,
                    total_bytes=int(np.prod(self.shape)) * self.dtype.itemsize, unit="pixels",
                    detail={"shape": self.shape, "dtype": str(self.dtype),
                            "tile": self.tile, "overlap": self.overlap, "windows": len(w)})

    def __iter__(self):
        for (y0, y1, x0, x1) in self.windows():
            nbytes = (y1 - y0) * (x1 - x0) * self.dtype.itemsize * self.trailing
            self._swap_lease(nbytes)
            yield np.asarray(self.a[y0:y1, x0:x1]), (y0, y1, x0, x1)

    def close(self):
        super().close()
        st = getattr(self, "_store", None)
        if st is not None:
            try:
                st.close()
            except Exception:
                pass


class TiffTileArray:
    """Numpy-style 2D slicing over a tiled TIFF, decoding only the tiles a window touches.

    A gigapixel ssDNA or H&E page never becomes resident: `a[y0:y1, x0:x1]` reads and decodes
    exactly the TIFF tiles that intersect the window. Striped (non-tiled) TIFFs fall back to
    row-strip reads via tifffile. Works with any zarr version because it uses none.
    """

    def __init__(self, path: str, level: int = 0):
        import tifffile
        self.path = path
        self._tf = tifffile.TiffFile(path)
        series = self._tf.series[0]
        self.page = series.levels[level].pages[0] if hasattr(series, "levels") else series.pages[0]
        self.shape = tuple(self.page.shape[:2])
        self.dtype = np.dtype(self.page.dtype)
        self.tiled = bool(self.page.is_tiled)
        if self.tiled:
            self.th, self.tw = int(self.page.tilelength), int(self.page.tilewidth)
            self.ny, self.nx = self.page.chunked[0], self.page.chunked[1]
        else:
            self.th = int(getattr(self.page, "rowsperstrip", self.shape[0]) or self.shape[0])
            self.tw = self.shape[1]
            self.ny = math.ceil(self.shape[0] / self.th)
            self.nx = 1

    def _tile(self, ty: int, tx: int) -> np.ndarray:
        i = ty * self.nx + tx
        off, cnt = int(self.page.dataoffsets[i]), int(self.page.databytecounts[i])
        if cnt == 0:
            return np.zeros((self.th, self.tw), self.dtype)
        fh = self._tf.filehandle
        fh.seek(off)
        data, _, shape = self.page.decode(fh.read(cnt), i)
        a = np.asarray(data)
        # decode gives (depth, length, width, samples); take the 2D plane
        a = a.reshape(shape)[0, :, :, 0] if a.ndim == 4 else np.squeeze(a)
        return a.astype(self.dtype, copy=False)

    def __getitem__(self, key):
        ys, xs = key if isinstance(key, tuple) else (key, slice(None))
        y0, y1, _ = ys.indices(self.shape[0]) if isinstance(ys, slice) else (ys, ys + 1, 1)
        x0, x1, _ = xs.indices(self.shape[1]) if isinstance(xs, slice) else (xs, xs + 1, 1)
        out = np.zeros((y1 - y0, x1 - x0), self.dtype)
        for ty in range(y0 // self.th, (y1 - 1) // self.th + 1):
            for tx in range(x0 // self.tw, (x1 - 1) // self.tw + 1):
                t = self._tile(ty, tx)
                ty0, tx0 = ty * self.th, tx * self.tw
                sy0, sx0 = max(y0, ty0), max(x0, tx0)
                sy1 = min(y1, ty0 + t.shape[0])
                sx1 = min(x1, tx0 + t.shape[1])
                if sy1 <= sy0 or sx1 <= sx0:
                    continue
                out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                    t[sy0 - ty0:sy1 - ty0, sx0 - tx0:sx1 - tx0]
        return out

    def close(self):
        try:
            self._tf.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Sinks — write results out as they are produced, never accumulate
# ---------------------------------------------------------------------------

class ParquetSink:
    """Append RecordBatches / tables to one Parquet file without holding them."""

    def __init__(self, path: str, compression: str = "zstd"):
        self.path, self.compression, self._w = path, compression, None
        self.rows = 0

    def write(self, batch):
        import pyarrow as pa, pyarrow.parquet as pq
        if isinstance(batch, pa.RecordBatch):
            tbl = pa.Table.from_batches([batch])
        elif isinstance(batch, pa.Table):
            tbl = batch
        else:
            tbl = pa.table(batch)
        if self._w is None:
            self._w = pq.ParquetWriter(self.path, tbl.schema, compression=self.compression)
        self._w.write_table(tbl)
        self.rows += tbl.num_rows

    def close(self):
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
