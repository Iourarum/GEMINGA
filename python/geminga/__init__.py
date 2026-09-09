"""GEMINGA — Memory-Efficient Arrow-Native Dataset Explorer & Reader.

Stream remote Parquet datasets (Hugging Face Hub first) with HTTP range requests:
inspect, sample, visualize, and feed training loops without downloading the dataset.

    import geminga as md
    ds = md.open_hub("HuggingFaceTB/smoltalk2", config="Mid", split="train")
    md.print_summary(ds)                      # footer-only: schema, rows, per-column min/max/nulls
    tbl = md.sample_table(ds, n_row_groups=8) # a spread-out sample as a pyarrow.Table
    for batch in ds.stream(columns=["text"], limit=100_000): ...
"""
from __future__ import annotations

from typing import Iterable, Iterator, Optional, Sequence

import pyarrow as pa

from ._core import (
    BatchIterator, Budget, DATASETS_SERVER, Dataset, FastxStream, Lease,
    __version__, human, parse_size,
)
from .formats import (
    TiffTileArray,
    ArrowIPCReader, H5adReader, MaskSink, ParquetSink, Plan, TiledImageReader,
)

__all__ = [
    # core
    "Dataset", "BatchIterator", "Budget", "Lease", "FastxStream", "Plan",
    "DATASETS_SERVER", "__version__", "human", "parse_size",
    # openers
    "open", "open_hub", "open_urls", "open_paths", "open_fastx", "open_h5ad",
    "open_arrow", "open_image",
    # readers / sinks
    "H5adReader", "ArrowIPCReader", "TiledImageReader", "TiffTileArray", "ParquetSink", "MaskSink",
    # helpers
    "to_table", "sample_table", "stream_frames", "iter_numpy", "iter_torch",
    "print_summary", "print_plan",
]

_FASTX = (".fq", ".fastq", ".fa", ".fasta", ".fna", ".ffn", ".faa", ".frn")
_IMAGE = (".tif", ".tiff", ".ome.tif", ".ome.tiff")


def open(target, budget: "Budget | None" = None, **kw):
    """Open anything: a Hub id, a URL, or a local path — dispatched on what it is.

    Hub id ("org/name")            -> Dataset          (remote Parquet, range-streamed)
    http(s):// URL                 -> Dataset
    *.parquet                      -> Dataset
    *.fastq/.fq/.fasta[.gz]        -> FastxStream      (Rust parser, byte-chunked)
    *.h5ad                         -> H5adReader       (CSR row blocks)
    *.arrow/.feather/.ipc          -> ArrowIPCReader
    *.tif/.tiff (incl. OME)        -> TiledImageReader (tiles sized to the budget)
    *.zarr                         -> TiledImageReader
    """
    import os as _os
    if isinstance(target, (list, tuple)):
        paths = list(target)
        if all(str(p).startswith("http") for p in paths):
            return open_urls(paths, **kw)
        ds = open_paths(paths, **kw)
        if budget is not None:
            ds.set_budget(budget)
        return ds
    t = str(target)
    low = t.lower()
    if low.startswith("http://") or low.startswith("https://"):
        ds = open_urls([t], **kw)
    elif not _os.path.exists(t) and t.count("/") == 1:
        ds = open_hub(t, **kw)
    elif low.endswith(".parquet") or _os.path.isdir(t) and not low.endswith(".zarr"):
        ds = open_paths([t], **kw)
    elif low.endswith(".gz") and any(low[:-3].endswith(e) for e in _FASTX) or low.endswith(_FASTX):
        return open_fastx(t, budget=budget, **kw)
    elif low.endswith(".h5ad"):
        return open_h5ad(t, budget=budget, **kw)
    elif low.endswith((".arrow", ".feather", ".ipc")):
        return open_arrow(t, budget=budget, **kw)
    elif low.endswith(_IMAGE) or low.endswith(".zarr"):
        return open_image(t, budget=budget, **kw)
    else:
        raise ValueError(f"don't know how to open {t!r}; use open_paths/open_fastx/open_h5ad/open_image explicitly")
    if budget is not None:
        ds.set_budget(budget)
    return ds


def open_fastx(path: str, chunk_bytes: str = "64MB", max_records: int = 0,
               budget: "Budget | None" = None) -> "FastxStream":
    """FASTA/FASTQ (optionally gzipped) in byte-sized chunks, parsed in Rust."""
    return FastxStream(path, chunk_bytes, max_records, budget)


def open_h5ad(path: str, chunk_bytes="256MB", budget: "Budget | None" = None,
              layer: "str | None" = None, obs_columns=None) -> "H5adReader":
    """scRNA-seq / scATAC-seq .h5ad in cell blocks."""
    return H5adReader(path, chunk_bytes, budget, layer, obs_columns)


def open_arrow(path: str, chunk_bytes="256MB", budget: "Budget | None" = None,
               columns=None) -> "ArrowIPCReader":
    return ArrowIPCReader(path, chunk_bytes, budget, columns)


def open_image(path: str, chunk_bytes="512MB", budget: "Budget | None" = None,
               overlap: int = 0, level: int = 0, component=None, tier: str = "ram"):
    """Tiled imaging reader for TIFF/OME-TIFF (pyramid `level`) or Zarr."""
    if str(path).lower().endswith(".zarr"):
        return TiledImageReader.from_zarr(path, component=component, chunk_bytes=chunk_bytes,
                                          budget=budget, overlap=overlap, tier=tier)
    return TiledImageReader.from_tiff(path, level=level, chunk_bytes=chunk_bytes,
                                      budget=budget, overlap=overlap, tier=tier)


def print_plan(obj, chunk_bytes: "str | None" = None, columns=None) -> None:
    """What streaming this would cost, before reading any data."""
    if isinstance(obj, Dataset):
        p = obj.plan(chunk_bytes, list(columns) if columns else None)
        print(f"parquet plan: {p['chunks']} chunks x {p['rows_per_chunk']:,} rows "
              f"(~{p['chunk_bytes_human']} resident), {p['total_rows']:,} rows total, "
              f"{p['uncompressed_total_human']} uncompressed")
    else:
        p = obj.plan()
        print(p)
        for k, v in p.detail.items():
            print(f"    {k}: {v}")


def open_hub(dataset: str, config: Optional[str] = None, split: Optional[str] = None,
             token: Optional[str] = None) -> Dataset:
    """Resolve a Hub dataset to its Parquet files (refs/convert/parquet). Nothing is downloaded."""
    return Dataset.from_hub(dataset, config, split, token)


def open_urls(urls: Sequence[str], token: Optional[str] = None,
              sizes: Optional[Sequence[int]] = None) -> Dataset:
    return Dataset.from_urls(list(urls), token, list(sizes) if sizes is not None else None)


def open_paths(paths: Sequence[str]) -> Dataset:
    return Dataset.from_paths(list(paths))


def to_table(batches: Iterable[pa.RecordBatch]) -> pa.Table:
    batches = list(batches)
    if not batches:
        return pa.table({})
    return pa.Table.from_batches(batches)


def sample_table(ds: Dataset, n_row_groups: int = 8, seed: Optional[int] = None,
                 columns: Optional[Sequence[str]] = None, batch_size: int = 65_536) -> pa.Table:
    """Row groups spread across the whole dataset, as one Table — the 'look at it in detail' path."""
    return to_table(ds.sample(n_row_groups, seed, list(columns) if columns else None, batch_size))


def stream_frames(ds: Dataset, backend: str = "pyarrow", **kw) -> Iterator:
    """Yield batches converted for a given backend: 'pyarrow' | 'pandas' | 'polars' | 'numpy'."""
    it = ds.stream(**kw)
    if backend == "pyarrow":
        yield from it
    elif backend == "pandas":
        for b in it:
            yield b.to_pandas()
    elif backend == "polars":
        import polars as pl
        for b in it:
            yield pl.from_arrow(b)
    elif backend == "numpy":
        yield from iter_numpy(ds, **kw)
    else:
        raise ValueError(f"unknown backend {backend!r}")


def iter_numpy(ds: Dataset, **kw) -> Iterator[dict]:
    """Yield {column: np.ndarray}; zero-copy for fixed-width columns without nulls."""
    for b in ds.stream(**kw):
        out = {}
        for name, col in zip(b.schema.names, b.columns):
            try:
                out[name] = col.to_numpy(zero_copy_only=True)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                out[name] = col.to_numpy(zero_copy_only=False)
        yield out


def iter_torch(ds: Dataset, **kw) -> Iterator[dict]:
    """Yield {column: torch.Tensor} for numeric columns (strings are passed through as lists)."""
    import numpy as np
    import torch
    for arrays in iter_numpy(ds, **kw):
        out = {}
        for name, arr in arrays.items():
            if isinstance(arr, np.ndarray) and arr.dtype.kind in "biuf":
                out[name] = torch.from_numpy(np.ascontiguousarray(arr))
            else:
                out[name] = arr.tolist()
        yield out


def _fmt_bytes(n) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def print_summary(ds: Dataset, max_files: int = 5) -> None:
    """Human-readable overview built from footers only — no data pages are transferred."""
    d = ds.describe()
    print(f"{d['origin']}")
    size = _fmt_bytes(d['total_size_bytes']) if d['total_size_bytes'] else "?"
    print(f"  files: {d['num_files']}   rows: {d['total_rows']:,}   row groups: {d['total_row_groups']}   "
          f"size on disk: {size}{'   (partial listing)' if d['partial'] else ''}")
    print("  schema:")
    for f in d["schema"]:
        print(f"    {f}")
    for fi in d["files"][:max_files]:
        print(f"  - {fi['file']}: {fi['num_rows']:,} rows, {fi['num_row_groups']} row groups, "
              f"{_fmt_bytes(fi['size_bytes'])}, created_by={fi['created_by']}")
    if d["num_files"] > max_files:
        print(f"  ... {d['num_files'] - max_files} more file(s)")
    print("  columns (min/max/nulls from row-group statistics):")
    for s in ds.stats():
        mn, mx = s.get("min", s.get("min_len")), s.get("max", s.get("max_len"))
        span = f"{mn!r} .. {mx!r}" if mn is not None else "no statistics"
        print(f"    {s['column']:<24} {s['physical_type']:<12} nulls={s['null_count']}  "
              f"{_fmt_bytes(s['compressed_bytes'])} compressed   {span}")
