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
from .view import (
    CostError, Extent, ImageView, Overview, ProgressiveCanvas, RowView, SeqView, View,
)
from ._core import (
    GemIndex, GemStream, SeqIndex, gem_extent, gem_header, revcomp, search_fastx,
)
from . import render
from .ngff import read_ome_zarr_meta, to_ome_zarr
from .obsm import ObsmReader, obsm_keys, open_obsm
from .gefindex import SpatialIndex, build_spatial_index, morton2
from .selections import (
    INTENTS, Selection, SelectionBook, dataset_key, describe, external_queries, leads,
    local_leads, select, select_polygon, select_values, selections, similar,
)
from .library import (
    STATUSES, Library, describe_any, flag_cells, flag_genes, flag_locus, flag_region,
    flag_sequence, flag_values, flag_variant, leads_any, library, parse_locus,
)
from .remote import (
    RemoteEntry, RemoteFile, fetch_selected, list_tree, open_remote_gef,
    print_tree_summary, summarise_tree, wget_mirror_command,
)
from .frames import (
    POLARS_AVAILABLE, frame_bytes, from_polars, lazy_window, narrow_arrow, scan,
    sink_parquet, to_polars,
)
from .workspace import Cell, CellState, Workspace, workspace
from .profile import ASSAYS, Evidence, Hypothesis, Profile, profile
from .catalog import CATALOG_SCHEMA, FACETS, Catalog, CatalogEntry, catalog
from .assist import Assistant, AssistantReply, PROTOCOL_NOTES, assistant
from .staging import Staging, StagedRegion, StagingFull, staging
from .vault import Vault, VaultMember, vault
from .devices import Device, DevicePool, describe_devices, device_pool, devices
from .handoff import FORMATS, KINDS, Harvest, ROI, harvest
from .codec import (
    BitpackCodec, BytesCodec, Codec, CodecError, CompressedStore, DeviceArray, Encoded, StoreEntry,
    available_codecs, bitpack, bytes_codec, check_codec, compressed_store, decode_csr, encode_csr,
    get_codec, ratio_report, register_codec,
)
from .prep import Prep, PrepResult, prep
from .journal import (
    JOURNAL_SCHEMA, DatasetTrail, Journal, JournalEntry, journal, journal_recording, use_journal,
)
from .reserve import Policy, Reservation, ReservationLost, ReservationServer, reserve, serve
from .harmonize import (
    ASSAY_LABELS, OBS_COLUMNS, PERTURBATION_TYPES, Alignment, ColumnMap, GeneMap, Harmonizer,
    Vocabulary, id_type,
)
from .orchestrate import (
    BLUEPRINT_SCHEMA, RESOLUTION_STATUSES, Blueprint, Coverage, Orchestrator, Piece, Puzzle, Resolution,
    find_accessions, orchestrator, read_descriptions, resolve_pieces,
)
from .acquire import (
    EGRESS_USD_PER_GB, AcquisitionPlan, Route, available_tools, classify, command_for,
    fetch, peek, plan_many, print_shortlist, print_tools, resolve,
)
from .acquire import plan as acquisition_plan
from .tenx import (
    FEATURE_TYPES, ModalityReport, MtxTriple, TenXMatrix, TenXView, modalities, open_tenx,
)
from .spatial import (
    MultiChannelTiff, VisiumBundle, XeniumBundle, open_visium, open_xenium, sniff_bundle,
)
from .neuro import (
    Cohort, DicomSeries, MaskSink3D, NiftiVolume, Subject, VolumeView, guess_sequence,
    open_volume,
)
from .stereoseq import (
    BIN_LEVELS, GefFile, GemView, StereoSeqDataset, StereoSeqView,
    index_gem, open_gef, open_gem, open_stereoseq,
)
from .formats import (
    TiffTileArray,
    ArrowIPCReader, H5adReader, MaskSink, ParquetSink, Plan, TiledImageReader,
)

# The Rust core stamps __version__ at compile time and can lag the package when only the Python
# side changed; the installed distribution's version is the one the package can vouch for.
try:
    from importlib.metadata import version as _dist_version
    __version__ = _dist_version("geminga")
except Exception:                                # a source checkout: the core's stamp
    pass

__all__ = [
    # core
    "Dataset", "BatchIterator", "Budget", "Lease", "FastxStream", "Plan",
    "DATASETS_SERVER", "__version__", "human", "parse_size",
    # openers
    "open", "open_hub", "open_urls", "open_paths", "open_fastx", "open_h5ad",
    "open_arrow", "open_image",
    # readers / sinks
    "H5adReader", "ArrowIPCReader", "TiledImageReader", "TiffTileArray", "ParquetSink", "MaskSink",
    # navigation
    "View", "RowView", "SeqView", "ImageView", "Overview", "Extent",
    "ProgressiveCanvas", "CostError", "SeqIndex", "index_fastx", "view",
    "search_fastx", "revcomp", "render",
    # polars
    "to_polars", "from_polars", "narrow_arrow", "scan", "lazy_window", "sink_parquet",
    "frame_bytes",
    "POLARS_AVAILABLE",
    # wgpu stage
    "Stage", "Camera", "TextAtlas", "stage_available", "adapter_info",
    "Explorer", "Event", "Telemetry", "explore",
    "to_ome_zarr", "read_ome_zarr_meta",
    "ObsmReader", "obsm_keys", "open_obsm", "CATALOG_SCHEMA", "FACETS", "KINDS",
    "SpatialIndex", "build_spatial_index", "morton2",
    "RemoteFile", "RemoteEntry", "list_tree", "summarise_tree", "print_tree_summary",
    "fetch_selected", "open_remote_gef", "wget_mirror_command",
    "Selection", "SelectionBook", "selections", "select", "select_polygon", "select_values",
    "describe", "similar", "leads", "local_leads", "external_queries", "dataset_key", "INTENTS",
    "Library", "library", "flag_locus", "flag_variant", "flag_sequence", "flag_cells",
    "flag_genes", "flag_values", "flag_region", "describe_any", "leads_any", "parse_locus",
    "STATUSES",
    # Stereo-seq
    "GefFile", "StereoSeqView", "GemView", "StereoSeqDataset", "BIN_LEVELS",
    "open_gef", "open_gem", "open_stereoseq", "index_gem",
    "GemIndex", "GemStream", "gem_header", "gem_extent",
    # workspace, profiling, catalog, assistant
    "Workspace", "Cell", "CellState", "workspace",
    "Profile", "Evidence", "Hypothesis", "profile", "ASSAYS",
    "Catalog", "CatalogEntry", "catalog",
    "Assistant", "AssistantReply", "assistant", "PROTOCOL_NOTES",
    # oversized local storage: vault -> staging -> device -> pipeline
    "Vault", "VaultMember", "vault",
    "Staging", "StagedRegion", "StagingFull", "staging",
    "Device", "DevicePool", "devices", "device_pool", "describe_devices",
    "Harvest", "ROI", "harvest", "FORMATS",
    # acquisition: route planning, pricing, peeking, delegation
    "Route", "AcquisitionPlan", "acquisition_plan", "resolve", "peek", "fetch",
    "plan_many", "print_shortlist", "available_tools", "print_tools", "command_for",
    "classify", "EGRESS_USD_PER_GB",
    # 10x and spatial bundles
    "TenXMatrix", "MtxTriple", "TenXView", "ModalityReport", "modalities", "open_tenx",
    "FEATURE_TYPES", "VisiumBundle", "XeniumBundle", "MultiChannelTiff", "open_visium",
    "open_xenium", "sniff_bundle",
    # neuroimaging
    "NiftiVolume", "DicomSeries", "VolumeView", "MaskSink3D", "Subject", "Cohort",
    "open_volume", "guess_sequence",
    # compressed resident blocks and shared-GPU reservations
    "Codec", "Encoded", "BitpackCodec", "BytesCodec", "CompressedStore", "StoreEntry", "DeviceArray",
    "bitpack", "bytes_codec", "compressed_store", "encode_csr", "decode_csr", "ratio_report",
    "check_codec", "register_codec", "get_codec", "available_codecs", "CodecError",
    "ReservationServer", "Reservation", "ReservationLost", "Policy", "reserve", "serve",
    # the standard preprocessing, out of core
    "Prep", "PrepResult", "prep",
    # the journal of streamed pieces
    "JOURNAL_SCHEMA", "DatasetTrail", "Journal", "JournalEntry", "journal", "journal_recording",
    "use_journal",
    # orchestration: descriptions -> pieces -> resolution -> harmonised, annotated builds
    "Orchestrator", "orchestrator", "Piece", "Resolution", "Blueprint", "Puzzle", "Coverage",
    "read_descriptions", "find_accessions", "resolve_pieces", "BLUEPRINT_SCHEMA", "RESOLUTION_STATUSES",
    "Harmonizer", "Vocabulary", "GeneMap", "Alignment", "ColumnMap", "id_type",
    "ASSAY_LABELS", "PERTURBATION_TYPES", "OBS_COLUMNS",
    # helpers
    "to_table", "sample_table", "stream_frames", "iter_numpy", "iter_torch",
    "print_summary", "print_plan", "print_cost",
]


def index_fastx(path: str, n_buckets: int = 2048, sidecar: "str | None" = None,
                reuse: bool = True) -> "SeqIndex":
    """Build (or reuse) the level-of-detail sidecar that makes a sequence file navigable.

    One pass over the file. Afterwards, zooming and cost estimation are free, and loading a
    window skips straight to it.
    """
    import os as _os
    sc = sidecar or (path + ".gemidx.json")
    if reuse and _os.path.exists(sc) and _os.path.getmtime(sc) >= _os.path.getmtime(path):
        try:
            return SeqIndex.load(sc)
        except Exception:
            pass
    idx = SeqIndex.build(path, n_buckets)
    try:
        idx.save(sc)
    except OSError:
        pass
    return idx


def view(target, budget: "Budget | None" = None, columns=None, **kw):
    """A zoomable, cost-aware view over anything GEMINGA can open.

        v = g.view("atlas.h5ad", budget=b)
        v.cost()                      # what would this take?  nothing loaded yet
        v.select_frac(0.4, 0.45).auto_zoom()
        v.overview().sparkline()
        data = v.load()               # budget-enforced
    """
    import os as _os
    if isinstance(target, _os.PathLike):    # pathlib.Path and friends: the path they name
        target = _os.fspath(target)
    t = str(target)
    if "::obsm/" in t:                      # "atlas.h5ad::obsm/X_geneformer"
        t, key = t.split("::obsm/", 1)
        kw.setdefault("obsm", key)
    low = t.lower()
    if isinstance(target, str) and (low.endswith(_FASTX) or
                                    (low.endswith(".gz") and any(low[:-3].endswith(e) for e in _FASTX))):
        return SeqView(index_fastx(t, **kw), t, budget)
    if isinstance(target, SeqIndex):
        return SeqView(target, None, budget)
    if isinstance(target, str) and (low.endswith(_IMAGE) or low.endswith(".zarr")):
        return ImageView(open_image(t, budget=budget, **kw).a, budget)
    if isinstance(target, TiledImageReader):
        return ImageView(target.a, budget)
    if isinstance(target, TiffTileArray):
        return ImageView(target, budget)
    if isinstance(target, str) and (low.endswith((".nii", ".nii.gz", ".dcm", ".dicom", ".ima"))
                                    or (_os.path.isdir(t) and any(
                                        f.lower().endswith((".dcm", ".ima"))
                                        for f in _os.listdir(t)[:50]))):
        return VolumeView(open_volume(t), budget)
    if isinstance(target, str) and low.endswith(".h5") and not low.endswith(".h5ad"):
        return open_tenx(t, budget=budget, **kw)
    if isinstance(target, str) and low.endswith(".gef"):
        return open_gef(t, budget=budget, **kw)
    if isinstance(target, str) and (low.endswith(".gem") or low.endswith(".gem.gz")):
        return open_gem(t, budget=budget, **kw)
    if isinstance(target, str) and low.endswith(".h5ad"):
        return RowView(open_h5ad(t, budget=budget, **kw), columns, budget, kind="cells")
    if isinstance(target, H5adReader):
        return RowView(target, columns, budget, kind="cells")
    if isinstance(target, Dataset):
        if budget is not None:
            target.set_budget(budget)
        return RowView(target, columns, budget)
    ds = open(target, budget=budget, **kw)
    if isinstance(ds, Dataset):
        return RowView(ds, columns, budget)
    if isinstance(ds, H5adReader):
        return RowView(ds, columns, budget, kind="cells")
    raise ValueError(f"no view for {target!r}")


def print_cost(v) -> None:
    """Human-readable answer to 'what will this cost me?' — before loading anything."""
    c = v.cost()
    win = " x ".join(f"{c['window'][i]:,}-{c['window'][i+1]:,}" for i in range(0, len(c['window']), 2))
    print(f"window {win}  zoom {c['zoom']} (stride {c['stride']})")
    print(f"  {c['units']:,} {c['unit']} — {c['fraction_of_dataset']:.2%} of the dataset")
    print(f"  {c['chunks']} chunk(s) x {c['resident_human']} resident, {c['total_human']} in total")
    if "decode_human" in c:
        print(f"  {c['tiles_touched']} tiles must be decoded whole: {c['decode_human']} touched")
    if c.get("fits_whole") is None:
        print("  no budget attached — nothing is enforced")
    elif c["fits_whole"]:
        print(f"  .load() fits: {c['budget_available_human']} free")
    elif c["fits"]:
        print(f"  .load() would not fit ({c['budget_available_human']} free) "
              f"— stream it with .chunks() instead")
    else:
        print(f"  DOES NOT FIT — {c['advice']}")

_FASTX = (".fq", ".fastq", ".fa", ".fasta", ".fna", ".ffn", ".faa", ".frn")
_IMAGE = (".tif", ".tiff", ".ome.tif", ".ome.tiff")


def open(target, budget: "Budget | None" = None, **kw):
    """Open anything: a Hub id, a URL, or a local path — dispatched on what it is.

    ::

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
    if isinstance(target, _os.PathLike):    # pathlib.Path and friends: the path they name
        target = _os.fspath(target)
    if isinstance(target, (list, tuple)):
        paths = [_os.fspath(p) if isinstance(p, _os.PathLike) else p for p in target]
        if all(str(p).startswith("http") for p in paths):
            return open_urls(paths, **kw)
        ds = open_paths(paths, **kw)
        if budget is not None:
            ds.set_budget(budget)
        return ds
    t = str(target)
    if "::obsm/" in t:                      # "atlas.h5ad::obsm/X_geneformer"
        t, key = t.split("::obsm/", 1)
        kw.setdefault("obsm", key)
    low = t.lower()
    if low.startswith("http://") or low.startswith("https://"):
        ds = open_urls([t], **kw)
    elif not _os.path.exists(t) and t.count("/") == 1:
        ds = open_hub(t, **kw)
    elif _os.path.isdir(t) and not low.endswith(".zarr"):
        # A directory is a bundle first, and a pile of Parquet files only if it is not one.
        b = sniff_bundle(t)
        if b["kind"] in ("visium", "visium-hd"):
            return open_visium(t, budget=budget, **kw)
        if b["kind"] == "xenium":
            return open_xenium(t, budget=budget)
        if b["kind"] == "cellranger" and b.get("matrix"):
            return open_tenx(b["matrix"], budget=budget, **kw)
        if b["kind"] == "mtx-triple":
            return MtxTriple(b["matrix"], budget=budget)
        if any(f.lower().endswith((".dcm", ".ima")) for f in _os.listdir(t)[:50]):
            return VolumeView(open_volume(t), budget)
        ds = open_paths([t], **kw)
    elif low.endswith(".parquet"):
        ds = open_paths([t], **kw)
    elif low.endswith(".gz") and any(low[:-3].endswith(e) for e in _FASTX) or low.endswith(_FASTX):
        return open_fastx(t, budget=budget, **kw)
    elif low.endswith(".h5") and not low.endswith(".h5ad"):
        return open_tenx(t, budget=budget, **kw)
    elif low.endswith((".nii", ".nii.gz", ".dcm", ".dicom", ".ima")):
        return VolumeView(open_volume(t), budget)
    elif low.endswith(".gef"):
        return open_gef(t, budget=budget, **kw)
    elif low.endswith(".gem") or low.endswith(".gem.gz"):
        return open_gem(t, budget=budget, **kw)
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
              layer: "str | None" = None, obs_columns=None, obsm: "str | None" = None,
              obs_names: bool = True):
    """scRNA-seq / scATAC-seq .h5ad in cell blocks; `obsm=` streams an embedding instead."""
    if obsm is not None:
        return ObsmReader(path, obsm, chunk_bytes, budget, obs_columns, obs_names)
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


def __getattr__(name):
    """`Stage`, `Camera`, `TextAtlas` and friends load wgpu only when first touched, so the
    package imports cleanly on a machine with no graphics stack at all."""
    if name in ("Stage", "Camera", "TextAtlas"):
        from . import stage as _s
        return getattr(_s, name)
    if name in ("Explorer", "Event", "Telemetry", "explore"):
        from . import explorer as _e
        return getattr(_e, name)
    if name == "stage_available":
        from .stage import available
        return available
    if name == "adapter_info":
        from .stage import adapter_info
        return adapter_info
    raise AttributeError(f"module 'geminga' has no attribute {name!r}")
