"""OME-Zarr (NGFF v0.4) export, so the interactive viewers can take over.

GEMINGA should not grow an image viewer. napari streams multiscale OME-Zarr lazily, loading
only what it displays; neuroglancer does the same in a browser; Vitessce reads it for spatial
omics. All three agree on one format, so the most useful thing GEMINGA can do for images is
write it well.

Two paths, because the sources differ in a way that matters:

**Stereo-seq GEF.** The file already *is* a pyramid — SAW wrote bin1, bin10, ... bin200. Each
level is rasterised at one pixel per bin and written as its own NGFF dataset, with the bin size
as the scale. Nothing is downsampled by GEMINGA; the instrument's own aggregation becomes the
viewer's zoom stack. Levels whose raster would be too large are skipped rather than silently
downsampled, and the skip is reported. Row `i` of a level is the bin whose y origin is
`y0 + i * bin` and column `j` the bin at `x0 + j * bin` — the same convention as an image, whose
row 0 is the first row of the registered picture — so a GEF export and an image export of the
same chip overlay without a flip.

**Images and volumes.** Level 0 is copied band by band, and each coarser level is built from
the one just written by 2x pooling, also band by band. Peak memory is a few bands, not the
image — the same bounded-resident promise as every reader. Pooling keeps the source dtype:
intensities are averaged and rounded (integers) so a uint16 image stays uint16 at every level,
and label images are subsampled (nearest) because the mean of two label ids is a third label.
An RGB(A) image is written with a channel axis, `(c, y, x)`, which is where NGFF v0.4 puts it.

NGFF v0.4 is written with zarr format 2, which is what napari-ome-zarr, neuroglancer and
ome-zarr-py all read today.

Nothing here deletes what it did not write. A target that exists is refused unless it is empty,
or (with `overwrite=True`) already an OME-Zarr of ours holding nothing but zarr data.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
from typing import Optional, Sequence

import numpy as np

from ._core import Budget, human, parse_size

__all__ = ["to_ome_zarr", "read_ome_zarr_meta", "OverwriteRefused"]

#: Stereo-seq DNB pitch — one GEF coordinate unit — in micrometres.
STEREOSEQ_UNIT_UM = 0.5


class OverwriteRefused(FileExistsError, ValueError):
    """The target already exists and writing to it would destroy something this library did
    not make (or something it made but the caller did not ask to replace)."""


# ---------------------------------------------------------------------------
# Refusing to delete what is not ours
# ---------------------------------------------------------------------------

_ZARR_META = (".zgroup", ".zarray", ".zattrs", ".zmetadata", "zarr.json")
_CHUNK_NAME = re.compile(r"\d+(\.\d+)*")


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def zarr_node(path: str):
    """`(kind, attrs)` for the zarr node stored at `path`: kind is 'array' or 'group', attrs the
    user attributes (v2 `.zattrs` or v3 `zarr.json`). `None` if `path` is not a zarr node.
    Reads the metadata files directly, so it works whatever zarr version is installed."""
    z3 = _read_json(os.path.join(path, "zarr.json"))
    if isinstance(z3, dict) and z3.get("node_type") in ("array", "group"):
        return z3["node_type"], dict(z3.get("attributes") or {})
    for marker, kind in ((".zarray", "array"), (".zgroup", "group")):
        if os.path.isfile(os.path.join(path, marker)):
            return kind, dict(_read_json(os.path.join(path, ".zattrs")) or {})
    return None


def _carries(attrs: dict, markers: Sequence[str]) -> bool:
    ome = attrs.get("ome")                                   # NGFF 0.5 nests its keys under "ome"
    return any(m in attrs or (isinstance(ome, dict) and m in ome) for m in markers)


def _foreign_entries(path: str, kind: str) -> list:
    """Top-level entries of a zarr node that are not zarr data — files somebody else put there."""
    foreign = []
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.islink(full):
            foreign.append(name)
        elif name in _ZARR_META:
            continue
        elif os.path.isdir(full):
            if kind == "array":
                ok = name == "c" or _CHUNK_NAME.fullmatch(name)        # v3 chunks / v2 "a/b/c"
            else:
                ok = any(os.path.exists(os.path.join(full, m)) for m in (".zarray", ".zgroup", "zarr.json"))
            if not ok:
                foreign.append(name)
        elif not (kind == "array" and _CHUNK_NAME.fullmatch(name)):    # v2 chunk files "0.1.2"
            foreign.append(name)
    return sorted(foreign)


def plan_target(path: str, overwrite: bool, *, markers: Sequence[str], what: str,
                reuse: bool = False) -> str:
    """Decide what writing to `path` may do. Returns 'create', 'reuse' or 'replace', or raises
    `OverwriteRefused` — deleting nothing, ever, on the way to the answer.

    * missing, or an empty directory                    -> 'create'
    * a zarr node of ours (an attribute in `markers`)   -> 'reuse' if `reuse` and not
                                                          `overwrite`, 'replace' if `overwrite`
                                                          and it holds only zarr data
    * anything else (foreign files, a plain zarr group, a file, a symlink) is refused, with or
      without `overwrite`.
    """
    path = os.fspath(path)
    if not os.path.lexists(path):
        return "create"
    if os.path.islink(path):
        raise OverwriteRefused(f"{path} is a symbolic link; refusing to write through or delete it")
    if not os.path.isdir(path):
        raise OverwriteRefused(f"{path} exists and is not a directory; refusing to touch it")
    entries = os.listdir(path)
    if not entries:
        return "create"
    node = zarr_node(path)
    if node is None or not _carries(node[1], markers):
        raise OverwriteRefused(
            f"{path} is a non-empty directory that is not {what} ({', '.join(sorted(entries)[:3])}"
            f"{', ...' if len(entries) > 3 else ''}); refusing to write into or delete it. "
            f"Choose a new path, or empty the directory yourself.")
    if not overwrite:
        if reuse:
            return "reuse"
        raise OverwriteRefused(f"{path} already holds {what}; pass overwrite=True to replace it")
    foreign = _foreign_entries(path, node[0])
    if foreign:
        raise OverwriteRefused(
            f"{path} holds {what} but also files that are not zarr data ({', '.join(foreign[:3])}"
            f"{', ...' if len(foreign) > 3 else ''}); overwrite=True would delete them. "
            f"Move them out of the way or choose a new path.")
    return "replace"


def clear_zarr_members(path: str) -> None:
    """Empty a directory `plan_target` approved for replacement (zarr data only)."""
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isdir(full) and not os.path.islink(full):
            shutil.rmtree(full)
        else:
            os.remove(full)


def _group(path: str, overwrite: bool):
    import zarr
    if plan_target(path, overwrite, markers=("multiscales",), what="an OME-Zarr") == "replace":
        clear_zarr_members(path)
    # "w-" never deletes: zarr itself refuses if anything is in the way.
    try:
        return zarr.open_group(path, mode="w-", zarr_format=2)
    except TypeError:                                           # zarr 2.x
        return zarr.open_group(path, mode="w-")


def _array(grp, name, shape, chunks, dtype):
    try:
        return grp.create_array(name, shape=shape, chunks=chunks, dtype=dtype,
                                fill_value=0)
    except AttributeError:                                      # zarr 2.x
        return grp.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype,
                                  fill_value=0)


def _axes(layout: str, unit: str) -> list:
    """NGFF axes for a layout string such as 'yx', 'zyx' or 'cyx'."""
    return [({"name": "c", "type": "channel"} if ch == "c"
             else {"name": ch, "type": "space", "unit": unit}) for ch in layout]


def _write_meta(grp, name: str, datasets: list, layout: str, unit: str, kind: str,
                extra: Optional[dict] = None):
    ms = {"version": "0.4", "name": name, "axes": _axes(layout, unit),
          "datasets": datasets, "type": kind}
    grp.attrs["multiscales"] = [ms]
    grp.attrs["geminga"] = {"writer": "geminga.to_ome_zarr", **(extra or {})}


# ---------------------------------------------------------------------------
# Stereo-seq GEF
# ---------------------------------------------------------------------------

def _gef_raster(gef, window: tuple, b: int, genes, side: int) -> np.ndarray:
    """MID per bin as a (side, side) image: row = y bin, column = x bin, both counted from the
    window's low corner. Binned with integer division on the stored coordinates — the float
    arithmetic `GefFile.rasterize` uses can drop a bin origin into the previous bin."""
    d = gef.read_window(window, b, genes)
    x0, _x1, y0, _y1 = window
    img = np.zeros((side, side), np.float32)
    if d["x"].size == 0:
        return img
    ix = np.clip((d["x"].astype(np.int64) - x0) // b, 0, side - 1)
    iy = np.clip((d["y"].astype(np.int64) - y0) // b, 0, side - 1)
    np.add.at(img.reshape(-1), iy * side + ix, d["count"].astype(np.float32))
    return img


def _from_gef(view_or_gef, path: str, genes, max_side: int, budget: Optional[Budget],
              unit_um: float, chunk: int, overwrite: bool) -> dict:
    gef = getattr(view_or_gef, "gef", view_or_gef)
    if genes is None and hasattr(view_or_gef, "gef"):
        # Only a *view* carries a gene selection. A GefFile has a genes() *method* listing
        # every gene, and taking that attribute blindly passed a bound method to rasterize.
        sel = getattr(view_or_gef, "genes", None)
        genes = list(sel) if isinstance(sel, (list, tuple)) else None
    ex = gef.extent
    span = max(ex[1] - ex[0], ex[3] - ex[2])
    grp = _group(path, overwrite)
    datasets, written, skipped = [], [], []
    cap = budget.available("ram") if budget is not None else None
    for b in gef.bins:                                          # finest first, as NGFF wants
        side = int(math.ceil(span / b))
        nbytes = side * side * 4
        if side > max_side:
            skipped.append({"bin": b, "side": side, "reason": f"> max_side {max_side}"})
            continue
        if cap is not None and nbytes > cap:
            skipped.append({"bin": b, "side": side,
                            "reason": f"raster {human(nbytes)} > {human(cap)} free"})
            continue
        lease = budget.acquire("ram", str(nbytes)) if budget is not None else None
        try:
            img = _gef_raster(gef, (ex[0], ex[0] + side * b, ex[2], ex[2] + side * b), b,
                              genes, side)
            arr = _array(grp, str(len(datasets)), (side, side), (min(chunk, side),) * 2,
                         "float32")
            arr[:] = img                                        # row i = y bin i, as an image
        finally:
            if lease is not None:
                lease.release()
        datasets.append({"path": str(len(datasets)), "coordinateTransformations": [
            {"type": "scale", "scale": [b * unit_um, b * unit_um]},
            {"type": "translation", "translation": [ex[2] * unit_um, ex[0] * unit_um]}]})
        written.append({"bin": b, "side": side, "bytes": nbytes})
    if not datasets:
        raise ValueError(f"no GEF level fits: {skipped}")
    _write_meta(grp, f"{gef.chip or os.path.basename(gef.path)}"
                     f"{' · ' + ','.join(genes) if genes else ''}",
                datasets, "yx", "micrometer", "sum",
                {"source": gef.path, "bins": [w["bin"] for w in written],
                 "skipped": skipped, "unit_um": unit_um, "genes": genes,
                 "orientation": "row i = y bin i, column j = x bin j"})
    return {"path": path, "levels": written, "skipped": skipped, "kind": "gef"}


# ---------------------------------------------------------------------------
# Images and volumes
# ---------------------------------------------------------------------------

def _classify(arr, channels_last: Optional[bool]):
    """('image' | 'rgb' | 'volume', shape) for anything array-like.

    `NiftiVolume` / `DicomSeries` (anything with `.slab`) are volumes whatever their shape.
    A bare 3-D array whose last axis is 3 or 4 is read as RGB(A) unless `channels_last=False`;
    `channels_last=True` forces that reading for any trailing size."""
    if hasattr(arr, "slab"):
        return "volume", tuple(int(s) for s in arr.shape[:3])
    shp = tuple(int(s) for s in arr.shape)
    if len(shp) == 2:
        return "image", shp
    if len(shp) == 3:
        rgb = (shp[2] in (3, 4)) if channels_last is None else bool(channels_last)
        return ("rgb" if rgb else "volume"), shp
    raise ValueError(f"cannot export a {len(shp)}-D array as OME-Zarr (expected an image, an "
                     f"RGB(A) image or a volume); select the plane or frame you want first")


_LAYOUT = {"image": "yx", "rgb": "cyx", "volume": "zyx"}


def _read_band(src, kind: str, a: int, b: int) -> np.ndarray:
    """Rows (or slices) a:b of the source, in the destination's axis order."""
    if kind == "volume":
        return np.asarray(src.slab(a, b)) if hasattr(src, "slab") else np.asarray(src[a:b])
    blk = np.asarray(src[a:b])
    if kind == "rgb":                                            # (rows, W, C) -> (C, rows, W)
        blk = np.ascontiguousarray(blk.transpose(2, 0, 1))
    return blk


def _pool(a: np.ndarray, axes: tuple, labels: bool) -> np.ndarray:
    """2x reduction over `axes`, in the input's own dtype.

    Intensities: the mean of each 2-block (odd edges padded by repeating the last cell, so the
    edge is not diluted), rounded to the nearest integer and clipped for integer dtypes.
    Labels: the first cell of each block — the mean of two label ids names a third object."""
    if labels:
        key = tuple(slice(None, None, 2) if i in axes else slice(None) for i in range(a.ndim))
        return np.ascontiguousarray(a[key])
    pad = [(0, a.shape[i] % 2) if i in axes else (0, 0) for i in range(a.ndim)]
    if any(p[1] for p in pad):
        a = np.pad(a, pad, mode="edge")
    shape, red = [], []
    for i, n in enumerate(a.shape):
        if i in axes:
            shape += [n // 2, 2]
            red.append(len(shape) - 1)
        else:
            shape.append(n)
    acc = np.float64 if a.dtype.itemsize >= 4 else np.float32
    m = a.reshape(shape).mean(axis=tuple(red), dtype=acc)
    if a.dtype.kind in "iu":
        info = np.iinfo(a.dtype)
        m = np.clip(np.rint(m), info.min, info.max)
    return m.astype(a.dtype, copy=False)


def _from_array(src, path: str, levels: int, chunk: int, unit: str, voxel: Sequence[float],
                band_bytes: int, overwrite: bool, name: str, kind: str, shape: tuple,
                labels: bool) -> dict:
    """`band_bytes` is a ceiling on what is resident at once, not a suggestion."""
    layout = _LAYOUT[kind]
    dtype = np.dtype(src.dtype)
    shape = tuple(shape)
    if kind == "rgb":
        shape = (shape[2], shape[0], shape[1])                   # stored (c, y, x)
        band_axis, pooled = 1, (1, 2)
    elif kind == "volume":
        band_axis, pooled = 0, (0, 1, 2)
    else:
        band_axis, pooled = 0, (0, 1)
    n_pool = len(pooled)
    grp = _group(path, overwrite)
    if kind == "volume":
        chunks = (min(16, shape[0]),) + tuple(min(chunk, s) for s in shape[1:])
    elif kind == "rgb":
        chunks = (1,) + tuple(min(chunk, s) for s in shape[1:])
    else:
        chunks = tuple(min(chunk, s) for s in shape)
    level0 = _array(grp, "0", shape, chunks, dtype)

    def at(lo, hi):
        return (slice(None),) * band_axis + (slice(lo, hi),)

    row = int(np.prod([s for i, s in enumerate(shape) if i != band_axis])) * dtype.itemsize
    # an RGB band is read as (rows, W, C) and transposed into a copy, so it is resident twice
    band = max(2, (band_bytes // max(row * (2 if kind == "rgb" else 1), 1)) // 2 * 2)
    peak = 0
    for a in range(0, shape[band_axis], band):                  # copy level 0 band by band
        b = min(a + band, shape[band_axis])
        blk = _read_band(src, kind, a, b)
        level0[at(a, b)] = blk.astype(dtype, copy=False)
        peak = max(peak, blk.nbytes * (2 if kind == "rgb" else 1))
    ndim = len(shape)
    factor = [2 if i in pooled else 1 for i in range(ndim)]
    scale = [float(v) for v in voxel]
    datasets = [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": list(scale)}]}]
    prev, prev_shape = level0, shape
    acc_bytes = 8 if dtype.itemsize >= 4 else 4
    for lv in range(1, levels):
        new_shape = tuple(max(1, math.ceil(s / 2)) if i in pooled else s
                          for i, s in enumerate(prev_shape))
        if min(new_shape[-2:]) < 16:
            break
        dst = _array(grp, str(lv), new_shape,
                     tuple(min(c, s) for c, s in zip(chunks, new_shape)), dtype)
        # Resident per step: the input band, a padded copy of it when an edge is odd, the
        # accumulator-width mean (a 2**n_pool-th of the band) and that mean cast back to the
        # source dtype. Sizing from the source dtype alone let a requested 8 MB band peak at 20.
        row_in = int(np.prod([s for i, s in enumerate(prev_shape) if i != band_axis]))
        per_row = row_in * dtype.itemsize * 2 + row_in / 2 ** n_pool * (acc_bytes + dtype.itemsize)
        step = max(2, int(band_bytes // max(per_row, 1)) // 2 * 2)
        for a in range(0, prev_shape[band_axis], step):
            b = min(a + step, prev_shape[band_axis])
            blk = np.asarray(prev[at(a, b)])
            ds = _pool(blk, pooled, labels)
            dst[at(a // 2, a // 2 + ds.shape[band_axis])] = ds
            padded = blk.nbytes if any(blk.shape[i] % 2 for i in pooled) else 0
            peak = max(peak, blk.nbytes + padded + ds.size * acc_bytes * (0 if labels else 1)
                       + ds.nbytes)
        scale = [v * f for v, f in zip(scale, factor)]
        datasets.append({"path": str(lv), "coordinateTransformations": [
            {"type": "scale", "scale": list(scale)}]})
        prev, prev_shape = dst, new_shape
    _write_meta(grp, name, datasets, layout, unit, "nearest" if labels else "mean",
                {"band_rows": band, "peak_band_bytes": peak, "dtype": str(dtype),
                 "labels": labels})
    return {"path": path, "levels": [{"path": d["path"]} for d in datasets],
            "kind": "volume" if kind == "volume" else "image", "peak_band_bytes": peak,
            "shape": shape, "dtype": str(dtype), "labels": labels}


def _unwrap(source):
    """The array behind a view, a reader or a sink; the source itself if it is one already.
    Explicit `is None` tests: `a or b` on an ndarray is a ValueError."""
    for attr in ("vol", "a", "z"):
        inner = getattr(source, attr, None)
        if inner is not None:
            return inner
    return source


def to_ome_zarr(source, path: str, *, levels: int = 5, genes: Optional[Sequence[str]] = None,
                max_side: int = 16384, budget: Optional[Budget] = None, chunk: int = 256,
                unit_um: float = STEREOSEQ_UNIT_UM, band: str = "64MB",
                overwrite: bool = False, labels: Optional[bool] = None,
                channels_last: Optional[bool] = None) -> dict:
    """Write a multiscale OME-Zarr that napari, neuroglancer and Vitessce can stream.

    `source` may be a Stereo-seq view or `GefFile` (its bin levels become the pyramid), an
    `ImageView`, `VolumeView`, `TiffTileArray`, `NiftiVolume`, `DicomSeries`, a `MaskSink3D`, or
    any array-like with `.shape` and slicing — including a plain ndarray. Returns what was
    written, including any GEF levels skipped as too large.

    `overwrite=False` (the default) refuses any non-empty target. `overwrite=True` replaces an
    existing OME-Zarr of ours, and nothing else: a directory with files that are not zarr data
    is never emptied. Both raise `OverwriteRefused` (a `FileExistsError`).

    `labels` says the data are label ids, which are pooled by subsampling instead of averaging;
    `None` decides from the source (a mask-named NIfTI or a `MaskSink3D` is a label image).
    `channels_last` settles a 3-D array whose last axis is 3 or 4: RGB(A) by default, a volume
    with `channels_last=False`.
    """
    path = os.fspath(path)
    if hasattr(source, "gef") or source.__class__.__name__ == "GefFile":
        return _from_gef(source, path, genes, max_side, budget, unit_um, chunk, overwrite)
    arr = _unwrap(source)
    kind, shape = _classify(arr, channels_last)
    if labels is None:
        labels = (source.__class__.__name__ in ("MaskSink3D", "MaskSink")
                  or getattr(arr, "sequence", None) == "MASK")
    voxel = getattr(arr, "voxel_mm", None)                       # (z, y, x) for every volume class
    nax = len(_LAYOUT[kind])
    if voxel is not None and kind == "volume":
        unit, vox = "millimeter", tuple(float(v) for v in voxel)
    else:
        unit, vox = "pixel", None
    if vox is None or len(vox) != nax:
        vox = (1.0,) * nax
    name = os.path.basename(str(getattr(arr, "path", "array")))
    return _from_array(arr, path, max(int(levels), 1), chunk, unit, vox, int(parse_size(band)),
                       overwrite, name, kind, shape, bool(labels))


def read_ome_zarr_meta(path: str) -> dict:
    """The multiscales block and level shapes, read back — for checking an export."""
    import zarr
    try:
        g = zarr.open_group(path, mode="r", zarr_format=2)
    except TypeError:
        g = zarr.open_group(path, mode="r")
    ms = g.attrs["multiscales"][0]
    shapes = [tuple(g[d["path"]].shape) for d in ms["datasets"]]
    return {"multiscales": ms, "shapes": shapes, "geminga": dict(g.attrs.get("geminga", {}))}
