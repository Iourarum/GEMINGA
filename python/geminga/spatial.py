"""Spatial bundles: a matrix, a set of coordinates, and an image that share a frame.

Visium, Visium HD, Xenium, CosMx and MERFISH each ship a *directory*, not a file, and the
directory is the dataset: expression in one member, positions in another, the tissue image in
a third, and a scale factor that ties them together. Opening the matrix alone throws away what
makes the data spatial.

Two bundles are first-class here because their layouts are stable and documented:

* **Space Ranger** (Visium): `filtered_feature_bc_matrix.h5` + `spatial/tissue_positions*.csv`
  + `spatial/scalefactors_json.json` + `tissue_hires_image.png`. Visium HD adds
  `binned_outputs/square_NNNum/` with the same structure per bin size — which is a pyramid, the
  same way a GEF is.
* **Xenium** (Xenium Ranger): `transcripts.parquet` with `x_location`/`y_location` per molecule,
  `cells.parquet`, `cell_feature_matrix.h5` (10x layout), `morphology.ome.tif`. The transcript
  table is Parquet, so GEMINGA's row-group pruning by coordinate applies directly.

CosMx and MERFISH ship large flat CSVs. They are recognised, sized, and the honest advice is
given: convert to Parquet once, then the same tools apply.

Multiplexed imaging (CODEX, IMC, mIF) is an OME-TIFF with a channel axis. `MultiChannelTiff`
reads one channel's tiles at a time, so a 40-plex slide costs one channel of memory, not forty.
"""
from __future__ import annotations

import glob
import json
import math
import os
from typing import Optional, Sequence

import numpy as np

from ._core import Budget, human

__all__ = ["VisiumBundle", "XeniumBundle", "MultiChannelTiff", "sniff_bundle",
           "open_visium", "open_xenium"]


def _first(d: str, patterns: Sequence[str]) -> Optional[str]:
    """First existing match of `patterns` under `d`. The patterns may carry wildcards; the
    directory is a literal path — `slide[1]/` or `run?*/` must not be read as a character class
    or a wildcard, which made every lookup under it find nothing."""
    base = glob.escape(d)
    for p in patterns:
        hits = sorted(glob.glob(os.path.join(base, p)))
        if hits:
            return hits[0]
    return None


# ---------------------------------------------------------------------------
# Visium / Space Ranger
# ---------------------------------------------------------------------------

class VisiumBundle:
    """A Space Ranger output directory, opened lazily.

    `positions` is the barcode → pixel map; `scale` converts full-resolution pixels to the
    bundled hires/lowres images. `bin_levels` is non-empty for Visium HD.
    """

    def __init__(self, path: str, budget: Optional[Budget] = None,
                 bin_um: Optional[int] = None):
        self.path = os.path.abspath(path)
        self.budget = budget
        root = self.path
        self.bin_levels = sorted(int(os.path.basename(p).split("_")[1].rstrip("um"))
                                 for p in glob.glob(os.path.join(glob.escape(root),
                                                                 "binned_outputs", "square_*um")))
        if self.bin_levels:
            b = bin_um or self.bin_levels[-1]
            if b not in self.bin_levels:
                b = min(self.bin_levels, key=lambda x: abs(x - (bin_um or 0)))
            self.bin_um = b
            root = os.path.join(root, "binned_outputs", f"square_{b:03d}um")
        else:
            self.bin_um = None
        self.root = root
        self.matrix_path = _first(root, ["filtered_feature_bc_matrix.h5",
                                         "raw_feature_bc_matrix.h5"])
        self.mtx_dir = _first(root, ["filtered_feature_bc_matrix", "raw_feature_bc_matrix"])
        sp = os.path.join(root, "spatial")
        self.positions_path = _first(sp, ["tissue_positions.csv", "tissue_positions.parquet",
                                          "tissue_positions_list.csv"])
        self.scalefactors_path = _first(sp, ["scalefactors_json.json"])
        self.hires = _first(sp, ["tissue_hires_image.png", "tissue_hires_image.tif*"])
        self.lowres = _first(sp, ["tissue_lowres_image.png"])
        self.fullres = _first(self.path, ["*.tif", "*.tiff", "*.btf"])
        if not (self.matrix_path or self.mtx_dir):
            raise ValueError(f"{path}: no feature-barcode matrix found")
        self._pos = None
        self._m = None

    @property
    def matrix(self):
        if self._m is None:
            from .tenx import TenXMatrix, MtxTriple
            self._m = (TenXMatrix(self.matrix_path, budget=self.budget) if self.matrix_path
                       else MtxTriple(self.mtx_dir, budget=self.budget))
        return self._m

    @property
    def scale(self) -> dict:
        if not self.scalefactors_path:
            return {}
        with open(self.scalefactors_path) as fh:
            return json.load(fh)

    @property
    def positions(self):
        """barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres."""
        if self._pos is None:
            if self.positions_path is None:
                raise ValueError("no tissue_positions file in spatial/")
            if self.positions_path.endswith(".parquet"):
                import pyarrow.parquet as pq
                self._pos = pq.read_table(self.positions_path)
            else:
                import pyarrow.csv as pacsv
                t = pacsv.read_csv(self.positions_path)
                if "barcode" not in t.schema.names:            # older list format, no header
                    t = pacsv.read_csv(self.positions_path,
                                       read_options=pacsv.ReadOptions(
                                           column_names=["barcode", "in_tissue", "array_row",
                                                         "array_col", "pxl_row_in_fullres",
                                                         "pxl_col_in_fullres"]))
                self._pos = t
        return self._pos

    def extent(self) -> tuple:
        """(x0, x1, y0, y1) in full-resolution pixels."""
        p = self.positions
        x = p.column("pxl_col_in_fullres").to_numpy()
        y = p.column("pxl_row_in_fullres").to_numpy()
        return int(x.min()), int(x.max()) + 1, int(y.min()), int(y.max()) + 1

    def in_window(self, window: tuple) -> np.ndarray:
        """Column indices of the matrix (cells) whose spot centre lies in (x0,x1,y0,y1)."""
        x0, x1, y0, y1 = window
        p = self.positions
        x = p.column("pxl_col_in_fullres").to_numpy()
        y = p.column("pxl_row_in_fullres").to_numpy()
        keep = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        bc = p.column("barcode").to_numpy(zero_copy_only=False)
        wanted = set(bc[keep].tolist())
        return np.flatnonzero(np.isin(self.matrix.barcodes, list(wanted)))

    def describe(self) -> dict:
        m = self.matrix
        d = {"path": self.path, "platform": "Visium HD" if self.bin_levels else "Visium",
             "bin_um": self.bin_um, "bin_levels": self.bin_levels,
             "n_spots": m.n_cells, "n_features": m.n_features,
             "modalities": {k: v.n_features for k, v in m.modalities.items()},
             "hires_image": self.hires, "lowres_image": self.lowres,
             "fullres_image": self.fullres, "scale": self.scale}
        try:
            d["extent_px"] = self.extent()
            d["in_tissue"] = int(self.positions.column("in_tissue").to_numpy().sum())
        except Exception:
            pass
        return d

    def close(self):
        if self._m is not None:
            self._m.close()

    def __repr__(self) -> str:
        d = self.describe()
        return (f"VisiumBundle({os.path.basename(self.path)}, {d['platform']}"
                f"{f' bin{self.bin_um}um' if self.bin_um else ''}, {d['n_spots']:,} spots, "
                f"{d['n_features']:,} features)")


# ---------------------------------------------------------------------------
# Xenium
# ---------------------------------------------------------------------------

class XeniumBundle:
    """A Xenium Ranger output directory. Transcripts are Parquet with per-molecule
    coordinates, so windowing is row-group pruning — the same mechanism as any Parquet."""

    def __init__(self, path: str, budget: Optional[Budget] = None):
        self.path = os.path.abspath(path)
        self.budget = budget
        self.transcripts = _first(path, ["transcripts.parquet"])
        self.cells = _first(path, ["cells.parquet", "cells.csv.gz"])
        self.matrix_path = _first(path, ["cell_feature_matrix.h5"])
        self.boundaries = _first(path, ["cell_boundaries.parquet", "nucleus_boundaries.parquet"])
        self.morphology = _first(path, ["morphology.ome.tif", "morphology_focus.ome.tif",
                                        "morphology_focus/*.ome.tif"])
        self.experiment = _first(path, ["experiment.xenium"])
        if not self.transcripts and not self.matrix_path:
            raise ValueError(f"{path}: neither transcripts.parquet nor cell_feature_matrix.h5")
        self._ds = None

    @property
    def transcript_dataset(self):
        """A GEMINGA Parquet Dataset over transcripts — prune()/search() by x/y work on it."""
        if self._ds is None:
            import geminga as g
            self._ds = g.open_paths([self.transcripts])
            if self.budget is not None:
                self._ds.set_budget(self.budget)
        return self._ds

    COORDINATES = ("x_location", "y_location")
    DEFAULT_COLUMNS = ("feature_name", "x_location", "y_location", "cell_id", "qv")

    def window(self, x0: float, x1: float, y0: float, y1: float,
               columns: Optional[Sequence[str]] = DEFAULT_COLUMNS):
        """Transcripts in a window: row groups pruned by x statistics, then filtered exactly.

        The exact filter needs the coordinate columns, so they are read whether or not
        `columns` names them — asking for `["feature_name"]` is fine — and dropped again
        afterwards: the table has the columns you asked for, in the order you asked."""
        ds = self.transcript_dataset
        want = list(self.DEFAULT_COLUMNS if columns is None else columns)
        cols = want + [c for c in self.COORDINATES if c not in want]
        batches = ds.search("x_location", x0, x1, cols)
        import pyarrow as pa
        if not batches:
            return pa.table({c: [] for c in want})
        t = pa.Table.from_batches(batches)
        # search() returns the row groups that *could* match; the exact predicate is ours.
        x = t.column("x_location").to_numpy()
        y = t.column("y_location").to_numpy()
        t = t.filter(pa.array((x >= x0) & (x < x1) & (y >= y0) & (y < y1)))
        return t.select(want) if len(cols) != len(want) else t

    def prune(self, x0: float, x1: float, columns=None) -> dict:
        return self.transcript_dataset.prune("x_location", x0, x1, list(columns) if columns else None)

    def describe(self) -> dict:
        d = {"path": self.path, "platform": "Xenium", "transcripts": self.transcripts,
             "cells": self.cells, "matrix": self.matrix_path, "morphology": self.morphology,
             "boundaries": self.boundaries}
        if self.experiment:
            try:
                with open(self.experiment) as fh:
                    e = json.load(fh)
                d["panel"] = e.get("panel_name") or e.get("panel_design_id")
                d["num_cells"] = e.get("num_cells")
                d["region"] = e.get("region_name")
                d["pixel_size_um"] = e.get("pixel_size")
            except Exception:
                pass
        if self.transcripts:
            try:
                st = {s["column"]: s for s in self.transcript_dataset.stats()}
                d["n_transcripts"] = self.transcript_dataset.describe()["total_rows"]
                if "x_location" in st and "y_location" in st:
                    d["extent_um"] = (st["x_location"]["min"], st["x_location"]["max"],
                                      st["y_location"]["min"], st["y_location"]["max"])
            except Exception:
                pass
        return d

    def __repr__(self) -> str:
        d = self.describe()
        return (f"XeniumBundle({os.path.basename(self.path)}, "
                f"{d.get('n_transcripts', '?'):,} transcripts, panel {d.get('panel', '?')})")


# ---------------------------------------------------------------------------
# Multiplexed imaging
# ---------------------------------------------------------------------------

class MultiChannelTiff:
    """A CODEX / IMC / mIF OME-TIFF: (C, Y, X) or (Y, X, C), read one channel at a time.

    A 40-plex slide costs one channel of memory, not forty — and a *window* of a channel costs
    the tiles it touches. Each channel is its own TIFF page, and a page is read segment by
    segment (a tile, or a strip of rows), decoding only the segments that intersect the window.
    `cost()` says which of three situations a read is in, before it happens:

    * **tiles** — a tiled page: the window's tiles are decoded, one at a time;
    * **strips** — rows are stored in strips: the strips covering the window's rows;
    * **full-plane** — the page is a single strip (or a layout that cannot be read in parts,
      such as separate colour planes): the whole plane *of that one channel* is decoded to
      serve any window, and `cost()` and its `note` say so.

    With a `budget`, the bytes a read will make resident — the output window plus the segment
    being decoded and its compressed bytes, or for a full plane the plane — are leased before
    anything is read, so a read that cannot fit is refused up front (`MemoryError`) instead of
    discovering it in the allocator. The lease is released when the call returns: the array you
    get back is yours. Channel names come from OME-XML when present, else `ch00..chNN`.
    """

    def __init__(self, path: str, budget: Optional[Budget] = None, tier: str = "ram"):
        import tifffile
        self.path = path
        self.budget, self.tier = budget, tier
        self._tf = tifffile.TiffFile(path)
        s = self._tf.series[0]
        self._series = s
        self.axes = s.axes                       # e.g. "CYX", "YXC", "YXS", "ZCYX"
        self.shape = tuple(int(x) for x in s.shape)
        self.dtype = np.dtype(s.dtype)
        ci = self.axes.find("C")
        si = self.axes.find("S")
        self.channel_axis = ci if ci >= 0 else (si if si >= 0 else -1)
        self.n_channels = self.shape[self.channel_axis] if self.channel_axis >= 0 else 1
        self.channel_names = self._names()
        self.tiled = bool(getattr(s.pages[0], "is_tiled", False))

    def _names(self) -> list:
        names = []
        try:
            ome = self._tf.ome_metadata
            if ome:
                import re
                names = re.findall(r'<Channel[^>]*Name="([^"]+)"', ome)
        except Exception:
            pass
        if len(names) != self.n_channels:
            names = [f"ch{i:02d}" for i in range(self.n_channels)]
        return names

    def plane_shape(self) -> tuple:
        y = self.shape[self.axes.find("Y")]
        x = self.shape[self.axes.find("X")]
        return (y, x)

    # ---- layout: which pages hold a channel, and how a page is cut up ----
    def _page_grid(self):
        """Page numbers laid out over the series' page axes (everything but Y, X and the
        in-page sample axis), or None if the series does not map onto its pages that way."""
        dims = [i for i, a in enumerate(self.axes) if a not in ("Y", "X", "S")]
        shape = tuple(self.shape[i] for i in dims)
        n = int(np.prod(shape, dtype=np.int64))                  # 1 for a single-page image
        if len(self._series.pages) != n:
            return None
        return dims, np.arange(n).reshape(shape)

    def _channel_pages(self, c: int):
        """(page numbers, shape of the leading axes, sample index) for channel `c`."""
        grid = self._page_grid()
        if grid is None:
            return None
        dims, pages = grid
        sample = 0
        if self.channel_axis >= 0 and self.channel_axis in dims:
            pages = np.take(pages, c, axis=dims.index(self.channel_axis))
        elif self.channel_axis >= 0:                     # channels are samples within the page
            sample = c
        return pages.ravel(), tuple(pages.shape), sample

    def _segments(self, kf):
        """(rows, cols, n_rows, n_cols, samples) of one page's tiles or strips, or None when
        the page cannot be read in parts."""
        if int(getattr(kf, "planarconfig", 1)) != 1 or int(getattr(kf, "imagedepth", 1)) != 1:
            return None
        h, w, spp = int(kf.imagelength), int(kf.imagewidth), int(kf.samplesperpixel)
        if kf.is_tiled:
            th, tw = int(kf.tilelength), int(kf.tilewidth)
        else:
            th, tw = min(int(kf.rowsperstrip or h), h), w
        ny, nx = math.ceil(h / th), math.ceil(w / tw)
        if len(kf.dataoffsets) != ny * nx:
            return None
        return th, tw, ny, nx, spp

    def _window(self, window) -> tuple:
        y, x = self.plane_shape()
        y0, y1, x0, x1 = (0, y, 0, x) if window is None else (int(v) for v in window)
        y0, x0 = min(max(y0, 0), y), min(max(x0, 0), x)
        return y0, max(min(y1, y), y0), x0, max(min(x1, x), x0)

    def _channel_index(self, c) -> int:
        if isinstance(c, str):
            if c not in self.channel_names:
                raise KeyError(f"no channel {c!r}; have {self.channel_names}")
            c = self.channel_names.index(c)
        c = int(c)
        if not 0 <= c < self.n_channels:
            raise IndexError(f"channel {c} out of range for {self.n_channels} channel(s)")
        return c

    def cost(self, channel=0, window: Optional[tuple] = None) -> dict:
        """What reading `window` = (y0, y1, x0, x1) of one channel decodes and holds. Reads nothing.

        `decode_bytes` is what is decompressed (a whole tile or strip is decoded however little
        of it the window needs); `resident_bytes` the peak held at once, which is what a budget
        is asked for: the output, plus the segment being decoded and its compressed bytes."""
        c = self._channel_index(channel)
        y0, y1, x0, x1 = self._window(window)
        isz = self.dtype.itemsize
        found = self._channel_pages(c)
        planes = 1 if found is None else int(np.prod(found[1], dtype=np.int64))
        out = (y1 - y0) * (x1 - x0) * isz * planes
        h, w = self.plane_shape()
        kf = self._series.pages[0].keyframe
        geo = None if found is None else self._segments(kf)
        d = {"channel": c, "window": (y0, y1, x0, x1), "planes": planes, "output_bytes": out}
        if found is None:                                # pages do not map onto the series
            stack = int(np.prod(self.shape, dtype=np.int64)) * isz
            d.update(strategy="full-stack", tiles_touched=0, decode_bytes=stack,
                     resident_bytes=2 * stack + out,
                     note=f"this file's pages do not map onto its series: the whole stack "
                          f"({human(stack)}) is decoded to serve any window")
        elif geo is None:                                # a page that cannot be read in parts
            page = h * w * int(kf.samplesperpixel) * isz
            d.update(strategy="full-plane", tiles_touched=1, decode_bytes=page * planes,
                     resident_bytes=2 * page + out,
                     note=f"this page layout cannot be read in parts: the full plane "
                          f"({human(page)}) of the channel is decoded to serve any window")
        else:
            th, tw, ny, nx, spp = geo
            touched = (((y1 - 1) // th - y0 // th + 1) * ((x1 - 1) // tw - x0 // tw + 1)
                       if y1 > y0 and x1 > x0 else 0)
            seg = th * tw * spp * isz
            whole = ny == 1 and nx == 1
            d.update(strategy="full-plane" if whole else ("tiles" if self.tiled else "strips"),
                     tiles_touched=touched, decode_bytes=touched * seg * planes,
                     resident_bytes=2 * seg + out)
            d["note"] = (f"not tiled and stored as one strip: the full plane ({human(seg)}) of "
                         f"this one channel is decoded to serve any window" if whole else
                         f"{touched} {'tile' if self.tiled else 'strip'}(s) of {human(seg)} "
                         f"decoded, one at a time")
        d["resident_human"] = human(d["resident_bytes"])
        d["decode_human"] = human(d["decode_bytes"])
        cap = self.budget.available(self.tier) if self.budget is not None else None
        d["fits"] = None if cap is None else d["resident_bytes"] <= cap
        return d

    def channel(self, c, y0: int = 0, y1: Optional[int] = None, x0: int = 0,
                x1: Optional[int] = None) -> np.ndarray:
        """One channel, one window, as an array: only that channel's page, and only the tiles
        (or strips) the window touches, are decoded — see the class docstring and `cost()`.
        A series with extra axes (Z, T) returns those axes in front: (Z, h, w)."""
        c = self._channel_index(c)
        y, x = self.plane_shape()
        y0, y1, x0, x1 = self._window((y0, y if y1 is None else y1, x0, x if x1 is None else x1))
        need = self.cost(c, (y0, y1, x0, x1))["resident_bytes"]
        lease = (self.budget.acquire(self.tier, str(int(need)))
                 if self.budget is not None and need > 0 else None)
        try:
            found = self._channel_pages(c)
            if found is None:                            # unusual layout: the honest slow path
                full = np.asarray(self._series.asarray())
                key = [slice(None)] * len(self.axes)
                key[self.axes.find("Y")], key[self.axes.find("X")] = slice(y0, y1), slice(x0, x1)
                if self.channel_axis >= 0:
                    key[self.channel_axis] = c
                return np.asarray(full[tuple(key)])
            pages, lead, sample = found
            out = np.zeros((len(pages), y1 - y0, x1 - x0), self.dtype)
            for k, pn in enumerate(pages):
                self._read(self._series.pages[int(pn)], sample, y0, y1, x0, x1, out[k])
            return out.reshape(lead + (y1 - y0, x1 - x0))        # lead is () for a plain CYX
        finally:
            if lease is not None:
                lease.release()

    def _read(self, page, sample: int, y0: int, y1: int, x0: int, x1: int, out: np.ndarray):
        """Fill `out` with a window of one channel of one page."""
        if y1 <= y0 or x1 <= x0:
            return
        kf = page.keyframe
        geo = self._segments(kf)
        if geo is None:                                  # whole page, then crop
            a = np.asarray(page.asarray())
            if a.ndim == 3:
                a = a[sample] if int(getattr(kf, "planarconfig", 1)) != 1 else a[..., sample]
            out[:] = a[y0:y1, x0:x1]
            return
        th, tw, _ny, nx, _spp = geo
        fh = self._tf.filehandle
        for ty in range(y0 // th, (y1 - 1) // th + 1):
            for tx in range(x0 // tw, (x1 - 1) // tw + 1):
                i = ty * nx + tx
                off, cnt = int(page.dataoffsets[i]), int(page.databytecounts[i])
                if cnt == 0:                             # a sparse segment is all zeros
                    continue
                fh.seek(off)
                seg, _idx, shape = kf.decode(fh.read(cnt), i)
                seg = np.asarray(seg).reshape(shape)     # (depth, rows, cols, samples)
                t = seg[0, :, :, min(sample, seg.shape[3] - 1)]
                ty0, tx0 = ty * th, tx * tw
                sy0, sx0 = max(y0, ty0), max(x0, tx0)
                sy1, sx1 = min(y1, ty0 + t.shape[0]), min(x1, tx0 + t.shape[1])
                if sy1 > sy0 and sx1 > sx0:
                    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                        t[sy0 - ty0:sy1 - ty0, sx0 - tx0:sx1 - tx0]

    def describe(self) -> dict:
        y, x = self.plane_shape()
        geo = self._segments(self._series.pages[0].keyframe)
        return {"path": self.path, "axes": self.axes, "shape": self.shape,
                "plane": (y, x), "n_channels": self.n_channels,
                "channels": self.channel_names[:12] + (["..."] if self.n_channels > 12 else []),
                "dtype": str(self.dtype), "tiled": self.tiled,
                "segments": (None if geo is None else
                             f"{geo[1]}x{geo[0]} tiles" if self.tiled else f"strips of {geo[0]} rows"),
                "bytes_per_channel": y * x * self.dtype.itemsize,
                "bytes_total": int(np.prod(self.shape)) * self.dtype.itemsize}

    def close(self):
        try:
            self._tf.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        y, x = self.plane_shape()
        return (f"MultiChannelTiff({os.path.basename(self.path)}, {self.n_channels} channels "
                f"x {y}x{x} {self.dtype}, {human(y * x * self.dtype.itemsize)} per channel)")


# ---------------------------------------------------------------------------
# Bundle sniffing — what is this directory?
# ---------------------------------------------------------------------------

def sniff_bundle(d: str) -> dict:
    """Recognise Space Ranger, Xenium, CosMx, MERFISH, Cell Ranger and cellbin layouts."""
    # A bundle is self-contained and shallow: Space Ranger has `spatial/`, Visium HD has
    # `binned_outputs/square_*/`, and that is as deep as it goes. Walking further would let a
    # parent directory holding several datasets be mistaken for one of them.
    # Only the sub-folders a bundle is *known* to carry are descended into. Anything else
    # is another dataset living next door, and must not be mistaken for part of this one.
    KNOWN_SUBDIRS = ("spatial", "binned_outputs", "morphology_focus", "analysis",
                     "filtered_feature_bc_matrix", "raw_feature_bc_matrix")
    files = set()
    for dp, dns, fs in os.walk(d):
        rel = os.path.relpath(dp, d)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth == 0:
            dns[:] = [x for x in dns if x in KNOWN_SUBDIRS]
        elif depth == 1 and rel == "binned_outputs":
            dns[:] = [x for x in dns if x.startswith("square_")]
        elif depth < 3:
            dns[:] = [x for x in dns if x in KNOWN_SUBDIRS]
        else:
            dns[:] = []
        for f in fs:
            files.add(os.path.relpath(os.path.join(dp, f), d))
        if len(files) > 5000:
            break
    low = {f.lower() for f in files}
    out: dict = {"kind": "directory", "images": [], "extras": {}, "advice": []}

    def has(*names):
        return any(any(l.endswith(n) for l in low) for n in names)

    if has("transcripts.parquet", "cell_feature_matrix.h5") and has("experiment.xenium"):
        out["kind"] = "xenium"
        out["matrix"] = _first(d, ["cell_feature_matrix.h5"])
        out["spatial"] = {"platform": "Xenium", "n_positions": None,
                          "extent": "per-transcript x/y in transcripts.parquet"}
        out["images"] = [f for f in files if "morphology" in f.lower()][:4]
        out["advice"].append("transcripts.parquet supports prune()/search() by x_location")
    elif has("scalefactors_json.json") or has("tissue_positions.csv", "tissue_positions_list.csv"):
        hd = [f for f in files if "binned_outputs" in f]
        out["kind"] = "visium-hd" if hd else "visium"
        out["matrix"] = _first(d, ["filtered_feature_bc_matrix.h5", "raw_feature_bc_matrix.h5",
                                   "binned_outputs/square_*um/filtered_feature_bc_matrix.h5"])
        if not out["matrix"]:
            out["matrix"] = _first(d, ["filtered_feature_bc_matrix", "raw_feature_bc_matrix"])
        out["spatial"] = {"platform": out["kind"], "n_positions": None,
                          "extent": "spatial/tissue_positions"}
        out["images"] = [f for f in files if "tissue_" in f.lower() and f.lower().endswith(
            (".png", ".tif", ".tiff", ".jpg"))][:4]
        if hd:
            out["extras"]["bin_levels_um"] = sorted({int(f.split("square_")[1].split("um")[0])
                                                    for f in hd if "square_" in f})
            out["advice"].append("Visium HD bin levels form a pyramid — use bin_um= to pick")
    elif has("_tx_file.csv", "_exprmat_file.csv", "_metadata_file.csv"):
        out["kind"] = "cosmx"
        out["spatial"] = {"platform": "CosMx", "n_positions": None,
                          "extent": "x_global_px / y_global_px in the tx file"}
        out["advice"].append("CosMx ships flat CSVs (often >10 GB) — convert *_tx_file.csv to "
                             "Parquet once (pyarrow.csv.open_csv streaming) and prune by x/y")
    elif has("detected_transcripts.csv", "cell_by_gene.csv"):
        out["kind"] = "merfish"
        out["spatial"] = {"platform": "MERFISH (Vizgen)", "n_positions": None,
                          "extent": "global_x / global_y in detected_transcripts.csv"}
        out["advice"].append("MERFISH ships CSVs — convert detected_transcripts.csv to Parquet "
                             "once; cell_by_gene.csv is dense and small enough to read directly")
    elif has("filtered_feature_bc_matrix.h5", "raw_feature_bc_matrix.h5"):
        out["kind"] = "cellranger"
        out["matrix"] = _first(d, ["filtered_feature_bc_matrix.h5", "raw_feature_bc_matrix.h5"])
        if has("fragments.tsv.gz"):
            out["extras"]["fragments"] = "fragments.tsv.gz (scATAC / multiome; tabix-indexed)"
            out["advice"].append("fragments.tsv.gz is bgzf+tabix — region queries via pysam, "
                                 "not GEMINGA yet")
        if has("protospacer_calls_per_cell.csv"):
            out["extras"]["guides"] = "protospacer_calls_per_cell.csv (CRISPR screen)"
    elif has("matrix.mtx.gz", "matrix.mtx"):
        out["kind"] = "mtx-triple"
        out["matrix"] = d if os.path.exists(os.path.join(d, "matrix.mtx.gz")) or \
            os.path.exists(os.path.join(d, "matrix.mtx")) else \
            os.path.dirname(_first(d, ["**/matrix.mtx.gz", "**/matrix.mtx"]) or d)
    elif has(".cellbin.gef", ".gef"):
        out["kind"] = "stomics"
        out["advice"].append("Stereo-seq GEF — use g.open_gef()")
    elif has(".imzml"):
        out["kind"] = "imzml"
        out["spatial"] = {"platform": "MALDI-MSI (imzML)", "n_positions": None,
                          "extent": "pixel grid in the .imzML XML"}
        out["advice"].append("imzML pairs an XML index with an .ibd binary — pyimzml reads "
                             "spectra per pixel; a GEMINGA reader is on the roadmap")
    tif = [f for f in files if f.lower().endswith((".ome.tif", ".ome.tiff", ".qptiff"))]
    if tif and out["kind"] == "directory":
        out["kind"] = "multiplexed-imaging"
        out["images"] = tif[:4]
        out["advice"].append("multi-channel OME-TIFF — MultiChannelTiff reads one channel at a time")
    return out


def open_visium(path: str, budget: Optional[Budget] = None, **kw) -> VisiumBundle:
    return VisiumBundle(path, budget, **kw)


def open_xenium(path: str, budget: Optional[Budget] = None) -> XeniumBundle:
    return XeniumBundle(path, budget)
