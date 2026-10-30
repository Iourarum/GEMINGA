"""Stereo-seq: GEF, GEM, and the images registered to them.

Stereo-seq is the case this library was designed around. A 1 cm chip carries 400 million DNA
nanoballs at 500 nm pitch; chips run to 13 x 13 cm, which is over 150x the area. A single bin-1
expression matrix routinely holds tens of millions of observations, and the paired ssDNA/H&E
image is gigapixel. None of it fits in a laptop, and most of it does not need to.

The useful accident is that **GEF is already a zoom pyramid**. SAW writes `/geneExp/bin1`,
`bin5`, `bin10` ... `bin200` into one HDF5 container, each a complete matrix at a coarser
spatial binning. That is exactly a level-of-detail stack, so zooming out on a GEF is not
decimation at all — it is switching to a level the instrument's own pipeline already computed.
A Stereo-seq view never decimates rows: the level *is* the decimation (`stride` is always 1).

The second useful structure is the per-level `gene` dataset, which stores `(gene, offset,
count)` triples pointing into `expression`. Reading one gene is therefore one contiguous HDF5
slice, not a scan: native predicate pushdown, the same idea as Parquet row-group pruning.

How a window is read, and what it costs:

* **Coordinates.** A row's `(x, y)` is the origin of its bin; the bin covers `[x, x + bin)`. A
  window `(x0, x1, y0, y1)` is half-open, and a bin belongs to it when the two *overlap*, so a
  small window inside one coarse bin still sees that bin. `extent` is half-open too: SAW writes
  the GEF's `maxX`/`maxY` attributes as the largest coordinate present (inclusive), so `extent`
  is one past them; a GEM's index stores max + 1 directly, so the two give the same window for
  the same chip.
* **Reads are leased blocks.** Without a spatial index a level is read in HDF5 row blocks; each
  block is leased while it is read and filtered to the window, and only the kept rows are kept.
  A gene's contiguous slice is read the same way, block by block. With the sidecar index
  (`gefindex`), only the row groups that can hold the window are streamed, through the budget.
  `load()` leases its planned output first, so its peak is one block plus what it keeps.
* **Estimates follow the tissue.** `estimate_rows` comes from the sidecar's row-group statistics
  when there is one, else from the coarsest level's occupancy: one small read of its most
  widespread genes gives the tissue's density per coarse bin, calibrated per level against that
  level's row count — never from the window's share of the chip area.
"""
from __future__ import annotations

import math
import os
import re
import time as _time
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .journal import _warn_once as _journal_warn, active as _journal_active, count_units, key_of_gef
from .journal import record as _journal_write
from ._core import Budget, GemIndex, GemStream, human, parse_size
from .formats import resolve_chunk_bytes
from .view import (
    CostError, Extent, Overview, View, _Accounted, _arrow_bytes, _grid_count, _grid_first,
    _Hold, _nbytes,
)

__all__ = ["GefFile", "StereoSeqView", "GemView", "StereoSeqDataset",
           "open_gef", "open_gem", "open_stereoseq", "index_gem", "BIN_LEVELS"]

#: Bin sizes SAW writes into a visualization GEF, coarsest last.
BIN_LEVELS = (1, 5, 10, 20, 50, 100, 150, 200, 500)

#: What one observation costs once materialized: x, y, count as int32. The on-disk compound
#: dtype (u4, u4, u2) is smaller, and using it would understate what load() actually holds.
RESIDENT_BYTES_PER_OBS = 12
#: A gene column of fixed-width unicode adds 4 bytes per character of the longest name.
GENE_COLUMN_BYTES_PER_OBS = 32

_DEFAULT_BLOCK = 16 << 20         # a read block when there is no budget to size it from
_MASK_BYTES = 2                   # the window filter's boolean mask, and one temporary
_OCC_GENES = 256                  # genes read from the coarsest level for the occupancy grid
_OCC_MAX_ROWS = 1 << 18           # ... and at most this many of their rows


def _decode(a) -> np.ndarray:
    """HDF5 gives fixed-width bytes for gene names; make them str."""
    arr = np.asarray(a)
    if arr.dtype.kind in ("S", "O"):
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])
    return arr


def _i32(a: np.ndarray) -> np.ndarray:
    """int32 without a copy when the values are 4-byte integers already (chip coordinates are)."""
    if a.dtype.kind in "iu" and a.dtype.itemsize == 4:
        return a.view(np.int32)
    return a.astype(np.int32)


@dataclass
class BinLevel:
    """One bin size inside a GEF — a complete matrix at that spatial resolution."""
    bin_size: int
    n_obs: int
    n_genes: int
    itemsize: int
    path: str

    @property
    def nbytes(self) -> int:
        return self.n_obs * self.itemsize

    def __repr__(self) -> str:
        return (f"BinLevel(bin{self.bin_size}, {self.n_obs:,} obs, {self.n_genes:,} genes, "
                f"{human(self.nbytes)})")


class _ReadCtx:
    """How a GEF read leases: the budget, its tier, and the block size rule (the chunk floor rule
    of `formats.resolve_chunk_bytes`, re-evaluated for every block, so blocks shrink as a load's
    output grows)."""

    def __init__(self, budget: Optional[Budget] = None, tier: str = "ram", chunk_bytes=None):
        self.budget, self.tier, self.chunk_bytes = budget, tier, chunk_bytes

    def cap(self) -> int:
        if self.budget is None:
            if self.chunk_bytes is None:
                return _DEFAULT_BLOCK
            return int(parse_size(self.chunk_bytes)) if isinstance(self.chunk_bytes, str) \
                else int(self.chunk_bytes)
        try:
            return int(resolve_chunk_bytes(self.chunk_bytes, _DEFAULT_BLOCK, self.budget, self.tier))
        except MemoryError:
            return max(int(self.budget.available(self.tier)), 1)

    def block_rows(self, per_row: int, chunk_rows: int = 0) -> int:
        rows = max(1, self.cap() // max(int(per_row), 1))
        if chunk_rows and rows >= chunk_rows:
            rows -= rows % chunk_rows              # whole HDF5 chunks: each decompressed once
        return rows


def _assemble(pieces: list, ctx: _ReadCtx, empty: dict) -> dict:
    """Join per-block pieces column by column. A column is allocated while its pieces still
    exist (that transient is leased), and each piece's part is dropped as soon as it is copied,
    so the peak is the output plus one column, not twice the output."""
    if not pieces:
        return dict(empty)
    if len(pieces) == 1:
        return pieces[0]
    n = sum(len(p["x"]) for p in pieces)
    out = {}
    for k in list(pieces[0].keys()):
        dt = np.result_type(*[p[k].dtype for p in pieces])
        with _Hold(ctx.budget, ctx.tier, n * dt.itemsize):
            col = np.empty(n, dt)
            o = 0
            for p in pieces:
                a = p[k]
                col[o:o + len(a)] = a
                o += len(a)
                p[k] = None
        out[k] = col
    return out


class GefFile:
    """A Stereo-seq GEF, opened lazily.

    Nothing is read beyond HDF5 metadata until you ask for a window. The bin levels, the chip
    extent and the gene list all come from dataset shapes and attributes.
    """

    def __init__(self, path, group: str = "geneExp"):
        """`path` is a local file, or any seekable file-like object — a `RemoteFile` over FTP
        or HTTP opens a GEF that stays on the server, since HDF5 is built for random access."""
        import h5py
        if hasattr(path, "read") and hasattr(path, "seek"):
            self._fileobj = path
            self.path = getattr(path, "url", repr(path))
            self.remote = True
        else:
            self._fileobj = None
            self.path = os.fspath(path)
            self.remote = False
        self.f = h5py.File(self._fileobj if self._fileobj is not None else self.path, "r")
        self.group_name = group
        if group not in self.f:
            avail = list(self.f.keys())
            self.f.close()
            raise ValueError(f"{path}: no {group!r} group; found {avail}. "
                             f"For a cellbin GEF use group='cellBin'.")
        self.grp = self.f[group]
        self.levels: dict[int, BinLevel] = {}
        for key in self.grp.keys():
            m = re.fullmatch(r"bin(\d+)", key)
            if not m:
                continue
            g = self.grp[key]
            if "expression" not in g:
                continue
            exp = g["expression"]
            self.levels[int(m.group(1))] = BinLevel(
                bin_size=int(m.group(1)),
                n_obs=int(exp.shape[0]),
                n_genes=int(g["gene"].shape[0]) if "gene" in g else 0,
                itemsize=int(exp.dtype.itemsize),
                path=f"{group}/{key}",
            )
        if not self.levels:
            self.f.close()
            raise ValueError(f"{path}: no bin levels under {group!r}")
        self.bins = sorted(self.levels)
        self._gene_cache: dict[int, tuple] = {}
        self._gene_rows: dict[int, tuple] = {}
        self._tiles: dict[int, bool] = {}
        self._sidx = None
        self._sidx_checked = False
        self._extent = None
        self._occ = None
        self._lam: dict[int, float] = {}
        self._journal_open()

    def _journal_open(self):
        """The first time a file is opened against a journal, write what it is: its extent,
        its bin levels and its gene list — so `rg GFAP journal.jsonl` later says which chips
        carried the gene, without reopening any of them. One record per file per journal."""
        j = _journal_active()
        if j is None:
            return
        try:
            key = key_of_gef(self)
            if j.opened(key):
                return
            coarse = self.bins[-1]
            names = [str(x) for x in self.genes(coarse)] if self.levels[coarse].n_genes else None
            j.record("open", key, window=list(self.extent), unit="spots",
                     level={"bins": list(self.bins)}, genes=names,
                     units=self.levels[self.bins[0]].n_obs, nbytes=self.file_bytes,
                     group=self.group_name)
        except Exception as err:                 # the diary never stops the open
            _journal_warn(f"journal: {err!r}")

    def _journal_record(self, op: str, window, b: int, genes, unit: str = "spots", **kw):
        if _journal_active() is None or not hasattr(self, "fingerprint"):   # a stand-in object
            return
        try:
            _journal_write(op, key_of_gef(self), window=list(window), level={"bin": int(b)},
                           unit=unit, genes=genes, **kw)
        except Exception as err:
            _journal_warn(f"journal: {err!r}")

    @property
    def file_bytes(self) -> int:
        if self._fileobj is not None:
            return int(getattr(self._fileobj, "size", 0))
        return os.path.getsize(self.path)

    def fingerprint(self) -> dict:
        """Identity for the spatial-index sidecar: path, size and mtime locally; URL and size
        for a remote file, whose mtime a plain FTP or HTTP server may not report."""
        from .gefindex import VERSION
        if self.remote:
            return {"url": self.path, "size": self.file_bytes, "version": VERSION}
        st = os.stat(self.path)
        return {"size": st.st_size, "mtime": int(st.st_mtime), "version": VERSION}

    # ---- spatial index ----
    @property
    def spatial_index(self):
        """The Z-order sidecar if one exists and still matches this file, else None. Looking
        for it creates nothing."""
        if not self._sidx_checked:
            self._sidx_checked = True
            try:
                from .gefindex import SpatialIndex
                si = SpatialIndex(self.path, fingerprint=self.fingerprint())
                self._sidx = si if si.valid and si.levels else None
            except Exception:
                self._sidx = None
        return self._sidx

    def _index_for(self, b: int):
        """The sidecar for level `b` when its file is there and opens; a sidecar is a soft
        cache, so anything else means the full-level block read."""
        si = self.spatial_index
        if si is None:
            return None
        try:
            return si if si.has(b) else None
        except Exception:
            return None

    def build_spatial_index(self, levels=None, **kw):
        """Write the sidecar so windows read only what they contain. One-time, out of core."""
        from .gefindex import build_spatial_index
        self._sidx = build_spatial_index(self, levels, **kw)
        self._sidx_checked = True
        return self._sidx

    def read_plan(self, window: tuple, bin_size: Optional[int] = None,
                  genes: Optional[Sequence[str]] = None) -> dict:
        """What reading a window will cost in bytes *read* — not rows kept. From footers."""
        b = self.pick_bin(bin_size)
        if genes:
            names, offs, cnts = self.gene_index(b)
            lookup = {n: i for i, n in enumerate(names)}
            n = sum(int(cnts[lookup[q]]) for q in genes if q in lookup)
            return {"path": "gene-slices", "read_bytes": n * self.levels[b].itemsize,
                    "read_human": human(n * self.levels[b].itemsize), "rows_read": n,
                    "level_bytes": self.levels[b].nbytes}
        si = self._index_for(b)
        if si is not None:
            p = si.plan(window, b)
            p["path"] = "spatial-index"
            p["rows_read"] = p["rows_scanned"]
            return p
        # Bytes actually on disk for this level — gzip-compressed HDF5 chunks — so that the
        # comparison with the (zstd-compressed) sidecar is like for like.
        try:
            disk = int(self.f[self.levels[b].path]["expression"].id.get_storage_size())
        except Exception:
            disk = self.levels[b].nbytes
        return {"path": "full-level", "read_bytes": disk, "rows_read": self.levels[b].n_obs,
                "read_human": human(disk),
                "level_bytes": self.levels[b].nbytes,
                "advice": "no spatial index for this level — GefFile.build_spatial_index() "
                          "makes windows read only what they contain"}

    # ---- geometry ----
    @property
    def extent(self) -> tuple[int, int, int, int]:
        """`(x0, x1, y0, y1)` in chip coordinates, half-open: `x1`/`y1` are one past the last
        coordinate. From the `minX`/`maxX`/`minY`/`maxY` attributes SAW writes — `maxX`/`maxY`
        are the largest coordinates present (inclusive), so `x1 = maxX + 1`; without them, from
        the coarsest level's bin origins — `x1` is then the end of the last occupied coarse bin,
        still exclusive and covering every observation."""
        if self._extent is None:
            exp = self.f[self.levels[self.bins[0]].path]["expression"]
            a = exp.attrs

            def attr(*names):
                for n in names:
                    if n in a:
                        return int(np.asarray(a[n]).ravel()[0])
                return None
            x0, y0, x1, y1 = attr("minX"), attr("minY"), attr("maxX"), attr("maxY")
            if x1 is not None:
                x1 += 1                                   # inclusive maximum -> half-open bound
            if y1 is not None:
                y1 += 1
            if None in (x0, y0, x1, y1):
                coarse = self.levels[self.bins[-1]]
                e = self.f[coarse.path]["expression"]
                b = coarse.bin_size
                mins, maxs = [None, None], [None, None]
                step = max(1, (8 << 20) // max(int(e.dtype.itemsize), 1))
                for s in range(0, coarse.n_obs, step):
                    rec = e[s:s + step]
                    for j, f in enumerate(("x", "y")):
                        v = rec[f]
                        lo, hi = int(v.min()), int(v.max())
                        mins[j] = lo if mins[j] is None else min(mins[j], lo)
                        maxs[j] = hi if maxs[j] is None else max(maxs[j], hi)
                    rec = None
                x0 = mins[0] if x0 is None else x0
                y0 = mins[1] if y0 is None else y0
                x1 = maxs[0] + b if x1 is None else x1
                y1 = maxs[1] + b if y1 is None else y1
            self._extent = (int(x0), int(x1), int(y0), int(y1))
        return self._extent

    @property
    def resolution(self) -> Optional[int]:
        for key in ("resolution", "Resolution"):
            if key in self.grp.attrs:
                return int(np.asarray(self.grp.attrs[key]).ravel()[0])
            if key in self.f.attrs:
                return int(np.asarray(self.f.attrs[key]).ravel()[0])
        return None

    @property
    def chip(self) -> Optional[str]:
        for key in ("StereoChip", "chipID", "sn", "SN"):
            for holder in (self.f.attrs, self.grp.attrs):
                if key in holder:
                    v = np.asarray(holder[key]).ravel()[0]
                    return v.decode() if isinstance(v, bytes) else str(v)
        return None

    # ---- genes ----
    def gene_index(self, bin_size: Optional[int] = None):
        """(names, offsets, counts) into the level's `expression` array.

        This is what makes a per-gene query one slice instead of a scan.
        """
        b = self.pick_bin(bin_size)
        if b in self._gene_cache:
            return self._gene_cache[b]
        g = self.f[self.levels[b].path]
        if "gene" not in g:
            raise ValueError(f"{self.path}: bin{b} has no gene index")
        gd = g["gene"][:]
        fields = gd.dtype.names or ()

        def pick(*cands):
            for c in cands:
                for f in fields:
                    if f.lower() == c:
                        return gd[f]
            return None
        names = pick("gene", "genename", "geneid", "symbol")
        offs = pick("offset", "offsets")
        cnts = pick("count", "counts", "cellcount", "genecount")
        if names is None or offs is None or cnts is None:
            raise ValueError(f"{self.path}: unexpected gene dtype {fields}")
        out = (_decode(names), np.asarray(offs, dtype=np.int64),
               np.asarray(cnts, dtype=np.int64))
        self._gene_cache[b] = out
        return out

    def genes(self, bin_size: Optional[int] = None) -> np.ndarray:
        return self.gene_index(bin_size)[0]

    def _ids_tile(self, b: int) -> bool:
        """Do the gene slices of level `b` cover its rows exactly, one after another? Only then
        is every row's gene known from the offsets (and `gene_id` returned)."""
        if b not in self._tiles:
            try:
                _, offs, cnts = self.gene_index(b)
                keep = cnts > 0
                o, c = offs[keep], cnts[keep]
                order = np.argsort(o, kind="stable")
                o, c = o[order], c[order]
                self._tiles[b] = (int(c.sum()) == self.levels[b].n_obs
                                  and (o.size == 0 or int(o[0]) == 0)
                                  and bool(np.all(o[1:] == o[:-1] + c[:-1])))
            except ValueError:
                self._tiles[b] = False
        return self._tiles[b]

    def _gene_ids(self, b: int, a: int, c: int) -> np.ndarray:
        """The gene (position in `genes(b)`) of rows [a, c) of level `b`, from the offsets alone:
        rows are stored gene by gene. -1 for a row no gene's slice covers."""
        if b not in self._gene_rows:
            _, offs, cnts = self.gene_index(b)
            keep = cnts > 0
            order = np.flatnonzero(keep)[np.argsort(offs[keep], kind="stable")]
            self._gene_rows[b] = (offs[order], offs[order] + cnts[order], order.astype(np.int32))
        starts, ends, ids = self._gene_rows[b]
        out = np.full(c - a, -1, np.int32)
        i = max(int(np.searchsorted(starts, a, side="right")) - 1, 0)
        while i < len(starts) and starts[i] < c:
            s, e = max(int(starts[i]), a), min(int(ends[i]), c)
            if e > s:
                out[s - a:e - a] = ids[i]
            i += 1
        return out

    # ---- level choice ----
    def pick_bin(self, bin_size: Optional[int] = None) -> int:
        if bin_size is None:
            return self.bins[0]
        if bin_size in self.levels:
            return bin_size
        return min(self.bins, key=lambda b: abs(b - bin_size))

    def bin_for_budget(self, window: tuple, cap_bytes: int,
                       per_obs: int = RESIDENT_BYTES_PER_OBS) -> int:
        """Finest bin whose window still fits `cap_bytes` once materialized."""
        for b in self.bins:                                    # fine -> coarse
            if self.estimate_rows(window, b) * per_obs <= cap_bytes:
                return b
        return self.bins[-1]

    # ---- estimates ----
    def estimate_rows(self, window: tuple, bin_size: int, *, budget: Optional[Budget] = None,
                      tier: str = "ram") -> int:
        """Observations of level `bin_size` whose bins overlap `window`.

        With the sidecar: from its row-group statistics (rows of every surviving row group, the
        partly covered ones by the share of their bounding box inside the window). Without: from
        the coarsest level's occupancy — the density of its most widespread genes per coarse bin,
        read once (a few hundred kilobytes) — calibrated so that the level's whole extent gives
        exactly its row count. `budget` leases that one read; if it cannot, the window's share
        of the chip area is used instead.
        """
        b = self.pick_bin(bin_size)
        n = self.levels[b].n_obs
        if n == 0:
            return 0
        w = tuple(int(v) for v in window)
        si = self._index_for(b)
        if si is not None:
            try:
                return int(min(si.estimate_rows(w, b), n))
            except Exception:
                pass
        occ = self._occupancy(budget, tier)
        if occ is None:
            return self._area_estimate(w, b)
        C, gx0, gy0, wt = occ
        lam = self._lambda(b)
        G = max(self.levels[b].n_genes, 1)
        ex0, ex1 = (w[0] // b) * b, -(-w[1] // b) * b         # the bins of level b that overlap
        ey0, ey1 = (w[2] // b) * b, -(-w[3] // b) * b
        ny, nx = wt.shape
        cx = gx0 + np.arange(nx, dtype=np.float64) * C
        cy = gy0 + np.arange(ny, dtype=np.float64) * C
        fx = np.clip(np.minimum(cx + C, ex1) - np.maximum(cx, ex0), 0, C) / C
        fy = np.clip(np.minimum(cy + C, ey1) - np.maximum(cy, ey0), 0, C) / C
        jx, jy = np.flatnonzero(fx), np.flatnonzero(fy)
        if not jx.size or not jy.size:
            return 0
        sub = wt[jy[0]:jy[-1] + 1, jx[0]:jx[-1] + 1]
        rows = (C / b) ** 2 * G * (-np.expm1(-lam * sub)) if math.isfinite(lam) else \
            np.full(sub.shape, (C / b) ** 2 * G)
        est = float(fy[jy[0]:jy[-1] + 1] @ rows @ fx[jx[0]:jx[-1] + 1])
        return int(round(min(max(est, 0.0), n)))

    def _area_estimate(self, w: tuple, b: int) -> int:
        ex0, ex1, ey0, ey1 = self.extent
        area = max(ex1 - ex0, 1) * max(ey1 - ey0, 1)
        bx0, bx1 = max((w[0] // b) * b, ex0), min(-(-w[1] // b) * b, ex1)
        by0, by1 = max((w[2] // b) * b, ey0), min(-(-w[3] // b) * b, ey1)
        sel = max(bx1 - bx0, 0) * max(by1 - by0, 0)
        return int(self.levels[b].n_obs * min(sel / area, 1.0))

    def _occupancy(self, budget: Optional[Budget] = None, tier: str = "ram"):
        """(C, x origin, y origin, weight grid): MID per coarse bin of the coarsest level's most
        widespread genes — where the tissue is, and how dense. Read once and cached."""
        if self._occ is not None:
            return self._occ if self._occ is not False else None
        C = self.bins[-1]
        try:
            names, offs, cnts = self.gene_index(C)
        except ValueError:
            self._occ = False
            return None
        ex = self.extent
        gx0, gy0 = (ex[0] // C) * C, (ex[2] // C) * C
        nx, ny = max(1, -(-(ex[1] - gx0) // C)), max(1, -(-(ex[3] - gy0) // C))
        order = np.argsort(-cnts, kind="stable")
        pick, rows = [], 0
        for i in order[:_OCC_GENES]:
            n = int(cnts[i])
            if n <= 0 or (pick and rows + n > _OCC_MAX_ROWS):
                break
            pick.append(i)
            rows += n
        exp = self.f[self.levels[C].path]["expression"]
        biggest = max([int(cnts[i]) for i in pick] or [0])
        need = nx * ny * 8 + biggest * (int(exp.dtype.itemsize) + 16)
        try:
            hold = _Hold(budget, tier, need)
        except CostError:
            return None                                       # not cached: try again with room
        try:
            wt = np.zeros((ny, nx), np.float64)
            for i in pick:
                lo, n = int(offs[i]), int(cnts[i])
                rec = exp[lo:lo + n]
                ix = np.clip((rec["x"].astype(np.int64) - gx0) // C, 0, nx - 1)
                iy = np.clip((rec["y"].astype(np.int64) - gy0) // C, 0, ny - 1)
                np.add.at(wt, (iy, ix), rec["count"] if "count" in (rec.dtype.names or ()) else 1)
                rec = ix = iy = None
        finally:
            hold.release()
        if not wt.any():
            self._occ = False
            return None
        wt += 1e-3 * wt.mean()                                # bins the picked genes missed
        self._occ = (C, gx0, gy0, wt)
        return self._occ

    def _lambda(self, b: int) -> float:
        """Saturation of level `b` per unit of occupancy weight, solved so that the whole grid
        holds exactly the level's rows: a coarse bin holds `(C/b)^2 * genes` rows at most."""
        if b not in self._lam:
            C, _, _, wt = self._occ
            n = float(self.levels[b].n_obs)
            cap = (C / b) ** 2 * max(self.levels[b].n_genes, 1)

            def total(lam):
                return float((cap * -np.expm1(-lam * wt)).sum())
            if n >= cap * wt.size * (1 - 1e-9):
                self._lam[b] = math.inf
            else:
                lo, hi = 0.0, 1.0 / max(float(wt.mean()), 1e-12)
                while total(hi) < n and hi < 1e300:
                    hi *= 2
                for _ in range(80):
                    mid = (lo + hi) / 2
                    if total(mid) < n:
                        lo = mid
                    else:
                        hi = mid
                self._lam[b] = hi
        return self._lam[b]

    # ---- reading ----
    def _scan_rows(self, b: int, r0: int, r1: int, window: tuple, ctx: _ReadCtx, *,
                   gene_ids: bool = False, gene_name: Optional[str] = None,
                   name_dtype=None):
        """Rows [r0, r1) of level `b` inside `window`, one leased HDF5 block at a time. Each
        block is leased (with room for its filter and its kept rows) before it is read; what is
        yielded is the kept rows, still leased until the generator is resumed."""
        exp = self.f[self.levels[b].path]["expression"]
        x0, x1, y0, y1 = window
        lx, ux = max(x0 - b + 1, 0), x1                      # bins that overlap the window
        ly, uy = max(y0 - b + 1, 0), y1
        if ux <= lx or uy <= ly or r1 <= r0:
            return
        fields = exp.dtype.names or ()
        out_row = RESIDENT_BYTES_PER_OBS + (4 if gene_ids else 0)
        if gene_name is not None:
            out_row += np.dtype(name_dtype).itemsize
        per_row = int(exp.dtype.itemsize) + _MASK_BYTES + (4 if gene_ids else 0) + out_row
        chunk_rows = int(exp.chunks[0]) if exp.chunks else 0
        a = r0
        while a < r1:
            c = min(a + ctx.block_rows(per_row, chunk_rows), r1)
            with _Hold(ctx.budget, ctx.tier, (c - a) * per_row) as hold:
                rec = exp[a:c]
                xs, ys = rec["x"], rec["y"]
                m = xs >= lx
                m &= xs < ux
                m &= ys >= ly
                m &= ys < uy
                k = int(np.count_nonzero(m))
                part = None
                if k:
                    part = {"x": _i32(xs[m]), "y": _i32(ys[m]),
                            "count": (rec["count"][m].astype(np.int32) if "count" in fields
                                      else np.ones(k, np.int32))}
                    if gene_ids:
                        part["gene_id"] = self._gene_ids(b, a, c)[m]
                    if gene_name is not None:
                        part["gene"] = np.full(k, gene_name, dtype=name_dtype)
                rec = xs = ys = m = None
                if part is not None:
                    hold.resize(_nbytes(part))
                    yield _Accounted(part)
                    part = None
            a = c

    def _iter_window(self, window: tuple, b: int, genes: Optional[Sequence[str]] = None,
                     gene_ids: bool = False, ctx: Optional[_ReadCtx] = None):
        """The window's rows of level `b`, piece by piece: gene slices for a gene selection,
        the sidecar's row groups when there is one, else the level in row blocks."""
        ctx = ctx or _ReadCtx()
        window = tuple(int(v) for v in window)
        gene_ids = bool(gene_ids) and not genes and self._ids_tile(b)
        if genes:
            names, offs, cnts = self.gene_index(b)
            lookup = {n: i for i, n in enumerate(names)}
            missing = [q for q in genes if q not in lookup]
            if missing:
                raise KeyError(f"not in bin{b}: {missing[:5]}"
                               f"{' ...' if len(missing) > 5 else ''}")
            dt = f"<U{max(max(len(q) for q in genes), 1)}"
            for q in genes:
                i = lookup[q]
                lo, n = int(offs[i]), int(cnts[i])
                if n:
                    yield from self._scan_rows(b, lo, lo + n, window, ctx, gene_name=q, name_dtype=dt)
            return
        si = self._index_for(b)
        if si is not None:
            yield from si.iter_window(window, b, gene_ids=gene_ids, budget=ctx.budget,
                                      tier=ctx.tier, chunk_bytes=ctx.cap())
            return
        yield from self._scan_rows(b, 0, self.levels[b].n_obs, window, ctx, gene_ids=gene_ids)

    def _empty(self, b: int, genes, gene_ids: bool) -> dict:
        out = {"x": np.zeros(0, np.int32), "y": np.zeros(0, np.int32),
               "count": np.zeros(0, np.int32)}
        if genes:
            out["gene"] = np.zeros(0, f"<U{max(max(len(q) for q in genes), 1)}")
        elif gene_ids and self._ids_tile(b):
            out["gene_id"] = np.zeros(0, np.int32)
        return out

    def _read(self, window: tuple, b: int, genes=None, stride: int = 1, gene_ids: bool = False,
              ctx: Optional[_ReadCtx] = None, hold: Optional[_Hold] = None):
        """Read a window into one dict of arrays. `hold` covers what is kept: it grows when the
        pieces exceed it and ends exactly at the output's size. Returns (dict, hold)."""
        ctx = ctx or _ReadCtx()
        hold = hold if hold is not None else _Hold(ctx.budget, ctx.tier)
        pieces, acc, seen = [], 0, 0
        stride = max(int(stride), 1)
        try:
            for item in self._iter_window(window, b, genes, gene_ids, ctx):
                part = item.data
                if stride > 1:                                 # one grid over the kept rows
                    n = len(part["x"])
                    first = (-seen) % stride
                    seen += n
                    part = {k: np.ascontiguousarray(v[first::stride]) for k, v in part.items()}
                acc += _nbytes(part)
                if acc > hold.bytes:
                    hold.add(acc - hold.bytes)
                pieces.append(part)
                part = item = None
            out = _assemble(pieces, ctx, self._empty(b, genes, gene_ids))
            pieces = None
            hold.resize(_nbytes(out))
        except BaseException:
            hold.release()
            raise
        out["bin_size"] = b
        return out, hold

    def read_window(self, window: tuple, bin_size: Optional[int] = None,
                    genes: Optional[Sequence[str]] = None, stride: int = 1,
                    with_gene_ids: bool = False, *, budget: Optional[Budget] = None,
                    tier: str = "ram", chunk_bytes=None):
        """Observations inside a spatial window as a dict of arrays (`x`, `y`, `count` as int32,
        plus `gene` names with `genes=`, or `gene_id` — positions in `genes(bin)` — with
        `with_gene_ids=True`, when the gene index covers the level's rows exactly).

        A bin is in the window when it overlaps it. With `genes=`, only those genes' slices are
        read — contiguous ranges given by the gene index — block by block. Otherwise the sidecar
        index streams the row groups that can hold the window, or the level is read in row
        blocks. With `budget=`, every block is leased before it is read and the rows kept are
        leased as they accumulate; the result is the caller's once returned.
        """
        b = self.pick_bin(bin_size)
        t0 = _time.perf_counter()
        out, hold = self._read(window, b, genes, stride, with_gene_ids,
                               _ReadCtx(budget, tier, chunk_bytes))
        hold.release()
        GefFile._journal_record(self, "read_window", window, b, genes, units=count_units(out),
                             nbytes=_nbytes(out), seconds=_time.perf_counter() - t0)
        return out

    def rasterize(self, window: tuple, bin_size: Optional[int] = None,
                  genes: Optional[Sequence[str]] = None, shape: int = 512, *,
                  budget: Optional[Budget] = None, tier: str = "ram",
                  chunk_bytes=None) -> np.ndarray:
        """A dense MID image of the window — what you actually look at. Pixel `(i, j)` holds the
        bins whose origin falls in its share of the window (integer arithmetic, so a bin-aligned
        grid puts every bin in its own pixel); a bin overlapping the window's low edge goes to the
        first pixel. Rows are streamed into the image a block at a time, never held."""
        b = self.pick_bin(bin_size)
        x0, x1, y0, y1 = (int(v) for v in window)
        shape = int(shape)
        ctx = _ReadCtx(budget, tier, chunk_bytes)
        sx, sy = max(x1 - x0, 1), max(y1 - y0, 1)
        wide, total = False, 0          # MIDs seen so far, exactly (a Python int)
        t0 = _time.perf_counter()
        with _Hold(budget, tier, shape * shape * 4) as hold:   # one 4-byte-per-pixel buffer, then the image
            acc = np.zeros((shape, shape), dtype=np.uint32)    # exact while no pixel can pass 2**32 - 1
            flat = acc.reshape(-1)
            for item in self._iter_window((x0, x1, y0, y1), b, genes, False, ctx):
                p = item.data
                with _Hold(budget, tier, len(p["x"]) * 16):
                    ix = (p["x"].astype(np.int64) - x0) * shape // sx
                    np.clip(ix, 0, shape - 1, out=ix)
                    iy = (p["y"].astype(np.int64) - y0) * shape // sy
                    np.clip(iy, 0, shape - 1, out=iy)
                    iy *= shape
                    iy += ix
                    ix = None
                    block = int(np.sum(p["count"], dtype=np.int64))
                    if not wide and total + block >= 2 ** 32:      # a pixel could now wrap: widen first
                        hold.add(shape * shape * 8)                # 8 bytes a pixel, beside the old 4
                        acc = acc.astype(np.float64)
                        flat = acc.reshape(-1)
                        wide = True
                    np.add.at(flat, iy, p["count"].astype(flat.dtype))
                    total += block
                    iy = None
                p = item = None
            if wide:
                img = acc.astype(np.float32)                       # the image's dtype, rounded once
            else:
                # The float32 image takes the accumulator's own bytes: element i is read and then
                # written in its own place, a block at a time, so no second array is ever made.
                src = acc.reshape(-1)
                img = src.view(np.float32)
                for s in range(0, src.size, 1 << 14):
                    img[s:s + (1 << 14)] = src[s:s + (1 << 14)]
                img = img.reshape(shape, shape)
        GefFile._journal_record(self, "rasterize", (x0, x1, y0, y1), b, genes, unit="pixels",
                                units=shape * shape, nbytes=img.nbytes,
                                seconds=_time.perf_counter() - t0, shape=shape, mids=total)
        return img

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass
        if self._fileobj is not None:
            try:
                self._fileobj.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *e):
        self.close()
        return False

    def __repr__(self) -> str:
        x0, x1, y0, y1 = self.extent
        return (f"GefFile({os.path.basename(self.path)}, bins={self.bins}, "
                f"x {x0}-{x1} y {y0}-{y1}, {human(self.file_bytes)}"
                f"{', remote' if self.remote else ''})")


class StereoSeqView(View):
    """Zoom and pan a Stereo-seq chip, with the bin pyramid doing the work.

    The window is `(x0, x1, y0, y1)` in chip coordinates, half-open. Zooming out picks a coarser
    bin that SAW already computed, so a whole-chip view costs about what a close-up costs — and
    since the level is the decimation, no rows are ever dropped (`stride` is 1). Past the
    coarsest level the view stays on it.

    `cost()` prices what is read and what is kept: `units` / `total_bytes` are the rows kept
    (estimated, see `GefFile.estimate_rows`), `read_bytes` what is read to find them,
    `resident_bytes` the largest chunk `chunks()` hands out, `peak_bytes` what `load()` needs
    (its output plus one read block). `chunks()` streams the read blocks, filtered to the window.
    """

    def __init__(self, gef: GefFile, budget: Optional[Budget] = None,
                 genes: Optional[Sequence[str]] = None, bin_size: Optional[int] = None, *,
                 owns_source: bool = True, chunk_bytes=None):
        self.gef = gef
        self.genes = list(genes) if genes else None
        self._forced_bin = bin_size
        super().__init__(budget, owns_source=owns_source, chunk_bytes=chunk_bytes)

    # ---- View hooks ----
    def _extent(self) -> Extent:
        x0, x1, y0, y1 = self.gef.extent
        return Extent("spots", (max(x1 - x0, 1), max(y1 - y0, 1)),
                      float(RESIDENT_BYTES_PER_OBS))

    def _per_obs(self) -> int:
        if not self.genes:
            return RESIDENT_BYTES_PER_OBS
        return RESIDENT_BYTES_PER_OBS + 4 * max(len(q) for q in self.genes)

    def _full_window(self):
        return self.gef.extent

    def _journal_level(self) -> dict:
        return {"bin": int(self.bin_size)}

    @property
    def stride(self) -> int:
        """Always 1: zoom chooses the bin level, and a level is complete."""
        return 1

    def _zoom_levels(self):
        return range(len(self.gef.bins))

    @property
    def bin_size(self) -> int:
        """The bin level the current zoom resolves to."""
        if self._forced_bin is not None:
            return self.gef.pick_bin(self._forced_bin)
        # Zoom level indexes the pyramid directly: level 0 is the finest bin SAW wrote,
        # each step out is the next coarser one it already computed.
        return self.gef.bins[min(max(self._zoom, 0), len(self.gef.bins) - 1)]

    def _ctx(self):
        return _ReadCtx(self.budget, self.tier, self.chunk_bytes)

    def _kept(self, b: int, window=None) -> int:
        """Rows of the window kept at level `b` (estimated)."""
        w = self._win if window is None else window
        est = self.gef.estimate_rows(w, b, budget=self.budget, tier=self.tier)
        if not self.genes:
            return est
        names, offs, cnts = self.gef.gene_index(b)
        lookup = {n: i for i, n in enumerate(names)}
        n = sum(int(cnts[lookup[q]]) for q in self.genes if q in lookup)
        return int(round(n * est / max(self.gef.levels[b].n_obs, 1)))

    def _read_shape(self, b: int, rp: dict) -> dict:
        """Block geometry for the read path: bytes a row costs while its block is read, rows per
        block now, the smallest block, and the block count."""
        exp = self.gef.f[self.gef.levels[b].path]["expression"]
        out_row = self._per_obs()
        if rp["path"] == "spatial-index":
            from .gefindex import BATCH_ROW_BYTES
            per_row = BATCH_ROW_BYTES + _MASK_BYTES + out_row
            chunk_rows = 0
        else:
            per_row = int(exp.dtype.itemsize) + _MASK_BYTES + out_row
            chunk_rows = int(exp.chunks[0]) if exp.chunks else 0
        rows_read = max(int(rp.get("rows_read", 0)), 0)
        cap = self._plan_cap()
        rows = max(1, cap // per_row)
        if chunk_rows and rows >= chunk_rows:
            rows -= rows % chunk_rows
        min_rows = min(max(rows_read, 1), chunk_rows or 4096)
        if self.genes:
            names, offs, cnts = self.gef.gene_index(b)
            lookup = {n: i for i, n in enumerate(names)}
            sizes = [int(cnts[lookup[q]]) for q in self.genes if q in lookup]
            n_blocks = sum(-(-s // rows) for s in sizes if s > 0)
        else:
            n_blocks = -(-rows_read // rows) if rows_read else 0
        return {"per_row": per_row, "rows": rows, "min_rows": min_rows,
                "n_blocks": max(n_blocks, 1), "rows_read": rows_read}

    def _chunk_plan(self, w, stride):
        b = self.bin_size
        rp = self.gef.read_plan(w, b, self.genes)
        shape = self._read_shape(b, rp)
        kept = self._kept(b, w)
        return shape["n_blocks"], min(shape["rows"], max(kept, 0)) * self._per_obs()

    def cost(self) -> dict:
        b = self.bin_size
        win = self._win
        rp = self.gef.read_plan(win, b, self.genes)
        shape = self._read_shape(b, rp)
        kept = self._kept(b)
        per_obs = self._per_obs()
        total = int(kept * per_obs)
        block = min(shape["rows"], max(shape["rows_read"], 1)) * shape["per_row"]
        min_block = shape["min_rows"] * shape["per_row"]
        widest = max(4 * max(len(q) for q in self.genes), 4) if self.genes else 4
        need = total + min_block + kept * widest               # output, one block, one column
        chunk_need = block
        per_chunk = min(shape["rows"], max(kept, 0)) * per_obs
        cap = self._free() if self.budget else None
        x0, x1, y0, y1 = win
        ex0, ex1, ey0, ey1 = self.gef.extent
        d = {
            "window": win,
            "zoom": self.zoom,
            "bin_size": b,
            "stride": 1,
            "unit": "observations",
            "units": int(kept),
            "resident_bytes": int(per_chunk),
            "resident_human": human(int(per_chunk)),
            "total_bytes": total,
            "total_human": human(total),
            "peak_bytes": int(need),
            "chunk_peak_bytes": int(chunk_need),
            "block_bytes": int(block),
            "chunks": int(shape["n_blocks"]),
            "fraction_of_dataset": min(
                (max(x1 - x0, 0) * max(y1 - y0, 0)) /
                max((ex1 - ex0) * (ey1 - ey0), 1), 1.0),
            "available_bins": self.gef.bins,
            "genes": self.genes,
            "read_bytes": int(rp["read_bytes"]),
            "read_human": human(int(rp["read_bytes"])),
            "read_path": rp["path"],
            "rows_read": int(shape["rows_read"]),
        }
        if "advice" in rp:
            d["read_advice"] = rp["advice"]
        if cap is not None:
            d["budget_available"] = cap
            d["budget_available_human"] = human(cap)
            d["fits"] = chunk_need <= cap
            d["fits_whole"] = need <= cap
            if not d["fits_whole"]:
                coarser = [x for x in self.gef.bins if x > b]
                d["advice"] = (
                    f"the window holds {human(total)} at bin{b} (and is read through "
                    f"{human(min_block)} blocks) but {human(cap)} is free; "
                    + (f"use bin{coarser[0]} (zoom_out), " if coarser else "")
                    + "narrow the window, select specific genes, stream it with .chunks(), "
                      f"or raise the '{self.tier}' budget")
        else:
            d["fits"] = d["fits_whole"] = None
        return d

    def _raster_check(self, b: int, side: int, what: str):
        """Can a `side` x `side` raster of the window at level `b` be built in the budget: the
        image, plus one read block and its pixel indices."""
        if self.budget is None:
            return
        rp = self.gef.read_plan(self._win, b, self.genes)
        shape = self._read_shape(b, rp)
        need = side * side * 4 + shape["min_rows"] * (shape["per_row"] + 16)
        free = self._free()
        if need > free:
            raise CostError(f"{what} needs {human(need)} (a {side}x{side} raster at bin{b} and a read "
                            f"block) but {human(free)} is free; ask for a smaller image, a "
                            f"coarser bin, or raise the '{self.tier}' budget")

    def _overview(self, w, n) -> Overview:
        """A thumbnail of the window from the coarsest bin that still resolves it into at least
        `min(n, 32)` bins a side, one pixel per bin (at most `n`, and 512, a side)."""
        side_max = max(1, min(int(n), 512))
        span = max(w[1] - w[0], w[3] - w[2], 1)
        want = min(side_max, 32)
        fit = [b for b in self.gef.bins if span / b >= want]
        b = fit[-1] if fit else self.gef.bins[0]
        side = int(min(side_max, max(1, -(-span // b))))
        self._raster_check(b, side, "the overview")
        img = self.gef.rasterize(w, b, self.genes, shape=side, budget=self.budget, tier=self.tier,
                                 chunk_bytes=self.chunk_bytes)
        rp = self.gef.read_plan(w, b, self.genes)
        return Overview("spots", w, np.arange(img.shape[1]), {"thumbnail": img},
                        cheap=False, source_bytes=int(rp["read_bytes"]))

    def _load(self, w, stride):
        """The window at the current bin: the planned output is leased first, then the level is
        read through blocks (gene slices, sidecar row groups, or row blocks), keeping only the
        rows that overlap the window."""
        b = self.bin_size
        self._hold(self._kept(b, w) * self._per_obs())
        out, _ = self.gef._read(w, b, self.genes, 1, False, self._ctx(), self._lease)
        return out

    def _chunks(self, w, stride):
        b = self.bin_size
        for item in self.gef._iter_window(w, b, self.genes, False, self._ctx()):
            item.data["bin_size"] = b
            yield item

    # ---- Stereo-seq specifics ----
    def select_genes(self, genes: Optional[Sequence[str]]) -> "StereoSeqView":
        """Restrict to specific genes. Their rows are contiguous in the GEF, so this is a
        slice, not a scan."""
        if genes:
            known = set(self.gef.genes(self.bin_size))
            missing = [g for g in genes if g not in known]
            if missing:
                raise KeyError(f"not in this chip: {missing[:5]}"
                               f"{' ...' if len(missing) > 5 else ''}")
        self.genes = list(genes) if genes else None
        return self

    def use_bin(self, bin_size: Optional[int]) -> "StereoSeqView":
        """Pin a bin level, or pass None to let zoom choose again."""
        self._forced_bin = bin_size
        return self

    def auto_bin(self, target: Optional[str] = None) -> "StereoSeqView":
        """Finest bin whose window loads within the budget (or `target` bytes), leaving 20 %
        headroom; the coarsest bin when none does."""
        if target is not None:
            cap = parse_size(target) if isinstance(target, str) else int(target)
        else:
            cap = self._free() if self.budget else 256 * 1024 ** 2
        cap = int(cap * 0.8)
        for b in self.gef.bins:
            self._forced_bin = b
            if self.cost()["peak_bytes"] <= cap:
                return self
        self._forced_bin = self.gef.bins[-1]
        return self

    def image(self, shape: int = 512) -> np.ndarray:
        """Rasterize the current window at the current bin level, checked against the budget
        first and streamed into the image block by block."""
        b = self.bin_size
        self._raster_check(b, int(shape), "the image")
        return self.gef.rasterize(self._win, b, self.genes, int(shape), budget=self.budget,
                                  tier=self.tier, chunk_bytes=self.chunk_bytes)

    def gene_profile(self, k: int = 20) -> list[tuple[str, int]]:
        """Top genes by MID inside the current window, at the current bin (ties in gene order).
        Streams the window once, summing MID per gene; with a gene selection, ranks those."""
        b = self.bin_size
        names = self.gef.genes(b)
        if self.budget is not None:
            rp = self.gef.read_plan(self._win, b, self.genes)
            shape = self._read_shape(b, rp)
            need = len(names) * 8 + shape["min_rows"] * (shape["per_row"] + 4)
            if need > self._free():
                raise CostError(f"ranking genes needs {human(need)} but {human(self._free())} is "
                                f"free; raise the '{self.tier}' budget")
        acc = np.zeros(len(names), np.float64)
        with _Hold(self.budget, self.tier, acc.nbytes):
            if self.genes:
                lookup = {n: i for i, n in enumerate(names)}
                for item in self.gef._iter_window(self._win, b, self.genes, False, self._ctx()):
                    p = item.data
                    acc[lookup[str(p["gene"][0])]] += float(p["count"].sum(dtype=np.int64))
            else:
                for item in self.gef._iter_window(self._win, b, None, True, self._ctx()):
                    p = item.data
                    ok = p["gene_id"] >= 0
                    np.add.at(acc, p["gene_id"][ok], p["count"][ok])
            order = np.argsort(-acc, kind="stable")[:k]
            return [(str(names[i]), int(acc[i])) for i in order]

    def __repr__(self) -> str:
        c = self.cost()
        fit = "" if c.get("fits_whole") is None else ("  fits" if c["fits_whole"] else "  TOO BIG")
        return (f"StereoSeqView(x {c['window'][0]}-{c['window'][1]} "
                f"y {c['window'][2]}-{c['window'][3]}, bin{c['bin_size']}, "
                f"~{c['units']:,} obs, {c['total_human']}{fit})")


class GemView(View):
    """Zoom and pan a GEM text matrix, backed by a one-pass spatial index.

    GEF is preferable when you have it — it carries the bin pyramid. GEM is what archives and
    converters emit, so it gets the same navigation from an index GEMINGA builds itself. The
    index's extent is already half-open (its max is one past the last coordinate), and so is
    the window. Zoom decimates the window's rows on one grid (in file order) for `load()` and
    `chunks()` alike; the parser drops rows outside the window before they reach memory.
    """

    def __init__(self, index: GemIndex, budget: Optional[Budget] = None,
                 path: Optional[str] = None, *, owns_source: bool = True, chunk_bytes=None):
        self.index = index
        self.path = path or index.path
        super().__init__(budget, owns_source=owns_source, chunk_bytes=chunk_bytes)

    def _extent(self) -> Extent:
        x0, x1, y0, y1 = self.index.extent
        return Extent("observations", (max(x1 - x0, 1), max(y1 - y0, 1)), 16.0)

    def _full_window(self):
        return tuple(int(v) for v in self.index.extent)

    def _rows(self, w) -> int:
        return int(self.index.rows_in(int(w[0]), int(w[1]), int(w[2]), int(w[3])))

    def _chunk_plan(self, w, stride):
        resident = _grid_count(0, self._rows(w), stride) * 16
        cap = self._plan_cap()
        return max(math.ceil(resident / cap), 1), min(resident, cap)

    def _load_need(self, w, stride, total):
        return int(total + min(self._plan_cap(), max(total, 1 << 16)))

    def cost(self) -> dict:
        d = super().cost()
        c = self.index.cost(int(self._win[0]), int(self._win[1]),
                            int(self._win[2]), int(self._win[3]))
        rows = _grid_count(0, int(c["rows"]), self.stride)
        d["units"] = rows
        d["unit"] = "observations"
        d["total_bytes"] = int(c["resident_bytes"] / max(int(c["rows"]), 1) * rows)
        d["total_human"] = human(d["total_bytes"])
        d["transfer_bytes"] = c["transfer_bytes"]
        d["peak_bytes"] = int(self._load_need(self._win, self.stride, d["total_bytes"]))
        cap = d.get("budget_available")
        if cap is not None:
            d["fits_whole"] = d["peak_bytes"] <= cap
            d.pop("advice", None)
            if not d["fits_whole"]:
                need = d["peak_bytes"] / max(cap, 1)
                d["advice"] = (f"the window holds {d['total_human']} but {human(cap)} is free; "
                               f"zoom out {max(1, math.ceil(math.log2(need) / 2))} level(s), "
                               f"narrow the window, or raise the '{self.tier}' budget")
        return d

    def _overview(self, w, n) -> Overview:
        side, flat = self.index.zoom(int(w[0]), int(w[1]), int(w[2]), int(w[3]), n)
        img = np.asarray(flat, dtype=np.float32).reshape(side, side)
        return Overview("observations", w, np.arange(side), {"thumbnail": img},
                        cheap=True, source_bytes=self.index.grid ** 2 * 8)

    def _pieces(self, w, stride: int, chunk: int):
        """The window's rows on the grid, batch by batch (each batch leased by the parser until
        the next one is asked for)."""
        import pyarrow as pa
        win = (int(w[0]), int(w[1]), int(w[2]), int(w[3]))
        st = GemStream(self.path, str(max(int(chunk), 1)), win, self.budget)
        seen = 0
        for batch in st:
            n = batch.num_rows
            first = _grid_first(seen, 0, stride) - seen
            seen += n
            if first < n:
                if stride == 1:
                    yield _Accounted(pa.Table.from_batches([batch]))
                else:
                    idx = np.arange(first, n, stride, dtype=np.int64)
                    yield _Accounted(pa.Table.from_batches([batch.take(pa.array(idx))]))
            batch = None
        st = None

    def _load(self, w, stride):
        import pyarrow as pa
        self._hold(self.cost()["total_bytes"])
        parts, acc = [], 0
        for item in self._pieces(w, stride, self._chunk_cap()):
            acc += _arrow_bytes(item.data)
            if acc > self._lease.bytes:
                self._lease.add(acc - self._lease.bytes)
            parts.append(item.data)
            item = None
        t = pa.concat_tables(parts) if len(parts) > 1 else parts[0] if parts else pa.table({})
        self._settle(acc)
        return t

    def _chunks(self, w, stride):
        yield from self._pieces(w, stride, self._chunk_cap())

    def top_genes(self, k: int = 20):
        return self.index.top_genes(k)


class StereoSeqDataset:
    """A chip as one object: expression plus the images registered to it.

    Stereo-seq analysis is multi-modal by construction — the ssDNA or H&E image shares a
    coordinate frame with the expression matrix, which is what makes cell segmentation
    meaningful. Holding them together means one window selection drives both. The views it
    hands out borrow its files: closing a view leaves the dataset open.
    """

    def __init__(self, gef: Optional[str] = None, image: Optional[str] = None,
                 budget: Optional[Budget] = None, group: str = "geneExp",
                 image_scale: float = 1.0, image_origin: tuple[int, int] = (0, 0)):
        self.budget = budget
        self.gef = GefFile(gef, group) if gef else None
        self.image_path = os.fspath(image) if image else None
        self.image_scale = float(image_scale)
        self.image_origin = tuple(image_origin)
        self._img = None
        if not self.gef and not image:
            raise ValueError("give at least one of gef= or image=")

    @property
    def image(self):
        """The registered image, opened lazily as a tiled array."""
        if self._img is None and self.image_path:
            from .formats import TiffTileArray
            self._img = TiffTileArray(self.image_path)
        return self._img

    def expression(self, **kw) -> StereoSeqView:
        if not self.gef:
            raise ValueError("no GEF attached")
        kw.setdefault("owns_source", False)
        return StereoSeqView(self.gef, self.budget, **kw)

    def morphology(self):
        from .view import ImageView
        if self.image is None:
            raise ValueError("no image attached")
        return ImageView(self.image, self.budget, owns_source=False)

    def chip_to_image(self, window: tuple) -> tuple:
        """Chip (x0,x1,y0,y1) -> image (y0,y1,x0,x1), applying scale and origin."""
        x0, x1, y0, y1 = window
        ox, oy = self.image_origin
        s = self.image_scale
        return (int((y0 - oy) * s), int((y1 - oy) * s),
                int((x0 - ox) * s), int((x1 - ox) * s))

    def _image_axes(self):
        from .formats import _spatial_axes
        a = self.image
        shape = tuple(int(n) for n in a.shape)
        _, ya, xa = _spatial_axes(a, shape)
        return shape, ya, xa

    def _morphology_grid(self, window: tuple, shape: int):
        """The image sampled on the expression raster's grid: pixel (i, j) is the image at the
        centre of the chip cell `rasterize` puts in pixel (i, j) — nearest neighbour, zero
        outside the image. Read with a step, so only about (2 x shape)^2 pixels are decoded into
        memory, however large the crop."""
        a = self.image
        full, ya_ax, xa_ax = self._image_axes()
        H, W = full[ya_ax], full[xa_ax]
        x0, x1, y0, y1 = window
        ox, oy = self.image_origin
        s = self.image_scale
        centres = (np.arange(shape) + 0.5) / shape
        ry = np.floor((y0 + centres * (y1 - y0) - oy) * s).astype(np.int64)
        rx = np.floor((x0 + centres * (x1 - x0) - ox) * s).astype(np.int64)
        iny, inx = (ry >= 0) & (ry < H), (rx >= 0) & (rx < W)
        if not iny.any() or not inx.any():
            return None
        ya, yb = int(ry[iny].min()), int(ry[iny].max()) + 1
        xa, xb = int(rx[inx].min()), int(rx[inx].max()) + 1
        sy, sx = max(1, (yb - ya) // shape), max(1, (xb - xa) // shape)
        ny, nx = _grid_count(ya, yb, sy), _grid_count(xa, xb, sx)
        px = int(np.dtype(a.dtype).itemsize * math.prod(
            n for i, n in enumerate(full) if i not in (ya_ax, xa_ax)))
        tile = int(getattr(a, "th", 512)) * int(getattr(a, "tw", 512)) * px
        need = (ny * nx + shape * shape) * px + tile
        if self.budget is not None:
            free = int(self.budget.available("ram"))
            if need > free:
                raise CostError(f"the morphology crop needs {human(need)} ({ny}x{nx} samples, a "
                                f"{shape}x{shape} grid and a decoded tile) but {human(free)} is "
                                f"free; ask for a smaller shape or raise the 'ram' budget")
        with _Hold(self.budget, "ram", need):
            key = [slice(None)] * len(full)
            key[ya_ax], key[xa_ax] = slice(ya, yb, sy), slice(xa, xb, sx)
            crop = np.moveaxis(np.asarray(a[tuple(key)]), (ya_ax, xa_ax), (0, 1))
            iy = np.clip((ry[iny] - ya) // sy, 0, ny - 1)
            ix = np.clip((rx[inx] - xa) // sx, 0, nx - 1)
            out = np.zeros((shape, shape) + crop.shape[2:], crop.dtype)
            out[np.ix_(np.flatnonzero(iny), np.flatnonzero(inx))] = crop[np.ix_(iy, ix)]
            crop = None
            out = np.moveaxis(out, (0, 1), (ya_ax, xa_ax))
        return out, (ya, yb, xa, xb)

    def paired(self, window: tuple, shape: int = 512):
        """Expression raster and morphology for the same chip window on the same `shape` x
        `shape` grid (pixel (i, j) of one is pixel (i, j) of the other), both checked against the
        budget before anything is read."""
        out = {}
        window = tuple(int(v) for v in window)
        if self.gef:
            v = StereoSeqView(self.gef, self.budget, owns_source=False)
            v.select(*window)
            out["expression"] = v.image(shape)
            out["bin_size"] = v.bin_size
        if self.image is not None:
            m = self._morphology_grid(window, int(shape))
            if m is not None:
                out["morphology"], out["morphology_window"] = m
        return out

    def close(self):
        if self.gef:
            self.gef.close()
        if self._img is not None:
            self._img.close()

    def __enter__(self):
        return self

    def __exit__(self, *e):
        self.close()
        return False

    def __repr__(self) -> str:
        bits = []
        if self.gef:
            bits.append(f"gef={os.path.basename(self.gef.path)} bins={self.gef.bins}")
        if self.image_path:
            bits.append(f"image={os.path.basename(self.image_path)}")
        return f"StereoSeqDataset({', '.join(bits)})"


# ---------------------------------------------------------------------------
# Openers
# ---------------------------------------------------------------------------

def open_gef(path: str, budget: Optional[Budget] = None, group: str = "geneExp",
             genes: Optional[Sequence[str]] = None,
             bin_size: Optional[int] = None) -> StereoSeqView:
    """A navigable view over a Stereo-seq GEF, using its own bin pyramid as zoom levels."""
    return StereoSeqView(GefFile(path, group), budget, genes, bin_size)


def index_gem(path: str, grid: Optional[int] = None, sidecar: Optional[str] = None,
              reuse: bool = True) -> GemIndex:
    """Build (or reuse) the spatial index that makes a GEM navigable.

    A sidecar newer than the GEM is reused when it was built with the requested `grid`; asking
    for another grid rebuilds it. `grid=None` takes whatever sidecar there is (built with 512
    when there is none). A sidecar that cannot be written beside the GEM goes under
    `~/.geminga/gefidx/` (where the GEF indexes go), is looked for there too, and a warning
    names both paths."""
    from .gefindex import _home_dir
    path = os.fspath(path)
    sc = sidecar or (path + ".gemidx.json")
    home = os.path.join(_home_dir(os.path.abspath(path)), os.path.basename(sc))
    if reuse:
        for cand in (sc, home):
            if os.path.exists(cand) and os.path.getmtime(cand) >= os.path.getmtime(path):
                try:
                    idx = GemIndex.load(cand)
                    if grid is None or int(idx.grid) == int(grid):
                        return idx
                except Exception:
                    pass
    idx = GemIndex.build(path, int(grid or 512))
    try:
        idx.save(sc)
        return idx
    except OSError as beside:
        if os.path.abspath(home) == os.path.abspath(sc):      # already the private copy: nowhere else
            warnings.warn(f"GEM index of {path} could not be saved at {sc} ({beside}); "
                          "it will be rebuilt next time", stacklevel=2)
            return idx
        try:
            os.makedirs(os.path.dirname(home), exist_ok=True)
            idx.save(home)
        except OSError as err:
            warnings.warn(f"GEM index of {path} could not be saved at {sc} ({beside}) or under "
                          f"{home} ({err}); it will be rebuilt next time", stacklevel=2)
            return idx
        warnings.warn(f"GEM index of {path} could not be saved at {sc} ({beside}); saved under "
                      f"{home} instead", stacklevel=2)
    return idx


def open_gem(path: str, budget: Optional[Budget] = None, grid: Optional[int] = None) -> GemView:
    """A navigable view over a Stereo-seq GEM matrix."""
    path = os.fspath(path)
    return GemView(index_gem(path, grid), budget, path)


def open_stereoseq(gef: Optional[str] = None, image: Optional[str] = None,
                   budget: Optional[Budget] = None, **kw) -> StereoSeqDataset:
    """A chip with its expression matrix and registered morphology image together."""
    return StereoSeqDataset(gef, image, budget, **kw)
