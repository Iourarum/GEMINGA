"""10x Genomics deliverables: the format most of a 2025-26 glioma atlas actually arrives in.

Cell Ranger, Space Ranger, Xenium Ranger and Cell Ranger ARC all emit the same two containers:
an HDF5 file with a `matrix/` group, or a MatrixMarket triple (`matrix.mtx.gz`,
`barcodes.tsv.gz`, `features.tsv.gz`). Both carry a `feature_type` column, and that column is
the single most useful piece of metadata in the file: it says whether the rows are
`Gene Expression`, `Antibody Capture` (CITE-seq ADT), `Peaks` (scATAC), `CRISPR Guide Capture`
(Perturb-seq / CRISPRi screens), or `Multiplexing Capture`. A multi-modal dataset is not a
special format — it is one 10x matrix with several feature types stacked.

So the reader here splits by feature type first and asks "what modalities are in this file"
before "how big is it". The matrix is CSC (features × barcodes), which means a *column block*
is a block of cells and reads as one contiguous slice of `data` and `indices`. Cell Ranger v2
files (a root group named for the genome, with `genes` / `gene_names`) open the same way.

**What a modality read costs.** CSC stores every feature of a cell together, so there is no
slice of `data` / `indices` that holds only the ADT rows of a block of cells: the whole column
block is read (all feature types) and the modality's rows are sliced out afterwards. Chunks are
therefore cut — from `indptr`, so a skewed matrix is planned from its real cell sizes — by the
bytes of the *full* column block, the lease covers that block, and `plan()` reports it as the
chunk size. What a modality changes is the size of what the consumer is handed, not what is
read. `modality_nnz()` counts a modality's entries per cell with one pass over `indices`
(cached) for callers that need the output size.

MatrixMarket is the awkward one: COO, text, gzipped, unindexed. It streams in byte-bounded
batches, but cannot be navigated without a full pass. The honest move is `to_h5()` — convert
once into staging, navigate forever. `to_h5()` stays in memory when the COO fits half the
free budget and otherwise makes two passes through memory-mapped scratch files (see there).
"""
from __future__ import annotations

import gzip
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

import numpy as np

from ._core import Budget, human
from .formats import Plan, _Governed, _largest, block_bytes, csr_bounds, resolve_chunk_bytes
from .view import Extent, Overview, View

__all__ = ["TenXMatrix", "MtxTriple", "TenXView", "ModalityReport", "modalities",
           "FEATURE_TYPES", "open_tenx"]

#: Cell Ranger feature_type strings and what they mean for analysis.
FEATURE_TYPES = {
    "Gene Expression":       ("RNA", "transcripts per cell"),
    "Antibody Capture":      ("ADT", "surface protein per cell — CITE-seq / TotalSeq"),
    "Peaks":                 ("ATAC", "chromatin accessibility peaks per cell"),
    "CRISPR Guide Capture":  ("GUIDE", "sgRNA UMIs per cell — Perturb-seq / CRISPRi / CROP-seq"),
    "Multiplexing Capture":  ("HTO", "hashing / CMO tags for sample demultiplexing"),
    "Custom":                ("CUSTOM", "user-defined features"),
}

_V2_KEYS = ("data", "indices", "indptr", "shape", "barcodes", "genes")      # a Cell Ranger v2 matrix group


@dataclass
class Modality:
    feature_type: str
    label: str
    n_features: int
    row_start: int              # contiguous row range in the stacked matrix
    row_end: int
    nnz: int = 0
    note: str = ""

    def __repr__(self) -> str:
        return f"{self.label}({self.n_features:,} features, {self.nnz:,} nnz)"


def _split_modalities(ftypes: np.ndarray) -> dict:
    out: dict[str, Modality] = {}
    for ft in dict.fromkeys(ftypes.tolist()):                    # first-seen order
        idx = np.flatnonzero(ftypes == ft)
        label, note = FEATURE_TYPES.get(ft, (ft.upper().replace(" ", "_"), ""))
        contiguous = bool(idx.size and idx[-1] - idx[0] + 1 == idx.size)
        m = Modality(ft, label, int(idx.size), int(idx[0]), int(idx[-1]) + 1, note=note)
        if not contiguous:
            m.note = "feature rows are interleaved; modality slicing falls back to a mask"
        out[label] = m
    return out


def _matrix_groups(f) -> tuple:
    """(v3-style candidates, v2-style candidates) among the root groups of an HDF5 file."""
    import h5py
    groups = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
    v3 = [k for k in groups if "matrix" in k.lower()]
    v2 = [k for k in groups if all(x in f[k] for x in _V2_KEYS)]
    return v3, v2


class TenXMatrix(_Governed):
    """A Cell Ranger HDF5 matrix, opened lazily, split by feature type.

    Blocks are *cells* (CSC columns), so a chunk is always a set of whole cells across every
    modality — the unit a downstream tool wants. Restrict to one modality with `modality=`
    and the chunks hold only that modality's rows; see the module docstring for why the read
    (and so the lease and the plan) is still the whole column block.

    Both the v3+ layout (``matrix/`` with ``features/``) and Cell Ranger v2's (a root group
    named for the genome, with ``genes`` and ``gene_names``) are understood. A v2 file with
    several genome groups needs ``group=`` to say which.
    """

    def __init__(self, path: str, chunk_bytes: str | int = "256MB",
                 budget: Optional[Budget] = None, modality: Optional[str] = None,
                 group: str = "matrix"):
        super().__init__(budget)
        import h5py
        self.path = path
        self.f = h5py.File(path, "r")
        try:
            self._setup(chunk_bytes, budget, modality, group)
        except BaseException:
            self.f.close()
            raise

    def _pick_group(self, group: str) -> str:
        import h5py
        f = self.f
        if group in f and isinstance(f[group], h5py.Group):
            return group
        v3, v2 = _matrix_groups(f)
        if v3:
            return v3[0]
        if len(v2) == 1:
            return v2[0]
        if len(v2) > 1:
            raise ValueError(f"{self.path}: several genome groups {v2}; pass group= to choose one")
        raise ValueError(f"{self.path}: no '{group}' group; keys {list(f.keys())}")

    def _setup(self, chunk_bytes, budget, modality, group):
        self.group = self._pick_group(group)
        self.g = self.f[self.group]
        self.shape = tuple(int(x) for x in self.g["shape"][:])       # (features, barcodes)
        self.n_features, self.n_cells = self.shape
        self.indptr = self.g["indptr"][:]                            # n_cells + 1
        self.nnz = int(self.indptr[-1])
        self.dtype = self.g["data"].dtype
        self._itemsize = self.dtype.itemsize + self.g["indices"].dtype.itemsize
        self.barcodes = _decode(self.g["barcodes"][:])
        feats = self.g["features"] if "features" in self.g else None
        if feats is not None:
            self.feature_ids = _decode(feats["id"][:])
            self.feature_names = _decode(feats["name"][:])
            ftypes = _decode(feats["feature_type"][:]) if "feature_type" in feats \
                else np.array(["Gene Expression"] * self.n_features)
            self.genome = _decode(feats["genome"][:]) if "genome" in feats else None
        elif "genes" in self.g:                                      # Cell Ranger v2: the group is the genome
            self.feature_ids = _decode(self.g["genes"][:])
            self.feature_names = _decode(self.g["gene_names"][:]) if "gene_names" in self.g \
                else self.feature_ids
            ftypes = np.array(["Gene Expression"] * self.n_features)
            self.genome = np.full(self.n_features, self.group)
        else:
            raise ValueError(f"{self.path}: group {self.group!r} has neither 'features' (Cell Ranger "
                             f"v3+) nor 'genes' (v2); it holds {sorted(self.g.keys())}")
        self.feature_types = ftypes
        self.modalities: dict[str, Modality] = self._split(ftypes)
        self.modality = modality
        if modality is not None and modality not in self.modalities:
            raise KeyError(f"no modality {modality!r}; this file has "
                           f"{list(self.modalities)}")
        self.chunk_bytes = resolve_chunk_bytes(chunk_bytes, "256MB", budget)
        self._bounds, self._bounds_for = None, None
        self._mod_nnz: dict = {}

    def _split(self, ftypes: np.ndarray) -> dict:
        return _split_modalities(ftypes)

    # ---- sizing ----
    def _rows_for(self, modality: Optional[str]) -> Optional[tuple]:
        if modality is None:
            return None
        m = self.modalities[modality]
        return (m.row_start, m.row_end)

    def _cell_bounds(self, c0: int = 0, c1: Optional[int] = None, cap: Optional[int] = None):
        """Cell boundaries ``[c0, ..., c1]`` of blocks whose full column block (data + indices,
        every feature type) is at most ``cap`` bytes (default ``chunk_bytes``) — except a single
        cell that alone is larger. Cut from ``indptr``, so a skewed matrix is planned exactly."""
        c1 = self.n_cells if c1 is None else int(c1)
        if cap is None and c0 == 0 and c1 == self.n_cells:
            if self._bounds is None or self._bounds_for != self.chunk_bytes:
                self._bounds = csr_bounds(self.indptr, self._itemsize, self.chunk_bytes)
                self._bounds_for = self.chunk_bytes
            return self._bounds
        return csr_bounds(self.indptr, self._itemsize, self.chunk_bytes if cap is None else cap, c0, c1)

    def _cells_per_chunk(self) -> int:
        """Cells in the biggest chunk by cell count."""
        d = np.diff(self._cell_bounds())
        return int(d.max()) if d.size else 0

    def _scan_cap(self) -> int:
        """Bytes one pass over the file may hold: the chunk size, but never more than is free
        right now, so a scan cannot wait on a lease held by an iteration of its own."""
        cap = self.chunk_bytes
        if self._budget is not None:
            cap = min(cap, int(self._budget.available(self._tier)))
        return max(cap, 16 * 1024)

    def plan(self) -> Plan:
        bounds = self._cell_bounds()
        sizes = block_bytes(self.indptr, bounds, self._itemsize)
        big = _largest(sizes, bounds, self.chunk_bytes, self._budget, self._tier,
                       "cell_start", "cell_end")
        cells = np.diff(bounds)
        detail = {"n_cells": self.n_cells, "n_features": self.n_features, "nnz": self.nnz,
                  "cells_per_chunk": int(cells[big["largest_chunk"]["index"]]) if sizes.size else 0,
                  "max_cells_per_chunk": int(cells.max()) if cells.size else 0,
                  "modalities": {k: v.n_features for k, v in self.modalities.items()},
                  "modality": self.modality, "chunk_bytes": self.chunk_bytes,
                  "file_size": os.path.getsize(self.path),
                  "modality_nnz": (int(self._mod_nnz[self.modality].sum())
                                   if self.modality in self._mod_nnz else None)}
        detail.update(big)
        if self.modality:
            detail["note"] = (f"CSC: the whole column block (every feature type) is read for each "
                              f"chunk and the {self.modality} rows are sliced out afterwards, so "
                              f"the lease and this plan cover the full block")
        return Plan(kind=f"10x-h5/{self.modality or 'all'}", source=self.path, n_chunks=len(sizes),
                    bytes_per_chunk=detail["largest_chunk_bytes"],
                    total_bytes=self.nnz * self._itemsize, unit="cells", detail=detail)

    def modality_nnz(self, modality: Optional[str] = None) -> np.ndarray:
        """Entries per cell that fall inside a modality, counted once from ``indices`` and cached.

        One pass over ``indices`` in blocks bounded by the budget; a modality that is the whole
        matrix is just ``diff(indptr)``. The result is the size of what a modality read hands
        over (times the entry size), as opposed to what it has to read.
        """
        modality = modality or self.modality
        m = self.modalities[modality] if modality else None
        if m is None or m.n_features == self.n_features:            # the whole matrix: indptr says it
            if "*" not in self._mod_nnz:
                self._mod_nnz["*"] = np.diff(self.indptr.astype(np.int64))
            return self._mod_nnz["*"]
        if modality in self._mod_nnz:
            return self._mod_nnz[modality]
        lut = self.feature_types == m.feature_type                  # n_features booleans
        ip = self.indptr.astype(np.int64)
        block = max(1024, self._scan_cap() // 16)                   # entries per step
        cum = np.zeros(self.n_cells + 1, np.int64)                  # mask entries before each ip[j]
        running = 0
        with self._transient(block * 16):                           # indices + mask + cumsum
            for lo in range(0, self.nnz, block):
                hi = min(lo + block, self.nnz)
                cs = np.cumsum(lut[self.g["indices"][lo:hi]], dtype=np.int64)
                j0 = int(np.searchsorted(ip, lo, side="right"))
                j1 = int(np.searchsorted(ip, hi, side="right"))
                if j1 > j0:
                    cum[j0:j1] = running + cs[ip[j0:j1] - lo - 1]
                running += int(cs[-1])
        per_cell = np.diff(cum)
        self._mod_nnz[modality] = per_cell
        return per_cell

    # ---- reading ----
    def block(self, c0: int, c1: int, modality: Optional[str] = None):
        """Cells c0:c1 as a scipy CSC matrix (features × cells), optionally one modality.

        CSC keeps all of a cell's features together, so the whole column block is read even for
        one modality (its rows are sliced out afterwards). No lease is taken here — the caller
        accounts for the block: ``__iter__`` leases it, ``TenXView`` holds what it hands out.
        """
        import scipy.sparse as sp
        c0, c1 = int(c0), int(c1)
        if not 0 <= c0 <= c1 <= self.n_cells:
            raise IndexError(f"cells {c0}:{c1} are outside 0:{self.n_cells}")
        modality = modality or self.modality
        lo, hi = int(self.indptr[c0]), int(self.indptr[c1])
        data = self.g["data"][lo:hi]
        indices = self.g["indices"][lo:hi]
        indptr = self.indptr[c0:c1 + 1] - lo
        X = sp.csc_matrix((data, indices, indptr), shape=(self.n_features, c1 - c0))
        del data, indices, indptr
        if modality:
            # Row-slicing copies the kept entries while the full block is alive, so the transient
            # is the block plus the modality's share of it (TenXView.cost()['peak_read_bytes']);
            # a mask-and-remap on the raw arrays was measured and was not smaller.
            r0, r1 = self._rows_for(modality)
            m = self.modalities[modality]
            if m.n_features != self.n_features:
                X = X[self.feature_types == m.feature_type] if "interleaved" in m.note else X[r0:r1]
        return X

    def __iter__(self) -> Iterator[dict]:
        bounds = self._cell_bounds()
        for i in range(len(bounds) - 1):
            c0, c1 = int(bounds[i]), int(bounds[i + 1])
            lo, hi = int(self.indptr[c0]), int(self.indptr[c1])
            nbytes = (hi - lo) * self._itemsize
            self._swap_lease(nbytes)                                 # lease first, then read
            X = self.block(c0, c1)
            out = {"X": X, "cell_start": c0, "cell_end": c1, "barcodes": self.barcodes[c0:c1],
                   "modality": self.modality or "all", "resident_bytes": nbytes}
            yield out
            out = X = None                                           # let go before the next read

    def features(self, modality: Optional[str] = None) -> np.ndarray:
        modality = modality or self.modality
        if not modality:
            return self.feature_names
        m = self.modalities[modality]
        if "interleaved" in m.note:
            return self.feature_names[self.feature_types == m.feature_type]
        return self.feature_names[m.row_start:m.row_end]

    def per_cell_totals(self, modality: Optional[str] = None, c0: int = 0,
                        c1: Optional[int] = None) -> np.ndarray:
        """UMI (or fragment / guide) total per cell, float64 — one pass over `data`.

        Reads in blocks the budget allows, each leased while it is read. ``c0:c1`` restricts
        the pass to a window of cells (the result is then that long). Empty cells total 0.
        """
        modality = modality or self.modality
        c1 = self.n_cells if c1 is None else int(c1)
        c0 = int(c0)
        out = np.zeros(c1 - c0, dtype=np.float64)
        whole = (not modality) or self.modalities[modality].n_features == self.n_features
        bounds = csr_bounds(self.indptr, self._itemsize, self._scan_cap(), c0, c1)
        for i in range(len(bounds) - 1):
            a, b = int(bounds[i]), int(bounds[i + 1])
            lo, hi = int(self.indptr[a]), int(self.indptr[b])
            with self._transient((hi - lo) * self._itemsize):
                if whole:
                    ip = self.indptr[a:b + 1].astype(np.int64) - lo
                    nz = np.diff(ip) > 0                             # reduceat needs non-empty segments
                    if nz.any():
                        d = self.g["data"][lo:hi]
                        out[a - c0:b - c0][nz] = np.add.reduceat(d, ip[:-1][nz], dtype=np.float64)
                else:
                    X = self.block(a, b, modality)
                    out[a - c0:b - c0] = np.asarray(X.sum(0, dtype=np.float64)).ravel()
        return out

    def close(self):
        super().close()
        try:
            self.f.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        mods = ", ".join(f"{k}={v.n_features:,}" for k, v in self.modalities.items())
        return (f"TenXMatrix({os.path.basename(self.path)}, {self.n_cells:,} cells x "
                f"{self.n_features:,} features [{mods}], {self.nnz:,} nnz)")


_MTX_RE = re.compile(r"^(?P<prefix>(?:.*[_.\-])?)matrix\.mtx(?:\.gz)?$", re.I)
_PLAIN = {"mtx": ("matrix.mtx.gz", "matrix.mtx"),
          "bc": ("barcodes.tsv.gz", "barcodes.tsv"),
          "ft": ("features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv")}


def _find(d: str, names: Sequence[str], prefix: Optional[str] = None) -> Optional[str]:
    """A file of directory `d` called one of `names`.

    With a `prefix` (possibly ``""``) only ``prefix + name`` is looked for. Without one the plain
    names are tried first and then GEO-style prefixed ones (``GSM123_matrix.mtx.gz``, separator
    ``_``, ``.`` or ``-``): if exactly one such file exists it is the answer; if several do the
    directory holds several samples and the call raises, listing them, instead of guessing.
    """
    for n in names:
        p = os.path.join(d, (prefix or "") + n)
        if os.path.exists(p):
            return p
    if prefix is not None:
        return None
    rx = re.compile(r"^(?:.*[_.\-])(?:" + "|".join(re.escape(n) for n in names) + r")$", re.I)
    found = sorted(f for f in os.listdir(d) if rx.match(f))
    if len(found) > 1:
        raise ValueError(f"{d}: several {names[0].split('.')[0]} files {found}; pass the path of "
                         f"one sample's matrix file (e.g. MtxTriple({os.path.join(d, found[0])!r}))")
    return os.path.join(d, found[0]) if found else None


def _locate_triple(path: str) -> tuple:
    """``(dir, matrix, barcodes, features)`` for a directory, or for the path of one matrix file.

    The plain Cell Ranger names win. Failing that, GEO-style prefixed names are accepted
    (``GSM123_matrix.mtx.gz`` with ``GSM123_barcodes.tsv.gz`` and ``GSM123_features.tsv.gz`` or
    ``GSM123_genes.tsv.gz``) when exactly one candidate of each kind exists. A directory that
    holds several samples is ambiguous and raises, listing them; naming one sample's matrix
    file picks that sample (and its companions, by prefix).
    """
    if os.path.isdir(path):
        d, pick = path, None
    else:
        d, pick = os.path.dirname(path) or ".", os.path.basename(path)
    prefix = ""
    if pick and _MTX_RE.match(pick):
        mtx, prefix = os.path.join(d, pick), _MTX_RE.match(pick).group("prefix")
    else:
        mtx = _find(d, _PLAIN["mtx"])
        if mtx:
            prefix = _MTX_RE.match(os.path.basename(mtx)).group("prefix")
    bc = _find(d, _PLAIN["bc"], prefix) or _find(d, _PLAIN["bc"])
    ft = _find(d, _PLAIN["ft"], prefix) or _find(d, _PLAIN["ft"])
    if not (mtx and bc and ft):
        missing = [n for n, v in (("matrix.mtx[.gz]", mtx), ("barcodes.tsv[.gz]", bc),
                                  ("features.tsv[.gz] (or genes.tsv)", ft)) if not v]
        raise ValueError(f"{d}: need matrix.mtx[.gz], barcodes.tsv[.gz], features.tsv[.gz] "
                         f"(or genes.tsv), plain or with a GEO-style prefix; missing {missing}")
    return d, mtx, bc, ft


class MtxTriple(_Governed):
    """`matrix.mtx.gz` + `barcodes.tsv.gz` + `features.tsv.gz`, streamed as COO batches.

    Text, gzipped, unindexed: it can be streamed under a budget, but not navigated. Use
    `to_h5()` to convert once into a real 10x HDF5 and get everything TenXMatrix offers.
    Plain Cell Ranger names and GEO-style prefixed ones are both found (`_locate_triple`), and
    the dimension line must agree with the barcodes and features files — a matrix written
    cells × features is reported as transposed rather than read as nonsense.
    """

    def __init__(self, path: str, chunk_bytes: str | int = "64MB",
                 budget: Optional[Budget] = None):
        super().__init__(budget)
        self.dir, self.mtx, self.bc_path, self.ft_path = _locate_triple(path)
        self.barcodes = np.array(_read_lines(self.bc_path))
        ft = [line.split("\t") for line in _read_lines(self.ft_path)]
        self.feature_ids = np.array([r[0] for r in ft])
        self.feature_names = np.array([r[1] if len(r) > 1 else r[0] for r in ft])
        self.feature_types = np.array([r[2] if len(r) > 2 else "Gene Expression" for r in ft])
        self.n_features, self.n_cells, self.nnz, self._header_len = self._header()
        self._check_dimensions()
        self.modalities = _split_modalities(self.feature_types)
        self.chunk_bytes = resolve_chunk_bytes(chunk_bytes, "64MB", budget)
        self.last_conversion: dict = {}
        self._bulk = None                                # the lease on a parsed matrix held in memory

    def _open(self):
        return gzip.open(self.mtx, "rt") if self.mtx.endswith(".gz") else open(self.mtx, "rt")

    def _header(self) -> tuple:
        with self._open() as fh:
            first = fh.readline()
            if not first.startswith("%%MatrixMarket"):
                raise ValueError(f"{self.mtx}: not MatrixMarket")
            n_hdr = 1
            for line in fh:
                n_hdr += 1
                if line.startswith("%"):
                    continue
                r, c, nnz = (int(x) for x in line.split()[:3])
                return r, c, nnz, n_hdr
        raise ValueError(f"{self.mtx}: no dimension line")

    def _check_dimensions(self):
        rows, cols = self.n_features, self.n_cells
        nf, nb = len(self.feature_ids), len(self.barcodes)
        if (rows, cols) == (nf, nb):
            return
        mtx, bc, ft = (os.path.basename(p) for p in (self.mtx, self.bc_path, self.ft_path))
        if (rows, cols) == (nb, nf) and nf != nb:
            raise ValueError(
                f"{self.mtx}: the dimension line says {rows} x {cols}, which is cells x features — "
                f"this matrix is transposed ({nb} barcodes, {nf} features); Cell Ranger writes "
                f"features x cells. Transpose it (scipy.io.mmread(...).T, mmwrite) before reading.")
        why = []
        if rows != nf:
            why.append(f"{ft} has {nf} entries but {mtx} has {rows} rows (features)")
        if cols != nb:
            why.append(f"{bc} has {nb} entries but {mtx} has {cols} columns (barcodes)")
        raise ValueError(f"{self.mtx}: dimension line {rows} x {cols} disagrees with its sidecar "
                         f"files: " + "; ".join(why))

    # ---- sizing ----
    _ENTRY = 12                                                    # (row, col, val) as int32/int32/f32

    def _entries_per_chunk(self) -> int:
        return max(1, self.chunk_bytes // self._ENTRY)

    def _fits_in_ram(self) -> bool:
        """The parsed matrix can be held (SciPy's C reader: ~24 bytes an entry with its working
        space) in half of what is free, with room left for a chunk on top."""
        if self._budget is None:
            return True
        avail = int(self._budget.available(self._tier))
        whole = self.nnz * 24
        return whole <= avail // 2 and whole + self._entries_per_chunk() * self._ENTRY <= avail

    def plan(self) -> Plan:
        per = self._entries_per_chunk()
        n = math.ceil(self.nnz / per)
        big = min(self.nnz, per) * self._ENTRY
        fits = big <= self.chunk_bytes and (self._budget is None
                                            or big <= int(self._budget.available(self._tier)))
        return Plan(kind="mtx-triple", source=self.mtx, n_chunks=n, bytes_per_chunk=big,
                    total_bytes=self.nnz * self._ENTRY, unit="entries",
                    detail={"n_cells": self.n_cells, "n_features": self.n_features,
                            "nnz": self.nnz, "navigable": False, "entries_per_chunk": per,
                            "chunk_bytes": self.chunk_bytes, "largest_chunk_bytes": big,
                            "fits": bool(fits), "parse": "scipy" if self._fits_in_ram() else "stream",
                            "modalities": {k: v.n_features for k, v in self.modalities.items()},
                            "advice": "convert once with to_h5() to navigate by cell block"})

    # ---- reading ----
    def _mmread(self):
        """The whole matrix as a SciPy COO via its C reader, or None for a dense 'array' file."""
        import scipy.io
        m = scipy.io.mmread(self.mtx)
        return m.tocoo() if hasattr(m, "tocoo") else None

    def __iter__(self) -> Iterator[dict]:
        """COO batches: {'row', 'col', 'val'} as 0-based int32/float32 arrays.

        Two paths, chosen by cost. When the whole matrix fits comfortably in the budget (or
        there is no budget), SciPy's C reader parses it in one go — about ten times faster
        than any text loop — and the result is handed out in budget-sized slices; the parsed
        matrix itself is leased for as long as it is held. Otherwise the text is read in blocks
        and parsed with `numpy.fromstring`, which is four times faster than splitting lines in
        Python; a batch is exactly `entries_per_chunk` entries (the last one the remainder), and
        never more than one batch plus a parse sub-block is resident.
        """
        per = self._entries_per_chunk()
        if self._fits_in_ram():
            self._bulk = (self._budget.acquire(self._tier, str(self.nnz * 24))
                          if self._budget is not None else None)     # lease before it is parsed
            try:
                m = self._mmread()
                if m is not None:
                    for i in range(0, m.nnz, per):
                        sl = slice(i, min(i + per, m.nnz))
                        yield self._emit(m.row[sl], m.col[sl], m.data[sl])
                    return
            finally:
                self._release_bulk()
        yield from self._stream(per)

    def _release_bulk(self):
        bulk, self._bulk = self._bulk, None
        if bulk is not None:
            bulk.release()

    def close(self):
        self._release_bulk()
        super().close()

    def _stream(self, per: int) -> Iterator[dict]:
        """Block-wise text parse: `per` entries a batch, a quarter of that parsed at a time."""
        with self._open() as fh:
            kind = fh.readline().lower()
            pattern = "pattern" in kind
            if "complex" in kind:
                raise NotImplementedError(f"{self.mtx}: complex MatrixMarket matrices are not supported")
            width = 2 if pattern else 3
            for _ in range(self._header_len - 1):
                fh.readline()
            sub = max(1000, per // 4)                    # entries per parse step: bounds the temporaries
            chars = 12.0                                 # running estimate of characters per entry
            rest, eof, carry, emitted = "", False, None, 0
            while emitted < self.nnz:
                want = min(per, self.nnz - emitted)
                self._swap_lease(want * self._ENTRY)     # lease first, then allocate
                r = np.empty(want, np.int32)
                c = np.empty(want, np.int32)
                v = np.empty(want, np.float32)
                n = 0
                while n < want:
                    if carry is not None:
                        rr, cc, vv = carry
                        carry = None
                    else:
                        if eof:
                            break
                        text, rest, eof = self._text_block(fh, rest, max(4096, int(sub * chars)))
                        if not text.strip():
                            continue
                        rr, cc, vv = self._parse_block(text, width, pattern)
                        chars = max(2.0 * width, len(text) / max(len(rr), 1))
                    take = min(len(rr), want - n)
                    r[n:n + take], c[n:n + take], v[n:n + take] = rr[:take], cc[:take], vv[:take]
                    n += take
                    if take < len(rr):
                        carry = (rr[take:], cc[take:], vv[take:])
                    rr = cc = vv = None
                if n < want:                             # the text ran out before the header's count
                    raise ValueError(f"{self.mtx}: the dimension line declares {self.nnz} entries "
                                     f"but the file holds {emitted + n}")
                emitted += n
                yield {"row": r, "col": c, "val": v}
                r = c = v = None
            tail = rest + ("" if eof else fh.read(1 << 16))
            if carry is not None or tail.strip():
                raise ValueError(f"{self.mtx}: more than the {self.nnz} entries the dimension "
                                 "line declares")

    @staticmethod
    def _text_block(fh, rest: str, size: int) -> tuple:
        """Read about `size` characters and cut at a newline: ``(whole lines, remainder, eof)``."""
        buf = rest
        while True:
            data = fh.read(size)
            if not data:
                return buf, "", True
            buf += data
            cut = buf.rfind("\n")
            if cut >= 0:
                return buf[:cut], buf[cut + 1:], False

    def _parse_block(self, text: str, width: int, pattern: bool) -> tuple:
        a = np.fromstring(text, dtype=np.float64, sep=" ")
        if a.size % width:
            raise ValueError(f"{self.mtx}: a line does not have {width} fields")
        a = a.reshape(-1, width)
        r = a[:, 0].astype(np.int32) - 1
        c = a[:, 1].astype(np.int32) - 1
        if len(r) and (r.min() < 0 or r.max() >= self.n_features or c.min() < 0
                       or c.max() >= self.n_cells):
            raise ValueError(f"{self.mtx}: an entry lies outside the {self.n_features} x "
                             f"{self.n_cells} matrix its dimension line declares")
        v = np.ones(len(a), np.float32) if pattern else a[:, 2].astype(np.float32)
        return r, c, v

    def _emit(self, r, c, v) -> dict:
        self._swap_lease(len(r) * self._ENTRY)                       # lease first, then convert
        return {"row": np.asarray(r, np.int32), "col": np.asarray(c, np.int32),
                "val": np.asarray(v, np.float32)}

    # ---- conversion ----
    def to_h5(self, dest: str, compression: str = "gzip") -> str:
        """COO text -> CSC HDF5 in Cell Ranger layout. Then use TenXMatrix.

        Two paths, chosen by cost. **In memory** when ``nnz * 24`` bytes (the parsed COO and its
        CSC) fit in half of what the budget has free, or there is no budget: SciPy's reader, one
        conversion, leased for its duration. **Spilled** otherwise: pass 1 streams the COO once
        to count entries per cell (this is the ``indptr``) and to learn whether the values are
        integers; pass 2 streams it again and scatters each entry to its cell's write cursor in
        two ``numpy.memmap`` files in a temporary directory beside ``dest``; the memmaps are
        then copied into the HDF5 datasets a block of cells at a time, indices sorted and
        duplicate coordinates summed. Nothing larger than a streaming chunk (plus its sort
        working space and one block of cells during the copy) is ever resident — the lease covers
        the streaming chunk and the copy block.

        **Spill size**: ``nnz * (4 + itemsize of data)`` bytes of scratch disk — 8 bytes an
        entry for the usual integer counts — next to ``dest`` for the duration of the call (the
        directory is removed on success and on failure), on top of the output file. If the budget
        declares a ``spill`` tier the bytes are taken from it first; if the disk has less free
        space the call fails before writing. ``last_conversion`` records which path ran.

        Values pass through float32 on the way in (as in iteration), so integers above 2**24 are
        not exact; integer-valued matrices are written as int32 (int64 beyond 2**31), others as
        float32.
        """
        dest = os.fspath(dest)
        if self._budget is None or self.nnz * 24 <= int(self._budget.available(self._tier)) // 2:
            self._to_h5_ram(dest, compression)
        else:
            self._to_h5_spill(dest, compression)
        return dest

    def _write_meta(self, g, X_shape, indptr, compression):
        g.create_dataset("indptr", data=np.asarray(indptr, np.int64))
        g.create_dataset("shape", data=np.array(X_shape, np.int32))
        g.create_dataset("barcodes", data=np.char.encode(self.barcodes.astype(str)))
        ft = g.create_group("features")
        ft.create_dataset("id", data=np.char.encode(self.feature_ids.astype(str)))
        ft.create_dataset("name", data=np.char.encode(self.feature_names.astype(str)))
        ft.create_dataset("feature_type", data=np.char.encode(self.feature_types.astype(str)))
        ft.create_dataset("_all_tag_keys", data=np.array([b"genome"]))
        ft.create_dataset("genome", data=np.array([b"unknown"] * self.n_features))

    @staticmethod
    def _value_dtype(all_int: bool, vmax: float):
        if all_int:
            return np.int32 if vmax < 2 ** 31 else np.int64
        return np.float32

    def _to_h5_ram(self, dest: str, compression: str):
        import h5py
        bulk = (self._budget.acquire(self._tier, str(self.nnz * 24))
                if self._budget is not None else None)               # lease before it is parsed
        try:
            m = self._mmread()
            if m is None:
                raise ValueError(f"{self.mtx}: dense ('array') MatrixMarket is not supported")
            X = m.tocsc()
            del m
            X.sum_duplicates()
            X.sort_indices()
            vals = X.data
            all_int = vals.dtype.kind in "iub" or bool(np.array_equal(vals, np.rint(vals)))
            vmax = float(np.abs(vals).max()) if vals.size else 0.0
            dt = self._value_dtype(all_int, vmax)
            comp = compression if X.nnz else None
            with h5py.File(dest, "w") as f:
                g = f.create_group("matrix")
                g.create_dataset("data", data=vals.astype(dt, copy=False), compression=comp)
                g.create_dataset("indices", data=X.indices.astype(np.int32, copy=False), compression=comp)
                self._write_meta(g, X.shape, X.indptr, compression)
            self.last_conversion = {"path": "ram", "spill_bytes": 0, "nnz": int(X.nnz)}
        finally:
            if bulk is not None:
                bulk.release()

    def _to_h5_spill(self, dest: str, compression: str):
        import h5py
        import scipy.sparse as sp
        nnz, nc = self.nnz, self.n_cells
        per = self._entries_per_chunk()
        parent = os.path.dirname(os.path.abspath(dest))

        # pass 1: entries per cell, and what the values are
        counts = np.zeros(nc, np.int64)
        all_int, vmax = True, 0.0
        for blk in self._stream(per):
            c, v = blk["col"], blk["val"]
            if len(c) * 8 >= nc:
                counts += np.bincount(c, minlength=nc)
            else:
                u, k = np.unique(c, return_counts=True)
                counts[u] += k
            if all_int and len(v):
                if np.array_equal(v, np.rint(v)):
                    vmax = max(vmax, float(np.abs(v).max()))
                else:
                    all_int = False
            blk = c = v = None
        self._swap_lease(0)
        indptr = np.zeros(nc + 1, np.int64)
        np.cumsum(counts, out=indptr[1:])
        del counts
        dt = np.dtype(self._value_dtype(all_int, vmax))
        spill_bytes = nnz * (dt.itemsize + 4)

        # scratch space: the budget's spill tier if it declares one, and the disk either way
        spill_lease = None
        if self._budget is not None and self._budget.stats()["spill"]["capacity"] > 0:
            spill_lease = self._budget.acquire("spill", str(spill_bytes))
        try:
            free = shutil.disk_usage(parent).free
            if free < spill_bytes + (4 << 20):
                raise OSError(28, f"no space for the {human(spill_bytes)} scratch files beside "
                                  f"{dest} ({human(free)} free)")
            with tempfile.TemporaryDirectory(prefix=".geminga-spill-", dir=parent) as tmp:
                dm = im = None
                try:
                    dm = np.memmap(os.path.join(tmp, "data.bin"), dtype=dt, mode="w+", shape=(nnz,))
                    im = np.memmap(os.path.join(tmp, "indices.bin"), dtype=np.int32, mode="w+",
                                   shape=(nnz,))
                    self._scatter(per, indptr, dm, im, dt)
                    written = self._copy_to_h5(dest, compression, indptr, dm, im, dt, sp, h5py)
                finally:
                    dm = im = None                           # unmap before the directory goes
            self.last_conversion = {"path": "spill", "spill_bytes": int(spill_bytes), "nnz": written}
        finally:
            if spill_lease is not None:
                spill_lease.release()

    def _scatter(self, per: int, indptr: np.ndarray, dm, im, dt):
        """Pass 2: stream the COO again, put every entry at its cell's write cursor."""
        cursor = indptr[:-1].copy()
        for blk in self._stream(per):
            r, c, v = blk["row"], blk["col"], blk["val"]
            step = max(1024, len(r) // 4)                    # keeps the sort's working space ~ one chunk
            for a in range(0, len(r), step):
                rr, cc, vv = r[a:a + step], c[a:a + step], v[a:a + step]
                order = np.argsort(cc, kind="stable")
                cs = cc[order]
                first = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
                size = np.diff(np.r_[first, len(cs)])
                pos = cursor[cs] + (np.arange(len(cs)) - np.repeat(first, size))
                dm[pos] = vv[order].astype(dt)
                im[pos] = rr[order]
                cursor[cs[first]] += size
            blk = r = c = v = None
        self._swap_lease(0)
        dm.flush()
        im.flush()

    def _copy_to_h5(self, dest, compression, indptr, dm, im, dt, sp, h5py) -> int:
        """Copy the scratch files into HDF5 a block of cells at a time, sorted and de-duplicated;
        returns the entries written (fewer than declared if coordinates repeat)."""
        nf, nc = self.n_features, self.n_cells
        work = 3 * (dt.itemsize + 4)                         # bytes a block holds while it is sorted
        bounds = csr_bounds(indptr, work, self.chunk_bytes)
        out_ptr = np.zeros(nc + 1, np.int64)
        written = 0
        with h5py.File(dest, "w") as f:
            g = f.create_group("matrix")
            chunk = (max(1, min(self.nnz, 1 << 18)),)
            d_data = g.create_dataset("data", shape=(0,), maxshape=(None,), chunks=chunk,
                                      dtype=dt, compression=compression)
            d_idx = g.create_dataset("indices", shape=(0,), maxshape=(None,), chunks=chunk,
                                     dtype=np.int32, compression=compression)
            for i in range(len(bounds) - 1):
                a, b = int(bounds[i]), int(bounds[i + 1])
                lo, hi = int(indptr[a]), int(indptr[b])
                with self._transient((hi - lo) * work):
                    blk = sp.csc_matrix((np.array(dm[lo:hi]), np.array(im[lo:hi]),
                                         indptr[a:b + 1] - lo), shape=(nf, b - a))
                    blk.sum_duplicates()                     # sorts each column, merges repeats
                    m = int(blk.nnz)
                    out_ptr[a + 1:b + 1] = written + blk.indptr[1:]
                    d_data.resize(written + m, axis=0)
                    d_idx.resize(written + m, axis=0)
                    d_data[written:written + m] = blk.data.astype(dt, copy=False)
                    d_idx[written:written + m] = blk.indices.astype(np.int32, copy=False)
                    written += m
                    blk = None
            self._write_meta(g, (nf, nc), out_ptr, compression)
        return written

    def __repr__(self) -> str:
        return (f"MtxTriple({os.path.basename(self.dir)}, {self.n_cells:,} cells x "
                f"{self.n_features:,} features, {self.nnz:,} nnz, not navigable)")


class TenXView(View):
    """Zoom/pan/cost over cells of a 10x matrix, one modality at a time.

    The window is a cell range; zoom decimates cells. `overview()` is per-cell totals for
    the modality — UMIs, ADT counts, fragments, guide UMIs — bucketed, which is the first QC
    plot anyone makes and costs one pass over `data` for the cells in the window.

    Two sizes matter and `cost()` reports both. CSC keeps all of a cell's features together, so
    a window of cells is *read* as whole column blocks whichever modality is viewed:
    `read_bytes` (and the per-chunk `resident_bytes`) are those blocks, cut from `indptr` to the
    view's chunk cap and leased while they are read. What the consumer is *handed* is the
    modality's rows only, decimated by the zoom: `total_bytes`, from the modality's exact entry
    count per cell (`TenXMatrix.modality_nnz`: one cached pass over `indices` the first time a
    cost is asked for, none when the modality is the whole matrix). The base class accounts
    what is handed out.
    """

    def __init__(self, m: TenXMatrix, modality: Optional[str] = None,
                 budget: Optional[Budget] = None):
        self.m = m
        self.modality = modality or m.modality or next(iter(m.modalities))
        if self.modality not in m.modalities:
            raise KeyError(f"no modality {self.modality!r}; have {list(m.modalities)}")
        super().__init__(budget)

    def _extent(self) -> Extent:
        # the cheap default (mean column block per cell, no I/O); cost() refines it per window
        per_cell = self.m.nnz / max(self.m.n_cells, 1) * self.m._itemsize
        return Extent("cells", (self.m.n_cells,), float(per_cell))

    def _full_window(self):
        return (0, self.m.n_cells)

    def _blocks(self, w):
        return self.m._cell_bounds(int(w[0]), int(w[1]), self._chunk_cap())

    def _chunk_plan(self, w, stride):
        bounds = self._blocks(w)
        sizes = block_bytes(self.m.indptr, bounds, self.m._itemsize)
        if stride > 1 and len(bounds) > 1:
            # `_decimated` skips a block that holds no cell of the decimation grid, so count only
            # the blocks that will actually be yielded.
            c0 = int(w[0])
            kept = [i for i in range(len(bounds) - 1)
                    if int(bounds[i]) + (-(int(bounds[i]) - c0)) % stride < int(bounds[i + 1])]
            sizes = sizes[kept] if kept else sizes[:0]
        return len(sizes), (int(sizes.max()) if sizes.size else 0)

    def cost(self) -> dict:
        """As the base class, with the window costed exactly: `total_bytes` is what the modality
        hands over (from its entry count per cell), `read_bytes` the column blocks read to get
        it, and `fits_whole` asks whether `load()` can read them — an undecimated window is
        leased whole while it is read, a decimated one a chunk at a time."""
        c0, c1 = int(self._win[0]), int(self._win[1])
        nnz = int(self.m.modality_nnz(self.modality)[c0:c1].sum())
        out = nnz * self.m._itemsize + (c1 - c0 + 1) * 4              # data + indices + a 32-bit indptr
        saved = self.extent.unit_bytes
        self.extent.unit_bytes = (out + 0.5) / max(c1 - c0, 1)         # +0.5: int() must not round down
        try:
            d = super().cost()
        finally:
            self.extent.unit_bytes = saved
        read = self._block_bytes(c0, c1)
        d["read_bytes"], d["read_human"] = read, human(read)
        d["peak_read_bytes"] = self._read_bytes(c0, c1)
        cap = d.get("budget_available")
        if cap is not None:
            if self.stride > 1:
                need = d["total_bytes"] + d["resident_bytes"]            # every kept part, plus one block
            elif self._whole_matrix():
                need = read                                              # what is read is what is held
            else:
                need = d["total_bytes"] + self._widest_cell(c0, c1)      # the output, plus one block
            d["fits_whole"] = need <= cap
            if d["fits_whole"]:
                d.pop("advice", None)
            else:
                d["advice"] = (f"loading this window needs {human(need)} (it holds {human(d['total_bytes'])} "
                               f"and reads whole column blocks, since CSC keeps every feature of a cell "
                               f"together) but {human(cap)} is free; narrow the window, zoom out, stream "
                               f"it with .chunks(), or raise the '{self.tier}' budget")
        return d

    def _whole_matrix(self) -> bool:
        return self.m.modalities[self.modality].n_features == self.m.n_features

    def _widest_cell(self, c0: int, c1: int) -> int:
        """Bytes of the largest single column block in the window: the least a read can lease."""
        d = np.diff(self.m.indptr[c0:c1 + 1].astype(np.int64))
        return int(d.max()) * self.m._itemsize if d.size else 0

    def _overview(self, w, n) -> Overview:
        c0, c1 = int(w[0]), int(w[1])
        tot = self.m.per_cell_totals(self.modality, c0, c1)
        if tot.size == 0:
            return Overview("cells", w, np.array([]), {}, cheap=False)
        k = max(1, math.ceil(tot.size / n))
        pad = (-tot.size) % k
        arr = np.concatenate([tot, np.full(pad, np.nan)]).reshape(-1, k)
        return Overview("cells", w, w[0] + (np.arange(arr.shape[0]) + 0.5) * k,
                        {f"{self.modality}_total_per_cell": np.nanmean(arr, 1),
                         f"{self.modality}_max_in_bucket": np.nanmax(arr, 1)},
                        counts=np.full(arr.shape[0], k), cheap=False,
                        source_bytes=int((self.m.indptr[c1] - self.m.indptr[c0]) * self.m._itemsize))

    def _block_bytes(self, a: int, b: int) -> int:
        return int((self.m.indptr[b] - self.m.indptr[a]) * self.m._itemsize)

    def _read_bytes(self, a: int, b: int) -> int:
        """What reading cells a:b for this modality materializes at its peak: the column block,
        plus the modality's rows copied out of it while the block is still alive. The lease is
        the block (the planned quantity); this is the transient `cost()` reports as
        `peak_read_bytes`, so a budget set to the byte knows what scipy's slice adds."""
        read = self._block_bytes(a, b)
        if self._whole_matrix():
            return read
        return read + int(self.m.modality_nnz(self.modality)[a:b].sum()) * self.m._itemsize

    def _load(self, w, stride):
        c0, c1 = int(w[0]), int(w[1])
        self.release()                      # the consumer has asked for the next result
        if stride <= 1:
            if self._whole_matrix():
                with _lease(self.budget, self.tier, self._block_bytes(c0, c1)):
                    return self.m.block(c0, c1, self.modality)
            return self._load_modality(c0, c1)
        import scipy.sparse as sp
        # Decimated: the kept parts accumulate, so each is leased as it arrives and the sum is
        # given back only once the caller's hold on the assembled result takes over.
        parts, held = [], []
        try:
            for part in self._decimated(w, stride, release=False):
                n = int(part.data.nbytes + part.indices.nbytes + part.indptr.nbytes)
                if self.budget is not None and n > 0:
                    held.append(self.budget.acquire(self.tier, str(n)))
                parts.append(part)
            return sp.hstack(parts, format="csc") if parts else self.m.block(c0, c0, self.modality)
        finally:
            for h in held:
                h.release()

    def _load_modality(self, c0: int, c1: int):
        """A modality of a window, assembled column block by column block into arrays sized from
        its exact entry count: the output is allocated once (and leased up front), each block is
        leased only while it is read, so a modality whose rows fit the budget loads even when the
        full column blocks do not."""
        import scipy.sparse as sp
        m = self.m
        per = m.modality_nnz(self.modality)[c0:c1]
        cum = np.zeros(len(per) + 1, np.int64)
        np.cumsum(per, out=cum[1:])
        nnz = int(cum[-1])
        out_bytes = nnz * m._itemsize + (c1 - c0 + 1) * 4
        cap = self._chunk_cap()
        if self.budget is not None:
            cap = min(cap, int(self.budget.available(self.tier)) - out_bytes)
        bounds = m._cell_bounds(c0, c1, max(cap, self._widest_cell(c0, c1), 1))
        indptr = np.zeros(c1 - c0 + 1, np.int64)
        with _lease(self.budget, self.tier, out_bytes):
            data = np.empty(nnz, m.dtype)
            indices = np.empty(nnz, np.int32)
            for i in range(len(bounds) - 1):
                a, b = int(bounds[i]), int(bounds[i + 1])
                with _lease(self.budget, self.tier, self._block_bytes(a, b)):
                    X = m.block(a, b, self.modality)
                lo, hi = int(cum[a - c0]), int(cum[b - c0])
                if X.nnz != hi - lo:                      # the counts and the slice disagree: do it plainly
                    return sp.hstack([m.block(int(bounds[j]), int(bounds[j + 1]), self.modality)
                                      for j in range(len(bounds) - 1)], format="csc")
                data[lo:hi] = X.data
                indices[lo:hi] = X.indices
                indptr[a - c0:b - c0 + 1] = X.indptr.astype(np.int64) + lo
                X = None
            return sp.csc_matrix((data, indices, indptr),
                                 shape=(m.modalities[self.modality].n_features, c1 - c0))

    def _decimated(self, w, stride, release=True):
        """Blocks of the window with every ``stride``-th cell kept (counted from the window's
        start). Each block is leased while it is read, and — when streaming — the hold from the
        previous chunk is given back first, since the consumer has asked for the next one."""
        c0 = int(w[0])
        bounds = self._blocks(w)
        for i in range(len(bounds) - 1):
            a, b = int(bounds[i]), int(bounds[i + 1])
            first = a + (-(a - c0)) % stride
            if first >= b:
                continue
            if release:
                self.release()
            with _lease(self.budget, self.tier, self._block_bytes(a, b)):
                X = self.m.block(a, b, self.modality)
                part = X[:, first - a::stride] if stride > 1 else X
                del X                           # the full block must not outlive its lease
            yield part
            del part

    def _chunks(self, w, stride):
        yield from self._decimated(w, stride)

    def features(self) -> np.ndarray:
        return self.m.features(self.modality)

    def __repr__(self) -> str:
        c = self.cost()
        return (f"TenXView({self.modality}, cells {c['window'][0]:,}-{c['window'][1]:,}, "
                f"stride {c['stride']}, {c['total_human']})")


class _lease:
    """``with _lease(budget, tier, n):`` — hold ``n`` bytes while a block is read."""

    def __init__(self, budget, tier, nbytes):
        self.budget, self.tier, self.nbytes, self.held = budget, tier, int(nbytes), None

    def __enter__(self):
        if self.budget is not None and self.nbytes > 0:
            self.held = self.budget.acquire(self.tier, str(self.nbytes))
        return self

    def __exit__(self, *exc):
        if self.held is not None:
            self.held.release()
        return False


# ---------------------------------------------------------------------------
# The overview: what modalities does this thing carry, and what can be done with each
# ---------------------------------------------------------------------------

@dataclass
class ModalityReport:
    """What a dataset directory or file carries, before any of it is loaded."""
    target: str
    kind: str
    n_cells: int = 0
    modalities: dict = field(default_factory=dict)      # label -> dict
    spatial: Optional[dict] = None
    images: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    advice: list = field(default_factory=list)
    file_bytes: int = 0

    def print(self) -> None:
        print(f"{self.target}  ({self.kind}, {human(self.file_bytes)})")
        if self.n_cells:
            print(f"  cells       : {self.n_cells:,}")
        for label, d in self.modalities.items():
            extra = f"  — {d['note']}" if d.get("note") else ""
            print(f"  {label:<12}: {d['n_features']:>8,} features{extra}")
        if self.spatial:
            s = self.spatial
            n = s.get("n_positions")
            print(f"  spatial     : {s.get('platform', '?')}, "
                  f"{f'{n:,} positions, ' if n else ''}{s.get('extent', '?')}")
        for im in self.images:
            print(f"  image       : {im}")
        for k, v in self.extras.items():
            print(f"  {k:<12}: {v}")
        for a in self.advice:
            print(f"  -> {a}")


def modalities(target: str) -> ModalityReport:
    """Look at a file or a Cell Ranger / Space Ranger / Xenium directory and say what is in it.

    Reads headers only. The point is to turn "30 rows in an inventory" into "which of these
    can I actually open, and what will I find inside".
    """
    from .spatial import sniff_bundle
    t = os.path.abspath(target)
    rep = ModalityReport(target=t, kind="unknown")
    if os.path.isdir(t):
        rep.file_bytes = sum(os.path.getsize(os.path.join(dp, f))
                             for dp, _, fs in os.walk(t) for f in fs)
        b = sniff_bundle(t)
        rep.kind = b["kind"]
        if rep.kind in ("directory", "multiplexed-imaging"):
            # not itself a bundle: report what its children are
            kids = []
            for name in sorted(os.listdir(t)):
                sub = os.path.join(t, name)
                if os.path.isdir(sub):
                    kind = sniff_bundle(sub)["kind"]
                    if kind != "directory":
                        kids.append(f"{name}/ ({kind})")
                elif name.lower().endswith((".h5", ".h5ad", ".gef", ".ome.tif", ".ome.tiff")):
                    kids.append(name)
            if kids:
                rep.kind = "collection"
                rep.extras["members"] = kids
                rep.advice.append("run g.modalities() on a member for its contents")
                return rep
        rep.spatial, rep.images, rep.extras = b.get("spatial"), b.get("images", []), b.get("extras", {})
        rep.advice.extend(b.get("advice", []))
        mat = b.get("matrix")
        if mat:
            try:
                m = TenXMatrix(mat) if mat.endswith(".h5") else MtxTriple(mat)
                rep.n_cells = m.n_cells
                rep.modalities = {k: {"n_features": v.n_features, "note": v.note}
                                  for k, v in m.modalities.items()}
                if isinstance(m, MtxTriple):
                    rep.advice.append("MatrixMarket is not navigable — MtxTriple.to_h5() once")
                m.close() if hasattr(m, "close") else None
            except Exception as e:
                rep.advice.append(f"matrix not readable: {e}")
        return rep
    rep.file_bytes = os.path.getsize(t)
    low = t.lower()
    if low.endswith(".h5"):
        try:
            m = TenXMatrix(t)
            rep.kind = "10x-h5"
            rep.n_cells = m.n_cells
            rep.modalities = {k: {"n_features": v.n_features, "note": v.note}
                              for k, v in m.modalities.items()}
            m.close()
        except Exception as e:
            rep.kind = "hdf5"
            rep.advice.append(f"not a 10x matrix: {e}")
    elif low.endswith(".h5ad"):
        rep.kind = "h5ad"
        try:
            import h5py
            with h5py.File(t, "r") as f:
                x = f["X"]
                shape = tuple(x.attrs["shape"]) if hasattr(x, "attrs") and "shape" in x.attrs \
                    else getattr(x, "shape", (0, 0))
                rep.n_cells = int(shape[0])
                rep.modalities["RNA"] = {"n_features": int(shape[1]), "note": ""}
                obsm = f.get("obsm")
                if obsm is not None:
                    for k in obsm.keys():
                        kl = k.lower()
                        if "protein" in kl or "adt" in kl:
                            node = obsm[k]
                            nf = int(node.shape[1]) if hasattr(node, "shape") else 0
                            rep.modalities["ADT"] = {"n_features": nf, "note": f"obsm['{k}']"}
                        if "spatial" in kl:
                            rep.spatial = {"platform": "h5ad obsm['spatial']",
                                           "n_positions": rep.n_cells}
                if "uns" in f and "spatial" in f["uns"]:
                    rep.images = [f"uns['spatial'] with {len(f['uns']['spatial'].keys())} library"]
        except Exception as e:
            rep.advice.append(f"h5ad not readable: {e}")
    else:
        rep.advice.append("not a matrix container; use g.profile() for sequences, volumes and images")
    return rep


def open_tenx(path: str, modality: Optional[str] = None, budget: Optional[Budget] = None,
              **kw) -> TenXView:
    """A navigable view over one modality of a 10x matrix (h5, or a triple converted first)."""
    if os.path.isdir(path) or path.endswith((".mtx", ".mtx.gz")):
        raise ValueError("MatrixMarket triples are not navigable; convert with "
                         "MtxTriple(path).to_h5(dest) and open the .h5")
    return TenXView(TenXMatrix(path, budget=budget, **kw), modality, budget)


# ---------------------------------------------------------------------------

def _decode(a) -> np.ndarray:
    arr = np.asarray(a)
    if arr.dtype.kind in ("S", "O"):
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])
    return arr


def _read_lines(path: str) -> list:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]
