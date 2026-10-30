"""MRI volumes: NIfTI and DICOM, read as slabs, navigated in three dimensions.

Lesion-segmentation datasets have a shape the rest of this library had not met: one *volume*
per scan, several sequences per session (T1w, T2w, FLAIR), several sessions per subject, and
an expert mask alongside. A study like MSLesSeg is seventy-five subjects times up to four
timepoints times three sequences — a few hundred volumes of 10–50 MB each. None of that is big
by omics standards. What is big is the pipeline that runs over it: a segmentation model wants
a whole volume, or a patch, or a batch of patches, on a GPU with a fixed amount of memory, and
the difference between "fits" and "does not" is decided per slab, not per file.

So the readers here are lazy in the axis that matters. A NIfTI header is 352 bytes and says
everything about shape, dtype and voxel size; a DICOM series can be catalogued from headers
alone with `stop_before_pixels=True`. Pixels arrive slab by slab, budget-accounted, and the
3-D `VolumeView` gives the same zoom / pan / cost / auto_zoom the 1-D and 2-D views have.

**Conventions, stated once.** Both volume classes present slabs as `(Z, Y, X)` arrays — for a
NIfTI that is the file's axes 2, 1, 0, for a DICOM series (slice, row, column) — and *the
array axes are never reoriented*: whatever order the scanner wrote is the order you get.
Everything that describes the array is in that same array-axis order, so `shape`, `voxel_mm` and
`axcodes` can be zipped: `voxel_mm[i]` is the voxel size along array axis `i` and `axcodes[i]`
names the anatomical direction in which index `i` increases ('S' = toward superior, as in
nibabel's `aff2axcodes`). A NIfTI whose `aff2axcodes` is `('A', 'S', 'L')` — a sagittal
acquisition — has `axcodes == ('L', 'S', 'A')`: the same codes, reversed because the array is the
file's axes 2, 1, 0 (`file_axcodes` keeps nibabel's order), and its array axis 0 runs left to
right, not head to foot. Wherever an axis is asked for, an
integer means the array axis and an anatomical name — `'axial'`, `'S'`, `'SI'`, `'coronal'` —
is resolved through `axcodes`: `volume.axis_index('axial')` is the array axis the axial planes
are stacked along, and `VolumeView.overview()`'s "axial_mip" is the projection along it.

**Values.** A NIfTI scaling (`scl_slope`/`scl_inter`) or a DICOM `RescaleSlope`/`RescaleIntercept`
is applied, and the output dtype is chosen once per volume: the stored dtype when nothing is
rescaled, otherwise the narrowest float that holds every stored value exactly (float32 for
16-bit data). `dtype`, `nbytes`, `cost()` and the slabs themselves all report that dtype, so a
`cost()` is a statement about what `load()` returns.

One honest limitation up front: a gzipped NIfTI cannot be seeked. Reading slab `[100:110]`
means decompressing everything before it. Memory stays bounded — that is the point — but time
is linear in the offset, and `cost()` says so. Uncompressed NIfTI reads by seek.
"""
from __future__ import annotations

import itertools
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

import numpy as np

from ._core import Budget, human
from .ngff import OverwriteRefused, clear_zarr_members, plan_target
from .view import Extent, Overview, View

__all__ = ["NiftiVolume", "DicomSeries", "VolumeView", "MaskSink3D",
           "Subject", "Cohort", "open_volume", "SEQUENCE_PATTERNS"]

#: How MRI sequences announce themselves in filenames and DICOM SeriesDescription.
#: `MASK` is the broad detector, applied as it always was when a name carries no modality at all;
#: `guess_sequence` weighs its words against the modality tokens when it does.
SEQUENCE_PATTERNS = {
    "FLAIR":  re.compile(r"flair", re.I),
    "T1w":    re.compile(r"(?<![a-z0-9])t1(w|_?mprage|_?gd|c?e?)?(?![a-z0-9])|mprage", re.I),
    "T2w":    re.compile(r"(?<![a-z0-9])t2(w|_?tse)?(?![a-z0-9])", re.I),
    "PDw":    re.compile(r"(?<![a-z0-9])pd(w)?(?![a-z0-9])", re.I),
    "SWI":    re.compile(r"swi|susceptibility", re.I),
    "DWI":    re.compile(r"dwi|dti|diffusion", re.I),
    "MASK":   re.compile(r"mask|lesion|seg|label|gt|consensus|wmh", re.I),
}

#: Words that make a file a mask whatever else the name says: nothing else is called a "mask".
_MASK_ROLE = re.compile(r"mask|label|consensus|ground[_ -]?truth", re.I)
#: Words that describe a mask but also turn up in dataset names (MSLesSeg, WMH2017, LesionTOADS):
#: they count only as whole words, and only when they come after the modality token.
_MASK_WORDS = frozenset({"seg", "segm", "segmentation", "gt", "lesion", "lesions", "wmh",
                         "manual", "annot", "annotation"})
#: Words in a stem: `MSLesSeg_P1` -> MS, Les, Seg, P, 1.
_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
#: The same descriptive words when they are glued straight onto a modality (`FLAIRlesion`),
#: where the word splitter cannot tell where the acronym ends.
_MASK_GLUED = re.compile(r"(?:seg(?:mentation)?|gt|lesions?|wmh|manual|annot(?:ation)?)(?![a-z])",
                         re.I)


def guess_sequence(name: str) -> str:
    """Best guess at the sequence from a filename or series description.

    Sequence names are conventionally the *last* token — `P01_T1_T2w.nii.gz` is a T2w scan at
    timepoint 1, not a T1w — so among the modality patterns that match, the one ending latest
    wins. A mask role word (`mask`, `label`, `consensus`, `ground truth`) anywhere makes the
    file a mask: `FLAIR_lesion_mask` is a mask, not a FLAIR. Descriptive words (`seg`, `gt`,
    `lesion`, `wmh`) are weaker, because datasets are named after them — `MSLesSeg_P1_T1_FLAIR`
    is a FLAIR — so next to a modality they count only as whole words (or glued straight onto
    it), and only when they follow it: `P1_T1_FLAIR_seg` is a mask, `WMH_challenge_P1_T2w` is a
    T2w. A name with no modality at all is a mask if any mask word appears anywhere in it.
    """
    stem = os.path.basename(name)
    for ext in (".nii.gz", ".nii", ".gz"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
    best, best_end, ends = "unknown", -1, set()
    for seq, rx in SEQUENCE_PATTERNS.items():
        if seq == "MASK":
            continue
        for m in rx.finditer(stem):
            ends.add(m.end())
            if m.end() > best_end:
                best, best_end = seq, m.end()
    if _MASK_ROLE.search(stem):
        return "MASK"
    if best == "unknown":                         # nothing to outrank: any mask word will do
        return "MASK" if SEQUENCE_PATTERNS["MASK"].search(stem) else "unknown"
    mask_end = max([m.end() for m in _WORD.finditer(stem) if m.group().lower() in _MASK_WORDS]
                   + [m.end() for m in _MASK_GLUED.finditer(stem) if m.start() in ends],
                   default=-1)
    return "MASK" if mask_end > best_end else best


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

_PLANES = {"axial": "SI", "coronal": "AP", "sagittal": "LR"}
_OPPOSITE = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}
_UNKNOWN = ("?", "?", "?")


def _axis_index(axcodes: tuple, axis) -> int:
    """Array axis for `axis`: an int is an array axis already; a plane name ('axial'), a
    direction ('S', 'i') or an axis pair ('SI', 'RL') is looked up in `axcodes`."""
    if isinstance(axis, (int, np.integer)) and not isinstance(axis, bool):
        if not -3 <= int(axis) < 3:
            raise ValueError(f"axis {axis} is out of range for a 3-D volume")
        return int(axis) % 3
    if not isinstance(axis, str):
        raise TypeError(f"axis must be an array axis (int) or an anatomical name, not {axis!r}")
    letters = _PLANES.get(axis.strip().lower())
    if letters is None:
        up = axis.strip().upper()
        if not up or len(up) > 2 or any(c not in _OPPOSITE for c in up) or \
                (len(up) == 2 and _OPPOSITE[up[0]] != up[1]):
            raise ValueError(f"unknown axis {axis!r}: use an int, 'axial' / 'coronal' / "
                             f"'sagittal', a direction such as 'S', or a pair such as 'SI'")
        letters = up if len(up) == 2 else up + _OPPOSITE[up]     # 'I' names the S-I axis too
    if tuple(axcodes) == _UNKNOWN:
        raise ValueError(f"the orientation of this volume is unknown (no affine or "
                         f"ImageOrientationPatient), so {axis!r} cannot be placed; pass an int "
                         f"array axis instead")
    for i, code in enumerate(axcodes):
        if code in letters:
            return i
    raise ValueError(f"no axis of this volume runs {'-'.join(letters)} (axcodes {tuple(axcodes)})")


def _plane_axes(axcodes: tuple) -> dict:
    """Array axis each anatomical plane is stacked along (the axis a projection collapses).
    An unknown orientation keeps the historical convention: slices on axis 0, rows on 1."""
    if tuple(axcodes) == _UNKNOWN:
        return {"axial": 0, "coronal": 1, "sagittal": 2}
    return {name: _axis_index(axcodes, name) for name in _PLANES}


def _axcodes_from_directions(dirs_lps) -> tuple:
    """Anatomical codes for three array-axis direction vectors given in DICOM's LPS frame.
    The patient axes are assigned to array axes by the permutation that best aligns them
    (what nibabel's `io_orientation` does, for the 3 x 3 case), so an oblique series still gets
    one code per axis."""
    m = np.asarray(dirs_lps, float).reshape(3, 3)
    best = max(itertools.permutations(range(3)),
               key=lambda p: sum(abs(m[i, p[i]]) for i in range(3)))
    return tuple(("LPS"[p] if m[i, p] >= 0 else "RAI"[p]) for i, p in enumerate(best))


def _ranges(shape: tuple, axis: int, z0, z1, y0, y1, x0, x1) -> list:
    """Per-array-axis `(lo, hi)` for a slab request: the first pair applies to `axis`, the other
    two to the remaining axes in ascending order. Clipped to the volume, never negative."""
    order = [axis] + [i for i in range(3) if i != axis]
    out = [None] * 3
    for ax, (lo, hi) in zip(order, ((z0, z1), (y0, y1), (x0, x1))):
        n = int(shape[ax])
        lo = min(max(int(lo), 0), n)
        hi = n if hi is None else min(max(int(hi), lo), n)
        out[ax] = (lo, hi)
    return out


# ---------------------------------------------------------------------------
# NIfTI
# ---------------------------------------------------------------------------

class NiftiVolume:
    """A NIfTI-1/2 image, header now, voxels on demand.

    `shape` is (Z, Y, X[, T]) after moving the slice axis first, so that "a slab" always means
    a contiguous run of slices regardless of how the file was written. `voxel_mm` and `axcodes`
    follow the same (Z, Y, X) order (see the module docstring); `file_axcodes` is nibabel's own
    answer in the file's (X, Y, Z) order, for checking against other tools.

    `dtype` is what slabs come back as. A file with `scl_slope`/`scl_inter` is scaled by
    nibabel into float64; here that is cast once, explicitly, to float32 (float64 where float32
    cannot hold the stored values) and `dtype` says so. `stored_dtype` is the on-disk type.
    """

    def __init__(self, path: str):
        import nibabel as nib
        self.path = path
        self.img = nib.load(path)
        self.header = self.img.header
        self._raw_shape = tuple(int(s) for s in self.img.shape)
        stored = np.dtype(self.img.get_data_dtype())
        if stored.byteorder not in ("=", "|"):
            stored = stored.newbyteorder("=")
        self.stored_dtype = stored
        proxy = self.img.dataobj
        slope, inter = getattr(proxy, "slope", 1.0), getattr(proxy, "inter", 0.0)
        self.scl_slope = 1.0 if slope is None or not np.isfinite(slope) else float(slope)
        self.scl_inter = 0.0 if inter is None or not np.isfinite(inter) else float(inter)
        self.scaled = self.scl_slope != 1.0 or self.scl_inter != 0.0
        self.dtype = (np.dtype(np.promote_types(stored, np.float32))
                      if self.scaled and stored.kind in "iuf" else stored)
        zooms = tuple(float(z) for z in self.header.get_zooms()[:3])
        self.voxel_mm = zooms[::-1] if len(zooms) == 3 else (1.0, 1.0, 1.0)
        self.gzipped = path.lower().endswith(".gz")
        self.sequence = guess_sequence(os.path.basename(path))
        self.affine = np.asarray(self.img.affine)
        # nibabel stores (X, Y, Z[, T]); we present (Z, Y, X[, T]).
        self.shape = self._to_zyx_shape(self._raw_shape)
        self.ndim = len(self.shape)
        try:
            codes = tuple(str(c) for c in nib.aff2axcodes(self.affine))[:3]
        except Exception:
            codes = _UNKNOWN
        self.file_axcodes = codes if len(codes) == 3 and len(self._raw_shape) >= 3 else _UNKNOWN
        self.axcodes = tuple(reversed(self.file_axcodes))

    @staticmethod
    def _to_zyx_shape(s: tuple) -> tuple:
        if len(s) >= 3:
            return (s[2], s[1], s[0]) + tuple(s[3:])
        return s

    @property
    def nbytes(self) -> int:
        return int(np.prod(self._raw_shape)) * self.dtype.itemsize

    @property
    def n_slices(self) -> int:
        return self.shape[0]

    def axis_index(self, axis) -> int:
        """Array axis for an int or an anatomical name — see the module docstring."""
        return _axis_index(self.axcodes, axis)

    def plane_axes(self) -> dict:
        """`{'axial': i, 'coronal': j, 'sagittal': k}`: the array axis each plane stacks along."""
        return _plane_axes(self.axcodes)

    def slab(self, z0: int, z1: int, y0: int = 0, y1: Optional[int] = None,
             x0: int = 0, x1: Optional[int] = None, t: Optional[int] = None,
             axis=0) -> np.ndarray:
        """Voxels for the run `z0:z1` along `axis`, cropped in the other two axes.

        With the default `axis=0` this is "slices z0:z1, cropped in-plane", as (Z, Y, X). Any
        other `axis` (an array axis, or an anatomical name resolved through `axcodes`) takes the
        run along that axis instead; `y0:y1` and `x0:x1` then crop the remaining two axes in
        ascending array order. The result keeps the array's (Z, Y, X) layout — nothing is
        reoriented — and is returned in `dtype`."""
        (z0, z1), (y0, y1), (x0, x1) = _ranges(self.shape, self.axis_index(axis),
                                               z0, z1, y0, y1, x0, x1)
        proxy = self.img.dataobj                     # ArrayProxy: reads only what is sliced
        key = (slice(x0, x1), slice(y0, y1), slice(z0, z1))
        if self.ndim == 4:
            key += (0 if t is None else int(t),)
        a = np.asarray(proxy[key], dtype=self.dtype)         # an explicit cast: nibabel's own
        return np.ascontiguousarray(np.transpose(a, (2, 1, 0)))     # scaling is float64

    def describe(self) -> dict:
        return {
            "path": self.path, "format": "NIfTI" + (".gz" if self.gzipped else ""),
            "shape_zyx": self.shape, "dtype": str(self.dtype),
            "stored_dtype": str(self.stored_dtype), "scaled": self.scaled,
            "voxel_mm": self.voxel_mm, "axcodes": self.axcodes, "nbytes": self.nbytes,
            "sequence": self.sequence,
            "seek": "linear (gzip)" if self.gzipped else "random",
            "datatype_code": int(self.header["datatype"]),
            "intent": str(self.header.get_intent()[0]) if hasattr(self.header, "get_intent") else "",
        }

    def close(self):
        try:
            self.img.uncache()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (f"NiftiVolume({os.path.basename(self.path)}, {self.sequence}, "
                f"{'x'.join(map(str, self.shape))} {self.dtype}, "
                f"{'x'.join(f'{v:.2g}' for v in self.voxel_mm)} mm, {human(self.nbytes)})")


# ---------------------------------------------------------------------------
# DICOM
# ---------------------------------------------------------------------------

class DicomSeries:
    """A directory of single-slice DICOM files, catalogued from headers only.

    `stop_before_pixels=True` means indexing a 500-file series reads a few kilobytes per
    file and no image data. Slices are ordered by their position along the slice normal, which
    is the physically correct order and survives files being numbered oddly.

    `voxel_mm` is (slice, row, column) — the array-axis order of a slab. The slice entry is the
    *spacing* between consecutive `ImagePositionPatient` projections onto the slice normal, not
    `SliceThickness` (which differs whenever slices overlap or leave gaps); thickness is used
    only when the series carries no positions. `axcodes` come from `ImageOrientationPatient`.
    """

    def __init__(self, path: str, recursive: bool = False):
        import pydicom
        self.path = path
        files = []
        if os.path.isdir(path):
            walker = os.walk(path) if recursive else [(path, [], os.listdir(path))]
            for dp, _, fs in walker:
                for f in fs:
                    if f.lower().endswith((".dcm", ".dicom", ".ima")) or "." not in f:
                        files.append(os.path.join(dp, f))
        else:
            files = [path]
        self.slices: list = []               # (position_along_normal, path, header)
        self.series: dict = {}
        for f in sorted(files):
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            except Exception:
                continue
            if not hasattr(ds, "Rows"):
                continue
            uid = str(getattr(ds, "SeriesInstanceUID", "?"))
            self.series.setdefault(uid, []).append(f)
            self.slices.append((0.0, f, ds))
        if not self.slices:
            raise ValueError(f"{path}: no readable DICOM slices")
        # Keep the largest series if the directory mixes several.
        if len(self.series) > 1:
            best = max(self.series, key=lambda u: len(self.series[u]))
            keep = set(self.series[best])
            self.slices = [s for s in self.slices if s[1] in keep]
        self._order_slices()
        h = self.slices[0][2]
        self.rows, self.cols = int(h.Rows), int(h.Columns)
        self.shape = (len(self.slices), self.rows, self.cols)
        bits = int(getattr(h, "BitsAllocated", 16))
        signed = int(getattr(h, "PixelRepresentation", 0)) == 1
        self.stored_dtype = np.dtype(f"{'i' if signed else 'u'}{bits // 8}")
        ps = getattr(h, "PixelSpacing", None) or [1.0, 1.0]
        self.slice_thickness_mm = float(getattr(h, "SliceThickness", 1.0) or 1.0)
        self.slice_spacing_mm = self._spacing()
        self.voxel_mm = (self.slice_spacing_mm, float(ps[0]), float(ps[1]))
        # One dtype for the series. Rescale tags are per slice (some MR vendors vary the slope
        # slice to slice), so the series is rescaled if any slice is, and every slice is then
        # returned in the same float type.
        self.rescaled = any(self._rescale(s[2]) != (1.0, 0.0) for s in self.slices)
        self.dtype = (np.dtype(np.promote_types(self.stored_dtype, np.float32))
                      if self.rescaled else self.stored_dtype)
        self.modality = str(getattr(h, "Modality", "?"))
        self.description = str(getattr(h, "SeriesDescription", ""))
        self.sequence = guess_sequence(self.description or os.path.basename(path))
        self.patient = str(getattr(h, "PatientID", ""))
        self.study_date = str(getattr(h, "StudyDate", ""))
        self.manufacturer = str(getattr(h, "Manufacturer", ""))
        self.field_strength = getattr(h, "MagneticFieldStrength", None)
        self.gzipped = False
        self.axcodes, self.affine = self._geometry(h)

    # ---- geometry -----------------------------------------------------------
    @staticmethod
    def _iop(ds):
        iop = getattr(ds, "ImageOrientationPatient", None)
        if iop is not None and len(iop) == 6:
            return np.array([float(v) for v in iop])
        return None

    def _order_slices(self):
        """Sort along the slice normal; fall back to InstanceNumber if any slice lacks a
        position (mixing millimetres with instance counts would order nothing)."""
        iop = self._iop(self.slices[0][2])
        normal = None if iop is None else np.cross(iop[:3], iop[3:])
        pos, geometric = [], normal is not None
        for _, f, ds in self.slices:
            ipp = getattr(ds, "ImagePositionPatient", None)
            if geometric and ipp is not None and len(ipp) == 3:
                pos.append(float(np.dot(np.array([float(v) for v in ipp]), normal)))
            else:
                geometric = False
                pos.append(float(getattr(ds, "InstanceNumber", 0) or 0))
        if not geometric:
            pos = [float(getattr(s[2], "InstanceNumber", 0) or 0) for s in self.slices]
        self._geometric = geometric
        self.slices = sorted(((p, s[1], s[2]) for p, s in zip(pos, self.slices)),
                             key=lambda s: (s[0], s[1]))

    def _spacing(self) -> float:
        """Millimetres between consecutive slices: from positions when the series has them."""
        if self._geometric and len(self.slices) > 1:
            steps = np.abs(np.diff([s[0] for s in self.slices]))
            step = float(np.median(steps))
            if step > 0:
                self.uniform_spacing = bool(np.allclose(steps, step, rtol=0.02, atol=1e-3))
                return step
        self.uniform_spacing = True
        return self.slice_thickness_mm

    def _geometry(self, h):
        """(axcodes, affine) from ImageOrientationPatient and the first slice position."""
        iop = self._iop(h)
        if iop is None:
            return _UNKNOWN, None
        r, c = iop[:3], iop[3:]                       # direction of increasing column / row
        n = np.cross(r, c)
        axcodes = _axcodes_from_directions([n, c, r])            # array axes: slice, row, column
        affine = None
        ipp = getattr(self.slices[0][2], "ImagePositionPatient", None)
        if self._geometric and ipp is not None:
            m = np.eye(4)                                        # file-index order: column, row, slice
            m[:3, 0], m[:3, 1], m[:3, 2] = r * self.voxel_mm[2], c * self.voxel_mm[1], n * self.voxel_mm[0]
            m[:3, 3] = [float(v) for v in ipp]
            m[:2] *= -1                                          # LPS (DICOM) -> RAS+ (NIfTI)
            affine = m + 0.0                                     # (and no negative zeros)
        return axcodes, affine

    @staticmethod
    def _rescale(ds) -> tuple:
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        return slope, inter

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * self.dtype.itemsize

    @property
    def n_slices(self) -> int:
        return self.shape[0]

    def axis_index(self, axis) -> int:
        """Array axis for an int or an anatomical name — see the module docstring."""
        return _axis_index(self.axcodes, axis)

    def plane_axes(self) -> dict:
        """`{'axial': i, 'coronal': j, 'sagittal': k}`: the array axis each plane stacks along."""
        return _plane_axes(self.axcodes)

    def slab(self, z0: int, z1: int, y0: int = 0, y1: Optional[int] = None,
             x0: int = 0, x1: Optional[int] = None, t: Optional[int] = None,
             axis=0) -> np.ndarray:
        """Read only the slice files the request touches, cropped in-plane.

        `axis` works as in `NiftiVolume.slab`. Pixels are rescaled with each slice's own slope
        and intercept, in `dtype` — computed after the crop, in the output type, so 150 stored
        with a slope of 0.01 is 1.5 and a CT intercept of -1024 is not wrapped into uint16."""
        import pydicom
        (z0, z1), (y0, y1), (x0, x1) = _ranges(self.shape, self.axis_index(axis),
                                               z0, z1, y0, y1, x0, x1)
        out = np.empty((z1 - z0, y1 - y0, x1 - x0), self.dtype)
        for i, (_, f, hdr) in enumerate(self.slices[z0:z1]):
            arr = pydicom.dcmread(f).pixel_array[y0:y1, x0:x1]
            if self.rescaled:
                slope, inter = self._rescale(hdr)
                arr = arr.astype(self.dtype) * self.dtype.type(slope) + self.dtype.type(inter)
            out[i] = arr
        return out

    def describe(self) -> dict:
        return {
            "path": self.path, "format": "DICOM series", "shape_zyx": self.shape,
            "dtype": str(self.dtype), "stored_dtype": str(self.stored_dtype),
            "rescaled": self.rescaled, "voxel_mm": self.voxel_mm, "axcodes": self.axcodes,
            "slice_spacing_mm": self.slice_spacing_mm,
            "slice_thickness_mm": self.slice_thickness_mm,
            "uniform_spacing": self.uniform_spacing, "nbytes": self.nbytes,
            "sequence": self.sequence, "modality": self.modality,
            "description": self.description, "patient": self.patient,
            "study_date": self.study_date, "manufacturer": self.manufacturer,
            "field_strength_T": self.field_strength, "n_files": len(self.slices),
            "other_series_in_dir": len(self.series) - 1, "seek": "random (per file)",
        }

    def close(self):
        pass

    def __repr__(self) -> str:
        return (f"DicomSeries({os.path.basename(self.path.rstrip('/'))}, {self.sequence}, "
                f"{'x'.join(map(str, self.shape))} {self.dtype}, {len(self.slices)} files, "
                f"{human(self.nbytes)})")


def open_volume(path: str):
    """NiftiVolume for .nii/.nii.gz, DicomSeries for a directory or .dcm file."""
    low = path.lower()
    if low.endswith((".nii", ".nii.gz", ".img", ".hdr")):
        return NiftiVolume(path)
    if os.path.isdir(path) or low.endswith((".dcm", ".dicom", ".ima")):
        return DicomSeries(path)
    raise ValueError(f"not a volume I recognise: {path}")


# ---------------------------------------------------------------------------
# The 3-D view
# ---------------------------------------------------------------------------

class VolumeView(View):
    """Zoom, pan and cost over a volume. Window is (z0, z1, y0, y1, x0, x1).

    Zoom decimates every axis by the stride; `select_frac(a, b)` with one pair applies to all
    three. `cost()` distinguishes what you keep from what a gzipped file forces you to read.
    """

    def __init__(self, vol, budget: Optional[Budget] = None, tier: str = "ram"):
        self.vol = vol
        super().__init__(budget, tier)

    def _extent(self) -> Extent:
        z, y, x = self.vol.shape[:3]
        return Extent("voxels", (z, y, x), float(self.vol.dtype.itemsize))

    def _full_window(self):
        z, y, x = self.vol.shape[:3]
        return (0, z, 0, y, 0, x)

    def _chunk_plan(self, w, stride):
        z0, z1, y0, y1, x0, x1 = w
        keep = (math.ceil((z1 - z0) / stride) * math.ceil((y1 - y0) / stride)
                * math.ceil((x1 - x0) / stride) * self.extent.unit_bytes)
        cap = self._chunk_cap()
        return max(math.ceil(keep / cap), 1), min(keep, cap)

    def cost(self) -> dict:
        d = super().cost()
        z0, z1, y0, y1, x0, x1 = self._win
        # What must be *read* to serve the window, as opposed to what is kept. Reading is in
        # the file's own type (int16), keeping is in the output type (float32 once scaled).
        stored = getattr(self.vol, "stored_dtype", self.vol.dtype)
        plane = self.vol.shape[1] * self.vol.shape[2] * np.dtype(stored).itemsize
        if getattr(self.vol, "gzipped", False):
            d["read_bytes"] = int(z1 * plane)              # everything up to the last slice
            d["seek"] = "linear (gzip): reads from the start to slice %d" % z1
        else:
            d["read_bytes"] = int((z1 - z0) * plane)       # whole slices, cropped after
            d["seek"] = "random"
        d["read_human"] = human(d["read_bytes"])
        d["slices"] = z1 - z0
        d["voxel_mm"] = self.vol.voxel_mm
        d["axcodes"] = getattr(self.vol, "axcodes", _UNKNOWN)
        d["dtype"] = str(self.vol.dtype)
        d["sequence"] = self.vol.sequence
        return d

    def _axis(self, axis) -> int:
        """Array axis for an int or an anatomical name (the volume resolves names)."""
        if isinstance(axis, (int, np.integer)) and not isinstance(axis, bool):
            return _axis_index(_UNKNOWN, axis)
        if hasattr(self.vol, "axis_index"):
            return self.vol.axis_index(axis)
        raise ValueError(f"{type(self.vol).__name__} has no orientation; pass an int axis")

    def _overview(self, w, n) -> Overview:
        """Three orthogonal maximum-intensity projections at low resolution.

        Each projection collapses the array axis its anatomical plane is stacked along —
        `axcodes` decides, so a sagittal acquisition's "axial_mip" runs head to foot, not along
        whatever happens to be array axis 0. The images keep the remaining two axes in array
        order (not reoriented). With no known orientation the old convention holds."""
        z0, z1, y0, y1, x0, x1 = w
        s = max(1, int(max(z1 - z0, y1 - y0, x1 - x0) / max(n, 1)))
        # Decimate the slab axis by reading every s-th slice; crop and stride in-plane.
        picks = list(range(z0, z1, s))
        stack = []
        for z in picks:
            stack.append(self.vol.slab(z, z + 1, y0, y1, x0, x1)[0, ::s, ::s])
        vol = np.stack(stack) if stack else np.zeros((1, 1, 1), self.vol.dtype)
        axes = (self.vol.plane_axes() if hasattr(self.vol, "plane_axes")
                else {"axial": 0, "coronal": 1, "sagittal": 2})
        mips = {f"{name}_mip": vol.max(axes[name]) for name in ("axial", "coronal", "sagittal")}
        return Overview("voxels", w, np.arange(vol.shape[2]),
                        {**mips, "thumbnail": mips["axial_mip"]},
                        cheap=False,
                        source_bytes=int(len(picks) * (y1 - y0) * (x1 - x0)
                                         * self.extent.unit_bytes))

    @staticmethod
    def _decimate(a: np.ndarray, stride: int) -> np.ndarray:
        """Every stride-th voxel, as its own array: a strided view would keep the whole
        full-resolution slab alive behind a lease sized for the decimated one."""
        return np.ascontiguousarray(a[::stride, ::stride, ::stride]) if stride > 1 else a

    def _load(self, w, stride):
        z0, z1, y0, y1, x0, x1 = w
        return self._decimate(self.vol.slab(z0, z1, y0, y1, x0, x1), stride)

    def _chunks(self, w, stride):
        """Yield the window slab by slab, sized to the chunk cap."""
        z0, z1, y0, y1, x0, x1 = w
        plane = (y1 - y0) * (x1 - x0) * self.extent.unit_bytes / (stride * stride)
        per = max(1, int(self._chunk_cap() / max(plane, 1)) * stride)
        for a in range(z0, z1, per):
            b = min(a + per, z1)
            yield self._decimate(self.vol.slab(a, b, y0, y1, x0, x1), stride), (a, b, y0, y1, x0, x1)

    def slices(self, axis=0) -> Iterator[tuple]:
        """One 2-D slice at a time along an axis — what a 2-D segmentation model eats.

        `axis` is an array axis or an anatomical name (`'axial'`, `'S'`, ...). Yields
        `(plane, window, i)`: `window` is the chunk the plane came from and `i` its index
        within that chunk. The window is cut into runs of whole planes along `axis`, each sized
        to the chunk cap and leased against the budget while its planes are handed out — so a
        coronal plane is a whole coronal plane (all of z), never a fragment of one, and
        the lease is released when the next run is read, when the loop ends, and when the
        generator is closed early."""
        a = self._axis(axis)
        self.check(whole=False)
        stride = self.stride
        w = list(self._win)
        lo, hi = w[2 * a], w[2 * a + 1]
        others = [math.ceil((w[2 * k + 1] - w[2 * k]) / stride) for k in range(3) if k != a]
        plane = others[0] * others[1] * self.extent.unit_bytes
        per = max(1, int(self._chunk_cap() / max(plane, 1))) * stride
        try:
            for p in range(lo, hi, per):
                win = list(w)
                win[2 * a], win[2 * a + 1] = p, min(p + per, hi)
                s = self._decimate(self.vol.slab(*win), stride)
                self._hold(int(s.nbytes))
                for i in range(s.shape[a]):
                    yield np.take(s, i, axis=a), tuple(win), i
        finally:
            self.release()

    def __repr__(self) -> str:
        c = self.cost()
        fit = ("" if c.get("fits_whole") is None
               else ("  fits" if c["fits_whole"] else "  stream only" if c["fits"] else "  TOO BIG"))
        w = c["window"]
        return (f"VolumeView({self.vol.sequence} z {w[0]}-{w[1]} y {w[2]}-{w[3]} "
                f"x {w[4]}-{w[5]}, stride {c['stride']}, {c['total_human']}{fit})")


# ---------------------------------------------------------------------------
# Writing masks
# ---------------------------------------------------------------------------

class MaskSink3D:
    """A full-resolution label volume written slab by slab, so a segmentation of a 512³ scan
    never needs the whole output in memory. Zarr on disk; `to_nifti()` for the tools that
    expect one file.

    The store carries a marker attribute, which is how it recognises its own directory later.
    `overwrite=False` (the default) *resumes* a store of the same shape and dtype and creates
    one if the path is free; `overwrite=True` replaces a store of its own. A directory holding
    anything else is refused either way (`OverwriteRefused`, a `FileExistsError`) — nothing in
    it is deleted.

    `voxels_labelled` is the number of *distinct* voxels currently labelled, however many
    overlapping patches (sliding-window inference overlaps by design) wrote them;
    `overlap_voxels` counts the labelled voxels that were written over an already-labelled one.
    `affine` travels with the store so `to_nifti()` can put the mask back in the scan's frame:
    build one with `MaskSink3D.like(volume, path)` to take shape and affine from the source.
    """

    MARKER = "geminga_masksink"

    def __init__(self, path: str, shape: tuple, dtype="uint8", chunk: int = 16,
                 affine: Optional[np.ndarray] = None, overwrite: bool = False):
        import zarr
        path = os.fspath(path)
        shape = tuple(int(s) for s in shape)
        if len(shape) != 3:
            raise ValueError(f"a 3-D mask needs a (Z, Y, X) shape, got {shape}")
        dtype = np.dtype(dtype)
        plan = plan_target(path, overwrite, markers=(self.MARKER,), what="a MaskSink3D store",
                           reuse=True)
        stored_affine = None
        if plan == "reuse":
            z = zarr.open(path, mode="a")
            if tuple(z.shape) != shape or np.dtype(z.dtype) != dtype:
                raise OverwriteRefused(
                    f"{path} holds a {tuple(z.shape)} {np.dtype(z.dtype)} mask, not a "
                    f"{shape} {dtype} one; pass overwrite=True to replace it")
            meta = dict(z.attrs.get(self.MARKER) or {})
            stored_affine = meta.get("affine")
            self.voxels_labelled = self._count(z)
        else:
            if plan == "replace":
                clear_zarr_members(path)
            z = zarr.open(path, mode="w-", shape=shape, chunks=(min(chunk, shape[0]), shape[1], shape[2]),
                          dtype=dtype)
            self.voxels_labelled = 0
        self.z = z
        self.path, self.slabs, self.overlap_voxels = path, 0, 0
        self.affine = (np.asarray(affine, float) if affine is not None
                       else None if stored_affine is None else np.asarray(stored_affine, float))
        if plan != "reuse" or (affine is not None and stored_affine is None):
            z.attrs[self.MARKER] = {"version": 1, "affine": None if self.affine is None
                                    else self.affine.tolist()}

    @classmethod
    def like(cls, source, path: str, dtype="uint8", **kw) -> "MaskSink3D":
        """A sink with the shape and affine of a `NiftiVolume`, `DicomSeries` or `VolumeView`."""
        vol = getattr(source, "vol", source)
        return cls(path, tuple(vol.shape[:3]), dtype=dtype,
                   affine=kw.pop("affine", getattr(vol, "affine", None)), **kw)

    @staticmethod
    def _count(z) -> int:
        """Labelled voxels already in a store, counted a chunk-row at a time."""
        step = max(int(z.chunks[0]), 1)
        return sum(int((np.asarray(z[a:a + step]) > 0).sum()) for a in range(0, z.shape[0], step))

    def write(self, labels: np.ndarray, window: tuple) -> int:
        """Store `labels` at `window` = (z0, z1, y0, y1, x0, x1). Returns the labelled voxels in
        this patch; `voxels_labelled` and `overlap_voxels` account for what it replaced."""
        z0, z1, y0, y1, x0, x1 = (int(v) for v in window)
        shape = self.z.shape
        for lo, hi, n in ((z0, z1, shape[0]), (y0, y1, shape[1]), (x0, x1, shape[2])):
            if not 0 <= lo <= hi <= n:
                raise ValueError(f"window {tuple(window)} lies outside the {tuple(shape)} volume")
        lab = np.broadcast_to(np.asarray(labels), (z1 - z0, y1 - y0, x1 - x0))
        was = np.asarray(self.z[z0:z1, y0:y1, x0:x1]) > 0
        now = lab > 0
        self.z[z0:z1, y0:y1, x0:x1] = lab
        self.slabs += 1
        n = int(now.sum())
        self.overlap_voxels += int((now & was).sum())
        self.voxels_labelled += n - int(was.sum())
        return n

    def to_nifti(self, path: str, affine: Optional[np.ndarray] = None) -> str:
        """Write the whole mask as one NIfTI (it is materialised: nibabel takes an array).

        The affine is the sink's own (from `affine=` at construction, `MaskSink3D.like`, or the
        store's metadata when it was reopened) unless one is given here. With none known this
        raises: an identity affine would place the mask in the wrong frame without a sound."""
        import nibabel as nib
        aff = affine if affine is not None else self.affine
        if aff is None:
            raise ValueError(
                "no affine for this mask: pass affine= to to_nifti(), or build the sink with "
                "MaskSink3D(..., affine=volume.affine) / MaskSink3D.like(volume, path). "
                "Writing an identity affine would silently put the mask in the wrong frame.")
        arr = np.transpose(np.asarray(self.z[:]), (2, 1, 0))     # back to (X, Y, Z)
        nib.save(nib.Nifti1Image(arr, np.asarray(aff, float)), path)
        return path

    def __repr__(self) -> str:
        return (f"MaskSink3D({self.path}, {'x'.join(map(str, self.z.shape))}, "
                f"{self.slabs} slabs, {self.voxels_labelled:,} labelled voxels)")


# ---------------------------------------------------------------------------
# Subjects, sessions, sequences — the longitudinal shape
# ---------------------------------------------------------------------------

def _natural(s) -> list:
    """Sort key putting 2 before 10 and numbers before words: 'ses2' < 'ses10'."""
    return [(0, int(t), "") if t.isdigit() else (1, 0, t.lower())
            for t in re.split(r"(\d+)", str(s)) if t != ""]


@dataclass
class Subject:
    id: str
    sessions: dict = field(default_factory=dict)   # session -> {sequence: path}

    @property
    def timepoints(self) -> list:
        return sorted(self.sessions, key=_natural)

    @property
    def longitudinal(self) -> bool:
        return len(self.sessions) > 1

    def sequences(self) -> set:
        return {s for ses in self.sessions.values() for s in ses}

    def path(self, session, sequence: str) -> Optional[str]:
        return self.sessions.get(session, {}).get(sequence)

    def __repr__(self) -> str:
        return (f"Subject({self.id}, {len(self.sessions)} session(s), "
                f"{sorted(self.sequences())})")


#: A directory that names a timepoint rather than a person: ses-2, session01, tp3, visit_1, T2.
_SESSION_DIR = re.compile(r"(?:ses(?:sion)?|tp|time(?:point)?|visit|v|t)[-_ ]?(\d+)", re.I)
#: Sequence tokens that carry a digit and so look like an id: T1, T2, PD2.
_SEQ_DIGITS = re.compile(r"(?:t[12]|pd\d?)", re.I)


def _nested_layout(rel: str):
    """The fallback for trees named by directory: `Center_01/Patient_02/T1/flair.nii.gz`.

    The subject is the directory path down to the deepest directory that carries an identifier
    (a digit) and is not a session folder — `Center_01/Patient_02`, so two centres' patient 02
    stay two subjects. Folders below it that name no one (`Raw_Data`, `Masks`) are containers,
    not subjects. A session folder (`ses-3`, `T2`, `tp1`) gives the session, else it is '1'.
    A flat directory falls back to the first identifier in the file name."""
    parts = rel.replace(os.sep, "/").split("/")
    dirs, fname = parts[:-1], parts[-1]
    ses, ident = "1", [i for i, d in enumerate(dirs) if re.search(r"\d", d)]
    for d in dirs:
        m = _SESSION_DIR.fullmatch(d)
        if m:
            ses = m.group(1)
    keep = [i for i in ident if not _SESSION_DIR.fullmatch(dirs[i])]
    if keep:
        return "/".join(dirs[: keep[-1] + 1]), ses
    for m in re.finditer(r"[A-Za-z]*\d+", fname):
        if not _SEQ_DIGITS.fullmatch(m.group()):
            return m.group(), ses
    return None, None


class Cohort:
    """Group a directory of volumes into subjects, sessions and sequences.

    Challenge datasets each have their own layout; this recognises the common ones by regex
    and lets you supply your own. What it gives back is the question a lesion study actually
    asks first: which subjects have which sequences at which timepoints, and which have masks.

    Files are visited in sorted order, so the result does not depend on the filesystem. Two
    files that map to the same (subject, session, sequence) are never allowed to replace one
    another: the first keeps the plain sequence name, the next become `FLAIR_2`, `FLAIR_3`, ...
    and a `UserWarning` names them. Sessions sort numerically (`ses-2` before `ses-10`).

    `layouts=` takes regular expressions with `sub` (and optionally `ses`) groups, matched
    against the path relative to `root` with `/` separators, or functions `rel -> (sub, ses)`
    returning `(None, None)` for "not mine"; the first that matches wins.
    """

    #: (subject, session) capture groups tried in order against the path relative to root. The
    #: last entry is a function, not a pattern: trees named by directory (`Center_01/Patient_02`)
    #: are keyed by their directory path rather than by the first number in the path.
    LAYOUTS = [
        re.compile(r"sub-(?P<sub>[A-Za-z0-9]+)[/_].*?ses-(?P<ses>[A-Za-z0-9]+)", re.I),  # BIDS
        re.compile(r"(?P<sub>P\d+|patient\d+|training\d+|test\d+)[/_](?:tp|time|visit|t)?"
                   r"(?P<ses>\d+)", re.I),                                                  # MSLesSeg-ish
        re.compile(r"(?P<sub>\d{2,3})[/_](?P<ses>\d{2,3})(?![0-9])", re.I),               # ISBI-ish
        _nested_layout,                                                                     # by directory
    ]

    def __init__(self, root: str, layouts: Optional[Sequence] = None,
                 exts: Sequence[str] = (".nii", ".nii.gz")):
        self.root = os.path.abspath(root)
        self.layouts = list(layouts) if layouts else self.LAYOUTS
        self.subjects: dict[str, Subject] = {}
        self.unassigned: list = []
        for dp, dns, fs in os.walk(self.root):
            dns[:] = sorted(d for d in dns if not d.startswith("."))
            for f in sorted(fs):
                if not f.lower().endswith(tuple(exts)):
                    continue
                full = os.path.join(dp, f)
                rel = os.path.relpath(full, self.root).replace(os.sep, "/")
                sub, ses = self._match(rel)
                if sub is None:
                    self.unassigned.append(rel)
                    continue
                seq = guess_sequence(f)
                slot = self.subjects.setdefault(sub, Subject(sub)).sessions.setdefault(ses, {})
                key = seq
                if key in slot:
                    n = 2
                    while f"{seq}_{n}" in slot:
                        n += 1
                    key = f"{seq}_{n}"
                    warnings.warn(
                        f"Cohort: subject {sub!r} session {ses!r} already has a {seq} volume "
                        f"({os.path.relpath(slot[seq], self.root)}); {rel} is kept as {key!r} "
                        f"instead of replacing it", UserWarning, stacklevel=2)
                slot[key] = full

    def _match(self, rel: str):
        for rx in self.layouts:
            if callable(rx):
                sub, ses = rx(rel)
                if sub is not None:
                    return sub, ses or "1"
                continue
            m = rx.search(rel)
            if m:
                d = m.groupdict()
                return d.get("sub"), d.get("ses", "1") or "1"
        return None, None

    def __len__(self) -> int:
        return len(self.subjects)

    def __iter__(self) -> Iterator[Subject]:
        return iter(self.subjects.values())

    def __getitem__(self, sub: str) -> Subject:
        return self.subjects[sub]

    def longitudinal(self) -> list:
        return [s for s in self.subjects.values() if s.longitudinal]

    def with_mask(self) -> list:
        return [s for s in self.subjects.values() if "MASK" in s.sequences()]

    def complete(self, sequences: Sequence[str] = ("T1w", "T2w", "FLAIR")) -> list:
        want = set(sequences)
        return [s for s in self.subjects.values()
                if all(want <= set(ses) for ses in s.sessions.values())]

    def paired_windows(self, sub: str, sequence: str, budget: Optional[Budget] = None):
        """The same window across every timepoint of one subject — the longitudinal look."""
        s = self.subjects[sub]
        views = {}
        for ses in s.timepoints:
            p = s.path(ses, sequence)
            if p:
                views[ses] = VolumeView(open_volume(p), budget)
        return views

    def report(self) -> dict:
        seqs: dict[str, int] = {}
        dups = 0
        for s in self.subjects.values():
            for q in s.sequences():
                base = re.sub(r"_\d+$", "", q)          # FLAIR_2 is a second FLAIR, not a sequence
                dups += base != q
                seqs[base] = seqs.get(base, 0) + 1
        return {"root": self.root, "subjects": len(self.subjects),
                "longitudinal": len(self.longitudinal()), "with_mask": len(self.with_mask()),
                "sequences": seqs, "unassigned": len(self.unassigned), "duplicates": dups,
                "sessions": sum(len(s.sessions) for s in self.subjects.values())}

    def print_report(self) -> None:
        r = self.report()
        print(f"cohort {r['root']}")
        print(f"  {r['subjects']} subjects, {r['sessions']} sessions, "
              f"{r['longitudinal']} longitudinal, {r['with_mask']} with masks")
        print("  sequences: " + ", ".join(f"{k}={v}" for k, v in sorted(r["sequences"].items())))
        if r["duplicates"]:
            print(f"  {r['duplicates']} duplicate sequence file(s) kept as FLAIR_2-style entries")
        if r["unassigned"]:
            print(f"  {r['unassigned']} file(s) did not match any layout — pass layouts=")
        for s in list(self.subjects.values())[:8]:
            tps = ", ".join(f"{t}:{'/'.join(sorted(s.sessions[t]))}" for t in s.timepoints)
            print(f"    {s.id:<12} {tps}")
        if len(self.subjects) > 8:
            print(f"    ... {len(self.subjects) - 8} more")

    def __repr__(self) -> str:
        r = self.report()
        return (f"Cohort({os.path.basename(self.root)}, {r['subjects']} subjects, "
                f"{r['longitudinal']} longitudinal, {r['with_mask']} masked)")
