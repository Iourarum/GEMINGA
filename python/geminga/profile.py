"""What *is* this dataset?

A draft, deliberately. The goal is not a classifier that emits one label; it is a scoring
frame that an analyst can argue with. The distinction matters more here than in most places:
scRNA-seq and snRNA-seq produce byte-identical file formats and differ in a property —
intronic fraction — that you can only see by looking at the reads or the metadata. A tool
that prints "scRNA-seq" with no hedging on such a file is lying to a student who does not yet
know to doubt it.

So `profile()` returns ranked hypotheses, each carrying the evidence that produced it and the
evidence that would settle it. Where the data cannot distinguish two options, it says so and
names the observation that would.

## Evidence sources, in order of how much they are worth

1. **Container structure** — HDF5 group layout, Parquet schema, FASTQ record shape. Nearly
   conclusive about *format*, nearly silent about *assay*.
2. **Metadata written by the pipeline** — `uns`/`obs` keys in an h5ad, GEF attributes, FASTQ
   header fields, GEM `#` headers. The strongest assay signal when present, and often absent.
3. **Content statistics** — read length distribution, GC, barcode-shaped prefixes, gene-symbol
   style, spatial coordinate ranges. Weak individually, useful in combination.
4. **The page it came from** — if a URL is given, the HTML around it usually names the assay
   outright. Cheap, and frequently the only thing that separates sc from sn.

## On topic modelling

The obvious suggestion for step 4 is LDA. It is the wrong tool at this scale: LDA needs a
corpus to fit and produces topics that then need naming, and here there is one document and
a known vocabulary. A weighted keyword score over curated terms is transparent, needs no
training, and is trivially auditable — the analyst sees which words fired. The upgrade path
when a corpus does exist is a sentence-embedding similarity against protocol descriptions,
not LDA; that is noted in the roadmap rather than pretended at.
"""
from __future__ import annotations

import functools
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["Evidence", "Hypothesis", "Profile", "profile", "ASSAYS", "MOLECULES"]


@dataclass
class Evidence:
    """One observation, and how much it moves belief.

    `supports` may name several hypotheses. That is not a convenience: evidence that cannot
    separate two assays must lift both, or it quietly settles the question it is blind to.
    "20,000 observations, so cell-level" argues for scRNA-seq *and* snRNA-seq equally — a
    nucleus is a row too.
    """
    source: str            # "structure" | "metadata" | "content" | "page" | "filename"
    detail: str
    weight: float          # log-odds contribution; negative argues against
    supports: Any          # a label, or a sequence of labels it bears on equally

    @property
    def labels(self) -> tuple:
        if isinstance(self.supports, str):
            return (self.supports,)
        return tuple(self.supports)

    def __repr__(self) -> str:
        sign = "+" if self.weight >= 0 else ""
        return (f"[{self.source}] {self.detail} "
                f"({sign}{self.weight:.1f} -> {', '.join(self.labels)})")


@dataclass
class Hypothesis:
    label: str
    kind: str              # "format" | "molecule" | "assay" | "modality"
    logodds: float = 0.0
    evidence: list = field(default_factory=list)
    probability: float = 0.0
    distinguish: str = ""  # what observation would settle it against its nearest rival

    def __repr__(self) -> str:
        return f"{self.label} ({self.probability:.0%})"


#: Molecule classes a dataset can carry. Deliberately coarse.
MOLECULES = ("DNA", "RNA", "protein", "lipid", "metabolite", "image", "mixed", "unknown")

#: Assay priors and their keyword evidence. Weights are log-odds, hand-set and meant to be
#: argued with — they are a starting prior, not a trained model.
ASSAYS: dict[str, dict] = {
    "scRNA-seq": dict(molecule="RNA", prior=-1.0, distinguish=
        "intronic fraction: scRNA-seq is exon-dominated (<30% intronic). Check the pipeline's "
        "intron/exon report, or the ratio of reads mapping to intronic space.",
        terms={"scrna": 3.0, "single-cell rna": 3.0, "single cell rna": 3.0, "10x genomics": 1.5,
               "chromium": 1.5, "cellranger": 2.0, "smart-seq": 2.0, "drop-seq": 2.0,
               "cell suspension": 1.5, "dissociated cells": 1.5}),
    "snRNA-seq": dict(molecule="RNA", prior=-1.6, distinguish=
        "intronic fraction: snRNA-seq is intron-rich (often >50%) because nuclear RNA is "
        "largely unspliced. Also look for 'nuclei isolation' in the methods.",
        terms={"snrna": 3.5, "single-nucleus": 3.5, "single nucleus": 3.5, "nuclei": 2.0,
               "nucleus rna": 3.0, "frozen tissue": 1.0, "nuclei isolation": 3.0,
               "include-introns": 2.5, "premrna": 2.0}),
    "bulk RNA-seq": dict(molecule="RNA", prior=-0.8, distinguish=
        "absence of per-cell structure: one expression vector per sample rather than per "
        "barcode. Check whether the matrix has thousands of columns or a handful.",
        terms={"bulk rna": 3.0, "rna-seq": 1.0, "total rna": 1.5, "polya": 1.0,
               "ribo-depletion": 1.5, "salmon": 1.0, "kallisto": 1.0, "featurecounts": 1.5}),
    "scATAC-seq": dict(molecule="DNA", prior=-1.8, distinguish=
        "features are genomic intervals (chr:start-end), not gene symbols. Look at var_names.",
        terms={"scatac": 3.5, "snatac": 3.5, "atac": 2.5, "atacseq": 2.5,
               "chromatin accessibility": 3.0, "tn5": 2.0, "peaks": 1.2, "fragments.tsv": 3.0}),
    "Stereo-seq": dict(molecule="RNA", prior=-2.0, distinguish=
        "a GEF container with a bin pyramid, or a GEM header naming a chip serial.",
        terms={"stereo-seq": 4.0, "stereoseq": 4.0, "stomics": 3.5, "saw pipeline": 3.0,
               "dnb": 2.0, "bin1": 2.0, "cellbin": 2.5, "mgi": 1.2, "bgi": 1.0}),
    "Visium / spot spatial": dict(molecule="RNA", prior=-2.2, distinguish=
        "the bundle layout names the platform: Space Ranger (Visium, 55 um spots or HD 2 um "
        "bins), Xenium (per-transcript parquet), CosMx/MERFISH (flat CSVs).",
        terms={"visium": 4.0, "spaceranger": 3.5, "spot barcode": 2.5, "xenium": 4.0,
               "cosmx": 4.0, "merfish": 4.0, "spatial transcriptomics": 1.5,
               "tissue_positions": 3.0, "in situ": 1.5}),
    "WGS / WES": dict(molecule="DNA", prior=-1.5, distinguish=
        "coverage uniformity across the genome (WGS) versus enrichment over exons (WES).",
        terms={"whole genome sequencing": 3.0, "wgs": 2.5, "exome": 3.0, "wes": 2.5,
               "variant calling": 1.5, "gvcf": 2.0, "bwa": 1.0}),
    "proteomics (MS)": dict(molecule="protein", prior=-2.2, distinguish=
        "spectra or peptide identifications rather than nucleotide sequence.",
        terms={"mass spectrometry": 3.0, "proteomics": 3.0, "peptide": 2.0, "maxquant": 3.0,
               "tmt": 2.0, "dia-nn": 3.0, "orbitrap": 2.5}),
    "lipidomics": dict(molecule="lipid", prior=-2.6, distinguish=
        "lipid species nomenclature (PC 34:1) in the feature names.",
        terms={"lipidomics": 4.0, "lipid species": 3.0, "lipidsearch": 3.0}),
    "MRI (structural neuroimaging)": dict(molecule="image", prior=-2.2, distinguish=
        "a NIfTI or DICOM volume whose sequence name (T1w/T2w/FLAIR) or DICOM Modality=MR "
        "identifies it; CT has Modality=CT and Hounsfield-range intensities.",
        terms={"mri": 3.0, "flair": 4.0, "t1-weighted": 3.5, "t2-weighted": 3.5, "t1w": 3.0,
               "t2w": 3.0, "mprage": 3.5, "nifti": 3.0, "lesion segmentation": 3.5,
               "white matter": 2.0, "multiple sclerosis": 2.5, "wmh": 3.0, "3t": 1.5,
               "1.5t": 1.5, "siemens": 1.0, "bids": 2.0, "fsl": 1.5, "freesurfer": 2.0}),
    "medical imaging (DICOM, other)": dict(molecule="image", prior=-2.8, distinguish=
        "DICOM Modality tag: CT, PT, US, XA. MR is handled as MRI above.",
        terms={"dicom": 2.0, "ct scan": 3.0, "pacs": 3.0, "radiology": 2.0,
               "hounsfield": 3.5, "pet-ct": 3.5}),
    "microscopy / histology": dict(molecule="image", prior=-2.0, distinguish=
        "a pyramidal image with a physical pixel size and no nucleotide data alongside.",
        terms={"h&e": 3.0, "immunofluorescence": 3.0, "ssdna": 2.5, "whole slide": 3.0,
               "ome-tiff": 2.5, "histology": 2.5, "microscopy": 2.0}),
}

_FORMAT_BY_EXT = {
    ".gef": ("Stereo-seq GEF", "RNA"), ".gem": ("Stereo-seq GEM", "RNA"),
    ".h5ad": ("AnnData h5ad", None), ".parquet": ("Parquet", None),
    ".fastq": ("FASTQ", None), ".fq": ("FASTQ", None), ".fasta": ("FASTA", None),
    ".fa": ("FASTA", None), ".fna": ("FASTA", "DNA"), ".faa": ("FASTA", "protein"),
    ".bam": ("BAM", None), ".cram": ("CRAM", None), ".sam": ("SAM", None), ".vcf": ("VCF", "DNA"),
    ".arrow": ("Arrow IPC", None), ".feather": ("Arrow IPC", None),
    ".tif": ("TIFF", "image"), ".tiff": ("TIFF", "image"), ".zarr": ("Zarr", "image"),
    ".dcm": ("DICOM", "image"), ".mzml": ("mzML", "protein"), ".mgf": ("MGF", "protein"),
    ".nii": ("NIfTI", "image"), ".nii.gz": ("NIfTI.gz", "image"), ".ima": ("DICOM", "image"),
}


#: Longer keywords that are also the opening letters of unrelated words, so they must end at a
#: token boundary too: 'nuclei' is not 'nucleic acid', 'salmon' (the quantifier) is not Salmonella.
_WHOLE_WORD = frozenset({"nuclei", "salmon"})


@functools.lru_cache(maxsize=None)
def _term_rx(term: str):
    """A keyword as a *token*, not a substring, in lower-case text.

    It must start at a token boundary. Short keywords (4 characters or fewer: 'wes', 'mri',
    '3t', 'atac') must also end at one, optionally pluralised, so 'wes' does not fire inside
    'western' or '3t' inside '13th' or 'NIH-3T3'. Longer ones may run on ('scrna' still matches
    'scRNAseq') unless listed in `_WHOLE_WORD`. Same rule as `harmonize.Vocabulary`, restated
    here to keep this module free of that import."""
    closed = len(term) <= 4 or term in _WHOLE_WORD
    tail = r"(?:e?s)?(?=$|[^a-z0-9])" if closed else ""
    return re.compile(r"(?:^|(?<=[^a-z0-9]))" + re.escape(term) + tail)


def _count_term(text: str, term: str) -> int:
    return len(_term_rx(term).findall(text))


@dataclass
class Profile:
    """The answer: ranked hypotheses, the evidence behind them, and what is still unresolved."""
    target: str
    format: Optional[str] = None
    molecule: Optional[str] = None
    assays: list = field(default_factory=list)      # list[Hypothesis]
    modality: str = "unknown"                        # "single" | "multi-modal"
    evidence: list = field(default_factory=list)     # list[Evidence]
    notes: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def best(self) -> Optional[Hypothesis]:
        return self.assays[0] if self.assays else None

    #: Evidence from these sources can tell two assays apart. Structure and filename
    #: identify the *container*, which is a different question and must not be mistaken
    #: for settling the assay — a .h5ad is equally an scRNA-seq or an snRNA-seq file.
    DISCRIMINATING = ("metadata", "content", "page")

    @property
    def discriminating_evidence(self) -> list:
        return [e for e in self.evidence
                if e.source in self.DISCRIMINATING and e.weight > 0]

    @property
    def confident(self) -> bool:
        """True only when the leader is clearly ahead *and* something actually discriminated.

        A container tells you the format, not the assay. Without metadata, content or page
        evidence, a high probability is just the prior talking, and saying so is the whole
        point of this module.
        """
        if not self.assays:
            return False
        if not self.discriminating_evidence:
            return False
        if len(self.assays) < 2:
            return self.assays[0].probability > 0.6
        lead, rival = self.assays[0], self.assays[1]
        # Evidence that lifts both the leader and the runner-up equally cannot have chosen
        # between them; only evidence naming the leader and not the rival counts here.
        separates = any(lead.label in e.labels and rival.label not in e.labels
                        for e in self.discriminating_evidence)
        return (separates and lead.probability > 0.6
                and lead.probability - rival.probability > 0.25)

    def to_dict(self) -> dict:
        return {
            "target": self.target, "format": self.format, "molecule": self.molecule,
            "modality": self.modality, "confident": self.confident,
            "assays": [{"label": h.label, "probability": round(h.probability, 4),
                        "distinguish": h.distinguish} for h in self.assays[:5]],
            "evidence": [{"source": e.source, "detail": e.detail,
                          "weight": e.weight, "supports": e.supports} for e in self.evidence],
            "notes": self.notes, "metadata": self.metadata,
        }

    def explain(self) -> str:
        out = [f"{self.target}",
               f"  format   : {self.format or 'unrecognised'}",
               f"  molecule : {self.molecule or 'unknown'}",
               f"  modality : {self.modality}"]
        if self.assays:
            out.append("  assay    :")
            for h in self.assays[:4]:
                out.append(f"      {h.probability:>5.0%}  {h.label}")
            lead = self.assays[0]
            if not self.confident:
                rival = self.assays[1].label if len(self.assays) > 1 else "the alternatives"
                if not self.discriminating_evidence:
                    out.append("  NOT SETTLED — nothing here distinguishes one assay from "
                               "another; the ranking is the prior plus the container.")
                else:
                    out.append(f"  NOT SETTLED — {lead.label} versus {rival}.")
                out.append(f"  To decide: {lead.distinguish}")
        if self.evidence:
            out.append("  evidence :")
            for e in sorted(self.evidence, key=lambda x: -abs(x.weight))[:8]:
                out.append(f"      {e}")
        for n in self.notes:
            out.append(f"  note     : {n}")
        return "\n".join(out)

    def __repr__(self) -> str:
        b = self.best
        return (f"Profile({os.path.basename(self.target)}, {self.format}, "
                f"{b.label if b else 'assay unknown'}"
                f"{'' if self.confident else ' — unsettled'})")


# ---------------------------------------------------------------------------
# Evidence collectors
# ---------------------------------------------------------------------------

def _from_filename(path: str, ev: list) -> tuple[Optional[str], Optional[str]]:
    low = os.path.basename(path).lower()
    fmt = mol = None
    for ext, (f, m) in _FORMAT_BY_EXT.items():
        if low.endswith(ext) or low.endswith(ext + ".gz"):
            fmt, mol = f, m
            ev.append(Evidence("filename", f"extension {ext} -> {f}", 2.0, f))
            break
    if ".cellbin." in low:
        ev.append(Evidence("filename", "cellbin in the name: cell-segmented Stereo-seq",
                           2.5, "Stereo-seq"))
    for token, label, w in (("scrna", "scRNA-seq", 2.0), ("snrna", "snRNA-seq", 2.0),
                            ("atac", "scATAC-seq", 1.5), ("scatac", "scATAC-seq", 1.5),
                            ("snatac", "scATAC-seq", 1.5), ("atacseq", "scATAC-seq", 1.5),
                            ("wgs", "WGS / WES", 1.5), ("visium", "Visium / spot spatial", 2.0)):
        if _count_term(low, token):
            ev.append(Evidence("filename", f"{token!r} in the filename", w, label))
    return fmt, mol


#: Classic TIFF (42) and BigTIFF (43), little- and big-endian. Two letters alone are not a magic:
#: 'MMP1,MMP2,...' begins with 'MM'.
_TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")

#: A SAM header line. SAM text begins with '@' too, and is not FASTQ.
_SAM_HEADER = re.compile(rb"^@(HD|SQ|RG|PG|CO)\t")


def _looks_like_fastq(head: bytes) -> bool:
    """'@' opens a FASTQ record, but also a SAM header. A record is id, sequence, '+', quality:
    when the '+' line is in view it must be there; when only the sequence line is, it must
    look like one."""
    if not head.startswith(b"@") or _SAM_HEADER.match(head):
        return False
    lines = head.split(b"\n")
    if len(lines) > 2 and lines[2]:
        return lines[2][:1] == b"+"
    if len(lines) > 1:
        return lines[1].strip().isalpha()
    return True                                    # one long line: nothing to check against


def _peek_bytes(path: str, n: int = 4096) -> bytes:
    try:
        with __import__("builtins").open(path, "rb") as fh:
            return fh.read(n)
    except OSError:
        return b""


def _from_structure(path: str, ev: list, notes: list, meta: dict) -> Optional[str]:
    """Look at the container itself. Cheap, and nearly conclusive about format."""
    head = _peek_bytes(path, 512)
    if not head:
        return None

    if head[:8] == b"\x89HDF\r\n\x1a\n":
        try:
            import h5py
            with h5py.File(path, "r") as f:
                keys = list(f.keys())
                meta["hdf5_root"] = keys
                if "geneExp" in keys or "cellBin" in keys:
                    grp = f.get("geneExp") or f.get("cellBin")
                    bins = sorted(int(k[3:]) for k in grp.keys() if k.startswith("bin")
                                  and k[3:].isdigit())
                    meta["bin_levels"] = bins
                    ev.append(Evidence("structure",
                                       f"GEF container with bin levels {bins or '(cellbin)'}",
                                       5.0, "Stereo-seq"))
                    return "Stereo-seq GEF"
                if {"X", "obs", "var"} <= set(keys):
                    ev.append(Evidence("structure", "AnnData layout (X/obs/var)", 3.0,
                                       ("scRNA-seq", "snRNA-seq", "scATAC-seq")))
                    _from_anndata(f, ev, notes, meta)
                    return "AnnData h5ad"
                if "matrix" in keys:
                    try:
                        ft = f["matrix/features/feature_type"][:]
                        ft = {x.decode() if isinstance(x, bytes) else str(x) for x in ft}
                    except Exception:
                        ft = {"Gene Expression"}
                    meta["feature_types"] = sorted(ft)
                    labels = ["scRNA-seq", "snRNA-seq"]
                    if "Peaks" in ft:
                        ev.append(Evidence("metadata", "feature_type Peaks", 4.0, "scATAC-seq"))
                    if "Antibody Capture" in ft:
                        ev.append(Evidence("metadata", "feature_type Antibody Capture: CITE-seq",
                                           3.0, ("scRNA-seq", "snRNA-seq")))
                        meta["cite_seq"] = True
                    if "CRISPR Guide Capture" in ft:
                        ev.append(Evidence("metadata", "feature_type CRISPR Guide Capture: a "
                                           "single-cell perturbation screen", 3.0,
                                           ("scRNA-seq", "snRNA-seq")))
                        meta["crispr_screen"] = True
                    ev.append(Evidence("structure", "10x HDF5 matrix group", 2.5, tuple(labels)))
                    return "10x HDF5"
        except ImportError:
            notes.append("h5py not installed — HDF5 structure not inspected")
        except Exception as e:
            notes.append(f"HDF5 open failed: {e}")
        return "HDF5"

    if head[:4] == b"PAR1":
        ev.append(Evidence("structure", "Parquet magic", 3.0, "Parquet"))
        return "Parquet"
    if head[:6] == b"ARROW1":
        return "Arrow IPC"
    if head[:2] == b"\x1f\x8b":
        notes.append("gzip container — inner format inferred from the name")
    if head[:4] in _TIFF_MAGIC:
        ev.append(Evidence("structure", "TIFF magic", 3.0, "microscopy / histology"))
        return "TIFF"
    # NIfTI-1: magic 'n+1\0' or 'ni1\0' at byte 344 (gzipped files identified by name)
    if len(head) >= 348 and head[344:347] in (b"n+1", b"ni1"):
        ev.append(Evidence("structure", "NIfTI-1 magic at byte 344", 5.0,
                           "MRI (structural neuroimaging)"))
        _from_nifti(path, ev, notes, meta)
        return "NIfTI"
    if head[:2] == b"\x1f\x8b" and path.lower().endswith(".nii.gz"):
        ev.append(Evidence("structure", "gzipped NIfTI by name", 4.0,
                           "MRI (structural neuroimaging)"))
        _from_nifti(path, ev, notes, meta)
        return "NIfTI.gz"
    # DICOM: 'DICM' at offset 128
    d = _peek_bytes(path, 160)
    if len(d) >= 132 and d[128:132] == b"DICM":
        _from_dicom(path, ev, notes, meta)
        return "DICOM"
    if _SAM_HEADER.match(head):
        return "SAM"
    if _looks_like_fastq(head):
        ev.append(Evidence("structure", "FASTQ record header", 3.0, "bulk RNA-seq"))
        return "FASTQ"
    if head.startswith(b">"):
        ev.append(Evidence("structure", "FASTA record header", 3.0, "bulk RNA-seq"))
        return "FASTA"
    if head.startswith(b"#FileFormat=GEM") or b"\ngeneID\tx\ty" in _peek_bytes(path, 2048):
        ev.append(Evidence("structure", "GEM header block", 5.0, "Stereo-seq"))
        return "Stereo-seq GEM"
    return None


def _from_nifti(path: str, ev: list, notes: list, meta: dict):
    try:
        from .neuro import NiftiVolume
        v = NiftiVolume(path)
        d = v.describe()
        meta.update({k: d[k] for k in ("shape_zyx", "voxel_mm", "sequence", "dtype")})
        if v.sequence not in ("unknown", "MASK"):
            ev.append(Evidence("metadata", f"sequence token {v.sequence} in the filename", 2.5,
                               "MRI (structural neuroimaging)"))
        if v.sequence == "MASK":
            ev.append(Evidence("metadata", "filename marks this as a mask / label volume", 2.0,
                               "MRI (structural neuroimaging)"))
            meta["is_label_volume"] = True
        if v.ndim == 4:
            ev.append(Evidence("content", f"4-D volume ({v.shape[3]} frames): fMRI or DWI",
                               1.5, "MRI (structural neuroimaging)"))
        v.close()
    except Exception as e:
        notes.append(f"NIfTI header not readable: {e}")


def _from_dicom(path: str, ev: list, notes: list, meta: dict):
    try:
        import pydicom
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        mod = str(getattr(ds, "Modality", ""))
        desc = str(getattr(ds, "SeriesDescription", ""))
        meta.update({"modality": mod, "series_description": desc,
                     "manufacturer": str(getattr(ds, "Manufacturer", "")),
                     "field_strength_T": getattr(ds, "MagneticFieldStrength", None)})
        if mod == "MR":
            ev.append(Evidence("metadata", "DICOM Modality=MR", 5.0,
                               "MRI (structural neuroimaging)"))
            from .neuro import guess_sequence
            seq = guess_sequence(desc)
            meta["sequence"] = seq
            if seq not in ("unknown", "MASK"):
                ev.append(Evidence("metadata", f"SeriesDescription names {seq}", 2.0,
                                   "MRI (structural neuroimaging)"))
        elif mod:
            ev.append(Evidence("metadata", f"DICOM Modality={mod}", 5.0,
                               "medical imaging (DICOM, other)"))
        else:
            ev.append(Evidence("structure", "DICOM magic at byte 128", 4.0,
                               ("MRI (structural neuroimaging)",
                                "medical imaging (DICOM, other)")))
    except Exception as e:
        notes.append(f"DICOM header not readable: {e}")
        ev.append(Evidence("structure", "DICOM magic at byte 128", 4.0,
                           ("MRI (structural neuroimaging)", "medical imaging (DICOM, other)")))


def _from_anndata(f, ev: list, notes: list, meta: dict):
    """The sc-versus-sn question lives here, when the pipeline bothered to record it."""
    def _try(label, fn):
        try:
            fn()
        except Exception as exc:
            notes.append(f"{label} not readable: {exc}")

    def _uns():
        uns = f.get("uns")
        keys = list(uns.keys()) if uns is not None else []
        meta["uns_keys"] = keys[:40]
        blob = " ".join(keys).lower()
        for token, label, w in (("intron", "snRNA-seq", 2.5), ("nucle", "snRNA-seq", 2.5),
                                ("spatial", "Visium / spot spatial", 2.0),
                                ("peak", "scATAC-seq", 2.0)):
            if token in blob:
                ev.append(Evidence("metadata", f"{token!r} among uns keys", w, label))

    def _var():
        var = f.get("var")
        if var is not None:
            idx = var.attrs.get("_index", b"_index")
            idx = idx.decode() if isinstance(idx, bytes) else idx
            node = var.get(idx)
            if node is not None and hasattr(node, "shape"):
                names = node[:50]
                names = [x.decode() if isinstance(x, bytes) else str(x) for x in names]
                meta["var_sample"] = names[:8]
                interval = sum(bool(re.match(r"^(chr)?[\dXYM]+[:_-]\d+", n)) for n in names)
                if interval > len(names) * 0.5:
                    ev.append(Evidence("metadata",
                                       "var names are genomic intervals, not gene symbols",
                                       4.0, "scATAC-seq"))
                else:
                    ev.append(Evidence("metadata", "var names look like gene symbols",
                                       1.0, ("scRNA-seq", "snRNA-seq", "bulk RNA-seq")))

    def _obs():
        obs = f.get("obs")
        if obs is not None:
            ok = list(obs.keys())
            meta["obs_keys"] = ok[:40]
            # Prefer a real array: in AnnData a categorical column is a *group* holding
            # codes and categories, and len() on that is 2, not the number of cells.
            n = None
            for k in ok:
                node = obs.get(k)
                if node is not None and getattr(node, "shape", None):
                    n = int(node.shape[0]); break
            if n is None:
                for k in ok:
                    codes = obs.get(f"{k}/codes")
                    if codes is not None and getattr(codes, "shape", None):
                        n = int(codes.shape[0]); break
            if n:
                meta["n_obs"] = int(n)
                if n < 200:
                    ev.append(Evidence("content", f"only {n} observations — sample-level, "
                                       f"not cell-level", 2.0, "bulk RNA-seq"))
                elif n > 2000:
                    ev.append(Evidence("content",
                                       f"{n:,} observations — per-cell, not per-sample "
                                       f"(a nucleus is a row too)",
                                       1.5, ("scRNA-seq", "snRNA-seq", "scATAC-seq")))

    _try("uns", _uns)
    _try("var", _var)
    _try("obs", _obs)


def _from_content(path: str, fmt: Optional[str], ev: list, notes: list, meta: dict):
    """A quick look at actual data. Weak signals, but they combine."""
    if fmt in ("FASTQ", "FASTA"):
        try:
            import geminga as g
            st = g.open_fastx(path, chunk_bytes="2MB", max_records=2000)
            batch = next(iter(st), None)
            if batch is None:
                return
            seqs = batch.column("seq").to_pylist()[:2000]
            ids = batch.column("id").to_pylist()[:50]
            lens = [len(s) for s in seqs]
            meta["read_length_mean"] = sum(lens) / len(lens)
            meta["n_sampled"] = len(seqs)
            alpha = set("".join(seqs[:50]).upper())
            nucl = alpha <= set("ACGTUNRYKMSWBDHV")
            meta["alphabet"] = "nucleotide" if nucl else "amino-acid-like"
            if not nucl:
                ev.append(Evidence("content", "non-nucleotide alphabet in the sequences",
                                   4.0, "proteomics (MS)"))
            gc = sum(s.upper().count("G") + s.upper().count("C") for s in seqs[:500])
            tot = sum(len(s) for s in seqs[:500]) or 1
            meta["gc_fraction"] = round(gc / tot, 4)
            if nucl and 40 <= (sum(lens) / len(lens)) <= 120:
                ev.append(Evidence("content",
                                   f"short reads (mean {sum(lens)/len(lens):.0f} bp) — "
                                   f"short-read sequencing, which says nothing about assay",
                                   0.4, ("bulk RNA-seq", "scRNA-seq", "snRNA-seq",
                                         "scATAC-seq", "WGS / WES")))
            if any(re.match(r"^[ACGT]{14,18}[+_]", i) for i in ids):
                ev.append(Evidence("content", "barcode-shaped read prefixes", 2.0,
                                   ("scRNA-seq", "snRNA-seq", "scATAC-seq")))
            blob = " ".join(ids).lower()
            for token, label, w in (("nuclei", "snRNA-seq", 2.0), ("atac", "scATAC-seq", 2.0)):
                if _count_term(blob, token):       # not 'atac' inside an index sequence
                    ev.append(Evidence("content", f"{token!r} in read identifiers", w, label))
        except Exception as e:
            notes.append(f"content scan incomplete: {e}")

    elif fmt == "Stereo-seq GEF":
        try:
            import geminga as g
            v = g.open_gef(path)
            meta["bin_levels"] = v.gef.bins
            meta["extent"] = v.gef.extent
            meta["n_genes"] = int(len(v.gef.genes(v.gef.bins[0])))
            meta["chip"] = v.gef.chip
            v.gef.close()
            ev.append(Evidence("content",
                               f"{meta['n_genes']:,} genes across bins {meta['bin_levels']}",
                               3.0, "Stereo-seq"))
        except Exception as e:
            notes.append(f"GEF content scan incomplete: {e}")


def _from_page(url: str, ev: list, notes: list, meta: dict, timeout: float = 8.0):
    """Grep the page a dataset came from. Usually the only thing that separates sc from sn."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "geminga/0.4"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read(400_000).decode("utf-8", "replace")
    except Exception as e:
        notes.append(f"could not fetch {url}: {e}")
        return
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw,
                  flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).lower()
    meta["page_chars"] = len(text)
    hits = {}
    for label, spec in ASSAYS.items():
        for term, w in spec["terms"].items():
            n = _count_term(text, term)
            if n:
                # Diminishing returns: the tenth mention is not ten times the first.
                contrib = w * min(1.0 + math.log10(n), 2.2)
                hits.setdefault(label, []).append((term, n, contrib))
    for label, found in hits.items():
        total = sum(c for _, _, c in found)
        top = ", ".join(f"{t}x{n}" for t, n, _ in sorted(found, key=lambda x: -x[2])[:3])
        ev.append(Evidence("page", f"source page mentions {top}", total, label))
    meta["page_terms"] = {k: [t for t, _, _ in v] for k, v in hits.items()}


# ---------------------------------------------------------------------------

_DICOM_EXT = (".dcm", ".ima")
_NIFTI_EXT = (".nii", ".nii.gz")


def _find_volumes(root: str, depth: int = 3, listing_cap: int = 200, fanout: int = 8,
                  dir_cap: int = 200) -> tuple:
    """DICOM and NIfTI files under `root`, looking up to `depth` directory levels down.

    Series are rarely at the top of an export: `study/series_1/IM0001.dcm`, `sub-01/anat/x.nii.gz`.
    Returns `(dicom_files, nifti_files)` from the first directory, breadth first, that holds
    either — so a count describes one series, not a whole archive. Bounded, because profiling
    must stay cheap on a slow mount: at most `listing_cap` entries read per directory, the first
    `fanout` subdirectories followed from each, and `dir_cap` directories visited. A Zarr store
    is one dataset, not a folder of series, and is not entered.
    """
    if root.rstrip("/\\").lower().endswith(".zarr"):
        return [], []
    queue, seen = [(root, 0)], 0
    while queue and seen < dir_cap:
        here, level = queue.pop(0)
        seen += 1
        try:
            names = sorted(os.listdir(here))[:listing_cap]
        except OSError:
            continue
        dicom, nifti, subdirs = [], [], []
        for name in names:
            full, low = os.path.join(here, name), name.lower()
            if os.path.isdir(full):
                if not name.startswith(".") and not low.endswith(".zarr"):
                    subdirs.append(full)
            elif low.endswith(_DICOM_EXT):
                dicom.append(full)
            elif low.endswith(_NIFTI_EXT):
                nifti.append(full)
        if dicom or nifti:
            return dicom, nifti
        if level < depth:
            queue.extend((d, level + 1) for d in subdirs[:fanout])
    return [], []


def _prior_probability(label: str) -> float:
    """What a hypothesis is worth before any evidence: its share of the prior odds."""
    weights = {k: math.exp(v["prior"]) for k, v in ASSAYS.items()}
    return weights[label] / sum(weights.values())


def profile(target: str, url: Optional[str] = None, scan_content: bool = True,
            fetch_page: bool = True) -> Profile:
    """Work out what a dataset is, and say how sure that is.

    `target` is a path; `url` is the page it came from, if you have it. Set `fetch_page=False`
    to stay offline.

    Returns a :class:`Profile` — ranked assay hypotheses with the evidence behind each, plus,
    where two remain plausible, the observation that would separate them.
    """
    ev: list = []
    notes: list = []
    meta: dict = {}
    p = Profile(target=str(target))

    fmt_name, mol = _from_filename(str(target), ev)
    if os.path.isdir(target):
        try:
            from .spatial import sniff_bundle
            b = sniff_bundle(str(target))
            if b["kind"] not in ("directory",):
                meta["bundle"] = b["kind"]
                lab = {"visium": "Visium / spot spatial", "visium-hd": "Visium / spot spatial",
                       "xenium": "Visium / spot spatial", "cosmx": "Visium / spot spatial",
                       "merfish": "Visium / spot spatial", "stomics": "Stereo-seq",
                       "cellranger": ("scRNA-seq", "snRNA-seq"),
                       "mtx-triple": ("scRNA-seq", "snRNA-seq"),
                       "imzml": "proteomics (MS)"}.get(b["kind"])
                if lab:
                    ev.append(Evidence("structure", f"{b['kind']} bundle layout", 4.0, lab))
                    fmt_name = b["kind"]
                    if b["kind"] in ("xenium", "cosmx", "merfish"):
                        meta["platform"] = b["spatial"]["platform"]
        except Exception as e:
            notes.append(f"bundle sniff failed: {e}")
        dcm, nii = _find_volumes(str(target))
        if dcm:
            _from_dicom(dcm[0], ev, notes, meta)
            meta["n_dicom_files_seen"] = len(dcm)
            fmt_name = "DICOM series"
        elif nii:
            fmt_name = _from_structure(nii[0], ev, notes, meta) or "NIfTI"
            meta["n_nifti_files_seen"] = len(nii)
    if os.path.exists(target) and not os.path.isdir(target):
        found = _from_structure(str(target), ev, notes, meta)
        if found:
            fmt_name = found
        if scan_content:
            _from_content(str(target), fmt_name, ev, notes, meta)
    elif not os.path.exists(target):
        notes.append("path does not exist locally — profiled from the name and page only")

    if url and fetch_page:
        _from_page(url, ev, notes, meta)

    # Accumulate log-odds per assay from its prior plus the evidence bearing on it.
    hyps: dict[str, Hypothesis] = {}
    for label, spec in ASSAYS.items():
        hyps[label] = Hypothesis(label, "assay", spec["prior"],
                                 distinguish=spec.get("distinguish", ""))
    for e in ev:
        for label in e.labels:
            if label in hyps:
                hyps[label].logodds += e.weight
                hyps[label].evidence.append(e)

    mx = max(h.logodds for h in hyps.values())
    exps = {k: math.exp(h.logodds - mx) for k, h in hyps.items()}
    tot = sum(exps.values()) or 1.0
    for k, h in hyps.items():
        h.probability = exps[k] / tot
    ranked = sorted(hyps.values(), key=lambda h: -h.probability)

    p.format = fmt_name
    p.assays = ranked
    p.evidence = ev
    p.notes = notes
    p.metadata = meta

    lead = ranked[0]
    # The leader's molecule is only a claim if evidence put it there. With nothing to go on the
    # front-runner is just the assay with the highest prior (about 21%), and saying "RNA" about
    # a file nobody has looked at would be the prior talking.
    p.molecule = mol
    if p.molecule is None and lead.probability > 0.2 \
            and lead.probability > _prior_probability(lead.label) + 1e-9:
        p.molecule = ASSAYS[lead.label]["molecule"]
    if meta.get("bin_levels") and fmt_name and "GEF" in str(fmt_name):
        p.modality = "multi-modal"
        notes.append("Stereo-seq chips normally ship with a registered ssDNA or H&E image — "
                     "look for a TIFF beside this file and open both with open_stereoseq().")
    elif fmt_name in ("FASTQ", "FASTA"):
        p.modality = "single"
    elif fmt_name in ("AnnData h5ad",) and "spatial" in str(meta.get("uns_keys", "")).lower():
        p.modality = "multi-modal"
    return p
