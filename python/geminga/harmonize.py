"""Making pieces from different sources say the same thing in the same words.

Two datasets that both measure PD-L1 in T cells will call it ``CD274``, ``PD-L1``,
``ENSG00000120217``, ``29126`` or ``Q9NZQ7`` depending on who wrote the file; one will say the
cells are ``human`` and the other ``Homo sapiens``; one has a ``donor`` column and the other a
``patient_id``. None of this is analysis, and all of it has to be settled before any analysis
can start. This module settles it, and — the part that matters — *writes down what it did*,
so a harmonised dataset carries the mapping that made it rather than hiding it.

Three small tools, one wrapper:

* :class:`Vocabulary` — controlled values for the catalog's facets (organism, assay, disease,
  access …) with synonym tables you can extend. ``"scRNA-seq; CROP-seq"`` becomes
  ``["scRNA-seq", "CROP-seq"]`` and ``"10x 3' (CRISPRi)"`` yields a platform, an assay and a
  perturbation type.
* :class:`GeneMap` — one gene, many names. Built from any table that has some of symbol,
  Ensembl, Entrez, UniProt, aliases and mouse orthologs (an HGNC dump, a 10x ``features.tsv``,
  or the ``Nodes`` sheet of a project workbook), it maps identifier lists into one key space
  and aligns the feature axes of several matrices with a loss report.
* :class:`ColumnMap` — canonical observation columns (``donor``, ``condition``,
  ``perturbation`` …) guessed from whatever the file called them.

:class:`Harmonizer` holds the three and produces the provenance record the orchestrator stores
beside a built dataset.
"""
from __future__ import annotations

import csv
import gzip
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from .catalog import FACETS

__all__ = ["Vocabulary", "GeneMap", "Alignment", "ColumnMap", "Harmonizer",
           "ASSAY_LABELS", "PERTURBATION_TYPES", "OBS_COLUMNS", "id_type"]


# ---------------------------------------------------------------------------
# Vocabulary — facet values
# ---------------------------------------------------------------------------

#: Canonical assay labels. Keys are the labels; values are lower-case synonyms that identify
#: them inside free text (matched as substrings after normalisation).
ASSAY_LABELS: dict[str, tuple] = {
    "scRNA-seq":            ("scrna-seq", "scrna seq", "scrnaseq", "single-cell rna", "single cell rna",
                             "10x 3'", "10x 5'", "drop-seq", "smart-seq", "sci-rna-seq", "10x scrna"),
    "snRNA-seq":            ("snrna-seq", "snrna seq", "snrnaseq", "single-nucleus rna",
                             "single nucleus rna"),
    "bulk RNA-seq":         ("bulk rna-seq", "bulk rna seq", "rna-seq (bulk)", "bulk transcriptom"),
    "scATAC-seq":           ("scatac-seq", "scatac seq", "scatacseq", "single-cell atac", "sci-atac"),
    "snATAC-seq":           ("snatac-seq", "snatac seq", "snatacseq", "single-nucleus atac"),
    "multiome RNA+ATAC":    ("multiome", "rna+atac", "rna + atac", "share-seq", "paired-seq"),
    "CITE-seq":             ("cite-seq", "citeseq", "totalseq", "adt"),
    "ECCITE-seq":           ("eccite-seq", "eccite"),
    "Perturb-seq":          ("perturb-seq", "perturbseq"),
    "CROP-seq":             ("crop-seq", "cropseq"),
    "Perturb-CITE-seq":     ("perturb-cite-seq", "perturb-cite"),
    "TAP-seq":              ("tap-seq", "tapseq"),
    "SCITO-seq":            ("scito-seq",),
    "pooled CRISPR screen": ("pooled crispr", "crispr screen", "genome-wide crispr", "crispr knockout screen",
                             "crispr-cas9 knockout library", "dropout screen", "facs-based", "facs sort"),
    "arrayed CRISPR":       ("arrayed crispr",),
    "spatial transcriptomics": ("spatial transcriptomics", "spatial rna"),
    "Stereo-seq":           ("stereo-seq", "stereoseq", "stomics"),
    "Visium":               ("visium",),
    "Xenium":               ("xenium",),
    "MERFISH":              ("merfish",),
    "CosMx":                ("cosmx",),
    "CODEX":                ("codex",),
    "MALDI-MSI":            ("maldi", "imaging mass spec"),
    "spatial proteomics":   ("spatial proteomics",),
    "WES":                  ("wes", "whole-exome", "whole exome"),
    "WGS":                  ("wgs", "whole-genome sequencing", "whole genome sequencing"),
    "proteomics":           ("proteomics", "mass spectrometry", "lc-ms"),
    "metabolomics":         ("metabolomics",),
    "lipidomics":           ("lipidomics",),
    "immunopeptidomics":    ("immunopeptidomics",),
    "methylation":          ("methylation", "bisulfite", "epic array"),
    "ChIP-seq":             ("chip-seq", "chipseq"),
    "CUT&Tag":              ("cut&tag", "cut&run"),
    "MRI":                  ("mri", "t1w", "t2w", "flair", "dwi", "nifti", "dicom"),
    "histology":            ("h&e", "histology", "whole-slide", "wsi"),
    "microarray":           ("microarray", "affymetrix", "illumina beadchip"),
    "shotgun metagenomics": ("shotgun metagenom", "metagenomic"),
    "16S":                  ("16s",),
    "L1000":                ("l1000",),
}

#: How the genes were perturbed, from free text.
PERTURBATION_TYPES: dict[str, tuple] = {
    "CRISPRko":     ("crisprko", "crispr-ko", "crispr ko", "knockout", "cas9 ko", "crisprn", "cas9 knockout",
                     "dropout"),
    "CRISPRi":      ("crispri", "crispr-i", "crispr interference", "dcas9-krab", "krab"),
    "CRISPRa":      ("crispra", "crispr-a", "crispr activation", "vp64", "sam activation"),
    "base editing": ("base edit", "base-edit", "abe8", "be4"),
    "prime editing": ("prime edit",),
    "ORF overexpression": ("orf", "overexpression", "tf-orf", "open reading frame"),
    "shRNA/siRNA":  ("shrna", "sirna", "knockdown"),
    "chemical":     ("small molecule", "drug", "compound", "chemical perturbation", "sci-plex"),
}

_ORGANISMS = {
    "Homo sapiens":  ("homo sapiens", "human", "hsapiens", "hs", "h. sapiens", "9606", "grch38", "grch37",
                      "hg38", "hg19"),
    "Mus musculus":  ("mus musculus", "mouse", "mmusculus", "mm", "m. musculus", "10090", "murine", "mm10",
                      "mm39", "grcm38", "grcm39"),
    "Rattus norvegicus": ("rattus", "rat", "10116"),
    "Danio rerio":   ("zebrafish", "danio"),
    "Drosophila melanogaster": ("drosophila", "fly"),
    "Macaca mulatta": ("macaque", "rhesus"),
}

_ACCESS = {
    "controlled": ("controlled", "dbgap", "ega", "duos", "managed access", "dac", "authorized access",
                   "restricted"),
    "on request": ("on request", "upon request", "by request", "contact the authors"),
    "registered": ("registration", "account required", "login required", "free account"),
    "open":       ("open", "public", "publicly", "free", "freely", "cc-by", "cc0", "geo", "zenodo", "figshare",
                   "anonymous"),
}

#: Synonyms longer than four characters that must still match as whole words: ``public`` is not
#: a reason to call a dataset open when the text says ``publication``.
_WHOLE_WORD = frozenset({"public"})

#: Words that describe an *effect*, not a technology. ``CRISPRi knockdown`` is CRISPRi, however
#: long ``knockdown`` is; these only decide a perturbation type when nothing names a technology.
_GENERIC_PERTURBATION = frozenset({"knockout", "knockdown", "dropout", "overexpression", "drug", "compound",
                                   "small molecule", "chemical perturbation"})
#: ``CRISPRa-style overexpression`` says what the library resembles, not what it is.
_STYLE_RX = re.compile(r"crispr[a-z]*[\s-](?:style|like)\b")

_DISEASE = {
    "glioblastoma": ("glioblastoma", "gbm"),
    "glioma": ("glioma", "lgg", "astrocytoma", "oligodendroglioma"),
    "colorectal cancer": ("colorectal", "crc", "colon cancer", "colon adenocarcinoma", "coad"),
    "lung adenocarcinoma": ("lung adenocarcinoma", "luad"),
    "lung squamous cell carcinoma": ("lusc", "lung squamous"),
    "non-small cell lung cancer": ("nsclc", "non-small cell"),
    "small cell lung cancer": ("sclc", "small cell lung"),
    "melanoma": ("melanoma",),
    "multiple sclerosis": ("multiple sclerosis", "ms", "ms lesion", "ms patients"),
    "T-cell leukaemia": ("t-cell leukaemia", "t-cell leukemia", "t-all", "jurkat"),
    "chronic myeloid leukaemia": ("cml", "k562"),
    "hepatocellular carcinoma": ("hepatocellular", "hcc", "hepg2"),
    "gastric cancer": ("gastric",),
    "breast cancer": ("breast cancer", "breast carcinoma"),
    "pancreatic cancer": ("pancreatic", "pdac"),
    "healthy": ("healthy", "normal", "control", "non-diseased"),
    "pan-cancer": ("pan-cancer", "pancancer", "various cancers", "multiple cancer"),
}

_CONTEXT = {
    "cell line": ("cell line", "cell lines", "immortalised", "immortalized", "-line resource"),
    "organoid": ("organoid",),
    "iPSC/ESC-derived": ("ipsc", "esc-derived", "embryonic stem", "induced pluripotent"),
    "primary culture": ("patient-derived", "primary culture", "co-culture", "primary cells", "pdx"),
    "tissue": ("tissue", "biopsy", "resection", "section", "post-mortem", "postmortem", "surgical"),
    "blood": ("pbmc", "blood", "plasma", "serum"),
    "in vivo": ("in vivo", "xenograft", "mouse model"),
    "resource": ("multi-line resource", "resource", "database", "atlas", "meta-layer", "compendium",
                 "repository"),
}


_WS = re.compile(r"\s+")
_SEP = re.compile(r"\s*[;|\n]\s*|\s*,\s*(?![^()]*\))")


def _missing(v) -> bool:
    """None, an empty string, or a NaN/NA/NaT of any array library — a cell with nothing in it."""
    if v is None or (isinstance(v, str) and v == ""):
        return True
    if isinstance(v, float):
        return v != v
    return type(v).__name__ in ("NAType", "NaTType")


def _norm(text) -> str:
    """Lower-case, collapse whitespace, strip decoration — the comparison form."""
    if text is None:
        return ""
    s = str(text).lower().replace("’", "'").replace("’", "'")
    return _WS.sub(" ", s).strip()


def _split(text) -> list:
    """Split a multi-valued cell on the separators people actually use."""
    if text is None:
        return []
    if isinstance(text, (list, tuple, set)):
        return [str(x).strip() for x in text if str(x).strip()]
    t = str(text)
    if ";" not in t and "," not in t and "|" not in t and "\n" not in t:   # the common case
        t = t.strip()
        return [t] if t else []
    return [p.strip() for p in _SEP.split(t) if p and p.strip()]


class Vocabulary:
    """Controlled values for the facets, with synonyms you can extend.

        v = Vocabulary()
        v.assays("scRNA-seq; CROP-seq; 10x 3' (CRISPRi)")   # ['scRNA-seq', 'CROP-seq']
        v.organism("human")                                   # 'Homo sapiens'
        v.add("assay", "TAP-seq", ["targeted perturb-seq"])
        v.facets_of({"species": "human", "modalities": "scRNA-seq", "disease_state": "GBM"})
    """

    def __init__(self):
        self.tables: dict[str, dict[str, tuple]] = {
            "assay": {k: tuple(v) for k, v in ASSAY_LABELS.items()},
            "perturbation": {k: tuple(v) for k, v in PERTURBATION_TYPES.items()},
            "organism": {k: tuple(v) for k, v in _ORGANISMS.items()},
            "access": {k: tuple(v) for k, v in _ACCESS.items()},
            "disease": {k: tuple(v) for k, v in _DISEASE.items()},
            "context": {k: tuple(v) for k, v in _CONTEXT.items()},
        }
        self._compiled: dict = {}

    # ---- extension ----
    def add(self, table: str, label: str, synonyms: Sequence[str]) -> "Vocabulary":
        t = self.tables.setdefault(table, {})
        t[label] = tuple(t.get(label, ())) + tuple(_norm(s) for s in synonyms)
        self._compiled.pop(table, None)
        return self

    def labels(self, table: str) -> list:
        return list(self.tables[table])

    # ---- matching ----
    @staticmethod
    def _alternative(syn: str) -> str:
        """Synonyms match at a token boundary on the left. Short ones (≤4 characters: 'wes',
        'mri', 'adt', 'ko') must also end at a boundary — optionally pluralised, from three
        characters up ('ms' is only ever 'ms', never 'mss') — so 'wes' does not fire inside
        'western'; longer ones may continue ('metagenom' → metagenomics), except the few listed
        in `_WHOLE_WORD` ('public' must not fire inside 'publication')."""
        body = re.escape(syn)
        if len(syn) <= 4 or syn in _WHOLE_WORD:
            return body + (r"(?:e?s)?" if len(syn) >= 3 else "") + r"(?=$|[^a-z0-9])"
        return body

    def _regex(self, table: str):
        """One compiled alternation per table, longest synonym first, plus synonym → label.

        A single `finditer` pass replaces one regex search per synonym per call — the
        difference between a millisecond and a second per row on a 300-synonym table."""
        hit = self._compiled.get(table)
        if hit is not None:
            return hit
        lookup: dict[str, str] = {}
        for label, syns in self.tables[table].items():
            for syn in syns:
                if syn and syn not in lookup:
                    lookup[syn] = label
        alts = [self._alternative(syn) for syn in sorted(lookup, key=len, reverse=True)]
        rx = re.compile(r"(?:^|(?<=[^a-z0-9]))(?:" + "|".join(alts) + ")") if alts else None
        self._compiled[table] = (rx, lookup)
        return rx, lookup

    def _matches(self, table: str, s: str) -> list:
        """(start, synonym, label) for every synonym occurrence in an already normalised text."""
        rx, lookup = self._regex(table)
        if rx is None:
            return []
        out = []
        for m in rx.finditer(s):
            syn = m.group(0)
            label = lookup.get(syn)
            if label is None:                                  # pluralised short synonym
                for cut in (1, 2):
                    label = lookup.get(syn[:-cut])
                    if label is not None:
                        syn = syn[:-cut]
                        break
            if label is not None:
                out.append((m.start(), syn, label))
        return out

    def _hits(self, table: str, text) -> list:
        """(start, synonym length, label) for every label whose synonym occurs in the text."""
        s = _norm(text)
        if not s:
            return []
        best: dict[str, tuple] = {}
        for start, syn, label in self._matches(table, s):
            cur = best.get(label)
            if cur is None or len(syn) > cur[1]:
                best[label] = (start, len(syn), label)
        return list(best.values())

    def _all(self, table: str, text) -> list:
        """Labels present in the text, longest matching synonym first, no duplicates."""
        return [label for _, _, label in sorted(self._hits(table, text), key=lambda x: -x[1])]

    def assays(self, text) -> list:
        """All canonical assay labels named in a cell, in the order they appear."""
        return [label for _, _, label in sorted(self._hits("assay", text))]

    def primary_assay(self, text) -> str:
        a = self.assays(text)
        # The measurement, not the perturbation wrapper: CROP-seq *is* scRNA-seq with guides.
        for label in a:
            if label not in ("Perturb-seq", "CROP-seq", "Perturb-CITE-seq", "TAP-seq",
                             "ECCITE-seq", "pooled CRISPR screen", "arrayed CRISPR"):
                return label
        return a[0] if a else ""

    def perturbation(self, text) -> str:
        """One perturbation type. A technology named in the text (CRISPRi, shRNA, TF-ORF …)
        beats a word that only describes the effect (knockdown, knockout, overexpression …), so
        ``"CRISPRi knockdown"`` is CRISPRi; within a tier the longest synonym wins."""
        s = _STYLE_RX.sub(" ", _norm(text))
        if not s:
            return ""
        tiers: tuple = ({}, {})                               # technologies, effect words
        for start, syn, label in self._matches("perturbation", s):
            best = tiers[1 if syn in _GENERIC_PERTURBATION else 0]
            if label not in best or len(syn) > best[label]:
                best[label] = len(syn)
        for best in tiers:
            if best:
                return sorted(best, key=lambda lab: -best[lab])[0]
        return ""

    def organisms(self, text) -> list:
        """Every organism named in a cell, in the order they appear: ``"human; mouse"`` →
        ``['Homo sapiens', 'Mus musculus']``. :meth:`organism` is the first of them."""
        return [label for _, _, label in sorted(self._hits("organism", text))]

    def organism(self, text) -> str:
        hits = self.organisms(text)
        return hits[0] if hits else (str(text).strip() if text else "")

    def access(self, text) -> str:
        """One access class. Controlled beats open: 'open matrices, raw reads under dbGaP'
        still means something is behind a DAC, and the note keeps the detail."""
        hits = self._all("access", text)
        for cls in ("controlled", "on request", "registered", "open"):
            if cls in hits:
                return cls
        return ""

    def disease(self, text) -> str:
        """One disease label. ``healthy`` ('control', 'normal') only counts when nothing else
        matched — ``"MS vs control"`` is multiple sclerosis."""
        hits = self._all("disease", text)
        if len(hits) > 1 and "healthy" in hits:
            hits = [h for h in hits if h != "healthy"]
        return hits[0] if hits else (str(text).strip() if text else "")

    def context(self, text) -> str:
        hits = self._all("context", text)
        return hits[0] if hits else ""

    def facets_of(self, row: dict, *, organism_keys=("species", "organism", "taxon"),
                  tissue_keys=("tissue_or_cell_type", "tissue", "cell_type", "cell type", "context_type",
                               "cell types / context", "cell lines / relevance to crc & lung"),
                  assay_keys=("modalities", "modality", "assay", "protocols used", "sequences_modalities",
                              "sequencing approach", "assay / data type", "readout"),
                  disease_keys=("disease_state", "disease", "cancer type", "cancer focus", "subtype",
                                "condition"),
                  platform_keys=("platform", "instrument", "chemistry"),
                  condition_keys=("perturbation_type", "perturbation", "treatment", "condition"),
                  sample_keys=("n_donors_or_samples", "n_samples", "subjects_scans", "samples"),
                  donor_keys=("donor", "donor_id", "donor id", "patient_id", "patient id", "subject_id",
                              "subject id", "participant_id", "participant id", "individual_id"),
                  columns: Optional[dict] = None) -> dict:
        """Facets in the catalog's fixed key set, from a description row with any column names.

        `columns` pins the column to read per facet, by the names `read_descriptions(columns=)`
        uses (``species``, ``tissue``, ``modalities``, ``disease``, ``platform``,
        ``perturbation_type``, ``n_samples``, ``donor``) or the facet names themselves; a pinned
        column is tried before the `*_keys` defaults."""
        low = {_norm(k): v for k, v in row.items() if k is not None}
        pins = columns or {}

        def first(keys, facet):
            for n in _PIN_NAMES[facet]:
                if pins.get(n):
                    keys = (pins[n], *keys)
                    break
            for k in keys:
                v = low.get(_norm(k))
                if not _missing(v):
                    return v
            return ""

        assay_text = first(assay_keys, "assay")
        condition_text = first(condition_keys, "condition")
        out = {
            "organism": self.organism(first(organism_keys, "organism")),
            "tissue": str(first(tissue_keys, "tissue")).strip(),
            "assay": self.primary_assay(assay_text) or str(assay_text).strip(),
            "disease": self.disease(first(disease_keys, "disease")),
            "sample": str(first(sample_keys, "sample")).strip(),
            "donor": str(first(donor_keys, "donor")).strip(),
            "condition": self.perturbation(condition_text) or str(condition_text).strip(),
            "platform": str(first(platform_keys, "platform")).strip(),
        }
        return {k: v for k, v in out.items() if k in FACETS}


#: Names a column pin may go by, per facet (`read_descriptions(columns=)` vocabulary first).
_PIN_NAMES = {"organism": ("organism", "species", "taxon"), "tissue": ("tissue",),
              "assay": ("assay", "modalities", "modality"), "disease": ("disease",),
              "sample": ("sample", "n_samples"), "donor": ("donor",),
              "condition": ("condition", "perturbation_type", "perturbation"), "platform": ("platform",)}


# ---------------------------------------------------------------------------
# GeneMap — identifiers
# ---------------------------------------------------------------------------

_ENSG = re.compile(r"^ENS[A-Z]*G\d{11}(\.\d+)?$")
_ENTREZ = re.compile(r"^\d{1,9}$")
_UNIPROT = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?$")
_HGNC = re.compile(r"^HGNC:\d+$", re.I)
_CD = re.compile(r"^CDw?\d{1,3}[a-z]?$", re.I)


def id_type(x: str) -> str:
    """'ensembl' | 'entrez' | 'uniprot' | 'hgnc' | 'cd' | 'symbol' — what kind of name this is.

    A *shape* test only: ``CD8A`` is 'cd', ``P2RY12`` and ``B3GNT2`` are 'uniprot', and all
    three are also valid gene symbols. Nothing that has to decide whether a name may stand for
    a symbol should lean on this (see :meth:`GeneMap.canonical`)."""
    s = str(x).strip()
    if _ENSG.match(s):
        return "ensembl"
    if _HGNC.match(s):
        return "hgnc"
    if _ENTREZ.match(s):
        return "entrez"
    if _CD.match(s):
        return "cd"
    if _UNIPROT.match(s) and not s.isalpha():
        return "uniprot"
    return "symbol"


_ID_FIELDS = ("ensembl", "entrez", "uniprot", "hgnc")
_JUNK_NAMES = frozenset({"nan", "none", "null", "n/a", "na", "nil"})


def _t(x) -> str:
    return x.strip() if isinstance(x, str) else ("" if x is None else str(x).strip())


def _strip_version(x) -> str:
    """``ENSG00000141510.12`` → ``ENSG00000141510``: the suffix names a release of the
    annotation, not a different gene. Anything that is not an Ensembl gene id comes back as
    it was (stripped of whitespace)."""
    s = _t(x)
    if s[:3] == "ENS" and "." in s and _ENSG.match(s):
        return s.split(".", 1)[0]
    return s


def _key(name) -> str:
    """The lookup form of a name: Ensembl versions dropped, case folded."""
    return _strip_version(name).upper()


def _symbol_like(name: str) -> bool:
    """May this name stand for a gene symbol when the map has never heard of it? Anything that
    is not junk and not shaped like an Ensembl, Entrez or HGNC id. Junk is an empty string, one
    with whitespace or a colon in it, a pandas ``Unnamed: 0`` index column, or a stand-in for
    nothing (``nan``, ``None``). CD designations (``CD8A``) and UniProt-looking symbols
    (``P2RY12``, ``B3GNT2``) are symbols; UniProt typing is only trusted when the map itself
    says the name is a UniProt id."""
    s = name.strip()
    if not s or ":" in s or _WS.search(s):
        return False
    low = s.lower()
    if low in _JUNK_NAMES or low.startswith("unnamed"):
        return False
    if s[0].isdigit():
        return not _ENTREZ.match(s)
    return not (s[:3].upper() == "ENS" and _ENSG.match(s.upper()))


@dataclass
class GeneRecord:
    symbol: str
    ensembl: str = ""
    entrez: str = ""
    uniprot: str = ""
    hgnc: str = ""
    aliases: tuple = ()
    ortholog: str = ""          # mouse symbol, when the table had one
    ortholog_status: str = ""
    extra: dict = field(default_factory=dict)

    def names(self) -> list:
        out = [self.symbol, self.ensembl, self.entrez, self.uniprot, self.hgnc, *self.aliases]
        return [n for n in out if n]


def _clean(rec: GeneRecord) -> GeneRecord:
    """The map's own copy of a record: ids as stripped strings, Ensembl versions dropped
    wherever an id is stored (ids and aliases alike), aliases de-duplicated and never repeating
    the record's own names. The caller's object is left alone."""
    sym = _t(rec.symbol)
    ens, ent, uni, hgnc = _strip_version(rec.ensembl), _t(rec.entrez), _t(rec.uniprot), _t(rec.hgnc)
    seen = {sym.upper()}
    seen.update(v.upper() for v in (ens, ent, uni, hgnc) if v)
    aliases = []
    for a in rec.aliases or ():
        a = _strip_version(a)
        if a and a.upper() not in seen:
            seen.add(a.upper())
            aliases.append(a)
    return GeneRecord(symbol=sym, ensembl=ens, entrez=ent, uniprot=uni, hgnc=hgnc, aliases=tuple(aliases),
                      ortholog=_t(rec.ortholog), ortholog_status=_t(rec.ortholog_status),
                      extra=dict(rec.extra) if rec.extra else {})


def _compatible(a: GeneRecord, b: GeneRecord) -> bool:
    """Can two records with one symbol be one gene? Not when they carry different ids of the
    same kind that neither of them already lists (two Ensembl genes sharing a symbol, or a
    human and a mouse gene that differ only by case)."""
    an, bn = {_key(n) for n in a.names()}, {_key(n) for n in b.names()}
    for k in _ID_FIELDS:
        x, y = getattr(a, k), getattr(b, k)
        if x and y and _key(x) != _key(y) and _key(y) not in an and _key(x) not in bn:
            return False
    return True


def _merge_into(a: GeneRecord, b: GeneRecord) -> None:
    """Union b into a: a's ids win, b fills what a lacks, every other name of b becomes an
    alias of a (first-seen order)."""
    for k in _ID_FIELDS:
        if not getattr(a, k) and getattr(b, k):
            setattr(a, k, getattr(b, k))
    have = {_key(n) for n in a.names()}
    fresh = []
    for n in b.names():
        if _key(n) not in have:
            have.add(_key(n))
            fresh.append(n)
    if fresh:
        a.aliases = tuple(a.aliases) + tuple(fresh)
    if not a.ortholog and b.ortholog:
        a.ortholog, a.ortholog_status = b.ortholog, b.ortholog_status or a.ortholog_status
    elif not a.ortholog_status and b.ortholog_status:
        a.ortholog_status = b.ortholog_status
    for k, v in b.extra.items():
        a.extra.setdefault(k, v)


@dataclass
class Alignment:
    """The shared feature axis of several matrices, and what each one lost to get there.

    `report[piece]` counts every input name exactly once: ``n_in = n_mapped + n_duplicate +
    n_unmapped``. ``n_mapped`` are the distinct ids kept (the first occurrence wins),
    ``n_duplicate`` the later names that mapped to an id already kept (the same feature twice,
    or an alias beside its symbol — their values are *not* summed), ``n_unmapped`` the names
    with no id in the key space (junk such as ``Unnamed: 0`` included). `panel_missing` lists the
    panel entries that map to nothing in the key space and so are not on the axis at all
    (repeated on every piece's report); `panel_duplicates` the entries that added nothing new."""
    key: str
    index: list                          # canonical ids, in output order
    positions: dict                      # piece -> np.ndarray of source positions (-1 = absent)
    report: dict                         # piece -> {n_in, n_mapped, n_duplicate, n_kept, n_unmapped, examples, ...}
    how: str = "intersection"
    panel_missing: list = field(default_factory=list)
    panel_duplicates: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.index)

    def take(self, piece: str, X, axis: int = 1):
        """Reorder/select the feature axis of a block to the shared index (missing → zero)."""
        import scipy.sparse as sp
        pos = self.positions[piece]
        present = pos >= 0
        if sp.issparse(X):
            X = X.tocsr() if axis == 1 else X.tocsc()
            out_shape = (X.shape[0], len(pos)) if axis == 1 else (len(pos), X.shape[1])
            sel = X[:, pos[present]] if axis == 1 else X[pos[present], :]
            if present.all():
                return sel
            # scatter the present columns into the full shared axis
            if axis == 1:
                cols = np.flatnonzero(present)
                coo = sel.tocoo()
                return sp.csr_matrix((coo.data, (coo.row, cols[coo.col])), shape=out_shape)
            rows = np.flatnonzero(present)
            coo = sel.tocoo()
            return sp.csc_matrix((coo.data, (rows[coo.row], coo.col)), shape=out_shape)
        X = np.asarray(X)
        if axis == 1:
            out = np.zeros((X.shape[0], len(pos)), dtype=X.dtype)
            out[:, present] = X[:, pos[present]]
        else:
            out = np.zeros((len(pos), X.shape[1]), dtype=X.dtype)
            out[present, :] = X[pos[present], :]
        return out

    def explain(self) -> str:
        lines = [f"shared axis: {self.n:,} features keyed by {self.key} ({self.how})"]
        if self.panel_missing:
            lines.append(f"  panel entries with no {self.key} id, left off the axis ({len(self.panel_missing)}): "
                         + ", ".join(self.panel_missing[:6]) + (" …" if len(self.panel_missing) > 6 else ""))
        for p, r in self.report.items():
            ex = ", ".join(r["examples"][:4])
            dup = r.get("n_duplicate", 0)
            lines.append(f"  {p:<28} {r['n_in']:>7,} in  {r['n_mapped']:>7,} mapped  "
                         f"{r['n_kept']:>7,} kept  {r['n_unmapped']:>6,} unmapped"
                         + (f"  {dup:,} duplicate" if dup else "")
                         + (f"  e.g. {ex}" if ex else ""))
        return "\n".join(lines)


class GeneMap:
    """One gene, many names — and a way to choose one.

    ::

        gm = GeneMap.from_hgnc("hgnc_complete_set.txt")
        gm = GeneMap.from_table(rows, symbol="primary_gene", ensembl="ensembl_human",
                                members="human_genes", aliases=("node_id",), group="node_id")
        gm.to(["CD274", "PD-L1", "ENSG00000120217"], key="ensembl")
        aln = gm.align({"a": var_a, "b": var_b}, key="ensembl", how="intersection")

    One record per gene: records added with the same symbol are merged (ids and aliases
    unioned) unless they carry different ids of one kind, in which case they stay apart as
    different genes that happen to share a name. Ensembl version suffixes are never stored.
    """

    KEYS = ("symbol", "ensembl", "entrez", "uniprot", "hgnc")

    def __init__(self, records: Iterable[GeneRecord] = ()):
        self.records: list[GeneRecord] = []
        self._index: dict[str, list] = {}
        self._by_symbol: dict[str, list] = {}
        for r in records:
            self.add(r)

    # ---- building ----
    def add(self, rec: GeneRecord) -> "GeneMap":
        """Add a record (the map keeps its own copy). A record whose symbol the map already has
        is merged into it when the ids do not contradict each other."""
        rec = _clean(rec)
        sym = rec.symbol.upper()
        for i in self._by_symbol.get(sym, ()) if sym else ():
            cur = self.records[i]
            if _compatible(cur, rec):
                known = {_key(n) for n in cur.names()}
                _merge_into(cur, rec)
                for n in cur.names():
                    k = _key(n)
                    if k not in known:
                        known.add(k)
                        self._index.setdefault(k, []).append(i)
                return self
        i = len(self.records)
        self.records.append(rec)
        if sym:
            self._by_symbol.setdefault(sym, []).append(i)
        done: set = set()
        for n in rec.names():
            k = _key(n)
            if k not in done:
                done.add(k)
                self._index.setdefault(k, []).append(i)
        return self

    @staticmethod
    def _k(name: str) -> str:
        return _key(name)                         # version suffix is not identity

    @classmethod
    def from_table(cls, rows: Iterable[dict], *, symbol: str = "symbol", ensembl: str = "",
                   entrez: str = "", uniprot: str = "", hgnc: str = "",
                   aliases: Sequence[str] = (), members: str = "", ortholog: str = "",
                   ortholog_status: str = "", group: str = "", split_aliases: bool = True) -> "GeneMap":
        """From any table of dicts. Column names are the caller's; multi-valued cells are split
        on the usual separators.

        A row may describe a *complex* — CD3 is CD3D, CD3E and CD3G — when `members` names a
        column listing every gene. Each member becomes its own record carrying the row's
        aliases, and the id columns are split positionally to match. Resolving the complex's
        name then hits several records: :meth:`members` returns them all, :meth:`ambiguous`
        says so, :meth:`expand` lists them all and :meth:`resolve` returns the first.

        On a row that describes one gene, an id cell with several values (HGNC's
        ``uniprot_ids = "P04745|P0DUB6"``) keeps the first as the gene's id and the rest as
        aliases; on a complex, id lists that do not pair up with the members are left blank
        rather than guessed.

        `group` names a column whose value labels the row as a whole — a CD designation, a
        pathway, a panel — and becomes the ``"group"`` key space: ``canonical("CD3E", "group")``
        is then ``"CD3"``.
        """
        gm = cls()
        for row in rows:
            sym = str(row.get(symbol) or "").strip()
            if not sym:
                continue
            al = []
            for a in aliases:
                v = row.get(a)
                al.extend(_split(v) if split_aliases else ([str(v)] if v else []))
            genes = _split(row.get(members)) if members else []
            if not genes:
                genes = [sym]
            cols = {}
            for k, c in (("ensembl", ensembl), ("entrez", entrez), ("uniprot", uniprot), ("hgnc", hgnc)):
                vals = _split(row.get(c)) if c else []
                if k == "entrez":
                    vals = [v.split(".")[0] for v in vals]
                cols[k] = vals
            grp = str(row.get(group) or "").strip() if group else (sym if len(genes) > 1 else "")
            for i, gsym in enumerate(genes):
                ids = {}
                spare = []
                for k, vals in cols.items():
                    if len(vals) == len(genes):
                        ids[k] = vals[i]
                    elif len(genes) == 1 and len(vals) > 1:
                        ids[k] = vals[0]
                        spare.extend(vals[1:])             # more than one id for the one gene
                    else:
                        ids[k] = ""                    # cannot pair positionally; do not guess
                al_i = [x for x in dict.fromkeys([*(a.strip() for a in al), *spare]) if x and x != gsym]
                gm.add(GeneRecord(
                    symbol=gsym, ensembl=ids.get("ensembl", ""), entrez=ids.get("entrez", ""),
                    uniprot=ids.get("uniprot", ""), hgnc=ids.get("hgnc", ""), aliases=tuple(al_i),
                    ortholog=str(row.get(ortholog) or "").strip() if ortholog and len(genes) == 1 else "",
                    ortholog_status=str(row.get(ortholog_status) or "").strip() if ortholog_status else "",
                    extra={"group": grp} if grp else {},
                ))
        return gm

    @classmethod
    def from_hgnc(cls, path: str) -> "GeneMap":
        """The HGNC complete set (TSV, optionally gzipped) — the reference everyone else copies."""
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        return cls.from_table(rows, symbol="symbol", ensembl="ensembl_gene_id", entrez="entrez_id",
                              uniprot="uniprot_ids", hgnc="hgnc_id",
                              aliases=("alias_symbol", "prev_symbol"))

    @classmethod
    def from_features(cls, ids: Sequence[str], names: Sequence[str]) -> "GeneMap":
        """From a 10x features.tsv pair (Ensembl id, symbol) — often all a matrix ships with."""
        gm = cls()
        for i, n in zip(ids, names):
            i, n = str(i).strip(), str(n).strip()
            gm.add(GeneRecord(symbol=n or i, ensembl=_strip_version(i) if _ENSG.match(i) else ""))
        return gm

    def merge(self, other: "GeneMap") -> "GeneMap":
        for r in other.records:
            self.add(r)
        return self

    # ---- lookup ----
    def resolve(self, name: str, key: Optional[str] = None) -> Optional[GeneRecord]:
        """The one record a name stands for. When several records answer to it, the one whose
        *symbol* is the name wins (an alias may be another gene's symbol); among equals, with
        `key` given, the first that actually has that id — not whichever came first in the
        table. Without a symbol owner the same preference applies among the alias owners."""
        hits = self._index.get(self._k(name))
        if not hits:
            return None
        if len(hits) == 1:
            return self.records[hits[0]]
        recs = [self.records[i] for i in hits]
        q = str(name).strip().upper()
        pool = [r for r in recs if r.symbol.upper() == q] or recs
        if key:
            for r in pool:
                if self._get(r, key):
                    return r
        return pool[0]

    def ambiguous(self, name: str) -> bool:
        return len(self._index.get(self._k(name), [])) > 1

    def members(self, name: str) -> list:
        """Every record a name resolves to — one for a gene, several for a complex or an alias
        shared between genes."""
        return [self.records[i] for i in self._index.get(self._k(name), [])]

    def _records_for(self, name: str) -> list:
        """The records a name *means*: those whose symbol it is, if any (an alias that happens to
        be another gene's symbol does not drag that other gene in), otherwise every record that
        answers to it — a complex, or an alias shared between genes."""
        recs = self.members(name)
        if len(recs) > 1:
            q = str(name).strip().upper()
            own = [r for r in recs if r.symbol.upper() == q]
            if own:
                return own
        return recs

    def _expand_one(self, name: str, key: str) -> list:
        ids = [v for v in (self._get(r, key) for r in self._records_for(name)) if v]
        if not ids:
            c = self.canonical(name, key)
            ids = [c] if c else []
        return ids

    def expand(self, names: Sequence[str], key: str = "symbol") -> list:
        """Names → key-space ids with complexes expanded to their members, duplicates dropped.
        A name that is some gene's symbol means that gene only; a name that is a complex's
        alias means every member."""
        out: list = []
        seen: set = set()
        for n in names:
            for c in self._expand_one(n, key):
                if c not in seen:
                    seen.add(c)
                    out.append(c)
        return out

    @staticmethod
    def _get(r: "GeneRecord", key: str) -> str:
        return r.extra.get("group", "") if key == "group" else getattr(r, key, "")

    def to(self, names: Sequence[str], key: str = "symbol") -> list:
        """Map names into one key space; unknown names come back as None. A complex name maps
        to its first member here (one value per name) — use :meth:`expand` for all of them."""
        if key not in self.KEYS and key not in ("ortholog", "group"):
            raise ValueError(f"key must be one of {self.KEYS}, 'ortholog' or 'group'")
        out = []
        for n in names:
            r = self.resolve(n, key)
            v = self._get(r, key) if r is not None else ""
            out.append(v or None)
        return out

    def canonical(self, name: str, key: str = "symbol") -> str:
        """The key-space id for one name. A name the map does not know falls back to itself when
        it already looks like that kind of id: an Ensembl id absent from a partial table is still
        an Ensembl id, and — for ``key="symbol"`` — anything that is not shaped like an Ensembl,
        Entrez or HGNC id and is not junk (whitespace, a colon, ``Unnamed: 0``) is a symbol,
        whether or not it resembles a CD designation or a UniProt accession (``CD8A``,
        ``P2RY12``)."""
        s = str(name).strip()
        r = self.resolve(s, key)
        if r is not None:
            v = self._get(r, key)
            if v:
                return v
        if key == "symbol":
            return s if _symbol_like(s) else ""
        if key == "ensembl" and _ENSG.match(s.upper()):
            return s.upper().split(".")[0]
        if key == "entrez" and _ENTREZ.match(s):
            return s
        return ""

    def __len__(self) -> int:
        return len(self.records)

    def __contains__(self, name) -> bool:
        return self._k(name) in self._index

    # ---- aligning matrices ----
    def align(self, features: dict, key: str = "symbol", how: str = "intersection",
              panel: Optional[Sequence[str]] = None) -> Alignment:
        """Shared feature axis across several pieces.

        `features` maps piece name -> its feature names (any id type, mixed is fine). `how` is
        ``intersection`` (features every piece has) or ``union`` (zero-filled where absent).
        `panel` restricts the axis to a gene list, in the panel's order — the thing a
        node-centred project actually wants; a panel entry naming a complex contributes every
        member. Nothing is lost silently: see :class:`Alignment` for what the report counts.

        Within a piece the first occurrence of an id wins; later names that map to the same
        id (a repeated column, an alias beside its symbol) are counted as duplicates and
        dropped, not summed.
        """
        if how not in ("intersection", "union"):
            raise ValueError("how must be 'intersection' or 'union'")
        mapped: dict[str, dict] = {}
        report: dict[str, dict] = {}
        for piece, names in features.items():
            pos: dict[str, int] = {}
            unmapped: list = []
            dups: list = []
            n_in = 0
            for i, n in enumerate(names):
                n_in += 1
                c = self.canonical(n, key)
                if not c:
                    unmapped.append(str(n))
                elif c in pos:
                    dups.append(str(n))              # first occurrence wins
                else:
                    pos[c] = i
            mapped[piece] = pos
            report[piece] = {"n_in": n_in, "n_mapped": len(pos), "n_duplicate": len(dups),
                             "n_unmapped": len(unmapped), "examples": unmapped[:8],
                             "duplicate_examples": dups[:8], "n_kept": 0, "panel_missing": []}
        panel_missing: list = []
        panel_dups: list = []
        if panel is not None:
            index = []
            seen_p: set = set()
            for g in panel:
                ids = self._expand_one(g, key)
                if not ids:
                    panel_missing.append(str(g))
                    continue
                fresh = [c for c in dict.fromkeys(ids) if c not in seen_p]
                if not fresh:
                    panel_dups.append(str(g))
                for c in fresh:
                    seen_p.add(c)
                    index.append(c)
        elif how == "intersection":
            sets = [set(m) for m in mapped.values()]
            common = set.intersection(*sets) if sets else set()
            first = next(iter(mapped.values()), {})
            index = [c for c in first if c in common]
        else:
            index = []
            seen = set()
            for m in mapped.values():
                for c in m:
                    if c not in seen:
                        seen.add(c)
                        index.append(c)
        positions = {}
        for piece, m in mapped.items():
            arr = np.array([m.get(c, -1) for c in index], dtype=np.int64)
            positions[piece] = arr
            report[piece]["n_kept"] = int((arr >= 0).sum())
            report[piece]["panel_missing"] = list(panel_missing)
        return Alignment(key=key, index=index, positions=positions, report=report, how=how,
                         panel_missing=panel_missing, panel_duplicates=panel_dups)

    def panel_table(self, panel: Sequence[str], key: str = "symbol", expand: bool = True) -> list:
        """One row per panel gene with every id the map knows — the `var` of a built dataset.
        A panel entry naming a complex gets one row per member (``query`` is the name asked
        for, ``id`` the member's id); with ``expand=False`` it gets the single row
        :meth:`resolve` gives, which is what a table indexed by an axis that is already
        expanded wants."""
        rows = []
        for g in panel:
            q = str(g)
            if not expand:
                r = self.resolve(g, key)
                rows.append(self._table_row(q, r, self.canonical(q, key) or q))
                continue
            recs = self._records_for(g)
            if not recs:
                rows.append(self._table_row(q, None, self.canonical(q, key) or q))
                continue
            for r in recs:
                rid = self._get(r, key)
                if not rid:
                    rid = (self.canonical(q, key) or q) if len(recs) == 1 else r.symbol
                rows.append(self._table_row(q, r, rid))
        return rows

    @staticmethod
    def _table_row(query: str, r: Optional[GeneRecord], rid: str) -> dict:
        return {
            "id": rid,
            "query": query,
            "group": r.extra.get("group", "") if r else "",
            "symbol": r.symbol if r else (query if _symbol_like(query) else ""),
            "ensembl": r.ensembl if r else "",
            "entrez": r.entrez if r else "",
            "uniprot": r.uniprot if r else "",
            "hgnc": r.hgnc if r else "",
            "aliases": ";".join(r.aliases) if r else "",
            "ortholog": r.ortholog if r else "",
            "ortholog_status": r.ortholog_status if r else "",
            "known": r is not None,
        }


# ---------------------------------------------------------------------------
# ColumnMap — observation metadata
# ---------------------------------------------------------------------------

#: Canonical obs columns and what files call them.
OBS_COLUMNS: dict[str, tuple] = {
    "cell_id":       ("cell_id", "cell", "barcode", "barcodes", "obs_names", "_index", "index", "cell_barcode",
                      "cellid", "cell_name"),
    "sample":        ("sample", "sample_id", "sampleid", "library", "library_id", "orig.ident", "gem_group",
                      "lane", "run"),
    "donor":         ("donor", "donor_id", "patient", "patient_id", "subject", "subject_id", "individual",
                      "participant"),
    "condition":     ("condition", "treatment", "stim", "stimulation", "group", "status", "disease_state",
                      "disease"),
    "perturbation":  ("perturbation", "perturbed_gene", "target", "target_gene", "gene_target", "knockout",
                      "ko", "gene", "perturbation_name", "feature_call", "guide_target"),
    "guide":         ("guide", "guide_id", "sgrna", "grna", "guide_identity", "protospacer", "feature_call_id",
                      "sgrna_id"),
    "cell_type":     ("cell_type", "celltype", "cell type", "annotation", "cluster_name", "cell_type_annotation",
                      "leiden_annotation", "type"),
    "batch":         ("batch", "batch_id", "experiment", "exp", "replicate", "rep"),
    "timepoint":     ("timepoint", "time", "time_point", "day", "hour", "hours", "stage"),
    "cluster":       ("cluster", "clusters", "leiden", "louvain", "seurat_clusters"),
    "n_counts":      ("n_counts", "total_counts", "ncounts", "numi", "n_umi", "umi_count", "nCount_RNA"),
    "n_genes":       ("n_genes", "n_genes_by_counts", "ngenes", "nfeature_rna", "nFeature_RNA"),
}


class ColumnMap:
    """Guess which of a file's obs columns carry the canonical fields, and rename them.

        cm = ColumnMap()
        cm.guess(["patient_id", "sgRNA", "leiden", "n_counts"])
        # {'donor': 'patient_id', 'guide': 'sgRNA', 'cluster': 'leiden', 'n_counts': 'n_counts'}
    """

    def __init__(self, extra: Optional[dict] = None):
        self.table = {k: tuple(_norm(x) for x in v) for k, v in OBS_COLUMNS.items()}
        for k, v in (extra or {}).items():
            self.table[k] = tuple(self.table.get(k, ())) + tuple(_norm(x) for x in v)

    def guess(self, columns: Sequence[str], fixed: Optional[dict] = None) -> dict:
        """canonical -> original. `fixed` pins choices the analyst already knows."""
        out = dict(fixed or {})
        cols = [str(c) for c in columns]
        low = {_norm(c): c for c in cols}
        for canon, syns in self.table.items():
            if canon in out:
                continue
            for s in syns:
                if s in low and low[s] not in out.values():
                    out[canon] = low[s]
                    break
        return out

    def apply(self, columns: dict, mapping: dict) -> dict:
        """Rename a dict of columns to canonical names; unmapped columns keep their names."""
        inv = {v: k for k, v in mapping.items()}
        return {inv.get(k, k): v for k, v in columns.items()}


# ---------------------------------------------------------------------------
# Harmonizer — the three together, with a record of what was done
# ---------------------------------------------------------------------------

class Harmonizer:
    """Vocabulary + GeneMap + ColumnMap, plus the provenance they produce.

        h = Harmonizer()
        h.genes = GeneMap.from_table(nodes, symbol="primary_gene", ensembl="ensembl_human")
        h.facets({"species": "human", "modalities": "scRNA-seq; CROP-seq"})
        h.report()
    """

    def __init__(self, vocabulary: Optional[Vocabulary] = None, genes: Optional[GeneMap] = None,
                 columns: Optional[ColumnMap] = None):
        self.vocabulary = vocabulary or Vocabulary()
        self.genes = genes or GeneMap()
        self.columns = columns or ColumnMap()
        self.log: list[dict] = []

    def facets(self, row: dict, columns: Optional[dict] = None) -> dict:
        """Facets of one description row; `columns` pins columns per facet (see
        :meth:`Vocabulary.facets_of`)."""
        return self.vocabulary.facets_of(row, columns=columns)

    def gene_key_for(self, names: Sequence[str]) -> str:
        """The id space most of these names are already in — the cheapest key to harmonise to."""
        kinds = [id_type(n) for n in names[:2000]]
        if not kinds:
            return "symbol"
        counts = {k: kinds.count(k) for k in set(kinds)}
        best = max(counts, key=counts.get)
        return best if best in GeneMap.KEYS else "symbol"

    def align(self, features: dict, key: str = "symbol", how: str = "intersection",
              panel: Optional[Sequence[str]] = None) -> Alignment:
        aln = self.genes.align(features, key=key, how=how, panel=panel)
        self.log.append({"step": "align", "key": key, "how": how, "n": aln.n,
                         "panel": len(panel) if panel is not None else None,
                         "panel_missing": list(aln.panel_missing),
                         "report": aln.report})
        return aln

    def obs_mapping(self, piece: str, columns: Sequence[str], fixed: Optional[dict] = None) -> dict:
        m = self.columns.guess(columns, fixed)
        self.log.append({"step": "obs-columns", "piece": piece, "mapping": m,
                         "unmapped": [c for c in columns if c not in m.values()]})
        return m

    def report(self) -> dict:
        return {
            "vocabulary": {t: len(v) for t, v in self.vocabulary.tables.items()},
            "genes": len(self.genes),
            "obs_columns": list(self.columns.table),
            "log": list(self.log),
        }
