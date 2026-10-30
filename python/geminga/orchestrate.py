"""Building datasets out of *descriptions* of datasets.

The pulsar this library is named after was called Geminga because, for years, there was nothing
at its position in the optical sky — *gh'è minga*, "it is not there", in Milanese. That is the
state of most datasets an analyst works with. They exist as a row in a spreadsheet: a first
author, an accession, a modality column, a note that says "raw reads under dbGaP". The data is
somewhere else, or nowhere yet, and a project is a puzzle assembled from such rows.

This module treats a row as a **piece** and a project as a **puzzle**, and gives four verbs:

* **describe** — read rows from a CSV, a workbook tab, a Parquet file or a list of dicts, in
  whatever columns the author used, and turn each into a :class:`Piece` with the accessions
  pulled out of its free text, its facets harmonised, and the genes it names by role.
* **resolve** — say, for every piece, whether it is *here* (a local file, a vault member, a
  catalog record), *one fetch away* (a free route, streamable or not, and what it would cost),
  *behind a DAC*, *paper only*, or *unresolved* — without transferring a byte.
* **harmonise + annotate** — map facets into the catalog's fixed vocabulary, gene lists into one
  identifier space, observation columns onto canonical names, and record every piece in the
  catalog with those facets; pieces that are not here become ``need-data`` flags in the
  library so later searches find them.
* **build** — execute a :class:`Blueprint`: open the pieces that are local, align their
  feature axes to one key (optionally to a gene *panel*), stream them under the budget, and
  write one :class:`~geminga.handoff.Harvest` with a provenance record naming the row each
  piece came from, the mapping applied, what was lost, and which pieces are holes. A hole
  never fails a build. It is written down, priced, and flagged.

Nothing here downloads. `geminga.acquire` prices routes and builds the commands; the analyst
or their download tool moves the bytes; the orchestrator notices next time it resolves.
"""
from __future__ import annotations

import bisect
import csv
import functools
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from ._core import Budget, human
from .acquire import AcquisitionPlan, classify
from .acquire import plan as acquisition_plan
from .catalog import FACETS
from .harmonize import Alignment, GeneMap, GeneRecord, Harmonizer, _missing, _norm, _split, id_type

__all__ = ["Piece", "Resolution", "Blueprint", "Puzzle", "Coverage", "Orchestrator", "orchestrator",
           "read_descriptions", "find_accessions", "resolve_pieces", "RESOLUTION_STATUSES", "BLUEPRINT_SCHEMA"]

BLUEPRINT_SCHEMA = "geminga.blueprint/1"

#: Resolution statuses, best first. A piece gets the best status any of its accessions earns.
RESOLUTION_STATUSES = ("local", "streamable", "fetchable", "controlled", "manual", "unresolved")

_STATUS_WORDS = {
    "local": "here", "streamable": "one fetch away, streamable", "fetchable": "one fetch away",
    "controlled": "behind a DAC", "manual": "paper only", "unresolved": "unresolved",
    # not resolution statuses: what a *hole* in a built puzzle can be besides the above
    "missing": "file missing", "unreadable": "unreadable",
}


# ---------------------------------------------------------------------------
# Accessions out of free text
# ---------------------------------------------------------------------------

#: An accession is delimited by anything that is not a letter or a digit. ``\b`` would not do:
#: it treats ``_`` as part of a word, so ``GSE12345_RAW.tar`` and ``SRR1448774_1.fastq.gz`` —
#: the form accessions take in file names — would contain no accession.
_L = r"(?<![A-Za-z0-9])"
_R = r"(?![A-Za-z0-9])"

#: Patterns in the order they are tried. Each is (regex, kind, case, hint): the canonical accession,
#: the `geminga.acquire.classify` kind where acquire knows the shape (otherwise a local kind name),
#: how to normalise the case of ids that people type in either ("upper"/"lower"/None), and a
#: lower-case literal the text must contain for the regex to be worth running at all.
_ACC_PATTERNS = [
    (re.compile(_L + r"(GSE\d{3,8})" + _R, re.I), None, "upper", "gse"),
    (re.compile(_L + r"(GSM\d{3,9})" + _R, re.I), None, "upper", "gsm"),
    (re.compile(_L + r"(GPL\d{2,7})" + _R, re.I), "geo_platform", "upper", "gpl"),
    (re.compile(_L + r"([SED]RR\d{5,10})" + _R, re.I), None, "upper", "rr"),
    (re.compile(_L + r"([SED]RX\d{5,10})" + _R, re.I), None, "upper", "rx"),
    (re.compile(_L + r"([SED]RP\d{5,10})" + _R, re.I), None, "upper", "rp"),
    (re.compile(_L + r"(PRJ[END][AB]\d{3,9})" + _R, re.I), None, "upper", "prj"),
    (re.compile(_L + r"(SAM[END][AG]?\d{5,12})" + _R), None, None, "sam"),
    (re.compile(_L + r"(EGA[SD]\d{11})" + _R), None, None, "ega"),
    (re.compile(_L + r"(phs\d{6}(?:\.v\d+(?:\.p\d+)?)?)" + _R, re.I), None, None, "phs"),
    (re.compile(_L + r"(E-[A-Z]{4}-\d{2,7})" + _R), None, None, "e-"),
    (re.compile(_L + r"(CRA\d{6,7})" + _R), None, None, "cra"),
    (re.compile(_L + r"(HRA\d{6,7})" + _R), "gsa_human", None, "hra"),
    (re.compile(_L + r"(CN[PSXRA]\d{7})" + _R), None, None, "cn"),
    (re.compile(_L + r"(STDS\d{7})" + _R), None, None, "stds"),
    (re.compile(_L + r"(SCP\d{3,5})" + _R), "scp", None, "scp"),
    (re.compile(_L + r"(DUOS-\d{6})" + _R), "duos", None, "duos"),
    (re.compile(_L + r"(ds\d{6})" + _R), None, None, "ds"),
    # hosts `acquire` cannot price: named here so that a row mentioning them is not "unresolved"
    (re.compile(_L + r"(PXD\d{4,9})" + _R, re.I), "pride", "upper", "pxd"),
    (re.compile(_L + r"(syn\d{4,12})" + _R, re.I), "synapse", "lower", "syn"),
    (re.compile(_L + r"(MTBLS\d{1,7})" + _R, re.I), "metabolights", "upper", "mtbls"),
    (re.compile(_L + r"([SED]RS\d{5,10})" + _R, re.I), "sra_sample", "upper", "rs"),
    (re.compile(_L + r"(GDS\d{2,6})" + _R, re.I), "geo_dataset", "upper", "gds"),
    (re.compile(_L + r"(PRJCA\d{4,9})" + _R, re.I), "gsa_project", "upper", "prjca"),
    (re.compile(_L + r"(OMIX\d{3,9})" + _R, re.I), "omix", "upper", "omix"),
    (re.compile(_L + r"(EGAF\d{11})" + _R), "ega_file", None, "egaf"),
    (re.compile(r"10\.5281/zenodo\.(\d{4,9})"), "zenodo", None, "10.5281/zenodo"),
    (re.compile(r"zenodo\.org/records?/(\d{4,9})", re.I), "zenodo", None, "zenodo.org"),
    (re.compile(_L + r"zenodo\s*(?:record\s*)?(\d{5,9})" + _R, re.I), "zenodo", None, "zenodo"),
    (re.compile(r"10\.6084/m9\.figshare\.(\d{5,9})"), "figshare", None, "10.6084/m9.figshare"),
    (re.compile(r"figshare\.com/[^\s)]*?/(\d{5,9})", re.I), "figshare", None, "figshare.com"),
    (re.compile(_L + r"figshare\+?\s*(\d{6,9})" + _R, re.I), "figshare", None, "figshare"),
    (re.compile(_L + r"(10\.17632/[a-z0-9]{6,20}\.\d{1,3})", re.I), None, None, "10.17632/"),
    (re.compile(_L + r"(10\.\d{4,9}/[^\s;,)\]\"']+)"), "doi", None, "10."),
    (re.compile(_L + r"((?:https?|ftp)://[^\s;,)\]\"'>]+)"), "url", None, "://"),
]


@dataclass
class Accession:
    id: str
    kind: str
    context: str = ""        # the cell it was found in

    def __repr__(self) -> str:
        return f"{self.id} ({self.kind})"


def find_accessions(text, context: str = "") -> list:
    """Every accession-shaped thing in a string, canonicalised so `geminga.acquire` can price
    it: a bare Zenodo record number becomes its DOI, a figshare article number likewise."""
    if text is None:
        return []
    s = str(text)
    out: list[Accession] = []
    seen = set()
    low = s.lower()
    for rx, kind, case, hint in _ACC_PATTERNS:
        if hint not in low:
            continue
        for m in rx.finditer(s):
            raw = m.group(1)
            if case == "upper":
                raw = raw.upper()
            elif case == "lower":
                raw = raw.lower()
            if kind == "zenodo":
                acc, k = f"10.5281/zenodo.{raw}", "zenodo_doi"
            elif kind == "figshare":
                acc, k = f"10.6084/m9.figshare.{raw}", "figshare_doi"
            elif kind == "doi":
                low = raw.lower()
                if "zenodo" in low or "figshare" in low or "17632" in low:
                    continue                          # already caught by the specific pattern
                acc, k = raw.rstrip(".)"), "doi"
            elif kind == "url":
                acc = raw.rstrip(".)")
                low = acc.lower()
                # A URL that merely carries an accession already found (a GEO query page, a
                # Zenodo record, a doi.org link) adds nothing; keep URLs that are the only lead.
                if any(x in low for x in ("zenodo.org/record", "figshare.com/articles", "doi.org/10.")) \
                        or any(sid in low for sid in seen if len(sid) >= 6):
                    k = "url-dup"
                else:
                    k = classify(acc)
            else:
                acc, k = raw, kind or classify(raw)
            if k == "url-dup":
                continue
            key = acc.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(Accession(acc, k, context))
    return out


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------

#: Column synonyms: canonical field -> names seen in the wild (compared after normalisation).
_COLUMNS: dict[str, tuple] = {
    "id":        ("dataset_id", "id", "dataset", "key", "name", "dataset / cohort", "dataset / resource",
                  "dataset / model", "resource", "study", "study / finding", "cohort / study", "project / study",
                  "dataset focus", "database", "tool / database", "model"),
    "title":     ("title", "dataset focus", "description", "content summary", "what it does", "content"),
    "reference": ("first_author_year", "reference", "citation", "year & source", "key reference / accession",
                  "status & citation", "journal", "first author"),
    "doi":       ("doi",),
    "accession": ("accession", "accessions", "accession(s)", "accession / host", "data_url", "url",
                  "url / access", "data url", "link", "links", "format & host", "host"),
    "species":   ("species", "organism", "taxon"),
    "context":   ("context_type", "context", "cell types / context", "population / country",
                  "country / region", "demographic"),
    "tissue":    ("tissue_or_cell_type", "tissue", "cell_type", "cell type", "cell lines / relevance to crc & lung"),
    "modalities": ("modalities", "modality", "assay", "assays", "protocols used", "sequences_modalities",
                   "sequencing approach", "assay / data type", "readout", "layer", "type"),
    "platform":  ("platform", "instrument", "chemistry", "delivery / method"),
    "disease":   ("disease_state", "disease", "cancer type", "cancer focus", "subtype", "cancer / tissue scope",
                  "cancer relevance"),
    "n_cells":   ("n_cells", "cells", "n cells", "scale", "size", "subjects_scans", "size (verified where stated)"),
    "n_samples": ("n_donors_or_samples", "n_samples", "samples", "donors", "n_donors", "subjects"),
    "n_adt":     ("n_adt", "n_proteins", "adt"),
    "panel":     ("protein_panel_list", "panel", "antibody panel"),
    "genes_protein": ("cd_protein_targets (listed)", "protein_targets", "cd_protein_targets", "adt_targets",
                      "proteins measured", "surface targets"),
    "genes_perturbed": ("cd_genes_perturbed", "genes_perturbed", "perturbed_genes", "what is perturbed",
                        "targets", "target genes", "factors used"),
    "genes_focus": ("cd_focus", "focus_genes", "genes_of_interest", "key_genes", "host molecule", "genes"),
    "perturbation_type": ("perturbation_type", "perturbation", "perturbagen type", "direction of reprogramming"),
    "perturbation_scope": ("perturbation_scope", "scope"),
    "perturbation_n_genes": ("perturbation_n_genes", "n_genes_perturbed", "n_targets"),
    "perturbation_readout": ("perturbation_readout", "readout"),
    "formats":   ("file_formats", "formats", "format", "file formats", "format & host"),
    "access":    ("access", "access note", "url / access", "accessibility"),
    "verified":  ("verified", "verification", "status"),
    "priority":  ("priority", "tier", "fit_0_3", "fit", "rank", "group"),
    "priority_reason": ("priority_reason", "why it matters here", "why it matters", "relevance to this project",
                        "relevance / how it fits this pipeline", "use in this project", "design & relevance to tf/drug analysis",
                        "relevance / caveat", "how to use it here", "tme relevance", "why it matters for this project",
                        "why it matters for the translation question", "recommended use", "key finding"),
    "notes":     ("notes", "note", "comment", "comments", "remarks", "caveat", "evidence"),
    "raw_counts": ("raw_counts_available", "raw_counts", "raw counts"),
    "evidence":  ("edge_evidence", "evidence", "evidence strength", "evidence basis"),
}


_UNITS = {"n_cells": ("cells", "nuclei", "spots", "bins", "profiles", "single-cell transcriptomes"),
          "n_samples": ("donors", "patients", "samples", "subjects", "lines", "models", "individuals", "screens"),
          "n_adt": ("adt", "proteins", "antibodies", "targets", "-plex")}


def _number(m) -> int:
    n = float(m.group(1).replace(",", ""))
    if m.group(2):
        n *= 1000 if m.group(2).lower() == "k" else 1_000_000
    return int(round(n))


_NUM = r"~?\s*>?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?:\s*([kKmM])(?![A-Za-z]))?(?![\d,])"
_NUM_RX = re.compile(_NUM)


@functools.lru_cache(maxsize=64)
def _unit_rx(unit: str):
    return re.compile(_NUM + r"\+?\s*(?:[a-z\-]+\s){0,2}" + re.escape(unit), re.I)


#: Nouns that make the number before them a count of something other than observations: with
#: `units` given, "44 datasets" is not 44 cells and "20 antibodies" is not 20 donors. (People,
#: scans and lines are observation-like sizes and stay countable — a "Scale" cell of "100 MS
#: patients" is a size.)
_OTHER_NOUN = re.compile(
    r"\+?\s*(?:[a-z\-]+\s){0,2}(?:datasets?|stud(?:y|ies)|cohorts?|papers?|publications?|projects?|atlas(?:es)?"
    r"|files?|genes?|features?|conditions?|time\s?points?|tissues?|cell\s+types?|batches|assays?|modalit(?:y|ies)"
    r"|platforms?|technolog(?:y|ies)|resources?|databases?|librar(?:y|ies)|lanes?|runs?|guides?|sgrnas?|perturbations?"
    r"|panels?|proteins?|antibod(?:y|ies)|targets?|markers?|protocols?|methods?|references?|sites?|centres?|centers?"
    r"|countries|figures?|tables?)\b|-?plex\b", re.I)


def _first_int(text, units: Sequence[str] = (), require_unit: bool = False) -> Optional[int]:
    """The count in '58,302 profiled; 19,834 after QC' (58302), '~1.7M cells' (1700000), or
    '44 datasets, 1.2M cells' when `units` says which word the number must precede. With
    `require_unit`, a number that precedes none of the unit words is not a count at all.

    Without `require_unit` but with `units`, a number that precedes no unit word still counts
    ('300', '58,302 profiled') — unless it is followed by a noun that names something else being
    counted (`_OTHER_NOUN`: datasets, studies, cohorts, genes …): '44 datasets' is None. NaN and
    infinity are not counts."""
    if text is None:
        return None
    if isinstance(text, (int, float, np.integer, np.floating)) and not isinstance(text, bool):
        f = float(text)
        return int(f) if np.isfinite(f) else None
    s = str(text)
    for u in units:
        m = _unit_rx(u).search(s)
        if m:
            return _number(m)
    if require_unit:
        return None
    for m in _NUM_RX.finditer(s):
        if units and _OTHER_NOUN.match(s, m.end()):
            continue
        return _number(m)
    return None


_GENE_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9\-\.]{0,19}$")
_GENE_STOP = frozenset({"none", "n/a", "na", "various", "all", "genome-wide", "unknown",
                        "ko", "kd", "wt", "oe", "ntc", "nt", "and", "or"})
_PAREN = re.compile(r"\(([^()]*)\)")
_GENE_JOIN = re.compile(r"\s*(?:/|&|\band\b|\bor\b)\s*", re.I)
_LIST_SEP = re.compile(r"\s*[,;|]\s*")


def _gene_ok(tok: str) -> bool:
    if _GENE_TOKEN.match(tok):
        return tok.lower() not in _GENE_STOP
    kind = id_type(tok)
    # an Entrez id needs three digits: a bare '12' in a gene cell is a count, not a gene
    return kind == "ensembl" or (kind == "entrez" and len(tok) >= 3)


def _genes(text) -> list:
    """Gene-like tokens from a cell such as 'CD276; CD274; CD47' — tolerant of prose: 'IL2 and
    IL7', 'CD3, CD4, and CD8', 'MYC/MAX' and 'TP53 (p53)' (the alias in brackets is a token of
    its own) all give their symbols; qualifiers ('KO'), 'N/A' and bare small numbers do not."""
    out = []
    if _missing(text):
        return out
    for piece in _split(text):
        piece = piece.strip()
        if piece.lower() in _GENE_STOP:                  # 'N/A', before '/' splits it
            continue
        bracketed = _PAREN.findall(piece)
        parts = [_PAREN.sub(" ", piece)] + [x for b in bracketed for x in _LIST_SEP.split(b)]
        for part in parts:
            for tok in _GENE_JOIN.split(part):
                tok = tok.strip().strip(".")
                if tok and _gene_ok(tok):
                    out.append(tok)
    return list(dict.fromkeys(out))


@dataclass
class Piece:
    """One described dataset — what a row of an inventory says about it, structured."""
    id: str
    title: str = ""
    reference: str = ""
    accessions: list = field(default_factory=list)        # list[Accession]
    facets: dict = field(default_factory=dict)            # keys from FACETS
    modalities: list = field(default_factory=list)        # canonical assay labels
    context: str = ""                                     # cell line | organoid | tissue | ...
    perturbation: dict = field(default_factory=dict)      # type, scope, n_genes, readout
    genes: dict = field(default_factory=dict)             # perturbed / protein / focus -> [symbols]
    panel: str = ""
    formats: list = field(default_factory=list)
    access: str = ""
    verified: str = ""
    priority: str = ""
    priority_reason: str = ""
    expected: dict = field(default_factory=dict)          # n_cells, n_samples, n_adt (ints) + raw text
    notes: str = ""
    row: dict = field(default_factory=dict)               # the source row, verbatim
    source: str = ""                                      # file#sheet:row
    local: list = field(default_factory=list)             # paths found on this machine
    status: str = "unresolved"
    via: str = ""                                         # the accession/route that earned the status

    @property
    def primary_assay(self) -> str:
        return self.facets.get("assay", "") or (self.modalities[0] if self.modalities else "")

    @property
    def all_genes(self) -> list:
        out = []
        for k in ("focus", "perturbed", "protein"):
            for g in self.genes.get(k, []):
                if g not in out:
                    out.append(g)
        return out

    def accession_ids(self) -> list:
        return [a.id for a in self.accessions]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["accessions"] = [asdict(a) for a in self.accessions]
        return d

    def line(self) -> str:
        n = self.expected.get("n_cells")
        via = self.via or (self.accessions[0].id if self.accessions else "-")
        return (f"{self.id[:40]:<40} {self.primary_assay[:22]:<22} {(f'{n:,}' if n else '-'):>10}  "
                f"{self.status:<11} {via}")

    def __repr__(self) -> str:
        return f"Piece({self.id}, {self.primary_assay or '?'}, {len(self.accessions)} accession(s), {self.status})"


def _col_lookup(columns: Sequence[str]) -> dict:
    """canonical field -> actual column name, by normalised synonym; first synonym wins."""
    low = {_norm(c): c for c in columns if c is not None}
    out = {}
    for canon, syns in _COLUMNS.items():
        for s in syns:
            if s in low and low[s] not in out.values():
                out[canon] = low[s]
                break
    return out


def _union_keys(rows: list) -> list:
    """Every key any row has, in first-seen order — rows of dicts rarely share one key set."""
    out: dict = {}
    for r in rows:
        for k in r:
            out.setdefault(k, None)
    return list(out)


def _dedupe_headers(headers: Sequence) -> list:
    """Two columns called ``Notes`` become ``Notes`` and ``Notes.1`` (as pandas does) instead of
    one column holding whichever came last; a generated name never takes one that exists."""
    names = [str(h) for h in headers]
    taken = set(names)
    seen: set = set()
    out = []
    for n in names:
        if n in seen:
            k = 1
            while f"{n}.{k}" in taken:
                k += 1
            n = f"{n}.{k}"
            taken.add(n)
        seen.add(n)
        out.append(n)
    return out


class _Rows(list):
    """Rows of dicts that remember the row of the sheet (or line of the file) each came from."""
    numbers: Optional[list] = None


def _clean_row(r: dict) -> dict:
    """A row with every empty cell — None, NaN, pandas NA — as the empty string, so a missing
    tissue is '' and not the text 'nan'."""
    return {k: ("" if _missing(v) else v) for k, v in r.items()}


def _rows_from(source, sheet: Optional[str] = None) -> tuple:
    """(rows as dicts, columns, label) from a CSV/TSV, XLSX tab, Parquet, JSONL, or dict list.

    Empty cells are ''. Repeated column headers are kept apart (``Notes``, ``Notes.1``). The rows of
    a CSV or workbook carry the real row/line number of each (``rows.numbers``), blank rows and
    all."""
    if isinstance(source, (list, tuple)):
        rows = [_clean_row(dict(r)) for r in source]
        return rows, _union_keys(rows), "rows"
    if hasattr(source, "to_pylist"):                     # pyarrow Table
        names = list(source.column_names)
        dd = _dedupe_headers(names)
        if dd != names:
            source = source.rename_columns(dd)
        return [_clean_row(r) for r in source.to_pylist()], dd, "table"
    if hasattr(source, "to_dict") and hasattr(source, "columns"):   # pandas / polars
        names = list(source.columns)
        dd = _dedupe_headers(names)
        if dd != [str(c) for c in names]:
            source = source.copy()
            source.columns = dd
            names = dd
        try:
            rows = source.to_dict("records")
        except TypeError:
            rows = source.to_dicts()
        return [_clean_row(r) for r in rows], names, "frame"
    path = os.path.expanduser(str(source))
    low = path.lower()
    label = os.path.basename(path)
    if low.endswith((".xlsx", ".xlsm")):
        try:
            import openpyxl
        except ImportError as e:
            raise ImportError("reading .xlsx needs openpyxl: pip install openpyxl") from e
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        names = wb.sheetnames
        if sheet is None:
            # The tab that is a dataset list: the one whose cells hold the most accessions
            # (not counting bare URLs and DOIs, which every reference list has).
            best, best_n = names[0], -1
            for n in names:
                count = 0
                for i, r in enumerate(wb[n].iter_rows(values_only=True)):
                    if i > 300:
                        break
                    for c in r:
                        if isinstance(c, str) and len(c) > 4:
                            count += sum(1 for a in find_accessions(c) if a.kind not in ("url", "doi"))
                if count > best_n:
                    best, best_n = n, count
            sheet = best
        if sheet not in names:
            raise KeyError(f"no sheet {sheet!r} in {label}; have {names}")
        it = wb[sheet].iter_rows(values_only=True)
        hdr = _dedupe_headers([str(h).strip() if h is not None else f"col{i}"
                               for i, h in enumerate(next(it, ()))])
        rows = _Rows()
        rows.numbers = []
        for n, r in enumerate(it, 2):                    # the sheet's own row numbers: row 1 is the header
            if r is None or all(c is None or str(c).strip() == "" for c in r):
                continue
            rows.append(_clean_row({hdr[i]: r[i] for i in range(min(len(hdr), len(r)))}))
            rows.numbers.append(n)
        wb.close()
        return rows, hdr, f"{label}#{sheet}"
    if low.endswith(".parquet"):
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        names = list(t.column_names)
        dd = _dedupe_headers(names)
        if dd != names:
            t = t.rename_columns(dd)
        return [_clean_row(r) for r in t.to_pylist()], dd, label
    if low.endswith((".jsonl", ".ndjson")):
        rows = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(_clean_row(json.loads(line)))
        return rows, _union_keys(rows), label
    if low.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
        rows = rows if isinstance(rows, list) else rows.get("pieces", rows.get("rows", []))
        rows = [_clean_row(r) for r in rows]
        return rows, _union_keys(rows), label
    delim = "\t" if low.endswith((".tsv", ".tab", ".txt")) else ","
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd = csv.reader(fh, delimiter=delim)
        header = next(rd, [])
        hdr = _dedupe_headers([h if h != "" else f"col{i}" for i, h in enumerate(header)])
        rows = _Rows()
        rows.numbers = []
        last = rd.line_num                               # lines used up by the header
        for rec in rd:
            first, last = last + 1, rd.line_num          # the record starts after the previous one ended
            if not rec or all(str(c).strip() == "" for c in rec):
                continue
            row = {h: (rec[i] if i < len(rec) else "") for i, h in enumerate(hdr)}
            if len(rec) > len(hdr):
                row[None] = rec[len(hdr):]
            rows.append(row)
            rows.numbers.append(first)
        return rows, hdr, label


def read_descriptions(source, sheet: Optional[str] = None, *, id_column: Optional[str] = None,
                      harmonizer: Optional[Harmonizer] = None, columns: Optional[dict] = None) -> list:
    """Rows of dataset descriptions -> :class:`Piece` objects.

    `source` is a CSV/TSV path, an .xlsx path (`sheet=` picks the tab; by default the first tab
    with an accession-like column), a Parquet/JSONL path, a pyarrow Table, a DataFrame, or a
    list of dicts. Column names are matched against a synonym table; `columns=` pins any the
    guesser gets wrong (``{"accession": "Key Reference / Accession"}``). Accessions are pulled
    from *every* text cell of the row, so a spreadsheet that mentions the GEO series only in
    its notes still resolves.

    Empty cells (None, NaN, NA) are ``""`` however the table was read, so a missing tissue is
    not the text "nan". Repeated column headers stay apart (``Notes``, ``Notes.1``) and every
    one is searched for accessions. A pin in `columns=` steers the piece's fields *and* its
    facets (``{"species": "Org"}`` makes ``facets["organism"]`` come from the ``Org`` column).
    ``Piece.source`` names the row as the sheet (or file) numbers it, blank rows included.
    Piece ids are made unique: a repeat gets ``_2``, ``_3`` … and never takes an id that exists.
    """
    h = harmonizer or Harmonizer()
    rows, cols, label = _rows_from(source, sheet)
    cmap = _col_lookup(cols)
    pins = {k: v for k, v in (columns or {}).items() if v in cols}
    cmap.update(pins)
    if id_column:
        cmap["id"] = id_column
    pieces: list[Piece] = []
    seen_ids: dict[str, int] = {}
    numbers = getattr(rows, "numbers", None) or list(range(2, len(rows) + 2))

    def cell(field_name):
        c = cmap.get(field_name)
        return row.get(c) if c else None

    for k, row in enumerate(rows):
        i = numbers[k]                                   # the row of the sheet this piece came from
        raw_id = cell("id")
        if raw_id is None or str(raw_id).strip() == "":
            raw_id = cell("title") or cell("reference") or f"row{i}"
        pid = re.sub(r"[^A-Za-z0-9_.\-]+", "_", str(raw_id).strip())[:80].strip("_") or f"row{i}"
        if pid in seen_ids:                              # 'A, A, A_2': every id ends up unique
            n = seen_ids[pid]
            while f"{pid}_{n + 1}" in seen_ids:
                n += 1
            seen_ids[pid] = n + 1
            pid = f"{pid}_{n + 1}"
        seen_ids[pid] = seen_ids.get(pid, 1)

        # accessions: the accession-ish columns first (their order matters), then everything else
        accs: list[Accession] = []
        ordered = [cmap[k] for k in ("accession", "doi", "reference") if k in cmap]
        seen_acc: set = set()
        for c in ordered + [c for c in cols if c not in ordered]:
            v = row.get(c)
            if v is None or isinstance(v, (int, float)) or len(str(v)) < 5:
                continue
            for a in find_accessions(v, context=str(c)):
                if a.id.lower() not in seen_acc:
                    seen_acc.add(a.id.lower())
                    accs.append(a)

        facets = h.facets(row, columns=pins) if pins else h.facets(row)
        mods_text = cell("modalities")
        modalities = h.vocabulary.assays(mods_text) if mods_text else []
        if not facets.get("assay") and modalities:
            facets["assay"] = modalities[0]
        p_type = h.vocabulary.perturbation(cell("perturbation_type")) or \
            h.vocabulary.perturbation(cell("modalities")) or h.vocabulary.perturbation(cell("platform"))
        perturbation = {}
        if p_type or cell("perturbation_scope") or cell("perturbation_n_genes"):
            perturbation = {"type": p_type or str(cell("perturbation_type") or "").strip(),
                            "scope": str(cell("perturbation_scope") or "").strip(),
                            "n_genes": _first_int(cell("perturbation_n_genes")),
                            "readout": str(cell("perturbation_readout") or "").strip()}
        genes = {}
        for role in ("perturbed", "protein", "focus"):
            gl = _genes(cell(f"genes_{role}"))
            if gl:
                genes[role] = gl
        expected = {}
        for k in ("n_cells", "n_samples", "n_adt"):
            v = cell(k)
            if v not in (None, ""):
                n = _first_int(v, _UNITS[k])
                if n is not None:
                    expected[k] = n
                expected[k + "_text"] = str(v).strip()
        # one "Scale" column often carries both: '371,223 cells; 28 MMRp + 34 MMRd patients'
        if "n_samples" not in expected and "n_cells_text" in expected:
            n = _first_int(expected["n_cells_text"], _UNITS["n_samples"], require_unit=True)
            if n is not None:
                expected["n_samples"] = n
        access_text = cell("access")
        access = h.vocabulary.access(access_text) if access_text else ""
        if not access and accs:
            if any(a.kind in ("ega_dataset", "ega_study", "ega_file", "dbgap", "duos") for a in accs):
                access = "controlled"
        pieces.append(Piece(
            id=pid, title=str(cell("title") or "").strip(), reference=str(cell("reference") or "").strip(),
            accessions=accs, facets=facets, modalities=modalities,
            context=h.vocabulary.context(cell("context")) or str(cell("context") or "").strip(),
            perturbation=perturbation, genes=genes, panel=str(cell("panel") or "").strip(),
            formats=_split(cell("formats")), access=access, verified=str(cell("verified") or "").strip(),
            priority=str(cell("priority") or "").strip(), priority_reason=str(cell("priority_reason") or "").strip(),
            expected=expected, notes=str(cell("notes") or "").strip(),
            row={str(k): (v if isinstance(v, (int, float, bool)) or v is None else str(v)) for k, v in row.items()},
            source=f"{label}:{i}",
        ))
    return pieces


# ---------------------------------------------------------------------------
# Resolution — is it there?
# ---------------------------------------------------------------------------

@dataclass
class Resolution:
    """What resolving one piece found, best route first."""
    piece: str
    status: str = "unresolved"
    local_paths: list = field(default_factory=list)
    plans: dict = field(default_factory=dict)          # accession -> AcquisitionPlan
    route_kind: str = ""
    route_url: str = ""
    bytes: int = 0
    cost_usd: float = 0.0
    needs: str = ""                                    # tool or approval needed
    note: str = ""
    via: str = ""                                      # accession that earned the status
    controlled: list = field(default_factory=list)     # accessions that need approval, if any
    cost_known: bool = True                            # False: metered route of unknown size

    @property
    def here(self) -> bool:
        return self.status == "local"

    def explain(self) -> str:
        lines = [f"{self.piece}: {_STATUS_WORDS.get(self.status, self.status)}"]
        for p in self.local_paths:
            lines.append(f"  local  {p}")
        if self.route_kind:
            lines.append(f"  via    {self.route_kind} {self.route_url}"
                         + (f"  {human(self.bytes)}" if self.bytes else "")
                         + (f"  ~${self.cost_usd:.2f}" if self.cost_usd else
                            ("  metered, size unknown" if not self.cost_known else ""))
                         + (f"  needs {self.needs}" if self.needs else ""))
        if self.controlled:
            lines.append(f"  also   controlled-access component(s): {', '.join(self.controlled)}")
        if self.note:
            lines.append(f"  note   {self.note}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"piece": self.piece, "status": self.status, "local_paths": list(self.local_paths),
                "route_kind": self.route_kind, "route_url": self.route_url, "bytes": self.bytes,
                "cost_usd": self.cost_usd, "cost_known": self.cost_known, "needs": self.needs,
                "note": self.note, "via": self.via,
                "controlled": list(self.controlled),
                "plans": {k: v.to_dict() for k, v in self.plans.items()}}


_SRA_FAMILY = ("sra_run", "sra_experiment", "sra_study", "bioproject", "biosample")
_DATA_SUFFIXES = (".h5ad", ".h5", ".hdf5", ".mtx", ".mtx.gz", ".gef", ".gem", ".gem.gz", ".parquet", ".arrow",
                  ".feather", ".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt.gz", ".zarr", ".loom", ".rds",
                  ".fastq.gz", ".fq.gz", ".bam", ".tar", ".tar.gz", ".zip", ".nii", ".nii.gz", ".tif", ".tiff")
_DATA_SUFFIXES_BY_LENGTH = tuple(sorted(_DATA_SUFFIXES, key=len, reverse=True))


def _needle_rx(needle: str):
    """A name belongs to an accession when the accession stands alone in it: not inside a longer
    token (`GSE123456` is not `GSE12345`; `SRR14487741` is not `SRR1448774`) and not followed by
    another digit. A letter or an underscore after it is fine (`GSE12345_RAW.tar`)."""
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![0-9])", re.I)


def _is_matrix_dir(subdirs: Sequence[str], files: Sequence[str]) -> bool:
    """A directory that is a dataset in its own right: a MatrixMarket triple, or a Cell Ranger
    bundle (`outs/`, `filtered_feature_bc_matrix[.h5]`, `raw_feature_bc_matrix[.h5]`)."""
    names = [n.lower() for n in (*subdirs, *files)]
    return ("matrix.mtx" in names or "matrix.mtx.gz" in names or "outs" in names
            or any(n.startswith(("filtered_feature_bc_matrix", "raw_feature_bc_matrix"))
                   or n.endswith("_feature_bc_matrix.h5") for n in names))


def _is_puzzle_dir(files: Sequence[str]) -> bool:
    """A directory `Orchestrator.build` wrote: its X_*/T_* files are outputs, never sources."""
    return "blueprint.json" in files and "manifest.json" in files


def _is_build_output(path: str) -> bool:
    d = os.path.dirname(os.path.abspath(path))
    return os.path.exists(os.path.join(d, "blueprint.json")) and os.path.exists(os.path.join(d, "manifest.json"))


def _data_entries(dirs: Sequence[str], max_files: int = 200_000) -> list:
    """(path, stem) for every dataset under `dirs`: files with a data suffix, `.zarr` stores, and
    matrix directories — in a fixed order. Directories that a build wrote, and anything hidden,
    are left out. The stem is the name without its data suffix."""
    out, n = [], 0
    for d in dirs:
        d = os.path.expanduser(str(d))
        if not os.path.isdir(d):
            continue
        for root, subdirs, files in os.walk(d):
            subdirs[:] = sorted(x for x in subdirs if not x.startswith("."))
            files = sorted(f for f in files if not f.startswith("."))
            if _is_puzzle_dir(files):
                subdirs[:] = []
                continue
            n += len(subdirs) + len(files)
            if root != d and _is_matrix_dir(subdirs, files):
                out.append((root, os.path.basename(root)))
            for name in subdirs[:]:
                if name.lower().endswith(".zarr"):
                    out.append((os.path.join(root, name), name[:-5]))
                    subdirs.remove(name)
            for name in files:
                low = name.lower()
                suffix = next((x for x in _DATA_SUFFIXES_BY_LENGTH if low.endswith(x)), None)
                if suffix:
                    out.append((os.path.join(root, name), name[:-len(suffix)]))
            if n >= max_files:
                return out
    return out


class _EntryIndex:
    """Find the entries whose stem holds an accession on its boundaries without a regex per
    entry: one substring search over all stems, then the boundary rule on the few that contain
    the text."""

    def __init__(self, entries: Sequence[tuple]):
        self.entries = list(entries)
        self._starts: list = []
        pos, parts = 0, []
        for _, stem in self.entries:
            low = stem.lower()                          # (case folding can change a string's length)
            self._starts.append(pos)
            parts.append(low)
            pos += len(low) + 1
        self._blob = "\x00".join(parts)

    def find(self, needles: Sequence[str]) -> list:
        hits: set = set()
        for needle in needles:
            w = needle.lower()
            if len(needle) < 4 or not self.entries:
                continue
            rx = _needle_rx(needle)
            at = self._blob.find(w)
            while at >= 0:
                i = bisect.bisect_right(self._starts, at) - 1
                if i not in hits and rx.search(self.entries[i][1]):
                    hits.add(i)
                at = self._blob.find(w, at + 1)
        return [self.entries[i][0] for i in sorted(hits)]


def _walk_for(needles: Sequence[str], dirs: Sequence[str], max_files: int = 200_000) -> list:
    """Datasets under `dirs` whose name carries one of the accessions: a data file (`_DATA_SUFFIXES`),
    a `.zarr` store, or a matrix directory, with the accession standing alone in its name
    (`_needle_rx`), case-insensitive. A figure called `atlas_figure.png`, a `GSE123456` file for
    `GSE12345` and the outputs of an earlier build are not hits."""
    wants = [n for n in needles if n and len(n) >= 4]
    if not wants:
        return []
    return _EntryIndex(_data_entries(dirs, max_files)).find(wants)


_HOSTS = ("figshare", "zenodo", "geo", "arrayexpress", "biostudies", "ega", "dbgap", "sra", "ena", "kaggle",
          "openneuro", "nitrc", "synapse", "dryad", "mendeley", "single cell portal", "cellxgene", "hca",
          "gsa", "cngb", "stomics", "hugging face", "huggingface", "ida", "loni", "tcia", "xnat", "physionet",
          "osf", "gdc", "tcga", "depmap", "biogrid", "proteomexchange", "pride", "metabolights")


_HOSTS_RX = re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(h) for h in sorted(_HOSTS, key=len, reverse=True))
                       + r")(?![a-z])")


def _hosts_named(text: str) -> list:
    found = {m.group(0) for m in _HOSTS_RX.finditer(" " + _norm(text) + " ")}
    return [h for h in _HOSTS if h in found]


def _status_of_plan(p: AcquisitionPlan, offline: bool) -> tuple:
    """(status, route) for one accession's plan."""
    if p.kind in _SRA_FAMILY and offline:
        return "fetchable", None
    usable = [r for r in p.routes if not r.needs_auth and r.kind != "manual"]
    if usable:
        best = usable[0]
        return ("streamable" if best.streamable else "fetchable"), best
    auth = [r for r in p.routes if r.needs_auth]
    if auth:
        return "controlled", auth[0]
    manual = [r for r in p.routes if r.kind == "manual"]
    if manual:
        return "manual", manual[0]
    return "unresolved", None


#: Accession kinds `acquire` cannot price, and what to say about them: (status, tool, note).
_UNPRICED = {
    "pride": ("fetchable", "", "PRIDE / ProteomeXchange: open, files over HTTPS or FTP from the PRIDE archive"),
    "metabolights": ("fetchable", "", "MetaboLights: open, files over FTP or HTTPS"),
    "synapse": ("fetchable", "synapseclient",
                "Synapse: needs a Synapse login (some projects also terms of use); pip install synapseclient"),
    "geo_dataset": ("fetchable", "", "a curated GEO DataSet (GDS): the data are the GSE series it derives from"),
    "sra_sample": ("fetchable", "", "free from ENA/SRA through the sample's runs; sizes known once online"),
    "gsa_project": ("fetchable", "", "CNCB-NGDC GSA project: open data free over FTP, human data needs an application"),
    "omix": ("fetchable", "", "NGDC OMIX: open archive, some entries need an application"),
    "ega_file": ("controlled", "pyega3", "EGA file: controlled access, needs a Data Access Committee approval and pyega3"),
}


def resolve_pieces(pieces: Sequence[Piece], *, vault=None, catalog=None, search_dirs: Sequence[str] = (),
                   offline: bool = True, timeout: float = 20.0) -> dict:
    """Resolve every piece: local first, then priced routes. Never transfers data.

    A piece is *local* when something that exists on this machine carries one of its accessions
    (or its id, when the id is itself accession-shaped) as a whole token in its name — a data
    file, a `.zarr` store or a matrix directory under `search_dirs`, a vault member — or when the
    catalog or an `attach()` names an existing path for it. A name that merely contains the
    accession (`GSE123456` for `GSE12345`) or a path that has since gone does not count.

    `offline=True` (the default) prices from accession shape alone — SRA-family accessions are
    reported *fetchable* with the size unknown, because ENA's file report is the only way to
    learn it. Set `offline=False` to ask ENA, GEO, Zenodo and figshare for sizes.
    """
    out: dict[str, Resolution] = {}
    if isinstance(search_dirs, (str, os.PathLike)):
        search_dirs = [search_dirs]
    cat_entries = catalog.entries() if catalog is not None else {}
    # Index the catalog once by every accession its entries mention (key, target, url, tags,
    # metadata), so matching pieces to entries is a lookup rather than pieces × entries.
    by_acc: dict[str, list] = {}
    for key, e in cat_entries.items():
        hay = " ".join([e.key, e.target, e.url, " ".join(e.tags), json.dumps(e.metadata, default=str)])
        for a in find_accessions(hay):
            by_acc.setdefault(a.id.lower(), []).append(e)
    dir_index = _EntryIndex(_data_entries(search_dirs)) if search_dirs else None
    for p in pieces:
        r = Resolution(piece=p.id)
        # a piece id names a dataset only when it is itself accession-shaped ('Atlas' is not)
        needles = list(dict.fromkeys(([p.id] if find_accessions(p.id) else []) + p.accession_ids()))
        found: set = set()

        def _add(path):
            real = os.path.abspath(path)
            if real not in found and os.path.exists(path):
                found.add(real)
                r.local_paths.append(path)
        # --- local: catalog, vault, directories
        cands = {p.id: cat_entries[p.id]} if p.id in cat_entries else {}
        for a in p.accession_ids():
            for e in by_acc.get(a.lower(), ()):
                cands.setdefault(e.key, e)
        for e in cands.values():
            for lp in (e.local_paths or []) + [e.target]:
                if os.path.exists(lp) and not _is_build_output(lp):   # note_local() records a build's outputs
                    _add(lp)
        if vault is not None:
            rxs = {n: _needle_rx(n) for n in needles}
            for needle in needles:
                for m in vault.find(needle):
                    if rxs[needle].search(m.path):         # find() is a substring test; this is the boundary
                        _add(vault.abspath(m.path) if hasattr(vault, "abspath") else m.path)
        if dir_index is not None:
            for hit in dir_index.find(needles):
                _add(hit)
        for lp in p.local:
            _add(lp)                                       # attached by hand, if still there
        # --- routes
        best_rank = len(RESOLUTION_STATUSES)
        for a in p.accessions:
            if a.kind in ("doi", "url", "scp", "duos", "geo_platform", "gsa_human", "unknown", *_UNPRICED):
                # not priceable by acquire; still informative
                needs = ""
                if a.kind == "duos":
                    st, note = "controlled", "DUOS-managed (Terra/AnVIL); needs a data access request"
                elif a.kind == "scp":
                    st, note = "fetchable", "Broad Single Cell Portal — processed matrices, free after sign-in"
                elif a.kind == "gsa_human":
                    st, note = "controlled", "GSA-Human needs an application to CNCB"
                elif a.kind == "url":
                    st, note = "manual", "a URL; g.acquire.peek() can probe it for size and Range support"
                elif a.kind in _UNPRICED:
                    st, needs, note = _UNPRICED[a.kind]
                else:
                    st, note = "manual", f"{a.kind} only — a landing page or a paper, not a file"
                if st == "controlled":
                    r.controlled.append(a.id)
                rank = RESOLUTION_STATUSES.index(st)
                if rank < best_rank:
                    best_rank, r.status, r.route_kind, r.route_url, r.note = rank, st, a.kind, a.id, note
                    r.needs = needs
                    r.via = a.id
                continue
            try:
                pl = acquisition_plan(a.id, timeout=timeout, offline=offline)
            except Exception as e:                       # pricing must never stop a resolve
                r.note = f"{a.id}: {e}"
                continue
            r.plans[a.id] = pl
            st, route = _status_of_plan(pl, offline)
            if st == "controlled":
                r.controlled.append(a.id)
            rank = RESOLUTION_STATUSES.index(st)
            if rank < best_rank:
                best_rank, r.status = rank, st
                r.via = a.id
                if route is not None:
                    r.route_kind, r.route_url = route.kind, route.url
                    r.bytes = route.bytes or pl.total_bytes
                    r.cost_usd = route.cost_usd
                    r.cost_known = bool(getattr(route, "cost_known", True))
                    r.needs = (route.tool if route.tool and not route.available else "") or \
                        ("approval" if route.needs_auth else "")
                    r.note = route.note
                else:
                    r.route_kind, r.route_url = a.kind, a.id
                    r.note = ("free from ENA/SRA; size and FASTQ/SRA routes are known once online "
                              "(resolve(offline=False))")
        if r.local_paths:
            r.status = "local"
        elif r.status == "unresolved" and not p.accessions:
            hosts = _hosts_named(" ".join(str(v) for v in p.row.values() if v is not None))
            r.note = (f"no accession in the row, but it names {', '.join(hosts)} — search there for "
                      f"{p.title or p.id}" if hosts else "no accession, DOI or URL in the description")
        p.status = r.status
        p.local = list(r.local_paths)
        p.via = r.via if r.status != "local" else (r.local_paths[0] if r.local_paths else "")
        out[p.id] = r
    return out


# ---------------------------------------------------------------------------
# Coverage — which piece says what about which gene
# ---------------------------------------------------------------------------

@dataclass
class Coverage:
    """Gene × piece evidence codes: P protein-measured, X perturbed (x implied by a genome-wide
    screen), F named as a focus, R present in a local matrix after harmonisation."""
    genes: list
    pieces: list
    codes: dict                     # (gene, piece) -> "PXF" string
    key: str = "symbol"
    unmapped: dict = field(default_factory=dict)   # name -> pieces that use it; not in the key space

    def code(self, gene: str, piece: str) -> str:
        return self.codes.get((gene, piece), "")

    def counts(self, gene: str) -> dict:
        s = "".join(self.codes.get((gene, p), "") for p in self.pieces)
        return {"P": s.count("P"), "X": s.count("X"), "x": s.count("x"), "F": s.count("F"), "R": s.count("R")}

    def rows(self) -> list:
        out = []
        for g in self.genes:
            row = {"gene": g}
            row.update({p: self.codes.get((g, p), "") for p in self.pieces})
            c = self.counts(g)
            row.update({"n_protein": c["P"], "n_perturbed": c["X"], "n_implied": c["x"],
                        "n_focus": c["F"], "n_measured": c["R"]})
            out.append(row)
        return out

    def table(self):
        import pyarrow as pa
        return pa.Table.from_pylist(self.rows())

    def to_csv(self, path: str) -> str:
        rows = self.rows()
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["gene"])
            w.writeheader()
            w.writerows(rows)
        return path

    def uncovered(self) -> list:
        """Genes no piece measures at protein level, perturbs explicitly, or holds locally."""
        return [g for g in self.genes if not any(ch in "PXR" for p in self.pieces
                                                  for ch in self.codes.get((g, p), ""))]

    def print(self, limit: int = 40) -> None:
        print(f"coverage: {len(self.genes)} genes x {len(self.pieces)} pieces   "
              f"P protein · X perturbed · x implied (genome-wide) · F focus · R measured locally")
        head = f"{'gene':<12}" + "".join(f"{p[:10]:>11}" for p in self.pieces[:12])
        print(head + ("  …" if len(self.pieces) > 12 else "") + f"{'P':>4}{'X':>4}{'F':>4}{'R':>4}")
        for g in self.genes[:limit]:
            c = self.counts(g)
            line = f"{g:<12}" + "".join(f"{self.codes.get((g, p), '·'):>11}" for p in self.pieces[:12])
            print(line + ("   " if len(self.pieces) > 12 else "") +
                  f"{c['P']:>4}{c['X']:>4}{c['F']:>4}{c['R']:>4}")
        if len(self.genes) > limit:
            print(f"  … {len(self.genes) - limit} more genes")
        un = self.uncovered()
        if un:
            print(f"  uncovered ({len(un)}): {', '.join(un[:12])}{' …' if len(un) > 12 else ''}")
        if self.unmapped:
            worst = sorted(self.unmapped.items(), key=lambda kv: -len(kv[1]))[:8]
            print(f"  names the gene map cannot place ({len(self.unmapped)}): "
                  + ", ".join(f"{n} ({len(ps)})" for n, ps in worst)
                  + " — add aliases to the map, or merge an HGNC dump, and these become evidence")


# ---------------------------------------------------------------------------
# Blueprint and Puzzle — what to build, and what got built
# ---------------------------------------------------------------------------

@dataclass
class Blueprint:
    """A recipe: which pieces, keyed how, restricted to what, carrying which columns."""
    name: str
    pieces: list                                  # piece ids, in order
    key: str = "symbol"                           # gene id space of the output
    how: str = "intersection"                     # or "union" — ignored when panel is set
    panel: Optional[list] = None                  # gene list; the output var axis
    obs: list = field(default_factory=lambda: ["sample", "donor", "condition", "perturbation", "guide", "cell_type"])
    modality: str = "RNA"                         # for 10x files with several feature types
    layer: Optional[str] = None                   # h5ad layer instead of X
    note: str = ""
    created: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Blueprint":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"schema": BLUEPRINT_SCHEMA, "blueprint": self.to_dict()}, fh, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "Blueprint":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return cls.from_dict(d.get("blueprint", d))


class Puzzle:
    """A built dataset: the pieces that were here, the holes that were not, and the record."""

    def __init__(self, harvest, blueprint: Blueprint, built: list, holes: list, alignment: Optional[Alignment],
                 provenance: dict, n_obs: int):
        self.harvest = harvest
        self.blueprint = blueprint
        self.built = built              # list[dict] per built piece
        self.holes = holes              # list[dict] per missing piece
        self.alignment = alignment
        self.provenance = provenance
        self.n_obs = n_obs

    @property
    def path(self) -> str:
        return self.harvest.path

    @property
    def n_vars(self) -> int:
        return self.alignment.n if self.alignment is not None else 0

    @property
    def complete(self) -> bool:
        return not self.holes

    def read_obs(self):
        return self.harvest.read("obs")

    def read_var(self):
        return self.harvest.read("var")

    def read_X(self, piece: str):
        """COO of one piece: cell (global row), gene (index into var), value."""
        b = next((b for b in self.built if b["piece"] == piece), None)
        return self.harvest.read(b["roi"] if b and "roi" in b else f"X_{piece}")

    def measured(self) -> dict:
        """{piece: [var ids present]} — what `Orchestrator.coverage(measured=...)` wants."""
        if self.alignment is None:
            return {}
        return {b["piece"]: [c for c, pos in zip(self.alignment.index, self.alignment.positions[b["piece"]])
                             if pos >= 0] for b in self.built if b["piece"] in self.alignment.positions}

    def to_anndata(self):
        """Assemble an AnnData in memory (needs anndata + scipy). For pieces that fit."""
        import anndata as ad
        import scipy.sparse as sp
        obs = self.read_obs().to_pandas()
        var = self.read_var().to_pandas()
        rows, cols, vals = [], [], []
        for b in self.built:
            if b.get("reader") == "table":                 # harmonised tables have no cell × gene matrix
                continue
            t = self.read_X(b["piece"])
            rows.append(t.column("cell").to_numpy()); cols.append(t.column("gene").to_numpy())
            vals.append(t.column("value").to_numpy())
        X = sp.csr_matrix((np.concatenate(vals) if vals else np.zeros(0, np.float32),
                           (np.concatenate(rows) if rows else np.zeros(0, np.int64),
                            np.concatenate(cols) if cols else np.zeros(0, np.int64))),
                          shape=(self.n_obs, self.n_vars))
        obs.index = obs["cell_id"].astype(str)
        var.index = var["id"].astype(str)
        a = ad.AnnData(X=X, obs=obs, var=var)
        a.uns["geminga_blueprint"] = self.blueprint.to_dict()
        a.uns["geminga_holes"] = [h["piece"] for h in self.holes]
        return a

    def print_table(self) -> None:
        bp = self.blueprint
        print(f"puzzle {bp.name} at {self.path}")
        print(f"  {len(self.built)} piece(s) built, {len(self.holes)} hole(s); "
              f"{self.n_obs:,} obs x {self.n_vars:,} vars keyed by {bp.key}"
              + (f" (panel of {len(bp.panel)})" if bp.panel else f" ({bp.how})"))
        for b in self.built:
            print(f"  + {b['piece']:<36} {b['n_obs']:>9,} obs  {b['n_kept']:>6,}/{b['n_vars_in']:<7,} vars kept  "
                  f"{b['reader']:<10} {human(b['resident_peak'])} peak")
        for h in self.holes:
            # a hole is never "here": a piece still marked local whose file is gone is missing
            word = "file missing" if h["status"] == "local" else _STATUS_WORDS.get(h["status"], h["status"])
            print(f"  - {h['piece']:<36} {word}"
                  + (f"  via {h['route_kind']}" if h.get("route_kind") else "")
                  + (f"  {human(h['bytes'])}" if h.get("bytes") else ""))

    def __repr__(self) -> str:
        return (f"Puzzle({self.blueprint.name}, {len(self.built)} built, {len(self.holes)} holes, "
                f"{self.n_obs:,} x {self.n_vars:,})")


# ---------------------------------------------------------------------------
# Piece readers — one small adapter per container
# ---------------------------------------------------------------------------

def _decode(v):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v])


def _obs_strings(v, n: int):
    """An obs column as a nullable Arrow string array: None and NaN stay null, not 'nan'."""
    import pyarrow as pa
    a = np.asarray(v)
    if a.dtype.kind in ("U", "S"):
        return pa.array(_as_str(a).tolist(), pa.string())
    if a.dtype.kind == "f":
        return pa.array([None if np.isnan(x) else (str(int(x)) if float(x).is_integer() else str(x))
                         for x in a.tolist()], pa.string())
    if a.dtype.kind == "O":
        return pa.array([None if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
                         for x in a.tolist()], pa.string())
    return pa.array([str(x) for x in a.tolist()], pa.string())


def _as_str(a) -> np.ndarray:
    """A numpy array of Python str, from bytes (S), str (U) or object arrays — vectorised."""
    a = np.asarray(a)
    if a.dtype.kind == "S":
        return np.char.decode(a, "utf-8")
    if a.dtype.kind == "U":
        return a
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a.tolist()], dtype=object)


class _StreamWriter:
    """Batch-at-a-time Parquet or Arrow IPC writer, so an output never has to exist in memory."""

    def __init__(self, path: str, schema, fmt: str):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.path, self.fmt, self.rows, self._closed = path, fmt, 0, False
        if fmt == "parquet":
            self._sink = None
            self._w = pq.ParquetWriter(path, schema, compression="zstd")
        elif fmt == "arrow":
            self._sink = pa.OSFile(path, "wb")
            self._w = pa.ipc.new_file(self._sink, schema)
        else:
            raise ValueError(f"a built puzzle streams to 'parquet' or 'arrow', not {fmt!r}")

    def write(self, batch) -> None:
        if batch.num_rows:
            self._w.write_batch(batch)
            self.rows += batch.num_rows

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._w.close()
        finally:
            if self._sink is not None:
                self._sink.close()

    def close_quietly(self) -> None:
        """Release the file after a failure; whatever else went wrong is the error to report."""
        try:
            self.close()
        except Exception:
            pass


def _ipc_batches(path: str):
    """The record batches of an Arrow IPC file, one at a time, memory-mapped."""
    import pyarrow as pa
    with pa.memory_map(path, "rb") as src:
        rd = pa.ipc.open_file(src)
        for i in range(rd.num_record_batches):
            yield rd.get_batch(i)


def _remove(*paths: str) -> None:
    for pth in paths:
        try:
            os.remove(pth)
        except OSError:
            pass


def _safe_name(pid: str) -> str:
    """A piece id as a file/region name: the characters of `read_descriptions` ids, nothing else."""
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", str(pid))[:150].strip("_") or "piece"


def _hole(p: "Piece", r: Optional["Resolution"], status: str, note: Optional[str] = None) -> dict:
    """A hole in a puzzle for a piece that is not (or no longer) here: its route, priced, and why."""
    return {"piece": p.id, "status": status, "route_kind": r.route_kind if r else "",
            "route_url": r.route_url if r else "", "bytes": r.bytes if r else 0,
            "cost_usd": r.cost_usd if r else 0.0, "accessions": p.accession_ids(), "source": p.source,
            "note": note if note is not None else (r.note if r else "no resolution"),
            "plan": next((pl.explain() for pl in r.plans.values()), "") if r else ""}


def _unreadable(p: "Piece", path: str, note: str, candidates: Sequence[str]) -> dict:
    """A hole for a piece whose file is here but cannot be used."""
    return {"piece": p.id, "status": "unreadable", "route_kind": "", "route_url": path,
            "bytes": os.path.getsize(path) if os.path.isfile(path) else 0, "cost_usd": 0.0,
            "accessions": p.accession_ids(), "source": p.source, "note": note, "plan": "",
            "candidates": list(candidates)}


def _why(errors: dict) -> str:
    return "; ".join(f"{path}: {err}" for path, err in errors.items())


def _unwrap_bundle(path: str) -> str:
    """What to open for a path: a file, or a directory holding the matrix triple — or, for a Cell
    Ranger bundle (`<dir>/outs/…`), its `filtered_feature_bc_matrix.h5` or matrix directory."""
    if not os.path.isdir(path):
        return path
    if any(os.path.exists(os.path.join(path, n)) for n in ("matrix.mtx.gz", "matrix.mtx")):
        return path
    for base in (path, os.path.join(path, "outs")):
        if not os.path.isdir(base):
            continue
        names = sorted(os.listdir(base))
        for stem in ("filtered_feature_bc_matrix", "raw_feature_bc_matrix"):
            for n in names:
                low = n.lower()
                full = os.path.join(base, n)
                if low.endswith(stem + ".h5") and os.path.isfile(full):
                    return full
            if stem in names and os.path.isdir(os.path.join(base, stem)):
                return os.path.join(base, stem)
    return path


class _PieceReader:
    """Uniform face over the containers a build understands.

    Two independent streams, both bounded by the reader's own budget lease: `coo_batches()`
    yields (cell, gene, value) triples already mapped onto the shared feature axis — no CSR is
    ever assembled for the whole piece — and `obs_batches()` yields the observation metadata in
    strides, read from the file as it goes.
    """

    STRIDE = 65_536

    def __init__(self, path: str, budget: Optional[Budget], blueprint: Blueprint, chunk_bytes=None):
        self.path = path
        self.opened = path = _unwrap_bundle(path)         # a Cell Ranger bundle opens its matrix
        self.kind = ""
        self.n_obs = 0
        self.var_names: np.ndarray = np.zeros(0, dtype=object)
        self._obs_names: Optional[np.ndarray] = None       # in memory for 10x/mtx; sliced from HDF5 for h5ad
        self._obs_cols: list = []
        self._cats: dict = {}
        self._reader = None
        low = path.lower()
        kw = {"budget": budget}
        if chunk_bytes:
            kw["chunk_bytes"] = chunk_bytes
        if low.endswith(".h5ad"):
            from .formats import H5adReader
            self.kind = "h5ad"
            self._reader = H5adReader(path, layer=blueprint.layer, **kw)
            self.n_obs = self._reader.n_obs
            self.var_names = np.asarray(self._reader.var_names, dtype=object)
            f = self._reader.f
            idx = f["obs"].attrs.get("_index", "_index")
            self._idx = idx.decode() if isinstance(idx, bytes) else idx
            self._obs_cols = [k for k in f["obs"].keys() if k != self._idx and not k.startswith("__")]
        elif low.endswith((".h5", ".hdf5")):
            from .tenx import TenXMatrix
            self.kind = "10x-h5"
            r = TenXMatrix(path, **kw)
            mod = blueprint.modality if blueprint.modality in r.modalities else None
            if mod is None and len(r.modalities) > 1:
                mod = next(iter(r.modalities))
            r.modality = mod
            self._reader = r
            self.n_obs = r.n_cells
            names = r.features(mod)
            if blueprint.key == "ensembl":
                ids = r.feature_ids if mod is None else (
                    r.feature_ids[r.modalities[mod].row_start:r.modalities[mod].row_end])
                names = np.where([id_type(x) == "ensembl" for x in ids], ids, names)
            self.var_names = np.asarray(names, dtype=object)
            self._obs_names = np.asarray(r.barcodes, dtype=object)
        elif os.path.isdir(path) or low.endswith((".mtx", ".mtx.gz")):
            from .tenx import MtxTriple
            self.kind = "mtx"
            r = MtxTriple(path, **kw)
            self._reader = r
            self.n_obs = r.n_cells
            names = r.feature_ids if blueprint.key == "ensembl" and all(
                id_type(x) == "ensembl" for x in r.feature_ids[:50]) else r.feature_names
            self.var_names = np.asarray(names, dtype=object)
            self._obs_names = np.asarray(r.barcodes, dtype=object)
        else:
            raise ValueError(f"{path}: not a container a build can align (h5ad, 10x .h5, or an mtx triple)")

    def obs_column_names(self) -> list:
        return list(self._obs_cols)

    def source_path(self) -> str:
        """The file whose bytes this piece is made of: the matrix of a triple, else what was opened."""
        if self.kind == "mtx":
            return getattr(self._reader, "mtx", None) or self.opened
        return self.opened

    # ---- X ----
    def coo_batches(self, positions: np.ndarray, base: int = 0):
        """(cell, gene, value) per block — cell global (`base` + local, int64), gene on the shared
        axis (int32), value as stored.

        One gather through a source-column → shared-axis lookup table, then a mask: no CSR
        column selection (which copies the block) and no per-piece matrix assembly. Every
        intermediate is dropped before the block is handed over, so the live set while the
        consumer writes is the three output arrays, and the reader's next block is not read on
        top of the previous one.
        """
        present = positions >= 0
        lut = np.full(len(self.var_names), -1, dtype=np.int32)   # source column -> shared index
        lut[positions[present]] = np.flatnonzero(present)

        def _emit(row, col, val, start):
            gene = lut[col]
            ok = gene >= 0
            if ok.all():
                cell = np.add(row, base + start, dtype=np.int64)
                return cell, gene, val
            cell = np.add(row[ok], base + start, dtype=np.int64)
            return cell, gene[ok], val[ok]

        if self.kind == "h5ad":
            for blk in self._reader:
                X = blk.pop("X")
                if hasattr(X, "tocoo"):
                    coo = X.tocoo(copy=False)
                    out = _emit(coo.row, coo.col, coo.data, blk["obs_start"])
                    del coo
                else:
                    row, col = np.nonzero(X)
                    out = _emit(row, col, np.asarray(X)[row, col], blk["obs_start"])
                    del row, col
                del X, blk
                yield out
                del out
        elif self.kind == "10x-h5":
            for blk in self._reader:                       # X is features × cells, CSC
                coo = blk.pop("X").tocoo(copy=False)
                out = _emit(coo.col, coo.row, coo.data, blk["cell_start"])
                del coo, blk
                yield out
                del out
        else:                                              # mtx: COO stream, rows are features
            for blk in self._reader:
                out = _emit(blk["col"], blk["row"], blk["val"], 0)
                del blk
                yield out
                del out

    # ---- obs ----
    def _obs_slice(self, name: str, start: int, end: int) -> np.ndarray:
        """One obs column, rows start:end, through the shared AnnData decoder (categorical codes
        of -1 become None, nullable groups are honoured, strings are str)."""
        from .formats import read_obs_column
        return read_obs_column(self._reader.f, name, start, end, cache=self._cats)

    def obs_batches(self, columns: Sequence[str], stride: Optional[int] = None):
        """(start, end, names, {column: values}) in strides — never the whole piece at once."""
        stride = stride or self.STRIDE
        for start in range(0, self.n_obs, stride):
            end = min(start + stride, self.n_obs)
            if self.kind == "h5ad":
                f = self._reader.f
                names = _as_str(f[f"obs/{self._idx}"][start:end]) if f"obs/{self._idx}" in f \
                    else np.array([str(i) for i in range(start, end)])
                vals = {c: self._obs_slice(c, start, end) for c in columns}
            else:
                names = self._obs_names[start:end]
                vals = {}
            yield start, end, names, vals

    def close(self):
        try:
            self._reader.close()
        except Exception:
            pass


_TABLE_SUFFIXES = (".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt", ".txt.gz", ".parquet")
_GENE_COLUMNS = ("gene", "genes", "symbol", "gene_symbol", "gene symbol", "official_symbol", "hgnc_symbol",
                 "identifier", "gene_id", "ensembl", "ensembl_id", "target", "target_gene")

#: "SYMBOL (entrez)" — the header of a gene column in a DepMap-style matrix.
_WIDE_HEADER = re.compile(r"^[A-Za-z][A-Za-z0-9\-\.]{0,19} \(\d+\)$")


def _is_table(path: str) -> bool:
    return path.lower().endswith(_TABLE_SUFFIXES)


def _table_piece(path: str, dest_dir: str, name: str, key: str, genes: GeneMap, budget: Optional[Budget],
                 chunk_bytes=None) -> dict:
    """Stream a CSV/TSV/Parquet table into `<dest_dir>/<name>.parquet`, harmonising gene names.

    Long tables (one gene per row) get a `gene_<key>` column beside their own gene column; wide
    tables (one gene per column, DepMap style ``"CD274 (29126)"``) get their gene columns renamed
    into the key space, with the original header kept in the Parquet schema metadata. Everything
    is done batch by batch: the chunk is the budget's floor rule, never the whole file.
    """
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq
    low = path.lower()
    if budget is not None and chunk_bytes is None:
        avail = budget.available("ram")
        chunk = max(min(1 << 20, avail), avail // 4)
    else:
        chunk = int(chunk_bytes or (64 << 20))
    dest = os.path.join(dest_dir, name + ".parquet")
    info = {"reader": "table", "rows": 0, "gene_column": "", "n_gene_columns": 0, "n_gene_columns_mapped": 0,
            "chunks": 0, "resident_peak": 0}
    lease = None

    def _lease(n):
        nonlocal lease
        if lease is not None:
            lease.release()
            lease = None
        if budget is not None and n > 0:
            lease = budget.acquire("ram", str(int(n)))

    if low.endswith(".parquet"):
        pf = pq.ParquetFile(path)
        batches = pf.iter_batches(batch_size=max(1024, int(chunk // max(1, pf.metadata.num_columns * 8))))
        first_schema = pf.schema_arrow
    else:
        delim = "\t" if ".tsv" in low or ".txt" in low else ","
        reader = pacsv.open_csv(path, read_options=pacsv.ReadOptions(block_size=int(min(chunk, 1 << 30))),
                                parse_options=pacsv.ParseOptions(delimiter=delim))
        batches = iter(reader)
        first_schema = reader.schema
    names = list(first_schema.names)
    # wide: many columns that look like genes ("CD274 (29126)", an Ensembl id, a name the map
    # knows). A column that only *survives* canonical() — every symbol-shaped header does, as
    # itself — is not evidence of a gene: a long hit table with Gene/Score/FDR/… columns is long.
    canon_cols = {}
    gene_shaped = 0
    for c in names:
        base = re.sub(r"\s*\(\d+\)$", "", str(c)).strip()
        cid = genes.canonical(base, key) if base else ""
        if cid:
            canon_cols[c] = cid
        if (base and base in genes) or _WIDE_HEADER.match(str(c)) or id_type(base) in ("ensembl", "entrez"):
            gene_shaped += 1
    wide = gene_shaped >= 20 or (len(names) >= 5 and gene_shaped >= 0.5 * len(names))
    gene_col = next((c for c in names if _norm(c) in _GENE_COLUMNS), "") if not wide else ""
    info["gene_column"] = gene_col
    writer = None
    try:
        for b in batches:
            b = pa.RecordBatch.from_pandas(b) if not isinstance(b, pa.RecordBatch) else b
            _lease(b.nbytes)
            if wide:
                new_names = [canon_cols.get(c, c) for c in b.schema.names]
                # two source columns may map to one gene; keep the first, suffix the rest
                seen: dict = {}
                for i, n in enumerate(new_names):
                    if n in seen:
                        new_names[i] = f"{n}__{seen[n]}"
                        seen[n] += 1
                    else:
                        seen[n] = 1
                b = pa.RecordBatch.from_arrays(b.columns, names=new_names)
                info["n_gene_columns"] = gene_shaped
                info["n_gene_columns_mapped"] = len(canon_cols)
            elif gene_col:
                col = b.column(b.schema.get_field_index(gene_col)).to_pylist()
                mapped = [genes.canonical(str(x), key) or "" for x in col]
                info["n_rows_mapped"] = info.get("n_rows_mapped", 0) + sum(1 for m in mapped if m)
                b = b.append_column(f"gene_{key}", pa.array(mapped, pa.string()))
            if writer is None:
                meta = {b"geminga.source": path.encode(), b"geminga.key": key.encode(),
                        b"geminga.original_columns": json.dumps(names).encode()}
                writer = pq.ParquetWriter(dest, b.schema.with_metadata(meta), compression="zstd")
            writer.write_batch(b)
            info["rows"] += b.num_rows
            info["chunks"] += 1
        if writer is None:                                # empty file
            writer = pq.ParquetWriter(dest, first_schema, compression="zstd")
    finally:                                              # a failed table leaves no open handle behind
        if writer is not None:
            writer.close()
        _lease(0)
    if budget is not None:
        info["resident_peak"] = int(budget.stats()["ram"]["peak"])
    info["path"] = dest
    return info


class Orchestrator:
    """Describe → resolve → harmonise/annotate → build, with everything written down.

    ::

        orc = g.orchestrator(catalog=g.catalog("~/inokan/catalog.jsonl"),
                             library=g.library("~/inokan/library.jsonl"), budget=g.Budget(ram="8GB"))
        orc.describe("InoKAN_CD_nodes_3_culture_organoids.xlsx", sheet="Datasets")
        orc.learn_genes(xlsx, sheet="Nodes", symbol="primary_gene", members="human_genes",
                        ensembl="ensembl_human (Geneformer key)", aliases=("node_id",), group="node_id")
        orc.resolve(search_dirs=["/mnt/nas/inokan"])
        orc.print_report()
        orc.annotate()
        orc.coverage(genes=["CD276", "CD274", "CD47"]).print()
        bp = orc.blueprint("cd-core", key="ensembl", panel=orc.panel_genes("ensembl"))
        puzzle = orc.build(bp, "~/handoff/cd-core")
    """

    def __init__(self, *, vault=None, catalog=None, library=None, budget: Optional[Budget] = None,
                 harmonizer: Optional[Harmonizer] = None):
        self.vault, self.catalog, self.library, self.budget = vault, catalog, library, budget
        self.harmonizer = harmonizer or Harmonizer()
        self.pieces: dict[str, Piece] = {}
        self.resolutions: dict[str, Resolution] = {}
        self.panels: dict[str, list] = {}
        self.sources: list[str] = []

    # ---- describe ----
    def describe(self, source, sheet: Optional[str] = None, **kw) -> list:
        pieces = read_descriptions(source, sheet, harmonizer=self.harmonizer, **kw)
        for p in pieces:
            if p.id in self.pieces:                        # a second sheet reusing an id: never overwrite
                n = len(self.pieces)
                while f"{p.id}_{n}" in self.pieces:
                    n += 1
                p.id = f"{p.id}_{n}"
            self.pieces[p.id] = p
        self.sources.append(pieces[0].source.rsplit(":", 1)[0] if pieces else str(source))
        return pieces

    def add(self, piece: Piece) -> Piece:
        self.pieces[piece.id] = piece
        return piece

    def learn_genes(self, source, sheet: Optional[str] = None, **from_table_kw) -> GeneMap:
        """Teach the harmoniser gene identities from a table (a project's node sheet, an HGNC
        dump via GeneMap.from_hgnc, a features.tsv via GeneMap.from_features)."""
        rows, cols, _ = _rows_from(source, sheet)
        gm = GeneMap.from_table(rows, **from_table_kw)
        self.harmonizer.genes.merge(gm)
        return self.harmonizer.genes

    def learn_panel(self, source, sheet: Optional[str] = None, *, panel: str = "panel",
                    gene: str = "gene_symbol", name: Optional[str] = None) -> dict:
        """Antibody panels: panel name -> gene symbols, so 'protein measured' can be inferred
        for any piece whose `panel` names one."""
        rows, cols, _ = _rows_from(source, sheet)
        for r in rows:
            pn = str(r.get(panel) or "").strip()
            gs = str(r.get(gene) or "").strip()
            if pn and gs and (name is None or pn == name):
                self.panels.setdefault(pn, [])
                if gs not in self.panels[pn]:
                    self.panels[pn].append(gs)
        return self.panels

    def panel_genes(self, key: str = "symbol") -> list:
        """Every gene the harmoniser knows, in one key space — the natural build panel for a
        node-centred project."""
        gm = self.harmonizer.genes
        return list(dict.fromkeys(v for v in (gm._get(r, key) for r in gm.records) if v))

    # ---- resolve ----
    def resolve(self, *, search_dirs: Sequence[str] = (), offline: bool = True, timeout: float = 20.0) -> dict:
        """Say where every piece is — here, one fetch away, behind a DAC, paper only — without
        moving a byte. See :func:`resolve_pieces`: a file under `search_dirs` (or a vault member)
        belongs to a piece when one of the piece's accessions stands alone in its name, and a
        path that has gone is not there."""
        self.resolutions = resolve_pieces(list(self.pieces.values()), vault=self.vault, catalog=self.catalog,
                                          search_dirs=search_dirs, offline=offline, timeout=timeout)
        return self.resolutions

    def attach(self, piece_id: str, path: str, note: str = "") -> Resolution:
        """The analyst says: this local file is that piece. Recorded in the catalog too."""
        p = self.pieces[piece_id]
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if path not in p.local:
            p.local.append(path)
        r = self.resolutions.get(piece_id) or Resolution(piece=piece_id)
        if path not in r.local_paths:
            r.local_paths.append(path)
        r.status = p.status = "local"
        self.resolutions[piece_id] = r
        if self.catalog is not None:
            self.catalog.record(piece_id, path, url=(p.accessions[0].id if p.accessions else ""),
                                note=note or p.title, facets=p.facets, local_paths=[path],
                                tags=list(p.modalities), accessions=p.accession_ids())
        return r

    # ---- report ----
    def by_status(self) -> dict:
        out = {s: [] for s in RESOLUTION_STATUSES}
        for p in self.pieces.values():
            out.setdefault(p.status, []).append(p)
        return out

    def report(self) -> dict:
        bs = self.by_status()
        known = [self.resolutions[p.id] for s in ("streamable", "fetchable") for p in bs[s]
                 if p.id in self.resolutions]
        return {
            "pieces": len(self.pieces),
            "by_status": {s: len(v) for s, v in bs.items()},
            "local_bytes": sum(os.path.getsize(lp) for p in bs["local"] for lp in p.local
                               if os.path.isfile(lp)),
            "fetch_bytes_known": sum(r.bytes for r in known),
            "fetch_sizes_known": sum(1 for r in known if r.bytes),
            "fetch_cost_usd": sum(r.cost_usd for r in known),
            "fetch_cost_unknown": sum(1 for r in known if not r.cost_known),
            "needs_tools": sorted({r.needs for r in known if r.needs and r.needs != "approval"}),
            "controlled_via": sorted({(self.resolutions[p.id].route_kind or "?") for p in bs["controlled"]
                                      if p.id in self.resolutions}),
            "controlled_components": sum(1 for r in self.resolutions.values()
                                         if r.controlled and r.status != "controlled"),
            "assays": {a: sum(1 for p in self.pieces.values() if p.primary_assay == a)
                       for a in sorted({p.primary_assay for p in self.pieces.values()})},
        }

    def print_report(self, limit: int = 80) -> None:
        rep = self.report()
        bs = self.by_status()
        n = rep["pieces"]
        print(f"gh'è minga — {n} piece(s) described from {len(self.sources)} source(s)")
        print(f"  here                 {len(bs['local']):>3}   {human(rep['local_bytes'])} on disk")
        print(f"  one fetch away       {len(bs['streamable']) + len(bs['fetchable']):>3}   "
              f"{len(bs['streamable'])} streamable · sizes known for {rep['fetch_sizes_known']} "
              f"({human(rep['fetch_bytes_known'])}) · metered ~${rep['fetch_cost_usd']:.2f}"
              + (f" + {rep['fetch_cost_unknown']} metered of unknown size" if rep["fetch_cost_unknown"] else "")
              + (f" · install {', '.join(rep['needs_tools'])}" if rep["needs_tools"] else ""))
        print(f"  behind a DAC         {len(bs['controlled']):>3}   "
              + (", ".join(rep["controlled_via"]) if rep["controlled_via"] else "-")
              + (f"   (+{rep['controlled_components']} open pieces with a controlled-access component)"
                 if rep["controlled_components"] else ""))
        print(f"  paper only           {len(bs['manual']):>3}   landing page, DOI or URL — no file to price")
        print(f"  unresolved           {len(bs['unresolved']):>3}   nothing accession-shaped in the row")
        print(f"\n  {'id':<40} {'assay':<22} {'cells':>10}  {'status':<11} via")
        order = [p for s in RESOLUTION_STATUSES for p in bs[s]]
        for p in order[:limit]:
            print("  " + p.line())
        if len(order) > limit:
            print(f"  … {len(order) - limit} more")

    # ---- annotate ----
    def annotate(self, *, to_catalog: bool = True, to_library: bool = True, intent: str = "need-data") -> dict:
        """Write every piece into the catalog (facets, accessions, status) and every hole into the
        library as a gene-list flag with the intent `need-data`, so `Library.leads()` can hunt for
        it later. Idempotent: a hole that already has an open flag gets that flag updated, and a
        hole that has since arrived gets its flag resolved. Returns
        {piece: {"catalog": key, "library": selection id}}."""
        out: dict[str, dict] = {}
        existing: dict[str, Any] = {}
        if to_library and self.library is not None:
            for sel in self.library.query(intent=intent):
                pid = sel.metadata.get("piece") if isinstance(sel.metadata, dict) else None
                if pid and sel.status != "resolved" and pid not in existing:
                    existing[pid] = sel
        for p in self.pieces.values():
            rec: dict = {}
            r = self.resolutions.get(p.id)
            if to_catalog and self.catalog is not None:
                target = p.local[0] if p.local else (p.accessions[0].id if p.accessions else p.id)
                e = self.catalog.record(
                    p.id, target, url=(r.route_url if r and r.route_url.startswith(("http", "ftp")) else ""),
                    note=p.title or p.priority_reason, tags=list(p.modalities) + ([p.context] if p.context else []),
                    facets=p.facets, local_paths=list(p.local),
                    accessions=p.accession_ids(), status=p.status, access=p.access, formats=p.formats,
                    perturbation=p.perturbation, genes=p.genes, expected=p.expected,
                    source=p.source, reference=p.reference, priority=p.priority)
                rec["catalog"] = e.key
            if to_library and self.library is not None:
                prior = existing.get(p.id)
                if p.status == "local":
                    if prior is not None:
                        self.library.resolve(prior, note=f"now local: {p.local[0] if p.local else ''}")
                        rec["library"] = prior.id
                        rec["library_resolved"] = True
                else:
                    meta = {"piece": p.id, "accessions": p.accession_ids(), "status": p.status,
                            "assay": p.primary_assay, "bytes": r.bytes if r else 0,
                            "cost_usd": r.cost_usd if r else 0.0, "source": p.source}
                    note = (f"{_STATUS_WORDS.get(p.status, p.status)}; "
                            + (f"via {r.route_kind} {r.route_url}" if r and r.route_kind else "no route"))
                    if prior is not None:
                        self.library.update(prior, note=note, metadata={**prior.metadata, **meta},
                                            tags=sorted(set(prior.tags) | {"orchestrator", p.status} | set(p.modalities)))
                        rec["library"] = prior.id
                    else:
                        from .library import flag_genes
                        sel = flag_genes(p.all_genes, name=f"{p.id}: not here",
                                         organism=p.facets.get("organism") or "Homo sapiens",
                                         intent=intent, note=note,
                                         tags=["orchestrator", p.status] + list(p.modalities),
                                         metadata=meta, lib=self.library)
                        rec["library"] = sel.id
            out[p.id] = rec
        return out

    # ---- coverage ----
    def coverage(self, genes: Optional[Sequence[str]] = None, pieces: Optional[Sequence[str]] = None,
                 key: str = "symbol", measured: Optional[dict] = None) -> Coverage:
        """Gene × piece evidence, computed from the descriptions (and, when given, from
        `measured` = {piece: var names} of local matrices — a build's alignment report supplies
        this automatically).

        A name that is a complex (CD3 = CD3D + CD3E) stands for all its members, on either side:
        a piece that lists CD3 gives evidence for CD3D and CD3E, and a requested CD3 is evidenced
        by any one of its members."""
        gm = self.harmonizer.genes
        ids = list(pieces) if pieces else list(self.pieces)
        if genes is None:
            genes = list(dict.fromkeys(g for pid in ids for g in self.pieces[pid].all_genes))
        memo: dict = {}

        def _expand(name):                               # the same few names recur in every piece
            c = memo.get(name)
            if c is None:
                c = memo[name] = gm.expand([name], key)
            return c

        canon = {g: set(_expand(g)) or {str(g)} for g in genes}
        codes: dict = {}
        unmapped: dict = {}

        def _canon(names, pid):
            out = set()
            for n in names:
                c = _expand(n)
                if c:
                    out.update(c)
                else:
                    out.add(n)
                    unmapped.setdefault(n, []).append(pid)
            return out

        for pid in ids:
            p = self.pieces[pid]
            protein = _canon(p.genes.get("protein", []), pid)
            for pn, gl in self.panels.items():
                if p.panel and (pn == p.panel or _norm(pn) in _norm(p.panel)):
                    for g in gl:
                        protein.update(_expand(g) or [g])
            perturbed = _canon(p.genes.get("perturbed", []), pid)
            focus = _canon(p.genes.get("focus", []), pid)
            genome_wide = "genome" in _norm(p.perturbation.get("scope", ""))
            present = set()
            if measured and pid in measured:
                present = {gm.canonical(g, key) or g for g in measured[pid]}
            for g in genes:
                c = canon[g]
                s = ""
                if not c.isdisjoint(protein):
                    s += "P"
                if not c.isdisjoint(perturbed):
                    s += "X"
                elif genome_wide and p.perturbation.get("type"):
                    s += "x"
                if not c.isdisjoint(focus):
                    s += "F"
                if not c.isdisjoint(present):
                    s += "R"
                if s:
                    codes[(g, pid)] = s
        return Coverage(genes=list(genes), pieces=ids, codes=codes, key=key,
                        unmapped={n: sorted(set(v)) for n, v in unmapped.items() if len(gm)})

    # ---- blueprint & build ----
    def blueprint(self, name: str, pieces: Optional[Sequence[str]] = None, *, key: str = "symbol",
                  how: str = "intersection", panel: Optional[Sequence[str]] = None,
                  obs: Optional[Sequence[str]] = None, modality: str = "RNA", layer: Optional[str] = None,
                  note: str = "", local_only: bool = False) -> Blueprint:
        ids = list(pieces) if pieces else [p.id for p in self.pieces.values()
                                            if (not local_only or p.status == "local")]
        missing = [i for i in ids if i not in self.pieces]
        if missing:
            raise KeyError(f"unknown piece(s) {missing}")
        bp = Blueprint(name=name, pieces=ids, key=key, how=how,
                       panel=list(panel) if panel is not None else None,
                       obs=list(obs) if obs is not None else
                       ["sample", "donor", "condition", "perturbation", "guide", "cell_type"],
                       modality=modality, layer=layer, note=note)
        return bp

    def build(self, blueprint: Blueprint, dest: str, *, fmt: str = "parquet", budget: Optional[Budget] = None,
              chunk_bytes=None, overwrite: bool = True, allow_empty: bool = False) -> Puzzle:
        """Execute a blueprint into a Harvest at `dest`. Pieces that are here get aligned and
        streamed; pieces that are not become holes in the record, never errors.

        What a build promises about its record:

        * a piece whose files are listed in ``Piece.local`` is opened from the first of them that
          opens (matrices before tables); every path considered is recorded as ``candidates``;
        * a piece whose files have since vanished is a hole of status ``missing``; one that cannot
          be opened, or that fails while it is being streamed, is a hole of status ``unreadable``
          and leaves no partial output — its observation rows are not in ``obs`` either;
        * each built piece carries ``source_sha256`` (the file its numbers came from — for a
          MatrixMarket triple, ``matrix.mtx[.gz]``) and ``output_sha256`` (the Parquet/Arrow file
          written); the old ``sha256`` key is the output hash, as it always was;
        * the manifest and dispatch files are written whatever happens, and regions left by an
          earlier build into the same `dest` that this build did not produce are dropped from
          it (and their files removed); with ``overwrite=False`` a `dest` that already holds
          regions is refused up front;
        * a blueprint whose pieces share no features is a ``ValueError`` — an n_vars=0 puzzle
          would be silent data loss — unless ``allow_empty=True``.
        """
        import pyarrow as pa
        from .handoff import FORMATS, ROI, Harvest, _sha256
        if fmt not in ("parquet", "arrow"):
            raise ValueError(f"a built puzzle streams to 'parquet' or 'arrow', not {fmt!r}")
        budget = budget or self.budget
        h = Harvest(dest, note=blueprint.note or f"puzzle {blueprint.name}", producer="geminga.orchestrate")
        if h.rois and not overwrite:
            raise ValueError(f"{h.path} already holds {len(h.rois)} region(s) from an earlier build; "
                             f"pass overwrite=True to rebuild it")
        self.harmonizer.log = []                         # the provenance below is this build's alone
        for left in os.listdir(h.path):                  # staging files of a build that was killed outright
            if left.startswith(".obs_") and left.endswith(".part"):
                _remove(os.path.join(h.path, left))
        t0 = time.time()
        if budget is not None:
            budget.reset_peak()
        suffix = FORMATS[fmt]["suffix"]
        fnames: dict[str, str] = {}                      # piece id -> the name its files and regions carry
        for pid in blueprint.pieces:
            base = nm = _safe_name(pid)
            k = 1
            while nm in fnames.values():
                k += 1
                nm = f"{base}_{k}"
            fnames[pid] = nm

        # 1. which pieces are here, and their feature axes
        readers: dict[str, _PieceReader] = {}
        tables: list[tuple] = []                         # (piece, [table paths to try], errors so far)
        holes: list[dict] = []
        features: dict[str, list] = {}
        tried: dict[str, dict] = {}                      # piece -> {"candidates": [...], "errors": {path: why}}
        for pid in blueprint.pieces:
            p = self.pieces[pid]
            r = self.resolutions.get(pid)
            existing = [lp for lp in p.local if os.path.exists(lp)]
            if not existing:
                gone = bool(p.local) or p.status == "local"
                holes.append(_hole(p, r, "missing" if gone else p.status,
                                   note=("file missing: " + ", ".join(p.local)) if p.local else
                                   ("file missing" if gone else None)))
                continue
            # A count matrix is the piece; a table beside it is annotation. Matrices first.
            cands = [c for c in existing if not _is_table(c)] + [c for c in existing if _is_table(c)]
            errors: dict = {}
            tried[pid] = {"candidates": cands, "errors": errors}
            opened = False
            for i, path in enumerate(cands):
                if _is_table(path):
                    tables.append((pid, cands[i:], errors))
                    opened = True
                    break
                try:
                    rd = _PieceReader(path, budget, blueprint, chunk_bytes)
                except Exception as e:
                    errors[path] = f"{type(e).__name__}: {e}"
                    continue
                readers[pid] = rd
                features[pid] = list(rd.var_names)
                opened = True
                break
            if not opened:
                holes.append(_unreadable(p, cands[0], _why(errors), cands))

        # 2. one feature axis
        aln = None
        try:
            if readers:
                aln = self.harmonizer.align(features, key=blueprint.key, how=blueprint.how, panel=blueprint.panel)
                if not allow_empty and (aln.n == 0 or not any(rep["n_kept"] for rep in aln.report.values())):
                    why = ("none of the panel's genes is in any of them" if blueprint.panel
                           else f"the {blueprint.how} of their features is empty")
                    raise ValueError(
                        f"blueprint {blueprint.name!r}: the alignment has no shared features (zero variables): the "
                        f"{len(readers)} piece(s) that are here share no features under key={blueprint.key!r} "
                        f"({why}). Check the key and the gene map, or pass allow_empty=True to build it anyway.\n"
                        f"{aln.explain()}")
            elif blueprint.panel:
                aln = self.harmonizer.genes.align({}, key=blueprint.key, panel=blueprint.panel)
        except BaseException:
            for rd in readers.values():
                rd.close()
            raise

        # 3. stream every local piece into the harvest, block by block — X as COO against the
        #    shared axis, obs in strides — so nothing larger than a block is ever held
        x_schema = pa.schema([("cell", pa.int64()), ("gene", pa.int32()), ("value", pa.float32())])
        obs_fields = ["cell_id", "piece"] + list(blueprint.obs) + [f"facet_{f}" for f in FACETS]
        obs_schema = pa.schema([(c, pa.string()) for c in obs_fields])
        obs_path = os.path.join(h.path, "obs" + suffix)
        built: list[dict] = []
        produced: set = set()                            # the regions this build wrote
        offset = 0
        n_obs = 0
        obs_w = None
        prov = None
        finished = False
        error = None

        def _register(name: str, path: str, rows: int, source: str, kind: str, note: str, stat: dict,
                      format: str = fmt) -> ROI:
            roi = ROI(name=name, source=source, note=note, targets=["puzzle"], kind=kind, stat=stat,
                      path=os.path.relpath(path, h.path), format=format, rows=rows, bytes=os.path.getsize(path),
                      sha256=_sha256(path), written=time.time())
            h.rois[name] = roi
            produced.add(name)
            return roi

        try:
            obs_w = _StreamWriter(obs_path, obs_schema, fmt)
            for pid, rd in list(readers.items()):
                p = self.pieces[pid]
                roi_name = f"X_{fnames[pid]}"
                pos = aln.positions[pid]
                cols = rd.obs_column_names()
                mapping = self.harmonizer.obs_mapping(pid, cols) if cols else {}
                if budget is not None:
                    budget.reset_peak()
                x_path = os.path.join(h.path, roi_name + suffix)
                part = os.path.join(h.path, f".obs_{fnames[pid]}.part")      # this piece's obs rows, until it is whole
                xw = ow = None
                failure = None
                try:
                    xw = _StreamWriter(x_path, x_schema, fmt)
                    n_chunks = 0
                    for cell, gene, val in rd.coo_batches(pos, base=offset):
                        xw.write(pa.record_batch([pa.array(cell), pa.array(gene),
                                                  pa.array(val.astype(np.float32, copy=False))], schema=x_schema))
                        del cell, gene, val
                        n_chunks += 1
                    xw.close()
                    x_rows = xw.rows
                    peak = budget.stats()["ram"]["peak"] if budget is not None else 0
                    facet_vals = [p.facets.get(f, "") for f in FACETS]
                    ow = _StreamWriter(part, obs_schema, "arrow")
                    for start, end, names, vals in rd.obs_batches([mapping[c] for c in blueprint.obs if c in mapping]):
                        n = end - start
                        arrays = [pa.array([f"{pid}:{x}" for x in names.tolist()], pa.string()),
                                  pa.array([pid] * n, pa.string())]
                        for c in blueprint.obs:
                            src = mapping.get(c)
                            v = vals.get(src) if src else None
                            arrays.append(pa.array([""] * n, pa.string()) if v is None else _obs_strings(v, n))
                        arrays.extend(pa.array([fv] * n, pa.string()) for fv in facet_vals)
                        ow.write(pa.record_batch(arrays, schema=obs_schema))
                    ow.close()
                    source_sha = _sha256(rd.source_path())
                except Exception as e:
                    failure = e
                except BaseException:
                    for w in (xw, ow):
                        if w is not None:
                            w.close_quietly()
                    _remove(x_path, part)
                    h.rois.pop(roi_name, None)
                    raise
                for w in (xw, ow):
                    if w is not None:
                        w.close_quietly()
                if failure is not None:                  # a hole, not a half-written piece
                    _remove(x_path, part)
                    h.rois.pop(roi_name, None)
                    holes.append(_unreadable(p, rd.path, f"{type(failure).__name__}: {failure}",
                                             tried[pid]["candidates"]))
                    del readers[pid]
                    aln.report.pop(pid, None)
                    aln.positions.pop(pid, None)
                    rd.close()
                    continue
                try:
                    for batch in _ipc_batches(part):
                        obs_w.write(batch)
                finally:
                    _remove(part)
                roi = _register(roi_name, x_path, x_rows, rd.path, "piece",
                                f"{p.primary_assay} · {p.reference}".strip(" ·"),
                                {"n_obs": rd.n_obs, "n_vars_in": len(rd.var_names), "n_kept": int((pos >= 0).sum()),
                                 "nnz": x_rows})
                rep = aln.report[pid]
                built.append({"piece": pid, "path": rd.path, "candidates": list(tried[pid]["candidates"]),
                              **({"candidate_errors": dict(tried[pid]["errors"])} if tried[pid]["errors"] else {}),
                              **({"opened": rd.opened} if rd.opened != rd.path else {}),
                              "source_sha256": source_sha, "output_sha256": roi.sha256, "sha256": roi.sha256,
                              "reader": rd.kind, "n_obs": rd.n_obs, "n_vars_in": rep["n_in"],
                              "n_mapped": rep["n_mapped"], "n_duplicate": rep.get("n_duplicate", 0),
                              "n_kept": rep["n_kept"], "unmapped_examples": rep["examples"],
                              "obs_mapping": mapping, "chunks": n_chunks, "resident_peak": int(peak),
                              "output": roi.path, "roi": roi_name, "rows": roi.rows, "bytes": roi.bytes,
                              "source": p.source, "accessions": p.accession_ids(), "facets": dict(p.facets)})
                offset += rd.n_obs
                rd.close()
            obs_w.close()
            _register("obs", obs_path, obs_w.rows, "geminga.orchestrate", "table",
                      "one row per observation across pieces", {"n_obs": obs_w.rows})

            # 3b. table pieces — screens, gene-effect matrices, hit lists — harmonised, not aligned
            for pid, paths, errors in tables:
                p = self.pieces[pid]
                roi_name = f"T_{fnames[pid]}"
                ok = False
                for path in paths:
                    if budget is not None:
                        budget.reset_peak()
                    try:
                        info = _table_piece(path, h.path, roi_name, blueprint.key, self.harmonizer.genes, budget,
                                            chunk_bytes)
                        source_sha = _sha256(path)
                    except Exception as e:
                        _remove(os.path.join(h.path, roi_name + ".parquet"))
                        h.rois.pop(roi_name, None)
                        errors[path] = f"{type(e).__name__}: {e}"
                        continue
                    roi = _register(roi_name, info["path"], info["rows"], path, "piece-table",
                                    f"{p.primary_assay} · {p.reference}".strip(" ·"),
                                    {"rows": info["rows"], "gene_column": info["gene_column"],
                                     "n_gene_columns": info["n_gene_columns"]}, format="parquet")
                    n_in = info["n_gene_columns"] or (info["rows"] if info["gene_column"] else 0)
                    n_ok = info["n_gene_columns_mapped"] or info.get("n_rows_mapped", 0)
                    built.append({"piece": pid, "path": path, "candidates": list(tried[pid]["candidates"]),
                                  **({"candidate_errors": dict(errors)} if errors else {}),
                                  "source_sha256": source_sha, "output_sha256": roi.sha256, "sha256": roi.sha256,
                                  "reader": "table", "n_obs": info["rows"], "n_vars_in": n_in, "n_mapped": n_ok,
                                  "n_kept": n_ok, "unmapped_examples": [], "obs_mapping": {}, "chunks": info["chunks"],
                                  "resident_peak": info["resident_peak"], "output": roi.path, "roi": roi_name,
                                  "rows": roi.rows, "bytes": roi.bytes, "source": p.source,
                                  "accessions": p.accession_ids(), "facets": dict(p.facets),
                                  "gene_column": info["gene_column"]})
                    ok = True
                    break
                if not ok:
                    holes.append(_unreadable(p, paths[0], _why(errors), tried[pid]["candidates"]))

            # 4. obs, var, pieces tables
            n_obs = offset
            if aln is not None:
                var_rows = self.harmonizer.genes.panel_table(aln.index, key=blueprint.key, expand=False)
                at = {c: i for i, c in enumerate(aln.index)}
                for vr in var_rows:
                    vr["id"] = vr["id"] or vr["query"]
                    for pid in readers:
                        vr[f"in_{pid}"] = bool(aln.positions[pid][at[vr["query"]]] >= 0)
                h.add_table("var", pa.Table.from_pylist(var_rows) if var_rows else
                            pa.table({"id": pa.array([], pa.string())}), fmt=fmt, kind="table",
                            targets=["puzzle"], note=f"feature axis keyed by {blueprint.key}", overwrite=overwrite)
                produced.add("var")
            built_ids = {b["piece"] for b in built}
            piece_rows = []
            for pid in blueprint.pieces:
                p = self.pieces[pid]
                piece_rows.append({"piece": pid, "built": pid in built_ids,
                                   "status": p.status, "title": p.title,
                                   "reference": p.reference, "assay": p.primary_assay,
                                   "modalities": ";".join(p.modalities), "accessions": ";".join(p.accession_ids()),
                                   **{f"facet_{f}": p.facets.get(f, "") for f in FACETS},
                                   "n_cells_expected": p.expected.get("n_cells"), "source": p.source})
            h.add_table("pieces", pa.Table.from_pylist(piece_rows), fmt=fmt, kind="table", targets=["puzzle"],
                        note="every piece the blueprint named, built or not", overwrite=overwrite)
            produced.add("pieces")

            # 5. provenance. Regions an earlier build of this dest left behind are not part of this one.
            for name in [n for n, r in h.rois.items() if n not in produced and "puzzle" in (r.targets or [])]:
                stale = h.rois.pop(name)
                full = os.path.join(h.path, stale.path)
                if os.path.dirname(os.path.realpath(full)) == os.path.realpath(h.path):
                    _remove(full)
            from . import __version__
            prov = {
                "schema": BLUEPRINT_SCHEMA,
                "producer": f"geminga {__version__}",
                "built": time.time(), "seconds": round(time.time() - t0, 3),
                "blueprint": blueprint.to_dict(),
                "harmonizer": self.harmonizer.report(),
                "alignment": {"key": aln.key, "how": aln.how, "n": aln.n, "report": aln.report,
                              "panel_missing": aln.panel_missing} if aln else None,
                "pieces": built,
                "holes": holes,
                "n_obs": n_obs, "n_vars": aln.n if aln else 0,
                "budget": budget.stats() if budget is not None else None,
                "sources": list(self.sources),
            }
            with open(os.path.join(h.path, "blueprint.json"), "w", encoding="utf-8") as fh:
                json.dump(prov, fh, indent=2, default=str)
            finished = True
        except BaseException as e:
            error = e
            raise
        finally:
            for rd in readers.values():                  # a piece that died leaves its file closed
                rd.close()
            if obs_w is not None:
                obs_w.close_quietly()
                if "obs" not in produced and os.path.exists(obs_path):
                    _register("obs", obs_path, obs_w.rows, "geminga.orchestrate", "table",
                              "one row per observation across the pieces that finished", {"n_obs": obs_w.rows})
            if not finished:                             # say what was done before it stopped
                try:
                    with open(os.path.join(h.path, "blueprint.json"), "w", encoding="utf-8") as fh:
                        json.dump({"schema": BLUEPRINT_SCHEMA, "complete": False,
                                   "error": f"{type(error).__name__}: {error}" if error else "",
                                   "blueprint": blueprint.to_dict(), "pieces": built, "holes": holes,
                                   "built": time.time()}, fh, indent=2, default=str)
                except Exception:
                    pass
            h.write_manifest()
            h.write_dispatch()
        if self.catalog is not None:
            for b in built:
                self.catalog.note_local(b["piece"], os.path.join(h.path, b["output"]), what=f"puzzle {blueprint.name}")
        return Puzzle(h, blueprint, built, holes, aln, prov, n_obs)

    # ---- tables ----
    def pieces_table(self):
        import pyarrow as pa
        rows = []
        for p in self.pieces.values():
            r = self.resolutions.get(p.id)
            rows.append({"piece": p.id, "title": p.title, "reference": p.reference, "status": p.status,
                         "assay": p.primary_assay, "modalities": ";".join(p.modalities),
                         "accessions": ";".join(p.accession_ids()),
                         **{f: p.facets.get(f, "") for f in FACETS},
                         "context": p.context, "access": p.access, "priority": p.priority,
                         "n_cells": p.expected.get("n_cells"), "n_samples": p.expected.get("n_samples"),
                         "perturbation": p.perturbation.get("type", ""), "scope": p.perturbation.get("scope", ""),
                         "genes_focus": ";".join(p.genes.get("focus", [])),
                         "genes_perturbed": ";".join(p.genes.get("perturbed", [])),
                         "genes_protein": ";".join(p.genes.get("protein", [])),
                         "route": r.route_kind if r else "", "bytes": r.bytes if r else 0,
                         "cost_usd": r.cost_usd if r else 0.0, "local": ";".join(p.local), "source": p.source})
        return pa.Table.from_pylist(rows)

    def to_csv(self, path: str) -> str:
        import pyarrow.csv as pacsv
        pacsv.write_csv(self.pieces_table(), path)
        return path

    def save(self, path: str) -> str:
        """Pieces, resolutions, antibody panels, the gene map and the observation-column synonyms
        as JSON, reloadable with `Orchestrator.load()` — the reloaded orchestrator aligns and
        builds exactly as this one does, not just describes the same pieces."""
        d = {"schema": "geminga.pieces/2", "sources": self.sources,
             "pieces": [p.to_dict() for p in self.pieces.values()],
             "resolutions": {k: v.to_dict() for k, v in self.resolutions.items()},
             "panels": self.panels,
             "genes": [asdict(r) for r in self.harmonizer.genes.records],
             "columns": {k: list(v) for k, v in self.harmonizer.columns.table.items()}}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str, **kw) -> "Orchestrator":
        """Read back what `save()` wrote (files from before the gene map was saved load too, with
        an empty one). Genes and column synonyms are added to the harmonizer passed in `kw`, if any."""
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        o = cls(**kw)
        o.sources = list(d.get("sources", []))
        o.panels = dict(d.get("panels", {}))
        for pd_ in d.get("pieces", []):
            accs = [Accession(**a) for a in pd_.pop("accessions", [])]
            p = Piece(**{k: v for k, v in pd_.items() if k in Piece.__dataclass_fields__})
            p.accessions = accs
            o.pieces[p.id] = p
        for k, rd in d.get("resolutions", {}).items():
            rd = dict(rd)
            rd.pop("plans", None)
            o.resolutions[k] = Resolution(**{kk: vv for kk, vv in rd.items() if kk in Resolution.__dataclass_fields__})
        for gd in d.get("genes", []):
            gd = {k: v for k, v in gd.items() if k in GeneRecord.__dataclass_fields__}
            gd["aliases"] = tuple(gd.get("aliases") or ())
            o.harmonizer.genes.add(GeneRecord(**gd))
        for k, v in d.get("columns", {}).items():
            o.harmonizer.columns.table[k] = tuple(v)
        return o

    def __len__(self) -> int:
        return len(self.pieces)

    def __repr__(self) -> str:
        bs = {s: len(v) for s, v in self.by_status().items() if v}
        return f"Orchestrator({len(self.pieces)} pieces, {bs})"


def orchestrator(**kw) -> Orchestrator:
    """Describe, resolve, harmonise, annotate and build datasets from descriptions of them."""
    return Orchestrator(**kw)
