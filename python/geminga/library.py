"""A library of things you flagged — any kind of thing, with its own coordinates.

Selections began as regions on a map. Analysis produces flags of many other shapes: a SNP that
looks miscalled, a read whose adapter trimming is suspect, a cluster of cells that might be
doublets, a gene list from a reviewer, a stretch of a culture series where the probe drifted.
They all want the same treatment — remember exactly *what* and *where*, say *why*, track what
happened next, and find more data about it — and they are far more useful in one place, because
the connections between them are where the questions are: the SNP sits in a gene that is enriched
in the region you marked, which is expressed in the cells you flagged.

**One record, many coordinate systems.** Every entry is a `Selection` whose `kind` fixes how its
coordinates are read, and those conventions are stored with it rather than assumed:

    rect / polygon   chip or image units
    interval         record, cell or row index
    column-range     a value range on a named column (time), optionally with a value predicate
    locus            chromosome, 0-based half-open start/end, genome build
    variant          chromosome, 1-based VCF position, ref, alt, rsID, build
    sequence         record ids in a FASTA/FASTQ, a 0-based sub-range, the sequence and its hash
    cells            barcodes (kept in a sidecar file when large), their count and hash
    genes            symbols or Ensembl ids, and the organism

**Review workflow.** `open → in-review → confirmed | refuted → resolved`, each transition kept in
the record's history with a timestamp and a note. `double-check` flags and `need-data` flags are
the two queues the library can list at any moment.

**A query language over the metadata.** `lib.query(kind=..., status=..., gene=..., locus=
"chr7:55.0M-55.3M", tag=..., meta={"tissue": "cortex"}, text=...)`. Anything you put in
`metadata` is queryable; `lib.frame()` gives the whole library as a Polars DataFrame for anything
the query method does not cover.

**Links.** `lib.related(x)` finds records sharing genes, overlapping loci, or the same dataset —
across kinds. Explicit links (`lib.link(a, b, "explains")`) are stored too.

**More data, per kind.** Sequences are searched for in your other FASTA/FASTQ files with the SIMD
matcher; genes are checked against every dataset you hold; externally, each kind gets the
repository that answers it — Ensembl and ClinVar for variants and genes, BLAST for sequences, GEO
and PubMed for expression. Queries are built, not run, unless you ask; everything found is stored
on the record.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
import warnings
from typing import Optional, Sequence
from urllib.parse import quote, urlencode

import numpy as np

from .selections import (INTENTS, Selection, SelectionBook, _forward, dataset_key,
                         describe as _describe, external_queries, leads as _leads, local_leads,
                         select, select_values)

__all__ = ["Library", "library", "flag_locus", "flag_variant", "flag_sequence", "flag_cells",
           "flag_genes", "flag_values", "flag_region", "parse_locus", "STATUSES",
           "describe_any", "leads_any"]

#: The review workflow. Transitions are free, but each is recorded.
STATUSES = ("open", "described", "searched", "in-review", "confirmed", "refuted", "resolved")

_GENOME_ALIASES = {"hg38": "GRCh38", "grch38": "GRCh38", "hg19": "GRCh37", "grch37": "GRCh37",
                   "mm10": "GRCm38", "grcm38": "GRCm38", "mm39": "GRCm39", "grcm39": "GRCm39"}


def _build(b: Optional[str]) -> str:
    return _GENOME_ALIASES.get(str(b or "GRCh38").lower(), b or "GRCh38")


def _chrom(c) -> str:
    c = str(c)
    return c[3:] if c.lower().startswith("chr") else c


def _num(s: str) -> int:
    s = s.replace(",", "").replace("_", "").strip().lower()
    mult = 1
    for suf, m in (("k", 1_000), ("m", 1_000_000), ("g", 1_000_000_000)):
        if s.endswith(suf):
            s, mult = s[:-1], m
    return int(round(float(s) * mult))


def parse_locus(text: str) -> tuple:
    """`chr7:55,019,017-55,211,628` or `7:55.0M-55.3M` -> ("7", start0, end0).

    Input is read the way genome browsers print it — 1-based, inclusive — and stored 0-based,
    half-open, which is what BED, pysam and every overlap test here use. Position 0 does not
    exist in a browser's coordinates; a start of 0 (`chr1:0-100`) is read as the first base
    rather than turned into an offset of -1, and the result never starts below 0.
    """
    m = re.match(r"^\s*([^:\s]+)\s*:\s*([\d.,_]+[kKmMgG]?)\s*[-–]\s*([\d.,_]+[kKmMgG]?)\s*$", text)
    if not m:
        m1 = re.match(r"^\s*([^:\s]+)\s*:\s*([\d.,_]+[kKmMgG]?)\s*$", text)
        if not m1:
            raise ValueError(f"not a locus: {text!r} (expected chr:start-end)")
        p = _num(m1.group(2))
        return _chrom(m1.group(1)), max(p - 1, 0), max(p, 1)
    a, b = _num(m.group(2)), _num(m.group(3))
    if b < a:
        raise ValueError(f"end before start in {text!r}")
    return _chrom(m.group(1)), max(a - 1, 0), b


def _new(kind, name, geometry, dataset, intent, note, tags, metadata, genes=None) -> Selection:
    if intent not in INTENTS:
        raise ValueError(f"intent must be one of {list(INTENTS)}")
    return Selection(id=uuid.uuid4().hex[:12], name=name, dataset=dataset or {}, kind=kind,
                     geometry=geometry, intent=intent, note=note, tags=list(tags),
                     metadata=dict(metadata or {}),
                     genes=[g for g in (genes or [])] or None,
                     history=[{"status": "open", "ts": time.time(), "note": "created"}])


def _ds(source) -> dict:
    return dataset_key(source) if source is not None else {}


# ---------------------------------------------------------------------------
# Constructors, one per kind
# ---------------------------------------------------------------------------

def flag_locus(locus: str, name: Optional[str] = None, *, build: str = "GRCh38",
               genes: Sequence[str] = (), intent: str = "double-check", note: str = "",
               tags: Sequence[str] = (), metadata: Optional[dict] = None, source=None,
               lib: Optional["Library"] = None) -> Selection:
    """A genomic interval. `locus` as a browser prints it: 1-based, inclusive."""
    c, s, e = parse_locus(locus)
    sel = _new("locus", name or f"{c}:{s + 1:,}-{e:,}",
               {"chrom": c, "start": s, "end": e, "build": _build(build),
                "coords": "0-based half-open"},
               _ds(source), intent, note, tags, metadata, genes)
    return lib.add(sel) if lib is not None else sel


def flag_variant(chrom=None, pos: Optional[int] = None, ref: str = "", alt: str = "", *,
                 rsid: Optional[str] = None, name: Optional[str] = None, build: str = "GRCh38",
                 genes: Sequence[str] = (), intent: str = "double-check", note: str = "",
                 tags: Sequence[str] = (), metadata: Optional[dict] = None, source=None,
                 lib: Optional["Library"] = None) -> Selection:
    """A SNP or small variant, as VCF writes it: `pos` is 1-based. An rsID alone is accepted
    when the position is not known yet; external lookup can fill it in."""
    if pos is None and not rsid:
        raise ValueError("a variant needs a position or an rsID")
    c = _chrom(chrom) if chrom is not None else None
    start = (int(pos) - 1) if pos is not None else None
    end = (start + max(len(ref), 1)) if start is not None else None
    geom = {"chrom": c, "pos": int(pos) if pos is not None else None, "ref": ref.upper(),
            "alt": alt.upper(), "rsid": rsid, "build": _build(build),
            "start": start, "end": end, "coords": "pos 1-based (VCF); start/end 0-based half-open"}
    label = name or rsid or f"{c}:{pos} {ref}>{alt}"
    sel = _new("variant", label, geom, _ds(source), intent, note, tags, metadata, genes)
    return lib.add(sel) if lib is not None else sel


def flag_sequence(sequence: Optional[str] = None, *, records: Sequence[str] = (),
                  start: Optional[int] = None, end: Optional[int] = None, source=None,
                  name: Optional[str] = None, genes: Sequence[str] = (),
                  intent: str = "double-check", note: str = "", tags: Sequence[str] = (),
                  metadata: Optional[dict] = None, lib: Optional["Library"] = None) -> Selection:
    """A sequence of interest — a read, a motif, an adapter, a contig region.

    Give the sequence itself, or record ids (with an optional 0-based [start, end) within them)
    in a FASTA/FASTQ `source`, or both. The sequence is hashed so identity can be checked later
    without trusting the name; sequences over 10 kb are kept by hash only.

    RNA is stored as DNA (`U` becomes `T`), but only for a nucleotide sequence, one written
    entirely in `ACGTUN`. In anything else a `U` is selenocysteine and is left alone.
    """
    seq = (sequence or "").upper()
    if seq and set(seq) <= set("ACGTUN"):
        seq = seq.replace("U", "T")
    seq = seq or None
    if seq is None and not records:
        raise ValueError("give a sequence, or record ids in a source file")
    geom = {"records": list(records), "start": start, "end": end,
            "length": len(seq) if seq else (end - start if start is not None and end else None),
            "sequence": seq if seq and len(seq) <= 10_000 else None,
            "sha256": hashlib.sha256(seq.encode()).hexdigest() if seq else None,
            "coords": "0-based half-open within the record"}
    label = name or (f"{seq[:12]}…" if seq and len(seq) > 12 else seq) or ",".join(records[:2])
    sel = _new("sequence", label, geom, _ds(source), intent, note, tags, metadata, genes)
    return lib.add(sel) if lib is not None else sel


def flag_cells(barcodes: Sequence[str], *, source=None, name: Optional[str] = None,
               modality: Optional[str] = None, genes: Sequence[str] = (),
               intent: str = "double-check", note: str = "", tags: Sequence[str] = (),
               metadata: Optional[dict] = None, lib: Optional["Library"] = None,
               sidecar_dir: str = "~/.geminga/cellsets") -> Selection:
    """A set of cells, by barcode. Large sets live in a sidecar file keyed by their hash, so the
    log stays small and the same set is never stored twice."""
    bc = sorted({str(b) for b in barcodes})
    h = hashlib.sha256("\n".join(bc).encode()).hexdigest()
    path = None
    if len(bc) > 64:
        d = os.path.expanduser(sidecar_dir); os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{h[:16]}.txt")
        if not os.path.exists(path):
            with open(path, "w") as fh:
                fh.write("\n".join(bc) + "\n")
    geom = {"n": len(bc), "barcodes": bc if path is None else None, "barcodes_path": path,
            "barcodes_sha256": h, "preview": bc[:5], "modality": modality}
    sel = _new("cells", name or f"{len(bc):,} cells", geom, _ds(source), intent, note, tags,
               metadata, genes)
    return lib.add(sel) if lib is not None else sel


def flag_genes(genes: Sequence[str], *, name: Optional[str] = None, organism: str = "Homo sapiens",
               id_type: Optional[str] = None, source=None, intent: str = "need-data",
               note: str = "", tags: Sequence[str] = (), metadata: Optional[dict] = None,
               lib: Optional["Library"] = None) -> Selection:
    """A gene list — markers, a reviewer's candidates, hits from a screen."""
    gl = [str(x).strip() for x in genes if str(x).strip()]
    it = id_type or ("ensembl" if gl and all(x.upper().startswith("ENS") for x in gl) else "symbol")
    geom = {"genes": gl, "n": len(gl), "id_type": it, "organism": organism}
    sel = _new("genes", name or ", ".join(gl[:3]) + ("…" if len(gl) > 3 else ""), geom,
               _ds(source), intent, note, tags, metadata, gl)
    return lib.add(sel) if lib is not None else sel


def flag_values(source, column: str, lo, hi, name: str, *, where: Optional[tuple] = None,
                intent: str = "double-check", note: str = "", tags: Sequence[str] = (),
                metadata: Optional[dict] = None, lib: Optional["Library"] = None) -> Selection:
    """Values in a time series: a window on `column`, optionally narrowed by a predicate on
    another column — `where=("o2_percent", "<", 8)` flags the readings, not just the window."""
    sel = select_values(source, column, lo, hi, name, intent=intent, note=note)
    if where is not None:
        c, op, v = where
        if op not in ("<", "<=", ">", ">=", "==", "!="):
            raise ValueError(f"unsupported operator {op!r}")
        sel.geometry["where"] = {"column": c, "op": op, "value": float(v)}
    sel.tags, sel.metadata = list(tags), dict(metadata or {})
    sel.history = [{"status": "open", "ts": time.time(), "note": "created"}]
    return lib.add(sel) if lib is not None else sel


def flag_region(view, name: str, bounds: Optional[tuple] = None, *, intent: str = "double-check",
                note: str = "", tags: Sequence[str] = (), metadata: Optional[dict] = None,
                lib: Optional["Library"] = None) -> Selection:
    """A spatial region or an interval — the original selection, with metadata and history."""
    sel = select(view, name, bounds, intent=intent, note=note, tags=tags)
    sel.metadata = dict(metadata or {})
    sel.history = [{"status": "open", "ts": time.time(), "note": "created"}]
    return lib.add(sel) if lib is not None else sel


# ---------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------

def _genes_of(s: Selection) -> set:
    out = {x.upper() for x in (s.genes or [])}
    if s.kind == "genes":
        out |= {x.upper() for x in s.geometry.get("genes", [])}
    out |= {x.upper() for x in s.metadata.get("genes", [])} if isinstance(
        s.metadata.get("genes"), list) else set()
    out |= {e["gene"].upper() for e in s.summary.get("enriched", [])[:10]}
    return out


def _interval(s: Selection) -> Optional[tuple]:
    g = s.geometry
    if s.kind in ("locus", "variant") and g.get("chrom") is not None and g.get("start") is not None:
        return (str(g["chrom"]), int(g["start"]), int(g["end"]), g.get("build"))
    return None


class Library(SelectionBook):
    """Everything you flagged, queryable by what it is, where it is, and what it concerns."""

    def __init__(self, path: str = "~/.geminga/library.jsonl"):
        super().__init__(os.path.abspath(os.path.expanduser(path)))     # see SelectionBook

    # ---- workflow ----
    def set_status(self, sel_or_id, status: str, note: str = "", by: str = "") -> Selection:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        cur = self.get(sel_or_id.id if isinstance(sel_or_id, Selection) else sel_or_id)
        if cur is None:
            raise KeyError("no such record")
        hist = list(cur.history) + [{"status": status, "ts": time.time(), "note": note,
                                     "by": by, "from": cur.status}]
        return self.update(cur, status=status, history=hist)

    def review(self, x, note: str = "", by: str = "") -> Selection:
        return self.set_status(x, "in-review", note, by)

    def confirm(self, x, note: str = "", by: str = "") -> Selection:
        return self.set_status(x, "confirmed", note, by)

    def refute(self, x, note: str = "", by: str = "") -> Selection:
        return self.set_status(x, "refuted", note, by)

    def resolve(self, x, note: str = "", by: str = "") -> Selection:
        return self.set_status(x, "resolved", note, by)

    def annotate(self, x, **metadata) -> Selection:
        """Merge metadata into a record; every key becomes queryable."""
        cur = self.get(x.id if isinstance(x, Selection) else x)
        md = dict(cur.metadata); md.update(metadata)
        return self.update(cur, metadata=md)

    def link(self, a, b, relation: str = "related", note: str = "") -> None:
        """An explicit, typed link — 'explains', 'contradicts', 'same-sample', …"""
        ra = self.get(a.id if isinstance(a, Selection) else a)
        rb = self.get(b.id if isinstance(b, Selection) else b)
        self.update(ra, links=list(ra.links) + [{"to": rb.id, "relation": relation, "note": note}])
        self.update(rb, links=list(rb.links) + [{"to": ra.id, "relation": f"~{relation}",
                                                  "note": note}])

    # ---- queues ----
    def to_double_check(self) -> list:
        return [s for s in self.all().values()
                if s.intent == "double-check" and s.status not in ("confirmed", "refuted",
                                                                   "resolved")]

    def open_questions(self) -> list:
        return [s for s in self.all().values()
                if s.intent == "need-data" and s.status not in ("resolved",)]

    # ---- queries ----
    def query(self, *, kind=None, intent=None, status=None, tag=None, gene=None, locus=None,
              build: Optional[str] = None, dataset=None, meta: Optional[dict] = None,
              text: Optional[str] = None, since: Optional[float] = None) -> list:
        """Filter the library. Every argument narrows; `kind`, `status` and `intent` accept a
        value or a tuple. `locus` is a browser-style string and matches overlapping loci and
        variants on the same build."""
        def want(v, x):
            return v is None or (x in v if isinstance(v, (tuple, list, set)) else x == v)
        iv = parse_locus(locus) if locus else None
        bld = _build(build) if build else None
        dkey = (dataset_key(dataset)["key"] if dataset is not None and not isinstance(dataset, dict)
                else (dataset or {}).get("key"))
        genes = {gene.upper()} if isinstance(gene, str) else {x.upper() for x in (gene or [])}
        out = []
        for s in self.all().values():
            if not (want(kind, s.kind) and want(intent, s.intent) and want(status, s.status)):
                continue
            if tag and tag not in s.tags:
                continue
            if dkey and s.dataset.get("key") != dkey:
                continue
            if since and s.created < since:
                continue
            if genes and not (genes & _genes_of(s)):
                continue
            if iv is not None or bld:
                ii = _interval(s)
                if ii is None:
                    continue
                if bld and ii[3] != bld:
                    continue
                if iv is not None and not (ii[0] == iv[0] and ii[1] < iv[2] and iv[1] < ii[2]):
                    continue
            if meta:
                ok = True
                for k, v in meta.items():
                    have = s.metadata.get(k)
                    if callable(v):
                        ok = ok and bool(have is not None and v(have))
                    else:
                        ok = ok and have == v
                if not ok:
                    continue
            if text:
                blob = " ".join([s.name, s.note, " ".join(s.tags), json.dumps(s.metadata,
                                 default=str), " ".join(_genes_of(s))]).lower()
                if text.lower() not in blob:
                    continue
            out.append(s)
        return sorted(out, key=lambda s: s.created)

    def related(self, x, by: Sequence[str] = ("genes", "locus", "dataset", "links")) -> list:
        """Other records connected to `x`, across kinds, with the reasons."""
        me = self.get(x.id if isinstance(x, Selection) else x)
        mg, mi = _genes_of(me), _interval(me)
        linked = {l["to"]: l["relation"] for l in me.links}
        out = []
        for s in self.all().values():
            if s.id == me.id:
                continue
            why = []
            if "genes" in by:
                sh = mg & _genes_of(s)
                if sh:
                    why.append("genes: " + ", ".join(sorted(sh)[:5]))
            if "locus" in by and mi is not None:
                si = _interval(s)
                if si and si[0] == mi[0] and si[3] == mi[3] and si[1] < mi[2] and mi[1] < si[2]:
                    why.append(f"overlaps {si[0]}:{si[1] + 1:,}-{si[2]:,}")
            if "dataset" in by and me.dataset.get("key") and s.dataset.get("key") == me.dataset.get("key"):
                why.append("same dataset")
            if "links" in by and s.id in linked:
                why.append(f"linked: {linked[s.id]}")
            if why:
                out.append((s, why))
        # explicit links first, then by number of reasons
        return sorted(out, key=lambda t: (not any(w.startswith("linked") for w in t[1]), -len(t[1])))

    def frame(self):
        """The whole library as a Polars DataFrame — for anything `query()` does not cover."""
        import polars as pl
        rows = []
        for s in self.all().values():
            iv = _interval(s)
            rows.append({"id": s.id, "name": s.name, "kind": s.kind, "intent": s.intent,
                         "status": s.status, "created": s.created, "updated": s.updated,
                         "dataset": s.dataset.get("path"), "genes": sorted(_genes_of(s)),
                         "chrom": iv[0] if iv else None, "start": iv[1] if iv else None,
                         "end": iv[2] if iv else None, "build": iv[3] if iv else None,
                         "tags": list(s.tags), "n_leads": len(s.leads),
                         "metadata": json.dumps(s.metadata, default=str), "note": s.note})
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def to_bed(self, path: str, build: Optional[str] = None) -> str:
        """Loci and variants as BED6 — for IGV, bedtools, and any genome browser.

        Strictly six columns. Intent and status travel inside the name field, because a seventh
        column would be read as `thickStart`, which must be an integer — an earlier version
        appended a comment there and produced a file IGV and bedtools reject.

        A BED file has no column for the genome build, so coordinates from two assemblies cannot
        share one: if the library holds loci on more than one build, pass `build=` to choose
        (only that build is written), otherwise this raises `ValueError`. Records that have no
        coordinates yet (an rsID-only variant) are skipped, with a warning saying how many.
        """
        bld = _build(build) if build else None
        rows, unplaced = [], 0
        for s in self.query(kind=("locus", "variant")):
            iv = _interval(s)
            if iv is None:
                if bld is None or _build(s.geometry.get("build")) == bld:
                    unplaced += 1
            elif bld is None or iv[3] == bld:
                rows.append((s, iv))
        builds = sorted({str(iv[3]) for _, iv in rows})
        if bld is None and len(builds) > 1:
            raise ValueError(f"the library holds loci on several genome builds {builds}, and a BED "
                             f"file cannot say which is which; pass build=... to write one of them")
        if unplaced:
            warnings.warn(f"to_bed: {unplaced} record(s) without coordinates were skipped "
                          f"(rsID-only variants have no position until they are looked up)",
                          UserWarning, stacklevel=2)
        lines = []
        for s, (c, a, b, _) in rows:
            name = f"{s.name}|{s.intent}|{s.status}".replace(" ", "_").replace("\t", "_")
            lines.append(f"chr{c}\t{a}\t{b}\t{name}\t0\t.\n")
        with open(path, "w") as fh:               # only after every check has passed
            fh.writelines(lines)
        return path

    def print_table(self, rows: Optional[list] = None) -> None:
        rows = rows if rows is not None else sorted(self.all().values(), key=lambda s: s.created)
        if not rows:
            print("(library empty)"); return
        print(f"  {'kind':<12} {'name':<26} {'intent':<13} {'status':<11} {'genes':<22} where")
        for s in rows:
            iv = _interval(s)
            where = (f"{iv[0]}:{iv[1] + 1:,}-{iv[2]:,} {iv[3]}" if iv else
                     os.path.basename(str(s.dataset.get("path", ""))) or "-")
            gs = ",".join(sorted(_genes_of(s))[:3])
            print(f"  {s.kind:<12} {s.name[:26]:<26} {s.intent:<13} {s.status:<11} "
                  f"{gs[:22]:<22} {where[:40]}")
        print(f"\n  to double-check: {len(self.to_double_check())}   "
              f"open questions: {len(self.open_questions())}")

    def __repr__(self) -> str:
        n = self.all()
        kinds: dict[str, int] = {}
        for s in n.values():
            kinds[s.kind] = kinds.get(s.kind, 0) + 1
        return f"Library({self.path}, " + ", ".join(f"{v} {k}" for k, v in kinds.items()) + ")"


def library(path: str = "~/.geminga/library.jsonl") -> Library:
    """Open (or create) the library of flagged things."""
    return Library(path)


# ---------------------------------------------------------------------------
# Describe and search, for every kind
# ---------------------------------------------------------------------------

def describe_any(sel: Selection, view=None, *, sources: Sequence[str] = (),
                 lib: Optional[Library] = None, **kw) -> dict:
    """Measure a record in whatever way its kind allows, and store the result on it."""
    if sel.kind in ("rect", "polygon", "interval") or (sel.kind == "column-range"
                                                       and "where" not in sel.geometry):
        return _describe(sel, view, book=lib, **kw)
    if sel.kind == "column-range":
        summ = _describe_where(sel)
    elif sel.kind == "sequence":
        summ = _describe_sequence(sel, sources)
    elif sel.kind == "genes":
        summ = _describe_genes(sel, view, sources)
    elif sel.kind == "cells":
        summ = _describe_cells(sel, view)
    elif sel.kind in ("variant", "locus"):
        g = sel.geometry
        summ = {"length": (g["end"] - g["start"]) if g.get("start") is not None else None,
                "build": g.get("build"),
                "note": "GEMINGA does not read VCF/BAM yet; the record carries coordinates and "
                        "links. Use leads_any() for Ensembl, ClinVar and dbSNP."}
        if lib is not None:
            summ["related"] = [f"{s.kind}:{s.name} ({'; '.join(w)})"
                               for s, w in lib.related(sel)[:10]]
    else:
        raise TypeError(f"no describe for kind {sel.kind!r}")
    sel.summary = summ
    if lib is not None:                        # the stored record decides, not this (maybe stale) object
        sel.status = lib._advance(sel, "described", summary=summ).status
    else:
        sel.status = _forward(sel.status, "described")
    return summ


def _describe_where(sel) -> dict:
    import polars as pl
    g = sel.geometry; w = g["where"]
    lf = pl.scan_parquet(sel.dataset["path"])
    t = pl.col(g["column"])
    ops = {"<": pl.Expr.__lt__, "<=": pl.Expr.__le__, ">": pl.Expr.__gt__, ">=": pl.Expr.__ge__,
           "==": pl.Expr.__eq__, "!=": pl.Expr.__ne__}
    win = lf.filter((t >= g["lo"]) & (t < g["hi"]))
    hit = win.filter(ops[w["op"]](pl.col(w["column"]), w["value"]))
    n_win = win.select(pl.len()).collect().item()
    h = hit.select([pl.len().alias("n"), pl.col(g["column"]).min().alias("first"),
                    pl.col(g["column"]).max().alias("last"),
                    pl.col(w["column"]).min().alias("vmin"),
                    pl.col(w["column"]).max().alias("vmax")]).collect()
    return {"rows_in_window": int(n_win), "rows_matching": int(h["n"][0]),
            "first": h["first"][0], "last": h["last"][0],
            "value_range": [h["vmin"][0], h["vmax"][0]],
            "predicate": f"{w['column']} {w['op']} {w['value']}"}


def _describe_sequence(sel, sources) -> dict:
    """Where else does this sequence occur, in files you hold? Both strands, SIMD matcher."""
    from ._core import search_fastx
    seq = sel.geometry.get("sequence")
    out = {"length": sel.geometry.get("length"), "occurrences": []}
    if not seq:
        out["note"] = "no sequence stored (only record ids or a hash); nothing to search for"
        return out
    gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
    out["gc"] = round(gc, 3)
    for p in sources:
        if not os.path.exists(p):
            continue
        r = search_fastx(p, [seq], True, 100_000, 0, None, True)
        out["occurrences"].append({"file": os.path.basename(p), "path": p,
                                   "records_matched": int(r["records_matched"]),
                                   "records_scanned": int(r["records_scanned"]),
                                   "fraction": round(r["records_matched"] /
                                                     max(r["records_scanned"], 1), 6)})
    return out


def _describe_genes(sel, view, sources) -> dict:
    genes = sel.geometry.get("genes", [])
    out = {"n": len(genes), "per_dataset": local_leads(genes, sources) if sources else []}
    if view is not None and hasattr(view, "gef"):
        gef = view.gef
        b = next((lv for lv in sorted(gef.bins) if lv >= 50), gef.bins[-1])
        names, offs, cnts = gef.gene_index(b)
        pos = {n.upper(): i for i, n in enumerate(names)}
        exp = gef.f[gef.levels[b].path]["expression"]
        rows = []
        for g in genes:
            i = pos.get(g.upper())
            if i is None:
                rows.append({"gene": g, "present": False}); continue
            lo, n = int(offs[i]), int(cnts[i])
            rec = exp[lo:lo + n]
            mid = float(np.asarray(rec["count"], float).sum())
            x, y = np.asarray(rec["x"], float), np.asarray(rec["y"], float)
            w = np.asarray(rec["count"], float)
            rows.append({"gene": g, "present": True, "mid": int(mid), "bins_detected": n,
                         "centroid": [round(float((x * w).sum() / max(w.sum(), 1)), 1),
                                      round(float((y * w).sum() / max(w.sum(), 1)), 1)]})
        out["in_view"] = rows
    return out


def _cell_barcodes(sel) -> list:
    g = sel.geometry
    if g.get("barcodes"):
        return g["barcodes"]
    with open(g["barcodes_path"]) as fh:
        return [l.strip() for l in fh if l.strip()]


def _describe_cells(sel, view) -> dict:
    """Totals and the most over-represented features in the set, against all cells."""
    m = getattr(view, "m", view)
    if m is None or not hasattr(m, "barcodes"):
        return {"n": sel.geometry["n"], "note": "pass the 10x matrix or view to measure the set"}
    want = set(_cell_barcodes(sel))
    idx = np.flatnonzero(np.isin(m.barcodes, list(want)))
    mod = sel.geometry.get("modality") or next(iter(m.modalities))
    names = m.features(mod)
    in_sum = np.zeros(len(names)); all_sum = np.zeros(len(names))
    per = m._cells_per_chunk()
    for c0 in range(0, m.n_cells, per):
        c1 = min(c0 + per, m.n_cells)
        X = m.block(c0, c1, mod)
        all_sum += np.asarray(X.sum(1)).ravel()
        loc = idx[(idx >= c0) & (idx < c1)] - c0
        if loc.size:
            in_sum += np.asarray(X[:, loc].sum(1)).ravel()
    frac = idx.size / max(m.n_cells, 1)
    expected = all_sum * frac
    lfc = np.log2((in_sum + 1) / (expected + 1))
    order = np.argsort(-lfc)
    return {"n": int(idx.size), "found_in_matrix": int(idx.size), "requested": len(want),
            "modality": mod, "fraction_of_cells": round(frac, 4),
            "enriched": [{"gene": str(names[i]), "log2fc": round(float(lfc[i]), 3),
                          "observed": int(in_sum[i]), "expected": round(float(expected[i]), 1)}
                         for i in order[:10] if in_sum[i] >= 5 and lfc[i] > 0]}


def _ensembl_species(organism: str) -> str:
    return organism.strip().lower().replace(" ", "_") if organism else "homo_sapiens"


def leads_any(sel: Selection, *, sources: Sequence[str] = (), context: Sequence[str] = (),
              organism: Optional[str] = None, online: bool = False,
              lib: Optional[Library] = None) -> list:
    """Where more data on this record might be — per kind, locally and in public repositories."""
    if sel.kind in ("rect", "polygon", "interval", "column-range"):
        return _leads(sel, context=context, organism=organism, local=sources, online=online,
                      book=lib)
    org = organism or sel.metadata.get("organism") or sel.geometry.get("organism") or "Homo sapiens"
    sp = _ensembl_species(org)
    found: list = []
    q: list = []
    if sel.kind == "sequence":
        seq = sel.geometry.get("sequence")
        occ = _describe_sequence(sel, sources)["occurrences"] if sources else []
        found += [{"kind": "local", "source": "your data", "title": o["file"], "url": o["path"],
                   "records_matched": o["records_matched"]} for o in occ if o["records_matched"]]
        if seq:
            q.append({"source": "BLAST", "title": f"blastn vs nt, {len(seq)} nt",
                      "url": "https://blast.ncbi.nlm.nih.gov/Blast.cgi?" + urlencode(
                          {"CMD": "Put", "PROGRAM": "blastn", "DATABASE": "nt",
                           "QUERY": seq[:8000]})})
    if sel.kind == "variant":
        g = sel.geometry
        if g.get("rsid"):
            q.append({"source": "Ensembl", "title": f"variation {g['rsid']}",
                      "url": f"https://rest.ensembl.org/variation/{sp}/{quote(g['rsid'])}"
                             f"?content-type=application/json"})
            for db in ("clinvar", "snp"):
                q.append({"source": "ClinVar" if db == "clinvar" else "dbSNP",
                          "title": g["rsid"],
                          "url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
                                 + urlencode({"db": db, "term": g["rsid"], "retmode": "json"})})
        if g.get("chrom") is not None and g.get("start") is not None:
            q.append({"source": "Ensembl", "title": "variants and genes overlapping",
                      "url": f"https://rest.ensembl.org/overlap/region/{sp}/{g['chrom']}:"
                             f"{g['start'] + 1}-{g['end']}?feature=gene;feature=variation;"
                             f"content-type=application/json"})
    if sel.kind == "locus":
        g = sel.geometry
        q.append({"source": "Ensembl", "title": "genes and variants in the locus",
                  "url": f"https://rest.ensembl.org/overlap/region/{sp}/{g['chrom']}:"
                         f"{g['start'] + 1}-{g['end']}?feature=gene;feature=variation;"
                         f"content-type=application/json"})
    genes = sorted(_genes_of(sel))
    if genes:
        if sources:
            found += local_leads(genes, sources, exclude=[sel.dataset.get("path")])
        for gname in genes[:5]:
            q.append({"source": "Ensembl", "title": f"gene {gname}",
                      "url": f"https://rest.ensembl.org/lookup/symbol/{sp}/{quote(gname)}"
                             f"?content-type=application/json"})
        for x in external_queries(genes, context, org, None):
            q.append({"source": x["source"], "title": x["term"], "url": x["url"]})
    for x in q:
        entry = {"kind": "query", **x, "accession": None}
        if online:
            try:
                import urllib.request
                with urllib.request.urlopen(urllib.request.Request(
                        x["url"], headers={"User-Agent": "geminga/0.14"}), timeout=20) as r:
                    entry["status_code"] = r.status
            except Exception as e:
                entry["error"] = f"{type(e).__name__}: {e}"
        found.append(entry)
    sel.leads = found
    sel.summary = dict(sel.summary)
    sel.summary["leads_checked"] = {"sources": [os.path.basename(p) for p in sources
                                                if os.path.exists(p)],
                                    "local_found": sum(1 for f in found if f["kind"] == "local")}
    if lib is not None:
        sel.status = lib._advance(sel, "searched", leads=found, summary=sel.summary).status
    else:
        sel.status = _forward(sel.status, "searched")
    return found
