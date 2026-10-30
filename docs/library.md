# The library

Analysis produces flags of many shapes: a SNP that looks miscalled, a read whose adapter trimming
is suspect, a cluster of cells that might be doublets, a reviewer's gene list, a stretch of a
culture series where the probe may have drifted, a corner of a tissue section. They all want the
same treatment — remember exactly *what* and *where*, say *why*, track what happened next, find
more data — and they are far more useful in one place, because the connections between them are
where the questions are.

```{image} _static/library.png
:alt: flagged records of seven kinds, linked by shared genes, overlapping loci and explicit links
:class: gem-figure
```

## Flag anything, with its own coordinates

```python
lib = g.library()                                        # ~/.geminga/library.jsonl

g.flag_variant("chr7", 55191822, "T", "G", rsid="rs121434568", genes=["EGFR"],
               note="call looks strand-biased", metadata={"sample": "S1"}, lib=lib)
g.flag_locus("chr7:55,019,017-55,211,628", "EGFR amplicon", lib=lib)
g.flag_sequence("AGATCGGAAGAGCACACGTC", name="adapter read-through", lib=lib)
g.flag_cells(barcodes, source="filtered_feature_bc_matrix.h5", name="sgEGFR cells", lib=lib)
g.flag_genes(["VEGFA", "CA9", "HILPDA"], name="reviewer's panel", lib=lib)
g.flag_values("culture.parquet", "t", t0, t1, "O2 dip", where=("o2_percent", "<", 8), lib=lib)
g.flag_region(view, "rim-east", box, lib=lib)            # spatial, as before
```

Coordinates are stored **with their convention**, never assumed:

| kind | coordinates | convention recorded |
|---|---|---|
| region | chip or image units | rectangle or polygon |
| values | a window on a named column, plus an optional predicate | `o2_percent < 8` flags readings, not just the window |
| locus | chromosome, start, end, build | entered as a browser prints it (1-based inclusive), stored 0-based half-open |
| variant | chromosome, position, ref, alt, rsID, build | position kept 1-based as in VCF; `hg38` normalised to `GRCh38` |
| sequence | record ids, a sub-range, the sequence, its SHA-256 | 0-based half-open within the record; over 10 kb kept by hash |
| cells | barcodes, count, hash | sets over 64 cells live in a sidecar file, stored once however ordered |
| genes | symbols or Ensembl ids, organism | id type detected |

## Review workflow

`open → in-review → confirmed | refuted → resolved`. Each transition is appended to the record's
history with a timestamp, a note and who made it:

```python
lib.review(snp, "pulled the BAM, checking strand counts", by="SV")
lib.refute(snp, "strand bias confirmed: artefact", by="SV")
lib.to_double_check()        # the queue of unresolved double-check flags
lib.open_questions()         # the queue of need-data flags
```

## Querying

```python
lib.query(gene="EGFR")                          # any kind that concerns EGFR
lib.query(locus="chr7:55.0M-55.3M")             # overlapping loci and variants, same build
lib.query(meta={"sample": "S1", "depth": lambda d: d < 20})
lib.query(kind=("variant", "cells"), status="in-review", text="strand")
lib.frame()                                     # the whole library as a Polars DataFrame
lib.to_bed("flags.bed")                         # loci and variants for IGV and bedtools
```

Anything put in `metadata` — sample, caller, instrument, reviewer, depth — is queryable.

## Links

`lib.related(record)` finds other records sharing genes, overlapping loci, sharing a dataset, or
explicitly linked — across kinds. In the demonstration, the flagged SNP linked itself to the EGFR
amplicon (overlap and gene), the sgEGFR cells and the reviewer's panel, with nobody drawing those
links by hand. Explicit links carry a relation: `lib.link(o2_dip, rim, "same biology: hypoxia")`.

## Finding more data, per kind

`describe_any()` measures a record in whatever way its kind allows; `leads_any()` finds more.

| kind | in your data | in public repositories |
|---|---|---|
| sequence | occurrences in your FASTA/FASTQ files, both strands, SIMD matcher | BLAST (`CMD=Put` URL) |
| variant | related records by locus and gene | Ensembl variation, ClinVar, dbSNP, Ensembl region overlap |
| locus | overlapping records | Ensembl genes and variants in the region |
| genes | which of your datasets measured them; totals and centroids in a chip | Ensembl lookup, GEO, PubMed |
| cells | features over-represented in the set against all cells | via their genes |
| values | rows matching the predicate in the window | via the series' metadata |
| region | enrichment against the section, similar regions | GEO, PubMed, ENA |

Queries are built, not run, unless `online=True`. Both GEO/PubMed (NCBI E-utilities) and Ensembl
(REST) are plain GETs, so PRISM or a browser can run them as well as GEMINGA can.

:::{admonition} What it does not do yet
:class: caution
GEMINGA does not read VCF or BAM, so a flagged variant carries coordinates and links but cannot
be re-checked against its reads from inside the library — the record says so. The online queries
are tested on recorded responses; the build environment cannot reach NCBI, EBI or Ensembl.
:::
