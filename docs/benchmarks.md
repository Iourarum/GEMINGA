# Benchmarks

Every number here is produced by a script in `examples/` and re-checked by CI. If one of them
does not reproduce on your machine, that is a bug worth filing.

## The budget invariant

One 256 MB budget, every format, one process — `examples/all_formats.py`:

| Input | Size | Chunks | Peak resident |
|---|---|---|---|
| `reads.fastq` (200k reads) | 61.7 MB | 4 | 18.0 MB |
| `reads.fastq.gz` | 28.6 MB | 4 | 18.0 MB |
| `sc.h5ad` (20k cells × 3k genes, CSR) | 24.2 MB | 3 | 8.0 MB |
| `tissue.ome.tif` (6144², uint16) | 72.0 MB | 25 tiles | 4.0 MB |
| `test_bins.parquet` (2M rows) | 24.0 MB | 20 | 14.4 MB |

Budget high-water across all of it: **18.0 MB of 256 MB, 47 grants, 0 waits, 0 denials.**

## Governor behaviour

From `examples/budget_behaviour.py`:

- **Adaptive** — the same Parquet file at a 32 MB budget streams 8 MB chunks; at 128 MB, 32 MB
  chunks. Peak tracks the budget, never the file.
- **Backpressure** — with 20 of 24 MB held elsewhere, the first chunk arrives after **1.50 s**
  instead of over-committing (`waits=1, wait_millis=1497`).
- **Fail fast** — a 64 MB chunk against a 4 MB budget raises `MemoryError` naming the fix.

## Preprocessing out of core

`examples/bench_prep.py 200000 512MB` — 200,000 cells × 20,000 genes, 66 M non-zeros, a
0.50 GB `.h5ad`, two laptop cores:

| pass | time | resident |
|---|---|---|
| 1 — QC + per-gene sums | 4.0 s | one block |
| 2 — HVG covariance (2,000 × 2,000) | 8.2 s | one block + one 16 k × 2,000 dense sub-block |
| 3 — projection to 50 PCs | 2.8 s | the same, output memory-mapped |
| total | **15 s** | budget peak **224 MB** of 512 MB, 0 denials; process RSS peak 0.57 GB (Python + numpy + scipy + h5py baseline ≈ 0.25 GB) |

The same run on 20,000 cells takes 0.8 s under 24 MB; time is linear in non-zeros, the
resident set is not. Results agree with a dense float64 computation to 3 × 10⁻⁵ (float32
output) and with scanpy 1.11 on the test matrix.

## The compressed store

`examples/bench_codec.py` (1 M elements each, one core): UMI counts 7.8× at 54 MB/s encode,
106 MB/s decode; sorted indices 10.7×; CSR indices with ~300 nnz per row 2.4×; row lengths
3.2×; random int32 1.0× (no loss, no gain). The sc.h5ad fixture's 23 MB matrix sits packed in
6.2 MB under an 8 MB budget whose peak stayed at 7.9 MB.

## The orchestrator

`examples/bench_orchestrate.py`, synthetic inputs, one laptop core. The *before* column is
0.16.0 as released; *after* is 0.16.1.

| Path | Before | After | What changed |
|---|---|---|---|
| `describe()` 5,000 inventory rows | 140.6 s | 1.7 s | one compiled alternation per vocabulary table instead of a regex per synonym per cell (the `re` cache thrashed past 512 patterns) |
| `annotate()` 2,000 pieces | 48.5 s | 0.5 s | the catalog and library folds are cached against the file's size and mtime and advanced on each append, so a record-after-lookup loop never re-parses the log |
| `Catalog.get()` ×200 over 2,000 entries | 2.4 s | 1 ms | one entry copied per call, not all of them |
| `resolve()` 2,000 pieces against a 2,000-entry catalog | 1.6 s | 0.2 s | entries indexed by every accession they mention |
| `GeneMap.expand()` 20,000 names | 2.5 s | 26 ms | set-backed de-duplication (it was `not in list`) |
| `GeneMap.from_table()` 20,000 rows | 740 ms | 330 ms | a fast path for cells with no separator |
| `build()` 200k × 2k cells (8 M nnz) h5ad, 64 MB budget | 6.4 s · 362 MB Python heap | 3.2 s · 75 MB | X and obs streamed block by block to the writers; one gather through a lookup table instead of CSR column selection; intermediates dropped before each block is handed over |
| `MtxTriple` 5 M entries, no budget | 7.1 s | 0.7 s | SciPy's C reader when the matrix fits half the budget |
| `MtxTriple` 5 M entries, 64 MB budget | 7.1 s | 1.7 s · 15.6 MB peak | block-wise `numpy.fromstring` parse, chunk floor rule applied |
| `Dataset.search` 10 M-row int64 column, full range | — | 0.87 s | unchanged: on par with pyarrow decoding the same row groups (0.91 s), so exact filtering costs nothing measurable |

The build's Python heap sits at roughly four to five times the reader's lease: the block itself,
its COO expansion, the int64 cell index and the mask are all live at once. That ratio is the
honest cost of the COO conversion, and the budget governs the lease, not the transient copies.

## Navigation

From `examples/navigate.py`:

| measurement | value |
|---|---|
| FASTQ index build (61.7 MB, 200k reads) | 4.2 s, one pass |
| Index sidecar size | 35 KB (**0.054 %** of the file) |
| Re-aggregate index to any zoom level | **< 1 ms** |
| Sidecar reuse on second open | 0.2 ms |
| GC across a 64× stride range | 0.5000 → 0.5008 |

## Stereo-seq

From `examples/stereoseq_tour.py`, on a 4000 × 4000 fixture chip with 3M observations:

| zoom | bin | observations | resident |
|---|---|---|---|
| 0 | bin1 | 2,946,567 | 28.1 MB |
| 3 | bin50 | 1,017,051 | 9.7 MB |
| 5 | bin200 | 116,221 | **1.1 MB** |

| operation | result |
|---|---|
| All 300 genes at bin10 | 2,560,898 obs in 0.99 s |
| One gene at bin10 | 14,893 obs in 0.007 s — **139×** |
| GEM index (600k rows) | 0.2 s, extent and per-gene totals |
| Window estimate vs actual load | 134,004 estimated, 132,260 actual (**1.3 %**) |
| Registration check | peak expression (0.81, 0.52) vs morphology (0.82, 0.47) |

## Search

| measurement | value |
|---|---|
| Sequence search throughput | 30 Mbase in 0.38 s — **80 Mbase/s**, single core |
| Parquet pruning, unsorted | 20/20 row groups, 100 % selectivity |
| Parquet pruning, sorted by the predicate column | 2/20 row groups, **9.2 %** |

## Searching the journal

A synthetic journal of 1,000,000 records (468 MB, `examples/bench_journal.py`), warm page
cache, best of three, every tool printing or returning the matching lines:

| query | hits | grep -n | rg -n | trigram index | blocks read |
|---|---|---|---|---|---|
| a selection id, in 10 lines | 10 | 518 ms | 174 ms | **3 ms** | 10 / 15,625 |
| a gene, in ~1 % of records | 9,868 | 455 ms | 162 ms | **80 ms** | 7,316 / 15,625 |
| a term in every record | 888,888 | 1,227 ms | **1,194 ms** | 1,649 ms | all |
| two characters (cannot narrow) | 749,141 | 1,881 ms | **1,215 ms** | 1,467 ms | all |

| measurement | value |
|---|---|
| Index build, 1 M lines | 8.0 s |
| Sidecar size | 16.4 MB (**3 %** of the journal); 2,965 distinct trigrams, the JSON keys as one-byte marks |
| `update()` after 10,000 appended lines | 37 ms |
| `Journal.search("GFAP", overlaps=…)`, gene and window | 191 ms (DuckDB `read_json_auto`: 640 ms) |

Rare terms — a gene, a file, a selection id — are what the index is for; a term in every
record is a scan whoever does it, and ripgrep scans faster.

## Throughput

| path | rate |
|---|---|
| FASTQ parse, plain | ~680 MB/s |
| FASTQ parse, gzipped | ~90 MB/s |

The 7.5× gap is decompression, not parsing. Replacing zlib with libdeflate or bgzf is the
single biggest available win, and it is on the roadmap.

## Remote Parquet

Measured against a local Range-capable HTTP server counting bytes independently:

| operation | bytes moved | requests |
|---|---|---|
| `describe()` + `stats()` for 2M rows | 65,536 (**0.26 %**) | 1 |
| `stream(columns=["x","y"], limit=250_000)` | 1.33 MB | 3 |
| `sample(n_row_groups=4)` — 400k rows | 5.0 MB | 4 |
