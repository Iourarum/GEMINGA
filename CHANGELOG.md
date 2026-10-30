# Changelog

## 0.18.0 — 2026-10-13

### Added
- `geminga.journal` — the durable trail of streamed pieces. Every `View.load()`, `chunks()`
  stream, `overview()`, `GefFile.read_window()` / `rasterize()`, `select()` / `select_polygon()`
  / `select_values()` and `similar()` appends one JSON record to `~/.geminga/journal.jsonl`
  (schema `geminga.journal/1`): the exact half-open window in the dataset's own coordinates,
  the level (`{bin}` or `{zoom, stride}`), the dataset key with the file's fingerprint, rows,
  bytes, seconds, and who asked. The explorer tags its reloads with the frame, `describe()` with
  the selection id, `Staging.extract` with the region key, `handoff` with the ROI and targets,
  the `Stage` with `stage`; `recording(**tags)` does the same for your own code and `quiet()`
  silences internal reads. The first opening of a GEF against a journal records its extent,
  bin levels and gene list, so a gene name grepped in the journal says which chips carried it.
  `Journal.search(text, overlaps, within, dataset, op, selection, since, until, genes)`,
  `grep()` (through `rg`, `grep` or Python), `datasets()` (per-dataset fold: pieces, bytes,
  levels, union of windows), `visited()`, `coverage()` (how many times each part of a chip was
  streamed), `replay(entry)` (the same file, level and window reopened; refuses a file whose
  size no longer matches), `duckdb()`, `to_parquet()`, `tail()`, `print_table()`;
  `python -m geminga.journal tail|grep|datasets|where|coverage|index|engine`;
  `GEMINGA_JOURNAL=path|0`, `g.use_journal()`. The test suite journals into a temporary file.
- `TrigramIndex` in the Rust core and `Journal.index()` — a trigram index over a line-oriented
  file in the shape of Microsoft's tgrep: blocks of 64 lines, delta-varint posting lists of
  block ids, trigrams present in every block kept as a one-byte mark, a literal query touches
  only the blocks holding all of its trigrams, lines appended since the last `update()` are
  scanned so results are complete while the index lags. Saved beside the file as `<file>.tgx`,
  extended incrementally. `Journal.search()`/`grep()` use it when it exists (`Journal.engine`).
  Numbers against grep, ripgrep, Python and DuckDB in `docs/benchmarks.md`
  (`examples/bench_journal.py`). `docs/journal.md`.

## 0.17.0 — 2026-10-11

### Added
- `geminga.prep` — the standard single-cell preprocessing out of core, under the budget:
  `Prep`/`prep()` with `qc()`, `filter_cells()`, `filter_genes()`, `hvg()` (Seurat flavour),
  `pca()` (exact, from the HVG covariance accumulated in one pass) and `run()`; `PrepResult`
  with `to_anndata()` (obs/var/obsm/varm, no X) and `write()`. Three passes over budget-sized
  blocks; checked against a dense float64 implementation and against scanpy 1.11 (totals and
  mitochondrial fractions equal, 99/100 HVGs). 200 k cells × 20 k genes under 512 MB in 15 s
  (`examples/bench_prep.py`). `docs/prep.md`.
- `geminga.codec` — compressed resident blocks (from the 2026-10-10 review session, now
  shipped): `BitpackCodec` (per-block frame-of-reference / zigzag-delta packing of 32-bit
  integers, host decoder vectorised over all blocks, GPU decoder in WGSL tested bit for bit
  against it), `BytesCodec` (zstd/lz4/zlib), `encode_csr`/`decode_csr`, `CompressedStore`
  (`put`/`get`/`get_device`/`evict`/`report`; packed bytes leased while stored, raw bytes only
  while held; `tier="vram"` keeps payloads in device buffers), `ratio_report`. New in this
  release: the plug-in contract — `check_codec()` conformance harness, `register_codec()` /
  `get_codec()` / `available_codecs()` with `geminga.codecs` entry points, `device_layout`
  (`lanes-u32` or `native`) on `Encoded`/`DeviceArray` — plus tables: `put_table()`,
  `get_table()` and `duckdb()` register packed tables as DuckDB views. `examples/plugin_codec.py`
  is the template. `docs/codec.md`.
- `geminga.reserve` — the shared-GPU reservation server and client (from the same session, now
  shipped): `ReservationServer`/`serve()`, `reserve()`, `Reservation` (budget whose `vram`
  capacity is the grant; heartbeat; `store()`), `Policy` (ttl, idle reclaim, share cap, queue).
  `python -m geminga.reserve --serve ...`. `docs/reserve.md`.
- `Budget.try_acquire(tier, size)` — one locked check-and-take; `workspace.try_acquire` uses it.
- `docs/honest.md` — what the library is and is not, written for someone deciding whether to
  adopt it; now the first page of the reference section.
- `tests/test_fixes_core.py`, `tests/test_fixes_audit.py`, `tests/test_prep.py`; the six
  `tests/test_fixes_*.py` suites and `CHANGES-A..F.md` notes from the review session (388
  regression tests: orchestrate/harmonize/handoff, stores and planners, infrastructure,
  visual/neuro/NGFF/spatial, readers, views/Stereo-seq/GEF index).

### Fixed (core, from an independent audit of 0.16.1)
- `Budget.acquire` (and the FASTQ/GEM streams) waited on the tier **holding the GIL**, so no
  Python thread could ever release the bytes being waited for: every wait froze the
  interpreter for the whole timeout and then failed. The wait now releases the GIL.
- The wait deadline restarted on every `release` (each one wakes every waiter), so a tier with
  sibling churn could block far beyond `timeout_s`. The deadline is measured from the first wait.
- Chunks were materialised first and leased afterwards; the planned chunk is now leased before
  it is read (waiting without the GIL) and the lease resized to what was materialised, so an
  oversize chunk is refused with nothing resident and backpressure applies before the read.
- The default chunk for Parquet streaming used a quarter of the tier's *capacity*; it now uses
  a quarter of what is *free*, floored at 1 MB but never above what is free (the chunk floor
  rule every other reader applied).
- `Dataset.sample()` read outside the attached budget; it is now governed batch by batch.
  `sample()` and `search()` return lists that are the caller's and are not leased — now said in
  their docstrings.
- `Budget(timeout_s=float("inf"))` panicked (`Duration::from_secs_f64`); it now means "wait as
  long as it takes" (capped at ten years).
- `GemIndex.build` ignored `#OffsetX`/`#OffsetY`, so on any GEM with offsets (normal SAW output)
  every window landed in cell (0, 0) and `GemView.cost()/overview()` were wrong; it now applies
  the offsets like `GemStream` and `gem_extent`. The sidecar carries a version and an older
  one is rebuilt rather than trusted.
- `TenXView`: a decimated stream kept the full column block alive beside its released lease;
  the block is now dropped inside the lease. A decimated `load()` leases the kept parts as they
  accumulate, and `cost()` prices it as the output plus one block (it priced the larger of the
  two). `cost()["peak_read_bytes"]` reports the transient a modality slice adds.

### Fixed (Python, same audit)
- `ProgressiveCanvas.add` binned by `(w-1)` and truncated (the last row and column got a sliver
  of the range, the first a double share) and binned non-finite coordinates into cell (0, 0).
  It now floors into `w` equal cells; rows with a null coordinate or weight are dropped and
  counted in `n_dropped`.
- `Staging.extract` left the view's lease held after the region was written.
- `RemoteFile.read` grew a bytearray block by block; it now fills one preallocated buffer.
- `H5adReader` and `TenXMatrix` kept int64 index arrays alive beside the int32 copies scipy
  made, across the yield (`ObsmReader` and `RowView` already did not).
- `index_gem` swallowed a failed sidecar save, so every open rebuilt the index; it now falls back
  to the private index home (`~/.geminga/gefidx/<hash>/`, the same key the vault uses) and warns
  once with both paths.
- `GefFile.rasterize` accumulated MID counts in float32 (exact only to 2^24 per pixel); the
  accumulator is now an exact uint32, widened to float64 only if a window's total would pass
  2^32, at the same four bytes per pixel.
- GEF `maxX`/`maxY` are read as the inclusive maxima SAW writes, so `extent` is one past them;
  before, the last column and row of a chip fell outside its own extent. Without the
  attributes the extent still comes from the coarsest level. Fixture writers and tests follow
  the SAW convention.
- `view._acquire` (every view and `_Hold` lease) used `available()` then `acquire()`, two calls
  between which another thread could take the bytes; it uses `Budget.try_acquire` when the
  core offers it.

## 0.16.1 — 2026-10-04
Performance (numbers in docs/benchmarks.md, script `examples/bench_orchestrate.py`)
- `Vocabulary`: one compiled alternation per table — `describe()` on 5,000 rows 140 s → 1.7 s.
- `Catalog` and `SelectionBook`/`Library`: the JSONL fold is cached against the file's size and
  mtime and advanced on the object's own appends; `get()` copies one entry — `annotate()` on
  2,000 pieces 48 s → 0.5 s. Other handles still see every append (the signature is re-checked
  on each call).
- `resolve_pieces`: catalog entries indexed by accession — 1.6 s → 0.2 s at 2,000 × 2,000.
- `GeneMap`: set-backed de-duplication in `expand()`/`align(panel=)`, a no-separator fast path
  in `_split` — `expand()` of 20,000 names 2.5 s → 26 ms.
- `Orchestrator.build()`: X and obs are streamed block by block into Parquet/Arrow writers
  (nothing larger than a block is held); COO mapping is one gather through a lookup table;
  intermediates are released before each block is handed over — 362 MB → 75 MB Python heap
  for a 200k-cell piece under a 64 MB budget, 6.4 s → 3.2 s.
- `MtxTriple`: SciPy's C reader when the matrix fits half the budget (10×), otherwise a
  block-wise `numpy.fromstring` parse (4×); the chunk floor rule now applies to it too;
  `pattern` matrices read as ones.
- Precompiled regexes for counts, gene tokens and host names; cells shorter than an accession
  are skipped.
Changed
- `Orchestrator.annotate()` is idempotent: a hole that already has an open `need-data` flag gets
  the flag updated rather than a duplicate, and a piece that has since arrived gets its flag
  resolved (`{"library_resolved": True}` in the result).
Added
- `examples/bench_orchestrate.py`; tests for the mtx fast/streaming paths, pattern files, and
  annotate() idempotency.

## 0.16.0 — 2026-09-26
Added
- `geminga.orchestrate`: `Orchestrator` / `orchestrator()`, `Piece`, `Resolution`, `Blueprint`,
  `Puzzle`, `Coverage`, `read_descriptions`, `find_accessions`, `resolve_pieces`,
  `RESOLUTION_STATUSES`, `BLUEPRINT_SCHEMA` (`geminga.blueprint/1`). Descriptions in CSV/TSV,
  workbook tabs (auto-picked by accession density), Parquet, JSONL, frames or dicts become pieces
  with accessions pulled from every text cell (GEO, SRA/ENA, BioProject, BioSample, EGA, dbGaP,
  ArrayExpress, GSA, CNGB, STOmics, SCP, DUOS, OpenNeuro, Zenodo, figshare, Mendeley, DOIs,
  URLs), facets in the catalog vocabulary, genes by role, counts read from prose. Resolution
  is offline by default and never transfers: local (catalog, vault, `search_dirs`, `attach()`),
  streamable, fetchable, controlled, manual, unresolved — with the earning route, price, and
  the controlled-access components of otherwise open pieces. `annotate()` writes catalog records
  and `need-data` library flags for holes; `coverage()` recomputes a gene × piece P/X/x/F/R
  evidence matrix and reports names the gene map cannot place. `build()` executes a blueprint
  under the budget into a `Harvest` — `.h5ad`, Cell Ranger `.h5`, mtx triples aligned on one key
  (or a gene panel) as COO per piece with shared `obs`/`var`; CSV/TSV/Parquet pieces streamed
  with gene columns harmonised — and writes `blueprint.json` provenance; holes are recorded,
  never fatal. `Puzzle.to_anndata()`, `.measured()`; `Orchestrator.save()/load()`.
- `geminga.harmonize`: `Vocabulary` (assay, perturbation, organism, access, disease, context
  synonyms; token-boundary matching; extensible), `GeneMap` (`from_table` with `members=` for
  complexes and `group=` for designations, `from_hgnc`, `from_features`; `to`, `expand`,
  `canonical`, `members`, `ambiguous`, `align` → `Alignment` with a per-piece loss report and
  `take()`, `panel_table`), `ColumnMap` (canonical obs columns), `Harmonizer`, `id_type`.
- `KINDS` gains `piece` and `piece-table` (additive; `geminga.harvest/2` unchanged).
- `examples/orchestrate_inokan.py`, `examples/orchestrate_figure.py`, `docs/orchestrate.md`,
  `tests/test_orchestrate.py` (36 tests).
Fixed
- `Dataset.search` now applies the range predicate to rows; it returned whole candidate row
  groups before, and its docstring promised otherwise. Nulls never match; a non-numeric column
  raises. `tests/test_navigation.py` covers exact counts, floats and nulls.
- The Rust core passes `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check`
  (doc-comment tabs, `map_or(true, …)`, dead-code annotations with reasons, PyO3 argument
  counts); a `[tool.ruff]` section makes `ruff check python examples` pass, so the CI lint job
  can go green for the first time.
- `examples/range_server.py` no longer `chdir`s the whole process; it broke every relative
  fixture path in tests that ran after `test_remote.py`.
- `Library.query` used a lambda assignment; `harmonize`/`orchestrate` imports trimmed.
Notes
- Wheel built for CPython 3.11 this time (the sandbox moved); the source builds on 3.9+.

## 0.15.0 — 2026-09-25
Added
- `ObsmReader`, `obsm_keys`, `open_obsm`; `g.open(path, obsm=key)` and the `path::obsm/key` form
  (from the HyVESTA work).
- `CATALOG_SCHEMA`, `FACETS`; `CatalogEntry.facets`, `.local_paths`, `.SCHEMA`, `.from_event`;
  `Catalog.facets_for`, `.facet_table`, `.validate`, `.note_local`; `record(facets=, local_paths=)`,
  `annotate(facets=)`; every catalog event carries `schema`.
- `ROI.kind`, `ROI.stat`; `Harvest.add_table`, `.for_kind`, `.significant`; `KINDS`; manifests
  `geminga.harvest/2`, dispatch `geminga.dispatch/2` (additive; `/1` files load).
- `docs/interop.md`.
Fixed
- The 1 MB chunk floor could exceed a small budget in `H5adReader`, `TenXMatrix`, the Arrow reader
  and every `View`; it is now capped at what is free.
- `__version__` now reports the installed distribution rather than the Rust core's compile stamp.
- Static-analysis pass: unused imports, empty f-strings, over-broad exception asserts in tests.

## 0.14.0 — the library: flags of every kind, workflow, queries, links, BED/GeoJSON/PRISM export
## 0.13.0 — selections: mark, describe, similar-region search, leads
## 0.12.0 — remote listing, selective fetch, GEFs opened over FTP/HTTP, CNGB accessions
## 0.11.0 — GEF spatial index (Z-order sidecar), per-location aggregation, glioma phantom
## 0.10.0 — HUD, interactive explorer, OME-Zarr export
## 0.9.0  — Polars integration, wgpu stage
## 0.8.0  — 10x by feature type, Visium/Xenium bundles, multiplexed OME-TIFF, modality overview
## 0.7.0  — MRI volumes and cohorts; budget accounting on every view
## 0.6.0  — acquisition route planning and remote peeking
## 0.5.0  — vault, staging, device pools, pipeline handoff
## 0.4.0  — workspaces and cells, profiling, catalog, assistant
## 0.3.0  — Stereo-seq GEF/GEM, multimodal registration, Sphinx docs
## 0.2.0  — navigation, cost before load, search, progressive rendering
## 0.1.0  — the budget governor and the first readers
