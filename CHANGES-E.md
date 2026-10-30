# CHANGES-E — reader correctness fixes

Scope: `python/geminga/{formats,tenx,obsm,frames}.py` and the tests in `tests/test_fixes_readers.py`
(new), `test_budget.py`, `test_tenx_spatial.py`, `test_hyvesta_additions.py`. Nothing else was edited;
the chunk-dict keys and reader attributes named in the brief are unchanged (keys were only added).

Process: each finding got a failing regression test first, then the fix. Against a snapshot of the
untouched code 78 of the 91 tests in `test_fixes_readers.py` fail (12 are controls that pass, 1 needs
`imagecodecs`); 89 of the 137 tests across the four files fail. Every streaming test runs under a `Budget`
at least 4x smaller than its input and asserts `peak <= capacity`, no denials, nothing left leased, and a
`tracemalloc` heap peak under 3x capacity (measured at most 1.8x, mostly 0.4x-1.1x). Randomised
checks outside the suite, all clean: 3,000 `TiffTileArray` selectors over 12 TIFF layouts vs numpy, 2,000
`csr_bounds` cases vs a naive cutter, 183 MatrixMarket stream cases (CRLF, blank lines, pattern, gz/plain) vs
SciPy, 80 `to_h5` matrices (duplicates, empty cells, floats, ram / spill / 2 KB-30 KB budgets) vs SciPy - the
only failures were budgets smaller than one cell's sort space, which raise `MemoryError` by design.

## Fixes

1. **obs columns (formats.py, obsm.py).** New `formats.read_obs_column(f, name, start, end)` dispatches
   on `encoding-type` and is used by `H5adReader` and `ObsmReader`.
   - `categorical` -> object array, code -1 is `None` (it used to index the last category);
     `nullable-integer` -> float64 with NaN (object + `None` only if a value is not exact in a float);
     `nullable-boolean` / `nullable-string-array` -> object with `None`; vlen and fixed-width strings
     -> `str` (they came back as `bytes`); plain numeric as stored. The pre-0.7 `__categories` layout works.
   - A missing column is a `KeyError` naming the columns that exist, raised at construction
     (`ObsmReader` used to skip it silently; `H5adReader(obs_columns=...)` raised `TypeError` on any
     categorical). `obs_columns="x"` is accepted as one column.
   - `H5adReader.obs_names` / cached `var_names` (from the `_index` attribute), new `obs_names=` kwarg
     (default False) to put `obs_names` in chunks, `layer="raw"` (`raw/X` + `raw/var`; a real layer called
     `raw` wins, `layer=".raw"` forces `.raw`), a missing layer lists the layers.
   - `ObsmReader` on a DataFrame-valued entry raises `ValueError` naming the entry, its columns and the
     keys it can stream; `obsm_keys` lists it as `(n_rows, n_cols, "mixed", "dataframe")` instead of omitting
     it; other unreadable groups raise `ValueError` rather than `AttributeError`. `obs_names` chunks are read
     per slice instead of materialising the whole index.
2. **Imaging (formats.py).** `TiffTileArray.shape/axes` are now the whole series (`(6,1500,1500)` "CYX" for
   the CODEX file, `(H,W,3)` "YXS", `(3,H,W)` "SYX", `(Z,C,H,W)` ...). Indexing is numpy's: ints, slices with
   any step (negative too), `Ellipsis`; an integer now drops its axis (it used to leave a length-1 one).
   Only the pages, sample planes and tiles a selection touches are decoded; tiled and striped, contiguous and
   planar, pyramid `level=`, ImageJ hyperstacks. Every decode gets `jpegtables=page.jpegtables` (and
   `jpegheader` for JPEG). New: `window(y0,y1,x0,x1,step)`, `y_axis`, `x_axis`, `s_axis`, `ndim`, `nbytes`,
   `spatial_shape`, `series=`; `th`/`tw`/`page` kept. Series that cannot be mapped onto pages raise
   `NotImplementedError` saying why. `TiledImageReader` tiles over the Y and X axes (from `axes=`, the array's
   `axes`, OME-Zarr `multiscales`, else the rule: short last axis (<=4) = interleaved samples, otherwise the
   last two axes), reads every other axis whole and counts it: side = `isqrt(chunk_bytes / (itemsize *
   channels))`. The 64 px floor is gone; if not even 16 px fits it raises `ChunkTooSmallError` (a `MemoryError`
   and a `ValueError`) saying why. `plan().total_bytes` is the true 27 MB for the CODEX file (was 4.5 MB), a CYX
   zarr under `chunk_bytes=1MB` makes <=1 MB tiles (was 8 MB), `plan().bytes_per_chunk` is the largest tile.
3. **Sparse chunking follows `indptr` (formats.py, obsm.py, tenx.py).** `csr_bounds` cuts with `searchsorted` over
   the cumulative bytes: every block's data+indices <= `chunk_bytes`, a single bigger row is its own block (rows
   are also capped so the sliced `indptr` stays within the chunk). `H5adReader`, `ObsmReader` (CSR) and
   `TenXMatrix` use it. `plan()` now reports the real largest chunk: `bytes_per_chunk` (so `Plan.fits` is honest),
   `detail["largest_chunk_bytes"]`, `["largest_chunk"]` (`index`, start/end, `bytes`), `["fits"]` (<= `chunk_bytes`
   and <= what the budget has free), `rows_per_chunk` = rows of the largest chunk, plus `max_rows_per_chunk`,
   `mean_chunk_bytes`, `mean_bytes_per_row`. The skewed h5ad (first 10% of rows 15x denser) now streams in
   <= 2 MB blocks under 8 MB (the first block was 12.5 MB and raised; `plan()` had said `fits`).
   - 10x: CSC keeps all of a cell's features together, so a modality read still reads the whole column block.
     That is now documented and honest: chunks are cut by the full block, the lease is exactly its bytes (taken
     before the read), `plan()` reports it and says so (`detail["note"]`), chunks gain `resident_bytes`. The
     2%-of-matrix guess is gone. `TenXMatrix.modality_nnz(mod)` counts a modality's entries per cell with one
     cached pass over `indices` (none when the modality is the whole matrix: `diff(indptr)`).
   - `TenXView`: `cost()["total_bytes"]` is what the modality hands over, exact from `modality_nnz` (972,004 B for
     the fixture's ADT; it said 58 KB), `read_bytes` the column blocks read, `resident_bytes` the largest block,
     `fits_whole` asks whether the rows plus one block fit. `load()` assembles a modality block by block into arrays
     sized from its entry count (output leased up front, each block only while read), so a modality whose rows fit
     the budget loads even when its full column blocks do not; chunks are cut from `indptr` and each block is
     leased after the previous hold is given back; strided loads decimate chunk by chunk; `overview()` reads only
     the window.
4. **10x layouts (tenx.py).** `per_cell_totals()` no longer raises `IndexError` when a chunk ends on an empty
   cell (`reduceat` over non-empty columns only), accumulates in float64, leases each block, takes `c0`/`c1`.
   Cell Ranger v2 files (root group named for the genome with `genes`/`gene_names`) open - the dead v2 branch is
   reachable; several genome groups need `group=`; `genome` is the group name. `MtxTriple` finds GEO-style
   prefixed names (`GSM123_matrix.mtx.gz`, `_barcodes`, `_features`/`_genes`, separators `_ . -`) when exactly one
   candidate of each kind exists (`_find` is now a real helper); several samples in one directory raise listing
   them, naming one sample's matrix file picks it. The dimension line must agree with the barcodes/features
   counts, otherwise `ValueError` says which file disagrees, and "transposed" for a cells x features matrix.
5. **`MtxTriple.to_h5` (tenx.py).** In memory when `nnz * 24 <= available // 2` (or no budget): SciPy's reader,
   the `nnz * 24` bytes leased. Otherwise two passes: count entries per cell (-> `indptr`, also learns whether
   values are integers), then scatter by per-cell write cursors into two `numpy.memmap` files in a temp directory
   beside `dest`, then copy into resizable HDF5 datasets a block of cells at a time (indices sorted, duplicate
   coordinates summed). The lease covers the streaming chunk and the copy block. Spill size = `nnz * (4 + itemsize
   of data)` bytes (8 B/entry for counts); taken from the `spill` tier when the budget declares one, free disk
   checked first, directory removed on success and on failure. `last_conversion` records `path` / `spill_bytes`.
   A 4M-entry matrix under 8 MB: Python heap peak 8.9 MB (194 MB before); the in-memory path 115 MB (195 MB
   before). Output dtype: int32 (int64 beyond) when the values are exactly integers (the test was
   `allclose`, which also accepted 100000.4), else float32.
   Iteration: the in-memory fast path now leases the parsed matrix and is taken only if it fits half the free
   budget with a chunk on top; the streaming path yields exactly `entries_per_chunk` entries a batch, parses a
   quarter of a batch at a time, checks indices and entry counts, and leases before allocating. The 1000-entry
   floor on a batch is removed (it exceeded budgets under 12 KB). `close()` releases the parsed-matrix lease.
6. **frames / sinks / Arrow IPC.** `lazy_window` filters, then selects, so the window may be on unprojected
   columns. With a `y` bound `explain` plans x and y: row groups surviving both (exact `row_groups_*`,
   `candidate_rows`), `bytes_to_fetch` scaled by compressed size (`bytes_to_fetch_estimated`), sub-plans under
   `"x"`/`"y"`; predicate columns count towards the bytes. `ParquetSink(path, compression, schema=None,
   lookahead_bytes=8MB)`: with `schema=` (Schema or dict) batches are conformed to it; without it a column's type
   is the first non-null type any batch gives it - batches are held back (<= `lookahead_bytes`) while a column
   is still all-null, and a column still all-null at the limit or at close is written as `string`; later batches
   are cast to the file's schema, an impossible cast is a `ValueError` naming the column. `ArrowIPCReader` applies
   the floor rule, splits a batch bigger than the chunk into `RecordBatch.slice`s (<= chunk, variable-width
   columns shrink-to-fit), leases `piece.nbytes`; for a file `plan()` replays the split so it names the exact
   largest slice (a stream can only estimate: `detail["estimated"]`).
7. **Chunk rule, one implementation (`formats.resolve_chunk_bytes`).** The floor rule applies when the parsed
   value equals the reader's default, so `256 << 20` and `"256 MB"` are the default too. A different value is
   the caller's choice: if it exceeds `budget.available()` the constructor raises `MemoryError` ("cannot fit
   chunk_bytes=100.0 MB in the 'ram' tier (4.0 MB available)") before anything is read or leased. Used by
   `H5adReader`, `TenXMatrix`, `MtxTriple`, `ArrowIPCReader`, `TiledImageReader` (its `tier`), `ObsmReader`.
   Readers also close their file handle if construction fails.

## For the view.py / stereoseq.py / spatial.py owners

- `TiffTileArray` is N-d now: to cut a window use `a.window(y0,y1,x0,x1)` or `a.y_axis`/`a.x_axis`; `a[y0:y1, x0:x1]`
  only means "the first two axes" for a 2-D array. `ImageView` takes `shape[0], shape[1]` as (h, w).
- `RowView._extent` divides `plan().bytes_per_chunk` by `detail["rows_per_chunk"]`: both describe the largest
  chunk now, so skewed data is costed conservatively; the mean is `detail["mean_bytes_per_row"]`.
- `MultiChannelTiff.channel()` still reads the whole series; `TiffTileArray(path)[c, y0:y1, x0:x1]` replaces it.

## Declined / deferred

- **`spatial.MultiChannelTiff`, `view.ImageView`/`RowView`, `orchestrate._PieceReader._obs_slice`, `__init__`
  exports and `open_h5ad(obs_names=)` wiring**: not my files. `_obs_slice` has its own categorical decode that
  breaks on nullable groups and should call `read_obs_column`; `read_obs_column`, `csr_bounds`,
  `resolve_chunk_bytes` are importable from `geminga.formats` but not re-exported.
- **Decode amplification**: a TIFF/zarr tile is decoded whole (512 x 512 x channels) whatever the window; that
  transient is not leased (the window is). Fixing it means aligning windows to the tile grid.
- **Modality reads that skip other features**: impossible with CSC without per-cell seeks; documented instead.
- **`to_h5` spill: a single cell whose entries exceed the chunk** (sorting needs the column) raises `MemoryError`;
  values pass through float32 (integers above 2**24 are inexact); symmetric/`array` MatrixMarket still unsupported.
- **CSC h5ad** is still refused (planned, `streamable: False`). **Arrow IPC streams** have no exact plan.
- **Dtype choices** left to the brief's "float with NaN or object with None": nullable-integer is always float64.
- **Skipped here**: the JPEGTables regression test needs `imagecodecs` (not installed; it passes when present
  and fails on the old code - checked with a scratch install on `PYTHONPATH`); the real-`anndata` test needs
  `anndata` (installed here, so it ran). Test files outside the four named were not run, apart from one
  accidental `-k "mtx or h5"` run that also selected tests from `test_orchestrate.py` (the `to_h5` consumer
  among them); all 32 selected tests passed. An ad-hoc check of `orchestrate._PieceReader` over the new
  readers matched the old output.
