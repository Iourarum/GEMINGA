# CHANGES-F: view, Stereo-seq and GEF-index fixes

Scope: `python/geminga/{view,stereoseq,gefindex}.py`, `__init__.py` (PathLike in `view()`/`open()`; no new
public names were needed), the new `tests/test_fixes_views.py`, plus a lint fix in `tests/test_navigation.py`.
`formats.py`, `tenx.py`, `obsm.py` and `frames.py` were not edited (only read and imported from: `resolve_chunk_bytes`,
`csr_bounds`, `_spatial_axes`, `TiffTileArray.window`).

Process: every finding got a regression test first. Run against a snapshot of the original four files (with
E's reader fixes in place), **37 of the 38 tests in `test_fixes_views.py` fail**, each for the reason it targets
(70 MB heap for a 3 MB window read, 63 MB read for a 500-record FASTQ window, a zoom 1 that halves the rows, ...).
The one that passes is a control (an unclamped 2-D zoom round trip already worked). Every memory test uses an input
at least 10x its budget and asserts budget `peak <= capacity`, no denials, nothing left leased, and a
`tracemalloc` heap peak under 3x the capacity, during `load()` and during `chunks()`. Arrow/Rust allocations
are invisible to `tracemalloc`, so those tests also bound the real high-water mark (`VmHWM`) or the bytes read
(`/proc/self/io`). GEF fixtures are always copied to a temp dir first.

Randomized checks outside the suite, against brute-force references (all clean):
* 120 Parquet windows (single file and 3 files with uneven row groups, strides 1-128, chosen chunk sizes).
* 90 h5ad windows (CSR fixture, dense `X`, 1-D `obsm`).
* 150 FASTX windows (plain FASTQ seek path, gzip scan path, multi-line FASTA).
* 90 image windows (YX, 6-channel CYX CODEX, RGB YXS) checking `load`, `chunks` counts and `tiles` assembly.
* 120 GEF windows (all bins, with/without sidecar, gene selections, 2-24 MB budgets) checking `load`,
  `chunks`, raster mass and `gene_profile`.
* 6,000 random navigation steps checking the window/zoom invariants.

## Fixes

1. **Stereo-seq zoom is the pyramid, never a stride (stereoseq.py, view.py).** `StereoSeqView.stride` is
   always 1. Zoom picks the bin level, and past the coarsest level the view stays on it. `auto_zoom` tries
   only the pyramid's levels. `cost()`, `load()` and `chunks()` return the same rows: at every zoom, the MID of
   the full chip and of a sub-window equal those of the same window read plainly at that bin.
   *Behaviour:* zoom 1 on a GEF used to return every other row of bin10; it now returns all of them. Other
   views keep stride-based decimation.

2. **GEF reads are leased blocks, with honest estimates (stereoseq.py, gefindex.py).**
   - **Plain reads.** Without a sidecar, a level is read in HDF5 row blocks (aligned to HDF5 chunks). Block
     size follows the chunk floor rule, re-evaluated per block, so blocks shrink as a load's output grows.
     Each block is leased before it is read (record + mask + room for its kept rows) and filtered to the
     window; the lease shrinks to the kept rows while they are handed over. `read_window(..., budget=, tier=,
     chunk_bytes=)` joins the pieces column by column, leasing one column as the transient. `load()`
     reserves its planned output first and grows the reservation only if the window is denser than
     estimated. So the peak is output + one block + one column. Without a budget, blocks are 16 MB.
   - **Gene reads.** Each selected gene's contiguous slice is read the same way, block by block.
     `cost()["read_bytes"]` is the slice's bytes (`read_path` "gene-slices"). The `gene` column's width (and
     the per-observation cost) comes from the longest selected name.
   - **Sidecar reads.** `SpatialIndex.iter_window` streams the surviving row groups through the dataset with
     the budget attached. The requested batch size is translated into the reader's footer-based sizing:
     the sidecar's encoded bytes per row are far below the 16 B a row takes decoded, and a "1 MB" batch
     came out at 2.2 MB. The exact window predicate is applied per batch and kept pieces are leased.
     `SpatialIndex.read` (kept, now with `budget=`) is built on it.
   - **`estimate_rows`.** With a sidecar: from row-group statistics (surviving groups, partly covered ones by
     the share of their box). Without one: from coarsest-level occupancy, as follows.
     - One cached read of the coarsest level's most widespread genes (at most 256 genes / 2^18 rows) gives
       MID per coarse bin.
     - A per-level saturation model, `(C/b)^2·G·(1-e^(-λ_b·w))`, turns that into rows per coarse bin.
       `λ_b` is solved so the whole grid holds exactly `n_obs(b)`.
     - The window's bin-aligned envelope intersects the coarse cells by area.
     - Accuracy over 60 random windows per level (bins 1, 20, 100) on each of the two fixture chips (those
       holding at least 1,000 rows): 0.82-1.49x of actual. Area fraction gave 0.21-12.7x on the same windows.
       The whole chip is exact.
     - Signature kept; keyword-only `budget=`/`tier=` lease that one read (area fallback if it cannot).
   - **`cost()` keys.**
     - `units`/`total_bytes`: estimated rows kept x bytes per observation.
     - `read_bytes`, `rows_read`, `read_path`, `block_bytes`.
     - `resident_bytes`: the largest chunk.
     - `peak_bytes`: output + smallest block + one column.
     - `chunk_peak_bytes`: one block.
     - `fits` is one block within free; `fits_whole` is `peak_bytes` within free.
     - `chunks`: blocks read (blocks with no row in the window are skipped).
   - **Rasters.** `image()`, `overview()` and `paired()` are checked before anything is read (raster + a read
     block; for the morphology, crop + grid + one decoded tile). Rows are streamed into the raster, never held.
   - *Behaviour:* `StereoSeqView.cost()["resident_bytes"]` is per chunk now (it was the total, which is still
     `total_bytes`). Tiny budgets now see `fits_whole=False` sooner because the read block is priced.

3. **Load paths lease the output first and read only what they keep (view.py).** `_load` reserves the planned
   output (`_hold`) before reading, then settles the hold to what was handed out.
   - **`ImageView`.** Handles N-d arrays (Y/X axes found as `TiledImageReader` does). It reads through
     `TiffTileArray.window(y0,y1,x0,x1,step)` or step slicing, so only tiles holding a kept pixel are decoded
     and only kept pixels are allocated; one decoded tile is leased as the transient. `overview(n)` reads with
     step `ceil(span/n)`, keeps RGB, and folds channels by max.
   - **`RowView` over Parquet.** Reads only the row groups that hold a grid row. Batches are sized from
     decoded widths (dictionary-encoded files have footer sizes 0.007 B/row) and decimated as they arrive;
     partial edge batches are copied. Multi-file datasets are read file by file: the Rust `offset` applies to
     the first file only, so it is no longer used.
   - **`RowView` over h5ad / obsm (CSR or dense).** Reads only rows `[lo, hi)`, in `csr_bounds` blocks cut
     inside the window. Output arrays are allocated once at their exact size (nnz from `indptr`) after the
     lease. Dense matrices use strided h5py reads. `cost()["total_bytes"]` is exact; `read_bytes` is the
     blocks actually read.
   - **`SeqView`.** Seeks to the closest index checkpoint at or before `lo`, skips to `lo` and stops at `hi`.
     - The index's `byte_offset`s are short by 1 byte per record (`index.rs` counts `id+seq+2(+qual+3)`).
       The view measures that drift once from the file size; it must be the same integer for every record.
     - It then verifies each checkpoint it uses: the bytes up to the next checkpoint must hold exactly that
       bucket's records (4 newlines per FASTQ record; `'>'` line starts for FASTA; no `\r`).
     - Otherwise (gzip, CRLF, irregular files) it reads from the start with the Rust reader, still stopping
       after `hi`.
     - The Python parser reproduces `FastxStream`'s schema and values (FASTA raw sequences keep their line
       breaks).
     - `cost()` reports `seek`, `seek_byte`, `skip_records`, `read_bytes`.
   - *Measured:*
     - 10-row h5ad window: 102 KB read (was 26 MB).
     - 500 records at the end of the 62 MB FASTQ: 674 KB read (was 63 MB).
     - Stride-8 load of the 72 MB OME-TIFF: 2.1 MB heap (was 152 MB).
     - Stride-32 Parquet load: real HWM under 0.5 MB (was 47 MB).

4. **Lease lifecycle (view.py, stereoseq.py).**
   - **`chunks()`** returns a stream object. It gives back the previous chunk's hold before reading the next,
     and adopts a producer's own lease instead of leasing again. Parquet chunk peak / largest chunk is 1.13x
     (was 2.13x). It closes, releasing everything, on exhaustion, on error, on leaving a `for` loop early, and
     on `release()`.
   - **`release()`** also closes a stream in flight and drops a formats reader's lease on its last chunk
     (`_swap_lease(0)`). Called from inside a producer, as `TenXView` does, it only drops the hold.
   - **`owns_source`.** Views carry it; `StereoSeqDataset.expression()`, `morphology()` and `paired()` lend
     views with `owns_source=False`. `close()` closes `src`/`a`/`m`/`vol`/`gef`/`index` only when owned, so
     `ImageView`'s TIFF and `TenXView`'s HDF5 file are now closed, and a dataset survives
     `with ds.expression() as v: ...`.
   - **Chunk size.** Every view takes `chunk_bytes=`, and `_chunk_cap()` is `formats.resolve_chunk_bytes`. A
     chosen size is honoured (capped at what is free). Planning counts the view's own hold as free, since
     `load()`/`chunks()` give it back before reading; `cost()` does too.
   - **Fail-fast leases.** A view's internal leases raise `CostError` at once when the tier cannot grant them,
     instead of waiting out the budget timeout on a single-threaded read.
   - `StereoSeqView.cost()["chunks"]` is the block count; it used to divide by the whole free capacity.

5. **Navigation (view.py).**
   - Zoom is signed internally (the `zoom` property clamps at 0). A `zoom_in` past full resolution is
     remembered, so `zoom_in(2)` then `zoom_out(2)` returns to zoom 0 on the full window, and the two are
     exact inverses on any unclamped window, 1-D or 2-D.
   - Zoom moves by as many levels as the window actually changed (area ratio per dimension), so `zoom_out`
     of a full window keeps its zoom.
   - Windows are moved inside the extent keeping their width: `pan(0.5)` x4 keeps 1,000 rows, and
     `zoom_in(center=0)` at an edge keeps half the span. `center` may be a tuple per axis.
   - `auto_zoom` tries the view's levels finest first and needs `peak_bytes` within 80 % of free (or `target`).
     When none fits it raises `CostError` naming the cheapest zoom (`err.cost`) and restores the zoom
     (it used to end at zoom 39).
   - One decimation grid, `(i - lo) % stride == 0`, is shared by `load()`, `chunks()` and `tiles()` for
     Parquet, h5ad, FASTX, images and GEM, whatever the chunk size.
   - `ImageView` chunks are groups of whole tiles sized to the chunk cap, and `cost()["chunks"]` counts them.
   - The `TenXView` chunk counts are not fixed here; see deferred.

6. **Wrong answers (stereoseq.py).**
   - `gene_profile(k)` streams the window at the current bin and sums MID per gene. Ties are broken by gene
     order; with a gene selection it ranks those genes.
   - `overview()` is of the window: the coarsest bin that resolves it into at least `min(n, 32)` bins a side,
     one pixel per bin, at most `n` (and 512) a side.
   - Bins are kept when they *overlap* the window (`x < x1 and x + bin > x0`) in block reads, sidecar pruning
     and filtering, and estimates. A 90-unit window inside one bin200 returns its 300 rows (was 0).
   - `paired(shape)`: the morphology is sampled on the expression raster's grid (pixel centres, nearest
     neighbour, zeros outside the image, per-axis read step), so both are `(shape, shape)`; N-d images keep
     their other axes.
   - Extent is half-open everywhere. GEF `maxX`/`maxY` are exclusive. Without them, the bound is the end of
     the last occupied coarsest bin. `GemView._full_window` now uses the GEM index extent as is, because
     `gem_extent` already returns max + 1; adding another 1 gave `(0, 4001, 0, 4001)`. GEF and GEM of the
     same chip now give the same window.
   - `rasterize` uses integer pixel arithmetic, `(x - x0) * shape // span`: x=290 at bin10 on 400 px used to
     land in pixel 28. It streams blocks into the image.
   - `gene_id` is returned only when the gene slices tile the level, as before, so `selections` never sees -1.

7. **The GEF sidecar (gefindex.py, stereoseq.py).**
   - **Locating it.** `_sidecar_dir(path, write=False)` creates nothing on lookup. `SpatialIndex` takes the
     first candidate whose fingerprint matches: beside the GEF, then `~/.geminga/gefidx/<hash>/`. So an
     existing valid sidecar beside a read-only GEF is used. Writing goes beside the GEF when `os.access`
     allows, else to the home fallback; the directory is created only then.
   - **`build_spatial_index(budget=)` is honoured.** The working set is the chunk floor rule on the budget
     (256 MB without one). The level is read in leased blocks, each Z-sorted into a raw run file. Runs are
     merged at most `fan` at a time (buffers of at least 1,024 rows; several passes when needed). The final
     Parquet is zstd with statistics.
     - It uses plain encoding: with dictionary pages plus zstd, pyarrow kept every numpy buffer it was handed
       alive.
     - Polars is no longer needed to build.
     - bin1 (2.95 M rows) under 2 MB: budget peak 1.0 MB, heap 1.3 MB, about 1.6 s. Before: 0 grants and a
       55 MB heap.
   - **Soft cache.** `levels`/`has()` require the level's file to exist and its footer to open; otherwise
     reads fall back to the full-level block read (it raised `OSError`).
   - **Partial rebuilds.** `force=True` rebuilds keep the other still-present levels in `index.json`, which
     is written atomically.
   - **`index_gem(grid=)`.** `grid=None` (the new default, also for `open_gem`) reuses any sidecar; an explicit
     grid that differs from the sidecar's rebuilds it.
   - `StereoSeqView.chunks()` streams the read blocks or row-group batches, filtered and leased; it used to
     yield the whole window as one chunk.

`__init__.py`: `view()` and `open()` accept `os.PathLike` (and lists of them) through `os.fspath`.

## Measured peaks for the regression inputs

The script uses the test fixtures and budgets. Imports and HDF5 were warmed up outside the measurements:
pyarrow imports pandas on its first `pa.array(numpy)`, which adds about 28 MB of module objects to `tracemalloc`.
"Real HWM" is `VmHWM` above the starting RSS. Every row left nothing leased.

| scenario | input | capacity | budget peak | heap peak | real HWM | bytes read |
|---|---:|---:|---:|---:|---:|---:|
| GEF window load, bin1, full-level blocks | 28.1 MB | 3.0 MB | 2.0 MB (0.67x) | 1.7 MB (0.57x) | 808.0 KB | 13.1 MB |
| GEF image(512), whole chip bin1 | 28.1 MB | 3.0 MB | 2.1 MB (0.69x) | 2.1 MB (0.70x) | 500.0 KB | 12.7 MB |
| GEF chunks(), whole chip bin1 (no sidecar) | 28.1 MB | 3.0 MB | 944.3 KB (0.31x) | 1.5 MB (0.50x) | 0 B | 13.1 MB |
| paired(256), bin1 + ssDNA | 28.1 MB | 3.0 MB | 1.2 MB (0.39x) | 1.6 MB (0.52x) | 136.0 KB | 14.7 MB |
| GEF gene slice load (BIG, 1.5 M rows) | 13.7 MB | 1.5 MB | 1.4 MB (0.95x) | 1.3 MB (0.90x) | 12.0 KB | 3.9 MB |
| GEF chunks(), whole chip bin1 via sidecar | 28.1 MB | 3.0 MB | 633.9 KB (0.21x) | 506.7 KB (0.16x) | 4.5 MB | 6.9 MB |
| GEF window load via sidecar | 28.1 MB | 3.0 MB | 1.9 MB (0.63x) | 1.5 MB (0.51x) | 68.0 KB | 941.3 KB |
| build_spatial_index(levels=[1]) under 2 MB | 28.1 MB | 2.0 MB | 1024.0 KB (0.50x) | 1.3 MB (0.64x) | 1.1 MB | 381.6 MB* |
| ImageView load, stride 8 | 72.0 MB | 4.0 MB | 1.6 MB (0.41x) | 2.1 MB (0.54x) | 0 B | 72.0 MB |
| ImageView overview(256) | 72.0 MB | 4.0 MB | 640.0 KB (0.16x) | 1.1 MB (0.29x) | 0 B | 72.0 MB |
| ImageView chunks(), stride 8 | 72.0 MB | 4.0 MB | 1.4 MB (0.36x) | 2.1 MB (0.54x) | 0 B | 72.0 MB |
| RowView(parquet) load, stride 32 | 68.7 MB | 4.0 MB | 3.6 MB (0.89x) | 22.8 KB (0.01x) | 4.0 KB | 42.1 MB |
| RowView(parquet) chunks(), stride 32 | 68.7 MB | 4.0 MB | 1.1 MB (0.28x) | 19.6 KB (0.00x) | 2.0 MB | 42.1 MB |
| SeqView window near the end (seek) | 61.7 MB | 4.0 MB | 1.3 MB (0.33x) | 824.7 KB (0.20x) | 16.0 KB | 673.5 KB |
| SeqView chunks(), whole file stride 128 | 61.7 MB | 4.0 MB | 1.1 MB (0.28x) | 3.0 KB (0.00x) | 0 B | 61.8 MB |
| RowView(h5ad) 10-row window | 22.9 MB | 2.0 MB | 22.6 KB (0.01x) | 41.9 KB (0.02x) | 68.0 KB | 101.3 KB |
| RowView(h5ad) stride 16 load | 22.9 MB | 2.0 MB | 1.6 MB (0.79x) | 1.4 MB (0.71x) | 152.0 KB | 26.0 MB |
| RowView(h5ad) chunks(), whole X stride 16 | 22.9 MB | 2.0 MB | 515.4 KB (0.25x) | 332.6 KB (0.16x) | 0 B | 31.0 MB |

\* The build's reads include its own run files, read back once per merge pass.

The same scenarios on the original code are the 37 failing regression tests: 70.7 MB heap for the GEF window,
152 MB for the image, a 47 MB real HWM for Parquet, 63 MB read for the FASTQ window, 26 MB read for 10 h5ad rows,
a 55 MB heap and no leases for the index build, and a 2.1x double lease per Parquet chunk.

## Declined / deferred

- **`TenXView` chunk counts at large strides (tenx.py, not mine).** Its `_chunk_plan` counts every column block,
  but `_decimated` skips blocks with no kept cell (10 planned vs 6 yielded at stride 512 under 300 KB). Fix
  there: count only blocks with `a + (-(a - c0)) % stride < b`. Its fd leak *is* fixed (`close()` now closes
  `m`), and it runs unchanged on the new `View` base (its tests pass).
- **Rust `SeqIndex` offsets (index.rs).** `byte_offset` drifts by 1 byte per record (`rec_bytes` misses a
  separator). It is worked around by calibration plus per-checkpoint verification, which also accepts a fixed
  index (drift 0). Files that fail verification (CRLF, `+id` lines of varying length) read from the start.
- **Rust `GemIndex` (gem.rs).** `gx`/`rows_in` treat `max_x` as inclusive (`max_x - min_x + 1`) although
  `gem_extent` returns max + 1, and `rows_in` includes the cell at `x1`. Estimates are slightly high; not mine.
- **Decode amplification.** A decimated image read without a pyramid level still decodes every tile that
  holds a kept pixel (stride 8 of the 72 MB OME-TIFF reads 72 MB; memory stays at one tile). Choosing a
  pyramid level per zoom needs level-aware `ImageView` construction. Likewise, decimated FASTX and Parquet
  reads (stride below a row group) read their whole window. Both are inherent, and `read_bytes` says so.
- **Unleased one-off reads.**
  - `GefFile.extent` for a GEF without `minX`/`maxX` reads the coarsest level once in 8 MB blocks, unleased.
  - The occupancy read behind `estimate_rows` is unleased when the caller passes no budget (Stage, Explorer):
    at most 2^18 rows plus a coarse grid, cached.
  - Rasters from `image()`/`overview()`/`paired()` are checked and leased while built but not held after
    they are returned (as before).
- **Unchanged behaviour.** CSC h5ad is still refused, as the readers do. A single row, tile or gene block
  larger than the free budget is a `CostError`, since nothing smaller can be read.
- **API notes for other owners.**
  - `cost()` gains `peak_bytes` and `chunk_peak_bytes`; `fits`/`fits_whole` include the read block.
  - `chunks()` returns an iterator object (with `close()`), not a generator.
  - `View._lease` is a `_Hold`.
  - `index_gem`/`open_gem` default `grid=None`.
  - `examples/remote_cngb.py` calls `_sidecar_dir(url)` to delete a sidecar; that still works and no longer
    creates the directory first.
  - README text describing Stereo-seq stride or `resident_bytes` as a total is now stale (not my file).
- **Tests and fixtures.**
  - The read-only-share test patches `gefindex.os.access`/`makedirs`/`open`, because the suite runs as root,
    where permission bits do not stop writes.
  - Timing the *original* code's own `test_stereoseq.py` (for comparison) let its write probe touch the mtime
    of `fixtures/chip.gef.gemidx/`. No file was left; the mtime was restored from the pristine tree. The new
    code never writes there: the directory stayed unchanged across full runs.
  - Test files outside the five named were not run. Ad-hoc smoke runs passed for `to_ome_zarr`,
    `selections.describe`/`similar`, `render.chip`/`navigator`, `View.polars()`, `VolumeView` and `TenXView`.

Final run: `PYTHONPATH=python timeout 250 python3 -m pytest tests/test_fixes_views.py tests/test_navigation.py
tests/test_stereoseq.py tests/test_gefindex.py tests/test_fixes_readers.py -q -x` gives
`156 passed, 1 skipped in 62.87s (0:01:02)`. `ruff check` on the eight owned files: all checks passed.
