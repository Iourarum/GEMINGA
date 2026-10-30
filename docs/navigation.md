# Navigation

`g.view(target, budget=...)` returns a view over anything GEMINGA can open.

## Moving

```python
v.select(lo, hi)                  # 1-D: rows, records, cells
v.select(x0, x1, y0, y1)          # 2-D: chip coordinates or pixels
v.select_frac(0.25, 0.30)         # same, as fractions of the extent
v.zoom_in(2, center=None)         # narrow; resolution rises to match
v.zoom_out(2)                     # widen; resolution drops to match
v.pan(0.5)                        # shift by half the window span
v.set_zoom(3)                     # resolution only
v.reset()                         # back to the full extent at zoom 0
```

`zoom_in` and `zoom_out` take `constant_cost=False` if you want to change extent without
changing resolution — the cost then grows as you would expect.

## Choosing a level automatically

```python
v.auto_zoom()                     # coarsest-but-still-detailed level that fits
v.auto_zoom("32MB")               # or against an explicit ceiling
v.auto_bin()                      # Stereo-seq: finest GEF bin that fits
```

```{list-table} `auto_zoom` on a 65.3 MB FASTQ
:header-rows: 1

* - budget
  - chosen zoom
  - stride
  - records
  - resident
* - 128 MB
  - 0
  - 1
  - 200,000
  - 65.3 MB
* - 16 MB
  - 3
  - 8
  - 25,000
  - 8.2 MB
* - 4 MB
  - 5
  - 32
  - 6,250
  - 2.0 MB
* - 512 KB
  - 8
  - 256
  - 781
  - 261 KB
```

## Overviews

An overview is a level-of-detail summary built from metadata, never from data pages.

```python
ov = v.overview(256)
ov.series.keys()                  # what is available for this modality
print(ov.sparkline("gc"))         # ▇▃▆▇▇▄▅▆▆▅▄▇▅▅▆▆▆▅▄▆▆▆▇▆▅▅▅▆▆▇
ov.cheap                          # False if data had to be sampled
ov.source_bytes                   # how much was consulted
```

| modality | where the overview comes from |
|---|---|
| Parquet | row-group statistics in the footer |
| h5ad | `indptr` — nnz per cell is free |
| FASTQ/FASTA | the index sidecar |
| Stereo-seq GEF | the coarsest bin level |
| Stereo-seq GEM | the density grid in the sidecar |
| Images | the tiles the window touches, decimated |

## The sequence index

A sequence file gets one pass; everything after is arithmetic.

```python
idx = g.index_fastx("reads.fastq")     # builds, or reuses reads.fastq.gemidx.json
idx.n_records, idx.total_bases
idx.zoom(0, None, 512)                  # re-aggregate to any resolution, sub-millisecond
idx.cost(150_000, 151_000)              # resident bytes, and the seek checkpoint
```

For a 61.7 MB FASTQ the sidecar is **35 KB (0.054 %)** and re-aggregation to any zoom level
takes **under a millisecond**.

## Loading versus streaming

```python
v.cost()["fits_whole"]     # can .load() return the whole window?
v.cost()["fits"]           # can .chunks() stream it?

data = v.load()            # raises CostError if it will not fit
for chunk in v.chunks():   # bounded; always works if `fits` is True
    ...
```
