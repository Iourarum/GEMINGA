# Stereo-seq

Stereo-seq is the case GEMINGA was designed around. A 1 cm chip carries 400 million DNA
nanoballs at 500 nm pitch; chips run to 13 × 13 cm, over 150× the area. A bin-1 expression
matrix routinely holds tens of millions of observations and the paired ssDNA or H&E image is
gigapixel. None of it fits in a laptop, and most of it does not need to.

## The GEF bin pyramid is the zoom stack

SAW writes `/geneExp/bin1`, `bin5`, `bin10` … `bin200` into one HDF5 container, each a complete
matrix at a coarser spatial binning. That is exactly a level-of-detail stack — so zooming out on
a GEF is not decimation, it is switching to a level the instrument's own pipeline already
computed.

```python
import geminga as g

v = g.open_gef("chip.gef", budget=g.Budget(ram="512MB"))
for z in range(len(v.gef.bins)):
    v.set_zoom(z)
    c = v.cost()
    print(f"zoom {z} -> bin{c['bin_size']:<4} {c['units']:>10,} obs  {c['resident_human']}")
```

```{list-table} Measured on a miniature chip (4000 × 4000 units, 3M observations)
:header-rows: 1

* - zoom
  - bin
  - observations
  - resident
* - 0
  - bin1
  - 2,946,567
  - 28.1 MB
* - 1
  - bin10
  - 2,560,898
  - 24.4 MB
* - 2
  - bin20
  - 2,165,831
  - 20.7 MB
* - 3
  - bin50
  - 1,017,051
  - 9.7 MB
* - 4
  - bin100
  - 377,405
  - 3.6 MB
* - 5
  - bin200
  - 116,221
  - **1.1 MB**
```

A whole-chip overview therefore costs 1.1 MB instead of 28 MB — and on a real 13 cm chip the
ratio is far larger, because the coarse levels grow with area while the fine ones grow with
observations.

## Let the budget pick the level

```python
v.select(12000, 16000, 9000, 13000)   # chip coordinates: x0, x1, y0, y1
v.auto_bin()                           # finest bin whose window still fits
```

When a selection does not fit, the advice names the bin rather than talking about bytes:

```text
the window holds 28.1 MB at bin1 but 2.0 MB is free; use bin10 (zoom_out),
narrow the window, select specific genes, or raise the 'ram' budget
```

## One gene is one slice, not a scan

Each bin level carries a `gene` dataset of `(gene, offset, count)` pointing into `expression`.
Reading a gene is therefore a contiguous HDF5 slice — native predicate pushdown, the same idea
as Parquet row-group pruning.

```python
v.select_genes(["Gapdh", "Actb"])
d = v.load()          # only those genes' rows ever enter memory
```

:::{admonition} Measured
:class: measured
At bin10 on the fixture chip: all 300 genes, 2,560,898 observations in **0.99 s**. One gene,
14,893 observations in **0.007 s** — **139× faster**, and the unwanted rows are never read.
:::

## Expression and morphology share one window

Stereo-seq is multi-modal by construction: the ssDNA or H&E image shares a coordinate frame with
the expression matrix, which is what makes cell segmentation meaningful. `StereoSeqDataset`
holds them together so one window selection drives both.

```python
ds = g.open_stereoseq(gef="chip.gef", image="chip_ssdna.tif",
                      budget=budget, image_scale=0.512)
pair = ds.paired((1800, 3000, 1600, 2800), shape=384)
pair["expression"]        # (384, 384) MID raster at the chosen bin
pair["morphology"]        # the registered image crop, same chip window
```

`image_scale` is image pixels per chip unit; `image_origin` shifts the frame if registration put
the image somewhere other than (0, 0).

```{image} _static/stereoseq_chip.png
:alt: whole chip, window, registered morphology, and one gene
:class: gem-figure
```

That figure is one call:

```python
g.render.chip(ds, window=(1800, 3000, 1600, 2800), genes=["Gene0007"],
              path="chip.png")
```

## GEM files

GEM is the tab-separated form everything converts to and from: `geneID x y MIDCount [ExonCount]`
under `#key=value` headers. It carries no pyramid, so GEMINGA builds a spatial index itself —
one pass producing chip extent, a density grid, per-gene totals, and the occupancy counts the
cost model runs on.

```python
gv = g.open_gem("chip.gem", budget=budget)   # indexes once, reuses the sidecar after
gv.index.top_genes(5)
gv.select_frac(0.5, 0.8, 0.5, 0.8)
g.print_cost(gv)
t = gv.load()                                 # pyarrow Table: gene, x, y, count
```

Parsing is Rust — gene names are dictionary-encoded, so a 30k-symbol vocabulary costs kilobytes
rather than a string per row. The sidecar is a fixed ~1.6 MB at the default `grid=512`,
regardless of file size: negligible for the 100 GB files this is aimed at, and you can drop it
to `grid=128` for small ones.

:::{note}
Prefer GEF when you have it. GEM support exists because archives, converters and collaborators
emit it, not because it is the better container.
:::

## Cell-bin GEF

`.cellbin.gef` stores expression per segmented cell rather than per square bin. Open it by
naming the group:

```python
v = g.open_gef("chip.cellbin.gef", group="cellBin", budget=budget)
```

There is no bin pyramid inside a cellbin file — cells are the smallest unit and the only one —
so zoom falls back to windowing. Pair it with the bin GEF when you want both.
