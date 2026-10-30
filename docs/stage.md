# The stage, and Polars

Two additions that keep GEMINGA inside the same Rust-and-wgpu stack as the rest of the
constellation: Polars as the dataframe layer, and a small wgpu canvas for looking at data.

## Polars

Arrow is already the currency — every tabular reader hands back `pyarrow` — and Polars is
Arrow with a Rust engine behind it, so the conversion is a pointer move.

```python
df = view.polars()                     # the current window, budget-enforced like load()
df = g.to_polars(matrix.block(0, 500, "ADT"))   # 10x block, or a Stereo-seq window, or sparse
lf = g.scan("transcripts.parquet")     # a LazyFrame; nothing read until .collect()
```

### Which pruner to use

GEMINGA and Polars both push predicates into Parquet, and the division is worth stating:

| | answers | works over |
|---|---|---|
| `Dataset.prune()` | *what would this cost*, from footers, before fetching | local **and** remote, via HTTP range requests |
| `pl.scan_parquet` | *execute this query well*, once decided | anything Polars can reach |

`lazy_window()` composes them in the right order — prune to learn the selectivity, then let
Polars run the exact predicate:

```python
lf, plan = xenium.lazy_window(1000, 1400, 500, 900,
                              columns=["feature_name", "x_location", "y_location"],
                              explain=True)
plan["row_groups_kept"], plan["selectivity"]   # 2 of 20, 10.0 %
lf.collect()                                    # 2,341 rows, 36.6 KB
```

The two paths agree exactly: `lf.collect().height == xenium.window(...).num_rows`, asserted in
the test suite.

`g.sink_parquet(lf, path)` streams a result larger than memory to disk, and
`Catalog.polars()` reads the append-only log straight into a frame.

:::{admonition} One boundary detail
:class: measured
Polars stores strings, binary and lists in their 64-bit Arrow forms — `large_string`, not
`string`. The data is identical, but a round trip would otherwise widen the schema of anything
GEMINGA had already written, and a `Harvest` region whose type changed between runs is a real
problem for the R and Ruby readers at the other end of the manifest. `from_polars()` narrows
them back by default; pass `compat=False` to keep Polars' native widths.
:::

## The stage

PRISM's canvas is a general node-graph renderer. GEMINGA needs a fraction of that, so the
stage is deliberately the fraction: **two pipelines, one draw call each**, and nothing that
knows what a pipeline step is.

```python
st = g.Stage(900, 700, point_budget="16MB")
st.show(view)                     # picks an LOD that fits, loads, uploads
st.axes()
st.label("GBM chip", 14, 14, scale=1.3)
st.render_png("chip.png")
```

**Points, instanced.** Every observation is one instance of a unit quad carrying position,
size and a scalar; a colormap texture turns the scalar into colour, and the fragment shader
discards outside the inscribed circle to get round points. One draw call for the layer.

**Text, from a glyph atlas.** A font is rasterised once into a texture and each character is an
instance sampling its own rectangle. ASCII, one size, no kerning or shaping — enough for axis
ticks, gene names and a cost readout, which is all GEMINGA needs. Labels live in **screen**
space so they do not grow when you zoom, which is the point of a label. `world=True` anchors
one to a data coordinate instead.

**The camera** is a centre and a span. No rotation, no projection matrix; a chip is flat.

### The renderer never decides how much to read

`show()` asks the view what a window would cost, coarsens — a coarser GEF bin, a larger stride
— until the estimate is under the stage's point budget, and only then calls `load()`. Measured
on a 2.9 M-observation chip:

| point budget | chose | points | steps |
|---|---|---|---|
| 2 MB | bin200 | 116,221 | 5 |
| 16 MB | bin50 | 1,017,051 | 3 |
| 64 MB | bin1 | 2,946,567 | 0 |

Draw calls stay at 2 throughout, at every zoom level.

```{image} _static/stage_chip.png
:alt: a Stereo-seq chip window rendered by the wgpu stage with axis ticks and labels
:class: gem-figure
```

## Knowing where you are, and what it costs

Three panels, all built from rectangles and the glyph atlas, all sharing the same two draw
calls as everything else:

```python
st.show(view, window=win)
st.hud(view, budget=b)          # minimap + detail ladder + budget gauges + axes
```

**Minimap.** A thumbnail of the whole dataset with the current window outlined in red, built
from the *coarsest* level available — the cheapest complete picture there is — so it costs the
same however deep you have zoomed. The caption gives the window as a fraction of the extent.

**Detail ladder.** The pyramid as rungs, with the level actually on screen filled in. On a GEF
these are the bin levels SAW wrote; elsewhere they are zoom levels.

**Budget gauges.** Per tier: capacity as the track, the high-water mark as a lighter fill with
a tick, and what is held right now as the solid bar. The peak is the number that matters —
"it fits" is a claim about the peak, not about this instant — so it gets the tick.

```{image} _static/hud_zoom.png
:alt: three zoom levels with minimap, detail ladder and budget gauges
:class: gem-figure
```

Left to right: the whole chip at bin100 (4.3 MB), a 30 % window at bin1 (8.4 MB), and a 6 %
window at bin1 (332 KB). The minimap box shrinks, the ladder highlight moves, and the budget
bar follows — nothing in the layout is placed by hand per panel.

### Several datasets at once

`hud(view, workspace=ws)` swaps the per-tier gauges for one gauge per cell: the partition it
reserved, and the peak it actually reached inside it.

```{image} _static/hud_workspace.png
:alt: a workspace HUD showing three cells, their partitions and their peaks
:class: gem-figure
```

208 MB reserved of 256 MB across three cells; `stereo` peaked at 33.7 MB inside its 96 MB
slice, and no cell can reach into another's.

### Panels and ticks do not collide

Each panel registers the screen box it occupies, and `axes()` skips any tick label that would
land underneath one. This matters because the rectangle pass runs *before* the text pass — a
panel drawn later can never paint over a label already queued, so avoiding the overlap has to
happen at layout time, not by draw order.

### Headless by default

It renders to a texture and returns a numpy array, so it runs in CI on a software adapter —
the figure above was produced on llvmpipe with no GPU present. `g.stage_available()` and
`g.adapter_info()` report what is there.

`wgpu` is an optional dependency, imported lazily: `import geminga` works on a machine with no
graphics stack at all, and only touching `g.Stage` pulls it in.

### What it deliberately is not

No widgets, no node graph, no scene tree, no layout engine, no animation. For those, export
OME-Zarr and use napari or neuroglancer, which have solved them. The stage exists for the one
thing those cannot do as directly: render a GEMINGA view at an LOD the budget chose.
