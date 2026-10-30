# Interactive navigation

The stage renders one frame. The explorer turns a stream of input — drags, scroll steps, key
presses — into a stream of frames, and its one real job is deciding **when a frame needs I/O**.

```python
ex = g.explore("chip.gef", budget=g.Budget(ram="2GB"), point_budget="16MB")
ex.run()            # a window: drag to pan, scroll to zoom about the cursor,
                    # 0 fit · arrows step · +/- zoom · g cycle genes · b bookmark · r recall · q quit
```

## Most frames need no I/O

The naive loop reloads on every camera move: a drag becomes a disk read per mouse event, and it
is wrong for a simpler reason — the data was in memory a moment ago. The explorer loads a
**margin** around what is on screen and serves every frame it can from that, reloading only
when the view leaves the loaded area or the zoom moves far enough that a different rung of the
pyramid would be chosen.

Rung changes have **hysteresis**: stepping finer waits until the finer rung fits with 15 % to
spare, stepping coarser waits until the current rung no longer fits. Without the band, a zoom
hovering near a boundary thrashes between two rungs.

```{image} _static/explorer_session.png
:alt: six moments from a recorded navigation session
:class: gem-figure
```

A scripted 21-frame session on the fixture chip: 6 reloads, 71 % of frames served from memory,
budget peak 11.6 MB of 96 MB. Frame 5 shows a detail worth noticing — filtering to one gene
makes the *minimap* show that gene too, so its hotspot lights up with the window sitting on it.

## Margin policy: slack against resolution

A margin buys smooth panning, but it multiplies the loaded area — 0.5 on every side is four
times the visible area — and a larger area forces a coarser rung. A fixed margin silently trades
resolution for slack. So the planner tries shrinking it and keeps whichever margin permits the
finest rung, preferring the widest among ties. How far it may shrink is the policy:

| policy | margins tried | reloads | from memory | frames at bin20 or finer | I/O time |
|---|---|---|---|---|---|
| `smooth` | 0.5 | 3 | 83 % | 3 / 17 | 0.43 s |
| **`balanced`** (default) | 0.5, 0.25, 0.125 | 3 | 83 % | **6 / 17** | 0.42 s |
| `sharp` | … down to 0 | 9 | 50 % | 13 / 17 | 2.10 s |

`balanced` strictly dominates `smooth` — same reloads, same I/O, twice the detail — which is why
it is the default. `sharp` buys the most detail at five times the I/O; pick it when you are
reading structure, not roaming.

## The budget holds across a session, not just per frame

Every reload goes through `view.load()`, so the previous region's lease is released before the
next is taken. The test suite drives fourteen zooms and drags under a 32 MB budget and asserts
`used <= cap` after every frame.

## Headless is the primary mode

All of the logic runs without a window, which is how it is tested and how a session goes into a
paper:

```python
frames = ex.record([g.Event("scroll", x=400, y=300, steps=1),
                    g.Event("drag", dx=-20, dy=8), ...])
ex.telemetry.summary()   # '21 frames, 6 reloads (71% served from memory), 38.7 MB read in 0.67s'
```

`run()` opens a glfw window when a display exists and feeds real mouse events into exactly the
same `handle()` method, so the window adds nothing the tests do not already cover.

Scope, stated: the explorer navigates **point-cloud views** — Stereo-seq GEF and GEM. Images
already have better interactive viewers; see below.

## Handing images to the viewers that already exist

```python
g.to_ome_zarr(view, "chip.ome.zarr")              # GEF: SAW's bin levels become the pyramid
g.to_ome_zarr(tiff, "slide.ome.zarr", band="64MB") # images and volumes: 2x mean pooling
```

Then `napari chip.ome.zarr`, or open it in neuroglancer or Vitessce.

For a GEF, nothing is downsampled by GEMINGA: each bin level is rasterised at one pixel per bin
and written as its own NGFF dataset with the bin size as the scale — 0.5 µm at bin1 to 100 µm at
bin200 — so the instrument's aggregation becomes the viewer's zoom stack. Every level sums to
the same total MID, asserted in the tests. Levels too large for `max_side` or the budget are
skipped and reported, never silently downsampled.

For images and volumes, level 0 is copied band by band and each coarser level is built from the
one just written. **`band` is a ceiling**: the pooling works in float32 and holds its input and
output at once, and the step is sized for that — an earlier version let a requested 8 MB band
peak at 20 MB, which the tests now forbid.

Written as NGFF v0.4 on zarr format 2, which napari-ome-zarr and neuroglancer read today, and
checked against `ome-zarr-py` as an independent reader.
