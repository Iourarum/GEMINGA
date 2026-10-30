# Visuals

Four figures, each answering a different question. Matplotlib is imported lazily and only in
`geminga.render`, so nothing in the core needs it.

## `navigator` — where am I, and what would this cost?

```python
g.render.navigator(v, "nav.png", title="GEMINGA — FASTQ navigator")
```

An overview strip drawn from metadata with the current window highlighted, a cost readout, and
the detail beneath it. The overview renders for a 60 GB file as fast as for a 60 MB one. If the
selection does not fit the budget, the detail panel says so instead of blowing up.

```{image} _static/nav_fastq.png
:alt: FASTQ navigator
:class: gem-figure
```

## `chip` — a Stereo-seq sheet

```python
g.render.chip(ds, window=(1800, 3000, 1600, 2800), genes=["Gene0007"], path="chip.png")
```

Whole-chip overview from the coarsest bin, the selected window's expression, the registered
morphology crop, and the named gene's own distribution.

## `hits` — search results on the overview axis

```python
g.render.hits(v, hits, "hits.png", patterns=[adapter])
```

Hit density aligned to the same axis as the overview, so a motif's distribution can be read
against coverage or quality.

## `progressive` — the picture as it stands, mid-stream

`ProgressiveCanvas` takes its bounds from column statistics (free), then histograms each batch
into a fixed grid, so it is O(grid), never O(rows).

```python
canvas = g.ProgressiveCanvas.from_dataset(ds, "x", "y", shape=(400, 400))
for batch in ds.stream(columns=["x", "y", "count"], chunk_bytes="8MB"):
    canvas.add_batch(batch, "x", "y", weight="count")
    if canvas.n_batches % 10 == 0:
        g.render.progressive(canvas, f"frame_{canvas.n_batches:03d}.png")
```

2,000,000 rows through a 400 × 400 canvas stays at **2.4 MB resident** from the first batch to
the last.

```{image} _static/progressive.png
:alt: progressive density canvas
:width: 520px
:class: gem-figure
```

## Palette

Figures use the project palette so they match the logo and docs:

| role | hex |
|---|---|
| ink | `#1A1A1A` |
| red | `#E8412F` |
| yellow | `#F5C518` |
| blue | `#2E77C4` |
| navy | `#0B2545` |
| paper | `#FFFCF3` |

Available as `g.render.PALETTE`.
