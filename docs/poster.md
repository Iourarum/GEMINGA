# Poster

## Abstract

**GEMINGA: navigating multi-modal spatial omics under a declared memory budget, shown on a
glioblastoma section**

Sebastian [Surname]¹ · ¹ [Affiliation], Bucharest, Romania

**Background.** A Stereo-seq chip holds millions of spatially barcoded observations beside a
gigapixel morphology image, and a glioma study adds single-nucleus, surface-protein and
imaging modalities. Tools for these data assume they fit in memory, or stream in rows without
bounding the bytes that become resident — so on the 8–16 GB machines most students have, jobs
fail unpredictably or are never attempted.

**Approach.** GEMINGA makes the memory budget an input. Every reader plans from metadata before
reading, reserves before materialising, and releases on advance. Datasets are navigated like
maps: zooming changes resolution, not extent, and for Stereo-seq the bin levels the instrument's
pipeline already wrote become the zoom stack. A Z-order–sorted sidecar lets spatial windows be
answered by Parquet row-group pruning, and an interactive explorer loads a margin around the
view with hysteresis, drawing a minimap, detail ladder and budget gauges on a wgpu canvas.

**Results.** On a glioblastoma-structured section (2.98 M observations, 303 genes), peak memory
stayed below the declared budget from 4 MB to 128 MB, with the library choosing bin200 at 4 MB
and bin1 by 64 MB. Before the spatial index, reading any window took ~200 ms regardless of size;
with it, read time scaled with the window — 22× faster at 1.6 % of the section and 49× faster
at 0.1 % — while reading 33× fewer compressed bytes and returning identical observations. Eight
niche markers — hypoxic rim (VEGFA, CA9), cellular tumour (SOX2, OLIG2), microvasculature
(PECAM1), myeloid (CD163) and infiltrated brain (SNAP25, MBP) — were each read as one contiguous
gene slice in 1–4 ms, a median 47× faster than a full load, with the speedup tracking how rare
the gene is. A recorded navigation session peaked at 6.6 MB within a 96 MB budget.

**Conclusion.** Declaring the budget rather than discovering it makes multi-modal glioma analysis
tractable on commodity hardware and makes resource claims checkable. Apache-2.0; 198 tests.

*The section is a designed phantom following glioblastoma anatomy; every software measurement
is real.*

## Figures

```{image} _static/poster_A_concept.png
:alt: the GEMINGA pipeline, from declared budget to handoff
:class: gem-figure
```

```{image} _static/poster_B_niches.png
:alt: glioblastoma niches, each read as one gene slice
:class: gem-figure
```

```{image} _static/poster_C_navigation.png
:alt: six moments of a recorded navigation session on the glioblastoma section
:class: gem-figure
```

```{image} _static/poster_D_measured.png
:alt: budget invariance, spatial index time and bytes, gene pushdown
:class: gem-figure
```

Regenerate every panel from a real chip with
`python examples/poster_figures.py --gef your_chip.gef --out poster/`.
