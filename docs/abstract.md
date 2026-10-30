# Conference abstract

> Draft for a one-page submission. Figures in brackets refer to the two panels produced by
> `examples/glioma_conference_figures.py`. Every number below is a real measurement of
> GEMINGA; the dataset it ran on is synthetic and glioma-shaped, because the build machine had
> no route to the data repositories. **Before submitting, re-run the script on a real dataset
> and replace the bracketed figures** — the script takes `--gef`, `--h5` and `--h5ad` and
> regenerates both panels unchanged.

## GEMINGA: memory-budgeted navigation of multi-modal glioma datasets on commodity hardware

**Sebastian [Surname]¹**
¹ [Affiliation], Bucharest, Romania

### Background

A single Stereo-seq chip records hundreds of millions of spatially barcoded observations beside
a gigapixel morphology image; a contemporary glioma study pairs that with single-nucleus
expression, surface protein, and multiplexed imaging. Analysis tools for these data assume the
dataset fits in memory, or stream in units of *rows* without bounding the *bytes* that become
resident. On the 8–16 GB machines most graduate students actually have, the result is a job
that fails unpredictably partway through, or an analysis that is never attempted. The scarce
resource is not compute; it is a predictable ceiling.

### Methods

GEMINGA (Gated, Extensible, Memory-budgeted Ingestion for N-modal Genomics and Arrays) inverts
the usual relationship between data and hardware. The analyst declares a budget across host
memory, device memory and disk. Every reader then derives its chunking from that budget,
estimates the cost of a selection from metadata *before* reading any of it, and holds an
accounted reservation for everything it returns, released when the consumer advances. Requests
that can never fit are refused with the corrective action named; requests that do not fit yet
apply backpressure rather than failing.

Datasets are navigated as maps: zooming changes *resolution*, not extent, so the cost of a view
stays roughly constant as it moves. For Stereo-seq, the bin levels SAW already writes into a GEF
serve directly as the zoom stack, and the per-level gene index makes a single-gene query a
contiguous read rather than a scan. The same contract covers 10x matrices split by
`feature_type`, Space Ranger and Xenium bundles, multiplexed OME-TIFF, NIfTI and DICOM volumes,
and remote Parquet read over HTTP range requests. The core is Rust with a Python API;
108 tests assert the budget invariant on every supported format.

### Results

Across declared budgets from 8 MB to 256 MB on the same chip, measured peak residency never
exceeded the declaration, with zero denials and a maximum peak-to-budget ratio of 0.77; the
library selected a coarser bin level automatically as the ceiling tightened [Fig 1a]. Walking
the GEF bin pyramid from bin1 to bin200 reduced resident cost from 33.7 MB to 1.3 MB — a 26-fold
drop — while keeping the whole chip in view [Fig 1b]. Reading a single gene through the GEF gene
index was a median 72× faster than loading all genes over the same window [Fig 1c]. Three
modalities — a Stereo-seq chip, a CITE-seq-plus-guide 10x matrix, and a single-cell atlas — were
held open simultaneously under one 256 MB ceiling, each in an isolated partition that could not
starve its siblings [Fig 1d]. A whole-chip overview at the coarsest bin required 1.1 MB of reads
and rendered in 0.01 s; zooming to a 1,150-unit window at bin20 and then to a single gene within
it cost 2.2 MB and 1,755 observations respectively [Fig 2a–c]. Feature-type slicing of the
paired 10x matrix separated 2,000 expression, 40 antibody and 24 guide features without reading
the others [Fig 2d]. Every extracted region carries a checksummed manifest recording its source,
window, bin level and gene selection, and the budget under which it was produced.

### Conclusions

Treating the memory budget as an input rather than a discovery makes multi-modal glioma analysis
tractable on hardware that is already on the desk, and makes the resource claims in a methods
section checkable. The approach generalises beyond spatial transcriptomics to imaging and
neuroimaging cohorts under one contract. GEMINGA is Apache-2.0 at [repository URL].

**Keywords:** spatial transcriptomics, Stereo-seq, multi-modal data, out-of-core computing,
glioma, reproducibility

---

## Figures

```{image} _static/fig1_budget.png
:alt: budget invariance, bin pyramid, gene pushdown, workspace partitioning
:class: gem-figure
```

**Figure 1.** (a) Declared budget against measured peak residency; the dashed line is the
declaration and the marker label is the bin level GEMINGA chose automatically. (b) Resident cost
and observation count at every bin level in the GEF. (c) Speedup of a single-gene query over a
full-matrix load of the same window. (d) Three datasets open at once, each partition shown
against its peak use.

```{image} _static/fig2_navigation.png
:alt: whole chip, window, single gene, and modality composition
:class: gem-figure
```

**Figure 2.** (a) Whole chip at the coarsest bin, with the selected window in red; the raster
grid matches the bin grid. (b) The same window at bin20. (c) One gene within that window, read
as a contiguous slice. (d) Feature types in the paired 10x matrix, with median counts per cell.

## Running it on real data

```bash
python examples/glioma_conference_figures.py \
    --gef  /data/GBM_chip.gef \
    --h5   /data/GBM_filtered_feature_bc_matrix.h5 \
    --h5ad /data/GBmap.h5ad \
    --out  figures/
```

Candidate datasets, none of which were reachable from the build machine — verify each accession
before relying on it:

| Modality | Candidate |
|---|---|
| Spatial, GBM | STOmicsDB `STDS0000040` (catalogued as 10x Visium, not Stereo-seq) |
| Spatial, GBM | `GSE237183` — Greenwald et al., 10x Visium |
| Spatial, GBM | Zenodo `10.5281/zenodo.17622242` — Xenium atlas |
| Spatial + sc + ATAC | Zenodo `10.5281/zenodo.8085502`; raw at GSA `HRA004942` |
| scRNA-seq, glioma | GBmap (~1.1M cells); `GSE131928` (Neftel et al.) |

:::{warning}
No public **human glioma Stereo-seq** chip surfaced in searching. The script accepts any GEF, so
a mouse glioma model chip works as a substitute — but the strongest version of this abstract
needs a real human one, and MGI/STOmics is the most direct route to it.
:::
