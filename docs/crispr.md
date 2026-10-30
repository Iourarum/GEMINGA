# GEMINGA and CRISPR screens — a one-page description

## What the tool is

GEMINGA is a memory-budgeted data layer for omics. You declare the bytes you have — RAM, VRAM,
disk — and it derives the chunking, enforces the ceiling, and refuses to over-commit. Every
reader plans from metadata before touching data, every view answers `cost()` before `load()`,
and every byte that becomes resident is accounted and released. It streams Parquet from the
Hugging Face Hub with HTTP range requests, walks Stereo-seq's bin pyramid as zoom levels, reads
10x matrices by feature type, slabs MRI volumes, and hands regions to R, Ruby, Perl or Python
tools through a manifest. Rust core, Python API, 108 tests, built for the machine a PhD student
actually has.

## How a CRISPR screen arrives

Single-cell CRISPR screens — Perturb-seq, CRISPRi, CRISPRa, CROP-seq, ECCITE-seq — ship as a
Cell Ranger matrix with a `CRISPR Guide Capture` feature type stacked under the expression
rows, plus `protospacer_calls_per_cell.csv` naming which guide each cell received. Pooled
screens with a bulk readout (MAGeCK-style) are small count tables and are not a streaming
problem. The single-cell case is: Tahoe-100M-scale chemical screens and genome-wide Perturb-seq
are hundreds of thousands to a hundred million cells.

## What already works

```python
m  = g.TenXMatrix("filtered_feature_bc_matrix.h5")   # RNA + GUIDE (+ ADT for ECCITE-seq)
gv = g.open_tenx(path, modality="GUIDE", budget=b)
gv.overview(64).sparkline("GUIDE_total_per_cell")    # guide UMIs per cell: the first QC plot
m.per_cell_totals("GUIDE")                            # zero-guide and multi-guide cells
```

The guide matrix is a contiguous row slice, so "which cells got which sgRNA" costs the guide
rows and nothing else. The RNA view over the same cells is one `select()` away. The profiler
flags `crispr_screen: True` when it sees the feature type.

## What adapting it properly means

**A perturbation index.** One pass over `protospacer_calls_per_cell.csv` (or the guide rows)
builds `guide -> cell indices` and `target gene -> cell indices`. Then "all cells carrying an
*EGFR* guide" is an index lookup that drives a `TenXView` window — the same pattern as the
Stereo-seq gene index and the FASTQ sidecar. Cost is answered before the cells are read.

**Perturbation-aware navigation.** `zoom_out()` on a screen should aggregate by guide, then
by target, then by pathway — the pyramid is *biological*, not spatial. A `PerturbView` whose
levels are cell / guide / gene / pathway gives the overview a screen actually needs: how many
cells per target, how balanced, which targets are under-covered.

**CRISPRi versus knockout.** The formats are identical and the biology is not: CRISPRi is
graded repression, and the interesting readout is often the *target's own* residual expression.
The profiler should read the guide names (dCas9-KRAB constructs, `NTC`/non-targeting labels)
and the CSV's `feature_call` field and say which it thinks the screen is — with the same
"here is what would settle it" it applies to sc versus sn.

**Chemical-genetic and combination screens.** Tahoe-100M, sci-Plex3, ComboSciPlex carry
perturbation identity as metadata rather than guide rows. Same index, different source column;
the handoff manifest already records `genes=` per region, so a `perturbations=` field beside
it is the natural extension.

**What it will not do.** Differential expression, energy-distance tests, GRN inference — that
is `pertpy`, `scPerturb` tooling, decoupleR. GEMINGA's contribution is that those tools get the
right 20,000 cells on a laptop instead of the wrong 2 million on a cluster.

## Effort

The perturbation index and `PerturbView` are a week on top of what exists; the CRISPRi/knockout
profiling rules are an afternoon. The pieces they rest on — feature-type slicing, per-cell
totals, budget accounting, the manifest — are already built and tested.
