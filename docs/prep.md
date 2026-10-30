# Preprocessing out of core

The five steps everyone runs on a count matrix — QC metrics, cell and gene filters,
library-size normalisation with log1p, highly variable genes, PCA — are each a sum over
cells. scanpy does them with the matrix in memory, which is why a 2-million-cell atlas does
not open on a laptop. `geminga.prep` does them in three passes over budget-sized blocks and
hands back what scanpy needs next: a 50-dimensional embedding per cell, with the QC columns
and the HVG flags.

```python
pp = g.prep("atlas.h5ad", budget=g.Budget(ram="4GB"))
pp.qc()                                                     # pass 1
pp.filter_cells(min_genes=200, max_pct_mito=20).filter_genes(min_cells=3)
pp.hvg(n_top=2000)                                          # from pass-1 sums, no new pass
res = pp.pca(n_comps=50, out="X_pca.npy")                   # passes 2 and 3, memory-mapped output
res.report
# {'passes': 3, 'blocks': 27, 'seconds': {...}, 'n_kept': 1_874_112, 'n_hvg': 2000, 'budget_peak': '1.0 GB'}

adata = res.to_anndata()                                    # obs, var, obsm['X_pca'], varm['PCs'] — no X
sc.pp.neighbors(adata); sc.tl.umap(adata); sc.tl.leiden(adata)
```

or all at once: `g.prep(path, budget).run(n_top=2000, n_comps=50)`.

## What each pass does

| pass | reads | accumulates | resident |
|---|---|---|---|
| 1 `qc()` | every block once | per cell: total counts, genes detected, mitochondrial and ribosomal fractions; per gene: cells detected, sums of normalised counts and their squares (for dispersion), sums of log1p values and their squares (for scaling) | one block |
| — `filter_*`, `hvg()` | nothing | the cell mask, the gene mask, Seurat-flavour dispersions binned by mean, the top *n* | — |
| 2 `pca()` | kept cells, HVG columns | the HVG × HVG covariance of the scaled, clipped values (2,000 genes → 32 MB whatever the cell count) | one block + one dense sub-block sized to the budget |
| 3 `pca()` | kept cells, HVG columns | the projection onto the leading eigenvectors, written block by block | one block + one sub-block; the output in RAM (leased) or a memory-mapped `.npy` |

The PCA is exact — an eigendecomposition of the accumulated covariance — not incremental or
randomised, so it does not depend on block order and matches a dense computation to float
precision. The tests compare every step with a dense float64 implementation and, when scanpy
is installed, with scanpy 1.11 itself: totals and mitochondrial fractions equal, 99 of 100
HVGs the same on the test matrix (the one difference is a single-gene bin, where scanpy
assigns a normalised dispersion of 1 and GEMINGA 0).

One honest difference: the per-gene statistics behind HVG selection and scaling are over
*all* cells seen in pass 1, not only the cells kept by `filter_cells`. Doing it the other way
costs a pass, and the ranking it changes is at the margin; `res.params["hvg_stats_over"]`
records which cells the statistics cover.

## Scale

`examples/bench_prep.py` writes a 200,000 × 20,000 matrix with 60 M non-zeros block by
block and runs the pipeline under a 512 MB budget; the numbers are in
[benchmarks](benchmarks.md). Time is linear in non-zeros and the resident set does not grow
with cells: a 2 M-cell atlas is ten times the time, not ten times the memory.

## Inputs

`.h5ad` (CSR `X` or a layer), Cell Ranger `.h5` (one feature type, `RNA` by default), or an
mtx triple directory — the last is converted once to `geminga_matrix.h5` beside itself,
because text cannot be read by cell range. CSC h5ad files are refused by the reader with the
conversion to run.

## What it does not do

Batch correction, integration, neighbours, embeddings, clustering, differential expression:
scanpy, Seurat and their ecosystems do these well, and after `prep` the matrix they need is
cells × 50. Doublet detection and ambient-RNA correction need the matrix per sample and are
not here either.
