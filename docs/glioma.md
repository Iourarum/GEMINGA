# Glioma multi-omics

The 2025–26 glioma literature ships almost nothing as a single matrix. A typical entry pairs
snRNA-seq with Visium or Xenium, adds CITE-seq surface protein or a CODEX panel, and lands the
whole thing as a Cell Ranger / Space Ranger / Xenium Ranger *directory*. This page maps that
inventory onto what GEMINGA can open, what it cannot yet, and where the honest boundary is.

## Start with the overview

```python
g.modalities("GSE2xxxxx/sample_01").print()
```

```text
/data/sample_01/filtered_feature_bc_matrix.h5  (10x-h5, 958.6 KB)
  cells       : 3,000
  RNA         :    2,000 features  — transcripts per cell
  ADT         :       40 features  — surface protein per cell — CITE-seq / TotalSeq
  GUIDE       :       24 features  — sgRNA UMIs per cell — Perturb-seq / CRISPRi / CROP-seq
```

Headers only. The `feature_type` column in a 10x matrix is the single most useful piece of
metadata in the file: it says which modalities are stacked in it. A multi-modal dataset is
not a special format — it is one matrix with several feature types, and GEMINGA slices them
as contiguous row ranges, the same trick the Stereo-seq gene index gives.

## The inventory, by how it ships

```{list-table}
:header-rows: 1

* - Protocol in the inventory
  - Ships as
  - GEMINGA
  - Note
* - scRNA-seq, snRNA-seq
  - 10x `.h5`, `.mtx.gz` triple, `.h5ad`
  - **`TenXMatrix`, `MtxTriple`, `H5adReader`**
  - `.mtx.gz` streams but cannot be navigated; `to_h5()` once
* - CITE-seq, scCITE-seq (300+ antibodies)
  - same matrix, `Antibody Capture` rows
  - **`modality="ADT"`**
  - a contiguous row slice; the 40–300 ADT rows are a few % of the file
* - scATAC-seq
  - 10x ATAC `.h5` (`Peaks`) + `fragments.tsv.gz`
  - **matrix yes**; fragments not yet
  - fragments are bgzf + tabix: region queries via pysam, on the roadmap
* - Perturb-seq / CRISPRi
  - `CRISPR Guide Capture` rows + `protospacer_calls_per_cell.csv`
  - **`modality="GUIDE"`**
  - see [the CRISPR page](crispr.md)
* - Visium, Visium HD
  - Space Ranger directory
  - **`VisiumBundle`**
  - matrix + `tissue_positions` + scalefactors + image in one frame; HD bin levels are a pyramid
* - Xenium
  - Xenium Ranger directory
  - **`XeniumBundle`**
  - `transcripts.parquet` is pruned by `x_location` row-group statistics
* - CosMx, MERFISH
  - flat CSVs, often >10 GB
  - recognised, sized, advised
  - convert `*_tx_file.csv` / `detected_transcripts.csv` to Parquet once
* - CODEX, IMC, mIF
  - multi-channel OME-TIFF
  - **`MultiChannelTiff`**
  - one channel at a time; 40-plex costs one channel of memory
* - Stereo-seq
  - GEF / GEM
  - **native**
  - [the Stereo-seq page](stereoseq.md)
* - MALDI-MSI, Raman
  - imzML + `.ibd`; HDF5/CSV spectra
  - recognised; not read
  - `pyimzml` reads spectra per pixel; a GEMINGA reader is on the roadmap
* - PacBio HiFi WGS, WES
  - BAM / CRAM / VCF
  - not read
  - `noodles` on the roadmap; `samtools view -L` is the right tool today
* - Patch-seq, 16S, LC-MS/MS
  - bespoke tables, FASTQ, mzML
  - FASTQ yes; tables via Parquet
  - profile them; the analysis lives elsewhere
```

Twenty-four of the thirty inventory rows list at least one protocol GEMINGA opens natively.
The six that do not are metabolomics, Raman, long-read DNA and mass-spec proteomics — none of
which are "large matrix" problems, and none of which this library should pretend to own.

## What the overview is for

A Xenium run and a Visium run of the same tumour share a tissue but not a coordinate frame.
Before any integration question, the useful facts are: how many cells, which panel, what extent,
what is registered to what. `modalities()` on each directory, then `describe()` on the bundle:

```python
xb = g.open_xenium("xenium/"); xb.describe()["extent_um"], xb.describe()["panel"]
vb = g.open_visium("visium/"); vb.describe()["extent_px"], vb.describe()["in_tissue"]
```

Then a window on the Xenium transcripts costs what Parquet pruning costs — on the fixture,
2 of 20 row groups for a 400-unit x-band — and the Visium spots inside a pixel window come
straight from `tissue_positions` without touching the matrix.

## Where the boundary is

GEMINGA opens, slices, sizes and hands off. It does not normalise, integrate, cluster or call
cell types — Scanpy, Seurat, squidpy and Giotto do, and the point of the handoff manifest is
to get regions to them without a full download in between. On a 16 GB laptop, "which 2,000
cells and which 300 genes from this 400,000-cell atlas" is the question GEMINGA answers; what
those cells *are* is the next tool's job.
