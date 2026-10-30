# What this is, and is not

An assessment written to be read before adopting the library, not after. It is deliberately
unflattering where the facts are, because a tool for people with small machines fails them
worst when it overpromises.

## The claim, and what backs it

The claim is narrow: *declare the bytes you have, and every reader, window and store stays
under them, telling you the cost before it loads.* That claim is backed. The budget is a
Rust governor with leases; every reader sizes its chunks from what is free; the test suite's
central invariant — peak resident never exceeds the budget — is checked on FASTQ, Parquet,
h5ad, Cell Ranger h5, mtx, GEF/GEM, OME-TIFF, NIfTI, DICOM and the compressed store, with
denials counted. When a window will not fit, you are told in bytes, before anything is read.

What the claim does not say is that your analysis will run. GEMINGA moves and governs bytes.
It does not cluster, embed, integrate, segment or test. Until 0.17 it did not even do the
five preprocessing steps everyone runs; `geminga.prep` now does those, out of core, and
hands scanpy a 50-dimensional embedding. Everything after that — neighbours, UMAP, Leiden,
differential expression, trajectories — is scanpy, Seurat, squidpy or your own code on a
matrix that now fits.

## Where it is useful today

- **Deciding whether to download.** `acquire.plan()` prices every route for an accession;
  `peek()` reads the first reads of a 40 GB FASTQ for a few hundred kilobytes; the
  orchestrator turns a spreadsheet of 56 dataset descriptions into a priced, resolved
  shopping list without transferring a byte. For a student choosing among two hundred
  candidate datasets this is the difference between a week and an afternoon.
- **Looking at data that does not fit.** A 2.2 TB Stereo-seq project on an FTP server can be
  listed, a chip opened in place, a window at bin50 rendered, a region cut and handed to a
  tool — on a laptop, over the network, without a mirror. The GEF pyramid as a zoom stack,
  the Z-order index for window pruning and the wgpu stage are the parts of this library that
  have no equivalent in anndata, SpatialData or Stereopy.
- **Keeping a project honest.** The catalog, the library of flagged things, the harvest
  manifests with checksums and the build provenance mean that six months later you can say
  which file, which window, which gene map and which budget produced a figure. Since 0.18 the
  journal writes down every window actually streamed — file, level, exact coordinates, rows,
  bytes, who asked — so the question "did this region ever get read at bin 1?" has an answer
  in a text file, and `replay` reopens the piece.
- **The standard pipeline on an atlas-sized matrix.** `prep` runs QC, filters, normalisation,
  HVG and exact PCA over a matrix of any number of cells in three passes with a block and
  one dense HVG sub-block resident; on 200 k cells × 20 k genes it stays under a 512 MB
  budget. The output is what scanpy needs next.
- **Sharing a GPU.** The reservation server and the compressed store make a 4 GB slice of a
  shared card hold what 20 GB used to, with leases that expire and idle slices reclaimed.

## Where it is not, yet

- **It is not a drop-in for scanpy's in-memory model.** `sc.pp.*` functions expect an
  `AnnData` with `X` in memory. `prep` reproduces their results (checked against scanpy
  1.11 in the tests), but you call `g.prep`, not `sc.pp`. A backed-mode adapter that lets
  existing scanpy scripts run unchanged on a governed reader is the obvious next step and
  is not built.
- **Every spatial fixture is synthetic.** The GEF reader follows the SAW format description
  and the Stereo-seq tests pass on files GEMINGA itself wrote. No real SAW, Space Ranger or
  Xenium output has been through it in this repository. The first real chip will find
  something; expect that.
- **The interactive explorer has never opened a real window.** It is tested headless
  (`record()`); `run()` needs a display none of the development environments had.
- **Online paths are tested on recorded responses.** ENA, GEO, Zenodo, figshare and CNGB were
  unreachable from where this was built; the request shapes are right, the parsing is tested
  on captured replies, and the first live run will tell.
- **Wheels are Linux x86_64 only, and the Python version follows the build machine.** Most
  PhD laptops are Apple Silicon or Windows; until the CI matrix produces those wheels,
  installing means building the Rust core locally.
- **The GPU paths are proved on a software adapter.** The bitpack kernel is bit-exact on
  llvmpipe. That proves correctness, not speed; on real hardware the decode is memory-bound,
  and nobody has measured it there yet.
- **The reservation server is cooperative.** It cannot stop a process from calling
  `cudaMalloc` past its grant. It makes GEMINGA's own readers and stores obey the number
  they were given, and it makes the sharing visible; it is not a sandbox.
- **No VCF/BAM, no imzML, no CosMx/MERFISH tables, no h5ad-backed explorer.** Listed so
  nobody discovers them the hard way.
- **The trigram index is not faster than ripgrep on everything.** It wins by orders of magnitude
  on rare terms and loses on terms that occur in every record, where it has to scan every block
  and ripgrep's scanner is better than ours. `docs/journal.md` says when each is the right tool;
  the journal is plain JSONL precisely so that `rg` always works on it.

## What would make it more useful for a PhD on a laptop

In the order the authors would do them:

1. **A backed `AnnData` over a governed reader**, so `sc.pp.normalize_total(adata)` runs on
   a 2 M-cell file by streaming blocks under the budget — the single change that would let
   existing notebooks run unchanged.
2. **Real-data validation**: one public Stereo-seq chip, one Visium, one Xenium, one 10x
   multiome, run end to end in CI from their public URLs, with the numbers in the docs.
3. **Wheels for macOS arm64 and Windows**, and a conda-forge recipe.
4. **Windows kept packed**: zoom out, keep the coarse level resident in the compressed
   store, zoom back without re-reading — the reason the codec contract exists.
5. **A row-aware integer codec** (delta within CSR rows, using the lengths the store already
   has) — the first thing a plug-in should beat, and the GPU compressor's natural target.
6. **One honest benchmark page against the alternatives** — anndata backed mode, TileDB-SOMA,
   SpatialData/Dask — on the same files and the same laptop, including where they win.

## How to read the test count

There are over 800 tests. About 400 of them are regression tests written during a
review-and-fix pass, each one written to fail on the code before the fix. That number says
the obvious defects of a young codebase were hunted and caught; it does not say the
library has been used in anger. The gap between those two is where a first real project
will land, and the issue tracker is the right place to put what it finds.
