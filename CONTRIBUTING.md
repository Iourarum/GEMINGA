# Contributing to GEMINGA

GEMINGA exists because a lot of good science is done on one laptop. If you have hit a memory
wall on real data, your bug report is the most valuable thing you can send us — attach the
`plan()` output and `budget.stats()`, and we will usually know where to look.

## Getting set up

```bash
git clone https://github.com/<you>/geminga && cd geminga
pip install maturin pytest
maturin develop --release          # builds the Rust core into your current venv
pip install -e ".[all]"            # h5py, tifffile, zarr, scipy for the Python readers
python examples/make_omics_fixtures.py fixtures
pytest -q
```

Rust 1.85 or newer. On a machine with fewer than 4 cores, `maturin develop` (no `--release`)
builds much faster and is fine for everything except benchmarking.

## What we are looking for

- **Format readers.** The bar is: produce a `Plan` before reading, acquire from the budget
  before materializing, release when the consumer asks for the next chunk. See
  `python/geminga/formats.py` — `H5adReader` is the shortest complete example.
- **Benchmarks on real datasets.** Especially anything that makes peak resident memory exceed
  the declared budget. That is a bug, not a tuning issue.
- **Small hardware reports.** "Ran X on a 2016 ThinkPad with 8 GB and it worked / did not" is
  genuinely useful and we will act on it.

## What we are not looking for (yet)

- Analysis algorithms. GEMINGA moves bytes; Scanpy, Seurat, squidpy and friends do the science.
- Cluster/MPI support. The target is one machine.
- New on-disk formats. We read what instruments and archives already emit.

## House rules for code

- Every new reader gets a test that asserts peak budget use stays under the declared cap.
- Rust: `cargo fmt`, `cargo clippy -- -D warnings`. Python: `ruff check`.
- Public functions get a docstring saying what crosses the wire and what becomes resident.
- No claim in the README without a number behind it that CI can reproduce.

## Reporting a memory bug

Include:
1. `reader.plan()` output,
2. `budget.stats()` after the failure,
3. peak RSS (`/usr/bin/time -v` on Linux),
4. the file's shape (rows/cells/pixels, dtype, compression).
