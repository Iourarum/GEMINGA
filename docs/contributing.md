# Contributing

GEMINGA exists because a lot of good science is done on one laptop. If you have hit a memory
wall on real data, your bug report is the most valuable thing you can send — attach `plan()` or
`cost()` output and `budget.stats()`, and we will usually know where to look.

## Setting up

```bash
git clone https://github.com/<user>/geminga && cd geminga
pip install maturin pytest
maturin develop --release
pip install -e ".[all]"
python examples/make_omics_fixtures.py fixtures
python examples/make_stereoseq_fixtures.py fixtures
pytest -q
```

## What we are looking for

- **Format readers.** The bar is: plan before reading, acquire before materializing, release on
  advance. See `python/geminga/formats.py` — `H5adReader` is the shortest complete example.
- **Real Stereo-seq files.** The fixtures are synthetic. If a real GEF trips the reader —
  different dtypes, missing attributes, a SAW version we have not seen — that is exactly the
  report we want.
- **Benchmarks that break the invariant.** Anything making peak resident memory exceed the
  declared budget is a bug, not a tuning issue.
- **Small hardware reports.** "Ran X on a 2016 ThinkPad with 8 GB and it worked / did not" is
  genuinely useful and we will act on it.

## What we are not looking for, yet

- Analysis algorithms. GEMINGA moves bytes; Scanpy, Seurat, Stereopy and squidpy do the science.
- Cluster or MPI support. The target is one machine.
- New on-disk formats. We read what instruments and archives already emit.

## House rules

- Every new reader gets a test asserting peak budget use stays under the declared cap.
- Rust: `cargo fmt`, `cargo clippy -- -D warnings`. Python: `ruff check`.
- Public functions get a docstring saying what crosses the wire and what becomes resident.
- No claim in the docs without a number behind it that CI can reproduce.

## Building these docs

```bash
pip install sphinx furo myst-parser sphinx-copybutton sphinx-design
sphinx-build -b html docs docs/_build/html
```
