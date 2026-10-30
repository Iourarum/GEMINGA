# Installation

## From a wheel

```bash
pip install geminga-0.3.0-*.whl
```

The wheel bundles the Rust core; nothing else is compiled at install time.

## From source

Needs Rust 1.85 or newer and [maturin](https://www.maturin.rs/).

```bash
git clone https://github.com/<user>/geminga && cd geminga
pip install maturin
maturin build --release && pip install target/wheels/geminga-*.whl
# or, inside a virtualenv, for development:
maturin develop --release
```

:::{tip}
On a machine with fewer than four cores, plain `maturin develop` (no `--release`) builds several
times faster and is fine for everything except benchmarking.
:::

## Optional dependencies by format

GEMINGA's core has no Python dependencies beyond `pyarrow`. Each format reader pulls its own:

| Format | Install |
|---|---|
| Stereo-seq GEF, h5ad | `pip install h5py scipy` |
| TIFF / OME-TIFF | `pip install tifffile` |
| Zarr / OME-Zarr, `MaskSink` | `pip install zarr` |
| Figures | `pip install matplotlib` |
| Dataframes, lazy Parquet | `pip install polars` |
| The wgpu stage | `pip install wgpu pillow` |

Everything at once: `pip install "geminga[all]"`.

## Verifying the install

```bash
python -c "import geminga as g; print(g.__version__, g.Budget(ram='1GB'))"
python examples/make_omics_fixtures.py fixtures
pytest -q
```

The suite asserts the budget invariant on every format, so a clean run means the ceiling holds
on your machine, not just ours.
