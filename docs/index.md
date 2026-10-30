# GEMINGA

```{image} _static/logo-wordmark.svg
:alt: GEMINGA
:width: 620px
:class: gem-figure
```

**G**ated, **E**xtensible, **M**emory-budgeted **I**ngestion for **N**-modal **G**enomics and **A**rrays

Stream any omics dataset through a memory budget you set, and navigate it like a map: zoom, pan,
select an interval, search it — and be told what each move costs *before* anything loads.

Built for **resource-aware PhDs**: the people who know exactly how much RAM they have, because
they have had to. A laptop, a shared node with a quota, a GPU with 8 GB, a disk that will not
hold the dataset. GEMINGA treats the budget as the input rather than the thing you discover when
the kernel kills your job.

```python
import geminga as g

budget = g.Budget(ram="6GB", vram="4GB", spill="100GB")

v = g.open_gef("chip.gef", budget=budget)   # Stereo-seq: bin1..bin200 are the zoom levels
v.select(12000, 16000, 9000, 13000)          # a window on the chip
v.cost()                                     # what would this take?  nothing loaded yet
v.auto_bin()                                 # finest bin that fits the budget
data = v.load()                              # budget-enforced
```

:::{admonition} The one invariant
:class: budget
Peak resident memory never exceeds the budget you declared. If it does, that is a bug, and
there is [an issue template](https://github.com/<user>/geminga/issues/new/choose) for it.
:::

```{toctree}
:maxdepth: 2
:caption: Getting started

install
concepts
quickstart
```

```{toctree}
:maxdepth: 2
:caption: Guides

stereoseq
glioma
abstract
poster
crispr
wetlab
neuro
workspace
vault
remote
acquire
interop
formats
navigation
search
visuals
stage
explorer
selections
library
journal
orchestrate
prep
codec
reserve
```

```{toctree}
:maxdepth: 2
:caption: Reference

honest
api
benchmarks
contributing
```
