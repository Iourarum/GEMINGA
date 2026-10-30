# Oversized local storage

The streaming side of GEMINGA assumes bytes are expensive to fetch and cheap to forget. This
page is about the opposite situation, which is at least as common and usually more annoying:
the dataset is **already downloaded**, it lives on an external drive or a NAS mount, and it is
larger than the machine's RAM, its VRAM, and quite possibly its internal disk. Nothing needs
fetching. What is scarce is somewhere to put the part you are working on.

The shape is a three-level hierarchy, each level bounded and each explicitly declared:

```text
vault (slow, huge, read-only)  ->  staging (fast, bounded)  ->  device (RAM/VRAM/XPU)  ->  pipeline
```

:::{admonition} The design choice that matters
:class: budget
**Staging holds extracted regions, not file copies.** Copying a 400 GB GEF onto a 200 GB NVMe
to work on a 2 GB corner of it is exactly the mistake this is meant to prevent. A region is read
from the vault through the same windowed readers as everything else, written to staging in an
analysis-ready form, and from then on every re-read is local. One pass over the slow link per
region, not per look.
:::

## The vault

```python
v = g.vault("/mnt/nas/stereoseq", note="18 TB array")
v.scan(profile=True)          # once; cached in a manifest beside the data
v.print_table()
```

```text
vault /mnt/nas/stereoseq  —  5 datasets, 203.1 MB, scanned 0.0 h ago
       size format             assay                    path
    72.0 MB TIFF               microscopy / histology ? tissue.ome.tif
    61.7 MB FASTQ              bulk RNA-seq ?           reads.fastq
    33.3 MB Stereo-seq GEF     Stereo-seq               chip.gef
    24.2 MB AnnData h5ad       scRNA-seq ?              sc.h5ad
  4 member(s) profiled but not settled — vault.find(unsettled=True)
```

Scanning walks the tree, records size and mtime, and runs [the profiler](workspace.md#what-is-this-dataset)
on each member — so the `?` marks are the same honest uncertainty as everywhere else, not a
scan failure. The manifest is written **beside the data** when the mount allows it, so the next
person to plug in that drive inherits the work; it falls back to `~/.geminga/` on a read-only
mount.

```python
v.find(assay="Stereo-seq")           # by what it is
v.find(unsettled=True)               # the queue worth a human look
v.oversized("16GB")                  # the members that actually need this workflow
view = v.open("chip.gef", budget=b)  # opened in place — nothing is copied
v.add_to(ws, "chip.gef", ram="3GB")  # or straight into a workspace cell
```

## The working area

```python
b  = g.Budget(ram="8GB", spill="500GB")
st = g.staging("/scratch/geminga", capacity="200GB", budget=b)
```

Staging reserves its capacity against the budget's `spill` tier, so two things cannot be
promised the same disk. It also checks the filesystem's real free space, because a configured
capacity is a claim and `df` is a fact.

### Extracting a region

```python
view = v.open("chip.gef", budget=g.Budget(ram="4GB")).use_bin(20)
r = st.extract("chip:lobe-a", view, window=(12000, 16000, 9000, 13000), pin=True)
```

That is the one pass over the slow link. Afterwards:

```python
tbl = st.read("chip:lobe-a")     # local, repeatable, fast
st.print_report()
```

```text
staging /scratch/geminga
  897.5 KB of 40.0 MB used in 3 region(s), 1 pinned  (6.1 GB left on the filesystem)
  key                             size        rows  hits  window
   chip:corner                  9.2 KB       5,273     0  3400x4000x3400x4000
   chip:lobe-b                257.2 KB     191,162     0  2400x3200x2200x3000
  *chip:lobe-a                631.1 KB     549,514     2  1000x2200x1000x2200
```

### Eviction

Least-recently-used, with pinning, because that matches how people work: you return to the
region you were just looking at, and the one that took twenty minutes to extract should survive
until you say otherwise. A region that cannot fit even after evicting everything unpinned raises
`StagingFull` naming the fix:

```text
StagingFull: a 12.4 GB region cannot fit a 8.0 GB working area. Narrow the window, use a
coarser bin or zoom level, project fewer columns, or give staging more room.
```

## Devices

```python
g.describe_devices()
```

```text
device         name                               free      total  notes
cpu            host                             3.6 GB     3.9 GB  host memory
cuda:0         NVIDIA RTX A6000                44.1 GB    47.5 GB
xpu:0          Intel Arc A770                  15.2 GB    16.0 GB
tenstorrent:0  Tenstorrent Tensix                  0 B        0 B  TT-Metalium manages its own memory
```

A `DevicePool` is a bounded allowance on one device, refused up front if it will not fit:

```python
pool = g.device_pool("cuda:0", "6GB", budget=b)
with pool.place(array, "lobe-a counts") as t:
    model(t)                       # t is a torch tensor on the device
```

Free figures come from the driver on CUDA and XPU, not from an estimate. Leaving the context
releases the allowance and, on CUDA, drops the reference so the caching allocator can reuse it.

:::{caution}
GEMINGA does not allocate device memory — torch, CuPy or a vendor runtime does. What it can do
is refuse an over-commitment and give you the number before you find out the hard way.

Accelerators that are not GPUs — Tenstorrent's Tensix parts, NPUs, TPUs — are listed
`accounting_only`. They get the bound but not the placement, because their memory is managed by
their own runtime and there is no general `.to(device)` to call. Moving the bytes stays yours
to do; refusing to over-commit is still done for you.
:::

## Handing regions to other tools

GEMINGA moves bytes. It does not fit models, render manifolds or run flux balance analysis, and
it should not start. What it owes a toolchain is a clean handoff.

```python
h = g.harvest("/scratch/handoff/chapter3", note="tumour margin survey")
h.add(view, "margin-a", fmt="parquet", targets=["regulus", "prism"])
h.add(view, "margin-b", fmt="arrow",   targets=["amethrine"])
h.add(view, "gene7",    fmt="ndjson",  targets=["prism"])
h.add_staged(st, "chip:lobe-a", targets=["regulus"])   # promote without re-reading the vault
h.print_table()
```

| format | who reads it |
|---|---|
| `arrow` | Python, R (`arrow`), Ruby (`red-arrow`), Julia |
| `parquet` | Python, R (`arrow`), Ruby (`red-arrow`), DuckDB |
| `ndjson` | anything, including Perl, line by line |
| `csv` | anything |
| `npy` | Python, R (`reticulate`), Julia |

Every region carries its source, window, bin or zoom level, gene selection, row count and a
SHA-256. `h.verify()` re-checks them all — cheap insurance before handing a set to a
collaborator.

### The dispatch file

Perl has no first-class Arrow reader, so an orchestrator written in it gets NDJSON for payloads
and, more usefully, a flat target-keyed index it can act on without understanding Arrow at all:

```python
h.write_dispatch()     # -> dispatch.json
```

```json
{
  "schema": "geminga.dispatch/1",
  "targets": {
    "regulus":   [{"name": "margin-a", "path": "/abs/margin-a.parquet",
                   "format": "parquet", "rows": 549514, "sha256": "...",
                   "window": [1000, 2200, 1000, 2200], "bin_size": 20}],
    "amethrine": [{"name": "margin-b", "path": "/abs/margin-b.arrow", "...": "..."}],
    "prism":     [{"name": "gene7", "path": "/abs/gene7.ndjson", "...": "..."}]
  }
}
```

One object per target, absolute paths, formats and row counts. A Perl driver reads it with any
JSON module and shells out to the R, Ruby or Python tool that owns each target. Nothing in
GEMINGA imports those tools or depends on them existing — the manifest is the whole contract.

## The chain end to end

```python
b = g.Budget(ram="8GB", vram="6GB", spill="200GB")

v  = g.vault("/mnt/nas/stereoseq").scan(profile=True)
st = g.staging("/scratch/geminga", capacity="150GB", budget=b)
h  = g.harvest("/scratch/handoff/ch3", note="margin survey")

view = v.open("chip.gef", budget=g.Budget(ram="4GB")).use_bin(20)
for name, win in regions:
    st.extract(name, view, window=win)
    h.add_staged(st, name, targets=["regulus", "prism"])

h.write_dispatch()
```

```text
vault    203.1 MB across 5 datasets (never copied)
staging  12.1 MB of 40.0 MB in 5 regions
harvest  5.6 MB in 4 regions for 3 tools
budget   spill 460.0 MB free of 500 MB
```
