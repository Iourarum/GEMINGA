# Core concepts

Four ideas carry the whole library. Everything else is a format reader.

## The budget is the input

Existing streaming APIs stream in *examples*: `batch_size`, `buffer_size`, both counted in rows.
Nothing tells you, or enforces, how many *bytes* are resident. On a 96 GB workstation you never
notice. On 16 GB you find out when the kernel kills the process.

GEMINGA inverts that. You declare bytes; the library derives the chunking.

```python
b = g.Budget(ram="6GB", vram="4GB", spill="100GB", timeout_s=120)
```

Three independent pools:

`ram`
: Host memory holding decoded chunks. Every reader acquires from it before materializing.

`vram`
: Device memory. GEMINGA does **not** allocate it — you do, via torch or cupy — but it refuses
  to let you over-commit. Acquire around `.to("cuda")`, release after the step.

`spill`
: Scratch space on disk for chunks you want to keep but not hold in RAM.

Requests that can never fit fail immediately, naming the fix. Requests that merely do not fit
*right now* block until something releases — backpressure, not OOM.

## Plan before you read

Every reader produces a plan from metadata that is already cheap to reach, before touching data:
Parquet row-group statistics, an h5ad's `indptr`, a GEF's bin levels, a sequence index sidecar.

```python
v.cost()
# {'units': 2_946_567, 'resident_bytes': 29_465_670, 'resident_human': '28.1 MB',
#  'chunks': 1, 'fits': True, 'fits_whole': True, ...}
```

`cost()` separates two questions that are usually confused:

`fits`
: Can this be streamed, one chunk at a time, with `.chunks()`?

`fits_whole`
: Can `.load()` hand you the entire window at once?

`.load()` gates on the second, because that is what it does. When the answer is no, the error
says what to change:

```text
CostError: the window holds 65.3 MB at zoom 0 but 16.0 MB is free; zoom out 3 level(s),
narrow the window to ~24% of its span, stream it with .chunks() instead of .load(),
or raise the 'ram' budget
```

## Zoom is resolution, not just extent

Zooming out loads a *coarser* view of a wider span, so every level costs about the same to draw.
That is what makes navigation feel instant on a laptop — and it is why `zoom_in()` raises
resolution as it narrows, the way a map does.

```{list-table} The same FASTQ, three zoom steps
:header-rows: 1

* - move
  - window
  - zoom
  - stride
  - records
  - resident
* - start
  - 0–200,000
  - 4
  - 16
  - 12,500
  - 4.0 MB
* - `zoom_in(4)`
  - 100,000–150,000
  - 2
  - 4
  - 12,500
  - 4.0 MB
* - `zoom_in(4)`
  - 118,750–131,250
  - 0
  - 1
  - 12,500
  - 4.0 MB
```

Decimation keeps distributions honest: GC across a 64× stride range moves from 0.5000 to 0.5008.

:::{admonition} Where the levels come from
:class: measured
For Stereo-seq GEF, the levels are not decimation at all — they are the bin sizes SAW already
computed. Zooming out switches to `bin10`, `bin50`, `bin200`. For Parquet the floor is the row
group; for images it is the tile; for sequences it is the index sidecar.
:::

## Search skips what it can prove it does not need

Two mechanisms, one idea: decide from cheap metadata what *could* match, fetch only that.

- **Parquet**: footer min/max rules out whole row groups before a byte leaves the server.
- **Stereo-seq GEF**: the per-bin `gene` dataset stores `(gene, offset, count)` into
  `expression`, so one gene is one contiguous slice.
- **Sequences**: Aho-Corasick matches every pattern — and every reverse complement — in one
  pass, so 300 adapters cost about what one costs.
