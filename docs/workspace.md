# Workspaces, profiling and the catalog

Three things a student needs once they have more than one dataset: several open at once
without them interfering, a way to find out what each one actually is, and a record of both.

## Several datasets, partitioned

```python
ws = g.workspace(ram="8GB", name="thesis")
ws.add("chip",  "run1.gef",      ram="3GB", note="tumour section")
ws.add("atlas", "reference.h5ad", ram="2GB")
ws.add("reads", "run1.fastq.gz",  ram="1GB")

ws["chip"].view.select(12000, 16000, 9000, 13000)
ws.print_report()
```

```text
thesis: 6.0 GB reserved of 8.0 GB, 2.0 GB free
  cell         state          slice      peak  loads  target
  chip         open          3.0 GB    412 MB      1  run1.gef
  atlas        open          2.0 GB    1.1 GB      1  reference.h5ad
  reads        declared      1.0 GB       0 B      0  run1.fastq.gz
```

A cell reserves its slice on creation, so a fourth cell that would over-commit the workspace
is refused *before* anything opens:

```text
MemoryError: cannot reserve 4GB ram for cell 'extra': the workspace has 2.0 GB free.
Close a cell, or give this one a smaller slice.
```

### Lifecycle

`suspend()` closes file handles and caches but remembers the window, zoom, bin and gene
selection; `resume()` puts them back. With `release_reservation=True` the slice returns to the
workspace so a sibling can grow into it.

```python
ws["reads"].suspend(release_reservation=True)   # 1 GB back to the pool
ws.add("big", "other.gef", ram="900MB")
ws["reads"].resume()                             # window restored exactly
```

### Failure stays in its cell

```python
ws.add("broken", "missing.gef", ram="10MB")
try:
    ws["broken"].view
except Exception:
    pass
ws["broken"].state   # 'failed'
ws["chip"].state     # 'open' — unaffected
```

### Working across cells

Datasets rarely share coordinates but often share structure. `select_frac` puts every cell at
the same *relative* position in its own extent — one pair applies to every axis, whatever its
dimensionality:

```python
for name, c in ws.select_frac(0.45, 0.55).items():
    print(name, c["units"], c["unit"], c["resident_human"])
```

```text
chip      29,465  observations   287.7 KB
atlas      2,000  cells            1.7 MB
reads     20,000  records          6.5 MB
```

`ws.select(...)` applies an absolute window instead, for cells that genuinely share a frame —
a GEF and the image registered to it. `ws.broadcast(fn)` runs anything over every cell,
collecting results and putting failures in `ws.errors`.

## Why cells are not containers

:::{admonition} The design decision, stated
:class: budget
The obvious reading of "isolate each dataset" is Docker, Apptainer or a FreeBSD jail. For
GEMINGA that is the wrong mechanism, and it is worth being explicit about why.

A jail isolates at the OS level: namespaces, a chroot, kernel-enforced limits. Of that list,
the only part GEMINGA needs is the resource limit — and the budget governor already is one,
at the cost of an atomic add. The other parts actively break the library's premise. Its whole
performance argument is that a decoded Arrow batch reaches Python with no copy; put the reader
behind a namespace boundary and every batch needs serialization or a shared-memory handshake.
A container starts in tens to hundreds of milliseconds; navigating an index here takes under
one. Isolation of that kind would cost precisely the thing it was meant to protect.

What is worth borrowing is the jail's *model*: a partitioned resource bound, a namespace, a
lifecycle, and failure containment. All four are in `Cell`, and all four are nearly free.
:::

Real OS-level isolation is still there for the case that earns it — a reader that might exceed
its budget in ways the governor cannot see, such as a C extension that leaks:

```python
n = ws["chip"].run_isolated(lambda v: len(v.load()["x"]))
```

This forks a subprocess under an `RLIMIT_AS` allowance and pickles the result back. It is
opt-in per call because it costs a fork, a fresh open and a round-trip — honest overhead for
honest protection. A runaway is killed by the allowance rather than by the OOM killer:

```text
RuntimeError: isolated run in cell 'chip' failed: MemoryError
```

:::{note}
The allowance is measured as *additional* address space. A forked child already carries the
parent's interpreter, numpy and Arrow mappings, so the raw cap is added to the child's measured
baseline — otherwise the limit would kill it before it did any work.
:::

## What is this dataset?

`g.profile()` does not return a label. It returns ranked hypotheses, the evidence behind each,
and — where two remain plausible — the observation that would settle it.

```python
p = g.profile("data.h5ad", url="https://portal.example/dataset/123")
print(p.explain())
```

```text
data.h5ad
  format   : AnnData h5ad
  molecule : RNA
  assay    :
        62%  scRNA-seq
        28%  bulk RNA-seq
         2%  snRNA-seq
  NOT SETTLED — scRNA-seq versus bulk RNA-seq.
  To decide: intronic fraction: scRNA-seq is exon-dominated (<30% intronic).
```

### Confidence requires evidence that discriminates

A container tells you the *format*, not the assay: a `.h5ad` is equally an scRNA-seq or an
snRNA-seq file. So `confident` is False unless something from **metadata, content or the source
page** actually separated the candidates. Structure and filename evidence identify the
container and never, on their own, settle the assay.

This is the whole reason the module exists. scRNA-seq and snRNA-seq produce byte-identical
formats and differ in a property — intronic fraction — you can only see by looking at reads or
metadata. A tool that prints one of those names with no hedging is lying to a student who does
not yet know to doubt it.

### Evidence sources

| source | what it settles | weight |
|---|---|---|
| container structure | format, near-conclusively | high for format, none for assay |
| pipeline metadata | assay, when the pipeline recorded it | highest available |
| content statistics | read length, GC, barcode shape, feature naming | weak alone, useful combined |
| the source page | often names the assay outright | frequently the only sc/sn signal |

### On topic modelling

LDA is the obvious suggestion for the page-scanning step and the wrong tool at this scale: it
needs a corpus to fit and yields topics that then need naming, while here there is one document
and a known vocabulary. A weighted keyword score over curated terms is transparent, needs no
training, and is auditable — the analyst sees which words fired and how often. The upgrade path
once a corpus exists is sentence-embedding similarity against protocol descriptions, not LDA.

:::{warning}
`ASSAYS` priors and term weights are hand-set and meant to be argued with. They are a starting
prior, not a trained model. Adjusting them for your field is expected.
:::

## Keeping track

```python
cat = g.catalog("~/.geminga/catalog.jsonl")
cat.record("chip1", "run1.gef", url=..., profile=p, note="tumour section")
cat.note_streamed("chip1", 28_100_000, "bin1 window")
cat.annotate("chip1", tags=["thesis", "ch3"], note="analyst confirmed: FF, not FFPE")
cat.print_table()
cat.unsettled()          # the queue of things automated profiling could not pin down
```

Append-only JSONL, on purpose: a text log survives a corrupted index, a half-written database
and a version bump; it diffs, greps and copies between machines. DuckDB is a query surface over
the same file rather than the storage itself, so nothing breaks if it is absent.

```python
con = cat.duckdb()
con.execute("SELECT key, assay FROM events WHERE event='record'").fetchall()
cat.to_parquet("catalog-snapshot.parquet")
```

Pass a catalog to a workspace and cell openings are logged automatically:

```python
ws = g.workspace(ram="8GB", catalog=cat)
```

## The optional assistant

GEMINGA ships no inference engine. Vendoring one would add gigabytes of weights and a CUDA or
Metal build dependency to a library whose premise is that the machine is small. Instead, a thin
adapter over whatever you already run:

| backend | endpoint | covers |
|---|---|---|
| `openai` | `/v1/chat/completions` | **mistral.rs**, llama.cpp `llama-server`, vLLM, LM Studio, hosted APIs |
| `ollama` | `/api/generate` | a local Ollama daemon |
| `template` | none | offline prose, always available |

```python
ai = g.assistant()                       # probes localhost, falls back to template
print(ai.explain_protocol("snRNA-seq"))  # ~300 words plus places to read further
print(ai.describe(p))                    # summarises a Profile, including what is unresolved
```

mistral.rs is the natural pairing for a Rust-core project — pure Rust on Candle, quantized
models, CUDA/Metal/CPU, and an OpenAI-compatible server that this adapter speaks. Note that
`rustformers/llm` is archived and should not be built on.

The `template` backend is not a stub: it carries checked offline prose for the common protocols
and is what runs when nothing is reachable, so **nothing in GEMINGA ever requires a model**.

:::{caution}
`AssistantReply.grounded` is False when a model generated the text. Model output is a lead, not
a citation — the system prompt forbids inventing URLs and accessions, but verify anything you
carry into a methods section.
:::
