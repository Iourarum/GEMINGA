# Quickstart

Five minutes, no data of your own required.

## Generate fixtures

```bash
python examples/make_omics_fixtures.py fixtures       # FASTQ, FASTA, h5ad, Arrow, OME-TIFF
python examples/make_adapter_fastq.py fixtures/reads_adapters.fastq
python examples/make_stereoseq_fixtures.py fixtures   # GEF with a real bin pyramid, GEM, ssDNA
```

## Set a budget and open something

```python
import geminga as g

b = g.Budget(ram="256MB")
ds = g.open("fixtures/chip.gef", budget=b)
```

`g.open()` dispatches on what the target is — a Hub id, a URL, or a path by extension.
`.gef`, `.gem`, `.h5ad`, `.parquet`, `.fastq[.gz]`, `.arrow`, `.tif`, `.zarr` all work.

## Ask what something costs

```python
v = g.view("fixtures/reads_adapters.fastq", budget=b)
g.print_cost(v)
```

```text
window 0-200,000  zoom 0 (stride 1)
  200,000 records — 100.00% of the dataset
  1 chunk(s) x 63.8 MB resident, 63.8 MB in total
  .load() fits: 256.0 MB free
```

## Navigate

```python
v.select_frac(0.55, 0.60)     # a slice of the extent
v.zoom_in(4)                   # narrow, resolution rises to match
v.auto_zoom()                  # or: coarsest level that still fits
print(v.overview().sparkline("gc"))
```

## Load, or stream if it will not fit

```python
try:
    data = v.load()
except g.CostError as e:
    print(e)                   # says exactly what to change
    for chunk in v.chunks():   # bounded streaming instead
        ...
```

## Search

```python
hits = v.search(["AGATCGGAAGAGC"], both_strands=True)
print(hits["n_hits"], "in", hits["records_matched"], "records")
print(v.hit_profile(hits, 60).sparkline("hits"))
```

## Draw

```python
g.render.navigator(v, "nav.png")            # overview + window + detail + cost
g.render.hits(v, hits, "hits.png")          # hit density on the overview axis
```

## Run the tours

```bash
python examples/navigate.py fixtures .
python examples/stereoseq_tour.py fixtures .
python examples/all_formats.py fixtures
python examples/budget_behaviour.py
```
