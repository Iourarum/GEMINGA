# Selections

Navigating a dataset produces questions: *this corner of the rim looks different*, *I need more
cells like these*, *what happened in the culture around day four*. A selection is that question
pinned to coordinates — and kept, with what you later found about it.

```{image} _static/selections.png
:alt: marking a region, measuring it, finding more like it — on a map and on a time series
:class: gem-figure
```

## Marking

```python
book = g.selections()                                  # ~/.geminga/selections.jsonl
sel = g.select(view, "rim-east", (2158, 2378, 1970, 2190), intent="need-data", book=book)
g.select_polygon(view, "margin", [(x, y), ...], book=book)
g.select_values("culture.parquet", "t", t0, t1, "hypoxia d3.9", book=book)   # a time window
```

Rectangles and polygons on spatial maps, intervals on records, cells and rows, and **value ranges
on a named column** — which is how a time series is selected, independent of row order on disk.
Each selection records the level of detail it was made at, any gene filter in force, and *why*:

| intent | drawn | meaning |
|---|---|---|
| `interesting` | red | worth coming back to |
| `need-data` | blue | an open question; `book.open_questions()` lists them |
| `compare` | violet | one side of a comparison |
| `candidate` | yellow | proposed by `similar()`, linked to its parent |
| `exclude` | grey | artefact, fold, tear |

In the explorer: **`s`** marks what is on screen, **`m`** describes the active selection and asks
for more like it, **`n`** steps through them. They are drawn on the map and on the minimap, and
persisted as they are made.

## Keeping track

The book is an **append-only JSON-lines log**: an event per line, nothing ever rewritten, a torn
last line survives a crash. Any language reads it — `book.write_dispatch()` writes a flat,
dataset-keyed index for PRISM, and `book.to_geojson()` gives spatial selections in the format
QuPath and the napari shapes layer exchange.

Every selection carries its dataset's path or URL **and fingerprint**, so it reattaches to the
same file next session — remote GEFs included — and is never silently applied to a different
file with the same name.

## 1. What is inside — `describe()`

For a Stereo-seq region: observations, total MID, density against the section, and genes
enriched or depleted **relative to the count expected** if the region had the section's
composition:

```text
VEGFA   +4.2   2,787 observed vs 156 expected
CA9     +4.1   2,662 vs 156
SNAP25 -11.1   0 vs 2,211          z = -47
```

Background totals come from a coarse level — per-gene totals do not depend on binning — and the
region is read through the spatial index when one exists.

:::{admonition} Why "observed vs expected"
:class: measured
An earlier version reported fold changes in counts-per-million with a small epsilon, and genes
simply absent from the region came out at −16.8: the pseudocount, not the biology. A one-molecule
pseudocount on both sides of observed-against-expected bounds that, and `z` says how surprising
a gap is for its size. Every number shown carries the counts it came from.
:::

For a time window: each numeric column inside against the whole series, as a z-score.

## 2. Where else looks like it — `similar()`

The section is tiled at the selection's own size (half-tile steps), each tile summarised as a gene
profile from one coarse level, and ranked by cosine similarity; near-duplicates are suppressed so
the result is *k* distinct places. On the glioma phantom, a patch marked on the east side of the
hypoxic rim returned six candidates **all on the same ring** — radii inside the rim band, angles
spread from −111° to +135° — without the search knowing a ring exists.

Time series are compared window by window: Polars' `group_by_dynamic` summarises every window's
numeric columns, and windows are ranked against the marked one. On a ten-day culture series with
four planted hypoxic episodes, marking one returned the other three as the top three (0.74–0.87),
with the next-best window at 0.36.

## 3. Data you do not have — `leads()`

```python
g.leads(sel, context=["glioblastoma", "hypoxia"], organism="Homo sapiens",
        local=vault_paths, online=True)
```

**Locally**, which datasets you already hold measured the selection's enriched genes, ranked by
coverage — and, when none did, a record saying which were checked, because an empty result is an
answer. **Externally**, queries against GEO and PubMed (NCBI E-utilities) and ENA (portal search)
built from those genes and your context terms. With `online=False` they are ready-to-run GET
requests any tool can issue; with `online=True` they are executed and accessions found are
attached, ready for `acquisition_plan()` — so "find more data" ends in a priced, dry-run download.

:::{note}
The environment these docs were built in cannot reach NCBI or EBI, so the online path is tested
on recorded responses rather than live ones.
:::
