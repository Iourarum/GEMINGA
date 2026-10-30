# Search

## Parquet predicate pushdown

Footer min/max rules out whole row groups before a byte of data leaves the server.

```python
ds = g.open("spatial.parquet", budget=b)
p = ds.prune("x", 12000, 12500, ["x", "y", "count"])
p["row_groups_kept"], p["bytes_to_fetch_human"], p["selectivity"]

batches = ds.search("x", 12000, 12500, ["x", "y", "count"])
```

`prune()` decides and reports; `search()` executes and applies the predicate exactly.

:::{warning}
Pushdown pays off exactly as much as your data is clustered. The same 2M rows:

| layout | row groups kept | bytes fetched | selectivity |
|---|---|---|---|
| unsorted | 20/20 | 9.7 MB of 9.7 MB | 100 % |
| sorted by `x` | 2/20 | 497 KB of 5.3 MB | **9.2 %** |

If your writer does not sort, pruning cannot help. Sort on the column you filter on.
:::

Row groups whose statistics are missing are always kept, and reported separately as
`kept_without_stats` — pruning is never allowed to lose a row.

## Stereo-seq gene selection

Each GEF bin level carries `(gene, offset, count)` into `expression`, so one gene is one
contiguous slice.

```python
v = g.open_gef("chip.gef", budget=b).use_bin(10)
v.select_genes(["Gapdh"])
d = v.load()
```

**139× faster** than reading all genes on the fixture chip, and the unwanted rows never enter
memory.

## Sequence search

Aho-Corasick matches every pattern, and optionally every reverse complement, in one pass — so
300 adapters cost about what one costs.

```python
v = g.view("reads.fastq", budget=b)
hits = v.search(["AGATCGGAAGAGC"], both_strands=True, max_hits=500_000)
hits["n_hits"], hits["records_matched"], hits["records_scanned"]
```

Restrict to the current window with `within_window=True`, which is how you localise:

```python
v.reset()
hits = v.search([adapter])                       # where is it?
print(v.hit_profile(hits, 60).sparkline("hits"))
# ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█████████▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁

v.select(110_000, 140_000)
v.search([adapter], within_window=True)          # 45.0 % of reads carry it here
v.select(0, 100_000)
v.search([adapter], within_window=True)          # 0 hits outside the band
```

:::{admonition} Measured
:class: measured
13,486 hits across 200,000 records, 30 Mbase scanned in **0.38 s (80 Mbase/s)**, single core.
:::

`g.revcomp(seq)` is exposed separately if you want it.
