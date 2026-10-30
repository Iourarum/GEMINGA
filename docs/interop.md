# Working with the other tools

GEMINGA reads; HyVESTA indexes and detects; Regulus, AMETHRINE and POLARIS render; BELLATRIX
computes; PRISM orchestrates. Three parts of GEMINGA exist so those tools can rely on it
without a per-tool adapter, and each is a contract rather than a convenience.

## Cell embeddings: `obsm`

Foundation-model embeddings — Geneformer, scGPT, scVI, a PCA — live in an h5ad under
`obsm/<key>`, beside `X` but with far fewer columns. They stream under the same contract as
everything else:

```python
r = g.open("atlas.h5ad", budget=b, obsm="X_geneformer", obs_columns=["donor"])
r = g.open("atlas.h5ad::obsm/X_geneformer", budget=b)     # the same, as one string
g.obsm_keys("atlas.h5ad")      # {'X_geneformer': (n_obs, 256, 'float32', 'array'), ...}
for chunk in r:                # {"X", "obsm_key", "obs_start", "obs_end", "n_vars",
    ...                        #  "resident_bytes", "obs_names", <obs columns>}
```

The chunk dict has the same keys as `H5adReader`'s, so a consumer written for counts works on
embeddings unchanged, and `obs_names` (barcodes) ride along because an embedding row is
worthless unless it maps back to a cell. Dense and CSR-encoded obsm both stream; CSC is refused
with the fix. Measured under a **64 KB** budget: 32 blocks, the lease held at any moment equal
to that block, zero denials, a bit-identical round trip.

Adding it exposed a latent bug shared by every reader: the 1 MB chunk floor could exceed a small
budget and plan a chunk that can never fit. The floor is now capped at what is free — in
`H5adReader`, `TenXMatrix`, the Arrow reader and every `View`.

## The catalog is a contract

Other tools read the catalog, so its record shape is frozen and versioned. Every event carries
`"schema": "geminga.catalog/1"`; `CatalogEntry.SCHEMA` documents each field with its JSON type;
`Catalog.validate()` checks a file against it. Adding a field keeps the number; renaming,
retyping or removing one bumps it. Older lines still fold.

Facets are what downstream tools key on, so they get a **controlled vocabulary**:

```python
FACETS = ("organism", "tissue", "assay", "disease", "sample", "donor", "condition", "platform")

cat.record("atlas", "atlas.h5ad", profile=p,
           facets={"organism": "Homo sapiens", "tissue": "glioblastoma"})
cat.annotate("atlas", facets={"donor": "MGH-01"})
cat.facets_for("atlas")     # every key present, '' where unset
cat.facet_table()           # one stable-column row per dataset
cat.note_local("atlas", "/scratch/atlas.h5ad")   # where copies and regions landed
```

Values are free strings — use UBERON, CL, MONDO or EFO terms where you have them — but the keys
are fixed, and a typo in a key **raises** rather than silently losing data. The profiler's assay
becomes a facet unless the analyst set one. HyVESTA reads the table and broadcasts dataset-level
facets onto rows.

## Harvest manifests carry `kind` and `stat`

A `Harvest` entry now says what it *is* and how sure it is, so a consumer can filter without
opening files:

```python
h = g.harvest("out/run1", producer="hyvesta")
h.add_table("bundle", cols, fmt="ndjson", kind="bundle", targets=["regulus", "prism"],
            stat={"resultant": 0.52, "p": 0.002, "q": 0.004})
h.for_kind("bundle"); h.significant(0.05)
h.write_dispatch()          # PRISM sees "kind" and "stat" on every entry
```

`kind` is `region` for GEMINGA's own cuts and `patch`, `relation`, `bundle`, `ridge`,
`enrichment` or `table` for other producers; `stat` is a free JSON object — a statistic, its
null, p and q. `add_table()` writes a table you already hold, with the same checksum, manifest
line and dispatch as a cut region — which is what lets HyVESTA write *through* `Harvest`
rather than imitate it, so the two producers emit identical layouts. Manifests are
`geminga.harvest/2` and dispatch files `geminga.dispatch/2`; both additive, and a `/1` file
still loads with `kind="region"`, `stat={}`.
