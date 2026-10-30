# Getting hold of data

GEMINGA does not download. `fasterq-dump`, `ascp`, `fastq-dl`, `kingfisher`, `pyega3` and
`aws s3` already exist and writing a worse one would help nobody. What is missing, and what
costs graduate students real money, is the step before: working out **which route to use**,
what it will cost, and whether the transfer is needed at all.

## The one fact worth internalising

**ENA mirrors SRA and serves FASTQ free** — over FTP, HTTPS and Aspera, already converted, with
no `fasterq-dump` step. A cloud route that bills you for the same bytes is almost always
avoidable.

```{list-table} The same data, three ways
:header-rows: 1

* - volume
  - ENA / AWS Open Data
  - AWS requester-pays
  - GCP requester-pays
* - one 10x run (2.4 GB)
  - **free**
  - $0.22
  - $0.29
* - a 58-sample LUAD atlas (180 GB)
  - **free**
  - $16.20
  - $21.60
* - a TCGA WGS slice (1.2 TB)
  - **free**
  - $108.00
  - $144.00
```

Egress figures are published list prices and change; the point is the order of magnitude.
`g.EGRESS_USD_PER_GB` holds them and is meant to be edited if your institution has a waiver.

## Planning before transferring

```python
p = g.acquisition_plan("SRR1448774")
print(p.explain())
```

```text
SRR1448774  (sra_run)
  2 file(s), 2.4 GB total
  ena-http          2 file(s)     2.4 GB      free  streamable
                 EBI mirror of SRA, already converted to FASTQ
  ena-ftp           2 file(s)     2.4 GB      free
  ena-aspera        2 file(s)     2.4 GB      free  install ascp
                 FASP, port 33001. Much faster on a long fat pipe; needs the proprietary ascp client
  aws-odp           1 file(s)         0 B      free
                 AWS Open Data Sponsorship Program — AWS pays the egress. Use --no-sign-request.
  -> take ena-http and inspect it in place before transferring anything
```

Routes are ranked free-and-streamable first. `plan_many()` prices a whole shortlist:

```python
g.print_shortlist(g.plan_many(["GSE178341", "SRR1448774", "EGAD00001005054"]))
```

## Peeking without transferring

ENA, GEO, Zenodo and most CDNs honour Range requests, so the first slice of a 40 GB FASTQ can
be read for a few hundred kilobytes:

```python
info = g.peek("SRR1448774", n_bytes="512KB")
# {'ok': True, 'format': 'FASTQ', 'read_length_mean': 101.0, 'gc_fraction': 0.4832,
#  'reads_sampled': 500, 'fraction_read': 0.0002, 'cost_avoided_usd': 0.0}
```

For a shortlist of two hundred candidates, this is the difference between a week of transfers
and an afternoon. It recognises FASTQ, FASTA, Parquet, HDF5 (h5ad/GEF/10x), MatrixMarket,
Stereo-seq GEM, TSV and CSV, and handles gzip members truncated at the range boundary.

Feed the result to [the profiler](workspace.md#what-is-this-dataset) to get ranked assay
hypotheses for a dataset you have not downloaded.

## Delegating the transfer

```python
g.print_tools()          # what is on PATH, and what each is for
out = g.fetch("SRR1448774", "/data")        # dry run by default
print(out["command"])
g.fetch("SRR1448774", "/data", dry_run=False)
```

:::{admonition} `fetch` is dry-run by default, on purpose
:class: budget
A transfer that might cost money or fill a disk should be a decision, not a side effect of
exploring. The default prints the exact command it would run.
:::

| route | tool | note |
|---|---|---|
| `ena-http`, `ena-ftp` | `aria2c` / `curl` / `wget` | free, Range-capable |
| `ena-aspera` | `ascp` | FASP is proprietary — `ascp` is the only client that speaks it |
| `aws-odp` | `aws` | `--no-sign-request`; AWS pays under the Open Data Sponsorship Program |
| `aws-requester`, `gcp-requester` | `aws`, `gcloud` | **billed to you** |
| `sra-prefetch`, `dbgap` | `prefetch` | dbGaP needs an `.ngc` key |
| `ega` | `pyega3` | needs Data Access Committee approval |

Missing tools and missing approvals are reported, never guessed around. Approval is named
first, because installing a client does not get you the data.

## Repository-specific notes

**GEO** supplementary files are free over HTTPS and Range-capable, so processed matrices
(`.h5ad`, `.mtx.gz`, `.tsv.gz`) can often be read in place. Raw reads for human cohorts usually
sit behind dbGaP or EGA instead.

**EGA and dbGaP** are controlled access. There is no way around the Data Access Committee, and
no route planner can invent one. Once approved, the transfer itself is free.

**CNCB-NGDC / GSA** (`CRA…`) serves open GSA freely; GSA-Human requires an application.

**STOmics DB** (`STDS…`) hosts Stereo-seq GEF and GEM, which GEMINGA reads natively — see
[the Stereo-seq guide](stereoseq.md).

**Dropbox** share links are HTML previews until `dl=1` is set; the planner rewrites them.
Range support is inconsistent, so streaming may fall back to a download.

**Aspera** is worth the setup on a long fat pipe — a trans-continental transfer that crawls
over TCP often runs an order of magnitude faster over FASP. The client is proprietary and
free-to-use; the planner builds the `ascp` command with the right host, port and key path.

## What cannot be worked around

Some things are genuinely closed, and a tool that pretended otherwise would waste your time:

- **Controlled-access human data** (EGA, dbGaP, GSA-Human) needs committee approval. Budget
  months, not minutes.
- **Requester-pays buckets with no public mirror** cost what they cost. Check for an ENA or
  Open Data copy first; if there is none, the bill is real.
- **Cold-storage tiers** may add a retrieval charge *and* a delay on top of egress.
- **FASP** cannot be reimplemented. `ascp` or nothing.
