# Datasets too big to download

A CNGB project is fetched, by the portal's own suggestion, with

```bash
wget -c -nH -np -r -R index.html* --cut-dirs 4 ftp://ftp.cngb.org/pub/CNSA/data2/<CNP>/<CNS>/
```

which mirrors the whole tree. For a multi-terabyte project that is the wrong first step on any
machine a student has: it needs the disk, it needs days, and it moves everything before you know
which files you want. GEMINGA replaces it with three steps, none of which needs the mirror.

## 1. List before fetching

```python
tree = g.list_tree("ftp://ftp.cngb.org/pub/CNSA/data2/CNP0002646/CNS0513001/")
g.print_tree_summary(tree, disk_free=shutil.disk_usage("/data").free)
```

Directory listings only — FTP `MLSD` carries sizes for free. On a mock of that layout:

```text
15 files, 1.9 TB
  type             files        size   share
  .fq.gz              10      1.9 TB  100.0%
  .tissue.gef          1     33.3 MB    0.0%
  .gef                 1     25.0 MB    0.0%
  analysis-ready (GEF / GEM / h5ad): 62.1 MB = 0.0% of the tree
  local disk free: 7.4 GB — the full mirror does NOT fit
```

In Stereo-seq projects the terabytes are overwhelmingly raw FASTQ; the analysis-ready files are
usually a small fraction. The listing tells you which case you are in before anything moves.

## 2. Fetch only what analysis needs

```python
sel = g.fetch_selected(tree, "/data/cnp0002646", include=["*.gef"])
sel["bytes_human"], sel["fits"]
for cmd in sel["commands"]:
    print(cmd)          # wget -c -x -nH -P /data/cnp0002646 'ftp://…/chip.gef'
```

One resumable command per file, keeping the directory layout. Nothing is run.

## 3. Or open the GEF where it sits

```python
v = g.open_remote_gef("ftp://ftp.cngb.org/…/SS200000135TL_D1.gef", budget=b)
v.gef.bins, v.gef.extent, v.gef.genes(1)[:10]     # metadata only
v.gef.read_window(v.gef.extent, 20, ["Vegfa"])     # one contiguous gene slice
```

`RemoteFile` is a seekable file over FTP `REST` offsets or HTTP `Range` requests, with a block
cache that is bounded and reserved against the budget. HDF5 is built for random access and h5py
accepts any seekable file-like object, so a GEF opens on the server.

| step, on a 25 MB GEF over FTP | network cost |
|---|---|
| open: bins, extent, 303-gene index | 7.0 MB in 7 requests |
| a 0.25 % window at bin1, no index | +13.0 MB — the whole level |
| build a local spatial index for bin1 | +15.0 MB, one sequential pass |
| the same window afterwards | **0 B**, 10 ms, identical to a local read |

:::{admonition} Where the savings are, stated plainly
:class: measured
Opening costs roughly a constant — HDF5 metadata is scattered through the file and each touch
pulls one block — so it is a large share of a small GEF and a tiny share of a multi-gigabyte one.
Fine spatial work on one chip still reads most of that chip's GEF once, to build its index. The
saving is against the *project*, not against the file: one chip's levels rather than terabytes
of FASTQ.
:::

The sidecar for a remote GEF lives in `~/.geminga/gefidx/`, keyed by URL and size, and is reused
on the next open.

## Costs that belong to the protocol

Bytes are counted as **received**. FTP's `RETR` takes a start offset but no end, so the server
keeps sending until the client hangs up, and up to one TCP window per request is sent and
discarded. HTTP `Range` has no such overshoot — prefer an HTTP mirror when one exists.

Every miss is one request, and over FTP each request opens a new data connection, so latency
dominates on a distant server. The default 1 MB block with read-ahead on sequential access is a
compromise; raise `block=` on a high-latency link.

## If this sandbox is where you run it

The environment these docs were built in could not reach `ftp.cngb.org` — its egress allowlist
blocks the host — so the numbers above come from a local FTP server laid out like the CNGB mirror,
with the raw FASTQs as sparse files. Add the host to your network settings and run
`python examples/remote_cngb.py ftp://ftp.cngb.org/pub/CNSA/data2/CNP0002646/CNS0513001/`
to repeat it against the real project.
