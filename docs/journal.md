# The journal — what was streamed, exactly where

Opening a Stereo-seq chip and looking around it reads a few hundred windows at a few bin sizes.
A week later the questions are: *which* windows, at which level, from which file, and did the
region in the figure ever get read at bin 1 or only at bin 50? The library and the selection
book answer for what you deliberately flagged. The journal answers for everything else: every
piece a reader streamed or a view investigated is written down as it happens, with its exact
coordinates, and the file is plain text.

```python
import geminga as g
v = g.open_gef("chip.gef", g.Budget(ram="4GB"))         # -> "open": extent, bin levels, gene list
v.select(1600, 3000, 1400, 2100).use_bin(50).load()      # -> "load": window, bin, rows, bytes
for blk in v.select(0, 4000, 0, 4000).use_bin(10).chunks():   # -> one "chunks" record at the end
    ...
v.gef.rasterize((0, 4000, 0, 4000), 200)                # -> "rasterize"
sel = g.select(v, "rim", intent="double-check")          # -> "select", carrying the selection id

g.journal().print_table()
```

```
2026-10-13 14:12:27 open        chip.gef   0,4000,0,4000      bins=[1, 10, 20, 50, 100, 200]  2,946,567 spots  33.3 MB  300 genes
2026-10-13 14:12:27 load        chip.gef   1600,3000,1400,2100  bin=50     106,371 spots    1.2 MB
2026-10-13 14:12:28 chunks      chip.gef   0,4000,0,4000      bin=10   2,946,567 spots   33.6 MB
2026-10-13 14:12:29 rasterize   chip.gef   0,4000,0,4000      bin=200      4,096 pixels   16.0 KB
2026-10-13 14:12:29 select      chip.gef   1600,3000,1400,2100  bin=50           0 spots      0 B
```

## What is written, and by whom

One JSON object per line in `~/.geminga/journal.jsonl` (schema `geminga.journal/1`):

```json
{"t": 1760000000.1, "op": "load",
 "dataset": {"path": "/data/chip.gef", "kind": "gef", "key": "962647ad33bf30be",
             "fingerprint": {"size": 34958244, "mtime": 1760000000, "version": 2}},
 "unit": "spots", "window": [1600, 3000, 1400, 2100], "level": {"bin": 50},
 "genes": null, "units": 106371, "bytes": 1276452, "seconds": 0.21,
 "via": "explorer", "selection": "", "note": "", "tags": ["frame=12"],
 "host": "lab-laptop", "pid": 4242, "session": "a1b2c3d4", "schema": "geminga.journal/1"}
```

| op | written by | window |
|---|---|---|
| `open` | the first time a file is opened against this journal | a GEF: the chip extent, `level` lists the bins, `genes` the gene list, `units` the bin-1 rows; an h5ad: `[0, n_obs]` and the var names as `genes`; a Parquet table: `[0, n_rows]` and its columns as `col=` tags |
| `load` | every `View.load()` — GEF, GEM, Parquet, h5ad, image, volume views alike | the view's window, `level` `{bin}` or `{zoom, stride}` |
| `chunks` | a `View.chunks()` stream when it ends (or is released early) | the window; `units`/`bytes` are the totals, `tags` has `chunks=N` |
| `overview` | `View.overview()` | the window; `units` is the bucket count |
| `read_window`, `rasterize` | `GefFile` direct calls | the window asked for; `rasterize` counts pixels |
| `select` | `g.select`, `select_polygon`, `select_values` | the pinned bounds; `selection` is the id |
| `similar` | `g.similar` — one record for the whole candidate scan | the selection's bounds; `tags` name the candidates |
| `note` | `journal.note(dataset, text)` | — |

Windows are exact and half-open in the dataset's own coordinates — chip units for a GEF, row
numbers for a table, record numbers for a FASTQ — the same convention as `View.window`. The
dataset key is a hash of the absolute path and the file size, with the fingerprint alongside,
so a record is never silently reattached to a different file of the same name.

`via` says what asked: the explorer tags its reloads with the frame they served, `describe()`
tags the reads it makes with the selection id, `Staging.extract` with the region key, `handoff`
with the ROI name and targets, the `Stage` with `stage`. Your own code can do the same:

```python
from geminga.journal import recording, quiet
with recording(via="figure-3", note="panel b"):
    data = v.load()                    # tagged
with quiet():
    v.load()                           # not written: an internal read you will record yourself
```

The journal never stops an analysis: a file that cannot be written warns once and recording
goes silent. `GEMINGA_JOURNAL=/path/file.jsonl` moves it, `GEMINGA_JOURNAL=0` turns it off,
`g.use_journal(path_or_None)` does the same from Python. The test suite journals into a
temporary file (`tests/conftest.py`), never into yours.

## Going back

```python
j = g.journal()
e = j.search(op="load", dataset="chip.gef", overlaps=(2000, 2100, 1800, 1900))[-1]
v = j.replay(e)          # the same file, bin and window, as a view ready to load()
```

`replay` refuses — `ValueError` — when the file's size no longer matches the fingerprint, and
`FileNotFoundError` when it is not on this machine; `check=False` opens it anyway.

`j.datasets()` folds the trail per dataset: pieces, bytes, levels seen, the union of all
windows looked at, how many sittings. `j.visited(dataset)` lists the windows; `j.coverage(dataset)`
counts, on a grid over the chip, how many times each part was streamed — the figure that shows
where nobody has looked yet.

```
$ python -m geminga.journal datasets
962647ad33bf30be chip.gef        gef    214 pieces  412.7 MB  2026-10-11…2026-10-13 [0,4000,0,4000]
$ python -m geminga.journal where 962647ad33bf30be
$ python -m geminga.journal coverage 962647ad33bf30be --shape 48
```

## Searching it: grep, ripgrep, DuckDB, or the trigram index

The file is JSONL so that the ordinary tools work on it first:

```bash
rg -F GFAP ~/.geminga/journal.jsonl              # every piece that involved the gene, every chip
rg -F '"op":"open"' ~/.geminga/journal.jsonl | jq .dataset.path     # every file ever opened
```

An `open` record carries the chip's gene list, so `rg GFAP` answers "which of the chips I ever
opened carried GFAP" without reopening any of them — including chips that were opened over FTP
and are no longer reachable.

From Python, `Journal.search` adds the predicates grep cannot express:

```python
j.search("GFAP")                                        # text, any field
j.search(overlaps=(1000, 2000, 1000, 2000), op="load")  # windows overlapping a region
j.search(within=(0, 4000, 0, 4000), dataset="chip.gef", since=time.time() - 86400)
j.search(selection=sel.id)                              # everything read while describing it
j.grep(r'"bin":(50|100)', regex=True)                   # raw lines, like rg -n
```

`search(text=...)` goes through the fastest engine present — `j.engine` says which: the
trigram index when one is built, else `rg`, else `grep`, else a Python scan. `j.duckdb()`
exposes the file as a DuckDB view for SQL.

### The trigram index

`j.index()` builds `<journal>.tgx` beside the file, in the Rust core, in the shape of
Microsoft's *tgrep*: the file is cut into blocks of 64 lines, every block's distinct
case-folded trigrams go into posting lists, a query's trigrams are intersected and only the
surviving blocks are read and scanned. Trigrams present in every block — a JSON log's keys —
are kept as a one-byte mark rather than a list, which is what keeps the index small. The index
covers a prefix of the file; `search()` scans whatever has been appended since, so it is never
wrong, only slower, until `j.index()` extends it (milliseconds for a day's records).

Measured on a synthetic journal of 1,000,000 records (468 MB; `examples/bench_journal.py`),
best of three, warm page cache, every tool printing or returning the matching lines:

| query | hits | grep -n | rg -n | Python scan | trigram | blocks read |
|---|---|---|---|---|---|---|
| a selection id, in 10 lines | 10 | 518 ms | 174 ms | 729 ms | **3 ms** | 10 / 15,625 |
| a gene, in ~1 % of records | 9,868 | 455 ms | 162 ms | 710 ms | **80 ms** | 7,316 / 15,625 |
| a term in every record | 888,888 | 1,227 ms | **1,194 ms** | 779 ms* | 1,649 ms | all |
| two characters (cannot narrow) | 749,141 | 1,881 ms | **1,215 ms** | 859 ms* | 1,467 ms | all |

\* the Python scan only counts; it does not hand the lines back. Building the index took 8 s;
the sidecar is 16 MB (3 % of the journal); appending 10,000 records and extending it took 37 ms.
`Journal.search("GFAP", overlaps=...)` — the gene *and* a window — answered in 191 ms through
the index against 640 ms for the same question in DuckDB over `read_json_auto`.

The shape of the result is what matters: the index wins by two orders of magnitude on rare
terms, by two on uncommon ones, and loses by a third to ripgrep on terms that occur in every
record, where it has to scan every block anyway and hands 900,000 lines back to Python.

So, grep or tgrep or faster? For a journal that fits in the page cache, `rg` already answers
in tens of milliseconds and the index is a convenience. The index is the right tool when the
journal is large (a lab's shared journal of years), on a cold disk, over a network mount, or
when the question is asked repeatedly — search-as-you-type in the explorer's HUD — and for the
`open` records' gene lists, where a gene name is exactly the rare term the index is built for.

## Schema

`geminga.journal.Entry.SCHEMA` lists every field with its JSON type; `JOURNAL_SCHEMA` is
bumped only on an incompatible change, and additive fields keep the number. The catalog
(`geminga.catalog/1`) records *datasets*; the selection book and library record *flags*; the
journal records *reads*. The three link through the dataset key and the selection id.
