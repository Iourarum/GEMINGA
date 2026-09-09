"""Inspect a Hub dataset without downloading it: schema, sizes, per-column stats, a spread sample.

    python examples/inspect_hub.py HuggingFaceTB/smoltalk2 Mid train
"""
import sys, time
import geminga as md

dataset, config, split = (sys.argv[1:] + [None, None, None])[:3]
t0 = time.perf_counter()
ds = md.open_hub(dataset, config=config, split=split)   # one JSON call to datasets-server
print(f"resolved {ds.num_files} parquet file(s) in {time.perf_counter()-t0:.2f}s")
md.print_summary(ds)                                     # one footer range-read per file
t1 = time.perf_counter()
tbl = md.sample_table(ds, n_row_groups=4, seed=42)       # a few row groups, spread across files
print(f"\nsample: {tbl.num_rows:,} rows x {tbl.num_columns} cols in {time.perf_counter()-t1:.2f}s")
print(tbl.slice(0, 5).to_pandas())
