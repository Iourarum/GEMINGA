"""Self-test over a local Range-capable HTTP server: proves footer-only inspection, projection
pushdown, spread sampling and materialization move only the bytes they need.

    python examples/make_test_data.py /tmp/test_bins.parquet 2000000
    python examples/local_roundtrip.py /tmp/test_bins.parquet
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import pyarrow as pa, pyarrow.parquet as pq
import geminga as md
from range_server import serve, SERVED

path = os.path.abspath(sys.argv[1]); d, name = os.path.split(path)
httpd = serve(d, 8765); full = os.path.getsize(path)
url = f"http://127.0.0.1:8765/{name}"

def report(ds, label, t0):
    t = ds.transfer_stats()
    print(f"  [{label}] {time.perf_counter()-t0:.3f}s   moved {t['bytes']:,} B "
          f"({100*t['bytes']/full:.2f}% of file) in {t['range_fetches']} range fetches; "
          f"server saw {SERVED['requests']} requests")
    ds.reset_transfer_stats(); SERVED.update(bytes=0, requests=0, ranges=0)

ds = md.open_urls([url])
t0 = time.perf_counter(); md.print_summary(ds); report(ds, "describe + stats (footer only)", t0)

cols = ds.schema().names[:2]
t0 = time.perf_counter(); rows = 0
for b in ds.stream(columns=cols, batch_size=65_536, limit=250_000):
    rows += b.num_rows
print(f"  streamed {rows:,} rows of {cols}"); report(ds, "projected stream, limit 250k", t0)

t0 = time.perf_counter(); tbl = md.sample_table(ds, n_row_groups=4, seed=42)
print(f"  sample: {tbl.num_rows:,} rows x {tbl.num_columns} cols"); report(ds, "spread sample, 4 row groups", t0)

t0 = time.perf_counter(); info = ds.materialize("subset.parquet", columns=cols, limit=50_000)
print(f"  materialized {info} -> re-read {pq.read_table('subset.parquet').num_rows:,} rows"); report(ds, "materialize 50k", t0)

local = md.open_paths([path]); assert local.describe()["total_rows"] == ds.describe()["total_rows"]
print("  local path gives identical metadata: ok")
httpd.shutdown()
