"""Feed a remote dataset into a training loop without holding it — or materialize a slice locally.

    python examples/stream_to_training.py HuggingFaceTB/smoltalk2 Mid train
"""
import sys, time
import geminga as md

dataset, config, split = (sys.argv[1:] + [None, None, None])[:3]
ds = md.open_hub(dataset, config=config, split=split)
schema = ds.schema()
cols = [f.name for f in schema][:3]          # project: only the columns the model needs
print("streaming columns", cols)

# 1) online: iterate batches straight into a loop (GIL released while the next batch loads)
rows, t0 = 0, time.perf_counter()
for i, batch in enumerate(ds.stream(columns=cols, batch_size=32_768, limit=200_000)):
    rows += batch.num_rows                     # batch is a pyarrow.RecordBatch (zero-copy)
    # tensors = {k: torch.from_numpy(v) for k, v in ...}  # see md.iter_torch
print(f"streamed {rows:,} rows in {i+1} batches, {time.perf_counter()-t0:.2f}s")

# 2) offline: keep a reproducible shard on disk for the training set
info = ds.materialize("subset.parquet", columns=cols, limit=50_000)
print("materialized", info)
