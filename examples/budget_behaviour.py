"""The governor's three behaviours: adaptive chunking, backpressure, and fail-fast."""
import threading, time
import geminga as g

PQ = "/home/claude/test_bins.parquet"
print("=== 1. the same dataset under three budgets: chunking adapts, peak tracks the budget ===")
print(f"{'budget':>8} {'chunk':>8} {'rows/chunk':>11} {'chunks':>7} {'peak resident':>14} {'time':>7}")
for cap in ("32MB", "128MB", "1GB"):
    B = g.Budget(ram=cap)
    ds = g.open(PQ, budget=B)
    plan = ds.plan()
    t0 = time.perf_counter(); n = rows = 0
    for b in ds.stream():
        n += 1; rows += b.num_rows
    st = B.stats()["ram"]
    print(f"{cap:>8} {g.human(plan['chunk_bytes']):>8} {plan['rows_per_chunk']:>11,} "
          f"{n:>7} {st['peak_human']:>14} {time.perf_counter()-t0:>6.2f}s")

print("\n=== 2. backpressure: a holder blocks the reader until it releases ===")
B = g.Budget(ram="24MB", timeout_s=10)
hold = B.acquire("ram", "20MB")          # simulate a model or a mask buffer holding memory
print(f"  held 20MB; {g.human(B.available('ram'))} left of 24MB")
def release_later():
    time.sleep(1.5); hold.release(); print("  ...holder released after 1.5s")
threading.Thread(target=release_later, daemon=True).start()
ds = g.open(PQ, budget=B)
t0 = time.perf_counter()
first = next(iter(ds.stream(chunk_bytes="8MB")))
st = B.stats()["ram"]
print(f"  first chunk of {first.num_rows:,} rows arrived after {time.perf_counter()-t0:.2f}s "
      f"(waits={st['waits']}, waited {st['wait_millis']}ms) — it blocked instead of over-committing")

print("\n=== 3. fail fast: a chunk that can never fit is refused, with a fix in the message ===")
B = g.Budget(ram="4MB")
try:
    for _ in g.open(PQ, budget=B).stream(chunk_bytes="64MB"):
        pass
except MemoryError as e:
    print("  MemoryError:", str(e))

print("\n=== 4. VRAM tier: accounting for memory GEMINGA does not allocate itself ===")
B = g.Budget(ram="256MB", vram="8GB")
ds = g.open(PQ, budget=B)
moved = 0
for batch in ds.stream(columns=["x", "y"], chunk_bytes="16MB", limit=1_000_000):
    nbytes = batch.get_total_buffer_size()
    with B.acquire("vram", str(nbytes)):      # reserve before .to('cuda'), release after the step
        moved += nbytes
print(f"  {g.human(moved)} cycled through the VRAM tier; peak {B.stats()['vram']['peak_human']} "
      f"of 8.0 GB, denials={B.stats()['vram']['denials']}")
