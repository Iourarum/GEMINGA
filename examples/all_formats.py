"""End-to-end self-test: every format streamed under one 256 MB budget, with the real
resident high-water mark measured by the OS (RSS), not just claimed.

    python examples/make_omics_fixtures.py /tmp/gem
    python examples/all_formats.py /tmp/gem
"""
import os, resource, sys, time
import numpy as np
import geminga as g

D = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gem"
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
base = rss()
B = g.Budget(ram="256MB", vram="2GB", spill="10GB", timeout_s=30)
print(f"budget: {B}\nbaseline RSS {g.human(base)}\n")

def band(name): print(f"--- {name} " + "-" * (58 - len(name)))

# 1. FASTQ, plain and gzipped, parsed in Rust
for f in ("reads.fastq", "reads.fastq.gz"):
    band(f)
    p = os.path.join(D, f); size = os.path.getsize(p)
    st = g.open(p, budget=B, chunk_bytes="16MB")
    t0 = time.perf_counter(); n = rows = 0; peak = 0
    for batch in st:
        n += 1; rows += batch.num_rows
        peak = max(peak, B.stats()["ram"]["used"])
    dt = time.perf_counter() - t0
    print(f"  {rows:,} reads in {n} chunks, {dt:.2f}s ({size/dt/1e6:.0f} MB/s), "
          f"quality={st.has_quality}, peak resident {g.human(peak)} of {g.human(size)} file")

# 2. FASTA
band("transcripts.fasta")
st = g.open(f"{D}/transcripts.fasta", budget=B, chunk_bytes="4MB")
tot = 0; n = 0
for b in st:
    n += 1; tot += b.num_rows
print(f"  {tot:,} sequences in {n} chunks, quality={st.has_quality}, cols={st.schema().names}")

# 3. h5ad, CSR row blocks
band("sc.h5ad")
r = g.open(f"{D}/sc.h5ad", budget=B, chunk_bytes="8MB", obs_columns=["batch"])
print("  ", r.plan())
cells = 0; nnz = 0; peak = 0; sums = []
for ch in r:
    cells += ch["X"].shape[0]; nnz += ch["X"].nnz
    sums.append(np.asarray(ch["X"].sum(1)).ravel())
    peak = max(peak, B.stats()["ram"]["used"])
r.close()
print(f"  streamed {cells:,} cells, {nnz:,} nonzeros, peak resident {g.human(peak)}; "
      f"mean UMI/cell {np.concatenate(sums).mean():.1f}")

# 4. Arrow IPC
band("qc.arrow")
r = g.open(f"{D}/qc.arrow", budget=B)
print("  ", r.plan())
rows = sum(b.num_rows for b in r); r.close()
print(f"  {rows:,} rows streamed")

# 5. Imaging: tile, "segment", write masks out incrementally
band("tissue.ome.tif")
r = g.open(f"{D}/tissue.ome.tif", budget=B, chunk_bytes="4MB", overlap=64)
plan = r.plan(); print("  ", plan)
sink = g.MaskSink(f"{D}/masks.zarr", shape=(r.h, r.w))
peak = 0; t0 = time.perf_counter()
for tile, win in r:
    # stand-in for Mask R-CNN / cellpose: threshold + connected components
    from scipy import ndimage
    lab, _ = ndimage.label(tile > 3000)
    sink.write(lab, win)
    peak = max(peak, B.stats()["ram"]["used"])
r.close()
print(f"  {plan.n_chunks} tiles of {r.tile}px in {time.perf_counter()-t0:.2f}s, "
      f"{sink.n_objects:,} objects -> {f'{D}/masks.zarr'}, peak resident {g.human(peak)}")

# 6. Parquet, chunk size derived from the budget
band("parquet (budget-derived chunking)")
pq = "/home/claude/test_bins.parquet"
if os.path.exists(pq):
    ds = g.open(pq, budget=B)
    g.print_plan(ds, chunk_bytes="8MB")
    rows = 0; peak = 0
    for b in ds.stream(columns=["x", "y"], chunk_bytes="8MB", limit=500_000):
        rows += b.num_rows; peak = max(peak, B.stats()["ram"]["used"])
    print(f"  streamed {rows:,} rows, peak resident {g.human(peak)}")

print(f"\n=== budget after everything ===")
for tier, s in B.stats().items():
    if s["capacity"]:
        print(f"  {tier:<6} peak {s['peak_human']:>10}  of {s['capacity_human']:>10}  "
              f"grants={s['grants']:<5} waits={s['waits']} denials={s['denials']}")
print(f"process RSS high-water: {g.human(rss())} (baseline was {g.human(base)}); "
      f"largest input on disk: {g.human(max(os.path.getsize(os.path.join(D,f)) for f in os.listdir(D) if os.path.isfile(os.path.join(D,f))))}")
