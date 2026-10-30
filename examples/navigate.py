"""Zoom, pan, search and draw — the v0.2 tour, on whatever you point it at.

    python examples/make_omics_fixtures.py fixtures
    python examples/make_adapter_fastq.py fixtures/reads_adapters.fastq
    python examples/navigate.py fixtures
"""
import os
import sys
import time
import numpy as np
import geminga as g

D = sys.argv[1] if len(sys.argv) > 1 else "fixtures"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
B = g.Budget(ram="256MB")
FQ = os.path.join(D, "reads_adapters.fastq")
ADAPTER = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"

print("1. index once — afterwards navigation costs nothing")
t0 = time.perf_counter(); idx = g.index_fastx(FQ)
sc = FQ + ".gemidx.json"
print(f"   {idx!r} in {time.perf_counter()-t0:.1f}s; sidecar is "
      f"{os.path.getsize(sc)/idx.file_bytes:.4%} of the file")

print("\n2. zoom like a map — the cost stays flat as you move")
v = g.view(FQ, budget=B).reset().set_zoom(4)
for _ in range(4):
    c = v.cost()
    print(f"   {str(c['window']):>22}  zoom {c['zoom']}  stride {c['stride']:>3}  "
          f"{c['units']:>7,} records  {c['total_human']:>9}")
    v.zoom_in(4, center=idx.n_records * 0.62)

print("\n3. ask before you load")
v.reset(); g.print_cost(v)
v.select_frac(0.55, 0.60); g.print_cost(v)

print("\n4. search the whole file, then localise")
v.reset()
t0 = time.perf_counter(); hits = v.search([ADAPTER[:20]], max_hits=500_000)
dt = time.perf_counter() - t0
print(f"   {hits['n_hits']:,} hits in {hits['records_matched']:,}/{hits['records_scanned']:,} "
      f"records, {hits['bases_scanned']/1e6:.0f} Mbase in {dt:.2f}s "
      f"({hits['bases_scanned']/dt/1e6:.0f} Mbase/s)")
print("   density:", v.hit_profile(hits, 60).sparkline("hits"))
rec = np.array([h["record"] for h in hits["hits"]])
if rec.size:
    lo, hi = int(rec.min()), int(rec.max())
    print(f"   contamination spans records {lo:,}-{hi:,}")
    inside = v.select(lo, hi).search([ADAPTER[:20]], max_hits=500_000)
    print(f"   inside that band: {inside['records_matched']/inside['records_scanned']:.1%} "
          f"of reads carry the adapter")

print("\n5. figures")
v.select(lo, hi) if rec.size else v.reset()
print("  ", g.render.navigator(v, os.path.join(OUT, "nav_fastq.png"),
                               title="GEMINGA — FASTQ navigator"))
v.reset()
print("  ", g.render.hits(v, hits, os.path.join(OUT, "nav_hits.png"),
                          patterns=[ADAPTER[:20]]))

tif = os.path.join(D, "tissue.ome.tif")
if os.path.exists(tif):
    iv = g.view(tif, budget=B).select(2400, 3600, 2600, 3800)
    print("  ", g.render.navigator(iv, os.path.join(OUT, "nav_image.png"),
                                   title="GEMINGA — image navigator"))

h5 = os.path.join(D, "sc.h5ad")
if os.path.exists(h5):
    hv = g.view(h5, budget=B)
    print("\n6. h5ad: the overview is free — nnz per cell comes from indptr")
    ov = hv.overview(60)
    print("   genes/cell:", ov.sparkline("genes_detected"))
    hv.select_frac(0.3, 0.35); g.print_cost(hv)
    print("  ", g.render.navigator(hv, os.path.join(OUT, "nav_h5ad.png"),
                                   title="GEMINGA — h5ad navigator"))

pq = os.path.join(D, "spatial.parquet")
if not os.path.exists(pq):
    pq = "/home/claude/test_bins_sorted.parquet"
if os.path.exists(pq):
    print("\n7. predicate pushdown, then a picture built while streaming")
    ds = g.open(pq, budget=B)
    p = ds.prune("x", 12000, 12500, ["x", "y", "count"])
    print(f"   {p['row_groups_kept']}/{p['row_groups_total']} row groups, "
          f"{p['bytes_to_fetch_human']} of {p['bytes_total_human']} ({p['selectivity']:.1%})")
    canvas = g.ProgressiveCanvas.from_dataset(ds, "x", "y", shape=(400, 400))
    for bt in ds.stream(columns=["x", "y", "count"], chunk_bytes="8MB"):
        canvas.add_batch(bt, "x", "y", weight="count")
    print("  ", canvas)
    print("  ", g.render.progressive(canvas, os.path.join(OUT, "progressive.png")))
