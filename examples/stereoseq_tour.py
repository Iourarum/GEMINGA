"""Stereo-seq end to end: pyramid zoom, gene pushdown, multimodal pairing.

    python examples/make_stereoseq_fixtures.py fixtures
    python examples/stereoseq_tour.py fixtures
"""
import os
import sys
import time
import geminga as g

D = sys.argv[1] if len(sys.argv) > 1 else "fixtures"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
GEF, GEM = os.path.join(D, "chip.gef"), os.path.join(D, "chip.gem")
IMG = os.path.join(D, "chip_ssdna.tif")
B = g.Budget(ram="512MB")

print("1. the GEF bin pyramid is the zoom stack — SAW already computed every level")
v = g.open_gef(GEF, budget=B).reset()
print(f"   {v.gef}")
for z in range(len(v.gef.bins)):
    v.set_zoom(z); c = v.cost()
    print(f"   zoom {z} -> bin{c['bin_size']:<4} {c['units']:>10,} obs  {c['resident_human']:>9}")

print("\n2. ask before you load; the advice names the bin")
v.reset().use_bin(1)
small = g.open_gef(GEF, budget=g.Budget(ram="2MB")).reset().use_bin(1)
print("  ", small.cost()["advice"])
small.use_bin(None).auto_bin()
print(f"   auto_bin -> bin{small.cost()['bin_size']}, {small.cost()['resident_human']}")

print("\n3. one gene is one contiguous slice, not a scan")
v.reset().use_bin(10)
t0 = time.perf_counter(); allg = v.select_genes(None).load(); t_all = time.perf_counter() - t0
top = v.gene_profile(1)[0][0]
t0 = time.perf_counter(); one = v.select_genes([top]).load(); t_one = time.perf_counter() - t0
print(f"   all genes: {allg['x'].size:>9,} obs in {t_all:.2f}s")
print(f"   {top:<9}: {one['x'].size:>9,} obs in {t_one:.3f}s "
      f"({t_all/max(t_one,1e-6):.0f}x faster)")

print("\n4. GEM gets the same navigation from an index we build once")
t0 = time.perf_counter(); gv = g.open_gem(GEM, budget=B)
print(f"   indexed {gv.index.n_rows:,} rows in {time.perf_counter()-t0:.1f}s; "
      f"top genes {[n for n, _ in gv.index.top_genes(3)]}")
gv.select_frac(0.5, 0.8, 0.5, 0.8); g.print_cost(gv)

print("\n5. expression and morphology share one window")
ds = g.open_stereoseq(gef=GEF, image=IMG, budget=B,
                      image_scale=2048 / (v.gef.extent[1] - v.gef.extent[0]))
w = (1800, 3000, 1600, 2800)
pair = ds.paired(w, shape=384)
print(f"   expression {pair['expression'].shape} at bin{pair['bin_size']}; "
      f"morphology {pair['morphology'].shape}")

print("\n6. figures")
print("  ", g.render.chip(ds, window=w, path=os.path.join(OUT, "stereoseq_chip.png"),
                          genes=[top]))
ds.close()
