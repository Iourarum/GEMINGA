"""What compression buys a budget — the numbers in docs/codec.md.

    python examples/bench_codec.py [h5ad]

Packs the arrays a count matrix is made of, times the reference codec on the host, and streams
a matrix through a CompressedStore to show how much of it a fixed tier now holds. Defaults to
the sc.h5ad fixture; give a real .h5ad to see your own ratios.
"""
import os
import sys
import time

import numpy as np

import geminga as g
from geminga import codec as C

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(FIX, "sc.h5ad")
rng = np.random.default_rng(0)
bp = C.BitpackCodec()


def t(label, f, nbytes):
    t0 = time.perf_counter()
    out = f()
    dt = time.perf_counter() - t0
    print(f"  {label:<44} {dt * 1000:8.1f} ms  {nbytes / dt / 2**20:8.0f} MB/s")
    return out


print("synthetic arrays (1M elements each)")
cases = {
    "UMI counts, Poisson(2)+1 (int32)": rng.poisson(2.0, 1_000_000).astype(np.int32) + 1,
    "sorted column indices, 30k genes (int32)": np.sort(rng.integers(0, 30000, 1_000_000)).astype(np.int32),
    "CSR indices, ~300 nnz per row (int32)": np.concatenate(
        [np.sort(rng.integers(0, 30000, rng.integers(100, 500))) for _ in range(3400)])[:1_000_000].astype(np.int32),
    "row lengths (int32)": rng.integers(100, 500, 1_000_000).astype(np.int32),
    "bin coordinates, Stereo-seq x (int32)": np.sort(rng.integers(0, 26000, 1_000_000)).astype(np.int32),
    "random int32 (incompressible)": rng.integers(0, 1 << 31, 1_000_000).astype(np.int32),
}
for name, a in cases.items():
    e = t(f"encode  {name}", lambda: bp.encode(a), a.nbytes)
    t(f"decode  {name}", lambda: bp.decode(e), a.nbytes)
    print(f"  {'':<44} ratio {e.ratio:5.2f}x   mean width {e.meta['mean_width']:4.1f} bits   "
          f"delta blocks {e.meta['delta_blocks'] / max(e.meta['n_blocks'], 1):.0%}")

if os.path.exists(path):
    print(f"\n{path}: streaming X into a compressed store")
    for cap in ("8MB", "32MB"):
        b = g.Budget(ram=cap)
        store = C.compressed_store(b, "ram")
        rd = g.H5adReader(path, budget=b)
        n = 0
        t0 = time.perf_counter()
        for i, blk in enumerate(rd):
            try:
                store.put(f"blk{i}", blk["X"])
            except MemoryError:
                break
            n += blk["obs_end"] - blk["obs_start"]
        rd.close()
        r = store.report()
        print(f"  budget {cap:>5}: {n:,} of {rd.n_obs:,} cells resident packed — "
              f"{r['raw_human']} of data in {r['packed_human']} ({r['ratio']:.1f}x), "
              f"{time.perf_counter() - t0:.2f} s, budget peak {b.stats()['ram']['peak_human']}")
        store.clear()
else:
    print(f"\n(no {path}; run examples/make_omics_fixtures.py for the fixture)")
