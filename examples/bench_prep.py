"""The standard pipeline on an atlas-sized matrix, under a budget — the numbers in docs/prep.md.

    python examples/bench_prep.py [n_cells] [budget] [work_dir]

Writes an .h5ad of `n_cells` × 20,000 genes (~1.5% dense, four planted cell types with marker
genes, 13 mitochondrial genes) block by block — the whole matrix is never in memory — and runs
QC → filters → HVG → PCA under the budget, reporting time per pass, the budget's peak, and the
process's real peak RSS.
"""
import os
import resource
import sys
import time

import h5py
import numpy as np

import geminga as g

n_obs = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
cap = sys.argv[2] if len(sys.argv) > 2 else "512MB"
R = sys.argv[3] if len(sys.argv) > 3 else "/tmp/gem/bench"
os.makedirs(R, exist_ok=True)
n_vars, blk, density = 20_000, 10_000, 0.015
P = os.path.join(R, f"atlas_{n_obs // 1000}k.h5ad")

if not os.path.exists(P):
    rng = np.random.default_rng(0)
    t0 = time.time()
    with h5py.File(P, "w") as f:
        f.attrs["encoding-type"] = "anndata"
        x = f.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["shape"] = np.array([n_obs, n_vars])
        dd = x.create_dataset("data", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=(1 << 20,))
        di = x.create_dataset("indices", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(1 << 20,))
        indptr = [0]
        for s0 in range(0, n_obs, blk):
            rows = min(blk, n_obs - s0)
            per_row = rng.binomial(n_vars, density, rows)                  # background nnz per cell
            lab = rng.integers(0, 4, rows)
            marker = rng.binomial(50, 0.6, rows)                            # marker nnz per cell (type block)
            data_blk, idx_blk = [], []
            for r in range(rows):
                cols = rng.choice(n_vars, per_row[r], replace=False)
                vals = np.round(rng.exponential(1.5, per_row[r])).astype(np.float32) + 1
                mcols = lab[r] * 50 + rng.choice(50, marker[r], replace=False)
                mvals = np.round(rng.exponential(6.0, marker[r])).astype(np.float32) + 1
                c = np.concatenate([cols, mcols])
                v = np.concatenate([vals, mvals])
                c, inv = np.unique(c, return_inverse=True)
                v = np.bincount(inv, weights=v).astype(np.float32)
                data_blk.append(v)
                idx_blk.append(c.astype(np.int32))
                indptr.append(indptr[-1] + c.size)
            d = np.concatenate(data_blk)
            i = np.concatenate(idx_blk)
            n0 = dd.shape[0]
            dd.resize((n0 + d.size,))
            di.resize((n0 + i.size,))
            dd[n0:] = d
            di[n0:] = i
            print(f"  wrote cells {s0:,}–{s0 + rows:,}  nnz so far {dd.shape[0]:,}  {time.time() - t0:.0f}s", flush=True)
        x.create_dataset("indptr", data=np.array(indptr, np.int64))
        obs = f.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=np.array([f"C{i}" for i in range(n_obs)]).astype("S"))
        var = f.create_group("var")
        var.attrs["_index"] = "_index"
        names = [f"MT-{i}" if i >= n_vars - 13 else f"G{i}" for i in range(n_vars)]
        var.create_dataset("_index", data=np.array(names).astype("S"))
    print(f"wrote {P}: {os.path.getsize(P) / 2**30:.2f} GB in {time.time() - t0:.0f}s")

b = g.Budget(ram=cap)
t0 = time.time()
res = g.prep(P, budget=b).run(n_top=2000, n_comps=50, min_genes=200, max_pct_mito=20, min_cells=3,
                              out=os.path.join(R, "X_pca.npy"))
dt = time.time() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
print(f"{res}\n  file {os.path.getsize(P) / 2**30:.2f} GB · {dt:.0f} s total {res.report['seconds']}\n"
      f"  budget {cap}: peak {b.stats()['ram']['peak_human']}, denials {b.stats()['ram']['denials']}, "
      f"grants {b.stats()['ram']['grants']} · process peak RSS {rss:.2f} GB\n"
      f"  explained variance ratio, first 5: {np.round(res.explained_variance_ratio[:5], 4)}")
