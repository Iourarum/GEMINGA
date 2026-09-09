"""High-detail look at a spatial dataset from a spread sample: 2D density + per-gene view.

    python examples/visualize_sample.py /path/or/url --hub org/name config split
"""
import sys, numpy as np, geminga as md

if sys.argv[1] == "--hub":
    ds = md.open_hub(*sys.argv[2:5])
elif sys.argv[1].startswith("http"):
    ds = md.open_urls([sys.argv[1]])
else:
    ds = md.open_paths([sys.argv[1]])

tbl = md.sample_table(ds, n_row_groups=6, seed=1, columns=["x", "y", "count"])
x, y, c = (tbl[k].to_numpy() for k in ("x", "y", "count"))
print(f"{tbl.num_rows:,} sampled bins; x {x.min()}..{x.max()}, y {y.min()}..{y.max()}")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
h = ax[0].hexbin(x, y, C=c, gridsize=220, reduce_C_function=np.sum, mincnt=1, cmap="magma")
ax[0].set_title("UMI counts per hex, 6-row-group sample"); ax[0].set_aspect("equal")
fig.colorbar(h, ax=ax[0], label="summed count")
ax[1].hist2d(x, y, bins=300, cmap="viridis"); ax[1].set_aspect("equal"); ax[1].set_title("bin density")
fig.tight_layout(); fig.savefig("sample_overview.png", dpi=130)
print("wrote sample_overview.png")
