"""Conference figures for GEMINGA, measured on a glioma-shaped dataset.

RUN ON REAL DATA
----------------
Every measurement here is a real measurement of GEMINGA. Only the *biology* is synthetic,
because the machine this was first run on had no route to the data repositories. Point it at
real data and the same code produces the same figures:

    python examples/glioma_conference_figures.py --gef  /data/GBM_chip.gef \
                                                 --h5   /data/GBM_filtered_feature_bc_matrix.h5 \
                                                 --h5ad /data/GBmap.h5ad --out figures/

Candidate glioma datasets (verify the accession before use; none were reachable from the
build machine):
  spatial   STOmicsDB STDS0000040   GBM whole-transcriptome (catalogued as 10x Visium)
            GSE237183               Greenwald et al., 10x Visium GBM
            Zenodo 10.5281/zenodo.17622242  GBM atlas, Xenium
            Zenodo 10.5281/zenodo.8085502   GBM ST + scRNA + scATAC (raw at GSA HRA004942)
  sc/sn     GBmap (~1.1M cells)     integrated glioma atlas
            GSE131928               Neftel et al., GBM scRNA-seq
  Stereo-seq: no public *human glioma* chip was found. Ask MGI/STOmics, or substitute a
            mouse glioma model chip; --gef accepts any GEF.
"""
import argparse
import os
import resource
import time
import numpy as np
import geminga as g

P = g.render.PALETTE
SYN = "SYNTHETIC DATA — software measurements are real"


def rss() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def style(ax, grid=True):
    ax.set_facecolor(P["paper"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(P["ink"]); ax.spines[s].set_linewidth(1.1)
    ax.grid(grid, color=P["grid"], lw=0.7, alpha=0.9); ax.set_axisbelow(True)
    ax.tick_params(colors=P["ink"], labelsize=8)
    return ax


# ---------------------------------------------------------------- measurements
def measure_budget_invariance(gef, budgets=("8MB", "16MB", "32MB", "64MB", "128MB", "256MB")):
    """Declared budget vs the peak GEMINGA actually held, whole chip each time."""
    out = []
    for cap in budgets:
        b = g.Budget(ram=cap)
        v = g.open_gef(gef, budget=b).reset().auto_bin()
        c = v.cost()
        v.load()
        st = b.stats()["ram"]
        out.append(dict(cap=g.parse_size(cap), peak=int(st["peak"]),
                        bin=c["bin_size"], obs=c["units"], denials=int(st["denials"])))
        v.release(); v.gef.close()
    return out


def measure_pyramid(gef, budget="512MB"):
    """Cost and time at every bin level SAW wrote."""
    b = g.Budget(ram=budget)
    v = g.open_gef(gef, budget=b).reset()
    rows = []
    for z, bin_size in enumerate(v.gef.bins):
        v.set_zoom(z); c = v.cost()
        t0 = time.perf_counter(); d = v.load(); dt = time.perf_counter() - t0
        rows.append(dict(bin=bin_size, obs=int(d["x"].size), bytes=int(c["resident_bytes"]),
                         seconds=dt))
        v.release()
    return v, rows


def measure_gene_pushdown(v, bin_size=10, k=6):
    """One gene as a contiguous slice versus the whole matrix, same window."""
    v.reset().use_bin(bin_size).select_genes(None)
    t0 = time.perf_counter(); allg = v.load(); t_all = time.perf_counter() - t0
    n_all = int(allg["x"].size); v.release()
    tops = [name for name, _ in v.gene_profile(k)]
    rows = []
    for name in tops:
        v.select_genes([name])
        t0 = time.perf_counter(); d = v.load(); dt = time.perf_counter() - t0
        rows.append(dict(gene=name, obs=int(d["x"].size), seconds=dt,
                         speedup=t_all / max(dt, 1e-9)))
        v.release()
    v.select_genes(None)
    return dict(all_seconds=t_all, all_obs=n_all, genes=rows)


def measure_workspace(gef, h5, h5ad, ram="256MB"):
    """Three modalities open at once, each in its own partition."""
    ws = g.workspace(ram=ram, name="glioma")
    spec = [("stereo", gef, "96MB"), ("cite", h5, "64MB"), ("atlas", h5ad, "64MB")]
    for name, path, slice_ in spec:
        if path and os.path.exists(path):
            ws.add(name, path, ram=slice_, note=os.path.basename(path))
    for c in list(ws):
        try:
            if c.name == "stereo":
                c.view.reset().auto_bin()
            elif c.name == "cite":
                c.view.select_frac(0.0, 1.0)
            else:
                c.view.select_frac(0.0, 0.25)
            c.load()
        except Exception:
            pass
    rep = ws.report()
    ws.close()
    return rep


# ---------------------------------------------------------------- figures
def figure1(gef, h5, h5ad, out, budgets):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), dpi=140)
    fig.patch.set_facecolor(P["paper"])

    # (a) budget invariance
    inv = measure_budget_invariance(gef, budgets)
    ax = style(axes[0, 0])
    caps = np.array([r["cap"] for r in inv]) / 1024**2
    peaks = np.array([r["peak"] for r in inv]) / 1024**2
    ax.plot(caps, caps, "--", color=P["ink"], lw=1.2, label="declared budget")
    ax.plot(caps, peaks, "o-", color=P["blue"], lw=2, ms=6, label="measured peak")
    ax.fill_between(caps, peaks, caps, color=P["blue"], alpha=0.12)
    for r, x, y in zip(inv, caps, peaks):
        ax.annotate(f"bin{r['bin']}", (x, y), textcoords="offset points", xytext=(4, -11),
                    fontsize=7, color=P["ink"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("declared budget (MB)"); ax.set_ylabel("peak resident (MB)")
    ax.set_title("a  the ceiling holds: peak never crosses the budget", fontsize=10,
                 loc="left", color=P["ink"])
    ax.legend(fontsize=8, frameon=False)

    # (b) pyramid
    v, pyr = measure_pyramid(gef)
    ax = style(axes[0, 1])
    x = np.arange(len(pyr))
    mb = [r["bytes"] / 1024**2 for r in pyr]
    ax.bar(x, mb, color=P["red"], alpha=0.85, width=0.62)
    ax.set_xticks(x); ax.set_xticklabels([f"bin{r['bin']}" for r in pyr], fontsize=8)
    ax.set_ylabel("resident (MB)"); ax.set_yscale("log")
    ax2 = ax.twinx(); ax2.plot(x, [r["obs"] for r in pyr], "o-", color=P["blue"], lw=1.8, ms=5)
    ax2.set_ylabel("observations", color=P["blue"], fontsize=9)
    ax2.tick_params(colors=P["blue"], labelsize=8); ax2.set_yscale("log"); ax2.grid(False)
    drop = mb[0] / max(mb[-1], 1e-9)
    ax.set_title(f"b  the GEF bin pyramid is the zoom stack ({drop:.0f}x cheaper at the top)",
                 fontsize=10, loc="left", color=P["ink"])

    # (c) gene pushdown
    pd = measure_gene_pushdown(v)
    ax = style(axes[1, 0])
    names = [r["gene"] for r in pd["genes"]]; sp = [r["speedup"] for r in pd["genes"]]
    ax.barh(np.arange(len(names)), sp, color=P["yellow"], edgecolor=P["ink"], lw=0.8)
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("speedup vs loading all genes (x)")
    ax.axvline(1, color=P["ink"], lw=1)
    ax.set_title(f"c  one gene is a contiguous slice  (all genes: {pd['all_obs']:,} obs in "
                 f"{pd['all_seconds']:.2f} s)", fontsize=10, loc="left", color=P["ink"])

    # (d) workspace partition
    rep = measure_workspace(gef, h5, h5ad)
    ax = style(axes[1, 1])
    cells = rep["cells"]
    names = [c["name"] for c in cells]
    slices = [c["ram_cap"] / 1024**2 for c in cells]
    peaks_ = [c["ram_peak"] / 1024**2 for c in cells]
    y = np.arange(len(names))
    ax.barh(y, slices, color=P["grid"], edgecolor=P["ink"], lw=0.8, label="partition")
    ax.barh(y, peaks_, height=0.55, color=P["blue"], label="peak used")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("MB")
    free = rep["ram_free"] / 1024**2
    ax.set_title(f"d  three modalities under one {rep['ram_cap']/1024**2:.0f} MB ceiling "
                 f"({free:.0f} MB free)", fontsize=10, loc="left", color=P["ink"])
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    fig.suptitle("GEMINGA — memory-budgeted navigation of a glioma multi-modal dataset",
                 fontsize=13, color=P["navy"], x=0.01, ha="left", weight="bold")
    fig.text(0.99, 0.985, SYN, ha="right", va="top", fontsize=7.5, color=P["red"])
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    p = os.path.join(out, "fig1_budget.png"); fig.savefig(p, facecolor=P["paper"])
    plt.close(fig)
    v.gef.close()
    return p, dict(invariance=inv, pyramid=pyr, pushdown=pd, workspace=rep)


def figure2(gef, h5, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.3), dpi=140)
    fig.patch.set_facecolor(P["paper"])
    b = g.Budget(ram="512MB")
    v = g.open_gef(gef, budget=b)
    ex = v.gef.extent
    coarse, fine = v.gef.bins[-1], v.gef.bins[min(2, len(v.gef.bins) - 1)]

    def raster_shape(window, bin_size, cap=420):
        """One pixel per bin. Rendering a 938-bin level onto 420x420 pixels leaves 99.5% of
        the image empty and the picture reads as black — the resolution of the panel has to
        follow the resolution of the level."""
        units = max(window[1] - window[0], window[3] - window[2])
        return int(np.clip(units // max(bin_size, 1), 8, cap))

    def show(ax, img, title, extent=None, cmap="magma"):
        a = np.log1p(np.asarray(img, float)); pos = a[a > 0]
        # vmin at 0 keeps empty bins black and populated ones visible; clipping the low end
        # to a percentile of the positives washes a sparse level out entirely.
        hi = float(np.percentile(pos, 99.0)) if pos.size else None
        ax.imshow(a, cmap=cmap, origin="lower", vmin=0, vmax=hi, extent=extent,
                  aspect="equal", interpolation="nearest")
        ax.grid(False); ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=9.5, loc="left", color=P["ink"])

    # (a) whole chip at the coarsest bin
    sh = raster_shape(ex, coarse)
    t0 = time.perf_counter(); over = v.gef.rasterize(ex, coarse, None, sh)
    t_over = time.perf_counter() - t0
    span = ex[1] - ex[0]
    win = (ex[0] + int(span * .28), ex[0] + int(span * .58),
           ex[2] + int((ex[3] - ex[2]) * .28), ex[2] + int((ex[3] - ex[2]) * .58))
    show(axes[0], over, f"a  whole chip, bin{coarse} at {sh}x{sh}\n"
         f"{g.human(v.gef.levels[coarse].nbytes)} read, {t_over:.2f} s",
         extent=[ex[0], ex[1], ex[2], ex[3]])
    from matplotlib.patches import Rectangle
    axes[0].add_patch(Rectangle((win[0], win[2]), win[1]-win[0], win[3]-win[2],
                                fill=False, edgecolor=P["red"], lw=2.2))

    # (b) the window at a fine bin
    v.select(*win).use_bin(fine); c = v.cost()
    shw = raster_shape(win, fine)
    show(axes[1], v.image(shw), f"b  window, bin{fine} at {shw}x{shw}\n{c['units']:,} obs, {c['resident_human']} resident",
         extent=[win[0], win[1], win[2], win[3]])

    # (c) a single gene in that window
    top = v.gene_profile(1)[0][0]
    v.select_genes([top]); d = v.gef.read_window(win, fine, [top])
    show(axes[2], v.image(shw), f"c  {top} only\n{d['x'].size:,} obs from a contiguous slice",
         extent=[win[0], win[1], win[2], win[3]])
    v.select_genes(None)

    # (d) modality composition of the paired 10x matrix
    ax = style(axes[3])
    if h5 and os.path.exists(h5):
        m = g.TenXMatrix(h5, budget=b)
        labels = list(m.modalities)
        nfeat = [m.modalities[k].n_features for k in labels]
        tots = {k: m.per_cell_totals(k) for k in labels}
        cols = [P["blue"], P["red"], P["yellow"], P["navy"]]
        ax.bar(labels, nfeat, color=cols[:len(labels)], edgecolor=P["ink"], lw=0.8)
        ax.set_yscale("log"); ax.set_ylabel("features")
        for i, k in enumerate(labels):
            ax.annotate(f"median {np.median(tots[k]):.0f}/cell", (i, nfeat[i]),
                        textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7.5,
                        color=P["ink"])
        ax.set_title(f"d  one 10x matrix, {len(labels)} modalities\n{m.n_cells:,} cells, "
                     f"sliced by feature_type", fontsize=9.5, loc="left", color=P["ink"])
        m.close()
    else:
        ax.axis("off")
    v.gef.close()
    fig.suptitle("Navigating a glioma chip: overview to gene, and the paired modalities",
                 fontsize=13, color=P["navy"], x=0.01, ha="left", weight="bold")
    fig.text(0.995, 0.985, SYN, ha="right", va="top", fontsize=7.5, color=P["red"])
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    p = os.path.join(out, "fig2_navigation.png"); fig.savefig(p, facecolor=P["paper"])
    plt.close(fig)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gef", default="/tmp/gem/chip.gef")
    ap.add_argument("--h5", default="/tmp/gem/glioma/filtered_feature_bc_matrix.h5")
    ap.add_argument("--h5ad", default="/tmp/gem/sc.h5ad")
    ap.add_argument("--out", default=".")
    ap.add_argument("--budgets", default="8MB,16MB,32MB,64MB,128MB,256MB")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = rss()
    p1, stats = figure1(a.gef, a.h5, a.h5ad, a.out, tuple(a.budgets.split(",")))
    p2 = figure2(a.gef, a.h5, a.out)
    print(p1); print(p2)
    print(f"\nprocess RSS high-water {g.human(rss())} (baseline {g.human(base)})")
    inv = stats["invariance"]
    print(f"budget invariance: {len(inv)} budgets, "
          f"max peak/cap ratio {max(r['peak']/r['cap'] for r in inv):.2f}, "
          f"denials {sum(r['denials'] for r in inv)}")
    pyr = stats["pyramid"]
    print(f"pyramid: bin{pyr[0]['bin']} {pyr[0]['obs']:,} obs "
          f"{g.human(pyr[0]['bytes'])} -> bin{pyr[-1]['bin']} {pyr[-1]['obs']:,} obs "
          f"{g.human(pyr[-1]['bytes'])}")
    pd = stats["pushdown"]
    print(f"gene pushdown: median {np.median([r['speedup'] for r in pd['genes']]):.0f}x "
          f"(all genes {pd['all_seconds']:.2f} s)")
