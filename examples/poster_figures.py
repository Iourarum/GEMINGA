"""Poster figures for GEMINGA on a glioblastoma-structured section.

    python examples/make_glioma_phantom.py fixtures/gbm
    python examples/poster_figures.py --gef fixtures/gbm/gbm_phantom.gef --out poster/

Every software measurement here is real. The tissue is a designed phantom whose layout follows
glioblastoma anatomy (examples/make_glioma_phantom.py); point --gef at a real chip to replace it.
"""
import argparse
import os
import time
import numpy as np
import geminga as g

P = g.render.PALETTE
TAG = "glioma-structured phantom · software measurements are real"


def plt_setup():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold",
                         "axes.titlesize": 12, "axes.labelsize": 10.5,
                         "xtick.labelsize": 9, "ytick.labelsize": 9})
    return plt


def style(ax, grid=True):
    ax.set_facecolor(P["paper"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(P["ink"])
    ax.grid(grid, color=P["grid"], lw=0.7); ax.set_axisbelow(True)
    return ax


# ------------------------------------------------------------------ A: concept
def fig_concept(out):
    plt = plt_setup()
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(14, 4.2), dpi=150)
    fig.patch.set_facecolor(P["paper"]); ax.set_facecolor(P["paper"]); ax.axis("off")
    ax.set_xlim(0, 14); ax.set_ylim(0, 4.2)
    boxes = [
        (0.2, "Declare a budget", "RAM · VRAM · disk\nthe ceiling is the input", P["red"]),
        (2.95, "Plan from metadata", "bin pyramid · gene index\nZ-order sidecar · footers", P["yellow"]),
        (5.7, "Load under a lease", "reserve before reading\nrelease on advance", P["blue"]),
        (8.45, "Navigate", "minimap · detail ladder\nmargin + hysteresis", P["blue"]),
        (11.2, "Hand off", "Arrow · Parquet · NDJSON\n+ checksummed manifest", P["yellow"]),
    ]
    for x, title, body, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.05), 2.55, 2.1, boxstyle="round,pad=0.04,rounding_size=0.18",
                                    fc="white", ec=col, lw=3))
        ax.text(x + 1.275, 2.62, title, ha="center", va="center", fontsize=11.8,
                weight="bold", color=P["navy"])
        ax.text(x + 1.275, 1.72, body, ha="center", va="center", fontsize=10.2,
                color=P["ink"], linespacing=1.45)
    for x in (2.75, 5.5, 8.25, 11.0):
        ax.add_patch(FancyArrowPatch((x, 2.1), (x + 0.2, 2.1), arrowstyle="-|>",
                                     mutation_scale=22, color=P["ink"], lw=2))
    ax.text(0.2, 3.72, "GEMINGA", fontsize=20, weight="bold", color=P["navy"])
    ax.text(2.55, 3.74, "Gated, Extensible, Memory-budgeted Ingestion for N-modal Genomics and Arrays",
            fontsize=11.5, color=P["blue"], va="center")
    ax.text(0.2, 0.42, "peak resident memory never exceeds the declared budget — asserted by the "
            "test suite on every format", fontsize=10.5, style="italic", color=P["ink"])
    ax.text(13.8, 0.42, "Rust core · Python API · 198 tests", fontsize=10,
            ha="right", color=P["ink"])
    p = os.path.join(out, "poster_A_concept.png")
    fig.savefig(p, facecolor=P["paper"], bbox_inches="tight"); plt.close(fig)
    return p


# ------------------------------------------------------------------ B: niches
NICHES = [("VEGFA", "hypoxic rim"), ("CA9", "hypoxic rim"), ("SOX2", "cellular tumour"),
          ("OLIG2", "OPC-like tumour"), ("PECAM1", "microvasculature"),
          ("CD163", "myeloid"), ("SNAP25", "infiltrated neurons"), ("MBP", "white matter")]


def fig_niches(gef, out, bin_size=20, side=200):
    plt = plt_setup()
    v = g.open_gef(gef, budget=g.Budget(ram="256MB"))
    ex = v.gef.extent
    reg = np.load(os.path.join(os.path.dirname(gef), "regions.npy")) \
        if os.path.exists(os.path.join(os.path.dirname(gef), "regions.npy")) else None
    fig = plt.figure(figsize=(15.5, 7.4), dpi=150); fig.patch.set_facecolor(P["paper"])
    gs = fig.add_gridspec(2, 6, width_ratios=[1.35, 1.35, 1, 1, 1, 1], wspace=0.22, hspace=0.34)

    ax = fig.add_subplot(gs[:, :2])
    t0 = time.perf_counter(); tot = v.gef.rasterize(ex, 50, None, 80); t_tot = time.perf_counter() - t0
    a = np.log1p(tot); ax.imshow(a, origin="lower", cmap="magma", extent=[ex[0], ex[1], ex[2], ex[3]],
                                 vmin=0, vmax=np.percentile(a[a > 0], 99), interpolation="nearest")
    if reg is not None:
        cx, cy, rn, rr, rt, rm = reg[:6]
        # the phantom's borders are lobed — r(t) = R (1 + 0.12 sin 5t) — so draw that, not circles
        th = np.linspace(0, 2 * np.pi, 400)
        wob = 1 + 0.12 * np.sin(5 * th)
        for r, col in ((rn, "white"), (rr, P["yellow"]), (rt, P["blue"]), (rm, "white")):
            ax.plot(cx + r * wob * np.cos(th), cy + r * wob * np.sin(th), ls="--", lw=1.4,
                    color=col, alpha=0.9)
        ax.text(cx, cy, "necrosis", ha="center", va="center", color="white", fontsize=9, weight="bold")
        ax.text(cx + rr * 0.72, cy + rr * 0.72, "rim", color=P["yellow"], fontsize=9, weight="bold")
        ax.text(cx + rt * 0.72, cy - rt * 0.72, "tumour", color="#9fd3ff", fontsize=9, weight="bold")
        ax.text(cx - rm * 0.95, cy + rm * 0.78, "margin", color="white", fontsize=9, weight="bold")
    ax.set_title(f"All genes, bin50 — {t_tot * 1000:.0f} ms", loc="left", color=P["navy"])
    ax.set_xticks([]); ax.set_yticks([])

    rows = []
    for k, (gene, lab) in enumerate(NICHES):
        ax = fig.add_subplot(gs[k // 4, 2 + k % 4])
        t0 = time.perf_counter()
        d = v.gef.read_window(ex, bin_size, [gene]); dt = time.perf_counter() - t0
        img = np.zeros((side, side), np.float32)
        ix = np.clip(((d["x"] - ex[0]) / (ex[1] - ex[0]) * (side - 1)).astype(int), 0, side - 1)
        iy = np.clip(((d["y"] - ex[2]) / (ex[3] - ex[2]) * (side - 1)).astype(int), 0, side - 1)
        np.add.at(img, (iy, ix), d["count"])
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, 1.2)
        ax.imshow(img, origin="lower", cmap="magma", vmin=0,
                  vmax=np.percentile(img[img > 0], 99.5) if (img > 0).any() else 1)
        ax.set_title(gene, loc="left", fontsize=12, color=P["navy"])
        ax.text(0.02, -0.07, f"{lab}\n{d['x'].size:,} obs · {dt * 1000:.0f} ms", transform=ax.transAxes,
                va="top", fontsize=8.6, color=P["ink"])
        ax.set_xticks([]); ax.set_yticks([])
        rows.append((gene, d["x"].size, dt))
    fig.suptitle("Glioblastoma niches, each read as one contiguous gene slice", x=0.01, ha="left",
                 fontsize=15, weight="bold", color=P["navy"])
    fig.text(0.995, 0.975, TAG, ha="right", fontsize=8.5, color=P["red"])
    p = os.path.join(out, "poster_B_niches.png")
    fig.savefig(p, facecolor=P["paper"], bbox_inches="tight"); plt.close(fig)
    v.gef.close()
    return p, rows


# ------------------------------------------------------------------ C: navigation
def fig_navigation(gef, out, w=700, h=520):
    from PIL import Image, ImageDraw, ImageFont
    from geminga import Event
    b = g.Budget(ram="96MB", vram="256MB", spill="1GB")
    ex = g.explore(gef, budget=b, width=w, height=h, point_budget="8MB")
    reg = np.load(os.path.join(os.path.dirname(gef), "regions.npy")) \
        if os.path.exists(os.path.join(os.path.dirname(gef), "regions.npy")) else None
    f = ex.full
    rim = (reg[0] + reg[3] * 0.95, reg[1]) if reg is not None else ((f[0] + f[1]) / 2, (f[2] + f[3]) / 2)
    margin = (reg[0] - reg[5] * 0.85, reg[1] + reg[5] * 0.3) if reg is not None else rim

    def px(wx, wy):
        cam = ex.stage.camera; sx, sy = cam.scale
        return ((wx - cam.cx) * sx + 1) / 2 * w, (1 - (wy - cam.cy) * sy) / 2 * h

    shots = [("whole section, fitted", ex.frame())]
    for _ in range(7):
        x, y = px(*rim); ex.handle(Event("scroll", x=x, y=y, steps=1)); img = ex.frame()
    shots.append(("zoomed onto the hypoxic rim", img))
    ex.bookmark("rim")
    ex.gene("VEGFA"); shots.append(("VEGFA only — the rim lights up", ex.frame()))
    ex.gene(None); ex.fit(); ex.frame()
    for _ in range(7):
        x, y = px(*margin); ex.handle(Event("scroll", x=x, y=y, steps=1)); img = ex.frame()
    shots.append(("the infiltrated margin", img))
    ex.gene("SNAP25"); shots.append(("SNAP25 — surviving neurons", ex.frame()))
    ex.gene(None); ex.recall("rim"); shots.append(("recall bookmark 'rim'", ex.frame()))

    gap, capt = 12, 30
    tw, th = w // 2 + 60, int((w // 2 + 60) * h / w)
    sheet = Image.new("RGB", (3 * tw + 2 * gap, 2 * (th + capt) + gap), (240, 238, 232))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    for k, (cap, arr) in enumerate(shots):
        im = Image.fromarray(arr[:, :, :3]).resize((tw, th), Image.LANCZOS)
        cx, cy = (k % 3) * (tw + gap), (k // 3) * (th + capt + gap)
        sheet.paste(im, (cx, cy + capt))
        d.text((cx + 4, cy + 7), f"{k + 1}. {cap}", fill=(11, 37, 69), font=font)
    p = os.path.join(out, "poster_C_navigation.png"); sheet.save(p)
    return p, ex.telemetry, b


# ------------------------------------------------------------------ D: measured
def fig_measured(gef, out):
    plt = plt_setup()
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), dpi=150)
    fig.patch.set_facecolor(P["paper"])

    # (a) budget invariance
    ax = style(axes[0])
    caps, peaks, bins = [], [], []
    for cap in ("4MB", "8MB", "16MB", "32MB", "64MB", "128MB"):
        b = g.Budget(ram=cap); v = g.open_gef(gef, budget=b).reset().auto_bin()
        v.load(); st = b.stats()["ram"]
        caps.append(g.parse_size(cap) / 2 ** 20); peaks.append(st["peak"] / 2 ** 20)
        bins.append(v.cost()["bin_size"]); v.release(); v.gef.close()
    ax.plot(caps, caps, "--", color=P["ink"], lw=1.3, label="declared budget")
    ax.plot(caps, peaks, "o-", color=P["blue"], lw=2.4, ms=7, label="measured peak")
    ax.fill_between(caps, peaks, caps, color=P["blue"], alpha=0.12)
    for x, y, bb in zip(caps, peaks, bins):
        ax.annotate(f"bin{bb}", (x, y), textcoords="offset points", xytext=(5, -13), fontsize=8.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("declared budget (MB)"); ax.set_ylabel("peak resident (MB)")
    ax.set_title("a  the ceiling holds", loc="left", color=P["navy"]); ax.legend(frameon=False, fontsize=9)

    # (b, c) spatial index: time and bytes read vs window size
    plain = g.open_gef(gef).gef; plain._sidx, plain._sidx_checked = None, True
    idx = g.open_gef(gef).gef
    if idx.spatial_index is None:
        idx.build_spatial_index()
    ex = idx.extent; cx, cy = (ex[0] + ex[1]) / 2, (ex[2] + ex[3]) / 2
    fracs = [1.0, 0.25, 0.063, 0.016, 0.004, 0.001]
    rows = []
    for fr in fracs:
        s = (ex[1] - ex[0]) * fr ** 0.5
        w = (int(cx - s / 2), int(cx + s / 2), int(cy - s / 2), int(cy + s / 2))
        t0 = time.perf_counter(); plain.read_window(w, 1); t_a = time.perf_counter() - t0
        t0 = time.perf_counter(); idx.read_window(w, 1); t_b = time.perf_counter() - t0
        rows.append((fr, t_a, t_b, plain.read_plan(w, 1)["read_bytes"], idx.read_plan(w, 1)["read_bytes"]))
    fr = [r[0] * 100 for r in rows]
    ax = style(axes[1])
    ax.plot(fr, [r[1] * 1000 for r in rows], "o-", color=P["red"], lw=2.4, ms=7, label="whole level")
    ax.plot(fr, [r[2] * 1000 for r in rows], "o-", color=P["blue"], lw=2.4, ms=7, label="Z-order sidecar")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel("window, % of the section"); ax.set_ylabel("read time (ms)")
    sp = rows[3][1] / rows[3][2]
    ax.set_title(f"b  time scales with the window ({sp:.0f}x at 1.6 %)", loc="left", color=P["navy"])
    ax.legend(frameon=False, fontsize=9)
    ax = style(axes[2])
    ax.plot(fr, [r[3] / 2 ** 20 for r in rows], "o-", color=P["red"], lw=2.4, ms=7, label="whole level")
    ax.plot(fr, [r[4] / 2 ** 20 for r in rows], "o-", color=P["blue"], lw=2.4, ms=7, label="Z-order sidecar")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel("window, % of the section"); ax.set_ylabel("compressed bytes read (MB)")
    rb = rows[-1][3] / rows[-1][4]
    ax.set_title(f"c  and so do bytes ({rb:.0f}x less at 0.1 %)", loc="left", color=P["navy"])
    ax.legend(frameon=False, fontsize=9)

    # (d) gene pushdown for glioma markers
    ax = style(axes[3])
    v = g.open_gef(gef).reset().use_bin(10)
    v.gef._sidx, v.gef._sidx_checked = None, True           # compare against the full read
    t0 = time.perf_counter(); allg = v.select_genes(None).load(); t_all = time.perf_counter() - t0; v.release()
    names, sps = [], []
    for gene, _ in NICHES:
        v.select_genes([gene]); t0 = time.perf_counter(); v.load(); dt = time.perf_counter() - t0; v.release()
        names.append(gene); sps.append(t_all / max(dt, 1e-9))
    y = np.arange(len(names))
    ax.barh(y, sps, color=P["yellow"], edgecolor=P["ink"], lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlabel("speedup vs loading all genes (x)")
    ax.set_title(f"d  one gene, one slice (median {np.median(sps):.0f}x)", loc="left", color=P["navy"])
    v.gef.close()

    fig.suptitle("Measured on a laptop-class budget", x=0.01, ha="left", fontsize=15,
                 weight="bold", color=P["navy"])
    fig.text(0.995, 0.975, TAG, ha="right", fontsize=8.5, color=P["red"])
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out, "poster_D_measured.png")
    fig.savefig(p, facecolor=P["paper"], bbox_inches="tight"); plt.close(fig)
    return p, dict(caps=caps, peaks=peaks, bins=bins, index=rows, pushdown=dict(zip(names, sps)),
                   all_genes_s=t_all, all_obs=int(allg["x"].size))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gef", default="/tmp/gem/gbm/gbm_phantom.gef")
    ap.add_argument("--out", default="poster")
    ap.add_argument("--only", default="ABCD")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if "A" in a.only: print(fig_concept(a.out))
    if "B" in a.only:
        p, rows = fig_niches(a.gef, a.out); print(p)
        for gene, n, dt in rows: print(f"   {gene:<7} {n:>8,} obs  {dt * 1000:5.0f} ms")
    if "C" in a.only:
        p, t, b = fig_navigation(a.gef, a.out); print(p)
        print("  ", t.summary(), "| peak", b.stats()["ram"]["peak_human"])
    if "D" in a.only:
        p, m = fig_measured(a.gef, a.out); print(p)
        print("   invariance:", [f"{c:.0f}->{pk:.1f}" for c, pk in zip(m["caps"], m["peaks"])])
        for fr, ta, tb, ba, bb in m["index"]:
            print(f"   {fr:>6.1%}  {ta * 1000:6.0f} -> {tb * 1000:5.0f} ms   {ba / 2**20:6.2f} -> {bb / 2**20:6.2f} MB")
        print(f"   pushdown median {np.median(list(m['pushdown'].values())):.0f}x "
              f"(all genes {m['all_obs']:,} obs in {m['all_genes_s']:.2f}s)")
