"""Pictures for navigation.

Three figures, each answering a different question:

* `navigator(view)` — where am I in this dataset, what does the neighbourhood look like, and
  what would loading the current selection cost? The overview strip is drawn from metadata, so
  it renders for a 60 GB file as fast as for a 60 MB one.
* `progressive(canvas)` — the picture as it stands right now, mid-stream.
* `hits(view, results)` — where the search matches fell, aligned to the same axis as the
  overview, so a motif's distribution can be read against coverage or quality.

Matplotlib is imported lazily and only here; nothing in the core needs it.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._core import human

__all__ = ["navigator", "progressive", "hits", "chip", "PALETTE"]

# The project palette, so figures match the logo and the docs.
PALETTE = {"ink": "#1A1A1A", "red": "#E8412F", "yellow": "#F5C518",
           "blue": "#2E77C4", "navy": "#0B2545", "paper": "#FFFCF3",
           "grid": "#D9DFE7"}


def _style(ax):
    ax.set_facecolor(PALETTE["paper"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(PALETTE["ink"])
        ax.spines[s].set_linewidth(1.2)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=PALETTE["ink"], labelsize=8)
    return ax


def _cost_text(c: dict) -> str:
    lines = [
        f"window   {c['window']}",
        f"zoom     {c['zoom']}  (stride {c['stride']})",
        f"{c['unit']:<8} {c['units']:,}",
        f"         {c['fraction_of_dataset']:.2%} of dataset",
        f"chunks   {c['chunks']} x {c['resident_human']}",
        f"total    {c['total_human']}",
    ]
    if "decode_human" in c:
        lines.append(f"decode   {c['tiles_touched']} tiles -> {c['decode_human']}")
    if c.get("fits_whole") is True:
        lines.append(f"load()   fits ({c['budget_available_human']} free)")
    elif c.get("fits_whole") is False:
        lines.append("load()   " + ("stream with .chunks()" if c.get("fits")
                                    else "TOO BIG even to stream"))
    return "\n".join(lines)


def navigator(view, path: Optional[str] = None, series: Optional[Sequence[str]] = None,
              n_buckets: int = 320, detail: bool = True, title: Optional[str] = None,
              dpi: int = 130):
    """Overview strip with the current window highlighted, the detail beneath it, and the cost.

    Nothing is loaded to draw the overview. The detail panel is drawn only if `detail=True`
    *and* the current selection fits the budget — otherwise it says so instead of blowing up.

    Row and record views get an overview strip per series. A 2-D window is drawn by what it is:
    an `ImageView`'s is (y0, y1, x0, x1), rows first with row 0 at the top; a point-cloud view's
    (a Stereo-seq GEF, a GEM) is (x0, x1, y0, y1) with y growing upward. The red rectangle is
    placed accordingly, so a window is never drawn transposed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from .view import ImageView

    full = view._full_window()
    saved = view.window
    ov_full = view.select(*full).overview(n_buckets)
    view.select(*saved)
    c = view.cost()

    # Two kinds of 2-D window, with opposite axis orders: an image is (y0, y1, x0, x1) — rows
    # first, row 0 at the top — and a point cloud (Stereo-seq GEF, GEM) is (x0, x1, y0, y1)
    # with y growing upward. Reading one as the other draws the window transposed.
    is_image = isinstance(view, ImageView)
    is_points = len(full) == 4 and not is_image
    flat = is_image or is_points
    if series:
        names = list(series)
    else:
        # Prefer series that actually vary — a flat line teaches nothing and costs a panel.
        cand = [k for k in ov_full.series if k != "thumbnail"]
        def varies(k):
            a = np.asarray(ov_full.series[k], dtype=float)
            a = a[np.isfinite(a)]
            return a.size > 1 and float(a.std()) > abs(float(a.mean())) * 1e-6 + 1e-12
        names = [k for k in cand if varies(k)][:3] or cand[:1]
    n_over = 1 if flat else max(len(names), 1)
    rows = n_over + (1 if detail else 0)

    fig = plt.figure(figsize=(11, 2.1 * n_over + (3.4 if detail else 0) + 0.5), dpi=dpi)
    fig.patch.set_facecolor(PALETTE["paper"])
    gs = fig.add_gridspec(rows, 5, hspace=0.55, wspace=0.42,
                          width_ratios=[1, 1, 1, 0.25, 1.25],
                          height_ratios=[1] * n_over + ([1.8] if detail else []))

    colors = [PALETTE["blue"], PALETTE["red"], PALETTE["yellow"]]

    if flat:
        ax = _style(fig.add_subplot(gs[0, :4]))
        thumb = ov_full.series.get("thumbnail")
        if is_image:
            if thumb is not None:
                ax.imshow(thumb, cmap="magma", aspect="equal",
                          extent=[full[2], full[3], full[1], full[0]])
            y0, y1, x0, x1 = saved
        else:
            if thumb is not None:                       # raster row 0 is the low-y edge
                ax.imshow(thumb, cmap="magma", aspect="equal", origin="lower",
                          extent=[full[0], full[1], full[2], full[3]])
            x0, x1, y0, y1 = saved
        ax.grid(False)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor=PALETTE["red"], linewidth=2.4))
        ax.set_title("overview — current window in red", fontsize=10, color=PALETTE["ink"],
                     loc="left")
    else:
        for i, nm in enumerate(names):
            ax = _style(fig.add_subplot(gs[i, :4]))
            y = np.asarray(ov_full.series[nm], dtype=float)
            ax.plot(ov_full.x, y, color=colors[i % 3], linewidth=1.4)
            ax.fill_between(ov_full.x, y, np.nanmin(y) if np.isfinite(y).any() else 0,
                            color=colors[i % 3], alpha=0.18)
            ax.axvspan(saved[0], saved[1], color=PALETTE["red"], alpha=0.16)
            for xv in saved[:2]:
                ax.axvline(xv, color=PALETTE["red"], linewidth=1.6)
            ax.set_ylabel(nm, fontsize=9, color=PALETTE["ink"])
            ax.set_xlim(full[0], full[1])
            if i == 0:
                ax.set_title(f"overview — {len(ov_full)} buckets from "
                             f"{'metadata' if ov_full.cheap else 'a sample'} "
                             f"({human(ov_full.source_bytes)} consulted)",
                             fontsize=10, color=PALETTE["ink"], loc="left")
            if i == len(names) - 1:
                ax.set_xlabel(ov_full.kind, fontsize=9, color=PALETTE["ink"])

    axc = fig.add_subplot(gs[0:n_over, 4])
    axc.axis("off")
    axc.text(0, 1, _cost_text(c), family="DejaVu Sans Mono", fontsize=7.6,
             va="top", ha="left", color=PALETTE["ink"],
             bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                       edgecolor=PALETTE["blue"], linewidth=1.4))

    if detail:
        axd = _style(fig.add_subplot(gs[n_over, :]))
        if c.get("fits_whole") is False and c.get("fits") is not True:
            axd.text(0.5, 0.5, "selection does not fit the budget\n" + c.get("advice", ""),
                     ha="center", va="center", fontsize=10, color=PALETTE["red"], wrap=True)
            axd.set_xticks([]); axd.set_yticks([])
        else:
            _detail_panel(axd, view, c, is_image)

    fig.suptitle(title or f"GEMINGA — {type(view).__name__}", fontsize=13,
                 color=PALETTE["navy"], x=0.01, ha="left", weight="bold")
    fig.subplots_adjust(left=0.065, right=0.995, top=0.925, bottom=0.085)
    if path:
        fig.savefig(path, facecolor=PALETTE["paper"])
        plt.close(fig)
        return path
    return fig


def _detail_panel(ax, view, c, is_image):
    """Draw whatever the current selection actually contains."""
    if is_image:
        img = view.load()
        ax.imshow(img, cmap="magma", aspect="equal")
        ax.grid(False)
        ax.set_title(f"detail — {img.shape[0]}x{img.shape[1]} px at stride {c['stride']}",
                     fontsize=10, loc="left", color=PALETTE["ink"])
        return
    data = view.load()
    if data is None:
        ax.text(0.5, 0.5, "empty selection", ha="center", va="center")
        return
    if isinstance(data, dict) and "x" in data:                    # Stereo-seq GEF window
        xs, ys = np.asarray(data["x"]), np.asarray(data["y"])
        if xs.size == 0:
            ax.text(0.5, 0.5, "empty selection", ha="center", va="center")
            return
        w = data.get("count")
        ax.hexbin(xs, ys, C=None if w is None else np.asarray(w, dtype=float),
                  reduce_C_function=np.sum, gridsize=110, cmap="magma", mincnt=1)
        ax.set_title(f"detail — {xs.size:,} observations at bin{data.get('bin_size', '?')}",
                     fontsize=10, loc="left", color=PALETTE["ink"])
        return
    if hasattr(data, "schema"):                                   # pyarrow.Table
        names = data.schema.names
        num = [n for n in names if data.column(n).type in
               (__import__("pyarrow").int32(), __import__("pyarrow").int64(),
                __import__("pyarrow").float32(), __import__("pyarrow").float64())]
        if {"x", "y"} <= set(names):
            xs = data.column("x").to_numpy(zero_copy_only=False)
            ys = data.column("y").to_numpy(zero_copy_only=False)
            ax.hexbin(xs, ys, gridsize=110, cmap="magma", mincnt=1)
            ax.set_title(f"detail — {data.num_rows:,} rows at stride {c['stride']}",
                         fontsize=10, loc="left", color=PALETTE["ink"])
            return
        if "seq" in names:
            lens = data.column("length").to_numpy(zero_copy_only=False) if "length" in names else None
            seqs = data.column("seq").to_pylist()
            gc = np.array([(s.count("G") + s.count("C")) / max(len(s), 1) for s in seqs])
            ax.hist(gc, bins=min(60, max(len(np.unique(gc)), 10)),
                    color=PALETTE["blue"], alpha=0.85)
            ax.set_xlabel("GC fraction"); ax.set_ylabel("records")
            ax.set_title(f"detail — {data.num_rows:,} records at stride {c['stride']}"
                         + (f", mean length {lens.mean():.0f}" if lens is not None else ""),
                         fontsize=10, loc="left", color=PALETTE["ink"])
            return
        if num:
            v = data.column(num[0]).to_numpy(zero_copy_only=False)
            ax.hist(v, bins=60, color=PALETTE["blue"], alpha=0.85)
            ax.set_xlabel(num[0]); ax.set_ylabel("rows")
            return
        ax.text(0.5, 0.5, f"{data.num_rows:,} rows, no numeric column to plot",
                ha="center", va="center")
        return
    if hasattr(data, "nnz"):                                       # scipy sparse (h5ad)
        per_cell = np.asarray((data > 0).sum(1)).ravel()
        ax.hist(per_cell, bins=60, color=PALETTE["blue"], alpha=0.85)
        ax.set_xlabel("features detected per cell"); ax.set_ylabel("cells")
        ax.set_title(f"detail — {data.shape[0]:,} cells x {data.shape[1]:,} features, "
                     f"{data.nnz:,} nonzeros", fontsize=10, loc="left", color=PALETTE["ink"])
        return
    ax.text(0.5, 0.5, repr(type(data)), ha="center", va="center")


def progressive(canvas, path: Optional[str] = None, log: bool = True,
                title: Optional[str] = None, dpi: int = 130):
    """Draw a ProgressiveCanvas as it currently stands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    img = canvas.image
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=dpi)
    fig.patch.set_facecolor(PALETTE["paper"])
    pos = img[img > 0]
    norm = LogNorm(vmin=max(pos.min(), 1e-9), vmax=img.max()) if (log and pos.size) else None
    im = ax.imshow(img, origin="lower", cmap="magma", norm=norm,
                   extent=[canvas.x0, canvas.x1, canvas.y0, canvas.y1], aspect="equal")
    fig.colorbar(im, ax=ax, label="accumulated weight")
    _style(ax)
    ax.grid(False)
    ax.set_title(title or f"{canvas.n_rows:,} rows in {canvas.n_batches} batches — "
                          f"{canvas.coverage:.0%} covered, {human(canvas.nbytes)} resident",
                 fontsize=10, loc="left", color=PALETTE["ink"])
    fig.tight_layout()
    if path:
        fig.savefig(path, facecolor=PALETTE["paper"]); plt.close(fig); return path
    return fig


def hits(view, results: dict, path: Optional[str] = None, n_buckets: int = 256,
         patterns: Optional[Sequence[str]] = None, dpi: int = 130):
    """Search hits as a density profile on the same axis as the overview."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prof = view.hit_profile(results, n_buckets)
    ov = view.overview(n_buckets)
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.6), dpi=dpi, sharex=True)
    fig.patch.set_facecolor(PALETTE["paper"])

    ax = _style(axes[0])
    key = "mean_q" if "mean_q" in ov.series else next(iter(ov.series))
    ax.plot(ov.x, ov.series[key], color=PALETTE["blue"], linewidth=1.3)
    ax.set_ylabel(key, fontsize=9)
    ax.set_title(f"{results['n_hits']:,} hits in {results['records_matched']:,} of "
                 f"{results['records_scanned']:,} records scanned"
                 + (f"  ·  patterns: {', '.join(patterns)}" if patterns else ""),
                 fontsize=10, loc="left", color=PALETTE["ink"])

    ax = _style(axes[1])
    ax.bar(prof.x, prof.series["hits"], width=(prof.x[1] - prof.x[0]) if len(prof) > 1 else 1,
           color=PALETTE["red"], alpha=0.9)
    ax.set_ylabel("hits", fontsize=9)
    ax.set_xlabel(prof.kind, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, facecolor=PALETTE["paper"]); plt.close(fig); return path
    return fig


def _norm(img):
    """Percentile-clipped log scaling — spatial MID counts are long-tailed enough that a raw
    log1p leaves a whole-chip view looking black."""
    a = np.log1p(np.asarray(img, dtype=np.float64))
    pos = a[a > 0]
    if pos.size == 0:
        return a, None, None
    return a, float(np.percentile(pos, 2)), float(np.percentile(pos, 99.5))


def chip(view_or_dataset, window=None, path: Optional[str] = None, shape: int = 420,
         genes: Optional[Sequence[str]] = None, dpi: int = 130):
    """A Stereo-seq chip sheet: whole-chip overview, the selected window's expression, the
    registered morphology crop, and — when genes are given — that gene's own distribution.

    The overview always comes from the coarsest bin in the GEF, so it renders in the same time
    for a 1 cm chip and a 13 cm one.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from .stereoseq import StereoSeqDataset, StereoSeqView

    ds = view_or_dataset if isinstance(view_or_dataset, StereoSeqDataset) else None
    v = ds.expression() if ds else view_or_dataset
    if not isinstance(v, StereoSeqView):
        raise TypeError("chip() needs a StereoSeqView or a StereoSeqDataset")
    if window:
        v.select(*window)
    win = v.window
    full = v.gef.extent
    c = v.cost()

    panels = [("whole chip", None), (f"window · bin{c['bin_size']}", None)]
    if ds is not None and ds.image is not None:
        panels.append(("morphology", None))
    if genes:
        panels.append((", ".join(genes[:2]) + ("..." if len(genes) > 2 else ""), None))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.5), dpi=dpi)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(PALETTE["paper"])

    over = v.gef.rasterize(full, v.gef.bins[-1], None, shape=shape)
    ax = axes[0]
    a, lo, hi = _norm(over)
    ax.imshow(a, cmap="magma", origin="lower", vmin=lo, vmax=hi,
              extent=[full[0], full[1], full[2], full[3]], aspect="equal")
    ax.add_patch(Rectangle((win[0], win[2]), win[1] - win[0], win[3] - win[2],
                           fill=False, edgecolor=PALETTE["red"], linewidth=2.2))
    ax.set_title(f"whole chip · bin{v.gef.bins[-1]}\n"
                 f"{human(v.gef.levels[v.gef.bins[-1]].nbytes)} read",
                 fontsize=9.5, loc="left", color=PALETTE["ink"])
    ax.grid(False); ax.tick_params(labelsize=7)

    ax = axes[1]
    a, lo, hi = _norm(v.image(shape))
    ax.imshow(a, cmap="magma", origin="lower", vmin=lo, vmax=hi,
              extent=[win[0], win[1], win[2], win[3]], aspect="equal")
    ax.set_title(f"window · bin{c['bin_size']} · {c['units']:,} obs\n{c['resident_human']} resident",
                 fontsize=9.5, loc="left", color=PALETTE["ink"])
    ax.grid(False); ax.tick_params(labelsize=7)

    i = 2
    if ds is not None and ds.image is not None:
        pair = ds.paired(win, shape=shape)
        if "morphology" in pair:
            ax = axes[i]
            ax.imshow(pair["morphology"], cmap="bone", origin="lower", aspect="equal")
            ax.set_title("morphology (registered)\nsame chip window", fontsize=9.5,
                         loc="left", color=PALETTE["ink"])
            ax.grid(False); ax.tick_params(labelsize=7)
        i += 1

    if genes:
        saved = v.genes
        ax = axes[i]
        v.select_genes(list(genes))
        gd = v.gef.read_window(win, v.bin_size, list(genes))
        a, lo, hi = _norm(v.image(shape))
        ax.imshow(a, cmap="magma", origin="lower", vmin=lo, vmax=hi,
                  extent=[win[0], win[1], win[2], win[3]], aspect="equal")
        ax.set_title(f"{', '.join(genes[:2])}\n{gd['x'].size:,} obs from a contiguous slice",
                     fontsize=9.5, loc="left", color=PALETTE["ink"])
        ax.grid(False); ax.tick_params(labelsize=7)
        v.select_genes(saved)

    fig.suptitle(f"GEMINGA — Stereo-seq {v.gef.chip or ''}".strip(), fontsize=13,
                 color=PALETTE["navy"], x=0.01, ha="left", weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if path:
        fig.savefig(path, facecolor=PALETTE["paper"]); plt.close(fig); return path
    return fig
