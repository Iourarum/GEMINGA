"""Selections, end to end: mark, describe, find more like it — on a map and on a time series.

    python examples/make_glioma_phantom.py fixtures/gbm
    python examples/make_culture_timeseries.py fixtures/culture.parquet
    python examples/selections_figure.py --gef fixtures/gbm/gbm_phantom.gef --ts fixtures/culture.parquet
"""
import argparse
import os
import numpy as np
import geminga as g
P = g.render.PALETTE

ap = argparse.ArgumentParser()
ap.add_argument("--gef", default="/tmp/gem/gbm/gbm_phantom.gef")
ap.add_argument("--ts", default="/tmp/gem/culture.parquet")
ap.add_argument("--out", default="/home/claude/poster/selections.png")
ap.add_argument("--book", default="/tmp/gem/sel_fig.jsonl")
a = ap.parse_args()
if os.path.exists(a.book):
    os.remove(a.book)
book = g.selections(a.book)

# --- the map: mark a rim patch while navigating, ask for more like it -------------------
reg = np.load(os.path.join(os.path.dirname(a.gef), "regions.npy"))
cx, cy, rn, rr = reg[:4]
r, s = (rn + rr) / 2 * 1.02, 110
ex = g.explore(a.gef, width=620, height=560, point_budget="8MB", selections=book)
sel = ex.select_view("rim-east", intent="need-data", bounds=(cx + r - s, cx + r + s, cy - s, cy + s))
summ = ex.describe_active()
cands = ex.more_like_this(k=6)
f = ex.full
span = (rr * 1.25) * 2
ex.goto((cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2))
img_overview = ex.frame()
ex.goto_selection(sel, pad=0.6)
img_close = ex.frame()

# --- the time series ---------------------------------------------------------------------
import polars as pl
ev = np.load(a.ts.replace(".parquet", "_events.npy"))
d0 = ev[1]
tsel = g.select_values(a.ts, "t", d0 * 86400, (d0 + 0.25) * 86400, "hypoxia d3.9",
                       intent="need-data", book=book)
tsum = g.describe(tsel, book=book)
tc = g.similar(tsel, k=4, book=book)
df = pl.read_parquet(a.ts)

# --- compose -------------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(17, 9.6), dpi=140)
fig.patch.set_facecolor(P["paper"])
gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 1.15], height_ratios=[1, 0.78],
                      hspace=0.32, wspace=0.16)
for k, (im, title) in enumerate([(img_overview, "a  mark a region, ask for more like it"),
                                 (img_close, "b  the marked patch, at bin1")]):
    ax = fig.add_subplot(gs[0, k]); ax.imshow(im[:, :, :3]); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=12.5, weight="bold", color=P["navy"])

ax = fig.add_subplot(gs[0, 2])
ax.set_facecolor(P["paper"])
en, de = summ["enriched"][:7], summ["depleted"][:5]
rows = en + de
y = np.arange(len(rows))[::-1]
ax.barh(y, [r["log2fc"] for r in rows], color=[P["red"]] * len(en) + [P["blue"]] * len(de),
        edgecolor=P["ink"], lw=0.6)
ax.set_yticks(y); ax.set_yticklabels([r["gene"] for r in rows], fontsize=9.5)
for yy, r in zip(y, rows):
    ax.text(r["log2fc"] + (0.25 if r["log2fc"] > 0 else -0.25), yy,
            f"{r['observed']:,} vs {r['expected']:,.0f}", va="center",
            ha="left" if r["log2fc"] > 0 else "right", fontsize=8, color=P["ink"])
ax.axvline(0, color=P["ink"], lw=1)
lim = max(abs(r["log2fc"]) for r in rows) * 1.55
ax.set_xlim(-lim, lim * 0.95)
ax.set_xlabel("log2 (observed + 1) / (expected + 1)")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.set_title("c  what is inside, against the section's composition", loc="left",
             fontsize=12.5, weight="bold", color=P["navy"])

ax = fig.add_subplot(gs[1, :])
ax.set_facecolor(P["paper"])
days = df["t"].to_numpy() / 86400
ax.plot(days, df["o2_percent"].to_numpy(), color=P["navy"], lw=0.6, alpha=0.8, label="dissolved O2 (%)")
ax2 = ax.twinx(); ax2.plot(days, df["ph"].to_numpy(), color=P["red"], lw=0.5, alpha=0.55, label="pH")
ax2.set_ylabel("pH", color=P["red"]); ax2.tick_params(colors=P["red"])
lo, hi = tsel.geometry["lo"] / 86400, tsel.geometry["hi"] / 86400
ax.axvspan(lo, hi, color=P["blue"], alpha=0.30)
ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.985, "marked", ha="center", va="top", fontsize=9,
        color=P["blue"], weight="bold")
for c in tc:
    c0, c1 = c.geometry["lo"] / 86400, c.geometry["hi"] / 86400
    strong = c.score > 0.6
    ax.axvspan(c0, c1, color=P["yellow"], alpha=0.45 if strong else 0.15)
    # weak candidates are labelled at the top, so an adjacent strong one never collides
    yl = ax.get_ylim()[0] + 0.4 if strong else ax.get_ylim()[1] - 0.9
    ax.text((c0 + c1) / 2, yl, f"{c.score:.2f}", ha="center", fontsize=9,
            color=P["ink"], weight="bold" if strong else "normal")
ax.set_xlabel("day"); ax.set_ylabel("dissolved O2 (%)")
for sp in ("top",):
    ax.spines[sp].set_visible(False)
ax.set_title("d  the same on a time series: one episode marked, its three siblings found "
             "(scores 0.74–0.87; the next best is 0.36)", loc="left", fontsize=12.5,
             weight="bold", color=P["navy"])
fig.suptitle("Selections — mark it, measure it, find more like it", x=0.01, ha="left",
             fontsize=16, weight="bold", color=P["navy"])
fig.text(0.995, 0.975, "glioma-structured phantom and synthetic culture series · software "
         "measurements are real", ha="right", fontsize=8.5, color=P["red"])
fig.savefig(a.out, facecolor=P["paper"], bbox_inches="tight")
print(a.out)
print(f"book: {len(book)} entries, {len(book.open_questions())} open question(s); "
      f"time-series candidates: {[round(c.score, 3) for c in tc]}")
