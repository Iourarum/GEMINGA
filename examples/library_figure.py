"""The library as a picture: every flagged thing, what connects it, and what state it is in."""
import sys
import numpy as np
import geminga as g
P = g.render.PALETTE
lib = g.library(sys.argv[1] if len(sys.argv) > 1 else "/tmp/gem/lib_demo.jsonl")
out = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/poster/library.png"
recs = sorted(lib.all().values(), key=lambda s: (s.kind, s.name))
KIND = {"rect": ("s", P["red"], "region"), "variant": ("D", P["blue"], "variant"),
        "locus": ("h", P["navy"], "locus"), "sequence": ("^", "#7a4fb5", "sequence"),
        "cells": ("o", "#2a9d8f", "cells"), "genes": ("p", P["yellow"], "genes"),
        "column-range": ("v", "#e76f51", "time values")}
STATUS_EDGE = {"open": P["ink"], "described": P["ink"], "searched": P["ink"],
               "in-review": P["yellow"], "confirmed": "#2a9d8f", "refuted": P["red"],
               "resolved": "#999999"}
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16, 7.6), dpi=140); fig.patch.set_facecolor(P["paper"])
ax = fig.add_axes([0.01, 0.04, 0.55, 0.86]); ax.set_facecolor(P["paper"]); ax.axis("off")
# Force-directed, so connected records sit near each other. A circle ordered by kind put linked
# records on opposite sides, and every edge crossed the middle with its label on top of the rest.
import networkx as nx
G = nx.Graph()
G.add_nodes_from(s.id for s in recs)
for s in recs:
    for o, why in lib.related(s, by=("genes", "locus", "links")):
        strong = any(w.startswith("overlaps") or w.startswith("linked") for w in why)
        G.add_edge(s.id, o.id, weight=3.0 if strong else 1.0)
lay = nx.kamada_kawai_layout(G, weight=None) if G.number_of_edges() else nx.circular_layout(G)
xy = np.array(list(lay.values())); xy = (xy - xy.mean(0)) / (np.abs(xy).max() or 1)
pos = {k: tuple(p) for k, p in zip(lay.keys(), xy)}
drawn = set()
for s in recs:
    for o, why in lib.related(s, by=("genes", "locus", "links")):
        key = tuple(sorted((s.id, o.id)))
        if key in drawn or o.id not in pos:
            continue
        drawn.add(key)
        (x0, y0), (x1, y1) = pos[s.id], pos[o.id]
        strong = any(w.startswith("overlaps") or w.startswith("linked") for w in why)
        ax.plot([x0, x1], [y0, y1], color=P["blue"] if strong else P["grid"],
                lw=2.4 if strong else 1.4, zorder=1)
        link = [w for w in why if w.startswith("linked")]
        lab = (link[0].replace("linked: ", "").lstrip("~") if link
               else why[0].replace("genes: ", ""))[:22]
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, lab, fontsize=7.5, ha="center", va="center",
                color=P["ink"], bbox=dict(boxstyle="round,pad=0.15", fc=P["paper"], ec="none"),
                zorder=2)
for s in recs:
    mk, col, _ = KIND.get(s.kind, ("o", "#888", s.kind))
    x, y = pos[s.id]
    ax.scatter([x], [y], s=520, marker=mk, color=col, edgecolor=STATUS_EDGE.get(s.status, P["ink"]),
               linewidth=3.2 if s.status in ("in-review", "confirmed", "refuted") else 1.2, zorder=3)
    ha = "left" if x > 0.05 else ("right" if x < -0.05 else "center")
    ax.text(x + 0.13 * np.sign(x or 1), y + 0.02, f"{s.name[:26]}\n{s.intent} · {s.status}", ha=ha, va="center",
            fontsize=8.6, color=P["navy"], zorder=4)
ax.set_xlim(-2.0, 2.0); ax.set_ylim(-1.25, 1.25)
from matplotlib.lines import Line2D
hand = [Line2D([], [], marker=m, color="w", markerfacecolor=c, markersize=11, label=l)
        for m, c, l in KIND.values()]
ax.legend(handles=hand, loc="lower left", fontsize=8.5, frameon=False, ncol=4,
          bbox_to_anchor=(0.0, -0.06))
ax.set_title("a  what was flagged, and what connects it", loc="left", fontsize=13,
             weight="bold", color=P["navy"])

tx = fig.add_axes([0.58, 0.04, 0.41, 0.86]); tx.axis("off")
tx.set_title("b  the record for one flag", loc="left", fontsize=13, weight="bold", color=P["navy"])
snp = next(s for s in recs if s.kind == "variant" and s.geometry.get("rsid"))
gm = snp.geometry
lines = [("what", f"variant {gm['rsid']}  {gm['ref']}>{gm['alt']}  gene {', '.join(snp.genes or [])}"),
         ("where", f"chr{gm['chrom']}:{gm['pos']:,} ({gm['build']}, VCF 1-based)"),
         ("why", snp.note),
         ("metadata", ", ".join(f"{k}={v}" for k, v in snp.metadata.items())),
         ("history", "")]
lines += [("", f"  {h['status']:<10} {h.get('note', '')[:44]}") for h in snp.history]
lines += [("related", "")]
lines += [("", f"  {o.kind:<9} {o.name[:22]:<22} {'; '.join(w)[:34]}") for o, w in lib.related(snp)[:4]]
lines += [("more data", "")]
lines += [("", f"  {l['source']:<8} {l['url'][:60]}") for l in snp.leads[:4]]
y = 0.96
for k, v in lines:
    tx.text(0.0, y, k, fontsize=9.5, weight="bold", color=P["blue"], family="DejaVu Sans Mono",
            transform=tx.transAxes, va="top")
    tx.text(0.17, y, v, fontsize=8.6, color=P["ink"], family="DejaVu Sans Mono",
            transform=tx.transAxes, va="top")
    y -= 0.047
fig.suptitle("The library — flag anything, with its own coordinates; find what connects it",
             x=0.01, ha="left", fontsize=15, weight="bold", color=P["navy"])
fig.text(0.995, 0.975, "demonstration records on synthetic data · software behaviour is real",
         ha="right", fontsize=8.5, color=P["red"])
fig.savefig(out, facecolor=P["paper"], bbox_inches="tight"); print(out)
