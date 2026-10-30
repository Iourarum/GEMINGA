"""The orchestrator as a picture: what a workbook of descriptions turns into.

    python examples/orchestrate_figure.py /tmp/gem/inokan out.png

Reads what `orchestrate_inokan.py` wrote (pieces.json, coverage_core.csv). Three panels: the
gh'è-minga tiles (every piece by where it actually is), the recomputed node × dataset evidence
matrix, and the accessions the descriptions turned out to contain.
"""
import csv
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

import geminga as g

P = g.render.PALETTE
SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gem/inokan"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/gem/inokan/orchestrator.png"
orc = g.Orchestrator.load(os.path.join(SRC, "pieces.json"))
pieces = list(orc.pieces.values())

STATUS = {"local": ("#2a9d8f", "here"), "streamable": (P["blue"], "one fetch away, streamable"),
          "fetchable": ("#8db9e6", "one fetch away"), "controlled": ("#e76f51", "behind a DAC"),
          "manual": ("#b9b9b9", "paper only"), "unresolved": (P["paper"], "unresolved")}

fig = plt.figure(figsize=(17, 9.2), dpi=140)
fig.patch.set_facecolor(P["paper"])

# ---- A. gh'è minga: one tile per piece --------------------------------------------------------
ax = fig.add_axes([0.03, 0.50, 0.44, 0.42]); ax.set_facecolor(P["paper"]); ax.axis("off")
cols = 8
order = [p for s in STATUS for p in pieces if p.status == s]
for i, p in enumerate(order):
    r, c = divmod(i, cols)
    col, _ = STATUS[p.status]
    ax.add_patch(Rectangle((c, -r), 0.94, 0.94, facecolor=col, edgecolor=P["ink"],
                           linewidth=0.6, hatch="////" if p.status == "unresolved" else None))
    name = p.id.split("_")[0][:9]
    n = p.expected.get("n_cells")
    ax.text(c + 0.47, -r + 0.58, name, ha="center", va="center", fontsize=6.6,
            color="white" if p.status in ("local", "streamable", "controlled") else P["ink"])
    lab = "" if not n else (f"{n/1e6:.1f}M" if n >= 1e6 else (f"{n/1000:.0f}k" if n >= 1000 else str(n)))
    ax.text(c + 0.47, -r + 0.28, lab,
            ha="center", va="center", fontsize=6, color="white" if p.status in ("local", "streamable", "controlled") else P["ink"])
rows = (len(order) + cols - 1) // cols
ax.set_xlim(-0.2, cols + 0.2); ax.set_ylim(-rows + 0.6, 1.3)
by = Counter(p.status for p in pieces)
ax.set_title("A  gh'è minga — 56 described pieces, resolved without transferring a byte", loc="left",
             fontsize=12, color=P["navy"], fontweight="bold")
hand = [Patch(facecolor=STATUS[s][0], edgecolor=P["ink"], hatch="////" if s == "unresolved" else None,
              label=f"{STATUS[s][1]} ({by.get(s, 0)})") for s in STATUS if by.get(s, 0) or s == "local"]
ax.legend(handles=hand, loc="lower left", bbox_to_anchor=(0.0, -0.16), ncol=3, fontsize=8, frameon=False)

# ---- B. the evidence matrix, recomputed from the descriptions ----------------------------------
with open(os.path.join(SRC, "coverage_core.csv"), newline="") as fh:
    cov_rows = list(csv.DictReader(fh))
piece_cols = [c for c in cov_rows[0] if c not in ("gene", "n_protein", "n_perturbed", "n_implied",
                                                   "n_focus", "n_measured")]
def score(r):
    return int(r["n_protein"]) * 3 + int(r["n_perturbed"]) * 2 + int(r["n_focus"])


top = sorted(cov_rows, key=lambda r: -score(r))[:24]
used = [c for c in piece_cols if any(r[c].replace("x", "") for r in top)]
ax = fig.add_axes([0.53, 0.30, 0.45, 0.62]); ax.set_facecolor(P["paper"])
CODE = {"P": ("#2a9d8f", "P  protein measured"), "X": (P["red"], "X  perturbed"),
        "F": (P["yellow"], "F  named as focus")}
for i, r in enumerate(top):
    for j, c in enumerate(used):
        code = r[c]
        if "x" in code:
            ax.plot(j, i, ".", color=P["grid"], markersize=3, zorder=1)
        letters = [ch for ch in "PXF" if ch in code]
        w = 0.9 / max(len(letters), 1)
        for k, ch in enumerate(letters):
            ax.add_patch(Rectangle((j - 0.45 + k * w, i - 0.45), w, 0.9, facecolor=CODE[ch][0],
                                   edgecolor="none", zorder=2))
ax.set_xlim(-0.6, len(used) - 0.4); ax.set_ylim(len(top) - 0.4, -0.6)
ax.set_xticks(range(len(used))); ax.set_xticklabels([u[:22] for u in used], rotation=90, fontsize=6.4)
ax.set_yticks(range(len(top))); ax.set_yticklabels([r["gene"] for r in top], fontsize=7.4)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.set_title("B  node × dataset evidence recomputed from the rows — 159 of 164 hand-coded cells agree",
             loc="left", fontsize=10.5, color=P["navy"], fontweight="bold")
ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in CODE.values()] +
          [plt.Line2D([], [], marker=".", color=P["grid"], linestyle="", label="x  implied by a genome-wide screen")],
          loc="upper left", bbox_to_anchor=(0.0, -0.36), ncol=2, fontsize=8, frameon=False)

# ---- C. what the prose contained --------------------------------------------------------------
ax = fig.add_axes([0.10, 0.07, 0.36, 0.33]); ax.set_facecolor(P["paper"])
kinds = Counter()
NAMES = {"geo_series": "GEO series", "geo_sample": "GEO sample", "bioproject": "BioProject", "sra_study": "SRA study",
         "sra_run": "SRA run", "zenodo_doi": "Zenodo", "figshare_doi": "figshare", "ega_study": "EGA",
         "ega_dataset": "EGA", "dbgap": "dbGaP", "scp": "Single Cell Portal", "duos": "DUOS",
         "arrayexpress": "ArrayExpress", "mendeley_doi": "Mendeley", "doi": "DOI only", "url": "URL only",
         "cngb": "CNGB", "gsa": "GSA", "openneuro": "OpenNeuro"}
for p in pieces:
    for a in p.accessions:
        kinds[NAMES.get(a.kind, a.kind)] += 1
items = sorted(kinds.items(), key=lambda kv: -kv[1])
ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1],
        color=[("#b9b9b9" if k in ("DOI only", "URL only") else ("#e76f51" if k in ("EGA", "dbGaP", "DUOS") else P["blue"]))
               for k, _ in items][::-1])
for i, (k, v) in enumerate(items[::-1]):
    ax.text(v + 0.5, i, str(v), va="center", fontsize=8, color=P["ink"])
ax.set_xlim(0, max(v for _, v in items) * 1.15)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(labelsize=8)
ax.set_title(f"C  {sum(kinds.values())} accessions pulled out of the free text, by repository",
             loc="left", fontsize=11, color=P["navy"], fontweight="bold")

fig.text(0.03, 0.015, "GEMINGA 0.16 · orchestrate_inokan.py on InoKAN_CD_nodes_3_culture_organoids.xlsx · "
         "offline resolution (no network): sizes appear once online", fontsize=8, color="#666")
fig.savefig(OUT, facecolor=fig.get_facecolor())
print("wrote", OUT)
