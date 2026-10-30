"""A glioblastoma-structured Stereo-seq phantom.

Not data. A designed tissue whose *layout* follows the textbook anatomy of a glioblastoma
section, so that figures made from it show the things a glioma reader looks for — and so that
nobody mistakes it for a measurement. Every figure built from it is labelled as such.

Regions, from the centre out:

  necrotic core            few transcripts, little gene diversity
  pseudopalisading rim     hypoxia / MES-like programme: VEGFA, CA9, HILPDA, ADM, NDRG1
  cellular tumour          Neftel-style states in patches — NPC-like (DLL3, SOX4),
                           OPC-like (OLIG2, PDGFRA), AC-like (GFAP, AQP4); SOX2 throughout
  microvascular prolif.    clusters of PECAM1, VWF, CD34 within the tumour
  myeloid niches           CD163, AIF1, CD68, perivascular and perinecrotic
  infiltrated margin       normal brain (SNAP25, SYT1 neurons; MBP, PLP1 oligodendrocytes)
                           with a thinning scatter of OLIG2/SOX2 tumour cells

    python examples/make_glioma_phantom.py fixtures/gbm
"""
import os
import sys

import h5py
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "fixtures/gbm"
SIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
N = int(sys.argv[3]) if len(sys.argv) > 3 else 3_000_000
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(2026)

PROGRAMS = {
    "hypoxia":  ["VEGFA", "CA9", "HILPDA", "ADM", "NDRG1", "BNIP3", "PGK1", "SLC2A1"],
    "npc":      ["DLL3", "SOX4", "SOX11", "STMN2"],
    "opc":      ["OLIG2", "PDGFRA", "OLIG1", "BCAN"],
    "ac":       ["GFAP", "AQP4", "CLU", "SPARCL1"],
    "stem":     ["SOX2", "NES", "PTPRZ1"],
    "vessel":   ["PECAM1", "VWF", "CD34", "CLDN5", "COL4A1"],
    "myeloid":  ["CD163", "AIF1", "CD68", "C1QA", "TYROBP"],
    "neuron":   ["SNAP25", "SYT1", "RBFOX3", "GAD1"],
    "oligo":    ["MBP", "PLP1", "MOG", "MOBP"],
    "cycling":  ["MKI67", "TOP2A"],
}
filler = [f"G{i:04d}" for i in range(260)]
genes = [g for gs in PROGRAMS.values() for g in gs] + filler
gidx = {g: i for i, g in enumerate(genes)}

c = np.array([SIDE * 0.47, SIDE * 0.52])            # tumour centre (x, y)
R_nec, R_rim, R_tum, R_mar = SIDE * 0.07, SIDE * 0.12, SIDE * 0.30, SIDE * 0.44

# positions: denser in the tumour, sparse in the necrosis, background brain elsewhere
xs = rng.uniform(0, SIDE, N * 2)
ys = rng.uniform(0, SIDE, N * 2)
d = np.hypot(xs - c[0], ys - c[1])
wobble = 1 + 0.12 * np.sin(np.arctan2(ys - c[1], xs - c[0]) * 5)   # an irregular border
dn = d / wobble
density = np.where(dn < R_nec, 0.12,
          np.where(dn < R_rim, 1.0,
          np.where(dn < R_tum, 0.8,
          np.where(dn < R_mar, 0.55, 0.35))))
keep = rng.random(N * 2) < density / density.max()
xs, ys, dn = xs[keep][:N], ys[keep][:N], dn[keep][:N]
n = xs.size

# region labels
region = np.select([dn < R_nec, dn < R_rim, dn < R_tum, dn < R_mar],
                   ["necrosis", "rim", "tumour", "margin"], "brain")

# Neftel-state patches inside the cellular tumour: a coarse random field
gx = (xs / (SIDE / 7)).astype(int); gy = (ys / (SIDE / 7)).astype(int)
state_field = rng.integers(0, 3, (8, 8))
state = np.array(["npc", "opc", "ac"])[state_field[gy.clip(0, 7), gx.clip(0, 7)]]

# vascular clusters and myeloid niches
vessels = c + rng.normal(0, R_tum * 0.55, (26, 2))
vd = np.min(np.hypot(xs[:, None] - vessels[None, :, 0], ys[:, None] - vessels[None, :, 1]), 1)
near_vessel = vd < SIDE * 0.012

def pick(prog):
    g = PROGRAMS[prog]
    return np.array([gidx[x] for x in g])[rng.integers(0, len(g), n)]

gid = rng.integers(len(genes) - len(filler), len(genes), n)          # background filler
u = rng.random(n)
is_tum = region == "tumour"
# cellular tumour: state programme, SOX2 everywhere, some cycling
m = is_tum & (u < 0.55)
gid[m] = np.where(state[m] == "npc", pick("npc")[m],
         np.where(state[m] == "opc", pick("opc")[m], pick("ac")[m]))
m = is_tum & (u >= 0.55) & (u < 0.70); gid[m] = pick("stem")[m]
m = is_tum & (u >= 0.70) & (u < 0.76); gid[m] = pick("cycling")[m]
# hypoxic rim
m = (region == "rim") & (u < 0.62); gid[m] = pick("hypoxia")[m]
m = (region == "rim") & (u >= 0.62) & (u < 0.78); gid[m] = pick("myeloid")[m]
# necrosis: sparse, mostly filler plus some myeloid debris
m = (region == "necrosis") & (u < 0.30); gid[m] = pick("myeloid")[m]
# margin: normal brain with thinning infiltration
m = (region == "margin") & (u < 0.35); gid[m] = pick("neuron")[m]
m = (region == "margin") & (u >= 0.35) & (u < 0.62); gid[m] = pick("oligo")[m]
m = (region == "margin") & (u >= 0.62) & (u < 0.72); gid[m] = pick("opc")[m]
m = (region == "brain") & (u < 0.45); gid[m] = pick("neuron")[m]
m = (region == "brain") & (u >= 0.45) & (u < 0.80); gid[m] = pick("oligo")[m]
# microvascular proliferation and perivascular myeloid, wherever vessels are
m = near_vessel & (u < 0.55) & (region != "brain"); gid[m] = pick("vessel")[m]
m = near_vessel & (u >= 0.85) & (region != "brain"); gid[m] = pick("myeloid")[m]

counts = rng.geometric(np.where(region == "necrosis", 0.7, 0.42), n).astype(np.int64)
xs = xs.astype(np.uint32); ys = ys.astype(np.uint32)

BINS = (1, 10, 20, 50, 100, 200)
gef = os.path.join(out, "gbm_phantom.gef")
with h5py.File(gef, "w") as f:
    grp = f.create_group("geneExp")
    grp.attrs["resolution"] = 500
    f.attrs["StereoChip"] = np.bytes_("GBM-PHANTOM")
    for b in BINS:
        bx, by = (xs // b) * b, (ys // b) * b
        keys = gid.astype(np.int64) * SIDE * SIDE + bx.astype(np.int64) * SIDE + by
        uniq, inv = np.unique(keys, return_inverse=True)
        agg = np.zeros(uniq.size, np.int64); np.add.at(agg, inv, counts)
        rec = np.zeros(uniq.size, np.dtype([("x", "<u4"), ("y", "<u4"), ("count", "<u2")]))
        rec["x"] = ((uniq % (SIDE * SIDE)) // SIDE).astype(np.uint32)
        rec["y"] = (uniq % SIDE).astype(np.uint32)
        rec["count"] = np.clip(agg, 0, 65535)
        ug = (uniq // (SIDE * SIDE)).astype(np.int64)
        g = grp.create_group(f"bin{b}")
        ds = g.create_dataset("expression", data=rec, compression="gzip", compression_opts=1)
        for k, v in (("minX", 0), ("minY", 0), ("maxX", SIDE - 1), ("maxY", SIDE - 1)):   # inclusive, as SAW
            ds.attrs[k] = np.uint32(v)
        cnt = np.bincount(ug, minlength=len(genes)).astype(np.uint32)
        off = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.uint32)
        grec = np.zeros(len(genes), np.dtype([("gene", "S32"), ("offset", "<u4"), ("count", "<u4")]))
        grec["gene"] = np.char.encode(np.array(genes)); grec["offset"] = off; grec["count"] = cnt
        g.create_dataset("gene", data=grec, compression="gzip", compression_opts=1)
        print(f"  bin{b:<4} {uniq.size:>10,} observations")
print(f"gef: {gef} ({os.path.getsize(gef):,} B), {len(genes)} genes")

# a registered ssDNA-like image: nuclei density follows the tissue, necrosis is pale
import tifffile
S = 2048
sy, sx = np.mgrid[0:S, 0:S] * (SIDE / S)
dd = np.hypot(sx - c[0], sy - c[1])
img = np.where(dd < R_nec, 900, np.where(dd < R_rim, 7000, np.where(dd < R_tum, 5200,
               np.where(dd < R_mar, 3600, 2600)))).astype(np.float32)
img += rng.normal(0, 700, (S, S)); img = img[::-1].clip(0, 65535).astype(np.uint16)
tif = os.path.join(out, "gbm_phantom_ssdna.tif")
tifffile.imwrite(tif, img, tile=(512, 512), photometric="minisblack")
print(f"image: {tif}, scale {S / SIDE:.3f} px per chip unit")
np.save(os.path.join(out, "regions.npy"),
        np.array([c[0], c[1], R_nec, R_rim, R_tum, R_mar, SIDE], np.float64))
