"""A miniature Stereo-seq chip: a GEF with a real bin pyramid, the matching GEM, and a
registered ssDNA-like image. Shaped like SAW output, small enough to ship."""
import os
import sys
import numpy as np
import h5py

out = sys.argv[1] if len(sys.argv) > 1 else "fixtures"
os.makedirs(out, exist_ok=True)
SIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 4000     # chip side in bin-1 units
NGENE = 300
BINS = (1, 10, 20, 50, 100, 200)
rng = np.random.default_rng(17)

genes = np.array([f"Gene{i:04d}" for i in range(NGENE)])
# two tissue lobes plus background, so zooming out shows structure
yy, xx = np.mgrid[0:SIDE, 0:SIDE]
field = np.zeros((SIDE, SIDE), np.float32)
for cy, cx, r, amp in [(SIDE*0.35, SIDE*0.38, SIDE*0.16, 1.0), (SIDE*0.66, SIDE*0.62, SIDE*0.10, 1.5)]:
    field += amp * np.exp(-(((yy-cy)**2 + (xx-cx)**2) / (2*r*r)))
field += 0.02
p = (field / field.sum()).ravel()

N = 3_000_000
idx = rng.choice(p.size, size=N, p=p)
ys, xs = np.divmod(idx, SIDE)
xs = xs.astype(np.uint32); ys = ys.astype(np.uint32)
# a few genes localised to one lobe, the rest spread — so gene selection is visible
gid = rng.integers(0, NGENE, N)
lobe2 = ((ys - SIDE*0.66)**2 + (xs - SIDE*0.62)**2) < (SIDE*0.12)**2
gid[lobe2 & (rng.random(N) < 0.5)] = 7
counts = (rng.geometric(0.45, N)).astype(np.uint8)

gef = os.path.join(out, "chip.gef")
with h5py.File(gef, "w") as f:
    grp = f.create_group("geneExp")
    grp.attrs["resolution"] = 500
    f.attrs["StereoChip"] = np.bytes_("SSMINI01")
    for b in BINS:
        bx, by = (xs // b) * b, (ys // b) * b
        key = bx.astype(np.int64) * SIDE + by + gid.astype(np.int64) * SIDE * SIDE
        order = np.argsort(gid, kind="stable")          # gene-contiguous, as SAW writes it
        bx, by, cc, gg = bx[order], by[order], counts[order], gid[order]
        # collapse duplicate (gene, binned x, y)
        keys = gg.astype(np.int64) * SIDE * SIDE + bx.astype(np.int64) * SIDE + by
        uniq, inv = np.unique(keys, return_inverse=True)
        agg = np.zeros(uniq.size, np.int64)
        np.add.at(agg, inv, cc)
        ux = ((uniq % (SIDE * SIDE)) // SIDE).astype(np.uint32)
        uy = (uniq % SIDE).astype(np.uint32)
        ug = (uniq // (SIDE * SIDE)).astype(np.int64)
        dt = np.dtype([("x", "<u4"), ("y", "<u4"), ("count", "<u2")])
        rec = np.zeros(uniq.size, dt)
        rec["x"], rec["y"], rec["count"] = ux, uy, np.clip(agg, 0, 65535)
        g = grp.create_group(f"bin{b}")
        d = g.create_dataset("expression", data=rec, compression="gzip", compression_opts=1)
        d.attrs["minX"] = np.uint32(0); d.attrs["minY"] = np.uint32(0)
        # SAW writes maxX/maxY as the largest coordinate (inclusive), not one past it
        d.attrs["maxX"] = np.uint32(SIDE - 1); d.attrs["maxY"] = np.uint32(SIDE - 1)
        cnt = np.bincount(ug, minlength=NGENE).astype(np.uint32)
        off = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.uint32)
        gdt = np.dtype([("gene", "S32"), ("offset", "<u4"), ("count", "<u4")])
        grec = np.zeros(NGENE, gdt)
        grec["gene"] = np.char.encode(genes); grec["offset"] = off; grec["count"] = cnt
        g.create_dataset("gene", data=grec, compression="gzip", compression_opts=1)
        print(f"  bin{b:<4} {uniq.size:>9,} observations")
print(f"gef: {gef} ({os.path.getsize(gef):,} B)")

gem = os.path.join(out, "chip.gem")
sub = slice(None, 600_000)
with open(gem, "w") as fh:
    fh.write("#FileFormat=GEMv0.1\n#SortedBy=None\n#BinSize=1\n")
    fh.write("#StereoChip=SSMINI01\n#OffsetX=0\n#OffsetY=0\n")
    fh.write("geneID\tx\ty\tMIDCount\n")
    for gi, x, y, c in zip(gid[sub], xs[sub], ys[sub], counts[sub]):
        fh.write(f"{genes[gi]}\t{x}\t{y}\t{c}\n")
print(f"gem: {gem} ({os.path.getsize(gem):,} B)")

import tifffile
S = 2048
sy, sx = np.mgrid[0:S, 0:S]
img = np.zeros((S, S), np.uint16)
for cy, cx, r, amp in [(S*0.35, S*0.38, S*0.16, 9000), (S*0.66, S*0.62, S*0.10, 12000)]:
    img += (amp * np.exp(-(((sy-cy)**2 + (sx-cx)**2) / (2*r*r)))).astype(np.uint16)
img += (rng.random((S, S)) * 300).astype(np.uint16)
tif = os.path.join(out, "chip_ssdna.tif")
tifffile.imwrite(tif, img, tile=(512, 512), photometric="minisblack")
print(f"image: {tif} ({os.path.getsize(tif):,} B), scale = {S/SIDE:.3f} px per chip unit")
