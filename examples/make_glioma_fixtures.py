"""Miniature 10x-style deliverables shaped like a 2025-26 glioma atlas: a CITE-seq + guide
multi-modal Cell Ranger h5, an mtx triple of the same, a Space Ranger (Visium) bundle, a
Xenium bundle with per-transcript parquet, and a 6-plex OME-TIFF."""
import os
import sys
import gzip
import json
import numpy as np
import h5py
import scipy.sparse as sp
import pyarrow as pa
import pyarrow.parquet as pq

out = sys.argv[1] if len(sys.argv) > 1 else "fixtures/glioma"
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(42)

# ---- one multi-modal matrix: 3000 cells; 2000 genes + 40 ADT + 24 guides ----
n_cells = 3000
genes = [f"GENE{i:04d}" for i in range(2000)]
adts = [f"ADT-{n}" for n in ("CD45","CD3","CD4","CD8","CD11b","CD14","CD16","CD19","CD20",
        "CD33","CD56","CD68","CD163","HLA-DR","PD-1","PD-L1","CTLA-4","TIM-3","LAG-3","CD44")] + \
       [f"ADT-marker{i:02d}" for i in range(20)]
guides = [f"sg{g}_{i}" for g in ("EGFR","PTEN","NF1","TP53","CDKN2A","NTC") for i in (1,2,3,4)]
feature_names = genes + adts + guides
ftypes = (["Gene Expression"] * len(genes) + ["Antibody Capture"] * len(adts)
          + ["CRISPR Guide Capture"] * len(guides))
n_feat = len(feature_names)

# GEX: sparse Poisson-ish; ADT: dense-ish; guides: ~1 guide per cell
gex = sp.random(len(genes), n_cells, density=0.04, format="csc", dtype=np.float32,
                random_state=1, data_rvs=lambda k: rng.poisson(2, k) + 1)
adt = sp.csc_matrix(rng.poisson(30, (len(adts), n_cells)).astype(np.float32))
gm = np.zeros((len(guides), n_cells), np.float32)
assign = rng.integers(0, len(guides), n_cells)
gm[assign, np.arange(n_cells)] = rng.poisson(40, n_cells) + 5
guide = sp.csc_matrix(gm)
X = sp.vstack([gex, adt, guide]).tocsc()
X.sum_duplicates()
barcodes = [f"{''.join(rng.choice(list('ACGT'), 16))}-1" for _ in range(n_cells)]

h5 = os.path.join(out, "filtered_feature_bc_matrix.h5")
with h5py.File(h5, "w") as f:
    g = f.create_group("matrix")
    g.create_dataset("data", data=X.data.astype(np.int32), compression="gzip")
    g.create_dataset("indices", data=X.indices.astype(np.int32), compression="gzip")
    g.create_dataset("indptr", data=X.indptr.astype(np.int64))
    g.create_dataset("shape", data=np.array(X.shape, np.int32))
    g.create_dataset("barcodes", data=np.array(barcodes, "S"))
    ft = g.create_group("features")
    ft.create_dataset("id", data=np.array([f"ID{i}" for i in range(n_feat)], "S"))
    ft.create_dataset("name", data=np.array(feature_names, "S"))
    ft.create_dataset("feature_type", data=np.array(ftypes, "S"))
    ft.create_dataset("genome", data=np.array(["GRCh38"] * n_feat, "S"))
print(f"  10x h5: {n_cells} cells x {n_feat} features ({X.nnz:,} nnz), {os.path.getsize(h5):,} B")

# Cell Ranger's guide-assignment table, as it ships
with open(os.path.join(out, "protospacer_calls_per_cell.csv"), "w") as fh:
    fh.write("cell_barcode,num_features,feature_call,num_umis\n")
    for bc, a in zip(barcodes, assign):
        fh.write(f"{bc},1,{guides[a]},{int(gm[a, barcodes.index(bc)])}\n")

# the same matrix as an mtx triple
md = os.path.join(out, "mtx"); os.makedirs(md, exist_ok=True)
coo = X.tocoo()
with gzip.open(os.path.join(md, "matrix.mtx.gz"), "wt") as fh:
    fh.write("%%MatrixMarket matrix coordinate integer general\n%\n")
    fh.write(f"{X.shape[0]} {X.shape[1]} {X.nnz}\n")
    for r, c, v in zip(coo.row, coo.col, coo.data):
        fh.write(f"{r+1} {c+1} {int(v)}\n")
with gzip.open(os.path.join(md, "barcodes.tsv.gz"), "wt") as fh:
    fh.write("\n".join(barcodes) + "\n")
with gzip.open(os.path.join(md, "features.tsv.gz"), "wt") as fh:
    fh.write("\n".join(f"ID{i}\t{n}\t{t}" for i, (n, t) in enumerate(zip(feature_names, ftypes))) + "\n")
print(f"  mtx triple: {os.path.getsize(os.path.join(md,'matrix.mtx.gz')):,} B")

# ---- Space Ranger (Visium) bundle: 1500 spots on a hex-ish grid over a 4000px image ----
vd = os.path.join(out, "visium"); os.makedirs(os.path.join(vd, "spatial"), exist_ok=True)
n_spots = 1500
rows, cols = np.divmod(np.arange(n_spots), 50)
px_col = (cols * 70 + (rows % 2) * 35 + 300).astype(int)
px_row = (rows * 60 + 300).astype(int)
vg = sp.random(len(genes), n_spots, density=0.06, format="csc", dtype=np.float32,
               random_state=2, data_rvs=lambda k: rng.poisson(3, k) + 1).tocsc()
vbc = [f"{''.join(rng.choice(list('ACGT'), 16))}-1" for _ in range(n_spots)]
with h5py.File(os.path.join(vd, "filtered_feature_bc_matrix.h5"), "w") as f:
    g = f.create_group("matrix")
    g.create_dataset("data", data=vg.data.astype(np.int32)); g.create_dataset("indices", data=vg.indices.astype(np.int32))
    g.create_dataset("indptr", data=vg.indptr.astype(np.int64)); g.create_dataset("shape", data=np.array(vg.shape, np.int32))
    g.create_dataset("barcodes", data=np.array(vbc, "S"))
    ft = g.create_group("features")
    ft.create_dataset("id", data=np.array([f"ID{i}" for i in range(len(genes))], "S"))
    ft.create_dataset("name", data=np.array(genes, "S"))
    ft.create_dataset("feature_type", data=np.array(["Gene Expression"] * len(genes), "S"))
    ft.create_dataset("genome", data=np.array(["GRCh38"] * len(genes), "S"))
in_tissue = ((px_col - 2000)**2 + (px_row - 1200)**2 < 1500**2).astype(int)
with open(os.path.join(vd, "spatial", "tissue_positions.csv"), "w") as fh:
    fh.write("barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n")
    for i in range(n_spots):
        fh.write(f"{vbc[i]},{in_tissue[i]},{rows[i]},{cols[i]},{px_row[i]},{px_col[i]}\n")
json.dump({"spot_diameter_fullres": 65.0, "tissue_hires_scalef": 0.5, "tissue_lowres_scalef": 0.15,
           "fiducial_diameter_fullres": 100.0}, open(os.path.join(vd, "spatial", "scalefactors_json.json"), "w"))
import tifffile
he = (rng.random((2000, 2000, 3)) * 60 + 180).astype(np.uint8)
yy, xx = np.mgrid[0:2000, 0:2000]
he[((xx - 1000)**2 + (yy - 600)**2) < 750**2] = [200, 140, 170]
tifffile.imwrite(os.path.join(vd, "spatial", "tissue_hires_image.tif"), he)
print(f"  visium: {n_spots} spots, {int(in_tissue.sum())} in tissue")

# ---- Xenium bundle: transcripts.parquet sorted by x (so pruning pays), cells, morphology ----
xd = os.path.join(out, "xenium"); os.makedirs(xd, exist_ok=True)
n_tx = 400_000
panel = [f"GENE{i:04d}" for i in range(300)]
tx = pa.table({
    "transcript_id": np.arange(n_tx, dtype=np.uint64),
    "cell_id": np.array([f"cell_{i}" for i in rng.integers(1, 8000, n_tx)]),
    "feature_name": np.array(panel)[rng.integers(0, 300, n_tx)],
    "x_location": rng.uniform(0, 6000, n_tx).astype(np.float32),
    "y_location": rng.uniform(0, 4500, n_tx).astype(np.float32),
    "z_location": rng.uniform(0, 12, n_tx).astype(np.float32),
    "qv": rng.uniform(15, 40, n_tx).astype(np.float32),
})
pq.write_table(tx.sort_by("x_location"), os.path.join(xd, "transcripts.parquet"),
               row_group_size=20_000, write_statistics=True)
pq.write_table(pa.table({"cell_id": np.array([f"cell_{i}" for i in range(1, 8000)]),
                         "x_centroid": rng.uniform(0, 6000, 7999).astype(np.float32),
                         "y_centroid": rng.uniform(0, 4500, 7999).astype(np.float32),
                         "transcript_counts": rng.poisson(50, 7999).astype(np.int32)}),
               os.path.join(xd, "cells.parquet"))
json.dump({"panel_name": "Xenium Human Brain Panel (synthetic)", "num_cells": 7999,
           "region_name": "GBM_core", "pixel_size": 0.2125}, open(os.path.join(xd, "experiment.xenium"), "w"))
morph = (rng.random((2250, 3000)) * 800).astype(np.uint16)
tifffile.imwrite(os.path.join(xd, "morphology.ome.tif"), morph, tile=(512, 512))
print(f"  xenium: {n_tx:,} transcripts, {os.path.getsize(os.path.join(xd,'transcripts.parquet')):,} B parquet")

# ---- 6-plex CODEX-like OME-TIFF (C, Y, X) ----
chans = ["DAPI", "CD45", "CD68", "GFAP", "SOX2", "PD-L1"]
stack = (rng.random((6, 1500, 1500)) * 500).astype(np.uint16)
stack[3][((xx[:1500,:1500] - 700)**2 + (yy[:1500,:1500] - 800)**2) < 400**2] += 3000
ome = ('<?xml version="1.0"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
       '<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" '
       'SizeC="6" SizeT="1" SizeX="1500" SizeY="1500" SizeZ="1">'
       + "".join(f'<Channel ID="Channel:0:{i}" Name="{c}" SamplesPerPixel="1"/>' for i, c in enumerate(chans))
       + '<TiffData/></Pixels></Image></OME>')
tifffile.imwrite(os.path.join(out, "codex_6plex.ome.tif"), stack, tile=(512, 512),
                 metadata={"axes": "CYX", "Channel": {"Name": chans}}, ome=True)
print("  codex: 6 channels x 1500x1500")
tot = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs)
print(f"fixtures in {out}: {tot:,} bytes")
