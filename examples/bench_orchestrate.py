"""The orchestrator's hot paths, timed — the numbers in docs/benchmarks.md.

    python examples/bench_orchestrate.py [work_dir]

Synthetic inputs only (a 5,000-row inventory, a 20,000-gene map, a 200k-cell h5ad), so it runs
anywhere in under a minute and needs no network.
"""
import os
import random
import sys
import time
import tracemalloc

import h5py
import numpy as np
import scipy.sparse as sp

import geminga as g
from geminga import harmonize as hz
from geminga import orchestrate as orc

R = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gem/bench"
os.makedirs(R, exist_ok=True)
rng = np.random.default_rng(0); random.seed(0)

def t(label, f):
    t0 = time.perf_counter(); out = f(); dt = time.perf_counter() - t0
    print(f"{label:<58} {dt*1000:9.1f} ms"); return out

# 1. describe on a 5,000-row inventory with InoKAN-like columns
hosts = ["GSE{:06d} (PRJNA{:06d})", "Zenodo 10.5281/zenodo.{:07d}; CELLxGENE", "E-MTAB-{:05d}; EGAS00001{:06d}",
         "Figshare+ {:08d}", "SCP{:04d}; DUOS-{:06d}", "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE{:06d}"]
mods = ["scRNA-seq; Perturb-seq", "scRNA-seq; CITE-seq; ECCITE-seq", "pooled CRISPR screen (FACS sort on surface marker)",
        "multiome RNA+ATAC; snRNA-seq; snATAC-seq; Perturb-seq", "scATAC-seq; CROP-seq", "T1w; T2w; T2-FLAIR"]
rows = []
for i in range(5000):
    rows.append({"dataset_id": f"ds{i}", "title": f"study {i}", "accession": random.choice(hosts).format(i, i, i), "data_url": f"https://example.org/{i}",
                 "species": random.choice(["human", "mouse", "Homo sapiens"]), "context_type": random.choice(["cell line", "organoid", "iPSC/ESC-derived"]),
                 "tissue_or_cell_type": "Jurkat T cells (anti-CD3/CD28 stimulated)", "modalities": random.choice(mods),
                 "platform": "10x 3' (dual-sgRNA CRISPRi)", "disease_state": random.choice(["T-cell leukaemia", "GBM", "healthy", "pan-cancer"]),
                 "n_cells": f"~{random.randint(1,900)},{random.randint(100,999)} high-quality cells", "n_donors_or_samples": "2 conditions",
                 "CD_protein_targets (listed)": "CD47; CD274", "perturbation_type": "CRISPRi", "perturbation_scope": "targeted",
                 "perturbation_n_genes": "39", "CD_genes_perturbed": "CD274; CD47", "CD_focus": "CD3D; CD3E; CD28; CD69; CD2",
                 "file_formats": "mtx; tsv", "access": "Open (GEO/SCP); raw reads controlled via EGA", "notes": "see also GSE000001 and the paper's supplement"})
o = orc.Orchestrator()
pieces = t("describe 5,000 rows", lambda: o.describe(rows))
t("resolve 5,000 pieces (offline)", lambda: o.resolve())
t("report()", lambda: o.report())
t("coverage 420 genes x 5,000 pieces", lambda: o.coverage(genes=[f"CD{i}" for i in range(1, 421)]))

# 2. GeneMap on an HGNC-sized table (20k genes), align 5 var lists of 18k with union/intersection/panel
genes = [f"G{i}" for i in range(20000)]
table = [{"symbol": s, "ensembl": f"ENSG{i:011d}", "entrez": str(i + 1), "alias": f"A{i}|B{i}"} for i, s in enumerate(genes)]
gm = t("GeneMap.from_table 20,000 rows", lambda: hz.GeneMap.from_table(table, symbol="symbol", ensembl="ensembl", entrez="entrez", aliases=("alias",)))
feats = {f"p{k}": [f"ENSG{i:011d}.{k}" for i in rng.permutation(20000)[:18000]] for k in range(5)}
t("align 5 x 18k union (ensembl)", lambda: gm.align(feats, key="ensembl", how="union"))
t("align 5 x 18k intersection", lambda: gm.align(feats, key="ensembl", how="intersection"))
t("align 5 x 18k panel of 2,000 (symbol)", lambda: gm.align(feats, key="symbol", panel=genes[:2000]))
t("expand 20k names", lambda: gm.expand(genes, key="ensembl"))
t("canonical x 18k (fallback path, unknown ids)", lambda: [gm.canonical(f"ENSG{i:011d}", "symbol") for i in range(30000, 48000)])

# 3. build a 200k-cell h5ad piece under a 64 MB budget: peak python memory and time
def write_h5ad(path, n_obs, n_vars, seed):
    X = sp.random(n_obs, n_vars, density=0.02, format="csr", dtype=np.float32, random_state=seed)
    with h5py.File(path, "w") as f:
        x = f.create_group("X"); x.attrs["encoding-type"] = "csr_matrix"; x.attrs["shape"] = np.array(X.shape)
        x.create_dataset("data", data=X.data); x.create_dataset("indices", data=X.indices); x.create_dataset("indptr", data=X.indptr)
        obs = f.create_group("obs"); obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=np.array([f"C{i}" for i in range(n_obs)]).astype("S"))
        d = obs.create_group("donor"); d.attrs["encoding-type"] = "categorical"
        d.create_dataset("codes", data=(np.arange(n_obs) % 7).astype(np.int8)); d.create_dataset("categories", data=np.array([f"D{i}" for i in range(7)]).astype("S"))
        var = f.create_group("var"); var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=np.array([f"ENSG{i:011d}" for i in range(n_vars)]).astype("S"))
    return X.nnz
p = f"{R}/big.h5ad"
if not os.path.exists(p):
    nnz = write_h5ad(p, 200_000, 2000, 1); print("wrote big.h5ad nnz", nnz)
b = g.Budget(ram="64MB")
o2 = orc.Orchestrator(budget=b); o2.harmonizer.genes = gm
o2.describe([{"dataset_id": "big", "accession": "GSE1", "modalities": "scRNA-seq"}]); o2.attach("big", p); o2.resolve()
bp = o2.blueprint("bench", key="ensembl", how="union")
tracemalloc.start()
pz = t("build 200k x 2k cells h5ad, 64MB budget, union", lambda: o2.build(bp, f"{R}/puzzle", overwrite=True))
cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
print(f"   python heap peak during build: {peak/2**20:.1f} MB; budget peak {b.stats()['ram']['peak_human']}; "
      f"X rows {pz.built[0]['rows']:,}; chunks {pz.built[0]['chunks']}")
