"""Out-of-core preprocessing: the passes must agree with the dense computation, under the budget."""
import os

import h5py
import numpy as np
import pytest
import scipy.sparse as sp

import geminga as g
from geminga.prep import Prep

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")


def _h5ad(path, X, var_names, obs_names=None):
    X = sp.csr_matrix(X)
    with h5py.File(path, "w") as f:
        f.attrs["encoding-type"] = "anndata"
        x = f.create_group("X"); x.attrs["encoding-type"] = "csr_matrix"; x.attrs["shape"] = np.array(X.shape)
        x.create_dataset("data", data=X.data.astype(np.float32)); x.create_dataset("indices", data=X.indices.astype(np.int32))
        x.create_dataset("indptr", data=X.indptr.astype(np.int64))
        obs = f.create_group("obs"); obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=np.array(obs_names if obs_names is not None else
                                                   [f"C{i}" for i in range(X.shape[0])]).astype("S"))
        var = f.create_group("var"); var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=np.array(var_names).astype("S"))


def _dense_reference(X, n_top, n_comps, min_genes, max_pct_mito, min_cells, mito, target_sum=1e4, max_value=10):
    """The same pipeline, dense and in float64: what scanpy does in memory."""
    X = X.astype(np.float64)
    tot = X.sum(1); n_genes = (X > 0).sum(1)
    pct_mito = np.where(tot > 0, 100 * X[:, mito].sum(1) / np.maximum(tot, 1e-12), 0)
    keep = (n_genes >= min_genes) & (pct_mito <= max_pct_mito)
    norm = X / np.maximum(tot, 1e-12)[:, None] * target_sum
    lg = np.log1p(norm)
    mean = norm.mean(0); var = norm.var(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = np.where(mean > 0, var / mean, 0); disp_log = np.where(disp > 0, np.log(disp), np.nan)
    mean_log = np.log1p(mean)
    cand = ((X > 0).sum(0) >= min_cells) & np.isfinite(disp_log) & (mean > 0)
    edges = np.linspace(mean_log[cand].min(), mean_log[cand].max(), 21)
    bins = np.clip(np.searchsorted(edges, mean_log, side="right") - 1, 0, 19)
    nd = np.full(X.shape[1], np.nan)
    for b in range(20):
        sel = cand & (bins == b)
        if sel.sum() > 1:
            d = disp_log[sel]; sd = d.std(ddof=1); nd[sel] = (d - d.mean()) / (sd if sd > 0 else 1)
        elif sel.sum() == 1:
            nd[sel] = 0
    order = np.argsort(-np.nan_to_num(nd, nan=-np.inf), kind="stable"); order = order[cand[order]]
    top = np.sort(order[:n_top])
    mu = lg[:, top].mean(0); sd = lg[:, top].std(0, ddof=1); sd[sd == 0] = 1
    Z = np.clip((lg[:, top] - mu) / sd, -max_value, max_value)[keep]
    C = np.cov(Z, rowvar=False); ev, V = np.linalg.eigh(C); o = np.argsort(ev)[::-1][:n_comps]
    comps = V[:, o].T
    for i in range(comps.shape[0]):
        j = np.argmax(np.abs(comps[i])); comps[i] *= np.sign(comps[i, j]) or 1
    Xp = (Z - Z.mean(0)) @ comps.T
    return {"keep": keep, "hvg": top, "X_pca": Xp, "ratio": ev[o] / np.trace(C), "pct_mito": pct_mito}


@pytest.fixture(scope="module")
def clustered(tmp_path_factory):
    """Three clusters of cells with distinct marker genes, some mito genes, a few bad cells."""
    rng = np.random.default_rng(7)
    n, m = 1500, 400
    base = rng.poisson(1.0, (n, m))
    labels = rng.integers(0, 3, n)
    for c in range(3):
        base[labels == c, c * 40:(c + 1) * 40] += rng.poisson(6.0, ((labels == c).sum(), 40))
    base[:, 390:] += rng.poisson(0.5, (n, 10))                   # "mito" genes
    bad = rng.choice(n, 30, replace=False)
    base[bad, 390:] += 80                                          # high-mito cells
    base[rng.choice(n, 20, replace=False), :] = 0                  # empty cells
    names = [f"G{i}" for i in range(390)] + [f"MT-ND{i}" for i in range(10)]
    p = str(tmp_path_factory.mktemp("prep") / "clusters.h5ad")
    _h5ad(p, base, names)
    return p, base.astype(np.float64), labels, np.array([x.startswith("MT-") for x in names])


def test_passes_match_the_dense_computation(clustered):
    p, X, labels, mito = clustered
    b = g.Budget(ram="2MB")
    pp = Prep(p, budget=b)
    res = pp.run(n_top=100, n_comps=10, min_genes=20, max_pct_mito=25, min_cells=3)
    ref = _dense_reference(X, 100, 10, 20, 25, 3, mito)
    assert np.array_equal(res.kept, ref["keep"]) and res.kept.sum() < len(X)         # filters bit
    assert np.allclose(res.obs["pct_mito"], ref["pct_mito"], atol=1e-9)
    assert set(res.hvg_index) == set(ref["hvg"])
    mine = np.asarray(res.X_pca)[res.kept]
    assert np.abs(mine - ref["X_pca"]).max() < 1e-3 * np.abs(ref["X_pca"]).max()
    assert np.allclose(res.explained_variance_ratio, ref["ratio"], atol=1e-6)
    assert res.report["passes"] == 3 and b.stats()["ram"]["denials"] == 0
    assert b.stats()["ram"]["peak"] <= 2 << 20


def test_pca_separates_the_planted_clusters(clustered):
    p, X, labels, _ = clustered
    res = Prep(p, budget=g.Budget(ram="4MB")).run(n_top=120, n_comps=5, min_genes=20, max_pct_mito=25)
    Z = np.asarray(res.X_pca)[res.kept][:, :2]
    lab = labels[res.kept]
    cents = np.array([Z[lab == c].mean(0) for c in range(3)])
    within = np.mean([np.linalg.norm(Z[lab == c] - cents[c], axis=1).mean() for c in range(3)])
    between = np.mean([np.linalg.norm(cents[i] - cents[j]) for i in range(3) for j in range(i + 1, 3)])
    assert between > 3 * within


def test_memmap_output_and_write_and_anndata(clustered, tmp_path):
    p, X, labels, _ = clustered
    out = str(tmp_path / "X_pca.npy")
    res = Prep(p, budget=g.Budget(ram="4MB")).run(n_top=80, n_comps=8, min_genes=20, out=out)
    assert isinstance(res.X_pca, np.memmap) and res.report["out_lease"] == "memmap"
    d = res.write(str(tmp_path / "prep"))
    assert sorted(os.listdir(d)) == ["X_pca.npy", "obs.parquet", "prep.json", "var.parquet"]
    assert np.load(os.path.join(d, "X_pca.npy")).shape == (int(res.kept.sum()), 8)
    pytest.importorskip("anndata")
    a = res.to_anndata()
    assert a.n_obs == int(res.kept.sum()) and a.obsm["X_pca"].shape == (a.n_obs, 8)
    assert a.var["highly_variable"].sum() == 80 and a.varm["PCs"].shape == (400, 8)
    assert np.abs(a.varm["PCs"][res.hvg_index] - res.components.T).max() < 1e-6   # float32 storage


def test_fixture_h5ad_and_tenx_h5_run_under_a_small_budget():
    h5ad = os.path.join(FIX, "sc.h5ad")
    if not os.path.exists(h5ad):
        pytest.skip("no sc.h5ad fixture")
    b = g.Budget(ram="16MB")
    res = Prep(h5ad, budget=b).run(n_top=300, n_comps=10, min_genes=10, max_pct_mito=100)
    assert res.X_pca.shape == (20_000, 10) and b.stats()["ram"]["peak"] <= 16 << 20 and b.stats()["ram"]["denials"] == 0
    tenx = os.path.join(FIX, "glioma", "filtered_feature_bc_matrix.h5")
    if os.path.exists(tenx):
        pp = Prep(tenx, budget=g.Budget(ram="16MB"))
        assert pp.src.kind == "10x-h5"
        r2 = pp.run(n_top=200, n_comps=5, min_genes=5, max_pct_mito=100)
        assert r2.X_pca.shape[1] == 5 and r2.kept.any()


def test_scanpy_agrees_when_installed(clustered):
    sc = pytest.importorskip("scanpy")
    ad = pytest.importorskip("anndata")
    p, X, labels, mito = clustered
    assert ad is not None
    res = Prep(p, budget=g.Budget(ram="4MB")).run(n_top=100, n_comps=10, min_genes=20, max_pct_mito=25, min_cells=3)
    import pandas as pd
    a = ad.AnnData(X=sp.csr_matrix(X.astype(np.float32)),
                   var=pd.DataFrame(index=[f"G{i}" for i in range(390)] + [f"MT-ND{i}" for i in range(10)]))
    a.var["mt"] = a.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, inplace=True)
    assert np.allclose(a.obs["total_counts"].values, res.obs["total_counts"])
    # scanpy reports NaN for zero-count cells where GEMINGA reports 0
    assert np.allclose(np.nan_to_num(a.obs["pct_counts_mt"].values), res.obs["pct_mito"], atol=1e-4)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=100, flavor="seurat", n_bins=20)
    # the gene *statistics* are over all cells in both; the sets should agree almost entirely
    sc_hvg = set(np.flatnonzero(a.var["highly_variable"].values))
    assert len(sc_hvg & set(res.hvg_index)) >= 90


def test_errors_are_clear():
    with pytest.raises(ValueError, match="h5ad"):
        Prep("nothing.txt")
    pp = Prep.__new__(Prep)
    pp._stats = None
    with pytest.raises(RuntimeError, match="qc"):
        pp.filter_cells()
