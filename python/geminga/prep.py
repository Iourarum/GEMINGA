"""The standard single-cell preprocessing, out of core, under the budget.

What a PhD student actually runs on a count matrix is the same five steps every time: QC
metrics, cell and gene filters, library-size normalisation with log1p, highly variable genes,
PCA. scanpy does them in memory, which is why a 2-million-cell atlas does not open on a laptop.
Every one of those steps is a sum over cells, so they can be done in passes over budget-sized
blocks — and once the cells are 50 numbers each instead of 30,000, neighbours, UMAP and
clustering fit anywhere.

This module does exactly that and nothing more:

1. **pass 1** — per-cell totals, genes detected, mitochondrial and ribosomal fractions; per-gene
   cells detected, and the sums needed for dispersion (on normalised counts) and for scaling
   (on log1p values);
2. **filters and HVG** — from those sums, no new pass: Seurat-flavour dispersion binned by mean,
   top *n*;
3. **pass 2** — the HVG × HVG covariance of the scaled, clipped values (2,000 genes → a 32 MB
   matrix, whatever the number of cells);
4. **pass 3** — projection onto the leading eigenvectors, written block by block.

The numbers match a dense scanpy-style computation to float precision on the same cells (the
test says how closely). Memory is the reader's block plus one dense sub-block of the HVG
columns, sized to the budget; nothing grows with the number of cells except the output, which
can be a memory-mapped file.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ._core import Budget, human

__all__ = ["Prep", "PrepResult", "prep"]


# ---------------------------------------------------------------------------
# Sources: anything that yields obs × var CSR blocks
# ---------------------------------------------------------------------------

class _Source:
    """Blocks of cells as CSR (cells × genes), with names. h5ad, Cell Ranger h5, or an mtx
    triple (converted once beside itself, because text cannot be read by cell range)."""

    def __init__(self, target, budget: Optional[Budget], chunk_bytes=None, layer: Optional[str] = None,
                 modality: str = "RNA"):
        self.budget = budget
        self.kind = ""
        self.path = str(target) if isinstance(target, (str, os.PathLike)) else ""
        kw = {"budget": budget}
        if chunk_bytes:
            kw["chunk_bytes"] = chunk_bytes
        low = self.path.lower()
        if self.path and low.endswith(".h5ad"):
            from .formats import H5adReader
            self.kind = "h5ad"
            self._make = lambda: H5adReader(self.path, layer=layer, **kw)
            r = self._make()
            self.n_obs, self.n_vars = r.n_obs, r.n_vars
            self.var_names = np.asarray(r.var_names).astype(str)
            self.obs_names = _h5ad_obs_names(r.f, self.n_obs)
            r.close()
        elif self.path and low.endswith((".h5", ".hdf5")):
            from .tenx import TenXMatrix
            self.kind = "10x-h5"
            mod = modality
            probe = TenXMatrix(self.path)
            if mod not in probe.modalities:
                mod = next(iter(probe.modalities))
            self.n_obs, self.n_vars = probe.n_cells, probe.modalities[mod].n_features
            self.var_names = np.asarray(probe.features(mod)).astype(str)
            self.obs_names = np.asarray(probe.barcodes).astype(str)
            probe.close()
            self._make = lambda: TenXMatrix(self.path, modality=mod, **kw)
        elif self.path and (os.path.isdir(self.path) or low.endswith((".mtx", ".mtx.gz"))):
            from .tenx import MtxTriple
            d = self.path if os.path.isdir(self.path) else os.path.dirname(self.path)
            h5 = os.path.join(d, "geminga_matrix.h5")
            if not os.path.exists(h5):
                MtxTriple(self.path, budget=budget).to_h5(h5)
            self.__init__(h5, budget, chunk_bytes, layer, modality)
            self.kind = "mtx→h5"
        else:
            raise ValueError(f"{target!r}: give an .h5ad, a Cell Ranger .h5, or an mtx triple directory")

    def blocks(self):
        r = self._make()
        try:
            if self.kind == "10x-h5":
                for blk in r:
                    yield blk["cell_start"], blk["cell_end"], blk["X"].T.tocsr()
            else:
                for blk in r:
                    X = blk["X"]
                    yield blk["obs_start"], blk["obs_end"], (X.tocsr() if hasattr(X, "tocsr") else _dense_to_csr(X))
        finally:
            r.close()


def _dense_to_csr(X):
    import scipy.sparse as sp
    return sp.csr_matrix(np.asarray(X))


def _h5ad_obs_names(f, n: int) -> np.ndarray:
    idx = f["obs"].attrs.get("_index", "_index")
    idx = idx.decode() if isinstance(idx, bytes) else idx
    if f"obs/{idx}" in f:
        v = f[f"obs/{idx}"][:]
        return np.char.decode(v, "utf-8") if v.dtype.kind == "S" else np.asarray(v).astype(str)
    return np.array([str(i) for i in range(n)])


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class PrepResult:
    """What the passes produced. `obs`/`var` are dicts of equal-length arrays (tables)."""
    obs: dict
    var: dict
    X_pca: Optional[np.ndarray] = None
    components: Optional[np.ndarray] = None            # (n_comps, n_hvg)
    explained_variance: Optional[np.ndarray] = None
    explained_variance_ratio: Optional[np.ndarray] = None
    hvg_index: Optional[np.ndarray] = None             # column indices of the HVGs, in order
    params: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return int(len(next(iter(self.obs.values())))) if self.obs else 0

    @property
    def kept(self) -> np.ndarray:
        return np.asarray(self.obs["kept"], bool)

    def obs_table(self):
        import pyarrow as pa
        return pa.table({k: pa.array(np.asarray(v)) for k, v in self.obs.items()})

    def var_table(self):
        import pyarrow as pa
        return pa.table({k: pa.array(np.asarray(v)) for k, v in self.var.items()})

    def to_anndata(self, kept_only: bool = True):
        """obs, var and obsm['X_pca'] — no X. scanpy's neighbors/umap/leiden run on this."""
        import anndata as ad
        import pandas as pd
        obs = pd.DataFrame({k: np.asarray(v) for k, v in self.obs.items()})
        obs.index = obs["obs_names"].astype(str)
        var = pd.DataFrame({k: np.asarray(v) for k, v in self.var.items()})
        var.index = var["var_names"].astype(str)
        keep = self.kept if kept_only else np.ones(self.n_obs, bool)
        a = ad.AnnData(obs=obs[keep], var=var)
        if self.X_pca is not None:
            a.obsm["X_pca"] = np.asarray(self.X_pca)[keep] if not kept_only else np.asarray(self.X_pca)[keep]
        a.uns["pca"] = {"variance": self.explained_variance, "variance_ratio": self.explained_variance_ratio,
                        "params": self.params}
        a.varm["PCs"] = np.zeros((len(var), 0 if self.components is None else self.components.shape[0]), np.float32)
        if self.components is not None:
            a.varm["PCs"][self.hvg_index] = self.components.T.astype(np.float32)
        return a

    def write(self, path: str) -> str:
        """obs.parquet, var.parquet, X_pca.npy (kept cells, in order) and prep.json."""
        import pyarrow.parquet as pq
        os.makedirs(path, exist_ok=True)
        pq.write_table(self.obs_table(), os.path.join(path, "obs.parquet"))
        pq.write_table(self.var_table(), os.path.join(path, "var.parquet"))
        if self.X_pca is not None:
            np.save(os.path.join(path, "X_pca.npy"), np.asarray(self.X_pca)[self.kept])
        with open(os.path.join(path, "prep.json"), "w", encoding="utf-8") as fh:
            json.dump({"schema": "geminga.prep/1", "params": self.params, "report": self.report,
                       "explained_variance_ratio": None if self.explained_variance_ratio is None
                       else self.explained_variance_ratio.tolist()}, fh, indent=2, default=str)
        return path

    def __repr__(self) -> str:
        k = int(self.kept.sum()) if self.obs else 0
        pcs = "" if self.X_pca is None else f", X_pca {self.X_pca.shape[1]} comps"
        return f"PrepResult({k:,}/{self.n_obs:,} cells kept, {int(np.asarray(self.var.get('highly_variable', [])).sum()):,} HVGs{pcs})"


# ---------------------------------------------------------------------------
# The passes
# ---------------------------------------------------------------------------

class Prep:
    """QC → filters → normalise/log1p → HVG → PCA, in passes over budget-sized blocks.

        pp = g.prep("atlas.h5ad", budget=g.Budget(ram="4GB"))
        pp.qc()                                              # pass 1
        pp.filter_cells(min_genes=200, max_pct_mito=20).filter_genes(min_cells=3)
        pp.hvg(n_top=2000)                                   # no new pass
        res = pp.pca(n_comps=50)                             # passes 2 and 3
        res.to_anndata()                                     # obs/var/obsm — scanpy from here
    """

    def __init__(self, target, budget: Optional[Budget] = None, chunk_bytes=None, layer: Optional[str] = None,
                 modality: str = "RNA", target_sum: float = 1e4):
        self.src = _Source(target, budget, chunk_bytes, layer, modality)
        self.budget = budget
        self.target_sum = float(target_sum)
        self.n_obs, self.n_vars = self.src.n_obs, self.src.n_vars
        self.var_names, self.obs_names = self.src.var_names, self.src.obs_names
        self.obs: dict = {}
        self.var: dict = {}
        self._stats: Optional[dict] = None
        self.cell_mask = np.ones(self.n_obs, bool)
        self.gene_mask = np.ones(self.n_vars, bool)
        self.hvg_index: Optional[np.ndarray] = None
        self.params: dict = {"target_sum": self.target_sum}
        self.report: dict = {"passes": 0, "blocks": 0, "seconds": {}}

    # ---- pass 1 ----
    def qc(self, mito_prefix: Sequence[str] = ("MT-", "mt-"), ribo_prefix: Sequence[str] = ("RPS", "RPL", "Rps", "Rpl"),
           count_pass: bool = True) -> "Prep":
        """Per-cell and per-gene metrics plus the per-gene sums the later steps need."""
        t0 = time.time()
        n_obs, n_vars = self.n_obs, self.n_vars
        is_mito = np.array([any(n.startswith(p) for p in mito_prefix) for n in self.var_names])
        is_ribo = np.array([any(n.startswith(p) for p in ribo_prefix) for n in self.var_names])
        total = np.zeros(n_obs, np.float64)
        n_genes = np.zeros(n_obs, np.int32)
        mito = np.zeros(n_obs, np.float64)
        ribo = np.zeros(n_obs, np.float64)
        cells_per_gene = np.zeros(n_vars, np.int64)
        gene_sum = np.zeros(n_vars, np.float64)
        s_norm = np.zeros(n_vars, np.float64)
        s_norm2 = np.zeros(n_vars, np.float64)
        s_log = np.zeros(n_vars, np.float64)
        s_log2 = np.zeros(n_vars, np.float64)
        nblocks = 0
        for start, end, X in self.src.blocks():
            nblocks += 1
            n_rows = end - start
            tot = np.asarray(X.sum(axis=1, dtype=np.float64)).ravel()
            total[start:end] = tot
            n_genes[start:end] = np.diff(X.indptr)
            mito[start:end] = np.asarray(X[:, is_mito].sum(axis=1, dtype=np.float64)).ravel() if is_mito.any() else 0.0
            ribo[start:end] = np.asarray(X[:, is_ribo].sum(axis=1, dtype=np.float64)).ravel() if is_ribo.any() else 0.0
            cells_per_gene += np.bincount(X.indices, minlength=n_vars)
            gene_sum += np.asarray(X.sum(axis=0, dtype=np.float64)).ravel()
            # normalised and log1p sums through the CSR arrays, in row slices so that the four
            # per-non-zero temporaries (row ids, normalised, log, squares) stay inside the chunk rule
            scale = np.where(tot > 0, self.target_sum / np.maximum(tot, 1e-12), 0.0).astype(np.float32)
            avg_nnz = max(X.nnz / max(n_rows, 1), 1.0)
            rows = self._sub_rows(n_rows, int(avg_nnz * 4))          # ~16 bytes of temporaries per nnz
            for s0 in range(0, n_rows, rows):
                s1_ = min(s0 + rows, n_rows)
                lo, hi = int(X.indptr[s0]), int(X.indptr[s1_])
                if hi == lo:
                    continue
                cols = X.indices[lo:hi]
                rid = np.repeat(np.arange(s0, s1_, dtype=np.int32), np.diff(X.indptr[s0:s1_ + 1]))
                norm = X.data[lo:hi].astype(np.float32) * scale[rid]
                del rid
                s_norm += np.bincount(cols, weights=norm, minlength=n_vars)
                np.square(norm, out=norm)
                s_norm2 += np.bincount(cols, weights=norm, minlength=n_vars)
                lg = np.log1p(X.data[lo:hi].astype(np.float32) * scale[np.repeat(
                    np.arange(s0, s1_, dtype=np.int32), np.diff(X.indptr[s0:s1_ + 1]))])
                s_log += np.bincount(cols, weights=lg, minlength=n_vars)
                np.square(lg, out=lg)
                s_log2 += np.bincount(cols, weights=lg, minlength=n_vars)
                del norm, lg, cols
            del X
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_mito = np.where(total > 0, 100.0 * mito / total, 0.0)
            pct_ribo = np.where(total > 0, 100.0 * ribo / total, 0.0)
        self.obs = {"obs_names": self.obs_names, "total_counts": total, "n_genes": n_genes,
                    "pct_mito": pct_mito, "pct_ribo": pct_ribo, "kept": np.ones(n_obs, bool)}
        self.var = {"var_names": self.var_names, "n_cells": cells_per_gene, "total_counts": gene_sum,
                    "mito": is_mito, "ribo": is_ribo}
        self._stats = {"n": n_obs, "s_norm": s_norm, "s_norm2": s_norm2, "s_log": s_log, "s_log2": s_log2,
                       "total": total, "n_genes": n_genes, "pct_mito": pct_mito}
        self.report["passes"] += 1
        self.report["blocks"] = nblocks
        self.report["seconds"]["qc"] = round(time.time() - t0, 3)
        return self

    # ---- filters (no pass) ----
    def filter_cells(self, min_counts: float = 0, min_genes: int = 0, max_pct_mito: float = 100.0,
                     max_counts: Optional[float] = None) -> "Prep":
        self._need_qc()
        o = self.obs
        m = (o["total_counts"] >= min_counts) & (o["n_genes"] >= min_genes) & (o["pct_mito"] <= max_pct_mito)
        if max_counts is not None:
            m &= o["total_counts"] <= max_counts
        self.cell_mask = m
        self.obs["kept"] = m.copy()
        self.params.update({"min_counts": min_counts, "min_genes": min_genes, "max_pct_mito": max_pct_mito,
                            "max_counts": max_counts})
        return self

    def filter_genes(self, min_cells: int = 1) -> "Prep":
        self._need_qc()
        self.gene_mask = self.var["n_cells"] >= min_cells
        self.var["kept"] = self.gene_mask.copy()
        self.params["min_cells"] = min_cells
        return self

    def _need_qc(self):
        if self._stats is None:
            raise RuntimeError("run qc() first")

    # ---- HVG (no pass) ----
    def hvg(self, n_top: int = 2000, n_bins: int = 20, exclude_mito: bool = False) -> "Prep":
        """Seurat-flavour highly variable genes from the pass-1 sums: dispersion = var/mean of the
        normalised counts, log-scaled, z-scored within bins of log1p(mean), top `n_top`.

        The per-gene statistics are over *all* cells seen in pass 1, not only the cells kept by
        `filter_cells` — a second pass would be needed for that, and the ranking it changes is
        at the margin. The parameters record which cells the statistics cover."""
        self._need_qc()
        st = self._stats
        n = st["n"]
        mean = st["s_norm"] / n
        var = st["s_norm2"] / n - mean ** 2
        var = np.maximum(var, 0) * (n / max(n - 1, 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            disp = np.where(mean > 0, var / mean, 0.0)
            disp_log = np.where(disp > 0, np.log(disp), np.nan)
        mean_log = np.log1p(mean)
        cand = self.gene_mask & np.isfinite(disp_log) & (mean > 0)
        if exclude_mito:
            cand &= ~self.var["mito"]
        norm_disp = np.full(self.n_vars, np.nan)
        if cand.any():
            edges = np.linspace(mean_log[cand].min(), mean_log[cand].max(), n_bins + 1)
            bins = np.clip(np.searchsorted(edges, mean_log, side="right") - 1, 0, n_bins - 1)
            for b in range(n_bins):
                sel = cand & (bins == b)
                if sel.sum() == 0:
                    continue
                d = disp_log[sel]
                if sel.sum() == 1:
                    norm_disp[sel] = 0.0           # scanpy: a single gene in a bin gets dispersion 1 -> here 0-centred
                else:
                    sd = d.std(ddof=1)
                    norm_disp[sel] = (d - d.mean()) / (sd if sd > 0 else 1.0)
        order = np.argsort(-np.nan_to_num(norm_disp, nan=-np.inf), kind="stable")
        order = order[cand[order]]
        top = order[:n_top]
        hv = np.zeros(self.n_vars, bool)
        hv[top] = True
        self.hvg_index = np.sort(top)
        self.var.update({"means": mean, "dispersions": disp_log, "dispersions_norm": norm_disp,
                         "highly_variable": hv})
        self.params.update({"n_top_genes": n_top, "n_bins": n_bins, "hvg_flavor": "seurat",
                            "hvg_stats_over": "all cells of pass 1"})
        return self

    # ---- passes 2 and 3 ----
    def _scale_params(self, idx: np.ndarray) -> tuple:
        st = self._stats
        n = st["n"]
        mu = st["s_log"][idx] / n
        var = st["s_log2"][idx] / n - mu ** 2
        sd = np.sqrt(np.maximum(var, 0) * (n / max(n - 1, 1)))
        sd[sd == 0] = 1.0
        return mu, sd

    def _dense_rows(self, X, idx: np.ndarray, mu, sd, max_value: Optional[float], tot):
        """Normalised, log1p, HVG-restricted, scaled, clipped block as dense float32 — one
        column-selected copy of the sparse rows, scaled in place, then densified once."""
        scale = (np.where(tot > 0, self.target_sum / np.maximum(tot, 1e-12), 0.0)).astype(np.float32)
        sub = X[:, idx].tocsr()
        sub.data = sub.data.astype(np.float32, copy=False)
        sub.data *= np.repeat(scale, np.diff(sub.indptr))
        np.log1p(sub.data, out=sub.data)
        D = sub.toarray()
        del sub
        D -= mu.astype(np.float32)
        D /= sd.astype(np.float32)
        if max_value is not None:
            np.clip(D, -max_value, max_value, out=D)
        return D

    def _kept_blocks(self):
        """Blocks restricted to kept cells: a row slice when every cell is kept (no copy),
        a masked copy otherwise; plus the global indices of the rows handed out."""
        for start, end, X in self.src.blocks():
            keep = self.cell_mask[start:end]
            if not keep.any():
                continue
            if keep.all():
                yield start, end, X, np.arange(start, end), self.obs["total_counts"][start:end]
            else:
                where = np.flatnonzero(keep)
                yield start, end, X[where], where + start, self.obs["total_counts"][start:end][keep]
            del X

    def _sub_rows(self, n_rows: int, n_cols: int) -> int:
        """Rows per dense sub-block so that it stays inside a quarter of what is free."""
        per_row = max(n_cols * 4, 1)
        if self.budget is None:
            return max(1, min(n_rows, (64 << 20) // per_row))
        avail = self.budget.available("ram")
        chunk = max(min(1 << 20, avail), avail // 4)
        return max(1, min(n_rows, int(chunk // per_row)))

    def pca(self, n_comps: int = 50, scale: bool = True, max_value: Optional[float] = 10.0,
            out: Optional[str] = None) -> PrepResult:
        """Pass 2: the HVG covariance of the scaled values over kept cells. Pass 3: the projection,
        written block by block into RAM (leased) or a memory-mapped `out` file."""
        self._need_qc()
        if self.hvg_index is None:
            self.hvg()
        idx = self.hvg_index
        k = len(idx)
        n_comps = min(n_comps, k, int(self.cell_mask.sum()) - 1)
        mu, sd = self._scale_params(idx)
        if not scale:
            sd = np.ones_like(sd)
        t0 = time.time()
        S = np.zeros((k, k), np.float64)
        s1 = np.zeros(k, np.float64)
        n_kept = 0
        lease = None
        for start, end, Xk, where, tot in self._kept_blocks():
            rows = self._sub_rows(Xk.shape[0], k)
            for s0 in range(0, Xk.shape[0], rows):
                s1_ = min(s0 + rows, Xk.shape[0])
                lease = self._swap(lease, (s1_ - s0) * k * 4)
                D = self._dense_rows(Xk[s0:s1_], idx, mu, sd, max_value, tot[s0:s1_])
                S += (D.T @ D).astype(np.float64)                 # float32 BLAS per block, float64 across
                s1 += D.sum(axis=0, dtype=np.float64)
                n_kept += D.shape[0]
                del D
            del Xk
        if lease is not None:
            lease.release()
        if n_kept < 2:
            raise ValueError("fewer than two cells kept; nothing to decompose")
        mean = s1 / n_kept
        C = (S - n_kept * np.outer(mean, mean)) / (n_kept - 1)
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1][:n_comps]
        comps = evecs[:, order].T                                   # (n_comps, k)
        # deterministic sign: largest-magnitude loading positive
        for i in range(comps.shape[0]):
            j = np.argmax(np.abs(comps[i]))
            if comps[i, j] < 0:
                comps[i] *= -1
        ev = np.maximum(evals[order], 0)
        total_var = float(np.trace(C))
        self.report["seconds"]["covariance"] = round(time.time() - t0, 3)
        self.report["passes"] += 1

        # pass 3: projection
        t0 = time.time()
        if out:
            X_pca = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(self.n_obs, n_comps))
            out_lease = None
        else:
            out_lease = self._swap(None, self.n_obs * n_comps * 4)
            X_pca = np.zeros((self.n_obs, n_comps), np.float32)
        W = comps.T.astype(np.float32)                              # (k, n_comps)
        mean32 = mean.astype(np.float32)
        lease = None
        for start, end, Xk, where, tot in self._kept_blocks():
            rows = self._sub_rows(Xk.shape[0], k)
            for s0 in range(0, Xk.shape[0], rows):
                s1_ = min(s0 + rows, Xk.shape[0])
                lease = self._swap(lease, (s1_ - s0) * k * 4)
                D = self._dense_rows(Xk[s0:s1_], idx, mu, sd, max_value, tot[s0:s1_])
                D -= mean32
                X_pca[where[s0:s1_]] = D @ W
                del D
            del Xk
        if lease is not None:
            lease.release()
        if hasattr(X_pca, "flush"):
            X_pca.flush()
        self.report["seconds"]["projection"] = round(time.time() - t0, 3)
        self.report["passes"] += 1
        self.report["n_kept"] = int(n_kept)
        self.report["n_hvg"] = int(k)
        if self.budget is not None:
            self.report["budget_peak"] = self.budget.stats()["ram"]["peak_human"]
        self.params.update({"n_comps": n_comps, "scale": scale, "max_value": max_value,
                            "pca": "exact, from the HVG covariance accumulated in one pass"})
        res = PrepResult(obs=dict(self.obs), var=dict(self.var), X_pca=X_pca, components=comps,
                         explained_variance=ev, explained_variance_ratio=ev / total_var if total_var else ev,
                         hvg_index=idx, params=dict(self.params), report=dict(self.report))
        res.report["out_lease"] = "memmap" if out else human(self.n_obs * n_comps * 4)
        if out_lease is not None:
            out_lease.release()                                      # the array now belongs to the caller
        return res

    def _swap(self, lease, nbytes: int):
        if lease is not None:
            lease.release()
        if self.budget is None or nbytes <= 0:
            return None
        return self.budget.acquire("ram", str(int(nbytes)))

    def run(self, n_top: int = 2000, n_comps: int = 50, min_genes: int = 200, min_counts: float = 0,
            max_pct_mito: float = 20.0, min_cells: int = 3, out: Optional[str] = None, **pca_kw) -> PrepResult:
        """All of it, with the usual defaults."""
        self.qc()
        self.filter_cells(min_counts=min_counts, min_genes=min_genes, max_pct_mito=max_pct_mito)
        self.filter_genes(min_cells=min_cells)
        self.hvg(n_top=n_top)
        return self.pca(n_comps=n_comps, out=out, **pca_kw)

    def __repr__(self) -> str:
        return f"Prep({self.src.kind}, {self.n_obs:,} cells x {self.n_vars:,} genes, {self.report['passes']} pass(es))"


def prep(target, budget: Optional[Budget] = None, **kw) -> Prep:
    """QC, filters, normalisation, HVG and PCA out of core — the standard pipeline under a budget."""
    return Prep(target, budget, **kw)
