"""Generate one small fixture per supported format, for the end-to-end self-test."""
import gzip, os, sys, numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gem"
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(11)

# --- FASTQ (bulk RNA-seq shaped) + gzipped copy -----------------------------
n_reads, rlen = 200_000, 150
bases = np.array(list("ACGT"))
with open(f"{out}/reads.fastq", "w") as fh:
    for i in range(n_reads):
        s = "".join(bases[rng.integers(0, 4, rlen)])
        q = "".join(chr(33 + min(41, int(v))) for v in rng.normal(35, 4, rlen).clip(2, 41))
        fh.write(f"@read{i}:lane1:{i%8}\n{s}\n+\n{q}\n")
with open(f"{out}/reads.fastq", "rb") as a, gzip.open(f"{out}/reads.fastq.gz", "wb") as b:
    b.write(a.read())
print(f"fastq: {n_reads:,} reads, {os.path.getsize(f'{out}/reads.fastq'):,} B "
      f"(gz {os.path.getsize(f'{out}/reads.fastq.gz'):,} B)")

# --- FASTA (transcriptome shaped) -------------------------------------------
with open(f"{out}/transcripts.fasta", "w") as fh:
    for i in range(2_000):
        L = int(rng.integers(300, 3000))
        s = "".join(bases[rng.integers(0, 4, L)])
        fh.write(f">ENST{i:08d} transcript_{i}\n")
        for j in range(0, L, 60):
            fh.write(s[j:j+60] + "\n")
print(f"fasta: 2,000 transcripts, {os.path.getsize(f'{out}/transcripts.fasta'):,} B")

# --- h5ad (scRNA-seq, CSR counts) -------------------------------------------
import anndata as ad, scipy.sparse as sp
n_obs, n_vars = 20_000, 3_000
X = sp.random(n_obs, n_vars, density=0.05, format="csr", dtype=np.float32,
              random_state=3, data_rvs=lambda k: rng.poisson(3, k).astype(np.float32) + 1)
a = ad.AnnData(X)
a.var_names = [f"Gene{i:05d}" for i in range(n_vars)]
a.obs_names = [f"Cell{i:06d}" for i in range(n_obs)]
a.obs["total_counts"] = np.asarray(X.sum(1)).ravel()
a.obs["batch"] = rng.integers(0, 4, n_obs).astype(np.int32)
a.write_h5ad(f"{out}/sc.h5ad")
print(f"h5ad: {n_obs:,} cells x {n_vars:,} genes, nnz={X.nnz:,}, "
      f"{os.path.getsize(f'{out}/sc.h5ad'):,} B")

# --- Arrow IPC ---------------------------------------------------------------
import pyarrow as pa
tbl = pa.table({"cell": np.arange(n_obs, dtype=np.int32),
                "umi": np.asarray(X.sum(1)).ravel().astype(np.float32),
                "genes": np.diff(X.indptr).astype(np.int32)})
with pa.OSFile(f"{out}/qc.arrow", "wb") as sink:
    with pa.ipc.new_file(sink, tbl.schema) as w:
        for b in tbl.to_batches(max_chunksize=4096):
            w.write_batch(b)
print(f"arrow: {tbl.num_rows:,} rows, {os.path.getsize(f'{out}/qc.arrow'):,} B")

# --- Imaging (tiled pyramidal OME-TIFF, ssDNA-ish) ---------------------------
import tifffile
H = W = 6144
img = np.zeros((H, W), dtype=np.uint16)
yy, xx = np.mgrid[0:H, 0:W]
for cy, cx, r in [(2000, 2200, 1400), (4200, 4000, 900)]:
    img += (np.exp(-(((yy-cy)**2 + (xx-cx)**2) / (2.0*r*r))) * 9000).astype(np.uint16)
img += (rng.random((H, W)) * 400).astype(np.uint16)
tifffile.imwrite(f"{out}/tissue.ome.tif", img, tile=(512, 512), photometric="minisblack",
                 metadata={"axes": "YX"})
print(f"tiff: {H}x{W} uint16, {os.path.getsize(f'{out}/tissue.ome.tif'):,} B")
print("fixtures in", out)
