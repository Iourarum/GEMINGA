"""Generate a Stereo-seq-shaped Parquet file: (x, y, gene, count) at bin level, many row groups.
Row groups are ~100k rows so a sample of a few groups is a few MB, like the Hub's converted files."""
import sys, numpy as np, pyarrow as pa, pyarrow.parquet as pq

n = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
rng = np.random.default_rng(7)
# a tissue-ish blob: two Gaussian lobes on a 20k x 20k bin grid
lobe = rng.random(n) < 0.6
x = np.where(lobe, rng.normal(7000, 1800, n), rng.normal(13000, 1200, n)).clip(0, 19999).astype(np.int32)
y = np.where(lobe, rng.normal(9000, 2200, n), rng.normal(11000, 900, n)).clip(0, 19999).astype(np.int32)
genes = np.array([f"Gene{i:04d}" for i in range(2000)])
gene = pa.array(genes[rng.integers(0, 2000, n)]).dictionary_encode()
count = rng.geometric(0.35, n).astype(np.int32)
umi_frac = rng.random(n).astype(np.float32)
t = pa.table({"x": x, "y": y, "gene": gene, "count": count, "umi_frac": umi_frac})
pq.write_table(t, sys.argv[1], row_group_size=100_000, compression="snappy",
               write_statistics=True)
meta = pq.read_metadata(sys.argv[1])
print(f"wrote {sys.argv[1]}: {meta.num_rows:,} rows, {meta.num_row_groups} row groups, "
      f"{meta.serialized_size:,} B footer")
