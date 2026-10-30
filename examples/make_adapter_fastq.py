"""A FASTQ with adapter read-through concentrated in one region of the file — the shape real
libraries have, and what makes a search-and-visualise demo mean something."""
import sys
import numpy as np
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gem/reads_adapters.fastq"
n, rlen = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000, 150
ADAPTER = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"
rng = np.random.default_rng(5)
bases = np.array(list("ACGT"))
lo, hi = int(n * 0.55), int(n * 0.70)          # the contaminated stretch
with open(out, "w") as fh:
    for i in range(n):
        s = "".join(bases[rng.integers(0, 4, rlen)])
        if lo <= i < hi and rng.random() < 0.45:
            start = int(rng.integers(40, rlen - 20))
            ins = ADAPTER[: rlen - start]
            s = s[:start] + ins + s[start + len(ins):]
        q = "".join(chr(33 + min(41, int(v))) for v in rng.normal(35, 4, rlen).clip(2, 41))
        fh.write(f"@read{i}\n{s}\n+\n{q}\n")
print(f"wrote {out}: {n:,} reads, adapter seeded in records {lo:,}-{hi:,}")
