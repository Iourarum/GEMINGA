"""Flag seven kinds of thing in one library, describe each, and let the links appear.

Every record here is a demonstration on synthetic data. The rsID and its coordinates are real
(EGFR L858R, GRCh38); the call being "refuted" is invented for the example.
"""
import os
import sys
import numpy as np
import geminga as g
path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gem/lib_demo.jsonl"
if os.path.exists(path):
    os.remove(path)
lib = g.library(path)
GBM, H5 = "/tmp/gem/gbm/gbm_phantom.gef", "/tmp/gem/glioma/filtered_feature_bc_matrix.h5"
FQ = ["/tmp/gem/reads_adapters.fastq", "/tmp/gem/reads.fastq"]

v = g.open_gef(GBM).use_bin(20)
reg = np.load("/tmp/gem/gbm/regions.npy"); cx, cy, rn, rr = reg[:4]; r, s = (rn + rr) / 2 * 1.02, 110
rim = g.flag_region(v, "rim-east", (cx + r - s, cx + r + s, cy - s, cy + s), intent="need-data",
                    metadata={"tissue": "glioblastoma", "sample": "phantom-1"}, lib=lib)
g.describe_any(rim, v, lib=lib)
snp = g.flag_variant("chr7", 55191822, "T", "G", rsid="rs121434568", genes=["EGFR"],
                     note="L858R-position call looks strand-biased",
                     metadata={"sample": "phantom-1", "caller": "GATK"}, lib=lib)
g.flag_variant("6", 43770613, "C", "T", genes=["VEGFA"], note="low depth, recheck",
               metadata={"sample": "phantom-1"}, lib=lib)
g.flag_locus("chr7:55,019,017-55,211,628", "EGFR amplicon", genes=["EGFR"], intent="need-data",
             metadata={"assay": "WGS"}, lib=lib)
adp = g.flag_sequence("AGATCGGAAGAGCACACGTC", name="adapter read-through",
                      note="appears in a band of reads; trimming may have failed", lib=lib)
g.describe_any(adp, sources=FQ, lib=lib)
m = g.TenXMatrix(H5)
Xg = m.block(0, m.n_cells, "GUIDE").tocsr()
gi = [i for i, n in enumerate(m.features("GUIDE")) if "EGFR" in n]
cells = g.flag_cells(m.barcodes[np.flatnonzero(np.asarray(Xg[gi].sum(0)).ravel() > 0)], source=H5,
                     name="sgEGFR cells", modality="RNA", genes=["EGFR"],
                     note="knockdown efficiency unclear", lib=lib)
g.describe_any(cells, m, lib=lib)
panel = g.flag_genes(["VEGFA", "CA9", "HILPDA", "EGFR"], name="hypoxia + EGFR panel",
                     metadata={"from": "reviewer 2"}, lib=lib)
g.describe_any(panel, v, lib=lib)
o2 = g.flag_values("/tmp/gem/culture.parquet", "t", 3.9 * 86400, 4.15 * 86400, "O2 dip day 3.9",
                   where=("o2_percent", "<", 8), note="probe drift or real hypoxia?",
                   metadata={"instrument": "PreSens"}, lib=lib)
g.describe_any(o2, lib=lib)

lib.link(o2, rim, "same biology: hypoxia", note="culture episode vs tissue niche")
lib.review(snp, "pulled the BAM, checking strand counts", by="SV")
lib.refute(snp, "strand bias confirmed: artefact", by="SV")
g.leads_any(lib.get(snp.id), lib=lib)
g.leads_any(lib.get(adp.id), sources=FQ, lib=lib)
g.leads_any(lib.get(panel.id), sources=[GBM, "/tmp/gem/chip.gef"], lib=lib)
lib.print_table()
