"""The library: flags of every kind, their coordinate conventions, workflow, queries and links."""
import json
import os
import numpy as np
import pytest
import geminga as g

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")
GBM = os.path.join(FIX, "gbm", "gbm_phantom.gef")
H5 = os.path.join(FIX, "glioma", "filtered_feature_bc_matrix.h5")
TS = os.path.join(FIX, "culture.parquet")
FQA = os.path.join(FIX, "reads_adapters.fastq")
FQ = os.path.join(FIX, "reads.fastq")


@pytest.fixture
def lib(tmp_path):
    return g.library(str(tmp_path / "lib.jsonl"))


@pytest.mark.parametrize("text,expect", [
    ("chr7:55,019,017-55,211,628", ("7", 55_019_016, 55_211_628)),
    ("7:55.0M-55.3M", ("7", 54_999_999, 55_300_000)),
    ("chrX:100", ("X", 99, 100)),
    ("chr1:1k-2k", ("1", 999, 2000)),
    ("chr1:0-100", ("1", 0, 100)),        # browsers count from 1: a 0 start is the first base, not -1
    ("chr1:0", ("1", 0, 1)),
])
def test_parse_locus_reads_browser_coordinates(text, expect):
    assert g.parse_locus(text) == expect


def test_parse_locus_rejects_nonsense():
    for bad in ("chr7", "7:200-100", "not a locus"):
        with pytest.raises(ValueError):
            g.parse_locus(bad)


def test_variant_coordinates_are_explicit(lib):
    v = g.flag_variant("chr7", 55191822, "T", "G", rsid="rs121434568", build="hg38", lib=lib)
    gm = v.geometry
    assert gm["chrom"] == "7" and gm["pos"] == 55191822               # VCF 1-based kept as given
    assert (gm["start"], gm["end"]) == (55191821, 55191822)           # and 0-based half-open
    assert gm["build"] == "GRCh38" and "1-based" in gm["coords"]
    assert g.flag_variant(rsid="rs1").geometry["start"] is None         # rsID alone is allowed
    with pytest.raises(ValueError):
        g.flag_variant("7")


def test_locus_queries_overlap_and_respect_build(lib):
    g.flag_variant("chr7", 55191822, "T", "G", rsid="rs121434568", lib=lib)
    g.flag_locus("chr7:55,019,017-55,211,628", "EGFR", lib=lib)
    g.flag_locus("chr7:55,019,017-55,211,628", "EGFR hg19", build="hg19", lib=lib)
    g.flag_locus("chr6:43.7M-43.8M", "VEGFA", lib=lib)
    names = {s.name for s in lib.query(locus="chr7:55.19M-55.20M")}
    assert names >= {"rs121434568", "EGFR"}
    assert {s.name for s in lib.query(locus="chr7:55.19M-55.20M", build="GRCh38")} == \
        {"rs121434568", "EGFR"}
    assert lib.query(locus="chr7:1-1000") == []
    assert [s.name for s in lib.query(locus="6:43,750,000-43,760,000")] == ["VEGFA"]


def test_workflow_keeps_history_and_queues(lib):
    v = g.flag_variant("7", 100, "A", "C", lib=lib)
    n = g.flag_genes(["EGFR"], lib=lib)
    assert v in lib.to_double_check() or v.id in {x.id for x in lib.to_double_check()}
    assert n.id in {x.id for x in lib.open_questions()}
    lib.review(v, "checking", by="SV")
    lib.refute(v, "artefact", by="SV")
    h = lib.get(v.id).history
    assert [x["status"] for x in h] == ["open", "in-review", "refuted"]
    assert h[-1]["from"] == "in-review" and h[-1]["by"] == "SV"
    assert v.id not in {x.id for x in lib.to_double_check()}
    with pytest.raises(ValueError):
        lib.set_status(v, "maybe")


def test_metadata_is_queryable(lib):
    a = g.flag_genes(["CA9"], metadata={"sample": "S1", "depth": 12}, lib=lib)
    g.flag_genes(["CA9"], metadata={"sample": "S2", "depth": 40}, lib=lib)
    assert [s.id for s in lib.query(meta={"sample": "S1"})] == [a.id]
    assert len(lib.query(meta={"depth": lambda d: d < 20})) == 1
    lib.annotate(a, tissue="cortex")
    assert lib.query(meta={"tissue": "cortex"})[0].id == a.id
    assert lib.query(text="cortex")[0].id == a.id
    assert len(lib.query(gene="ca9")) == 2                              # case-insensitive


def test_related_crosses_kinds_and_links(lib):
    snp = g.flag_variant("7", 55191822, "T", "G", genes=["EGFR"], lib=lib)
    loc = g.flag_locus("chr7:55,019,017-55,211,628", "amp", genes=["EGFR"], lib=lib)
    cel = g.flag_cells(["AAAC-1", "AAAG-1"], name="cells", genes=["EGFR"], lib=lib)
    seq = g.flag_sequence("ACGTACGTAC", lib=lib)
    tsr = g.flag_genes(["VEGFA"], lib=lib)
    lib.link(seq, tsr, "control")
    rel = {o.id: w for o, w in lib.related(snp)}
    assert loc.id in rel and any(w.startswith("overlaps") for w in rel[loc.id])
    assert cel.id in rel and seq.id not in rel
    rel2 = {o.id: w for o, w in lib.related(seq)}
    assert tsr.id in rel2 and "linked: control" in rel2[tsr.id]
    assert lib.related(tsr)[0][0].id == seq.id                          # explicit links first


def test_cells_large_sets_go_to_a_sidecar(lib, tmp_path):
    small = g.flag_cells([f"C{i}-1" for i in range(10)], lib=lib)
    big = g.flag_cells([f"C{i}-1" for i in range(500)], lib=lib, sidecar_dir=str(tmp_path / "cs"))
    assert small.geometry["barcodes"] and small.geometry["barcodes_path"] is None
    assert big.geometry["barcodes"] is None and os.path.exists(big.geometry["barcodes_path"])
    again = g.flag_cells([f"C{i}-1" for i in reversed(range(500))], lib=lib,
                         sidecar_dir=str(tmp_path / "cs"))
    assert again.geometry["barcodes_sha256"] == big.geometry["barcodes_sha256"]   # order-free
    assert len(os.listdir(tmp_path / "cs")) == 1                        # stored once


def test_sequence_identity_is_hashed(lib):
    a = g.flag_sequence("acgu", lib=lib)
    assert a.geometry["sequence"] == "ACGT"                              # RNA to DNA, upper
    assert len(a.geometry["sha256"]) == 64
    long = g.flag_sequence("A" * 20_000)
    assert long.geometry["sequence"] is None and long.geometry["sha256"]
    with pytest.raises(ValueError):
        g.flag_sequence()


@pytest.mark.skipif(not os.path.exists(FQA), reason="no adapter FASTQ")
def test_sequence_is_found_in_local_files(lib):
    s = g.flag_sequence("AGATCGGAAGAGCACACGTC", lib=lib)
    d = g.describe_any(s, sources=[FQA, FQ], lib=lib)
    occ = {o["file"]: o["records_matched"] for o in d["occurrences"]}
    assert occ[os.path.basename(FQA)] > 1000 and occ.get(os.path.basename(FQ), 0) == 0
    L = g.leads_any(lib.get(s.id), sources=[FQA, FQ], lib=lib)
    assert any(l["kind"] == "local" for l in L) and any(l["source"] == "BLAST" for l in L)
    assert "CMD=Put" in next(l for l in L if l["source"] == "BLAST")["url"]


@pytest.mark.skipif(not os.path.exists(TS), reason="no time series")
def test_value_flags_count_the_readings(lib):
    ev = np.load(TS.replace(".parquet", "_events.npy"))
    f = g.flag_values(TS, "t", ev[1] * 86400, (ev[1] + 0.25) * 86400, "dip",
                      where=("o2_percent", "<", 8), lib=lib)
    d = g.describe_any(f, lib=lib)
    assert 0 < d["rows_matching"] < d["rows_in_window"]
    assert d["value_range"][1] < 8
    with pytest.raises(ValueError):
        g.flag_values(TS, "t", 0, 1, "x", where=("o2_percent", "~", 8))


@pytest.mark.skipif(not os.path.exists(H5), reason="no 10x fixture")
def test_cell_sets_are_measured_against_all_cells(lib):
    m = g.TenXMatrix(H5)
    Xg = m.block(0, m.n_cells, "GUIDE").tocsr()
    gi = [i for i, n in enumerate(m.features("GUIDE")) if "EGFR" in n]
    idx = np.flatnonzero(np.asarray(Xg[gi].sum(0)).ravel() > 0)
    c = g.flag_cells(m.barcodes[idx], source=H5, modality="GUIDE", lib=lib)
    d = g.describe_any(c, m, lib=lib)
    assert d["n"] == idx.size and 0.1 < d["fraction_of_cells"] < 0.25
    top = {e["gene"] for e in d["enriched"]}
    assert top and all("EGFR" in t for t in list(top)[:4])              # their own guides


@pytest.mark.skipif(not os.path.exists(GBM), reason="no phantom")
def test_gene_panel_described_in_a_chip(lib):
    v = g.open_gef(GBM)
    p = g.flag_genes(["VEGFA", "CA9", "NOT_A_GENE"], lib=lib)
    d = g.describe_any(p, v, lib=lib)
    rows = {r["gene"]: r for r in d["in_view"]}
    assert rows["VEGFA"]["present"] and rows["VEGFA"]["mid"] > 0
    assert rows["NOT_A_GENE"]["present"] is False


def test_variant_leads_point_at_the_right_repositories(lib):
    v = g.flag_variant("7", 55191822, "T", "G", rsid="rs121434568", genes=["EGFR"], lib=lib)
    L = g.leads_any(v, lib=lib)
    srcs = [l["source"] for l in L]
    assert {"Ensembl", "ClinVar", "dbSNP", "GEO", "PubMed"} <= set(srcs)
    urls = " ".join(l["url"] for l in L)
    assert "variation/homo_sapiens/rs121434568" in urls
    assert "overlap/region/homo_sapiens/7:55191822-55191822" in urls
    assert lib.get(v.id).status == "searched"


def test_bed_is_strictly_six_columns(lib, tmp_path):
    g.flag_variant("7", 55191822, "T", "G", rsid="rs1 x", lib=lib)
    g.flag_locus("chr6:100-200", "a locus", lib=lib)
    rows = [l.split("\t") for l in open(lib.to_bed(str(tmp_path / "x.bed"))).read().splitlines()]
    assert {len(r) for r in rows} == {6}
    assert all(r[1].isdigit() and r[2].isdigit() and int(r[1]) < int(r[2]) for r in rows)
    assert all(" " not in r[3] for r in rows)


def test_frame_and_old_logs(lib, tmp_path):
    g.flag_variant("7", 100, "A", "C", lib=lib)
    g.flag_genes(["EGFR"], lib=lib)
    df = lib.frame()
    assert df.height == 2 and {"kind", "chrom", "start", "genes"} <= set(df.columns)
    # a log written before metadata/history/links existed still loads
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps({"event": "add", "id": "x1", "name": "old", "dataset": {},
                               "kind": "interval", "geometry": {"lo": 0, "hi": 1}}) + "\n")
    rec = g.library(str(old)).get("x1")
    assert rec.metadata == {} and rec.history == [] and rec.links == []
