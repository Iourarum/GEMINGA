"""The dataset orchestrator: descriptions -> pieces -> resolution -> harmonised, annotated builds."""
import csv
import gzip
import json
import os

import numpy as np
import pytest
import scipy.sparse as sp

import geminga as g
from geminga import harmonize as hz
from geminga import orchestrate as orc

# ----------------------------------------------------------------------------- fixtures ----
GENES = ["CD274", "CD47", "GAPDH", "ACTB", "PTPRC", "CD3E", "CD3D", "NT5E"]
ENS = {"CD274": "ENSG00000120217", "CD47": "ENSG00000196776", "GAPDH": "ENSG00000111640",
       "ACTB": "ENSG00000075624", "PTPRC": "ENSG00000081237", "CD3E": "ENSG00000198851",
       "CD3D": "ENSG00000167286", "NT5E": "ENSG00000135318", "CD276": "ENSG00000103855"}

NODES = [{"node_id": "CD274", "primary_gene": "CD274", "human_genes": "CD274", "ensembl": ENS["CD274"],
          "entrez": "29126", "uniprot": "Q9NZQ7", "hgnc": "HGNC:17635", "mouse": "Cd274", "status": "one-to-one"},
         {"node_id": "CD47", "primary_gene": "CD47", "human_genes": "CD47", "ensembl": ENS["CD47"],
          "entrez": "961", "uniprot": "Q08722", "hgnc": "HGNC:1682", "mouse": "Cd47", "status": "one-to-one"},
         {"node_id": "CD276", "primary_gene": "CD276", "human_genes": "CD276", "ensembl": ENS["CD276"],
          "entrez": "80381", "uniprot": "Q5ZPR3", "hgnc": "HGNC:19137", "mouse": "Cd276", "status": "one-to-one"},
         {"node_id": "CD3", "primary_gene": "CD3D", "human_genes": "CD3D; CD3E",
          "ensembl": f"{ENS['CD3D']};{ENS['CD3E']}", "entrez": "915;916", "uniprot": "P04234;P07766",
          "hgnc": "HGNC:1673;HGNC:1674", "mouse": "", "status": "one-to-one"},
         {"node_id": "CD45", "primary_gene": "PTPRC", "human_genes": "PTPRC", "ensembl": ENS["PTPRC"],
          "entrez": "5788", "uniprot": "P08575", "hgnc": "HGNC:9666", "mouse": "Ptprc", "status": "one-to-one"},
         {"node_id": "CD73", "primary_gene": "NT5E", "human_genes": "NT5E", "ensembl": ENS["NT5E"],
          "entrez": "4907", "uniprot": "P21589", "hgnc": "HGNC:8021", "mouse": "Nt5e", "status": "one-to-one"},
         {"node_id": "GAPDH", "primary_gene": "GAPDH", "human_genes": "GAPDH", "ensembl": ENS["GAPDH"]},
         {"node_id": "ACTB", "primary_gene": "ACTB", "human_genes": "ACTB", "ensembl": ENS["ACTB"]}]


def node_map():
    return hz.GeneMap.from_table(NODES, symbol="primary_gene", members="human_genes", ensembl="ensembl",
                                 entrez="entrez", uniprot="uniprot", hgnc="hgnc", aliases=("node_id",),
                                 ortholog="mouse", ortholog_status="status")


def write_h5ad(path, n_obs, genes, seed):
    import h5py
    X = sp.random(n_obs, len(genes), density=0.4, format="csr", dtype=np.float32, random_state=seed)
    X.data = np.round(X.data * 10)
    with h5py.File(path, "w") as f:
        f.attrs["encoding-type"] = "anndata"
        x = f.create_group("X"); x.attrs["encoding-type"] = "csr_matrix"; x.attrs["shape"] = np.array(X.shape)
        x.create_dataset("data", data=X.data); x.create_dataset("indices", data=X.indices)
        x.create_dataset("indptr", data=X.indptr)
        obs = f.create_group("obs"); obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=np.array([f"A{i}" for i in range(n_obs)]).astype("S"))
        d = obs.create_group("patient_id"); d.attrs["encoding-type"] = "categorical"
        d.create_dataset("codes", data=(np.arange(n_obs) % 3).astype(np.int8))
        d.create_dataset("categories", data=np.array(["P1", "P2", "P3"]).astype("S"))
        obs.create_dataset("stim", data=np.array(["rest", "stim"] * (n_obs // 2)).astype("S"))
        var = f.create_group("var"); var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=np.array(genes).astype("S"))
    return X


def write_mtx(d, n_cells, genes, seed, guide=True):
    os.makedirs(d, exist_ok=True)
    feats = [(ENS[gn], gn, "Gene Expression") for gn in genes]
    if guide:
        feats.append(("sgCD274-1", "sgCD274-1", "CRISPR Guide Capture"))
    X = sp.random(len(feats), n_cells, density=0.5, format="coo", dtype=np.float32, random_state=seed)
    X.data = np.round(X.data * 5) + 1
    with gzip.open(f"{d}/matrix.mtx.gz", "wt") as fh:
        fh.write("%%MatrixMarket matrix coordinate real general\n%\n")
        fh.write(f"{len(feats)} {n_cells} {X.nnz}\n")
        for r, c, v in zip(X.row, X.col, X.data):
            fh.write(f"{r + 1} {c + 1} {v:g}\n")
    with gzip.open(f"{d}/barcodes.tsv.gz", "wt") as fh:
        for i in range(n_cells):
            fh.write(f"C{i}-1\n")
    with gzip.open(f"{d}/features.tsv.gz", "wt") as fh:
        for f in feats:
            fh.write("\t".join(f) + "\n")
    return X.tocsc()


ROWS = [
    {"dataset_id": "A_jurkat", "title": "piece A", "accession": "GSE00001 (PRJNA000001)", "species": "human",
     "modalities": "scRNA-seq; CROP-seq", "CD_focus": "CD274; CD47", "n_cells": "~300 high-quality cells",
     "perturbation_type": "CRISPRko", "perturbation_scope": "targeted", "CD_genes_perturbed": "CD274",
     "disease_state": "T-cell leukaemia", "tissue_or_cell_type": "Jurkat T cells", "access": "open"},
    {"dataset_id": "B_k562", "title": "piece B", "accession": "GSE00002", "species": "human",
     "modalities": "scRNA-seq; Perturb-seq", "CD_focus": "CD276", "perturbation_type": "CRISPRi",
     "perturbation_scope": "genome-wide", "platform": "10x 3' (dual-sgRNA CRISPRi)"},
    {"dataset_id": "C_thp1", "title": "piece C", "accession": "PRJNA000003", "species": "human",
     "modalities": "scRNA-seq; CITE-seq", "CD_protein_targets (listed)": "CD47; CD274", "CD_focus": "CD47"},
    {"dataset_id": "D_missing", "title": "piece D (not here)", "accession": "EGAS00001000004", "species": "human",
     "modalities": "scRNA-seq", "CD_focus": "CD3; NT5E", "access": "controlled (EGA)"},
    {"dataset_id": "E_paper", "title": "piece E (paper only)", "accession": "10.1000/xyz", "species": "mouse",
     "modalities": "scATAC-seq"},
    {"dataset_id": "F_nothing", "title": "no accession at all", "species": "human", "modalities": "WES",
     "notes": "data said to be on figshare, no record given"},
]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    # Files are named for the accession of the piece they hold: search_dirs finds a dataset by the
    # accession in its name (as a whole token), not by whatever the piece happens to be called.
    root = str(tmp_path_factory.mktemp("orch"))
    XA = write_h5ad(f"{root}/GSE00001_A_jurkat.h5ad", 300, GENES, 1)
    XC = write_mtx(f"{root}/PRJNA000003_C_thp1_mtx", 120, ["CD47", "CD276", "GAPDH", "PTPRC", "CD274"], 2)
    XB = write_mtx(f"{root}/B_src", 80, ["ACTB", "CD274", "CD3E", "PTPRC", "CD276"], 3)
    from geminga.tenx import MtxTriple
    MtxTriple(f"{root}/B_src").to_h5(f"{root}/GSE00002_B_k562.h5")
    return {"root": root, "XA": XA, "XB": XB, "XC": XC}


# --------------------------------------------------------------------- 1. accessions ----
@pytest.mark.parametrize("text, ids, kinds", [
    ("GSE92872 (PRJNA358686)", ["GSE92872", "PRJNA358686"], ["geo_series", "bioproject"]),
    ("DepMap 24Q4 (Figshare+ 27993248); dbGaP phs003444.v1.p1", ["phs003444.v1.p1", "10.6084/m9.figshare.27993248"],
     ["dbgap", "figshare_doi"]),
    ("SCP1064 (processed); DUOS-000124 (raw, controlled)", ["SCP1064", "DUOS-000124"], ["scp", "duos"]),
    ("Zenodo 10.5281/zenodo.14161275; CELLxGENE", ["10.5281/zenodo.14161275"], ["zenodo_doi"]),
    ("https://zenodo.org/records/13350497", ["10.5281/zenodo.13350497"], ["zenodo_doi"]),
    ("EGAS00001002885 (managed); ERP121676 (ENA, open); Zenodo record 4333872",
     ["ERP121676", "EGAS00001002885", "10.5281/zenodo.4333872"], ["sra_study", "ega_study", "zenodo_doi"]),
    ("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184742", ["GSE184742"], ["geo_series"]),
    ("https://link.springer.com/article/10.1186/s13046-024-03027-6", ["10.1186/s13046-024-03027-6"], ["doi"]),
    ("ftp://ftp.cngb.org/pub/CNSA/data2/CNP0002646/CNS0513001/", ["CNP0002646", "CNS0513001"], ["cngb", "cngb"]),
    ("Mendeley Data 10.17632/7cxccpv4hg.1", ["10.17632/7cxccpv4hg.1"], ["mendeley_doi"]),
    ("E-MTAB-13148 (scRNA-seq); E-MTAB-13144 (multiome)", ["E-MTAB-13148", "E-MTAB-13144"],
     ["arrayexpress", "arrayexpress"]),
    ("BioGRID ORCS screens 1063-1066 (PMID 30995489)", [], []),
    ("https://www.chemspider.com/", ["https://www.chemspider.com/"], ["url"]),
])
def test_find_accessions(text, ids, kinds):
    found = orc.find_accessions(text)
    assert [a.id for a in found] == ids
    assert [a.kind for a in found] == kinds


def test_url_that_only_carries_a_found_accession_is_dropped():
    found = orc.find_accessions("GSE1234; https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE1234")
    assert [a.id for a in found] == ["GSE1234"]


# --------------------------------------------------------------------- 2. vocabulary ----
def test_vocabulary_assays_and_perturbation():
    v = hz.Vocabulary()
    assert v.assays("scRNA-seq; CROP-seq") == ["scRNA-seq", "CROP-seq"]
    assert v.primary_assay("scRNA-seq; CROP-seq; pooled CRISPR screen (FACS sort)") == "scRNA-seq"
    assert v.assays("10x 3' + ADT (20-plex) + direct guide capture") == ["scRNA-seq", "CITE-seq"]
    assert v.assays("T1w; T2w; T2-FLAIR axial+sagittal") == ["MRI"]
    assert v.assays("Western blot and WES") == ["WES"]           # 'wes' must not fire inside 'western'
    assert v.assays("multiome RNA+ATAC; snRNA-seq; snATAC-seq; Perturb-seq") == \
        ["multiome RNA+ATAC", "snRNA-seq", "snATAC-seq", "Perturb-seq"]
    assert v.perturbation("genome-wide CRISPR-Cas9 knockout library") == "CRISPRko"
    assert v.perturbation("CRISPRi H1 library") == "CRISPRi"
    assert v.perturbation("targeted base editing") == "base editing"
    assert v.perturbation("barcoded TF-ORF library (CRISPRa-style overexpression)") == "ORF overexpression"


def test_vocabulary_organism_access_disease_context_and_extension():
    v = hz.Vocabulary()
    assert v.organism("human") == "Homo sapiens"
    assert v.organism("Mouse (C57BL/6)") == "Mus musculus"
    assert v.organism("hs") == "Homo sapiens"
    assert v.organism("Danio") == "Danio rerio"
    assert v.access("Open (GEO/SCP); raw reads controlled via EGA") == "controlled"
    assert v.access("Figshare CC-BY-4.0 (DICOM+NIfTI)") == "open"
    assert v.access("upon request") == "on request"
    assert v.disease("GBM") == "glioblastoma"
    assert v.disease("100 MS patients, Iranian cohort") == "multiple sclerosis"
    assert v.disease("various cancers") == "pan-cancer"
    assert v.context("iPSC/ESC-derived") == "iPSC/ESC-derived"
    assert v.context("patient-derived culture/co-culture") == "primary culture"
    assert v.context("multi-line resource") == "resource"
    v.add("assay", "TAP-seq", ["targeted perturb-seq"])
    assert v.assays("targeted Perturb-seq of enhancers") == ["TAP-seq", "Perturb-seq"] or \
        "TAP-seq" in v.assays("targeted Perturb-seq of enhancers")
    f = v.facets_of({"species": "human", "modalities": "scRNA-seq; CROP-seq", "disease_state": "T-cell leukaemia",
                     "tissue_or_cell_type": "Jurkat T cells", "perturbation_type": "CRISPRko"})
    assert set(f) <= set(g.FACETS)
    assert f["organism"] == "Homo sapiens" and f["assay"] == "scRNA-seq" and f["condition"] == "CRISPRko"


# ------------------------------------------------------------------------ 3. GeneMap ----
def test_genemap_resolves_across_id_spaces_and_expands_complexes():
    gm = node_map()
    assert gm.to(["CD274", "ENSG00000120217", "29126", "Q9NZQ7", "HGNC:17635"], key="symbol") == ["CD274"] * 5
    assert gm.to(["CD45", "CD73", "NOPE"], key="ensembl") == [ENS["PTPRC"], ENS["NT5E"], None]
    assert gm.to(["CD274"], key="ortholog") == ["Cd274"]
    assert gm.ambiguous("CD3") and not gm.ambiguous("CD3E")
    assert [r.symbol for r in gm.members("CD3")] == ["CD3D", "CD3E"]
    assert gm.expand(["CD3", "CD274"], key="ensembl") == [ENS["CD3D"], ENS["CD3E"], ENS["CD274"]]
    assert gm.canonical("ENSG00000000001.5", "ensembl") == "ENSG00000000001"   # unknown but well-formed
    assert gm.canonical("ENSG00000120217.12", "symbol") == "CD274"             # version suffix ignored
    assert [hz.id_type(x) for x in ["CD274", "ENSG00000120217", "29126", "Q9NZQ7", "HGNC:1", "CDw12", "P06126"]] == \
        ["cd", "ensembl", "entrez", "uniprot", "hgnc", "cd", "uniprot"]


def test_genemap_align_intersection_union_and_panel():
    gm = node_map()
    feats = {"a": ["CD274", "CD47", "GAPDH", "ZZZ1"], "b": [ENS["CD274"], "ACTB", "CD47"]}
    inter = gm.align(feats, key="ensembl", how="intersection")
    assert inter.index == [ENS["CD274"], ENS["CD47"]]
    assert inter.positions["a"].tolist() == [0, 1] and inter.positions["b"].tolist() == [0, 2]
    assert inter.report["a"]["n_unmapped"] == 1 and inter.report["a"]["examples"] == ["ZZZ1"]
    union = gm.align(feats, key="ensembl", how="union")
    assert union.n == 4 and union.positions["b"].tolist() == [0, 2, -1, 1]
    panel = gm.align(feats, key="symbol", panel=["CD276", "CD47"])
    assert panel.index == ["CD276", "CD47"] and panel.positions["a"].tolist() == [-1, 1]
    X = sp.csr_matrix(np.arange(8, dtype=np.float32).reshape(2, 4))
    out = panel.take("a", X).toarray()
    assert out.tolist() == [[0, 1], [0, 5]]
    assert "keyed by symbol" in panel.explain()


def test_genemap_from_features_and_hgnc(tmp_path):
    gm = hz.GeneMap.from_features([ENS["CD274"], ENS["CD47"], "sg1"], ["CD274", "CD47", "sg1"])
    assert gm.to(["CD47"], key="ensembl") == [ENS["CD47"]] and gm.resolve("sg1").ensembl == ""
    p = tmp_path / "hgnc.tsv"
    p.write_text("hgnc_id\tsymbol\talias_symbol\tprev_symbol\tensembl_gene_id\tentrez_id\tuniprot_ids\n"
                 "HGNC:17635\tCD274\tB7-H1|PD-L1\tPDCD1LG1\tENSG00000120217\t29126\tQ9NZQ7\n")
    hg = hz.GeneMap.from_hgnc(str(p))
    assert hg.to(["PD-L1", "PDCD1LG1", "B7-H1"], key="symbol") == ["CD274"] * 3


def test_columnmap_guesses_canonical_obs_columns():
    cm = hz.ColumnMap()
    m = cm.guess(["patient_id", "sgRNA", "leiden", "n_counts", "orig.ident", "gene", "_index", "weird"])
    assert m == {"cell_id": "_index", "sample": "orig.ident", "donor": "patient_id", "perturbation": "gene",
                 "guide": "sgRNA", "cluster": "leiden", "n_counts": "n_counts"}
    assert cm.guess(["x"], fixed={"donor": "x"}) == {"donor": "x"}


# ------------------------------------------------------------------- 4. descriptions ----
def test_read_descriptions_from_rows_with_uneven_keys():
    pieces = orc.read_descriptions(ROWS)
    assert [p.id for p in pieces] == ["A_jurkat", "B_k562", "C_thp1", "D_missing", "E_paper", "F_nothing"]
    a = pieces[0]
    assert a.accession_ids() == ["GSE00001", "PRJNA000001"]
    assert a.modalities == ["scRNA-seq", "CROP-seq"] and a.primary_assay == "scRNA-seq"
    assert a.facets["organism"] == "Homo sapiens" and a.facets["disease"] == "T-cell leukaemia"
    assert a.genes == {"perturbed": ["CD274"], "focus": ["CD274", "CD47"]}
    assert a.expected["n_cells"] == 300 and a.perturbation["type"] == "CRISPRko"
    assert pieces[2].genes["protein"] == ["CD47", "CD274"]        # a column only some rows have
    assert pieces[1].perturbation == {"type": "CRISPRi", "scope": "genome-wide", "n_genes": None, "readout": ""}
    assert pieces[3].access == "controlled"
    assert pieces[4].facets["organism"] == "Mus musculus"
    assert pieces[0].row["dataset_id"] == "A_jurkat" and pieces[0].source == "rows:2"


def test_read_descriptions_csv_and_other_dialect(tmp_path):
    p = tmp_path / "inv.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Modality", "Dataset / Cohort", "Reference", "Accession(s)", "Scale", "Access"])
        w.writerow(["scRNA-seq", "Colon atlas", "Pelka 2021", "GEO GSE178341; Broad SCP1162; raw dbGaP phs002407",
                    "371,223 cells; 28 MMRp + 34 MMRd patients", "Open (GEO/SCP); raw controlled via dbGaP"])
        w.writerow(["MRI", "MS3SEG", "figshare", "", "100 MS patients", "Figshare CC-BY"])
    pieces = orc.read_descriptions(str(p))
    assert pieces[0].id == "Colon_atlas" and pieces[0].accession_ids() == ["GSE178341", "phs002407", "SCP1162"]
    assert pieces[0].expected["n_cells"] == 371223 and pieces[0].expected["n_samples"] in (28, 34)
    assert pieces[0].access == "controlled" and pieces[0].primary_assay == "scRNA-seq"
    assert pieces[1].primary_assay == "MRI" and pieces[1].accessions == []
    assert pieces[1].expected["n_cells"] == 100


def test_read_descriptions_xlsx_picks_the_dataset_tab(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active; ws.title = "README"
    ws.append(["Project", "something"]); ws.append(["URL", "https://example.org"])
    ds = wb.create_sheet("Datasets")
    ds.append(["dataset_id", "accession", "modalities", "species", "n_cells"])
    ds.append(["one", "GSE00010", "scRNA-seq", "human", 1000])
    ds.append(["two", "E-MTAB-1", "Stereo-seq", "mouse", "2.5M bins"])
    ds.append([None, None, None, None, None])
    p = tmp_path / "book.xlsx"; wb.save(p)
    pieces = orc.read_descriptions(str(p))
    assert [x.id for x in pieces] == ["one", "two"] and pieces[0].source.endswith("#Datasets:2")
    assert pieces[1].expected["n_cells"] == 2_500_000 and pieces[1].facets["organism"] == "Mus musculus"
    with pytest.raises(KeyError):
        orc.read_descriptions(str(p), sheet="Nope")


def test_counts_do_not_read_units_as_multipliers():
    assert orc._first_int("100 MS patients", orc._UNITS["n_cells"]) == 100
    assert orc._first_int("~1.7M cells", orc._UNITS["n_cells"]) == 1_700_000
    assert orc._first_int("44 datasets, >1.2M cells", orc._UNITS["n_cells"]) == 1_200_000
    assert orc._first_int("610k cells across", orc._UNITS["n_cells"]) == 610_000
    assert orc._first_int("n/a") is None


# ------------------------------------------------------------------------ 5. resolve ----
def test_resolve_offline_statuses_and_precedence(world):
    root = world["root"]
    o = g.orchestrator()
    o.describe(ROWS)
    res = o.resolve(search_dirs=[root], offline=True)
    st = {k: v.status for k, v in res.items()}
    assert st == {"A_jurkat": "local", "B_k562": "local", "C_thp1": "local", "D_missing": "controlled",
                  "E_paper": "manual", "F_nothing": "unresolved"}
    assert res["A_jurkat"].local_paths == [f"{root}/GSE00001_A_jurkat.h5ad"]
    assert res["D_missing"].needs == "pyega3" and res["D_missing"].controlled == ["EGAS00001000004"]
    assert "figshare" in res["F_nothing"].note                     # named a host, gave no record
    o2 = g.orchestrator(); o2.describe(ROWS); r2 = o2.resolve(offline=True)
    assert r2["A_jurkat"].status == "streamable" and r2["A_jurkat"].via == "GSE00001"
    assert r2["C_thp1"].status == "fetchable" and "ENA" in r2["C_thp1"].note
    rep = o2.report()
    assert rep["by_status"]["streamable"] == 2 and rep["by_status"]["controlled"] == 1
    assert o2.pieces["A_jurkat"].line().split()[-1] == "GSE00001"


def test_resolve_finds_catalog_local_paths(world, tmp_path):
    cat = g.catalog(str(tmp_path / "cat.jsonl"))
    cat.record("whatever", world["root"] + "/GSE00001_A_jurkat.h5ad", accessions=["GSE00001"])
    o = g.orchestrator(catalog=cat); o.describe(ROWS[:1]); res = o.resolve()
    assert res["A_jurkat"].status == "local" and res["A_jurkat"].local_paths[0].endswith("GSE00001_A_jurkat.h5ad")


def test_attach_records_in_catalog(world, tmp_path):
    cat = g.catalog(str(tmp_path / "cat.jsonl"))
    o = g.orchestrator(catalog=cat); o.describe(ROWS[3:4])
    o.attach("D_missing", world["root"] + "/GSE00001_A_jurkat.h5ad", note="the DAC came through")
    assert o.pieces["D_missing"].status == "local"
    e = cat.get("D_missing")
    assert e.local_paths and e.facets["organism"] == "Homo sapiens" and e.metadata["accessions"] == ["EGAS00001000004"]
    with pytest.raises(FileNotFoundError):
        o.attach("D_missing", "/nope/none.h5ad")


# ----------------------------------------------------------------- 6. annotate/coverage ----
def test_annotate_writes_catalog_and_need_data_flags(world, tmp_path):
    cat = g.catalog(str(tmp_path / "cat.jsonl")); lib = g.library(str(tmp_path / "lib.jsonl"))
    o = g.orchestrator(catalog=cat, library=lib); o.describe(ROWS); o.resolve(search_dirs=[world["root"]])
    out = o.annotate()
    assert set(out) == set(o.pieces) and "library" not in out["A_jurkat"] and "library" in out["D_missing"]
    assert cat.facets_for("A_jurkat")["assay"] == "scRNA-seq" and cat.get("D_missing").metadata["status"] == "controlled"
    assert cat.validate() == []
    flags = lib.query(intent="need-data")
    assert {f.metadata["piece"] for f in flags} == {"D_missing", "E_paper", "F_nothing"}
    d = next(f for f in flags if f.metadata["piece"] == "D_missing")
    assert d.genes == ["CD3", "NT5E"] and "behind a DAC" in d.note and "controlled" in d.tags


def test_annotate_is_idempotent_and_resolves_arrivals(world, tmp_path):
    cat = g.catalog(str(tmp_path / "cat.jsonl")); lib = g.library(str(tmp_path / "lib.jsonl"))
    o = g.orchestrator(catalog=cat, library=lib); o.describe(ROWS[3:5]); o.resolve()
    first = o.annotate(); second = o.annotate()
    assert first == second and len(lib.query(intent="need-data")) == 2        # no duplicate flags
    o.attach("D_missing", world["root"] + "/GSE00001_A_jurkat.h5ad")                      # the DAC came through
    third = o.annotate()
    assert third["D_missing"]["library"] == first["D_missing"]["library"] and third["D_missing"]["library_resolved"]
    assert lib.get(first["D_missing"]["library"]).status == "resolved"
    assert lib.get(first["E_paper"]["library"]).status == "open"


def test_coverage_codes(world):
    o = g.orchestrator(); o.describe(ROWS); o.harmonizer.genes.merge(node_map())
    cov = o.coverage(genes=["CD274", "CD47", "CD276", "CD3", "NT5E", "GAPDH"])
    assert cov.code("CD274", "A_jurkat") == "XF" and cov.code("CD274", "C_thp1") == "P"
    assert cov.code("CD276", "B_k562") == "xF"                     # implied by a genome-wide screen
    assert cov.code("CD3", "D_missing") == "F" and cov.uncovered() == ["CD276", "CD3", "NT5E", "GAPDH"]
    assert cov.counts("CD47") == {"P": 1, "X": 0, "x": 1, "F": 2, "R": 0}
    t = cov.table(); assert t.num_rows == 6 and "n_protein" in t.column_names
    o.panels["Hao 2021"] = ["NT5E"]; o.pieces["E_paper"].panel = "Hao 2021 (228)"
    assert o.coverage(genes=["NT5E"]).code("NT5E", "E_paper") == "P"


# -------------------------------------------------------------------------- 7. build ----
@pytest.fixture(scope="module")
def built(world, tmp_path_factory):
    root = world["root"]
    cat = g.catalog(f"{root}/cat.jsonl"); lib = g.library(f"{root}/lib.jsonl"); b = g.Budget(ram="32MB")
    o = g.orchestrator(catalog=cat, library=lib, budget=b)
    o.describe(ROWS); o.harmonizer.genes.merge(node_map())
    o.attach("B_k562", f"{root}/GSE00002_B_k562.h5")
    o.resolve(search_dirs=[root])
    bp = o.blueprint("cd-mini", key="symbol", panel=["CD274", "CD47", "CD276", "PTPRC", "CD3E"], note="test")
    pz = o.build(bp, f"{root}/puzzle")
    return o, pz, b


def test_build_aligns_three_containers_and_records_holes(world, built):
    o, pz, b = built
    assert [x["piece"] for x in pz.built] == ["A_jurkat", "B_k562", "C_thp1"]
    assert [x["reader"] for x in pz.built] == ["h5ad", "10x-h5", "mtx"]
    assert [h["piece"] for h in pz.holes] == ["D_missing", "E_paper", "F_nothing"]
    assert pz.n_obs == 500 and pz.n_vars == 5 and not pz.complete
    assert sorted(os.listdir(pz.path)) == sorted(["X_A_jurkat.parquet", "X_B_k562.parquet", "X_C_thp1.parquet",
                                                  "obs.parquet", "var.parquet", "pieces.parquet",
                                                  "manifest.json", "blueprint.json", "dispatch.json"])
    assert all(pz.harvest.verify().values())
    var = pz.read_var().to_pylist()
    assert [v["id"] for v in var] == ["CD274", "CD47", "CD276", "PTPRC", "CD3E"]
    assert var[0]["ensembl"] == ENS["CD274"] and var[0]["in_A_jurkat"] and not var[1]["in_B_k562"]
    obs = pz.read_obs()
    assert obs.num_rows == 500 and obs.column("piece").to_pylist()[299:301] == ["A_jurkat", "B_k562"]
    assert obs.column("donor").to_pylist()[:3] == ["P1", "P2", "P3"] and obs.column("condition")[1].as_py() == "stim"
    assert obs.column("facet_assay")[0].as_py() == "scRNA-seq"
    pieces = pz.harvest.read("pieces").to_pylist()
    assert {p["piece"]: p["built"] for p in pieces}["D_missing"] is False


def test_build_values_match_sources(world, built):
    o, pz, _ = built
    var = [v["id"] for v in pz.read_var().to_pylist()]

    def column(piece, gene, n, offset):
        t = pz.read_X(piece).to_pandas(); sub = t[t.gene == var.index(gene)]
        out = np.zeros(n); out[sub.cell.values - offset] = sub.value.values
        return out
    assert np.allclose(column("A_jurkat", "CD274", 300, 0), world["XA"][:, 0].toarray().ravel())
    assert np.allclose(column("B_k562", "CD276", 80, 300), world["XB"][4, :].toarray().ravel())
    assert np.allclose(column("C_thp1", "CD47", 120, 380), world["XC"][0, :].toarray().ravel())
    # the guide row of the mtx triples must never leak into an RNA build
    assert pz.read_X("C_thp1").column("gene").to_numpy().max() < 5


def test_build_provenance_and_budget(world, built):
    o, pz, b = built
    prov = json.load(open(os.path.join(pz.path, "blueprint.json")))
    assert prov["schema"] == g.BLUEPRINT_SCHEMA and prov["blueprint"]["name"] == "cd-mini"
    assert prov["alignment"]["n"] == 5 and prov["alignment"]["report"]["A_jurkat"]["n_kept"] == 4
    a = prov["pieces"][0]
    assert a["obs_mapping"] == {"donor": "patient_id", "condition": "stim"} and a["sha256"] and a["source"] == "rows:2"
    assert all(x["resident_peak"] > 0 for x in prov["pieces"])
    assert prov["holes"][0]["status"] == "controlled" and prov["holes"][0]["accessions"] == ["EGAS00001000004"]
    assert prov["budget"]["ram"]["capacity"] == 32 << 20 and b.stats()["ram"]["used"] == 0
    man = json.load(open(os.path.join(pz.path, "manifest.json")))
    assert man["schema"] == "geminga.harvest/2" and {r["kind"] for r in man["regions"]} == {"piece", "table"}
    assert os.path.exists(os.path.join(pz.path, "dispatch.json"))
    assert any(e.get("event") == "local" and e["key"] == "A_jurkat" for e in o.catalog.events())


def test_puzzle_measured_feeds_coverage_and_anndata(world, built):
    o, pz, _ = built
    m = pz.measured()
    assert m["B_k562"] == ["CD274", "CD276", "PTPRC", "CD3E"]
    cov = o.coverage(genes=["CD274", "CD47"], measured=m)
    assert cov.code("CD274", "A_jurkat") == "XFR" and cov.code("CD47", "B_k562") == "x"
    pytest.importorskip("anndata")
    a = pz.to_anndata()
    assert a.shape == (500, 5) and a.uns["geminga_holes"] == ["D_missing", "E_paper", "F_nothing"]
    assert a.X[:300, 0].toarray().ravel().sum() == world["XA"][:, 0].sum()


def test_build_union_without_panel_and_ensembl_key(world, tmp_path):
    root = world["root"]
    o = g.orchestrator(budget=g.Budget(ram="16MB")); o.describe(ROWS[:3]); o.harmonizer.genes.merge(node_map())
    o.attach("B_k562", f"{root}/GSE00002_B_k562.h5"); o.resolve(search_dirs=[root])
    bp = o.blueprint("union", key="ensembl", how="union")
    pz = o.build(bp, str(tmp_path / "u"))
    ids = [v["id"] for v in pz.read_var().to_pylist()]
    assert len(ids) == 9 and all(i.startswith("ENSG") for i in ids) and pz.complete
    inter = o.build(o.blueprint("inter", key="ensembl", how="intersection"), str(tmp_path / "i"))
    assert set(v["symbol"] for v in inter.read_var().to_pylist()) == {"CD274", "PTPRC"}


def test_build_table_pieces_long_and_wide(tmp_path):
    long_p = tmp_path / "F_orcs.tsv"
    with open(long_p, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t"); w.writerow(["#SCREEN_ID", "OFFICIAL_SYMBOL", "SCORE.1", "HIT"])
        for i, s in enumerate(["CD274", ENS["CD47"], "GAPDH", "NOPE", "CD3"]):
            w.writerow([1066, s, i * 0.1, "YES" if i % 2 else "NO"])
    wide_p = tmp_path / "G_depmap.csv"
    letters = [chr(65 + i) + "X" for i in range(24)]
    with open(wide_p, "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["ModelID"] + [f"{gn} ({i})" for i, gn in enumerate(GENES + letters)] + ["CD274 (999)"]
        w.writerow(cols)
        for m in range(30):
            w.writerow([f"ACH-{m:06d}"] + list(np.round(np.random.default_rng(m).standard_normal(len(cols) - 1), 3)))
    o = g.orchestrator(budget=g.Budget(ram="8MB")); o.harmonizer.genes.merge(node_map())
    o.describe([{"dataset_id": "F_orcs", "accession": "10.1002/pro.3978", "modalities": "pooled CRISPR screen"},
                {"dataset_id": "G_depmap", "accession": "10.6084/m9.figshare.27993248", "modalities": "pooled CRISPR screen"},
                {"dataset_id": "H_bad", "accession": "GSE0", "modalities": "scRNA-seq"}])
    o.attach("F_orcs", str(long_p)); o.attach("G_depmap", str(wide_p))
    bad = tmp_path / "H_bad.h5ad"; bad.write_bytes(b"not hdf5"); o.attach("H_bad", str(bad))
    pz = o.build(o.blueprint("tables", key="ensembl"), str(tmp_path / "t"))
    import pyarrow.parquet as pq
    tf = pq.read_table(os.path.join(pz.path, "T_F_orcs.parquet"))
    assert tf.column("gene_ensembl").to_pylist() == [ENS["CD274"], ENS["CD47"], ENS["GAPDH"], "", ENS["CD3D"]]
    tg = pq.read_table(os.path.join(pz.path, "T_G_depmap.parquet"))
    assert tg.column_names[1] == ENS["CD274"] and tg.column_names[-1] == ENS["CD274"] + "__1"
    assert json.loads(tg.schema.metadata[b"geminga.original_columns"])[1] == "CD274 (0)"
    kinds = {r.name: r.kind for r in pz.harvest}
    assert kinds["T_F_orcs"] == "piece-table" and kinds["T_G_depmap"] == "piece-table"
    assert [h["piece"] for h in pz.holes] == ["H_bad"] and pz.holes[0]["status"] == "unreadable"
    f = next(x for x in pz.built if x["piece"] == "F_orcs")
    assert f["reader"] == "table" and f["n_kept"] == 4 and f["gene_column"] == "OFFICIAL_SYMBOL"


def test_blueprint_roundtrip_and_orchestrator_save_load(world, tmp_path):
    bp = orc.Blueprint("x", ["a", "b"], key="ensembl", panel=["CD274"])
    p = bp.save(str(tmp_path / "bp.json"))
    assert orc.Blueprint.load(p).to_dict() == bp.to_dict()
    o = g.orchestrator(); o.describe(ROWS); o.resolve(search_dirs=[world["root"]])
    o.save(str(tmp_path / "pieces.json"))
    o2 = orc.Orchestrator.load(str(tmp_path / "pieces.json"))
    assert len(o2) == 6 and o2.resolutions["A_jurkat"].status == "local"
    assert o2.pieces["A_jurkat"].accession_ids() == ["GSE00001", "PRJNA000001"]
    assert o2.pieces_table().num_rows == 6 and o2.to_csv(str(tmp_path / "p.csv"))
    with pytest.raises(KeyError):
        o.blueprint("bad", pieces=["nope"])
