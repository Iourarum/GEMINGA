"""Regression tests for the confirmed correctness defects in orchestrate / harmonize / handoff.

One section per finding (the numbers match CHANGES-A.md). Each test was written to fail on the
code as it stood before the fix and states the behaviour the fix guarantees.
"""
import csv
import gzip
import hashlib
import json
import os

import numpy as np
import pytest
import scipy.sparse as sp

import geminga as g
from geminga import harmonize as hz
from geminga import orchestrate as orc

# ----------------------------------------------------------------------------- helpers ----
ENS = {"CD274": "ENSG00000120217", "CD47": "ENSG00000196776", "CD3D": "ENSG00000167286",
       "CD3E": "ENSG00000198851", "CD276": "ENSG00000103855", "PTPRC": "ENSG00000081237"}


def _touch(path, data=b"x"):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return str(path)


def _h5ad(path, n_obs, genes, seed=0, donor="categorical"):
    """A minimal AnnData-layout .h5ad. `donor="nullable"` writes a nullable-integer obs column
    (values+mask; readable since the shared decoder landed); `donor="alien"` writes a group
    with an encoding no reader understands."""
    import h5py
    X = sp.random(n_obs, len(genes), density=0.5, format="csr", dtype=np.float32, random_state=seed)
    X.data = np.round(X.data * 10) + 1
    with h5py.File(path, "w") as f:
        f.attrs["encoding-type"] = "anndata"
        x = f.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["shape"] = np.array(X.shape)
        x.create_dataset("data", data=X.data)
        x.create_dataset("indices", data=X.indices)
        x.create_dataset("indptr", data=X.indptr)
        obs = f.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=np.array([f"C{i}" for i in range(n_obs)]).astype("S"))
        if donor == "categorical":
            d = obs.create_group("donor")
            d.attrs["encoding-type"] = "categorical"
            d.create_dataset("codes", data=(np.arange(n_obs) % 3).astype(np.int8))
            d.create_dataset("categories", data=np.array(["P1", "P2", "P3"]).astype("S"))
        elif donor == "nullable":
            d = obs.create_group("donor")
            d.attrs["encoding-type"] = "nullable-integer"
            d.create_dataset("values", data=np.arange(n_obs))
            d.create_dataset("mask", data=np.zeros(n_obs, dtype=bool))
        else:
            d = obs.create_group("donor")
            d.attrs["encoding-type"] = "alien-encoding-9000"
            d.create_dataset("payload", data=np.arange(n_obs))
        var = f.create_group("var")
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=np.array(genes).astype("S"))
    return X


def _mtx_dir(d, n_cells, genes, seed=0):
    os.makedirs(d, exist_ok=True)
    feats = [(f"ENSG{i + 1:011d}", gn, "Gene Expression") for i, gn in enumerate(genes)]
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
        for ft in feats:
            fh.write("\t".join(ft) + "\n")
    return X.tocsc()


def _node_map():
    nodes = [{"node_id": "CD274", "primary_gene": "CD274", "human_genes": "CD274", "ensembl": ENS["CD274"]},
             {"node_id": "CD47", "primary_gene": "CD47", "human_genes": "CD47", "ensembl": ENS["CD47"]},
             {"node_id": "CD3", "primary_gene": "CD3D", "human_genes": "CD3D; CD3E",
              "ensembl": f"{ENS['CD3D']};{ENS['CD3E']}"}]
    return hz.GeneMap.from_table(nodes, symbol="primary_gene", members="human_genes", ensembl="ensembl",
                                 aliases=("node_id",), group="node_id")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def _local_piece(o, pid, path, accession="GSE00001", modalities="scRNA-seq"):
    o.describe([{"id": pid, "accession": accession, "modalities": modalities}])
    o.attach(pid, str(path))


# ============================================================ 1. local files and their owners ==
def test_resolve_does_not_adopt_another_datasets_file(tmp_path):
    d = str(tmp_path)
    _touch(f"{d}/GSE123456_other_study.h5ad")
    o = g.orchestrator()
    o.describe([{"id": "wanted", "accession": "GSE12345"}])
    res = o.resolve(search_dirs=[d])
    assert res["wanted"].status != "local" and res["wanted"].local_paths == []
    assert o.pieces["wanted"].local == []
    _touch(f"{d}/GSE12345_counts.h5ad")                          # the real one, once it arrives
    res = o.resolve(search_dirs=[d])
    assert res["wanted"].status == "local" and res["wanted"].local_paths == [f"{d}/GSE12345_counts.h5ad"]


def test_walk_for_matches_on_accession_boundaries_and_only_data_files(tmp_path):
    d = str(tmp_path)
    for name in ["GSE123456_other_study.h5ad", "SRR14487741.fastq.gz", "atlas_figure.png", "GSE12345_RAW.tar",
                 "GSE12345.h5ad", "notes_GSE12345.txt", "gse12345_lower.csv"]:
        _touch(f"{d}/{name}")
    _touch(f"{d}/GSE54321/matrix.mtx.gz")                          # a matrix directory
    _touch(f"{d}/GSE54321_figures/plot.png")                       # a directory that is not
    _touch(f"{d}/GSE99999/outs/filtered_feature_bc_matrix/matrix.mtx.gz")   # a Cell Ranger bundle

    def rel(hits):
        return sorted(os.path.relpath(h, d) for h in hits)
    assert rel(orc._walk_for(["GSE12345"], [d])) == ["GSE12345.h5ad", "GSE12345_RAW.tar", "gse12345_lower.csv"]
    assert orc._walk_for(["SRR1448774"], [d]) == []                # SRR14487741 is another run
    assert orc._walk_for(["Atlas"], [d]) == []                     # a figure is not a dataset
    assert rel(orc._walk_for(["GSE54321"], [d])) == ["GSE54321"]
    assert rel(orc._walk_for(["GSE99999"], [d])) == ["GSE99999"]
    assert rel(orc._walk_for(["SRR14487741"], [d])) == ["SRR14487741.fastq.gz"]


def test_piece_id_is_a_needle_only_when_it_is_accession_shaped(tmp_path):
    d = str(tmp_path)
    _touch(f"{d}/Atlas.h5ad")
    _touch(f"{d}/SRR1448774.h5ad")
    o = g.orchestrator()
    o.add(orc.Piece(id="Atlas"))
    o.add(orc.Piece(id="SRR1448774"))
    res = o.resolve(search_dirs=[d])
    assert res["Atlas"].status == "unresolved" and res["Atlas"].local_paths == []
    assert res["SRR1448774"].status == "local" and res["SRR1448774"].local_paths == [f"{d}/SRR1448774.h5ad"]


def test_vault_members_are_matched_with_the_same_boundary_rule(tmp_path):
    root = tmp_path / "vault"
    _touch(root / "GSE123456_other.h5ad")
    _touch(root / "GSE12345_cells.h5ad")
    v = g.vault(str(root))
    v.scan()
    o = g.orchestrator(vault=v)
    o.describe([{"id": "wanted", "accession": "GSE12345"}])
    res = o.resolve()
    assert res["wanted"].local_paths == [str(root / "GSE12345_cells.h5ad")]


def test_vanished_files_do_not_keep_a_piece_local(tmp_path):
    f = _touch(tmp_path / "GSE00077.h5ad")
    cat = g.catalog(str(tmp_path / "cat.jsonl"))
    gone = str(tmp_path / "gone.h5ad")
    cat.record("rec", "GSE00077", local_paths=[gone], accessions=["GSE00077"])
    o = g.orchestrator(catalog=cat)
    o.describe([{"id": "p", "accession": "GSE00077"}])
    res = o.resolve()
    assert res["p"].status != "local" and res["p"].local_paths == []      # a stale catalog path
    o.attach("p", f)
    assert o.resolve()["p"].status == "local"
    os.remove(f)
    res = o.resolve()                                                    # attached, then deleted
    assert res["p"].status != "local" and o.pieces["p"].local == []


def test_build_outputs_noted_in_the_catalog_are_not_taken_for_sources(tmp_path):
    src = str(tmp_path / "GSE00001.h5ad")
    _h5ad(src, 20, ["AAA1", "AAA2", "AAA3"])
    cat = g.catalog(str(tmp_path / "cat.jsonl"))
    o = g.orchestrator(catalog=cat)
    _local_piece(o, "p", src)
    o.build(o.blueprint("b"), str(tmp_path / "out"))              # note_local() records X_p.parquet
    assert any(lp.endswith("X_p.parquet") for lp in cat.get("p").local_paths)
    os.remove(src)
    res = o.resolve()
    assert res["p"].status != "local" and o.pieces["p"].local == []


def test_build_marks_a_vanished_file_as_missing_and_says_so(tmp_path, capsys):
    src = str(tmp_path / "GSE00001.h5ad")
    _h5ad(src, 20, ["AAA1", "AAA2"])
    o = g.orchestrator()
    _local_piece(o, "p", src)
    os.remove(src)                                                # gone since the last resolve
    pz = o.build(o.blueprint("b"), str(tmp_path / "out"))
    assert [h["piece"] for h in pz.holes] == ["p"] and pz.holes[0]["status"] == "missing"
    pz.print_table()
    out = capsys.readouterr().out
    assert "file missing" in out and "here" not in out


def test_build_tries_the_candidate_paths_in_turn_and_records_them(tmp_path):
    bad = _touch(tmp_path / "GSE00001_bad.h5ad", b"not hdf5")
    good = str(tmp_path / "GSE00001_good.h5ad")
    _h5ad(good, 20, ["AAA1", "AAA2"])
    o = g.orchestrator()
    o.describe([{"id": "p", "accession": "GSE00001", "modalities": "scRNA-seq"}])
    o.pieces["p"].local = [bad, good]
    o.pieces["p"].status = "local"
    pz = o.build(o.blueprint("b"), str(tmp_path / "out"))
    assert pz.holes == [] and pz.built[0]["path"] == good
    assert pz.built[0]["candidates"] == [bad, good]
    prov = json.load(open(os.path.join(pz.path, "blueprint.json")))
    assert prov["pieces"][0]["candidates"] == [bad, good]
    # nothing opens: one hole that names every path it tried
    o.pieces["p"].local = [bad, _touch(tmp_path / "GSE00001_worse.h5ad", b"also not hdf5")]
    pz2 = o.build(o.blueprint("b2"), str(tmp_path / "out2"))
    assert pz2.built == [] and pz2.holes[0]["status"] == "unreadable"
    assert len(pz2.holes[0]["candidates"]) == 2 and all(p in pz2.holes[0]["note"] for p in pz2.holes[0]["candidates"])


def test_build_opens_a_cell_ranger_bundle_found_by_accession(tmp_path):
    d = tmp_path / "data"
    _mtx_dir(str(d / "GSE99999" / "outs" / "filtered_feature_bc_matrix"), 30, ["AAA1", "AAA2", "AAA3"])
    o = g.orchestrator()
    o.describe([{"id": "p", "accession": "GSE99999", "modalities": "scRNA-seq"}])
    o.resolve(search_dirs=[str(d)])
    assert o.pieces["p"].status == "local"
    pz = o.build(o.blueprint("b"), str(tmp_path / "out"))
    assert pz.holes == [] and pz.built[0]["reader"] == "mtx" and pz.n_obs == 30


# ====================================================================== 2. symbols and junk ==
@pytest.mark.parametrize("sym", ["CD8A", "P2RY12", "B3GNT2", "CD3E", "HLA-DRB1", "MT-CO1", "C1orf112", "7SK"])
def test_symbol_key_keeps_valid_symbols_that_look_like_other_ids(sym):
    assert hz.GeneMap().canonical(sym, "symbol") == sym


@pytest.mark.parametrize("junk", ["Unnamed: 0", "two words", "a:b", "ENSG00000000001", "12345", "HGNC:5", "",
                                  "nan", "Unnamed: 7_level_0"])
def test_symbol_key_refuses_ids_of_other_spaces_and_junk(junk):
    assert hz.GeneMap().canonical(junk, "symbol") == ""


def test_uniprot_typing_is_trusted_only_on_a_map_hit():
    gm = hz.GeneMap.from_table([{"symbol": "CD274", "uniprot": "Q9NZQ7"}], uniprot="uniprot")
    assert gm.canonical("Q9NZQ7", "symbol") == "CD274"             # a hit: it is the UniProt id
    assert gm.canonical("Q9NZQ8", "symbol") == "Q9NZQ8"            # no hit: a legal symbol-shaped name


def test_align_reports_duplicates_junk_and_unmappable_panel_genes():
    gm = hz.GeneMap.from_table([{"symbol": "CD274", "alias": "PD-L1"}], aliases=("alias",))
    feats = {"a": ["CD8A", "CD8A", "Unnamed: 0", "P2RY12", "two words", "a:b", "12345", "CD274", "PD-L1"]}
    aln = gm.align(feats, key="symbol")
    assert aln.index == ["CD8A", "P2RY12", "CD274"]
    assert aln.positions["a"].tolist() == [0, 3, 7]                # the first occurrence wins
    r = aln.report["a"]
    assert (r["n_in"], r["n_mapped"], r["n_duplicate"], r["n_unmapped"]) == (9, 3, 2, 4)
    assert r["n_in"] == r["n_mapped"] + r["n_duplicate"] + r["n_unmapped"]
    assert set(r["examples"]) == {"Unnamed: 0", "two words", "a:b", "12345"}
    assert "duplicate" in aln.explain()

    em = hz.GeneMap.from_table([{"symbol": "CD274", "ensembl": ENS["CD274"]}], ensembl="ensembl")
    pn = em.align({"a": ["CD274"], "b": [ENS["CD274"]]}, key="ensembl",
                  panel=["CD274", "NOT_A_GENE", "ENSG00000000009", "CD274"])
    assert pn.index == [ENS["CD274"], "ENSG00000000009"]
    assert pn.panel_missing == ["NOT_A_GENE"]
    assert pn.report["a"]["panel_missing"] == ["NOT_A_GENE"] and pn.report["b"]["panel_missing"] == ["NOT_A_GENE"]
    assert pn.report["a"]["n_kept"] == 1
    assert em.align({"a": ["CD274"]}, key="ensembl").report["a"]["panel_missing"] == []


# ================================================================ 3. the wrong gene's ids ==
@pytest.mark.parametrize("flip", [False, True])
def test_records_with_the_same_symbol_merge_whatever_the_row_order(flip):
    rows = [{"symbol": "TP53", "ensembl": "", "entrez": ""},
            {"symbol": "TP53", "ensembl": "ENSG00000141510", "entrez": "7157"}]
    gm = hz.GeneMap.from_table(rows[::-1] if flip else rows, symbol="symbol", ensembl="ensembl", entrez="entrez")
    assert len(gm) == 1
    assert gm.to(["TP53"], key="entrez") == ["7157"] and gm.to(["7157"], key="ensembl") == ["ENSG00000141510"]


def test_add_unions_ids_and_aliases_of_a_symbol_and_does_not_touch_the_callers_record():
    gm = hz.GeneMap()
    first = hz.GeneRecord(symbol="TP53", aliases=("p53",))
    gm.add(first)
    gm.add(hz.GeneRecord(symbol="tp53", aliases=("LFS1",), uniprot="P04637"))
    assert len(gm) == 1 and set(gm.resolve("TP53").aliases) == {"p53", "LFS1"}
    assert gm.to(["LFS1"], key="uniprot") == ["P04637"] and gm.ambiguous("TP53") is False
    assert first.aliases == ("p53",) and first.uniprot == ""        # the map keeps its own copy


def test_conflicting_same_symbol_records_stay_apart_and_resolve_by_the_requested_key():
    gm = hz.GeneMap([hz.GeneRecord(symbol="TBCE", ensembl="ENSG00000000001"),
                     hz.GeneRecord(symbol="TBCE", ensembl="ENSG00000000002", entrez="7")])
    assert len(gm) == 2
    assert gm.to(["TBCE"], key="entrez") == ["7"]                   # not whichever row came first
    assert gm.expand(["TBCE"], key="ensembl") == ["ENSG00000000001", "ENSG00000000002"]
    assert gm.canonical("ENSG00000000002", "ensembl") == "ENSG00000000002"


def test_resolve_among_alias_hits_prefers_the_record_that_has_the_requested_id():
    gm = hz.GeneMap([hz.GeneRecord(symbol="A1", aliases=("OLD",)),
                     hz.GeneRecord(symbol="A2", aliases=("OLD",), ensembl="ENSG00000000002")])
    assert gm.resolve("OLD").symbol == "A1" and gm.resolve("OLD", key="ensembl").symbol == "A2"
    assert gm.to(["OLD"], key="ensembl") == ["ENSG00000000002"]
    assert gm.canonical("OLD", "ensembl") == "ENSG00000000002"


def test_expand_returns_the_symbol_owner_only_but_still_expands_a_shared_alias():
    gm = hz.GeneMap([hz.GeneRecord(symbol="OLDNAME", aliases=("X",), ensembl="ENSG00000000001"),
                     hz.GeneRecord(symbol="X", ensembl="ENSG00000000002")])
    assert gm.expand(["X"], key="ensembl") == ["ENSG00000000002"]
    assert gm.to(["X"], key="ensembl") == ["ENSG00000000002"]
    cx = hz.GeneMap([hz.GeneRecord(symbol="A", aliases=("CX",), ensembl="ENSG00000000001"),
                     hz.GeneRecord(symbol="B", aliases=("CX",), ensembl="ENSG00000000002")])
    assert cx.expand(["CX"], key="ensembl") == ["ENSG00000000001", "ENSG00000000002"]


def test_multivalued_ids_on_a_single_gene_row_keep_the_first_and_alias_the_rest(tmp_path):
    p = tmp_path / "hgnc.tsv"
    p.write_text("hgnc_id\tsymbol\talias_symbol\tprev_symbol\tensembl_gene_id\tentrez_id\tuniprot_ids\n"
                 "HGNC:477\tAMY1A\t\t\tENSG00000237763\t276\tP04745|P0DUB6\n")
    gm = hz.GeneMap.from_hgnc(str(p))
    assert gm.to(["AMY1A"], key="uniprot") == ["P04745"]
    assert gm.to(["P0DUB6"], key="symbol") == ["AMY1A"] and "P0DUB6" in gm.resolve("AMY1A").aliases
    # a complex row whose id list cannot be paired positionally is still not guessed at
    cx = hz.GeneMap.from_table([{"s": "CD3", "m": "CD3D; CD3E", "u": "P04234|P07766|P09693"}], symbol="s",
                               members="m", uniprot="u")
    assert cx.to(["CD3D", "CD3E"], key="uniprot") == [None, None]


def test_ensembl_version_suffixes_are_stripped_where_ids_are_stored():
    a = hz.GeneMap.from_features(["ENSG00000141510.12"], ["TP53"])
    b = hz.GeneMap.from_features(["ENSG00000141510.14"], ["TP53"])
    assert a.resolve("TP53").ensembl == "ENSG00000141510"
    a.merge(b)
    assert len(a) == 1 and a.expand(["TP53"], key="ensembl") == ["ENSG00000141510"]
    gm = hz.GeneMap()
    gm.add(hz.GeneRecord(symbol="X", ensembl="ENSG00000000001.7", aliases=("ENSG00000000003.2",)))
    assert gm.panel_table(["X"], key="ensembl")[0]["ensembl"] == "ENSG00000000001"
    assert gm.to(["ENSG00000000003"], key="symbol") == ["X"]
    assert gm.resolve("X").aliases == ("ENSG00000000003",)


def test_complex_names_cover_all_their_members_in_panels_tables_and_coverage():
    gm = _node_map()
    aln = gm.align({"a": ["CD3D", "CD3E", "CD274"]}, key="symbol", panel=["CD3", "CD274"])
    assert aln.index == ["CD3D", "CD3E", "CD274"] and aln.positions["a"].tolist() == [0, 1, 2]
    rows = gm.panel_table(["CD3", "CD274"], key="ensembl")
    assert [r["id"] for r in rows] == [ENS["CD3D"], ENS["CD3E"], ENS["CD274"]]
    assert [r["query"] for r in rows] == ["CD3", "CD3", "CD274"] and all(r["known"] for r in rows)
    o = g.orchestrator()
    o.describe([{"id": "p", "accession": "GSE00001", "CD_focus": "CD3", "CD_protein_targets (listed)": "CD3E"}])
    o.harmonizer.genes.merge(gm)
    cov = o.coverage(genes=["CD3", "CD3D", "CD3E", "CD47"])
    assert cov.code("CD3", "p") == "PF" and cov.code("CD3D", "p") == "F" and cov.code("CD3E", "p") == "PF"
    assert cov.code("CD47", "p") == ""


def test_a_build_panel_naming_a_complex_gets_every_member_on_the_axis(tmp_path):
    src = str(tmp_path / "GSE00001.h5ad")
    _h5ad(src, 20, ["CD3D", "CD3E", "CD274"])
    o = g.orchestrator()
    o.harmonizer.genes.merge(_node_map())
    _local_piece(o, "p", src)
    pz = o.build(o.blueprint("b", panel=["CD3", "CD274"], key="symbol"), str(tmp_path / "out"))
    var = pz.read_var().to_pylist()
    assert [v["id"] for v in var] == ["CD3D", "CD3E", "CD274"] and all(v["in_p"] for v in var)
    assert pz.n_vars == 3 and pz.built[0]["n_kept"] == 3


# ============================================================================ 4. build() ==
def _two_piece_orchestrator(tmp_path, n_a=20, n_c=15):
    a = str(tmp_path / "GSE00001_a.h5ad")
    XA = _h5ad(a, n_a, ["AAA1", "AAA2", "AAA3", "AAA4"], seed=1)
    c_dir = str(tmp_path / "PRJNA000003_c")
    XC = _mtx_dir(c_dir, n_c, ["AAA2", "AAA3", "AAA5"], seed=2)
    o = g.orchestrator(budget=g.Budget(ram="32MB"))
    o.describe([{"id": "A", "accession": "GSE00001", "modalities": "scRNA-seq"},
                {"id": "C", "accession": "PRJNA000003", "modalities": "scRNA-seq"}])
    o.attach("A", a)
    o.attach("C", c_dir)
    return o, a, c_dir, XA, XC


def test_rebuilding_into_the_same_dest_drops_regions_this_build_did_not_produce(tmp_path):
    o, *_ = _two_piece_orchestrator(tmp_path)
    dest = str(tmp_path / "puz")
    first = o.build(o.blueprint("first", key="symbol", how="union"), dest)
    assert {r.name for r in first.harvest} == {"X_A", "X_C", "obs", "var", "pieces"}
    second = o.build(o.blueprint("second", pieces=["A"], key="symbol"), dest)
    assert {r.name for r in second.harvest} == {"X_A", "obs", "var", "pieces"}
    man = json.load(open(os.path.join(dest, "manifest.json")))
    assert {r["name"] for r in man["regions"]} == {"X_A", "obs", "var", "pieces"}
    assert not os.path.exists(os.path.join(dest, "X_C.parquet"))
    disp = json.load(open(os.path.join(dest, "dispatch.json")))
    assert "X_C" not in {i["name"] for items in disp["targets"].values() for i in items}
    assert all(g.harvest(dest).verify().values())
    assert second.read_obs().num_rows == second.n_obs == 20


def test_rebuilding_leaves_regions_another_producer_put_in_the_harvest(tmp_path):
    o, *_ = _two_piece_orchestrator(tmp_path)
    dest = str(tmp_path / "puz")
    h = g.harvest(dest)
    h.add_table("patches", {"seed": np.arange(3)}, kind="patch", targets=["regulus"])
    pz = o.build(o.blueprint("first", key="symbol", how="union"), dest)
    assert {r.name for r in pz.harvest} == {"patches", "X_A", "X_C", "obs", "var", "pieces"}
    pz2 = o.build(o.blueprint("second", pieces=["A"], key="symbol"), dest)
    assert {r.name for r in pz2.harvest} == {"patches", "X_A", "obs", "var", "pieces"}
    assert os.path.exists(os.path.join(dest, "patches.parquet")) and all(g.harvest(dest).verify().values())


def test_rebuilding_with_overwrite_false_refuses_before_touching_anything(tmp_path):
    o, *_ = _two_piece_orchestrator(tmp_path)
    dest = str(tmp_path / "puz")
    o.build(o.blueprint("first", key="symbol", how="union"), dest)
    before = {n: _sha(os.path.join(dest, n)) for n in os.listdir(dest)}
    with pytest.raises(ValueError, match="overwrite"):
        o.build(o.blueprint("again", pieces=["A"], key="symbol"), dest, overwrite=False)   # would change obs
    assert {n: _sha(os.path.join(dest, n)) for n in os.listdir(dest)} == before


def test_harmonizer_log_in_the_provenance_is_this_builds_alone(tmp_path):
    o, *_ = _two_piece_orchestrator(tmp_path)
    p1 = o.build(o.blueprint("one", key="symbol", how="union"), str(tmp_path / "p1"))
    p2 = o.build(o.blueprint("two", key="symbol", how="union"), str(tmp_path / "p2"))
    for pz in (p1, p2):
        log = json.load(open(os.path.join(pz.path, "blueprint.json")))["harmonizer"]["log"]
        assert [s["step"] for s in log].count("align") == 1
        assert sorted(s["piece"] for s in log if s["step"] == "obs-columns") == ["A"]     # C is a triple: no obs columns
    assert len(p2.provenance["harmonizer"]["log"]) == len(p1.provenance["harmonizer"]["log"])


def test_a_piece_records_the_hash_of_its_source_and_of_its_output_separately(tmp_path):
    o, a, c_dir, *_ = _two_piece_orchestrator(tmp_path)
    pz = o.build(o.blueprint("b", key="symbol", how="union"), str(tmp_path / "puz"))
    by = {b["piece"]: b for b in pz.built}
    assert by["A"]["source_sha256"] == _sha(a)
    assert by["C"]["source_sha256"] == _sha(os.path.join(c_dir, "matrix.mtx.gz"))      # the triple's matrix
    for pid in ("A", "C"):
        out = os.path.join(pz.path, by[pid]["output"])
        assert by[pid]["output_sha256"] == _sha(out) == pz.harvest.rois[f"X_{pid}"].sha256
        assert by[pid]["source_sha256"] != by[pid]["output_sha256"]
        assert by[pid]["sha256"] == by[pid]["output_sha256"]          # the old key keeps its old meaning
    prov = json.load(open(os.path.join(pz.path, "blueprint.json")))
    assert prov["pieces"][0]["source_sha256"] == _sha(a)


def test_a_table_piece_hashes_its_source_file_too(tmp_path):
    t = tmp_path / "GSE00005_hits.csv"
    t.write_text("gene,score\nCD274,1\nCD47,2\n")
    o = g.orchestrator()
    _local_piece(o, "T", t, accession="GSE00005", modalities="pooled CRISPR screen")
    pz = o.build(o.blueprint("b"), str(tmp_path / "puz"))
    b = pz.built[0]
    assert b["reader"] == "table" and b["source_sha256"] == _sha(str(t)) != b["output_sha256"]


def test_a_piece_that_fails_while_streaming_becomes_a_hole_and_leaves_no_partial_output(tmp_path):
    o, *_ = _two_piece_orchestrator(tmp_path)
    bad = str(tmp_path / "GSE00009_bad.h5ad")
    _h5ad(bad, 12, ["AAA1", "AAA2", "AAA3"], seed=3, donor="alien")        # obs column no reader can decode
    o.describe([{"id": "B", "accession": "GSE00009", "modalities": "scRNA-seq"}])
    o.attach("B", bad)
    dest = str(tmp_path / "puz")
    pz = o.build(o.blueprint("b", pieces=["A", "B", "C"], key="symbol", how="union"), dest)
    assert [b["piece"] for b in pz.built] == ["A", "C"]
    assert [(h["piece"], h["status"]) for h in pz.holes] == [("B", "unreadable")]
    assert "donor" in pz.holes[0]["note"] or "encoding" in pz.holes[0]["note"]
    assert sorted(os.listdir(dest)) == sorted(["X_A.parquet", "X_C.parquet", "obs.parquet", "var.parquet",
                                               "pieces.parquet", "manifest.json", "blueprint.json", "dispatch.json"])
    assert pz.read_obs().num_rows == pz.n_obs == 35                       # no rows of the failed piece
    assert pz.read_X("C").column("cell").to_numpy().min() >= 20          # C follows A: B took no rows
    assert all(g.harvest(dest).verify().values())
    assert [p["built"] for p in pz.harvest.read("pieces").to_pylist()] == [True, False, True]
    assert "in_B" not in pz.read_var().column_names and "B" not in pz.alignment.report
    assert o.budget.stats()["ram"]["used"] == 0                           # the failed piece gave its lease back


def test_a_failure_in_the_middle_of_a_piece_removes_the_partial_x_file(tmp_path, monkeypatch):
    o, *_ = _two_piece_orchestrator(tmp_path)
    real = orc._PieceReader.coo_batches

    def flaky(self, positions, base=0):
        it = real(self, positions, base)
        yield next(it)
        if os.path.basename(self.path).startswith("GSE00001"):
            raise RuntimeError("boom mid-stream")
        yield from it
    monkeypatch.setattr(orc._PieceReader, "coo_batches", flaky)
    dest = str(tmp_path / "puz")
    pz = o.build(o.blueprint("b", key="symbol", how="union"), dest)
    assert [h["piece"] for h in pz.holes] == ["A"] and "boom mid-stream" in pz.holes[0]["note"]
    assert not os.path.exists(os.path.join(dest, "X_A.parquet"))
    assert [b["piece"] for b in pz.built] == ["C"] and pz.read_obs().num_rows == 15
    assert pz.built[0]["output_sha256"] == _sha(os.path.join(dest, "X_C.parquet"))
    assert o.budget.stats()["ram"]["used"] == 0


def test_the_manifest_is_written_even_when_the_build_dies(tmp_path, monkeypatch):
    o, *_ = _two_piece_orchestrator(tmp_path)

    def boom(self, panel, key="symbol", **kw):
        raise RuntimeError("late failure")
    monkeypatch.setattr(hz.GeneMap, "panel_table", boom)
    dest = str(tmp_path / "puz")
    with pytest.raises(RuntimeError, match="late failure"):
        o.build(o.blueprint("b", key="symbol", how="union"), dest)
    man = json.load(open(os.path.join(dest, "manifest.json")))
    assert {r["name"] for r in man["regions"]} == {"X_A", "X_C", "obs"}
    assert all(g.harvest(dest).verify().values())
    assert not [n for n in os.listdir(dest) if n.startswith(".") or n.endswith(".part")]


def test_a_build_with_no_shared_features_is_an_error_unless_allowed(tmp_path):
    a, b = str(tmp_path / "GSE00001.h5ad"), str(tmp_path / "GSE00002.h5ad")
    _h5ad(a, 10, ["AAA1", "AAA2"])
    _h5ad(b, 10, ["BBB1", "BBB2"], seed=2)
    o = g.orchestrator()
    o.describe([{"id": "A", "accession": "GSE00001"}, {"id": "B", "accession": "GSE00002"}])
    o.attach("A", a)
    o.attach("B", b)
    with pytest.raises(ValueError, match="share no features"):
        o.build(o.blueprint("b", key="symbol", how="intersection"), str(tmp_path / "o1"))
    with pytest.raises(ValueError, match="share no features"):
        o.build(o.blueprint("b", key="symbol", panel=["ZZZ1"], how="union"), str(tmp_path / "o2"))
    assert not os.path.exists(str(tmp_path / "o1" / "manifest.json"))     # nothing half-written
    pz = o.build(o.blueprint("b", key="symbol", how="intersection"), str(tmp_path / "o3"), allow_empty=True)
    assert pz.n_vars == 0 and pz.n_obs == 20
    union = o.build(o.blueprint("u", key="symbol", how="union"), str(tmp_path / "o4"))
    assert union.n_vars == 4                                            # a union is never empty here


def test_a_piece_id_that_is_not_a_file_name_cannot_write_outside_dest(tmp_path):
    src = str(tmp_path / "GSE00001.h5ad")
    _h5ad(src, 10, ["AAA1", "AAA2"])
    o = g.orchestrator()
    o.add(orc.Piece(id="../esc/ape x", accessions=[orc.Accession("GSE00001", "geo_series")], local=[src]))
    dest = tmp_path / "deep" / "puz"
    pz = o.build(o.blueprint("b"), str(dest))
    assert pz.holes == [] and not (tmp_path / "deep" / "esc").exists()
    assert all(os.path.dirname(os.path.realpath(os.path.join(pz.path, r.path))) == os.path.realpath(pz.path)
               for r in pz.harvest)


def test_xlsx_rows_keep_their_real_sheet_row_numbers(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Datasets"
    ws.append(["id", "accession", "modalities"])
    ws.append(["one", "GSE00010", "scRNA-seq"])
    ws.append([None, None, None])
    ws.append(["two", "GSE00011", "scRNA-seq"])
    ws.cell(row=7, column=1, value="three")
    ws.cell(row=7, column=2, value="GSE00012")
    p = tmp_path / "book.xlsx"
    wb.save(p)
    pieces = orc.read_descriptions(str(p))
    assert [(x.id, x.source.rsplit(":", 1)[1]) for x in pieces] == [("one", "2"), ("two", "4"), ("three", "7")]


def test_csv_rows_keep_their_real_line_numbers(tmp_path):
    p = tmp_path / "inv.csv"
    p.write_text('id,accession,notes\none,GSE00010,"multi\nline"\n\ntwo,GSE00011,x\n,,\nthree,GSE00012,y\n')
    pieces = orc.read_descriptions(str(p))
    assert [(x.id, x.source.rsplit(":", 1)[1]) for x in pieces] == [("one", "2"), ("two", "5"), ("three", "7")]


# ============================================================================= 5. facets ==
def test_column_pins_reach_the_facets_not_only_the_piece_fields():
    rows = [{"dataset": "x1", "Org": "mouse", "Tissue type": "lung", "Method": "scRNA-seq; CROP-seq",
             "Dx": "GBM", "Chip": "10x 3' v3", "Treatment": "CRISPRi", "Donor ID": "D9"}]
    plain = orc.read_descriptions(rows)[0]
    assert plain.facets["organism"] == "" and plain.facets["tissue"] == ""
    pins = {"species": "Org", "tissue": "Tissue type", "modalities": "Method", "disease": "Dx",
            "platform": "Chip", "perturbation_type": "Treatment", "donor": "Donor ID"}
    p = orc.read_descriptions(rows, columns=pins)[0]
    assert p.facets["organism"] == "Mus musculus" and p.facets["tissue"] == "lung"
    assert p.facets["assay"] == "scRNA-seq" and p.facets["disease"] == "glioblastoma"
    assert p.facets["platform"] == "10x 3' v3" and p.facets["condition"] == "CRISPRi" and p.facets["donor"] == "D9"
    assert p.modalities == ["scRNA-seq", "CROP-seq"]
    h = hz.Harmonizer()
    assert h.facets(rows[0], columns={"species": "Org"})["organism"] == "Mus musculus"


def test_missing_values_from_pandas_polars_and_arrow_are_empty_not_nan(tmp_path):
    pd = pytest.importorskip("pandas")
    pl = pytest.importorskip("polars")
    import pyarrow as pa
    data = [{"id": "a", "tissue": None, "n_cells": None, "modalities": "scRNA-seq", "notes": None},
            {"id": "b", "tissue": "lung", "n_cells": 5, "modalities": "scRNA-seq", "notes": "hi"}]
    frames = {"pandas": pd.DataFrame(data).astype({"n_cells": "float64"}), "polars": pl.DataFrame(data),
              "arrow": pa.Table.from_pylist(data), "list": data}
    for name, src in frames.items():
        a, b = orc.read_descriptions(src)
        assert a.facets["tissue"] == "" and "n_cells" not in a.expected and a.notes == "", name
        assert a.row["tissue"] == "" and a.row["n_cells"] == "", name
        assert b.facets["tissue"] == "lung" and b.expected["n_cells"] == 5 and b.notes == "hi", name
    assert orc._first_int(float("nan")) is None and orc._first_int(np.float64("nan"), orc._UNITS["n_cells"]) is None


def test_disease_ms_is_a_whole_word_and_healthy_loses_to_any_disease():
    v = hz.Vocabulary()
    assert v.disease("MS vs control") == "multiple sclerosis"
    assert v.disease("100 MS patients, Iranian cohort") == "multiple sclerosis"
    assert v.disease("ms") == "multiple sclerosis" and v.disease("RRMS, MS") == "multiple sclerosis"
    assert v.disease("glioma vs normal brain") == "glioma"
    assert v.disease("healthy control") == "healthy" and v.disease("normal") == "healthy"
    assert v.disease("mssql export") != "multiple sclerosis"
    assert v.disease("MSS") != "multiple sclerosis" and v.disease("MSS colorectal") == "colorectal cancer"


@pytest.mark.parametrize("text, expected", [
    ("CRISPRi knockdown", "CRISPRi"), ("CRISPRko knockout", "CRISPRko"), ("shRNA knockdown", "shRNA/siRNA"),
    ("CRISPRa overexpression screen", "CRISPRa"), ("CRISPRi + small molecule", "CRISPRi"),
    ("CRISPRi knockout library", "CRISPRi"), ("genome-wide CRISPR-Cas9 knockout library", "CRISPRko"),
    ("barcoded TF-ORF library (CRISPRa-style overexpression)", "ORF overexpression"),
    ("knockdown", "shRNA/siRNA"), ("knockout", "CRISPRko"), ("drug", "chemical"), ("", ""),
])
def test_explicit_perturbation_technologies_beat_generic_effect_words(text, expected):
    assert hz.Vocabulary().perturbation(text) == expected


def test_organism_keeps_the_first_and_organisms_lists_all():
    v = hz.Vocabulary()
    assert v.organism("human; mouse") == "Homo sapiens" and v.organism("mouse; human") == "Mus musculus"
    assert v.organisms("human; mouse") == ["Homo sapiens", "Mus musculus"]
    assert v.organisms("Mus musculus (mouse) and human") == ["Mus musculus", "Homo sapiens"]
    assert v.organisms("") == [] and v.organisms("axolotl") == [] and v.organism("axolotl") == "axolotl"


def test_open_access_needs_a_whole_word_not_a_substring():
    v = hz.Vocabulary()
    assert v.access("data in the publication") == ""
    assert v.access("see the supplementary publication files") == ""
    assert v.access("publicly available") == "open" and v.access("public") == "open"
    assert v.access("Open (GEO)") == "open" and v.access("CC-BY-4.0") == "open"
    assert v.access("Open (GEO/SCP); raw reads controlled via EGA") == "controlled"
    assert orc.read_descriptions([{"id": "x", "access": "data in the publication"}])[0].access == ""


def test_facets_carry_a_donor_when_the_row_names_one():
    v = hz.Vocabulary()
    assert v.facets_of({"donor_id": "MGH-01"})["donor"] == "MGH-01"
    assert v.facets_of({"Patient ID": "S7"})["donor"] == "S7"
    assert v.facets_of({"subject": "S7"}, donor_keys=("subject",))["donor"] == "S7"
    assert v.facets_of({"x": 1})["donor"] == ""
    assert orc.read_descriptions([{"dataset_id": "x", "donor": "D1"}])[0].facets["donor"] == "D1"


# =========================================================================== 6. parsing ==
@pytest.mark.parametrize("text, ids, kinds", [
    ("GSE12345_RAW.tar", ["GSE12345"], ["geo_series"]),
    ("data/GSE12345_counts.csv", ["GSE12345"], ["geo_series"]),
    ("sample_GSE12345_counts", ["GSE12345"], ["geo_series"]),
    ("SRR1448774_1.fastq.gz", ["SRR1448774"], ["sra_run"]),
    ("gse12345", ["GSE12345"], ["geo_series"]),
    ("gsm123456 and srp123456 and prjna123456", ["GSM123456", "SRP123456", "PRJNA123456"],
     ["geo_sample", "sra_study", "bioproject"]),
    ("err1234567 drr7654321", ["ERR1234567", "DRR7654321"], ["sra_run", "sra_run"]),
    ("PXD012345", ["PXD012345"], ["pride"]),
    ("deposited as syn12345678 (Synapse)", ["syn12345678"], ["synapse"]),
    ("MTBLS1234", ["MTBLS1234"], ["metabolights"]),
    ("SRS1234567; ERS7654321", ["SRS1234567", "ERS7654321"], ["sra_sample", "sra_sample"]),
    ("GDS1234", ["GDS1234"], ["geo_dataset"]),
    ("PRJCA000123", ["PRJCA000123"], ["gsa_project"]),
    ("OMIX000123", ["OMIX000123"], ["omix"]),
    ("EGAF00001234567", ["EGAF00001234567"], ["ega_file"]),
    ("GSE12345abc", [], []), ("xGSE12345", [], []), ("GSE", [], []), ("ERR123", [], []),
])
def test_find_accessions_treats_underscore_as_a_boundary_and_knows_more_hosts(text, ids, kinds):
    found = orc.find_accessions(text)
    assert [a.id for a in found] == ids and [a.kind for a in found] == kinds


@pytest.mark.parametrize("text, expected", [
    ("IL2 and IL7", ["IL2", "IL7"]), ("IL2 & IL7", ["IL2", "IL7"]), ("CD3, CD4, and CD8", ["CD3", "CD4", "CD8"]),
    ("MYC/MAX", ["MYC", "MAX"]), ("TP53 (p53)", ["TP53", "p53"]), ("CD274 (PD-L1); CD47", ["CD274", "PD-L1", "CD47"]),
    ("CD274 or PD-L1", ["CD274", "PD-L1"]), ("CD274; CD47", ["CD274", "CD47"]), ("CD274 (KO)", ["CD274"]),
    ("12", []), ("N/A", []), ("genome-wide", []), ("genome-wide (18,000 genes)", []), ("none", []),
    ("ENSG00000120217; 29126", ["ENSG00000120217", "29126"]), ("CD274; CD274", ["CD274"]),
])
def test_gene_cells_are_split_on_and_ampersand_slash_and_parentheses(text, expected):
    assert orc._genes(text) == expected


def test_duplicate_column_headers_are_kept_not_collapsed(tmp_path):
    p = tmp_path / "dup.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "Notes", "Notes", "Notes.1"])
        w.writerow(["a", "first note", "see GSE00042", "third"])
    pc = orc.read_descriptions(str(p))[0]
    assert pc.notes == "first note" and pc.accession_ids() == ["GSE00042"]
    assert len(pc.row) == 4 and set(pc.row.values()) == {"a", "first note", "see GSE00042", "third"}
    assert pc.row["Notes"] == "first note" and pc.row["Notes.1"] == "third" and pc.row["Notes.2"] == "see GSE00042"
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["id", "Notes", "Notes"])
    ws.append(["a", "first note", "see GSE00042"])
    wb.save(tmp_path / "dup.xlsx")
    px = orc.read_descriptions(str(tmp_path / "dup.xlsx"))[0]
    assert px.row == {"id": "a", "Notes": "first note", "Notes.1": "see GSE00042"} and px.accession_ids() == ["GSE00042"]
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame([["a", "first note", "see GSE00042"]], columns=["id", "Notes", "Notes"])
    assert orc.read_descriptions(df)[0].row == {"id": "a", "Notes": "first note", "Notes.1": "see GSE00042"}


def test_piece_ids_are_unique_however_the_rows_collide():
    rows = [{"id": "A", "accession": "GSE00001"}, {"id": "A", "accession": "GSE00002"},
            {"id": "A_2", "accession": "GSE00003"}, {"id": "A", "accession": "GSE00004"},
            {"id": "A_2", "accession": "GSE00005"}]
    ids = [p.id for p in orc.read_descriptions(rows)]
    assert len(set(ids)) == 5 and ids[0] == "A"
    o = g.orchestrator()
    o.describe(rows)
    o.describe(rows)
    o.describe(rows)
    assert len(o) == 15 and len(set(o.pieces)) == 15 and all(k == p.id for k, p in o.pieces.items())


def test_a_count_is_not_a_number_that_precedes_something_else():
    cells = orc._UNITS["n_cells"]
    assert orc._first_int("44 datasets", cells) is None
    assert orc._first_int("12 studies; 3 cohorts", cells) is None
    assert orc._first_int("50 libraries", cells) is None and orc._first_int("228 antibodies", cells) is None
    assert orc._first_int("20-plex", cells) is None and orc._first_int("20-plex", orc._UNITS["n_adt"]) == 20
    assert orc._first_int("44 public datasets, 3.1M cells", cells) == 3_100_000
    assert orc._first_int("44 datasets, >1.2M cells", cells) == 1_200_000
    assert orc._first_int("300", cells) == 300 and orc._first_int("~1,000", cells) == 1000      # bare numbers stay counts
    assert orc._first_int("100 MS patients", cells) == 100                                     # a size, as before
    assert orc._first_int("44 datasets", cells, require_unit=False) is None
    assert orc._first_int("44 datasets") == 44                                                  # no units asked: first number
    row = orc.read_descriptions([{"id": "x", "n_cells": "44 datasets"}])[0]
    assert "n_cells" not in row.expected and row.expected["n_cells_text"] == "44 datasets"


# ============================================================================ 7. handoff ==
@pytest.mark.parametrize("name", ["../escaped", "a/b", "a b", "", "..", ".", "x\\y", "a:b", "é"])
def test_harvest_refuses_names_that_are_not_file_names(tmp_path, name):
    h = g.harvest(str(tmp_path / "root"))
    with pytest.raises(ValueError, match="name"):
        h.add_table(name, {"x": np.arange(3)})

    class V:
        window = (0, 1, 0, 1)
        path = "p"

        def load(self):
            return {"x": np.arange(2)}
    with pytest.raises(ValueError, match="name"):
        h.add(V(), name)
    assert len(h) == 0 and not (tmp_path / "escaped.parquet").exists()
    assert sorted(os.listdir(tmp_path / "root")) == []


def test_harvest_accepts_ordinary_names_and_will_not_write_through_an_escaping_symlink(tmp_path):
    h = g.harvest(str(tmp_path / "root"))
    assert h.add_table("X_my-piece.v2", {"x": np.arange(3)}).path == "X_my-piece.v2.parquet"
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"precious")
    os.symlink(outside, tmp_path / "root" / "evil.parquet")
    with pytest.raises(ValueError, match="outside"):
        h.add_table("evil", {"x": np.arange(3)})
    assert outside.read_bytes() == b"precious" and "evil" not in h.rois


def test_dict_columns_of_every_array_like_type_are_written_and_scalars_are_refused(tmp_path):
    pd = pytest.importorskip("pandas")
    pl = pytest.importorskip("polars")
    import pyarrow as pa
    h = g.harvest(str(tmp_path / "h"))
    h.add_table("series", {"a": pd.Series([1, 2, 3]), "b": pl.Series([4, 5, 6]), "c": pa.chunked_array([[7, 8, 9]]),
                           "d": np.arange(3), "e": ["x", "y", "z"], "f": (1.5, 2.5, 3.5), "g": pa.array([1, 2, 3])})
    t = h.read("series")
    assert t.column_names == ["a", "b", "c", "d", "e", "f", "g"]
    assert t.column("b").to_pylist() == [4, 5, 6] and t.column("c").to_pylist() == [7, 8, 9]
    with pytest.raises(TypeError, match="scalar"):
        h.add_table("scalar", {"x": np.arange(3), "n": 5})
    with pytest.raises(TypeError, match="column"):
        h.add_table("weird", {"x": object()})
    assert "scalar" not in h.rois and "weird" not in h.rois
    assert not os.path.exists(os.path.join(h.path, "scalar.parquet"))


def test_ragged_nested_lists_become_list_columns(tmp_path):
    h = g.harvest(str(tmp_path / "h"))
    h.add_table("nested", {"x": [[1, 2], [3], []], "y": [np.array([1.0, 2.0]), np.array([3.0]), None],
                           "z": np.zeros((3, 2)), "w": ["a", None, "c"]})
    t = h.read("nested")
    assert t.column("x").to_pylist() == [[1, 2], [3], []]
    assert t.column("y").to_pylist() == [[1.0, 2.0], [3.0], None]
    assert t.column("z").to_pylist() == [[0.0, 0.0]] * 3 and t.column("w").to_pylist() == ["a", None, "c"]


def test_npy_rows_is_the_first_dimension(tmp_path):
    h = g.harvest(str(tmp_path / "h"))
    r = h.add_table("arr", np.zeros((4, 3)), fmt="npy")
    assert r.rows == 4 and h.read("arr").shape == (4, 3)
    assert h.add_table("vec", np.arange(7), fmt="npy").rows == 7
    assert h.add_table("zero", np.array(5.0), fmt="npy").rows == 1


def test_sparse_matrices_keep_their_shape_and_precision(tmp_path):
    import pyarrow as pa
    h = g.harvest(str(tmp_path / "h"))
    X = sp.random(10, 7, density=0.3, format="csr", dtype=np.float64, random_state=1)
    X.data = X.data + 1e-9                                           # needs more than float32 to survive
    r = h.add_table("sp", X)
    assert r.stat["shape"] == [10, 7] and r.stat["dtype"] == "float64"
    t = h.read("sp")
    assert t.schema.field("value").type == pa.float64() and t.num_rows == X.nnz
    dense = np.zeros((10, 7))
    dense[t.column("row").to_numpy(), t.column("col").to_numpy()] = t.column("value").to_numpy()
    np.testing.assert_array_equal(dense, X.toarray())
    r32 = h.add_table("sp32", X.astype(np.float32))
    assert r32.stat["dtype"] == "float32" and h.read("sp32").schema.field("value").type == pa.float32()
    empty_tail = sp.csr_matrix(([1.0], ([0], [0])), shape=(5, 4))
    assert h.add_table("tail", empty_tail, stat={"note": "kept"}).stat == {"note": "kept", "shape": [5, 4],
                                                                          "dtype": "float64"}
    man = json.load(open(h.manifest_path))
    assert next(e for e in man["regions"] if e["name"] == "tail")["stat"]["shape"] == [5, 4]


def test_add_staged_honours_overwrite_and_sanitises_a_derived_name(tmp_path):
    import re

    import pyarrow as pa
    import pyarrow.parquet as pq
    st = g.staging(str(tmp_path / "stage"), capacity="100MB")

    def writer(n):
        def _w(path):
            pq.write_table(pa.table({"x": list(range(n))}), path)
            return {"rows": n, "format": "parquet"}
        return _w
    st.put("chip:a", writer(3))
    st.put("weird key/with:chars", writer(4))
    h = g.harvest(str(tmp_path / "h"))
    r = h.add_staged(st, "chip:a")
    assert r.name == "chip-a" and r.rows == 3
    with pytest.raises(ValueError, match="already exists"):
        h.add_staged(st, "chip:a")
    assert h.rois["chip-a"] is r
    st.put("chip:a", writer(5), overwrite=True)
    assert h.add_staged(st, "chip:a", overwrite=True).rows == 5 and h.rois["chip-a"].rows == 5
    r3 = h.add_staged(st, "weird key/with:chars")
    assert re.fullmatch(r"[A-Za-z0-9._-]+", r3.name) and os.path.exists(os.path.join(h.path, r3.path))
    with pytest.raises(ValueError, match="name"):
        h.add_staged(st, "chip:a", name="../x")
    assert all(h.verify().values())


def test_a_damaged_manifest_is_an_error_not_an_empty_harvest(tmp_path):
    root = tmp_path / "h"
    h = g.harvest(str(root))
    h.add_table("a", {"x": np.arange(3)})
    good = (root / "manifest.json").read_text()
    for bad in ["{not json", "", "[]", '{"regions": [{"source": "no name"}]}', '{"regions": "nope"}']:
        (root / "manifest.json").write_text(bad)
        with pytest.raises(ValueError, match="manifest"):
            g.harvest(str(root))
        assert (root / "manifest.json").read_text() == bad            # not rewritten by the failed attempt
    (root / "manifest.json").write_text(good)
    again = g.harvest(str(root))
    assert len(again) == 1 and again.add_table("b", {"x": np.arange(2)}) and len(g.harvest(str(root))) == 2


# ================================================================================ 8. thin ==
def test_a_long_hit_table_with_many_columns_is_not_taken_for_a_wide_matrix(tmp_path):
    t = tmp_path / "GSE00006_hits.csv"
    with open(t, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Gene", "Score", "FDR", "LFC", "Rank", "Hit", "Notes"])
        for i, s in enumerate(["CD274", "CD47", "GAPDH", "ZZZ9"]):
            w.writerow([s, i, 0.1, 0.5, i, "yes", "n"])
    o = g.orchestrator()
    _local_piece(o, "H", t, accession="GSE00006", modalities="pooled CRISPR screen")
    pz = o.build(o.blueprint("b", key="symbol"), str(tmp_path / "out"))
    b = pz.built[0]
    assert b["reader"] == "table" and b["gene_column"] == "Gene" and b["n_kept"] == 4
    t_out = pz.harvest.read("T_H")
    assert t_out.column_names == ["Gene", "Score", "FDR", "LFC", "Rank", "Hit", "Notes", "gene_symbol"]
    assert t_out.column("gene_symbol").to_pylist() == ["CD274", "CD47", "GAPDH", "ZZZ9"]


def test_a_wide_gene_matrix_is_still_recognised_by_its_symbol_entrez_headers(tmp_path):
    t = tmp_path / "GSE00007_wide.csv"
    cols = ["ModelID"] + [f"G{i} ({i + 100})" for i in range(25)]
    with open(t, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for m in range(3):
            w.writerow([f"ACH-{m}"] + [m + i for i in range(25)])
    o = g.orchestrator()
    _local_piece(o, "W", t, accession="GSE00007", modalities="pooled CRISPR screen")
    pz = o.build(o.blueprint("b", key="symbol"), str(tmp_path / "out"))
    out = pz.harvest.read("T_W").column_names
    assert out[0] == "ModelID" and out[1:4] == ["G0", "G1", "G2"] and pz.built[0]["gene_column"] == ""


def test_save_and_load_carry_the_gene_map_so_a_reloaded_orchestrator_builds_identically(tmp_path):
    src = str(tmp_path / "GSE00001.h5ad")
    _h5ad(src, 20, ["CD3D", "CD3E", "CD274"])
    o = g.orchestrator()
    o.harmonizer.genes.merge(_node_map())
    o.harmonizer.genes.add(hz.GeneRecord(symbol="XYZ1", aliases=("Old-XYZ",), uniprot="P12345"))
    o.harmonizer.columns = hz.ColumnMap(extra={"donor": ("sujeto",)})
    _local_piece(o, "p", src)
    path = o.save(str(tmp_path / "pieces.json"))
    o2 = orc.Orchestrator.load(path)
    gm, gm2 = o.harmonizer.genes, o2.harmonizer.genes
    assert len(gm2) == len(gm) == 5
    for name in ["CD3", "CD3E", "CD274", "Old-XYZ", "P12345", ENS["CD3D"]]:
        for key in ("symbol", "ensembl", "group", "uniprot"):
            assert gm2.expand([name], key=key) == gm.expand([name], key=key), (name, key)
    assert o2.harmonizer.columns.guess(["sujeto"]) == {"donor": "sujeto"}
    bp = o.blueprint("b", panel=["CD3", "CD274"], key="ensembl")
    a, b = o.build(bp, str(tmp_path / "a")), o2.build(bp, str(tmp_path / "b"))
    assert a.read_var().to_pylist() == b.read_var().to_pylist() and a.n_vars == b.n_vars == 3
    # files written before the gene map was persisted still load
    d = json.load(open(path))
    d.pop("genes")
    d.pop("columns", None)
    json.dump(d, open(tmp_path / "old.json", "w"))
    assert len(orc.Orchestrator.load(str(tmp_path / "old.json")).harmonizer.genes) == 0


# ============================================================================ extras ==
def test_a_cell_ranger_bundle_with_an_h5_opens_the_h5(tmp_path):
    from geminga.tenx import MtxTriple
    src = str(tmp_path / "src")
    _mtx_dir(src, 25, ["AAA1", "AAA2", "AAA3"])
    bundle = tmp_path / "data" / "GSE88888" / "outs"
    os.makedirs(bundle)
    MtxTriple(src).to_h5(str(bundle / "filtered_feature_bc_matrix.h5"))
    o = g.orchestrator()
    o.describe([{"id": "p", "accession": "GSE88888", "modalities": "scRNA-seq"}])
    o.resolve(search_dirs=str(tmp_path / "data"))                       # a single directory is fine too
    assert o.pieces["p"].status == "local"
    pz = o.build(o.blueprint("b"), str(tmp_path / "out"))
    b = pz.built[0]
    assert b["reader"] == "10x-h5" and b["opened"].endswith("filtered_feature_bc_matrix.h5") and pz.n_obs == 25
    assert b["source_sha256"] == _sha(b["opened"])


def test_a_puzzle_of_matrices_and_tables_reads_back_as_a_whole(tmp_path):
    pytest.importorskip("anndata")
    a = str(tmp_path / "GSE00001.h5ad")
    XA = _h5ad(a, 20, ["AAA1", "AAA2", "AAA3"], seed=1)
    t = tmp_path / "GSE00005_hits.csv"
    t.write_text("gene,score\nAAA1,1\nAAA2,2\n")
    o = g.orchestrator()
    o.describe([{"id": "A", "accession": "GSE00001", "modalities": "scRNA-seq"},
                {"id": "T", "accession": "GSE00005", "modalities": "pooled CRISPR screen"}])
    o.attach("A", a)
    o.attach("T", str(t))
    pz = o.build(o.blueprint("b", key="symbol"), str(tmp_path / "out"))
    assert [b["piece"] for b in pz.built] == ["A", "T"] and set(pz.measured()) == {"A"}
    ad = pz.to_anndata()
    assert ad.shape == (20, 3) and ad.X.sum() == XA.sum()


def test_an_interrupt_in_the_middle_of_a_piece_leaves_a_truthful_manifest(tmp_path, monkeypatch):
    o, *_ = _two_piece_orchestrator(tmp_path)
    real = orc._PieceReader.coo_batches

    def interrupted(self, positions, base=0):
        it = real(self, positions, base)
        yield next(it)
        if os.path.basename(self.path).startswith("PRJNA"):
            raise KeyboardInterrupt
        yield from it
    monkeypatch.setattr(orc._PieceReader, "coo_batches", interrupted)
    dest = str(tmp_path / "puz")
    with pytest.raises(KeyboardInterrupt):
        o.build(o.blueprint("b", key="symbol", how="union"), dest)
    man = json.load(open(os.path.join(dest, "manifest.json")))
    assert {r["name"] for r in man["regions"]} == {"X_A", "obs"}
    assert all(g.harvest(dest).verify().values())
    assert not os.path.exists(os.path.join(dest, "X_C.parquet"))
    assert not [n for n in os.listdir(dest) if n.startswith(".")]
    assert json.load(open(os.path.join(dest, "blueprint.json")))["complete"] is False


def test_hosts_acquire_cannot_price_get_a_status_and_a_tool_without_the_network():
    rows = [{"id": "pride", "accession": "PXD012345"}, {"id": "syn", "accession": "syn12345678"},
            {"id": "mtbls", "accession": "MTBLS1234"}, {"id": "sample", "accession": "SRS1234567"},
            {"id": "gds", "accession": "GDS1234"}, {"id": "gsa", "accession": "PRJCA000123"},
            {"id": "omix", "accession": "OMIX000123"}, {"id": "egaf", "accession": "EGAF00001234567"}]
    o = g.orchestrator()
    o.describe(rows)
    res = o.resolve(offline=True)
    assert {k: v.status for k, v in res.items()} == {
        "pride": "fetchable", "syn": "fetchable", "mtbls": "fetchable", "sample": "fetchable", "gds": "fetchable",
        "gsa": "fetchable", "omix": "fetchable", "egaf": "controlled"}
    assert res["syn"].needs == "synapseclient" and res["egaf"].needs == "pyega3"
    assert res["egaf"].controlled == ["EGAF00001234567"] and o.pieces["egaf"].access == "controlled"
    assert o.report()["needs_tools"] == ["synapseclient"]


def test_the_scalar_metadata_of_a_views_dict_payload_is_not_a_column(tmp_path):
    class V:
        window = (0, 10, 0, 10)
        path = "chip"
        zoom = 0

        def load(self):
            return {"x": np.arange(4), "y": np.arange(4) * 2, "bin_size": 50}      # as a GEF view returns it
    h = g.harvest(str(tmp_path / "h"))
    r = h.add(V(), "win")
    assert r.rows == 4 and h.read("win").column_names == ["x", "y"]
    assert h.add(V(), "win2", data={"x": np.arange(2), "tag": "note"}).rows == 2          # data= is the same


def test_missing_values_in_a_row_are_not_text_at_the_vocabulary_level_either():
    v = hz.Vocabulary()
    f = v.facets_of({"tissue": float("nan"), "species": None, "modalities": "scRNA-seq"})
    assert f["tissue"] == "" and f["organism"] == "" and f["assay"] == "scRNA-seq"


def test_a_gene_record_with_numeric_ids_is_stored_as_text():
    gm = hz.GeneMap([hz.GeneRecord(symbol="CD274", entrez=29126, aliases=("PD-L1", 12345))])
    assert gm.to(["29126", "PD-L1", "12345"], key="symbol") == ["CD274"] * 3
    assert gm.resolve("CD274").entrez == "29126"


def test_a_hole_still_marked_local_prints_as_a_missing_file(capsys):
    hole = {"piece": "p", "status": "local", "route_kind": "", "bytes": 0}
    from types import SimpleNamespace
    pz = orc.Puzzle(SimpleNamespace(path="/tmp/puzzle-x"), orc.Blueprint("b", ["p"]), [], [hole], None, {}, 0)
    pz.print_table()
    out = capsys.readouterr().out
    assert "file missing" in out and "here" not in out


def test_matrices_are_tried_before_tables_and_a_table_that_fails_falls_through_to_the_next(tmp_path):
    csv_path = tmp_path / "GSE00001_meta.csv"
    csv_path.write_text("gene,score\nAAA1,1\n")
    h5 = str(tmp_path / "GSE00001_x.h5ad")
    _h5ad(h5, 10, ["AAA1", "AAA2"])
    o = g.orchestrator()
    o.describe([{"id": "p", "accession": "GSE00001", "modalities": "scRNA-seq"}])
    o.pieces["p"].local = [str(csv_path), h5]                      # the table is listed first
    o.pieces["p"].status = "local"
    pz = o.build(o.blueprint("b"), str(tmp_path / "o1"))
    assert pz.built[0]["reader"] == "h5ad" and pz.built[0]["candidates"] == [h5, str(csv_path)]

    bad = _touch(tmp_path / "GSE00002_bad.parquet", b"not parquet")
    good = tmp_path / "GSE00002_good.csv"
    good.write_text("gene,score\nAAA1,1\nAAA2,2\n")
    o2 = g.orchestrator()
    o2.describe([{"id": "t", "accession": "GSE00002", "modalities": "pooled CRISPR screen"}])
    o2.pieces["t"].local = [bad, str(good)]
    o2.pieces["t"].status = "local"
    pz2 = o2.build(o2.blueprint("b"), str(tmp_path / "o2"))
    b = pz2.built[0]
    assert pz2.holes == [] and b["path"] == str(good) and b["candidates"] == [bad, str(good)]
    assert list(b["candidate_errors"]) == [bad]                      # why the first one was passed over
    assert pz2.harvest.read("T_t").num_rows == 2


def test_piece_ids_that_collide_once_made_safe_still_get_their_own_files(tmp_path):
    a, b = str(tmp_path / "GSE00001.h5ad"), str(tmp_path / "GSE00002.h5ad")
    _h5ad(a, 6, ["AAA1", "AAA2"], seed=1)
    _h5ad(b, 7, ["AAA1", "AAA2"], seed=2)
    o = g.orchestrator()
    o.add(orc.Piece(id="x/y", accessions=[orc.Accession("GSE00001", "geo_series")], local=[a], status="local"))
    o.add(orc.Piece(id="x y", accessions=[orc.Accession("GSE00002", "geo_series")], local=[b], status="local"))
    pz = o.build(o.blueprint("b"), str(tmp_path / "out"))
    assert pz.n_obs == 13 and [x["piece"] for x in pz.built] == ["x/y", "x y"]
    assert {x["roi"] for x in pz.built} == {"X_x_y", "X_x_y_2"}
    assert pz.read_X("x y").num_rows > 0 and all(pz.harvest.verify().values())
