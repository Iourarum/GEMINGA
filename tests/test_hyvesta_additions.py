"""The HyVESTA-driven additions: obsm streaming, the frozen catalog schema, kind/stat on handoff."""
import json
import numpy as np
import pytest
import geminga as g


def write_h5ad_like(path, n_obs=2000, n_vars=300, dim=64, seed=0, sparse_obsm=False):
    """An .h5ad in AnnData's on-disk layout, written with h5py alone: X as csr_matrix group,
    obs with _index + one categorical + one plain column, var/_index, obsm/X_geneformer."""
    import h5py
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    X = sp.random(n_obs, n_vars, density=0.05, format="csr", dtype=np.float32, random_state=seed)
    E = rng.standard_normal((n_obs, dim)).astype(np.float32)
    names = np.array([f"AAAC{i:06d}-1" for i in range(n_obs)]).astype("S")
    donors = np.array(["donor_1", "donor_2", "donor_3"]).astype("S")
    codes = rng.integers(0, 3, n_obs).astype(np.int8)
    with h5py.File(path, "w") as f:
        f.attrs["encoding-type"] = "anndata"; f.attrs["encoding-version"] = "0.1.0"
        x = f.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"; x.attrs["encoding-version"] = "0.1.0"
        x.attrs["shape"] = np.array(X.shape, dtype=np.int64)
        x.create_dataset("data", data=X.data); x.create_dataset("indices", data=X.indices)
        x.create_dataset("indptr", data=X.indptr)
        obs = f.create_group("obs"); obs.attrs["_index"] = "_index"
        obs.attrs["encoding-type"] = "dataframe"
        obs.create_dataset("_index", data=names)
        d = obs.create_group("donor"); d.attrs["encoding-type"] = "categorical"
        d.create_dataset("codes", data=codes); d.create_dataset("categories", data=donors)
        obs.create_dataset("n_counts", data=np.asarray(X.sum(1)).ravel().astype(np.float32))
        var = f.create_group("var"); var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=np.array([f"G{i}" for i in range(n_vars)]).astype("S"))
        obsm = f.create_group("obsm")
        if sparse_obsm:
            Es = sp.csr_matrix(np.where(rng.random(E.shape) < 0.3, E, 0).astype(np.float32))
            gph = obsm.create_group("X_geneformer")
            gph.attrs["encoding-type"] = "csr_matrix"
            gph.attrs["shape"] = np.array(Es.shape, dtype=np.int64)
            gph.create_dataset("data", data=Es.data); gph.create_dataset("indices", data=Es.indices)
            gph.create_dataset("indptr", data=Es.indptr)
            E = Es.toarray()
        else:
            obsm.create_dataset("X_geneformer", data=E)
        obsm.create_dataset("X_pca", data=E[:, :10].astype(np.float64))
    return E, names.astype(str), donors.astype(str)[codes]


@pytest.fixture(scope="module")
def h5(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("h5") / "atlas.h5ad")
    E, names, donors = write_h5ad_like(p)
    return p, E, names, donors


# ---------------------------------------------------------------- 1. obsm ----
def test_obsm_keys_from_metadata_only(h5):
    p, E, _, _ = h5
    keys = g.obsm_keys(p)
    assert keys["X_geneformer"] == (E.shape[0], E.shape[1], "float32", "array")
    assert keys["X_pca"][1] == 10


def test_obsm_reader_plans_streams_and_holds_the_budget(h5):
    p, E, names, donors = h5
    budget = g.Budget(ram="64KB")                       # tiny: forces many small blocks
    r = g.open(p, budget=budget, obsm="X_geneformer", obs_columns=["donor", "n_counts"])
    plan = r.plan()
    assert plan.kind == "h5ad-obsm/array" and plan.unit == "cells"
    assert plan.detail["dim"] == 64 and plan.n_chunks > 1
    assert plan.fits(budget)
    seen, got, peak = 0, [], 0
    for ch in r:
        assert set(ch) >= {"X", "obsm_key", "obs_start", "obs_end", "n_vars", "resident_bytes",
                           "obs_names", "donor", "n_counts"}
        assert ch["obsm_key"] == "X_geneformer" and ch["n_vars"] == 64
        n = ch["obs_end"] - ch["obs_start"]
        assert ch["X"].shape == (n, 64) and ch["resident_bytes"] == ch["X"].nbytes
        used = budget.stats()["ram"]["used"]
        assert used == ch["resident_bytes"]                 # exactly this block is held
        peak = max(peak, used)
        got.append(ch["X"])
        assert list(ch["obs_names"]) == list(names[ch["obs_start"]:ch["obs_end"]])
        assert list(ch["donor"]) == list(donors[ch["obs_start"]:ch["obs_end"]])
        seen += n
    r.close()
    assert seen == E.shape[0]
    np.testing.assert_array_equal(np.vstack(got), E)
    assert budget.stats()["ram"]["used"] == 0
    assert peak <= 64 << 10 and budget.stats()["ram"]["denials"] == 0


def test_obsm_reader_same_dict_shape_as_x(h5):
    """A consumer written against H5adReader chunks works on obsm chunks unchanged."""
    p, E, _, _ = h5
    xr = g.open(p, obs_columns=["n_counts"])
    er = g.open(p, obsm="X_pca", obs_columns=["n_counts"], obs_names=False)
    cx, ce = next(iter(xr)), next(iter(er))
    xr.close(); er.close()
    common = {"X", "obs_start", "obs_end", "n_vars", "resident_bytes", "n_counts"}
    assert common <= set(cx) and common <= set(ce)
    assert ce["n_vars"] == 10


def test_obsm_target_syntax_and_missing_key(h5):
    p, E, _, _ = h5
    r = g.open(f"{p}::obsm/X_pca")
    assert r.plan().detail["obsm_key"] == "X_pca"
    r.close()
    with pytest.raises(KeyError, match="available"):
        g.open(p, obsm="X_nope")


def test_obsm_sparse_encoding(tmp_path):
    p = str(tmp_path / "sparse.h5ad")
    E, _, _ = write_h5ad_like(p, n_obs=500, dim=32, sparse_obsm=True)
    r = g.open(p, obsm="X_geneformer", chunk_bytes="8KB")
    assert r.plan().kind == "h5ad-obsm/csr_matrix"
    got = np.vstack([ch["X"].toarray() for ch in r])
    r.close()
    np.testing.assert_array_equal(got, E)


def test_chunk_floor_never_exceeds_a_small_budget(h5):
    """The shared 1 MB floor used to plan a chunk larger than the budget itself."""
    p, _, _, _ = h5
    b = g.Budget(ram="200KB")
    for r in (g.open(p, budget=b), g.open(p, budget=b, obsm="X_geneformer")):
        assert r.chunk_bytes <= 200 << 10
        assert r.plan().fits(b)
        r.close()
    v = g.view(p, budget=g.Budget(ram="300KB"))
    assert v._chunk_cap() <= 300 << 10


def test_obsm_obs_columns_decode_missing_codes_and_demand_real_columns(tmp_path):
    """A categorical code of -1 is missing, not the last category; an absent column is an error."""
    import h5py
    p = str(tmp_path / "gaps.h5ad")
    E, names, donors = write_h5ad_like(p, n_obs=200)
    with h5py.File(p, "a") as f:
        f["obs/donor/codes"][[3, 150]] = -1
        d = f["obsm"].create_group("meta")                         # a DataFrame-valued entry
        d.attrs["encoding-type"] = "dataframe"
        d.attrs["_index"] = "_index"
        d.attrs.create("column-order", np.array(["u", "v"], dtype=object), dtype=h5py.string_dtype())
        d.create_dataset("_index", data=f["obs/_index"][:])
        d.create_dataset("u", data=np.zeros(200))
        d.create_dataset("v", data=np.ones(200))
    r = g.open(p, obsm="X_pca", obs_columns=["donor"], chunk_bytes="2KB")
    got = [v for ch in r for v in ch["donor"]]
    r.close()
    assert got[3] is None and got[150] is None
    assert [v for i, v in enumerate(got) if i not in (3, 150)] == [v for i, v in enumerate(donors)
                                                                    if i not in (3, 150)]
    with pytest.raises(KeyError, match="no_such"):
        g.open(p, obsm="X_pca", obs_columns=["donor", "no_such"])
    assert g.obsm_keys(p)["meta"] == (200, 2, "mixed", "dataframe")
    with pytest.raises(ValueError, match=r"meta.*u.*v.*X_geneformer|meta.*u.*v.*X_pca"):
        g.open(p, obsm="meta")


def test_obsm_chunk_rule_follows_the_value_not_the_spelling(h5):
    p, _, _, _ = h5
    b = g.Budget(ram="200KB")
    for spelling in ("256MB", "256 MB", 256 << 20):
        r = g.open(p, budget=b, obsm="X_geneformer", chunk_bytes=spelling)
        assert r.chunk_bytes == 200 << 10 and r.plan().fits(b)
        r.close()
    with pytest.raises(MemoryError, match="chunk_bytes"):
        g.open(p, budget=b, obsm="X_geneformer", chunk_bytes="64MB")


def test_obsm_sparse_blocks_follow_indptr(tmp_path):
    p = str(tmp_path / "skew.h5ad")
    E, _, _ = write_h5ad_like(p, n_obs=600, dim=64, sparse_obsm=True)
    r = g.open(p, obsm="X_geneformer", chunk_bytes="4KB")
    plan = r.plan()
    ip = r.indptr
    sizes = [(int(ip[c["obs_end"]]) - int(ip[c["obs_start"]])) * 8 for c in r]
    r.close()
    assert max(sizes) <= 4096 and plan.n_chunks == len(sizes) and plan.bytes_per_chunk == max(sizes)


# ---------------------------------------------------------------- 2. catalog ----
def test_catalog_records_carry_schema_and_facets(tmp_path):
    cat = g.catalog(str(tmp_path / "cat.jsonl"))
    cat.record("atlas", "/data/atlas.h5ad", url="https://example.org/ds/1",
               facets={"organism": "Homo sapiens", "tissue": "glioblastoma",
                       "platform": "10x 3' v3"},
               local_paths=["/scratch/atlas.h5ad"])
    cat.annotate("atlas", facets={"donor": "MGH-01"}, note="analyst confirmed")
    cat.note_local("atlas", "/scratch/regions/atlas-lobe-a.parquet", "harvest")
    e = cat.get("atlas")
    assert e.facets == {"organism": "Homo sapiens", "tissue": "glioblastoma",
                        "platform": "10x 3' v3", "donor": "MGH-01"}
    assert e.note == "analyst confirmed"
    assert len(e.local_paths) == 2 and e.local_paths[1].endswith("lobe-a.parquet")
    f = cat.facets_for("atlas")
    assert set(f) == set(g.FACETS) and f["tissue"] == "glioblastoma" and f["disease"] == ""
    rows = cat.facet_table()
    assert rows[0]["key"] == "atlas" and list(rows[0])[:2] == ["key", "target"]
    for ev in cat.events():
        assert ev["schema"] == g.CATALOG_SCHEMA
    assert cat.validate() == []
    assert cat.find("glioblastoma")[0].key == "atlas"       # facets are searchable
    assert cat.polars().height >= 3                          # the Polars graft still works


def test_catalog_rejects_unknown_facet_key(tmp_path):
    cat = g.catalog(str(tmp_path / "cat.jsonl"))
    with pytest.raises(KeyError, match="unknown facet"):
        cat.record("x", "/x", facets={"tisue": "brain"})


def test_catalog_reads_pre_schema_lines(tmp_path):
    p = tmp_path / "old.jsonl"
    p.write_text(json.dumps({"event": "record", "key": "old", "target": "/o.gef",
                             "kind": "Stereo-seq GEF", "assay": "Stereo-seq",
                             "ts": 1.0, "tags": [], "metadata": {}}) + "\n")
    cat = g.catalog(str(p))
    e = cat.get("old")
    assert e.kind == "Stereo-seq GEF" and e.facets == {} and e.local_paths == []
    assert cat.facets_for("old")["assay"] == ""
    assert cat.validate() == []
    cat.record("old", "/o.gef")                             # re-recording promotes it
    assert cat.facets_for("old")["assay"] == "Stereo-seq"


def test_catalog_schema_doc_matches_dataclass():
    assert set(g.CatalogEntry.__dataclass_fields__) == set(g.CatalogEntry.SCHEMA)


# ---------------------------------------------------------------- 3. handoff ----
def test_add_table_carries_kind_and_stat(tmp_path):
    h = g.harvest(str(tmp_path / "h"), note="t", producer="hyvesta")
    rng = np.random.default_rng(0)
    tbl = {"seed": np.arange(5), "dim": np.array([2, 2, 1, 2, 3]),
           "centroid": [rng.standard_normal(4).tolist() for _ in range(5)],
           "label": np.array(list("ABABC"), dtype=object)}
    r = h.add_table("patches", tbl, fmt="parquet", kind="patch", targets=["regulus"],
                    stat={"n": 5, "scale": 3.68}, source="GBM_atlas")
    assert r.kind == "patch" and r.stat["n"] == 5 and r.rows == 5 and r.sha256
    t = h.read("patches")
    assert t.column("centroid")[0].as_py() == tbl["centroid"][0]          # list column
    assert t.column("label").to_pylist() == list("ABABC")
    h.add_table("bundle", {"row_a": np.arange(3), "row_b": np.arange(3) + 10},
                fmt="ndjson", kind="bundle", targets=["regulus", "prism"],
                stat={"resultant": 0.52, "p": 0.002, "q": 0.004})
    h.add_table("noise", {"x": np.arange(2)}, kind="bundle", stat={"p": 0.6})
    man = json.load(open(h.manifest_path))
    assert man["schema"] == "geminga.harvest/2" and man["producer"] == "hyvesta"
    assert {e["kind"] for e in man["regions"]} == {"patch", "bundle"}
    assert [e.name for e in h.significant(0.05)] == ["bundle"]
    assert [e.name for e in h.for_kind("bundle")] == ["bundle", "noise"]
    d = json.load(open(h.write_dispatch()))
    assert d["schema"] == "geminga.dispatch/2" and d["producer"] == "hyvesta"
    assert d["targets"]["prism"][0]["kind"] == "bundle"
    assert d["targets"]["prism"][0]["stat"]["p"] == 0.002
    assert all(h.verify().values())


def test_v1_manifest_still_loads(tmp_path):
    root = tmp_path / "old"; root.mkdir()
    (root / "a.ndjson").write_text('{"x": 1}\n')
    man = {"schema": "geminga.harvest/1", "producer": "geminga", "created": 1.0, "note": "",
           "root": str(root), "regions": [{"name": "a", "source": "s", "window": [0, 1, 0, 1],
                                          "targets": ["prism"], "path": "a.ndjson",
                                          "format": "ndjson", "rows": 1, "bytes": 9,
                                          "sha256": "x", "written": 1.0}]}
    (root / "manifest.json").write_text(json.dumps(man))
    h = g.harvest(str(root))
    assert h.rois["a"].kind == "region" and h.rois["a"].stat == {}
    assert h.rois["a"].window == (0, 1, 0, 1)
