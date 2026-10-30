"""Regression tests for the reader fixes in CHANGES-E.md.

Every test that streams something does so under a `Budget` at least 4x smaller than the input,
and checks three things: the budget never went over (`peak <= capacity`, no denials), the
Python heap stayed under 3x the capacity (the reader's own transient copies are allowed,
materialising the file is not), and `plan()` predicted the largest chunk that was actually
handed out. Numbering follows the findings: 1 obs columns, 2 imaging, 3 sparse chunking,
4 10x layouts, 5 `to_h5`, 6 frames / sinks / Arrow IPC, 7 the chunk floor rule.
"""
from __future__ import annotations

import gc
import gzip
import os
import tracemalloc

import h5py
import numpy as np
import pytest
import scipy.sparse as sp

import geminga as g

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")
GLIOMA = os.path.join(FIX, "glioma")
CODEX = os.path.join(GLIOMA, "codex_6plex.ome.tif")
XENIUM = os.path.join(GLIOMA, "xenium", "transcripts.parquet")
MB = 1 << 20


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

class heap:
    """tracemalloc around a block; `.peak` is the Python-heap high-water mark inside it."""

    def __enter__(self):
        gc.collect()
        tracemalloc.start()
        return self

    def __exit__(self, *exc):
        _, self.peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return False


def budget(ram: str) -> g.Budget:
    return g.Budget(ram=ram, timeout_s=5)          # a lease that cannot be granted fails in 5 s


def governed(b: g.Budget, cap: int, h: heap, factor: int = 3) -> None:
    st = b.stats()["ram"]
    assert st["peak"] <= cap, st
    assert st["denials"] == 0, st
    assert st["used"] == 0, st                     # everything was handed back
    assert h.peak < cap * factor, f"heap peak {h.peak:,} >= {factor} x {cap:,}"


def _missing(v):
    return v is None or (isinstance(v, (float, np.floating)) and v != v)


def same(actual, expected):
    """Element-wise equality where None and NaN are the same 'missing'."""
    actual, expected = list(actual), list(expected)
    assert len(actual) == len(expected), (len(actual), len(expected))
    for i, (a, e) in enumerate(zip(actual, expected)):
        if _missing(e):
            assert _missing(a), (i, a, e)
        else:
            assert not _missing(a) and not isinstance(a, bytes), (i, a, e)
            assert a == e or (isinstance(e, float) and abs(a - e) < 1e-5), (i, a, e)


# ---------------------------------------------------------------------------
# AnnData-shaped files written with h5py alone (encoding 0.2.0 layout)
# ---------------------------------------------------------------------------

def _enc(node, kind, version="0.2.0"):
    node.attrs["encoding-type"] = kind
    node.attrs["encoding-version"] = version
    return node


def _vlen(grp, name, values):
    ds = grp.create_dataset(name, data=np.asarray(values, dtype=object), dtype=h5py.string_dtype())
    return _enc(ds, "string-array")


def _arr(grp, name, values):
    return _enc(grp.create_dataset(name, data=values), "array")


def _csr(grp, name, indptr, indices, data, shape):
    x = _enc(grp.create_group(name), "csr_matrix", "0.1.0")
    x.attrs["shape"] = np.array(shape, dtype=np.int64)
    x.create_dataset("data", data=data)
    x.create_dataset("indices", data=indices)
    x.create_dataset("indptr", data=indptr)


def _frame(grp, name, index_name="_index", n=0):
    f = _enc(grp.create_group(name), "dataframe")
    f.attrs["_index"] = index_name
    return f


def write_obs_zoo(path, n=48, n_vars=12, seed=0):
    """An .h5ad with one obs column of every encoding AnnData writes; returns the truth."""
    rng = np.random.default_rng(seed)
    X = sp.random(n, n_vars, density=0.3, format="csr", dtype=np.float32, random_state=seed)
    names = [f"cell{i:03d}" for i in range(n)]
    cats = np.array(["alpha", "beta", "gamma"])
    codes = rng.integers(0, 3, n).astype(np.int8)
    codes[[3, 17, 40]] = -1
    ncats = np.array([10, 20, 30], dtype=np.int64)
    ncodes = rng.integers(0, 3, n).astype(np.int8)
    ncodes[[0, 5]] = -1
    nint = rng.integers(-50, 50, n).astype(np.int64)
    nint_mask = np.zeros(n, bool)
    nint_mask[[2, 9, 33]] = True
    nbool = rng.random(n) < 0.5
    nbool_mask = np.zeros(n, bool)
    nbool_mask[[1, 8]] = True
    nstr = np.array([f"s{i}" for i in range(n)], dtype=object)
    nstr_mask = np.zeros(n, bool)
    nstr_mask[[4, 44]] = True
    flt = rng.random(n).astype(np.float32)
    flt[[6, 7]] = np.nan
    vstr = np.array([f"gene-ünï-{i}" for i in range(n)], dtype=object)
    fixed = np.array([f"fx{i:02d}" for i in range(n)]).astype("S6")

    truth = {
        "donor": [None if c < 0 else str(cats[c]) for c in codes],
        "batch": [None if c < 0 else int(ncats[c]) for c in ncodes],
        "age": [float("nan") if m else float(v) for v, m in zip(nint, nint_mask)],
        "flag": [None if m else bool(v) for v, m in zip(nbool, nbool_mask)],
        "label": [None if m else str(v) for v, m in zip(nstr, nstr_mask)],
        "score": [float(v) for v in flt],
        "barcode": list(vstr),
        "fixed": [v.decode() for v in fixed],
        "n": list(range(n)),
        "ok": [bool(i % 3) for i in range(n)],
    }
    with h5py.File(path, "w") as f:
        _enc(f, "anndata", "0.1.0")
        _csr(f, "X", X.indptr, X.indices, X.data, X.shape)
        obs = _frame(f, "obs")
        _vlen(obs, "_index", names)
        d = _enc(obs.create_group("donor"), "categorical")
        d.attrs["ordered"] = False
        _arr(d, "codes", codes)
        _vlen(d, "categories", cats)
        b = _enc(obs.create_group("batch"), "categorical")
        b.attrs["ordered"] = False
        _arr(b, "codes", ncodes)
        _arr(b, "categories", ncats)
        a = _enc(obs.create_group("age"), "nullable-integer", "0.1.0")
        _arr(a, "values", nint)
        _arr(a, "mask", nint_mask)
        fl = _enc(obs.create_group("flag"), "nullable-boolean", "0.1.0")
        _arr(fl, "values", nbool)
        _arr(fl, "mask", nbool_mask)
        ls = _enc(obs.create_group("label"), "nullable-string-array", "0.1.0")
        _vlen(ls, "values", nstr)
        _arr(ls, "mask", nstr_mask)
        _arr(obs, "score", flt)
        _vlen(obs, "barcode", vstr)
        _enc(obs.create_dataset("fixed", data=fixed), "string-array")
        _arr(obs, "n", np.arange(n, dtype=np.int64))
        _arr(obs, "ok", np.array(truth["ok"]))
        var = _frame(f, "var")
        _vlen(var, "_index", [f"G{i}" for i in range(n_vars)])
        obsm = _enc(f.create_group("obsm"), "dict", "0.1.0")
        _arr(obsm, "X_pca", rng.standard_normal((n, 5)).astype(np.float32))
        df = _frame(obsm, "meta")                     # a DataFrame-valued obsm entry
        _vlen(df, "_index", names)
        _arr(df, "umap1", rng.standard_normal(n))
        _arr(df, "umap2", rng.standard_normal(n))
        df.attrs.create("column-order", np.array(["umap1", "umap2"], dtype=object),
                        dtype=h5py.string_dtype())
        # .raw: the same cells, a different (longer) var axis
        raw = _enc(f.create_group("raw"), "raw", "0.1.0")
        R = sp.random(n, n_vars + 3, density=0.3, format="csr", dtype=np.float32, random_state=seed + 1)
        _csr(raw, "X", R.indptr, R.indices, R.data, R.shape)
        rvar = _frame(raw, "var")
        _vlen(rvar, "_index", [f"R{i}" for i in range(n_vars + 3)])
    return {"names": names, "truth": truth, "X": X, "R": R,
            "var": [f"G{i}" for i in range(n_vars)], "rvar": [f"R{i}" for i in range(n_vars + 3)]}


@pytest.fixture(scope="module")
def zoo(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("zoo") / "zoo.h5ad")
    return p, write_obs_zoo(p)


# ---------------------------------------------------------------------------
# synthetic CSR / 10x / mtx / tiff / ipc inputs
# ---------------------------------------------------------------------------

def _rows(rng, n_vars, k):
    return np.sort(rng.choice(n_vars, size=k, replace=False)).astype(np.int32)


def write_csr_h5ad(path, nnz_per_row, n_vars, seed=0, obs_columns=True, indptr_dtype=np.int64):
    """A CSR .h5ad whose rows have exactly `nnz_per_row[i]` entries."""
    rng = np.random.default_rng(seed)
    nnz_per_row = np.asarray(nnz_per_row)
    n = len(nnz_per_row)
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(nnz_per_row, out=indptr[1:])
    indices = np.concatenate([_rows(rng, n_vars, int(k)) for k in nnz_per_row]).astype(np.int32)
    data = rng.random(int(indptr[-1])).astype(np.float32)
    with h5py.File(path, "w") as f:
        _enc(f, "anndata", "0.1.0")
        _csr(f, "X", indptr.astype(indptr_dtype), indices, data, (n, n_vars))
        obs = _frame(f, "obs")
        _vlen(obs, "_index", [f"c{i}" for i in range(n)])
        if obs_columns:
            d = _enc(obs.create_group("grp"), "categorical")
            d.attrs["ordered"] = False
            codes = (np.arange(n) % 4).astype(np.int8)
            codes[::97] = -1
            _arr(d, "codes", codes)
            _vlen(d, "categories", ["a", "b", "c", "d"])
        var = _frame(f, "var")
        _vlen(var, "_index", [f"G{i}" for i in range(n_vars)])
    return indptr, indices, data


@pytest.fixture(scope="module")
def skewed(tmp_path_factory):
    """20,000 cells; the first 10% are 15x denser than the rest: 4.8M nnz, 38 MB."""
    p = str(tmp_path_factory.mktemp("skew") / "skewed.h5ad")
    k = np.full(20_000, 100)
    k[:2_000] = 1_500
    indptr, indices, data = write_csr_h5ad(p, k, 3_000)
    return p, indptr


def write_10x_h5(path, n_cells, n_rna=2000, n_adt=50, n_guide=24, seed=0, empty=(), v2=False):
    """Cell Ranger HDF5 (CSC, features x barcodes): RNA sparse, ADT dense, one guide a cell."""
    rng = np.random.default_rng(seed)
    n_feat = n_rna + n_adt + n_guide
    empty = set(int(e) % n_cells for e in empty)
    ind, dat, ptr = [], [], [0]
    for c in range(n_cells):
        if c in empty:
            ptr.append(ptr[-1])
            continue
        r = _rows(rng, n_rna, int(rng.integers(min(60, n_rna), min(140, n_rna) + 1)))
        a = np.arange(n_rna, n_rna + n_adt, dtype=np.int32)
        gd = (np.array([n_rna + n_adt + int(rng.integers(0, n_guide))], dtype=np.int32)
              if n_guide else np.zeros(0, np.int32))
        idx = np.concatenate([r, a, gd])
        ind.append(idx)
        dat.append(rng.integers(1, 60, len(idx)).astype(np.int32))
        ptr.append(ptr[-1] + len(idx))
    indices = np.concatenate(ind) if ind else np.zeros(0, np.int32)
    data = np.concatenate(dat) if dat else np.zeros(0, np.int32)
    indptr = np.asarray(ptr, np.int64)
    ref = sp.csc_matrix((data, indices, indptr), shape=(n_feat, n_cells))
    barcodes = np.array([f"BC{i:06d}-1" for i in range(n_cells)]).astype("S")
    ids = np.array([f"ENSG{i:07d}" for i in range(n_feat)]).astype("S")
    names = np.array([f"G{i}" for i in range(n_feat)]).astype("S")
    ftype = np.array([b"Gene Expression"] * n_rna + [b"Antibody Capture"] * n_adt
                     + [b"CRISPR Guide Capture"] * n_guide)
    with h5py.File(path, "w") as f:
        if v2:
            m = f.create_group("GRCh38")
            m.create_dataset("data", data=data)
            m.create_dataset("indices", data=indices.astype(np.int64))
            m.create_dataset("indptr", data=indptr)
            m.create_dataset("shape", data=np.array([n_feat, n_cells], np.int32))
            m.create_dataset("barcodes", data=barcodes)
            m.create_dataset("genes", data=ids)
            m.create_dataset("gene_names", data=names)
        else:
            m = f.create_group("matrix")
            m.create_dataset("data", data=data)
            m.create_dataset("indices", data=indices)
            m.create_dataset("indptr", data=indptr)
            m.create_dataset("shape", data=np.array([n_feat, n_cells], np.int32))
            m.create_dataset("barcodes", data=barcodes)
            ft = m.create_group("features")
            ft.create_dataset("id", data=ids)
            ft.create_dataset("name", data=names)
            ft.create_dataset("feature_type", data=ftype)
            ft.create_dataset("genome", data=np.array([b"GRCh38"] * n_feat))
    return ref, {"n_rna": n_rna, "n_adt": n_adt, "n_guide": n_guide}


@pytest.fixture(scope="module")
def tenx_big(tmp_path_factory):
    """25,000 cells, 2,074 features, ~3.5M nnz (28 MB of data+indices)."""
    p = str(tmp_path_factory.mktemp("tenx") / "big.h5")
    ref, info = write_10x_h5(p, 25_000)
    return p, ref, info


def write_mtx_dir(path, X, prefix="", sep="_", kind="integer", gz=True, header_dims=None,
                  features_name="features", n_barcodes=None, n_features=None):
    """A MatrixMarket triple for scipy matrix X (features x cells), entries in X's own order."""
    os.makedirs(path, exist_ok=True)
    X = X.tocoo()
    op = (lambda p: gzip.open(p, "wt", compresslevel=1)) if gz else (lambda p: open(p, "wt"))
    ext = ".gz" if gz else ""
    stem = f"{prefix}{sep}" if prefix else ""
    r, c = (header_dims or X.shape)
    fmt = (lambda v: str(int(v))) if kind == "integer" else (lambda v: repr(float(v)))
    with op(os.path.join(path, f"{stem}matrix.mtx{ext}")) as fh:
        fh.write(f"%%MatrixMarket matrix coordinate {kind} general\n%\n{r} {c} {X.nnz}\n")
        fh.write("\n".join(f"{a + 1} {b + 1} {fmt(v)}"
                           for a, b, v in zip(X.row.tolist(), X.col.tolist(), X.data.tolist())))
        fh.write("\n")
    with op(os.path.join(path, f"{stem}barcodes.tsv{ext}")) as fh:
        fh.writelines(f"BC{i}-1\n" for i in range(X.shape[1] if n_barcodes is None else n_barcodes))
    with op(os.path.join(path, f"{stem}{features_name}.tsv{ext}")) as fh:
        fh.writelines(f"ENSG{i:06d}\tG{i}\tGene Expression\n"
                      for i in range(X.shape[0] if n_features is None else n_features))
    return path


# ===========================================================================
# 7. the chunk floor rule is about the value, and an explicit chunk fails fast
# ===========================================================================

def _six_readers(tmp_path_factory_dir):
    """(name, default sentinel bytes, factory(chunk_bytes, budget)) for the six governed readers."""
    d = str(tmp_path_factory_dir)
    h5ad = os.path.join(d, "f.h5ad")
    write_csr_h5ad(h5ad, np.full(4000, 200), 2000, obs_columns=False)     # 6.4 MB
    with h5py.File(h5ad, "a") as f:
        _arr(f.create_group("obsm"), "X_e", np.zeros((4000, 16), np.float32))
    tenx = os.path.join(d, "f.h5")
    write_10x_h5(tenx, 2000, n_rna=500, n_adt=20, n_guide=8)
    mtx = write_mtx_dir(os.path.join(d, "mtx"), sp.random(300, 200, density=0.3, format="coo",
                                                          random_state=1) * 5)
    import pyarrow as pa
    arrow = os.path.join(d, "f.arrow")
    with pa.OSFile(arrow, "wb") as sink, pa.ipc.new_file(sink, pa.schema([("a", pa.int64())])) as w:
        for i in range(4):
            w.write_batch(pa.record_batch([pa.array(np.arange(i * 200_000, (i + 1) * 200_000))],
                                          names=["a"]))
    zpath = os.path.join(d, "z.zarr")
    import zarr
    z = zarr.open(zpath, mode="w", shape=(1024, 1024), chunks=(256, 256), dtype="uint16")
    z[:] = 7
    return [
        ("h5ad", 256 * MB, lambda cb, b: g.H5adReader(h5ad, cb, b)),
        ("obsm", 256 * MB, lambda cb, b: g.ObsmReader(h5ad, "X_e", cb, b)),
        ("tenx", 256 * MB, lambda cb, b: g.TenXMatrix(tenx, cb, b)),
        ("mtx", 64 * MB, lambda cb, b: g.MtxTriple(mtx, cb, b)),
        ("arrow", 256 * MB, lambda cb, b: g.ArrowIPCReader(arrow, cb, b)),
        ("image", 512 * MB, lambda cb, b: g.TiledImageReader.from_zarr(zpath, chunk_bytes=cb, budget=b)),
    ]


@pytest.fixture(scope="module")
def six(tmp_path_factory):
    return _six_readers(tmp_path_factory.mktemp("six"))


@pytest.mark.parametrize("which", ["h5ad", "obsm", "tenx", "mtx", "arrow", "image"])
@pytest.mark.parametrize("spelling", ["int", "spaced", "compact"])
def test_default_chunk_gets_the_floor_rule_whatever_its_spelling(six, which, spelling):
    name, default, make = next(s for s in six if s[0] == which)
    mb = default // MB
    chunk = {"int": default, "spaced": f"{mb} MB", "compact": f"{mb}MB"}[spelling]
    for ram in ("8MB", "600KB"):
        b = budget(ram)
        avail = b.available("ram")
        r = make(chunk, b)                      # used to raise MemoryError once iterated
        assert r.chunk_bytes == max(min(1 << 20, avail), avail // 4), (name, ram)
        assert r.chunk_bytes <= avail
        r.close()


@pytest.mark.parametrize("which", ["h5ad", "obsm", "tenx", "mtx", "arrow", "image"])
def test_explicit_chunk_above_available_fails_before_reading(six, which):
    name, default, make = next(s for s in six if s[0] == which)
    b = budget("4MB")
    with pytest.raises(MemoryError) as e:
        make("100MB", b)
    msg = str(e.value)
    assert "chunk_bytes" in msg and "100.0 MB" in msg and "4.0 MB" in msg, msg
    assert b.stats()["ram"]["grants"] == 0           # nothing was ever leased
    ok = make("1MB", b)                              # an explicit chunk that fits is honoured
    assert ok.chunk_bytes == MB
    ok.close()


def test_explicit_chunk_without_budget_is_untouched(six):
    for name, default, make in six:
        r = make("3MB", None)
        assert r.chunk_bytes == 3 * MB, name
        r.close()


# ===========================================================================
# 1. obs columns: one reader for every encoding, shared by H5adReader and ObsmReader
# ===========================================================================

def test_read_obs_column_handles_every_encoding(zoo):
    from geminga.formats import read_obs_column
    p, z = zoo
    with h5py.File(p, "r") as f:
        for name, truth in z["truth"].items():
            same(read_obs_column(f, name, 0, 48), truth)
            same(read_obs_column(f, name, 5, 31), truth[5:31])     # a slice, not the whole column
            same(read_obs_column(f, name, 44, 48), truth[44:48])
        with pytest.raises(KeyError, match="nope"):
            read_obs_column(f, "nope", 0, 5)


def test_categorical_missing_codes_are_missing_not_the_last_category(zoo):
    from geminga.formats import read_obs_column
    p, z = zoo
    with h5py.File(p, "r") as f:
        donor = read_obs_column(f, "donor", 0, 48)
        assert donor.dtype == object
        for i in (3, 17, 40):                           # code -1
            assert donor[i] is None, (i, donor[i])
        assert {"alpha", "beta", "gamma"} >= {v for v in donor if v is not None}
        batch = read_obs_column(f, "batch", 0, 48)
        assert batch[0] is None and batch[5] is None


def test_nullable_columns_have_stable_dtypes(zoo):
    from geminga.formats import read_obs_column
    p, _ = zoo
    with h5py.File(p, "r") as f:
        age = read_obs_column(f, "age", 0, 48)
        assert age.dtype == np.float64 and np.isnan(age[2]) and np.isnan(age[9])
        assert read_obs_column(f, "age", 10, 20).dtype == np.float64    # no NA in this slice
        flag = read_obs_column(f, "flag", 0, 48)
        assert flag.dtype == object and flag[1] is None and flag[8] is None
        assert all(isinstance(v, bool) for v in flag if v is not None)


def test_string_columns_are_str_not_bytes(zoo):
    from geminga.formats import read_obs_column
    p, z = zoo
    with h5py.File(p, "r") as f:
        for name in ("barcode", "fixed", "label", "donor"):
            col = read_obs_column(f, name, 0, 48)
            assert all(isinstance(v, str) for v in col if v is not None), name
        assert read_obs_column(f, "barcode", 0, 1)[0] == "gene-ünï-0"


def test_h5ad_reader_obs_columns_every_encoding(zoo):
    p, z = zoo
    cols = list(z["truth"])
    r = g.H5adReader(p, chunk_bytes=300, obs_columns=cols)          # many tiny chunks
    got = {c: [] for c in cols}
    n_chunks = 0
    for ch in r:
        n_chunks += 1
        for c in cols:
            assert len(ch[c]) == ch["obs_end"] - ch["obs_start"]
            got[c].extend(ch[c])
    r.close()
    assert n_chunks > 3
    for c in cols:
        same(got[c], z["truth"][c])


def test_obsm_reader_obs_columns_every_encoding(zoo):
    p, z = zoo
    cols = list(z["truth"])
    r = g.ObsmReader(p, "X_pca", chunk_bytes=200, obs_columns=cols)
    got = {c: [] for c in cols}
    names = []
    for ch in r:
        names.extend(ch["obs_names"])
        for c in cols:
            got[c].extend(ch[c])
    r.close()
    assert names == z["names"]
    for c in cols:
        same(got[c], z["truth"][c])


@pytest.mark.parametrize("reader", ["h5ad", "obsm"])
def test_missing_obs_column_is_an_error_not_a_skip(zoo, reader):
    p, _ = zoo
    with pytest.raises(KeyError, match="no_such_col"):
        if reader == "h5ad":
            g.H5adReader(p, obs_columns=["donor", "no_such_col"])
        else:
            g.ObsmReader(p, "X_pca", obs_columns=["donor", "no_such_col"])


def test_h5ad_obs_names_and_raw_layer(zoo):
    p, z = zoo
    r = g.H5adReader(p)
    assert list(r.obs_names) == z["names"]
    assert list(r.var_names) == z["var"] and r.n_vars == 12
    r.close()
    raw = g.H5adReader(p, layer="raw")
    assert raw.n_vars == 15 and raw.n_obs == 48
    assert list(raw.var_names) == z["rvar"]
    X = sp.vstack([ch["X"] for ch in raw]).toarray()
    np.testing.assert_array_equal(X, z["R"].toarray())
    assert raw.plan().kind == "h5ad/csr_matrix"
    raw.close()


def test_obsm_dataframe_valued_entry(zoo):
    p, _ = zoo
    keys = g.obsm_keys(p)
    assert keys["X_pca"][3] == "array"
    assert "meta" in keys and keys["meta"][3] == "dataframe"
    assert keys["meta"][0] == 48 and keys["meta"][1] == 2
    with pytest.raises(ValueError) as e:
        g.ObsmReader(p, "meta")
    msg = str(e.value)
    assert "meta" in msg and "umap1" in msg and "umap2" in msg and "X_pca" in msg, msg


def test_h5ad_obs_columns_agree_with_anndata(tmp_path):
    ad = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    n = 40
    rng = np.random.default_rng(3)
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(n)])
    obs["cat"] = pd.Categorical(rng.choice(["a", "b", "c", None], n))
    obs["nint"] = pd.array([None if i % 7 == 0 else i for i in range(n)], dtype="Int64")
    obs["nbool"] = pd.array([None if i % 5 == 0 else bool(i % 2) for i in range(n)], dtype="boolean")
    obs["flt"] = np.where(np.arange(n) % 6 == 0, np.nan, rng.random(n)).astype(np.float32)
    obs["str"] = [f"s{i}" for i in range(n)]
    obs["int"] = np.arange(n)
    a = ad.AnnData(X=sp.random(n, 8, density=0.4, format="csr", dtype=np.float32, random_state=2), obs=obs)
    a.raw = a.copy()
    a.obsm["frame"] = pd.DataFrame({"u": rng.random(n), "v": rng.random(n)}, index=a.obs_names)
    path = str(tmp_path / "real.h5ad")
    a.write_h5ad(path)
    cols = list(obs.columns)
    r = g.H5adReader(path, chunk_bytes=100, obs_columns=cols)
    got = {c: [] for c in cols}
    for ch in r:
        for c in cols:
            got[c].extend(ch[c])
    assert list(r.obs_names) == list(a.obs_names)
    r.close()
    for c in cols:
        want = obs[c].astype(object).where(obs[c].notna(), None).tolist()
        same(got[c], want)
    assert g.obsm_keys(path)["frame"][3] == "dataframe"
    assert list(g.H5adReader(path, layer="raw").var_names) == list(a.raw.var_names)


def test_obs_columns_stream_under_a_budget(tmp_path):
    """Categorical + string obs columns ride along with X inside the budget."""
    p = str(tmp_path / "wide.h5ad")
    write_csr_h5ad(p, np.full(30_000, 120), 1_000)                  # 3.6M nnz, 29 MB
    cap = 6 * MB
    b = budget("6MB")
    r = g.H5adReader(p, budget=b, obs_columns=["grp"])
    seen, none = 0, 0
    with heap() as h:
        for ch in r:
            seen += ch["obs_end"] - ch["obs_start"]
            none += sum(v is None for v in ch["grp"])
        r.close()
    governed(b, cap, h)
    assert seen == 30_000 and none == len(range(0, 30_000, 97))


# ===========================================================================
# 2. imaging: channels, planar samples, slice steps, tile size, JPEG tables
# ===========================================================================

@pytest.fixture(scope="module")
def tiffs(tmp_path_factory):
    tifffile = pytest.importorskip("tifffile")
    d = tmp_path_factory.mktemp("tifs")
    rng = np.random.default_rng(0)
    H, W = 300, 260
    gray = rng.integers(0, 60000, (H, W), dtype=np.uint16)
    rgb = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    cyx = rng.integers(0, 60000, (5, H, W), dtype=np.uint16)
    zcyx = rng.integers(0, 60000, (3, 2, H, W), dtype=np.uint16)
    out = {}

    def w(name, arr, **kw):
        path = str(d / name)
        tifffile.imwrite(path, arr, **kw)
        out[name.split(".")[0]] = path
    w("gray.tif", gray, tile=(64, 64))
    w("deflate.tif", gray, tile=(64, 64), compression="deflate")
    w("rgb.tif", rgb, tile=(64, 64), photometric="rgb")
    w("planar.tif", np.moveaxis(rgb, -1, 0), tile=(64, 64), photometric="rgb", planarconfig="separate")
    w("planar_strips.tif", np.moveaxis(rgb, -1, 0), photometric="rgb", planarconfig="separate",
      rowsperstrip=16)
    w("cyx.ome.tif", cyx, tile=(64, 64), photometric="minisblack", metadata={"axes": "CYX"}, ome=True)
    w("cyx_strips.tif", cyx, photometric="minisblack", rowsperstrip=32)
    w("zcyx.ome.tif", zcyx, tile=(64, 64), photometric="minisblack", metadata={"axes": "ZCYX"},
      ome=True)
    return out


def _keys(nd):
    full = [slice(None)] * nd

    def with_(**at):
        k = list(full)
        for pos, v in at.items():
            k[int(pos[1:])] = v
        return tuple(k)
    last, prev = nd - 1, nd - 2
    ks = [
        tuple(full),
        with_(**{f"a{prev}": slice(0, 100, 4), f"a{last}": slice(0, 100, 4)}),
        with_(**{f"a{prev}": slice(250, 10, -5), f"a{last}": slice(None, None, -1)}),
        with_(**{f"a{prev}": slice(60, 200), f"a{last}": slice(30, 190, 7)}),
        with_(**{f"a{prev}": 3, f"a{last}": slice(10, 20)}),
        with_(**{f"a{last}": -1}),
        with_(**{f"a{prev}": slice(290, 500), f"a{last}": slice(0, 10)}),
        (Ellipsis, slice(2, 9, 2)),
        with_(**{f"a{prev}": slice(63, 130), f"a{last}": slice(0, 3)}),
    ]
    if nd >= 3:
        ks.append(with_(a0=1, **{f"a{last}": slice(5, 70, 3)}))
        ks.append(with_(a0=slice(None, None, -2)))
        ks.append(with_(a0=slice(1, 3)))
    if nd == 4:
        ks.append((1, 1, slice(0, 70), slice(0, 70)))
        ks.append((slice(None), 0, slice(60, 70), slice(0, 5)))
    return ks


@pytest.mark.parametrize("name,axes,shape", [
    ("gray", "YX", (300, 260)),
    ("deflate", "YX", (300, 260)),
    ("rgb", "YXS", (300, 260, 3)),
    ("planar", "SYX", (3, 300, 260)),
    ("planar_strips", "SYX", (3, 300, 260)),
    ("cyx", "CYX", (5, 300, 260)),
    ("cyx_strips", "QYX", (5, 300, 260)),
    ("zcyx", "ZCYX", (3, 2, 300, 260)),
])
def test_tiff_array_is_numpy_for_any_layout_and_selector(tiffs, name, axes, shape):
    import tifffile
    ref = tifffile.imread(tiffs[name])
    a = g.TiffTileArray(tiffs[name])
    assert a.shape == shape == ref.shape and a.axes == axes
    assert a.dtype == ref.dtype and a.ndim == len(shape)
    for k in _keys(len(shape)):
        got, want = a[k], ref[k]
        assert got.shape == want.shape, (k, got.shape, want.shape)
        assert got.dtype == want.dtype
        assert np.array_equal(got, want), k
    a.close()


def test_tiff_array_decodes_only_the_planes_and_tiles_it_needs(tiffs):
    a = g.TiffTileArray(tiffs["cyx"])
    calls = []
    real = a.page.decode

    def spy(data, index, **kw):
        calls.append(index)
        return real(data, index, **kw)
    a.page.decode = spy
    a[2, 0:10, 0:10]
    assert len(calls) == 1                         # one channel, one 64 px tile
    calls.clear()
    a[:, 0:10, 0:10]
    assert len(calls) == 5                         # every channel, still one tile each
    calls.clear()
    a[1, 60:70, 60:70]
    assert len(calls) == 4                         # straddles a tile corner
    calls.clear()
    a[0, 0:300:100, 0:260:100]                     # a step: only the tiles that hold sampled pixels
    assert len(calls) == 3 * 3
    a.close()


def test_tiff_array_forwards_jpegtables_to_decode(tiffs):
    for name in ("gray", "cyx"):
        a = g.TiffTileArray(tiffs[name])
        seen = []
        real = a.page.decode

        def spy(data, index, **kw):
            seen.append(kw)
            return real(data, index, **kw)
        a.page.decode = spy
        a.page.jpegtables = b"TABLES"
        a[(0,) * (a.ndim - 2) + (slice(0, 70), slice(0, 70))]
        assert seen and all(kw.get("jpegtables") == b"TABLES" for kw in seen), name
        a.close()


def _split_jpeg(stream: bytes):
    """(tables-only stream, abbreviated stream): what a TIFF keeps in JPEGTables and in each tile."""
    pos, tables, rest = 2, [], [b"\xff\xd8"]
    while True:
        marker = stream[pos + 1]
        if marker == 0xDA:                                    # SOS: the entropy data runs to the end
            rest.append(stream[pos:])
            break
        seg = stream[pos:pos + 2 + int.from_bytes(stream[pos + 2:pos + 4], "big")]
        (tables if marker in (0xDB, 0xC4) else rest).append(seg)
        pos += len(seg)
    return b"\xff\xd8" + b"".join(tables) + b"\xff\xd9", b"".join(rest)


def test_jpeg_tiles_with_a_jpegtables_tag_decode(tmp_path):
    """Abbreviated JPEG tiles are undecodable without the page's JPEGTables."""
    imagecodecs = pytest.importorskip("imagecodecs")
    import tifffile
    yy, xx = np.mgrid[0:300, 0:260]
    img = ((np.sin(yy / 17.0) + np.cos(xx / 23.0)) * 60 + 128).clip(0, 255).astype(np.uint8)
    tiles, tables = [], None
    for ty in range(0, 300, 64):
        for tx in range(0, 260, 64):
            t = np.zeros((64, 64), np.uint8)
            blk = img[ty:ty + 64, tx:tx + 64]
            t[:blk.shape[0], :blk.shape[1]] = blk
            tb, ab = _split_jpeg(imagecodecs.jpeg_encode(t, level=92))
            tables = tables or tb
            tiles.append(ab)
    path = str(tmp_path / "abbrev.tif")
    tifffile.imwrite(path, iter(tiles), shape=img.shape, dtype=np.uint8, tile=(64, 64),
                     photometric="minisblack", compression="jpeg", jpegtables=tables)
    ref = tifffile.imread(path)
    a = g.TiffTileArray(path)
    assert a.page.jpegtables == tables
    assert np.array_equal(a[10:200:2, 5:250:3], ref[10:200:2, 5:250:3])
    r = g.TiledImageReader(a, chunk_bytes=20_000)
    out = np.zeros_like(ref)
    for tile, (y0, y1, x0, x1) in r:
        out[y0:y1, x0:x1] = tile
    assert np.array_equal(out, ref)
    a.close()


@pytest.mark.skipif(not os.path.exists(CODEX), reason="run examples/make_glioma_fixtures.py")
def test_codex_series_keeps_its_channel_axis():
    import tifffile
    a = g.TiffTileArray(CODEX)
    assert a.shape == (6, 1500, 1500) and a.axes == "CYX"
    ref = tifffile.imread(CODEX)
    assert np.array_equal(a[2, 100:200, 300:420], ref[2, 100:200, 300:420])
    assert np.array_equal(a[:, 1400:1500, 1400:1500], ref[:, 1400:1500, 1400:1500])
    a.close()


@pytest.mark.skipif(not os.path.exists(CODEX), reason="run examples/make_glioma_fixtures.py")
def test_tiled_reader_counts_every_channel_of_a_tile():
    cap = 4 * MB
    b = budget("4MB")                                     # the file is 27 MB
    r = g.open(CODEX, budget=b)
    p = r.plan()
    assert r.shape == (6, 1500, 1500)
    assert p.total_bytes == 6 * 1500 * 1500 * 2
    assert r.chunk_bytes == MB and p.bytes_per_chunk <= MB
    seen = np.zeros((1500, 1500), bool)
    n, biggest = 0, 0
    with heap() as h:
        for tile, (y0, y1, x0, x1) in r:
            assert tile.shape == (6, y1 - y0, x1 - x0)
            assert b.stats()["ram"]["used"] == tile.nbytes <= MB
            biggest = max(biggest, tile.nbytes)
            seen[y0:y1, x0:x1] = True
            n += 1
        r.close()
    governed(b, cap, h)
    assert seen.all() and n == p.n_chunks == p.detail["windows"]
    assert biggest == p.bytes_per_chunk                   # the plan named the largest tile


def test_tiled_reader_reassembles_rgb_cyx_and_planar(tiffs):
    import tifffile
    for name in ("rgb", "planar", "cyx", "zcyx", "gray"):
        ref = tifffile.imread(tiffs[name])
        r = g.open(tiffs[name], chunk_bytes=40_000)
        axes = r.a.axes
        yi, xi = axes.index("Y"), axes.index("X")
        out = np.zeros_like(ref)
        for tile, (y0, y1, x0, x1) in r:
            assert tile.nbytes <= 40_000
            key = [slice(None)] * ref.ndim
            key[yi], key[xi] = slice(y0, y1), slice(x0, x1)
            out[tuple(key)] = tile
        r.close()
        assert np.array_equal(out, ref), name
        p = r.plan()
        assert p.total_bytes == ref.nbytes and p.bytes_per_chunk <= 40_000


def test_tiled_reader_tiles_a_cyx_zarr_over_its_last_two_axes(tmp_path):
    zarr = pytest.importorskip("zarr")
    path = str(tmp_path / "cyx.zarr")
    data = np.random.default_rng(1).integers(0, 65535, (8, 1024, 1024), dtype=np.uint16)
    z = zarr.open(path, mode="w", shape=data.shape, chunks=(1, 256, 256), dtype="uint16")
    z[:] = data
    cap = 4 * MB
    b = budget("4MB")                                     # 16 MB of pixels
    r = g.TiledImageReader.from_zarr(path, chunk_bytes="1MB", budget=b)
    p = r.plan()
    assert p.bytes_per_chunk <= MB
    n, biggest, ys, xs = 0, 0, set(), set()
    with heap() as h:
        for tile, (y0, y1, x0, x1) in r:
            assert tile.shape == (8, y1 - y0, x1 - x0) and tile.nbytes <= MB
            if n % 5 == 0:
                assert np.array_equal(tile, data[:, y0:y1, x0:x1])
            biggest = max(biggest, tile.nbytes)
            ys.add(y1)
            xs.add(x1)
            n += 1
        r.close()
    governed(b, cap, h)
    assert max(ys) == 1024 and max(xs) == 1024
    assert n == p.n_chunks and biggest == p.bytes_per_chunk


class _Stub:
    """Shape, dtype and slicing only — an image too big to allocate."""

    def __init__(self, shape, dtype="uint16", axes=None):
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        if axes:
            self.axes = axes

    def __getitem__(self, key):
        shape = [len(range(*k.indices(n))) for k, n in zip(key, self.shape)]
        return np.zeros(shape, self.dtype)


def test_tiled_reader_axes_come_from_the_array_or_a_rule():
    r = g.TiledImageReader(_Stub((4096, 4096, 3), "uint8"), chunk_bytes=300_000)
    tile, (y0, y1, x0, x1) = next(iter(r))
    assert tile.shape == (y1 - y0, x1 - x0, 3) and tile.nbytes <= 300_000
    r = g.TiledImageReader(_Stub((6, 2048, 2048)), chunk_bytes=300_000)
    tile, (y0, y1, x0, x1) = next(iter(r))
    assert tile.shape == (6, y1 - y0, x1 - x0) and tile.nbytes <= 300_000
    r = g.TiledImageReader(_Stub((2048, 2048, 7)), chunk_bytes=300_000, axes="YXC")
    tile, (y0, y1, x0, x1) = next(iter(r))
    assert tile.shape == (y1 - y0, x1 - x0, 7)
    r = g.TiledImageReader(_Stub((3, 2, 512, 512), axes="TCYX"), chunk_bytes=300_000)
    tile, _ = next(iter(r))
    assert tile.shape[:2] == (3, 2) and tile.nbytes <= 300_000


def test_a_tile_that_cannot_fit_is_a_budget_failure_and_a_bad_argument():
    a = _Stub((64, 4096, 4096))
    for exc in (MemoryError, ValueError):
        with pytest.raises(exc, match="chunk_bytes"):
            g.TiledImageReader(a, chunk_bytes=8 * 1024)


def test_tiled_reader_drops_the_64px_floor_or_says_why_not():
    a = _Stub((64, 4096, 4096))                              # 128 bytes a pixel
    r = g.TiledImageReader(a, chunk_bytes=64 * 1024)
    assert 16 <= r.tile < 64                                 # the floor would have been 512 KB
    assert r.plan().bytes_per_chunk <= 64 * 1024
    with pytest.raises(MemoryError, match=r"16 px"):
        g.TiledImageReader(a, chunk_bytes=8 * 1024)
    with pytest.raises(MemoryError, match=r"64 channels|channels"):
        g.TiledImageReader(a, chunk_bytes=8 * 1024)
    tiny = g.TiledImageReader(_Stub((10, 12)), chunk_bytes=1024)     # an image smaller than 16 px
    assert tiny.plan().n_chunks == 1


# ===========================================================================
# 3. sparse chunking follows indptr, and plan() names the real largest chunk
# ===========================================================================

def test_h5ad_skewed_rows_are_cut_by_indptr(skewed):
    path, indptr = skewed
    cap = 8 * MB
    b = budget("8MB")                                        # the file is 38 MB
    r = g.H5adReader(path, budget=b)
    assert r.chunk_bytes == 2 * MB
    p = r.plan()
    sizes, spans = [], []
    with heap() as h:
        for ch in r:
            s, e = ch["obs_start"], ch["obs_end"]
            true = (int(indptr[e]) - int(indptr[s])) * 8
            X = ch["X"]
            assert ch["resident_bytes"] == true == X.data.nbytes + X.indices.nbytes
            assert b.stats()["ram"]["used"] == true
            sizes.append(true)
            spans.append((s, e))
        r.close()
    governed(b, cap, h)
    assert max(sizes) <= r.chunk_bytes
    assert spans[0][0] == 0 and spans[-1][1] == 20_000
    assert all(a[1] == c[0] for a, c in zip(spans, spans[1:]))
    assert spans[0][1] - spans[0][0] < 300                   # the dense head is cut into small blocks
    assert p.n_chunks == len(sizes)
    assert p.bytes_per_chunk == max(sizes) == p.detail["largest_chunk_bytes"]
    assert p.detail["fits"] is True and p.fits(b)
    assert p.total_bytes == sum(sizes)


def test_h5ad_plan_does_not_claim_a_chunk_fits_when_it_does_not(tmp_path):
    k = np.full(6_000, 50)
    k[2_500] = 20_000                                        # one row of 160,000 bytes
    path = str(tmp_path / "bigrow.h5ad")
    write_csr_h5ad(path, k, 30_000, obs_columns=False)
    tiny = budget("100KB")
    r = g.H5adReader(path, budget=tiny)
    p = r.plan()
    assert p.detail["largest_chunk_bytes"] == 160_000 == p.bytes_per_chunk
    assert p.detail["largest_chunk"]["obs_start"] == 2_500 and p.detail["largest_chunk"]["obs_end"] == 2_501
    assert p.detail["fits"] is False and not p.fits(tiny)
    with pytest.raises(MemoryError, match="cannot fit"):
        for _ in r:
            pass
    r.close()
    roomy = budget("640KB")                                  # 4x smaller than the 2.5 MB file
    r = g.H5adReader(path, 64 * 1024, roomy)
    p = r.plan()
    assert p.detail["fits"] is False and p.fits(roomy)       # over its chunk size, inside the budget
    with heap() as h:
        chunks = [(ch["obs_start"], ch["obs_end"], ch["resident_bytes"]) for ch in r]
        r.close()
    governed(roomy, 640 * 1024, h)
    assert (2_500, 2_501, 160_000) in chunks                 # its own chunk
    assert max(c[2] for c in chunks) == 160_000 == p.bytes_per_chunk
    assert sorted(c[2] for c in chunks)[-2] <= 64 * 1024     # everything else respects the chunk


def test_obsm_sparse_entries_are_cut_by_indptr_too(tmp_path):
    n, dim = 5_000, 400
    k = np.full(n, 6)
    k[:500] = 90                                             # the dense head
    path = str(tmp_path / "o.h5ad")
    indptr, indices, data = write_csr_h5ad(path, k, dim, obs_columns=False)
    with h5py.File(path, "a") as f:
        f.create_group("obsm")
        _csr(f["obsm"], "X_sp", indptr, indices, data, (n, dim))
    r = g.ObsmReader(path, "X_sp", chunk_bytes=20_000)
    p = r.plan()
    sizes = []
    for ch in r:
        true = (int(indptr[ch["obs_end"]]) - int(indptr[ch["obs_start"]])) * 8
        assert ch["resident_bytes"] == true
        sizes.append(true)
    r.close()
    assert max(sizes) <= 20_000
    assert p.n_chunks == len(sizes) and p.bytes_per_chunk == max(sizes)


def test_tenx_modality_chunks_lease_the_real_column_block(tenx_big):
    path, ref, info = tenx_big
    n_rna, n_adt = info["n_rna"], info["n_adt"]
    cap = 6 * MB
    b = budget("6MB")                                        # 28 MB of data+indices
    m = g.TenXMatrix(path, budget=b, modality="ADT")
    assert m.chunk_bytes == int(1.5 * MB)
    p = m.plan()
    sizes = []
    with heap() as h:
        for ch in m:
            c0, c1 = ch["cell_start"], ch["cell_end"]
            block = (int(m.indptr[c1]) - int(m.indptr[c0])) * 8
            # CSC: the whole column block (every feature type) is read, so that is what is leased
            assert b.stats()["ram"]["used"] == block == ch["resident_bytes"]
            X = ch["X"]
            assert X.shape == (n_adt, c1 - c0)
            assert (X != ref[n_rna:n_rna + n_adt, c0:c1]).nnz == 0
            assert list(ch["barcodes"][:1]) == [f"BC{c0:06d}-1"]
            sizes.append(block)
        m.close()
    governed(b, cap, h)
    assert max(sizes) <= m.chunk_bytes
    assert p.n_chunks == len(sizes) and p.bytes_per_chunk == max(sizes)
    assert p.detail["largest_chunk_bytes"] == max(sizes) and p.detail["fits"] is True
    assert "column block" in p.detail["note"]                # plan() says what is really read


def test_tenx_modality_nnz_is_counted_once_from_indices(tenx_big):
    path, ref, info = tenx_big
    n_rna, n_adt = info["n_rna"], info["n_adt"]
    m = g.TenXMatrix(path)
    per = m.modality_nnz("ADT")
    assert per.shape == (m.n_cells,) and int(per.sum()) == ref[n_rna:n_rna + n_adt].nnz
    assert np.array_equal(per, np.diff(ref[n_rna:n_rna + n_adt].tocsc().indptr))
    assert m.modality_nnz("ADT") is per                      # cached
    assert np.array_equal(m.modality_nnz("RNA") + m.modality_nnz("GUIDE") + per, np.diff(m.indptr))
    m.close()


def test_tenx_view_of_a_small_modality_is_costed_by_what_it_reads(tenx_big):
    path, ref, info = tenx_big
    n_rna, n_adt = info["n_rna"], info["n_adt"]
    cap = 6 * MB
    b = budget("6MB")
    v = g.open_tenx(path, modality="ADT", budget=b)
    m = v.m
    c = v.cost()
    ncell = m.n_cells
    out_bytes = ref[n_rna:n_rna + n_adt].nnz * 8 + (ncell + 1) * 4        # exactly what the ADT rows hold
    assert c["total_bytes"] == out_bytes                                   # it used to be 2% of the matrix
    assert c["read_bytes"] == m.nnz * 8 > out_bytes                        # CSC reads every feature of a cell
    assert c["fits_whole"] is False and "column blocks" in c["advice"]
    cells, worst = 0, 0
    with heap() as h:
        for X in v.chunks():
            assert X.shape[0] == n_adt
            worst = max(worst, X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)
            cells += X.shape[1]
        v.release()
    assert cells == ncell
    governed(b, cap, h)
    # the per-chunk figure the view reports covers the biggest column block it reads
    assert worst <= c["resident_bytes"] <= v._chunk_cap()
    v2 = g.open_tenx(path, modality="ADT", budget=b).select_frac(0.0, 0.1)
    c2 = v2.cost()
    X = v2.load()
    held = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
    assert X.shape == (n_adt, 2500)
    assert c2["total_bytes"] == held == b.stats()["ram"]["used"]           # cost() named what load() holds
    assert c2["read_bytes"] > held and c2["fits_whole"] is True
    v2.release()
    v3 = g.open_tenx(path, modality="RNA", budget=b).select_frac(0.2, 0.3)
    c3 = v3.cost()
    assert c3["total_bytes"] == (lambda Y: Y.data.nbytes + Y.indices.nbytes + Y.indptr.nbytes)(v3.load())
    v3.release()


def test_a_modality_loads_when_its_rows_fit_even_if_the_column_blocks_do_not(tenx_big):
    path, ref, info = tenx_big
    n_rna, n_adt = info["n_rna"], info["n_adt"]
    cap = 3 * MB // 2
    b = budget("1536KB")                                      # the 28 MB file; the window reads 2.8 MB
    v = g.open_tenx(path, modality="ADT", budget=b).select_frac(0.0, 0.1)
    c = v.cost()
    assert c["read_bytes"] > cap > c["total_bytes"] and c["fits_whole"] is True
    with heap() as h:
        X = v.load()
        held = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
        assert b.stats()["ram"]["used"] == held == c["total_bytes"]
        v.close()
    governed(b, cap, h)
    assert (X != ref[n_rna:n_rna + n_adt, 0:2500]).nnz == 0     # the comparison is not the reader's heap
    tight = budget("900KB")                                   # the rows alone (1.01 MB) no longer fit
    v = g.open_tenx(path, modality="ADT", budget=tight).select_frac(0.0, 0.1)
    assert v.cost()["fits_whole"] is False
    with pytest.raises(g.CostError, match="stream it with .chunks"):
        v.load()
    assert sum(X.shape[1] for X in v.chunks()) == 2500        # but it streams


def test_an_interleaved_modality_loads_and_counts_like_a_contiguous_one(tmp_path):
    nf, nc = 40, 300
    X = sp.random(nf, nc, density=0.2, format="csc", dtype=np.float32, random_state=2)
    X.data = np.ceil(X.data * 10).astype(np.int32)
    X.sort_indices()
    p = str(tmp_path / "inter.h5")
    with h5py.File(p, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("data", data=X.data)
        m.create_dataset("indices", data=X.indices.astype(np.int32))
        m.create_dataset("indptr", data=X.indptr.astype(np.int64))
        m.create_dataset("shape", data=np.array([nf, nc], np.int32))
        m.create_dataset("barcodes", data=np.array([f"B{i}".encode() for i in range(nc)]))
        ft = m.create_group("features")
        ft.create_dataset("id", data=np.array([f"E{i}".encode() for i in range(nf)]))
        ft.create_dataset("name", data=np.array([f"G{i}".encode() for i in range(nf)]))
        ft.create_dataset("feature_type", data=np.array([b"Gene Expression", b"Antibody Capture"] * 20))
    adt = X.tocsr()[np.arange(1, nf, 2)].tocsc()
    mt = g.TenXMatrix(p, chunk_bytes=2000)
    assert "interleaved" in mt.modalities["ADT"].note
    assert np.array_equal(mt.modality_nnz("ADT"), np.diff(adt.indptr))
    assert np.array_equal(mt.per_cell_totals("ADT"), np.asarray(adt.sum(0)).ravel())
    b = g.Budget(ram="1MB", timeout_s=5)
    v = g.open_tenx(p, modality="ADT", budget=b).select(37, 211)
    assert (v.load() != adt[:, 37:211]).nnz == 0
    v.set_zoom(1)
    assert (v.load() != adt[:, 37:211:2]).nnz == 0
    assert (sp.hstack(list(v.chunks()), format="csc") != adt[:, 37:211:2]).nnz == 0
    v.release()
    assert b.stats()["ram"]["used"] == 0 and b.stats()["ram"]["denials"] == 0


@pytest.mark.skipif(not os.path.exists(os.path.join(GLIOMA, "filtered_feature_bc_matrix.h5")),
                    reason="run examples/make_glioma_fixtures.py")
def test_tenx_fixture_adt_view_streams_inside_a_small_budget():
    h5 = os.path.join(GLIOMA, "filtered_feature_bc_matrix.h5")
    b = budget("640KB")                                      # the matrix holds 2.9 MB
    v = g.open_tenx(h5, modality="ADT", budget=b)
    with heap() as h:
        n = sum(X.shape[1] for X in v.chunks())
        v.close()
    governed(b, 640 * 1024, h)
    assert n == 3000


# ===========================================================================
# 4. per_cell_totals, Cell Ranger v2, GEO-style mtx names, dimension checks
# ===========================================================================

def test_per_cell_totals_survive_empty_cells(tmp_path):
    p = str(tmp_path / "empties.h5")
    ref, info = write_10x_h5(p, 400, n_rna=100, n_adt=5, n_guide=3, empty=(0, 7, 8, 9, 150, 398, 399))
    for chunk in (3_000, 20_000, 10 ** 9):
        m = g.TenXMatrix(p, chunk_bytes=chunk)
        np.testing.assert_array_equal(m.per_cell_totals(), np.asarray(ref.sum(0)).ravel())
        for mod, rows in (("RNA", slice(0, 100)), ("ADT", slice(100, 105)), ("GUIDE", slice(105, 108))):
            np.testing.assert_array_equal(m.per_cell_totals(mod), np.asarray(ref[rows].sum(0)).ravel())
        m.close()


def test_per_cell_totals_inside_a_budget(tenx_big):
    path, ref, info = tenx_big
    cap = 6 * MB
    b = budget("6MB")
    m = g.TenXMatrix(path, budget=b)
    with heap() as h:
        tot = m.per_cell_totals()
        m.close()
    governed(b, cap, h)
    np.testing.assert_array_equal(tot, np.asarray(ref.sum(0)).ravel())


def test_cell_ranger_v2_layout_opens(tmp_path):
    p = str(tmp_path / "v2.h5")
    ref, _ = write_10x_h5(p, 300, n_rna=60, n_adt=0, n_guide=0, v2=True)
    for kw in ({}, {"group": "GRCh38"}):
        m = g.TenXMatrix(p, **kw)
        assert (m.n_features, m.n_cells) == (60, 300) and list(m.modalities) == ["RNA"]
        assert m.feature_ids[0] == "ENSG0000000" and m.feature_names[3] == "G3"
        assert m.barcodes[2] == "BC000002-1" and set(m.genome) == {"GRCh38"}
        assert (m.block(10, 90) != ref[:, 10:90]).nnz == 0
        assert m.per_cell_totals().sum() == ref.sum()
        m.close()
    rep = g.modalities(p)
    assert rep.kind == "10x-h5" and rep.n_cells == 300


def test_cell_ranger_v2_with_several_genomes_asks_which(tmp_path):
    p = str(tmp_path / "multi.h5")
    write_10x_h5(p, 50, n_rna=30, n_adt=0, n_guide=0, v2=True)
    with h5py.File(p, "a") as f:
        f.copy("GRCh38", "mm10")
    with pytest.raises(ValueError, match=r"GRCh38.*mm10|mm10.*GRCh38"):
        g.TenXMatrix(p)
    assert g.TenXMatrix(p, group="mm10").n_cells == 50


def test_not_a_10x_file_still_says_so(tmp_path):
    p = str(tmp_path / "other.h5")
    with h5py.File(p, "w") as f:
        f.create_dataset("x", data=np.arange(3))
    with pytest.raises(ValueError, match="no 'matrix' group"):
        g.TenXMatrix(p)


def _coo_of(mt):
    parts = list(mt)
    return (np.concatenate([q["row"] for q in parts]), np.concatenate([q["col"] for q in parts]),
            np.concatenate([q["val"] for q in parts]))


def test_mtx_triple_accepts_geo_style_prefixed_names(tmp_path):
    rng = np.random.default_rng(2)
    X = sp.random(40, 30, density=0.3, format="coo", random_state=2)
    X.data = rng.integers(1, 9, X.nnz).astype(float)
    for features_name in ("features", "genes"):
        d = write_mtx_dir(str(tmp_path / features_name), X, prefix="GSM4339769", features_name=features_name)
        mt = g.MtxTriple(d)
        assert (mt.n_features, mt.n_cells, mt.nnz) == (40, 30, X.nnz)
        assert mt.barcodes[0] == "BC0-1" and mt.feature_names[1] == "G1"
        r, c, v = _coo_of(mt)
        assert (sp.coo_matrix((v, (r, c)), shape=X.shape) != X).nnz == 0
    d = write_mtx_dir(str(tmp_path / "dash"), X, prefix="s1", sep="-", gz=False)
    assert g.MtxTriple(d).nnz == X.nnz


def test_mtx_triple_with_several_prefixed_samples_needs_a_choice(tmp_path):
    X1 = sp.random(40, 30, density=0.2, format="coo", random_state=3)
    X2 = sp.random(40, 25, density=0.2, format="coo", random_state=4)
    d = str(tmp_path / "two")
    write_mtx_dir(d, X1, prefix="GSM1")
    write_mtx_dir(d, X2, prefix="GSM2")
    with pytest.raises(ValueError, match=r"GSM1.*GSM2"):
        g.MtxTriple(d)
    one = g.MtxTriple(os.path.join(d, "GSM2_matrix.mtx.gz"))      # naming the matrix picks the sample
    assert one.n_cells == 25 and one.nnz == X2.nnz
    assert g.MtxTriple(os.path.join(d, "GSM1_matrix.mtx.gz")).n_cells == 30


def test_find_accepts_prefixed_names_only_when_unambiguous(tmp_path):
    from geminga.tenx import _find
    d = tmp_path / "f"
    d.mkdir()
    assert _find(str(d), ("matrix.mtx.gz", "matrix.mtx")) is None
    (d / "GSM1_matrix.mtx.gz").write_bytes(b"")
    assert _find(str(d), ("matrix.mtx.gz", "matrix.mtx")).endswith("GSM1_matrix.mtx.gz")
    assert _find(str(d), ("matrix.mtx.gz", "matrix.mtx"), "") is None        # a prefix is exact
    assert _find(str(d), ("matrix.mtx.gz", "matrix.mtx"), "GSM1_").endswith("GSM1_matrix.mtx.gz")
    (d / "matrix.mtx").write_bytes(b"")                                       # plain names win
    assert _find(str(d), ("matrix.mtx.gz", "matrix.mtx")) == str(d / "matrix.mtx")
    (d / "matrix.mtx").unlink()
    (d / "GSM2_matrix.mtx.gz").write_bytes(b"")
    with pytest.raises(ValueError, match=r"GSM1.*GSM2"):
        _find(str(d), ("matrix.mtx.gz", "matrix.mtx"))


def test_mtx_triple_unprefixed_names_still_win(tmp_path):
    X = sp.random(20, 10, density=0.3, format="coo", random_state=1)
    d = write_mtx_dir(str(tmp_path / "plain"), X)
    write_mtx_dir(d, sp.random(20, 7, density=0.3, format="coo", random_state=1), prefix="GSM9")
    assert g.MtxTriple(d).n_cells == 10


def test_mtx_dimension_line_must_match_features_and_barcodes(tmp_path):
    X = sp.random(40, 30, density=0.3, format="coo", random_state=2)
    d = write_mtx_dir(str(tmp_path / "short"), X, n_barcodes=31)
    with pytest.raises(ValueError, match=r"barcodes.*31.*30|30.*31"):
        g.MtxTriple(d)
    d = write_mtx_dir(str(tmp_path / "feat"), X, n_features=41)
    with pytest.raises(ValueError, match=r"features.*41|41.*40"):
        g.MtxTriple(d)
    d = write_mtx_dir(str(tmp_path / "transposed"), X, header_dims=(30, 40), n_barcodes=30, n_features=40)
    with pytest.raises(ValueError, match="transposed"):
        g.MtxTriple(d)


# ===========================================================================
# 5. to_h5: in memory when it fits, otherwise two passes through memmaps
# ===========================================================================

@pytest.fixture(scope="module")
def mtx_big(tmp_path_factory):
    """2,000 x 4,000, 1.2M entries, in random order (not column-sorted): 14 MB as COO."""
    rng = np.random.default_rng(5)
    X = sp.random(2_000, 4_000, density=0.15, format="coo", dtype=np.float64, random_state=5)
    X.data = rng.integers(1, 50, X.nnz).astype(float)
    d = str(tmp_path_factory.mktemp("mtxbig") / "m")
    write_mtx_dir(d, X)
    return d, X


def _same_matrix(h5, X):
    m = g.TenXMatrix(h5)
    got = m.block(0, m.n_cells)
    m.close()
    assert got.shape == X.shape
    assert (got != X.tocsc()).nnz == 0
    return got


def test_mtx_to_h5_spills_instead_of_holding_the_matrix(mtx_big, tmp_path):
    d, X = mtx_big
    cap = 3 * MB
    b = budget("3MB")                                        # nnz * 24 = 29 MB; the COO is 14 MB
    mt = g.MtxTriple(d, budget=b)
    dest = str(tmp_path / "out" / "m.h5")
    os.makedirs(os.path.dirname(dest))
    with heap() as h:
        assert mt.to_h5(dest) == dest
        mt.close()
    governed(b, cap, h)
    assert mt.last_conversion["path"] == "spill"
    assert mt.last_conversion["spill_bytes"] == X.nnz * 8    # int32 data + int32 indices
    assert os.listdir(os.path.dirname(dest)) == ["m.h5"]     # the temp directory beside dest is gone
    got = _same_matrix(dest, X)
    assert got.has_sorted_indices
    with h5py.File(dest, "r") as f:
        assert f["matrix/data"].dtype == np.int32 and f["matrix/indices"].dtype == np.int32
        assert f["matrix/indptr"].dtype == np.int64
        assert list(f["matrix/features/feature_type"][:2]) == [b"Gene Expression"] * 2


def test_mtx_to_h5_stays_in_memory_when_it_fits(mtx_big, tmp_path):
    d, X = mtx_big
    mt = g.MtxTriple(d)                                      # no budget
    mt.to_h5(str(tmp_path / "ram.h5"))
    assert mt.last_conversion["path"] == "ram"
    _same_matrix(str(tmp_path / "ram.h5"), X)
    need = X.nnz * 24
    b = budget(f"{need * 3}")                                # nnz * 24 <= available // 2
    mt = g.MtxTriple(d, budget=b)
    mt.to_h5(str(tmp_path / "ram2.h5"))
    mt.close()
    assert mt.last_conversion["path"] == "ram"
    assert need <= b.stats()["ram"]["peak"] <= need * 3 and b.stats()["ram"]["used"] == 0
    b = budget(f"{need * 2 - 1000}")                         # one byte past the half-budget rule
    mt = g.MtxTriple(d, budget=b)
    mt.to_h5(str(tmp_path / "spill.h5"))
    mt.close()
    assert mt.last_conversion["path"] == "spill"
    a, c = g.TenXMatrix(str(tmp_path / "ram2.h5")), g.TenXMatrix(str(tmp_path / "spill.h5"))
    assert (a.block(0, a.n_cells) != c.block(0, c.n_cells)).nnz == 0
    a.close()
    c.close()


def test_mtx_to_h5_takes_its_scratch_space_from_a_declared_spill_tier(mtx_big, tmp_path):
    d, X = mtx_big
    b = g.Budget(ram="3MB", spill="64MB", timeout_s=5)
    mt = g.MtxTriple(d, budget=b)
    mt.to_h5(str(tmp_path / "a.h5"))
    mt.close()
    st = b.stats()
    assert st["spill"]["peak"] == mt.last_conversion["spill_bytes"] == X.nnz * 8 and st["spill"]["used"] == 0
    tight = g.Budget(ram="3MB", spill="1MB", timeout_s=5)             # a spill tier too small for it
    mt = g.MtxTriple(d, budget=tight)
    with pytest.raises(MemoryError, match="spill"):
        mt.to_h5(str(tmp_path / "b.h5"))
    mt.close()
    assert not os.path.exists(tmp_path / "b.h5") and os.listdir(tmp_path) == ["a.h5"]


def test_mtx_streams_exact_batches_inside_a_small_budget(mtx_big):
    d, X = mtx_big
    cap = 3 * MB
    b = budget("3MB")                                        # the COO is 14.4 MB; parsed it is 29 MB
    mt = g.MtxTriple(d, budget=b)
    p = mt.plan()
    assert p.detail["parse"] == "stream" and p.detail["fits"]
    sizes = []
    with heap() as h:
        for blk in mt:
            n = len(blk["row"])
            assert b.stats()["ram"]["used"] == 12 * n
            sizes.append(n)
        mt.close()
    governed(b, cap, h)
    per = p.detail["entries_per_chunk"]
    assert sum(sizes) == X.nnz == mt.nnz
    assert sizes[:-1] == [per] * (len(sizes) - 1) and 0 < sizes[-1] <= per      # exact batches, remainder last
    assert p.n_chunks == len(sizes) and p.bytes_per_chunk == 12 * per


def test_dense_h5ad_streams_inside_a_small_budget(tmp_path):
    path = str(tmp_path / "dense.h5ad")
    X = np.random.default_rng(0).random((12_000, 400)).astype(np.float32)      # 19 MB
    with h5py.File(path, "w") as f:
        _enc(f, "anndata", "0.1.0")
        _arr(f, "X", X)
        _arr(f.create_group("layers"), "twice", X * 2)
        obs = _frame(f, "obs")
        _vlen(obs, "_index", [f"c{i}" for i in range(12_000)])
    cap = 4 * MB
    b = budget("4MB")
    r = g.H5adReader(path, budget=b)
    p = r.plan()
    rows = 0
    with heap() as h:
        for ch in r:
            assert ch["X"].shape[0] == ch["obs_end"] - ch["obs_start"] and ch["resident_bytes"] == ch["X"].nbytes
            assert b.stats()["ram"]["used"] == ch["X"].nbytes
            assert np.array_equal(ch["X"][:1], X[ch["obs_start"]:ch["obs_start"] + 1])
            rows += ch["X"].shape[0]
        r.close()
    governed(b, cap, h)
    assert rows == 12_000 and p.bytes_per_chunk <= r.chunk_bytes == MB and p.detail["fits"]
    assert p.detail["rows_per_chunk"] == MB // (400 * 4)
    tw = g.H5adReader(path, layer="twice", chunk_bytes="2MB")
    assert np.array_equal(next(iter(tw))["X"], (X * 2)[:tw.plan().detail["rows_per_chunk"]])
    tw.close()


def test_closing_a_half_read_reader_gives_every_byte_back(mtx_big, skewed, tenx_big):
    b = budget("64MB")
    mt = g.MtxTriple(mtx_big[0], budget=b)               # nnz * 24 fits: the parsed matrix is leased
    it = iter(mt)
    next(it)
    assert b.stats()["ram"]["used"] > mtx_big[1].nnz * 24
    mt.close()
    assert b.stats()["ram"]["used"] == 0
    for r in (g.H5adReader(skewed[0], budget=b), g.TenXMatrix(tenx_big[0], budget=b)):
        next(iter(r))
        assert b.stats()["ram"]["used"] > 0
        r.close()
        assert b.stats()["ram"]["used"] == 0


def test_mtx_with_more_entries_than_declared_is_an_error(tmp_path):
    X = sp.random(20, 10, density=0.4, format="coo", random_state=1)
    d = write_mtx_dir(str(tmp_path / "long"), X)
    path = os.path.join(d, "matrix.mtx.gz")
    with gzip.open(path, "rt") as fh:
        text = fh.read()
    with gzip.open(path, "wt") as fh:
        fh.write(text + "\n1 1 5\n")                                  # one entry past the header's count
    with pytest.raises(ValueError, match="more than"):
        list(g.MtxTriple(d, budget=budget("2KB"), chunk_bytes=600))
    with gzip.open(path, "wt") as fh:
        fh.write(text.rsplit("\n", 2)[0] + "\n")                      # one entry short
    with pytest.raises(ValueError, match="holds"):
        list(g.MtxTriple(d, budget=budget("2KB"), chunk_bytes=600))


def test_mtx_to_h5_both_paths_agree_on_floats_and_duplicates(tmp_path):
    rows = np.array([0, 3, 3, 1, 2, 3, 0, 2, 1, 1] * 400)
    cols = np.array([1, 0, 0, 2, 2, 1, 3, 3, 0, 2] * 400)        # (3,0) and (1,2) repeat
    vals = np.round(np.random.default_rng(0).random(rows.size) * 4 + 0.25, 3)
    X = sp.coo_matrix((vals, (rows, cols)), shape=(4, 4))
    d = str(tmp_path / "dup")
    os.makedirs(d)
    with gzip.open(os.path.join(d, "matrix.mtx.gz"), "wt") as fh:
        fh.write(f"%%MatrixMarket matrix coordinate real general\n4 4 {rows.size}\n")
        fh.writelines(f"{r + 1} {c + 1} {v!r}\n"
                      for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist()))
    for name, n in (("barcodes", 4), ("features", 4)):
        with gzip.open(os.path.join(d, f"{name}.tsv.gz"), "wt") as fh:
            fh.writelines((f"BC{i}\n" if name == "barcodes" else f"E{i}\tG{i}\tGene Expression\n")
                          for i in range(n))
    ram = g.MtxTriple(d).to_h5(str(tmp_path / "ram.h5"))
    spill = g.MtxTriple(d, budget=budget("40KB"))
    spill.to_h5(str(tmp_path / "spill.h5"))
    assert spill.last_conversion["path"] == "spill"
    want = X.tocsc()
    want.sum_duplicates()
    for path in (ram, str(tmp_path / "spill.h5")):
        m = g.TenXMatrix(path)
        got = m.block(0, 4)
        assert got.nnz == want.nnz and np.allclose(got.toarray(), want.toarray(), atol=1e-4)
        with h5py.File(path, "r") as f:
            assert f["matrix/data"].dtype == np.float32
        m.close()


# ===========================================================================
# 6. frames, ParquetSink, Arrow IPC
# ===========================================================================

def _row_group_file(path, n=60_000, groups=12):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = np.random.default_rng(0)
    t = pa.table({"x": rng.random(n) * 1000,                  # every group spans the whole x range
                  "y": np.sort(rng.random(n)) * 1000,         # row groups are y-contiguous
                  "gene": rng.choice(["a", "b", "c"], n),
                  "z": rng.random(n)})
    pq.write_table(t, path, row_group_size=n // groups)
    return t


def test_lazy_window_filters_before_it_projects(tmp_path):
    pytest.importorskip("polars")
    from geminga.frames import lazy_window
    p = str(tmp_path / "pts.parquet")
    t = _row_group_file(p)
    lf = lazy_window(p, "x", 100, 300, "y", 400, 700, columns=["gene"])    # predicate columns not projected
    out = lf.collect()
    assert out.columns == ["gene"]
    x, y = t["x"].to_numpy(), t["y"].to_numpy()
    assert out.height == int(((x >= 100) & (x < 300) & (y >= 400) & (y < 700)).sum())
    assert lazy_window(p, "x", 100, 300, columns=["z", "gene"]).collect().columns == ["z", "gene"]


def test_lazy_window_explain_uses_the_y_bound(tmp_path):
    pytest.importorskip("polars")
    from geminga.frames import lazy_window
    p = str(tmp_path / "pts.parquet")
    _row_group_file(p)
    _, x_only = lazy_window(p, "x", 100, 300, columns=["gene"], explain=True)
    _, both = lazy_window(p, "x", 100, 300, "y", 400, 500, columns=["gene"], explain=True)
    assert x_only["row_groups_kept"] == x_only["row_groups_total"] == 12      # x alone prunes nothing
    assert both["row_groups_kept"] < 12 and both["candidate_rows"] < x_only["candidate_rows"]
    assert 0 < both["selectivity"] < x_only["selectivity"]
    kept = both["row_groups_per_file"][0]
    assert kept == list(range(kept[0], kept[-1] + 1)) and len(kept) == both["row_groups_kept"]
    assert both["x"]["row_groups_kept"] == 12 and both["y"]["row_groups_kept"] == both["row_groups_kept"]


@pytest.mark.skipif(not os.path.exists(XENIUM), reason="run examples/make_glioma_fixtures.py")
def test_lazy_window_explain_y_never_widens_the_x_plan():
    pytest.importorskip("polars")
    from geminga.frames import lazy_window
    _, x_only = lazy_window(XENIUM, "x_location", 1000, 1400, explain=True)
    _, both = lazy_window(XENIUM, "x_location", 1000, 1400, "y_location", 500, 900, explain=True)
    assert both["row_groups_kept"] <= x_only["row_groups_kept"] < x_only["row_groups_total"]


def test_parquet_sink_survives_an_all_null_first_batch(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    first = pa.record_batch({"id": [1, 2], "note": pa.array([None, None]), "n": pa.array([None, None])})
    later = pa.record_batch({"id": [3, 4], "note": ["a", "b"], "n": [5, 6]})
    p = str(tmp_path / "sink.parquet")
    with g.ParquetSink(p) as s:
        s.write(first)
        s.write(later)
        s.write(pa.record_batch({"id": [5], "note": pa.array([None]), "n": pa.array([None])}))
    assert s.rows == 5
    t = pq.read_table(p)
    assert t["note"].to_pylist() == [None, None, "a", "b", None]
    assert t["n"].to_pylist() == [None, None, 5, 6, None]            # the real type, not a string
    assert pa.types.is_integer(t.schema.field("n").type)


def test_parquet_sink_declared_schema_and_unresolved_columns(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    p = str(tmp_path / "decl.parquet")
    schema = pa.schema([("id", pa.int64()), ("score", pa.float32())])
    with g.ParquetSink(p, schema=schema) as s:
        s.write(pa.record_batch({"id": [1], "score": pa.array([None])}))
        s.write({"id": np.array([2, 3]), "score": np.array([0.5, 1.5])})
    t = pq.read_table(p)
    assert t.schema.field("score").type == pa.float32()
    assert t["score"].to_pylist() == [None, 0.5, 1.5]
    p2 = str(tmp_path / "nulls.parquet")                             # never resolved: written as string
    with g.ParquetSink(p2) as s:
        s.write(pa.record_batch({"id": [1, 2], "gone": pa.array([None, None])}))
    t2 = pq.read_table(p2)
    assert t2["gone"].to_pylist() == [None, None] and pa.types.is_string(t2.schema.field("gone").type)
    p3 = str(tmp_path / "bad.parquet")
    with g.ParquetSink(p3) as s:
        s.write(pa.record_batch({"id": [1], "v": [1.5]}))
        with pytest.raises(ValueError, match="'v'"):
            s.write(pa.record_batch({"id": [2], "v": ["text"]}))


def _ipc_file(path, rows, stream=False, batches=1, cols=3):
    import pyarrow as pa
    rng = np.random.default_rng(0)
    schema = pa.schema([(f"c{i}", pa.int64()) for i in range(cols)])
    new = pa.ipc.new_stream if stream else pa.ipc.new_file
    with pa.OSFile(path, "wb") as sink, new(sink, schema) as w:
        for _ in range(batches):
            w.write_batch(pa.record_batch([pa.array(rng.integers(0, 1 << 40, rows // batches))
                                           for _ in range(cols)], schema=schema))


@pytest.mark.parametrize("stream", [False, True])
def test_arrow_ipc_splits_a_batch_bigger_than_the_chunk(tmp_path, stream):
    path = str(tmp_path / ("s.arrows" if stream else "f.arrow"))
    _ipc_file(path, 1_000_000, stream=stream)                        # one 24 MB batch
    cap = 4 * MB
    b = budget("4MB")
    r = g.ArrowIPCReader(path, budget=b)
    assert r.chunk_bytes == MB
    p = r.plan()
    rows, sizes = 0, []
    with heap() as h:
        for batch in r:
            assert batch.nbytes <= MB
            assert b.stats()["ram"]["used"] == batch.nbytes
            rows += batch.num_rows
            sizes.append(batch.nbytes)
        r.close()
    governed(b, cap, h)
    assert rows == 1_000_000 and len(sizes) > 20
    if not stream:                                                   # a file knows its batches: exact
        assert p.n_chunks == len(sizes) and p.bytes_per_chunk == max(sizes)
        assert p.detail["fits"] is True
    else:
        assert p.bytes_per_chunk <= MB


def test_arrow_ipc_budget_covers_only_selected_columns_and_each_slice(tmp_path):
    path = str(tmp_path / "f.arrow")
    _ipc_file(path, 600_000, batches=3, cols=4)
    b = budget("2MB")
    r = g.ArrowIPCReader(path, budget=b, columns=["c1", "c3"])
    total = 0
    with heap() as h:
        for batch in r:
            assert batch.schema.names == ["c1", "c3"]
            assert b.stats()["ram"]["used"] == batch.nbytes <= r.chunk_bytes
            total += batch.num_rows
        r.close()
    governed(b, 2 * MB, h)
    assert total == 600_000


def test_arrow_ipc_without_a_budget_keeps_batches_whole(tmp_path):
    path = str(tmp_path / "f.arrow")
    _ipc_file(path, 90_000, batches=3)
    r = g.ArrowIPCReader(path)
    assert [b.num_rows for b in r] == [30_000] * 3
    r.close()
