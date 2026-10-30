"""10x matrices split by feature_type, spatial bundles, multiplexed imaging, and the overview."""
import gzip
import os
import numpy as np
import pytest
import geminga as g

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")
G = os.path.join(FIX, "glioma")
H5 = os.path.join(G, "filtered_feature_bc_matrix.h5")
need = pytest.mark.skipif(not os.path.exists(H5), reason="run examples/make_glioma_fixtures.py")


@need
def test_feature_types_become_modalities():
    m = g.TenXMatrix(H5)
    assert set(m.modalities) == {"RNA", "ADT", "GUIDE"}
    assert m.modalities["RNA"].n_features == 2000
    assert m.modalities["ADT"].n_features == 40
    assert m.modalities["GUIDE"].n_features == 24
    assert m.modalities["ADT"].row_start == 2000 and m.modalities["ADT"].row_end == 2040
    assert m.block(0, 500, "ADT").shape == (40, 500)
    assert m.block(0, 500).shape == (2064, 500)
    m.close()


@need
def test_per_modality_totals_are_per_modality():
    m = g.TenXMatrix(H5)
    gt, rt, at = (m.per_cell_totals(k) for k in ("GUIDE", "RNA", "ADT"))
    assert 30 < np.median(gt) < 60                     # one guide at ~45 UMIs per cell
    assert np.allclose(gt + rt + at, m.per_cell_totals())
    assert (gt == 0).sum() == 0
    m.close()


@need
def test_tenx_view_accounts_and_slices_one_modality():
    b = g.Budget(ram="32MB")
    v = g.open_tenx(H5, modality="RNA", budget=b).select_frac(0.5, 0.6)
    X = v.load()
    assert X.shape == (2000, 300)
    used = b.stats()["ram"]["used"]
    assert used == X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
    ov = v.reset().overview(40)
    assert "RNA_total_per_cell" in ov.series and len(ov) <= 40
    with pytest.raises(KeyError):
        g.open_tenx(H5, modality="PEAKS")


@need
def test_mtx_triple_streams_then_converts_identically(tmp_path):
    mt = g.MtxTriple(os.path.join(G, "mtx"), budget=g.Budget(ram="16MB"))
    assert not mt.plan().detail["navigable"]
    n = sum(len(b["row"]) for b in mt)
    assert n == mt.nnz
    h5 = mt.to_h5(str(tmp_path / "conv.h5"))
    a, c = g.TenXMatrix(H5), g.TenXMatrix(h5)
    assert a.nnz == c.nnz and set(a.modalities) == set(c.modalities)
    assert np.array_equal(a.block(100, 200).toarray(), c.block(100, 200).toarray())
    with pytest.raises(ValueError, match="not navigable"):
        g.open_tenx(os.path.join(G, "mtx"))


@need
def test_modality_chunks_are_planned_from_the_blocks_they_read():
    """An ADT view used to be planned at 2% of the matrix and iterate into a MemoryError."""
    cap = 640 * 1024                                     # the matrix holds 2.9 MB of data+indices
    b = g.Budget(ram="640KB", timeout_s=5)
    m = g.TenXMatrix(H5, budget=b, modality="ADT")
    p = m.plan()
    assert p.n_chunks > 1 and p.bytes_per_chunk <= m.chunk_bytes and p.detail["fits"]
    sizes, n_adt = [], 0
    for ch in m:
        c0, c1 = ch["cell_start"], ch["cell_end"]
        block = (int(m.indptr[c1]) - int(m.indptr[c0])) * 8
        assert b.stats()["ram"]["used"] == block == ch["resident_bytes"]     # the whole column block
        assert ch["X"].shape == (40, c1 - c0)
        sizes.append(block)
        n_adt += ch["X"].nnz
    m.close()
    assert b.stats()["ram"]["peak"] <= cap and b.stats()["ram"]["denials"] == 0
    assert p.bytes_per_chunk == max(sizes) and p.n_chunks == len(sizes)
    assert n_adt == int(g.TenXMatrix(H5).modality_nnz("ADT").sum())


@need
def test_mtx_triple_converts_in_two_passes_under_a_small_budget(tmp_path):
    b = g.Budget(ram="1MB", timeout_s=5)                 # nnz * 24 = 8.7 MB is far past half of it
    mt = g.MtxTriple(os.path.join(G, "mtx"), budget=b)
    h5 = mt.to_h5(str(tmp_path / "spilled.h5"))
    mt.close()
    assert mt.last_conversion["path"] == "spill" and mt.last_conversion["spill_bytes"] == mt.nnz * 8
    assert b.stats()["ram"]["peak"] <= 1 << 20 and b.stats()["ram"]["used"] == 0
    assert os.listdir(tmp_path) == ["spilled.h5"]
    a, c = g.TenXMatrix(H5), g.TenXMatrix(h5)
    assert a.nnz == c.nnz and set(a.modalities) == set(c.modalities)
    assert np.array_equal(a.block(0, 3000).toarray(), c.block(0, 3000).toarray())
    assert np.array_equal(a.per_cell_totals(), c.per_cell_totals())


@need
def test_mtx_triple_refuses_a_transposed_or_mismatched_matrix(tmp_path):
    import shutil
    d = str(tmp_path / "bad")
    shutil.copytree(os.path.join(G, "mtx"), d)
    with gzip.open(os.path.join(d, "barcodes.tsv.gz"), "rt") as fh:
        lines = fh.readlines()
    with gzip.open(os.path.join(d, "barcodes.tsv.gz"), "wt") as fh:
        fh.writelines(lines[:-1])                        # one barcode short
    with pytest.raises(ValueError, match=r"barcodes.*2999.*3000"):
        g.MtxTriple(d)


def _mtx_dir(path, X, kind="real"):
    import gzip
    os.makedirs(path, exist_ok=True)
    X = X.tocoo()
    with gzip.open(os.path.join(path, "matrix.mtx.gz"), "wt") as fh:
        fh.write(f"%%MatrixMarket matrix coordinate {kind} general\n% a comment\n{X.shape[0]} {X.shape[1]} {X.nnz}\n")
        for r, c, v in zip(X.row, X.col, X.data):
            fh.write(f"{r + 1} {c + 1}\n" if kind == "pattern" else f"{r + 1} {c + 1} {v:g}\n")
    with gzip.open(os.path.join(path, "barcodes.tsv.gz"), "wt") as fh:
        fh.writelines(f"B{i}\n" for i in range(X.shape[1]))
    with gzip.open(os.path.join(path, "features.tsv.gz"), "wt") as fh:
        fh.writelines(f"ENSG{i:011d}\tG{i}\tGene Expression\n" for i in range(X.shape[0]))
    return X


def test_mtx_fast_and_streaming_paths_agree(tmp_path):
    import scipy.sparse as sp
    X = _mtx_dir(str(tmp_path / "m"), sp.random(1000, 500, density=0.1, format="coo", dtype=np.float32, random_state=3))

    def collect(mt):
        parts = list(mt)
        return (np.concatenate([p["row"] for p in parts]), np.concatenate([p["col"] for p in parts]),
                np.concatenate([p["val"] for p in parts]), len(parts))
    fast = collect(g.MtxTriple(str(tmp_path / "m")))                              # fits: SciPy path
    tiny = g.Budget(ram="300KB")                                                  # 50,000 nnz × 24 B > 150 KB
    slow = collect(g.MtxTriple(str(tmp_path / "m"), budget=tiny))                 # must stream
    assert slow[3] > 1 and fast[3] == 1
    for a, b in zip(fast[:3], slow[:3]):
        assert np.array_equal(a, b)
    assert np.array_equal(fast[0], X.row) and np.allclose(fast[2], X.data)
    assert tiny.stats()["ram"]["denials"] == 0 and tiny.stats()["ram"]["peak"] <= 300 * 1024


def test_mtx_pattern_matrices_read_as_ones(tmp_path):
    import scipy.sparse as sp
    X = _mtx_dir(str(tmp_path / "p"), sp.random(50, 40, density=0.3, format="coo", random_state=1), kind="pattern")
    mt = g.MtxTriple(str(tmp_path / "p"), budget=g.Budget(ram="100KB"))
    parts = list(mt._stream(per=500))
    vals = np.concatenate([p["val"] for p in parts])
    assert len(vals) == X.nnz and np.all(vals == 1)


@need
def test_visium_bundle_registers_matrix_positions_image():
    vb = g.open_visium(os.path.join(G, "visium"))
    d = vb.describe()
    assert d["platform"] == "Visium" and d["n_spots"] == 1500
    assert d["in_tissue"] == 1203 and d["scale"]["tissue_hires_scalef"] == 0.5
    assert vb.hires and os.path.exists(vb.hires)
    idx = vb.in_window((1500, 2500, 800, 1600))
    assert 0 < idx.size < 1500
    x0, x1, y0, y1 = d["extent_px"]
    assert vb.in_window((x0, x1, y0, y1)).size == 1500
    vb.close()


@need
def test_xenium_prunes_transcripts_by_coordinate():
    xb = g.open_xenium(os.path.join(G, "xenium"))
    d = xb.describe()
    assert d["n_transcripts"] == 400_000 and "Brain" in d["panel"]
    p = xb.prune(1000, 1400)
    assert p["row_groups_kept"] < p["row_groups_total"] / 3      # sorted by x, so it pays
    t = xb.window(1000, 1400, 500, 900)
    x, y = t.column("x_location").to_numpy(), t.column("y_location").to_numpy()
    assert x.min() >= 1000 and x.max() < 1400 and y.min() >= 500 and y.max() < 900
    assert 0 < t.num_rows < 20_000


@need
def test_multichannel_tiff_reads_one_channel():
    mc = g.MultiChannelTiff(os.path.join(G, "codex_6plex.ome.tif"))
    assert mc.n_channels == 6 and mc.channel_names == ["DAPI", "CD45", "CD68", "GFAP", "SOX2", "PD-L1"]
    gf = mc.channel("GFAP", 600, 1000, 500, 900)
    assert gf.shape == (400, 400)
    assert gf.mean() > 5 * mc.channel("DAPI", 600, 1000, 500, 900).mean()
    assert np.array_equal(mc.channel(3, 0, 10, 0, 10), mc.channel("GFAP", 0, 10, 0, 10))
    mc.close()


@need
def test_bundle_sniffing_is_shallow_and_correct():
    assert g.sniff_bundle(os.path.join(G, "xenium"))["kind"] == "xenium"
    assert g.sniff_bundle(os.path.join(G, "visium"))["kind"] == "visium"
    assert g.sniff_bundle(os.path.join(G, "mtx"))["kind"] == "mtx-triple"
    assert g.sniff_bundle(G)["kind"] != "xenium"                 # a neighbour is not this bundle
    r = g.modalities(os.path.join(G, "visium"))
    assert r.n_cells == 1500 and r.spatial and r.images
    r = g.modalities(H5)
    assert set(r.modalities) == {"RNA", "ADT", "GUIDE"}


@need
def test_profile_sees_cite_and_crispr_in_a_10x_h5():
    p = g.profile(H5, fetch_page=False)
    assert p.format == "10x HDF5"
    assert p.metadata.get("cite_seq") and p.metadata.get("crispr_screen")
    assert not p.confident                                        # sc vs sn is still open
    x = g.profile(os.path.join(G, "xenium"), fetch_page=False)
    assert x.format == "xenium" and x.best.label == "Visium / spot spatial"


@need
def test_open_dispatches_bundles_and_h5():
    assert isinstance(g.open(H5), g.TenXView)
    assert isinstance(g.open(os.path.join(G, "visium")), g.VisiumBundle)
    assert isinstance(g.open(os.path.join(G, "xenium")), g.XeniumBundle)
    assert isinstance(g.open(os.path.join(G, "mtx")), g.MtxTriple)
