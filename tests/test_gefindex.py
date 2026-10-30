"""The GEF spatial index: same answers, a fraction of the reading."""
import os
import shutil
import numpy as np
import pytest
import geminga as g

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")
GEF = os.path.join(FIX, "chip.gef")
pytest.importorskip("polars")
need = pytest.mark.skipif(not os.path.exists(GEF), reason="run the fixture scripts first")


@pytest.fixture
def gef_copy(tmp_path):
    dst = tmp_path / "chip.gef"
    shutil.copy(GEF, dst)
    return str(dst)


def test_morton_interleaves_bits():
    assert g.morton2(np.array([0]), np.array([0]))[0] == 0
    assert g.morton2(np.array([1]), np.array([0]))[0] == 1
    assert g.morton2(np.array([0]), np.array([1]))[0] == 2
    assert g.morton2(np.array([2]), np.array([0]))[0] == 4
    assert g.morton2(np.array([3]), np.array([3]))[0] == 15
    # Z-order: the four cells of each 2x2 block are consecutive
    xs, ys = np.meshgrid(np.arange(4), np.arange(4))
    k = g.morton2(xs.ravel(), ys.ravel())
    assert sorted(k.tolist()) == list(range(16))
    big = g.morton2(np.array([260_000]), np.array([260_000]))[0]    # a 13 cm chip at bin1
    assert big > 0


@need
def test_index_returns_identical_observations(gef_copy):
    plain = g.open_gef(gef_copy).gef
    plain._sidx, plain._sidx_checked = None, True
    gf = g.open_gef(gef_copy).gef
    gf.build_spatial_index()
    for w in [(1000, 1400, 1200, 1600), (0, 4000, 0, 4000), (3900, 4000, 3900, 4000)]:
        for b in (1, 20):
            a, c = plain.read_window(w, b), gf.read_window(w, b)
            assert sorted(zip(a["x"], a["y"], a["count"])) == \
                   sorted(zip(c["x"], c["y"], c["count"]))


@need
def test_small_windows_read_a_fraction_of_the_level(gef_copy):
    gf = g.open_gef(gef_copy).gef
    before = gf.read_plan((1000, 1400, 1200, 1600), 1)
    assert before["path"] == "full-level" and "advice" in before
    gf.build_spatial_index()
    after = gf.read_plan((1000, 1400, 1200, 1600), 1)
    assert after["path"] == "spatial-index"
    assert after["read_bytes"] * 5 < before["read_bytes"]            # compressed vs compressed
    tiny = gf.read_plan((2000, 2040, 2000, 2040), 1)
    assert tiny["read_bytes"] <= after["read_bytes"]


@need
def test_cost_reports_read_and_kept(gef_copy):
    gf = g.open_gef(gef_copy).gef
    gf.build_spatial_index()
    v = g.open_gef(gef_copy).use_bin(1).select(1000, 1400, 1200, 1600)
    c = v.cost()
    assert c["read_path"] == "spatial-index"
    assert c["read_bytes"] > 0 and c["resident_bytes"] > 0


@need
def test_stale_index_is_ignored(gef_copy):
    gf = g.open_gef(gef_copy).gef
    gf.build_spatial_index(levels=[20])
    assert g.SpatialIndex(gef_copy).valid
    os.utime(gef_copy, (1, 1))                                          # the GEF "changed"
    assert not g.SpatialIndex(gef_copy).valid
    assert g.open_gef(gef_copy).gef.spatial_index is None


@need
def test_partial_index_falls_back_per_level(gef_copy):
    gf = g.open_gef(gef_copy).gef
    gf.build_spatial_index(levels=[20])
    fresh = g.open_gef(gef_copy).gef
    assert fresh.read_plan((0, 500, 0, 500), 20)["path"] == "spatial-index"
    assert fresh.read_plan((0, 500, 0, 500), 1)["path"] == "full-level"


@need
def test_gene_queries_keep_using_gene_slices(gef_copy):
    gf = g.open_gef(gef_copy).gef
    gf.build_spatial_index()
    top = g.open_gef(gef_copy).use_bin(10).gene_profile(1)[0][0]
    assert gf.read_plan((0, 4000, 0, 4000), 10, [top])["path"] == "gene-slices"
