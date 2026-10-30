"""Stereo-seq: the bin pyramid as a zoom stack, gene slices, and multimodal registration."""
import os
import numpy as np
import pytest
import geminga as g

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")
GEF, GEM = os.path.join(FIX, "chip.gef"), os.path.join(FIX, "chip.gem")
IMG = os.path.join(FIX, "chip_ssdna.tif")
need_gef = pytest.mark.skipif(not os.path.exists(GEF),
                              reason="run examples/make_stereoseq_fixtures.py first")
need_gem = pytest.mark.skipif(not os.path.exists(GEM), reason="no GEM fixture")


@need_gef
def test_bin_levels_are_discovered_and_ordered():
    v = g.open_gef(GEF)
    assert v.gef.bins == sorted(v.gef.bins)
    assert v.gef.bins[0] == 1
    for b in v.gef.bins:
        assert v.gef.levels[b].n_obs > 0
    # coarser bins must hold strictly fewer observations — that is what makes them cheaper
    counts = [v.gef.levels[b].n_obs for b in v.gef.bins]
    assert counts == sorted(counts, reverse=True)
    v.gef.close()


@need_gef
def test_zoom_walks_the_pyramid_and_shrinks_cost():
    v = g.open_gef(GEF, budget=g.Budget(ram="256MB")).reset()
    seen, resident = [], []
    for z in range(len(v.gef.bins)):
        v.set_zoom(z)
        c = v.cost()
        seen.append(c["bin_size"])
        resident.append(c["resident_bytes"])
    assert seen == v.gef.bins                     # one zoom level per bin SAW wrote
    assert resident == sorted(resident, reverse=True)
    assert resident[-1] < resident[0] / 10        # coarsest is at least 10x cheaper
    v.gef.close()


@need_gef
def test_cost_is_metadata_only_and_advice_names_a_bin():
    v = g.open_gef(GEF, budget=g.Budget(ram="2MB")).reset().use_bin(1)
    c = v.cost()
    assert c["fits_whole"] is False
    assert "bin" in c["advice"]
    v.use_bin(None).auto_bin()
    assert v.cost()["fits_whole"] is True
    v.gef.close()


@need_gef
def test_gene_selection_reads_only_that_gene():
    v = g.open_gef(GEF, budget=g.Budget(ram="256MB")).reset().use_bin(10)
    names, offs, cnts = v.gef.gene_index(10)
    assert len(names) == len(offs) == len(cnts)
    assert np.all(np.diff(offs) >= 0)             # offsets are monotonic: slices are contiguous
    top = v.gene_profile(1)[0][0]
    allg = v.select_genes(None).load()
    one = v.select_genes([top]).load()
    assert 0 < one["x"].size < allg["x"].size
    assert set(np.unique(one["gene"])) == {top}
    with pytest.raises(KeyError):
        v.select_genes(["NotAGene"])
    v.gef.close()


@need_gef
def test_window_selection_actually_clips():
    v = g.open_gef(GEF, budget=g.Budget(ram="256MB")).reset().use_bin(20)
    x0, x1, y0, y1 = 1000, 2000, 1200, 2200
    d = v.select(x0, x1, y0, y1).load()
    assert d["x"].size > 0
    assert d["x"].min() >= x0 and d["x"].max() < x1
    assert d["y"].min() >= y0 and d["y"].max() < y1
    v.gef.close()


@need_gef
def test_rasterize_shape_and_mass():
    v = g.open_gef(GEF, budget=g.Budget(ram="256MB")).reset().use_bin(100)
    img = v.image(256)
    assert img.shape == (256, 256)
    d = v.load()
    assert img.sum() == pytest.approx(float(d["count"].sum()), rel=1e-6)
    v.gef.close()


@need_gem
def test_gem_index_and_window():
    gv = g.open_gem(GEM, budget=g.Budget(ram="128MB"))
    assert gv.index.n_rows > 0 and gv.index.n_genes > 0
    x0, x1, y0, y1 = gv.index.extent
    assert x1 > x0 and y1 > y0
    top = gv.index.top_genes(3)
    assert len(top) == 3 and top[0][1] >= top[1][1]
    gv.select_frac(0.4, 0.7, 0.4, 0.7)
    est = gv.cost()["units"]
    t = gv.load()
    assert t.num_rows > 0
    assert abs(t.num_rows - est) / max(est, 1) < 0.25     # occupancy estimate is close
    xs = t.column("x").to_numpy()
    assert xs.min() >= gv.window[0] and xs.max() < gv.window[1]


@need_gef
def test_multimodal_windows_agree():
    if not os.path.exists(IMG):
        pytest.skip("no image fixture")
    v = g.open_gef(GEF)
    side = v.gef.extent[1] - v.gef.extent[0]
    v.gef.close()
    ds = g.open_stereoseq(gef=GEF, image=IMG, budget=g.Budget(ram="256MB"),
                          image_scale=2048 / side)
    pair = ds.paired((1600, 3000, 1400, 2800), shape=256)
    assert "expression" in pair and "morphology" in pair
    def peak(a):
        # Centroid of the smoothed top 1 %. The single brightest pixel of a bin1 raster is
        # Poisson noise anywhere inside the lobe; an earlier argmax version of this test
        # passed by luck and failed when an unrelated binning fix moved which pixel won.
        from scipy.ndimage import gaussian_filter
        sm = gaussian_filter(np.asarray(a, float), sigma=max(a.shape) / 40)
        yy, xx = np.nonzero(sm >= np.percentile(sm, 99))
        w = sm[yy, xx]
        return (yy * w).sum() / w.sum() / a.shape[0], (xx * w).sum() / w.sum() / a.shape[1]
    pe, pm = peak(pair["expression"]), peak(pair["morphology"])
    assert abs(pe[0] - pm[0]) < 0.1 and abs(pe[1] - pm[1]) < 0.1   # registration holds
    ds.close()


@need_gef
def test_budget_is_never_exceeded_on_a_chip():
    cap = 32 * 1024 ** 2
    b = g.Budget(ram="32MB", timeout_s=20)
    v = g.open_gef(GEF, budget=b).reset().auto_bin()
    v.load()
    assert b.stats()["ram"]["peak"] <= cap
    assert b.stats()["ram"]["denials"] == 0
    v.gef.close()
