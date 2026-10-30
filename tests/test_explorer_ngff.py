"""Interactive navigation driven headless, and OME-Zarr export."""
import os
import numpy as np
import pytest
import geminga as g

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")
GEF = os.path.join(FIX, "chip.gef")
TIF = os.path.join(FIX, "tissue.ome.tif")
NII = os.path.join(FIX, "ms", "train", "P01", "T1", "P01_T1_FLAIR.nii")
need = pytest.mark.skipif(not os.path.exists(GEF), reason="run the fixture scripts first")
needs_gpu = pytest.mark.skipif(not g.stage_available(), reason="no wgpu adapter")


def _hot(ex):
    f = ex.full
    wx, wy = f[0] + (f[1] - f[0]) * 0.62, f[2] + (f[3] - f[2]) * 0.66
    cam = ex.stage.camera
    sx, sy = cam.scale
    return ((wx - cam.cx) * sx + 1) / 2 * ex.width, (1 - (wy - cam.cy) * sy) / 2 * ex.height


@needs_gpu
@need
def test_drags_inside_the_margin_cost_no_io():
    ex = g.explore(GEF, budget=g.Budget(ram="96MB"), width=320, height=240, point_budget="8MB")
    ex.frame()
    before = ex.telemetry.reloads
    for _ in range(5):
        ex.handle(g.Event("drag", dx=-6, dy=3))
        ex.frame()
    assert ex.telemetry.reloads == before              # every small drag served from memory
    assert ex.telemetry.served_from_memory > 0.8


@needs_gpu
@need
def test_zoom_in_eventually_reaches_a_finer_rung():
    ex = g.explore(GEF, budget=g.Budget(ram="96MB"), width=320, height=240, point_budget="8MB")
    ex.frame()
    start = ex.loaded_rung
    x, y = _hot(ex)
    for _ in range(12):
        ex.handle(g.Event("scroll", x=x, y=y, steps=1))
        ex.frame()
    assert ex.loaded_rung < start
    assert ex.telemetry.rung_changes >= 1


@needs_gpu
@need
def test_ladder_matches_the_loaded_rung_every_frame():
    ex = g.explore(GEF, width=320, height=240, point_budget="8MB")
    x, y = _hot(ex)
    ex.frame()
    for _ in range(8):
        ex.handle(g.Event("scroll", x=x, y=y, steps=1))
        ex.frame()
        assert ex.view.bin_size == ex.loaded_rung       # the ladder reads view.bin_size


@needs_gpu
@need
def test_budget_holds_across_a_session():
    cap = 32 * 1024 ** 2
    b = g.Budget(ram="32MB")
    ex = g.explore(GEF, budget=b, width=320, height=240, point_budget="8MB")
    x, y = _hot(ex)
    ex.frame()
    for i in range(14):
        ex.handle(g.Event("scroll", x=x, y=y, steps=1) if i < 8 else g.Event("drag", dx=-40))
        ex.frame()
        assert b.stats()["ram"]["used"] <= cap
    assert b.stats()["ram"]["peak"] <= cap and b.stats()["ram"]["denials"] == 0


@needs_gpu
@need
def test_policies_trade_detail_for_io():
    out = {}
    for pol in ("smooth", "balanced", "sharp"):
        ex = g.explore(GEF, width=320, height=240, point_budget="8MB", policy=pol)
        x, y = _hot(ex)
        ex.frame()
        rungs = []
        for i in range(12):
            ex.handle(g.Event("scroll", x=x, y=y, steps=1) if i < 6 else g.Event("drag", dx=-20))
            ex.frame()
            rungs.append(ex.loaded_rung)
        out[pol] = (np.mean(rungs), ex.telemetry.reloads)
    assert out["balanced"][0] <= out["smooth"][0]       # at least as fine
    assert out["sharp"][0] <= out["balanced"][0]
    assert out["sharp"][1] >= out["balanced"][1]        # finer costs reloads
    with pytest.raises(ValueError):
        g.explore(GEF, policy="turbo")


@needs_gpu
@need
def test_bookmarks_and_gene_jump():
    ex = g.explore(GEF, width=320, height=240, point_budget="8MB")
    x, y = _hot(ex)
    ex.frame()
    for _ in range(4):
        ex.handle(g.Event("scroll", x=x, y=y, steps=1))
    ex.bookmark("here")
    here = (ex.stage.camera.cx, ex.stage.camera.cy, ex.stage.camera.span)
    ex.fit(); ex.frame()
    ex.recall("here")
    assert (ex.stage.camera.cx, ex.stage.camera.cy, ex.stage.camera.span) == here
    top = ex.view.gene_profile(1)[0][0]
    ex.gene(top)
    ex.frame()
    assert ex.view.genes == [top]
    ex.gene(None)
    assert ex.view.genes is None


@needs_gpu
@need
def test_camera_is_clamped_to_the_data():
    ex = g.explore(GEF, width=320, height=240)
    for _ in range(20):
        ex.handle(g.Event("scroll", x=160, y=120, steps=-1))       # zoom far out
    f = ex.full
    assert ex.stage.camera.span <= max(f[1] - f[0], f[3] - f[2]) * 1.06
    for _ in range(50):
        ex.handle(g.Event("drag", dx=500, dy=-500))                 # drag far away
    assert f[0] <= ex.stage.camera.cx <= f[1] and f[2] <= ex.stage.camera.cy <= f[3]


@needs_gpu
@need
def test_record_returns_frames():
    ex = g.explore(GEF, width=200, height=160)
    out = ex.record([{"kind": "scroll", "x": 100, "y": 80, "steps": 1},
                     {"kind": "drag", "dx": -5}])
    assert len(out) == 3 and all(f.shape == (160, 200, 4) for _, f in out)


@needs_gpu
def test_explorer_refuses_non_point_views():
    if not os.path.exists(NII):
        pytest.skip("no MRI fixture")
    with pytest.raises(TypeError, match="to_ome_zarr"):
        g.Explorer(g.view(NII))


# ---- OME-Zarr ----
@need
def test_gef_levels_become_the_ngff_pyramid(tmp_path):
    b = g.Budget(ram="256MB")
    r = g.to_ome_zarr(g.open_gef(GEF), str(tmp_path / "c.ome.zarr"), budget=b)
    m = g.read_ome_zarr_meta(str(tmp_path / "c.ome.zarr"))
    bins = [l["bin"] for l in r["levels"]]
    assert bins == sorted(bins)
    scales = [d["coordinateTransformations"][0]["scale"][0] for d in m["multiscales"]["datasets"]]
    assert scales == [bn * 0.5 for bn in bins]                     # 0.5 um per DNB unit
    assert m["multiscales"]["version"] == "0.4"
    assert b.stats()["ram"]["peak"] <= 256 * 1024 ** 2
    small = g.to_ome_zarr(g.open_gef(GEF), str(tmp_path / "s.ome.zarr"), max_side=500)
    assert small["skipped"] and all(s["side"] > 500 for s in small["skipped"])


@need
def test_rasterised_mass_is_preserved(tmp_path):
    import zarr
    gef = g.open_gef(GEF).gef
    g.to_ome_zarr(gef, str(tmp_path / "m.ome.zarr"), max_side=1000)
    grp = zarr.open_group(str(tmp_path / "m.ome.zarr"), mode="r")
    ms = grp.attrs["multiscales"][0]
    totals = [float(np.asarray(grp[d["path"]][:]).sum()) for d in ms["datasets"]]
    assert max(totals) - min(totals) < 1e-3 * max(totals)          # every level sums the same


def test_image_pyramid_respects_the_band_ceiling(tmp_path):
    if not os.path.exists(TIF):
        pytest.skip("no image fixture")
    t = g.TiffTileArray(TIF)
    for band in ("4MB", "16MB"):
        r = g.to_ome_zarr(t, str(tmp_path / f"t{band}.ome.zarr"), levels=4, band=band)
        assert r["peak_band_bytes"] <= g.parse_size(band)
    import zarr
    a0 = np.asarray(zarr.open_group(str(tmp_path / "t4MB.ome.zarr"), mode="r")["0"][:256, :256])
    assert np.array_equal(a0, t[0:256, 0:256])


def test_volume_export_keeps_voxel_size(tmp_path):
    if not os.path.exists(NII):
        pytest.skip("no MRI fixture")
    g.to_ome_zarr(g.NiftiVolume(NII), str(tmp_path / "v.ome.zarr"), levels=3)
    m = g.read_ome_zarr_meta(str(tmp_path / "v.ome.zarr"))
    assert [a["name"] for a in m["multiscales"]["axes"]] == ["z", "y", "x"]
    assert m["multiscales"]["datasets"][0]["coordinateTransformations"][0]["scale"] == [3.0, 1.0, 1.0]
    assert m["shapes"][1] == (24, 48, 48)


def test_independent_reader_accepts_the_export(tmp_path):
    pytest.importorskip("ome_zarr")
    if not os.path.exists(TIF):
        pytest.skip("no image fixture")
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader
    g.to_ome_zarr(g.TiffTileArray(TIF), str(tmp_path / "r.ome.zarr"), levels=3)
    nodes = list(Reader(parse_url(str(tmp_path / "r.ome.zarr")))())
    assert len(nodes[0].data) == 3
