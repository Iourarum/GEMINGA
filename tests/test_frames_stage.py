"""Polars conversion and the wgpu stage."""
import os
import numpy as np
import pytest
import geminga as g

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")
GEF = os.path.join(FIX, "chip.gef")
XEN = os.path.join(FIX, "glioma", "xenium")
H5 = os.path.join(FIX, "glioma", "filtered_feature_bc_matrix.h5")
pl = pytest.importorskip("polars")
need = pytest.mark.skipif(not os.path.exists(GEF), reason="run the fixture scripts first")
needs_gpu = pytest.mark.skipif(not g.stage_available(), reason="no wgpu adapter")


# ---- polars ----
def test_to_polars_from_every_shape():
    import pyarrow as pa
    import scipy.sparse as sp
    t = pa.table({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    assert g.to_polars(t).shape == (3, 2)
    assert g.to_polars(t.to_batches()[0]).shape == (3, 2)
    assert g.to_polars({"x": np.arange(5), "y": np.arange(5.0)}).shape == (5, 2)
    d = g.to_polars(sp.random(6, 4, density=0.5, format="csr", random_state=0))
    assert d.columns == ["row", "col", "value"] and d.height > 0
    assert g.to_polars(np.arange(4)).shape == (4, 1)
    assert g.to_polars(None).height == 0
    with pytest.raises(TypeError):
        g.to_polars(np.zeros((2, 2, 2)))


def test_arrow_roundtrip_preserves_schema_by_default():
    import pyarrow as pa
    t = pa.table({"g": ["a", "b"], "n": [1, 2]})
    back = g.from_polars(g.to_polars(t))
    assert back.equals(t)                      # compat=True narrows large_string -> string
    wide = g.from_polars(g.to_polars(t), compat=False)
    assert wide.schema.field("g").type == pa.large_string()
    assert wide.to_pydict() == t.to_pydict()   # only the width differs, never the data
    nd = g.from_polars(g.to_polars(t), to="numpy")
    assert set(nd) == {"g", "n"}
    with pytest.raises(ValueError):
        g.from_polars(g.to_polars(t), to="feather")


def test_frame_bytes_tracks_size():
    small = g.to_polars({"x": np.arange(100, dtype=np.int32)})
    big = g.to_polars({"x": np.arange(100_000, dtype=np.int32)})
    assert g.frame_bytes(big) > g.frame_bytes(small) * 100


@need
def test_view_polars_and_budget():
    b = g.Budget(ram="128MB")
    v = g.open_gef(GEF, budget=b).use_bin(20).select(1000, 2200, 1000, 2200)
    df = v.polars()
    assert set(df.columns) >= {"x", "y", "count"}
    assert df["x"].min() >= 1000 and df["x"].max() < 2200
    assert df.height == v.load()["x"].size
    v.gef.close()


@need
def test_lazy_window_prunes_then_collects():
    xb = g.open_xenium(XEN)
    lf, plan = xb.lazy_window(1000, 1400, 500, 900,
                              columns=["feature_name", "x_location", "y_location"],
                              explain=True)
    assert plan["row_groups_kept"] < plan["row_groups_total"]
    out = lf.collect()
    assert out.height > 0
    x = out["x_location"].to_numpy(); y = out["y_location"].to_numpy()
    assert x.min() >= 1000 and x.max() < 1400 and y.min() >= 500 and y.max() < 900
    # the lazy path and the GEMINGA path must agree exactly
    assert out.height == xb.window(1000, 1400, 500, 900).num_rows


@need
def test_sink_parquet_streams(tmp_path):
    xb = g.open_xenium(XEN)
    lf = g.scan(xb.transcripts).filter(pl.col("x_location") < 500)
    p = g.sink_parquet(lf, str(tmp_path / "out.parquet"))
    assert os.path.exists(p)
    assert pl.read_parquet(p).height == lf.collect().height


def test_catalog_polars(tmp_path):
    cat = g.catalog(str(tmp_path / "c.jsonl"))
    cat.record("a", "/x.gef", note="one")
    cat.record("b", "/y.h5ad", note="two")
    df = cat.polars()
    assert df.height == 2 and "key" in df.columns


# ---- stage ----
@needs_gpu
def test_stage_renders_points_and_text(tmp_path):
    st = g.Stage(320, 240)
    rng = np.random.default_rng(0)
    st.camera = g.Camera.fit((0, 100, 0, 100), 320, 240)
    st.points(rng.uniform(0, 100, 5000), rng.uniform(0, 100, 5000), rng.random(5000))
    st.label("hello", 10, 10)
    img = st.render()
    assert img.shape == (240, 320, 4)
    assert st.draw_calls == 2
    assert st.stats()["glyphs"] == 5
    # something was actually drawn over the background
    bg = np.array([254, 252, 243])
    assert (np.abs(img[:, :, :3].astype(int) - bg).sum(2) > 30).mean() > 0.05
    p = st.render_png(str(tmp_path / "s.png"))
    assert os.path.getsize(p) > 1000


@needs_gpu
def test_camera_fit_zoom_pan():
    c = g.Camera.fit((0, 100, 0, 100), 400, 400)
    assert c.cx == 50 and c.cy == 50
    x0, x1, y0, y1 = c.visible
    assert x0 < 0 < 100 < x1 or (x1 - x0) >= 100
    before = c.span
    c.zoom(2)
    assert c.span == pytest.approx(before / 2)
    c.zoom(2, about=(0, 0))
    assert c.cx < 50
    cx = c.cx; c.pan(0.5, 0)
    assert c.cx > cx


@needs_gpu
def test_text_atlas_layout():
    a = g.TextAtlas(14)
    assert a.image.ndim == 2 and a.atlas_w > 100
    lay = a.layout("AB", 0, 0)
    assert len(lay) == 2
    (r0, uv0), (r1, uv1) = lay
    assert r1[0] > r0[0]                     # advances rightward
    for uv in (uv0, uv1):
        assert 0 <= uv[0] < uv[2] <= 1 and 0 <= uv[1] < uv[3] <= 1
    assert a.layout(" ", 0, 0) == []          # space emits no quad
    assert a.measure("iii") < a.measure("MMM")


@needs_gpu
@need
def test_show_picks_lod_from_the_cost_model():
    b = g.Budget(ram="256MB")
    got = {}
    for cap in ("2MB", "64MB"):
        st = g.Stage(320, 240, point_budget=cap)
        v = g.open_gef(GEF, budget=b).reset()
        ch = st.show(v)
        got[cap] = ch
        assert ch["points"] <= ch["cap"]
        assert ch["estimate"] <= ch["cap"]
        v.gef.close()
    assert got["2MB"]["bin_size"] > got["64MB"]["bin_size"]     # tighter budget, coarser bin
    assert got["2MB"]["points"] < got["64MB"]["points"]


@needs_gpu
def test_point_cap_decimates_rather_than_failing():
    st = g.Stage(200, 200)
    st.MAX_POINTS = 1000
    st.points(np.arange(10_000, dtype=np.float32), np.arange(10_000, dtype=np.float32))
    assert st.stats()["points"] <= 1000


@needs_gpu
@need
def test_hud_draws_and_reserves():
    b = g.Budget(ram="64MB", vram="128MB")
    st = g.Stage(420, 320, point_budget="4MB")
    v = g.open_gef(GEF, budget=b).reset()
    ex = v.gef.extent
    st.show(v, window=(ex[0], (ex[0] + ex[1]) // 2, ex[2], (ex[2] + ex[3]) // 2))
    st.hud(v, budget=b)
    s = st.stats()
    assert s["rects"] > 20 and s["glyphs"] > 30
    assert st.draw_calls == 3
    assert st._reserved                              # panels claimed screen space
    # no tick label may sit inside a reserved box
    for lb in st._labels:
        if lb.text.replace(",", "").replace("-", "").isdigit():
            assert st._free(lb.x, lb.y, st.atlas.measure(lb.text, 0.85), st.atlas.size)
    img = st.render()
    assert img.shape == (320, 420, 4)
    v.gef.close()


@needs_gpu
def test_outline_is_four_rects():
    st = g.Stage(64, 64)
    st.outline(2, 2, 20, 20)
    assert len(st._rects) == 4
    st.clear_rects()
    assert not st._rects and not st._reserved


@needs_gpu
@need
def test_workspace_panel_shows_every_cell():
    ws = g.workspace(ram="128MB", name="t")
    ws.add("a", GEF, ram="48MB")
    ws.add("b", GEF, ram="32MB")
    st = g.Stage(400, 300)
    st.workspace_panel(ws)
    names = {l.text for l in st._labels}
    assert "a" in names and "b" in names
    assert any("128.0 MB" in l.text for l in st._labels)
    ws.close()


def test_aggregate_sums_duplicate_locations():
    x = np.array([5, 5, 5, 9], dtype=np.int32)
    y = np.array([2, 2, 2, 4], dtype=np.int32)
    v = np.array([1, 2, 3, 10])
    ax_, ay, av = g.Stage.aggregate(x, y, v)
    got = dict(zip(zip(ax_.tolist(), ay.tolist()), av.tolist()))
    assert got == {(5, 2): 6.0, (9, 4): 10.0}
    fx = np.array([0.5, 0.5]); fy = np.array([1.5, 1.5])      # float coords: left alone
    assert g.Stage.aggregate(fx, fy, v[:2])[0] is fx


@needs_gpu
def test_min_point_size_is_in_the_uniform():
    st = g.Stage(400, 300, min_point_px=2.0)
    st.points(np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32))
    st._upload_uniforms()
    assert st.min_point_px == 2.0


@need
def test_rasterize_uses_the_last_row_and_column():
    v = g.open_gef(GEF)
    ex = v.gef.extent
    img = v.gef.rasterize(ex, 50, None, (ex[1] - ex[0]) // 50)
    assert img[:, -1].sum() > 0 and img[-1, :].sum() > 0          # no permanently empty edge
    v.gef.close()
