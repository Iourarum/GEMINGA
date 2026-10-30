"""Selections: marked regions, their measurements, similar-region search, and leads."""
import json
import os
import numpy as np
import pytest
import geminga as g

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")
GBM = os.path.join(FIX, "gbm", "gbm_phantom.gef")
TS = os.path.join(FIX, "culture.parquet")
FQ = os.path.join(FIX, "reads.fastq")
need_gbm = pytest.mark.skipif(not os.path.exists(GBM), reason="run make_glioma_phantom.py")
need_ts = pytest.mark.skipif(not os.path.exists(TS), reason="run make_culture_timeseries.py")


def _rim(v):
    reg = np.load(os.path.join(os.path.dirname(GBM), "regions.npy"))
    cx, cy, rn, rr = reg[:4]
    r, s = (rn + rr) / 2 * 1.02, 110
    return (cx + r - s, cx + r + s, cy - s, cy + s), (cx, cy, rn, rr)


def test_geometry_and_contains():
    s = g.Selection("a", "r", {}, "rect", {"x0": 0, "x1": 10, "y0": 0, "y1": 5})
    assert s.area == 50 and list(s.contains([1, 11], [1, 1])) == [True, False]
    p = g.Selection("b", "p", {}, "polygon", {"points": [[0, 0], [10, 0], [0, 10]]})
    assert p.area == pytest.approx(50)
    assert list(p.contains([1, 9], [1, 9])) == [True, False]
    t = g.Selection("c", "r2", {}, "rect", {"x0": 5, "x1": 15, "y0": 0, "y1": 5})
    assert s.overlaps(t) == pytest.approx(0.5)
    f = p.to_feature()
    assert f["geometry"]["coordinates"][0][0] == f["geometry"]["coordinates"][0][-1]


def test_book_is_append_only_and_folds(tmp_path):
    b = g.selections(str(tmp_path / "s.jsonl"))
    s1 = g.Selection("id1", "one", {"key": "k", "path": "/x"}, "interval", {"lo": 0, "hi": 5})
    s2 = g.Selection("id2", "two", {"key": "k", "path": "/x"}, "interval", {"lo": 5, "hi": 9},
                     intent="need-data")
    b.add(s1); b.add(s2)
    b.update("id1", note="looked at it", status="resolved")
    b.delete("id2")
    lines = open(b.path).read().splitlines()
    assert len(lines) == 4 and all(json.loads(l) for l in lines)       # nothing rewritten
    cur = b.all()
    assert set(cur) == {"id1"} and cur["id1"].note == "looked at it"
    with open(b.path, "a") as fh:
        fh.write('{"event": "add", "id": "torn"')                        # a crash mid-write
    assert set(b.all()) == {"id1"}
    b.add(g.Selection("id3", "three", {"key": "k", "path": "/x"}, "interval", {"lo": 9, "hi": 10}))
    assert set(g.selections(b.path).all()) == {"id1", "id3"}            # the next write is not glued on
    with pytest.raises(KeyError):
        b.update("nope", note="x")


@need_gbm
def test_selection_reattaches_by_fingerprint(tmp_path):
    b = g.selections(str(tmp_path / "s.jsonl"))
    v = g.open_gef(GBM)
    g.select(v, "here", (100, 200, 100, 200), book=b)
    assert len(b.find(dataset=g.open_gef(GBM))) == 1               # same file, new session
    assert b.find(dataset={"key": "someone-else"}) == []


@need_gbm
def test_describe_finds_the_hypoxia_programme():
    v = g.open_gef(GBM).use_bin(20)
    box, _ = _rim(v)
    sel = g.select(v, "rim", box)
    m = g.describe(sel, v)
    top = {e["gene"] for e in m["enriched"][:8]}
    assert {"VEGFA", "CA9"} <= top
    assert all(e["log2fc"] > 0 and e["observed"] > e["expected"] for e in m["enriched"])
    dep = {e["gene"] for e in m["depleted"][:6]}
    assert dep & {"SNAP25", "SYT1", "RBFOX3", "GAD1", "MBP", "PLP1"}
    assert all(abs(e["log2fc"]) < 14 for e in m["depleted"])       # no pseudocount blow-ups
    assert sel.status == "described"


@need_gbm
def test_polygon_filters_inside_its_bounding_box():
    v = g.open_gef(GBM).use_bin(20)
    box, _ = _rim(v)
    x0, x1, y0, y1 = box
    rect = g.select(v, "r", box)
    tri = g.select_polygon(v, "t", [(x0, y0), (x1, y0), (x0, y1)])
    a, b = g.describe(rect, v), g.describe(tri, v)
    assert 0 < b["observations"] < a["observations"]


@need_gbm
def test_similar_regions_lie_on_the_same_ring():
    v = g.open_gef(GBM).use_bin(20)
    box, (cx, cy, rn, rr) = _rim(v)
    sel = g.select(v, "rim", box)
    c = g.similar(sel, v, k=6)
    assert len(c) >= 4
    assert all(x.parent == sel.id and x.intent == "candidate" for x in c)
    radii = [np.hypot((x.bounds[0] + x.bounds[1]) / 2 - cx, (x.bounds[2] + x.bounds[3]) / 2 - cy)
             for x in c]
    assert all(rn * 0.8 < r < rr * 1.25 for r in radii)            # the rim, not the tumour
    angles = [np.degrees(np.arctan2((x.bounds[2] + x.bounds[3]) / 2 - cy,
                                    (x.bounds[0] + x.bounds[1]) / 2 - cx)) for x in c]
    assert max(angles) - min(angles) > 120                         # spread around it
    for i, a in enumerate(c):                                      # distinct places
        for b in c[i + 1:]:
            assert a.overlaps(b) <= 0.3


@need_ts
def test_time_range_describe_and_similar_windows():
    ev = np.load(TS.replace(".parquet", "_events.npy"))
    d0 = ev[1]
    sel = g.select_values(TS, "t", d0 * 86400, (d0 + 0.25) * 86400, "hyp")
    m = g.describe(sel)
    assert m["series"]["o2_percent"]["z"] < -1.5
    c = g.similar(sel, k=4)
    hits = [x for x in c if any(abs(x.geometry["lo"] / 86400 - e) < 0.15 for e in ev if e != d0)]
    assert len(hits) == 3                                          # all three siblings
    assert min(h.score for h in hits) > max(x.score for x in c if x not in hits) + 0.2


def test_interval_on_a_record_view():
    if not os.path.exists(FQ):
        pytest.skip("no FASTQ fixture")
    v = g.view(FQ)
    n = v.extent.n
    sel = g.select(v, "chunk", (n * 0.4, n * 0.5))
    m = g.describe(sel, v)
    assert m["fraction"] == pytest.approx(0.1, abs=0.01) and m["series"]
    c = g.similar(sel, v, k=3)
    assert c and all(x.kind == "interval" for x in c)


@need_gbm
def test_leads_record_what_was_checked(tmp_path):
    v = g.open_gef(GBM).use_bin(20)
    box, _ = _rim(v)
    sel = g.select(v, "rim", box)
    g.describe(sel, v)
    other = os.path.join(FIX, "chip.gef")
    L = g.leads(sel, context=["glioblastoma"], organism="Homo sapiens",
                local=[other, GBM] if os.path.exists(other) else [GBM])
    lc = sel.summary["leads_checked"]
    assert lc["local_checked"] >= (1 if os.path.exists(other) else 0)
    assert GBM not in [l.get("url") for l in L if l["kind"] == "local"]   # not itself
    q = [l for l in L if l["kind"] == "query"]
    assert {x["source"] for x in q} == {"GEO", "PubMed", "ENA"}
    assert all(x["url"].startswith("https://") for x in q)
    assert "VEGFA" in q[0]["url"] and sel.status == "searched"


def test_local_leads_rank_by_coverage(tmp_path):
    if not os.path.exists(GBM):
        pytest.skip("no phantom")
    L = g.local_leads(["VEGFA", "CA9", "NOTAGENE"], [GBM])
    assert L and L[0]["coverage"] == pytest.approx(2 / 3, abs=1e-3)
    assert set(L[0]["genes_present"]) == {"VEGFA", "CA9"}


def test_query_results_parse_without_a_network():
    from geminga.selections import parse_query_result
    geo = parse_query_result({"source": "GEO", "db": "gds"},
                             {"esearchresult": {"idlist": ["200123", "200456"]}},
                             fetch_summaries=False)
    assert [l["id"] for l in geo] == ["200123", "200456"]
    ena = parse_query_result({"source": "ENA"},
                             [{"study_accession": "PRJEB1", "study_title": "GBM hypoxia"}])
    assert ena[0]["accession"] == "PRJEB1" and ena[0]["url"].endswith("PRJEB1")


@need_gbm
def test_dispatch_and_geojson_for_prism(tmp_path):
    b = g.selections(str(tmp_path / "s.jsonl"))
    v = g.open_gef(GBM).use_bin(20)
    box, _ = _rim(v)
    sel = g.select(v, "rim", box, intent="need-data", book=b)
    g.describe(sel, v, book=b)
    doc = json.load(open(b.write_dispatch(str(tmp_path / "d.json"))))
    ds = next(iter(doc["datasets"].values()))
    assert ds["selections"][0]["top_genes"][:1] and doc["schema"] == "geminga.selections/1"
    gj = b.to_geojson()
    assert gj["type"] == "FeatureCollection" and len(gj["features"]) == 1
    assert b.open_questions()[0].id == sel.id


@pytest.mark.skipif(not g.stage_available(), reason="no wgpu adapter")
@need_gbm
def test_explorer_marks_and_draws_selections(tmp_path):
    b = g.selections(str(tmp_path / "s.jsonl"))
    ex = g.explore(GBM, width=320, height=240, point_budget="4MB", selections=b)
    box, _ = _rim(ex.view)
    sel = ex.select_view("rim", intent="need-data", bounds=box)
    ex.describe_active()
    c = ex.more_like_this(k=3)
    ex.goto((ex.full[0], ex.full[1], ex.full[2], ex.full[3]))
    ex.frame()
    assert len(ex.selections()) == 1 + len(c)
    assert len(b.find(dataset=ex.view)) == 1 + len(c)             # persisted as they were made
    n_rects = ex.stage.stats()["rects"]
    ex.frame()
    ex.goto_selection(sel)
    assert ex.active is sel
    ex.handle(g.Event("key", key="n"))                              # step to the next one
    assert ex.active is not None and ex.active.id != sel.id
    x0, x1, y0, y1 = ex.active.bounds
    vx0, vx1, vy0, vy1 = ex.visible()
    assert vx0 <= x0 and x1 <= vx1 and vy0 <= y0 and y1 <= vy1      # and the camera went there
    before = len(ex.selections())
    ex.handle(g.Event("key", key="s"))                              # mark from the keyboard
    assert len(ex.selections()) == before + 1
    assert n_rects > 20
