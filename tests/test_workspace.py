"""Cells partition the budget, contain failures, and never over-commit the workspace."""
import os
import pytest
import geminga as g

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")
GEF = os.path.join(FIX, "chip.gef")
FQ = os.path.join(FIX, "reads.fastq")
need = pytest.mark.skipif(not os.path.exists(GEF), reason="run the fixture scripts first")


@need
def test_cells_reserve_and_cannot_overcommit():
    ws = g.workspace(ram="100MB", timeout_s=5)               # a shortfall must not wait this out
    ws.add("a", GEF, ram="60MB")
    ws.add("b", GEF, ram="30MB")
    assert ws.total_reserved("ram") == 90 * 1024 ** 2
    with pytest.raises(MemoryError, match="cannot reserve"):
        ws.add("c", GEF, ram="40MB")
    ws.close()


@need
def test_suspend_returns_the_slice_and_keeps_position():
    ws = g.workspace(ram="100MB")
    ws.add("a", GEF, ram="60MB")
    win = ws["a"].view.select(1000, 2000, 1000, 2000).window
    free = ws.budget.available("ram")
    ws["a"].suspend(release_reservation=True)
    assert ws.budget.available("ram") > free
    ws["a"].resume()
    assert ws["a"].view.window == win
    ws.close()


@need
def test_failure_is_contained():
    ws = g.workspace(ram="100MB")
    ws.add("good", GEF, ram="40MB")
    _ = ws["good"].view
    ws.add("bad", os.path.join(FIX, "definitely-missing.gef"), ram="10MB")
    with pytest.raises((OSError, ValueError, KeyError)):
        _ = ws["bad"].view
    assert ws["bad"].state == g.CellState.FAILED
    assert ws["good"].state == g.CellState.OPEN
    assert ws["bad"].report()["last_error"]
    ws.close()


@need
def test_select_frac_broadcasts_across_dimensionality():
    ws = g.workspace(ram="200MB")
    ws.add("chip", GEF, ram="80MB")
    if os.path.exists(FQ):
        ws.add("reads", FQ, ram="60MB")
    out = ws.select_frac(0.4, 0.6)
    assert set(out) <= set(ws.cells)
    assert not getattr(ws, "errors", {})
    for c in out.values():
        assert c["units"] > 0
    ws.close()


@need
def test_isolated_run_returns_and_enforces():
    ws = g.workspace(ram="200MB")
    ws.add("chip", GEF, ram="60MB")
    ws["chip"].view.select(1000, 2000, 1000, 2000)
    n = ws["chip"].run_isolated(lambda v: int(v.load()["x"].size))
    assert n > 0
    with pytest.raises(RuntimeError, match="isolated run"):      # killed by the allowance
        ws["chip"].run_isolated(lambda v: [bytearray(200 * 1024 ** 2) for _ in range(8)])
    ws.close()


@need
def test_profile_ranks_and_admits_uncertainty():
    p = g.profile(GEF, fetch_page=False)
    assert p.format == "Stereo-seq GEF"
    assert p.best.label == "Stereo-seq" and p.best.probability > 0.8
    probs = [h.probability for h in p.assays]
    assert probs == sorted(probs, reverse=True)
    assert abs(sum(probs) - 1.0) < 1e-6

    h5 = os.path.join(FIX, "sc.h5ad")
    if os.path.exists(h5):
        q = g.profile(h5, fetch_page=False)
        labels = [h.label for h in q.assays[:3]]
        # a container cannot settle sc versus sn, and the profile must not pretend it did
        assert "scRNA-seq" in labels and "snRNA-seq" in labels
        assert not q.confident
        assert "intronic" in q.best.distinguish.lower()


def test_confidence_requires_discriminating_evidence():
    p = g.profile("/nowhere/thing.h5ad", fetch_page=False)
    assert not p.confident
    assert p.discriminating_evidence == []


def test_catalog_roundtrip(tmp_path):
    cat = g.catalog(str(tmp_path / "cat.jsonl"))
    cat.record("k1", "/data/a.gef", note="first")
    cat.note_streamed("k1", 1000)
    cat.note_streamed("k1", 2000)
    cat.annotate("k1", tags=["x"], molecule="RNA")
    e = cat.get("k1")
    assert e.bytes_streamed == 3000 and e.tags == ["x"] and e.molecule == "RNA"
    assert len(cat.find("a.gef")) == 1
    assert cat.get("missing") is None


def test_assistant_always_answers_offline():
    ai = g.Assistant(backend="template")
    assert not ai.available
    r = ai.explain_protocol("snRNA-seq")
    assert r.backend == "template" and r.grounded
    assert 80 < len(r.text.split()) < 400
    assert "intron" in r.text.lower()          # the point of the sn note
    assert r.sources
