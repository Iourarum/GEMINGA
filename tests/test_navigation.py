"""v0.2: zoom, cost-before-load, and search that skips what it can prove it does not need."""
import os
import numpy as np
import pytest
import geminga as g

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")
FQ = os.path.join(FIX, "reads_adapters.fastq")
ADAPTER = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"
need = pytest.mark.skipif(not os.path.exists(FQ), reason="run examples/make_adapter_fastq.py first")


@need
def test_index_is_tiny_and_reusable():
    idx = g.index_fastx(FQ)
    sc = FQ + ".gemidx.json"
    assert idx.n_records > 0
    assert os.path.getsize(sc) < idx.file_bytes * 0.01      # sidecar under 1% of the file
    assert g.index_fastx(FQ).n_records == idx.n_records     # reused, not rebuilt


@need
def test_zoom_is_metadata_only_and_instant():
    import time
    idx = g.index_fastx(FQ)
    t0 = time.perf_counter()
    for target in (8, 64, 512, 4096):
        b = idx.zoom(0, None, target)
        assert 0 < len(b) <= max(target, idx.n_buckets)
        assert b[0]["start"] == 0
    assert time.perf_counter() - t0 < 1.0                   # no file access at any level


@need
def test_cost_is_reported_before_any_load():
    v = g.view(FQ, budget=g.Budget(ram="256MB")).reset()
    c = v.cost()
    assert c["units"] == v.extent.n
    assert c["fraction_of_dataset"] == pytest.approx(1.0)
    v.select_frac(0.5, 0.6)
    c2 = v.cost()
    assert c2["total_bytes"] < c["total_bytes"] / 5
    assert c2["fraction_of_dataset"] == pytest.approx(0.1, abs=0.01)


@need
def test_zoom_keeps_cost_flat_and_distribution_honest():
    v = g.view(FQ, budget=g.Budget(ram="256MB")).reset().set_zoom(4)
    first = v.cost()["total_bytes"]
    for _ in range(2):
        v.zoom_in(4, center=v.extent.n * 0.6)
        assert v.cost()["total_bytes"] == pytest.approx(first, rel=0.05)
    v.reset()
    means = []
    for z in (0, 3, 5):
        v.set_zoom(z)
        t = v.load()
        seqs = t.column("seq").to_pylist()
        means.append(np.mean([(s.count("G") + s.count("C")) / len(s) for s in seqs]))
    assert max(means) - min(means) < 0.01                   # decimation preserves the shape


@need
def test_does_not_fit_raises_with_a_usable_fix():
    v = g.view(FQ, budget=g.Budget(ram="8MB")).reset()
    c = v.cost()
    assert c["fits_whole"] is False and c["fits"] is True
    with pytest.raises(g.CostError, match="zoom out"):
        v.load()
    rows = sum(t.num_rows for t in v.chunks())              # streaming still works
    assert rows == v.extent.n


@need
def test_auto_zoom_lands_inside_the_budget():
    for cap in ("64MB", "8MB", "1MB"):
        v = g.view(FQ, budget=g.Budget(ram=cap)).reset().auto_zoom()
        c = v.cost()
        assert c["fits_whole"] is True
        assert c["total_bytes"] <= g.parse_size(cap)


@need
def test_search_finds_the_seeded_band_and_nothing_outside():
    v = g.view(FQ, budget=g.Budget(ram="128MB")).reset()
    hits = v.search([ADAPTER[:20]], max_hits=500_000)
    assert hits["n_hits"] > 0 and not hits["truncated"]
    rec = np.array([h["record"] for h in hits["hits"]])
    lo, hi = int(rec.min()), int(rec.max())
    assert v.select(0, lo).search([ADAPTER[:20]])["n_hits"] == 0
    assert v.select(hi + 1, v.extent.n).search([ADAPTER[:20]])["n_hits"] == 0


@need
def test_revcomp_and_both_strands():
    assert g.revcomp("AGATCGGAAGAGC") == "GCTCTTCCGATCT"
    v = g.view(FQ, budget=g.Budget(ram="128MB")).reset()
    fwd = v.search([ADAPTER[:20]], both_strands=False)["n_hits"]
    both = v.search([ADAPTER[:20]], both_strands=True)["n_hits"]
    assert both >= fwd


def test_pruning_never_loses_rows_and_pays_off_when_clustered(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = 400_000
    x = np.arange(n) % 20_000
    t = pa.table({"x": x.astype(np.int32), "y": (x * 7 % 19_000).astype(np.int32)})
    unsorted_p = tmp_path / "u.parquet"
    sorted_p = tmp_path / "s.parquet"
    pq.write_table(t, unsorted_p, row_group_size=20_000, write_statistics=True)
    pq.write_table(t.sort_by("x"), sorted_p, row_group_size=20_000, write_statistics=True)

    lo, hi = 5_000, 5_200
    exact = int(((x >= lo) & (x <= hi)).sum())
    for path, expect_gain in ((unsorted_p, False), (sorted_p, True)):
        ds = g.open(str(path))
        p = ds.prune("x", lo, hi, ["x", "y"])
        assert p["candidate_rows"] >= exact           # pruning is never allowed to lose rows
        batches = ds.search("x", lo, hi, ["x", "y"])
        got = sum(bt.num_rows for bt in batches)
        assert got == exact                           # pruning picks row groups; search returns rows
        xs = np.concatenate([bt.column("x").to_numpy() for bt in batches])
        assert xs.min() >= lo and xs.max() <= hi
        if expect_gain:
            assert p["selectivity"] < 0.5             # clustering is what makes it pay
            assert p["candidate_rows"] > exact        # and the candidate set was wider than the answer
    with pytest.raises(ValueError, match="no column"):
        g.open(str(sorted_p)).prune("nope", 0, 1)


def test_search_is_exact_on_floats_and_skips_nulls(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    y = np.arange(1000, dtype=np.float64) * 0.5
    t = pa.table({"y": pa.array([None if i % 100 == 0 else v for i, v in enumerate(y)], pa.float64()),
                  "s": pa.array([str(i) for i in range(1000)])})
    path = tmp_path / "f.parquet"
    pq.write_table(t, path, row_group_size=100)
    ds = g.open(str(path))
    out = ds.search("y", 10.25, 12.0, ["s"])
    ys = np.concatenate([bt.column("y").to_numpy() for bt in out])
    assert ys.tolist() == [10.5, 11.0, 11.5, 12.0]
    assert set(out[0].schema.names) == {"s", "y"}     # the searched column is projected in
    assert sum(bt.num_rows for bt in ds.search("y", -1, 1e9)) == 990   # nulls never match
    with pytest.raises(ValueError, match="numeric"):
        ds.search("s", 0, 1)


def test_progressive_canvas_is_bounded_and_accurate():
    c = g.ProgressiveCanvas((0, 100, 0, 100), shape=(64, 64))
    before = c.nbytes
    rng = np.random.default_rng(0)
    total = 0
    for _ in range(20):
        xs, ys = rng.uniform(0, 100, 5000), rng.uniform(0, 100, 5000)
        c.add(xs, ys)
        total += 5000
    assert c.nbytes == before                          # O(grid), not O(rows)
    assert c.n_rows == total
    assert c.image.sum() == total
    assert 0.9 < c.coverage <= 1.0


@need
def test_overview_series_and_sparkline():
    v = g.view(FQ, budget=g.Budget(ram="64MB")).reset()
    ov = v.overview(50)
    assert ov.cheap and "gc" in ov.series and len(ov) <= 50
    assert len(ov.sparkline("gc")) > 0
