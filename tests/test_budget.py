"""The invariant that matters: peak resident bytes never exceed the declared budget."""
import os
import numpy as np
import pytest
import geminga as g

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")
need = pytest.mark.skipif(not os.path.isdir(FIX), reason="run examples/make_omics_fixtures.py first")


def test_parse_and_format_sizes():
    assert g.parse_size("1KB") == 1024
    assert g.parse_size("1.5GiB") == 1610612736
    assert g.parse_size("4096") == 4096
    assert "GB" in g.human(3 * 1024 ** 3)


def test_tier_isolation_and_release():
    b = g.Budget(ram="10MB", vram="1GB")
    with b.acquire("ram", "8MB"):
        assert b.available("ram") == 2 * 1024 ** 2
        assert b.available("vram") == 1024 ** 3      # tiers are independent
    assert b.available("ram") == 10 * 1024 ** 2      # released on exit


def test_impossible_request_fails_fast():
    b = g.Budget(ram="1MB")
    with pytest.raises(MemoryError, match="cannot fit"):
        b.acquire("ram", "2MB")


def test_unknown_tier():
    with pytest.raises(ValueError):
        g.Budget(ram="1MB").acquire("nvme", "1KB")


@need
@pytest.mark.parametrize("name,kw", [
    ("reads.fastq", {"chunk_bytes": "8MB"}),
    ("reads.fastq.gz", {"chunk_bytes": "8MB"}),
    ("sc.h5ad", {"chunk_bytes": "8MB"}),
    ("qc.arrow", {}),
    ("tissue.ome.tif", {"chunk_bytes": "4MB"}),
    ("glioma/codex_6plex.ome.tif", {}),                 # six channels: every tile counts all six
    ("glioma/mtx", {}),
])
def test_peak_never_exceeds_budget(name, kw):
    cap = 64 * 1024 ** 2
    b = g.Budget(ram="64MB", timeout_s=20)
    r = g.open(os.path.join(FIX, name), budget=b, **kw)
    n = 0
    for _ in r:
        n += 1
        assert b.stats()["ram"]["used"] <= cap
    assert n > 0
    assert b.stats()["ram"]["peak"] <= cap
    assert b.stats()["ram"]["denials"] == 0
    if hasattr(r, "close"):
        r.close()


@need
def test_plan_is_free_and_honest():
    r = g.open(os.path.join(FIX, "sc.h5ad"), chunk_bytes="8MB")
    p = r.plan()
    assert p.n_chunks > 0 and p.bytes_per_chunk > 0
    assert p.detail["n_obs"] == 20_000
    sizes = [c["resident_bytes"] for c in r]
    seen = sum(c["X"].shape[0] for c in r)
    assert seen == p.detail["n_obs"]
    # the plan names the largest chunk that was really handed out
    assert p.n_chunks == len(sizes) and p.bytes_per_chunk == max(sizes) == p.detail["largest_chunk_bytes"]
    assert max(sizes) <= 8 * 1024 ** 2 and p.detail["fits"]
    r.close()


@need
def test_plan_total_counts_every_channel():
    cap = 4 * 1024 ** 2
    b = g.Budget(ram="4MB", timeout_s=20)
    r = g.open(os.path.join(FIX, "glioma", "codex_6plex.ome.tif"), budget=b)
    p = r.plan()
    assert p.total_bytes == 6 * 1500 * 1500 * 2 and p.bytes_per_chunk <= r.chunk_bytes <= cap
    big = max(t.nbytes for t, _ in r)
    assert big == p.bytes_per_chunk and b.stats()["ram"]["peak"] <= cap
    r.close()


@need
def test_explicit_chunk_larger_than_the_budget_fails_before_reading():
    b = g.Budget(ram="4MB")
    with pytest.raises(MemoryError, match="chunk_bytes"):
        g.open(os.path.join(FIX, "sc.h5ad"), budget=b, chunk_bytes="64MB")
    assert b.stats()["ram"]["grants"] == 0
    r = g.open(os.path.join(FIX, "sc.h5ad"), budget=b, chunk_bytes=256 << 20)   # the default, spelled out
    assert r.chunk_bytes == 1 << 20
    r.close()


@need
def test_tiff_tiles_match_full_read():
    import tifffile
    path = os.path.join(FIX, "tissue.ome.tif")
    full = tifffile.imread(path)
    a = g.TiffTileArray(path)
    for win in [(0, 512, 0, 512), (1000, 1600, 2000, 2700), (5800, 6144, 5900, 6144)]:
        y0, y1, x0, x1 = win
        assert np.array_equal(a[y0:y1, x0:x1], full[y0:y1, x0:x1])
    assert np.array_equal(a[100:900:7, 50:1500:16], full[100:900:7, 50:1500:16])      # steps are honoured
    a.close()


@need
def test_multichannel_tiff_keeps_every_channel():
    import tifffile
    path = os.path.join(FIX, "glioma", "codex_6plex.ome.tif")
    full = tifffile.imread(path)
    a = g.TiffTileArray(path)
    assert a.shape == full.shape == (6, 1500, 1500) and a.axes == "CYX"
    for c in range(6):
        assert np.array_equal(a[c, 700:1100, 300:900], full[c, 700:1100, 300:900])
    assert np.array_equal(a.window(1200, 1500, 1200, 1500), full[:, 1200:1500, 1200:1500])
    a.close()
