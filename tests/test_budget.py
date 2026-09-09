"""The invariant that matters: peak resident bytes never exceed the declared budget."""
import os
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
    seen = sum(c["X"].shape[0] for c in r)
    assert seen == p.detail["n_obs"]
    r.close()


@need
def test_tiff_tiles_match_full_read():
    import numpy as np, tifffile
    path = os.path.join(FIX, "tissue.ome.tif")
    full = tifffile.imread(path)
    a = g.TiffTileArray(path)
    for win in [(0, 512, 0, 512), (1000, 1600, 2000, 2700), (5800, 6144, 5900, 6144)]:
        y0, y1, x0, x1 = win
        assert np.array_equal(a[y0:y1, x0:x1], full[y0:y1, x0:x1])
    a.close()
