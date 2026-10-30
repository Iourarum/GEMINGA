"""Regression tests for the Rust-core defects found in the 0.17 audit: budget waits that held
the GIL, a deadline restarted by every release, chunks materialised before they were leased,
the planned chunk sized from capacity instead of what is free, `sample()` outside the budget,
`inf` timeouts panicking, and the GEM index ignoring the chip's offsets."""
import os
import threading
import time

import numpy as np
import pytest

import geminga as g

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")


def test_a_waiting_acquire_releases_the_gil_so_another_thread_can_hand_bytes_back():
    b = g.Budget(ram="1KB", timeout_s=3)
    hold = b.acquire("ram", "800")

    def releaser():
        time.sleep(0.3)
        hold.release()
    t0 = time.time()
    threading.Thread(target=releaser).start()
    lease = b.acquire("ram", "500")                                  # used to block the releaser for 3 s
    assert time.time() - t0 < 1.5
    lease.release()
    assert b.stats()["ram"]["waits"] == 1 and b.stats()["ram"]["denials"] == 0


def test_the_deadline_is_measured_from_the_first_wait_not_restarted_by_churn():
    b = g.Budget(ram="1KB", timeout_s=1.0)
    hold = b.acquire("ram", "900")
    stop = threading.Event()

    def churn():
        while not stop.is_set():
            x = b.acquire("ram", "50")
            time.sleep(0.05)
            x.release()
            time.sleep(0.05)
    th = threading.Thread(target=churn)
    th.start()
    t0 = time.time()
    with pytest.raises(MemoryError, match="timed out"):
        b.acquire("ram", "500")
    elapsed = time.time() - t0
    stop.set()
    th.join()
    hold.release()
    assert 0.9 <= elapsed < 2.0


def test_try_acquire_is_one_locked_step():
    b = g.Budget(ram="1KB")
    lease = b.try_acquire("ram", "600")
    assert lease is not None and b.try_acquire("ram", "600") is None
    assert b.try_acquire("ram", "5000") is None                      # never fits: no wait, no raise
    lease.release()
    assert b.try_acquire("ram", "1000") is not None
    assert b.stats()["ram"]["denials"] == 1


def test_infinite_timeout_means_wait_not_panic():
    b = g.Budget(ram="1MB", timeout_s=float("inf"))
    with b.acquire("ram", "1KB"):
        pass
    b2 = g.Budget(ram="1MB", timeout_s=1e300)
    b2.acquire("ram", "1KB").release()


def test_a_chunk_is_leased_before_it_is_read():
    if not os.path.exists(os.path.join(FIX, "reads.fastq")):
        pytest.skip("no FASTQ fixture")
    b = g.Budget(ram="2MB", timeout_s=1)
    with pytest.raises(MemoryError, match="cannot fit 16.0 MB"):     # the plan, not the 18 MB that was read
        next(iter(g.FastxStream(os.path.join(FIX, "reads.fastq"), "16MB", 0, b)))
    assert b.stats()["ram"]["peak"] == 0 and b.stats()["ram"]["denials"] == 1
    small = g.Budget(ram="4MB")
    n = 0
    for batch in g.FastxStream(os.path.join(FIX, "reads.fastq"), "1MB", 0, small):
        n += batch.num_rows
        assert small.stats()["ram"]["used"] >= 1          # held while the batch is out
    assert n == 200_000 and small.stats()["ram"]["peak"] <= 4 << 20


def test_default_chunk_follows_what_is_free():
    if not os.path.exists(os.path.join(FIX, "culture.parquet")):
        pytest.skip("no parquet fixture")
    b = g.Budget(ram="64MB")
    ds = g.open(os.path.join(FIX, "culture.parquet"), budget=b)
    assert ds.plan()["chunk_bytes"] == 16 << 20
    with b.acquire("ram", "60MB"):
        assert ds.plan()["chunk_bytes"] == 1 << 20                    # a quarter of 4 MB, floored at 1 MB
    with b.acquire("ram", "63.5MB"):
        assert ds.plan()["chunk_bytes"] == 512 << 10                  # never above what is free


def test_sample_reads_under_the_budget():
    if not os.path.exists(os.path.join(FIX, "culture.parquet")):
        pytest.skip("no parquet fixture")
    b = g.Budget(ram="8MB")
    ds = g.open(os.path.join(FIX, "culture.parquet"), budget=b)
    out = ds.sample(n_row_groups=2)
    assert sum(x.num_rows for x in out) > 0 and b.stats()["ram"]["grants"] >= 1


def test_gem_index_applies_the_chip_offsets(tmp_path):
    rng = np.random.default_rng(0)
    n = 20_000
    xs = rng.integers(0, 1000, n)
    ys = rng.integers(0, 1000, n)
    p = tmp_path / "off.gem"
    with open(p, "w") as fh:
        fh.write("#FileFormat=GEMv0.1\n#SortedBy=None\n#BinSize=1\n#StereoChip=X\n#OffsetX=5000\n#OffsetY=7000\n")
        fh.write("geneID\tx\ty\tMIDCount\n")
        for x, y in zip(xs, ys):
            fh.write(f"G{x % 7}\t{x}\t{y}\t1\n")
    idx = g.GemIndex.build(str(p), 8)
    x0, x1, y0, y1 = idx.extent
    assert x0 == 5000 and y0 == 7000                                   # the index sees the same coordinates GemStream does
    mid = (x0 + x1) // 2
    left = idx.rows_in(x0, mid, y0, y1)
    right = idx.rows_in(mid, x1, y0, y1)
    assert abs(left - (xs + 5000 < mid).sum()) < n * 0.02 and abs(right - (xs + 5000 >= mid).sum()) < n * 0.02
    assert left > n * 0.4 and right > n * 0.4                          # not everything in cell (0,0)
    sc = str(p) + ".gemidx.json"
    idx.save(sc)
    assert g.GemIndex.load(sc).rows_in(x0, mid, y0, y1) == left


def test_old_gem_index_sidecars_are_rebuilt_not_trusted(tmp_path):
    import json
    p = tmp_path / "t.gem"
    with open(p, "w") as fh:
        fh.write("#FileFormat=GEMv0.1\n#BinSize=1\n#OffsetX=0\n#OffsetY=0\ngeneID\tx\ty\tMIDCount\nG\t1\t1\t1\nG\t2\t2\t1\n")
    idx = g.GemIndex.build(str(p), 4)
    sc = str(p) + ".gemidx.json"
    idx.save(sc)
    d = json.load(open(sc))
    d["version"] = 1                                                    # an index from before the offset fix
    json.dump(d, open(sc, "w"))
    with pytest.raises(ValueError, match="older GEMINGA"):
        g.GemIndex.load(sc)
    time.sleep(0.01)
    os.utime(sc)                                                        # newer than the GEM: would have been reused
    rebuilt = g.open_gem(str(p))                                        # index_gem falls through to a rebuild
    assert rebuilt is not None and json.load(open(sc))["version"] == 2
