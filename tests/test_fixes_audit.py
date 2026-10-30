"""Regression tests for the audit fixes, one per defect.

Each test fails on the code as it was and passes after its fix:
1 ProgressiveCanvas binning and nulls, 2 Staging.extract leaving the view's lease,
3 RemoteFile.read copies, 4 h5ad chunks held across the yield, 5 index_gem save fallback,
6 GefFile.rasterize precision, 7 GEF maxX/maxY read as inclusive maxima.
"""
from __future__ import annotations

import gc
import os
import sys
import tracemalloc
from types import SimpleNamespace

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import geminga as g
from geminga import stereoseq as S

MB = 1 << 20
EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")


# 1 ---------------------------------------------------------------------------

def test_canvas_gives_each_column_an_equal_share_and_nulls_light_no_cell():
    c = g.ProgressiveCanvas((0, 1, 0, 1), shape=(4, 4))
    xs = np.linspace(0, 1, 100_000, endpoint=False)                  # uniform over [0, 1)
    c.add(xs, np.full(xs.size, 0.5))
    per_column = c.counts.sum(axis=0)
    assert np.abs(per_column / (xs.size / 4) - 1).max() < 0.01        # every column within 1 %
    assert c.n_rows == xs.size and c.counts.sum() == xs.size

    edge = g.ProgressiveCanvas((0, 1, 0, 1), shape=(4, 4))
    edge.add([0.0, 1.0], [0.0, 1.0])                                  # the top edge is the last cell
    assert edge.counts[0, 0] == 1 and edge.counts[3, 3] == 1

    nulls = g.ProgressiveCanvas((0, 1, 0, 1), shape=(4, 4))
    nulls.add(np.array([np.nan, np.nan, np.nan]), np.array([0.1, 0.5, 0.9]))
    assert nulls.counts.sum() == 0 and nulls.grid.sum() == 0.0
    assert nulls.coverage == 0.0 and nulls.n_rows == 0 and nulls.n_dropped == 3

    nw = g.ProgressiveCanvas((0, 1, 0, 1), shape=(2, 2), weight=True)
    nw.add([0.1, 0.9], [0.1, 0.9], weights=[np.nan, 2.0])                # a null weight: no cell either
    assert np.isfinite(nw.grid).all() and nw.grid.sum() == 2.0
    assert nw.n_rows == 1 and nw.n_dropped == 1 and nw.counts.sum() == 1


# 2 ---------------------------------------------------------------------------

def test_extract_gives_the_view_its_budget_back_once_the_file_is_written(tmp_path):
    src = str(tmp_path / "pts.parquet")
    rng = np.random.default_rng(0)
    n = 200_000
    pq.write_table(pa.table({"x": rng.uniform(0, 100, n), "y": rng.uniform(0, 100, n)}), src,
                   row_group_size=20_000)
    b = g.Budget(ram="64MB", spill="256MB", timeout_s=5)
    view = g.view(src, budget=b)
    st = g.staging(str(tmp_path / "stage"), capacity="32MB", budget=b)
    before = b.stats()["ram"]["used"]
    r = st.extract("roi", view, window=(0, 120_000))
    assert r.rows == 120_000
    assert b.stats()["ram"]["used"] == before                 # the staged file holds no RAM
    view.close()


# 3 ---------------------------------------------------------------------------

def test_remote_read_fills_one_buffer_across_blocks_and_peaks_near_two_copies(tmp_path):
    sys.path.insert(0, os.path.abspath(EXAMPLES))
    from range_server import serve
    data = np.random.default_rng(3).integers(0, 256, 17 * MB, dtype=np.uint8).tobytes()
    (tmp_path / "blob.bin").write_bytes(data)
    httpd = serve(str(tmp_path), 0)
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/blob.bin"
        rf = g.RemoteFile(url, block="64KB", cache="256KB", readahead=1)
        off, n = 100_003, 5 * 65_536 + 777                     # several blocks, starting mid-block
        rf.seek(off)
        assert rf.read(n) == data[off:off + n]
        rf.seek(0)
        rf.drop_cache()
        n = 16 * MB
        gc.collect()
        tracemalloc.start()
        try:
            out = rf.read(n)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert type(out) is bytes and out == data[:n] and rf.tell() == n
        # the result and one buffer of its size; a buffer grown by appends and then copied
        # again peaks above this (it over-allocates by up to 1/8 and copies every block twice)
        assert peak < 2 * n + MB, f"peak {peak:,} bytes for a {n:,}-byte read"
        rf.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


# 4 ---------------------------------------------------------------------------

def _write_csr_h5ad(path, n_obs=30_000, n_vars=400, seed=0):
    """An h5ad whose X and obsm/emb are CSR matrices with int64 indices."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(3, 14, n_obs)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    indices = np.concatenate([np.sort(rng.choice(n_vars, k, replace=False))
                              for k in counts]).astype(np.int64)
    data = rng.integers(1, 9, indices.size).astype(np.float32)

    def csr(group):
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["shape"] = np.array([n_obs, n_vars], np.int64)
        group.create_dataset("data", data=data)
        group.create_dataset("indices", data=indices)
        group.create_dataset("indptr", data=indptr)

    with h5py.File(path, "w") as f:
        csr(f.create_group("X"))
        csr(f.create_group("obsm").create_group("emb"))


def _worst_live_over_leased(make_stream, budget):
    """Warm up one pass (lazy imports, row bounds), then the worst ratio, over the chunks of a
    second pass, of traced live bytes at the yield to the bytes the budget holds for the chunk."""
    for _ in make_stream():
        pass
    gc.collect()
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        n, worst = 0, 0.0
        for _chunk in make_stream():
            leased = budget.stats()["ram"]["used"]
            live = tracemalloc.get_traced_memory()[0] - base
            worst = max(worst, live / max(leased, 1))
            n += 1
    finally:
        tracemalloc.stop()
    return n, worst


def test_h5ad_chunks_hold_no_int64_copies_across_the_yield(tmp_path):
    path = str(tmp_path / "csr64.h5ad")
    _write_csr_h5ad(path)
    b1, b2, b3 = (g.Budget(ram="64MB", timeout_s=5) for _ in range(3))
    h5 = g.H5adReader(path, chunk_bytes="1MB", budget=b1)
    obsm = g.ObsmReader(path, "emb", chunk_bytes="1MB", budget=b2, obs_names=False)
    rows = g.RowView(g.H5adReader(path, chunk_bytes="1MB", budget=b3), None, b3,
                     kind="cells", chunk_bytes="1MB")
    try:
        results = {"H5adReader": _worst_live_over_leased(lambda: iter(h5), b1),
                   "ObsmReader": _worst_live_over_leased(lambda: iter(obsm), b2),
                   "RowView.chunks": _worst_live_over_leased(rows.chunks, b3)}
    finally:
        h5.close()
        obsm.close()
        rows.close()
    for name, (n, worst) in results.items():
        assert n >= 3, (name, n)
        assert worst < 1.15, f"{name}: live bytes at a yield were {worst:.2f} x the leased bytes"


# 5 ---------------------------------------------------------------------------

def _write_gem(path, n=400, seed=1):
    rng = np.random.default_rng(seed)
    with open(path, "w") as fh:
        fh.write("#FileFormat=GEMv0.1\n#SortedBy=None\n#BinSize=1\n#StereoChip=SSMINI01\n"
                 "#OffsetX=0\n#OffsetY=0\ngeneID\tx\ty\tMIDCount\n")
        for _ in range(n):
            fh.write(f"Gene{rng.integers(0, 20):04d}\t{rng.integers(0, 500)}\t"
                     f"{rng.integers(0, 500)}\t1\n")
    return str(path)


def test_index_gem_saves_under_the_private_home_when_beside_the_gem_cannot_be_written(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    drive = tmp_path / "drive"
    drive.mkdir()
    gem = _write_gem(drive / "chip.gem")
    sc = gem + ".gemidx.json"
    from geminga.gefindex import _home_dir
    private = os.path.join(_home_dir(os.path.abspath(gem)), os.path.basename(sc))
    os.chmod(drive, 0o555)                       # a read-only directory ...
    try:
        if os.access(drive, os.W_OK):            # ... that root writes into anyway: a directory
            os.chmod(drive, 0o755)               # in the sidecar's place fails the save as well
            os.mkdir(sc)
        real, builds = S.GemIndex, []

        class Counting:
            @staticmethod
            def build(p, grid):
                builds.append(p)
                return real.build(p, grid)

            @staticmethod
            def load(p):
                return real.load(p)

        monkeypatch.setattr(S, "GemIndex", Counting)
        with pytest.warns(UserWarning) as rec:
            idx = g.index_gem(gem)
        msgs = [str(w.message) for w in rec if issubclass(w.category, UserWarning)]
        assert len(msgs) == 1, msgs
        assert sc in msgs[0] and private in msgs[0], msgs[0]
        assert os.path.isfile(private) and idx.n_rows > 0
        again = g.index_gem(gem)                  # the private copy is found: nothing is rebuilt
        assert len(builds) == 1 and again.n_rows == idx.n_rows
    finally:
        os.chmod(drive, 0o755)


# 6 ---------------------------------------------------------------------------

def _fake_gef(pieces):
    """Just enough of a GefFile for rasterize: one piece of rows per list, all at pixel (0, 0)."""
    items = [SimpleNamespace(data={"x": np.zeros(len(p), np.int32),
                                   "y": np.zeros(len(p), np.int32),
                                   "count": np.asarray(p, np.int32)}) for p in pieces]
    return SimpleNamespace(pick_bin=lambda b=None: 1, _iter_window=lambda *a, **k: iter(items))


def test_rasterize_sums_a_pixel_exactly_past_float32_precision():
    # one pixel receives 2**24 + 4 MIDs in five pieces; a float32 running sum stops at 2**24
    img = S.GefFile.rasterize(_fake_gef([[2 ** 24], [1], [1], [1], [1]]), (0, 4, 0, 4), 1, shape=2)
    assert img.dtype == np.float32 and img.shape == (2, 2)
    assert img[0, 0] == 2 ** 24 + 4
    assert img.sum() == 2 ** 24 + 4
    # past 2**32 MIDs in all, the count is carried wider, so it does not wrap
    img = S.GefFile.rasterize(_fake_gef([[2 ** 31 - 1], [2 ** 31 - 1], [5]]), (0, 4, 0, 4), 1, shape=2)
    assert img[0, 0] == np.float32(2 ** 32 + 3)


# 7 ---------------------------------------------------------------------------

def _saw_like_gef(path, side=300, bins=(1, 50, 100)):
    """A GEF whose attributes follow SAW: minX/minY the smallest coordinates, maxX/maxY the
    largest ones present (inclusive). The data reaches the far corner, (side-1, side-1)."""
    rng = np.random.default_rng(3)
    n = 2000
    xs = np.concatenate([rng.integers(0, side, n - 1), [side - 1]]).astype(np.int64)
    ys = np.concatenate([rng.integers(0, side, n - 1), [side - 1]]).astype(np.int64)
    with h5py.File(path, "w") as f:
        grp = f.create_group("geneExp")
        for b in bins:
            key = (xs // b) * b * (side + 1) + (ys // b) * b
            uniq, inv = np.unique(key, return_inverse=True)
            cnt = np.bincount(inv.ravel(), minlength=uniq.size)
            rec = np.zeros(uniq.size, [("x", "<u4"), ("y", "<u4"), ("count", "<u2")])
            rec["x"], rec["y"], rec["count"] = uniq // (side + 1), uniq % (side + 1), cnt
            gp = grp.create_group(f"bin{b}")
            d = gp.create_dataset("expression", data=rec)
            d.attrs["minX"] = d.attrs["minY"] = np.uint32(0)
            d.attrs["maxX"], d.attrs["maxY"] = np.uint32(int(rec["x"].max())), np.uint32(int(rec["y"].max()))
            gr = np.zeros(1, [("gene", "S32"), ("offset", "<u4"), ("count", "<u4")])
            gr["gene"], gr["offset"], gr["count"] = b"GeneA", 0, uniq.size
            gp.create_dataset("gene", data=gr)
    return str(path)


def test_gef_extent_is_one_past_the_inclusive_maxima_saw_writes(tmp_path):
    gf = g.GefFile(_saw_like_gef(tmp_path / "saw.gef"))
    try:
        x0, x1, y0, y1 = gf.extent
        assert (x0, y0) == (0, 0) and (x1, y1) == (300, 300)             # was (299, 299): the last
        rows = gf.read_window((x0, x1, y0, y1), 1)                         # column and row fell out
        xs, ys = np.asarray(rows["x"]), np.asarray(rows["y"])
        assert xs.max() == 299 and ys.max() == 299
        assert ((xs == 299) & (ys == 299)).any()                           # the far corner is in
    finally:
        gf.close()
    # the same chip with the attributes missing: computed from the coarsest level, same window
    noattr = _saw_like_gef(tmp_path / "noattr.gef")
    with h5py.File(noattr, "r+") as f:
        for b in (1, 50, 100):
            for k in ("minX", "minY", "maxX", "maxY"):
                del f[f"geneExp/bin{b}/expression"].attrs[k]
    gf2 = g.GefFile(noattr)
    assert gf2.extent == (0, 300, 0, 300)
    gf2.close()
