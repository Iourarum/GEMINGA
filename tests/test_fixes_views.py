"""Regression tests for the view / Stereo-seq / GEF-index fixes in CHANGES-F.md.

Numbering follows the findings: 1 Stereo-seq zoom is the pyramid, not a stride; 2 GEF reads are
leased blocks; 3 load paths read only what they keep; 4 lease lifecycle; 5 navigation; 6 wrong
answers; 7 the GEF sidecar.

Every memory test runs on an input at least 10x the budget and checks that the budget's peak
stayed within its capacity (no denials, nothing left leased) and that the Python heap
(`tracemalloc`) stayed under 3x the capacity while `load()` / `chunks()` ran. Arrow and the Rust
readers allocate outside the Python heap, so their tests also bound the process's real
high-water mark (`VmHWM`) and the bytes read from the file (`/proc/self/io`).

GEF fixtures are always copied to a temporary directory first: nothing here writes into
`fixtures/`.
"""
from __future__ import annotations

import gc
import json
import os
import pathlib
import shutil
import tracemalloc

import h5py
import numpy as np
import pytest

import geminga as g

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")
GEF = os.path.join(FIX, "chip.gef")
GEM = os.path.join(FIX, "chip.gem")
IMG = os.path.join(FIX, "chip_ssdna.tif")
TIF = os.path.join(FIX, "tissue.ome.tif")
FQ = os.path.join(FIX, "reads.fastq")
FA = os.path.join(FIX, "transcripts.fasta")
H5AD = os.path.join(FIX, "sc.h5ad")
TENX = os.path.join(FIX, "glioma", "filtered_feature_bc_matrix.h5")
MB = 1 << 20

need_gef = pytest.mark.skipif(not os.path.exists(GEF), reason="run examples/make_stereoseq_fixtures.py")
need_tif = pytest.mark.skipif(not os.path.exists(TIF), reason="no OME-TIFF fixture")
need_fq = pytest.mark.skipif(not os.path.exists(FQ + ".gemidx.json"), reason="no indexed FASTQ fixture")
need_h5ad = pytest.mark.skipif(not os.path.exists(H5AD), reason="no h5ad fixture")


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

class heap:
    """tracemalloc around a block; `.peak` is the Python-heap high-water mark inside it."""

    def __enter__(self):
        gc.collect()
        tracemalloc.start()
        return self

    def __exit__(self, *exc):
        _, self.peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return False


def _status(key: str) -> int:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024
    return 0


class rss:
    """The process's real high-water mark above where it started (Linux), for memory that
    Arrow and the Rust core allocate outside the Python heap."""

    def __enter__(self):
        gc.collect()
        try:
            with open("/proc/self/clear_refs", "w") as fh:
                fh.write("5")
            self.ok = True
        except OSError:
            self.ok = False
        self.base = _status("VmRSS")
        return self

    def __exit__(self, *exc):
        self.peak = _status("VmHWM") - self.base if self.ok else 0
        return False


class reads:
    """Bytes read through read() syscalls inside the block (`/proc/self/io` rchar)."""

    @staticmethod
    def _rchar() -> int:
        with open("/proc/self/io") as fh:
            for line in fh:
                if line.startswith("rchar:"):
                    return int(line.split()[1])
        return 0

    def __enter__(self):
        self.r0 = self._rchar()
        return self

    def __exit__(self, *exc):
        self.bytes = self._rchar() - self.r0
        return False


def budget(ram: str) -> g.Budget:
    return g.Budget(ram=ram, timeout_s=5)


def governed(b: g.Budget, cap: int, h: heap, factor: int = 3) -> None:
    st = b.stats()["ram"]
    assert st["peak"] <= cap, st
    assert st["denials"] == 0, st
    assert st["used"] == 0, st                     # everything was handed back
    assert h.peak < cap * factor, f"heap peak {h.peak:,} >= {factor} x {cap:,}"


def nbytes(d) -> int:
    if isinstance(d, dict):
        return sum(v.nbytes for v in d.values() if isinstance(v, np.ndarray))
    return int(d.nbytes)


def ref_window(path: str, b: int, w: tuple, genes=None) -> np.ndarray:
    """Rows of level `b` whose bin overlaps window `w`, read the plain way (the whole level)."""
    with h5py.File(path, "r") as f:
        e = f[f"geneExp/bin{b}/expression"][:]
        gi = f[f"geneExp/bin{b}/gene"][:]
    gid = np.repeat(np.arange(len(gi)), gi["count"].astype(np.int64))
    x0, x1, y0, y1 = w
    x, y = e["x"].astype(np.int64), e["y"].astype(np.int64)
    m = (x < x1) & (x + b > x0) & (y < y1) & (y + b > y0)
    if genes is not None:
        names = [n.decode() for n in gi["gene"]]
        m &= np.isin(gid, [names.index(q) for q in genes])
    out = np.zeros(int(m.sum()), dtype=[("x", "i8"), ("y", "i8"), ("count", "i8"), ("gene", "i8")])
    out["x"], out["y"], out["count"], out["gene"] = x[m], y[m], e["count"][m], gid[m]
    return out


def same_rows(d: dict, ref: np.ndarray) -> None:
    got = sorted(zip(d["x"].tolist(), d["y"].tolist(), d["count"].tolist()))
    want = sorted(zip(ref["x"].tolist(), ref["y"].tolist(), ref["count"].tolist()))
    assert got == want


def write_gef(path, x, y, gene, count, names, bins=(1, 100), chunk=4096):
    """A GEF laid out the way SAW writes one: per level, `expression` rows grouped gene by gene,
    and `gene` = (name, offset, count) pointing into them."""
    x, y = np.asarray(x, np.int64), np.asarray(y, np.int64)
    gene, count = np.asarray(gene, np.int64), np.asarray(count, np.int64)
    with h5py.File(path, "w") as f:
        f.attrs["StereoChip"] = np.bytes_(b"TEST")
        grp = f.create_group("geneExp")
        grp.attrs["resolution"] = 500
        for b in bins:
            bx, by = x // b * b, y // b * b
            key = (gene << 42) | (by << 21) | bx
            order = np.argsort(key, kind="stable")
            k = key[order]
            uniq, start = np.unique(k, return_index=True)
            cnt = np.add.reduceat(count[order], start)
            gid = uniq >> 42
            rows = np.zeros(len(uniq), dtype=[("x", "<u4"), ("y", "<u4"), ("count", "<u2")])
            rows["x"] = uniq & ((1 << 21) - 1)
            rows["y"] = (uniq >> 21) & ((1 << 21) - 1)
            rows["count"] = np.minimum(cnt, 65535)
            lg = grp.create_group(f"bin{b}")
            ds = lg.create_dataset("expression", data=rows, chunks=(min(chunk, len(rows)),),
                                   compression="gzip")
            ds.attrs["minX"], ds.attrs["minY"] = np.uint32(0), np.uint32(0)
            ds.attrs["maxX"], ds.attrs["maxY"] = np.uint32(x.max()), np.uint32(y.max())   # inclusive, as SAW
            per = np.bincount(gid, minlength=len(names))
            gd = np.zeros(len(names), dtype=[("gene", "S32"), ("offset", "<u4"), ("count", "<u4")])
            gd["gene"] = [n.encode() for n in names]
            gd["count"] = per
            gd["offset"] = np.concatenate([[0], np.cumsum(per)[:-1]])
            lg.create_dataset("gene", data=gd)
    return path


@pytest.fixture(scope="session")
def gef_ro(tmp_path_factory):
    """A private copy of the chip, never indexed: reads take the full-level path."""
    d = tmp_path_factory.mktemp("gef_ro")
    dst = d / "chip.gef"
    shutil.copy(GEF, dst)
    return str(dst)


@pytest.fixture
def gef_copy(tmp_path):
    dst = tmp_path / "chip.gef"
    shutil.copy(GEF, dst)
    return str(dst)


@pytest.fixture(scope="session")
def big_gene_gef(tmp_path_factory):
    """One gene with 1.5 M rows at bin1 — a 15 MB slice — and two small ones."""
    rng = np.random.default_rng(7)
    n = 1_500_000
    x = np.concatenate([rng.integers(0, 4000, n), rng.integers(0, 4000, 3000)])
    y = np.concatenate([rng.integers(0, 4000, n), rng.integers(0, 4000, 3000)])
    gene = np.concatenate([np.zeros(n, np.int64), np.repeat([1, 2], 1500)])
    count = rng.integers(1, 4, len(x))
    p = tmp_path_factory.mktemp("biggene") / "big.gef"
    return write_gef(str(p), x, y, gene, count, ["BIG", "SMALL1", "SMALL2"])


@pytest.fixture(scope="session")
def fq_view_factory():
    def make(b=None):
        return g.SeqView(g.SeqIndex.load(FQ + ".gemidx.json"), FQ, b)
    return make


@pytest.fixture(scope="session")
def big_parquet(tmp_path_factory):
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = 2_000_000
    p = tmp_path_factory.mktemp("pq") / "rows.parquet"
    t = pa.table({"a": np.arange(n, dtype=np.int64),
                  "b": np.random.default_rng(0).random(n),
                  "s": pa.array(np.char.mod("%016d", np.arange(n)))})
    pq.write_table(t, p, row_group_size=100_000)
    return str(p)


# ===========================================================================
# 1. Stereo-seq zoom: the pyramid level IS the decimation
# ===========================================================================

@need_gef
def test_stereo_zoom_walks_the_pyramid_without_dropping_rows(gef_ro):
    v = g.open_gef(gef_ro, budget=budget("256MB"))
    gf = v.gef
    chip_mid = int(ref_window(gef_ro, 1, gf.extent)["count"].sum())
    for z, b in enumerate(gf.bins):
        v.reset().set_zoom(z)
        c = v.cost()
        assert c["bin_size"] == b and c["stride"] == 1 and v.stride == 1
        d = v.load()
        ref = ref_window(gef_ro, b, v.window)
        assert d["x"].size == ref.size                       # every row of the level, none dropped
        assert int(d["count"].sum()) == int(ref["count"].sum()) == chip_mid
        assert sum(ch["x"].size for ch in v.chunks()) == d["x"].size
    v.select(1000, 1900, 1200, 2100)
    for z, b in enumerate(gf.bins):
        v.set_zoom(z)
        assert int(v.load()["count"].sum()) == int(ref_window(gef_ro, b, v.window)["count"].sum())
    v.close()


@need_gef
def test_stereo_zoom_beyond_the_pyramid_stays_on_the_coarsest_level(gef_ro):
    v = g.open_gef(gef_ro).reset()
    v.set_zoom(len(v.gef.bins) + 3)
    assert v.bin_size == v.gef.bins[-1] and v.stride == 1 and v.cost()["stride"] == 1
    v.close()


# ===========================================================================
# 2. GEF window reads: leased blocks, honest estimates, checked rasters
# ===========================================================================

@need_gef
def test_gef_window_read_streams_the_level_in_leased_blocks(gef_ro):
    cap = 3 * MB                                             # bin1 is 29.5 MB on disk, 35 MB resident
    b = budget("3MB")
    w = (1000, 1400, 1200, 1600)
    v = g.open_gef(gef_ro, budget=b).use_bin(1).select(*w)
    with heap() as h:
        d = v.load()
        assert b.stats()["ram"]["used"] == nbytes(d)          # what is held is what was handed out
        v.release()
    governed(b, cap, h)
    same_rows(d, ref_window(gef_ro, 1, w))
    c = v.cost()
    assert c["read_path"] == "full-level" and c["fits_whole"] is True
    assert c["peak_bytes"] > c["total_bytes"]                # the read block is priced too


@need_gef
def test_gene_reads_cost_their_slice_and_stream_it(big_gene_gef):
    cap = 3 * MB // 2                                        # the BIG slice is 15 MB on disk
    b = budget("1536KB")
    w = (100, 700, 200, 900)
    v = g.open_gef(big_gene_gef, budget=b).use_bin(1).select(*w).select_genes(["BIG"])
    c = v.cost()
    names, offs, cnts = v.gef.gene_index(1)
    assert c["read_path"] == "gene-slices"
    assert c["read_bytes"] == int(cnts[0]) * v.gef.levels[1].itemsize
    with heap() as h:
        d = v.load()
        v.release()
    governed(b, cap, h)
    same_rows(d, ref_window(big_gene_gef, 1, w, ["BIG"]))
    assert set(np.unique(d["gene"])) == {"BIG"}
    with heap() as h:
        n = sum(ch["x"].size for ch in v.chunks())
    governed(b, cap, h)
    assert n == d["x"].size
    v.close()


@need_gef
def test_image_overview_and_paired_are_checked_and_bounded(gef_copy):
    tiny = g.open_gef(gef_copy, budget=budget("512KB")).use_bin(1)
    with pytest.raises(g.CostError):
        tiny.image(512)                                      # a 1 MB raster cannot fit 512 KB
    with pytest.raises(g.CostError):
        g.open_gef(gef_copy, budget=budget("16KB")).overview(64)
    cap = 3 * MB
    b = budget("3MB")
    v = g.open_gef(gef_copy, budget=b).use_bin(1)            # the whole chip at bin1: 35 MB of rows
    with heap() as h:
        img = v.image(512)
        ov = v.overview(128)
    governed(b, cap, h)
    assert img.shape == (512, 512)
    assert float(img.sum()) == float(ref_window(gef_copy, 1, v.gef.extent)["count"].sum())
    assert ov.series["thumbnail"].sum() == img.sum()         # the same chip, the same MID
    side = v.gef.extent[1] - v.gef.extent[0]
    v.close()
    ds = g.open_stereoseq(gef=gef_copy, image=IMG, budget=b, image_scale=2048 / side)
    with heap() as h:
        pair = ds.paired((1600, 3000, 1400, 2800), shape=256)
    governed(b, cap, h)
    assert pair["expression"].shape == (256, 256)
    small = g.open_stereoseq(gef=gef_copy, image=IMG, budget=budget("256KB"), image_scale=2048 / side)
    with pytest.raises(g.CostError):
        small.paired((1600, 3000, 1400, 2800), shape=512)
    ds.close()
    small.close()


@need_gef
def test_estimate_rows_follows_the_tissue_not_the_area(gef_ro):
    gf = g.GefFile(gef_ro)
    rng = np.random.default_rng(3)
    ratios = []
    for b in (1, 20, 100):
        with h5py.File(gef_ro, "r") as f:
            e = f[f"geneExp/bin{b}/expression"][:]
        x, y = e["x"].astype(np.int64), e["y"].astype(np.int64)
        for _ in range(40):
            s = int(rng.integers(200, 2500))
            x0, y0 = (int(v) for v in rng.integers(0, 4000 - s, 2))
            w = (x0, x0 + s, y0, y0 + s)
            actual = int(((x < w[1]) & (x + b > w[0]) & (y < w[3]) & (y + b > w[2])).sum())
            if actual >= 1000:
                ratios.append(gf.estimate_rows(w, b) / actual)
    r = np.array(ratios)
    assert len(r) > 60
    assert 0.9 < np.median(r) < 1.1
    assert r.min() > 0.6 and r.max() < 1.6, (r.min(), r.max())   # area alone: 0.09x .. 3.5x
    assert gf.estimate_rows(gf.extent, 1) == gf.levels[1].n_obs   # the whole chip is exact
    gf.close()


@need_gef
def test_indexed_reads_stream_row_groups_through_the_budget(gef_copy):
    pytest.importorskip("pyarrow")
    g.GefFile(gef_copy).build_spatial_index(levels=[1])
    cap = 3 * MB
    b = budget("3MB")
    v = g.open_gef(gef_copy, budget=b).use_bin(1)
    c = v.cost()
    assert c["read_path"] == "spatial-index"
    assert c["fits_whole"] is False and c["fits"] is True     # 35 MB of rows: stream them
    n = 0
    with heap() as h:
        for ch in v.chunks():
            n += ch["x"].size
    governed(b, cap, h)
    assert n == v.gef.levels[1].n_obs
    w = (1000, 1400, 1200, 1600)
    est = v.gef.estimate_rows(w, 1)
    with heap() as h:
        d = v.select(*w).load()
        v.release()
    governed(b, cap, h)
    same_rows(d, ref_window(gef_copy, 1, w))
    assert 0.7 < est / d["x"].size < 1.4                    # row-group statistics, not area
    v.close()


# ===========================================================================
# 3. Load paths: lease the output first, read only what is kept
# ===========================================================================

@need_tif
def test_image_load_and_overview_read_with_a_step():
    tifffile = pytest.importorskip("tifffile")
    cap = 4 * MB                                             # the image is 75 MB
    b = budget("4MB")
    v = g.view(TIF, budget=b).set_zoom(3)                    # stride 8: 768 x 768 x 2 B
    assert v.cost()["fits_whole"] is True
    with heap() as h:
        a = v.load()
        assert b.stats()["ram"]["used"] == a.nbytes
        ov = v.overview(256)
        v.release()
    governed(b, cap, h)
    ref = tifffile.imread(TIF)
    assert np.array_equal(a, ref[::8, ::8])
    assert np.array_equal(ov.series["thumbnail"], ref[::24, ::24])
    del ref
    with heap() as h:
        n = sum(t.nbytes for t in v.chunks())
    governed(b, cap, h)
    assert n == a.nbytes
    v.close()


def test_parquet_zoom_decimates_while_reading(big_parquet):
    pa = pytest.importorskip("pyarrow")
    cap = 4 * MB                                             # 2 M rows, ~70 MB once decoded
    b = budget("4MB")
    v = g.view(big_parquet, budget=b).set_zoom(5)            # stride 32: 62,500 rows
    v.select(0, 20_000).load()                               # warm the reader up
    v.release().reset().set_zoom(5)
    with rss() as r, heap() as h:
        t = v.load()
        v.release()
    governed(b, cap, h)
    assert r.peak < 3 * cap, f"real high-water mark {r.peak:,}"     # the whole window is 70 MB
    assert t.num_rows == 62_500
    assert np.array_equal(t.column("a").to_numpy(), np.arange(0, 2_000_000, 32))
    assert isinstance(t, pa.Table)
    with rss() as r, heap() as h:
        n = sum(c.num_rows for c in v.chunks())
    governed(b, cap, h)
    assert r.peak < 3 * cap, f"real high-water mark {r.peak:,}"
    assert n == 62_500


@need_fq
def test_fastq_window_near_the_end_seeks_with_the_index(fq_view_factory):
    import pyarrow as pa
    cap = 4 * MB                                             # the file is 62 MB
    b = budget("4MB")
    v = fq_view_factory(b)
    v.select(10_000, 10_010).load()                          # warm up (and verify a checkpoint)
    v.release().select(190_000, 190_500)
    with reads() as io, heap() as h:
        t = v.load()
        v.release()
    governed(b, cap, h)
    assert io.bytes < 2 * MB, f"read {io.bytes:,} bytes"     # it used to read 98 % of the file
    c = v.cost()
    assert c["seek"] == "index" and c["read_bytes"] < 2 * MB
    seen, ref = 0, []
    for batch in g.FastxStream(FQ, "8MB"):
        lo = seen
        seen += batch.num_rows
        if seen > 190_000:
            ref.append(batch.slice(max(190_000 - lo, 0), min(190_500, seen) - max(190_000, lo)))
        if seen >= 190_500:
            break
    ref = pa.Table.from_batches(ref)
    assert t.num_rows == 500
    assert t.equals(ref.cast(t.schema))
    with reads() as io, heap() as h:
        got = pa.concat_tables(list(v.chunks()))
    governed(b, cap, h)
    assert io.bytes < 2 * MB and got.equals(t)


@need_h5ad
def test_h5ad_window_reads_only_its_rows():
    import scipy.sparse as sp
    cap = 2 * MB                                             # X is 24 MB
    b = budget("2MB")
    v = g.view(H5AD, budget=b)
    with h5py.File(H5AD, "r") as f:
        X = sp.csr_matrix((f["X/data"][:], f["X/indices"][:], f["X/indptr"][:]), shape=(20000, 3000))
    v.select(17_000, 17_010)
    with reads() as io, heap() as h:
        got = v.load()
        v.release()
    governed(b, cap, h)
    assert io.bytes < 512 * 1024, f"read {io.bytes:,} bytes for 10 rows"
    assert (got != X[17_000:17_010]).nnz == 0
    v.select(3_001, 19_000).set_zoom(4)                      # stride 16: 1,000 rows of 16,000
    with heap() as h:
        got = v.load()
        assert b.stats()["ram"]["used"] == nbytes(got.data) + got.indices.nbytes + got.indptr.nbytes
        assert v.cost()["total_bytes"] == got.data.nbytes + got.indices.nbytes + got.indptr.nbytes
        v.release()
        parts = list(v.chunks())
        v.release()
    governed(b, cap, h)
    assert (got != X[3_001:19_000:16]).nnz == 0
    assert (sp.vstack(parts) != X[3_001:19_000:16]).nnz == 0
    v.close()


# ===========================================================================
# 4. Lease lifecycle
# ===========================================================================

@need_h5ad
def test_leaving_a_chunk_loop_early_gives_everything_back():
    b = budget("2MB")
    v = g.view(H5AD, budget=b)
    for _X in v.chunks():
        break
    gc.collect()
    assert b.stats()["ram"]["used"] == 0
    it = v.chunks()
    next(it)
    v.release()                                              # release() closes a stream in flight
    assert b.stats()["ram"]["used"] == 0
    with pytest.raises(StopIteration):
        next(it)
    v.close()


@need_h5ad
def test_release_drops_the_readers_lease_as_well():
    b = budget("2MB")
    src = g.open_h5ad(H5AD, budget=b)
    v = g.RowView(src, budget=b, kind="cells").select(0, 1000)
    v.load()
    v.release()
    assert b.stats()["ram"]["used"] == 0
    for _ch in src:                                          # the reader on its own keeps a lease on
        break                                                # the chunk it handed out last
    assert b.stats()["ram"]["used"] > 0
    v.release()                                              # the view gives that back as well
    assert b.stats()["ram"]["used"] == 0
    v.close()


def test_chunks_adopt_the_readers_lease(big_parquet):
    b = budget("8MB")
    v = g.view(big_parquet, budget=b).select(0, 600_000)
    biggest = 0
    for t in v.chunks():
        biggest = max(biggest, t.get_total_buffer_size())
    st = b.stats()["ram"]
    assert st["used"] == 0
    assert st["peak"] <= biggest * 1.25 + 4096, (st["peak"], biggest)   # once, not 2-3 times


@need_gef
def test_closing_a_view_lent_by_a_dataset_leaves_the_dataset_open(gef_copy):
    side = 4000
    ds = g.open_stereoseq(gef=gef_copy, image=IMG, budget=budget("64MB"), image_scale=2048 / side)
    with ds.expression() as v:
        v.cost()
    with ds.morphology() as m:
        m.cost()
    pair = ds.paired((1000, 1400, 1000, 1400), shape=64)
    assert pair["expression"].shape == (64, 64) and pair["morphology"].shape == (64, 64)
    ds.close()


@need_tif
def test_close_closes_the_file_behind_image_and_tenx_views():
    v = g.view(TIF)
    fh = v.a._tf.filehandle
    v.close()
    assert fh.closed
    if os.path.exists(TENX):
        t = g.open_tenx(TENX)
        f = t.m.f
        t.close()
        assert not f.id.valid


@need_h5ad
def test_view_chunk_size_follows_the_shared_rule():
    b = budget("8MB")
    v = g.RowView(g.open_h5ad(H5AD, budget=b), budget=b, kind="cells", chunk_bytes="300KB")
    assert v._chunk_cap() == 300 * 1024                      # a chosen size is honoured
    w = g.RowView(g.open_h5ad(H5AD, budget=b), budget=b, kind="cells")
    assert w._chunk_cap() == 2 * MB                          # the floor rule: a quarter of 8 MB
    v.close()
    w.close()


@need_gef
def test_stereo_cost_counts_the_chunks_it_streams(gef_ro):
    cap = 2 * MB                                             # the whole chip at bin1: 35 MB of rows
    b = budget("2MB")
    v = g.open_gef(gef_ro, budget=b).use_bin(1)
    c = v.cost()
    assert c["fits_whole"] is False and c["fits"] is True
    n = rows = 0
    with heap() as h:
        for ch in v.chunks():
            n += 1
            rows += ch["x"].size
    governed(b, cap, h)
    assert n == c["chunks"] > 1
    assert rows == v.gef.levels[1].n_obs
    v.close()


# ===========================================================================
# 5. Navigation
# ===========================================================================

@need_fq
def test_zoom_in_and_out_are_inverses(fq_view_factory):
    v = fq_view_factory().reset()
    full, n = v.window, v.extent.n
    v.zoom_in(2).zoom_out(2)
    assert v.zoom == 0 and v.window == full and v.cost()["units"] == n
    v.zoom_out(2)
    assert v.zoom == 0                                       # the window could not widen
    v.select(50_000, 150_000).set_zoom(3)
    w0 = v.window
    v.zoom_in(2)
    assert v.zoom == 2 and v.window == (75_000, 125_000)
    v.zoom_out(2)
    assert v.zoom == 3 and v.window == w0


@need_tif
def test_two_d_zoom_round_trip():
    v = g.view(TIF).select(1000, 3000, 2000, 4000).set_zoom(2)
    w0 = v.window
    v.zoom_out(2)
    assert v.zoom == 3 and v.window == (0, 4000, 1000, 5000)
    v.zoom_in(2)
    assert v.zoom == 2 and v.window == w0
    v.close()


@need_fq
def test_pan_and_edge_zoom_keep_the_width(fq_view_factory):
    v = fq_view_factory().select(0, 1000)
    for _ in range(4):
        v.pan(0.5)
        assert v.window[1] - v.window[0] == 1000
    assert v.window == (2000, 3000)
    v.select(v.extent.n - 1000, v.extent.n).pan(0.5)
    assert v.window == (v.extent.n - 1000, v.extent.n)       # at the edge it stays, uncut
    v.reset().zoom_in(2, center=0)
    assert v.window == (0, v.extent.n // 2)                  # half the span, not a quarter


@need_fq
def test_auto_zoom_raises_when_nothing_fits(fq_view_factory):
    v = fq_view_factory(budget("1KB")).reset().set_zoom(2)
    with pytest.raises(g.CostError, match="cheapest"):
        v.auto_zoom()
    assert v.zoom == 2                                        # left where it was
    v2 = fq_view_factory(budget("1MB")).reset().auto_zoom()
    c = v2.cost()
    assert c["fits_whole"] is True and c["peak_bytes"] <= 0.8 * MB


@need_fq
def test_decimation_is_one_grid_for_load_and_chunks(fq_view_factory):
    v = fq_view_factory(budget("4MB")).select(1234, 9877).set_zoom(2)
    ids = v.load().column("id").to_pylist()
    v.release()
    want = [f"read{i}:" for i in range(1234, 9877, 4)]
    assert [s.split("lane")[0] for s in ids] == want
    got = [s for t in v.chunks() for s in t.column("id").to_pylist()]
    assert got == ids                                        # 1 MB chunks split the window
    for cb in ("64KB", "300KB"):
        v.chunk_bytes = cb
        got = [s for t in v.chunks() for s in t.column("id").to_pylist()]
        assert got == ids


def test_parquet_decimation_ignores_the_chunk_size(big_parquet):
    v = g.view(big_parquet, budget=budget("64MB")).select(777, 512_345).set_zoom(4)
    a = v.load().column("a").to_numpy()
    assert np.array_equal(a, np.arange(777, 512_345, 16))
    v.release()
    for cb in ("200KB", "3MB"):
        v.chunk_bytes = cb
        got = np.concatenate([t.column("a").to_numpy() for t in v.chunks()])
        assert np.array_equal(got, a)


def test_image_tiles_fit_together_into_load():
    v = g.view(IMG).select(101, 1900, 333, 1777).set_zoom(2)
    full = v.load()
    y0, _, x0, _ = v.window
    canvas = np.zeros_like(full)
    for tile, (a, _b, c, _d) in v.tiles():
        ya, xa = (a - y0) // 4, (c - x0) // 4
        canvas[ya:ya + tile.shape[0], xa:xa + tile.shape[1]] = tile
    assert np.array_equal(canvas, full)
    assert sum(t.size for t in v.chunks()) == full.size
    v.close()


# ===========================================================================
# 6. Wrong answers
# ===========================================================================

@need_gef
def test_gene_profile_ranks_genes_inside_the_window(gef_ro):
    w = (2500, 3600, 300, 1500)
    v = g.open_gef(gef_ro, budget=budget("64MB")).use_bin(10).select(*w)
    ref = ref_window(gef_ro, 10, w)
    mid = np.bincount(ref["gene"], weights=ref["count"])
    names = v.gef.genes(10)
    top = np.argsort(-mid, kind="stable")[:3]
    assert v.gene_profile(3) == [(str(names[i]), int(mid[i])) for i in top]
    v.close()


@need_gef
def test_overview_is_of_the_window(gef_ro):
    v = g.open_gef(gef_ro, budget=budget("64MB"))
    whole = v.reset().overview(64).series["thumbnail"]
    w = (1000, 2000, 1200, 2200)                             # aligned to every bin: MID is level-free
    sub = v.select(*w).overview(64).series["thumbnail"]
    assert whole.shape != sub.shape or not np.array_equal(whole, sub)
    assert float(sub.sum()) == float(ref_window(gef_ro, 1, w)["count"].sum())
    v.close()


@need_gef
def test_coarse_bins_that_overlap_the_window_are_kept(gef_ro):
    gf = g.GefFile(gef_ro)
    w = (1010, 1100, 1010, 1100)                             # inside one bin200, off its origin
    d = gf.read_window(w, 200)
    ref = ref_window(gef_ro, 200, w)
    assert d["x"].size == ref.size > 0
    same_rows(d, ref)
    gf.close()


@need_gef
def test_paired_rasters_share_one_grid(gef_copy):
    ds = g.open_stereoseq(gef=gef_copy, image=IMG, budget=budget("64MB"), image_scale=2048 / 4000)
    pair = ds.paired((1600, 3000, 1400, 2100), shape=256)
    assert pair["expression"].shape == (256, 256)
    assert pair["morphology"].shape == (256, 256)
    ds.close()


@need_gef
def test_extent_is_half_open_for_gef_and_gem(gef_ro, tmp_path):
    gf = g.GefFile(gef_ro)
    assert gf.extent == (0, 4000, 0, 4000)                   # maxX/maxY = 3999 are inclusive; extent is not
    assert int(ref_window(gef_ro, 1, gf.extent)["x"].max()) == 3999
    gf.close()
    if os.path.exists(GEM):
        gem = tmp_path / "chip.gem"
        shutil.copy(GEM, gem)
        gv = g.open_gem(str(gem))
        assert gv._full_window() == (0, 4000, 0, 4000)       # inclusive max + 1: the same window


@need_gef
def test_rasterize_puts_every_bin_in_its_own_pixel(gef_ro):
    gf = g.GefFile(gef_ro)
    img = gf.rasterize((0, 4000, 0, 4000), 10, shape=400)    # one pixel per bin10
    with h5py.File(gef_ro, "r") as f:
        e = f["geneExp/bin10/expression"][:]
    ref = np.zeros((400, 400))
    np.add.at(ref, (e["y"] // 10, e["x"] // 10), e["count"])
    assert np.array_equal(img, ref)                          # x = 290 used to land in pixel 28
    gf.close()


# ===========================================================================
# 7. The GEF sidecar
# ===========================================================================

@pytest.fixture
def readonly(monkeypatch):
    """Make a directory tree read-only for geminga.gefindex — a NAS share or a read-only mount.
    The tests run as root, so the permission bits alone would not stop a write."""
    import builtins
    from geminga import gefindex
    roots = []
    real_access, real_open, real_makedirs = os.access, builtins.open, os.makedirs

    def inside(p):
        p = os.path.abspath(str(p))
        return any(p == r or p.startswith(r + os.sep) for r in roots)

    def access(p, mode, *a, **k):
        if mode & os.W_OK and inside(p):
            return False
        return real_access(p, mode, *a, **k)

    def opener(p, mode="r", *a, **k):
        if any(c in mode for c in "wax+") and inside(p):
            raise PermissionError(13, "read-only", str(p))
        return real_open(p, mode, *a, **k)

    def makedirs(p, *a, **k):
        if inside(p) and not os.path.isdir(p):
            raise PermissionError(13, "read-only", str(p))
        return real_makedirs(p, *a, **k)

    monkeypatch.setattr(gefindex.os, "access", access)
    monkeypatch.setattr(gefindex.os, "makedirs", makedirs)
    monkeypatch.setattr(gefindex, "open", opener, raising=False)
    return roots


@need_gef
def test_sidecar_lookup_creates_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    d = tmp_path / "data"
    d.mkdir()
    p = str(d / "chip.gef")
    shutil.copy(GEF, p)
    gf = g.GefFile(p)
    assert gf.spatial_index is None
    assert gf.read_plan((0, 500, 0, 500), 1)["path"] == "full-level"
    assert sorted(os.listdir(d)) == ["chip.gef"]             # no sidecar directory, no probe file
    assert not (tmp_path / "home").exists()
    gf.close()


@need_gef
def test_a_read_only_gef_uses_the_sidecar_beside_it(tmp_path, monkeypatch, readonly):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    d = tmp_path / "share"
    d.mkdir()
    p = str(d / "chip.gef")
    shutil.copy(GEF, p)
    g.GefFile(p).build_spatial_index(levels=[20])            # written while the share was writable
    readonly.append(str(d))
    gf = g.GefFile(p)
    assert gf.spatial_index is not None
    assert gf.read_plan((0, 500, 0, 500), 20)["path"] == "spatial-index"
    assert not (tmp_path / "home").exists()
    gf.close()
    q = str(tmp_path / "share2")
    os.makedirs(q)
    shutil.copy(GEF, os.path.join(q, "chip.gef"))
    readonly.append(q)
    idx = g.GefFile(os.path.join(q, "chip.gef")).build_spatial_index(levels=[200])
    assert idx.dir.startswith(str(tmp_path / "home" / ".geminga" / "gefidx"))
    assert g.GefFile(os.path.join(q, "chip.gef")).spatial_index is not None


@need_gef
def test_building_the_index_honours_the_budget(gef_copy):
    pa = pytest.importorskip("pyarrow")
    pa.array(np.zeros(1, np.uint32))                         # pyarrow imports pandas on first use:
    cap = 2 * MB                                             # ~28 MB of modules, not data. bin1: 29.5 MB
    b = budget("2MB")
    gf = g.GefFile(gef_copy)
    with heap() as h:
        idx = gf.build_spatial_index(levels=[1], budget=b)
    governed(b, cap, h)
    assert b.stats()["ram"]["grants"] > 0                     # it did lease what it read
    assert idx.has(1)
    w = (1000, 1400, 1200, 1600)
    same_rows(g.GefFile(gef_copy).read_window(w, 1), ref_window(gef_copy, 1, w))


@need_gef
def test_a_missing_level_file_falls_back_to_the_block_read(gef_copy):
    g.GefFile(gef_copy).build_spatial_index(levels=[20, 100])
    os.remove(os.path.join(gef_copy + ".gemidx", "bin20.parquet"))
    gf = g.GefFile(gef_copy)
    w = (1000, 1400, 1200, 1600)
    assert gf.read_plan(w, 20)["path"] == "full-level"
    assert gf.read_plan(w, 100)["path"] == "spatial-index"
    same_rows(gf.read_window(w, 20), ref_window(gef_copy, 20, w))
    gf.close()


@need_gef
def test_rebuilding_one_level_keeps_the_others(gef_copy):
    g.GefFile(gef_copy).build_spatial_index(levels=[10, 200])
    g.GefFile(gef_copy).build_spatial_index(levels=[200], force=True)
    with open(os.path.join(gef_copy + ".gemidx", "index.json")) as fh:
        meta = json.load(fh)
    assert sorted(meta["levels"]) == ["10", "200"]
    assert g.SpatialIndex(gef_copy).levels == [10, 200]


def test_index_gem_rebuilds_for_another_grid(tmp_path):
    if not os.path.exists(GEM):
        pytest.skip("no GEM fixture")
    p = str(tmp_path / "chip.gem")
    shutil.copy(GEM, p)
    assert g.index_gem(p, grid=256).grid == 256
    assert g.index_gem(p, grid=128).grid == 128              # it used to hand back the 256 sidecar
    assert g.index_gem(p).grid == 128                         # no grid asked: whatever is there


def test_view_and_open_accept_paths():
    if not (os.path.exists(TIF) and os.path.exists(H5AD)):
        pytest.skip("fixtures missing")
    v = g.view(pathlib.Path(TIF))
    assert isinstance(v, g.ImageView)
    v.close()
    r = g.view(pathlib.Path(H5AD))
    assert isinstance(r, g.RowView)
    r.close()
    h = g.open(pathlib.Path(H5AD))
    assert isinstance(h, g.H5adReader)
    h.close()
