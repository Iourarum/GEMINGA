"""Compressed resident blocks (codec + store) and the shared-GPU reservation server."""
import importlib
import time

import numpy as np
import pytest
import scipy.sparse as sp

import geminga as g
from geminga import codec as C

R = importlib.import_module("geminga.reserve")      # `geminga.reserve` the attribute is the function

rng = np.random.default_rng(0)


# ------------------------------------------------------------------ bit-packing codec ----
@pytest.mark.parametrize("name, arr", [
    ("counts", rng.poisson(2.0, 100_003).astype(np.int32) + 1),
    ("sorted", np.sort(rng.integers(0, 30000, 50_000)).astype(np.int32)),
    ("negative", rng.integers(-1000, 1000, 10_001).astype(np.int32)),
    ("extremes", np.array([0, 2**31 - 1, -2**31, 5, -1, 1], np.int32)),
    ("uint32", np.array([0, 2**32 - 1, 7, 2**31], np.uint32)),
    ("int16", rng.integers(-300, 300, 999).astype(np.int16)),
    ("uint8", rng.integers(0, 255, 1000).astype(np.uint8)),
    ("one", np.array([42], np.int32)),
    ("empty", np.zeros(0, np.int32)),
    ("int64-small", rng.integers(0, 1 << 20, 3000).astype(np.int64)),
    ("2d", rng.integers(0, 50, (64, 33)).astype(np.int32)),
    ("rows", np.concatenate([np.sort(rng.integers(0, 2000, rng.integers(5, 300))) for _ in range(500)]).astype(np.int32)),
])
def test_bitpack_roundtrips_exactly(name, arr):
    bp = C.BitpackCodec()
    enc = bp.encode(arr)
    out = bp.decode(enc)
    assert out.dtype == arr.dtype and out.shape == arr.shape
    assert np.array_equal(out, arr)
    assert enc.raw_bytes == arr.nbytes and enc.nbytes >= 0
    assert enc.meta["n"] == arr.size and enc.meta["n_blocks"] == (arr.size + C.BLOCK - 1) // C.BLOCK


def test_bitpack_ratios_are_what_omics_integers_give():
    bp = C.BitpackCodec()
    counts = rng.poisson(2.0, 500_000).astype(np.int32) + 1          # UMI-like
    idx = np.sort(rng.integers(0, 30000, 300_000)).astype(np.int32)   # a sorted index column
    assert bp.encode(counts).ratio > 6
    assert bp.encode(idx).ratio > 8
    e = bp.encode(counts)
    assert 2 < e.meta["mean_width"] < 5 and e.meta["delta_blocks"] <= e.meta["n_blocks"]


def test_bitpack_refuses_floats_and_out_of_range():
    bp = C.BitpackCodec()
    with pytest.raises(TypeError, match="integers"):
        bp.encode(np.zeros(4, np.float32))
    with pytest.raises(ValueError, match="int32 range"):
        bp.encode(np.array([1 << 40], np.int64))
    with pytest.raises(ValueError, match="uint32"):
        bp.encode(np.array([1 << 40], np.uint64))


def test_bitpack_block_choice_is_per_block():
    bp = C.BitpackCodec()
    a = np.concatenate([np.arange(0, 128 * 4, 4, dtype=np.int32),      # delta wins: constant step
                        rng.integers(0, 8, 128).astype(np.int32)])      # FOR wins: random small
    e = bp.encode(a)
    table = np.asarray(e.payload["table"]).reshape(-1, 4)
    modes = (table[:, 2] >> 8) & 0xFF
    assert modes.tolist() == [1, 0]
    assert np.array_equal(bp.decode(e), a)


@pytest.fixture(scope="module")
def device():
    wgpu = pytest.importorskip("wgpu")
    try:
        adapter = wgpu.gpu.request_adapter_sync()
        return adapter.request_device_sync()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"no wgpu adapter: {e}")


@pytest.mark.parametrize("arr", [
    rng.poisson(2.0, 20_001).astype(np.int32) + 1,
    np.sort(rng.integers(0, 30000, 9_000)).astype(np.int32),
    rng.integers(-1000, 1000, 2_000).astype(np.int32),
    np.array([0, 2**31 - 1, -2**31, 5], np.int32),
    np.array([0, 2**32 - 1, 7], np.uint32),
    np.zeros(0, np.int32),
])
def test_bitpack_gpu_decode_matches_the_host(device, arr):
    bp = C.BitpackCodec()
    enc = bp.encode(arr)
    buf = bp.decode_device(enc, device)
    words = np.frombuffer(device.queue.read_buffer(buf), np.uint32)[: enc.meta["n"]]
    out = C._lanes_to_dtype(words, enc.dtype, enc.shape)
    assert np.array_equal(out, arr)
    up = bp.upload(enc, device)                                      # the store's path: payload on the device
    assert up.on_device and up.nbytes == enc.nbytes
    buf2 = bp.decode_device(up, device)
    words2 = np.frombuffer(device.queue.read_buffer(buf2), np.uint32)[: enc.meta["n"]]
    assert np.array_equal(C._lanes_to_dtype(words2, enc.dtype, enc.shape), arr)
    with pytest.raises(ValueError, match="device"):
        bp.decode(up)


# -------------------------------------------------------------------- bytes + sparse ----
def test_bytes_codec_roundtrips_floats_and_strings():
    bc = C.BytesCodec()
    x = rng.standard_normal(10_000).astype(np.float32)
    e = bc.encode(x)
    assert np.array_equal(bc.decode(e), x) and e.codec.startswith("bytes-")
    s = np.array(["a", "bb", "ccc"])
    assert np.array_equal(bc.decode(bc.encode(s)), s)
    with pytest.raises(ValueError, match="backend"):
        C.BytesCodec(backend="definitely-not-a-backend")


def test_bytes_codec_backend_choice():
    assert C.BytesCodec("zlib").backend == "zlib"
    assert C.BytesCodec().backend in ("zstd", "lz4", "zlib")


def test_csr_pack_roundtrip_integer_and_float_data():
    X = sp.random(2000, 500, density=0.05, format="csr", dtype=np.float32, random_state=1)
    X.data = np.round(X.data * 9) + 1                                 # integer counts in a float container
    pack = C.encode_csr(X)
    Y = C.decode_csr(pack)
    assert (X != Y).nnz == 0 and Y.dtype == X.dtype and pack["raw_bytes"] / pack["nbytes"] > 3
    assert pack["parts"]["data"].codec == "bitpack"
    F = sp.random(300, 200, density=0.1, format="csc", dtype=np.float64, random_state=2)
    pf = C.encode_csr(F)
    assert pf["format"] == "csc" and pf["parts"]["data"].codec.startswith("bytes-")
    G = C.decode_csr(pf)
    assert sp.isspmatrix_csc(G) and abs(F - G).max() == 0


# ---------------------------------------------------------------------------- store ----
def test_store_accounts_packed_bytes_and_scratch_leases():
    b = g.Budget(ram="32MB")
    st = g.compressed_store(b, "ram")
    counts = rng.poisson(2.0, 1_000_000).astype(np.int32) + 1          # 4 MB raw
    e = st.put("counts", counts)
    assert e.packed_bytes < counts.nbytes / 5 and b.stats()["ram"]["used"] == e.packed_bytes
    X = sp.random(3000, 1000, density=0.05, format="csr", dtype=np.float32, random_state=3)
    X.data = np.round(X.data * 5) + 1
    ex = st.put("X", X)
    assert ex.kind == "csr" and b.stats()["ram"]["used"] == e.packed_bytes + ex.packed_bytes
    with st.get("counts") as c:
        assert np.array_equal(c, counts)
        assert b.stats()["ram"]["used"] == e.packed_bytes + ex.packed_bytes + counts.nbytes   # scratch while held
    assert b.stats()["ram"]["used"] == e.packed_bytes + ex.packed_bytes
    with st.get("X") as Y:
        assert (X != Y).nnz == 0
    r = st.report()
    assert r["blocks"] == 2 and r["ratio"] > 3 and r["effective_capacity"] > r["capacity"] * 3
    with pytest.raises(KeyError):
        st.put("counts", counts)
    st.put("counts", counts[:10], overwrite=True)
    st.evict("X")
    assert "X" not in st and b.stats()["ram"]["used"] == st.entry("counts").packed_bytes
    st.clear()
    assert len(st) == 0 and b.stats()["ram"]["used"] == 0 and b.stats()["ram"]["denials"] == 0


def test_store_refuses_what_does_not_fit_packed():
    b = g.Budget(ram="64KB", timeout_s=1)
    st = g.compressed_store(b, "ram")
    st.put("ok", rng.poisson(2.0, 30_000).astype(np.int32) + 1)      # ~20 KB packed
    with pytest.raises(MemoryError):
        st.put("big", rng.integers(0, 1 << 30, 100_000).astype(np.int32))   # incompressible, 400 KB
    assert "big" not in st and b.stats()["ram"]["used"] == st.entry("ok").packed_bytes


def test_store_floats_go_through_the_bytes_codec():
    b = g.Budget(ram="16MB")
    st = g.compressed_store(b, "ram")
    x = np.repeat(np.float32(1.5), 100_000)
    e = st.put("x", x)
    assert e.codec.startswith("bytes-") and e.packed_bytes < x.nbytes / 10
    with st.get("x") as y:
        assert np.array_equal(y, x)


def test_store_vram_tier_keeps_blocks_on_the_device(device):
    b = g.Budget(ram="16MB", vram="2MB")
    st = g.compressed_store(b, "vram", device=device)
    counts = rng.poisson(2.0, 200_000).astype(np.int32) + 1            # 800 KB raw, ~100 KB packed
    e = st.put("counts", counts)
    assert e.on_device and b.stats()["vram"]["used"] == e.packed_bytes < 160_000
    with st.get_device("counts") as da:
        assert b.stats()["vram"]["used"] == e.packed_bytes + counts.nbytes      # decoded buffer leased in vram
        assert np.array_equal(da.read(), counts)
    assert b.stats()["vram"]["used"] == e.packed_bytes
    with st.get("counts") as c:                                          # host copy: decoded on device, read back
        assert np.array_equal(c, counts)
    big = rng.poisson(2.0, 3_000_000).astype(np.int32) + 1               # 12 MB raw, ~1.5 MB packed: fits packed
    st.put("big", big)
    with pytest.raises(MemoryError):                                     # ...but not decoded in 2 MB of vram
        st.get_device("big")
    assert b.stats()["vram"]["used"] == st.entry("counts").packed_bytes + st.entry("big").packed_bytes
    X = sp.random(500, 300, density=0.1, format="csr", dtype=np.float32, random_state=4)
    X.data = np.round(X.data * 3) + 1
    st.put("X", X)
    with st.get("X") as Y:
        assert (X != Y).nnz == 0
    with pytest.raises(ValueError, match="sparse"):
        st.get_device("X")
    st.clear()
    assert b.stats()["vram"]["used"] == 0


def test_ratio_report_lists_every_array():
    r = C.ratio_report({"c": rng.poisson(2.0, 10_000).astype(np.int32), "f": rng.standard_normal(1000).astype(np.float32)})
    assert set(r) == {"c", "f"} and r["c"]["codec"] == "bitpack" and r["f"]["codec"].startswith("bytes-")


# ---------------------------------------------------------------------- reservations ----
@pytest.fixture
def server():
    srv = R.ReservationServer({"gpu0": "24GB"}, host="127.0.0.1", port=0,
                              policy=R.Policy(ttl_s=2.0, idle_s=0.6, max_share=0.5)).start()
    yield srv
    srv.stop()


def test_reserve_grants_partial_and_queues_fairly(server):
    a = g.reserve(server.url, vram="16GB", user="alice", wait=3, ram="1GB")
    assert a.bytes == 16 << 30 and a.budget.stats()["vram"]["capacity"] == 16 << 30
    assert a.budget.stats()["ram"]["capacity"] == 1 << 30
    b = g.reserve(server.url, vram="16GB", min_vram="4GB", user="bob", wait=3)
    assert b.bytes == 8 << 30                                             # what was free, above his minimum
    st = server.status()["devices"]["gpu0"]
    assert st["free"] == 0 and {h["user"] for h in st["holders"]} == {"alice", "bob"}
    with pytest.raises(TimeoutError, match="alice"):                       # nothing free, nobody idle yet
        g.reserve(server.url, vram="4GB", user="carol", wait=0.3, poll=0.1)
    a.release(); b.release()
    assert server.status()["devices"]["gpu0"]["granted"] == 0


def test_idle_slices_are_reclaimed_only_when_needed_and_active_ones_never(server):
    a = g.reserve(server.url, vram="16GB", user="alice", wait=3)
    b = g.reserve(server.url, vram="8GB", user="bob", wait=3)
    lease = b.budget.acquire("vram", "1GB")                                # bob is working
    b.renew()
    time.sleep(0.8)                                                        # alice has been idle > idle_s
    t0 = time.time()
    c = g.reserve(server.url, vram="8GB", user="carol", wait=5, poll=0.2)  # takes alice's slice
    assert time.time() - t0 < 3 and c.bytes == 8 << 30
    assert not a.renew() and a.lost and "reclaimed" in a.reason
    with pytest.raises(R.ReservationLost):
        a.check()
    assert b.renew() and not b.lost                                        # active: untouched
    holders = {h["user"]: h["bytes"] for h in server.status()["devices"]["gpu0"]["holders"]}
    assert holders == {"bob": 8 << 30, "carol": 8 << 30}
    d = g.reserve(server.url, vram="4GB", user="dave", wait=3)             # 8 GB free: no reclaim
    assert not c.lost and d.bytes == 4 << 30
    events = [e["event"] for e in server.status()["events"]]
    assert events.count("reclaimed-idle") == 1
    lease.release()
    for r in (b, c, d):
        r.release()


def test_grants_expire_without_a_heartbeat(server):
    code, r = R._post(server.url, "/reserve", {"user": "ghost", "bytes": 1 << 30})
    assert code == 200 and r["granted"]
    time.sleep(2.3)                                                        # ttl_s is 2
    assert server.status()["devices"]["gpu0"]["granted"] == 0
    code, r = R._post(server.url, "/renew", {"token": r["token"]})
    assert code == 410 and "expired" in r["reason"]


def test_share_cap_applies_while_others_wait(server):
    a = g.reserve(server.url, vram="8GB", user="alice", wait=3)
    time.sleep(0.8)                                                        # alice idle
    b = g.reserve(server.url, vram="24GB", min_vram="1GB", user="bob", wait=3)
    assert b.bytes == 16 << 30                                             # what was free, nobody waiting -> full cap
    # carol waits; now bob's and alice's shares are capped at 50% for new grants, and idle alice yields
    c = g.reserve(server.url, vram="8GB", user="carol", wait=3, poll=0.2)
    assert c.bytes == 8 << 30
    a.release(); b.release(); c.release()


def test_bad_requests_do_not_take_the_server_down(server):
    code, r = R._post(server.url, "/reserve", {"user": "x", "bytes": 1 << 20, "device": "nope"})
    assert code == 400 and "no device" in r["error"]
    code, r = R._post(server.url, "/renew", {})
    assert code == 400
    code, r = R._post(server.url, "/nope", {})
    assert code == 404
    assert server.status()["devices"]["gpu0"]["granted"] == 0
    assert "gpu0" in R.format_status(server.status())


def test_reservation_store_lives_inside_the_slice(server):
    res = g.reserve(server.url, vram="1MB", user="eve", wait=3, ram="8MB")
    st = res.store(tier="ram")                                              # the host-side store, same budget
    st.put("c", rng.poisson(2.0, 50_000).astype(np.int32) + 1)
    assert st.report()["capacity"] == res.budget.stats()["ram"]["capacity"]
    with pytest.raises(ValueError, match="wgpu device"):
        res.store()                                                         # vram tier needs a device
    res.release()
    assert res.lost and repr(res).endswith("lost)")


# ------------------------------------------------------------ plug-in contract & tables ----
class _Rle:
    name = "rle-test"

    def encode(self, arr):
        a = np.asarray(arr)
        if a.dtype.kind not in "iu" or a.dtype.itemsize > 4:
            raise TypeError("integers only")
        v = a.reshape(-1)
        if v.size:
            change = np.flatnonzero(np.diff(v) != 0) + 1
            starts = np.concatenate([[0], change])
            lengths = np.diff(np.concatenate([starts, [v.size]]))
            runs = np.stack([v[starts].astype(np.int64) & 0xFFFFFFFF, lengths], axis=1).astype(np.uint32)
        else:
            runs = np.zeros((0, 2), np.uint32)
        return C.Encoded(self.name, a.dtype.str, tuple(a.shape), int(runs.nbytes), int(a.nbytes),
                         {"runs": runs}, {"n": int(v.size)})

    def decode(self, enc):
        runs = np.asarray(enc.payload["runs"], np.uint32).reshape(-1, 2)
        lanes = np.repeat(runs[:, 0], runs[:, 1]) if len(runs) else np.zeros(0, np.uint32)
        return C._lanes_to_dtype(lanes, enc.dtype, enc.shape)


class _Broken(_Rle):
    name = "broken"

    def decode(self, enc):
        out = super().decode(enc)
        return out + 1 if out.size else out


def test_check_codec_passes_the_builtins_and_a_conforming_plugin():
    for name in ("bitpack", "bytes"):
        rep = C.check_codec(name)
        assert not rep["failed"] and rep["passed"]
    rep = C.check_codec(_Rle())
    assert "counts int32" in rep["passed"] and any(d.startswith("float32") for d in rep["declined"])


def test_check_codec_names_the_case_a_bad_codec_fails():
    with pytest.raises(C.CodecError, match="not the inverse"):
        C.check_codec(_Broken())
    rep = C.check_codec(_Broken(), strict=False)
    assert rep["failed"] and "counts int32" not in rep["passed"]


def test_codecs_are_found_by_name_and_refused_when_unknown():
    C.register_codec("rle-test", _Rle)
    assert "rle-test" in C.available_codecs() and C.get_codec("rle-test").name == "rle-test"
    st = g.compressed_store(g.Budget(ram="4MB"), "ram", codec="rle-test")
    flags = (rng.random(100_000) < 0.03).astype(np.int32)
    e = st.put("flags", flags)
    assert e.codec == "rle-test" and e.packed_bytes < flags.nbytes / 5
    with st.get("flags") as f:
        assert np.array_equal(f, flags)
    with pytest.raises(KeyError, match="no codec"):
        C.get_codec("definitely-not-registered")
    with pytest.raises(TypeError, match="look like a codec"):
        C.get_codec(object())
    st2 = g.compressed_store(g.Budget(ram="4MB", vram="4MB"), "ram", codec="rle-test", device=object())
    st2.put("x", flags)
    with pytest.raises(ValueError, match="no device decoder"):
        st2.get_device("x")


def test_tables_pack_per_column_and_come_back_exactly():
    pa = pytest.importorskip("pyarrow")
    b = g.Budget(ram="32MB")
    st = g.compressed_store(b, "ram")
    t = pa.table({"cell": np.arange(50_000, dtype=np.int32), "count": (rng.poisson(2, 50_000) + 1).astype(np.int32),
                  "x": rng.standard_normal(50_000).astype(np.float32),
                  "barcode": [f"ACGT{i:06d}" for i in range(50_000)], "ok": rng.random(50_000) < 0.5})
    e = st.put("t", t)
    assert e.kind == "table" and e.shape == (50_000, 5) and e.packed_bytes < e.raw_bytes / 2
    assert b.stats()["ram"]["used"] == e.packed_bytes
    with st.get_table("t") as back:
        assert back.equals(t)
        assert b.stats()["ram"]["used"] == e.packed_bytes + e.raw_bytes
    assert b.stats()["ram"]["used"] == e.packed_bytes
    with pytest.raises(ValueError, match="nulls"):
        st.put("n", pa.table({"a": pa.array([1, None, 3])}))
    assert "n" not in st
    with pytest.raises(ValueError, match="not a table"):
        st.put("arr", np.arange(10, dtype=np.int32))
        st.get_table("arr")


def test_tables_register_with_duckdb_and_release_their_lease():
    duckdb = pytest.importorskip("duckdb")
    pa = pytest.importorskip("pyarrow")
    b = g.Budget(ram="16MB")
    st = g.compressed_store(b, "ram")
    st.put("hits", pa.table({"gene": rng.integers(0, 100, 20_000).astype(np.int32),
                             "score": rng.standard_normal(20_000).astype(np.float32)}))
    con = duckdb.connect()
    holds = st.duckdb(con)
    n, = con.execute("select count(*) from hits where score > 0").fetchone()
    assert 8_000 < n < 12_000 and b.stats()["ram"]["used"] > st.entry("hits").packed_bytes
    for h in holds.values():
        h.release()
    assert b.stats()["ram"]["used"] == st.entry("hits").packed_bytes


def test_device_array_native_layout_reads_the_dtype_as_stored(device):
    x = rng.standard_normal(1000).astype(np.float32)
    import wgpu
    buf = device.create_buffer_with_data(data=x, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    da = C.DeviceArray(buf, x.dtype.str, x.shape, x.size, device, layout="native")
    assert np.array_equal(da.read(), x)
