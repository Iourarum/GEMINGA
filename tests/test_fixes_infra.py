"""Regression tests for the infrastructure fixes listed in CHANGES-C.md.

Most of these were written against the code *before* the fix and failed on it; the docstring or
comment on a test says how. A few guard behaviour the fix had to keep, or a hazard found while
fixing (lock order, pins surviving a merge, a stalled body). Servers are local mocks (nothing
touches the network), views are small fakes where the real chip is not the point, and each test
gets a private ``HOME`` so the sidecars and fallback manifests under ``~/.geminga`` cannot leak
between tests or into the user's own.
"""
from __future__ import annotations

import functools
import http.server
import importlib
import io
import json
import logging
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

import geminga as g

# `geminga.vault`, `.staging` and `.devices` are shadowed on the package by the functions of the
# same name, so the modules have to be fetched by their full dotted name.
vault_mod = importlib.import_module("geminga.vault")
devices_mod = importlib.import_module("geminga.devices")

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")
GEF = os.path.join(FIX, "chip.gef")
DICOM_DIR = os.path.join(FIX, "ms", "dicom", "P04_FLAIR")
needs_gef = pytest.mark.skipif(not os.path.exists(GEF), reason="run the fixture scripts first")
needs_dicom = pytest.mark.skipif(not os.path.isdir(DICOM_DIR), reason="run the fixture scripts first")

MB = 1024 ** 2


@pytest.fixture(autouse=True)
def _private_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture(scope="module")
def chip(tmp_path_factory):
    """The fixture chip, copied: opening a GEF can leave a sidecar directory beside it, and the
    shared fixtures are not where tests should be leaving things."""
    d = tmp_path_factory.mktemp("chip")
    shutil.copy(GEF, d / "chip.gef")
    return str(d / "chip.gef")


def _rand(n: int, seed: int = 1) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


def _touch(path, n: int = 16) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * n)
    return str(path)


# ===========================================================================================
# 1. RemoteFile: wrong or short bytes
# ===========================================================================================

class _Origin:
    """A throwaway HTTP origin serving one resource at /blob.bin, with a switch for each way a
    real server lets a range reader down."""

    def __init__(self, data: bytes, *, ranges=True, head_length=True, head_status=200,
                 get_length=True, truncate=0, stall=0, shift=0, total=None, listing=None):
        self.data, self.ranges = data, ranges
        self.stall = stall            # ranged replies still to stop half way, connection held open
        self.head_length, self.head_status, self.get_length = head_length, head_status, get_length
        self.truncate = truncate      # ranged replies still to cut in half; -1 = every one
        self.shift, self.total = shift, total
        self.listing = listing        # HTML served for any URL ending in "/"
        self.log: list = []
        origin = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_HEAD(self):
                origin._head(self)

            def do_GET(self):
                origin._get(self)

        class Server(http.server.ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                pass                   # clients hang up mid-body on purpose

        self.httpd = Server(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02},
                         daemon=True).start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    @property
    def url(self) -> str:
        return self.base + "blob.bin"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def _head(self, h):
        self.log.append(("HEAD", None))
        if self.head_status != 200:
            h.send_error(self.head_status)
            return
        h.send_response(200)
        if self.head_length:
            h.send_header("Content-Length", str(len(self.data)))
        h.end_headers()

    def _get(self, h):
        rng = h.headers.get("Range")
        self.log.append(("GET", rng))
        try:
            if self.listing is not None and h.path.endswith("/"):
                body = self.listing.encode()
                h.send_response(200)
                h.send_header("Content-Length", str(len(body)))
                h.end_headers()
                h.wfile.write(body)
                return
            m = re.fullmatch(r"bytes=(\d+)-(\d*)", rng or "")
            if not (self.ranges and m):
                h.send_response(200)
                if self.get_length:
                    h.send_header("Content-Length", str(len(self.data)))
                h.end_headers()
                h.wfile.write(self.data)
                return
            a = int(m.group(1))
            b = min(int(m.group(2) or len(self.data) - 1), len(self.data) - 1)
            if a >= len(self.data):
                h.send_response(416)
                h.send_header("Content-Range", f"bytes */{len(self.data)}")
                h.end_headers()
                return
            body = self.data[a:b + 1]
            first = a + self.shift
            h.send_response(206)
            h.send_header("Content-Range",
                          f"bytes {first}-{first + len(body) - 1}/{self.total or len(self.data)}")
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            if self.stall and len(body) > 1:
                if self.stall > 0:
                    self.stall -= 1
                h.wfile.write(body[:len(body) // 2])
                h.wfile.flush()
                time.sleep(1.5)                          # ...and say nothing more
                return
            if self.truncate and len(body) > 1:         # (the 1-byte size probe is left alone)
                if self.truncate > 0:
                    self.truncate -= 1
                body = body[:len(body) // 2]
            h.wfile.write(body)
        except OSError:
            pass


@pytest.fixture
def origin():
    made = []

    def make(data, **kw):
        o = _Origin(data, **kw)
        made.append(o)
        return o

    yield make
    for o in made:
        o.close()


def test_server_that_ignores_range_never_yields_wrong_bytes(tmp_path):
    """Before: a 200 carrying the whole file was read as if it were the requested range, so a
    read at 2 MB silently returned the bytes of offset 0."""
    data = _rand(3 * MB)
    pub = tmp_path / "pub"
    pub.mkdir()
    (pub / "blob.bin").write_bytes(data)

    class Stock(http.server.SimpleHTTPRequestHandler):      # http.server's own handler: no Range
        def log_message(self, *a):
            pass

    class Server(http.server.ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            pass

    httpd = Server(("127.0.0.1", 0), functools.partial(Stock, directory=str(pub)))
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    try:
        rf = g.RemoteFile(f"http://127.0.0.1:{httpd.server_address[1]}/blob.bin",
                          block="256KB", cache="2MB")
        assert rf.size == len(data)
        rf.seek(0)
        assert rf.read(1000) == data[:1000]                 # a 200 *can* answer the first block
        rf.seek(2 * MB)
        with pytest.raises(IOError, match="(?i)range"):
            rf.read(1000)
        rf.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_range_ignoring_origin_without_length_is_still_refused_past_the_first_block(origin):
    data = _rand(MB)
    o = origin(data, ranges=False, head_length=True, get_length=False)
    rf = g.RemoteFile(o.url, block="64KB", cache="1MB")
    assert rf.read(500) == data[:500]
    rf.seek(500_000)
    with pytest.raises(IOError):
        rf.read(10)


def test_truncated_range_reply_is_retried_once_and_not_cached_short(origin):
    """Before: a 206 body cut short was cached as a full block; a read that ran across the cut
    returned fewer bytes than asked, with no error."""
    data = _rand(MB)
    o = origin(data, truncate=1)                    # the first ranged reply loses its second half
    rf = g.RemoteFile(o.url, block="256KB", cache="2MB")
    start = 256 * 1024 + 128 * 1024 - 100           # 100 bytes before where the first reply stops
    rf.seek(start)
    assert rf.read(1000) == data[start:start + 1000]
    assert rf.requests == 2                          # the cut reply, then the retry
    assert all(len(b) == rf.block for b in rf._cache.values())
    rf.close()


def test_persistently_short_replies_raise_instead_of_returning_short_data(origin):
    data = _rand(MB)
    o = origin(data, truncate=-1)
    rf = g.RemoteFile(o.url, block="256KB", cache="2MB")
    rf.seek(256 * 1024 + 128 * 1024 - 100)
    with pytest.raises(IOError, match="(?i)short"):
        rf.read(1000)
    assert not rf._cache                              # nothing short was kept


def test_a_reply_that_stalls_half_way_is_retried_and_then_reported(origin):
    data = _rand(MB)
    o = origin(data, stall=1)
    rf = g.RemoteFile(o.url, block="256KB", cache="2MB", timeout=0.5)
    rf.seek(1000)
    assert rf.read(5000) == data[1000:6000]                 # the stalled reply, then the retry
    assert rf.requests == 2
    o.stall = -1
    rf.seek(600_000)
    with pytest.raises(IOError, match="(?i)short.*(timed out|timeout)"):
        rf.read(10)


def test_reply_for_a_different_range_than_asked_is_refused(origin):
    data = _rand(MB)
    o = origin(data, shift=4096)                      # 206 whose Content-Range starts 4096 late
    rf = g.RemoteFile(o.url, block="64KB", cache="1MB")
    rf.seek(100_000)
    with pytest.raises(IOError, match="(?i)range"):
        rf.read(10)


def test_file_that_changed_size_between_head_and_get_is_refused(origin):
    data = _rand(MB)
    o = origin(data, total=len(data) + 10)
    rf = g.RemoteFile(o.url, block="64KB", cache="1MB")
    rf.seek(100_000)
    with pytest.raises(IOError, match="(?i)chang"):
        rf.read(10)


def test_block_larger_than_half_the_cache_still_reads_every_byte(origin):
    """Before: with block=1MB and cache=2MB the read-ahead span exceeded the cache, so the block
    just fetched was evicted at once; reads returned 1 MB, 0, 1 MB alternately and copyfileobj
    stopped after the first MB of ten."""
    data = _rand(10 * MB, seed=7)
    o = origin(data)
    rf = g.RemoteFile(o.url, block="1MB", cache="2MB")
    out = io.BytesIO()
    shutil.copyfileobj(rf, out)
    assert out.getvalue() == data
    assert len(rf._cache) <= rf.cache_cap
    rf.seek(0)
    assert rf.read() == data
    rf.close()


@pytest.mark.parametrize("block,cache,readahead", [("1MB", "1MB", 8), ("256KB", "768KB", 4),
                                                   ("128KB", "2MB", 64)])
def test_any_block_cache_readahead_combination_reads_correctly(origin, block, cache, readahead):
    data = _rand(3 * MB + 12_345, seed=3)
    o = origin(data)
    rf = g.RemoteFile(o.url, block=block, cache=cache, readahead=readahead)
    rng = np.random.default_rng(0)
    for off in [0, 1, len(data) - 70_000] + list(rng.integers(0, len(data) - 50_000, 12)):
        rf.seek(int(off))
        assert rf.read(50_000) == data[int(off):int(off) + 50_000]
    rf.drop_cache()                                       # so the scan below starts cold, whatever the
    rf.seek(0)                                            # random reads above happened to leave cached
    got = bytearray()
    while len(got) < len(data):                           # a sequential scan, in awkward sizes
        piece = rf.read(77_777)
        assert len(piece) == min(77_777, len(data) - len(got))      # never short before the end
        got += piece
    assert bytes(got) == data and rf.read(10) == b""
    assert len(rf._cache) <= rf.cache_cap
    rf.close()


def test_size_comes_from_a_ranged_get_when_head_carries_no_length(origin):
    """Before: a HEAD without Content-Length gave size 0 and read() returned b'' silently."""
    data = _rand(500_000)
    o = origin(data, head_length=False)
    rf = g.RemoteFile(o.url, block="64KB", cache="1MB")
    assert rf.size == len(data)
    rf.seek(400_000)
    assert rf.read(1000) == data[400_000:401_000]


def test_size_comes_from_a_ranged_get_when_head_is_refused(origin):
    data = _rand(200_000)
    o = origin(data, head_status=405)
    rf = g.RemoteFile(o.url, block="64KB", cache="1MB")
    assert rf.size == len(data)
    assert rf.read(100) == data[:100]


def test_unknown_size_is_an_error_not_an_empty_file(origin):
    o = origin(_rand(1000), ranges=False, head_length=False, get_length=False)
    with pytest.raises(IOError, match="(?i)size"):
        g.RemoteFile(o.url)


def test_empty_file_is_a_real_empty_file(origin):
    o = origin(b"")
    rf = g.RemoteFile(o.url)
    assert rf.size == 0 and rf.read() == b""


def test_negative_and_unknown_seeks_are_rejected(origin):
    o = origin(_rand(10_000))
    rf = g.RemoteFile(o.url, block="4KB", cache="64KB")
    with pytest.raises(ValueError):
        rf.seek(-1)
    with pytest.raises(ValueError):
        rf.seek(0, 7)
    assert rf.seek(-100, 2) == 9_900
    rf.close()
    with pytest.raises(ValueError):
        rf.read(1)


def test_shortfall_in_the_ram_budget_fails_at_once_not_after_the_timeout(origin):
    """Before: Budget.acquire waited out the whole timeout for RAM that nobody was going to give back."""
    o = origin(_rand(1000))
    b = g.Budget(ram="100MB", timeout_s=3)
    hold = b.acquire("ram", "90MB")
    t0 = time.monotonic()
    with pytest.raises(MemoryError, match="(?i)free"):
        g.RemoteFile(o.url, block="1MB", cache="64MB", budget=b)
    assert time.monotonic() - t0 < 1.5
    hold.release()


def test_a_failed_open_releases_the_cache_reservation(origin):
    o = origin(_rand(1000), head_status=404, ranges=False)
    b = g.Budget(ram="100MB")
    with pytest.raises(Exception):
        g.RemoteFile(o.url, block="1MB", cache="8MB", budget=b)
    assert b.available("ram") == 100 * MB


def test_listing_survives_a_file_whose_size_the_server_will_not_give(origin):
    listing = '<html><a href="blob.bin">blob.bin</a></html>'
    o = origin(_rand(1000), ranges=False, head_length=False, get_length=False, listing=listing)
    tree = g.list_tree(o.base)
    assert [e.name for e in tree] == ["blob.bin"] and tree[0].size == 0


@pytest.fixture
def ftp_origin(tmp_path):
    pytest.importorskip("pyftpdlib")
    from pyftpdlib.authorizers import DummyAuthorizer
    from pyftpdlib.handlers import FTPHandler
    from pyftpdlib.log import config_logging
    from pyftpdlib.servers import ThreadedFTPServer
    config_logging(level=logging.ERROR)
    root = tmp_path / "ftp"
    root.mkdir()
    data = _rand(400_000, seed=3)
    (root / "blob.bin").write_bytes(data)
    auth = DummyAuthorizer()
    auth.add_anonymous(str(root))

    class Handler(FTPHandler):
        authorizer = auth

    srv = ThreadedFTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"ftp://127.0.0.1:{srv.address[1]}/blob.bin", data
    srv.close_all()


def test_ftp_short_transfer_is_retried_once(ftp_origin, monkeypatch):
    url, data = ftp_origin
    rf = g.RemoteFile(url, block="64KB", cache="1MB")
    real, calls = rf._ftp_range, []

    def flaky(start, length, retry=True):
        calls.append(start)
        out = real(start, length, retry)
        return out[:len(out) // 2] if len(calls) == 1 else out      # the first transfer dies half way

    monkeypatch.setattr(rf, "_ftp_range", flaky)
    rf.seek(30_000)
    assert rf.read(10_000) == data[30_000:40_000]                   # crosses the cut at 32 KB
    assert len(calls) == 2
    assert all(len(b) == rf.block for i, b in rf._cache.items() if i < 6)
    rf.close()


def test_ftp_persistently_short_transfer_raises(ftp_origin, monkeypatch):
    url, data = ftp_origin
    rf = g.RemoteFile(url, block="64KB", cache="1MB")
    real = rf._ftp_range
    monkeypatch.setattr(rf, "_ftp_range", lambda s, n, retry=True: real(s, n, retry)[:100])
    with pytest.raises(IOError, match="(?i)short"):
        rf.read(1000)
    assert not rf._cache
    rf.close()


def test_ftp_server_that_gives_no_size_is_an_error(ftp_origin, monkeypatch):
    import ftplib
    url, _ = ftp_origin
    monkeypatch.setattr(ftplib.FTP, "size", lambda self, name: None)
    with pytest.raises(IOError, match="(?i)size"):
        g.RemoteFile(url)


def test_ftp_block_larger_than_half_the_cache_reads_every_byte(ftp_origin):
    url, data = ftp_origin
    rf = g.RemoteFile(url, block="128KB", cache="256KB")
    out = io.BytesIO()
    shutil.copyfileobj(rf, out)
    assert out.getvalue() == data
    rf.close()


# ===========================================================================================
# 2. Workspace: run_isolated
# ===========================================================================================

class FakeView:
    """Just enough of a view for a Cell: a position, and the calls that put it back."""

    def __init__(self, target, budget=None, **kw):
        self.target, self.budget = target, budget
        self.window, self.zoom, self.genes, self._forced_bin = (0, 100, 0, 100), 0, None, None
        self.closed = False

    def select(self, *bounds):
        self.window = tuple(bounds)
        return self

    def set_zoom(self, level):
        self.zoom = level
        return self

    def select_genes(self, genes):
        self.genes = list(genes) if genes else None
        return self

    def use_bin(self, b):
        self._forced_bin = b
        return self

    def cost(self):
        return {"units": 1}

    def close(self):
        self.closed = True


def _open_fake(target, budget=None, **kw):
    return FakeView(target, budget, **kw)


def _position(view):
    return view.window, view.zoom, view.genes, view._forced_bin


def test_run_isolated_returns_a_result_larger_than_a_pipe_buffer():
    """Before: the parent joined the child before reading its result; a child blocked in
    Queue.put (anything over ~64 KB) never exited, so the call sat out its whole timeout."""
    ws = g.workspace(ram="512MB")
    ws.add("x", "nowhere", ram="64MB", opener=_open_fake)
    t0 = time.monotonic()
    out = ws["x"].run_isolated(lambda v: b"\x07" * 8_000_000, timeout=10)
    assert out == b"\x07" * 8_000_000
    assert time.monotonic() - t0 < 8
    ws.close()


def test_run_isolated_reports_a_child_that_dies_without_waiting_out_the_timeout():
    ws = g.workspace(ram="512MB")
    ws.add("x", "nowhere", ram="64MB", opener=_open_fake)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="isolated run"):
        ws["x"].run_isolated(lambda v: os._exit(3), timeout=30)
    assert time.monotonic() - t0 < 10


def test_run_isolated_times_out_a_child_that_never_answers():
    ws = g.workspace(ram="512MB")
    ws.add("x", "nowhere", ram="64MB", opener=_open_fake)
    with pytest.raises(TimeoutError):
        ws["x"].run_isolated(lambda v: time.sleep(60), timeout=1)


def test_run_isolated_surfaces_an_unpicklable_result_as_an_error():
    ws = g.workspace(ram="512MB")
    ws.add("x", "nowhere", ram="64MB", opener=_open_fake)
    with pytest.raises(RuntimeError, match="isolated run.*(?i:pickl|lambda|local)"):
        ws["x"].run_isolated(lambda v: (lambda: 1), timeout=10)


def test_run_isolated_sees_the_live_position_not_the_last_suspend():
    """Before: the child was given `_saved`, which only suspend()/close() fill, and it never
    replayed bin or genes."""
    ws = g.workspace(ram="512MB")
    ws.add("x", "nowhere", ram="64MB", opener=_open_fake)
    v = ws["x"].view
    v.use_bin(50).select_genes(["G1", "G2"]).select(10, 20, 30, 40)
    v.set_zoom(2)
    assert ws["x"].run_isolated(_position, timeout=20) == ((10, 20, 30, 40), 2, ["G1", "G2"], 50)
    # park it, bring it back, move: the child must follow the move, not the old suspend
    ws["x"].suspend()
    ws["x"].resume()
    ws["x"].view.select(1, 2, 3, 4).set_zoom(0)
    assert ws["x"].run_isolated(_position, timeout=20) == ((1, 2, 3, 4), 0, ["G1", "G2"], 50)
    ws.close()


@needs_gef
def test_run_isolated_replays_bin_genes_and_window_on_a_real_chip(chip):
    ws = g.workspace(ram="256MB")
    ws.add("chip", chip, ram="128MB")
    v = ws["chip"].view
    genes = [str(x) for x in v.gef.genes(v.gef.bins[0])[:3]]
    v.use_bin(50).select_genes(genes).select(1000, 2000, 1200, 2200)
    got = ws["chip"].run_isolated(lambda w: (w.window, w.bin_size, w.genes), timeout=60)
    assert got == (v.window, v.bin_size, genes)
    ws.close()


def test_report_window_follows_the_live_view_after_a_resume():
    """Before: report()['window'] preferred the snapshot from the last suspend over the view."""
    ws = g.workspace(ram="100MB")
    ws.add("x", "nowhere", ram="10MB", opener=_open_fake)
    c = ws["x"]
    c.view.select(10, 20, 30, 40)
    c.suspend()
    assert c.report()["window"] == (10, 20, 30, 40)
    c.resume()
    assert c.report()["window"] == (10, 20, 30, 40)
    c.view.select(1, 2, 3, 4)
    assert c.report()["window"] == (1, 2, 3, 4)
    ws.close()


# ===========================================================================================
# 3. Workspace: reservations
# ===========================================================================================

def test_touching_a_parked_cell_takes_its_slice_back():
    """Before: a cell parked with release_reservation=True reopened through .view without a lease,
    so it ran 'open' while the workspace believed the RAM was free."""
    ws = g.workspace(ram="100MB", timeout_s=2)
    ws.add("a", "x", ram="60MB", opener=_open_fake)
    ws.add("b", "y", ram="30MB", opener=_open_fake)
    a = ws["a"]
    _ = a.view
    a.suspend(release_reservation=True)
    assert ws.budget.available("ram") == 70 * MB
    assert ws.report()["ram_reserved"] == 30 * MB
    _ = a.view                                          # any touch is a resume
    assert a.state == g.CellState.OPEN
    assert ws.budget.available("ram") == 10 * MB
    rep = ws.report()
    assert rep["ram_reserved"] == 90 * MB
    assert rep["ram_reserved"] + rep["ram_free"] == rep["ram_cap"]
    ws.close()
    assert ws.budget.available("ram") == 100 * MB


def test_touching_a_parked_cell_whose_slice_was_taken_fails_loudly_and_stays_parked():
    ws = g.workspace(ram="100MB", timeout_s=3)
    ws.add("a", "x", ram="60MB", opener=_open_fake)
    ws.add("b", "y", ram="30MB", opener=_open_fake)
    a = ws["a"]
    _ = a.view
    a.suspend(release_reservation=True)
    ws.add("c", "z", ram="60MB", opener=_open_fake)     # fits in what a gave back
    t0 = time.monotonic()
    with pytest.raises(MemoryError, match="(?i)free"):
        _ = a.view
    assert time.monotonic() - t0 < 1.5
    assert a.state == g.CellState.SUSPENDED
    rep = ws.report()
    assert rep["ram_reserved"] == 90 * MB <= rep["ram_cap"]
    ws.close()


def test_add_with_insufficient_ram_fails_at_once_and_names_the_free_figure():
    """Before: ws.add blocked for the workspace's whole timeout (two minutes by default) before
    raising the same MemoryError."""
    ws = g.workspace(ram="100MB", timeout_s=3)
    ws.add("a", "x", ram="60MB", opener=_open_fake)
    ws.add("b", "y", ram="30MB", opener=_open_fake)
    t0 = time.monotonic()
    with pytest.raises(MemoryError, match=r"cannot reserve.*10\.0 MB"):
        ws.add("c", "z", ram="40MB", opener=_open_fake)
    assert time.monotonic() - t0 < 1.5
    assert "c" not in ws
    assert ws.report()["ram_reserved"] == 90 * MB
    ws.close()


def test_a_failed_resume_gives_back_what_it_took_and_the_next_resume_takes_every_tier():
    """Before: a resume that got the RAM lease but not the VRAM one kept the RAM lease, and the
    next resume saw a non-empty lease list and skipped the VRAM lease for good."""
    ws = g.workspace(ram="100MB", vram="30MB", timeout_s=2)
    a = ws.add("a", "x", ram="30MB", vram="20MB", opener=_open_fake)
    _ = a.view
    a.suspend(release_reservation=True)
    b = ws.add("b", "y", ram="10MB", vram="20MB", opener=_open_fake)     # takes the VRAM a left
    with pytest.raises(MemoryError):
        a.resume()
    assert ws.budget.available("ram") == 90 * MB          # a's attempt left nothing behind
    b.close()
    a.resume()
    assert a.state == g.CellState.OPEN
    assert ws.budget.available("ram") == 70 * MB
    assert ws.budget.available("vram") == 10 * MB         # both tiers held again
    assert a.reserved("ram") == 30 * MB and a.reserved("vram") == 20 * MB
    ws.close()
    assert ws.budget.available("vram") == 30 * MB


def test_lazy_open_from_many_threads_opens_exactly_once():
    """Before: every thread that found the view unopened opened its own; all but one leaked."""
    opens = []

    def slow_open(target, budget=None, **kw):
        opens.append(1)
        time.sleep(0.2)
        return FakeView(target, budget, **kw)

    ws = g.workspace(ram="100MB")
    ws.add("x", "nowhere", ram="10MB", opener=slow_open)
    c = ws["x"]
    gate, seen = threading.Barrier(8), []

    def touch():
        gate.wait()
        seen.append(c.view)

    threads = [threading.Thread(target=touch) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(opens) == 1 and len({id(v) for v in seen}) == 1
    assert c.stats.opens == 1
    ws.close()


def test_parking_dropping_and_adding_cells_from_several_threads_neither_deadlocks_nor_leaks():
    ws = g.workspace(ram="100MB", timeout_s=2)
    anchor = ws.add("anchor", "x", ram="10MB", opener=_open_fake)
    errors = []

    def parker():                                    # park and resume one cell, over and over
        try:
            for _ in range(150):
                anchor.suspend(release_reservation=True)
                _ = anchor.view
        except MemoryError:
            pass                                     # a sibling held the slice for a moment: fine
        except Exception as e:                       # pragma: no cover
            errors.append(e)

    def churner(i):                                  # add and drop others in the meantime
        try:
            for j in range(100):
                name = f"c{i}-{j}"
                ws.add(name, "y", ram="20MB", opener=_open_fake)
                _ = ws[name].view
                ws.drop(name)
        except MemoryError:
            pass
        except Exception as e:                       # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=parker)] + [threading.Thread(target=churner, args=(i,))
                                                   for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not any(t.is_alive() for t in threads), "deadlocked"
    assert not errors
    ws.close()
    assert ws.budget.available("ram") == 100 * MB             # nothing leaked on the way


def test_dropping_a_cell_while_another_thread_resumes_it_cannot_deadlock(monkeypatch):
    """A cell takes its own lock and then the workspace's; drop() must not do the reverse."""
    ws = g.workspace(ram="100MB", timeout_s=2)
    c = ws.add("x", "t", ram="10MB", opener=_open_fake)
    c.suspend(release_reservation=True)
    dropping = threading.Event()
    real_close, real_reserve = g.Cell.close, g.Cell._reserve

    def close(self):
        dropping.set()                    # the dropper is about to need this cell's lock...
        return real_close(self)

    def reserve(self):
        dropping.wait(5)                  # ...so hold the cell's lock until it is under way
        time.sleep(0.2)
        return real_reserve(self)         # ...and only then ask for the workspace's

    monkeypatch.setattr(g.Cell, "close", close)
    monkeypatch.setattr(g.Cell, "_reserve", reserve)
    outcome = []

    def resume():
        try:
            _ = c.view
        except Exception as e:
            outcome.append(type(e).__name__)

    threads = [threading.Thread(target=resume, daemon=True),
               threading.Thread(target=lambda: ws.drop("x"), daemon=True)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)
    assert not any(t.is_alive() for t in threads), "deadlocked"
    assert "x" not in ws and c.state == g.CellState.CLOSED
    assert ws.budget.available("ram") == 100 * MB


def test_a_closed_cell_is_not_resurrected_by_leaving_a_with_block():
    ws = g.workspace(ram="100MB")
    c = ws.add("x", "nowhere", ram="10MB", opener=_open_fake)
    _ = c.view
    c.close()
    with c:
        pass
    assert c.state == g.CellState.CLOSED
    assert ws.budget.available("ram") == 100 * MB


def test_a_view_that_fails_to_restore_does_not_stay_half_open():
    class Broken(FakeView):
        def select(self, *bounds):
            raise RuntimeError("cannot restore")

    ws = g.workspace(ram="100MB")
    c = ws.add("x", "nowhere", ram="10MB", opener=lambda t, budget=None, **k: Broken(t, budget))
    c._saved = {"window": (1, 2, 3, 4)}
    with pytest.raises(RuntimeError, match="cannot restore"):
        _ = c.view
    assert c.state == g.CellState.FAILED and c._view is None
    ws.close()


# ===========================================================================================
# 4. Staging
# ===========================================================================================

def _blob(n: int, **info):
    def writer(path):
        with open(path, "wb") as fh:
            fh.write(b"x" * n)
        return {"rows": n, "format": "bin", **info}
    return writer


def _disk_bytes(st) -> int:
    return sum(os.path.getsize(os.path.join(st.path, f)) for f in os.listdir(st.path)
               if os.path.isfile(os.path.join(st.path, f)) and f != st.INDEX and not f.endswith(".tmp"))


class _FakeChip:
    """Enough of a Stereo-seq view for Staging.extract: a window, a bin, genes, a cost and a load."""
    path = "/vault/chip.gef"

    def __init__(self):
        self.window, self.genes, self._bin, self.zoom, self.loads = (0, 1000, 0, 1000), None, 50, 0, 0

    @property
    def bin_size(self):
        return self._bin

    def use_bin(self, b):
        self._bin = b
        return self

    def select(self, *w):
        self.window = tuple(w)
        return self

    def select_genes(self, genes):
        self.genes = list(genes) if genes else None
        return self

    def cost(self):
        return {"fits_whole": True}

    def load(self):
        self.loads += 1
        x0, x1, _, _ = self.window
        xs = np.arange(x0, x1, self._bin, dtype=np.int32)
        return {"x": xs, "y": xs.copy(), "count": np.ones(len(xs), np.int32), "bin_size": self._bin}


class _FakeTile:
    path = "/vault/tile.tif"
    window = (0, 10, 0, 10)

    def select(self, *w):
        return self

    def cost(self):
        return {"fits_whole": True}

    def load(self):
        return np.arange(100, dtype="float32").reshape(10, 10)


def test_index_stores_paths_relative_to_the_root(tmp_path):
    st = g.staging(str(tmp_path / "s"), capacity="10MB")
    st.put("r1", _blob(100))
    idx = json.load(open(st._index_path))
    assert not os.path.isabs(idx["r1"]["path"])
    assert st.get("r1").path.startswith(st.path) and os.path.isabs(st.get("r1").path)


def test_evicting_in_a_copied_staging_dir_leaves_the_original_alone(tmp_path):
    """Before: the index held absolute paths, so the copy's index pointed at the original's files."""
    a = g.staging(str(tmp_path / "a"), capacity="10MB")
    for k in ("r1", "r2"):
        a.put(k, _blob(1000))
    shutil.copytree(a.path, tmp_path / "b")
    b = g.staging(str(tmp_path / "b"), capacity="10MB")
    assert {r.key for r in b} == {"r1", "r2"}
    assert all(r.path.startswith(b.path) for r in b)
    b.clear(include_pinned=True)
    assert not os.listdir(b.path) or os.listdir(b.path) == [b.INDEX]
    assert all(os.path.exists(r.path) for r in a)
    assert {r.key for r in g.staging(a.path)} == {"r1", "r2"}


def test_an_old_index_with_absolute_paths_is_rebased_into_the_copy(tmp_path):
    a = g.staging(str(tmp_path / "a"), capacity="10MB")
    a.put("r1", _blob(1000))
    idx = json.load(open(a._index_path))
    idx["r1"]["path"] = a.get("r1").path                       # the format older versions wrote
    json.dump(idx, open(a._index_path, "w"))
    shutil.copytree(a.path, tmp_path / "b")
    b = g.staging(str(tmp_path / "b"), capacity="10MB")
    assert b.get("r1").path.startswith(b.path)
    b.evict("r1")
    assert os.path.exists(a.get("r1").path)


def test_an_edited_index_cannot_delete_files_outside_the_root(tmp_path):
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"precious")
    st = g.staging(str(tmp_path / "s"), capacity="10MB")
    st.put("r", _blob(10))
    idx = json.load(open(st._index_path))
    idx["abs"] = dict(idx["r"], key="abs", path=str(victim))
    idx["dotdot"] = dict(idx["r"], key="dotdot", path="../victim.bin")
    idx["self"] = dict(idx["r"], key="self", path=st.INDEX)
    json.dump(idx, open(st._index_path, "w"))
    st2 = g.staging(str(tmp_path / "s"), capacity="10MB")
    assert {r.key for r in st2} == {"r"}                       # the forged entries are refused
    st2.clear(include_pinned=True)
    assert victim.read_bytes() == b"precious"
    # and a region forged in memory is refused at the point of deletion too
    st2.regions["abs"] = g.StagedRegion(key="abs", path=str(victim), source="", bytes=8)
    st2.evict("abs")
    assert victim.read_bytes() == b"precious"


def test_a_symlink_inside_the_root_cannot_be_followed_out_of_it(tmp_path):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"precious")
    st = g.staging(str(tmp_path / "s"), capacity="10MB")
    st.put("r", _blob(10))
    link = os.path.join(st.path, "link-0000000000000000.bin")
    os.symlink(outside, link)
    idx = json.load(open(st._index_path))
    idx["ln"] = dict(idx["r"], key="ln", path=os.path.basename(link))
    json.dump(idx, open(st._index_path, "w"))
    st2 = g.staging(st.path, capacity="10MB")
    assert "ln" not in st2
    st2.clear(include_pinned=True)
    assert outside.read_bytes() == b"precious"


def test_same_key_with_a_different_bin_and_window_is_re_extracted(tmp_path):
    """Before: extract('roi', ...) a second time returned the first region whatever the view said."""
    st = g.staging(str(tmp_path / "s"), capacity="50MB")
    view = _FakeChip().use_bin(50)
    r1 = st.extract("roi", view, window=(0, 1000, 0, 1000))
    assert r1.rows == 20
    view.use_bin(200)
    r2 = st.extract("roi", view, window=(0, 8000, 0, 8000))
    assert r2.window == (0, 8000, 0, 8000) and r2.rows == 40
    assert st.read("roi").num_rows == 40
    assert len(st) == 1 and st.used == r2.bytes
    # the same window at another bin is another region as well
    r3 = st.extract("roi", view.use_bin(50), window=(0, 8000, 0, 8000))
    assert r3.rows == 160 and st.read("roi").num_rows == 160


def test_same_key_with_other_genes_or_columns_is_re_extracted(tmp_path):
    st = g.staging(str(tmp_path / "s"), capacity="50MB")
    view = _FakeChip()
    st.extract("roi", view, window=(0, 1000, 0, 1000))
    st.extract("roi", view, window=(0, 1000, 0, 1000), genes=["A", "B"])
    assert view.loads == 2
    assert st.get("roi").meta["sig"]["genes"] == ["A", "B"]


def test_same_key_with_the_same_view_is_a_hit_and_loads_nothing(tmp_path):
    st = g.staging(str(tmp_path / "s"), capacity="50MB")
    view = _FakeChip()
    r1 = st.extract("roi", view, window=(0, 1000, 0, 1000))
    r2 = st.extract("roi", view, window=(0, 1000, 0, 1000))
    assert r1 is r2 and view.loads == 1 and r2.hits >= 1
    # ...also across a restart, where JSON has turned the window tuple into a list
    st2 = g.staging(st.path, capacity="50MB")
    r3 = st2.extract("roi", view, window=(0, 1000, 0, 1000))
    assert view.loads == 1 and r3.rows == r1.rows


def test_a_re_extracted_region_keeps_its_pin(tmp_path):
    st = g.staging(str(tmp_path / "s"), capacity="50MB")
    view = _FakeChip()
    st.extract("roi", view, window=(0, 1000, 0, 1000), pin=True)
    r = st.extract("roi", view.use_bin(200), window=(0, 1000, 0, 1000))
    assert r.pinned


def test_a_failed_re_extraction_leaves_the_old_region_in_place(tmp_path):
    st = g.staging(str(tmp_path / "s"), capacity="50MB")
    view = _FakeChip()
    before = st.extract("roi", view, window=(0, 1000, 0, 1000))
    view.load = lambda: (_ for _ in ()).throw(RuntimeError("link went down"))
    with pytest.raises(RuntimeError, match="link went down"):
        st.extract("roi", view.use_bin(200), window=(0, 1000, 0, 1000))
    assert st.get("roi").rows == before.rows and os.path.exists(st.get("roi").path)
    assert st.used == _disk_bytes(st)


@needs_gef
def test_re_extracting_a_key_on_a_real_chip_follows_bin_and_window(tmp_path, chip):
    st = g.staging(str(tmp_path / "s"), capacity="64MB")
    view = g.open_gef(chip, budget=g.Budget(ram="128MB"))
    w1, w2 = (1000, 2200, 1000, 2200), (500, 3000, 500, 3000)
    r1 = st.extract("roi", view.use_bin(50), window=w1)
    r2 = st.extract("roi", view.use_bin(200), window=w2)
    assert r2.window == w2 and r2.rows != r1.rows
    assert st.read("roi").num_rows == view.select(*w2).load()["x"].size
    view.close()


def test_a_writer_that_raises_after_writing_leaves_nothing_behind(tmp_path):
    """Before: 3 MB sat on disk while used == 0."""
    st = g.staging(str(tmp_path / "s"), capacity="50MB")

    def bad(path):
        with open(path, "wb") as fh:
            fh.write(b"x" * 3_000_000)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        st.put("k", bad)
    assert st.used == 0 and "k" not in st
    assert _disk_bytes(st) == 0
    assert [f for f in os.listdir(st.path) if f != st.INDEX] == []


def test_a_writer_that_does_not_create_its_file_is_an_error(tmp_path):
    st = g.staging(str(tmp_path / "s"), capacity="50MB")
    with pytest.raises(RuntimeError, match="(?i)did not (create|write)"):
        st.put("k", lambda path: {"rows": 1})
    assert "k" not in st and _disk_bytes(st) == 0


def test_a_writer_that_calls_np_save_on_the_path_still_stages_an_array(tmp_path):
    """The extract() bug, as a custom writer would hit it: np.save adds '.npy' to a name without it."""
    st = g.staging(str(tmp_path / "s"), capacity="50MB")
    arr = np.arange(100, dtype="float32")
    r = st.put("k", lambda path: (np.save(path, arr), {"rows": arr.size})[1], suffix=".parquet")
    assert r.format == "npy" and r.bytes > 0 and r.bytes == os.path.getsize(r.path)
    assert np.array_equal(st.read("k"), arr)
    assert st.used == _disk_bytes(st) and len(os.listdir(st.path)) == 2       # the file, and the index


def test_a_writer_that_adds_some_other_extension_is_cleaned_up_and_reported(tmp_path):
    st = g.staging(str(tmp_path / "s"), capacity="50MB")

    def adds_foo(path):
        with open(path + ".foo", "wb") as fh:
            fh.write(b"x" * 1000)
        return {"rows": 1}

    with pytest.raises(RuntimeError, match="(?i)did not create"):
        st.put("k", adds_foo, suffix=".bin")
    assert "k" not in st and _disk_bytes(st) == 0


def test_startup_reconciles_the_directory_with_the_index(tmp_path):
    """A crash can leave an interrupted write, or a file whose index entry was lost. Whatever the
    policy for those, what staging says it uses must be what is on the disk."""
    st = g.staging(str(tmp_path / "s"), capacity="50MB")
    st.put("kept", _blob(1000))
    stray = os.path.join(st.path, "stray-0123456789abcdef.parquet")
    with open(stray, "wb") as fh:
        fh.write(b"y" * 5000)
    with open(os.path.join(st.path, "half-0123456789abcdef.99999999.part.parquet"), "wb") as fh:
        fh.write(b"z" * 7000)                      # an interrupted write by a process that is gone
    foreign = os.path.join(st.path, "notes.txt")
    with open(foreign, "wb") as fh:
        fh.write(b"w" * 300)
    st2 = g.staging(st.path, capacity="50MB")
    assert st2.used == _disk_bytes(st2)
    assert "kept" in st2 and st2.get("kept").bytes == 1000
    assert not [f for f in os.listdir(st2.path) if ".part" in f]
    assert os.path.exists(foreign)                 # a file that is not ours is never deleted
    st2.clear()                                    # ...whatever else was adopted is evictable
    assert os.path.exists(foreign)


def test_unrelated_files_in_the_directory_are_counted_never_deleted_and_named_when_they_block(tmp_path):
    root = tmp_path / "s"
    root.mkdir()
    (root / "thesis.pdf").write_bytes(b"p" * (8 * MB))
    st = g.staging(str(root), capacity="10MB")
    assert st.used == 8 * MB and st.report()["orphans"] == 1
    with pytest.raises(g.StagingFull, match=r"8\.0 MB of it is files in the directory"):
        st.put("big", _blob(5 * MB))
    st.clear(include_pinned=True)                 # staging's own regions only: this is not its file
    assert (root / "thesis.pdf").exists() and st.used == 8 * MB
    assert st.put("small", _blob(MB)).bytes == MB
    st.clear(include_pinned=True)
    assert (root / "thesis.pdf").exists() and list(r.key for r in st) == ["orphan:thesis.pdf"]
    st.evict("orphan:thesis.pdf")                 # only by name, and then it is the caller's doing
    assert not (root / "thesis.pdf").exists()


def test_orphans_are_evicted_before_real_regions_when_space_is_short(tmp_path):
    st = g.staging(str(tmp_path / "s"), capacity="10MB")
    stray = os.path.join(st.path, "lost-0123456789abcdef.parquet")
    with open(stray, "wb") as fh:
        fh.write(b"y" * (6 * MB))
    st2 = g.staging(st.path, capacity="10MB")
    assert st2.used == 6 * MB
    st2.put("new", _blob(7 * MB))                  # does not fit beside the orphan
    assert "new" in st2 and not os.path.exists(stray)
    assert st2.used == _disk_bytes(st2)


def test_arrays_land_as_npy_whatever_suffix_was_left_at_the_default(tmp_path):
    """Before: np.save appended '.npy' to the '.parquet' slot, so bytes == 0 and read() failed."""
    st = g.staging(str(tmp_path / "s"), capacity="50MB")
    r = st.extract("tile", _FakeTile())
    assert r.format == "npy" and r.bytes > 0 and r.bytes == os.path.getsize(r.path)
    assert np.array_equal(st.read("tile"), np.arange(100, dtype="float32").reshape(10, 10))
    r2 = st.extract("tile2", _FakeTile(), suffix=".npy")
    assert r2.bytes > 0 and np.array_equal(st.read("tile2"), st.read("tile"))
    assert st.used == _disk_bytes(st)


def test_two_staging_objects_on_one_directory_do_not_lose_each_others_keys(tmp_path):
    """Before: each object saved its own in-memory dict over the index, so the last writer won."""
    a = g.staging(str(tmp_path / "s"), capacity="50MB")
    b = g.staging(str(tmp_path / "s"), capacity="50MB")
    a.put("k1", _blob(100))
    b.put("k2", _blob(200))
    a.put("k3", _blob(300))
    assert {r.key for r in g.staging(a.path)} == {"k1", "k2", "k3"}
    assert "k2" in a and a.get("k2").bytes == 200                  # a notices b's region
    c = g.staging(str(tmp_path / "s"), capacity="50MB")
    c.put("k4", _blob(400))
    assert a.used == b.used == c.used == 100 + 200 + 300 + 400   # ...and everyone counts the same bytes
    b.evict("k1")                                                  # ...and b's eviction of a's
    assert "k1" not in g.staging(a.path)
    assert not os.path.exists(os.path.join(a.path, os.path.basename(a._slot("k1", ".parquet"))))


def test_concurrent_writers_on_separate_objects_lose_nothing(tmp_path):
    root = str(tmp_path / "s")
    objs = [g.staging(root, capacity="50MB") for _ in range(4)]
    errors = []

    def work(i):
        try:
            for j in range(15):
                objs[i].put(f"t{i}-{j}", _blob(500 + j))
        except Exception as e:                      # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    fresh = g.staging(root, capacity="50MB")
    assert len(fresh) == 60 and fresh.used == _disk_bytes(fresh)


def _put_some(root, tag, n):
    st = g.staging(root, capacity="50MB")
    for j in range(n):
        st.put(f"{tag}-{j}", _blob(100 + j))


def _fork():
    if not hasattr(os, "fork"):
        pytest.skip("needs fork")
    return multiprocessing.get_context("fork")


def test_processes_sharing_one_directory_lose_nothing(tmp_path):
    ctx = _fork()
    root = str(tmp_path / "s")
    procs = [ctx.Process(target=_put_some, args=(root, f"p{i}", 15)) for i in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    assert [p.exitcode for p in procs] == [0, 0, 0, 0]
    fresh = g.staging(root, capacity="50MB")
    assert len(fresh) == 60 and fresh.used == _disk_bytes(fresh)


def _hold_the_directory(root, ready, holding, seconds):
    st = g.staging(root, capacity="10MB")
    ready.wait(10)
    with st._dirlock:
        holding.set()
        time.sleep(seconds)


def _timed_put(root, ready, holding, out):
    st = g.staging(root, capacity="10MB")
    ready.set()
    holding.wait(10)
    t0 = time.monotonic()
    st.put("k", _blob(10))
    out.put(time.monotonic() - t0)


def test_a_second_process_waits_for_the_directory_lock(tmp_path):
    pytest.importorskip("fcntl")
    ctx = _fork()
    root = str(tmp_path / "s")
    ready, holding, out = ctx.Event(), ctx.Event(), ctx.Queue()
    a = ctx.Process(target=_hold_the_directory, args=(root, ready, holding, 1.0))
    b = ctx.Process(target=_timed_put, args=(root, ready, holding, out))
    a.start()
    b.start()
    waited = out.get(timeout=30)
    a.join(30)
    b.join(30)
    assert waited >= 0.6, f"the put went straight through ({waited:.2f}s) while the lock was held"


def test_one_object_is_safe_across_threads(tmp_path):
    st = g.staging(str(tmp_path / "s"), capacity="50MB")
    errors = []

    def work(i):
        try:
            for j in range(15):
                st.put(f"t{i}-{j}", _blob(500))
        except Exception as e:                      # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(st) == 60 and st.used == 60 * 500 == _disk_bytes(st)
    assert len(g.staging(st.path, capacity="50MB")) == 60


def test_a_put_that_cannot_fit_beside_the_pins_evicts_nothing(tmp_path):
    """Before: every unpinned region was evicted before StagingFull told you it was hopeless."""
    st = g.staging(str(tmp_path / "s"), capacity="10MB")
    st.put("pinned", _blob(8 * MB), pin=True)
    st.put("a", _blob(MB // 2))
    st.put("b", _blob(MB // 2))
    with pytest.raises(g.StagingFull, match="pinned"):
        st.put("big", _blob(5 * MB))
    assert {"pinned", "a", "b"} == {r.key for r in st}
    assert st.used == _disk_bytes(st) == 9 * MB
    with pytest.raises(g.StagingFull, match="cannot fit"):          # larger than the whole area
        st.put("huge", _blob(11 * MB))
    assert {"pinned", "a", "b"} == {r.key for r in st} and _disk_bytes(st) == 9 * MB


def test_a_put_that_fits_after_eviction_evicts_least_recently_used_first(tmp_path):
    st = g.staging(str(tmp_path / "s"), capacity="10MB")
    st.put("old", _blob(4 * MB))
    st.put("mid", _blob(4 * MB))
    st.get("old")                                   # old is now the more recently used
    st.put("new", _blob(4 * MB))
    assert {r.key for r in st} == {"old", "new"}
    assert st.used == _disk_bytes(st)


def test_pins_and_touches_survive_a_restart_and_another_object(tmp_path):
    st = g.staging(str(tmp_path / "s"), capacity="10MB")
    st.put("a", _blob(100))
    st.put("b", _blob(100))
    st.pin("a")
    other = g.staging(st.path, capacity="10MB")
    assert other.get("a").pinned and not other.get("b").pinned
    other.pin("a", False)
    assert not g.staging(st.path, capacity="10MB").get("a").pinned
    assert st.get("a").pinned is False                                # st notices the unpin
    for _ in range(3):
        st.get("b")
    st.pin("b")                                                       # any write carries touches along
    again = g.staging(st.path, capacity="10MB")
    assert again.get("b").hits >= 4 and again.get("b").pinned


def test_overwriting_a_key_under_another_suffix_replaces_the_file(tmp_path):
    st = g.staging(str(tmp_path / "s"), capacity="10MB")
    first = st.put("k", _blob(100), suffix=".bin")
    second = st.put("k", _blob(300), suffix=".dat", overwrite=True)
    assert first.path != second.path and second.path.endswith(".dat")
    assert not os.path.exists(first.path) and os.path.getsize(second.path) == 300
    assert len(st) == 1 and st.used == 300 == _disk_bytes(st)
    same = st.put("k", _blob(50), suffix=".dat", overwrite=True)      # same name: replaced in place
    assert same.path == second.path and os.path.getsize(same.path) == 50 and st.used == 50


def test_a_put_asking_again_with_a_different_window_or_source_is_written_again(tmp_path):
    st = g.staging(str(tmp_path / "s"), capacity="10MB")
    st.put("k", _blob(100), source="vault/a.gef", window=(0, 10, 0, 10))
    assert st.put("k", _blob(999), source="vault/a.gef", window=(0, 10, 0, 10)).bytes == 100   # a hit
    assert st.put("k", _blob(200), source="vault/a.gef", window=(0, 20, 0, 20)).bytes == 200
    assert st.put("k", _blob(300), source="vault/b.gef", window=(0, 20, 0, 20)).bytes == 300
    assert st.put("k", _blob(999)).bytes == 300                       # nothing to contradict: a hit
    assert st.used == 300 == _disk_bytes(st)


def test_try_acquire_gives_the_lease_or_none_and_never_waits():
    from geminga.workspace import try_acquire
    b = g.Budget(ram="10MB", timeout_s=30)
    t0 = time.monotonic()
    lease = try_acquire(b, "ram", 6 * MB)
    assert lease is not None and b.available("ram") == 4 * MB
    assert try_acquire(b, "ram", 6 * MB) is None                      # does not fit now
    assert try_acquire(b, "ram", 11 * MB) is None                     # could never fit
    assert try_acquire(b, "vram", 1) is None                          # a tier with nothing in it
    assert time.monotonic() - t0 < 1
    lease.release()
    assert try_acquire(b, "ram", 10 * MB) is not None


def test_staging_shortfall_in_the_spill_budget_fails_at_once(tmp_path):
    """Before: the Staging constructor waited out the budget's timeout for disk nobody would free."""
    b = g.Budget(ram="10MB", spill="100MB", timeout_s=3)
    first = g.staging(str(tmp_path / "s1"), capacity="60MB", budget=b)
    t0 = time.monotonic()
    with pytest.raises(MemoryError, match=r"cannot reserve.*40\.0 MB"):
        g.staging(str(tmp_path / "s2"), capacity="60MB", budget=b)
    assert time.monotonic() - t0 < 1.5
    first.release()
    assert b.available("spill") == 100 * MB


# ===========================================================================================
# 5. Vault
# ===========================================================================================

def test_scan_indexes_nifti_and_10x_matrices(tmp_path):
    """Before: .nii, .nii.gz, .h5 and .mtx(.gz) were not in the list of known suffixes."""
    root = tmp_path / "v"
    rels = ["mri/t1.nii", "mri/t2.NII.GZ", "tenx/filtered.h5", "tenx/mtx/matrix.mtx",
            "tenx/mtx2/matrix.mtx.gz", "plain/notes.txt"]
    for rel in rels:
        _touch(root / rel)
    v = g.vault(str(root)).scan()
    assert {m.path for m in v} == set(rels) - {"plain/notes.txt"}


def test_a_dicom_directory_is_one_member_not_one_per_slice(tmp_path):
    root = tmp_path / "v"
    for i in range(30):
        _touch(root / "mri" / "P01" / f"slice_{i:03d}.dcm", 100)
    for i in range(5):
        _touch(root / "mri" / "P02" / f"s{i}.IMA", 50)
    _touch(root / "other" / "reads.fastq")
    v = g.vault(str(root)).scan()
    assert sorted(m.path for m in v) == ["mri/P01", "mri/P02", "other/reads.fastq"]
    m = v.get("mri/P01")
    assert m.bytes == 3000 and m.meta["n_files"] == 30 and m.name == "P01"
    assert os.path.isdir(v.abspath("mri/P01"))
    assert v.get("mri/P02").bytes == 250
    again = g.vault(str(root)).scan()                       # and a rescan keeps it that way
    assert sorted(m.path for m in again) == sorted(m.path for m in v)


def test_a_directory_with_dicom_slices_and_other_files_keeps_the_other_files(tmp_path):
    root = tmp_path / "v"
    for i in range(3):
        _touch(root / "study" / f"s{i}.dcm", 10)
    _touch(root / "study" / "recon.nii.gz", 40)
    v = g.vault(str(root)).scan()
    assert sorted(m.path for m in v) == ["study", "study/recon.nii.gz"]


@needs_dicom
def test_a_real_dicom_series_profiles_and_opens_as_one_member(tmp_path):
    root = tmp_path / "v"
    shutil.copytree(DICOM_DIR, root / "P04_FLAIR")
    v = g.vault(str(root)).scan(profile=True)
    assert [m.path for m in v] == ["P04_FLAIR"]
    m = v.get("P04_FLAIR")
    assert m.format == "DICOM series" and m.profiled and m.meta["n_files"] == 48
    view = v.open("P04_FLAIR")
    assert type(view).__name__ == "VolumeView"
    view.close()


def test_vaults_with_the_same_folder_name_do_not_share_a_fallback_manifest(tmp_path):
    """Before: the fallback manifest was ~/.geminga/vault-<basename>.json, so /a/data and /b/data
    were one vault as far as the manifest went."""
    roots = []
    for side in ("a", "b"):
        r = tmp_path / side / "data"
        r.mkdir(parents=True)
        (r / ".geminga-vault.json").mkdir()          # nothing can be written there: use ~/.geminga
        roots.append(r)
    _touch(roots[0] / "one.gef")
    _touch(roots[1] / "two.gef")
    _touch(roots[1] / "three.gef")
    v1 = g.vault(str(roots[0])).scan()
    v2 = g.vault(str(roots[1]))
    assert len(v2) == 0                              # a fresh vault inherits nothing from v1
    v2.scan()
    assert v1.manifest_path != v2.manifest_path
    assert {m.name for m in g.vault(str(roots[0]))} == {"one.gef"}
    assert {m.name for m in g.vault(str(roots[1]))} == {"two.gef", "three.gef"}


def test_a_manifest_left_by_an_older_version_is_used_only_for_its_own_root(tmp_path, _private_home):
    a, b = tmp_path / "a" / "data", tmp_path / "b" / "data"
    for r in (a, b):
        r.mkdir(parents=True)
        (r / ".geminga-vault.json").mkdir()
    _touch(a / "one.gef")
    legacy = _private_home / ".geminga" / "vault-data.json"
    legacy.parent.mkdir(parents=True)
    payload = {"root": str(a), "scanned_at": 1.0,
               "members": {"one.gef": {"path": "one.gef", "bytes": 16, "mtime": 1.0}}}
    legacy.write_text(json.dumps(payload))
    assert [m.path for m in g.vault(str(a))] == ["one.gef"]        # its own: inherited
    assert len(g.vault(str(b))) == 0                               # someone else's: ignored


def test_readonly_vault_keeps_its_manifest_off_the_data(tmp_path):
    root = tmp_path / "nas" / "stereoseq"
    _touch(root / "chip.gef")
    before = sorted(os.listdir(root))
    v = g.vault(str(root), readonly=True).scan()
    assert sorted(os.listdir(root)) == before
    assert v.manifest_path.startswith(os.path.expanduser("~/.geminga"))
    assert [m.path for m in g.vault(str(root), readonly=True)] == ["chip.gef"]


def test_a_manifest_shipped_beside_the_data_is_read_even_when_the_vault_is_readonly(tmp_path):
    root = tmp_path / "drive"
    _touch(root / "chip.gef")
    g.vault(str(root)).scan()                                      # prepared on a workstation
    assert os.path.exists(root / ".geminga-vault.json")
    v = g.vault(str(root), readonly=True)
    assert [m.path for m in v] == ["chip.gef"]
    v.scan()
    assert v.manifest_path.startswith(os.path.expanduser("~/.geminga"))


def test_get_by_bare_name_requires_a_unique_match(tmp_path):
    """Before: get('matrix.gef') returned whichever of several same-named files came first."""
    root = tmp_path / "v"
    for rel in ("a/matrix.gef", "b/matrix.gef", "c/only.gef"):
        _touch(root / rel)
    v = g.vault(str(root)).scan()
    with pytest.raises((KeyError, ValueError)) as ei:
        v.get("matrix.gef")
    assert "a/matrix.gef" in str(ei.value) and "b/matrix.gef" in str(ei.value)
    with pytest.raises((KeyError, ValueError)):
        v.abspath("matrix.gef")
    assert v.get("a/matrix.gef").path == "a/matrix.gef"
    assert v.get("only.gef").path == "c/only.gef"
    assert v.get(str(root / "b" / "matrix.gef")).path == "b/matrix.gef"        # absolute path
    assert v.get("./c/only.gef").path == "c/only.gef"
    assert v.get("nope.gef") is None
    assert v.get(root / "c" / "only.gef").path == "c/only.gef"                   # a Path, too
    assert "matrix.gef" in v                                       # it *is* in the vault


def test_find_exact_matches_whole_tokens_in_the_path(tmp_path):
    root = tmp_path / "v"
    for rel in ("GSM1234567.h5ad", "GSM1234_a.h5ad", "sub/GSM1234.gef", "sub/xGSM1234.gef",
                "sub-GSM1234/data.h5", "SRR9_1.fastq.gz"):
        _touch(root / rel)
    v = g.vault(str(root)).scan()
    assert len(v.find("GSM1234")) == 5                                              # substring, as before
    assert _paths(v.find("GSM1234", exact=True)) == [
        "GSM1234_a.h5ad", "sub-GSM1234/data.h5", "sub/GSM1234.gef"]
    assert v.find("GSM12", exact=True) == []
    assert _paths(v.find(".gef", exact=True)) == ["sub/GSM1234.gef", "sub/xGSM1234.gef"]   # a suffix is a token
    assert _paths(v.find("sub-", exact=True)) == ["sub-GSM1234/data.h5"]
    assert v.find("sub-", exact=False) and v.find("ub-GSM", exact=True) == []
    assert _paths(v.find("srr9", exact=True)) == ["SRR9_1.fastq.gz"]
    assert _paths(v.find("data.h5", exact=True)) == ["sub-GSM1234/data.h5"]
    assert len(v.find("", exact=True)) == 6


def _paths(hits):
    return sorted(m.path for m in hits)


def _fastq(path, n=200):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    with open(path, "w") as fh:
        for i in range(n):
            seq = "".join(rng.choice(list("ACGT"), 60))
            fh.write(f"@read{i}\n{seq}\n+\n{'I' * 60}\n")
    return str(path)


def _gem(path, n=400):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    with open(path, "w") as fh:
        fh.write("#FileFormat=GEMv0.1\n#SortedBy=None\n#BinSize=1\n#StereoChip=SSMINI01\n"
                 "#OffsetX=0\n#OffsetY=0\ngeneID\tx\ty\tMIDCount\n")
        for _ in range(n):
            fh.write(f"Gene{rng.integers(0, 20):04d}\t{rng.integers(0, 500)}\t{rng.integers(0, 500)}\t1\n")
    return str(path)


def _tree(root):
    """Every file and directory under root: an empty .gemidx/ is as much a trace as a file."""
    return sorted(os.path.relpath(os.path.join(dp, n), root)
                  for dp, dns, fs in os.walk(root) for n in dns + fs)


def _sidecars_under_home(home):
    base = home / ".geminga" / "gefidx"
    return sorted(os.path.relpath(os.path.join(dp, f), base)
                  for dp, _, fs in os.walk(base) for f in fs) if base.exists() else []


def test_opening_a_member_of_a_readonly_vault_writes_nothing_beside_the_data(tmp_path, _private_home):
    """Before: the .gemidx.json sidecar of a FASTQ/GEM (and the .gemidx/ directory of a GEF) were
    written next to the data whatever the vault was told about it."""
    root = tmp_path / "nas"
    _fastq(root / "reads.fastq")
    _gem(root / "chip.gem")
    before = _tree(root)
    v = g.vault(str(root), readonly=True).scan()
    fq = v.open("reads.fastq")
    assert fq.cost()["units"] > 0
    gem = v.open("chip.gem")
    assert gem.cost()["units"] > 0
    assert _tree(root) == before
    side = _sidecars_under_home(_private_home)
    assert any(s.endswith("reads.fastq.gemidx.json") for s in side)
    assert any(s.endswith("chip.gem.gemidx.json") for s in side)
    # the second open finds the sidecar it left: same cost, no rebuild
    t = os.path.getmtime(_private_home / ".geminga" / "gefidx" / side[0])
    v.open("reads.fastq").cost()
    assert os.path.getmtime(_private_home / ".geminga" / "gefidx" / side[0]) == t


@needs_gef
def test_opening_a_gef_of_a_readonly_vault_does_not_create_its_sidecar_directory(tmp_path):
    root = tmp_path / "nas"
    root.mkdir()
    shutil.copy(GEF, root / "chip.gef")
    v = g.vault(str(root), readonly=True).scan()
    view = v.open("chip.gef")
    view.cost()
    view.select(1000, 1300, 1200, 1500).load()
    assert _tree(root) == ["chip.gef"]
    view.close()


@needs_gef
def test_an_index_built_for_a_member_of_a_readonly_vault_is_built_off_the_drive(tmp_path, _private_home):
    root = tmp_path / "nas"
    root.mkdir()
    shutil.copy(GEF, root / "chip.gef")
    v = g.vault(str(root), readonly=True).scan()
    view = v.open("chip.gef")
    coarsest = view.gef.bins[-1]
    si = view.gef.build_spatial_index(levels=[coarsest])
    assert si.valid and si.levels == [coarsest]
    assert si.dir.startswith(str(_private_home / ".geminga" / "gefidx"))
    assert _tree(root) == ["chip.gef"]
    view.close()
    again = v.open("chip.gef")                              # a new view finds it, and still writes nothing
    assert again.gef.spatial_index is not None and again.gef.spatial_index.levels == [coarsest]
    assert _tree(root) == ["chip.gef"]
    again.close()


def test_a_directory_that_is_not_writable_is_treated_as_read_only(tmp_path, _private_home, monkeypatch):
    root = tmp_path / "nas"
    _fastq(root / "reads.fastq")
    v = g.vault(str(root)).scan()
    before = _tree(root)
    monkeypatch.setattr(vault_mod, "_writable", lambda d: False, raising=False)
    v.open("reads.fastq").cost()
    assert _tree(root) == before
    assert any(s.endswith("reads.fastq.gemidx.json") for s in _sidecars_under_home(_private_home))


def test_a_writable_vault_still_keeps_its_sidecar_beside_the_data(tmp_path):
    root = tmp_path / "scratch"
    _fastq(root / "reads.fastq")
    v = g.vault(str(root)).scan()
    v.open("reads.fastq").cost()
    assert os.path.exists(root / "reads.fastq.gemidx.json")


def test_a_fresh_sidecar_beside_read_only_data_is_used_rather_than_rebuilt(tmp_path, _private_home):
    root = tmp_path / "drive"
    _fastq(root / "reads.fastq")
    g.vault(str(root)).scan().open("reads.fastq").cost()           # prepared where it was writable
    sidecar = root / "reads.fastq.gemidx.json"
    assert sidecar.exists()
    stamp = sidecar.stat().st_mtime_ns
    v = g.vault(str(root), readonly=True)
    assert v.open("reads.fastq").cost()["units"] > 0
    assert sidecar.stat().st_mtime_ns == stamp
    assert _sidecars_under_home(_private_home) == []                # nothing was rebuilt elsewhere


def test_mtx_members_open_as_a_matrix_triple(tmp_path):
    root = tmp_path / "v" / "sample"
    root.mkdir(parents=True)
    (root / "matrix.mtx").write_text("%%MatrixMarket matrix coordinate integer general\n"
                                     "3 2 3\n1 1 5\n2 1 1\n3 2 7\n")
    (root / "barcodes.tsv").write_text("AAAC\nAAAG\n")
    (root / "features.tsv").write_text("g1\tG1\tGene Expression\ng2\tG2\tGene Expression\n"
                                       "g3\tG3\tGene Expression\n")
    v = g.vault(str(tmp_path / "v")).scan()
    assert [m.path for m in v] == ["sample/matrix.mtx"]
    t = v.open("sample/matrix.mtx")
    assert (t.n_features, t.n_cells, t.nnz) == (3, 2, 3)


_RO_SCRIPT = r'''
import os, sys
root, home, pkg = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ["HOME"] = home
sys.path.insert(0, pkg)
import geminga as g
before = sorted(os.listdir(root))
v = g.vault(root)
v.scan()
assert len(v) == 2, list(v)
for name in ("reads.fastq", "chip.gef"):
    view = v.open(name)
    view.cost()
    view.close()
assert sorted(os.listdir(root)) == before, (before, sorted(os.listdir(root)))
side = [f for _, _, fs in os.walk(os.path.join(home, ".geminga")) for f in fs]
assert any(f.endswith("reads.fastq.gemidx.json") for f in side), side
print("RO-OK")
'''


@needs_gef
def test_on_a_real_read_only_mount_nothing_is_written_and_nothing_fails(tmp_path, _private_home):
    if shutil.which("unshare") is None or os.geteuid() != 0:
        pytest.skip("needs root and unshare to make a read-only mount")
    root = tmp_path / "ro"
    root.mkdir()
    shutil.copy(GEF, root / "chip.gef")
    _fastq(root / "reads.fastq")
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(g.__file__)))
    cmd = ["unshare", "-m", "sh", "-c",
           'mount --bind "$1" "$1" && mount -o remount,bind,ro "$1" && exec "$2" -c "$3" "$1" "$4" "$5"',
           "sh", str(root), sys.executable, _RO_SCRIPT, str(_private_home), pkg]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if p.returncode != 0 and re.search(r"(?i)not permitted|superuser|permission denied|unshare failed",
                                       p.stderr):
        pytest.skip(f"cannot make a read-only mount here: {p.stderr.strip()[:120]}")
    assert p.returncode == 0 and "RO-OK" in p.stdout, p.stdout + p.stderr
    assert sorted(os.listdir(root)) == ["chip.gef", "reads.fastq"]


# ===========================================================================================
# 6. Devices
# ===========================================================================================

class _FakeOpenVINO:
    """`import openvino` and nothing else is installed on a lot of machines; what the Core reports
    is the only evidence there is an NPU."""

    def __init__(self, available, names=None):
        names = names or {}
        outer = self

        class Core:
            available_devices = list(available)

            def get_property(self, device, prop):
                if device in names:
                    return names[device]
                raise RuntimeError("no such property")

        outer.Core = Core


def test_importing_openvino_alone_does_not_list_an_npu(monkeypatch):
    """Before: any machine with the openvino wheel installed reported an NPU."""
    monkeypatch.setitem(sys.modules, "openvino", _FakeOpenVINO(["CPU", "GPU"]))
    assert not [d for d in g.devices() if d.kind == "npu"]


def test_an_npu_in_the_cores_device_list_is_listed(monkeypatch):
    monkeypatch.setitem(sys.modules, "openvino",
                        _FakeOpenVINO(["CPU", "GPU.0", "NPU"], {"NPU": "Intel(R) AI Boost"}))
    npus = [d for d in g.devices() if d.kind == "npu"]
    assert len(npus) == 1 and npus[0].accounting_only and npus[0].id == "npu:0"
    assert npus[0].name == "Intel(R) AI Boost"
    assert [d.id for d in g.devices(include_accounting_only=False) if d.kind == "npu"] == []


def test_an_openvino_that_cannot_enumerate_devices_lists_nothing(monkeypatch):
    class Broken:
        class Core:
            def __init__(self):
                raise RuntimeError("plugin load failed")

    monkeypatch.setitem(sys.modules, "openvino", Broken)
    assert not [d for d in g.devices() if d.kind == "npu"]


def _host(*gpus):
    cpu = g.Device("cpu", 0, "host", 16 * 1024 ** 3, 8 * 1024 ** 3)
    return [cpu] + [g.Device(kind, i, f"{kind}-{i}", 8 * 1024 ** 3, 8 * 1024 ** 3) for kind, i in gpus]


@pytest.mark.parametrize("name", ["cuda:1", "CUDA:1", "rocm:1", "hip:1", " hip:1 "])
def test_every_gpu_spelling_resolves_to_the_rocm_device_on_a_rocm_host(monkeypatch, name):
    """Before: a ROCm host lists 'rocm:N' but torch (and everyone else) says 'cuda:N'."""
    monkeypatch.setattr(devices_mod, "devices", lambda *a, **k: _host(("rocm", 0), ("rocm", 1)))
    pool = g.device_pool(name, "1MB")
    assert pool.device.id == "rocm:1"
    assert pool.device.torch_device() == "cuda:1"


@pytest.mark.parametrize("name", ["cuda", "rocm", "hip", "HIP:0"])
def test_a_bare_gpu_name_is_device_zero(monkeypatch, name):
    monkeypatch.setattr(devices_mod, "devices", lambda *a, **k: _host(("rocm", 0), ("rocm", 1)))
    assert g.device_pool(name, "1MB").device.id == "rocm:0"


def test_cuda_names_still_resolve_on_a_cuda_host_and_rocm_names_do_not(monkeypatch):
    monkeypatch.setattr(devices_mod, "devices", lambda *a, **k: _host(("cuda", 0)))
    assert g.device_pool("cuda:0", "1MB").device.id == "cuda:0"
    assert g.device_pool("cuda", "1MB").device.id == "cuda:0"
    with pytest.raises(ValueError, match="no device"):
        g.device_pool("rocm:0", "1MB")                 # there is no AMD GPU to give
    with pytest.raises(ValueError, match="no device"):
        g.device_pool("cuda:3", "1MB")


def test_other_device_names_are_unchanged(monkeypatch):
    monkeypatch.setattr(devices_mod, "devices", lambda *a, **k: _host(("cuda", 0)))
    assert g.device_pool("cpu", "1MB").device.id == "cpu"
    assert g.device_pool("auto", "1MB").device.id == "cuda:0"
    with pytest.raises(ValueError, match="no device"):
        g.device_pool("gpu:99", "1MB")


def test_device_pool_shortfall_in_the_budget_fails_at_once():
    """Before: the pool constructor waited out the budget's timeout for allowance nobody would free."""
    b = g.Budget(ram="100MB", timeout_s=3)
    hold = b.acquire("ram", "90MB")
    t0 = time.monotonic()
    with pytest.raises(MemoryError, match=r"(?i)free"):
        g.device_pool("cpu", "64MB", budget=b)
    assert time.monotonic() - t0 < 1.5
    hold.release()
    pool = g.device_pool("cpu", "64MB", budget=b)
    assert b.available("ram") == 36 * MB
    pool.release()
