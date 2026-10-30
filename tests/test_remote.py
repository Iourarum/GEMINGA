"""Datasets on a server: list without downloading, open in place, pull only what is asked."""
import logging
import os
import shutil
import socket
import threading
import numpy as np
import pytest
import geminga as g

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")
GEF = os.path.join(FIX, "chip.gef")
pytest.importorskip("pyftpdlib")
need = pytest.mark.skipif(not os.path.exists(GEF), reason="run the fixture scripts first")


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture(scope="module")
def ftp_tree(tmp_path_factory):
    from pyftpdlib.authorizers import DummyAuthorizer
    from pyftpdlib.handlers import FTPHandler
    from pyftpdlib.servers import ThreadedFTPServer
    from pyftpdlib.log import config_logging
    config_logging(level=logging.ERROR)
    root = tmp_path_factory.mktemp("mirror")
    chip = root / "pub" / "CNP0000001" / "CNS0000001" / "chipA"
    chip.mkdir(parents=True)
    if os.path.exists(GEF):
        shutil.copy(GEF, chip / "chipA.gef")
    raw = root / "pub" / "CNP0000001" / "CNS0000001" / "CNX0000001"
    raw.mkdir(parents=True)
    with open(raw / "lane1_read_1.fq.gz", "wb") as fh:
        fh.truncate(50 * 1024 ** 3)                          # sparse: 50 GB listed, 0 used
    (root / "pub" / "CNP0000001" / "CNS0000001" / "md5.txt").write_text("x\n")
    a = DummyAuthorizer(); a.add_anonymous(str(root))
    h = FTPHandler; h.authorizer = a; h.passive_ports = range(61000, 61200)
    port = _free_port()
    srv = ThreadedFTPServer(("127.0.0.1", port), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"ftp://127.0.0.1:{port}/pub/CNP0000001/CNS0000001/", str(root)
    srv.close_all()


def test_cngb_accessions_classify_and_route():
    for acc in ("CNP0002646", "CNS0513001", "CNX0000001", "CNR0000001"):
        assert g.classify(acc) == "cngb"
    r = g.acquisition_plan("CNP0002646", offline=True).routes[0]
    assert r.kind == "cngb-ftp" and "list_tree" in r.note


def test_listing_reports_sizes_without_downloading(ftp_tree):
    url, root = ftp_tree
    tree = g.list_tree(url)
    names = {e.name: e for e in tree}
    assert names["lane1_read_1.fq.gz"].size == 50 * 1024 ** 3
    s = g.summarise_tree(tree)
    assert s["by_type"][0][0] == ".fq.gz"                    # where the bytes are
    only_gef = g.list_tree(url, include=["*.gef"])
    assert all(e.name.endswith(".gef") for e in only_gef)


@need
def test_fetch_selected_builds_commands_not_a_mirror(ftp_tree):
    url, _ = ftp_tree
    sel = g.fetch_selected(g.list_tree(url), "/tmp/x", include=["*.gef"])
    assert sel["count"] == 1 and sel["bytes"] == os.path.getsize(GEF)
    assert sel["commands"][0].startswith("wget -c -x -nH")
    assert "-r" not in g.fetch_selected(g.list_tree(url), include=["*.gef"])["commands"][0].split()
    assert "-r" in g.wget_mirror_command(url).split()


@need
def test_remote_file_reads_exact_bytes_over_ftp(ftp_tree):
    url, _ = ftp_tree
    rf = g.RemoteFile(url + "chipA/chipA.gef", block="256KB", cache="4MB")
    local = open(GEF, "rb").read()
    assert rf.size == len(local)
    rng = np.random.default_rng(0)
    for off in [0, 1, 262_143, 262_144, len(local) - 10] + list(rng.integers(0, len(local) - 5000, 20)):
        rf.seek(int(off))
        assert rf.read(4096) == local[int(off):int(off) + 4096]
    rf.seek(len(local) - 70_000)
    assert rf.read(10 ** 9) == local[-70_000:]                 # an oversized read stops at EOF
    assert rf.read(10) == b""
    s = rf.stats()
    assert s["requests"] > 0 and s["cache_hits"] > 0
    rf.drop_cache()
    assert not rf._cache
    rf.close()


@need
def test_remote_file_over_http_range(tmp_path):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
    from range_server import serve
    shutil.copy(GEF, tmp_path / "c.gef")
    port = _free_port()
    httpd = serve(str(tmp_path), port)
    try:
        rf = g.RemoteFile(f"http://127.0.0.1:{port}/c.gef", block="512KB")
        local = open(GEF, "rb").read()
        rf.seek(123_456)
        assert rf.read(70_000) == local[123_456:193_456]
        rf.close()
    finally:
        httpd.shutdown()


@need
def test_remote_gef_matches_local_and_index_builds_from_remote(ftp_tree):
    url, _ = ftp_tree
    b = g.Budget(ram="128MB")
    v = g.open_remote_gef(url + "chipA/chipA.gef", budget=b, cache="32MB")
    assert v.gef.remote and v.gef.bins == g.open_gef(GEF).gef.bins
    assert b.stats()["ram"]["used"] >= 32 * 1024 ** 2         # the cache is reserved
    local = g.open_gef(GEF).gef
    local._sidx, local._sidx_checked = None, True
    from geminga.gefindex import _sidecar_dir
    shutil.rmtree(_sidecar_dir(v.gef.path), ignore_errors=True)
    w = (1000, 1300, 1200, 1500)
    a = v.gef.read_window(w, 20)
    c = local.read_window(w, 20)
    assert sorted(zip(a["x"], a["y"], a["count"])) == sorted(zip(c["x"], c["y"], c["count"]))
    si = v.gef.build_spatial_index(levels=[20])
    assert si.valid and si.levels == [20] and not si.dir.endswith(".gemidx")   # local cache dir
    rf = v.gef._fileobj
    before = rf.bytes_transferred
    d = v.gef.read_window(w, 20)
    assert rf.bytes_transferred == before                     # answered from the local sidecar
    assert sorted(zip(d["x"], d["y"], d["count"])) == sorted(zip(c["x"], c["y"], c["count"]))
    v.gef.close()
    assert b.stats()["ram"]["used"] == 0                      # closing released the cache


def test_remote_file_rejects_other_schemes():
    with pytest.raises(ValueError):
        g.RemoteFile("s3://bucket/key")
