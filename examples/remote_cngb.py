"""A CNGB-style project over FTP, handled without mirroring it.

Against the real server:
    python examples/remote_cngb.py ftp://ftp.cngb.org/pub/CNSA/data2/CNP0002646/CNS0513001/

With no argument it builds a local mock of that layout (raw FASTQs as sparse files, so the tree
reports terabytes while using none) and serves it on 127.0.0.1:2121.
"""
import gzip
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import geminga as g


def mock_mirror(root="/tmp/cngbmirror"):
    base = os.path.join(root, "pub/CNSA/data2/CNP0002646/CNS0513001")
    if os.path.exists(root):
        shutil.rmtree(root)
    chip = os.path.join(base, "SS200000135TL_D1")
    os.makedirs(chip)
    shutil.copy("/tmp/gem/gbm/gbm_phantom.gef", os.path.join(chip, "SS200000135TL_D1.gef"))
    shutil.copy("/tmp/gem/chip.gef", os.path.join(chip, "SS200000135TL_D1.tissue.gef"))
    shutil.copy("/tmp/gem/gbm/gbm_phantom_ssdna.tif", os.path.join(chip, "SS200000135TL_D1_ssDNA.tif"))
    with open("/tmp/gem/chip.gem", "rb") as a, gzip.open(os.path.join(chip, "SS200000135TL_D1.gem.gz"), "wb") as b:
        b.write(a.read())
    # raw reads: sparse files, reported at full size in listings, zero bytes on disk
    for lane in range(1, 6):
        for r in (1, 2):
            d = os.path.join(base, f"CNX00{lane:05d}", f"CNR00{lane:05d}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"V300{lane:06d}_L0{lane}_read_{r}.fq.gz"), "wb") as fh:
                fh.truncate(int(215e9) if r == 1 else int(208e9))
    with open(os.path.join(base, "md5.txt"), "w") as fh:
        fh.write("# md5 manifest\n")
    return root, base


def serve(root, port=2121):
    from pyftpdlib.authorizers import DummyAuthorizer
    from pyftpdlib.handlers import FTPHandler
    from pyftpdlib.servers import ThreadedFTPServer
    from pyftpdlib.log import config_logging
    config_logging(level=logging.ERROR)
    a = DummyAuthorizer(); a.add_anonymous(root)
    h = FTPHandler; h.authorizer = a; h.passive_ports = range(60000, 60200)
    srv = ThreadedFTPServer(("127.0.0.1", port), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        root, base = mock_mirror()
        on_disk = int(subprocess.run(["du", "-sb", root], capture_output=True, text=True).stdout.split()[0])
        serve(root)
        url = "ftp://127.0.0.1:2121/pub/CNSA/data2/CNP0002646/CNS0513001/"
        print(f"mock mirror at {url}  (actually using {g.human(on_disk)} of disk)\n")

    print("=== 1. list the tree — directory listings only ===")
    t0 = time.perf_counter(); tree = g.list_tree(url)
    print(f"  listed in {time.perf_counter() - t0:.2f}s")
    g.print_tree_summary(tree, disk_free=shutil.disk_usage("/tmp").free)
    print("\n  the suggested mirror command would fetch all of it:")
    print("   ", g.wget_mirror_command(url))

    print("\n=== 2. fetch only what analysis needs ===")
    sel = g.fetch_selected(tree, "/data/cnp0002646", include=["*.gef"])
    print(f"  {sel['count']} GEF file(s), {sel['bytes_human']} — fits on disk: {sel['fits']}")
    for c in sel["commands"]:
        print("   ", c)

    print("\n=== 3. or open the GEF where it sits, over FTP ===")
    gef_url = [e.url for e in tree if e.name.endswith(".gef") and ".tissue" not in e.name][0]
    if len(sys.argv) == 1:
        # A sidecar from an earlier run is keyed by URL and size and would be reused — right
        # for a user, wrong for a demo that claims to measure first-contact costs.
        from geminga.gefindex import _sidecar_dir
        shutil.rmtree(_sidecar_dir(gef_url), ignore_errors=True)
    b = g.Budget(ram="256MB")
    t0 = time.perf_counter(); v = g.open_remote_gef(gef_url, budget=b)
    rf = v.gef._fileobj
    print(f"  opened in {time.perf_counter() - t0:.2f}s: {v.gef}")
    print(f"  bins {v.gef.bins}, {len(v.gef.genes(1))} genes, extent {v.gef.extent}")
    print(f"  so far: {rf.stats()['transferred_human']} of {rf.stats()['size_human']} in {rf.requests} requests")

    def step(label, fn):
        before, req = rf.bytes_transferred, rf.requests
        t0 = time.perf_counter(); out = fn(); dt = time.perf_counter() - t0
        print(f"  {label:<44} +{g.human(rf.bytes_transferred - before):>9} in "
              f"{rf.requests - req:>3} req, {dt:5.2f}s")
        return out

    ex = v.gef.extent
    step("whole-section overview at bin200", lambda: v.gef.rasterize(ex, 200, None, 20))
    step("one gene (VEGFA) across the section, bin20", lambda: v.gef.read_window(ex, 20, ["VEGFA"]))
    step("same gene again (cache)", lambda: v.gef.read_window(ex, 20, ["VEGFA"]))
    w = (1700, 1900, 1900, 2100)
    step("a 0.25% window at bin1, no index", lambda: v.gef.read_window(w, 1))

    print("\n=== 4. build a local spatial index from the remote file, for bin1 only ===")
    rf.drop_cache()     # charge the build its own cost, not the previous step's
    si = step("one sequential pass over bin1 + bin200", lambda: v.gef.build_spatial_index(levels=[1, 200]))
    print(f"  sidecar: {si}")
    d_remote = step("the same window, now via the local sidecar", lambda: v.gef.read_window(w, 1))
    local = g.open_gef("/tmp/gem/gbm/gbm_phantom.gef").gef
    local._sidx, local._sidx_checked = None, True
    d_local = local.read_window(w, 1)
    same = sorted(zip(d_remote["x"], d_remote["y"], d_remote["count"])) == \
        sorted(zip(d_local["x"], d_local["y"], d_local["count"]))
    print(f"  identical to reading the local file: {same} ({d_local['x'].size:,} observations)")

    s = rf.stats()
    tot = sum(e.size for e in tree)
    print(f"\n  total over the network: {s['transferred_human']} "
          f"({s['fraction_of_file']:.0%} of this GEF, {s['bytes_transferred'] / tot:.6%} of the "
          f"project), cache {s['cache_hits']} hits / {s['cache_misses']} misses, "
          f"budget peak {b.stats()['ram']['peak_human']}")
