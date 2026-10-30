"""Oversized local storage: vault -> staging -> device -> handoff, each bounded."""
import json
import os
import shutil
import numpy as np
import pytest
import geminga as g

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")
GEF = os.path.join(FIX, "chip.gef")
need = pytest.mark.skipif(not os.path.exists(GEF), reason="run the fixture scripts first")


@pytest.fixture(scope="module")
def vault_dir(tmp_path_factory):
    """A scratch vault: the chip and a few small files, copied. Scanning a vault writes its
    manifest (and opening a member its sidecars) into the scanned directory, so the shared
    fixtures are never scanned themselves."""
    root = tmp_path_factory.mktemp("vault")
    for name in ("chip.gef", "culture.parquet", "qc.arrow"):
        if os.path.exists(os.path.join(FIX, name)):
            shutil.copy(os.path.join(FIX, name), root / name)
    g.vault(str(root)).scan()
    return str(root)


@need
def test_vault_indexes_without_copying(vault_dir):
    v = g.vault(vault_dir)
    v.scan(profile=True)
    assert len(v) > 0
    assert v.total_bytes > 0
    m = v.get("chip.gef")
    assert m is not None and m.format == "Stereo-seq GEF"
    assert os.path.exists(v.abspath("chip.gef"))
    # scanning must never have written a copy of the data anywhere
    assert v.manifest_path.endswith(".json")
    assert os.path.getsize(v.manifest_path) < 1_000_000
    assert v.find(fmt="GEF")
    assert v.oversized("1KB")


@need
def test_vault_manifest_is_reused(vault_dir):
    v1 = g.vault(vault_dir)
    v1.scan(profile=True)
    n = len(v1)
    v2 = g.vault(vault_dir)                 # fresh object, same manifest
    assert len(v2) == n
    assert v2.get("chip.gef").format == "Stereo-seq GEF"


def test_staging_reserves_against_the_budget(tmp_path):
    b = g.Budget(ram="10MB", spill="100MB", timeout_s=5)      # a shortfall must not wait this out
    st = g.staging(str(tmp_path / "s1"), capacity="60MB", budget=b)
    assert b.available("spill") == 40 * 1024 ** 2
    with pytest.raises(MemoryError, match="cannot reserve"):
        g.staging(str(tmp_path / "s2"), capacity="60MB", budget=b)
    st.release()
    assert b.available("spill") == 100 * 1024 ** 2


@need
def test_extract_evicts_lru_and_respects_pins(tmp_path, vault_dir):
    st = g.staging(str(tmp_path / "s"), capacity="2MB")
    v = g.vault(vault_dir)
    view = v.open("chip.gef", budget=g.Budget(ram="128MB")).use_bin(20)
    st.extract("a", view, window=(1000, 2200, 1000, 2200), pin=True)
    st.extract("b", view, window=(2400, 3200, 2200, 3000))
    assert {"a", "b"} <= set(r.key for r in st)
    for i in range(4):                       # push past capacity
        st.extract(f"f{i}", view, window=(1000 + i * 40, 2400, 1000, 2400))
    assert "a" in st                         # pinned survives
    assert st.used <= st.capacity
    with pytest.raises(g.StagingFull, match="cannot fit"):
        tiny = g.staging(str(tmp_path / "t"), capacity="8KB")
        view.use_bin(1)
        tiny.extract("huge", view, window=(0, 4000, 0, 4000))


@need
def test_staged_region_reads_back_identically(tmp_path, vault_dir):
    st = g.staging(str(tmp_path / "s"), capacity="40MB")
    v = g.vault(vault_dir)
    view = v.open("chip.gef", budget=g.Budget(ram="128MB")).use_bin(20)
    win = (1000, 2200, 1000, 2200)
    r = st.extract("roi", view, window=win)
    direct = view.select(*win).load()
    tbl = st.read("roi")
    assert tbl.num_rows == r.rows == direct["x"].size
    assert int(tbl.column("count").to_numpy().sum()) == int(direct["count"].sum())
    assert st.get("roi").hits >= 1


def test_devices_always_lists_the_host():
    ds = g.devices()
    assert ds and ds[0].kind == "cpu"
    assert ds[0].total > 0
    assert ds[0].tier == "ram"
    for d in ds:
        assert d.id and isinstance(d.accounting_only, bool)


def test_device_pool_refuses_overcommit():
    b = g.Budget(ram="64MB")
    pool = g.device_pool("cpu", "16MB", budget=b)
    small = np.zeros(1_000_000, dtype="float32")        # 4 MB
    with pool.place(small, "small") as t:
        assert t is not None
        assert pool.report()["used"] == small.nbytes
    assert pool.report()["used"] == 0                   # released on exit
    with pytest.raises(MemoryError, match="does not fit"):
        pool.place(np.zeros(10_000_000, dtype="float64"))
    pool.release()
    assert b.available("ram") == 64 * 1024 ** 2
    with pytest.raises(ValueError):
        g.device_pool("gpu:99", "1MB")


@need
def test_harvest_writes_every_format_and_a_manifest(tmp_path, vault_dir):
    v = g.vault(vault_dir)
    view = v.open("chip.gef", budget=g.Budget(ram="128MB")).use_bin(50)
    view.select(1000, 2600, 1000, 2600)
    h = g.harvest(str(tmp_path / "h"), note="test")
    for fmt in ("parquet", "arrow", "ndjson", "csv"):
        r = h.add(view, f"r-{fmt}", fmt=fmt, targets=["regulus"])
        assert r.rows > 0 and r.bytes > 0 and r.sha256
        assert os.path.exists(os.path.join(h.path, r.path))
    assert all(h.verify().values())
    for fmt in ("parquet", "arrow", "ndjson", "csv"):
        assert h.read(f"r-{fmt}").num_rows > 0
    with pytest.raises(ValueError, match="already exists"):
        h.add(view, "r-parquet", fmt="parquet")
    # the manifest survives a fresh object
    again = g.harvest(str(tmp_path / "h"))
    assert len(again) == 4
    assert again.rois["r-arrow"].targets == ["regulus"]


@need
def test_dispatch_is_flat_and_target_keyed(tmp_path, vault_dir):
    v = g.vault(vault_dir)
    view = v.open("chip.gef", budget=g.Budget(ram="128MB")).use_bin(50)
    view.select(1000, 2600, 1000, 2600)
    h = g.harvest(str(tmp_path / "h"))
    h.add(view, "a", fmt="parquet", targets=["regulus", "prism"])
    h.add(view, "b", fmt="ndjson", targets=["prism"])
    h.add(view, "c", fmt="arrow", targets=["amethrine"])
    doc = json.load(open(h.write_dispatch()))
    assert doc["schema"] == "geminga.dispatch/2"
    assert all("kind" in i and "stat" in i for items in doc["targets"].values() for i in items)
    assert set(doc["targets"]) == {"regulus", "prism", "amethrine"}
    assert len(doc["targets"]["prism"]) == 2
    for item in doc["targets"]["prism"]:
        assert os.path.isabs(item["path"]) and os.path.exists(item["path"])
        assert item["sha256"] and item["rows"] > 0
    assert h.for_target("amethrine")[0].name == "c"
    # NDJSON must genuinely be one JSON object per line, for a Perl reader
    nd = [i for i in doc["targets"]["prism"] if i["format"] == "ndjson"][0]
    with open(nd["path"]) as fh:
        first = json.loads(fh.readline())
    assert "x" in first and "y" in first
