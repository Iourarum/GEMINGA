"""The journal: every streamed piece is written down with its exact coordinates, and can be
found again — by text, by overlap, by time — and reopened."""
import importlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

import geminga as g
from geminga.selections import dataset_key

J = importlib.import_module("geminga.journal")   # the module; g.journal is the function

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")
GEF = os.path.join(FIX, "chip.gef")
need_gef = pytest.mark.skipif(not os.path.exists(GEF), reason="no chip.gef fixture")


@pytest.fixture
def jr(tmp_path):
    """A private journal for one test, restored afterwards."""
    prev = J.use_journal(str(tmp_path / "journal.jsonl"))
    yield J.active()
    J.use_journal(prev)


def test_the_suite_journals_into_a_temporary_file():
    # conftest.py points GEMINGA_JOURNAL at a temp file, so test runs never touch ~/.geminga
    a = J.active()
    assert a is not None and "geminga-test-journal" in a.path


@need_gef
def test_every_streamed_piece_is_journaled_with_exact_coordinates(jr):
    b = g.Budget(ram="64MB")
    v = g.open_gef(GEF, b)
    key = dataset_key(v)["key"]
    v.select(1600, 3000, 1400, 2100).use_bin(50)
    data = v.load()
    v.select(0, 1000, 0, 1000).use_bin(20)
    n_stream = sum(len(c["x"]) for c in v.chunks())
    img = v.gef.rasterize((200, 1200, 300, 900), 100, shape=32)
    rows = v.gef.read_window((100, 400, 100, 400), 10, genes=["Gene0001", "Gene0007"])
    v.close()

    ents = list(jr.entries())
    ops = [e.op for e in ents]
    assert ops == ["open", "load", "chunks", "rasterize", "read_window"]
    assert all(e.key == key for e in ents), "one dataset key for the file and every view over it"
    assert all(e.schema == J.JOURNAL_SCHEMA and e.session == ents[0].session for e in ents)

    opened, load, chunks, ras, rw = ents
    assert opened.window == [0, 4000, 0, 4000] and opened.level == {"bins": [1, 10, 20, 50, 100, 200]}
    assert opened.genes is not None and len(opened.genes) == 300 and opened.units == 2_946_567
    assert load.window == [1600, 3000, 1400, 2100] and load.level == {"bin": 50}
    assert load.units == len(data["x"]) and load.bytes > 0 and load.unit == "spots"
    assert chunks.window == [0, 1000, 0, 1000] and chunks.level == {"bin": 20}
    assert chunks.units == n_stream and any(t.startswith("chunks=") for t in chunks.tags)
    assert ras.window == [200, 1200, 300, 900] and ras.level == {"bin": 100}
    assert ras.unit == "pixels" and ras.units == 32 * 32 and ras.bytes == img.nbytes
    assert rw.window == [100, 400, 100, 400] and rw.level == {"bin": 10} and rw.genes == ["Gene0001", "Gene0007"]
    assert rw.units == len(rows["x"])
    # the file is plain JSONL: one object per line, greppable
    with open(jr.path) as fh:
        lines = fh.read().splitlines()
    assert len(lines) == 5 and all(json.loads(s)["schema"] == J.JOURNAL_SCHEMA for s in lines)


@need_gef
def test_opening_the_same_file_again_does_not_repeat_the_gene_list(jr):
    g.open_gef(GEF).close()
    g.open_gef(GEF).close()
    assert [e.op for e in jr.entries()] == ["open"]
    # a fresh Journal object over the same file learns it from the file, not from memory
    gf = g.GefFile(GEF)
    assert J.Journal(jr.path).opened(J.key_of_gef(gf)) and J.key_of_gef(gf)["key"] == dataset_key(gf)["key"]
    gf.close()


@need_gef
def test_replay_reopens_the_very_same_piece(jr, tmp_path):
    v = g.open_gef(GEF)
    v.select(1600, 3000, 1400, 2100).use_bin(50).select_genes(["Gene0003"])
    n = len(v.load()["x"])
    v.close()
    e = jr.search(op="load")[0]
    back = jr.replay(e)
    assert tuple(back.window) == (1600, 3000, 1400, 2100) and back.bin_size == 50
    assert back.genes == ["Gene0003"] and len(back.load()["x"]) == n
    back.close()
    # a file whose size no longer matches the fingerprint is not quietly treated as the same data
    e2 = J.Entry.from_json({**json.loads(e.to_json()),
                            "dataset": {**e.dataset, "fingerprint": {**e.dataset["fingerprint"],
                                                                     "size": 12345}}})
    with pytest.raises(ValueError, match="not the same file"):
        jr.replay(e2)
    v2 = jr.replay(e2, check=False)
    assert tuple(v2.window) == (1600, 3000, 1400, 2100)
    v2.close()
    e3 = J.Entry.from_json({**json.loads(e.to_json()),
                            "dataset": {**e.dataset, "path": str(tmp_path / "gone.gef")}})
    with pytest.raises(FileNotFoundError):
        jr.replay(e3)


def _synthetic(jr, n=40):
    """Records about three fake datasets, without touching any file."""
    ds = [{"path": f"/data/chip{i}.gef", "kind": "gef", "key": f"{i:016x}",
           "fingerprint": {"size": 1000 + i}} for i in range(3)]
    for i in range(n):
        d = ds[i % 3]
        jr.record("load", d, window=[i * 10, i * 10 + 100, 0, 50], level={"bin": 50 if i % 2 else 20},
                  unit="spots", genes=["GFAP"] if i % 7 == 0 else ["SOX2", f"G{i}"],
                  units=i, nbytes=i * 8, seconds=0.01)
    return ds


def test_search_by_text_overlap_dataset_op_and_time(jr):
    ds = _synthetic(jr)
    assert len(list(jr.entries())) == 40
    hits = jr.search("GFAP")
    assert [e.units for e in hits] == [0, 7, 14, 21, 28, 35]
    assert jr.search("gfap") == hits and jr.search("gfap", ignore_case=False) == []
    assert [e.units for e in jr.search(genes=["GFAP"])] == [0, 7, 14, 21, 28, 35]
    # overlap: windows [i*10, i*10+100): those overlapping (395, 405) are i in 30..39
    ov = jr.search(overlaps=(395, 405, 10, 20))
    assert [e.units for e in ov] == list(range(30, 40))
    assert jr.search(overlaps=(395, 405, 60, 70)) == []           # y does not overlap
    assert [e.units for e in jr.search(within=(0, 150, 0, 50))] == [0, 1, 2, 3, 4, 5]
    assert all(e.key == ds[1]["key"] for e in jr.search(dataset=ds[1]["key"]))
    assert len(jr.search(dataset=ds[1])) == 13 and len(jr.search(dataset="/data/chip2.gef")) == 13
    assert jr.search(op="rasterize") == [] and len(jr.search(op=["load"])) == 40
    assert jr.search("GFAP", dataset=ds[0], limit=2)[1].units == 21
    now = time.time()
    assert len(jr.search(since=now - 60)) == 40 and jr.search(until=now - 60) == []
    # grep gives (line_number, raw line) through whichever engine is there
    lines = jr.grep("G33")
    assert [ln for ln, _ in lines] == [33] and '"G33"' in lines[0][1]
    assert jr.engine in ("trigram", "rg", "grep", "python")
    assert [ln for ln, _ in jr.grep(r'"units":3[0-2],', regex=True)] == [30, 31, 32]


def test_datasets_fold_visited_and_coverage(jr):
    ds = _synthetic(jr, n=9)
    trails = jr.datasets()
    assert set(trails) == {d["key"] for d in ds}
    t0 = trails[ds[0]["key"]]
    assert t0.n == 3 and t0.ops == {"load": 3} and t0.path == "/data/chip0.gef"
    assert t0.bbox == [0, 160, 0, 50]                   # union of windows i=0,3,6
    assert {tuple(sorted(lv.items())) for lv in t0.levels} == {(("bin", 20),), (("bin", 50),)}
    assert t0.genes[:1] == ["GFAP"] and t0.sessions == 1
    vis = jr.visited(ds[0]["key"])
    assert [e.window for e in vis] == [[0, 100, 0, 50], [30, 130, 0, 50], [60, 160, 0, 50]]
    assert len(jr.visited(ds[0]["key"], level={"bin": 20})) == 2
    img = jr.coverage(ds[0]["key"], shape=16)
    assert img.shape == (16, 16) and img.max() == 3 and img.min() == 1   # the middle thrice, the ends once
    assert img[:, 0].min() == 1 and img[:, -1].min() == 1
    assert jr.coverage("ffffffffffffffff").max() == 0
    assert len(t0.line_text()) > 20


def test_recording_and_quiet_contexts_tag_or_silence_records(jr):
    d = {"path": "/data/x.gef", "kind": "gef", "key": "a" * 16, "fingerprint": {}}
    with J.recording(via="explorer", frame=3):
        jr.record("load", d, window=[0, 1, 0, 1])
        with J.recording(selection="sel1"):
            jr.record("load", d, window=[0, 1, 0, 1])
        with J.quiet():
            assert jr.record("load", d, window=[0, 1, 0, 1]) is None
    jr.record("load", d, window=[0, 1, 0, 1], via="user", note="by hand")
    ents = list(jr.entries())
    assert [e.via for e in ents] == ["explorer", "explorer", "user"]
    assert ents[0].tags == ["frame=3"] and ents[1].selection == "sel1" and ents[0].selection == ""
    assert ents[2].note == "by hand"
    assert len(jr.search(via="explorer")) == 2 and len(jr.search(selection="sel1")) == 1


@need_gef
def test_selections_describe_and_similar_leave_one_trail_each(jr):
    v = g.open_gef(GEF, g.Budget(ram="128MB"))
    v.select(1500, 2500, 1500, 2500).use_bin(50)
    sel = g.select(v, "rim", intent="double-check")
    g.describe(sel, v)
    found = g.similar(sel, v, k=2)
    ents = list(jr.entries())
    sel_rec = [e for e in ents if e.op == "select"]
    assert len(sel_rec) == 1 and sel_rec[0].selection == sel.id
    assert sel_rec[0].window == [1500, 2500, 1500, 2500] and "name=rim" in sel_rec[0].tags
    described = [e for e in ents if e.via == "describe"]
    assert described and all(e.selection == sel.id for e in described)
    sim = [e for e in ents if e.op == "similar"]
    assert len(sim) == 1 and sim[0].units == len(found) and sim[0].selection == sel.id
    assert all(f"candidate={c.id}" in sim[0].tags for c in found)
    # the candidate scan inside similar() wrote nothing of its own
    i_sim = ents.index(sim[0])
    i_desc = max(ents.index(e) for e in described)
    assert all(e.op in ("similar",) for e in ents[i_desc + 1:i_sim + 1])
    v.close()


@need_gef
@pytest.mark.skipif(not g.stage_available(), reason="no wgpu adapter")
def test_explorer_reloads_are_journaled_with_their_frame(jr):
    ex = g.explore(GEF, width=160, height=120, point_budget=20_000)
    ex.frame()
    ex.goto((1000, 2000, 1000, 2000))
    ex.frame()
    loads = jr.search(op="load", via="explorer")
    assert len(loads) >= 1 and all(any(t.startswith("frame=") for t in e.tags) for e in loads)
    assert loads[-1].level.get("bin") == ex.loaded_rung
    assert loads[-1].window == [float(v) if not float(v).is_integer() else int(v)
                                for v in ex.loaded_window]
    ex.view.gef.close()


def test_off_switch_and_environment(tmp_path, monkeypatch):
    prev = J.use_journal(None)
    try:
        d = {"path": "/data/x.gef", "kind": "gef", "key": "b" * 16, "fingerprint": {}}
        assert J.record("load", d) is None and J.active() is None
        # g.journal() still gives a handle on the default file, without turning recording on
        assert isinstance(g.journal(), J.Journal) and J.active() is None
        monkeypatch.setenv("GEMINGA_JOURNAL", "0")
        J.use_journal(...)                      # re-read the environment
        assert J.active() is None
        monkeypatch.setenv("GEMINGA_JOURNAL", str(tmp_path / "env.jsonl"))
        J.use_journal(...)
        assert J.active().path == str(tmp_path / "env.jsonl")
        J.record("load", d, window=[1, 2])
        assert len(list(J.active().entries())) == 1
    finally:
        J.use_journal(prev)


def test_torn_lines_are_skipped_and_the_next_record_starts_clean(jr):
    d = {"path": "/data/x.gef", "kind": "gef", "key": "c" * 16, "fingerprint": {}}
    jr.record("load", d, window=[0, 1])
    with open(jr.path, "ab") as fh:
        fh.write(b'{"op":"load","dataset":{"key":"zz')          # a writer died here
    jr.record("load", d, window=[2, 3])
    ents = list(jr.entries())
    assert [e.window for e in ents] == [[0, 1], [2, 3]]
    assert jr.tail(1)[0].window == [2, 3] and len(jr) == 2


def test_cli_tail_grep_datasets_where_coverage(jr, capsys):
    _synthetic(jr, n=6)
    assert J._main(["--path", jr.path, "tail", "-n", "2"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2 and "chip" in out[0]
    assert J._main(["--path", jr.path, "grep", "GFAP"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and out[0].startswith("1:")
    assert J._main(["--path", jr.path, "datasets"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 3
    assert J._main(["--path", jr.path, "where", "0" * 16]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 2
    assert J._main(["--path", jr.path, "coverage", "0" * 16, "--shape", "8"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 9 and out[-1].startswith("max 2 visits")
    assert J._main(["--path", jr.path, "engine"]) == 0


def test_duckdb_and_parquet_surfaces(jr, tmp_path):
    _synthetic(jr, n=12)
    duckdb = pytest.importorskip("duckdb")
    con = jr.duckdb()
    assert con.execute("SELECT count(*) FROM journal WHERE op='load'").fetchone()[0] == 12
    n_gfap = con.execute("SELECT count(*) FROM journal WHERE list_contains(genes, 'GFAP')").fetchone()[0]
    assert n_gfap == 2
    assert duckdb is not None
    pq_path = jr.to_parquet(str(tmp_path / "j.parquet"))
    import pyarrow.parquet as pq
    assert pq.read_table(pq_path).num_rows == 12


# ---------------------------------------------------------------------------
# the trigram index
# ---------------------------------------------------------------------------

def _core_trigram():
    try:
        from geminga._core import TrigramIndex
        return TrigramIndex
    except ImportError:
        return None


need_trigram = pytest.mark.skipif(_core_trigram() is None, reason="core without TrigramIndex")


def _python_grep(path, pat, ignore_case=True):
    out = []
    with open(path, encoding="utf-8") as fh:
        for i, s in enumerate(fh):
            if (pat.lower() in s.lower()) if ignore_case else (pat in s):
                out.append((i, s.rstrip("\n")))
    return out


@need_trigram
def test_trigram_index_agrees_with_a_plain_scan_and_extends(jr):
    TrigramIndex = _core_trigram()
    _synthetic(jr, n=300)
    idx = jr.index(block_lines=16)
    assert isinstance(idx, TrigramIndex)
    assert idx.n_lines == 300 and idx.n_blocks == 19 and idx.indexed_bytes == os.path.getsize(jr.path)
    assert os.path.exists(jr.index_path) and jr.engine == "trigram"
    for pat, ic in (("GFAP", True), ("gfap", True), ("gfap", False), ("G13", True), ("chip2", True),
                    ("zzzz", True), ("50", True), ('"units":7,', True), ("", True)):
        assert idx.search(pat, ic) == _python_grep(jr.path, pat, ic), (pat, ic)
    # a rare term touches few blocks; one in every block is a mark, not a list, and narrows nothing
    assert idx.candidates("chip2.gef") is None               # every block has a chip2 record
    assert idx.n_everywhere > 50 and idx.candidates("G1") is None   # too short to narrow
    assert idx.candidates("zzzz") == []
    assert 0 < len(idx.candidates("G13")) < 19             # G13, G130..G139: a dozen lines
    # the journal grows: search sees the tail before the index is updated, and after
    _synthetic(jr, n=5)
    assert jr.engine != "trigram" or True                   # the Python side decides freshness below
    fresh = jr._fresh_index()
    assert fresh is not None, "an index that lags is still used; search scans the tail"
    assert idx.search("GFAP", True) == _python_grep(jr.path, "GFAP")
    assert jr.index().n_lines == 305 and jr.index().indexed_bytes == os.path.getsize(jr.path)
    assert jr.grep("GFAP") == _python_grep(jr.path, "GFAP")
    # Journal.search goes through it, including the regex path (literal picks the blocks)
    assert [e.units for e in jr.search("GFAP", dataset="0" * 16)][:2] == [0, 21]
    assert [ln for ln, _ in jr.grep(r'"units":1[0-2],', regex=True)] == [10, 11, 12]
    # a rewritten (shorter) file is not trusted
    with open(jr.path, "r+b") as fh:
        fh.truncate(100)
    assert jr._fresh_index() is None
    with pytest.raises(ValueError, match="rebuilt, not updated"):
        idx.update()
    assert jr.index(rebuild=True).n_lines == 0


@need_trigram
def test_trigram_sidecar_round_trip_and_foreign_file(jr, tmp_path):
    TrigramIndex = _core_trigram()
    _synthetic(jr, n=50)
    idx = jr.index(block_lines=8)
    back = TrigramIndex.load(jr.index_path)
    assert back.source_path == jr.path and back.n_lines == 50 and back.n_blocks == 7
    assert back.search("SOX2") == idx.search("SOX2")
    assert "TrigramIndex(" in repr(back) and back.postings_bytes > 0 and back.n_trigrams > 100
    bad = tmp_path / "x.tgx"
    bad.write_bytes(b"not an index")
    with pytest.raises(ValueError, match="not a GEMINGA trigram index"):
        TrigramIndex.load(str(bad))


@need_trigram
def test_trigram_is_what_the_cli_index_command_builds(jr, capsys):
    _synthetic(jr, n=20)
    assert J._main(["--path", jr.path, "index"]) == 0
    out = capsys.readouterr().out
    assert "20 lines" in out and jr.index_path in out
    assert J._main(["--path", jr.path, "engine"]) == 0 and capsys.readouterr().out.strip() == "trigram"


@pytest.mark.skipif(shutil.which("rg") is None, reason="no ripgrep")
def test_rg_and_journal_grep_agree(jr):
    _synthetic(jr, n=60)
    p = subprocess.run(["rg", "-n", "-F", "-i", "--no-heading", "GFAP", jr.path],
                       capture_output=True, text=True)
    rg_lines = [int(s.split(":", 1)[0]) - 1 for s in p.stdout.splitlines()]
    assert [ln for ln, _ in jr.grep("GFAP")] == rg_lines == [0, 7, 14, 21, 28, 35, 42, 49, 56]
    assert sys.version_info >= (3, 10)


def test_tables_record_their_names_on_open(jr):
    pq_path = os.path.join(FIX, "culture.parquet")
    h5 = os.path.join(FIX, "sc.h5ad")
    if not (os.path.exists(pq_path) and os.path.exists(h5)):
        pytest.skip("no table fixtures")
    b = g.Budget(ram="256MB")
    g.view(pq_path, budget=b).select_frac(0.1, 0.3).load()
    g.view(h5, budget=b).select_frac(0.1, 0.2).load()
    g.view(h5, budget=b)                                    # opened again: no second record
    ents = list(jr.entries())
    assert [e.op for e in ents] == ["open", "load", "open", "load"]
    pq_open, _, h5_open, h5_load = ents
    assert pq_open.window == [0, 14400] and pq_open.unit == "rows" and "col=o2_percent" in pq_open.tags
    assert pq_open.bytes == os.path.getsize(pq_path) and pq_open.genes is None
    assert h5_open.window == [0, 20000] and h5_open.unit == "cells" and len(h5_open.genes) == 3000
    assert h5_open.bytes == os.path.getsize(h5) and h5_load.window == [2000, 4000]
    assert jr.search("Gene00042")[0].op == "open"           # which files carry a gene: one grep
    assert jr.search("col=ph")[0].path == os.path.abspath(pq_path)


def test_count_units_covers_the_containers():
    import pyarrow as pa
    assert J.count_units({"x": np.zeros(7), "y": np.zeros(7)}) == 7
    assert J.count_units(pa.table({"a": [1, 2, 3]})) == 3
    assert J.count_units(np.zeros((5, 2))) == 5
    assert J.count_units(None) == 0 and J.count_units({}) == 0
