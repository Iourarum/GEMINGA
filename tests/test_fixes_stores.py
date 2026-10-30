"""Regression tests for defects in the stores (catalog, selections, library) and in the planners
(acquire, profile). Each test names the behaviour that used to be wrong.

Everything here runs offline: network calls are replaced by local mocks, and the fixtures that
the wider suite uses are only touched by the guard tests at the end, which skip without them.
"""
import gzip
import importlib
import json
import os
import random
import struct
import sys
import time
import urllib.request
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import pytest
import geminga as g

# `geminga.acquire` is a module, but `geminga.profile` / `.catalog` / `.selections` / `.library`
# are shadowed by the functions of the same name, so reach the modules through importlib.
acq = importlib.import_module("geminga.acquire")
sel_mod = importlib.import_module("geminga.selections")

FIX = os.path.abspath(os.environ.get("GEMINGA_FIXTURES", "fixtures"))


def _sel(sid, **kw):
    """A small interval selection with no dataset behind it."""
    return g.Selection(sid, kw.pop("name", sid), kw.pop("dataset", {"key": "k", "path": "/x"}),
                       "interval", {"lo": 0, "hi": 5}, **kw)


def _is_json(line):
    try:
        json.loads(line)
        return True
    except ValueError:
        return False


# ============================================================================
# 1. Torn tails and interleaved appends
# ============================================================================

def test_selection_append_after_a_torn_tail_keeps_both_records(tmp_path):
    b = g.selections(str(tmp_path / "s.jsonl"))
    b.add(_sel("id1"))
    with open(b.path, "a") as fh:
        fh.write('{"event": "add", "id": "torn", "name"')            # a writer died mid-line
    b.add(_sel("id2"))
    assert set(g.selections(b.path).all()) == {"id1", "id2"}          # a fresh reader loses neither
    assert set(b.all()) == {"id1", "id2"}
    lines = open(b.path).read().splitlines()
    assert [_is_json(x) for x in lines] == [True, False, True]       # the tear sits on its own line


def test_catalog_append_after_a_torn_tail_keeps_both_records(tmp_path):
    cat = g.catalog(str(tmp_path / "c.jsonl"))
    cat.record("a", "/a")
    with open(cat.path, "a") as fh:
        fh.write('{"event": "record", "key": "torn", "target"')
    cat.record("b", "/b")
    assert set(g.catalog(cat.path).entries()) == {"a", "b"}
    assert set(cat.entries()) == {"a", "b"}


# Windows may stamp a file's mtime at close, so "the cache is still current" cannot be asserted there.
warm = pytest.mark.skipif(sys.platform == "win32", reason="mtime is stamped at close on Windows")


@warm
def test_a_warm_cache_is_advanced_not_dropped_across_a_tail_repair(tmp_path):
    b = g.selections(str(tmp_path / "s.jsonl"))
    b.add(_sel("id1"))
    with open(b.path, "a") as fh:
        fh.write('{"event": "add", "id": "torn"')
    assert set(b.all()) == {"id1"}                                    # cache built over the torn file
    b.add(_sel("id2"))
    assert b._fold is not None and b._fold[0] == b._signature()      # the repair's newline is counted
    assert set(b.all()) == {"id1", "id2"}
    assert set(g.selections(b.path).all()) == {"id1", "id2"}
    cat = g.catalog(str(tmp_path / "c.jsonl"))
    cat.record("a", "/a")
    with open(cat.path, "a") as fh:
        fh.write('{"event": "record", "key": "torn"')
    assert set(cat.entries()) == {"a"}
    cat.record("b", "/b")
    assert cat._fold is not None and cat._fold[0] == cat._signature()
    assert set(cat.entries()) == {"a", "b"}
    assert set(g.catalog(cat.path).entries()) == {"a", "b"}


def test_a_log_that_starts_with_a_torn_fragment_still_takes_the_next_record(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"event": "record", "key": "torn"')                 # the whole file is a tear
    cat = g.catalog(str(p))
    cat.record("a", "/a")
    assert set(g.catalog(str(p)).entries()) == {"a"}
    q = tmp_path / "s.jsonl"
    q.write_text('{"event": "add", "id": "torn"')
    b = g.selections(str(q))
    b.add(_sel("id1"))
    assert set(g.selections(str(q)).all()) == {"id1"}


def test_clean_logs_gain_no_blank_lines(tmp_path):
    b = g.selections(str(tmp_path / "s.jsonl"))
    for i in range(3):
        b.add(_sel(f"id{i}"))
    cat = g.catalog(str(tmp_path / "c.jsonl"))
    for i in range(3):
        cat.record(f"k{i}", "/x")
    for p in (b.path, cat.path):
        text = open(p).read()
        assert text.endswith("\n") and "\n\n" not in text and len(text.splitlines()) == 3


def _interleave(store, line):
    """Make `store._signature` append `line` to the log exactly once, right after it has read the
    signature: the moment another process would slip in between a cache check and a write."""
    real = store._signature
    state = {"fired": False}

    def hooked():
        sig = real()
        if not state["fired"]:
            state["fired"] = True
            with open(store.path, "a", encoding="utf-8") as fh:
                fh.write(line)
        return sig

    store._signature = hooked
    return state


def test_selection_cache_is_dropped_when_another_process_appends_mid_write(tmp_path):
    b = g.selections(str(tmp_path / "s.jsonl"))
    b.add(_sel("id1"))
    assert set(b.all()) == {"id1"}                                    # cache is warm and current
    other = json.dumps({"event": "add", **asdict(_sel("other"))}) + "\n"
    state = _interleave(b, other)
    b.add(_sel("id2"))
    del b._signature
    assert state["fired"]
    assert set(b.all()) == {"id1", "other", "id2"}                    # not just {id1, id2}
    assert set(g.selections(b.path).all()) == {"id1", "other", "id2"}


def test_catalog_cache_is_dropped_when_another_process_appends_mid_write(tmp_path):
    cat = g.catalog(str(tmp_path / "c.jsonl"))
    cat.record("a", "/a")
    assert set(cat.entries()) == {"a"}
    other = json.dumps({"event": "record", "key": "other", "target": "/o", "ts": 1.0,
                        "schema": g.CATALOG_SCHEMA}) + "\n"
    state = _interleave(cat, other)
    cat.note_streamed("a", 10)
    del cat._signature
    assert state["fired"]
    assert set(cat.entries()) == {"a", "other"}
    assert cat.get("a").bytes_streamed == 10
    assert set(g.catalog(cat.path).entries()) == {"a", "other"}


@warm
def test_clean_appends_keep_the_cache_warm(tmp_path):
    """The fix must not simply disable the cache: a lone writer's loop still never re-parses."""
    b = g.selections(str(tmp_path / "s.jsonl"))
    b.add(_sel("id1"))
    b.all()
    b.add(_sel("id2"))
    assert b._fold is not None and b._fold[0] == b._signature()
    assert set(b.all()) == {"id1", "id2"}
    cat = g.catalog(str(tmp_path / "c.jsonl"))
    cat.record("a", "/a")
    cat.entries()
    cat.note_streamed("a", 5)
    assert cat._fold is not None and cat._fold[0] == cat._signature()
    assert cat.get("a").bytes_streamed == 5
    assert g.catalog(cat.path).get("a").bytes_streamed == 5


# ============================================================================
# 2. describe / leads never move a record backwards
# ============================================================================

def test_describe_any_and_leads_any_never_pull_a_reviewed_record_back(tmp_path):
    lib = g.library(str(tmp_path / "lib.jsonl"))
    s = g.flag_genes(["EGFR"], lib=lib)
    lib.confirm(s, "checked by hand")
    assert s.status == "open"                                         # the caller's copy is stale
    g.leads_any(s, lib=lib)
    assert lib.get(s.id).status == "confirmed" and s.status == "confirmed"
    assert lib.get(s.id).leads                                        # the leads were still stored
    g.describe_any(s, lib=lib)
    assert lib.get(s.id).status == "confirmed"
    for st in ("in-review", "refuted", "resolved"):
        t = g.flag_genes(["CA9"], lib=lib)
        lib.set_status(t, st)
        g.leads_any(t, lib=lib)
        g.describe_any(t, lib=lib)
        assert lib.get(t.id).status == st


def test_status_advances_open_described_searched_and_no_further_back(tmp_path):
    lib = g.library(str(tmp_path / "lib.jsonl"))
    t = g.flag_genes(["CA9"], lib=lib)
    g.describe_any(t, lib=lib)
    assert lib.get(t.id).status == "described"
    g.leads_any(t, lib=lib)
    assert lib.get(t.id).status == "searched"
    t.status = "open"                                                 # a stale copy
    g.describe_any(t, lib=lib)
    assert lib.get(t.id).status == "searched"                         # not back to 'described'
    lib.set_status(t, "open", "explicit moves stay free")
    assert lib.get(t.id).status == "open"
    u = g.flag_genes(["VEGFA"], lib=lib)                              # open -> searched directly
    g.leads_any(u, lib=lib)
    assert lib.get(u.id).status == "searched"


def test_advance_helper_is_the_single_gate(tmp_path):
    b = g.selections(str(tmp_path / "s.jsonl"))
    b.add(_sel("a"))
    assert b._advance("a", "searched").status == "searched"
    assert b._advance("a", "described").status == "searched"          # never backwards
    b.update("a", status="in-review")
    assert b._advance("a", "searched").status == "in-review"          # nor over a reviewer
    assert b.get("a").status == "in-review"
    with pytest.raises(KeyError):
        b._advance("nope", "described")


def test_selection_describe_and_leads_respect_review_status(tmp_path):
    pl = pytest.importorskip("polars")
    pq = str(tmp_path / "series.parquet")
    pl.DataFrame({"t": list(range(100)), "v": [float(i) for i in range(100)]}).write_parquet(pq)
    b = g.selections(str(tmp_path / "s.jsonl"))
    sel = g.select_values(pq, "t", 10, 20, "win", book=b)
    b.update(sel, status="in-review")
    assert sel.status == "open"                                       # stale in the caller's hands
    g.describe(sel, book=b)
    assert b.get(sel.id).status == "in-review" and sel.status == "in-review"
    assert b.get(sel.id).summary["rows"] == 10
    g.leads(sel, book=b)
    assert b.get(sel.id).status == "in-review" and b.get(sel.id).leads
    fresh = g.select_values(pq, "t", 30, 40, "other", book=b)
    g.describe(fresh, book=b)
    assert b.get(fresh.id).status == "described"
    g.leads(fresh, book=b)
    assert b.get(fresh.id).status == "searched" and fresh.status == "searched"
    # without a book the object itself is the record
    s3 = g.Selection("x", "x", {}, "interval", {"lo": 0, "hi": 1}, status="confirmed")
    g.leads(s3)
    assert s3.status == "confirmed"


# ============================================================================
# 3. Naive datetimes are UTC, whatever the machine's zone
# ============================================================================

@pytest.fixture
def set_tz():
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is not available on this platform")
    old = os.environ.get("TZ")

    def _set(name):
        os.environ["TZ"] = name
        time.tzset()

    yield _set
    if old is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = old
    time.tzset()


def test_naive_datetimes_convert_as_utc_in_every_zone(set_tz):
    naive = datetime(2020, 1, 1, 12, 0, 0)
    seen = []
    for name, west in (("UTC", 0), ("IST-5:30", -19800), ("PST8", 28800)):
        set_tz(name)
        assert time.timezone == west                                  # the zone really changed
        seen.append(sel_mod._num(naive))
    assert seen == [1577880000.0] * 3
    aware = datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=2)))
    assert sel_mod._num(aware) == 1577880000.0 - 7200                 # aware values are untouched
    assert sel_mod._num(5) == 5.0 and sel_mod._num("2.5") == 2.5


def test_a_datetime_window_selects_the_same_rows_in_any_zone(tmp_path, set_tz):
    pl = pytest.importorskip("polars")
    pq = str(tmp_path / "ts.parquet")
    pl.DataFrame({"t": [datetime(2020, 1, 1, h) for h in range(24)],
                  "v": [float(h) for h in range(24)]}).write_parquet(pq)
    means = {}
    for name in ("UTC", "IST-5:30", "PST8"):
        set_tz(name)
        sel = g.select_values(pq, "t", datetime(2020, 1, 1, 6), datetime(2020, 1, 1, 12), "morning")
        m = g.describe(sel)
        means[name] = (m["rows"], m["series"]["v"]["inside"])
    assert set(means.values()) == {(6, 8.5)}                          # hours 6..11, not a shifted set


# ============================================================================
# 4. Crashes, bad intervals, lossy sequences, relative paths
# ============================================================================

def test_repr_of_records_without_coordinates_does_not_crash():
    items = [g.flag_sequence("ACGTACGTAC"), g.flag_sequence(records=["r1"]),
             g.flag_variant(rsid="rs334"), g.flag_sequence("ACGT" * 30, name="long")]
    for s in items:
        assert repr(s).startswith("Selection(")
        assert len(s.bounds) == 2 and None not in s.bounds
        assert s.area >= 0
    assert g.flag_sequence("ACGTACGTAC").bounds == (0, 10)
    assert g.flag_variant(rsid="rs334").overlaps(g.flag_locus("chr1:1-100")) == 0.0
    placed = g.flag_variant("7", 100, "A", "C")
    assert "99" in repr(placed) and "100" in repr(placed)


def test_to_bed_skips_records_without_coordinates(tmp_path):
    lib = g.library(str(tmp_path / "lib.jsonl"))
    g.flag_variant(rsid="rs334", lib=lib)
    g.flag_variant("7", 100, "A", "C", lib=lib)
    g.flag_locus("chr6:100-200", "locus", lib=lib)
    with pytest.warns(UserWarning, match="without coordinates"):
        path = lib.to_bed(str(tmp_path / "x.bed"))
    rows = [x.split("\t") for x in open(path).read().splitlines()]
    assert len(rows) == 2 and {len(r) for r in rows} == {6}


def test_to_bed_will_not_mix_genome_builds(tmp_path):
    lib = g.library(str(tmp_path / "lib.jsonl"))
    g.flag_locus("chr1:100-200", "on38", lib=lib)
    g.flag_locus("chr1:100-200", "on37", build="hg19", lib=lib)
    with pytest.raises(ValueError, match="GRCh37"):
        lib.to_bed(str(tmp_path / "mixed.bed"))
    assert not os.path.exists(tmp_path / "mixed.bed")                 # nothing half-written
    t37 = open(lib.to_bed(str(tmp_path / "37.bed"), build="hg19")).read()
    t38 = open(lib.to_bed(str(tmp_path / "38.bed"), build="GRCh38")).read()
    assert "on37" in t37 and "on38" not in t37
    assert "on38" in t38 and "on37" not in t38
    one = g.library(str(tmp_path / "one.jsonl"))                      # a single build needs no argument
    g.flag_locus("chr2:5-10", "only", lib=one)
    assert "only" in open(one.to_bed(str(tmp_path / "one.bed"))).read()


@pytest.mark.parametrize("text,expect", [
    ("chr1:0-100", ("1", 0, 100)),            # browsers are 1-based: a 0 start is clamped, not -1
    ("chr1:0", ("1", 0, 1)),
    ("chrM:0-16569", ("M", 0, 16569)),
    ("chr1:1-100", ("1", 0, 100)),            # the ordinary case is unchanged
    ("chr7:55,019,017-55,211,628", ("7", 55_019_016, 55_211_628)),
])
def test_parse_locus_never_yields_a_negative_start(text, expect):
    assert g.parse_locus(text) == expect


def test_flag_sequence_only_converts_u_to_t_in_nucleotides():
    assert g.flag_sequence("MKVLUAGGH").geometry["sequence"] == "MKVLUAGGH"   # selenocysteine
    assert g.flag_sequence("acgu").geometry["sequence"] == "ACGT"
    assert g.flag_sequence("ACGUNNACGU").geometry["sequence"] == "ACGTNNACGT"
    assert g.flag_sequence("uuuu").geometry["sequence"] == "TTTT"


def test_relative_dataset_paths_are_absolutised(tmp_path, monkeypatch):
    data, elsewhere = tmp_path / "data", tmp_path / "elsewhere"
    data.mkdir()
    elsewhere.mkdir()
    (data / "rel.parquet").write_bytes(b"PAR1" + b"\0" * 20)
    monkeypatch.chdir(data)
    book = g.selections(str(tmp_path / "s.jsonl"))
    sel = g.select_values("rel.parquet", "t", 0, 1, "w", book=book)
    full = str(data / "rel.parquet")
    assert sel.dataset["path"] == full
    assert g.dataset_key("rel.parquet") == g.dataset_key(full)
    later = g.dataset_key("not_written_yet.parquet")                  # absolute even before it exists
    assert later["path"] == str(data / "not_written_yet.parquet") and later["fingerprint"] == {"size": None}
    url = "https://example.org/data/x.parquet"
    assert g.dataset_key(url)["path"] == url                          # URLs are left alone
    monkeypatch.chdir(elsewhere)
    assert len(book.find(dataset=full)) == 1                          # was 0: keys came from "rel.parquet"
    assert len(book.find(dataset=g.dataset_key(full))) == 1
    lib = g.library(str(tmp_path / "lib.jsonl"))
    f = g.flag_genes(["EGFR"], source=full, lib=lib)
    assert [s.id for s in lib.query(dataset=full)] == [f.id]
    assert os.path.isabs(f.dataset["path"])


def test_a_relative_selection_can_be_described_from_another_directory(tmp_path, monkeypatch):
    pl = pytest.importorskip("polars")
    data, elsewhere = tmp_path / "data", tmp_path / "elsewhere"
    data.mkdir()
    elsewhere.mkdir()
    pl.DataFrame({"t": list(range(50)), "v": [1.0] * 50}).write_parquet(str(data / "series.parquet"))
    monkeypatch.chdir(data)
    sel = g.select_values("series.parquet", "t", 10, 20, "w")
    monkeypatch.chdir(elsewhere)
    assert g.describe(sel)["rows"] == 10


@pytest.mark.parametrize("opener", [g.selections, g.library, g.catalog])
def test_relative_log_paths_do_not_fork_after_chdir(tmp_path, monkeypatch, opener):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()

    def write(store, i):
        if isinstance(store, g.Catalog):
            store.record(f"k{i}", "/x")
        else:
            store.add(_sel(f"id{i}"))

    monkeypatch.chdir(a)
    store = opener("log.jsonl")
    write(store, 1)
    monkeypatch.chdir(b)
    write(store, 2)
    assert os.path.isabs(store.path)
    assert len(open(a / "log.jsonl").read().splitlines()) == 2
    assert not (b / "log.jsonl").exists()


# ============================================================================
# 5. acquire
# ============================================================================

def test_offline_sra_run_plan_is_a_real_route_not_unrecognised():
    p = g.acquisition_plan("SRR1448774", offline=True)
    kinds = [r.kind for r in p.routes]
    assert "manual" not in kinds and kinds[0] == "ena-http" and "sra-prefetch" in kinds
    ena = p.routes[0]
    assert ena.url == "https://www.ebi.ac.uk/ena/browser/view/SRR1448774"
    assert ena.bytes == 0 and ena.streamable and ena.cost_usd == 0
    assert ena.note == "size known once online"
    pre = next(r for r in p.routes if r.kind == "sra-prefetch")
    assert pre.tool == "prefetch" and "fasterq-dump" in pre.note and pre.accession == "SRR1448774"
    assert not any("unrecognised" in r.note for r in p.routes)
    assert p.best.kind == "ena-http" and p.explain()


@pytest.mark.parametrize("acc", ["ERR164407", "DRR000001", "SRX123", "SRP043510",
                                 "PRJNA229998", "SAMN12345678"])
def test_offline_sra_family_never_says_unrecognised(acc):
    p = g.acquisition_plan(acc, offline=True)
    assert [r.kind for r in p.routes] == ["ena-http", "sra-prefetch"]
    assert acc in p.routes[0].url and p.routes[1].accession == acc
    assert not any("unrecognised" in r.note for r in p.routes)
    if p.kind != "sra_run":                                           # prefetch needs runs: say so
        assert "run accessions" in p.routes[1].note


def test_unknown_shapes_are_still_reported_as_unrecognised():
    r = g.acquisition_plan("nonsense", offline=True).routes[0]
    assert r.kind == "manual" and "unrecognised accession shape" in r.note


def test_an_sra_accession_ena_does_not_know_is_not_called_unrecognised(monkeypatch):
    monkeypatch.setattr(acq, "_ena_filereport", lambda accession, timeout=20.0: [])
    r = g.acquisition_plan("SRR1448774").routes[0]
    assert r.kind == "manual" and "unrecognised accession shape" not in r.note and "ENA" in r.note


@pytest.mark.parametrize("acc,tail", [
    ("GSE123", "series/GSEnnn/GSE123"),
    ("GSE99", "series/GSEnnn/GSE99"),
    ("GSE1234", "series/GSE1nnn/GSE1234"),
    ("GSE178341", "series/GSE178nnn/GSE178341"),
    ("GSM123", "samples/GSMnnn/GSM123"),
    ("GSM1234567", "samples/GSM1234nnn/GSM1234567"),
])
def test_geo_directory_stub_replaces_the_last_three_digits(acc, tail):
    assert acq.resolve(acc, offline=True)[0].url == f"https://ftp.ncbi.nlm.nih.gov/geo/{tail}/suppl/"


@pytest.mark.parametrize("acc", ["s3://bucket/key", "gs://bucket/key"])
def test_a_metered_route_of_unknown_size_is_not_free(acc):
    p = g.acquisition_plan(acc, offline=True)
    r = p.routes[0]
    assert p.cheapest_cost is None
    assert r.cost_known is False and r.cost_estimate is None and "free" not in repr(r)
    text = p.explain()
    assert "unknown size (metered)" in text and "free" not in text


def test_known_sizes_are_still_priced_and_a_free_route_still_settles_the_cost():
    metered = g.Route("aws-requester", "s3://x", 10 * 1024 ** 3, "aws", "aws-requester")
    p = g.AcquisitionPlan("x", "s3", [metered])
    assert p.cheapest_cost == pytest.approx(0.9) and metered.cost_known
    assert "$0.90" in p.explain() and "free" not in p.explain()
    unknown = g.Route("aws-requester", "s3://x", 0, "aws", "aws-requester")
    mixed = g.AcquisitionPlan("x", "sra_run", [g.Route("ena-http", "https://x", 0, "", "free"), unknown])
    assert mixed.cheapest_cost == 0.0
    assert "size unknown" in mixed.explain()
    assert g.Route("ena-http", "https://x", 0, "", "free").cost_known                # free is known
    assert g.Route("aws-odp", "s3://x", 0, "aws", "aws-odp").cost_known             # sponsored is known


def test_shortlist_copes_with_unknown_costs(capsys):
    plans = g.plan_many(["s3://b/k", "GSE178341", "SRR1448774"], offline=True)
    g.print_shortlist(plans)
    out = capsys.readouterr().out
    assert "unknown" in out and "available free" in out


ROW = {"run_accession": "SRR1",
       "fastq_ftp": "ftp.sra.ebi.ac.uk/vol1/fastq/SRR1/SRR1_1.fastq.gz;"
                    "ftp.sra.ebi.ac.uk/vol1/fastq/SRR1/SRR1_2.fastq.gz",
       "fastq_bytes": "1000000000;1000000000", "fastq_md5": "aa;bb",
       "sra_ftp": "ftp.sra.ebi.ac.uk/vol1/srr/SRR1/SRR1", "sra_bytes": "1600000000"}


@pytest.fixture
def ena_row(monkeypatch):
    monkeypatch.setattr(acq, "_ena_filereport", lambda accession, timeout=20.0: [dict(ROW)])


def test_plan_counts_one_representation_per_run(ena_row):
    p = g.acquisition_plan("SRR1")
    assert p.files == 2 and p.total_bytes == 2_000_000_000            # was 3 files, 3.6e9
    line = next(x for x in p.explain().splitlines() if x.strip().startswith("ena-http"))
    assert " 2 file(s)" in line and "1.9 GB" in line
    assert len(p.routes) > 3                                          # the .sra route is still offered
    assert any(r.filename == "SRR1" for r in p.routes)


def test_a_row_without_a_run_accession_is_still_one_run(monkeypatch):
    row = {k: v for k, v in ROW.items() if k != "run_accession"}
    monkeypatch.setattr(acq, "_ena_filereport", lambda accession, timeout=20.0: [row])
    p = g.acquisition_plan("SRR1")
    assert (p.files, p.total_bytes) == (2, 2_000_000_000)


def test_a_run_with_only_an_sra_container_counts_the_container(monkeypatch):
    row = {"run_accession": "SRR2", "sra_ftp": "ftp.x/vol1/srr/SRR2", "sra_bytes": "500"}
    monkeypatch.setattr(acq, "_ena_filereport", lambda accession, timeout=20.0: [row])
    p = g.acquisition_plan("SRR2")
    assert (p.files, p.total_bytes) == (1, 500)


def test_each_run_of_a_study_contributes_its_own_representation(monkeypatch):
    rows = [dict(ROW), dict(ROW, run_accession="SRR3",
                            fastq_ftp="ftp.x/SRR3.fastq.gz", fastq_bytes="7", sra_ftp="ftp.x/SRR3",
                            sra_bytes="9"),
            {"run_accession": "SRR4", "sra_ftp": "ftp.x/SRR4", "sra_bytes": "11"}]
    monkeypatch.setattr(acq, "_ena_filereport", lambda accession, timeout=20.0: rows)
    p = g.acquisition_plan("PRJNA1")
    assert p.files == 4 and p.total_bytes == 2_000_000_000 + 7 + 11


def test_fetch_dry_run_builds_a_command_for_every_mate(ena_row):
    out = g.fetch("SRR1", "/tmp/d", dry_run=True)
    assert out["ok"] and out["dry_run"] and out["route"] == "ena-http"
    assert len(out["commands"]) == 2
    assert "SRR1_1.fastq.gz" in out["commands"][0] and "SRR1_2.fastq.gz" in out["commands"][1]
    assert "SRR1_1.fastq.gz" in out["command"] and "SRR1_2.fastq.gz" in out["command"]
    assert "vol1/srr" not in out["command"]                           # not the .sra alternative
    assert len(out["urls"]) == 2 and out["url"] == out["urls"][0]
    assert out["bytes"] == 2_000_000_000
    p = g.acquisition_plan("SRR1")
    one = g.fetch("SRR1", "/tmp/d", route=p.routes[0], dry_run=True)  # an explicit route is just that one
    assert len(one["commands"]) == 1 and one["command"] == one["commands"][0]


def _gz_fastq(n, seed=7):
    rng = random.Random(seed)
    qual = "#$%&()*+,-./0123456789:;<=>?@ABCDEFGHIJ"
    recs = "".join(f"@read{i}/1\n{''.join(rng.choices('ACGT', k=100))}\n+\n"
                   f"{''.join(rng.choices(qual, k=100))}\n" for i in range(n))
    return gzip.compress(recs.encode(), mtime=0)


def test_sniff_decodes_a_truncated_gzip_slice():
    blob = _gz_fastq(6000)
    assert len(blob) > 3 * 65536
    s = acq._sniff(blob[:65536], "reads.fastq.gz")
    assert s["gzipped"] and s["format"] == "FASTQ"                    # was 'unknown'
    assert s["read_length_mean"] == 100.0 and s["reads_sampled"] > 100 and s["first_id"].startswith("@read0")
    assert 0.4 < s["gc_fraction"] < 0.6
    assert "cut" in s["note"]
    junk = acq._sniff(b"\x1f\x8b\x08\x00garbage-not-a-stream", "x.gz")   # nothing decodes: still no raise
    assert junk["format"] == "unknown" and "note" in junk


def test_sniff_reads_across_gzip_members():
    one = gzip.compress(b"@r1\nACGTACGTAC\n+\nIIIIIIIIII\n" * 3, mtime=0)
    two = gzip.compress(b"@r2\nACGTACGTAC\n+\nIIIIIIIIII\n" * 3, mtime=0)
    s = acq._sniff(one + two, "bgzf.fastq.gz")
    assert s["format"] == "FASTQ" and s["reads_sampled"] == 6 and "note" not in s


# ============================================================================
# 6. profile
# ============================================================================

class _Page:
    """What urllib.request.urlopen returns, minus the network."""

    def __init__(self, html):
        self._b = html.encode()

    def read(self, n=-1):
        return self._b if n is None or n < 0 else self._b[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def serve(monkeypatch):
    def _serve(html):
        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Page(html))
    return _serve


def _page_labels(p):
    return {lab for e in p.evidence if e.source == "page" for lab in e.labels}


def test_page_keywords_match_whole_tokens_not_substrings(serve):
    serve("<html><body><p>We ran a western blot on the 13th of March, and a second western "
          "blot, and a third western blot.</p><p>NIH-3T3 cells at 3T3 density.</p></body></html>")
    p = g.profile("/nowhere/notes.dat", url="http://example.invalid/p")
    labels = _page_labels(p)
    assert "WGS / WES" not in labels and "MRI (structural neuroimaging)" not in labels
    assert not p.confident


def test_page_keywords_still_fire_on_real_mentions(serve):
    serve("<p>Whole-exome sequencing (WES) of tumours; T1-weighted MRI on a 3T scanner (Siemens). "
          "ATAC-seq and snATAC-seq and scATACseq peaks.</p>")
    p = g.profile("/nowhere/notes.dat", url="http://example.invalid/p")
    detail = " ".join(e.detail for e in p.evidence if e.source == "page")
    assert "wesx1" in detail and "3tx1" in detail
    labels = _page_labels(p)
    assert {"WGS / WES", "MRI (structural neuroimaging)", "scATAC-seq"} <= labels


def test_keywords_that_open_unrelated_words_end_at_a_boundary_too(serve):
    serve("<p>Total nucleic acid was extracted from Salmonella cultures.</p>")
    p = g.profile("/nowhere/notes.dat", url="http://example.invalid/p")
    assert "snRNA-seq" not in _page_labels(p) and "bulk RNA-seq" not in _page_labels(p)
    serve("<p>Nuclei were isolated and reads quantified with Salmon.</p>")
    q = g.profile("/nowhere/notes.dat", url="http://example.invalid/p")
    assert {"snRNA-seq", "bulk RNA-seq"} <= _page_labels(q)


def test_filename_tokens_are_tokens_too():
    def fn(name):
        return {e.detail for e in g.profile(name, fetch_page=False).evidence if e.source == "filename"}
    assert any("'atac'" in d for d in fn("/nowhere/sample_atac.bed"))
    assert any("'atac'" in d for d in fn("/nowhere/pbmc_scatac.h5ad")) or \
        any("'scatac'" in d for d in fn("/nowhere/pbmc_scatac.h5ad"))
    assert any("wgs" in d for d in fn("/nowhere/NA12878.wgs.cram"))
    assert not any("wgs" in d for d in fn("/nowhere/showgsample.txt"))
    assert not any("atac" in d for d in fn("/nowhere/datatact.txt"))


def test_read_identifiers_are_tokenised_for_content_evidence(tmp_path):
    """An index sequence in an Illumina read id is not the word 'atac'."""
    fq = tmp_path / "idx.fastq"
    fq.write_text("".join(f"@A00123:45:HXXXX:1:1101:{i}:1 1:N:0:ATACGGTA\nACGTACGTACGT\n+\nIIIIIIIIIIII\n"
                          for i in range(20)))
    p = g.profile(str(fq), fetch_page=False)
    assert not any("atac" in e.detail for e in p.evidence if e.source == "content")


def test_a_csv_starting_with_mm_is_not_a_tiff(tmp_path):
    f = tmp_path / "markers.dat"
    f.write_text("MMP1,MMP2,MMP9\n1,2,3\n")
    p = g.profile(str(f), fetch_page=False)
    assert p.format != "TIFF" and not any("TIFF" in e.detail for e in p.evidence)
    g_ = tmp_path / "iis.dat"
    g_.write_text("II,JJ\n1,2\n")
    assert g.profile(str(g_), fetch_page=False).format != "TIFF"


@pytest.mark.parametrize("magic", [b"II*\x00", b"MM\x00*", b"II+\x00\x08\x00\x08\x00", b"MM\x00+\x00\x08\x00\x00"])
def test_real_tiff_and_bigtiff_magics_are_still_recognised(tmp_path, magic):
    f = tmp_path / "scan.dat"
    f.write_bytes(magic + b"\x08\x00\x00\x00" + b"\0" * 64)
    p = g.profile(str(f), fetch_page=False)
    assert p.format == "TIFF" and p.best.label == "microscopy / histology"


def _dicom_bytes(modality=b"MR"):
    """A minimal Part-10 file: preamble, 'DICM', a File Meta group naming Explicit VR Little
    Endian, and one data element, (0008,0060) Modality."""
    ts = b"1.2.840.10008.1.2.1\x00"
    meta = b"\x02\x00\x10\x00" + b"UI" + struct.pack("<H", len(ts)) + ts
    group = b"\x02\x00\x00\x00" + b"UL" + struct.pack("<H", 4) + struct.pack("<I", len(meta))
    mod = modality + (b" " if len(modality) % 2 else b"")
    return (b"\0" * 128 + b"DICM" + group + meta
            + b"\x08\x00\x60\x00" + b"CS" + struct.pack("<H", len(mod)) + mod)


def _nifti_bytes(shape=(4, 4, 4)):
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<8h", hdr, 40, 3, *shape, 1, 1, 1, 1)
    struct.pack_into("<h", hdr, 70, 4)                                # int16
    struct.pack_into("<h", hdr, 72, 16)
    struct.pack_into("<8f", hdr, 76, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0)
    struct.pack_into("<f", hdr, 108, 352.0)
    hdr[344:348] = b"n+1\x00"
    return bytes(hdr) + b"\0\0\0\0" + b"\0" * (2 * shape[0] * shape[1] * shape[2])


def test_a_dicom_series_one_level_down_is_recognised(tmp_path):
    series = tmp_path / "study" / "series_1"
    series.mkdir(parents=True)
    for i in range(3):
        (series / f"IM{i:04d}.dcm").write_bytes(_dicom_bytes())
    p = g.profile(str(tmp_path / "study"), fetch_page=False)
    assert p.format == "DICOM series" and p.metadata["n_dicom_files_seen"] == 3
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "a.IMA").write_bytes(_dicom_bytes())
    assert g.profile(str(flat), fetch_page=False).format == "DICOM series"


def test_dicom_header_metadata_survives_the_recursion(tmp_path):
    pytest.importorskip("pydicom")
    series = tmp_path / "a" / "b"
    series.mkdir(parents=True)
    (series / "x.dcm").write_bytes(_dicom_bytes(b"MR"))
    p = g.profile(str(tmp_path), fetch_page=False)
    assert p.format == "DICOM series" and p.metadata.get("modality") == "MR"
    assert p.best.label == "MRI (structural neuroimaging)"


def test_directory_recursion_stops_at_depth_three(tmp_path):
    deep_ok = tmp_path / "ok" / "a" / "b" / "c"
    deep_ok.mkdir(parents=True)
    (deep_ok / "x.dcm").write_bytes(_dicom_bytes())
    assert g.profile(str(tmp_path / "ok"), fetch_page=False).format == "DICOM series"
    too_deep = tmp_path / "no" / "a" / "b" / "c" / "d"
    too_deep.mkdir(parents=True)
    (too_deep / "x.dcm").write_bytes(_dicom_bytes())
    assert g.profile(str(tmp_path / "no"), fetch_page=False).format != "DICOM series"


def test_a_wide_tree_is_searched_breadth_first_and_a_zarr_store_is_not_entered(tmp_path):
    for i in range(1, 31):
        (tmp_path / f"sub-{i:02d}" / "anat").mkdir(parents=True)
    (tmp_path / "sub-01" / "anat" / "sub-01_T1w.nii").write_bytes(_nifti_bytes())
    assert g.profile(str(tmp_path), fetch_page=False).format == "NIfTI"
    store = tmp_path / "volumes.zarr"
    (store / "a").mkdir(parents=True)
    (store / "a" / "x.dcm").write_bytes(_dicom_bytes())
    assert g.profile(str(store), fetch_page=False).format != "DICOM series"
    assert g.profile(str(store) + os.sep, fetch_page=False).format != "DICOM series"


def test_nifti_volumes_in_nested_directories_are_recognised(tmp_path):
    d = tmp_path / "sub-01" / "anat"
    d.mkdir(parents=True)
    (d / "sub-01_T1w.nii").write_bytes(_nifti_bytes())
    p = g.profile(str(tmp_path), fetch_page=False)
    assert p.format == "NIfTI" and p.best.label == "MRI (structural neuroimaging)"
    assert p.metadata["n_nifti_files_seen"] == 1
    assert p.discriminating_evidence or p.evidence                    # the header was actually read


def test_an_existing_directory_is_not_reported_missing(tmp_path):
    p = g.profile(str(tmp_path), fetch_page=False)
    assert not any("does not exist" in n for n in p.notes)
    q = g.profile(str(tmp_path / "nope.h5ad"), fetch_page=False)
    assert any("does not exist" in n for n in q.notes)


def test_sam_is_not_reported_as_fastq(tmp_path):
    text = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:1000\nr1\t0\tchr1\t10\t60\t4M\t*\t0\t0\tACGT\tIIII\n"
    sam = tmp_path / "aln.sam"
    sam.write_text(text)
    assert g.profile(str(sam), fetch_page=False).format == "SAM"
    anon = tmp_path / "aln.dat"
    anon.write_text(text)
    assert g.profile(str(anon), fetch_page=False).format == "SAM"
    fq = tmp_path / "reads.dat"
    fq.write_text("@r1\nACGTACGTAC\n+\nIIIIIIIIII\n@r2\nACGTACGTAA\n+\nIIIIIIIIII\n")
    assert g.profile(str(fq), fetch_page=False).format == "FASTQ"      # real FASTQ is untouched
    long_read = tmp_path / "long.dat"
    long_read.write_text("@nano1 runid=x\n" + "ACGT" * 400 + "\n+\n" + "I" * 1600 + "\n")
    assert g.profile(str(long_read), fetch_page=False).format == "FASTQ"   # header + one long line visible


@pytest.mark.parametrize("name", ["/nowhere/thing.xyz", "/nowhere/thing.h5ad", "/nowhere/thing"])
def test_no_molecule_is_claimed_from_the_prior_alone(name):
    p = g.profile(name, fetch_page=False)
    assert p.molecule is None and not p.confident


def test_a_molecule_follows_real_evidence():
    p = g.profile("/nowhere/NA12878.wgs.cram", fetch_page=False)
    assert p.best.label == "WGS / WES" and p.molecule == "DNA"
    q = g.profile("/nowhere/chip.gef", fetch_page=False)              # the container names it
    assert q.molecule == "RNA"


# ============================================================================
# 7. catalog
# ============================================================================

def test_annotate_rejects_unknown_fields_instead_of_dropping_them(tmp_path):
    cat = g.catalog(str(tmp_path / "c.jsonl"))
    cat.record("a", "/a")
    n = len(open(cat.path).read().splitlines())
    with pytest.raises(TypeError, match=r"facets="):
        cat.annotate("a", tissue="cortex", disease="GBM")
    with pytest.raises(TypeError, match="colour"):
        cat.annotate("a", colour="red")
    with pytest.raises(TypeError, match="colour"):
        cat.annotate("a", note="fine", colour="red")                  # one bad field spoils the call
    assert len(open(cat.path).read().splitlines()) == n               # and nothing was logged
    cat.annotate("a", facets={"tissue": "cortex", "disease": "GBM"}, note="ok", tags=["t"])
    e = cat.get("a")
    assert e.facets == {"tissue": "cortex", "disease": "GBM"} and e.note == "ok" and e.tags == ["t"]
    with pytest.raises(KeyError, match="unknown facet"):
        cat.annotate("a", facets={"tisue": "x"})                      # facet typos still say so


def _profile(label, p=0.6):
    return g.Profile(target="/x.h5ad", format="AnnData h5ad", molecule="RNA",
                     assays=[g.Hypothesis(label, "assay", probability=p)])


def test_a_new_profile_updates_the_profilers_assay_facet(tmp_path):
    cat = g.catalog(str(tmp_path / "c.jsonl"))
    cat.record("k", "/k", profile=_profile("scRNA-seq"))
    assert cat.facets_for("k")["assay"] == "scRNA-seq"
    assert cat.get("k").metadata["assay_from_profiler"] is True
    cat.record("k", "/k", profile=_profile("snRNA-seq"))
    assert cat.facets_for("k")["assay"] == "snRNA-seq" and cat.get("k").assay == "snRNA-seq"
    assert cat.get("k").metadata["assay_from_profiler"] is True
    assert g.catalog(cat.path).facets_for("k")["assay"] == "snRNA-seq"   # a fresh reader folds the same
    cat.record("k", "/k", note="no profile this time")                   # nothing to refresh from
    assert cat.facets_for("k")["assay"] == "snRNA-seq"


def test_an_explicit_assay_facet_is_never_overwritten_by_the_profiler(tmp_path):
    cat = g.catalog(str(tmp_path / "c.jsonl"))
    cat.record("k", "/k", profile=_profile("scRNA-seq"), facets={"assay": "10x v3 scRNA-seq"})
    cat.record("k", "/k", profile=_profile("snRNA-seq"))
    assert cat.facets_for("k")["assay"] == "10x v3 scRNA-seq"
    assert cat.get("k").assay == "snRNA-seq"                          # the profiler's own field still moves
    assert cat.get("k").metadata["assay_from_profiler"] is False
    cat.record("j", "/j", profile=_profile("scRNA-seq"))              # profiler default first ...
    cat.annotate("j", facets={"assay": "Smart-seq2"})                 # ... then the analyst decides
    cat.record("j", "/j", profile=_profile("snRNA-seq"))
    assert cat.facets_for("j")["assay"] == "Smart-seq2"
    assert g.catalog(cat.path).facets_for("j")["assay"] == "Smart-seq2"
    cat.record("j", "/j", facets={"assay": "Smart-seq3"})             # explicit again, via record
    cat.record("j", "/j", profile=_profile("scRNA-seq"))
    assert cat.facets_for("j")["assay"] == "Smart-seq3"


def test_logs_from_before_the_flag_are_treated_as_analyst_set(tmp_path):
    """An older record with an assay facet and no flag may have been typed by a person; keep it."""
    p = tmp_path / "old.jsonl"
    p.write_text(json.dumps({"event": "record", "key": "old", "target": "/o", "assay": "Stereo-seq",
                             "facets": {"assay": "Stereo-seq"}, "ts": 1.0}) + "\n")
    cat = g.catalog(str(p))
    cat.record("old", "/o", profile=_profile("scRNA-seq"))
    assert cat.facets_for("old")["assay"] == "Stereo-seq"
    assert cat.get("old").assay == "scRNA-seq"


# ============================================================================
# Guards: behaviour that must not change while the above is fixed
# ============================================================================

need = pytest.mark.skipif(not os.path.exists(os.path.join(FIX, "chip.gef")), reason="fixtures not built")


@need
def test_guard_gef_and_h5ad_profiles_are_unchanged():
    p = g.profile(os.path.join(FIX, "chip.gef"), fetch_page=False)
    assert p.format == "Stereo-seq GEF" and p.best.label == "Stereo-seq" and p.best.probability > 0.8
    assert p.molecule == "RNA"
    h5 = os.path.join(FIX, "sc.h5ad")
    if os.path.exists(h5):
        q = g.profile(h5, fetch_page=False)
        assert {"scRNA-seq", "snRNA-seq"} <= {h.label for h in q.assays[:3]}
        assert not q.confident and "intronic" in q.best.distinguish.lower()
        assert q.molecule == "RNA"


@pytest.mark.skipif(not os.path.exists(os.path.join(FIX, "ms", "dicom", "P04_FLAIR")), reason="no MS fixtures")
def test_guard_top_level_dicom_series_and_nifti_files_are_unchanged():
    d = g.profile(os.path.join(FIX, "ms", "dicom", "P04_FLAIR"), fetch_page=False)
    assert d.format == "DICOM series" and d.best.label.startswith("MRI")
    nii = os.path.join(FIX, "ms", "train", "P01", "T1", "P01_T1_FLAIR.nii")
    if os.path.exists(nii):
        p = g.profile(nii, fetch_page=False)
        assert p.format == "NIfTI" and p.best.label.startswith("MRI")


@pytest.mark.skipif(not os.path.exists(os.path.join(FIX, "tissue.ome.tif")), reason="no tiff fixture")
def test_guard_a_real_tiff_is_still_a_tiff():
    p = g.profile(os.path.join(FIX, "tissue.ome.tif"), fetch_page=False)
    assert p.format == "TIFF" and p.best.label == "microscopy / histology"
    assert any(e.detail == "TIFF magic" for e in p.evidence)


def test_guard_confidence_semantics():
    assert not g.profile("/nowhere/thing.h5ad", fetch_page=False).confident
    ev = [g.Evidence("page", "source page mentions snrnax1", 3.5, "snRNA-seq")]
    p = g.Profile(target="x", assays=[g.Hypothesis("snRNA-seq", "assay", probability=0.8),
                                      g.Hypothesis("scRNA-seq", "assay", probability=0.1)], evidence=ev)
    assert p.confident
