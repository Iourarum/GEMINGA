"""Route planning: classify, price, build commands — all without touching the network."""
import pytest
import geminga as g


@pytest.mark.parametrize("acc,kind", [
    ("GSE178341", "geo_series"), ("GSM1234", "geo_sample"),
    ("SRR1448774", "sra_run"), ("ERR164407", "sra_run"), ("SRX123", "sra_experiment"),
    ("SRP043510", "sra_study"), ("PRJNA229998", "bioproject"),
    ("EGAD00001005054", "ega_dataset"), ("EGAS00001004030", "ega_study"),
    ("E-MTAB-8410", "arrayexpress"), ("CRA001963", "gsa"), ("STDS0000002", "stomics"),
    ("phs001835.v1.p1", "dbgap"), ("s3://bucket/key", "s3"), ("gs://bucket/key", "gs"),
    ("https://zenodo.org/x", "url"), ("not-an-accession", "unknown"),
])
def test_classify(acc, kind):
    assert g.classify(acc) == kind


def test_offline_plans_never_raise_and_always_advise():
    for acc in ("GSE178341", "EGAD00001005054", "STDS0000002", "CRA001963",
                "phs001835", "E-MTAB-8410", "nonsense", "SRR1448774", "SRP043510",
                "PRJNA229998", "s3://bucket/key", "gs://bucket/key"):
        p = g.acquisition_plan(acc, offline=True)
        assert p.routes, acc
        assert p.explain()
        assert all(r.note or r.kind != "manual" for r in p.routes)
        # "unrecognised" is for shapes classify() does not know, not for SRA accessions offline
        assert (g.classify(acc) == "unknown") == any("unrecognised" in r.note for r in p.routes), acc


def test_controlled_access_is_flagged_not_attempted():
    for acc in ("EGAD00001005054", "phs001835"):
        p = g.acquisition_plan(acc, offline=True)
        assert p.routes[0].needs_auth
        out = g.fetch(acc, "/tmp", route=p.routes[0], dry_run=True)
        assert out["ok"] is False and "credential" in out["reason"].lower()


def test_dropbox_share_link_becomes_a_download():
    url = "https://www.dropbox.com/scl/fi/xyz/atlas.h5ad?rlkey=k&dl=0"
    r = g.acquisition_plan(url, offline=True).routes[0]
    assert r.kind == "dropbox"
    assert "dl=1" in r.url and "dl=0" not in r.url
    assert r.cost_usd == 0


def test_routes_rank_free_and_streamable_first():
    rows = [
        g.Route("aws-requester", "s3://x", 10 * 1024 ** 3, "aws", "aws-requester"),
        g.Route("ena-http", "https://x", 10 * 1024 ** 3, "", "free", streamable=True),
        g.Route("sra-prefetch", "SRR1", 0, "prefetch", "free"),
    ]
    from geminga.acquire import _rank
    best = sorted(rows, key=_rank)[0]
    assert best.kind == "ena-http" and best.streamable


def test_cost_arithmetic():
    r = g.Route("aws-requester", "s3://x", 100 * 1024 ** 3, "aws", "aws-requester")
    assert r.cost_usd == pytest.approx(9.0, rel=1e-6)
    free = g.Route("ena-http", "https://x", 100 * 1024 ** 3, "", "free")
    assert free.cost_usd == 0.0
    assert g.EGRESS_USD_PER_GB["aws-odp"] == 0.0


def test_commands_are_built_for_every_route_kind():
    cases = [
        ("ena-aspera", "era-fasp@fasp.sra.ebi.ac.uk:vol1/fastq/SRR1/SRR1.fastq.gz", "ascp"),
        ("aws-odp", "s3://sra-pub-run-odp/sra/SRR1/SRR1", "aws"),
        ("aws-requester", "s3://private/x", "aws"),
        ("gcp-requester", "gs://private/x", "gcloud"),
        ("sra-prefetch", "SRR1", "prefetch"),
        ("ega", "EGAD1", "pyega3"),
    ]
    for kind, url, tool in cases:
        cmd = g.command_for(g.Route(kind, url, 0, tool, "free", accession="SRR1"), "/data")
        assert cmd[0] == tool
        assert "/data" in cmd
    http = g.command_for(g.Route("ena-http", "https://x/y.fastq.gz", 0, "", "free"), "/data")
    assert http[0] in ("aria2c", "curl", "wget")
    assert "--no-sign-request" in g.command_for(
        g.Route("aws-odp", "s3://b/k", 0, "aws", "aws-odp"), "/d")
    assert "--request-payer" in g.command_for(
        g.Route("aws-requester", "s3://b/k", 0, "aws", "aws-requester"), "/d")


def test_fetch_is_dry_run_by_default():
    r = g.Route("generic-http", "https://example.invalid/x.gz", 0, "", "free")
    out = g.fetch("https://example.invalid/x.gz", "/tmp", route=r)
    assert out["dry_run"] is True and out["ok"] is True
    assert "nothing transferred" in out["note"]


def test_missing_tool_is_reported_not_raised():
    r = g.Route("ena-aspera", "era-fasp@h:/p", 0, "definitely-not-a-real-binary", "free")
    out = g.fetch("SRR1", "/tmp", route=r)
    assert out["ok"] is False and "not installed" in out["reason"]


def test_sniff_recognises_the_formats_in_the_inventory():
    from geminga.acquire import _sniff
    fq = b"@r1 1:N:0\nACGTACGTAC\n+\nIIIIIIIIII\n" * 20
    s = _sniff(fq)
    assert s["format"] == "FASTQ" and s["read_length_mean"] == 10.0
    assert 0.4 < s["gc_fraction"] < 0.6
    assert _sniff(b">seq1 desc\nACGT\n")["format"] == "FASTA"
    assert _sniff(b"PAR1....")["format"] == "Parquet"
    assert _sniff(b"\x89HDF\r\n\x1a\n....")["format"].startswith("HDF5")
    assert _sniff(b"%%MatrixMarket matrix coordinate integer general\n3 3 3\n")["format"] \
        == "MatrixMarket"
    assert _sniff(b"#FileFormat=GEMv0.1\n#BinSize=1\ngeneID\tx\ty\tMIDCount\n")["format"] \
        == "Stereo-seq GEM"


def test_peek_reports_failure_without_raising():
    out = g.peek("https://definitely-not-a-host.invalid/x.fastq.gz",
                 route=g.Route("generic-http", "https://definitely-not-a-host.invalid/x.gz",
                               0, "", "free"))
    assert out["ok"] is False and "reason" in out


def test_shortlist_totals(capsys):
    plans = g.plan_many(["GSE178341", "EGAD00001005054", "STDS0000002"], offline=True)
    assert len(plans) == 3
    g.print_shortlist(plans)
    assert "available free" in capsys.readouterr().out


def test_available_tools_shape():
    t = g.available_tools()
    assert "ascp" in t and "fasterq-dump" in t and "pyega3" in t
    for _name, d in t.items():
        assert set(d) == {"path", "present", "what"} and d["what"]
