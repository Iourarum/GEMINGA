"""Getting hold of a dataset without paying for it twice.

GEMINGA does not download. There are good tools for that already — `fasterq-dump`, `ascp`,
`fastq-dl`, `kingfisher`, `pyega3`, `aws s3` — and writing a worse one would help nobody. What
is missing, and what costs graduate students real money, is the step before: working out
*which route to use*, what it will cost, and whether the download is needed at all.

So this module does three things and delegates the fourth.

**Resolve.** An accession becomes a concrete list of routes. `SRR1448774` is not one file at
one place; it is an SRA object at NCBI, a FASTQ at EBI, an `.sra` in the AWS Open Data
Program, and sometimes a GCP object. These differ enormously in price.

**Price.** Each route carries its byte count and an egress estimate. The single most useful
fact in this module is that **ENA mirrors SRA and serves FASTQ free** over FTP, HTTPS and
Aspera, already converted, with no `fasterq-dump` step. A cloud route that charges you for the
same bytes is almost always avoidable, and the planner says so out loud.

**Peek without downloading.** ENA's HTTPS endpoints honour Range requests. That means the
first few thousand reads of a 40 GB FASTQ can be inspected — read length, GC, barcode shape,
adapter content — for a few hundred kilobytes. For a shortlist of two hundred candidate
datasets, that is the difference between a week of transfers and an afternoon.

**Delegate.** The actual bytes are moved by whichever tool is installed. This module builds
the command, checks the tool is there, and gets out of the way. FASP in particular is
proprietary: `ascp` is the only thing that speaks it, and pretending otherwise would be a lie.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

from ._core import human

__all__ = ["Route", "AcquisitionPlan", "resolve", "plan", "peek", "fetch",
           "available_tools", "EGRESS_USD_PER_GB", "ENA_FIELDS"]

#: Rough published egress prices, USD per GB, for the first tier. Estimates only — providers
#: change these, and your institution may have a waiver or a Data Egress Waiver programme.
#: The point of the number is the order of magnitude, not the third decimal.
EGRESS_USD_PER_GB = {
    "free": 0.0,
    "aws-odp": 0.0,        # Open Data Sponsorship Program: AWS pays
    "aws-requester": 0.09,
    "gcp-requester": 0.12,
    "azure": 0.087,
}

ENA_PORTAL = "https://www.ebi.ac.uk/ena/portal/api/filereport"
ENA_FIELDS = ("run_accession,experiment_accession,sample_accession,study_accession,"
              "fastq_ftp,fastq_aspera,fastq_bytes,fastq_md5,submitted_ftp,submitted_bytes,"
              "sra_ftp,sra_bytes,read_count,base_count,instrument_platform,library_strategy,"
              "library_source,scientific_name")

_ACC = {
    "sra_run": re.compile(r"^[SED]RR\d+$"),
    "sra_experiment": re.compile(r"^[SED]RX\d+$"),
    "sra_study": re.compile(r"^[SED]RP\d+$"),
    "bioproject": re.compile(r"^PRJ[END][AB]\d+$"),
    "biosample": re.compile(r"^SAM[END][AG]?\d+$"),
    "geo_series": re.compile(r"^GSE\d+$"),
    "geo_sample": re.compile(r"^GSM\d+$"),
    "ega_dataset": re.compile(r"^EGAD\d+$"),
    "ega_study": re.compile(r"^EGAS\d+$"),
    "arrayexpress": re.compile(r"^E-[A-Z]{4}-\d+$"),
    "gsa": re.compile(r"^CRA\d+$"),
    "stomics": re.compile(r"^STDS\d+$"),
    "cngb": re.compile(r"^CN[PSXRA]\d{7}$"),
    "dbgap": re.compile(r"^phs\d+", re.I),
    "figshare_doi": re.compile(r"^(doi:)?10\.6084/m9\.figshare\.\d+", re.I),
    "zenodo_doi": re.compile(r"^(doi:)?10\.5281/zenodo\.\d+", re.I),
    "openneuro": re.compile(r"^ds\d{6}$"),
    "mendeley_doi": re.compile(r"^(doi:)?10\.17632/[a-z0-9]+\.\d+", re.I),
}


def classify(accession: str) -> str:
    a = accession.strip()
    for kind, rx in _ACC.items():
        if rx.match(a):
            return kind
    if a.startswith(("http://", "https://")):
        low = a.lower()
        if "kaggle.com/datasets/" in low:
            return "kaggle"
        if "figshare.com" in low:
            return "figshare_url"
        if "nitrc.org" in low:
            return "nitrc"
        if "data.mendeley.com" in low:
            return "mendeley_url"
        return "url"
    if a.startswith("s3://"):
        return "s3"
    if a.startswith("gs://"):
        return "gs"
    return "unknown"


@dataclass
class Route:
    """One concrete way to get one file, with its price attached."""
    kind: str                  # "ena-http" | "ena-ftp" | "ena-aspera" | "aws-odp" | ...
    url: str
    bytes: int = 0
    tool: str = ""             # the binary that moves it, "" when plain HTTP will do
    cost_class: str = "free"
    streamable: bool = False   # HTTP Range works, so GEMINGA can read it in place
    needs_auth: bool = False
    md5: str = ""
    note: str = ""
    accession: str = ""
    filename: str = ""
    representation: str = ""   # "fastq" | "sra" | "": ENA offers one run's reads as either

    @property
    def metered(self) -> bool:
        """Does moving these bytes cost money (egress billed to you)?"""
        return EGRESS_USD_PER_GB.get(self.cost_class, 0.0) > 0

    @property
    def cost_known(self) -> bool:
        """False for a metered route whose size is not known yet. 0 bytes x $0.09/GB is 0 only
        arithmetically; it is not "free", it is "not priced yet"."""
        return not self.metered or self.bytes > 0

    @property
    def cost_usd(self) -> float:
        """USD, for the callers that add it up; 0.0 when `cost_known` is False — check that
        first, or use `cost_estimate`, which is None in that case."""
        return (self.bytes / 1024 ** 3) * EGRESS_USD_PER_GB.get(self.cost_class, 0.0)

    @property
    def cost_estimate(self) -> Optional[float]:
        return self.cost_usd if self.cost_known else None

    @property
    def available(self) -> bool:
        return not self.tool or shutil.which(self.tool) is not None

    def __repr__(self) -> str:
        bits = [self.kind, human(self.bytes)]
        if not self.cost_known:
            bits.append("metered, size unknown")
        elif self.cost_usd > 0:
            bits.append(f"~${self.cost_usd:.2f}")
        else:
            bits.append("free")
        if self.streamable:
            bits.append("streamable")
        if self.needs_auth:
            bits.append("auth")
        if self.tool and not self.available:
            bits.append(f"needs {self.tool}")
        return f"Route({', '.join(bits)})"


#: Preference order. Free and streamable first; anything metered last.
_RANK = {"cngb-ftp": 1, "kaggle": 6, "ena-http": 0, "ena-ftp": 1, "ena-aspera": 2, "aws-odp": 3, "generic-http": 4,
         "dropbox": 4, "geo-ftp": 4, "sra-prefetch": 5, "gs-public": 6,
         "aws-requester": 8, "gcp-requester": 8, "ega": 9, "dbgap": 9, "manual": 10}


def _rank(r: Route) -> tuple:
    # a metered route of unknown size is not "free": within a kind it sorts after priced ones
    return (_RANK.get(r.kind, 7), r.cost_usd if r.cost_known else float("inf"),
            0 if r.available else 1)


def _get_json(url: str, timeout: float = 20.0):
    req = urllib.request.Request(url, headers={"User-Agent": "geminga/0.6"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _ena_filereport(accession: str, timeout: float = 20.0) -> list:
    q = urllib.parse.urlencode({"accession": accession, "result": "read_run",
                                "fields": ENA_FIELDS, "format": "tsv", "limit": "0"})
    text = _get_json(f"{ENA_PORTAL}?{q}", timeout)
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return []
    head = lines[0].split("\t")
    return [dict(zip(head, l.split("\t"))) for l in lines[1:]]


def _routes_from_ena_row(row: dict) -> list:
    out: list[Route] = []
    acc = row.get("run_accession", "")
    sizes = [int(b) for b in (row.get("fastq_bytes") or "").split(";") if b.strip().isdigit()]
    md5s = [m for m in (row.get("fastq_md5") or "").split(";") if m]
    ftps = [p for p in (row.get("fastq_ftp") or "").split(";") if p]
    asps = [p for p in (row.get("fastq_aspera") or "").split(";") if p]

    for i, p in enumerate(ftps):
        n = sizes[i] if i < len(sizes) else 0
        md5 = md5s[i] if i < len(md5s) else ""
        fn = p.rsplit("/", 1)[-1]
        # ENA serves the same path over HTTPS, and that endpoint honours Range requests —
        # which is what lets GEMINGA inspect a file without transferring it.
        out.append(Route("ena-http", "https://" + p, n, "", "free", streamable=True,
                         md5=md5, accession=acc, filename=fn, representation="fastq",
                         note="EBI mirror of SRA, already converted to FASTQ"))
        out.append(Route("ena-ftp", "ftp://" + p, n, "", "free", md5=md5,
                         accession=acc, filename=fn, representation="fastq"))
    for i, p in enumerate(asps):
        n = sizes[i] if i < len(sizes) else 0
        out.append(Route("ena-aspera", f"era-fasp@{p}", n, "ascp", "free",
                         accession=acc, filename=p.rsplit("/", 1)[-1], representation="fastq",
                         note="FASP, port 33001. Much faster on a long fat pipe; "
                              "needs the proprietary ascp client"))
    if row.get("sra_ftp"):
        for p in row["sra_ftp"].split(";"):
            if p:
                out.append(Route("ena-http", "https://" + p,
                                 int((row.get("sra_bytes") or "0").split(";")[0] or 0),
                                 "", "free", streamable=True, accession=acc,
                                 filename=p.rsplit("/", 1)[-1], representation="sra",
                                 note=".sra container; needs fasterq-dump to become FASTQ"))
    if acc:
        out.append(Route("aws-odp",
                         f"s3://sra-pub-run-odp/sra/{acc}/{acc}", 0, "aws", "aws-odp",
                         accession=acc,
                         note="AWS Open Data Sponsorship Program — AWS pays the egress. "
                              "Use --no-sign-request. Needs fasterq-dump afterwards."))
        out.append(_prefetch_route(acc))
    return out


_SRA_KINDS = ("sra_run", "sra_experiment", "sra_study", "bioproject", "biosample")


def _ena_browser_url(accession: str) -> str:
    return f"https://www.ebi.ac.uk/ena/browser/view/{accession}"


def _prefetch_route(acc: str, note: str = "") -> Route:
    return Route("sra-prefetch", acc, 0, "prefetch", "free", accession=acc,
                 note=note or "NCBI SRA Toolkit: `prefetch` downloads the .sra, `fasterq-dump` "
                              "converts it to FASTQ. Falls back to cloud sources that may be "
                              "metered depending on the run's storage tier")


def _offline_sra_routes(accession: str, kind: str) -> list:
    """What an SRA-family accession's *shape* alone licenses us to say, with no network.

    ENA mirrors all of SRA, so its record page is always a route, and the size is only learned
    from ENA's file report — hence bytes 0 and a note that says so, not a refusal. The NCBI
    toolkit route is offered too, with a caveat when the accession is not a run: `prefetch`
    takes runs, so a study or an experiment has to be expanded to its runs first.
    """
    out = [Route("ena-http", _ena_browser_url(accession), 0, "", "free", streamable=True,
                 accession=accession, note="size known once online")]
    caveat = "" if kind == "sra_run" else (
        f"NCBI SRA Toolkit takes run accessions (SRR/ERR/DRR): expand this {kind} to its runs "
        f"first (ENA's file report lists them once online), then `prefetch` and `fasterq-dump` "
        f"each")
    out.append(_prefetch_route(accession, caveat))
    return out


def _geo_stub(accession: str) -> str:
    """GEO files series and samples under a directory whose name has the last three digits
    replaced by 'nnn', whatever the length: GSE178341 -> GSE178nnn, GSE1234 -> GSE1nnn, and
    everything below a thousand (GSE123, GSE99) -> GSEnnn."""
    prefix, digits = accession[:3], accession[3:]
    return prefix + digits[:-3] + "nnn"


def _one_per_run(routes: list) -> list:
    """The routes that together deliver the data *once*.

    ENA offers one run's reads both as FASTQ files and as an .sra container; those are
    alternatives, not additions, so a run that has FASTQ contributes its FASTQ files (every mate)
    and not the container. A run with only the container contributes that. Routes with no
    representation recorded (GEO, figshare, plain URLs) all count, as before."""
    by_run: dict = {}
    for r in routes:
        # an ENA route with no run accession (a hand-made row) still belongs to *a* run
        by_run.setdefault(r.accession or ("" if r.representation else r.url), []).append(r)
    out: list = []
    for rs in by_run.values():
        fastq = [r for r in rs if r.representation == "fastq"]
        out.extend(fastq or rs)
    return out


def resolve(accession: str, timeout: float = 20.0, offline: bool = False) -> list:
    """Every way to get this accession, cheapest and most streamable first.

    Falls back to a documented manual route rather than raising, because "I cannot reach EBI
    right now" should not look the same as "this data does not exist".
    """
    kind = classify(accession)
    routes: list[Route] = []

    if kind in _SRA_KINDS:
        if offline:
            routes.extend(_offline_sra_routes(accession, kind))
        else:
            try:
                for row in _ena_filereport(accession, timeout):
                    routes.extend(_routes_from_ena_row(row))
            except Exception as e:
                routes.append(Route("manual", ENA_PORTAL, note=f"ENA lookup failed ({e}); "
                                    f"try the browser at {_ena_browser_url(accession)}"))
            else:
                if not routes:                    # ENA answered, and has no files to offer
                    routes.append(Route("manual", _ena_browser_url(accession), accession=accession,
                                        note=f"ENA lists no read files for this {kind} accession "
                                             f"(not public yet, withdrawn, or mistyped); check the "
                                             f"record page"))

    elif kind in ("geo_series", "geo_sample"):
        stub = _geo_stub(accession)
        base = (f"https://ftp.ncbi.nlm.nih.gov/geo/"
                f"{'series' if kind == 'geo_series' else 'samples'}/{stub}/{accession}/suppl/")
        routes.append(Route("geo-ftp", base, 0, "", "free", streamable=True,
                            accession=accession,
                            note="GEO supplementary files — free, and the HTTPS endpoint "
                                 "supports Range, so matrices can be read in place"))
        routes.append(Route("manual",
                            f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
                            note="Landing page; raw reads for human cohorts are usually "
                                 "under dbGaP or EGA instead"))

    elif kind in ("ega_dataset", "ega_study"):
        routes.append(Route("ega", accession, 0, "pyega3", "free", needs_auth=True,
                            accession=accession,
                            note="Controlled access. Requires a Data Access Committee "
                                 "approval and EGA credentials; transfer is free once "
                                 "granted. pip install pyega3"))

    elif kind == "dbgap":
        routes.append(Route("dbgap", accession, 0, "prefetch", "free", needs_auth=True,
                            accession=accession,
                            note="Controlled access; needs an approved dbGaP project and "
                                 "an .ngc key passed to prefetch"))

    elif kind == "arrayexpress":
        routes.append(Route("generic-http",
                            f"https://www.ebi.ac.uk/biostudies/files/{accession}/", 0, "",
                            "free", streamable=True, accession=accession,
                            note="BioStudies/ArrayExpress — free over HTTPS, Aspera also "
                                 "available from EBI"))

    elif kind == "gsa":
        routes.append(Route("generic-http",
                            f"https://download.cncb.ac.cn/gsa/{accession}/", 0, "", "free",
                            accession=accession,
                            note="CNCB-NGDC Genome Sequence Archive. Open GSA is free; "
                                 "GSA-Human needs an application"))

    elif kind == "cngb":
        routes.append(Route("cngb-ftp", "ftp://ftp.cngb.org/pub/CNSA/", 0, "", "free",
                            streamable=True, accession=accession,
                            note="CNGB / CNSA: free anonymous FTP. The data{N} directory is shown "
                                 "on the project page. Do not mirror it: g.list_tree(url) lists "
                                 "every file with sizes, and g.open_remote_gef(url) opens a GEF "
                                 "in place over FTP REST"))
    elif kind == "stomics":
        routes.append(Route("generic-http",
                            f"https://db.cngb.org/stomics/datasets/{accession}", 0, "", "free",
                            accession=accession,
                            note="STOmics DB — Stereo-seq GEF/GEM, which GEMINGA reads "
                                 "natively. Files are served over HTTPS and Aspera"))

    elif kind in ("figshare_doi", "figshare_url"):
        doi = accession.replace("doi:", "")
        art = re.search(r"figshare\.(\d+)", doi)
        url = (f"https://api.figshare.com/v2/articles/{art.group(1)}" if art
               else accession)
        routes.append(Route("generic-http", url, 0, "", "free", streamable=True,
                            accession=accession,
                            note="figshare — free, CC-BY. The API lists files with sizes and "
                                 "direct download URLs that redirect to S3 and honour Range. "
                                 "MS3SEG and MSLesSeg live here"))
    elif kind == "zenodo_doi":
        rec = re.search(r"zenodo\.(\d+)", accession)
        url = f"https://zenodo.org/api/records/{rec.group(1)}" if rec else accession
        routes.append(Route("generic-http", url, 0, "", "free", streamable=True,
                            accession=accession,
                            note="Zenodo — free, stable DOI, Range-capable; the API lists "
                                 "files and checksums"))
    elif kind == "openneuro":
        routes.append(Route("aws-odp", f"s3://openneuro.org/{accession}", 0, "aws", "aws-odp",
                            accession=accession,
                            note="OpenNeuro is in the AWS Open Data Sponsorship Program — "
                                 "--no-sign-request, AWS pays. BIDS layout, which Cohort "
                                 "reads directly"))
        routes.append(Route("generic-http", f"https://openneuro.org/datasets/{accession}",
                            0, "", "free", accession=accession,
                            note="or the openneuro CLI: pip install openneuro-py"))
    elif kind in ("mendeley_doi", "mendeley_url"):
        routes.append(Route("generic-http", accession if kind == "mendeley_url" else
                            f"https://doi.org/{accession.replace('doi:', '')}", 0, "", "free",
                            accession=accession,
                            note="Mendeley Data — free; the landing page links per-file "
                                 "downloads, no API token needed"))
    elif kind == "kaggle":
        slug = re.search(r"kaggle\.com/datasets/([^/?#]+/[^/?#]+)", accession)
        routes.append(Route("kaggle", slug.group(1) if slug else accession, 0, "kaggle",
                            "free", needs_auth=True, accession=accession,
                            note="Free, but the kaggle CLI needs an API token "
                                 "(~/.kaggle/kaggle.json) and some datasets require accepting "
                                 "terms in the browser first"))
    elif kind == "nitrc":
        routes.append(Route("generic-http", accession, 0, "", "free", accession=accession,
                            note="NITRC — free; most projects need a (free) account to reach "
                                 "the file listing, so a bare URL may return the login page"))
    elif kind == "url":
        routes.append(_url_route(accession, timeout, offline))

    elif kind == "s3":
        routes.append(Route("aws-requester", accession, 0, "aws", "aws-requester",
                            note="Check for the Open Data Program first — if the bucket is "
                                 "sponsored, --no-sign-request costs nothing"))
    elif kind == "gs":
        routes.append(Route("gcp-requester", accession, 0, "gcloud", "gcp-requester",
                            note="Requester-pays buckets bill your project for egress; "
                                 "check for a public mirror before paying"))
    if not routes:                      # only a shape no branch above knows reaches this
        routes.append(Route("manual", accession,
                            note=f"unrecognised accession shape ({kind}); give a URL, or "
                                 f"look it up in the source repository"))
    return sorted(routes, key=_rank)


def _url_route(url: str, timeout: float = 10.0, offline: bool = False) -> Route:
    """Classify a bare URL, converting the shареable forms that are really downloads."""
    low = url.lower()
    kind, note = "generic-http", ""
    if "dropbox.com" in low:
        kind = "dropbox"
        # A Dropbox share link is an HTML preview until dl=1 is set.
        parts = urllib.parse.urlsplit(url)
        q = dict(urllib.parse.parse_qsl(parts.query))
        q["dl"] = "1"
        url = urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(q)))
        note = ("Dropbox share link rewritten with dl=1. Public links need no token; "
                "Range support is inconsistent, so streaming may fall back to download")
    elif "onedrive" in low or "1drv.ms" in low:
        note = "OneDrive share links often redirect; expect one hop before the real file"
    elif "zenodo.org" in low:
        note = "Zenodo — free, stable, and Range-capable"
    elif "figshare" in low:
        note = "figshare — free; large files redirect to S3"
    r = Route(kind, url, 0, "", "free", note=note)
    if not offline:
        r.bytes, r.streamable = _probe(url, timeout)
    return r


def _probe(url: str, timeout: float = 10.0) -> tuple:
    """Size and Range support, from one HEAD and one one-byte GET."""
    size, ranged = 0, False
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "geminga/0.6"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            size = int(r.headers.get("Content-Length") or 0)
            ranged = (r.headers.get("Accept-Ranges", "").lower() == "bytes")
    except Exception:
        pass
    if not ranged:
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=0-0",
                                                       "User-Agent": "geminga/0.6"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                ranged = (r.status == 206)
                if not size:
                    cr = r.headers.get("Content-Range", "")
                    if "/" in cr:
                        size = int(cr.rsplit("/", 1)[-1])
        except Exception:
            pass
    return size, ranged


@dataclass
class AcquisitionPlan:
    """What it would take to get a dataset, before committing to it."""
    accession: str
    kind: str
    routes: list = field(default_factory=list)
    files: int = 0
    total_bytes: int = 0

    @property
    def best(self) -> Optional[Route]:
        usable = [r for r in self.routes if r.available and not r.needs_auth]
        return (usable or self.routes or [None])[0]

    @property
    def cheapest_cost(self) -> Optional[float]:
        """The cheapest way in, in USD. 0.0 when some route is free; None when nothing is free
        and a metered route's size is not known yet (it could be the cheapest, and 0 bytes x
        $0.09/GB is not a price)."""
        if not self.routes:
            return 0.0
        known = [r.cost_usd for r in self.routes if r.cost_known]
        if known and min(known) == 0:
            return 0.0
        if len(known) < len(self.routes):
            return None
        return min(known)

    @property
    def streamable(self) -> bool:
        return any(r.streamable for r in self.routes)

    def by_kind(self) -> dict:
        out: dict[str, list] = {}
        for r in self.routes:
            out.setdefault(r.kind, []).append(r)
        return out

    def explain(self) -> str:
        lines = [f"{self.accession}  ({self.kind})"]
        if self.total_bytes:
            lines.append(f"  {self.files} file(s), {human(self.total_bytes)} total")
        seen = set()
        for r in self.routes:
            if r.kind in seen:
                continue
            seen.add(r.kind)
            group = _one_per_run([x for x in self.routes if x.kind == r.kind])
            n = sum(x.bytes for x in group)
            cost = sum(x.cost_usd for x in group)
            marks = []
            if r.streamable:
                marks.append("streamable")
            if r.needs_auth:
                marks.append("needs approval")
            if r.tool and not r.available:
                marks.append(f"install {r.tool}")
            if not all(x.cost_known for x in group):
                price = "unknown size (metered)"
            else:
                price = "free" if cost == 0 else f"~${cost:.2f}"
            lines.append(f"  {r.kind:<14} {len(group):>3} file(s)  {human(n):>10}  "
                         f"{price:>8}  {' · '.join(marks)}")
            if r.note:
                lines.append(f"                 {r.note}")
        b = self.best
        if b is not None:
            lines.append(f"  -> take {b.kind}" +
                         (" and inspect it in place before transferring anything"
                          if b.streamable else ""))
        metered = [r for r in self.routes if r.metered]
        if metered and self.cheapest_cost == 0:
            priced = [r.cost_usd for r in metered if r.cost_known]
            how = f"~${max(priced):.2f}" if priced else "size unknown"
            lines.append(f"  a metered route exists ({how}); you do not need it")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"accession": self.accession, "kind": self.kind, "files": self.files,
                "total_bytes": self.total_bytes, "streamable": self.streamable,
                "routes": [asdict(r) for r in self.routes]}


def plan(accession: str, timeout: float = 20.0, offline: bool = False) -> AcquisitionPlan:
    """Resolve an accession and price every route, without transferring anything.

    `files` and `total_bytes` count what it takes to get the data once: ENA offers a run's reads
    as FASTQ files and as an `.sra` container, and those are alternatives, so FASTQ is counted
    and the container only when there is no FASTQ."""
    routes = resolve(accession, timeout, offline)
    # One representation per run: the .sra container and the FASTQ files are the same reads.
    primary = _one_per_run([r for r in routes if r.kind in ("ena-http", "geo-ftp", "generic-http",
                                                            "dropbox")])
    return AcquisitionPlan(accession=accession, kind=classify(accession), routes=routes,
                           files=len({r.filename for r in primary if r.filename}) or
                                 len(primary),
                           total_bytes=sum(r.bytes for r in primary
                                           if r.kind in ("ena-http", "generic-http",
                                                         "dropbox")))


# ---------------------------------------------------------------------------
# Inspect without transferring
# ---------------------------------------------------------------------------

def peek(accession_or_url: str, n_bytes: str = "512KB", timeout: float = 30.0,
         route: Optional[Route] = None) -> dict:
    """Read the first slice of a remote file and say what it looks like.

    The point: for a shortlist of candidate datasets, this answers "is this what I think it
    is" for a few hundred kilobytes each instead of terabytes of transfer. Works wherever the
    server honours Range, which ENA, GEO, Zenodo and most CDNs do.
    """
    from ._core import parse_size
    want = int(parse_size(n_bytes))
    if route is None:
        p = plan(accession_or_url, timeout)
        cands = [r for r in p.routes if r.streamable and r.url.startswith("http")]
        if not cands:
            return {"ok": False, "reason": "no Range-capable route for this accession",
                    "plan": p.explain()}
        route = cands[0]
    try:
        req = urllib.request.Request(route.url,
                                     headers={"Range": f"bytes=0-{want - 1}",
                                              "User-Agent": "geminga/0.6"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read(want)
            partial = (r.status == 206)
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}", "url": route.url}

    out = {"ok": True, "url": route.url, "bytes_read": len(raw), "partial": partial,
           "total_bytes": route.bytes, "cost_avoided_usd": route.cost_usd}
    if route.bytes:
        out["fraction_read"] = len(raw) / route.bytes
    out.update(_sniff(raw, route.filename or route.url))
    return out


def _gunzip_prefix(raw: bytes, limit: int = 1 << 20) -> tuple:
    """The start of a gzip stream that may be cut off, as `(decoded bytes, complete)`.

    A byte range ends wherever it ends, usually mid-stream, and `gzip.GzipFile.read` answers a
    truncated stream by raising — discarding everything it had already decoded. A raw
    `zlib` decompressor just hands back what it could decode, which is all a peek needs. Members
    are chained, so block-gzipped (BGZF) and concatenated files read past their first member.
    `complete` is False if the input ended mid-member, hit `limit`, or turned out damaged.
    """
    import zlib
    out, total, rest = [], 0, raw
    while rest and total < limit:
        if not rest.strip(b"\x00"):                # zero padding after the last member
            return b"".join(out), True
        d = zlib.decompressobj(wbits=31)           # 16 + MAX_WBITS: expect a gzip header
        try:
            chunk = d.decompress(rest, limit - total)
        except zlib.error:
            return b"".join(out), False
        out.append(chunk)
        total += len(chunk)
        if not d.eof:                               # ran out of input, or out of room
            return b"".join(out), False
        rest = d.unused_data
    return b"".join(out), not rest


def _sniff(raw: bytes, name: str = "") -> dict:
    """What is in these bytes? Deliberately shallow — the profiler does the real work."""
    info: dict = {"gzipped": raw[:2] == b"\x1f\x8b"}
    body = raw
    if info["gzipped"]:
        body, complete = _gunzip_prefix(raw)
        if not body:
            info["note"] = "gzip header seen but nothing decoded from this slice (too short, or damaged)"
        elif not complete:
            info["note"] = ("gzip stream cut at the range boundary — normal; what decoded is what "
                            "was inspected")
    head = body[:200_000].decode("utf-8", "replace")
    lines = head.splitlines()
    if head.startswith("@") and len(lines) >= 4:
        seqs = lines[1::4][:500]
        seqs = [s for s in seqs if s and set(s.upper()) <= set("ACGTUN")]
        info["format"] = "FASTQ"
        if seqs:
            info["reads_sampled"] = len(seqs)
            info["read_length_mean"] = round(sum(len(s) for s in seqs) / len(seqs), 1)
            gc = sum(s.upper().count("G") + s.upper().count("C") for s in seqs)
            info["gc_fraction"] = round(gc / max(sum(len(s) for s in seqs), 1), 4)
            info["first_id"] = lines[0][:120]
    elif head.startswith(">"):
        info["format"] = "FASTA"
        info["first_id"] = lines[0][:120]
    elif raw[:4] == b"PAR1":
        info["format"] = "Parquet"
    elif raw[:8] == b"\x89HDF\r\n\x1a\n":
        info["format"] = "HDF5 (h5ad / GEF / 10x)"
    elif head.startswith("%%MatrixMarket"):
        info["format"] = "MatrixMarket"
        info["header"] = lines[1][:120] if len(lines) > 1 else ""
    elif head.startswith("#FileFormat=GEM"):
        info["format"] = "Stereo-seq GEM"
        info["header"] = "; ".join(l for l in lines[:8] if l.startswith("#"))[:200]
    elif "\t" in head[:2000]:
        info["format"] = "TSV"
        info["columns"] = lines[0].split("\t")[:12] if lines else []
    elif "," in head[:2000]:
        info["format"] = "CSV"
        info["columns"] = lines[0].split(",")[:12] if lines else []
    else:
        info["format"] = "unknown"
    return info


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------

#: External tools this module knows how to drive. It never bundles or reimplements them.
TOOLS = {
    "ascp":      "IBM Aspera FASP client (proprietary; ships with Aspera Connect / CLI)",
    "aspera":    "aspera-cli wrapper around ascp",
    "prefetch":  "NCBI SRA Toolkit — downloads .sra containers",
    "fasterq-dump": "NCBI SRA Toolkit — converts .sra to FASTQ (replaces fastq-dump)",
    "fastq-dl":  "pip install fastq-dl — tries ENA then SRA, handles retries",
    "kingfisher": "pip install kingfisher — tries ena-ascp, ena-ftp, aws-odp, prefetch in order",
    "pyega3":    "pip install pyega3 — EGA controlled-access client",
    "aws":       "AWS CLI — use --no-sign-request on Open Data buckets",
    "gcloud":    "Google Cloud CLI (gcloud storage / gsutil)",
    "curl":      "generic HTTP/FTP",
    "wget":      "generic HTTP/FTP",
    "aria2c":    "multi-connection HTTP; often several times faster than curl on one file",
    "kaggle":    "pip install kaggle — needs ~/.kaggle/kaggle.json",
    "openneuro": "pip install openneuro-py — BIDS datasets from OpenNeuro",
}


def available_tools() -> dict:
    """Which transfer tools are on PATH, and what each is for."""
    return {name: {"path": shutil.which(name), "present": shutil.which(name) is not None,
                   "what": what} for name, what in TOOLS.items()}


def print_tools() -> None:
    for name, d in available_tools().items():
        mark = "yes" if d["present"] else " no"
        print(f"  [{mark}] {name:<14} {d['what']}")


def command_for(route: Route, dest: str = ".", threads: int = 4) -> list:
    """The command line that would move this route's bytes. Built, not run."""
    d = os.path.abspath(os.path.expanduser(dest))
    if route.kind == "ena-aspera":
        host = route.url.split("@", 1)[-1]
        key = os.environ.get("ASPERA_KEY", "$ASPERA_KEY")
        return ["ascp", "-QT", "-l", "300m", "-P", "33001", "-i", key,
                f"era-fasp@fasp.sra.ebi.ac.uk:{host.split(':', 1)[-1] if ':' in host else host}",
                d]
    if route.kind in ("ena-http", "geo-ftp", "generic-http", "dropbox"):
        if shutil.which("aria2c"):
            return ["aria2c", "-x", str(threads), "-s", str(threads), "-d", d, route.url]
        if shutil.which("curl"):
            return ["curl", "-L", "--retry", "5", "-C", "-", "-o",
                    os.path.join(d, route.filename or route.url.rsplit("/", 1)[-1]),
                    route.url]
        return ["wget", "-c", "-P", d, route.url]
    if route.kind == "ena-ftp":
        return ["wget", "-c", "-P", d, route.url]
    if route.kind == "aws-odp":
        return ["aws", "s3", "cp", "--no-sign-request", route.url, d]
    if route.kind == "aws-requester":
        return ["aws", "s3", "cp", "--request-payer", "requester", route.url, d]
    if route.kind == "gcp-requester":
        return ["gcloud", "storage", "cp", "-u", "$GCP_BILLING_PROJECT", route.url, d]
    if route.kind == "gs-public":
        return ["gcloud", "storage", "cp", route.url, d]
    if route.kind == "sra-prefetch":
        return ["prefetch", "--output-directory", d, route.accession]
    if route.kind == "dbgap":
        return ["prefetch", "--ngc", os.environ.get("DBGAP_NGC", "$DBGAP_NGC"),
                "--output-directory", d, route.accession]
    if route.kind == "ega":
        return ["pyega3", "-cf", os.environ.get("EGA_CREDENTIALS", "$EGA_CREDENTIALS"),
                "fetch", route.accession, "--output-dir", d]
    if route.kind == "kaggle":
        return ["kaggle", "datasets", "download", "-d", route.url, "-p", d, "--unzip"]
    return ["# no automated command for this route:", route.url]


def fetch(accession: str, dest: str = ".", route: Optional[Route] = None,
          dry_run: bool = True, timeout: float = 20.0, threads: int = 4) -> dict:
    """Move the bytes, using whichever tool is installed.

    Defaults to `dry_run=True` on purpose. A transfer that might cost money or fill a disk
    should be a decision, not a side effect of exploring.

    With no `route`, the best route's *kind* is taken whole: every file of it (both mates of a
    paired run, every run of a study), one representation per run, so FASTQ and not the `.sra`
    beside it. The result lists them in `commands` and `urls`; `command` is the same commands
    joined with `&&`. Pass `route=` to move exactly one route.
    """
    p = plan(accession, timeout) if route is None else None
    r = route or (p.best if p else None)
    if r is None:
        return {"ok": False, "reason": "no route found"}
    # A planned fetch takes every file of the chosen kind — both mates of a pair, every run of a
    # study — as one representation per run (FASTQ, not the .sra beside it). An explicit
    # `route=` is exactly that one route.
    group = _one_per_run([x for x in p.routes if x.kind == r.kind]) if p is not None else [r]
    cmds = [command_for(x, dest, threads) for x in group]
    out = {"route": r.kind, "url": group[0].url, "urls": [x.url for x in group],
           "bytes": sum(x.bytes for x in group), "cost_usd": sum(x.cost_usd for x in group),
           "command": " && ".join(" ".join(c) for c in cmds),
           "commands": [" ".join(c) for c in cmds], "dry_run": dry_run}
    # Approval is the deeper blocker: installing a client does not get you the data. Say so
    # first, and mention the missing tool in the same breath rather than one at a time.
    if r.needs_auth:
        extra = ("" if r.available else
                 f" You will also need {r.tool}: {TOOLS.get(r.tool, '')}.")
        out.update(ok=False,
                   reason=f"this route needs approved credentials; run the command yourself "
                          f"once they are in place.{extra}")
        return out
    if r.tool and not r.available:
        out.update(ok=False, reason=f"{r.tool} is not installed. {TOOLS.get(r.tool, '')}")
        return out
    if dry_run:
        out["ok"] = True
        out["note"] = "nothing transferred — pass dry_run=False to run this"
        return out
    os.makedirs(os.path.abspath(os.path.expanduser(dest)), exist_ok=True)
    try:
        for cmd in cmds:                              # one after another; stop at the first failure
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
            out["ok"] = proc.returncode == 0
            out["returncode"] = proc.returncode
            out["stderr"] = proc.stderr[-2000:]
            if proc.returncode != 0:
                break
    except Exception as e:
        out.update(ok=False, reason=f"{type(e).__name__}: {e}")
    return out


def plan_many(accessions: Sequence[str], timeout: float = 20.0,
              offline: bool = False) -> dict:
    """Price a whole shortlist at once — what the inventory question actually looks like."""
    out = {}
    for a in accessions:
        try:
            out[a] = plan(a, timeout, offline)
        except Exception as e:
            out[a] = AcquisitionPlan(a, classify(a),
                                     [Route("manual", a, note=f"{type(e).__name__}: {e}")])
    return out


def print_shortlist(plans: dict) -> None:
    total_free = total_metered = 0
    print(f"  {'accession':<18} {'kind':<14} {'files':>5} {'size':>10} {'cheapest':>9}  route")
    for a, p in plans.items():
        b = p.best
        cost = p.cheapest_cost                     # None: metered, and no size to price yet
        total_free += p.total_bytes if cost == 0 else 0
        total_metered += p.total_bytes if cost is not None and cost > 0 else 0
        label = "free" if cost == 0 else ("unknown" if cost is None else f"${cost:.2f}")
        print(f"  {a:<18} {p.kind:<14} {p.files:>5} {human(p.total_bytes):>10} {label:>9}  "
              f"{b.kind if b else '-'}{'  (streamable)' if p.streamable else ''}")
    print(f"\n  {human(total_free)} available free"
          + (f", {human(total_metered)} only on metered routes" if total_metered else ""))
