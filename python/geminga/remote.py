"""Working with a dataset too big to download, on the server that holds it.

The suggested way to fetch a CNGB project is

    wget -c -nH -np -r -R index.html* --cut-dirs 4 ftp://ftp.cngb.org/pub/CNSA/data2/<CNP>/<CNS>/

which mirrors the whole tree. For a project of several terabytes that is the wrong first step on
any machine a student has: it needs the disk, it needs days, and it moves everything before you
know which files you want. The alternative has three parts, and none of them requires the mirror.

**List before fetching.** `list_tree()` walks an FTP or HTTP directory tree and returns every
file with its size, reading only directory listings. Totals by file type tell you where the
terabytes actually are — usually raw FASTQ — and what the analysis-ready files cost.

**Open in place.** `RemoteFile` is a seekable file over FTP `REST` offsets or HTTP `Range`
requests, with a block cache. HDF5 is designed for random access and h5py accepts any seekable
file-like object, so a GEF can be opened *on the server*: its bin levels, extent and gene index
come from a few hundred kilobytes of metadata, a coarse overview from one small level, and a
gene from one contiguous slice.

**Fetch only what a question needs.** `fetch_selected()` builds `wget` commands for chosen files
rather than the tree. And for fine-grained spatial work on one chip, the GEF spatial index can be
built *from the remote file* into a local sidecar: one sequential pass over the levels you care
about, after which windows are answered locally in milliseconds.

The cache is bounded and, given a Budget, reserved against it; every byte that crosses the
network is counted in `.stats()`.
"""
from __future__ import annotations

import fnmatch
import ftplib
import http.client
import io
import os
import posixpath
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Sequence

from ._core import Budget, human, parse_size
from .workspace import try_acquire

__all__ = ["RemoteFile", "RemoteEntry", "list_tree", "summarise_tree", "print_tree_summary",
           "fetch_selected", "open_remote_gef", "wget_mirror_command"]

_UA = "geminga/0.12"

_CRANGE = re.compile(r"\s*bytes\s+(\d+)-(\d+)/(\d+|\*)\s*$", re.I)       # Content-Range of a 206
_CRANGE_NONE = re.compile(r"\s*bytes\s+\*/(\d+)\s*$", re.I)               # ...and of a 416
_HEAD_REFUSED = (400, 403, 405, 501)       # servers that will not answer HEAD but will answer GET


class _RangeRefused(IOError):
    """The server answered a Range request with something other than the range asked for.
    Asking again gets the same answer, so this is not retried."""


def _length(header: Optional[str]) -> Optional[int]:
    return int(header) if header is not None and header.strip().isdigit() else None


def _http_size(url: str, timeout: float = 30.0) -> int:
    """The size of an HTTP resource: HEAD's Content-Length, else the total from a one-byte
    ranged GET (`Content-Range: bytes 0-0/<total>`). Raises IOError when the server will say
    neither — a size of 0 would make every later read return b"" without a word."""
    headers = {"User-Agent": _UA, "Accept-Encoding": "identity"}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=headers),
                                    timeout=timeout) as r:
            n = _length(r.headers.get("Content-Length"))
            if n is not None:
                return n
    except urllib.error.HTTPError as e:
        if e.code not in _HEAD_REFUSED:
            raise
    probe = urllib.request.Request(url, headers={**headers, "Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(probe, timeout=timeout) as r:
            if r.status == 206:
                m = _CRANGE.match(r.headers.get("Content-Range") or "")
                if m and m.group(3) != "*":
                    return int(m.group(3))
            elif r.status == 200:                       # ignores Range, but may say how long
                n = _length(r.headers.get("Content-Length"))
                if n is not None:
                    return n
    except urllib.error.HTTPError as e:
        m = _CRANGE_NONE.match(e.headers.get("Content-Range") or "") if e.code == 416 else None
        if m is None:
            raise
        return int(m.group(1))                          # an empty file satisfies no range
    raise IOError(f"cannot determine the size of {url}: the server gave no Content-Length for "
                  f"HEAD and no total in a ranged GET")


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

@dataclass
class RemoteEntry:
    url: str
    path: str                  # relative to the listing root
    size: int
    kind: str = "file"         # "file" | "dir"
    modify: str = ""

    @property
    def name(self) -> str:
        return posixpath.basename(self.path)

    @property
    def ext(self) -> str:
        n = self.name.lower()
        for e in (".cellbin.gef", ".tissue.gef", ".gef", ".gem.gz", ".gem", ".fq.gz",
                  ".fastq.gz", ".fq", ".fastq", ".h5ad", ".h5", ".tif", ".tiff",
                  ".ome.tif", ".bam", ".csv", ".txt", ".json", ".md5"):
            if n.endswith(e):
                return e
        return os.path.splitext(n)[1] or "(none)"


def _ftp_connect(host: str, port: int = 21, user: str = "anonymous", passwd: str = "",
                 timeout: float = 30.0) -> ftplib.FTP:
    ftp = ftplib.FTP(timeout=timeout)
    ftp.connect(host, port)
    ftp.login(user, passwd or "geminga@")
    ftp.voidcmd("TYPE I")
    ftp.set_pasv(True)
    return ftp


def _split(url: str):
    u = urllib.parse.urlsplit(url)
    return u.scheme, u.hostname, u.port, urllib.parse.unquote(u.path) or "/"


def _ftp_list(ftp: ftplib.FTP, path: str) -> list:
    """(name, kind, size, modify) for one directory. MLSD first, LIST as the fallback."""
    out = []
    try:
        for name, facts in ftp.mlsd(path, facts=["type", "size", "modify"]):
            if name in (".", ".."):
                continue
            t = facts.get("type", "")
            kind = "dir" if t in ("dir", "cdir", "pdir") else "file"
            if t in ("cdir", "pdir"):
                continue
            out.append((name, kind, int(facts.get("size", 0) or 0), facts.get("modify", "")))
        return out
    except ftplib.error_perm:
        pass
    lines: list = []
    ftp.retrlines(f"LIST {path}", lines.append)
    for ln in lines:
        parts = ln.split(None, 8)
        if len(parts) < 9:
            continue
        perms, size, name = parts[0], parts[4], parts[8]
        if name in (".", ".."):
            continue
        out.append((name, "dir" if perms.startswith("d") else "file",
                    int(size) if size.isdigit() else 0, " ".join(parts[5:8])))
    return out


def _http_list(url: str, timeout: float = 30.0) -> list:
    """Parse an autoindex HTML page into (name, kind, size, modify)."""
    req = urllib.request.Request(url if url.endswith("/") else url + "/",
                                 headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        html = r.read().decode("utf-8", "replace")
    out = []
    for href in re.findall(r'href="([^"?#]+)"', html):
        if href.startswith(("/", "http", "..")) or href in ("./",):
            continue
        name = urllib.parse.unquote(href)
        out.append((name.rstrip("/"), "dir" if href.endswith("/") else "file", 0, ""))
    return out


def list_tree(url: str, *, max_depth: int = 6, max_files: int = 200_000,
              include: Optional[Sequence[str]] = None, timeout: float = 30.0,
              sizes: bool = True) -> list:
    """Every file under `url`, with sizes, from directory listings alone. Nothing is downloaded.

    Works over FTP (MLSD, falling back to LIST) and HTTP autoindex pages. For HTTP, sizes come
    from one HEAD per file when `sizes=True`, which is slower; FTP listings carry them for free.
    `include` filters by glob on the file name, e.g. ``["*.gef", "*.h5ad"]``.
    """
    scheme, host, port, root = _split(url)
    files: list[RemoteEntry] = []
    if scheme == "ftp":
        ftp = _ftp_connect(host, port or 21, timeout=timeout)
        try:
            stack = [(root.rstrip("/") or "/", 0)]
            while stack and len(files) < max_files:
                d, depth = stack.pop()
                for name, kind, size, mod in _ftp_list(ftp, d):
                    full = posixpath.join(d, name)
                    rel = posixpath.relpath(full, root.rstrip("/") or "/")
                    if kind == "dir":
                        if depth < max_depth:
                            stack.append((full, depth + 1))
                        continue
                    if include and not any(fnmatch.fnmatch(name.lower(), p.lower())
                                           for p in include):
                        continue
                    port_s = f":{port}" if port else ""
                    files.append(RemoteEntry(f"ftp://{host}{port_s}{full}", rel, size,
                                             "file", mod))
        finally:
            try:
                ftp.quit()
            except Exception:
                pass
    elif scheme in ("http", "https"):
        stack = [(url.rstrip("/") + "/", 0)]
        base = url.rstrip("/") + "/"
        while stack and len(files) < max_files:
            d, depth = stack.pop()
            for name, kind, size, mod in _http_list(d, timeout):
                full = d + name + ("/" if kind == "dir" else "")
                if kind == "dir":
                    if depth < max_depth:
                        stack.append((full, depth + 1))
                    continue
                if include and not any(fnmatch.fnmatch(name.lower(), p.lower())
                                       for p in include):
                    continue
                if sizes and not size:
                    try:
                        size = _http_size(full, timeout)
                    except (OSError, ValueError, http.client.HTTPException):
                        size = 0          # unknown: say so in the listing, do not abandon it
                files.append(RemoteEntry(full, full[len(base):], size, "file", mod))
    else:
        raise ValueError(f"list_tree needs an ftp:// or http(s):// URL, got {url!r}")
    return sorted(files, key=lambda e: e.path)


def summarise_tree(entries: Sequence[RemoteEntry]) -> dict:
    by_ext: dict[str, list] = {}
    for e in entries:
        by_ext.setdefault(e.ext, []).append(e)
    rows = sorted(((ext, len(v), sum(x.size for x in v)) for ext, v in by_ext.items()),
                  key=lambda r: -r[2])
    return {"files": len(entries), "bytes": sum(e.size for e in entries), "by_type": rows,
            "largest": sorted(entries, key=lambda e: -e.size)[:10]}


def print_tree_summary(entries: Sequence[RemoteEntry], disk_free: Optional[int] = None) -> None:
    s = summarise_tree(entries)
    print(f"{s['files']:,} files, {human(s['bytes'])}")
    print(f"  {'type':<14} {'files':>7} {'size':>11} {'share':>7}")
    for ext, n, b in s["by_type"]:
        print(f"  {ext:<14} {n:>7,} {human(b):>11} {b / max(s['bytes'], 1):>7.1%}")
    ana = sum(b for ext, n, b in s["by_type"] if ext in (".gef", ".tissue.gef", ".cellbin.gef",
                                                          ".h5ad", ".gem.gz", ".gem"))
    print(f"  analysis-ready (GEF / GEM / h5ad): {human(ana)} "
          f"= {ana / max(s['bytes'], 1):.1%} of the tree")
    if disk_free is not None:
        print(f"  local disk free: {human(disk_free)} — the full mirror "
              f"{'fits' if s['bytes'] < disk_free else 'does NOT fit'}")


# ---------------------------------------------------------------------------
# A seekable file on a server
# ---------------------------------------------------------------------------

class RemoteFile(io.RawIOBase):
    """Random access to a remote file over FTP REST or HTTP Range, with a bounded block cache.

    h5py, pyarrow and tifffile all accept a seekable file-like object, so this is enough to
    open a GEF, a Parquet file or a TIFF where it sits. Each cache miss costs one request; FTP
    opens a data connection per miss, so the block size is larger than you would pick for
    HTTP and sequential reads trigger read-ahead.

    What it will not do is hand back bytes it cannot vouch for. A reply must be the range that
    was asked for (HTTP 206 with a matching Content-Range; a 200 only answers a request that
    starts at byte 0), and it must be as long as asked. A short reply is requested once more and
    then raised as an IOError; it is never cached. A server that will not say how big the file
    is makes opening it an error, not an empty file.
    """

    RETRIES = 1          # a reply that comes back short is asked for once more, then it is an error

    def __init__(self, url: str, *, block: str = "1MB", cache: str = "64MB",
                 budget: Optional[Budget] = None, readahead: int = 4, timeout: float = 60.0):
        super().__init__()
        # close() must be safe on a half-built object, so what it touches exists first.
        self._ftp: Optional[ftplib.FTP] = None
        self._lease = None
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._lock = threading.Lock()
        self._size = 0
        self._stalled = ""
        self.url = url
        self.scheme, self.host, self.port, self.rpath = _split(url)
        if self.scheme not in ("ftp", "http", "https"):
            raise ValueError(f"RemoteFile supports ftp and http(s), got {url!r}")
        self.block = int(parse_size(block))
        if self.block <= 0:
            raise ValueError(f"block must be positive, got {block!r}")
        self.cache_cap = max(int(parse_size(cache)) // self.block, 2)
        self.readahead = max(1, int(readahead))
        self.timeout = timeout
        self._pos = 0
        self._last_block = -2
        self.requests = 0
        self.bytes_transferred = 0
        self.hits = 0
        self.misses = 0
        try:
            if budget is not None:
                need = self.cache_cap * self.block
                self._lease = try_acquire(budget, "ram", need)
                if self._lease is None:
                    raise MemoryError(
                        f"cannot reserve {human(need)} of RAM for the read cache of {url}: only "
                        f"{human(budget.available('ram'))} is free in the budget. Use a smaller "
                        f"cache=, or free some first.")
            self._size = self._fetch_size()
        except BaseException:
            self.close()                         # give back the cache reservation and the socket
            raise

    # ---- io protocol ----
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if whence == 0:
            new = offset
        elif whence == 1:
            new = self._pos + offset
        elif whence == 2:
            new = self._size + offset
        else:
            raise ValueError(f"invalid whence ({whence}, should be 0, 1 or 2)")
        if new < 0:
            raise ValueError(f"negative seek position {new}")
        self._pos = new
        return new

    @property
    def size(self) -> int:
        return self._size

    def readinto(self, b) -> int:
        view = memoryview(b).cast("B")
        data = self.read(len(view))
        view[:len(data)] = data
        return len(data)

    def read(self, n: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if n is None or n < 0:
            n = self._size - self._pos
        n = max(0, min(n, self._size - self._pos))
        if n == 0:
            return b""
        out = bytearray(n)                       # the size is known: one buffer, filled in place
        mv = memoryview(out)
        pos, filled = self._pos, 0
        end = pos + n
        while pos < end:
            bi = pos // self.block
            blk = self._block(bi)
            off = pos - bi * self.block
            take = min(end - pos, len(blk) - off)
            if take <= 0:                        # unreachable while blocks are validated; loud if not
                raise IOError(f"{self.url}: block {bi} holds {len(blk)} bytes, "
                              f"too few to reach offset {pos}")
            mv[filled:filled + take] = memoryview(blk)[off:off + take]
            pos += take
            filled += take
        mv.release()
        self._pos = pos                          # only a read that completed moves the position
        return bytes(out)                        # the one copy: bytes is what callers get

    def close(self):
        if self._ftp is not None:
            try:
                self._ftp.quit()
            except Exception:
                pass
            self._ftp = None
        if self._lease is not None:
            self._lease.release()
            self._lease = None
        self._cache.clear()
        super().close()

    # ---- blocks ----
    def _block(self, bi: int) -> bytes:
        with self._lock:
            blk = self._cache.get(bi)
            if blk is not None:
                self._cache.move_to_end(bi)
                self.hits += 1
                self._last_block = bi
                return blk
            self.misses += 1
            span = 1
            if bi == self._last_block + 1:
                # Sequential access: fetch several blocks in one request — but never more than
                # the cache can keep beside an older block. A span as large as the cache evicts
                # the very block it was fetched for (block=1MB, cache=2MB read 1 MB, nothing,
                # 1 MB, nothing).
                span = max(1, min(self.readahead, self.cache_cap - 1,
                                  -(-self._size // self.block) - bi))
                for k in range(1, span):                 # no need to fetch what we already hold
                    if bi + k in self._cache:
                        span = k
                        break
            start = bi * self.block
            data = self._fetch(start, min(self.block * span, self._size - start))
            fetched = {}
            for k in range(span):
                piece = data[k * self.block:(k + 1) * self.block]
                if len(piece) == min(self.block, self._size - (bi + k) * self.block):
                    fetched[bi + k] = piece              # a short block is never kept
            if bi not in fetched:                        # _fetch checked, so this cannot happen
                raise IOError(f"{self.url}: block {bi} came back short")
            for k, piece in fetched.items():
                self._cache[k] = piece
            while len(self._cache) > self.cache_cap:
                self._cache.popitem(last=False)
            self._last_block = bi + span - 1
            return fetched[bi]

    # ---- transport ----
    def _conn(self) -> ftplib.FTP:
        if self._ftp is None:
            self._ftp = _ftp_connect(self.host, self.port or 21, timeout=self.timeout)
        return self._ftp

    def _fetch_size(self) -> int:
        if self.scheme == "ftp":
            n = self._conn().size(self.rpath)
            if n is None:
                raise IOError(f"cannot determine the size of {self.url}: the FTP server gave no "
                              f"size for it (no 213 reply to SIZE)")
            return int(n)
        return _http_size(self.url, self.timeout)

    def _fetch(self, start: int, length: int) -> bytes:
        """Exactly `length` bytes from `start`, or an IOError.

        `length` was clipped to the file by the caller, so a shorter reply is a transfer that
        died or a server that sent less than it was asked for — never end-of-file. It is asked
        for again `RETRIES` times and then raised, so a short block cannot be passed on, or
        cached, as if it were whole.
        """
        length = min(length, self._size - start)               # nothing exists past end-of-file
        if length <= 0:
            return b""
        got, why = 0, ""
        for _ in range(self.RETRIES + 1):
            self.requests += 1
            self._stalled = ""
            if self.scheme == "ftp":
                data = self._ftp_range(start, length)
            else:
                data = self._http_range(start, length)
            self.bytes_transferred += len(data)
            if len(data) == length:
                return data
            got, why = len(data), f" ({self._stalled})" if self._stalled else ""
        raise IOError(f"short read from {self.url}: asked for {length} bytes at offset {start} "
                      f"and got {got} after {self.RETRIES + 1} tries{why}")

    def _http_range(self, start: int, length: int) -> bytes:
        """`length` bytes at `start` — or whatever the server sent, if it was the right range.

        A server that does not honour Range answers 200 with the whole file. That is acceptable
        for a request that starts at byte 0 (the body is cut to the length asked for) and
        nothing else: reading it as if it were the range at `start` returns the wrong bytes
        without any error.
        """
        req = urllib.request.Request(self.url, headers={
            "Range": f"bytes={start}-{start + length - 1}", "User-Agent": _UA,
            "Accept-Encoding": "identity"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            if r.status == 206:
                cr = r.headers.get("Content-Range") or ""
                m = _CRANGE.match(cr)
                if m is None:
                    raise _RangeRefused(f"{self.url}: a 206 reply without a readable "
                                        f"Content-Range header ({cr!r})")
                if int(m.group(1)) != start:
                    raise _RangeRefused(f"{self.url}: asked for the range starting at byte "
                                        f"{start}, but the reply's Content-Range is {cr!r}")
                if m.group(3) != "*" and int(m.group(3)) != self._size:
                    raise _RangeRefused(f"{self.url} changed on the server: it was {self._size} "
                                        f"bytes when opened and the reply says {m.group(3)}")
            elif r.status == 200:
                if start != 0:
                    raise _RangeRefused(
                        f"{self.url}: the server ignored the Range request and answered 200 "
                        f"with the whole file, so offset {start} cannot be read here. Only the "
                        f"start of the file is reachable; use a server that honours Range, or "
                        f"fetch the file.")
            else:
                raise _RangeRefused(f"{self.url}: HTTP {r.status} is not an answer to a "
                                    f"Range request")
            try:
                return r.read(length)
            except http.client.IncompleteRead as e:
                return bytes(e.partial)
            except OSError as e:                  # stalled or reset part-way through the body
                self._stalled = f"{type(e).__name__}: {e}"
                return b""

    def _ftp_range(self, start: int, length: int, retry: bool = True) -> bytes:
        """RETR from an offset and hang up after `length` bytes. The server reports the
        aborted transfer on the control connection; that reply is consumed and ignored.
        The caller checks that `length` bytes arrived."""
        try:
            ftp = self._conn()
            conn = ftp.transfercmd(f"RETR {self.rpath}", rest=start)
            buf = bytearray()
            try:
                while len(buf) < length:
                    chunk = conn.recv(min(1 << 16, length - len(buf)))
                    if not chunk:
                        break
                    buf += chunk
            finally:
                conn.close()
            try:
                ftp.voidresp()
            except ftplib.all_errors:
                pass
            return bytes(buf)
        except (ftplib.all_errors, OSError):
            if not retry:
                raise
            try:
                if self._ftp is not None:
                    self._ftp.close()
            except Exception:
                pass
            self._ftp = None
            return self._ftp_range(start, length, retry=False)

    def drop_cache(self) -> "RemoteFile":
        """Forget cached blocks, so the next reads show their true network cost."""
        with self._lock:
            self._cache.clear()
            self._last_block = -2
        return self

    def stats(self) -> dict:
        """Bytes are counted as *received*. Over FTP the server cannot be told where to stop —
        RETR has a start offset but no end — so it keeps sending until the client hangs up,
        and up to one TCP window per request is sent and discarded. HTTP Range has no such
        overshoot."""
        return {"url": self.url, "size": self._size, "size_human": human(self._size),
                "requests": self.requests, "bytes_transferred": self.bytes_transferred,
                "transferred_human": human(self.bytes_transferred),
                "fraction_of_file": self.bytes_transferred / max(self._size, 1),
                "cache_hits": self.hits, "cache_misses": self.misses,
                "block": human(self.block), "cache_blocks": self.cache_cap}

    def __repr__(self) -> str:
        s = self.stats()
        return (f"RemoteFile({posixpath.basename(self.rpath)}, {s['size_human']}, "
                f"{s['transferred_human']} transferred in {s['requests']} requests)")


def open_remote_gef(url: str, budget: Optional[Budget] = None, *, block: str = "1MB",
                    cache: str = "64MB", group: str = "geneExp"):
    """A Stereo-seq view over a GEF that stays on the server.

    Metadata, coarse levels and gene slices are cheap. A fine spatial window reads its whole
    level unless a spatial index exists — build one locally from the remote file with
    `view.gef.build_spatial_index(levels=[...])`, which makes one sequential pass per level.
    """
    from .stereoseq import GefFile, StereoSeqView
    rf = RemoteFile(url, block=block, cache=cache, budget=budget)
    return StereoSeqView(GefFile(rf, group), budget)


# ---------------------------------------------------------------------------
# Fetching only what is wanted
# ---------------------------------------------------------------------------

def wget_mirror_command(url: str, cut_dirs: int = 4) -> str:
    """The whole-tree mirror, as CNGB suggests it — shown for comparison, not recommended."""
    return f"wget -c -nH -np -r -R 'index.html*' --cut-dirs {cut_dirs} {url}"


def fetch_selected(entries: Sequence[RemoteEntry], dest: str = ".",
                   include: Sequence[str] = ("*.gef",), exclude: Sequence[str] = (),
                   max_bytes: Optional[str] = None) -> dict:
    """Commands for just the files you want, with the total and a disk check.

    Returns the file list, total size, and a `wget -c -x -nH` command per file (`-x` keeps the
    directory layout so paths stay meaningful). Nothing is run.
    """
    import shutil
    pick = [e for e in entries
            if any(fnmatch.fnmatch(e.name.lower(), p.lower()) for p in include)
            and not any(fnmatch.fnmatch(e.name.lower(), p.lower()) for p in exclude)]
    pick.sort(key=lambda e: e.size)
    cap = parse_size(max_bytes) if max_bytes else None
    chosen, total = [], 0
    for e in pick:
        if cap is not None and total + e.size > cap:
            continue
        chosen.append(e)
        total += e.size
    try:
        free = shutil.disk_usage(os.path.abspath(os.path.expanduser(dest)) if os.path.exists(
            os.path.expanduser(dest)) else ".").free
    except OSError:
        free = 0
    d = os.path.expanduser(dest)
    return {"files": chosen, "count": len(chosen), "bytes": total, "bytes_human": human(total),
            "disk_free": free, "fits": total < free,
            "commands": [f"wget -c -x -nH -P {d} '{e.url}'" for e in chosen]}
