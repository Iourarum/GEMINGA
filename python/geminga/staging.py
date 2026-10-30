"""A bounded working area between slow storage and memory.

The streaming side of GEMINGA assumes bytes are expensive to fetch and cheap to forget. This
module is for the opposite situation, which is at least as common: the dataset is already
downloaded, it lives on an external drive or a NAS mount, and it is larger than the machine's
RAM, its VRAM, and quite possibly its internal disk. Nothing needs fetching. What is scarce is
somewhere to put the part you are working on.

So the shape here is a three-level hierarchy, each level bounded and each explicitly declared:

    vault (slow, huge, read-only)  ->  staging (fast, bounded)  ->  device (RAM/VRAM/XPU)

The important design choice is that **staging holds extracted regions, not file copies**.
Copying a 400 GB GEF onto a 200 GB NVMe to work on a 2 GB corner of it is the mistake this is
meant to prevent. A region is read from the vault through the same windowed readers the rest
of the library uses, written to staging in an analysis-ready form, and from then on every
re-read is local and fast. One pass over the slow link per region, not per look.

Eviction is least-recently-used, with pinning, because that matches how people actually work:
you return to the region you were just looking at, and you want the one you spent twenty
minutes extracting to survive until you say otherwise.

The index is a convenience; the directory is the truth. Regions are written to a `.part` file
and renamed into place only when complete and accounted for, so a writer that fails leaves
nothing behind; the index stores paths relative to the root and refuses any that resolve
outside it, so a copied directory or a hand-edited index cannot make an eviction delete
somebody else's file; and several `Staging` objects (or processes) on one directory merge each
other's entries instead of overwriting them. At startup the directory is reconciled with the
index: interrupted writes are removed, and files the index does not know are counted, not lost.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import os
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterator, Optional

from ._core import Budget, human, parse_size
from .workspace import try_acquire

__all__ = ["Staging", "StagedRegion", "staging"]


@dataclass
class StagedRegion:
    """One extracted region, sitting in the working area."""
    key: str
    path: str
    source: str
    bytes: int
    rows: int = 0
    window: Optional[tuple] = None
    format: str = ""
    note: str = ""
    pinned: bool = False
    created: float = 0.0
    accessed: float = 0.0
    hits: int = 0
    meta: dict = field(default_factory=dict)

    def touch(self) -> "StagedRegion":
        self.accessed = time.time()
        self.hits += 1
        return self

    def __repr__(self) -> str:
        pin = " pinned" if self.pinned else ""
        return (f"StagedRegion({self.key}, {human(self.bytes)}, {self.rows:,} rows, "
                f"{self.hits} hits{pin})")


_FIELDS = tuple(f.name for f in dataclasses.fields(StagedRegion))


class StagingFull(RuntimeError):
    """Raised when a region cannot fit even after evicting everything unpinned."""


# ---------------------------------------------------------------------------
# One lock per directory
# ---------------------------------------------------------------------------

class _DirLock:
    """Serialises index changes on one staging directory: a re-entrant lock for the threads of
    this process (shared by every `Staging` object on the directory) and an `flock` on the
    directory itself for other processes, where the platform has one. No lock file is created."""

    def __init__(self, path: str):
        self.path = path
        self._rlock = threading.RLock()
        self._depth = 0
        self._fd: Optional[int] = None

    def __enter__(self):
        self._rlock.acquire()
        if self._depth == 0:
            self._fd = self._flock()
        self._depth += 1
        return self

    def __exit__(self, *exc):
        self._depth -= 1
        if self._depth == 0 and self._fd is not None:
            try:
                os.close(self._fd)            # closing the descriptor releases the flock
            except OSError:
                pass
            self._fd = None
        self._rlock.release()
        return False

    def _flock(self) -> Optional[int]:
        try:
            import fcntl
            fd = os.open(self.path, os.O_RDONLY)
        except (ImportError, OSError):
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError:                       # e.g. a filesystem without locks: carry on unlocked
            os.close(fd)
            return None
        return fd


_DIR_LOCKS: dict = {}
_DIR_LOCKS_GUARD = threading.Lock()
_INFLIGHT: set = set()                         # .part files being written by this process
_PARTS = itertools.count()

# <stem>-<16 hex>.<pid>[.<n>].part<suffix>  — a write in progress or abandoned
_PART_RE = re.compile(r"^(?P<stem>.*)-(?P<h>[0-9a-f]{16})\.(?P<pid>\d+)(?:\.\d+)?\.part(?P<sfx>\..*)?$")
# <stem>-<16 hex><suffix>                    — a finished region, as _slot() names it
_SLOT_RE = re.compile(r"^(?P<stem>.*)-(?P<h>[0-9a-f]{16})(?P<sfx>\..*)?$")


def _dir_lock(path: str) -> _DirLock:
    key = os.path.realpath(path)
    with _DIR_LOCKS_GUARD:
        lk = _DIR_LOCKS.get(key)
        if lk is None:
            lk = _DIR_LOCKS[key] = _DirLock(key)
        return lk


def _writer_alive(pid: int) -> bool:
    """Might `pid` still be writing? Only a process we can positively see to be gone says no."""
    if os.name != "posix":
        return True                           # os.kill(pid, 0) is not a probe elsewhere
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _jsonable(x):
    """What `x` will look like after a trip through the index, so that it can be compared."""
    return None if x is None else json.loads(json.dumps(x, default=str))


def _region_from(d) -> Optional[StagedRegion]:
    try:
        r = StagedRegion(**{k: d[k] for k in _FIELDS if k in d})
        r.bytes = int(r.bytes)
        r.path, r.key = str(r.path), str(r.key)
        if r.window is not None:
            r.window = tuple(r.window)            # JSON gave us a list
        if not isinstance(r.meta, dict):
            r.meta = {}
        return r
    except (TypeError, ValueError):
        return None


class Staging:
    """A size-capped directory that holds extracted regions.

        st = g.staging("/scratch/geminga", capacity="200GB")
        p  = st.extract("chip:tumour", vault_path, window=(12000,16000,9000,13000))
        tbl = st.read(p)            # local and fast from here on
    """

    INDEX = "_geminga_staging.json"

    def __init__(self, path: str, capacity: str = "50GB",
                 budget: Optional[Budget] = None, tier: str = "spill"):
        self.path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(self.path, exist_ok=True)
        self.capacity = parse_size(capacity)
        self.budget, self.tier = budget, tier
        self.regions: dict[str, StagedRegion] = {}
        self._lease = None
        self._dirlock = _dir_lock(self.path)
        self._index_sig = None
        self._dropped = 0
        if budget is not None:
            # Reserve the working area against the spill tier, so a workspace cannot
            # promise the same disk to two different things. Never waits: the disk is not
            # going to appear because we stood there.
            self._lease = try_acquire(budget, tier, self.capacity)
            if self._lease is None:
                raise MemoryError(
                    f"cannot reserve {human(self.capacity)} of '{tier}' for staging at "
                    f"{self.path}: only {human(budget.available(tier))} is free")
        try:
            with self._dirlock:
                self._reconcile()
        except BaseException:
            self.release()
            raise

    # ---- index ----
    @property
    def _index_path(self) -> str:
        return os.path.join(self.path, self.INDEX)

    def _signature(self):
        try:
            st = os.stat(self._index_path)
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _inside(self, p: str) -> Optional[str]:
        """`p` — relative to the root, or absolute — as a path that really is inside the root,
        or None. 'Inside' is decided on the real path, so neither `..` nor a symlink can argue
        its way out, and the index (or its temporaries) is never anybody's region."""
        if not p or "\0" in p:
            return None
        if not any(c in p for c in (os.sep, os.altsep or os.sep)) and p not in (".", ".."):
            # a bare name in the root, which is how every region is stored: there is nothing to
            # resolve unless the file itself is a link — one lstat instead of a realpath each
            full = os.path.join(self.path, p)
            if not self._reserved(p) and not os.path.islink(full):
                return full
        full = os.path.normpath(p if os.path.isabs(p) else os.path.join(self.path, p))
        real, root = os.path.realpath(full), os.path.realpath(self.path)
        if not real.startswith(root + os.sep):
            return None
        if os.path.dirname(real) == root and self._reserved(os.path.basename(real)):
            return None
        return full

    def _reserved(self, name: str) -> bool:
        """The index, and the temporaries it is written through."""
        return name == self.INDEX or name.startswith(self.INDEX + ".")

    def _read_index(self) -> dict:
        """The regions the index on disk lists whose files are really there, with every path
        resolved against this root. Paths that resolve outside it are refused, not followed."""
        out: dict = {}
        self._dropped = 0
        try:
            with open(self._index_path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return out
        if not isinstance(raw, dict):
            return out
        for k, v in raw.items():
            r = _region_from(v) if isinstance(v, dict) else None
            inside = self._inside(r.path) if r is not None else None
            if inside is None and r is not None and os.path.isabs(r.path):
                # an index written with absolute paths (older versions), carried along with its
                # directory to a new place: the file is here, beside the index, under its name
                inside = self._inside(os.path.basename(r.path))
            if r is None or inside is None or not os.path.isfile(inside):   # the disk is the truth
                self._dropped += 1
                continue
            r.path = inside
            out[str(k)] = r
        return out

    def _merge_disk(self, skip=()) -> None:
        """Take in what other Staging objects (or processes) have written to the index since we
        last looked, and forget what they evicted. Entries are updated in place, so a region
        returned earlier is still the one in `regions`. The index on disk wins, except that the
        newest `accessed` and the highest `hits` of either side survive: touches are kept in
        memory until the next write."""
        sig = self._signature()
        disk = self._read_index()
        for k, d in disk.items():
            if k in skip:
                continue
            mine = self.regions.get(k)
            if mine is None:
                self.regions[k] = d
                continue
            d.accessed, d.hits = max(d.accessed, mine.accessed), max(d.hits, mine.hits)
            for name in _FIELDS:
                setattr(mine, name, getattr(d, name))
        for k in [k for k, r in self.regions.items() if k not in disk and not os.path.isfile(r.path)]:
            del self.regions[k]
        self._index_sig = sig

    def _load_index(self) -> None:
        """Re-read the index from disk (the old private name, for callers that reloaded by hand)."""
        with self._dirlock:
            self._merge_disk()

    def _refresh(self) -> None:
        if self._signature() != self._index_sig:
            with self._dirlock:
                self._merge_disk()

    def _row(self, r: StagedRegion) -> dict:
        d = asdict(r)
        d["path"] = os.path.relpath(r.path, self.path)
        return d

    def _save_index(self, skip=(), merge: bool = True) -> None:
        """Write the index. By default merge first, so whatever another object added since we
        last looked stays. A caller that has just merged under the same lock, and then changed
        something, passes `merge=False`: the merge lets the disk win, and would undo the change."""
        with self._dirlock:
            if merge:
                self._merge_disk(skip)
            payload = {k: self._row(r) for k, r in self.regions.items()}
            tmp = f"{self._index_path}.{os.getpid()}.{threading.get_ident()}.tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, default=str)
                os.replace(tmp, self._index_path)
                self._index_sig = self._signature()
            except OSError:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _reconcile(self) -> None:
        """Make the directory and the index agree about what is in the working area.

        Index entries whose files are gone are dropped; interrupted writes (`.part` files whose
        writer is gone) are removed; and a file the index does not know is *counted*: adopted
        as a region named `orphan:<file>`. One of staging's own (named like a slot) is an
        ordinary unpinned region and will be evicted first when space is wanted. Anything
        else is *foreign*: pinned, counted against the capacity, and never deleted by staging's
        own housekeeping (`clear()` skips it too) — only an explicit `evict(key)` removes it.
        """
        self._merge_disk()
        changed = self._dropped > 0
        known = {os.path.realpath(r.path) for r in self.regions.values()}
        for name in sorted(os.listdir(self.path)):
            full = os.path.join(self.path, name)
            if (self._reserved(name) or name.startswith(".") or os.path.islink(full)
                    or not os.path.isfile(full)):
                continue
            if os.path.realpath(full) in known:
                continue
            m = _PART_RE.match(name)
            if m:
                pid = int(m.group("pid"))
                mine = pid == os.getpid()
                if (mine and os.path.realpath(full) in _INFLIGHT) or (not mine and _writer_alive(pid)):
                    continue
                self._discard(full)
                changed = True
                continue
            try:
                st = os.stat(full)
            except OSError:
                continue
            key = f"orphan:{name}"
            self.regions[key] = StagedRegion(
                key=key, path=full, source="(in the directory, not in the index)",
                bytes=st.st_size, format=os.path.splitext(name)[1].lstrip("."),
                note="adopted at startup", pinned=_SLOT_RE.match(name) is None,
                created=st.st_mtime, accessed=st.st_mtime,
                meta={"orphan": True, "foreign": _SLOT_RE.match(name) is None})
            changed = True
        if changed:
            self._save_index()

    # ---- accounting ----
    @property
    def used(self) -> int:
        self._refresh()                       # another object's regions use this space too
        return sum(r.bytes for r in list(self.regions.values()))

    @property
    def free(self) -> int:
        return max(self.capacity - self.used, 0)

    def disk_free(self) -> int:
        try:
            return shutil.disk_usage(self.path).free
        except OSError:
            return 0

    def _make_room(self, need: int, replacing: Optional[StagedRegion] = None) -> int:
        """Evict least-recently-used unpinned regions until `need` bytes fit — or, if they
        cannot fit however much is evicted, evict nothing and say why. `replacing` is a region
        about to be overwritten by the one being made room for: its bytes are as good as free."""
        if need > self.capacity:
            raise StagingFull(
                f"a {human(need)} region cannot fit a {human(self.capacity)} working area. "
                f"Narrow the window, use a coarser bin or zoom level, project fewer columns, "
                f"or give staging more room.")
        others = [r for r in list(self.regions.values()) if r is not replacing]
        pinned = sum(r.bytes for r in others if r.pinned)
        if pinned + need > self.capacity:
            foreign = sum(r.bytes for r in others if r.pinned and r.meta.get("orphan"))
            extra = (f" ({human(foreign)} of it is files in the directory that staging did not "
                     f"write — move them out, or give staging a directory of its own)"
                     if foreign else "")
            raise StagingFull(
                f"cannot free {human(need)}: {human(pinned)} is pinned and unevictable{extra}. "
                f"Unpin something, or raise the capacity.")
        used = sum(r.bytes for r in others)
        victims = sorted((r for r in others if not r.pinned), key=lambda r: r.accessed or r.created)
        chosen = []
        while used + need > self.capacity and victims:
            v = victims.pop(0)
            used -= v.bytes
            chosen.append(v)
        self._evict_many([v.key for v in chosen])
        return sum(v.bytes for v in chosen)

    # ---- writing ----
    def _slot(self, key: str, suffix: str) -> str:
        h = hashlib.blake2b(key.encode(), digest_size=8).hexdigest()
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)[:60]
        return os.path.join(self.path, f"{safe}-{h}{suffix}")

    def _part_slot(self, key: str, suffix: str) -> str:
        """Where a region is written before it is anything: the slot's name with `.<pid>.<n>.part`
        before the suffix, so a writer that goes by the extension still sees the right one."""
        slot = self._slot(key, suffix)
        stem = slot[:len(slot) - len(suffix)] if suffix else slot
        return f"{stem}.{os.getpid()}.{next(_PARTS)}.part{suffix}"

    def _discard(self, path: str) -> None:
        """Remove a file of ours and anything a writer made beside it by adding its own
        extension (np.save appends '.npy' to a name that lacks it)."""
        base = os.path.basename(path)
        try:
            names = [n for n in os.listdir(os.path.dirname(path)) if n == base or n.startswith(base + ".")]
        except OSError:
            names = [base]
        for n in names:
            try:
                os.remove(os.path.join(os.path.dirname(path), n))
            except OSError:
                pass

    def _remove_file(self, path: str) -> bool:
        """Delete a region's file — but only a file that really is inside the working area."""
        inside = self._inside(path)
        if inside is None:
            return False
        try:
            os.remove(inside)
            return True
        except OSError:
            return False

    def _same(self, old: StagedRegion, source: str, window, sig) -> bool:
        """Is `old` the region a caller asking for `key` again means? With a signature, it must
        match exactly; without one, the source and window must not contradict what is stored."""
        if sig is not None:
            return old.meta.get("sig") == sig
        if source and old.source and old.source != source:
            return False
        if window is not None and old.window is not None and tuple(old.window) != tuple(window):
            return False
        return True

    def put(self, key: str, writer: Callable[[str], dict], suffix: str = ".parquet",
            source: str = "", window: Optional[tuple] = None, note: str = "",
            pin: bool = False, overwrite: bool = False, sig: Optional[dict] = None) -> StagedRegion:
        """Materialize a region by calling `writer(path)`, which returns `{rows, format, ...}`.

        The writer decides the on-disk form; staging only owns the space accounting and the
        eviction policy. `path` is a scratch name with the suffix you asked for; the file is
        moved into place when the writer returns, and removed if it raises. A writer may return
        `suffix` to say the file should be filed under another one. `np.save(path, a)` works as
        it stands (the `.npy` it adds is taken as the file). Any other writer that returns without
        creating the file it was given is an error, not an empty region.

        Asking again for a key that is staged returns the staged region only if it is the same
        region: with `sig` (see `extract`) the signatures must be equal, without it `source`
        and `window` must not contradict what was stored. Otherwise it is written again, and
        the old one is replaced only once the new one is complete.
        """
        sig = _jsonable(sig)
        self._refresh()
        with self._dirlock:
            old = self.regions.get(key)
            if old is not None and not overwrite and self._same(old, source, window, sig):
                return old.touch()
        # We do not know the size until it is written, so write first and reconcile after —
        # but refuse up front if the filesystem itself cannot take a reasonable attempt.
        if self.disk_free() < 64 * 1024 ** 2:
            raise StagingFull(f"{self.path} has only {human(self.disk_free())} left on the "
                              f"filesystem, whatever the configured capacity says")
        part = self._part_slot(key, suffix)
        _INFLIGHT.add(os.path.realpath(part))
        done = False
        try:
            info = writer(part) or {}
            if not os.path.isfile(part) and os.path.isfile(part + ".npy"):
                # np.save(path, a) writes path + '.npy' unless the name already ends so: its own
                # convention, unambiguous, so take what it wrote rather than lose the region
                os.replace(part + ".npy", part)
                info = {"format": "npy", "suffix": ".npy", **info}
            if not os.path.isfile(part):
                raise RuntimeError(
                    f"the writer for {key!r} did not create the file it was given ({part}); "
                    f"a writer that adds an extension of its own to the name must be handed a "
                    f"file object instead")
            size = os.path.getsize(part)
            final = self._slot(key, info.get("suffix", suffix))
            with self._dirlock:
                self._merge_disk()
                old = self.regions.get(key)
                if size > self.free + (old.bytes if old is not None else 0):
                    self._make_room(size, replacing=old)
                os.replace(part, final)
                for k2, r2 in list(self.regions.items()):      # whatever sat on that name is gone
                    if k2 != key and os.path.realpath(r2.path) == os.path.realpath(final):
                        del self.regions[k2]
                if old is not None and os.path.realpath(old.path) != os.path.realpath(final):
                    self._remove_file(old.path)
                now = time.time()
                meta = {k: v for k, v in info.items() if k not in ("rows", "format", "source", "suffix")}
                if sig is not None:
                    meta["sig"] = sig
                r = StagedRegion(key=key, path=final, source=source or info.get("source", ""),
                                 bytes=size, rows=int(info.get("rows", 0)), window=window,
                                 format=info.get("format", suffix.lstrip(".")), note=note,
                                 pinned=pin or (old is not None and old.pinned),
                                 created=now, accessed=now, meta=meta)
                self.regions[key] = r
                self._save_index(merge=False)
            done = True
            return r
        finally:
            _INFLIGHT.discard(os.path.realpath(part))
            if not done:                       # a writer that failed leaves nothing behind
                self._discard(part)

    def extract(self, key: str, view, window: Optional[tuple] = None,
                columns: Optional[list] = None, genes: Optional[list] = None,
                suffix: str = ".parquet", pin: bool = False, note: str = "",
                overwrite: bool = False) -> StagedRegion:
        """Pull one region out of a vault-resident dataset and land it in the working area.

        `view` is any GEMINGA view — the windowing, bin level and gene selection already set
        on it are what gets extracted. This is the single pass over the slow link.

        The region remembers what it was cut from (source, window, bin, zoom, genes, columns:
        `meta["sig"]`). Asking again for the same key with the same view state is a hit and
        reads nothing; asking with a different one — another bin, another window — replaces the
        region, because the key names a place in the working area, not the data that was last put
        there.
        """
        if window:
            view.select(*window)
        if genes is not None and hasattr(view, "select_genes"):
            view.select_genes(genes)
        win = tuple(view.window)
        src = getattr(view, "path", None) or str(getattr(view, "target", "")) or repr(view)
        if hasattr(view, "gef"):
            src = view.gef.path
        sig = _jsonable({"source": src, "window": list(win), "bin": _bin_of(view),
                         "zoom": getattr(view, "zoom", None), "genes": _names(getattr(view, "genes", None)),
                         "columns": list(columns) if columns else None, "format": suffix})

        self._refresh()
        have = self.regions.get(key)
        if have is not None and not overwrite and self._same(have, src, win, sig):
            return have.touch()

        def _write(dest: str) -> dict:
            import numpy as np
            import pyarrow as pa
            import pyarrow.parquet as pq
            try:
                from .journal import recording as _recording
                with _recording(via="extract", note=f"staging:{key}"):
                    data = view.load()
                if isinstance(data, dict) and "x" in data:                 # Stereo-seq
                    cols = {k: np.asarray(v) for k, v in data.items()
                            if isinstance(v, np.ndarray)}
                    tbl = pa.table(cols)
                elif hasattr(data, "num_rows"):                            # pyarrow Table
                    tbl = data.select(columns) if columns else data
                elif hasattr(data, "tocoo"):                               # scipy sparse
                    coo = data.tocoo()
                    tbl = pa.table({"row": coo.row.astype("int32"),
                                    "col": coo.col.astype("int32"),
                                    "value": coo.data.astype("float32")})
                elif isinstance(data, np.ndarray):                         # image tile
                    with open(dest, "wb") as fh:   # a file object: np.save adds '.npy' to a path
                        np.save(fh, data)
                    return {"rows": int(data.size), "format": "npy", "suffix": ".npy",
                            "shape": list(data.shape), "dtype": str(data.dtype)}
                else:
                    raise TypeError(f"cannot stage a {type(data).__name__}")
                pq.write_table(tbl, dest, compression="zstd")
                return {"rows": tbl.num_rows, "format": "parquet",
                        "columns": tbl.schema.names}
            finally:
                # the rows are on disk now (or the write failed): the view's reservation goes back
                if hasattr(view, "release"):
                    view.release()

        if suffix == ".parquet" and hasattr(view, "load"):
            probe = view.cost()
            if probe.get("fits_whole") is False:
                raise StagingFull(
                    f"the region does not fit the view's own memory budget, so it cannot be "
                    f"extracted in one piece: {probe.get('advice', '')}")
        return self.put(key, _write, suffix=suffix, source=src, window=win,
                        note=note, pin=pin, overwrite=overwrite, sig=sig)

    # ---- reading ----
    def get(self, key: str) -> Optional[StagedRegion]:
        self._refresh()
        r = self.regions.get(key)
        return r.touch() if r else None

    def read(self, key: str):
        """Read a staged region back — local, so this is fast and repeatable."""
        r = self.get(key)
        if r is None:
            raise KeyError(f"no staged region {key!r}; have {list(self.regions)}")
        self._save_index()
        if r.format == "npy":
            import numpy as np
            return np.load(r.path)
        import pyarrow.parquet as pq
        return pq.read_table(r.path)

    def path_of(self, key: str) -> str:
        r = self.get(key)
        if r is None:
            raise KeyError(f"no staged region {key!r}")
        return r.path

    def __contains__(self, key: str) -> bool:
        self._refresh()
        return key in self.regions

    def __iter__(self) -> Iterator[StagedRegion]:
        self._refresh()
        return iter(list(self.regions.values()))

    def __len__(self) -> int:
        self._refresh()
        return len(self.regions)

    # ---- lifecycle ----
    def pin(self, key: str, on: bool = True) -> "Staging":
        with self._dirlock:
            self._merge_disk()
            if key in self.regions:
                self.regions[key].pinned = on
                self._save_index(merge=False)
        return self

    def evict(self, key: str) -> "Staging":
        self._evict_many([key])
        return self

    def _evict_many(self, keys) -> None:
        """Drop regions and delete their files — each only if it really is inside the root —
        with one index write for the lot."""
        with self._dirlock:
            self._merge_disk()
            gone = []
            for k in keys:
                r = self.regions.pop(k, None)
                if r is not None:
                    self._remove_file(r.path)
                    gone.append(k)
            if gone:
                self._save_index(merge=False)

    def clear(self, include_pinned: bool = False) -> "Staging":
        """Evict every unpinned region (every region, with `include_pinned`). Files in the
        directory that staging did not write are not regions to clear, pinned or not."""
        with self._dirlock:
            self._merge_disk()
            self._evict_many([k for k, r in self.regions.items()
                              if (include_pinned or not r.pinned) and not r.meta.get("foreign")])
        return self

    def release(self) -> "Staging":
        """Give the reservation back to the budget. Files on disk survive."""
        if self._lease is not None:
            self._lease.release()
            self._lease = None
        return self

    # ---- reporting ----
    def report(self) -> dict:
        regions = list(self.regions.values())
        return {
            "path": self.path,
            "capacity": self.capacity, "capacity_human": human(self.capacity),
            "used": self.used, "used_human": human(self.used),
            "free": self.free, "free_human": human(self.free),
            "filesystem_free_human": human(self.disk_free()),
            "regions": len(regions),
            "pinned": sum(1 for r in regions if r.pinned),
            "orphans": sum(1 for r in regions if r.meta.get("orphan")),
            "hits": sum(r.hits for r in regions),
        }

    def print_report(self) -> None:
        r = self.report()
        print(f"staging {r['path']}")
        print(f"  {r['used_human']} of {r['capacity_human']} used in {r['regions']} region(s), "
              f"{r['pinned']} pinned  ({r['filesystem_free_human']} left on the filesystem)")
        if not self.regions:
            return
        print(f"  {'key':<26} {'size':>9} {'rows':>11} {'hits':>5}  window")
        for reg in sorted(self.regions.values(), key=lambda x: -(x.accessed or 0)):
            w = ("x".join(str(v) for v in reg.window) if reg.window else "-")[:34]
            print(f"  {('*' if reg.pinned else ' ')}{reg.key:<25} {human(reg.bytes):>9} "
                  f"{reg.rows:>11,} {reg.hits:>5}  {w}")

    def __enter__(self):
        return self

    def __exit__(self, *e):
        self.release()
        return False

    def __repr__(self) -> str:
        return (f"Staging({self.path}, {human(self.used)}/{human(self.capacity)}, "
                f"{len(self.regions)} regions)")


def _bin_of(view):
    """The bin level a view's load() will read, where it has one; None for views without bins."""
    try:
        b = getattr(view, "bin_size", None)
        return int(b) if b is not None else None
    except Exception:
        return None


def _names(x):
    return sorted(str(v) for v in x) if isinstance(x, (list, tuple)) else None


def staging(path: str, capacity: str = "50GB", budget: Optional[Budget] = None,
            **kw) -> Staging:
    """A bounded working area for regions pulled out of slow, oversized storage."""
    return Staging(path, capacity, budget, **kw)
