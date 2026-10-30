"""The slow, huge, read-only side: an external drive or a NAS mount full of datasets.

A vault is scanned once and remembered. Scanning walks the tree, records size and mtime, and —
optionally — profiles each member so you can ask "which of these is single-nucleus" without
opening anything by hand. The manifest is a JSON file written *beside the data* by default, so
the next person to mount that drive inherits the work.

What a vault deliberately does not do is copy. Members are opened in place through the same
windowed readers as everything else, and only extracted regions land in staging. On a NAS that
matters: the difference between one pass over the slow link per region and one pass per look.

Nor does it write beside data it was told, or can see, is read-only. On a read-only mount — or
with `readonly=True`, for a drive you simply do not want touched — the manifest goes to
`~/.geminga/`, keyed by the vault's absolute path, and the sidecar indexes that opening a FASTQ,
GEM or GEF would leave next to the file go to `~/.geminga/gefidx/` instead. A manifest or sidecar
that is already beside the data is still read.

A DICOM series is a directory of one file per slice; it is one member (the directory), not one
per slice.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Iterator, Optional

from ._core import human

__all__ = ["Vault", "VaultMember", "vault", "AmbiguousMember"]

_KNOWN = (".gef", ".gem", ".h5ad", ".parquet", ".arrow", ".feather", ".fastq", ".fq",
          ".fasta", ".fa", ".fna", ".tif", ".tiff", ".zarr", ".bam", ".cram",
          ".nii", ".nii.gz", ".h5", ".mtx", ".mtx.gz",
          ".fastq.gz", ".fq.gz", ".fasta.gz", ".fa.gz", ".gem.gz")
# A DICOM series is one file per slice, so the directory is the member. These are the suffixes
# g.view() and profile() recognise a DICOM directory by.
_DICOM = (".dcm", ".ima")
_SEQ = (".fq", ".fastq", ".fa", ".fasta", ".fna", ".ffn", ".faa", ".frn")


class AmbiguousMember(KeyError, ValueError):
    """A bare file name that more than one member of the vault answers to."""

    def __init__(self, name: str, candidates):
        self.name, self.candidates = name, list(candidates)
        super().__init__(
            f"{name!r} matches {len(self.candidates)} members: {', '.join(self.candidates)}. "
            f"Name it by its path relative to the vault root, e.g. {self.candidates[0]!r}.")

    def __str__(self) -> str:                 # KeyError would show the message in quotes
        return str(self.args[0])


def _writable(directory: str) -> bool:
    """Can a sidecar be written in `directory`? `access` is False on a read-only mount even for
    root. A false 'yes' (an NFS export that squashes root) costs only a sidecar that cannot be
    saved, which every writer of one already survives."""
    return os.access(directory, os.W_OK | os.X_OK)


def _geminga_home() -> str:
    return os.path.expanduser("~/.geminga")


def _path_key(path: str) -> str:
    """The key gefindex gives a file's private sidecar directory under ~/.geminga/gefidx."""
    return hashlib.blake2b(os.path.abspath(path).encode(), digest_size=8).hexdigest()


@dataclass
class VaultMember:
    path: str
    bytes: int
    mtime: float
    format: str = ""
    assay: str = ""
    assay_confidence: float = 0.0
    molecule: str = ""
    modality: str = ""
    confident: bool = False
    profiled: bool = False
    note: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    def __repr__(self) -> str:
        a = f"{self.assay} {self.assay_confidence:.0%}" if self.assay else "unprofiled"
        return f"VaultMember({self.name}, {human(self.bytes)}, {a})"


class Vault:
    """A read-only store of datasets too big to move.

        v = g.vault("/mnt/nas/stereoseq", capacity_note="18 TB array")
        v.scan(profile=True)          # once; cached in a manifest beside the data
        v.print_table()
        v.find("stereo")
        view = v.open("SS200000135TL_D1.gef", budget=b)

    `readonly=True` keeps the manifest and every sidecar off the drive even where it could be
    written to; a drive that really is read-only gets the same treatment without being asked.
    """

    MANIFEST = ".geminga-vault.json"

    def __init__(self, root: str, manifest: Optional[str] = None, note: str = "",
                 readonly: bool = False):
        self.root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(self.root):
            raise NotADirectoryError(f"{self.root} is not a directory")
        self.note = note
        self.readonly = bool(readonly)
        self._explicit = manifest is not None
        self._primary = manifest or os.path.join(self.root, self.MANIFEST)
        self.manifest_path = self._home_manifest()
        self.members: dict[str, VaultMember] = {}
        self.scanned_at = 0.0
        self._load()

    # ---- manifest ----
    def _fallback_manifest(self) -> str:
        """Where the manifest goes when it cannot go beside the data. Keyed by a hash of the
        absolute root: two drives that both have a folder called `data` are two vaults."""
        h = hashlib.blake2b(self.root.encode(), digest_size=8).hexdigest()
        return os.path.join(_geminga_home(), f"vault-{os.path.basename(self.root) or 'root'}-{h}.json")

    def _legacy_manifest(self) -> str:
        """What older versions called it: the folder name alone, shared by every vault with one."""
        return os.path.join(_geminga_home(), "vault-" + os.path.basename(self.root) + ".json")

    def _home_manifest(self) -> str:
        """Where a new manifest is tried first."""
        return self._fallback_manifest() if self.readonly and not self._explicit else self._primary

    def _load(self):
        # The manifest beside the data travels with the drive, so it is taken whatever path the
        # drive was mounted at. One in ~/.geminga is only ever this vault's own: it must name
        # this root, which keeps a same-named folder elsewhere from inheriting it.
        for p, travels in ((self._primary, True), (self._fallback_manifest(), False),
                           (self._legacy_manifest(), False)):
            if not os.path.exists(p):
                continue
            try:
                with open(p, encoding="utf-8") as fh:
                    d = json.load(fh)
                if not travels and d.get("root") != self.root:
                    continue
                members = {k: VaultMember(**v) for k, v in d.get("members", {}).items()}
            except Exception:
                continue
            self.scanned_at = d.get("scanned_at", 0.0)
            self.members = members
            self.manifest_path = self._home_manifest() if p == self._legacy_manifest() else p
            return

    def save(self) -> str:
        """Write the manifest beside the data, or to ~/.geminga if the mount is read-only."""
        payload = {"root": self.root, "note": self.note, "scanned_at": self.scanned_at,
                   "members": {k: asdict(v) for k, v in self.members.items()}}
        targets = [self._fallback_manifest()] if self.readonly and not self._explicit else [
            self.manifest_path, self._fallback_manifest()]
        for p in dict.fromkeys(targets):
            try:
                os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
                with open(p, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, default=str)
                self.manifest_path = p
                return p
            except OSError:
                continue
        return ""

    # ---- scanning ----
    def scan(self, profile: bool = False, rescan: bool = False, max_files: int = 100_000,
             fetch_page: bool = False, verbose: bool = False) -> "Vault":
        """Walk the tree. With `profile=True`, work out what each member is.

        Profiling opens each file's header — cheap even over a slow mount, because none of the
        readers need more than a few kilobytes to identify a container. Content scans are the
        expensive part and are skipped for members above `content_limit`.
        """
        from .profile import profile as _profile
        seen, n = {}, 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            # a .zarr directory is one dataset, not a folder to descend into
            for d in list(dirnames):
                if d.endswith(".zarr"):
                    dirnames.remove(d)
                    filenames.append(d)
            cands = []                       # (path, key, bytes, mtime, meta)
            slices = [fn for fn in filenames if fn.lower().endswith(_DICOM) and not fn.startswith(".")]
            if slices:                       # a DICOM series: the directory is the member
                sizes = []
                for fn in slices:
                    try:
                        st = os.stat(os.path.join(dirpath, fn))
                    except OSError:
                        continue
                    sizes.append((st.st_size, st.st_mtime))
                if sizes:
                    cands.append((dirpath, os.path.relpath(dirpath, self.root),
                                  sum(b for b, _ in sizes), max(t for _, t in sizes),
                                  {"n_files": len(sizes)}))
            for fn in filenames:
                low = fn.lower()
                if fn.startswith(".") or fn == self.MANIFEST or low.endswith(_DICOM):
                    continue
                if not any(low.endswith(e) for e in _KNOWN):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                cands.append((full, os.path.relpath(full, self.root), _size_of(full, st),
                              st.st_mtime, {}))
            for full, rel, nbytes, mtime, meta in cands:
                prev = self.members.get(rel)
                m = VaultMember(path=rel, bytes=nbytes, mtime=mtime, meta=dict(meta))
                if (prev and not rescan and abs(prev.mtime - m.mtime) < 1
                        and prev.bytes == m.bytes and prev.profiled):
                    seen[rel] = prev                     # unchanged: keep what we know
                else:
                    if profile:
                        try:
                            p = _profile(full, scan_content=m.bytes < (2 << 30),
                                         fetch_page=fetch_page)
                            m.format = p.format or ""
                            m.molecule = p.molecule or ""
                            m.modality = p.modality
                            m.confident = p.confident
                            if p.best:
                                m.assay = p.best.label
                                m.assay_confidence = float(p.best.probability)
                            m.meta.update({k: v for k, v in p.metadata.items()
                                           if k in ("bin_levels", "n_genes", "n_obs", "extent",
                                                    "chip", "read_length_mean", "gc_fraction")})
                            if not p.confident and p.best:
                                m.meta["unsettled_because"] = p.best.distinguish
                            m.profiled = True
                        except Exception as e:
                            m.note = f"profile failed: {type(e).__name__}"
                    seen[rel] = m
                    if verbose:
                        print(f"  {rel}  {human(m.bytes)}  {m.assay or '-'}")
                n += 1
                if n >= max_files:
                    break
            if n >= max_files:
                break
        self.members = seen
        self.scanned_at = time.time()
        self.save()
        return self

    # ---- querying ----
    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self) -> Iterator[VaultMember]:
        return iter(self.members.values())

    def __contains__(self, rel: str) -> bool:
        rel = os.fspath(rel)
        return (rel in self.members or self._rel(rel) in self.members
                or any(m.name == rel for m in self.members.values()))

    def _rel(self, p: str) -> Optional[str]:
        """`p` as a key of `members`: normalised, and relative to the root if it was absolute."""
        q = os.path.normpath(p)
        if os.path.isabs(q):
            q = os.path.relpath(q, self.root)
            if q == ".." or q.startswith(".." + os.sep):
                return None
        return q

    def get(self, rel: str) -> Optional[VaultMember]:
        """A member by its path relative to the root (or an absolute path into the vault), or by
        bare file name — if exactly one member has that name. A name several members share raises
        `AmbiguousMember` (a KeyError and a ValueError) listing them, rather than quietly
        choosing one."""
        rel = os.fspath(rel)
        if rel in self.members:
            return self.members[rel]
        key = self._rel(rel)
        if key in self.members:
            return self.members[key]
        same = [m for m in self.members.values() if m.name == rel]
        if len(same) > 1:
            raise AmbiguousMember(rel, sorted(m.path for m in same))
        return same[0] if same else None

    def abspath(self, rel: str) -> str:
        m = self.get(rel)
        if m is None:
            raise KeyError(f"{rel!r} is not in this vault")
        return os.path.normpath(os.path.join(self.root, m.path))

    def find(self, needle: str = "", assay: str = "", fmt: str = "",
             min_bytes: int = 0, unsettled: Optional[bool] = None, exact: bool = False) -> list:
        """Members matching all the given filters, largest first.

        `needle` is a substring of the path, format, assay, molecule or note — so "GSM123" also
        finds GSM1234567. With `exact=True` it must be a whole token of the *path*: bounded on
        both sides by anything that is not a letter or digit (a separator, `_`, `.`, `-`, `/`, or
        either end), which is what matching an accession needs. (A needle that itself begins or
        ends with a separator, such as ".gef", is bounded there already.)
        """
        n = needle.lower()
        token = None
        if n and exact:       # a boundary is only asked for where the needle itself ends in a letter or digit
            token = re.compile((r"(?<![a-z0-9])" if n[0].isalnum() else "") + re.escape(n)
                               + (r"(?![a-z0-9])" if n[-1].isalnum() else ""))
        out = []
        for m in self.members.values():
            if token is not None:
                if not token.search(m.path.lower()):
                    continue
            elif n and n not in f"{m.path} {m.format} {m.assay} {m.molecule} {m.note}".lower():
                continue
            if assay and assay.lower() not in m.assay.lower():
                continue
            if fmt and fmt.lower() not in m.format.lower():
                continue
            if m.bytes < min_bytes:
                continue
            if unsettled is not None and (not m.confident) != unsettled:
                continue
            out.append(m)
        return sorted(out, key=lambda m: -m.bytes)

    @property
    def total_bytes(self) -> int:
        return sum(m.bytes for m in self.members.values())

    def oversized(self, ram: str) -> list:
        """Members larger than a given memory figure — the ones that need this workflow."""
        from ._core import parse_size
        cap = parse_size(ram)
        return [m for m in self.members.values() if m.bytes > cap]

    # ---- opening ----
    def open(self, rel: str, budget=None, **kw):
        """Open a member in place. Nothing is copied — and, where the drive is read-only (or the
        vault was told it is), nothing is written to it either: the sidecar index that opening a
        FASTQ, GEM or GEF would leave beside the file goes under ~/.geminga/gefidx instead."""
        return self._open_path(self.abspath(rel), budget, **kw)

    def _open_path(self, path: str, budget=None, **kw):
        import geminga as g
        low = path.lower()
        if low.endswith((".mtx", ".mtx.gz")):
            from .tenx import MtxTriple
            return MtxTriple(path, budget=budget, **kw)
        if not self.readonly and _writable(os.path.dirname(path)):
            return g.view(path, budget=budget, **kw)
        if low.endswith(_SEQ) or (low.endswith(".gz") and any(low[:-3].endswith(e) for e in _SEQ)):
            kw.setdefault("sidecar", self._sidecar(path))
        elif low.endswith((".gem", ".gem.gz")):
            from .stereoseq import GemView, index_gem
            return GemView(index_gem(path, kw.pop("grid", None), sidecar=self._sidecar(path)),
                           budget, path)
        elif low.endswith(".gef"):
            # A GEF's spatial index is found without writing (beside the file, else under
            # ~/.geminga/gefidx) and built beside the file only when that directory may be
            # written: gefindex decides by os.access. A vault declared read-only is stricter —
            # an explicit build must land off the drive even where the bits would allow it.
            view = g.view(path, budget=budget, **kw)
            gf = getattr(view, "gef", None)
            if gf is not None and not getattr(gf, "remote", False):
                from .gefindex import _candidates
                private = _candidates(path)[-1]
                build = gf.build_spatial_index

                def build_privately(levels=None, **bkw):
                    bkw.setdefault("directory", private)
                    os.makedirs(private, exist_ok=True)
                    return build(levels, **bkw)
                gf.build_spatial_index = build_privately
            return view
        return g.view(path, budget=budget, **kw)

    def _sidecar(self, path: str) -> str:
        """Where the `.gemidx.json` of `path` lives when it cannot be written beside it. One that
        is already beside the data and still fresh is used in place — reading needs no write
        access, and a drive often arrives with its indexes built. Otherwise it is a private copy
        under ~/.geminga/gefidx, keyed the way gefindex keys its own."""
        beside = path + ".gemidx.json"
        try:
            if os.path.getmtime(beside) >= os.path.getmtime(path):
                return beside
        except OSError:
            pass
        d = os.path.join(_geminga_home(), "gefidx", _path_key(path))
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, os.path.basename(beside))

    def add_to(self, ws, rel: str, name: Optional[str] = None, ram: str = "1GB", **kw):
        """Put a member into a workspace as its own budget-partitioned cell."""
        m = self.get(rel)
        if m is None:
            raise KeyError(f"{rel!r} is not in this vault")
        kw.setdefault("opener", lambda target, budget=None, **k: self._open_path(target, budget, **k))
        return ws.add(name or os.path.splitext(m.name)[0], self.abspath(rel),
                      ram=ram, note=m.assay or m.format, **kw)

    # ---- reporting ----
    def report(self) -> dict:
        by_assay: dict[str, int] = {}
        for m in self.members.values():
            by_assay[m.assay or "unprofiled"] = by_assay.get(m.assay or "unprofiled", 0) + 1
        return {"root": self.root, "members": len(self.members),
                "total_bytes": self.total_bytes, "total_human": human(self.total_bytes),
                "scanned_at": self.scanned_at, "by_assay": by_assay,
                "unsettled": sum(1 for m in self.members.values()
                                 if m.profiled and not m.confident)}

    def print_table(self, needle: str = "", limit: int = 30) -> None:
        r = self.report()
        age = ("never" if not r["scanned_at"]
               else f"{(time.time() - r['scanned_at']) / 3600:.1f} h ago")
        print(f"vault {r['root']}  —  {r['members']} datasets, {r['total_human']}, "
              f"scanned {age}")
        rows = self.find(needle)
        if not rows:
            print("  (nothing matched)" if needle else "  (empty — run .scan(profile=True))")
            return
        print(f"  {'size':>9} {'format':<18} {'assay':<24} path")
        for m in rows[:limit]:
            flag = "" if m.confident or not m.profiled else " ?"
            print(f"  {human(m.bytes):>9} {(m.format or '-'):<18} "
                  f"{((m.assay or '-') + flag):<24} {m.path}")
        if len(rows) > limit:
            print(f"  ... {len(rows) - limit} more")
        if r["unsettled"]:
            print(f"  {r['unsettled']} member(s) profiled but not settled — "
                  f"vault.find(unsettled=True)")

    def __repr__(self) -> str:
        return f"Vault({self.root}, {len(self.members)} datasets, {human(self.total_bytes)})"


def _size_of(path: str, st) -> int:
    if os.path.isdir(path):                             # .zarr and friends
        total = 0
        for dp, _, fs in os.walk(path):
            for f in fs:
                try:
                    total += os.path.getsize(os.path.join(dp, f))
                except OSError:
                    pass
        return total
    return st.st_size


def vault(root: str, **kw) -> Vault:
    """Index a slow, oversized store of datasets — an external drive or a NAS mount."""
    return Vault(root, **kw)
