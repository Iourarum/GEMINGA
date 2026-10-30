"""A spatial index for GEF, so a window costs what is in it.

A GEF stores each bin level's `expression` array in *gene* order — that is what makes a
single-gene query one contiguous slice. The price is that a *spatial* query has no order to
exploit: rows for a small window are scattered through the whole array, so a window has to read
the entire level (in leased blocks) and filter it. `cost()` reports both the rows kept and the
bytes read, so the difference is visible.

The fix is a second copy of each level in spatial order, and the order is the whole trick.
Rows are sorted by the **Z-order (Morton) code** of their grid cell. Z-order keeps nearby
cells nearby in the file, so any fixed-size run of rows covers a compact square-ish patch of
the chip. Written as Parquet with fixed-size row groups, each row group's footer statistics on
`x` and `y` then describe a small box, and every row group whose box misses the window is ruled
out before a byte of it is read. The same statistics estimate how many rows a window holds.

The sidecar is built out of core and inside a budget: the level is read in leased blocks, each
block sorted into a run file, and the runs merged k ways (in more than one pass when there are
many), so building a level larger than memory never needs the level in memory.

**Where the sidecar lives.** Beside the GEF (`chip.gef.gemidx/`), or under
`~/.geminga/gefidx/<hash>/` when the GEF's directory cannot be written (a read-only mount, a
NAS share). Looking for a sidecar creates nothing: an existing, valid one is used wherever it
is — beside a read-only GEF included. It records the GEF's size and mtime and is ignored if
either changes. A sidecar is a soft cache: a level whose file is missing or unreadable is read
from the GEF instead, and rebuilding some levels keeps the others.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from typing import Optional

import numpy as np

from ._core import Budget, human

__all__ = ["build_spatial_index", "SpatialIndex", "morton2"]

VERSION = 2          # 2: true 2-D Morton codes

#: Arrow bytes of one sidecar row as it is streamed: x, y, count, gene as uint32.
BATCH_ROW_BYTES = 16
_RUN_ROW_BYTES = 48          # a row while a run is built: record, gene id, key, order, sorted copy
_MERGE_ROW_BYTES = 96        # a row in a merge buffer, its candidate copy, order and output
_MIN_MERGE_ROWS = 256        # the smallest per-run merge buffer worth a read
_MERGE_TARGET_ROWS = 1024    # merge few enough runs at once that buffers hold this many
_MAX_FAN = 64
_MASKS = ((16, 0x0000FFFF0000FFFF), (8, 0x00FF00FF00FF00FF), (4, 0x0F0F0F0F0F0F0F0F),
          (2, 0x3333333333333333), (1, 0x5555555555555555))


def _part1by1(v: np.ndarray) -> np.ndarray:
    """Spread the low 32 bits of v so there is one zero between each: ...b2 0 b1 0 b0.

    An earlier version used the *3-D* masks, which leave two zeros between bits. Because
    inserting constant zeros at fixed positions preserves comparison order, the row order —
    and so every pruning result — was unaffected; the codes themselves were simply not 2-D
    Morton codes, which a function named `morton2` has to return. Done in place with one
    temporary, so a block costs 16 bytes a row here, not five fresh arrays.
    """
    v = np.asarray(v).astype(np.uint64)
    v &= np.uint64(0x00000000FFFFFFFF)
    t = np.empty_like(v)
    for shift, mask in _MASKS:
        np.left_shift(v, np.uint64(shift), out=t)
        v |= t
        v &= np.uint64(mask)
    return v


def morton2(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Interleave two coordinate grids into one Z-order key. Handles up to 2**32 per axis —
    a 13 cm chip at bin1 is ~260,000 units."""
    key = _part1by1(ix)
    t = _part1by1(iy)
    np.left_shift(t, np.uint64(1), out=t)
    key |= t
    return key


def _is_remote(p: str) -> bool:
    return str(p).startswith(("ftp://", "http://", "https://"))


def _home_dir(key: str) -> str:
    h = hashlib.blake2b(key.encode(), digest_size=8).hexdigest()
    return os.path.join(os.path.expanduser("~/.geminga/gefidx"), h)


def _candidates(gef_path: str) -> list:
    """Where a sidecar for `gef_path` may be, in the order a lookup tries them."""
    if _is_remote(gef_path):
        return [_home_dir(gef_path)]                    # a remote GEF's sidecar can only be local
    return [gef_path + ".gemidx", _home_dir(os.path.abspath(gef_path))]


def _writable(d: str) -> bool:
    """May a sidecar be written at `d`? Asked of the directory (or the one that would hold it),
    without creating or touching anything."""
    if os.path.isdir(d):
        return os.access(d, os.W_OK | os.X_OK)
    parent = os.path.dirname(os.path.abspath(d)) or "."
    return os.path.isdir(parent) and os.access(parent, os.W_OK | os.X_OK)


def _sidecar_dir(gef_path: str, write: bool = False) -> str:
    """The sidecar directory for `gef_path`.

    Looking up (`write=False`) creates nothing: the directory beside the GEF if it holds an
    `index.json`, else the home fallback if that does, else the beside path (absent). For
    writing, beside the GEF when that may be written (`os.access`), else the home fallback; the
    chosen directory is created then.
    """
    cands = _candidates(gef_path)
    if not write:
        for d in cands:
            if os.path.exists(os.path.join(d, "index.json")):
                return d
        return cands[0]
    d = cands[0] if (len(cands) > 1 and _writable(cands[0])) else cands[-1]
    os.makedirs(d, exist_ok=True)
    return d


def _fingerprint(gef_path: str) -> dict:
    st = os.stat(gef_path)
    return {"size": st.st_size, "mtime": int(st.st_mtime), "version": VERSION}


def _read_meta(d: str) -> dict:
    p = os.path.join(d, "index.json")
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


class SpatialIndex:
    """The sidecar for one GEF: which levels are indexed, and a pruning reader for each."""

    def __init__(self, gef_path: str, directory: Optional[str] = None,
                 fingerprint: Optional[dict] = None):
        self.gef_path = os.fspath(gef_path) if not isinstance(gef_path, str) else gef_path
        self._fp = fingerprint
        self._ds: dict = {}
        self._stats: dict = {}
        self._per_row: dict = {}
        if directory is not None:
            self.dir, self.meta = directory, _read_meta(directory)
        else:
            self.dir, self.meta = self._find()
        self.meta_path = os.path.join(self.dir, "index.json")

    def _fingerprint(self) -> Optional[dict]:
        if self._fp is not None:
            return self._fp
        try:
            return _fingerprint(self.gef_path)
        except OSError:
            return None

    def _find(self):
        """The first candidate whose sidecar matches the GEF; else the first that has one."""
        fp = self._fingerprint()
        first = None
        for d in _candidates(self.gef_path):
            meta = _read_meta(d)
            if not meta:
                continue
            if meta.get("gef") == fp:
                return d, meta
            if first is None:
                first = (d, meta)
        return first or (_sidecar_dir(self.gef_path), {})

    @property
    def valid(self) -> bool:
        return bool(self.meta) and self.meta.get("gef") == self._fingerprint()

    @property
    def levels(self) -> list:
        """Levels the sidecar lists *and* still has a file for."""
        if not self.valid:
            return []
        return sorted(int(b) for b in self.meta.get("levels", {})
                      if os.path.exists(self.path(int(b))))

    def has(self, bin_size: int) -> bool:
        """Is level `bin_size` usable — listed, present, and readable?"""
        if bin_size not in self.levels:
            return False
        try:
            self._rg_stats(bin_size)
            self.dataset(bin_size)
            return True
        except Exception:
            return False

    def path(self, bin_size: int) -> str:
        return os.path.join(self.dir, f"bin{bin_size}.parquet")

    def dataset(self, bin_size: int):
        if bin_size not in self._ds:
            from ._core import Dataset
            self._ds[bin_size] = Dataset.from_paths([self.path(bin_size)])
        return self._ds[bin_size]

    def _rg_stats(self, b: int) -> dict:
        """Per row group: rows, compressed bytes and the x / y range — from the footer only."""
        if b not in self._stats:
            import pyarrow.parquet as pq
            md = pq.read_metadata(self.path(b))
            names = md.schema.names
            jx, jy = names.index("x"), names.index("y")
            n = md.num_row_groups
            out = {k: np.zeros(n, np.float64) for k in ("xmin", "xmax", "ymin", "ymax")}
            rows = np.zeros(n, np.int64)
            comp = np.zeros(n, np.int64)
            for i in range(n):
                rg = md.row_group(i)
                rows[i] = rg.num_rows
                comp[i] = sum(rg.column(j).total_compressed_size for j in range(rg.num_columns))
                for j, (lo, hi) in ((jx, ("xmin", "xmax")), (jy, ("ymin", "ymax"))):
                    st = rg.column(j).statistics
                    if st is not None and st.has_min_max:
                        out[lo][i], out[hi][i] = float(st.min), float(st.max)
                    else:                                  # no statistics: it cannot be pruned
                        out[lo][i], out[hi][i] = -np.inf, np.inf
            out["rows"], out["comp"] = rows, comp
            self._stats[b] = out
        return self._stats[b]

    def _survivors(self, window: tuple, b: int):
        """Row groups whose box can hold a bin that overlaps the window."""
        st = self._rg_stats(b)
        x0, x1, y0, y1 = (int(v) for v in window)
        lx, ly = x0 - b + 1, y0 - b + 1                     # a bin origin > x0 - b overlaps
        keep = ((st["xmax"] >= lx) & (st["xmin"] <= x1 - 1)
                & (st["ymax"] >= ly) & (st["ymin"] <= y1 - 1))
        return st, keep

    def plan(self, window: tuple, bin_size: int) -> dict:
        """Which row groups a window needs, and what reading them costs — from footers."""
        st, keep = self._survivors(window, bin_size)
        idx = np.flatnonzero(keep)
        comp = int(st["comp"][idx].sum())
        total = int(st["comp"].sum())
        return {"row_groups": idx.tolist(), "row_groups_total": int(len(keep)),
                "rows_scanned": int(st["rows"][idx].sum()),
                "read_bytes": comp, "read_human": human(comp), "level_bytes": total,
                "selectivity": comp / max(total, 1)}

    def estimate_rows(self, window: tuple, bin_size: int) -> float:
        """Rows of the window, from row-group statistics: every surviving row group's rows,
        those only partly inside by the share of their bounding box the window's bins cover."""
        b = bin_size
        st, keep = self._survivors(window, b)
        if not keep.any():
            return 0.0
        x0, x1, y0, y1 = (int(v) for v in window)
        ex0, ex1 = (x0 // b) * b, -(-x1 // b) * b
        ey0, ey1 = (y0 // b) * b, -(-y1 // b) * b
        bx0, bx1 = st["xmin"][keep], st["xmax"][keep] + b
        by0, by1 = st["ymin"][keep], st["ymax"][keep] + b
        fx = np.clip(np.minimum(bx1, ex1) - np.maximum(bx0, ex0), 0, None) / np.maximum(bx1 - bx0, 1)
        fy = np.clip(np.minimum(by1, ey1) - np.maximum(by0, ey0), 0, None) / np.maximum(by1 - by0, 1)
        f = np.nan_to_num(np.clip(fx, 0, 1) * np.clip(fy, 0, 1), nan=1.0)
        return float((st["rows"][keep] * f).sum())

    def _reader_chunk(self, b: int, cols: list, rows: int) -> int:
        """The `chunk_bytes` that makes the Parquet reader cut batches of about `rows` rows. It
        sizes batches from the footer's encoded bytes per row, which for these small integer
        columns is far below the 4 bytes a value takes once decoded."""
        key = (b, tuple(cols))
        if key not in self._per_row:
            rpm = int(self.dataset(b).plan(str(1 << 20), list(cols))["rows_per_chunk"])
            self._per_row[key] = (1 << 20) / max(rpm, 1)
        return max(1, int(rows * self._per_row[key]))

    def iter_window(self, window: tuple, bin_size: int, *, gene_ids: bool = False,
                    budget: Optional[Budget] = None, tier: str = "ram", chunk_bytes=None):
        """The window's rows, piece by piece: the surviving row groups are streamed through the
        dataset with `budget` attached (each batch leased by the reader until the next is asked
        for), filtered exactly to the window as they come; each kept piece is leased while it is
        handed over."""
        from .stereoseq import RESIDENT_BYTES_PER_OBS, _i32
        from .view import _Accounted, _Hold, _nbytes
        b = int(bin_size)
        p = self.plan(window, b)
        if not p["row_groups"]:
            return
        x0, x1, y0, y1 = (int(v) for v in window)
        lx, ly = x0 - b + 1, y0 - b + 1
        out_row = RESIDENT_BYTES_PER_OBS + (4 if gene_ids else 0)
        per_row = BATCH_ROW_BYTES + 2 + out_row
        cap = int(chunk_bytes) if chunk_bytes else 16 << 20
        ds = self.dataset(b)
        ds.set_budget(budget)
        cols = ["x", "y", "count"] + (["gene"] if gene_ids else [])
        rust = self._reader_chunk(b, cols, max(cap // per_row, 1))
        for batch in ds.stream(columns=cols, row_groups=p["row_groups"], chunk_bytes=str(rust)):
            n = batch.num_rows
            with _Hold(budget, tier, n * (per_row - BATCH_ROW_BYTES)) as hold:
                xs = batch.column("x").to_numpy()
                ys = batch.column("y").to_numpy()
                m = xs >= max(lx, 0)
                m &= xs < x1
                m &= ys >= max(ly, 0)
                m &= ys < y1
                k = int(np.count_nonzero(m))
                part = None
                if k:
                    part = {"x": _i32(xs[m]), "y": _i32(ys[m]),
                            "count": batch.column("count").to_numpy()[m].astype(np.int32)}
                    if gene_ids:
                        part["gene_id"] = _i32(batch.column("gene").to_numpy()[m])
                xs = ys = m = None
                batch = None
                if part is not None:
                    hold.resize(_nbytes(part))
                    yield _Accounted(part)
                    part = None

    def read(self, window: tuple, bin_size: int, genes_lookup: Optional[np.ndarray] = None,
             stride: int = 1, with_gene_ids: bool = False, *, budget: Optional[Budget] = None,
             tier: str = "ram") -> dict:
        """Observations in the window: prune on x and y, stream survivors, filter exactly."""
        from .stereoseq import _assemble, _ReadCtx
        ctx = _ReadCtx(budget, tier)
        pieces, seen = [], 0
        for item in self.iter_window(window, bin_size, gene_ids=with_gene_ids, budget=budget,
                                     tier=tier, chunk_bytes=ctx.cap()):
            part = item.data
            if stride > 1:
                first = (-seen) % stride
                seen += len(part["x"])
                part = {k: np.ascontiguousarray(v[first::stride]) for k, v in part.items()}
            pieces.append(part)
        empty = {"x": np.zeros(0, np.int32), "y": np.zeros(0, np.int32),
                 "count": np.zeros(0, np.int32)}
        if with_gene_ids:
            empty["gene_id"] = np.zeros(0, np.int32)
        out = _assemble(pieces, ctx, empty)
        out["bin_size"] = bin_size
        out["_plan"] = self.plan(window, bin_size)
        return out

    def __repr__(self) -> str:
        return (f"SpatialIndex({os.path.basename(self.gef_path)}, levels {self.levels}, "
                f"{'valid' if self.valid else 'stale or missing'})")


# ---------------------------------------------------------------------------
# Building: sorted runs, then a k-way merge, all inside the budget
# ---------------------------------------------------------------------------

_COLS = ("x", "y", "count", "gene")
_RUN_COLS = (("z", np.uint64), ("x", np.uint32), ("y", np.uint32), ("count", np.uint32),
             ("gene", np.uint32))


def _schema():
    import pyarrow as pa
    return pa.schema([pa.field(c, pa.uint32(), False) for c in _COLS])


class _Run:
    """A sorted run on disk, raw: its `n` rows' columns one after another (z as uint64, then
    x, y, count, gene as uint32). Written once, read front to back a buffer at a time."""

    def __init__(self, path: str, n: int):
        self.path, self.n, self.pos, self.fh = path, int(n), 0, None
        self.off, o = {}, 0
        for c, dt in _RUN_COLS:
            self.off[c] = o
            o += self.n * np.dtype(dt).itemsize

    def append(self, cols: dict):
        """Write the next rows (a run is preallocated: `n` is known before it is written)."""
        if self.fh is None:
            self.fh = open(self.path, "wb")
            self.fh.truncate(sum(self.n * np.dtype(dt).itemsize for _, dt in _RUN_COLS))
        k = len(cols["z"])
        for c, dt in _RUN_COLS:
            self.fh.seek(self.off[c] + self.pos * np.dtype(dt).itemsize)
            np.ascontiguousarray(cols[c], dtype=dt).tofile(self.fh)
        self.pos += k

    def finish(self) -> "_Run":
        if self.fh is not None:
            self.fh.close()
        self.fh, self.pos = None, 0
        return self

    def read(self, k: int) -> Optional[dict]:
        if self.pos >= self.n:
            return None
        if self.fh is None:
            self.fh = open(self.path, "rb")
        k = min(int(k), self.n - self.pos)
        out = {}
        for c, dt in _RUN_COLS:
            self.fh.seek(self.off[c] + self.pos * np.dtype(dt).itemsize)
            out[c] = np.fromfile(self.fh, dtype=dt, count=k)
        self.pos += k
        return out

    def close(self, remove: bool = False):
        if self.fh is not None:
            self.fh.close()
            self.fh = None
        if remove:
            try:
                os.remove(self.path)
            except OSError:
                pass


def _sorted_block(exp, gf, b: int, a: int, c: int, has_count: bool) -> dict:
    """Rows [a, c) of level `b` sorted by Z-order, as uint64 `z` and uint32 columns."""
    rec = exp[a:c]
    key = morton2(rec["x"] // np.uint32(b), rec["y"] // np.uint32(b))
    order = np.argsort(key, kind="stable")
    out = {"z": key[order]}
    key = None
    out["x"] = rec["x"].astype(np.uint32)[order]
    out["y"] = rec["y"].astype(np.uint32)[order]
    out["count"] = (rec["count"].astype(np.uint32)[order] if has_count
                    else np.ones(c - a, np.uint32))
    rec = None
    out["gene"] = gf._gene_ids(b, a, c).view(np.uint32)[order]
    return out


class _Out:
    """The sidecar's Parquet file: rows in row groups of exactly `group` rows (the last may be
    short), zstd, with the min/max statistics pruning depends on."""

    def __init__(self, path: str, group: int):
        import pyarrow.parquet as pq
        self.group = group
        # Plain encoding: with dictionary pages and zstd, pyarrow keeps the numpy buffers it was
        # handed alive until the process ends — the whole level, outside any lease.
        self.w = pq.ParquetWriter(path, _schema(), compression="zstd", write_statistics=True,
                                  use_dictionary=False)
        self.pending, self.n = [], 0

    def append(self, cols: dict):
        self.pending.append(cols)
        self.n += len(cols["z"])
        while self.n >= self.group:
            self._flush(self.group)

    def _flush(self, k: int):
        import pyarrow as pa
        take, rest, got = [], [], 0
        for p in self.pending:
            if got >= k:
                rest.append(p)
                continue
            m = min(k - got, len(p["z"]))
            take.append({c: p[c][:m] for c in _COLS})
            if m < len(p["z"]):
                rest.append({c: p[c][m:] for c in p})
            got += m
        cols = [pa.array(np.concatenate([t[c] for t in take])) for c in _COLS]
        self.w.write_table(pa.Table.from_arrays(cols, schema=_schema()), row_group_size=k)
        self.pending, self.n = rest, self.n - got

    def finish(self):
        if self.n:
            self._flush(self.n)
        self.w.close()


def _merge(runs: list, out, work: int, budget, tier):
    """Merge sorted runs into `out` (a `_Run` or the final `_Out`). Each run is read through a
    buffer of `work` shared out; at every step every row at or below the smallest buffered
    maximum is final, so those rows are merged (a stable sort of the candidates) and written."""
    from .view import _Hold
    k = len(runs)
    buf = max(_MIN_MERGE_ROWS, work // (k * _MERGE_ROW_BYTES))
    with _Hold(budget, tier, min(work, k * buf * _MERGE_ROW_BYTES)):
        bufs = [r.read(buf) for r in runs]
        while True:
            active = [i for i in range(k) if bufs[i] is not None]
            if not active:
                break
            bound = min(bufs[i]["z"][-1] for i in active)
            parts = []
            for i in active:
                z = bufs[i]["z"]
                j = int(np.searchsorted(z, bound, side="right"))
                if j:
                    parts.append({c: a[:j] for c, a in bufs[i].items()})
                bufs[i] = runs[i].read(buf) if j == len(z) else {c: a[j:] for c, a in bufs[i].items()}
            z = np.concatenate([p["z"] for p in parts])
            order = np.argsort(z, kind="stable")
            cols = {"z": z[order]}
            for c in _COLS:
                cols[c] = np.concatenate([p[c] for p in parts])[order]
            parts = z = order = None
            out.append(cols)
            cols = None


def _sort_level(gf, b: int, dest: str, tmp: str, work: int, budget, tier, group: int) -> int:
    """Write level `b` of `gf` to `dest` sorted by Z-order, never holding more than `work`.

    A level that fits one block is sorted in memory and written. Otherwise every block of
    `work // _RUN_ROW_BYTES` rows becomes a sorted run on disk, and runs are merged at most
    `fan` at a time — several passes when there are more."""
    from .view import _Hold
    exp = gf.f[gf.levels[b].path]["expression"]
    n = gf.levels[b].n_obs
    if n == 0:
        return 0
    has_count = "count" in (exp.dtype.names or ())
    rows = max(1, work // _RUN_ROW_BYTES)
    if n <= rows:
        with _Hold(budget, tier, n * _RUN_ROW_BYTES):
            out = _Out(dest, group)
            try:
                out.append(_sorted_block(exp, gf, b, 0, n, has_count))
            finally:
                out.finish()
        return n
    runs = []
    for a in range(0, n, rows):
        c = min(a + rows, n)
        with _Hold(budget, tier, (c - a) * _RUN_ROW_BYTES):
            r = _Run(os.path.join(tmp, f"run{len(runs):06d}.bin"), c - a)
            r.append(_sorted_block(exp, gf, b, a, c, has_count))
            runs.append(r.finish())
    fan = max(2, min(_MAX_FAN, work // (_MERGE_TARGET_ROWS * _MERGE_ROW_BYTES)))
    level = 0
    try:
        while len(runs) > fan:                             # merge in passes until one pass will do
            nxt = []
            for s in range(0, len(runs), fan):
                grp = runs[s:s + fan]
                if len(grp) == 1:
                    nxt.append(grp[0])
                    continue
                m = _Run(os.path.join(tmp, f"m{level}-{s // fan:06d}.bin"), sum(r.n for r in grp))
                _merge(grp, m, work, budget, tier)
                for r in grp:
                    r.close(remove=True)
                nxt.append(m.finish())
            runs, level = nxt, level + 1
        out = _Out(dest, group)
        try:
            _merge(runs, out, work, budget, tier)
        finally:
            out.finish()
    finally:
        for r in runs:
            r.close()
    return n


def build_spatial_index(gef, levels=None, *, row_group_rows: int = 16_384,
                        budget: Optional[Budget] = None, directory: Optional[str] = None,
                        force: bool = False, verbose: bool = False) -> SpatialIndex:
    """Write the Z-order sidecar for some or all bin levels of a GEF.

    `gef` may be a path, a `GefFile`, or a Stereo-seq view. Levels already indexed and still
    matching the GEF are skipped unless `force=True`; rebuilding some levels keeps the others in
    `index.json`. With `budget`, everything the build holds is leased: the level is read in
    blocks that fit a quarter of what is free (at least 1 MB), each sorted into a run, and the
    runs merged inside the same allowance. Without one it works in 256 MB.
    """
    from .formats import resolve_chunk_bytes
    from .stereoseq import GefFile
    gf = getattr(gef, "gef", gef)
    own = False
    if isinstance(gf, (str, os.PathLike)):
        gf, own = GefFile(os.fspath(gf)), True
    try:
        fp = gf.fingerprint()
        d = directory or _sidecar_dir(gf.path, write=True)
        os.makedirs(d, exist_ok=True)
        idx = SpatialIndex(gf.path, d, fingerprint=fp)
        meta = {"gef": fp, "levels": {}}
        if idx.valid:                                      # keep the levels that are still there
            meta["levels"] = {k: v for k, v in idx.meta.get("levels", {}).items()
                              if os.path.exists(idx.path(int(k)))}
        want = list(levels) if levels is not None else list(gf.bins)
        for b in want:
            if b not in gf.levels:
                raise ValueError(f"{gf.path} has no bin{b}; levels: {gf.bins}")
        work = (resolve_chunk_bytes(None, 256 << 20, budget) if budget is not None else 256 << 20)
        for b in want:
            if not force and str(b) in meta["levels"] and os.path.exists(idx.path(b)):
                continue
            t0 = time.perf_counter()
            dest = idx.path(b)
            tmp = os.path.join(d, f".build-bin{b}-{os.getpid()}")
            os.makedirs(tmp, exist_ok=True)
            part = dest + f".{os.getpid()}.partial"
            try:
                rows = _sort_level(gf, b, part, tmp, work, budget, "ram", row_group_rows)
                if rows:
                    os.replace(part, dest)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
                if os.path.exists(part):
                    os.remove(part)
            if rows == 0:
                continue
            meta["levels"][str(b)] = {"rows": rows, "bytes": os.path.getsize(dest),
                                      "row_group_rows": row_group_rows,
                                      "seconds": round(time.perf_counter() - t0, 3)}
            if verbose:
                print(f"  bin{b:<4} {rows:>10,} rows -> {human(os.path.getsize(dest))} "
                      f"in {time.perf_counter() - t0:.1f}s")
        meta["genes"] = [str(g) for g in gf.genes(gf.bins[0])]
        tmp_meta = os.path.join(d, f".index.json.{os.getpid()}")
        with open(tmp_meta, "w") as fh:
            json.dump(meta, fh)
        os.replace(tmp_meta, os.path.join(d, "index.json"))
        return SpatialIndex(gf.path, d, fingerprint=fp)
    finally:
        if own:
            gf.close()
