"""Navigate a dataset the way you navigate a map: zoom, pan, select an interval — and be told
what it costs *before* anything is loaded.

The three ideas here:

**Overview before detail.** Every modality can produce a level-of-detail summary from metadata
that is already cheap to reach: Parquet row-group statistics, an h5ad's `indptr`, a sequence
index sidecar, an image's tile grid. Drawing a 60 GB file at zoom 0 costs a few kilobytes.

**Cost before load.** `view.cost()` returns resident bytes, transfer bytes and chunk count for
the current window at the current zoom, computed from the same metadata. It also says whether
it fits the attached Budget, and if not, what to change.

**Zoom is resolution, not just extent.** Zooming out does not load less of the dataset — it
loads a *coarser* representation of the same span. Every level costs about the same to draw,
which is what makes navigation feel instant on a laptop.

The memory contract every view keeps:

* `cost()` prices what `load()` and `chunks()` do. `total_bytes` is what `load()` hands out,
  `peak_bytes` what it needs while reading (the output plus one read block), `resident_bytes`
  the largest chunk `chunks()` hands out. `fits_whole` compares `peak_bytes`, and `fits` one
  chunk plus its read block, with what the budget has free (counting what the view itself holds,
  since it gives that back before reading).
* Bytes are leased **before** they are allocated. `load()` reserves its planned output first, then
  reads in blocks sized to what is left, each block leased while it is read.
* Decimation is one global grid: at stride `s` the unit `i` of a window starting at `lo` is kept
  when `(i - lo) % s == 0` — the same units from `load()`, `chunks()` and `tiles()`, whatever the
  chunk size.
* A chunk is accounted until the next one is requested, the stream is closed (a `break` closes
  it), or `release()` is called. A chunk a reader already leased is not leased a second time.
"""
from __future__ import annotations

import math
import os
import time as _time
import weakref
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

import numpy as np

from .journal import _warn_once as _journal_warn, active as _journal_active, count_units
from .journal import record as _journal_record
from .selections import dataset_key as _dataset_key
from ._core import Budget, human, parse_size
from .formats import _spatial_axes, csr_bounds, resolve_chunk_bytes

__all__ = ["Extent", "Overview", "View", "RowView", "SeqView", "ImageView",
           "ProgressiveCanvas", "CostError"]

_DEFAULT_CHUNK = 64 << 20         # what a view streams in when no budget says otherwise


class CostError(MemoryError):
    """Raised when a window cannot be loaded within the attached budget."""


def _nbytes(obj) -> int:
    """Resident size of whatever a view handed out, across the types views return."""
    if obj is None:
        return 0
    if isinstance(obj, dict):
        return sum(_nbytes(v) for v in obj.values())
    for attr in ("nbytes",):                               # numpy, pyarrow buffers
        v = getattr(obj, attr, None)
        if isinstance(v, (int, np.integer)):
            return int(v)
    if hasattr(obj, "get_total_buffer_size"):              # pyarrow Table / RecordBatch
        return int(obj.get_total_buffer_size())
    if hasattr(obj, "data") and hasattr(obj, "indices"):   # scipy sparse
        return int(obj.data.nbytes + obj.indices.nbytes + getattr(obj, "indptr", np.zeros(0)).nbytes)
    return 0


def _arrow_bytes(obj) -> int:
    """Memory an Arrow table or batch keeps alive — whole buffers, not just a slice of them."""
    if obj is None:
        return 0
    if hasattr(obj, "get_total_buffer_size"):
        return int(obj.get_total_buffer_size())
    return _nbytes(obj)


def _acquire(budget: Budget, tier: str, nbytes: int):
    """A lease for `nbytes`, or `CostError` at once when the tier cannot grant it. A view's
    reads run on the caller's thread, so waiting for the budget to drain would only time out.
    One locked check-and-take (`Budget.try_acquire`) when the core offers it, so two threads
    cannot both pass an `available()` check for the same bytes; the two-call form otherwise."""
    n = int(nbytes)
    take = getattr(budget, "try_acquire", None)
    if take is not None:
        lease = take(tier, str(n))
        if lease is not None:
            return lease
        free = int(budget.available(tier))
    else:
        free = int(budget.available(tier))
        if n <= free:
            return budget.acquire(tier, str(n))
    raise CostError(f"needs {human(n)} in the '{tier}' tier but only {human(free)} "
                    "is free; narrow the window, zoom out, or raise the budget")


class _Hold:
    """A reservation made of one or more budget leases, given back together.

    `add` leases more bytes, `resize` grows or shrinks it to an exact size, `release` gives
    everything back. Without a budget it only keeps count. Usable as a context manager for a
    transient (a block being read, a column being concatenated).
    """
    __slots__ = ("budget", "tier", "leases", "bytes")

    def __init__(self, budget: Optional[Budget] = None, tier: str = "ram", nbytes: int = 0):
        self.budget, self.tier, self.leases, self.bytes = budget, tier, [], 0
        if nbytes:
            self.add(nbytes)

    def add(self, nbytes) -> "_Hold":
        n = int(nbytes)
        if n <= 0:
            return self
        if self.budget is not None:
            self.leases.append(_acquire(self.budget, self.tier, n))
        self.bytes += n
        return self

    def resize(self, nbytes) -> "_Hold":
        n = max(int(nbytes), 0)
        if n > self.bytes:
            self.add(n - self.bytes)
        elif n < self.bytes:
            self.release()
            self.add(n)
        return self

    def release(self):
        leases, self.leases, self.bytes = self.leases, [], 0
        for lease in leases:
            try:
                lease.release()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


class _FreeAs:
    """Budget-shaped stand-in reporting a fixed free size, so the chunk floor rule can be asked
    about the room a read *will* have."""
    __slots__ = ("n",)

    def __init__(self, n: int):
        self.n = int(n)

    def available(self, tier: str = "ram") -> int:
        return self.n


class _Accounted:
    """A chunk whose bytes its producer already holds a lease for, until the producer resumes."""
    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data


def _grid_count(lo: int, hi: int, s: int) -> int:
    """Units of [lo, hi) on the stride grid."""
    return max(0, -(-(int(hi) - int(lo)) // max(int(s), 1)))


def _grid_first(start: int, lo: int, s: int) -> int:
    """The first unit >= start on the grid anchored at lo."""
    return int(start) + (-(int(start) - int(lo))) % max(int(s), 1)


def _grid_hit(a: int, b: int, lo: int, hi: int, s: int) -> bool:
    """Does [a, b) hold a unit of [lo, hi) on the grid?"""
    start, end = max(a, lo), min(b, hi)
    return start < end and _grid_first(start, lo, s) < end


@dataclass
class Extent:
    """The navigable domain of a dataset."""
    kind: str                      # "rows" | "records" | "cells" | "pixels"
    size: tuple                    # (n,) or (height, width)
    unit_bytes: float              # mean resident bytes per unit at full resolution

    @property
    def n(self) -> int:
        return int(np.prod(self.size))

    def __repr__(self) -> str:
        s = " x ".join(f"{d:,}" for d in self.size)
        return f"Extent({s} {self.kind}, ~{human(int(self.unit_bytes))}/unit)"


@dataclass
class Overview:
    """A level-of-detail summary. `cheap` is False if data pages had to be read to build it."""
    kind: str
    window: tuple
    x: np.ndarray                              # bucket centres, in domain units
    series: dict = field(default_factory=dict) # name -> values, one per bucket
    counts: Optional[np.ndarray] = None
    cheap: bool = True
    source_bytes: int = 0                      # bytes consulted to build this

    def __len__(self) -> int:
        return len(self.x)

    def __repr__(self) -> str:
        return (f"Overview({self.kind}, {len(self)} buckets over {self.window}, "
                f"series={list(self.series)}, {'metadata-only' if self.cheap else 'sampled'})")

    def sparkline(self, name: Optional[str] = None, width: int = 60) -> str:
        """A dependency-free glyph plot — useful in a terminal or a log."""
        name = name or next(iter(self.series))
        v = np.asarray(self.series[name], dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return "(no data)"
        if v.size > width:
            idx = np.linspace(0, v.size - 1, width).astype(int)
            v = v[idx]
        lo, hi = float(v.min()), float(v.max())
        blocks = "▁▂▃▄▅▆▇█"
        if hi - lo < 1e-12:
            return blocks[3] * len(v)
        q = ((v - lo) / (hi - lo) * (len(blocks) - 1)).round().astype(int)
        return "".join(blocks[i] for i in q)


class _ChunkStream:
    """What `View.chunks()` returns: an iterator that holds the current chunk's reservation and
    gives it back when the next chunk is asked for, when it is closed — a `break` out of the loop
    drops it, which closes it — or when the view's `release()` is called."""

    def __init__(self, view: "View"):
        self.view = view
        self.inner = None
        self.done = False
        self.busy = False
        self.n_chunks = 0                       # for the journal: what the stream handed out
        self.nbytes = 0
        self.units = 0
        self.t0 = _time.perf_counter()

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        v = self.view
        if self.inner is None:
            v.check(whole=False)
            v.release()                         # whatever the view handed out before is done with
            v._stream = weakref.ref(self)
            self.inner = iter(v._chunks(v._win, v.stride))
        v._drop_hold()                          # ... and so is the previous chunk
        self.busy = True
        try:
            item = next(self.inner)
        except BaseException:
            self.busy = False
            self.close()
            raise
        self.busy = False
        self.n_chunks += 1
        if isinstance(item, _Accounted):        # the producer's own lease covers it
            self.nbytes += _nbytes(item.data)
            self.units += count_units(item.data)
            return item.data
        data = item[0] if isinstance(item, tuple) else item
        v._hold(_nbytes(data))
        self.nbytes += _nbytes(data)
        self.units += count_units(data)
        return item

    def close(self):
        if self.done:
            return
        v = self.view
        if self.busy:                           # release() from inside the producer: its last hold only
            v._drop_hold()
            return
        self.done = True
        inner, self.inner = self.inner, None
        try:
            if inner is not None and hasattr(inner, "close"):
                inner.close()
        finally:
            if v._stream is not None and v._stream() is self:
                v._stream = None
            v._drop_hold()
            v._release_source()
            if self.n_chunks:
                v._journal_record("chunks", units=self.units, nbytes=self.nbytes,
                                  seconds=_time.perf_counter() - self.t0, chunks=self.n_chunks)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class View:
    """Zoom/pan/select over one dataset, with cost estimation before loading.

    Subclasses implement `_extent`, `_full_window`, `_overview`, `_load` and `_chunk_plan`, and
    usually `_chunks`. `owns_source=False` makes `close()` leave the source open (a dataset
    that hands out several views over one file); `chunk_bytes` overrides the chunk floor rule.
    """

    def __init__(self, budget: Optional[Budget] = None, tier: str = "ram", *,
                 owns_source: bool = True, chunk_bytes=None):
        self.budget, self.tier = budget, tier
        self.owns_source = bool(owns_source)
        self.chunk_bytes = chunk_bytes
        self._lease: Optional[_Hold] = None    # the reservation behind what load()/chunks() last returned
        self._stream = None                    # weak reference to the chunk stream in flight
        self.extent: Extent = self._extent()
        self._win = self._full_window()
        self._zoom = 0                         # signed: below 0 = zoomed in past full resolution

    # ---- accounting ----
    def _drop_hold(self):
        h, self._lease = self._lease, None
        if h is not None:
            h.release()

    def _hold(self, nbytes: int):
        """Reserve `nbytes` for the data about to be handed out, giving back the previous hold first."""
        self._drop_hold()
        self._lease = _Hold(self.budget, self.tier).add(nbytes)

    def _settle(self, nbytes: int):
        """Make the current hold exactly `nbytes` — what was actually handed out."""
        if self._lease is None:
            self._lease = _Hold(self.budget, self.tier)
        self._lease.resize(nbytes)

    def _held(self) -> int:
        return int(self._lease.bytes) if self._lease is not None else 0

    # ---- the journal ----
    def _journal_level(self) -> dict:
        """How this view names its level of detail in a journal record."""
        return {"zoom": int(self.zoom), "stride": int(self.stride)}

    def _journal_record(self, op: str, **kw):
        """Write one record about the current window to the active journal, if any."""
        if _journal_active() is None:
            return
        try:
            _journal_record(op, self, window=self._win, level=self._journal_level(),
                            unit=self.extent.kind, genes=getattr(self, "genes", None), **kw)
        except Exception as err:                # never let the diary stop the read
            _journal_warn(f"journal: {err!r}")

    def _sources(self):
        for attr in ("src", "a", "m", "vol", "gef", "index"):
            obj = getattr(self, attr, None)
            if obj is not None:
                yield obj

    def _release_source(self):
        """Give back a reader's lease on the chunk it produced last (formats readers keep it until
        their next chunk otherwise)."""
        for obj in self._sources():
            swap = getattr(obj, "_swap_lease", None)
            if callable(swap):
                try:
                    swap(0)
                except Exception:
                    pass

    def release(self) -> "View":
        """Give back the reservation for the last thing load()/chunks() produced, close a chunk
        stream still in flight, and drop the source reader's lease on its last chunk."""
        st = self._stream() if self._stream is not None else None
        if st is not None:
            if st.busy:                          # called by the producer itself, mid-chunk
                self._drop_hold()
                return self
            st.close()
        self._drop_hold()
        self._release_source()
        return self

    def close(self):
        """Release, then close the sources this view owns (not one a dataset lent it)."""
        self.release()
        if not self.owns_source:
            return
        for obj in self._sources():
            fn = getattr(obj, "close", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *e):
        self.close()
        return False

    # ---- subclass hooks ----
    def _extent(self) -> Extent: raise NotImplementedError
    def _full_window(self) -> tuple: raise NotImplementedError
    def _overview(self, window, n_buckets) -> Overview: raise NotImplementedError
    def _load(self, window, stride): raise NotImplementedError
    def _chunk_plan(self, window, stride) -> tuple[int, int]:
        """(n_chunks, resident_bytes of the largest chunk) for the current selection."""
        raise NotImplementedError

    def _load_need(self, window, stride, total: int) -> int:
        """Peak bytes `load()` needs: what it hands out plus the transient it reads through."""
        return int(total)

    def _chunk_need(self, window, stride, per_chunk: int) -> int:
        """Peak bytes one chunk of `chunks()` needs while it is read."""
        return int(per_chunk)

    def _zoom_levels(self):
        """The zoom levels `auto_zoom` tries, finest first."""
        span = max([self._win[i + 1] - self._win[i] for i in range(0, len(self._win), 2)] + [1])
        return range(0, max(0, math.ceil(math.log2(max(span, 1)))) + 1)

    # ---- navigation ----
    @property
    def window(self) -> tuple:
        return self._win

    @property
    def zoom(self) -> int:
        return max(self._zoom, 0)

    @property
    def stride(self) -> int:
        """Decimation factor implied by the zoom level: 1 = every unit, 2 = every other."""
        return 1 << max(self._zoom, 0)

    def reset(self) -> "View":
        self._win, self._zoom = self._full_window(), 0
        return self

    def select(self, *bounds) -> "View":
        """Explicit interval: `select(lo, hi)` for 1-D, `select(y0, y1, x0, x1)` for images."""
        self._win = self._clamp(tuple(int(b) for b in bounds))
        return self

    def select_frac(self, *fracs) -> "View":
        """Same, in fractions of the full extent: `select_frac(0.25, 0.30)`.

        Give one pair and it applies to every axis — "the middle tenth of this thing",
        whatever its dimensionality. Give a pair per axis for finer control.
        """
        full = self._full_window()
        n_axes = len(full) // 2
        if len(fracs) == 2 and n_axes > 1:
            fracs = tuple(fracs) * n_axes
        if len(fracs) != len(full):
            raise ValueError(
                f"{type(self).__name__} has {n_axes} axis/axes; pass 2 fractions to apply to "
                f"all of them, or {len(full)} for one pair per axis (got {len(fracs)})")
        out = []
        for i in range(0, len(full), 2):
            lo, hi = full[i], full[i + 1]
            a, b = fracs[i], fracs[i + 1]
            out += [lo + (hi - lo) * a, lo + (hi - lo) * b]
        return self.select(*out)

    def zoom_in(self, factor: float = 2.0, center=None, constant_cost: bool = True) -> "View":
        """Narrow the window. By default resolution rises to match, the way a map does — so
        the amount of data on screen, and therefore the cost, stays roughly constant.

        `center` is a coordinate for a 1-D view, or one per axis. The window keeps its width at
        an edge (it moves inside the extent rather than being cut). Zooming in past full
        resolution is remembered, so `zoom_out` by the same factor returns to the same zoom."""
        return self._rescale(1.0 / float(factor), center,
                             -self._levels(factor) if constant_cost else 0)

    def zoom_out(self, factor: float = 2.0, center=None, constant_cost: bool = True) -> "View":
        """Widen the window; resolution drops to match, keeping the cost roughly constant —
        by as many levels as the window actually widened (none when it already spans the
        extent). Pass `constant_cost=False` to widen without decimating (cost then grows)."""
        return self._rescale(float(factor), center,
                             self._levels(factor) if constant_cost else 0)

    @staticmethod
    def _levels(factor) -> int:
        f = float(factor)
        return max(int(round(math.log2(f))), 0) if f > 0 else 0

    def _rescale(self, k: float, center, dz: int) -> "View":
        old = self._win
        new = self._scaled(old, k, center)
        self._win = new
        if dz:
            ratio = self._units(new) / max(self._units(old), 1e-12)
            if dz < 0:
                ratio = 1.0 / max(ratio, 1e-12)
            dims = max(len(new) // 2, 1)
            moved = math.floor(math.log2(ratio) / dims + 0.5) if ratio > 1 else 0
            step = min(abs(dz), max(moved, 0))
            self._zoom += step if dz > 0 else -step
        return self

    def pan(self, frac: float = 0.5, axis: int = 0) -> "View":
        """Move the window by `frac` of its width along `axis`; at an edge it stops there,
        keeping its width."""
        w = list(self._win)
        i = axis * 2
        full = self._full_window()
        span = w[i + 1] - w[i]
        w[i:i + 2] = self._place(w[i] + span * frac, span, full[i], full[i + 1])
        self._win = tuple(w)
        return self

    def _scaled(self, w: tuple, k: float, center) -> tuple:
        full = self._full_window()
        n = len(w) // 2
        if isinstance(center, (tuple, list)):
            centers = list(center) + [None] * (n - len(center))
        else:
            centers = [center if n == 1 else None] + [None] * (n - 1)
        out = []
        for j in range(n):
            lo, hi = w[2 * j], w[2 * j + 1]
            f0, f1 = full[2 * j], full[2 * j + 1]
            c = centers[j] if centers[j] is not None else (lo + hi) / 2
            width = min(max((hi - lo) * k, 1), f1 - f0)
            out += self._place(c - width / 2, width, f0, f1)
        return tuple(out)

    @staticmethod
    def _place(start: float, width: float, f0: int, f1: int) -> list:
        """[a, a + width) moved inside [f0, f1) without changing its width — unless it is wider
        than the extent, which it then spans."""
        span = int(f1 - f0)
        wd = max(1, min(int(math.floor(width + 0.5)), span)) if span > 0 else 0
        a = int(math.floor(start + 0.5))
        a = max(int(f0), min(a, int(f1) - wd))
        return [a, a + wd]

    def _clamp(self, w: tuple) -> tuple:
        full = self._full_window()
        out = []
        for i in range(0, len(w), 2):
            lo, hi = int(max(full[i], min(w[i], full[i + 1]))), int(max(full[i], min(w[i + 1], full[i + 1])))
            if hi <= lo:
                hi = min(lo + 1, full[i + 1])
            out += [lo, hi]
        return tuple(out)

    def set_zoom(self, level: int) -> "View":
        self._zoom = max(0, int(level))
        return self

    def auto_zoom(self, target: Optional[str] = None) -> "View":
        """Pick the finest zoom whose load fits the budget (or `target` bytes), leaving 20 %
        headroom for the consumer. Raises `CostError` naming the cheapest zoom when none fits;
        the zoom is then left where it was.

        This is the answer to "just show me this, whatever it takes to fit."
        """
        if target is not None:
            cap = parse_size(target) if isinstance(target, str) else int(target)
        elif self.budget is not None:
            cap = self._free()
        else:
            cap = 256 * 1024 ** 2
        cap = int(cap * 0.8)
        start, best = self._zoom, None
        for z in self._zoom_levels():
            self._zoom = z
            c = self.cost()
            need = int(c.get("peak_bytes", c["total_bytes"]))
            if need <= cap and c.get("fits_whole") is not False:
                return self
            if best is None or need < best[0]:
                best = (need, z)
        self._zoom = start
        need, z = best if best else (0, start)
        err = CostError(f"no zoom fits: the cheapest is zoom {z}, which needs {human(need)} to "
                        f"load, and {human(cap)} is the target (80 % of what is free); narrow the "
                        f"window, stream it with .chunks(), or raise the '{self.tier}' budget")
        err.cost = {"zoom": z, "peak_bytes": need, "target": cap}
        raise err

    # ---- cost ----
    def _free(self) -> int:
        """What the budget has free for this view: what is free plus what the view holds, which
        it gives back before it reads anything."""
        return int(self.budget.available(self.tier)) + self._held() if self.budget else 0

    def cost(self) -> dict:
        """What loading the current window at the current zoom would take. Reads nothing."""
        win, s = self._win, self.stride
        n_chunks, per_chunk = self._chunk_plan(win, s)
        units = self._kept_units(win, s)
        total = int(units * self.extent.unit_bytes)      # truncated: subclasses add 0.5 to round
        need = int(self._load_need(win, s, total))
        chunk_need = int(self._chunk_need(win, s, int(per_chunk)))
        cap = self._free() if self.budget else None
        d = {
            "window": win,
            "zoom": self.zoom,
            "stride": s,
            "units": int(units),
            "unit": self.extent.kind,
            "resident_bytes": int(per_chunk),
            "resident_human": human(int(per_chunk)),
            "total_bytes": total,
            "total_human": human(total),
            "peak_bytes": need,
            "chunk_peak_bytes": chunk_need,
            "chunks": int(n_chunks),
            "fraction_of_dataset": (self._units(win) / max(self.extent.n, 1)),
        }
        if cap is not None:
            d["budget_available"] = cap
            d["budget_available_human"] = human(cap)
            # Two different questions. `fits` = can this be streamed one chunk at a time.
            # `fits_whole` = can load() hand you the entire window at once, reading through one
            # block — that is what load() actually does, so that is the one that gates it.
            d["fits"] = chunk_need <= cap
            d["fits_whole"] = need <= cap
            if not d["fits_whole"]:
                d["advice"] = self._advice(need, cap)
        else:
            d["fits"] = d["fits_whole"] = None
        return d

    def _advice(self, need: int, cap: int) -> str:
        ratio = need / max(cap, 1)
        dims = max(len(self._win) // 2, 1)
        levels = max(1, math.ceil(math.log2(max(ratio, 1.0001)) / dims))
        return (f"the window needs {human(need)} at zoom {self.zoom} but {human(cap)} is free; "
                f"zoom out {levels} level(s), narrow the window to ~{100 / ratio:.0f}% of its span, "
                f"stream it with .chunks() instead of .load(), or raise the '{self.tier}' budget")

    def _units(self, w) -> float:
        u = 1.0
        for i in range(0, len(w), 2):
            u *= max(w[i + 1] - w[i], 0)
        return u

    def _kept_units(self, w, stride) -> int:
        """Units of the window on the decimation grid: what load() returns."""
        u = 1
        for i in range(0, len(w), 2):
            u *= _grid_count(w[i], w[i + 1], stride)
        return u

    def check(self, whole: bool = True) -> "View":
        """Raise CostError if the current selection cannot be satisfied. `whole=False` only
        asks whether it can be streamed."""
        c = self.cost()
        if whole and c.get("fits_whole") is False:
            raise CostError(c.get("advice", "does not fit the budget"))
        if not whole and c.get("fits") is False:
            need = int(c.get("chunk_peak_bytes", c.get("resident_bytes", 0)))
            raise CostError(
                f"even one chunk needs {human(need)} (the chunk and the block it is read through) "
                f"but {human(int(c.get('budget_available', 0)))} is free; narrow the window, zoom "
                f"out, or raise the '{self.tier}' budget")
        return self

    def chunks(self):
        """Stream the current window at the current zoom, one budget-sized chunk at a time.
        Use this when `cost()['fits_whole']` is False but `fits` is True. Each chunk is
        accounted against the budget until the next one is requested; leaving the loop early
        (or `release()`) gives the reservation back."""
        return _ChunkStream(self)

    def _chunks(self, window, stride):
        yield self._load(window, stride)

    def _chunk_cap(self, free: Optional[int] = None) -> int:
        """The chunk size: the floor rule of `formats.resolve_chunk_bytes` — a quarter of what is
        free, floored at 1 MB but never above what is free — unless `chunk_bytes` was set. `free`
        overrides what the budget says is free (planning counts what the view holds as free)."""
        if self.budget is None:
            return int(parse_size(self.chunk_bytes)) if isinstance(self.chunk_bytes, str) else \
                int(self.chunk_bytes or _DEFAULT_CHUNK)
        b = self.budget if free is None else _FreeAs(free)
        try:
            return int(resolve_chunk_bytes(self.chunk_bytes, _DEFAULT_CHUNK, b, self.tier))
        except MemoryError:                    # a chosen size larger than what is free: what is free
            return max(int(b.available(self.tier)), 1)

    def _plan_cap(self) -> int:
        """The chunk size `chunks()` will use: the view gives back what it holds before reading."""
        return self._chunk_cap(self._free()) if self.budget is not None else self._chunk_cap()

    # ---- data ----
    def overview(self, n_buckets: int = 256) -> Overview:
        t0 = _time.perf_counter()
        ov = self._overview(self._win, n_buckets)
        self._journal_record("overview", units=int(n_buckets), seconds=_time.perf_counter() - t0)
        return ov

    def load(self):
        """Materialize the current window at the current zoom, budget-enforced and accounted.
        The reservation is held until the next load()/chunks() call or `release()`. The piece —
        file, level, exact window, rows and bytes — is written to the active journal."""
        self.check()
        self.release()
        t0 = _time.perf_counter()
        try:
            data = self._load(self._win, self.stride)
        except BaseException:
            self._drop_hold()
            raise
        if self._lease is None or self._lease.bytes == 0:     # the subclass did not account it
            self._hold(_nbytes(data))
        self._journal_record("load", units=count_units(data), nbytes=self._held(),
                             seconds=_time.perf_counter() - t0)
        return data

    def __repr__(self) -> str:
        c = self.cost()
        fit = ("" if c.get("fits_whole") is None
               else ("  fits" if c["fits_whole"] else "  stream only" if c["fits"] else "  TOO BIG"))
        return (f"{type(self).__name__}(window={self._win}, zoom={self.zoom}, "
                f"{c['units']:,} {c['unit']}, ~{c['resident_human']}/chunk x {c['chunks']}{fit})")


# ---------------------------------------------------------------------------
# Row-oriented: Parquet, h5ad
# ---------------------------------------------------------------------------

def _is_parquet(src) -> bool:
    return hasattr(src, "stream") and hasattr(src, "describe")


def _is_h5(src) -> bool:
    return hasattr(src, "node") and hasattr(src, "encoding") and hasattr(src, "n_obs")


_MIN_BATCH_ROWS = 256         # the Rust Parquet reader never makes a batch smaller than this


class RowView(View):
    """A view over anything with rows: Parquet row groups, h5ad cells.

    Zoom decimates rows on the global grid (`(i - lo) % stride == 0`), which keeps
    distributions honest while shrinking the resident footprint by exactly the stride. Reads
    touch only the window: Parquet row groups that hold a row of it (on the grid), h5ad rows
    `[lo, hi)` cut into blocks from `indptr`.
    """

    def __init__(self, source, columns: Optional[Sequence[str]] = None,
                 budget: Optional[Budget] = None, kind: str = "rows", *,
                 owns_source: bool = True, chunk_bytes=None):
        self.src = source
        self.columns = list(columns) if columns else None
        self.kind = kind
        self._desc = None
        self._layout = None
        self._schema = None
        self._footer_row = None
        if budget is not None and _is_parquet(source):
            source.set_budget(budget)          # every batch the dataset produces is accounted
        super().__init__(budget, owns_source=owns_source, chunk_bytes=chunk_bytes)
        self._journal_open()

    def _journal_open(self):
        """The first time a table is opened against a journal, write what it is: its row count
        and, for an h5ad, its gene (var) names; for Parquet, its column names as `col=` tags —
        so a gene or a column grepped in the journal says which files you ever opened have it."""
        if _journal_active() is None:
            return
        try:
            key = _dataset_key(self)
            j = _journal_active()
            if j.opened(key):
                return
            genes, tags = None, []
            if _is_h5(self.src):
                try:
                    names = self.src.var_names
                except (KeyError, ValueError, OSError):      # no var group: a bare matrix
                    names = None
                if names is not None and len(names) <= 100_000:
                    genes = [str(x) for x in names]
            elif _is_parquet(self.src):
                tags = [f"col={str(f).split(':', 1)[0].strip()}" for f in self._describe().get("schema", [])]
            else:
                return
            _journal_record("open", key, window=list(self._full_window()), level={},
                            unit=self.extent.kind, genes=genes, units=int(self.extent.n),
                            nbytes=int(self._describe().get("total_size_bytes", 0)
                                       or key["fingerprint"].get("size") or 0), tags=tags)
        except Exception as err:
            _journal_warn(f"journal: {err!r}")

    def _describe(self):
        if self._desc is None:
            self._desc = self.src.describe() if _is_parquet(self.src) else {}
        return self._desc

    def _extent(self) -> Extent:
        if _is_parquet(self.src):
            rows = int(self._describe()["total_rows"])
            p = self.src.plan(None, self.columns)
            per = float(p["bytes_per_row"])
            if self.columns:
                per = max(p["chunk_bytes"] / max(p["rows_per_chunk"], 1), 1.0)
            per = max(per, self._fixed_row_bytes())
        else:                                            # H5adReader / ObsmReader
            rows = int(self.src.n_obs)
            p = self.src.plan()
            per = p.bytes_per_chunk / max(p.detail["rows_per_chunk"], 1)
        return Extent(self.kind, (rows,), per)

    def _fixed_row_bytes(self) -> float:
        """Arrow bytes a row's fixed-width columns take once decoded — a floor under the footer
        estimate, which dictionary and run-length encoding can make far smaller than memory."""
        try:
            sch = self._arrow_schema()
        except Exception:
            return 0.0
        per = 0.0
        for f in sch:
            try:
                per += f.type.bit_width / 8.0
            except ValueError:                           # variable width: 4-byte offsets at least
                per += 4.0
        return per

    def _arrow_schema(self):
        if self._schema is None:
            import pyarrow as pa
            sch = self.src.schema()
            if self.columns:
                sch = pa.schema([sch.field(c) for c in self.columns])
            self._schema = sch
        return self._schema

    def _full_window(self):
        return (0, self.extent.size[0])

    # ---- Parquet ----
    def _pq_layout(self) -> list:
        """[(file index, [(row group index, first row, end row), ...]), ...] — global rows."""
        if self._layout is None:
            out, off = [], 0
            for fi, f in enumerate(self._describe()["files"]):
                rgs = []
                for j, rg in enumerate(f["row_groups"]):
                    n = int(rg["num_rows"])
                    rgs.append((int(rg.get("index", j)), off, off + n))
                    off += n
                out.append((fi, rgs))
            self._layout = out
        return self._layout

    def _pq_footer_row(self) -> float:
        """Bytes a row takes by the footer's (encoded) sizes — what the Rust reader cuts batches
        by. Dictionary and run-length encoding can make it far smaller than the decoded row."""
        if getattr(self, "_footer_row", None) is None:
            try:
                rpm = int(self.src.plan(str(1 << 20), self.columns)["rows_per_chunk"])
                self._footer_row = (1 << 20) / max(rpm, 1)
            except Exception:
                self._footer_row = float(self.extent.unit_bytes)
        return self._footer_row

    def _pq_reader_chunk(self, cap: int) -> int:
        """The `chunk_bytes` that makes the reader's batches about `cap` bytes once decoded."""
        rows = max(int(cap // max(self.extent.unit_bytes, 1)), 1)
        return max(int(rows * self._pq_footer_row()), 1)

    def _pq_rows_per_batch(self, chunk: int) -> int:
        try:
            return int(self.src.plan(str(self._pq_reader_chunk(chunk)), self.columns)["rows_per_chunk"])
        except Exception:
            return max(_MIN_BATCH_ROWS, int(chunk // max(self.extent.unit_bytes, 1)))

    @staticmethod
    def _pq_batches_hit(a: int, b: int, lo: int, hi: int, s: int, rpb: int) -> int:
        """Batches of row group [a, b) — the reader cuts it every `rpb` rows — that hold a grid
        row of [lo, hi): the pieces `_pq_pieces` yields from it."""
        j0 = max(0, (max(lo, a) - a) // rpb)
        j1 = (min(hi, b) - 1 - a) // rpb
        if j1 < j0:
            return 0
        if s == 1:
            return j1 - j0 + 1
        if j1 - j0 > 4096:                      # far too many to walk: every batch wider than
            return min(j1 - j0 + 1, _grid_count(max(lo, a), min(hi, b), s))   # s has one
        return sum(_grid_hit(a + j * rpb, min(a + (j + 1) * rpb, b), lo, hi, s)
                   for j in range(j0, j1 + 1))

    def _pq_pieces(self, lo: int, hi: int, s: int, chunk: int):
        """Batches of the window's grid rows, read from the row groups that hold one. A batch
        wholly inside at stride 1 is passed on as is; anything else is a copy of the kept rows,
        so nothing outside the window stays alive. The dataset's lease covers each batch until
        the next one is fetched."""
        import pyarrow as pa
        for fi, rgs in self._pq_layout():
            sel = [(i, a, b) for (i, a, b) in rgs if _grid_hit(a, b, lo, hi, s)]
            if not sel:
                continue
            it = self.src.stream(columns=self.columns, row_groups=[i for i, _, _ in sel],
                                 files=[fi], chunk_bytes=str(self._pq_reader_chunk(chunk)))
            k, pos = 0, sel[0][1]
            for batch in it:
                off, n = 0, batch.num_rows
                while off < n:
                    m = min(n - off, sel[k][2] - pos)
                    start, end = max(pos, lo), min(pos + m, hi)
                    if start < end:
                        first = _grid_first(start, lo, s)
                        if first < end:
                            if s == 1 and off == 0 and m == n and start == pos and end == pos + m:
                                yield batch
                            else:
                                idx = np.arange(first - pos + off, end - pos + off, s, dtype=np.int64)
                                yield batch.take(pa.array(idx))
                    off += m
                    pos += m
                    if pos >= sel[k][2] and k + 1 < len(sel):
                        k += 1
                        pos = sel[k][1]
                batch = None
            if pos >= hi:
                break

    # ---- h5ad (H5adReader / ObsmReader) ----
    def _h5_cols(self) -> int:
        return int(getattr(self.src, "n_vars", None) or getattr(self.src, "dim", 1))

    def _h5_check(self):
        if getattr(self.src, "encoding", None) == "csc_matrix":
            raise NotImplementedError(
                "CSC is column-major; reading a window of rows would touch every column chunk. "
                "Convert to CSR first (anndata: adata.X = adata.X.tocsr()).")

    def _h5_widest(self, lo: int, hi: int) -> int:
        """Bytes of the widest single row in [lo, hi): the smallest block a read can use."""
        src = self.src
        if src.encoding != "csr_matrix":
            return self._h5_cols() * int(src._itemsize)
        ip, best = src.indptr, 0
        for a in range(lo, hi, 1 << 20):
            b = min(hi, a + (1 << 20))
            if b > a:
                best = max(best, int(np.max(ip[a + 1:b + 1] - ip[a:b])))
        return best * int(src._itemsize)

    def _h5_out_bytes(self, lo: int, hi: int, s: int) -> int:
        """Exact bytes of the rows load() hands out (CSR: data + indices + indptr)."""
        src = self.src
        n_out = _grid_count(lo, hi, s)
        if src.encoding == "csr_matrix":
            nnz = self._h5_nnz(lo, hi, s)
            idx = 4 if max(nnz, self._h5_cols()) < 2 ** 31 else 8
            return nnz * (np.dtype(src.dtype).itemsize + idx) + (n_out + 1) * idx
        return n_out * self._h5_cols() * np.dtype(src.dtype).itemsize

    def _h5_nnz(self, lo: int, hi: int, s: int) -> int:
        if hi <= lo:
            return 0
        ip = self.src.indptr
        return int(np.sum(ip[lo + 1:hi + 1:s], dtype=np.int64) - np.sum(ip[lo:hi:s], dtype=np.int64))

    @staticmethod
    def _h5_per(cap: int, s: int) -> int:
        """The `csr_bounds` chunk for a read cap: a block's data + indices and its indptr are each
        at most this, and when decimating its kept copy is too — so the block lease fits `cap`."""
        return max(int(cap) // (4 if s > 1 else 2), 1)

    def _h5_blocks(self, lo: int, hi: int, s: int, cap: int):
        """Kept rows of [lo, hi) block by block, as (first row, matrix). Each block is leased
        before it is read; what is yielded stays leased until the next block is asked for."""
        import scipy.sparse as sp
        src = self.src
        self._h5_check()
        node, cols = src.node, self._h5_cols()
        if src.encoding == "csr_matrix":
            ip, isz = src.indptr, int(src._itemsize)
            bounds = csr_bounds(ip, isz, self._h5_per(cap, s), lo, hi).tolist()
            for a, b in zip(bounds[:-1], bounds[1:]):
                first = _grid_first(a, lo, s)
                if first >= b:
                    continue
                p0, p1 = int(ip[a]), int(ip[b])
                blk = (p1 - p0) * isz + (b - a + 1) * 8
                with _Hold(self.budget, self.tier, blk * (2 if s > 1 else 1)) as hold:
                    X = sp.csr_matrix((node["data"][p0:p1], node["indices"][p0:p1],
                                       np.asarray(ip[a:b + 1]) - p0), shape=(b - a, cols))
                    if s > 1:
                        X = X[first - a::s]
                        hold.resize(_nbytes(X))
                    yield first, X
                    X = None
            return
        row = cols * np.dtype(src.dtype).itemsize
        step = max(1, int(cap) // max(row, 1)) * s
        for a in range(lo, hi, step):
            b = min(a + step, hi)
            first = _grid_first(a, lo, s)
            if first >= b:
                continue
            k = _grid_count(first, b, s)
            with _Hold(self.budget, self.tier, k * row):
                X = np.asarray(node[first:b:s])
                yield first, (X[:, None] if X.ndim == 1 else X)
                X = None

    def _h5_load(self, lo: int, hi: int, s: int):
        """The window's grid rows into arrays allocated once, at their exact size, after the
        output is leased; blocks are read one at a time on top of it."""
        import scipy.sparse as sp
        src = self.src
        self._h5_check()
        n_out, cols = _grid_count(lo, hi, s), self._h5_cols()
        self._hold(self._h5_out_bytes(lo, hi, s))
        cap = self._chunk_cap()
        if src.encoding == "csr_matrix":
            ip = src.indptr
            nnz = self._h5_nnz(lo, hi, s)
            idt = np.int32 if max(nnz, cols) < 2 ** 31 else np.int64
            data = np.empty(nnz, src.dtype)
            indices = np.empty(nnz, idt)
            indptr = np.zeros(n_out + 1, idt)
            if n_out:
                np.subtract(ip[lo + 1:hi + 1:s], ip[lo:hi:s], out=indptr[1:], casting="unsafe")
                np.cumsum(indptr[1:], out=indptr[1:])
            o = 0
            for _first, X in self._h5_blocks(lo, hi, s, cap):
                m = int(X.nnz)
                data[o:o + m] = X.data
                indices[o:o + m] = X.indices
                o += m
                X = None
            return sp.csr_matrix((data, indices, indptr), shape=(n_out, cols), copy=False)
        out = np.empty((n_out, cols), src.dtype)
        o = 0
        for _first, X in self._h5_blocks(lo, hi, s, cap):
            out[o:o + X.shape[0]] = X
            o += X.shape[0]
            X = None
        return out

    # ---- View hooks ----
    def _chunk_plan(self, w, stride):
        lo, hi = int(w[0]), int(w[1])
        cap = self._plan_cap()
        if _is_parquet(self.src):
            rpb = max(self._pq_rows_per_batch(cap), 1)
            n = 0
            for _fi, rgs in self._pq_layout():
                for (_i, a, b) in rgs:
                    if _grid_hit(a, b, lo, hi, stride):
                        n += self._pq_batches_hit(a, b, lo, hi, stride, rpb)
            return max(n, 1), int(min(rpb, max(hi - lo, 1)) * self.extent.unit_bytes)
        if _is_h5(self.src) and self.src.encoding == "csr_matrix":
            bounds = csr_bounds(self.src.indptr, self.src._itemsize, self._h5_per(cap, stride), lo, hi)
            n, big = 0, 0
            ip, isz = self.src.indptr, int(self.src._itemsize)
            for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
                if _grid_first(a, lo, stride) < b:
                    n += 1
                    big = max(big, (int(ip[b]) - int(ip[a])) * isz + (b - a + 1) * 8)
            return max(n, 1), big
        if _is_h5(self.src):
            row = self._h5_cols() * np.dtype(self.src.dtype).itemsize
            kept = max(1, cap // max(row, 1))
            n_out = _grid_count(lo, hi, stride)
            return max(math.ceil(n_out / kept), 1), min(kept, max(n_out, 1)) * row
        rows = _grid_count(lo, hi, stride)
        per_chunk_rows = min(max(int(cap / max(self.extent.unit_bytes, 1)), 1), max(rows, 1))
        return math.ceil(rows / per_chunk_rows) if rows else 1, per_chunk_rows * self.extent.unit_bytes

    def _load_need(self, w, stride, total):
        lo, hi = int(w[0]), int(w[1])
        if _is_parquet(self.src):
            return int(total + _MIN_BATCH_ROWS * self.extent.unit_bytes)
        if _is_h5(self.src):
            return int(total + 2 * self._h5_widest(lo, hi))
        return int(total)

    def _chunk_need(self, w, stride, per_chunk):
        if _is_h5(self.src) and stride > 1:
            return 2 * int(per_chunk)
        return int(per_chunk)

    def cost(self) -> dict:
        d = super().cost()
        if _is_h5(self.src) and self.src.encoding != "csc_matrix":
            # exact, from indptr: what the decimated rows hold
            lo, hi = int(self._win[0]), int(self._win[1])
            total = self._h5_out_bytes(lo, hi, self.stride)
            need = int(self._load_need(self._win, self.stride, total))
            d.update(total_bytes=total, total_human=human(total), peak_bytes=need,
                     read_bytes=self._h5_read_bytes(lo, hi, self.stride))
            d["read_human"] = human(d["read_bytes"])
            if d.get("budget_available") is not None:
                d["fits_whole"] = need <= d["budget_available"]
                d.pop("advice", None)
                if not d["fits_whole"]:
                    d["advice"] = self._advice(need, d["budget_available"])
        return d

    def _h5_read_bytes(self, lo: int, hi: int, s: int) -> int:
        """Bytes read from the file: whole CSR blocks (a block is read whole even when it keeps
        one row in `s`), or for a dense matrix the HDF5 chunks holding a kept row."""
        src = self.src
        if src.encoding == "csr_matrix":
            ip, isz = src.indptr, int(src._itemsize)
            if s == 1:
                return (int(ip[hi]) - int(ip[lo])) * isz
            bounds = csr_bounds(ip, isz, self._h5_per(self._plan_cap(), s), lo, hi).tolist()
            return sum((int(ip[b]) - int(ip[a])) * isz for a, b in zip(bounds[:-1], bounds[1:])
                       if _grid_first(a, lo, s) < b)
        row = self._h5_cols() * np.dtype(src.dtype).itemsize
        ch = getattr(src.node, "chunks", None)
        per = int(ch[0]) if ch else 1
        if s >= per:                                       # each kept row in a chunk of its own
            return _grid_count(lo, hi, s) * min(per, max(hi - lo, 1)) * row
        return (hi - lo) * row

    def _overview(self, w, n) -> Overview:
        """Parquet: from row-group statistics. h5ad: from `indptr`. Neither reads data pages."""
        if _is_parquet(self.src):
            d = self._describe()
            x, size, rows = [], [], []
            off = 0
            for f in d["files"]:
                for rg in f["row_groups"]:
                    lo, hi = off, off + rg["num_rows"]
                    off = hi
                    if hi <= w[0] or lo >= w[1]:
                        continue
                    x.append((lo + hi) / 2)
                    size.append(rg["uncompressed_bytes"] / max(rg["num_rows"], 1))
                    rows.append(rg["num_rows"])
            ov = Overview(kind="rows", window=w, x=np.array(x),
                          series={"bytes_per_row": np.array(size)},
                          counts=np.array(rows), cheap=True,
                          source_bytes=sum(len(str(f)) for f in d["files"]))
            # add per-column min/max spans where statistics exist
            for s in self.src.stats():
                if s.get("min") is not None and isinstance(s["min"], (int, float)):
                    ov.series[f"{s['column']}:span"] = np.full(
                        len(x), float(s["max"]) - float(s["min"]))
            return ov
        # h5ad: nnz per cell is free from indptr
        ip = np.asarray(self.src.indptr)
        nnz = np.diff(ip)[w[0]:w[1]]
        if nnz.size == 0:
            return Overview("cells", w, np.array([]), {}, cheap=True)
        k = max(1, math.ceil(nnz.size / n))
        pad = (-nnz.size) % k
        arr = np.concatenate([nnz, np.zeros(pad, nnz.dtype)]).reshape(-1, k)
        cnt = np.full(arr.shape[0], k)
        if pad:
            cnt[-1] = k - pad
        return Overview("cells", w, w[0] + (np.arange(arr.shape[0]) + 0.5) * k,
                        {"genes_detected": arr.sum(1) / cnt,
                         "max_in_bucket": arr.max(1)},
                        counts=cnt, cheap=True, source_bytes=ip.nbytes)

    def _chunks(self, w, stride):
        import pyarrow as pa
        lo, hi = int(w[0]), int(w[1])
        if _is_parquet(self.src):
            for p in self._pq_pieces(lo, hi, stride, self._chunk_cap()):
                yield _Accounted(pa.Table.from_batches([p]))
            return
        if _is_h5(self.src):
            for _first, X in self._h5_blocks(lo, hi, stride, self._chunk_cap()):
                yield _Accounted(X)
            return
        yield from self._legacy(lo, hi, stride)

    def _legacy(self, lo, hi, stride):
        """Any other row source yielding {'X', 'obs_start', 'obs_end'} chunks: read in order,
        stop after the window."""
        for ch in self.src:
            if ch["obs_end"] <= lo:
                continue
            if ch["obs_start"] >= hi:
                break
            a, b = max(lo, ch["obs_start"]), min(hi, ch["obs_end"])
            first = _grid_first(a, lo, stride)
            if first < b:
                yield ch["X"][first - ch["obs_start"]: b - ch["obs_start"]: stride]

    def _load(self, w, stride):
        import pyarrow as pa
        lo, hi = int(w[0]), int(w[1])
        if _is_parquet(self.src):                             # Parquet Dataset
            self._hold(int(_grid_count(lo, hi, stride) * self.extent.unit_bytes))
            pieces, acc = [], 0
            for p in self._pq_pieces(lo, hi, stride, self._chunk_cap()):
                acc += _arrow_bytes(p)
                if acc > self._lease.bytes:                   # denser than the footers said
                    self._lease.add(acc - self._lease.bytes)
                pieces.append(p)
            t = (pa.Table.from_batches(pieces) if pieces
                 else self._arrow_schema().empty_table())
            self._settle(acc)
            return t
        if _is_h5(self.src):
            return self._h5_load(lo, hi, stride)
        import scipy.sparse as sp
        rows = list(self._legacy(lo, hi, stride))
        return sp.vstack(rows) if rows else None


# ---------------------------------------------------------------------------
# Sequence files
# ---------------------------------------------------------------------------

def _fastx_schema(with_qual: bool):
    import pyarrow as pa
    f = [pa.field("id", pa.utf8(), False), pa.field("seq", pa.utf8(), False),
         pa.field("length", pa.int32(), False)]
    if with_qual:
        f.append(pa.field("qual", pa.utf8(), True))
    return pa.schema(f)


def _utf8(values: list):
    """bytes -> a utf8 Arrow array, invalid bytes replaced the way the Rust reader does."""
    import pyarrow as pa
    try:
        return pa.array(values, pa.binary()).cast(pa.utf8())
    except (pa.ArrowInvalid, UnicodeDecodeError):
        return pa.array([v.decode("utf-8", "replace") for v in values], pa.utf8())


_COMPRESSED = (".gz", ".bgz", ".zst", ".bz2", ".xz")


class SeqView(View):
    """A view over a FASTA/FASTQ, backed by a SeqIndex sidecar.

    Zoom is re-aggregation of index buckets — instant at any level, no file access. Only
    `load()` touches the file, and only from the index checkpoint at or before the window: it
    seeks there, skips records up to the window, and stops after it. A compressed file cannot
    seek, so it is read from the start, still stopping after the window.

    The index's checkpoint offsets leave out a fixed number of bytes per record (the record
    separator), which the view measures once from the file size and confirms for every
    checkpoint it uses by counting the records between it and the next one; when that does not
    hold, it reads from the start instead of guessing.
    """

    def __init__(self, index, path: Optional[str] = None, budget: Optional[Budget] = None, *,
                 owns_source: bool = True, chunk_bytes=None):
        self.index = index
        self.path = path or index.path
        self._calib = None              # bytes per record the checkpoints miss; False: cannot seek
        self._verified: dict = {}
        self._ck = None
        super().__init__(budget, owns_source=owns_source, chunk_bytes=chunk_bytes)

    def _extent(self) -> Extent:
        return Extent("records", (int(self.index.n_records),),
                      self.index.bytes_per_record + (20 if self.index.has_quality else 12))

    def _full_window(self):
        return (0, self.extent.size[0])

    # ---- seeking ----
    def _checkpoints(self):
        if self._ck is None:
            b = self.index.zoom(0, None, 1 << 30)            # every bucket
            self._ck = (np.array([d["start"] for d in b], np.int64),
                        np.array([d["n"] for d in b], np.int64),
                        np.array([d["byte_offset"] for d in b], np.int64))
        return self._ck

    def _calibrate(self):
        if self._calib is None:
            self._calib = False
            ix = self.index
            n = int(ix.n_records)
            try:
                size = os.path.getsize(self.path)
            except OSError:
                size = -1
            if (ix.seekable and n > 0 and size == int(ix.file_bytes)
                    and not str(self.path).lower().endswith(_COMPRESSED)):
                rem = size - int(round(ix.bytes_per_record * n))
                if rem >= 0 and rem % n == 0:
                    self._calib = rem // n
        return self._calib

    def _seek_plan(self, lo: int):
        """(checkpoint, its first record, byte offset, next checkpoint's offset, records in it),
        or None when the read starts at the beginning of the file."""
        d = self._calibrate()
        if d is False or lo <= 0:
            return None
        starts, ns, offs = self._checkpoints()
        c = int(np.searchsorted(starts, lo, side="right")) - 1
        if c <= 0:
            return None
        t = int(offs[c]) + d * int(starts[c])
        t_next = (int(offs[c + 1]) + d * int(starts[c + 1]) if c + 1 < len(starts)
                  else int(self.index.file_bytes))
        return c, int(starts[c]), t, t_next, int(ns[c])

    def _verify(self, c: int, t: int, t_next: int, n_c: int) -> bool:
        """Is `t` the first byte of record `starts[c]`? Counted, not assumed: the bytes up to the
        next checkpoint must hold exactly `n_c` records and end on a line end."""
        if c in self._verified:
            return self._verified[c]
        fastq = bool(self.index.has_quality)
        ok = False
        try:
            with open(self.path, "rb") as fh:
                fh.seek(t - 1)
                if fh.read(2) != (b"\n@" if fastq else b"\n>"):
                    raise ValueError
                fh.seek(t)
                remaining, nl, hdr, last, cr = t_next - t, 0, 0, b"\n", 0
                with _Hold(self.budget, self.tier, 2 * min(remaining, 1 << 18) + 64):
                    while remaining > 0:
                        buf = fh.read(min(remaining, 1 << 18))
                        if not buf:
                            break
                        remaining -= len(buf)
                        a = np.frombuffer(buf, np.uint8)
                        nl += int(np.count_nonzero(a == 10))
                        cr += int(np.count_nonzero(a == 13))
                        if not fastq:            # a header is a '>' that starts a line
                            hdr += int(np.count_nonzero((a[1:] == 62) & (a[:-1] == 10)))
                            hdr += int(last == b"\n" and buf[:1] == b">")
                        last = buf[-1:]
                        a = None
                ok = (remaining == 0 and last == b"\n" and cr == 0
                      and (nl == 4 * n_c if fastq else hdr == n_c))
        except (OSError, ValueError):
            ok = False
        self._verified[c] = ok
        return ok

    def _read_plan(self, lo: int, hi: int) -> dict:
        p = self._seek_plan(lo)
        per = int(self.index.file_bytes) / max(int(self.index.n_records), 1)
        if p is not None and self._verified.get(p[0], True):
            c, rec0, t, t_next, n_c = p
            end = max(hi, rec0 + n_c)                       # the checkpoint is counted once
            return {"seek": "index", "seek_byte": t, "seek_from_record": rec0,
                    "skip_records": lo - rec0, "read_bytes": int((end - rec0) * per)}
        why = ("compressed: read from the start" if (str(self.path).lower().endswith(_COMPRESSED)
                                                       or not self.index.seekable)
               else "from the start")
        return {"seek": why, "seek_byte": 0, "seek_from_record": 0, "skip_records": lo,
                "read_bytes": int(hi * per)}

    # ---- reading ----
    def _pieces(self, lo: int, hi: int, s: int, chunk: int):
        """Kept records of [lo, hi), batch by batch, each leased while it is handed over."""
        p = self._seek_plan(lo)
        if p is not None and self._verify(p[0], p[2], p[3], p[4]):
            yield from self._parse_from(p[2], p[1], lo, hi, s, chunk)
            return
        yield from self._scan(lo, hi, s, chunk)

    def _scan(self, lo: int, hi: int, s: int, chunk: int):
        """The Rust reader from the start of the file (each batch leased by it), stopping after hi."""
        import pyarrow as pa
        from ._core import FastxStream
        bpr = max(float(self.index.bytes_per_record), 1.0)     # the reader's target is payload;
        payload = int(chunk * bpr / (bpr + 16))                 # Arrow adds offsets per record
        st = FastxStream(self.path, str(max(payload, 1)), 0, self.budget)
        seen = 0
        for batch in st:
            a = seen
            seen += batch.num_rows
            if seen > lo:
                start, end = max(a, lo), min(seen, hi)
                first = _grid_first(start, lo, s)
                if first < end:
                    if s == 1 and start == a and end == seen:
                        yield _Accounted(pa.Table.from_batches([batch]))
                    else:
                        idx = np.arange(first - a, end - a, s, dtype=np.int64)
                        yield _Accounted(pa.Table.from_batches([batch.take(pa.array(idx))]))
            batch = None
            if seen >= hi:
                break
        st = None

    def _parse_from(self, offset: int, rec: int, lo: int, hi: int, s: int, chunk: int):
        """Records from byte `offset` (record `rec`) onwards, parsed here: skipped up to `lo`,
        kept on the grid, stopped at `hi`. A block of the file and its lines are leased while
        they are parsed; the batch handed over is leased until the next one is asked for."""
        import pyarrow as pa
        fastq = bool(self.index.has_quality)
        schema = _fastx_schema(fastq)
        block = max(int(chunk) // 3, 1 << 12)
        carry, pending, left = b"", [], 0          # `left`: bytes carried into the next block
        with open(self.path, "rb") as fh:
            fh.seek(offset)
            eof = False
            while rec < hi and not eof:
                hold = _Hold(self.budget, self.tier, 3 * (block + left))
                try:
                    data = fh.read(block)
                    eof = not data
                    buf = carry + data if carry else data
                    data = None
                    lines = buf.split(b"\n")
                    buf = None
                    carry = b"" if eof else lines.pop()
                    if eof and lines and lines[-1] == b"":
                        lines.pop()
                    if pending:
                        lines = pending + lines
                    recs, pending = (_fastq_records(lines) if fastq
                                     else _fasta_records(lines, eof))
                    lines = None
                    n = len(recs)
                    start, end = max(rec, lo), min(rec + n, hi)
                    piece = None
                    if start < end and _grid_first(start, lo, s) < end:
                        keep = recs[_grid_first(start, lo, s) - rec:end - rec:s]
                        ids = [r[0][1:] for r in keep]
                        seqs = [r[1] for r in keep]
                        nb = sum(map(len, ids)) + sum(map(len, seqs)) + 16 * len(keep)
                        quals = None
                        if fastq:
                            quals = [r[2] for r in keep]
                            nb += sum(map(len, quals))
                        hold.resize(hold.bytes + nb)
                        cols = [_utf8(ids), _utf8(seqs),
                                pa.array(np.fromiter(map(len, seqs), np.int32, len(seqs)))]
                        if fastq:
                            cols.append(_utf8(quals))
                        piece = pa.Table.from_arrays(cols, schema=schema)
                        keep = ids = seqs = quals = cols = None
                    recs = None
                    rec += n
                    left = len(carry) + sum(map(len, pending))
                    if piece is not None:
                        hold.resize(_arrow_bytes(piece) + 3 * left)
                        yield _Accounted(piece)
                        piece = None
                finally:
                    hold.release()

    # ---- View hooks ----
    def _chunk_plan(self, w, stride):
        lo, hi = int(w[0]), int(w[1])
        cap = self._plan_cap()
        resident = _grid_count(lo, hi, stride) * self.extent.unit_bytes
        read = self._read_plan(lo, hi)["read_bytes"]
        n = max(math.ceil(max(read, 1) / cap), 1) if stride > 1 else max(math.ceil(resident / cap), 1)
        return n, min(int(resident), cap)

    def _load_need(self, w, stride, total):
        return int(total + 4 * self.extent.unit_bytes)      # the block it parses can shrink to a record

    def cost(self) -> dict:
        d = super().cost()
        rp = self._read_plan(int(self._win[0]), int(self._win[1]))
        d.update(rp)
        d["read_human"] = human(rp["read_bytes"])
        return d

    def _overview(self, w, n) -> Overview:
        b = self.index.zoom(int(w[0]), int(w[1]), n)
        if not b:
            return Overview("records", w, np.array([]), {}, cheap=True)
        x = np.array([(d["start"] + d["end"]) / 2 for d in b])
        s = {"mean_len": np.array([d["mean_len"] for d in b]),
             "gc": np.array([d["gc"] for d in b]),
             "n_frac": np.array([d["n_frac"] for d in b])}
        if self.index.has_quality:
            s["mean_q"] = np.array([np.nan if d["mean_q"] is None else d["mean_q"] for d in b])
        return Overview("records", w, x, s,
                        counts=np.array([d["n"] for d in b]), cheap=True,
                        source_bytes=self.index.n_buckets * 80)

    def _load(self, w, stride):
        """Records in the window as a pyarrow.Table: the planned output is leased first, then the
        file is read from the checkpoint before the window, keeping only the grid records."""
        import pyarrow as pa
        lo, hi = int(w[0]), int(w[1])
        self._hold(int(_grid_count(lo, hi, stride) * self.extent.unit_bytes))
        pieces, acc = [], 0
        for item in self._pieces(lo, hi, stride, self._chunk_cap()):
            p = item.data
            acc += _arrow_bytes(p)
            if acc > self._lease.bytes:                       # longer records than the index mean
                self._lease.add(acc - self._lease.bytes)
            pieces.append(p)
            p = item = None
        t = (pa.concat_tables(pieces) if len(pieces) > 1 else pieces[0] if pieces
             else _fastx_schema(bool(self.index.has_quality)).empty_table())
        self._settle(acc)
        return t

    def _chunks(self, w, stride):
        """Yield the window as budget-sized tables without ever holding all of it."""
        yield from self._pieces(int(w[0]), int(w[1]), stride, self._chunk_cap())

    def search(self, patterns, both_strands: bool = True, max_hits: int = 100_000,
               within_window: bool = True, first_per_record: bool = False) -> dict:
        """Multi-pattern search, optionally restricted to the current window."""
        from ._core import search_fastx
        lo, hi = (int(self._win[0]), int(self._win[1])) if within_window else (0, None)
        if isinstance(patterns, str):
            patterns = [patterns]
        return search_fastx(self.path, list(patterns), both_strands, max_hits,
                            lo, hi, first_per_record)

    def hit_profile(self, hits: dict, n_buckets: int = 256) -> Overview:
        """Turn search hits into a zoom-aligned density profile you can draw next to the data."""
        lo, hi = self._win
        rec = np.array([h["record"] for h in hits["hits"]], dtype=float)
        edges = np.linspace(lo, hi, n_buckets + 1)
        dens, _ = np.histogram(rec, bins=edges)
        return Overview("records", (lo, hi), (edges[:-1] + edges[1:]) / 2,
                        {"hits": dens.astype(float)}, counts=dens, cheap=False,
                        source_bytes=int(hits["bases_scanned"]))


def _fastq_records(lines: list):
    """Complete 4-line FASTQ records from `lines`, and the lines left over."""
    n = len(lines) // 4
    recs = []
    for i in range(0, 4 * n, 4):
        h, sq, p, q = lines[i], lines[i + 1], lines[i + 2], lines[i + 3]
        if h[:1] != b"@" or p[:1] != b"+" or len(sq) != len(q):
            raise ValueError(f"malformed FASTQ record at {h[:40]!r}")
        recs.append((h, sq, q))
    return recs, lines[4 * n:]


def _fasta_records(lines: list, eof: bool):
    """Complete FASTA records (header, sequence lines joined by newlines, as the Rust reader's
    raw sequence has them); the last record is complete only at the end of the file."""
    starts = [i for i, ln in enumerate(lines) if ln[:1] == b">"]
    if not starts:
        return [], lines
    if starts[0] != 0:
        raise ValueError(f"FASTA text before a header: {lines[0][:40]!r}")
    ends = starts[1:] + [len(lines)]
    done = len(starts) if eof else len(starts) - 1
    recs = [(lines[a], b"\n".join(lines[a + 1:b])) for a, b in zip(starts[:done], ends[:done])]
    return recs, ([] if eof else lines[starts[-1]:])


# ---------------------------------------------------------------------------
# Imaging
# ---------------------------------------------------------------------------

class ImageView(View):
    """A view over a tiled image: the zoom/pan model people already expect from a map.

    The window is `(y0, y1, x0, x1)` over the image's Y and X axes (found from its `axes`, as
    `TiledImageReader` does); every other axis — channels, samples — rides along. Zoom
    decimates pixels on the global grid, and reads go through `window(..., step)`, so only the
    tiles holding a kept pixel are decoded and only the kept pixels are allocated. Cost
    accounts honestly for the fact that a TIFF tile must be decoded whole even if you keep one
    pixel in sixteen — `decode_bytes` is what actually gets touched, `resident_bytes` is what
    you keep.
    """

    def __init__(self, array, budget: Optional[Budget] = None, tier: str = "ram", *,
                 owns_source: bool = True, chunk_bytes=None, axes: Optional[str] = None):
        self.a = array
        self.shape = tuple(int(n) for n in array.shape)
        self.axes, self.y_axis, self.x_axis = _spatial_axes(array, self.shape, axes)
        self.h, self.w = self.shape[self.y_axis], self.shape[self.x_axis]
        self.dtype = np.dtype(array.dtype)
        self.trailing = int(np.prod([n for i, n in enumerate(self.shape)
                                     if i not in (self.y_axis, self.x_axis)] or [1]))
        th, tw = getattr(array, "th", None), getattr(array, "tw", None)
        if th is None and getattr(array, "chunks", None) is not None:
            try:
                th, tw = array.chunks[self.y_axis], array.chunks[self.x_axis]
            except Exception:
                th = tw = None
        self.th = int(th or 512)
        self.tw = int(tw or 512)
        super().__init__(budget, tier, owns_source=owns_source, chunk_bytes=chunk_bytes)

    def _extent(self) -> Extent:
        return Extent("pixels", (self.h, self.w), self.dtype.itemsize * self.trailing)

    def _full_window(self):
        return (0, self.h, 0, self.w)

    @property
    def _px(self) -> int:
        return int(self.dtype.itemsize * self.trailing)

    @property
    def _tile_bytes(self) -> int:
        """One decoded tile: the transient a read goes through."""
        return int(self.th * self.tw * self._px)

    def _read(self, y0: int, y1: int, x0: int, x1: int, s: int) -> np.ndarray:
        """Rows y0:y1:s and columns x0:x1:s, every other axis whole, decoding only the tiles that
        hold a kept pixel."""
        a = self.a
        if hasattr(a, "window"):
            return np.asarray(a.window(y0, y1, x0, x1, s))
        key = [slice(None)] * len(self.shape)
        key[self.y_axis], key[self.x_axis] = slice(y0, y1, s), slice(x0, x1, s)
        return np.asarray(a[tuple(key)])

    @staticmethod
    def _runs(lo: int, hi: int, s: int, tile: int) -> list:
        """[(first, end, count)] per tile of an axis holding grid coordinates of [lo, hi)."""
        out = []
        if hi <= lo:
            return out
        for t in range(lo // tile, (hi - 1) // tile + 1):
            a, b = max(t * tile, lo), min((t + 1) * tile, hi)
            f = _grid_first(a, lo, s)
            if f < b:
                k = _grid_count(f, b, s)
                out.append((f, f + (k - 1) * s + 1, k))
        return out

    def _groups(self, w, s: int, cap: int) -> list:
        """Rectangles of whole tiles (the ones holding a kept pixel) whose decimated output fits
        `cap` — the chunks of `chunks()`, so the count `cost()` reports is the count yielded."""
        y0, y1, x0, x1 = w
        ry, rx = self._runs(y0, y1, s, self.th), self._runs(x0, x1, s, self.tw)
        if not ry or not rx:
            return []
        px = self._px
        tall = max(r[2] for r in ry)
        bands, cur, n = [], [], 0
        for r in rx:
            if cur and (n + r[2]) * tall * px > cap:
                bands.append(cur)
                cur, n = [], 0
            cur.append(r)
            n += r[2]
        bands.append(cur)
        out = []
        for band in bands:
            cols = sum(r[2] for r in band)
            cur, n = [], 0
            for r in ry:
                if cur and (n + r[2]) * cols * px > cap:
                    out.append((cur[0][0], cur[-1][1], band[0][0], band[-1][1], n, cols))
                    cur, n = [], 0
                cur.append(r)
                n += r[2]
            out.append((cur[0][0], cur[-1][1], band[0][0], band[-1][1], n, cols))
        return out

    def _tiles_touched(self, w, s: int = 1) -> int:
        y0, y1, x0, x1 = w
        return len(self._runs(y0, y1, s, self.th)) * len(self._runs(x0, x1, s, self.tw))

    def _chunk_plan(self, w, stride):
        g = self._groups(w, stride, self._plan_cap())
        if not g:
            return 1, 0
        return len(g), max(ny * nx for *_, ny, nx in g) * self._px

    def _load_need(self, w, stride, total):
        return int(total + self._tile_bytes)

    def _chunk_need(self, w, stride, per_chunk):
        return int(per_chunk + self._tile_bytes)

    def cost(self) -> dict:
        d = super().cost()
        t = self._tiles_touched(self._win, self.stride)
        d["tiles_touched"] = t
        d["decode_bytes"] = int(t * self._tile_bytes)
        d["decode_human"] = human(d["decode_bytes"])
        d["read_bytes"], d["read_human"] = d["decode_bytes"], d["decode_human"]
        return d

    def _out_shape(self, ny: int, nx: int) -> tuple:
        sh = list(self.shape)
        sh[self.y_axis], sh[self.x_axis] = ny, nx
        return tuple(sh)

    def _overview(self, w, n) -> Overview:
        """A thumbnail of the window with at most `n` pixels per side, read with a step — only
        the kept pixels are allocated and only the tiles holding them decoded."""
        y0, y1, x0, x1 = w
        s = max(1, math.ceil(max(y1 - y0, x1 - x0) / max(int(n), 1)))
        ny, nx = _grid_count(y0, y1, s), _grid_count(x0, x1, s)
        need = ny * nx * self._px + self._tile_bytes
        if self.budget is not None and need > self._free():
            raise CostError(f"a {ny}x{nx} thumbnail needs {human(need)} but {human(self._free())} "
                            f"is free; ask for fewer buckets or raise the '{self.tier}' budget")
        with _Hold(self.budget, self.tier, need):
            img = self._read(y0, y1, x0, x1, s)
            if img.ndim > 2:                       # one plane to look at: RGB stays, channels fold
                rgb = self.axes.endswith("S") and img.shape[-1] in (3, 4) and img.ndim == 3
                if not rgb:
                    keep = (self.y_axis, self.x_axis)
                    img = img.max(axis=tuple(i for i in range(img.ndim) if i not in keep))
        return Overview("pixels", w, np.arange(img.shape[1]),
                        {"thumbnail": img}, cheap=False,
                        source_bytes=int(self._tiles_touched(w, s) * self._tile_bytes))

    def _load(self, w, stride):
        """The window at the current zoom: the output is leased before anything is read, then
        filled straight from the tiles that hold a kept pixel (one decoded tile at a time)."""
        y0, y1, x0, x1 = w
        out = _grid_count(y0, y1, stride) * _grid_count(x0, x1, stride) * self._px
        self._hold(out)
        with _Hold(self.budget, self.tier, self._tile_bytes):
            a = self._read(y0, y1, x0, x1, stride)
        self._settle(a.nbytes)
        return a

    def _chunks(self, w, stride):
        for (ya, yb, xa, xb, ny, nx) in self._groups(w, stride, self._chunk_cap()):
            with _Hold(self.budget, self.tier, ny * nx * self._px + self._tile_bytes) as hold:
                a = self._read(ya, yb, xa, xb, stride)
                hold.resize(a.nbytes)
                yield _Accounted(a)
                a = None

    def tiles(self) -> Iterator[tuple[np.ndarray, tuple]]:
        """Stream the window tile by tile at the current zoom, so a preview can build up. Every
        tile holds the window's grid pixels (`(y - y0) % stride == 0`), so the tiles fit together
        into exactly what `load()` returns. Each tile is accounted until the next is asked for."""
        y0, y1, x0, x1 = self._win
        s = self.stride
        try:
            for (fy, ey, _ny) in self._runs(y0, y1, s, self.th):
                for (fx, ex, _nx) in self._runs(x0, x1, s, self.tw):
                    self._drop_hold()
                    with _Hold(self.budget, self.tier, self._tile_bytes):
                        self._hold(_grid_count(fy, ey, s) * _grid_count(fx, ex, s) * self._px)
                        a = self._read(fy, ey, fx, ex, s)
                    yield a, (fy, ey, fx, ex)
                    a = None
        finally:
            self._drop_hold()


# ---------------------------------------------------------------------------
# Progressive canvas — a picture that sharpens as chunks arrive
# ---------------------------------------------------------------------------

class ProgressiveCanvas:
    """Accumulate a 2-D density image from streamed batches, drawable at any moment.

    This is the "build a visual representation while it downloads" piece: bounds come from
    Parquet footer statistics (free), then each batch is histogrammed into a fixed grid, so
    memory is O(grid) no matter how many rows go through it.
    """

    def __init__(self, bounds: tuple[float, float, float, float], shape=(512, 512),
                 weight: bool = False):
        self.x0, self.x1, self.y0, self.y1 = map(float, bounds)
        self.shape = shape
        self.grid = np.zeros(shape, dtype=np.float64)
        self.counts = np.zeros(shape, dtype=np.int64)
        self.n_rows = 0
        self.n_dropped = 0          # rows with a null or non-finite coordinate: no cell, not in n_rows
        self.n_batches = 0
        self.weight = weight

    @classmethod
    def from_dataset(cls, ds, x: str = "x", y: str = "y", shape=(512, 512), **kw):
        """Bounds straight from column statistics — no data pages read."""
        st = {s["column"]: s for s in ds.stats()}
        for c in (x, y):
            if c not in st or st[c].get("min") is None:
                raise ValueError(f"column {c!r} has no min/max statistics to bound the canvas")
        return cls((st[x]["min"], st[x]["max"], st[y]["min"], st[y]["max"]), shape, **kw)

    def add(self, xs, ys, weights=None) -> "ProgressiveCanvas":
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        wt = None if weights is None else np.asarray(weights, dtype=np.float64)
        keep = np.isfinite(xs) & np.isfinite(ys)      # a null has no cell: counted apart, never binned
        if wt is not None:
            keep &= np.isfinite(wt)                   # a null weight would poison its cell with NaN
        if not keep.all():
            self.n_dropped += int(keep.size - np.count_nonzero(keep))
            xs, ys = xs[keep], ys[keep]
            wt = None if wt is None else wt[keep]
        h, w = self.shape
        # floor, then clip: each cell gets an equal share of the range, and x1 falls in the last cell
        ix = np.clip(np.floor((xs - self.x0) / max(self.x1 - self.x0, 1e-9) * w), 0, w - 1).astype(np.int64)
        iy = np.clip(np.floor((ys - self.y0) / max(self.y1 - self.y0, 1e-9) * h), 0, h - 1).astype(np.int64)
        flat = iy * w + ix
        if wt is None:
            np.add.at(self.counts.reshape(-1), flat, 1)
            np.add.at(self.grid.reshape(-1), flat, 1.0)
        else:
            np.add.at(self.counts.reshape(-1), flat, 1)
            np.add.at(self.grid.reshape(-1), flat, wt)
        self.n_rows += xs.size
        self.n_batches += 1
        return self

    def add_batch(self, batch, x: str = "x", y: str = "y", weight: Optional[str] = None):
        cols = batch.schema.names
        for c in (x, y):
            if c not in cols:
                raise ValueError(f"batch has no column {c!r}; has {cols}")
        return self.add(batch.column(x).to_numpy(zero_copy_only=False),
                        batch.column(y).to_numpy(zero_copy_only=False),
                        batch.column(weight).to_numpy(zero_copy_only=False) if weight else None)

    @property
    def image(self) -> np.ndarray:
        return self.grid

    @property
    def coverage(self) -> float:
        """Fraction of grid cells that have received at least one point."""
        return float((self.counts > 0).mean())

    @property
    def nbytes(self) -> int:
        return self.grid.nbytes + self.counts.nbytes

    def __repr__(self) -> str:
        return (f"ProgressiveCanvas({self.shape[0]}x{self.shape[1]}, {self.n_rows:,} rows in "
                f"{self.n_batches} batches, {self.coverage:.1%} covered, "
                f"{human(self.nbytes)} resident)")
