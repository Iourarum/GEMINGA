"""Several datasets open at once, each in its own budget partition.

## Why this is not a container

The obvious reading of "isolate each dataset" is a container: Docker, Apptainer, a FreeBSD
jail. For GEMINGA that would be the wrong tool, and it is worth saying why rather than
quietly not doing it.

A jail isolates at the OS level — namespaces, a chroot, resource limits enforced by the
kernel. What GEMINGA actually needs from that list is the resource limit, and it already has
one: the budget governor is a cgroup-like hard bound that costs an atomic add. The other
two-thirds of what a container gives you are things this library specifically must *not*
have. Its whole performance argument is that a decoded Arrow batch crosses into the user's
Python process with no copy. Put the reader behind a namespace boundary and every batch has
to be serialized or handed through shared memory that must be set up and torn down. A
container start is tens to hundreds of milliseconds; navigating an index here is under a
millisecond. Isolation of that kind would cost the thing it is meant to protect.

So what is worth borrowing from jails is the *model*, not the mechanism:

- **A partitioned resource bound.** A cell reserves from the workspace budget on open and
  returns it on close. Nothing it does can starve a sibling. This is the jail's memory limit,
  implemented as nested accounting.
- **A namespace.** Cells are addressed by name, not by whatever variable happens to hold
  them. No global state, no accidental sharing.
- **A lifecycle.** `open`, `suspend`, `resume`, `close` — with resources genuinely released
  at each step, not just dereferenced.
- **Failure containment.** A reader that raises marks its own cell failed; siblings keep
  working and the workspace stays usable.

Real OS-level isolation remains available where it actually earns its cost — `run_isolated()`
forks a subprocess under an address-space rlimit, which is the right answer for a reader you
do not trust to stay inside its budget (a C library with a leak, an experimental parser).
It is opt-in per call because it is slow, and slow is the honest trade there.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from ._core import Budget, human, parse_size

__all__ = ["Cell", "Workspace", "CellState", "workspace"]


def try_acquire(budget: Budget, tier: str, nbytes: int):
    """Reserve `nbytes` of `tier` if they are free right now: the lease, or None.

    `Budget.acquire` waits for a tier that is merely full, up to the budget's timeout. That is
    right for a reader whose sibling hands a chunk back a moment later and wrong for a
    *reservation* — a cell's slice, a staging area, a device pool, a read cache — which only a
    decision by the caller can free: the wait is two minutes of nothing followed by the same
    MemoryError. The core's `Budget.try_acquire` makes the check and the grant one locked step,
    so two callers cannot both see the same free bytes; the two-call fallback is kept for a
    budget-like object without it.
    """
    n = int(nbytes)
    if hasattr(budget, "try_acquire"):             # the core's one-lock check-and-take (0.17+)
        return budget.try_acquire(tier, str(n))
    if n > budget.available(tier):
        return None
    return budget.acquire(tier, str(n))


def _apply_position(view, saved: dict) -> None:
    """Put a freshly opened view back where `saved` says it was: bin, genes, window, zoom.

    The order matters: genes are checked against the bin in force, and a window is clamped
    against the extent. A cell restores through this after a suspend, and `run_isolated` replays
    through it in the child, so the two cannot drift apart.
    """
    if not saved or view is None:
        return
    if saved.get("bin") is not None and hasattr(view, "use_bin"):
        view.use_bin(saved["bin"])
    if saved.get("genes") and hasattr(view, "select_genes"):
        view.select_genes(saved["genes"])
    if saved.get("window"):
        view.select(*saved["window"])
    if saved.get("zoom"):
        view.set_zoom(saved["zoom"])


class CellState:
    DECLARED = "declared"
    OPEN = "open"
    SUSPENDED = "suspended"
    CLOSED = "closed"
    FAILED = "failed"


@dataclass
class CellStats:
    opened_at: Optional[float] = None
    opens: int = 0
    loads: int = 0
    bytes_peak: int = 0
    errors: int = 0
    last_error: Optional[str] = None


class Cell:
    """One dataset, held inside its own slice of the workspace budget.

    A cell is lazy: declaring it costs a reservation and nothing else. The underlying reader
    opens on first use and can be suspended — freeing file handles and caches — without
    losing the window, zoom or gene selection you had set.

    The slice is a real reservation. It is taken without waiting (a slice that is not free now
    will not be freed by waiting for it), it is given back by `close()` or by
    `suspend(release_reservation=True)`, and touching a cell that was parked without its slice
    takes the slice again — or raises MemoryError and leaves the cell parked.
    """

    def __init__(self, name: str, target: Any, workspace: "Workspace",
                 ram: str = "1GB", vram: str = "0", spill: str = "0",
                 opener: Optional[Callable] = None, note: str = "", **open_kw):
        self.name = name
        self.target = target
        self.ws = workspace
        self.note = note
        self.open_kw = open_kw
        self._opener = opener
        self.state = CellState.DECLARED
        self.stats = CellStats()
        self._view = None
        self._saved: dict = {}
        self._lock = threading.RLock()

        self.caps = {"ram": parse_size(ram), "vram": parse_size(vram),
                     "spill": parse_size(spill)}
        self._asked = {"ram": ram, "vram": vram, "spill": spill}
        # Reserve from the parent budget now, so a cell can never be opened that the
        # workspace could not afford. This is the jail's memory limit.
        self._leases: dict = {}
        self._reserve()
        self.budget = Budget(ram=ram, vram=vram, spill=spill,
                             timeout_s=workspace.timeout_s)

    # ---- lifecycle ----
    def _reserve(self) -> None:
        """Take this cell's whole slice from the workspace — every tier or none of them.

        Never waits. If a tier cannot give its share, whatever was already taken for this
        attempt goes back before the MemoryError leaves, so a failed attempt holds nothing and
        the next one starts clean.
        """
        got: dict = {}
        try:
            with self.ws._lock:                   # check-then-acquire must not interleave
                for tier, n in self.caps.items():
                    if not n:
                        continue
                    lease = try_acquire(self.ws.budget, tier, n)
                    if lease is None:
                        free = human(self.ws.budget.available(tier))
                        have = f"{free} free" if tier == "ram" else f"{free} of {tier} free"
                        raise MemoryError(
                            f"cannot reserve {self._asked[tier]} {tier} for cell {self.name!r}: "
                            f"the workspace has {have}. Close a cell, or give this one a "
                            f"smaller slice.")
                    got[tier] = lease
        except BaseException:
            for lease in got.values():
                lease.release()
            raise
        self._leases = got

    def _release_leases(self):
        leases, self._leases = self._leases, {}
        for lease in leases.values():
            try:
                lease.release()
            except Exception:
                pass

    def reserved(self, tier: str = "ram") -> int:
        """Bytes of `tier` this cell holds from the workspace right now — zero while parked
        without its slice, whatever the nominal cap says."""
        lease = self._leases.get(tier)
        return int(lease.bytes) if lease is not None else 0

    @property
    def view(self):
        """The underlying view, opened on first access.

        Any touch of a cell that is not open is a resume: the slice is taken again if it was
        given back, and the reader is reopened where it was. Safe to touch from several threads;
        the reader is opened once.
        """
        v = self._view
        if v is not None:
            return v
        with self._lock:
            if self._view is None:
                self._open()
            return self._view

    def _open(self):
        if self.state == CellState.CLOSED:
            raise RuntimeError(f"cell {self.name!r} is closed")
        if not self._leases and any(self.caps.values()):
            self._reserve()                            # MemoryError leaves the state as it was
        import geminga as g
        try:
            fn = self._opener or g.view
            self._view = fn(self.target, budget=self.budget, **self.open_kw)
            self._restore()
            self.state = CellState.OPEN
            self.stats.opens += 1
            self.stats.opened_at = time.time()
        except Exception as e:                       # containment: the cell fails, not
            self._close_view()                       # the workspace — and a half-restored
            self.state = CellState.FAILED            # reader is closed, not left to be reused
            self.stats.errors += 1
            self.stats.last_error = f"{type(e).__name__}: {e}"
            raise

    def _save(self):
        v = self._view
        if v is None:
            return
        genes = getattr(v, "genes", None)
        self._saved = {"window": getattr(v, "window", None),
                       "zoom": getattr(v, "zoom", 0),
                       "genes": list(genes) if isinstance(genes, (list, tuple)) else None,
                       "bin": getattr(v, "_forced_bin", None)}

    def _restore(self):
        _apply_position(self._view, self._saved)

    def suspend(self, release_reservation: bool = False) -> "Cell":
        """Close file handles and caches but keep the cell's position. Resumes transparently.

        With `release_reservation=True` the budget slice goes back to the workspace too — use
        it when you want a sibling to grow while this one is parked.
        """
        with self._lock:
            if self.state == CellState.CLOSED:
                return self
            self._save()
            self._close_view()
            if release_reservation:
                self._release_leases()
            self.state = CellState.SUSPENDED
        return self

    def resume(self) -> "Cell":
        """Reopen a parked cell where it was, taking its slice back first if it was given up.

        All or nothing: if the workspace cannot give every tier back, nothing is held and the
        cell stays parked, so a later resume starts from the same place.
        """
        with self._lock:
            _ = self.view
        return self

    def _close_view(self):
        v, self._view = self._view, None
        for obj in (v, getattr(v, "gef", None), getattr(v, "src", None)):
            close = getattr(obj, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def close(self) -> "Cell":
        with self._lock:
            self._save()
            self._close_view()
            self._release_leases()
            self.state = CellState.CLOSED
        return self

    # ---- work ----
    def cost(self) -> dict:
        return self.view.cost()

    def load(self):
        d = self.view.load()
        self.stats.loads += 1
        self.stats.bytes_peak = max(self.stats.bytes_peak,
                                    int(self.budget.stats()["ram"]["peak"]))
        return d

    def overview(self, n: int = 256):
        return self.view.overview(n)

    def run(self, fn: Callable, *a, **kw):
        """Call `fn(view, *a, **kw)`, containing any failure inside this cell."""
        try:
            out = fn(self.view, *a, **kw)
            self.stats.bytes_peak = max(self.stats.bytes_peak,
                                        int(self.budget.stats()["ram"]["peak"]))
            return out
        except Exception as e:
            self.state = CellState.FAILED
            self.stats.errors += 1
            self.stats.last_error = f"{type(e).__name__}: {e}"
            raise

    def run_isolated(self, fn: Callable, *a, timeout: float = 300, **kw):
        """Run `fn(view, *a, **kw)` in a forked subprocess under a hard address-space limit.

        This is the genuine OS-level jail, and it is opt-in because it costs what OS-level
        isolation always costs: a fork, a fresh open of the dataset, and a pickle round-trip
        for the result. Worth it when a reader might exceed its budget in ways the governor
        cannot see — a C extension that leaks, an experimental parser — and not worth it
        otherwise. Returns whatever `fn` returns, provided it pickles.

        The child starts where the cell is *now* — bin, genes, window and zoom of the live view,
        not the ones it had when it was last parked.
        """
        import multiprocessing as mp
        import resource

        with self._lock:
            self._save()
            saved = dict(self._saved)
        cap = self.caps["ram"]
        target, kwargs, opener = self.target, dict(self.open_kw), self._opener
        ram = str(cap)

        def _child(conn):
            try:
                # RLIMIT_AS bounds the *whole* address space, and a forked child already
                # carries the parent's interpreter, numpy, h5py and Arrow mappings. Setting
                # the raw cap here would kill the child before it did any work. What the
                # caller means is "this much more", so measure the baseline and add to it.
                base = 0
                try:
                    with __import__("builtins").open("/proc/self/status") as fh:
                        for line in fh:
                            if line.startswith("VmSize:"):
                                base = int(line.split()[1]) * 1024
                                break
                except OSError:
                    base = 512 * 1024 ** 2
                hard = base + cap + (256 * 1024 ** 2)      # margin for the fresh reader
                resource.setrlimit(resource.RLIMIT_AS, (hard, hard))
                import geminga as g
                b = Budget(ram=ram)
                v = (opener or g.view)(target, budget=b, **kwargs)
                _apply_position(v, saved)
                # send() pickles here, in this thread, so a result that cannot be pickled is
                # reported as that rather than lost in a feeder thread.
                conn.send(("ok", fn(v, *a, **kw)))
            except BaseException as e:                    # MemoryError included, on purpose
                try:
                    conn.send(("err", f"{type(e).__name__}: {e}", traceback.format_exc()[-2000:]))
                except BaseException:
                    pass                                  # the pipe closes; the parent says so
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
        recv, send = ctx.Pipe(duplex=False)
        p = ctx.Process(target=_child, args=(send,))
        try:
            p.start()
        except BaseException:
            recv.close()
            send.close()
            raise
        send.close()              # ours: with it closed, a child that dies leaves EOF, not silence
        deadline = None if timeout is None else time.monotonic() + timeout
        msg = None
        try:
            # The result is read *before* the child is joined. A child that is still writing a
            # result bigger than the pipe's buffer cannot exit until somebody reads it, so
            # joining first sat out the whole timeout for anything over ~64 KB.
            while True:
                left = None if deadline is None else deadline - time.monotonic()
                if recv.poll(0.25 if left is None else max(0.0, min(left, 0.25))):
                    try:
                        msg = recv.recv()
                    except (EOFError, OSError):
                        msg = None                        # died before or while answering
                    break
                if not p.is_alive():
                    if recv.poll(0):
                        continue                          # it answered and then exited
                    break
                if left is not None and left <= 0:
                    p.terminate()
                    raise TimeoutError(
                        f"isolated run in cell {self.name!r} exceeded {timeout}s")
        finally:
            recv.close()
            p.join(10)
            if p.is_alive():
                p.terminate()
                p.join()
        if msg is None:
            raise RuntimeError(
                f"isolated run in cell {self.name!r} died without a result "
                f"(exit {p.exitcode}) — most likely it exceeded its {human(cap)} "
                f"address-space allowance, which is exactly what the allowance is for")
        if msg[0] == "ok":
            return msg[1]
        self.stats.errors += 1
        self.stats.last_error = msg[1]
        raise RuntimeError(f"isolated run in cell {self.name!r} failed: {msg[1]}\n{msg[2]}")

    # ---- introspection ----
    def report(self) -> dict:
        s = self.budget.stats()
        return {
            "name": self.name,
            "target": str(self.target),
            "state": self.state,
            "note": self.note,
            "ram_cap": self.caps["ram"],
            "ram_cap_human": human(self.caps["ram"]),
            "ram_reserved": self.reserved("ram"),
            "ram_peak": int(s["ram"]["peak"]),
            "ram_peak_human": human(int(s["ram"]["peak"])),
            "ram_used": int(s["ram"]["used"]),
            "grants": int(s["ram"]["grants"]),
            "waits": int(s["ram"]["waits"]),
            "denials": int(s["ram"]["denials"]),
            "loads": self.stats.loads,
            "errors": self.stats.errors,
            "last_error": self.stats.last_error,
            "window": self._window(),
        }

    def _window(self):
        """Where the cell is: the live view's window if it is open, else where it was parked."""
        v = self._view
        if v is not None:
            w = getattr(v, "window", None)
            if w is not None:
                return w
        return self._saved.get("window")

    def __enter__(self):
        return self

    def __exit__(self, *e):
        self.suspend()
        return False

    def __repr__(self) -> str:
        r = self.report()
        return (f"Cell({self.name!r}, {self.state}, {r['ram_cap_human']} slice, "
                f"peak {r['ram_peak_human']})")


class Workspace:
    """Several cells under one budget ceiling.

    The workspace is the thing a student actually sits in front of: three datasets partially
    open, each bounded, none able to starve the others, all addressable by name.

        ws = g.workspace(ram="8GB")
        ws.add("chip", "run1.gef", ram="3GB", note="tumour section")
        ws.add("ref",  "atlas.h5ad", ram="2GB")
        ws.add("reads", "run1.fastq.gz", ram="1GB")

        ws["chip"].view.select(12000, 16000, 9000, 13000)
        ws.print_report()
    """

    def __init__(self, ram: str = "4GB", vram: str = "0", spill: str = "0",
                 timeout_s: float = 120, name: str = "workspace",
                 catalog: Optional["Any"] = None):
        self.name = name
        self.timeout_s = timeout_s
        self.budget = Budget(ram=ram, vram=vram, spill=spill, timeout_s=timeout_s)
        self.cells: dict[str, Cell] = {}
        self.catalog = catalog
        self._lock = threading.RLock()       # membership, and check-then-acquire on the budget

    # ---- membership ----
    def add(self, name: str, target: Any, ram: str = "1GB", vram: str = "0",
            spill: str = "0", opener: Optional[Callable] = None, note: str = "",
            **open_kw) -> Cell:
        with self._lock:
            if name in self.cells and self.cells[name].state != CellState.CLOSED:
                raise ValueError(f"cell {name!r} already exists; close it first or pick "
                                 f"another name")
            cell = Cell(name, target, self, ram, vram, spill, opener, note, **open_kw)
            self.cells[name] = cell
        if self.catalog is not None:
            try:
                self.catalog.note_opened(name, target, ram, note)
            except Exception:
                pass
        return cell

    def open(self, name: str, target: Any, **kw) -> Cell:
        """Add a cell and open it immediately."""
        c = self.add(name, target, **kw)
        _ = c.view
        return c

    def __getitem__(self, name: str) -> Cell:
        if name not in self.cells:
            raise KeyError(f"no cell {name!r}; have {list(self.cells)}")
        return self.cells[name]

    def __contains__(self, name: str) -> bool:
        return name in self.cells

    def __iter__(self) -> Iterator[Cell]:
        return iter(self.cells.values())

    def __len__(self) -> int:
        return len(self.cells)

    def drop(self, name: str) -> "Workspace":
        cell = self[name]
        cell.close()                       # outside our lock: a cell takes its own lock, then ours
        with self._lock:
            if self.cells.get(name) is cell:
                del self.cells[name]
        return self

    def close(self) -> "Workspace":
        for c in list(self.cells.values()):
            c.close()
        return self

    # ---- working across cells ----
    def active(self) -> list[Cell]:
        return [c for c in self.cells.values()
                if c.state in (CellState.OPEN, CellState.DECLARED, CellState.SUSPENDED)]

    def broadcast(self, fn: Callable, only: Optional[list[str]] = None,
                  skip_failed: bool = True) -> dict:
        """Run `fn(cell)` over every cell, collecting results and containing failures.

        Returns `{name: result}` for successes and puts exceptions in `.errors`.
        """
        out, errs = {}, {}
        for c in self.cells.values():
            if only and c.name not in only:
                continue
            if skip_failed and c.state in (CellState.FAILED, CellState.CLOSED):
                continue
            try:
                out[c.name] = fn(c)
            except Exception as e:
                errs[c.name] = f"{type(e).__name__}: {e}"
        self.errors = errs
        return out

    def select_frac(self, *fracs, only: Optional[list[str]] = None) -> dict:
        """Put every cell at the same *relative* position in its own extent.

        Datasets rarely share coordinates, but they often share structure — the middle of a
        run, the same quartile of a chip. This is the cheapest useful form of alignment.
        """
        return self.broadcast(lambda c: c.view.select_frac(*fracs).cost(), only=only)

    def select(self, *bounds, only: Optional[list[str]] = None) -> dict:
        """Put every cell at the same absolute window — for cells that genuinely do share a
        coordinate frame, such as a GEF and the image registered to it."""
        return self.broadcast(lambda c: c.view.select(*bounds).cost(), only=only)

    def costs(self, only: Optional[list[str]] = None) -> dict:
        return self.broadcast(lambda c: c.cost(), only=only)

    def overviews(self, n: int = 256, only: Optional[list[str]] = None) -> dict:
        return self.broadcast(lambda c: c.overview(n), only=only)

    def total_reserved(self, tier: str = "ram") -> int:
        """What the cells hold from this workspace right now: their live leases, not their
        nominal caps — a cell parked without its slice reserves nothing."""
        return sum(c.reserved(tier) for c in list(self.cells.values()))

    # ---- reporting ----
    def report(self) -> dict:
        s = self.budget.stats()
        return {
            "workspace": self.name,
            "ram_cap": int(s["ram"]["capacity"]),
            "ram_reserved": self.total_reserved("ram"),
            "ram_free": int(self.budget.available("ram")),
            "cells": [c.report() for c in self.cells.values()],
        }

    def print_report(self) -> None:
        r = self.report()
        print(f"{r['workspace']}: {human(r['ram_reserved'])} reserved of "
              f"{human(r['ram_cap'])}, {human(r['ram_free'])} free")
        if not r["cells"]:
            print("  (no cells)")
            return
        print(f"  {'cell':<12} {'state':<10} {'slice':>9} {'peak':>9} {'loads':>6}  target")
        for c in r["cells"]:
            tgt = c["target"]
            tgt = ("..." + tgt[-38:]) if len(tgt) > 41 else tgt
            print(f"  {c['name']:<12} {c['state']:<10} {c['ram_cap_human']:>9} "
                  f"{c['ram_peak_human']:>9} {c['loads']:>6}  {tgt}")
            if c["last_error"]:
                print(f"  {'':<12} └ {c['last_error'][:90]}")

    def __enter__(self):
        return self

    def __exit__(self, *e):
        self.close()
        return False

    def __repr__(self) -> str:
        states = {}
        for c in self.cells.values():
            states[c.state] = states.get(c.state, 0) + 1
        bits = ", ".join(f"{v} {k}" for k, v in sorted(states.items()))
        return (f"Workspace({self.name!r}, {len(self.cells)} cells"
                f"{' — ' + bits if bits else ''}, "
                f"{human(self.budget.available('ram'))} free)")


def workspace(ram: str = "4GB", **kw) -> Workspace:
    """Open a workspace: several datasets under one ceiling, each in its own partition."""
    return Workspace(ram=ram, **kw)
