"""Where the working set actually lands: host RAM, a GPU, an XPU, or an accelerator.

The budget governor has always had a `vram` tier that it accounts but does not allocate. This
module makes that concrete. It discovers what the machine has, reports how much is genuinely
free on each device, and hands out budget-backed allowances so a placement that would not fit
is refused before the allocator is asked.

The honesty this needs: GEMINGA does not allocate device memory. Torch, CuPy or a vendor
runtime does. What GEMINGA can do is refuse to let you over-commit, and tell you the number
before you find out the hard way. For CUDA and XPU it can also read the *real* free figure
from the driver, which is better than any estimate.

Accelerators that are not GPUs — Tenstorrent's Tensix parts, NPUs, TPUs — get the accounting
but not the placement, because their memory is managed by their own runtime and there is no
general `.to(device)` to call. That is stated rather than papered over: an `accounting_only`
device will bound your working set and refuse over-commitment, and moving the bytes is yours
to do.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from ._core import Budget, human, parse_size
from .workspace import try_acquire

__all__ = ["Device", "DevicePool", "devices", "device_pool"]

# torch calls an AMD GPU "cuda:N" (its HIP build keeps the CUDA names), while devices() lists it as
# "rocm:N". Every spelling of a GPU resolves to the kinds it can mean: "cuda" to either vendor,
# because that is the name torch gives both; "rocm"/"hip" to AMD only.
_GPU_NAMES = {"cuda": ("cuda", "rocm"), "rocm": ("rocm",), "hip": ("rocm",)}


@dataclass
class Device:
    """One place bytes can live."""
    kind: str                  # "cpu" | "cuda" | "xpu" | "mps" | "rocm" | "tenstorrent" | "npu"
    index: int = 0
    name: str = ""
    total: int = 0
    free: int = 0
    accounting_only: bool = False
    note: str = ""

    @property
    def id(self) -> str:
        return self.kind if self.kind == "cpu" else f"{self.kind}:{self.index}"

    @property
    def tier(self) -> str:
        """Which budget tier this device draws on."""
        return "ram" if self.kind == "cpu" else "vram"

    def torch_device(self) -> str:
        if self.kind in ("cuda", "xpu", "mps"):
            return self.kind if self.kind == "mps" else f"{self.kind}:{self.index}"
        if self.kind == "rocm":
            return f"cuda:{self.index}"          # ROCm masquerades as cuda in torch
        return "cpu"

    def __repr__(self) -> str:
        extra = "  (accounting only)" if self.accounting_only else ""
        return (f"Device({self.id}, {self.name or '-'}, "
                f"{human(self.free)} free of {human(self.total)}){extra}")


def _cpu_memory() -> tuple[int, int]:
    try:
        import psutil
        v = psutil.virtual_memory()
        return int(v.total), int(v.available)
    except Exception:
        pass
    try:
        info = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, _, rest = line.partition(":")
                info[k.strip()] = int(rest.split()[0]) * 1024
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        return total, avail
    except OSError:
        pass
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return total, total // 2
    except (ValueError, OSError):
        return 0, 0


def devices(include_accounting_only: bool = True) -> list:
    """Everything this machine can hold a working set on.

    All probes are optional imports in try blocks — a missing torch or vendor runtime means
    that device simply is not listed, never an import error.
    """
    found: list[Device] = []
    total, avail = _cpu_memory()
    found.append(Device("cpu", 0, "host", total, avail))

    try:
        import torch
        if torch.cuda.is_available():
            hip = bool(getattr(torch.version, "hip", None))
            for i in range(torch.cuda.device_count()):
                try:
                    free, tot = torch.cuda.mem_get_info(i)
                except Exception:
                    props = torch.cuda.get_device_properties(i)
                    tot = int(props.total_memory)
                    free = tot - int(torch.cuda.memory_reserved(i))
                found.append(Device("rocm" if hip else "cuda", i,
                                    torch.cuda.get_device_name(i), int(tot), int(free)))
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            for i in range(xpu.device_count()):
                tot = free = 0
                try:
                    props = xpu.get_device_properties(i)
                    tot = int(getattr(props, "total_memory", 0))
                    free = tot - int(xpu.memory_reserved(i))
                except Exception:
                    pass
                name = ""
                try:
                    name = xpu.get_device_name(i)
                except Exception:
                    pass
                found.append(Device("xpu", i, name or "Intel XPU", tot, free))
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            # Unified memory: the ceiling is host RAM, and torch reports a recommended max.
            try:
                tot = int(torch.mps.recommended_max_memory())
            except Exception:
                tot = total
            found.append(Device("mps", 0, "Apple unified", tot,
                                max(tot - int(getattr(torch.mps, "current_allocated_memory",
                                                      lambda: 0)()), 0)))
    except Exception:
        pass

    if include_accounting_only:
        for mod, kind, label, note in (
            ("ttnn", "tenstorrent", "Tenstorrent Tensix",
             "TT-Metalium manages its own memory; GEMINGA bounds the working set but does "
             "not place it"),
            ("pybuda", "tenstorrent", "Tenstorrent (TT-Buda)",
             "model-level runtime; place via TT-Buda, bound it here"),
            ("openvino", "npu", "OpenVINO NPU",
             "placement is the OpenVINO runtime's job; accounting only"),
        ):
            try:
                lib = __import__(mod)
            except Exception:
                continue
            if mod == "openvino":
                # `import openvino` proves the wheel is installed, not that there is an NPU in
                # the machine: only a device the runtime itself reports counts.
                found.extend(_openvino_npus(lib, label, note))
                continue
            found.append(Device(kind, 0, label, 0, 0, accounting_only=True, note=note))
    return found


def _openvino_npus(ov, label: str, note: str) -> list:
    """One accounting-only Device per NPU in `openvino.Core().available_devices`."""
    try:
        core_cls = getattr(ov, "Core", None)
        if core_cls is None:
            from openvino.runtime import Core as core_cls
        core = core_cls()
        names = [str(n) for n in core.available_devices]
    except Exception:
        return []
    out = []
    for n in names:
        base, _, idx = n.partition(".")
        if base.upper() != "NPU":
            continue
        name = label
        try:
            full = core.get_property(n, "FULL_DEVICE_NAME")
            if isinstance(full, str) and full:
                name = full
        except Exception:
            pass
        out.append(Device("npu", int(idx) if idx.isdigit() else len(out), name, 0, 0,
                          accounting_only=True, note=note))
    return out


def describe_devices() -> None:
    """Print what is available and what each one is good for."""
    ds = devices()
    print(f"{'device':<14} {'name':<28} {'free':>10} {'total':>10}  notes")
    for d in ds:
        note = d.note or ("host memory" if d.kind == "cpu" else "")
        print(f"{d.id:<14} {(d.name or '-')[:28]:<28} {human(d.free):>10} "
              f"{human(d.total):>10}  {note[:44]}")
    if not any(d.kind != "cpu" for d in ds):
        print("\nNo accelerator found. Everything still works; the working set lives in RAM.")


class DevicePool:
    """A bounded allowance on one device, backed by the budget.

        pool = g.device_pool("cuda:0", "6GB", budget=b)
        with pool.place(array) as t:      # refused up front if it will not fit
            ...
    """

    def __init__(self, device: Any = "cpu", capacity: str = "1GB",
                 budget: Optional[Budget] = None):
        self.device = _resolve(device)
        self.capacity = parse_size(capacity)
        self.budget = budget
        self._lease = None
        self._held: list = []
        if self.device.total and self.capacity > self.device.total:
            raise ValueError(
                f"{self.device.id} has {human(self.device.total)} in total; asking for "
                f"{human(self.capacity)} cannot work")
        if self.device.free and self.capacity > self.device.free:
            raise MemoryError(
                f"{self.device.id} has only {human(self.device.free)} free right now "
                f"(of {human(self.device.total)}); asked for {human(self.capacity)}. "
                f"Something else is using it, or free some first.")
        if budget is not None:
            self._lease = try_acquire(budget, self.device.tier, self.capacity)
            if self._lease is None:                 # never waits: nobody is going to hand it back
                raise MemoryError(
                    f"the budget's '{self.device.tier}' tier has "
                    f"{human(budget.available(self.device.tier))} free, less than the "
                    f"{human(self.capacity)} this pool wants")
        self.own = Budget(**{self.device.tier: str(self.capacity)})

    # ---- placement ----
    def fits(self, nbytes: int) -> bool:
        return nbytes <= self.own.available(self.device.tier)

    def nbytes_of(self, obj) -> int:
        for attr in ("nbytes",):
            if hasattr(obj, attr):
                return int(getattr(obj, attr))
        if hasattr(obj, "element_size") and hasattr(obj, "nelement"):
            return int(obj.element_size() * obj.nelement())
        if hasattr(obj, "get_total_buffer_size"):
            return int(obj.get_total_buffer_size())
        raise TypeError(f"cannot size a {type(obj).__name__}")

    def place(self, obj, name: str = ""):
        """Move an array to the device, reserving first so it cannot over-commit.

        Returns a context manager; leaving it releases the allowance (and, on CUDA/XPU,
        drops the tensor reference so the caching allocator can reuse it).
        """
        n = self.nbytes_of(obj)
        if not self.fits(n):
            raise MemoryError(
                f"{human(n)} does not fit the {human(self.capacity)} allowance on "
                f"{self.device.id} ({human(self.own.available(self.device.tier))} free). "
                f"Narrow the region, coarsen the zoom or bin level, or raise the pool.")
        lease = self.own.acquire(self.device.tier, str(n))
        moved = obj
        if not self.device.accounting_only and self.device.kind != "cpu":
            try:
                import torch
                t = obj if hasattr(obj, "to") else torch.as_tensor(obj)
                moved = t.to(self.device.torch_device())
            except Exception as e:
                lease.release()
                raise RuntimeError(
                    f"could not place on {self.device.id}: {e}. The allowance was returned; "
                    f"install a torch build for this device, or use a cpu pool.") from None
        return _Placement(self, moved, lease, n, name)

    def release(self) -> "DevicePool":
        if self._lease is not None:
            self._lease.release()
            self._lease = None
        return self

    def report(self) -> dict:
        s = self.own.stats()[self.device.tier]
        return {"device": self.device.id, "name": self.device.name,
                "capacity": self.capacity, "capacity_human": human(self.capacity),
                "used": int(s["used"]), "used_human": human(int(s["used"])),
                "peak_human": s["peak_human"], "grants": int(s["grants"]),
                "accounting_only": self.device.accounting_only}

    def __enter__(self):
        return self

    def __exit__(self, *e):
        self.release()
        return False

    def __repr__(self) -> str:
        r = self.report()
        return (f"DevicePool({r['device']}, {r['used_human']}/{r['capacity_human']}, "
                f"peak {r['peak_human']})")


class _Placement:
    """A tensor (or array) living inside a pool's allowance."""

    def __init__(self, pool: DevicePool, obj, lease, nbytes: int, name: str):
        self.pool, self.obj, self._lease = pool, obj, lease
        self.nbytes, self.name = nbytes, name

    def __enter__(self):
        return self.obj

    def __exit__(self, *e):
        self.release()
        return False

    def release(self):
        self.obj = None
        if self._lease is not None:
            self._lease.release()
            self._lease = None
        if self.pool.device.kind in ("cuda", "rocm"):
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    def __repr__(self) -> str:
        return (f"Placement({self.name or 'anon'} on {self.pool.device.id}, "
                f"{human(self.nbytes)})")


def _resolve(device: Any) -> Device:
    if isinstance(device, Device):
        return device
    want = str(device).strip().lower()
    found = devices()
    for d in found:
        if d.id == want or (want == d.kind and d.index == 0):
            return d
    kind, sep, idx = want.partition(":")
    if kind in _GPU_NAMES and (idx.isdigit() or not sep):
        for d in found:
            if d.kind in _GPU_NAMES[kind] and d.index == (int(idx) if sep else 0):
                return d
    if want in ("auto", "best"):
        real = [d for d in found if d.kind != "cpu" and not d.accounting_only]
        return max(real, key=lambda d: d.free) if real else found[0]
    raise ValueError(f"no device {device!r}; available: {[d.id for d in found]}")


def device_pool(device: Any = "cpu", capacity: str = "1GB",
                budget: Optional[Budget] = None) -> DevicePool:
    """A bounded allowance on one device, refused up front if it will not fit."""
    return DevicePool(device, capacity, budget)
