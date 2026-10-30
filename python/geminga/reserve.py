"""Reserving a slice of a shared GPU, and giving it back.

A lab GPU server has a few cards and many users. The failure modes are familiar: one notebook
holds 20 GB of VRAM for a week while its owner is on holiday; a job that needs 2 GB cannot
start because three idle sessions hold 8 GB each; nobody can tell who has what. GEMINGA's
answer is small and cooperative: a **reservation server** hands out VRAM in bytes, with a
lease that must be renewed, reclaims what is idle when someone is waiting, caps any one user's
share while others queue, and shows the whole picture in one page. A client turns a grant
into a :class:`~geminga.Budget` whose `vram` tier is exactly the bytes granted, so every
reader and :class:`~geminga.CompressedStore` in that process stays inside the slice.

The server does not enforce anything at the driver level — it cannot stop a process from
calling `cudaMalloc` — so this is a contract between people who want the card shared well,
not a sandbox. What it does enforce is the bookkeeping: a grant is bytes, a grant expires, a
grant that sits idle while others wait is taken back, and the server's `/status` is the truth
about who holds what. Pair it with the compressed store and a 4 GB slice holds what 20 GB used
to.

Run a server (stdlib only, no dependencies)::

    python -m geminga.reserve --serve --port 8765 --device gpu0=24GB --device gpu1=24GB

Reserve from a notebook::

    res = g.reserve("http://gpu-server:8765", vram="4GB", user="seb", ram="8GB")
    budget = res.budget                      # vram capacity == the bytes granted
    store = res.store(device=wgpu_device)    # a CompressedStore in the slice
    ...
    res.release()                            # or leave the `with` block
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from ._core import Budget, human, parse_size

__all__ = ["ReservationServer", "Reservation", "ReservationLost", "reserve", "serve", "Policy"]


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

@dataclass
class Policy:
    """How the card is shared."""
    ttl_s: float = 300.0          # a grant not renewed within this is expired
    idle_s: float = 120.0         # a holder reporting no activity for this long is reclaimable
    max_share: float = 0.5        # one user's share of a device while someone else is waiting
    max_wait_s: float = 3600.0    # a queued request older than this is dropped
    min_grant: int = 64 << 20     # never grant less than this (unless asked for less)


@dataclass
class Holder:
    token: str
    user: str
    device: str
    bytes: int
    granted_at: float
    expires_at: float
    active_at: float
    used_bytes: int = 0
    renews: int = 0
    label: str = ""

    def idle_for(self, now: float) -> float:
        return now - self.active_at


@dataclass
class Waiter:
    id: str
    user: str
    device: Optional[str]
    bytes: int
    min_bytes: int
    since: float
    label: str = ""


class _State:
    def __init__(self, devices: dict, policy: Policy):
        self.devices = {name: int(cap) for name, cap in devices.items()}
        self.policy = policy
        self.holders: dict[str, Holder] = {}
        self.queue: list[Waiter] = []
        self.lock = threading.Lock()
        self.events: list[dict] = []          # a short log, newest last
        self._grants_for_waiters: dict = {}   # request id -> Holder granted while its owner was queued
        self.pickup_s = 15.0                  # a queued grant not collected within this is released

    # ---- accounting ----
    def granted(self, device: str, user: Optional[str] = None) -> int:
        return sum(h.bytes for h in self.holders.values()
                   if h.device == device and (user is None or h.user == user))

    def free(self, device: str) -> int:
        return self.devices[device] - self.granted(device)

    def _log(self, kind: str, **kw) -> None:
        self.events.append({"ts": time.time(), "event": kind, **kw})
        del self.events[:-200]

    # ---- the policy, applied on every request ----
    def sweep(self, now: float) -> None:
        """Expire, grant what fits, and only then reclaim idle slices — one at a time, longest
        idle first, and only as many as the queue needs. An active holder is never preempted,
        whatever its share; the share cap applies when a grant is made."""
        pol = self.policy
        for tok, h in list(self.holders.items()):
            if now > h.expires_at:
                del self.holders[tok]
                self._log("expired", user=h.user, device=h.device, bytes=h.bytes)
        for rid, h in list(self._grants_for_waiters.items()):      # granted, never collected
            if now - h.granted_at > self.pickup_s:
                self._grants_for_waiters.pop(rid, None)
                self.holders.pop(h.token, None)
                self._log("uncollected", user=h.user, device=h.device, bytes=h.bytes)
        self.queue = [w for w in self.queue if now - w.since <= pol.max_wait_s]
        self._grant_waiting(now)
        for w in list(self.queue):
            while any(x.id == w.id for x in self.queue):
                idle = [h for h in self.holders.values()
                        if h.idle_for(now) > pol.idle_s and (w.device is None or h.device == w.device)]
                if not idle:
                    break
                h = max(idle, key=lambda h: h.idle_for(now))
                del self.holders[h.token]
                self._log("reclaimed-idle", user=h.user, device=h.device, bytes=h.bytes,
                          idle_s=round(h.idle_for(now)), for_user=w.user)
                self._grant_waiting(now)

    def _someone_waiting_for(self, device: str) -> bool:
        return any(w.device in (None, device) for w in self.queue)

    def _cap_for(self, user: str, device: str, others_waiting: bool) -> int:
        """Most this user may hold on the device right now."""
        cap = self.devices[device]
        if not others_waiting:
            return cap
        return int(cap * self.policy.max_share)

    def _try_grant(self, w: Waiter, now: float) -> Optional[Holder]:
        devices = [w.device] if w.device else sorted(self.devices, key=lambda d: -self.free(d))
        others = any(x.user != w.user for x in self.queue if x.id != w.id)
        for d in devices:
            if d not in self.devices:
                continue
            free = self.free(d)
            room = self._cap_for(w.user, d, others) - self.granted(d, w.user)
            grant = min(w.bytes, free, room)
            if grant >= max(w.min_bytes, 1) and (grant >= self.policy.min_grant or grant >= w.bytes):
                tok = secrets.token_hex(12)
                h = Holder(tok, w.user, d, int(grant), now, now + self.policy.ttl_s, now, label=w.label)
                self.holders[tok] = h
                self._log("granted", user=w.user, device=d, bytes=int(grant), asked=w.bytes)
                return h
        return None

    def _grant_waiting(self, now: float) -> None:
        remaining = []
        for w in self.queue:                   # FIFO; a request that cannot be served stays queued
            h = self._try_grant(w, now)
            if h is None:
                remaining.append(w)
            else:
                self._grants_for_waiters[w.id] = h
        self.queue = remaining

    # ---- requests ----
    def reserve(self, user: str, nbytes: int, device: Optional[str], min_bytes: int, req_id: Optional[str],
                label: str) -> dict:
        now = time.time()
        with self.lock:
            self.sweep(now)
            if req_id and req_id in self._grants_for_waiters:
                h = self._grants_for_waiters.pop(req_id)
                return self._grant_reply(h)
            if device is not None and device not in self.devices:
                raise KeyError(f"no device {device!r}; have {list(self.devices)}")
            if any(w.id == req_id for w in self.queue):
                return self._queued_reply(req_id, now)
            w = Waiter(req_id or secrets.token_hex(8), user, device, int(nbytes), int(min_bytes), now, label)
            h = self._try_grant(w, now) if not self.queue else None
            if h is not None:
                return self._grant_reply(h)
            self.queue.append(w)
            self._log("queued", user=user, bytes=int(nbytes), device=device)
            return self._queued_reply(w.id, now)

    def _grant_reply(self, h: Holder) -> dict:
        return {"granted": True, "token": h.token, "device": h.device, "bytes": h.bytes,
                "expires_at": h.expires_at, "ttl_s": self.policy.ttl_s, "idle_s": self.policy.idle_s}

    def _queued_reply(self, req_id: str, now: float) -> dict:
        pos = next((i for i, w in enumerate(self.queue) if w.id == req_id), -1)
        return {"granted": False, "request_id": req_id, "position": pos + 1, "queue": len(self.queue),
                "retry_after": 2.0, "devices": self.device_table(now)}

    def renew(self, token: str, used_bytes: int, active: bool) -> dict:
        now = time.time()
        with self.lock:
            self.sweep(now)
            h = self.holders.get(token)
            if h is None:
                return {"ok": False, "reason": "reservation expired or was reclaimed while idle; reserve again"}
            h.expires_at = now + self.policy.ttl_s
            h.used_bytes = int(used_bytes)
            h.renews += 1
            if active or used_bytes > 0:
                h.active_at = now
            return {"ok": True, "expires_at": h.expires_at, "idle_for": round(h.idle_for(now), 1),
                    "waiting": len(self.queue)}

    def release(self, token: str) -> dict:
        now = time.time()
        with self.lock:
            h = self.holders.pop(token, None)
            if h is not None:
                self._log("released", user=h.user, device=h.device, bytes=h.bytes)
            self.sweep(now)
            return {"ok": h is not None}

    def cancel(self, req_id: str) -> dict:
        """A waiting client gives up: leave the queue, and give back a grant made meanwhile."""
        now = time.time()
        with self.lock:
            before = len(self.queue)
            self.queue = [w for w in self.queue if w.id != req_id]
            h = self._grants_for_waiters.pop(req_id, None)
            if h is not None:
                self.holders.pop(h.token, None)
                self._log("cancelled", user=h.user, device=h.device, bytes=h.bytes)
            self.sweep(now)
            return {"ok": before != len(self.queue) or h is not None}

    def device_table(self, now: float) -> dict:
        out = {}
        for d, cap in self.devices.items():
            hs = [h for h in self.holders.values() if h.device == d]
            out[d] = {"capacity": cap, "granted": sum(h.bytes for h in hs), "free": cap - sum(h.bytes for h in hs),
                      "holders": [{"user": h.user, "bytes": h.bytes, "used_bytes": h.used_bytes,
                                   "idle_s": round(h.idle_for(now), 1), "expires_in": round(h.expires_at - now, 1),
                                   "label": h.label, "held_s": round(now - h.granted_at, 1)} for h in hs]}
        return out

    def status(self) -> dict:
        now = time.time()
        with self.lock:
            self.sweep(now)
            return {"devices": self.device_table(now),
                    "queue": [{"user": w.user, "bytes": w.bytes, "device": w.device,
                               "waiting_s": round(now - w.since, 1)} for w in self.queue],
                    "policy": self.policy.__dict__, "events": self.events[-20:], "now": now}


class _Handler(BaseHTTPRequestHandler):
    state: _State = None   # set per server

    def log_message(self, *a):          # quiet
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path.startswith("/status"):
            self._send(200, self.state.status())
        else:
            self._send(404, {"error": "GET /status, POST /reserve /renew /release /cancel"})

    def do_POST(self):
        try:
            b = self._body()
            if self.path == "/reserve":
                r = self.state.reserve(str(b.get("user") or "anonymous"), int(b["bytes"]), b.get("device"),
                                       int(b.get("min_bytes") or b["bytes"]), b.get("request_id"),
                                       str(b.get("label") or ""))
                self._send(200 if r["granted"] else 202, r)
            elif self.path == "/renew":
                r = self.state.renew(str(b["token"]), int(b.get("used_bytes") or 0), bool(b.get("active", True)))
                self._send(200 if r["ok"] else 410, r)
            elif self.path == "/release":
                self._send(200, self.state.release(str(b["token"])))
            elif self.path == "/cancel":
                self._send(200, self.state.cancel(str(b["request_id"])))
            else:
                self._send(404, {"error": "unknown endpoint"})
        except KeyError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:   # a bad request must not take the server down
            self._send(400, {"error": f"{type(e).__name__}: {e}"})


class ReservationServer:
    """The server. `devices` maps a name to its capacity ("24GB" or bytes).

        srv = ReservationServer({"gpu0": "24GB"}, port=8765).start()
        ...
        srv.stop()
    """

    def __init__(self, devices: dict, host: str = "0.0.0.0", port: int = 8765, policy: Optional[Policy] = None):
        caps = {name: int(parse_size(str(v))) if not isinstance(v, int) else v for name, v in devices.items()}
        if not caps:
            raise ValueError("give at least one device, e.g. {'gpu0': '24GB'}")
        self.state = _State(caps, policy or Policy())
        handler = type("Handler", (_Handler,), {"state": self.state})
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self.host, self.port = self.httpd.server_address[0], self.httpd.server_address[1]
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}"

    def start(self) -> "ReservationServer":
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def status(self) -> dict:
        return self.state.status()

    def print_status(self) -> None:
        print(format_status(self.status()))


def format_status(st: dict) -> str:
    lines = []
    for d, info in st["devices"].items():
        lines.append(f"{d}: {human(info['granted'])} of {human(info['capacity'])} granted, "
                     f"{human(info['free'])} free")
        for h in info["holders"]:
            lines.append(f"   {h['user']:<14} {human(h['bytes']):>9}  using {human(h['used_bytes']):>9}  "
                         f"idle {h['idle_s']:>6.0f}s  expires in {h['expires_in']:>5.0f}s  {h['label']}")
    if st["queue"]:
        lines.append("waiting:")
        for w in st["queue"]:
            lines.append(f"   {w['user']:<14} {human(w['bytes']):>9}  for {w['waiting_s']:.0f}s"
                         + (f"  on {w['device']}" if w["device"] else ""))
    return "\n".join(lines)


def serve(devices: dict, host: str = "0.0.0.0", port: int = 8765, policy: Optional[Policy] = None) -> ReservationServer:
    """Start a reservation server in a background thread and return it."""
    return ReservationServer(devices, host, port, policy).start()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class ReservationLost(RuntimeError):
    """The server expired or reclaimed this reservation; reserve again."""


def _post(url: str, path: str, obj: dict, timeout: float = 10.0) -> tuple:
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except Exception:
            return e.code, {"error": str(e)}


class Reservation:
    """A granted slice of a device, kept alive by a heartbeat, turned into a Budget.

    `budget.vram` capacity is the bytes granted. The heartbeat renews the lease every
    `ttl/3` seconds and reports what the budget's vram tier is actually using, which is what
    the server's idle rule looks at: a slice nobody is streaming through gets reclaimed when
    someone else is waiting, and `lost` becomes True (`check()` raises).
    """

    def __init__(self, url: str, user: str, token: str, device: str, nbytes: int, ttl_s: float,
                 ram: Optional[str] = None, spill: Optional[str] = None, label: str = ""):
        self.url, self.user, self.token, self.device, self.bytes = url, user, token, device, int(nbytes)
        self.ttl_s, self.label = ttl_s, label
        self.budget = Budget(ram=ram or "0", vram=str(self.bytes), spill=spill or "0")
        self.lost = False
        self.reason = ""
        self._last_grants = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()

    # ---- liveness ----
    def _activity(self) -> tuple:
        st = self.budget.stats()["vram"]
        active = st["used"] > 0 or st["grants"] != self._last_grants
        self._last_grants = st["grants"]
        return int(st["used"]), bool(active)

    def renew(self, active: Optional[bool] = None) -> bool:
        used, act = self._activity()
        code, r = _post(self.url, "/renew", {"token": self.token, "used_bytes": used,
                                             "active": act if active is None else active})
        if code != 200:
            self.lost, self.reason = True, r.get("reason") or r.get("error") or f"HTTP {code}"
            return False
        return True

    def _heartbeat(self) -> None:
        period = max(1.0, self.ttl_s / 3.0)
        while not self._stop.wait(period):
            if self.lost:
                return
            try:
                self.renew()
            except Exception as e:      # the server being unreachable is not a loss yet
                self.reason = str(e)

    def touch(self) -> None:
        """Tell the server this reservation is in use even if nothing is leased right now."""
        self.renew(active=True)

    def check(self) -> None:
        if self.lost:
            raise ReservationLost(f"reservation on {self.device} lost: {self.reason}")

    # ---- what the grant buys ----
    def store(self, codec=None, device=None, tier: str = "vram"):
        """A CompressedStore inside the slice (needs a wgpu device for tier='vram')."""
        from .codec import CompressedStore
        return CompressedStore(self.budget, tier, codec, device)

    def release(self) -> None:
        self._stop.set()
        if not self.lost:
            try:
                _post(self.url, "/release", {"token": self.token}, timeout=5.0)
            except Exception:
                pass
        self.lost = True
        self.reason = self.reason or "released"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    def __repr__(self) -> str:
        state = "lost" if self.lost else "held"
        return f"Reservation({human(self.bytes)} on {self.device} @ {self.url}, {state})"


def reserve(url: str, vram: str = "4GB", user: Optional[str] = None, device: Optional[str] = None,
            min_vram: Optional[str] = None, wait: float = 120.0, ram: Optional[str] = None,
            spill: Optional[str] = None, label: str = "", poll: float = 2.0) -> Reservation:
    """Ask a reservation server for `vram` bytes; block up to `wait` seconds in the queue.

    `min_vram` accepts a smaller grant when the full amount is not free (the server gives the
    most it can, never less than this). Raises TimeoutError when the wait runs out — with the
    server's device table in the message, so the caller can see who holds what.
    """
    user = user or os.environ.get("USER") or os.environ.get("USERNAME") or socket.gethostname()
    want = int(parse_size(vram))
    least = int(parse_size(min_vram)) if min_vram else want
    req_id = secrets.token_hex(8)
    deadline = time.time() + wait
    last: dict = {}
    while True:
        code, r = _post(url, "/reserve", {"user": user, "bytes": want, "device": device, "min_bytes": least,
                                          "request_id": req_id, "label": label})
        if code == 200 and r.get("granted"):
            return Reservation(url, user, r["token"], r["device"], r["bytes"], float(r.get("ttl_s", 300.0)),
                               ram=ram, spill=spill, label=label)
        if code not in (200, 202):
            raise RuntimeError(f"reservation server refused: {r.get('error', r)}")
        last = r
        if time.time() >= deadline:
            try:
                _post(url, "/cancel", {"request_id": req_id}, timeout=5.0)    # leave the queue
            except Exception:
                pass
            table = "\n".join(f"  {d}: {human(i['granted'])} of {human(i['capacity'])} granted to "
                              f"{', '.join(h['user'] for h in i['holders']) or 'nobody'}"
                              for d, i in last.get("devices", {}).items())
            raise TimeoutError(f"no {human(want)} slice free within {wait:.0f}s (queue position "
                               f"{last.get('position')} of {last.get('queue')}):\n{table}")
        time.sleep(min(poll, float(r.get("retry_after", poll))))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m geminga.reserve",
                                 description="GEMINGA VRAM reservation server / status")
    ap.add_argument("--serve", action="store_true", help="run the server in the foreground")
    ap.add_argument("--device", action="append", default=[], metavar="NAME=SIZE",
                    help="a device and its capacity, e.g. gpu0=24GB (repeatable)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ttl", type=float, default=300.0, help="seconds a grant lives without a renew")
    ap.add_argument("--idle", type=float, default=120.0, help="seconds of inactivity before a slice is reclaimable")
    ap.add_argument("--max-share", type=float, default=0.5, help="one user's share while others wait")
    ap.add_argument("--status", metavar="URL", help="print a server's status and exit")
    a = ap.parse_args(argv)
    if a.status:
        with urllib.request.urlopen(a.status.rstrip("/") + "/status", timeout=10) as r:
            print(format_status(json.loads(r.read())))
        return 0
    if a.serve:
        devices = dict(d.split("=", 1) for d in a.device) or {"gpu0": "8GB"}
        srv = ReservationServer(devices, a.host, a.port, Policy(ttl_s=a.ttl, idle_s=a.idle, max_share=a.max_share))
        print(f"reservation server on {srv.url}: " + ", ".join(f"{k}={human(v)}" for k, v in srv.state.devices.items()))
        try:
            srv.httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(_main())
