# Reserving a slice of a shared GPU

A lab GPU server has a few cards and many users, and the failure modes are familiar: one
notebook holds 20 GB of VRAM for a week while its owner is on holiday; a job that needs 2 GB
cannot start because three idle sessions hold 8 GB each; nobody can tell who has what.
GEMINGA's answer is small and cooperative — a **reservation server** that hands out VRAM in
bytes with leases that must be renewed, reclaims idle slices when someone is waiting, caps one
user's share while others queue, and shows the whole picture on one page. A client turns a
grant into a `Budget` whose `vram` tier is exactly the bytes granted, so every reader and
every [compressed store](codec.md) in that process stays inside the slice.

## Run the server

```
python -m geminga.reserve --serve --port 8765 --device gpu0=24GB --device gpu1=24GB \
       --ttl 300 --idle 120 --max-share 0.5
python -m geminga.reserve --status http://gpu-server:8765
```

Standard library only; one process; state in memory. Restarting it forgets every grant, which
is the right thing for a server that just rebooted.

## Reserve from a notebook

```python
res = g.reserve("http://gpu-server:8765", vram="4GB", user="seb", ram="8GB",
                min_vram="2GB", wait=120, label="glioma chip, bin20")
res.bytes, res.device                   # what was granted, where
budget = res.budget                     # Budget(ram="8GB", vram=<granted>)
store = res.store(device=wgpu_device)   # a CompressedStore in the slice
...
res.release()                           # or leave the `with res:` block
```

`reserve()` blocks in the queue up to `wait` seconds and raises `TimeoutError` with the
device table in the message — who holds what — when the wait runs out; it also withdraws the
queued request so a grant made later does not go to a client that gave up. `min_vram` accepts
a smaller grant when the full amount is not free; the server gives the most it can, never less
than that.

A heartbeat renews the lease every third of the TTL and reports what the budget's `vram` tier
is actually using. That report is what the idle rule looks at: a slice nothing is leased from,
and nothing has been leased from recently, is idle. `res.touch()` says "in use" explicitly when
the work is outside the budget's view; `res.lost` turns true (and `res.check()` raises
`ReservationLost`) once the server has taken the slice back.

## The policy

| rule | default | meaning |
|---|---|---|
| `ttl_s` | 300 | a grant not renewed within this expires |
| `idle_s` | 120 | a holder reporting no activity for this long is reclaimable |
| `max_share` | 0.5 | one user's share of a device while someone else is waiting |
| `max_wait_s` | 3600 | a queued request older than this is dropped |
| `min_grant` | 64 MB | partial grants are never smaller than this |

Reclaiming is deliberate: expired grants go first; then, only while a request is waiting that
cannot be served, idle slices are taken back one at a time, longest idle first, until the
request fits. **An active holder is never preempted**, whatever its share — the share cap
applies when a grant is made, not to work in progress. A grant made for a queued client that
never collects it is released after fifteen seconds. Requests are served in order; a request
that cannot be served does not block smaller ones behind it from being granted when the queue
is swept, but it keeps its place.

`GET /status` returns, per device, the capacity, what is granted and free, every holder with
its user, bytes, bytes actually used, idle time, time to expiry and label, plus the queue and
the last twenty events (granted, queued, reclaimed-idle, expired, released, cancelled).

## What this is not

It does not enforce anything at the driver level. A process can still call `cudaMalloc`
past its grant; the server cannot stop it. This is a contract between people who want the card
shared well, with bookkeeping that is honest and visible, not a sandbox. Pairing it with
NVIDIA MPS/MIG or `CUDA_VISIBLE_DEVICES` per slice is the operator's choice; GEMINGA's part is
to make its own readers and stores obey the number they were given.
