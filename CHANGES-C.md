# CHANGES-C — infrastructure fixes (remote, workspace, staging, vault, devices)

Files changed: `python/geminga/{remote,workspace,staging,vault,devices}.py`, `tests/test_vault_staging.py`,
`tests/test_workspace.py`, and the new `tests/test_fixes_infra.py` (103 regression tests). `tests/test_remote.py`
is untouched and passes.

Test accounting: 91 of the new tests were written before any fix. 73 of them failed on the old code, each for
the reason it names; 15 passed on it because they guard behaviour that had to survive; 3 (read-everything across
block/cache/read-ahead combinations) passed because they were too weak, were then made deterministic, and now
fail on the old logic too. The other 12 were added afterwards for hazards found while fixing: a mid-body stall,
lock order in `drop`, pins surviving a merge, overwrite under another suffix, a `put` asked again with another
source or window, foreign files in a staging directory, index builds for read-only members, `try_acquire`, a
custom writer that calls `np.save`, the cross-process lock, and four processes on one directory. In the two
existing files, `test_vault_staging.py` now scans a scratch copy through a module fixture (it used to scan
`fixtures/` itself, and several of its tests depended on the manifest that left there), and its budget test and
`test_workspace.py`'s use a 5 s budget timeout so that a regression fails fast instead of waiting 120 s.

Shared helper: `workspace.try_acquire(budget, tier, nbytes)` returns a lease or `None` and never waits.
`Budget.acquire` blocks for the budget's whole timeout (120 s by default) on a tier that is merely full;
that is right for a reader waiting on a sibling and wrong for a *reservation*. It is used for cell slices,
`Staging(budget=)`, `DevicePool(budget=)` and `RemoteFile(budget=)`. (Two existing tests sat out 120 s each on
this, which was essentially all of the 247 s the old three-file run took; the same command now takes ~12 s.)

## 1. remote.py — `RemoteFile` returned wrong or short bytes

- **Replies must be the range asked for.** HTTP 206 needs a `Content-Range` whose start is the offset
  requested (and whose total is the size seen at open); a 200 answers only a request starting at byte 0
  and is cut to the length asked for. Any other reply raises `IOError` naming the problem (no retry: asking
  again gets the same answer). Behaviour change: a server that ignores `Range` used to return the bytes of
  offset 0 for a read at 2 MB; now only the first block is readable and the rest raises.
- **Exact length, retried once, never cached short.** `_fetch` clips to end-of-file itself, then requires
  `len(data) == length`; a short body (or a stall/reset mid-body) is requested once more, then `IOError`
  ("short read ... after 2 tries"). A block that is not full-size (except the file's last) is never put in
  the cache. FTP takes the same path, so an aborted RETR is retried and then refused instead of cached.
- **Unknown size is an error, not an empty file.** Size is HEAD's `Content-Length`; when HEAD lacks it or is
  refused (400/403/405/501) it is the total from a one-byte ranged GET (`Content-Range: bytes 0-0/N`, or
  `*/0` on a 416 for an empty file); otherwise `IOError`. FTP `SIZE` returning nothing raises too.
  `list_tree(sizes=True)` uses the same probe but records 0 for a size the server will not give instead of
  abandoning the listing.
- **Read-ahead can no longer evict what it fetched.** The span is capped at `cache_cap - 1` blocks and at
  the end of the file, and stops at the first block already cached; the requested block is returned from
  the fetch, not looked up again. `block="1MB", cache="2MB"` used to return 1 MB and then nothing (so
  `copyfileobj` copied 1 MB of 10; checked against the old logic); it now reads all of it. `read()` raises
  rather than return short.
- **Hardening in the same code:** `seek` rejects negative positions and unknown `whence`; `read`/`seek` on a
  closed file raise `ValueError`; `readinto` takes any contiguous buffer; a constructor that fails (404, no
  size, no budget) releases its cache reservation and socket; `block <= 0` is rejected; requests send
  `Accept-Encoding: identity` so a compressing server cannot shift the offsets.

## 2. workspace.py — `run_isolated`

- **No more hang on results over ~64 KB.** The result now travels over a `Pipe` that the parent reads
  *before* joining the child (the old `Queue` child blocked in `put` until the join timed out). A child that
  dies is seen at once (EOF) instead of after the timeout; an unpicklable result comes back as that error
  (it used to vanish in the queue's feeder thread and be reported as "died"); `timeout=None` still means no
  limit; a failed fork closes both pipe ends. `RLIMIT_AS` semantics (baseline + cap + 256 MB) are unchanged.
- **Runs against the live position, not the last suspend.** It calls the same `_save()` as `suspend()` first
  and the child replays bin, genes, window and zoom through `_apply_position`, the function `_restore()` now
  also uses, so the two cannot drift. (The child used to get `_saved`, empty unless the cell had been parked,
  and never replayed bin or genes.)
- **`report()["window"]`** is the open view's window if there is one, else where the cell was parked (it used
  to prefer the snapshot from the last suspend). Each cell report also gains `ram_reserved`.

## 3. workspace.py — reservations

- **`cell.view` is the resume path.** Touching a cell parked with `suspend(release_reservation=True)` takes
  its slice again first; if the workspace cannot give it, `MemoryError` and the cell stays `suspended`
  (before: it reopened with no lease, state `open`, over capacity). `total_reserved()` and
  `report()["ram_reserved"]` now sum the leases actually held (`Cell.reserved(tier)`), so reserved + free ==
  capacity. Behaviour change: `total_reserved` no longer counts the nominal cap of a parked cell.
- **Slices never wait.** `ws.add(...)` (and every re-reservation) raises `MemoryError` at once, with the free
  figure, instead of blocking for the workspace timeout first. The message keeps its old wording
  ("cannot reserve 40MB ram for cell 'c': the workspace has 10.0 MB free. ...") and names the tier when it
  is not RAM.
- **A failed `resume()` holds nothing.** Leases are taken for all tiers or none; before, the RAM lease taken
  for a resume whose VRAM lease failed stayed in the list, so the next resume saw a non-empty list and
  skipped VRAM for good.
- **Lazy open is locked.** Each cell has an `RLock` (double-checked in `view`), so N threads touching a new
  cell open it once instead of leaking N-1 readers; the workspace has one for membership and
  check-then-acquire. Lock order is cell then workspace everywhere: `drop()` closes the cell outside the
  workspace lock (inside it, a concurrent resume of that cell deadlocked; a test forces that interleaving).
- **Two small fixes found on the way:** `suspend()` on a closed cell is a no-op (`with cell:` after `close()`
  used to turn it back into `suspended` and let it reopen); a view that fails while being restored is closed
  and dropped (it used to stay half-open under state `failed`).

## 4. staging.py — the index no longer trusted over the key, the disk and the path

- **Paths are relative to the root and must stay inside it.** The index stores `path` relative to the
  staging directory; on load every path is resolved with `realpath` and refused (entry dropped, file
  untouched) if it leaves the root, names the index or its temporaries, or is a symlink out; deletion
  re-checks the same thing. A copied staging directory therefore evicts its own files, never the original's,
  and an edited index cannot delete `/etc/anything`. Indexes written by older versions (absolute paths) are
  rebased by file name onto the directory they are found in. `StagedRegion.path` stays absolute in memory.
- **Same key, different request, different region.** `extract` records what a region was cut from
  (`meta["sig"]`: source, window, bin, zoom, genes, columns, format) and on a key hit compares; a mismatch
  re-extracts (`extract("roi", bin50, W1)` then `extract("roi", bin200, W2)` used to return the first
  object). The old region stays until the new one is complete, keeps its pin, and a failed re-extraction
  leaves it in place. `put(..., sig=)` is the new optional argument; without it `put` compares `source` and
  `window` when both sides have them. An identical request is still a hit that loads nothing, across
  restarts too.
- **A writer that fails leaves nothing.** Regions are written to `<stem>-<hash>.<pid>.<n>.part<suffix>` (the
  real extension stays last, for writers that go by it) and `os.replace`d into place only after the space
  accounting; a writer that raises has the partial and anything beside it removed, and one that returns
  without creating its file is a `RuntimeError` rather than an empty region — except that `np.save(path, a)`
  is taken as it stands (the `.npy` it adds is adopted, format `npy`). Behaviour change: a writer is handed a
  scratch name, not the final one.
- **Startup reconciles directory and index.** Entries whose file is gone are dropped, `.part` files whose
  writer is no longer running (pid check; live writers, here or in another process, are left alone) are
  removed, and any other file the index does not know is *counted* as a region `orphan:<file>`: unpinned and
  first to be evicted when it is one of staging's own slot names, otherwise pinned and "foreign" (never
  deleted by `clear()` or eviction, only by `evict(key)`). So `used` equals what is on disk, and
  `report()["orphans"]` says how many. `StagingFull` names the foreign bytes when they are what blocks you.
- **Arrays stage correctly.** `np.save` now gets a file object and the region is filed as `.npy` whatever
  suffix was left at the `.parquet` default (it used to write `...parquet.npy`, leaving `bytes == 0` and a
  `read()` that failed); a writer may return `suffix` to be filed under another one.
- **Several `Staging` objects (or processes) on one directory merge instead of overwrite.** Every write
  re-reads the index and merges under a per-directory re-entrant lock plus an `flock` on the directory
  itself (no lock file); lookups notice another object's index change by its mtime/size. Merge rule: the
  index on disk wins, the newest `accessed`/highest `hits` of either side survive, regions are updated in
  place so references stay valid. Tests run four processes and four threads writing to one directory: no key
  is lost and `used` equals what is on disk; another test holds the lock in one process and times a `put`
  in a second.
- **A `put` that cannot fit evicts nothing.** `_make_room` checks `pinned + size <= capacity` before touching
  anything (it used to evict every unpinned region and then raise); a region being replaced counts as free.
- `Staging(budget=)` reserves without waiting (see the shared helper) and releases the lease if startup
  fails; `clear()` and eviction under space pressure write the index once for the whole batch; `used`,
  `free` and `report()` count other objects' regions too; `report()` gains `orphans`.

## 5. vault.py

- **Known suffixes:** `.nii`, `.nii.gz`, `.h5`, `.mtx`, `.mtx.gz` are indexed. `.mtx(.gz)` members open as
  `MtxTriple` (`g.view` cannot open them); `.h5` and NIfTI already had openers.
- **A DICOM directory is one member.** Slices named `.dcm` or `.ima` (the suffixes `g.view` and `profile`
  take a directory by) make their directory the member: path = the directory, `bytes` = the slices' total,
  `meta["n_files"]`, `format` "DICOM series" when profiled; `.dcm` is no longer in `_KNOWN`, so slices are not
  members of their own. Mixed directories keep their other files as members. (A rescan drops per-slice entries
  left in an old manifest.) `scan` also keeps an unchanged profiled member only if its size is unchanged
  too, and profiling no longer replaces `meta`.
- **Fallback manifest is keyed by the vault's absolute path:** `~/.geminga/vault-<name>-<blake2b(root)>.json`.
  One in `~/.geminga` is accepted only if it names this root (so two `data` folders no longer share a
  manifest); the old `vault-<name>.json` is read under the same rule and migrated on the next save. The
  manifest beside the data is still taken whatever path the drive is mounted at.
- **`get()` needs a unique match.** A bare name several members share raises `AmbiguousMember` (a
  `KeyError` *and* a `ValueError`) listing the paths; an exact relative path always wins. `get`/`abspath`
  also accept `./a/b`, an absolute path into the vault, or a `Path`.
- **`find(needle, exact=False)`**: with `exact=True` the needle must be a whole token of the *path* (bounded by
  anything that is not a letter or digit, or by either end; a needle that starts or ends with a separator,
  e.g. ".gef", is bounded there already). Substring behaviour is unchanged by default.
- **Nothing is written beside read-only data.** New `Vault(readonly=True)` plus automatic detection
  (`os.access`) of a directory that is not writable: the manifest goes to `~/.geminga`, FASTQ/GEM sidecars
  (`.gemidx.json`) go to `~/.geminga/gefidx/<hash>/` through their `sidecar=` argument (a fresh one already
  beside the data is read in place). A GEF is pointed at an index it can read without writing — one beside the
  file if there is one, else the private directory — and `view.gef.build_spatial_index()` builds into the
  private directory. `add_to()` opens through the same path. On a real read-only bind mount, plain `g.view`
  ignores an index shipped beside the GEF (the whole level is read) while `Vault.open` uses it in place.

## 6. devices.py

- **No phantom NPU.** `import openvino` is no longer evidence of hardware: an NPU is listed only for each
  `NPU`/`NPU.n` that `openvino.Core().available_devices` reports (named from `FULL_DEVICE_NAME` when it is a
  string; a runtime that cannot enumerate lists nothing).
- **GPU spellings:** `cuda`, `cuda:N` resolve to the CUDA or ROCm device with that index (torch's name for
  both); `rocm:N` and `hip:N` resolve to the ROCm device; whitespace and case are ignored. `rocm:0` on an
  NVIDIA-only host is still "no device" (there is no AMD GPU to give). `DevicePool(budget=)` reserves
  without waiting.

## Declined / deferred

- **Tenstorrent (`ttnn`, `pybuda`) devices are still listed on import alone.** Same flaw as openvino's; deferred
  because the right check is the runtime's own device enumeration and there is no TT runtime or hardware here
  to verify an API against — a wrong guess would hide real devices.
- **gefindex.py / stereoseq.py (not mine):** `_sidecar_dir` ignores an existing valid `.gemidx/` when the
  directory is not writable (it probes by writing), `GefFile.spatial_index` creates `.gemidx/` as a side
  effect of looking, and `open_gem` has no `sidecar` argument. The vault works around all three (best
  effort, via `GefFile._sidx`/`_sidx_checked` and `index_gem(sidecar=)`; it leaves the view as it came if
  those change). The real fix is in gefindex: use a valid existing index read-only, create nothing on lookup.
- **orchestrate.py:647 (not mine)** still calls `vault.find(needle)`; it should pass `exact=True` for accession
  needles now that it exists. **`__init__.py`** does not export `AmbiguousMember` or `try_acquire`
  (`from geminga.vault import AmbiguousMember` works; `geminga.vault` the attribute is the function).
- **Docs and changelog (not mine):** `docs/vault.md` (fallback manifest name, `readonly`, DICOM members,
  `exact`), `docs/workspace.md` (touching a parked cell re-reserves or raises), `docs/remote.md` (errors
  instead of silent short reads), `CHANGELOG.md`.
- **No whole-file fallback for servers that ignore `Range`.** Buffering the file would defeat the point of
  opening it in place; the first block works, the rest is a clear `IOError`. No retry of connect-time errors
  or 5xx either — only short replies, as specified.
- **Foreign files in a staging directory are counted, never deleted.** The alternative (remove them) can
  destroy someone's data in a shared scratch directory; the cost is that such a directory looks fuller than it
  is, which `StagingFull` and `report()["orphans"]` say. Give staging a directory of its own.
- **Index cost is still O(regions) per write** (now one read plus one write, not one write); fine for
  thousands of regions, a journal would be needed beyond that. `flock` is best effort (absent on Windows,
  possibly unsupported on some network filesystems, where only the in-process lock applies); the `.part`
  liveness check is POSIX-only (elsewhere another writer's partials are left alone).
- **`run_isolated` is still fork-only** (the child is a closure); unchanged.
- **Outside my files:** `ruff check` reports E401 at `tests/test_budget.py:72` and
  `tests/test_navigation.py:104,132`; left alone. The real-read-only-mount test needs root and `unshare` and
  skips otherwise. I removed `fixtures/.geminga-vault.json`, which the old vault tests (and the baseline run)
  had written into the shared fixtures; the vault tests now scan a scratch copy. Existing tests that open
  `fixtures/chip.gef` in place still touch `fixtures/chip.gef.gemidx/`, as before.
