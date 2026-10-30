# CHANGES-B: store and planner defect fixes

Edited: `python/geminga/{catalog,selections,library,acquire,profile}.py`. Tests: new
`tests/test_fixes_stores.py` (83 tests, all offline; fixture-dependent guards skip without
fixtures; `polars`/`pydicom` tests use `importorskip`), plus small additions to
`tests/test_selections.py`, `tests/test_library.py`, `tests/test_acquire.py`. No other module was
touched. The first 79 tests were written and run red before any fix: 58 failed, the other 21 are
guards and positive controls that must keep passing. Four later tests (warm cache across a tail
repair, wide tree and Zarr store, `nuclei`/`salmon`, accession-less ENA row) were written
together with the code they cover. Public signatures are unchanged; additions are listed per item.

## Fixes

1. **Torn tail and interleaved append** (`catalog.py` `Catalog._append`, `selections.py`
   `SelectionBook._append`). Both now go through one helper, `catalog._append_line(path, line,
   expect)`.
   - It opens the log `a+b` and `fstat`s the handle. If the file is non-empty and its last byte
     is not `\n` it writes `\n` and the record in one `write`, so a crashed writer's fragment
     stays on its own line. Before, the next record was glued onto the fragment and a fresh
     reader lost both.
   - The cached fold is kept (advanced) only if the handle's pre-write `(size, mtime_ns)` equals
     the signature the cache was built against **and** the post-write size is pre-write size plus
     the bytes written (including the repair newline). Otherwise `_fold = None` and the next read
     re-folds. Before, the fold was re-stamped with a fresh `stat`, which could vouch for another
     process's append that the cache never folded.
   - Behaviour change: a record written after a crash is no longer lost; a concurrent append can
     no longer leave a stale cache that claims to be current. A lone writer's loop still never
     re-parses (cache stays warm). Tests: torn tail (both stores, and a log that is only a
     fragment), warm cache across a repair, interleaved append by monkeypatching `_signature`
     (both stores), clean logs gain no blank lines. Also stress-checked with 6 concurrent
     processes writing 20 KB lines: no record lost or glued.

2. **Status clobber** (`selections.py`, `library.py`). New `SelectionBook._advance(sel_or_id,
   status, **fields)` is the single gate: it re-reads the stored record, moves status only
   forward along `open -> described -> searched`, and writes `fields` in the same event. Anything
   off that path (`in-review`, `confirmed`, `refuted`, `resolved`) is never replaced.
   `selections.describe`, `selections.leads`, `library.describe_any`, `library.leads_any` use it
   (with a book/lib) or the same `_forward` rule on the object (without). The caller's possibly
   stale object is re-synced to the stored status. Behaviour change: `lib.confirm(s)` then
   `leads_any(s, lib=lib)` leaves `confirmed` (was `searched`); a stale `open` copy passed to
   `describe` no longer rewrites `searched`/`in-review` to `described`. `set_status` stays free.
   Docstring of `Selection.status` now lists the review states.

3. **Naive datetimes** (`selections._num`, `select_values` docstring). A datetime without a
   `tzinfo` is read as UTC (`replace(tzinfo=timezone.utc)`), matching Polars `dt.epoch`; aware
   datetimes keep their offset; numbers are unchanged. Behaviour change: the same window selects
   the same rows under any `TZ` (before: shifted by the local offset). Tests set `TZ` with
   `time.tzset()` (POSIX zone strings, asserts `time.timezone` really changed) and compare the
   rows selected from a Parquet file.

4. **Crashes, intervals, sequences, paths** (`selections.py`, `library.py`).
   - `Selection.bounds` is `None`-safe for `variant`/`locus`/`sequence` (0 instead of `None`), a
     new `Selection.placed` property says whether those coordinates mean anything, and
     `__repr__` prints `[?]` for unplaced records. `repr(flag_sequence("ACGT..."))` and
     `repr(flag_variant(rsid="rs334"))` no longer raise; `area`/`overlaps` no longer raise.
   - `Library.to_bed` skips records without coordinates (emits a `UserWarning` with the count),
     raises `ValueError` when the library holds loci on more than one build unless `build=` is
     given (then only that build is written), and writes nothing if it raises. Before: `TypeError`
     on an rsID-only variant, and GRCh37/GRCh38 loci mixed in one file.
   - `parse_locus("chr1:0-100")` -> `("1", 0, 100)` (was start -1); `"chr1:0"` -> `("1", 0, 1)`.
     Docstring states browser coordinates are 1-based and position 0 is clamped to the first base.
   - `flag_sequence` converts `U` to `T` only when the sequence is entirely `ACGTUN`;
     `"MKVLUAGGH"` is stored as given (was `MKVLTAGGH`).
   - Absolute paths: `SelectionBook.__init__` (hence `Library`, whose `__init__` also does it) and
     `dataset_key` store `os.path.abspath(os.path.expanduser(...))` for any non-URL, non-repr
     string, whether or not the file exists yet (URLs, anything containing `://`, are left
     alone). Behaviour change: `select_values("rel.parquet")` can be described from another
     directory, `find(dataset=<abs path>)` finds it, and a relative log path no longer forks into
     a second file after `chdir`. Beyond the list, `Catalog.__init__` got the same fix (identical
     fork bug, one line).

5. **acquire.py** (`Route` gained `representation`, `metered`, `cost_known`, `cost_estimate`).
   - Offline SRA family: `plan("SRR1448774", offline=True)` now routes `ena-http`
     (`https://www.ebi.ac.uk/ena/browser/view/<acc>`, bytes 0, streamable, note "size known once
     online") then `sra-prefetch` (note names `prefetch` and `fasterq-dump`). For non-run
     accessions (SRX/SRP/PRJ/SAM) the prefetch note says it needs runs first. "unrecognised
     accession shape" is now only for shapes `classify` does not know; an online lookup that
     returns no rows gets its own message instead of "unrecognised".
   - GEO stub: last three digits become `nnn` at every length (`GSE123`/`GSE99` -> `GSEnnn`,
     `GSE1234` -> `GSE1nnn`, `GSM123` -> `GSMnnn`; longer ones unchanged).
   - Unknown cost: `AcquisitionPlan.cheapest_cost` is `None` when nothing is free and a metered
     route has no size (0.0 if any route is free); `explain()` prints "unknown size (metered)",
     `Route.__repr__` says "metered, size unknown", `print_shortlist` prints "unknown", and such
     routes sort after priced ones of the same kind. `Route.cost_usd` stays a float (0.0 when
     unknown) because `orchestrate.py` sums it; use `cost_known`/`cost_estimate` to tell.
   - One representation per run: `plan()` (`files`, `total_bytes`) and the per-kind rows of
     `explain()` count a run's FASTQ files and ignore its `.sra` container when FASTQ exists (the
     container counts if it is all there is). Two mates plus an `.sra` is now 2 files, not 3
     (3.6e9 became the FASTQ sum). Rows without a run accession are treated as one run.
   - `fetch(dry_run=True)` builds a command for every route of the chosen kind, one
     representation per run, both mates included: new keys `commands` (list) and `urls`;
     `command` joins them with ` && `; `url` is the first; `bytes`/`cost_usd` are sums. An
     explicit `route=` is still exactly that route. A real run executes them in order and stops
     at the first failure.
   - `_sniff` decodes gzip with `zlib.decompressobj(wbits=31)`, chaining members, and keeps the
     partial output of a truncated stream: the first 64 KB of a gz FASTQ is now `FASTQ` with read
     stats (was `unknown` with a note claiming enough was decoded). The note now says what
     actually happened.

6. **profile.py**.
   - Keywords match at token boundaries (rule copied from `harmonize.Vocabulary._alternative`,
     reimplemented, no import): short terms (<= 4 chars) are whole words (optional plural), longer
     ones need only a left boundary. "western" no longer scores `wes`, "13th"/"NIH-3T3" no longer
     score `3t`, "bin100" no longer scores `bin1`. Applied to page text, filename tokens, and read
     identifiers (an index `ATACGGTA` is no longer "atac" content evidence). `nuclei` and `salmon`
     also need a right boundary (`_WHOLE_WORD`) so "nucleic acid" and Salmonella do not score.
     To keep real signals that the old substring match caught: added terms `snatac`, `atacseq`
     (page) and filename tokens `scatac`, `snatac`, `atacseq`.
   - TIFF needs the full magic: `II*\0`, `MM\0*`, and BigTIFF `II+\0`, `MM\0+` (BigTIFF kept so
     large whole-slide files are not lost). "MMP1,MMP2,..." is no longer TIFF.
   - Directories: DICOM and NIfTI are searched up to 3 directory levels down, breadth first,
     first directory that holds any wins; at most 200 entries per directory, 8 subdirectories
     followed per level, 200 directories; `.zarr` stores are not entered. NIfTI-only directories
     now get `format` "NIfTI"/"NIfTI.gz", the volume's evidence, and `n_nifti_files_seen`.
   - The "path does not exist locally" note is only added when the path does not exist.
   - `.sam` is `SAM` (extension and header sniff); FASTQ now needs the `+` line when it is in view
     (or a sequence-looking line when only that is), and SAM header lines (`@HD`, `@SQ`, `@RG`,
     `@PG`, `@CO`) are excluded.
   - `molecule` is asserted from the leading assay only if that assay's probability is above its
     prior-only probability (and above 20%): `profile("/nowhere/thing.h5ad")` has no molecule
     (was RNA, the 21% prior). Visible side effect: `Vault` members such as Parquet/Arrow/gz-FASTQ
     files with no evidence now show an empty molecule.
   - `Profile.confident` is untouched.

7. **catalog.py**.
   - `annotate` raises `TypeError` for keyword arguments that are not `CatalogEntry` fields,
     before anything is logged; facet keys get the hint `pass facets={'tissue': ...}`. Before it
     logged them and dropped them at fold time.
   - The profiler's assay is mirrored into `facets["assay"]` only while the analyst has not named
     one: `metadata["assay_from_profiler"]` is `True` while the facet is the profiler's (a newer
     profile refreshes it), `False` once set via `record(facets=...)` or `annotate(facets=...)`
     (the fold sets it on annotate), and absent when there is no assay. Logs written before the
     flag that already carry an `assay` facet are treated as analyst-set and left alone.

## Declined / deferred

- **Append locking.** The tail-repair check and the write are not atomic across processes (a
  process that checks the tail while another is part-way through a multi-page line sees no `\n`
  yet), so concurrent writers can add an occasional blank line. No record is lost or glued
  (stress test), every reader here skips blank lines, and Polars reads them. Closing it needs
  `flock`/`msvcrt` and NFS fallbacks; not worth it.
- **`_advance` is read-then-write**, so two processes (one confirming, one running `leads`) can
  still interleave. A race-free version would log an "advance" op resolved at fold time, changing
  the event format other readers (PRISM) see. Left as is.
- **Logs written with relative dataset paths** keep their old `dataset.path`/`key`; they will not
  match new absolute keys in `find()`/`query()`. No migration of an append-only log.
- **Same silent-drop pattern, not in the list:** `Catalog.annotate` on a key never recorded still
  folds to nothing; `SelectionBook.update(**unknown)` is logged then dropped at fold;
  `Library.annotate/link` on an unknown id raise `AttributeError` rather than `KeyError`;
  `flag_variant(pos=0)` stores start -1.
- **Not matched by design:** short keywords glued to digits/letters (`TMT10`, `wgs30x`,
  `DNBSEQ`); extension-less DICOM files in directories are still not detected (detection is by
  `.dcm`/`.ima`, as before); NIfTI-2 headers get no structure evidence (as for single files).
- **Offline SRA route is a landing page.** It is `streamable=True` as specified, so a caller who
  passes `plan(acc, offline=True).best` to `peek(route=...)` would sniff HTML. `peek()` itself
  builds an online plan, which never contains this route.
- **Not run:** the other test files (instructed). Their `profile()` assertions (workspace, neuro,
  tenx_spatial) were replayed by hand against the fixtures and pass, `Vault.scan(profile=True)` was
  run on a scratch copy of `fixtures/`, and `examples/library_demo.py` ran end to end. The
  orchestrate, remote and neuro tests that touch `acquire`/`catalog` were checked by reading only
  (orchestrate short-circuits offline SRA plans before looking at routes).

## Follow-ups outside my files

- `orchestrate.py` copies `route.cost_usd` into `Resolution.cost_usd` and sums it; an unknown-size
  metered route therefore still reports $0.00 there. It should read `Route.cost_estimate` /
  `cost_known`.
- `CHANGELOG.md`, `docs/acquire.md`, `docs/library.md`, `docs/selections.md`: worth a line each
  (offline SRA routes, `cheapest_cost` may be `None`, `fetch` returns `commands`, `to_bed` build
  rule, absolute paths, `annotate` strictness, `assay_from_profiler`).

## Verification

`cd /home/claude/work/B/geminga && PYTHONPATH=python python3 -m pytest tests/test_selections.py
tests/test_library.py tests/test_acquire.py tests/test_fixes_stores.py
tests/test_hyvesta_additions.py -q -x` -> `158 passed, 1 warning in 7.76s` (the warning is the
existing matplotlib `get_cmap` deprecation in `stage.py`). `ruff check` on the five modules and
four test files: all checks passed. Baseline before the work was 73 passed.
