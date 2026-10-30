# CHANGES-D: visual, neuroimaging, OME-Zarr and spatial fixes

Touched: `python/geminga/{neuro,ngff,spatial,stage,explorer,render}.py`; `tests/test_neuro.py`
(the `voxel_mm` assertion is now `(3.0, 1.0, 1.0)`, one `axcodes` assertion, two `guess_sequence`
cases); new `tests/test_fixes_visual.py` (80 regression tests, one section per finding).

Process: the tests were written first and run against the code as reviewed. 55 of the first 63
failed, each for the stated reason. The other 8 passed: 7 pin behaviour that must not change
(unscaled dtype kept, plain scatter points stay centred, the image `navigator`, interleaved-channel
reads, ...), and the eighth (an Explorer `record()` over an empty region) passed only because its
scenario never triggered an empty load, so it was reworked until it failed too. The new tests never
write into `fixtures/`: they copy the chip, build their own DICOM/NIfTI/TIFF/GEF inputs, and use
`tmp_path`.

## Fixes

1. **Rescale and dtype (neuro.py).** The output dtype is chosen once per volume and reported
   everywhere.
   - `DicomSeries`: the stored dtype if no slice carries a non-trivial `RescaleSlope`/
     `RescaleIntercept`, otherwise `promote_types(stored, float32)` (float32 for 16-bit data,
     float64 for 32-bit, which float32 cannot hold exactly). Slope and intercept are per slice
     (some MR vendors vary the slope), but the dtype is per series. `slab()` crops first, then
     rescales *in the output dtype*: 150 stored with slope 0.01 is 1.5, and a CT intercept of
     -1024 on uint16 no longer wraps to 64536. `dtype`, `nbytes` and `describe()` follow it; new
     `stored_dtype`, `rescaled`.
   - `NiftiVolume`: nibabel resets the in-memory header's slope/intercept after loading, so the
     scaling is read from `dataobj.slope/inter`. Scaled files get the same dtype rule and an
     explicit cast in `slab()` (nibabel's own scaling is float64); new `stored_dtype`, `scaled`,
     `scl_slope`, `scl_inter`. `VolumeView`'s `unit_bytes` comes from `vol.dtype`, so `cost()`
     equals `load()` (it under-reported 4x for a scaled int16). `read_bytes` still counts the
     file's own itemsize.
2. **Geometry (neuro.py).**
   - New `axcodes` on both classes, in array-axis order like `shape` and `voxel_mm`. NIfTI uses
     `nibabel.aff2axcodes` reversed, because the array is the file's axes 2, 1, 0; nibabel's own
     order stays available as `file_axcodes`. DICOM derives them from
     `ImageOrientationPatient`, assigning patient axes to array axes by the best permutation, so
     an oblique series still gets one code per axis. Unknown orientation is `('?', '?', '?')`.
   - `axis_index(...)` resolves an int (array axis), a plane (`'axial'`), a direction (`'S'`) or
     a pair (`'SI'`) through `axcodes`; named axes on an unknown orientation raise. Array axes are
     never reoriented, and this is documented.
   - `slab(..., axis=)` takes the first range along that axis. `y0:y1` and `x0:x1` crop the
     remaining two in ascending array order. The default `axis=0` is unchanged. Ranges are now
     clipped to the volume (a DICOM `slab` past the end used to return uninitialised rows).
   - `VolumeView.overview()` collapses the axis each plane is stacked along according to
     `axcodes`, so "axial_mip" of a sagittal acquisition runs head to foot instead of along array
     axis 0. Unknown orientation keeps the old convention.
   - `voxel_mm` is `(z, y, x)` from both classes (NIfTI was `(x, y, z)`). `ngff` no longer
     compensates for the old NIfTI order.
   - DICOM slice spacing is the median step of the `ImagePositionPatient` projections onto the
     slice normal (3.6 where `SliceThickness` says 3.0). `SliceThickness` is used only when
     positions are missing. Slices sort by that projection, or by InstanceNumber if any slice
     lacks a position. New `slice_spacing_mm`, `slice_thickness_mm`, `uniform_spacing`, and a
     RAS+ `affine` (cross-checked against nibabel for axial and oblique series).
3. **Never delete what is not ours (ngff.py, neuro.py).**
   - `plan_target()` decides before anything is touched. Missing or empty means create. A zarr
     node carrying `multiscales` (or `MaskSink3D`'s marker attribute) with nothing but zarr data
     in it is replaceable, and only with `overwrite=True`. Anything else is refused: foreign
     files, a plain zarr group, a regular file, a symlink. It raises `OverwriteRefused`, a
     `FileExistsError` and `ValueError`.
   - `to_ome_zarr` and `MaskSink3D` now default to `overwrite=False`. The old `_group()` called
     zarr's `mode="w"` unconditionally, which deletes the directory, so even `overwrite=False`
     wiped foreign files. It is now `mode="w-"` after the guard, so zarr itself refuses too.
     Replacement deletes zarr members only.
   - `MaskSink3D` writes a marker attribute and its `overwrite=False` resumes a store of the same
     shape and dtype, validated: `mode="a"` used to adopt a mismatched array silently. It
     recounts `voxels_labelled` and recovers the affine on reopen.
4. **GEF raster orientation (ngff.py).** The `[::-1]` flip is gone, so row `i` covers y bin `i`,
   matching the `translation` metadata and the image export. Levels are binned with integer
   division from `read_window` (`_gef_raster`) instead of `GefFile.rasterize`, whose float
   arithmetic drops some bin origins into the previous bin (see deferred).
   Tests export the GEF and the registered ssDNA image of the chip fixture and require the
   brighter lobe on the same row/column fraction (0.66 / 0.62), and an exact comparison against
   integer binning.
5. **Ticks and bins (stage.py).**
   - `axes()` picks round values (1/2/5 x 10^k, at most `ticks + 1`) inside the area labels may
     occupy, and centres each label on `Camera.world_to_px` of the value it prints: "3,224" was
     45 px from where 3224 maps. The lowest y label stays above the x label row. New
     `Camera.world_to_px` / `px_to_world` (`Stage.world_to_px` and `Explorer.px_to_world`
     delegate).
   - `points(..., cells=True)` draws each point as the square `[x, x+size) x [y, y+size)`,
     anchored at its low corner. `show()` and `Explorer` pass it for GEF views, so a bin covers
     exactly its footprint (a round point centred on the origin sat half a bin off) and
     neighbouring bins tile into a surface. The shader gets a `style` uniform (the camera buffer
     grows 32 to 48 bytes). Plain scatter points are unchanged. Tests render a known bin on
     square, wide and tall viewports, and through `show()` on a small GEF with a real bin pyramid.
6. **Empty layers (stage.py).** `points()` with zero points no longer builds a zero-length
   buffer (`TypeError: memoryview: cannot cast view with zeros`): it records an empty layer,
   `render()` guards on the buffer, `draw_calls` counts only layers that draw. Tests: the direct
   call; an Explorer that goes from a populated frame to an empty window (a real, empty
   reload: the finest bin fits by estimate); an Explorer whose first load is empty; and
   `show()`/`hud()`/`navigator` over an empty window.
7. **Stage, Explorer, render, spatial.**
   - `Stage.aggregate`: packs `(x, y)` relative to the minima (the old `x<<32 | y` borrowed from
     `x` for negative `y`: `(-1, -3)` came back as `(-2, 4294967293)`), with a row-wise `unique` fallback
     where 63 bits would overflow.
   - `render.navigator`: an `ImageView` window is `(y0, y1, x0, x1)`, a point-cloud view's
     (GEF, GEM) is `(x0, x1, y0, y1)` with y up; the overview extent, origin and red rectangle
     follow what the view is. GEF windows also crashed in the detail panel, which now hexbins
     them.
   - `explore()` accepts any `os.PathLike`.
   - `XeniumBundle.window(columns=...)` always reads `x_location`/`y_location` for the exact
     filter and drops them again unless asked for: you get the columns you asked for, in order.
     `columns=None` means the default set.
   - `_first()` and `VisiumBundle` use `glob.escape` on the directory (`slide[1]/`, `run?*/`).
   - `MultiChannelTiff` reads the requested channel's page segment by segment (tile, or strip of
     rows) and decodes only the segments the window touches (`s.asarray()` of the whole stack
     before: 320 MB peak for a 0.12 MB window of an 8 x 4096^2 file; 1.2 MB now, measured on that
     file). New `cost(channel, window)` reports `strategy` (`tiles`, `strips`, `full-plane`, or
     `full-stack` for a series whose pages do not map), `decode_bytes`, `resident_bytes`, `fits`
     and a `note` that says so when a whole plane must be decoded. New optional `budget=`: the
     resident bytes are leased before reading and released on return, so an unaffordable read is
     refused up front with `MemoryError`. Handles CYX, ZCYX (extra axes stay in front), YXS
     (sample selection) and separate-plane pages (full plane, leased). Fuzzed ad hoc against
     `tifffile` for tiled/strips/one-strip/deflate/odd sizes.
8. **Thin list.**
   - `Cohort`: files are visited in sorted order. Trees named by directory
     (`Center_01/Patient_02/...`) are keyed by their directory path down to the deepest
     identifier folder that is not a session folder: `Center_01/Patient_02`, with `Raw_Data/` and
     `Masks/` as containers and `ses-3`/`T2`/`tp1` as the session. A duplicate (subject,
     session, sequence) is kept as `FLAIR_2`, `FLAIR_3` and a `UserWarning` names both files, so
     nothing is overwritten silently. Sessions sort naturally (`2` before `10`). `layouts=` also
     takes functions. `report()` counts `FLAIR_2` as a FLAIR and reports `duplicates`. The three
     explicit layouts (BIDS, MSLesSeg, ISBI) are unchanged and tried first.
   - `guess_sequence`: role words (`mask`, `label`, `consensus`, `ground truth`) still win
     anywhere. Next to a modality, descriptive words (`seg`, `gt`, `lesion`, `wmh`) count only as
     whole words (or glued straight onto it, `FLAIRlesion`) and only after it. `MSLesSeg_P1_T1_FLAIR`
     is a FLAIR, `P1_T1_FLAIR_seg` and `FLAIR_lesion_mask` are masks. A name with no modality at
     all behaves exactly as before, and `SEQUENCE_PATTERNS["MASK"]` is unchanged. Against the old
     rule on 41 realistic names, 6 differ, all with a descriptive word (`MSLesSeg`, `LesionSeg`,
     `wmh_challenge`, `lesion_`, `segmented_`) *before* the modality.
   - `VolumeView.slices(axis)` cuts the window into runs of *whole* planes along the axis (axis 1
     and 2 used to yield fragments of a plane per z-chunk), leases each run, and releases on
     advance, at the end and on early close. Accepts anatomical names. Checked equal to `load()`
     for every axis, stride and an offset window. `_load`/`_chunks` hand out contiguous
     decimated copies, so the lease matches what is retained.
   - `MaskSink3D.voxels_labelled` counts distinct labelled voxels (write reads the region it
     replaces); new `overlap_voxels`; `write()` returns the patch's own count as before, validates
     the window and broadcasts a scalar label.
   - `to_nifti(path, affine=None)` uses the sink's affine and raises when there is none (no
     silent identity). The affine is stored with the store; new `MaskSink3D.like(volume, path)`
     takes shape and affine from a `NiftiVolume`, `DicomSeries` or `VolumeView`.
   - `to_ome_zarr`: sources are unwrapped with explicit `is None` tests (`ImageView(ndarray)` hit
     `a or b` on an array). RGB(A) `(H, W, 3|4)` is written `(c, y, x)` with a channel axis
     (`channels_last=` overrides the guess for a 3-D array). `labels=` (auto for a MASK NIfTI or a
     `MaskSink3D`) selects nearest pooling; intensities are averaged and rounded.
   - Pyramid levels keep the source dtype (uint16 stayed float32 from level 1). Band sizing now
     counts the padded copy, the accumulator and the cast-back; the existing 4 MB/16 MB band test
     passes.

## Behaviour changes to know about

- `to_ome_zarr` and `MaskSink3D` default to `overwrite=False` (was `True`).
- `NiftiVolume.voxel_mm` is `(z, y, x)`. `describe()["voxel_mm"]` and the profile metadata that
  copies it change order with it.
- Scaled NIfTI/DICOM slabs are float32 (float64 for 32-bit data) where they were int or float64.
- `XeniumBundle.window` no longer returns columns it was not asked for.
- `Stage.axes()` prints round tick values.
- A `MaskSink3D` directory written by an older version has no marker and is refused, with or
  without `overwrite`; add the `geminga_masksink` attribute or write to a new path.

## Declined / deferred

- **`GefFile.rasterize` mis-bins (stereoseq.py, not mine).** `((x - x0) / span * shape).astype(int)`
  puts an exact bin origin at `k - epsilon`, so the cell lands one bin low: along one axis of the
  chip, 24 of 4000 indices at bin 1, 18 of 400 at bin 10, 7 of 200 at bin 20. It also feeds the
  minimap and `StereoSeqView.image()`, and shows as faint grid lines in `render.chip()`'s window
  panel. Fix there with integer division, or `floor(... + 1e-9)`. The export avoids it.
- **Coarse-bin thumbnails.** `StereoSeqView._overview` (stereoseq.py) and `render.chip()` (mine,
  left alone) rasterise the coarsest level (bin 200: 20 cells across) into a few-hundred-pixel grid, so
  the overview is a lattice of isolated dots; the grid side should be the level's side. Not in
  the findings, and the navigator and chip figures in `assets/` would change.
- **`geminga.view(Path)` (`__init__.py`).** It dispatches on `isinstance(target, str)`, so a
  `Path` falls through to `open()`. `explore()` converts with `os.fspath`; `view()` needs the same.
- **Docs** (`docs/neuro.md`, `docs/explorer.md`, README): not mine. They should mention the
  `(z, y, x)` `voxel_mm`, the `overwrite=False` default, `MultiChannelTiff(budget=)` and
  `to_ome_zarr(labels=)`.
- **Label pyramids are nearest, not majority**: nearest is what ome-zarr-py does for labels and
  is exact in dtype and values; a majority vote erodes one-voxel lesions.
- **`MaskSink3D.to_nifti` still materialises the whole mask** (nibabel takes an array); now in the
  docstring.
- **Scaled NIfTI transient memory**: nibabel decodes a scaled slab to float64 before the cast, so
  a slab briefly costs 3x its float32 size. Avoiding it needs nibabel's private unscaled reader.
- **Zarr write buffers**: `peak_band_bytes` accounts our arrays, not zarr's codec concurrency (a
  6144^2 export at a 4 MB band traces ~10 MB). Unchanged by this work.
- **`VolumeView.load()` with a stride** still reads the full-resolution window and then
  decimates; `cost()['read_bytes']` says so. Redesign, not a fix.
- **Cross-owner coupling**: `VolumeView` relies on `View._hold/_chunk_cap/check/release`; if the
  `view.py` owner renames them, `neuro.py` must follow.
- **Fixtures**: opening any GEF from `fixtures/` makes `gefindex._sidecar_dir` create and delete
  a `.write-test` file in its (empty) `.gemidx` directory, changing that directory's mtime. The
  existing tests do this; no fixture file was modified, and the new tests copy the chip first.

## Verification

`ruff check` is clean on all owned files.
`PYTHONPATH=python python3 -m pytest tests/test_neuro.py tests/test_explorer_ngff.py tests/test_frames_stage.py tests/test_fixes_visual.py -q -x`
134 passed in 31.20s
