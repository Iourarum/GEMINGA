# MRI volumes

Lesion-segmentation datasets have a shape the rest of this library had not met: one *volume*
per scan, several sequences per session (T1w, T2w, FLAIR), several sessions per subject, and an
expert mask alongside. MSLesSeg is 75 subjects × up to 4 timepoints × 3 sequences — a few
hundred volumes of 10–50 MB. Not big by omics standards. What is big is the pipeline: a
segmentation model wants a whole volume, or a patch, or a batch of patches, on a GPU with a
fixed amount of memory, and "fits" is decided per slab, not per file.

## Readers that are lazy in the axis that matters

```python
v = g.NiftiVolume("P01_T1_FLAIR.nii")     # 352-byte header: shape, dtype, voxel size, sequence
v.slab(10, 14, 20, 60, 30, 70)            # slices 10–13, cropped in-plane, as (Z, Y, X)

d = g.DicomSeries("P04_FLAIR/")           # headers only, stop_before_pixels=True
d.describe()                               # modality, series description, field strength, ...
d.slab(20, 23)                             # reads exactly three slice files
```

DICOM slices are ordered by their position along the slice normal (from
`ImagePositionPatient` and `ImageOrientationPatient`), which is the physically correct order
and survives files being numbered oddly. A directory mixing several series keeps the largest.

:::{admonition} Gzipped NIfTI cannot be seeked
:class: measured
Reading slab `[30:40]` of a `.nii.gz` means decompressing everything before it. Memory stays
bounded — that is the point — but time is linear in the offset, and `cost()` says so:

```text
gzipped: keep 61.2 KB but must read 720.0 KB — linear (gzip): reads from the start to slice 40
plain  : keep 61.2 KB, reads 180.0 KB — random
```

If you are going to revisit a volume, store it uncompressed in staging.
:::

## The 3-D view

`g.view("scan.nii.gz", budget=b)` gives a `VolumeView` with a `(z0, z1, y0, y1, x0, x1)`
window and the same zoom / pan / cost / `auto_zoom` as every other view. One pair of fractions
applies to all three axes:

```python
v.select_frac(0.25, 0.75)         # the central half in z, y and x
v.zoom_in(2)                       # resolution rises to match, cost stays flat
for slab, window in v.chunks():    # slab-wise, budget-sized
    ...
for sl, window, i in v.slices(axis=0):   # one 2-D slice at a time, for a 2-D model
    ...
```

`overview()` returns three orthogonal maximum-intensity projections at low resolution.

## Cohorts: subjects, sessions, sequences

```python
co = g.Cohort("MSLesSeg/train")
co.print_report()
```

```text
cohort MSLesSeg/train
  3 subjects, 5 sessions, 2 longitudinal, 3 with masks
  sequences: FLAIR=3, MASK=3, T1w=3, T2w=3
    P01          1:FLAIR/MASK/T1w/T2w, 2:FLAIR/MASK/T1w/T2w
```

Layouts are recognised by regex — BIDS (`sub-01/ses-02`), MSLesSeg-style (`P01/T1`),
ISBI-style (`training01_02`) — and you can pass your own. Then the questions a lesion study
asks first: `co.longitudinal()`, `co.with_mask()`, `co.complete(("T1w", "T2w", "FLAIR"))`.

Sequence names are conventionally the **last** token: `P01_T1_T2w.nii.gz` is a T2w scan at
timepoint 1. `guess_sequence` takes the latest-ending match, and a mask token anywhere wins.

### The longitudinal look

```python
views = co.paired_windows("P01", "MASK", budget=b)
for ses, vv in views.items():
    print(ses, int((vv.select(14, 34, 30, 70, 20, 70).load() > 0).sum()))
# 1: 371 lesion voxels     2: 757 lesion voxels   <- a new lesion, and growth
```

## Writing masks

`MaskSink3D` writes a full-resolution label volume slab by slab to Zarr, so segmenting a 512³
scan never needs the whole output in memory; `to_nifti()` produces the single file downstream
tools expect. Measured on the fixture: Dice 1.000 against the expert mask, budget peak 4 MB.

## Every view now accounts its load

A review finding worth stating: until v0.7, `View.load()` gated on the budget via `check()`
but never *reserved* — peak stayed at 0 B. Every view now holds a lease for the last thing it
returned, released on the next `load()`/`chunks()` call, on `release()`, or on leaving a
`with` block.

```text
after load:    used 360.0 KB, peak 360.0 KB   (data is 360.0 KB)
after release: used 0 B
```

## Where these datasets live

| host | route | note |
|---|---|---|
| figshare (MS3SEG, MSLesSeg) | free, Range-capable | API lists files with sizes and S3-redirect URLs |
| Zenodo (Shifts 2022) | free, Range-capable | DOI-stable; API gives checksums |
| OpenNeuro (`ds00xxxx`) | **AWS Open Data — AWS pays** | `--no-sign-request`; BIDS layout, which `Cohort` reads directly |
| NITRC (ISBI 2015) | free, account needed | a bare URL may return the login page |
| Kaggle mirrors | free, API token needed | some datasets require accepting terms in a browser |
| Mendeley Data | free | per-file links on the landing page |
| FLI-IAM (MSSEG-1/2) | registration | challenge portal; no automated route |
| UK Biobank | application **and fee** | not a data-transfer problem; a governance one |
| dbGaP (`phs000171`) | controlled | needs an approved project and an `.ngc` key |

The profiler recognises NIfTI (magic at byte 344), gzipped NIfTI by name, DICOM (magic at
byte 128, then `Modality=MR`), and reads the sequence from the filename or
`SeriesDescription`. Masks are flagged `is_label_volume`.
