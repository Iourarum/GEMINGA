"""Regression tests for the visual / neuroimaging / OME-Zarr / spatial review findings.

One section per finding (numbers match CHANGES-D.md). Every test here failed against the code
as reviewed; each builds its own small inputs so none depends on a fixture except where a
fixture *is* the point (the chip, the Visium and CODEX bundles).
"""
import json
import os
import shutil

import numpy as np
import pytest

import geminga as g

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")
GEF = os.path.join(FIX, "chip.gef")
SSDNA = os.path.join(FIX, "chip_ssdna.tif")
VISIUM = os.path.join(FIX, "glioma", "visium")
need_chip = pytest.mark.skipif(not (os.path.exists(GEF) and os.path.exists(SSDNA)),
                               reason="run the fixture scripts first")
need_visium = pytest.mark.skipif(not os.path.exists(VISIUM), reason="no Visium fixture")
needs_gpu = pytest.mark.skipif(not g.stage_available(), reason="no wgpu adapter")


@pytest.fixture(scope="session")
def chip_gef(tmp_path_factory):
    """The chip fixture, copied: opening a GEF probes its `.gemidx` sidecar directory with a
    write test, and these tests must not write into fixtures/ — not even that."""
    if not os.path.exists(GEF):
        pytest.skip("no chip fixture")
    dst = tmp_path_factory.mktemp("chip") / "chip.gef"
    shutil.copy(GEF, dst)
    return str(dst)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def _dicom_series(root, vol, *, slope=None, inter=None, slopes=None, signed=False,
                  thickness=3.0, step=3.0, iop=(1, 0, 0, 0, 1, 0), positions=True,
                  modality="MR", desc="AX FLAIR", spacing=(1.0, 1.0)):
    """Write `vol` (Z, Y, X) as single-slice DICOM files and return the directory."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    root = str(root)
    os.makedirs(root, exist_ok=True)
    suid = generate_uid()
    normal = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
    for z in range(vol.shape[0]):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.PatientID = "T"
        ds.Modality = modality
        ds.SeriesDescription = desc
        ds.SeriesInstanceUID = suid
        ds.Rows, ds.Columns = int(vol.shape[1]), int(vol.shape[2])
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1 if signed else 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelSpacing = list(spacing)
        ds.SliceThickness = thickness
        ds.InstanceNumber = z + 1
        if positions:
            ds.ImageOrientationPatient = list(iop)
            ds.ImagePositionPatient = [round(float(v), 3) for v in normal * step * z]
        s = slopes[z] if slopes is not None else slope
        if s is not None:
            ds.RescaleSlope = s
            ds.RescaleIntercept = 0.0 if inter is None else inter
        ds.PixelData = np.ascontiguousarray(vol[z].astype(np.int16 if signed else np.uint16)).tobytes()
        ds.save_as(os.path.join(root, f"s{(z * 7) % vol.shape[0]:03d}_{z:03d}.dcm"),
                   enforce_file_format=True)
    return root


def _nifti(path, arr_xyz, affine=None, dtype=None, slope=None, inter=None):
    import nibabel as nib
    img = nib.Nifti1Image(np.asarray(arr_xyz), np.eye(4) if affine is None else affine)
    if dtype is not None:
        img.header.set_data_dtype(dtype)
    if slope is not None:
        img.header.set_slope_inter(slope, inter or 0.0)
    nib.save(img, str(path))
    return str(path)


#: a sagittal acquisition: file axes (0, 1, 2) point to (A, S, L), voxels 1 x 2 x 5 mm
SAGITTAL = np.array([[0, 0, -5.0, 0], [1.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 0, 1.0]])


def _sagittal_volume(tmp_path):
    arr = np.zeros((10, 12, 8), np.int16)              # file order: A-P x S-I x L-R
    arr[3, 9, 5] = 1000                                # one bright voxel
    return _nifti(tmp_path / "sag.nii", arr, SAGITTAL, dtype=np.int16)


# ---------------------------------------------------------------------------
# 1. rescale slope / intercept: dtype chosen once, cost() == load()
# ---------------------------------------------------------------------------

def test_dicom_rescale_slope_is_not_truncated_into_the_stored_dtype(tmp_path):
    stored = np.full((4, 8, 8), 150, np.uint16)
    d = g.DicomSeries(_dicom_series(tmp_path / "s", stored, slope=0.01, inter=0.0))
    a = d.slab(0, 4)
    assert d.dtype == np.float32 and a.dtype == np.float32
    assert np.allclose(a, 1.5)                          # 150 * 0.01, not int(1)
    assert d.nbytes == a.nbytes                         # nbytes reports what slab() hands out
    v = g.VolumeView(d, g.Budget(ram="16MB"))
    assert v.cost()["total_bytes"] == v.load().nbytes   # cost() matches load()
    assert d.describe()["dtype"] == "float32"


def test_dicom_ct_intercept_does_not_wrap_unsigned_storage(tmp_path):
    hu = np.linspace(-1000, 400, 4 * 6 * 6).reshape(4, 6, 6).round().astype(np.int32)
    stored = (hu + 1024).astype(np.uint16)             # CT stored unsigned, intercept -1024
    d = g.DicomSeries(_dicom_series(tmp_path / "ct", stored, slope=1.0, inter=-1024.0,
                                    modality="CT", desc="CT chest"))
    a = d.slab(0, 4)
    assert a.dtype.kind == "f" and a.min() < 0          # the old code wrapped to 64536
    assert np.array_equal(a, hu.astype(np.float32))


def test_dicom_without_rescale_keeps_the_stored_dtype(tmp_path):
    stored = (np.arange(3 * 5 * 5).reshape(3, 5, 5) * 40).astype(np.int16)
    d = g.DicomSeries(_dicom_series(tmp_path / "p", stored, signed=True))
    assert d.dtype == np.int16 and d.slab(0, 3).dtype == np.int16
    d2 = g.DicomSeries(_dicom_series(tmp_path / "q", stored, signed=True, slope=1.0, inter=0.0))
    assert d2.dtype == np.int16                          # identity rescale is not a rescale
    assert np.array_equal(d2.slab(0, 3), stored)


def test_dicom_per_slice_slopes_share_one_dtype(tmp_path):
    stored = np.full((3, 4, 4), 100, np.uint16)
    d = g.DicomSeries(_dicom_series(tmp_path / "v", stored, slopes=[0.5, 1.0, 2.0]))
    a = d.slab(0, 3)
    assert a.dtype == np.float32                         # one dtype for the series, not per slice
    assert [float(a[i, 0, 0]) for i in range(3)] == [50.0, 100.0, 200.0]


def test_scaled_nifti_reports_the_dtype_it_returns(tmp_path):
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 200, (6, 5, 4)).astype(np.int16)           # file order (x, y, z)
    p = _nifti(tmp_path / "scaled.nii", raw, dtype=np.int16, slope=0.5, inter=10.0)
    v = g.NiftiVolume(p)
    s = v.slab(0, v.shape[0])
    assert s.dtype == v.dtype == np.float32 and v.stored_dtype == np.int16
    assert np.allclose(s, raw.transpose(2, 1, 0) * 0.5 + 10.0)
    assert v.describe()["dtype"] == "float32"
    view = g.VolumeView(v, g.Budget(ram="16MB"))
    assert view.cost()["total_bytes"] == view.load().nbytes == s.nbytes   # was 4x under-reported
    v.close()


def test_unscaled_nifti_keeps_its_dtype(tmp_path):
    raw = np.arange(6 * 5 * 4).reshape(6, 5, 4).astype(np.int16)
    v = g.NiftiVolume(_nifti(tmp_path / "plain.nii", raw, dtype=np.int16))
    assert v.dtype == np.int16 and v.slab(0, 4).dtype == np.int16
    v.close()


# ---------------------------------------------------------------------------
# 2. geometry: axcodes, voxel_mm order, DICOM spacing
# ---------------------------------------------------------------------------

def test_nifti_exposes_axcodes_and_voxel_mm_in_array_axis_order(tmp_path):
    v = g.NiftiVolume(_sagittal_volume(tmp_path))
    assert v.shape == (8, 12, 10)                        # (Z, Y, X) of the *file's* axes 2, 1, 0
    assert v.file_axcodes == ("A", "S", "L")             # nibabel's own answer
    assert v.axcodes == ("L", "S", "A")                  # the same, per array axis of a slab
    assert v.voxel_mm == (5.0, 2.0, 1.0)                 # (z, y, x), matching shape
    assert v.axis_index("axial") == 1 and v.axis_index("S") == 1 and v.axis_index("IS") == 1
    assert v.axis_index("coronal") == 2 and v.axis_index("sagittal") == 0
    assert v.axis_index(0) == 0 and v.axis_index("L") == 0
    with pytest.raises(ValueError):
        v.axis_index("up")
    v.close()


def test_axial_mip_follows_the_anatomy_not_array_axis_zero(tmp_path):
    v = g.VolumeView(g.NiftiVolume(_sagittal_volume(tmp_path)))
    ov = v.overview(256)
    ax, co, sa = ov.series["axial_mip"], ov.series["coronal_mip"], ov.series["sagittal_mip"]
    # the bright voxel sits at array index (z, y, x) = (5, 9, 3)
    assert ax.shape == (8, 10) and np.unravel_index(ax.argmax(), ax.shape) == (5, 3)
    assert co.shape == (8, 12) and np.unravel_index(co.argmax(), co.shape) == (5, 9)
    assert sa.shape == (12, 10) and np.unravel_index(sa.argmax(), sa.shape) == (9, 3)
    assert ov.series["thumbnail"].shape == ax.shape
    v.close()


def test_axial_mip_is_unchanged_for_an_axial_acquisition(tmp_path):
    arr = np.zeros((6, 7, 5), np.int16)                  # file order x, y, z; RAS+
    arr[2, 3, 4] = 50
    p = _nifti(tmp_path / "ax.nii", arr, np.diag([1.0, 1.0, 3.0, 1.0]), dtype=np.int16)
    vol = g.NiftiVolume(p)
    assert vol.axcodes == ("S", "A", "R")
    ov = g.VolumeView(vol).overview(64)
    assert ov.series["axial_mip"].shape == (7, 6)        # max over the slice axis, as before
    vol.close()


def test_slab_axis_resolves_anatomical_names(tmp_path):
    vol = g.NiftiVolume(_sagittal_volume(tmp_path))
    full = vol.slab(0, vol.shape[0])
    # three axial slices (S-I runs along array axis 1): rows 8:11 of axis 1
    ax = vol.slab(8, 11, axis="axial")
    assert ax.shape == (8, 3, 10) and np.array_equal(ax, full[:, 8:11, :])
    # by letter, with a crop in the two remaining axes (array order: z then x)
    s = vol.slab(8, 11, 4, 6, 2, 5, axis="I")
    assert s.shape == (2, 3, 3) and np.array_equal(s, full[4:6, 8:11, 2:5])
    assert vol.slab(2, 4, axis=2).shape == (8, 12, 2)
    vol.close()


def test_dicom_slice_spacing_comes_from_positions_not_thickness(tmp_path):
    stored = np.zeros((5, 6, 7), np.uint16)
    d = g.DicomSeries(_dicom_series(tmp_path / "gap", stored, thickness=3.0, step=3.6,
                                    spacing=(0.9, 0.8)))
    assert d.voxel_mm == pytest.approx((3.6, 0.9, 0.8))   # (z, y, x); PixelSpacing is (row, col)
    assert d.describe()["slice_thickness_mm"] == 3.0
    assert d.axcodes == ("S", "P", "L")                   # axial, DICOM's LPS patient frame
    assert d.axis_index("axial") == 0


def test_dicom_without_positions_falls_back_to_slice_thickness(tmp_path):
    stored = np.zeros((4, 5, 5), np.uint16)
    d = g.DicomSeries(_dicom_series(tmp_path / "nopos", stored, thickness=2.5, positions=False))
    assert d.voxel_mm[0] == 2.5
    assert d.axcodes == ("?", "?", "?")
    with pytest.raises(ValueError, match="orientation"):
        d.axis_index("S")                                # a named axis needs a known orientation
    ov = g.VolumeView(d).overview(32)                    # but the overview still works: it
    assert ov.series["axial_mip"].shape == (5, 5)        # keeps the old slice-axis convention


def test_sagittal_dicom_axial_projection_goes_through_the_row_axis(tmp_path):
    # rows run inferior, columns run posterior, slices step toward the patient's right
    stored = np.zeros((4, 6, 5), np.uint16)
    stored[1, 4, 3] = 900
    d = g.DicomSeries(_dicom_series(tmp_path / "sagd", stored, iop=(0, 1, 0, 0, 0, -1)))
    assert d.axcodes == ("R", "I", "P")
    assert d.axis_index("axial") == 1
    ov = g.VolumeView(d).overview(32)
    assert ov.series["axial_mip"].shape == (4, 5)
    assert np.unravel_index(ov.series["axial_mip"].argmax(), (4, 5)) == (1, 3)


# ---------------------------------------------------------------------------
# 3. overwrite: never delete what is not ours (OME-Zarr and MaskSink3D)
# ---------------------------------------------------------------------------

def _tiny_image():
    return np.arange(64 * 80, dtype=np.uint16).reshape(64, 80)


def test_to_ome_zarr_refuses_a_directory_with_foreign_files(tmp_path):
    keep = tmp_path / "thesis"
    keep.mkdir()
    (keep / "chapter1.txt").write_text("irreplaceable")
    for kw in ({}, {"overwrite": True}):                  # the default must not wipe either
        with pytest.raises(FileExistsError):
            g.to_ome_zarr(_tiny_image(), str(keep), levels=2, **kw)
        assert (keep / "chapter1.txt").read_text() == "irreplaceable"


def test_to_ome_zarr_default_does_not_replace_an_existing_export(tmp_path):
    out = str(tmp_path / "a.ome.zarr")
    g.to_ome_zarr(_tiny_image(), out, levels=2)
    before = sorted(os.listdir(out))
    with pytest.raises(FileExistsError):
        g.to_ome_zarr(_tiny_image() + 1, out, levels=2)
    assert sorted(os.listdir(out)) == before
    r = g.to_ome_zarr(_tiny_image() + 1, out, levels=2, overwrite=True)   # explicit: allowed
    assert r["kind"] == "image"
    import zarr
    assert int(zarr.open_group(out, mode="r")["0"][0, 0]) == 1


def test_to_ome_zarr_overwrite_allows_missing_and_empty_targets(tmp_path):
    g.to_ome_zarr(_tiny_image(), str(tmp_path / "new.ome.zarr"), levels=2, overwrite=True)
    (tmp_path / "empty").mkdir()
    g.to_ome_zarr(_tiny_image(), str(tmp_path / "empty"), levels=2, overwrite=True)
    assert g.read_ome_zarr_meta(str(tmp_path / "empty"))["shapes"][0] == (64, 80)


def test_to_ome_zarr_overwrite_refuses_an_export_that_has_foreign_files_inside(tmp_path):
    out = tmp_path / "mixed.ome.zarr"
    g.to_ome_zarr(_tiny_image(), str(out), levels=2)
    (out / "notes.txt").write_text("keep me")
    with pytest.raises(FileExistsError):
        g.to_ome_zarr(_tiny_image(), str(out), levels=2, overwrite=True)
    assert (out / "notes.txt").exists()


def test_to_ome_zarr_overwrite_refuses_a_plain_zarr_group(tmp_path):
    import zarr
    out = str(tmp_path / "plain.zarr")
    grp = zarr.open_group(out, mode="w", zarr_format=2)
    grp.create_array("data", shape=(4,), dtype="int32", fill_value=0)
    with pytest.raises(FileExistsError):
        g.to_ome_zarr(_tiny_image(), out, levels=2, overwrite=True)
    assert "data" in zarr.open_group(out, mode="r")


def test_gef_export_does_not_wipe_a_foreign_directory(tmp_path, chip_gef):
    keep = tmp_path / "keep"
    keep.mkdir()
    (keep / "x.txt").write_text("x")
    with pytest.raises(FileExistsError):
        g.to_ome_zarr(g.open_gef(chip_gef), str(keep), max_side=300, overwrite=True)
    assert (keep / "x.txt").exists()


def test_masksink_does_not_wipe_a_foreign_directory(tmp_path):
    keep = tmp_path / "labels"
    keep.mkdir()
    (keep / "ground_truth.txt").write_text("precious")
    for kw in ({}, {"overwrite": True}):
        with pytest.raises(FileExistsError):
            g.MaskSink3D(str(keep), shape=(4, 6, 6), **kw)
        assert (keep / "ground_truth.txt").read_text() == "precious"


def test_masksink_overwrite_replaces_only_its_own_store(tmp_path):
    p = str(tmp_path / "pred.zarr")
    s1 = g.MaskSink3D(p, shape=(4, 6, 6))
    s1.write(np.ones((4, 6, 6), np.uint8), (0, 4, 0, 6, 0, 6))
    with pytest.raises(FileExistsError, match="overwrite"):
        g.MaskSink3D(p, shape=(5, 6, 6))                   # a different shape cannot resume
    s2 = g.MaskSink3D(p, shape=(5, 6, 6), overwrite=True)  # its own store: replace is fine
    assert s2.z.shape == (5, 6, 6) and int(np.asarray(s2.z[:]).sum()) == 0


def test_masksink_resumes_by_default_and_remembers_what_it_labelled(tmp_path):
    p = str(tmp_path / "resume.zarr")
    s1 = g.MaskSink3D(p, shape=(4, 6, 6))
    s1.write(np.ones((2, 6, 6), np.uint8), (0, 2, 0, 6, 0, 6))
    s2 = g.MaskSink3D(p, shape=(4, 6, 6))
    assert s2.voxels_labelled == 72                        # picked up from the store
    s2.write(np.ones((2, 6, 6), np.uint8), (2, 4, 0, 6, 0, 6))
    assert int(np.asarray(s2.z[:]).sum()) == 144 and s2.voxels_labelled == 144


# ---------------------------------------------------------------------------
# 8. neuro leftovers
# ---------------------------------------------------------------------------

def test_guess_sequence_modality_token_beats_a_dataset_name_containing_seg():
    assert g.guess_sequence("MSLesSeg_P1_T1_FLAIR.nii.gz") == "FLAIR"
    assert g.guess_sequence("MSLesSeg_P1_T1_MASK.nii.gz") == "MASK"
    assert g.guess_sequence("WMH_challenge_P1_T2w.nii.gz") == "T2w"
    assert g.guess_sequence("P1_T1_FLAIR_seg.nii.gz") == "MASK"        # role suffix still wins
    assert g.guess_sequence("FLAIR_lesion_mask.nii.gz") == "MASK"
    assert g.guess_sequence("sub-01_gt.nii") == "MASK"
    assert g.guess_sequence("weight_map_FLAIR.nii") == "FLAIR"         # 'gt' inside a word
    assert g.guess_sequence("Consensus.nii.gz") == "MASK"
    assert g.guess_sequence("training01_02_mask1.nii") == "MASK"
    assert g.guess_sequence("FLAIRlesion.nii") == "MASK"               # glued straight onto it
    assert g.guess_sequence("FlairSeg.nii") == "MASK" and g.guess_sequence("T1gt.nii") == "MASK"
    assert g.guess_sequence("LesionSeg_P3_FLAIR.nii.gz") == "FLAIR"


def test_slices_along_every_axis_lease_per_chunk_and_yield_whole_planes(tmp_path):
    rng = np.random.default_rng(1)
    raw = rng.integers(0, 1000, (40, 32, 24)).astype(np.int16)         # file order x, y, z
    p = _nifti(tmp_path / "v.nii", raw, np.diag([1.0, 1.0, 1.0, 1.0]), dtype=np.int16)
    full = g.NiftiVolume(p).slab(0, 24)                                # (24, 32, 40)
    for axis in (0, 1, 2):
        b = g.Budget(ram="32KB", timeout_s=2.0)                        # forces several chunks
        v = g.VolumeView(g.NiftiVolume(p), b).reset()
        seen, leased = 0, []
        for plane, win, i in v.slices(axis=axis):
            leased.append(b.stats()["ram"]["used"])
            if seen == 0:
                assert plane.shape == tuple(s for k, s in enumerate(full.shape) if k != axis)
            seen += 1
        assert seen == full.shape[axis]
        assert min(leased) > 0, f"axis {axis}: planes were yielded with nothing leased"
        assert max(leased) <= 32 * 1024
        assert b.stats()["ram"]["grants"] > 1                          # leased chunk by chunk
        assert b.stats()["ram"]["used"] == 0                           # and released at the end
        v.close()


@pytest.mark.parametrize("zoom", [0, 1, 2])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_slices_content_matches_load_for_every_axis_stride_and_window(tmp_path, axis, zoom):
    rng = np.random.default_rng(2)
    raw = rng.integers(0, 1000, (50, 41, 33)).astype(np.int16)         # file order x, y, z
    p = _nifti(tmp_path / "w.nii", raw, np.diag([1.0, 1.0, 2.0, 1.0]), dtype=np.int16)
    win = (3, 30, 5, 38, 7, 45)                                        # an offset window
    ref = g.VolumeView(g.NiftiVolume(p), g.Budget(ram="64MB")).select(*win).set_zoom(zoom)
    b = g.Budget(ram="6KB", timeout_s=2.0)                             # a few planes a chunk
    v = g.VolumeView(g.NiftiVolume(p), b).select(*win).set_zoom(zoom)
    planes = [pl for pl, _w, _i in v.slices(axis=axis)]
    assert np.array_equal(np.stack(planes, axis=axis), ref.load())
    assert b.stats()["ram"]["used"] == 0


def test_slices_release_their_lease_when_closed_early(tmp_path):
    p = _nifti(tmp_path / "e.nii", np.zeros((30, 20, 10), np.int16), np.eye(4), dtype=np.int16)
    b = g.Budget(ram="2KB", timeout_s=2.0)
    it = g.VolumeView(g.NiftiVolume(p), b).slices(axis=0)
    next(it)
    assert b.stats()["ram"]["used"] > 0
    it.close()
    assert b.stats()["ram"]["used"] == 0


def test_slices_accepts_an_anatomical_axis(tmp_path):
    v = g.VolumeView(g.NiftiVolume(_sagittal_volume(tmp_path)))
    planes = [pl for pl, _w, _i in v.slices(axis="axial")]
    assert len(planes) == 12 and planes[0].shape == (8, 10)


def _touch(root, *rels):
    for r in rels:
        p = os.path.join(root, *r.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").close()


def test_cohort_keys_nested_trees_by_their_directory_path(tmp_path):
    _touch(str(tmp_path),
           "Center_01/Patient_01/FLAIR.nii.gz", "Center_01/Patient_02/FLAIR.nii.gz",
           "Center_01/Patient_02/T1.nii.gz", "Center_02/Patient_01/FLAIR.nii.gz")
    co = g.Cohort(str(tmp_path))
    assert set(co.subjects) == {"Center_01/Patient_01", "Center_01/Patient_02",
                                "Center_02/Patient_01"}
    assert co["Center_01/Patient_02"].sequences() == {"FLAIR", "T1w"}
    assert co["Center_02/Patient_01"].sequences() == {"FLAIR"}
    assert not co.unassigned


def test_cohort_nested_tree_with_container_folders_merges_one_patient(tmp_path):
    _touch(str(tmp_path),
           "Center_01/Patient_01/Raw_Data/3DFLAIR.nii.gz",
           "Center_01/Patient_01/Preprocessed_Data/FLAIR_preprocessed.nii.gz",
           "Center_01/Patient_01/Masks/Consensus.nii.gz",
           "Center_01/Patient_02/Preprocessed_Data/FLAIR_preprocessed.nii.gz")
    with pytest.warns(UserWarning, match="FLAIR"):
        co = g.Cohort(str(tmp_path))
    assert set(co.subjects) == {"Center_01/Patient_01", "Center_01/Patient_02"}
    ses = co["Center_01/Patient_01"].sessions["1"]
    assert {"FLAIR", "FLAIR_2", "MASK"} == set(ses)                    # nothing overwritten
    # walked in sorted order: Masks, Preprocessed_Data, Raw_Data
    assert ses["FLAIR"].endswith("Preprocessed_Data/FLAIR_preprocessed.nii.gz")
    assert ses["FLAIR_2"].endswith("Raw_Data/3DFLAIR.nii.gz")


def test_cohort_never_overwrites_a_duplicate_silently(tmp_path):
    _touch(str(tmp_path), "train/P01/T1/P01_T1_FLAIR.nii", "train/P01/T1/redo/P01_T1_FLAIR.nii.gz",
           "train/P01/T1/P01_T1_MASK.nii.gz")
    with pytest.warns(UserWarning, match="P01"):
        co = g.Cohort(str(tmp_path))
    ses = co["P01"].sessions["1"]
    assert set(ses) == {"FLAIR", "FLAIR_2", "MASK"}
    assert {os.path.basename(ses["FLAIR"]), os.path.basename(ses["FLAIR_2"])} == {
        "P01_T1_FLAIR.nii", "P01_T1_FLAIR.nii.gz"}
    # deterministic whatever the filesystem's listing order: sorted path wins the bare name
    assert ses["FLAIR"].endswith("P01_T1_FLAIR.nii") and "redo" in ses["FLAIR_2"]


def test_cohort_sorts_numeric_sessions_numerically(tmp_path):
    _touch(str(tmp_path), *[f"sub-01/ses-{n}/anat/sub-01_ses-{n}_T1w.nii.gz" for n in (10, 2, 1)])
    co = g.Cohort(str(tmp_path))
    assert co["01"].timepoints == ["1", "2", "10"]
    assert co["01"].longitudinal


def test_masksink_counts_unique_voxels_across_overlapping_patches(tmp_path):
    s = g.MaskSink3D(str(tmp_path / "o.zarr"), shape=(8, 8, 8))
    ones = np.ones((4, 4, 4), np.uint8)
    assert s.write(ones, (0, 4, 0, 4, 0, 4)) == 64
    assert s.write(ones, (2, 6, 2, 6, 2, 6)) == 64              # per-patch count unchanged
    assert s.voxels_labelled == 120                             # 64 + 64 - 8 shared
    assert s.overlap_voxels == 8
    assert int((np.asarray(s.z[:]) > 0).sum()) == s.voxels_labelled
    s.write(np.zeros((4, 4, 4), np.uint8), (0, 4, 0, 4, 0, 4))  # erasing lowers the count
    assert s.voxels_labelled == int((np.asarray(s.z[:]) > 0).sum()) == 56   # patch 2 minus shared


def test_masksink_to_nifti_requires_an_affine_and_propagates_the_source(tmp_path):
    import nibabel as nib
    p = _nifti(tmp_path / "src.nii", np.zeros((6, 5, 4), np.int16), SAGITTAL, dtype=np.int16)
    vol = g.NiftiVolume(p)
    bare = g.MaskSink3D(str(tmp_path / "bare.zarr"), shape=vol.shape)
    bare.write(np.ones((1, 5, 6), np.uint8), (0, 1, 0, 5, 0, 6))
    with pytest.raises(ValueError, match="affine"):
        bare.to_nifti(str(tmp_path / "bare.nii.gz"))            # no silent identity
    out = bare.to_nifti(str(tmp_path / "given.nii.gz"), affine=vol.affine)
    assert np.allclose(nib.load(out).affine, vol.affine)
    like = g.MaskSink3D.like(vol, str(tmp_path / "like.zarr"))
    assert like.z.shape == vol.shape and np.allclose(like.affine, vol.affine)
    out2 = like.to_nifti(str(tmp_path / "like.nii.gz"))
    assert np.allclose(nib.load(out2).affine, vol.affine)
    again = g.MaskSink3D(str(tmp_path / "like.zarr"), shape=vol.shape)   # reopened: remembers it
    assert np.allclose(again.affine, vol.affine)
    vol.close()


# ---------------------------------------------------------------------------
# 4. GEF raster orientation agrees with the image export
# ---------------------------------------------------------------------------

def _peak(a):
    """(row, column) fraction of the centroid of the cells within 80% of the maximum."""
    a = np.asarray(a, np.float64)
    ys, xs = np.nonzero(a >= 0.8 * a.max())
    return (ys.mean() + 0.5) / a.shape[0], (xs.mean() + 0.5) / a.shape[1]


@need_chip
def test_gef_and_image_exports_put_a_feature_on_the_same_row(tmp_path, chip_gef):
    import zarr
    gef_out, img_out = str(tmp_path / "gef.ome.zarr"), str(tmp_path / "img.ome.zarr")
    g.to_ome_zarr(g.open_gef(chip_gef), gef_out, max_side=100)
    g.to_ome_zarr(g.TiffTileArray(SSDNA), img_out, levels=6)
    gm, im = g.read_ome_zarr_meta(gef_out), g.read_ome_zarr_meta(img_out)
    gi = gm["shapes"].index((40, 40))                      # bin 100 over the 4000-unit chip
    ii = im["shapes"].index((64, 64))
    grow, gcol = _peak(zarr.open_group(gef_out, mode="r")[str(gi)][:])
    irow, icol = _peak(zarr.open_group(img_out, mode="r")[str(ii)][:])
    # the generator puts the brighter lobe at (row, col) = (0.66, 0.62) of the chip in both
    assert grow == pytest.approx(irow, abs=0.04) and gcol == pytest.approx(icol, abs=0.04)
    assert grow == pytest.approx(0.66, abs=0.04) and gcol == pytest.approx(0.62, abs=0.04)


@need_chip
def test_gef_export_row_i_covers_y_bin_i(tmp_path, chip_gef):
    import zarr
    out = str(tmp_path / "rows.ome.zarr")
    gef = g.open_gef(chip_gef).gef
    g.to_ome_zarr(gef, out, max_side=100)
    m = g.read_ome_zarr_meta(out)
    k = [d["path"] for d in m["multiscales"]["datasets"]][m["shapes"].index((40, 40))]
    arr = zarr.open_group(out, mode="r")[k][:]
    ex = gef.extent
    d = gef.read_window(ex, 100)
    want = np.zeros((40, 40), np.float32)
    np.add.at(want, (d["y"] // 100, d["x"] // 100), d["count"].astype(np.float32))
    assert np.array_equal(arr, want)                        # row = y bin, column = x bin


# ---------------------------------------------------------------------------
# 8. OME-Zarr inputs and pyramid dtype
# ---------------------------------------------------------------------------

def test_to_ome_zarr_takes_an_imageview_over_an_ndarray(tmp_path):
    img = _tiny_image()
    out = str(tmp_path / "iv.ome.zarr")
    r = g.to_ome_zarr(g.ImageView(img), out, levels=3)      # "truth value of an array" before
    assert r["kind"] == "image"
    m = g.read_ome_zarr_meta(out)
    assert m["shapes"] == [(64, 80), (32, 40), (16, 20)]
    import zarr
    assert np.array_equal(zarr.open_group(out, mode="r")["0"][:], img)


def test_rgb_image_is_exported_with_a_channel_axis(tmp_path):
    import zarr
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 255, (64, 80, 3)).astype(np.uint8)
    for src, name in ((rgb, "a"), (g.ImageView(rgb), "b")):
        out = str(tmp_path / f"{name}.ome.zarr")
        r = g.to_ome_zarr(src, out, levels=3)               # broadcast error before
        m = g.read_ome_zarr_meta(out)
        axes = m["multiscales"]["axes"]
        assert [a["name"] for a in axes] == ["c", "y", "x"]
        assert [a["type"] for a in axes] == ["channel", "space", "space"]
        assert m["shapes"] == [(3, 64, 80), (3, 32, 40), (3, 16, 20)]
        grp = zarr.open_group(out, mode="r")
        assert np.array_equal(grp["0"][:], rgb.transpose(2, 0, 1))
        assert all(grp[str(i)].dtype == np.uint8 for i in range(3))
        scales = [d["coordinateTransformations"][0]["scale"] for d in m["multiscales"]["datasets"]]
        assert scales == [[1, 1, 1], [1, 2, 2], [1, 4, 4]]   # channels are never downsampled
        assert r["kind"] == "image"


def test_intensity_pyramid_keeps_dtype_and_rounds_the_mean(tmp_path):
    import zarr
    rng = np.random.default_rng(4)
    for dt in (np.uint16, np.int16, np.uint8):
        info = np.iinfo(dt)
        img = rng.integers(info.min, info.max, (64, 80), dtype=dt)
        out = str(tmp_path / f"i_{np.dtype(dt).name}.ome.zarr")
        g.to_ome_zarr(img, out, levels=3)
        grp = zarr.open_group(out, mode="r")
        assert [grp[str(i)].dtype for i in range(3)] == [np.dtype(dt)] * 3   # float32 before
        want = np.rint(img.astype(np.float64).reshape(32, 2, 40, 2).mean((1, 3))).astype(dt)
        assert np.array_equal(grp["1"][:], want)
    f = rng.random((64, 80)).astype(np.float32)
    g.to_ome_zarr(f, str(tmp_path / "f.ome.zarr"), levels=2)
    assert zarr.open_group(str(tmp_path / "f.ome.zarr"), mode="r")["1"].dtype == np.float32


def test_odd_sized_pyramid_levels_still_keep_dtype(tmp_path):
    import zarr
    img = np.full((63, 81), 7, np.uint16)
    out = str(tmp_path / "odd.ome.zarr")
    g.to_ome_zarr(img, out, levels=3)
    grp = zarr.open_group(out, mode="r")
    assert grp["1"].shape == (32, 41) and grp["1"].dtype == np.uint16
    assert int(grp["1"][-1, -1]) == 7                        # edge cells are not diluted by padding


def test_label_pyramid_is_nearest_and_never_invents_labels(tmp_path):
    import zarr
    rng = np.random.default_rng(5)
    lab = rng.choice(np.array([0, 3, 7], np.uint8), size=(24, 32, 40))
    for src, kw, name in ((lab, {"labels": True}, "kw"),):
        out = str(tmp_path / f"{name}.ome.zarr")
        g.to_ome_zarr(src, out, levels=2, **kw)
        grp = zarr.open_group(out, mode="r")
        l1 = grp["1"][:]
        assert l1.dtype == np.uint8 and set(np.unique(l1)) <= {0, 3, 7}
        assert np.array_equal(l1, lab[::2, ::2, ::2])
        assert g.read_ome_zarr_meta(out)["multiscales"]["type"] == "nearest"
    # a mask volume is recognised without being told
    p = _nifti(tmp_path / "P01_T1_MASK.nii.gz", lab.transpose(2, 1, 0), dtype=np.uint8)
    out = str(tmp_path / "mask.ome.zarr")
    g.to_ome_zarr(g.NiftiVolume(p), out, levels=2)
    l1 = zarr.open_group(out, mode="r")["1"][:]
    assert set(np.unique(l1)) <= {0, 3, 7} and l1.dtype == np.uint8
    # ... and so is the sink that wrote it
    sink = g.MaskSink3D(str(tmp_path / "pred.zarr"), shape=lab.shape)
    sink.write(lab, (0, 24, 0, 32, 0, 40))
    out = str(tmp_path / "sink.ome.zarr")
    g.to_ome_zarr(sink, out, levels=2)
    assert np.array_equal(zarr.open_group(out, mode="r")["1"][:], lab[::2, ::2, ::2])
    # an intensity override averages, and says so
    g.to_ome_zarr(g.NiftiVolume(p), str(tmp_path / "forced.ome.zarr"), levels=2, labels=False)
    assert g.read_ome_zarr_meta(str(tmp_path / "forced.ome.zarr"))["multiscales"]["type"] == "mean"


def test_volume_export_voxel_scale_is_z_y_x_for_nifti_and_dicom(tmp_path):
    import zarr
    out = str(tmp_path / "sag.ome.zarr")
    g.to_ome_zarr(g.NiftiVolume(_sagittal_volume(tmp_path)), out, levels=2)
    m = g.read_ome_zarr_meta(out)
    assert m["multiscales"]["datasets"][0]["coordinateTransformations"][0]["scale"] == [5.0, 2.0, 1.0]
    d = g.DicomSeries(_dicom_series(tmp_path / "dd", np.zeros((16, 24, 32), np.uint16), step=3.6,
                                    spacing=(0.9, 0.8)))
    out2 = str(tmp_path / "dcm.ome.zarr")
    g.to_ome_zarr(d, out2, levels=2)
    sc = g.read_ome_zarr_meta(out2)["multiscales"]["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert sc == pytest.approx([3.6, 0.9, 0.8])
    assert zarr.open_group(out2, mode="r")["0"].shape == (16, 24, 32)


# ---------------------------------------------------------------------------
# builders for the stage / explorer sections
# ---------------------------------------------------------------------------

def _mini_gef(path, xs, ys, cs=None, side=1000, bins=(1, 100, 200)):
    """A valid little GEF: one gene, a real bin pyramid, bin-origin coordinates."""
    import h5py
    xs, ys = np.asarray(xs, np.int64), np.asarray(ys, np.int64)
    cs = np.ones(xs.size, np.int64) if cs is None else np.asarray(cs, np.int64)
    with h5py.File(path, "w") as f:
        grp = f.create_group("geneExp")
        f.attrs["StereoChip"] = np.bytes_("MINI")
        for b in bins:
            key = (xs // b) * b * (side + 1) + (ys // b) * b
            uniq, inv = np.unique(key, return_inverse=True)
            agg = np.zeros(uniq.size, np.int64)
            np.add.at(agg, inv.ravel(), cs)
            rec = np.zeros(uniq.size, [("x", "<u4"), ("y", "<u4"), ("count", "<u2")])
            rec["x"], rec["y"], rec["count"] = uniq // (side + 1), uniq % (side + 1), agg
            gp = grp.create_group(f"bin{b}")
            d = gp.create_dataset("expression", data=rec)
            d.attrs["minX"] = d.attrs["minY"] = np.uint32(0)
            d.attrs["maxX"] = d.attrs["maxY"] = np.uint32(side - 1)         # inclusive, as SAW
            gdt = np.dtype([("gene", "S32"), ("offset", "<u4"), ("count", "<u4")])
            gr = np.zeros(1, gdt)
            gr["gene"], gr["offset"], gr["count"] = b"GeneA", 0, uniq.size
            gp.create_dataset("gene", data=gr)
    return str(path)


def _bg_mask(img, bg=(0.996, 0.988, 0.953)):
    bg8 = np.round(np.array(bg) * 255).astype(int)
    return np.abs(img[:, :, :3].astype(int) - bg8).sum(2) > 30


# ---------------------------------------------------------------------------
# 5. stage: ticks sit where their value maps; a bin covers its footprint
# ---------------------------------------------------------------------------

def _label_ink(st, lb):
    rects = [r for r, _uv in st.atlas.layout(lb.text, lb.x, lb.y, lb.scale)]
    return (min(r[0] for r in rects), max(r[0] + r[2] for r in rects),
            min(r[1] for r in rects), max(r[1] + r[3] for r in rects))


@needs_gpu
@pytest.mark.parametrize("cam", [
    ("fit", (0, 4000, 0, 4000), 480, 360),
    ("fit", (0, 4000, 0, 4000), 300, 500),
    ("cam", (1234.5, 3224.0, 60.0), 480, 360),
    ("cam", (-50.0, -70.0, 9000.0), 640, 400),
])
def test_tick_labels_are_drawn_where_their_value_maps(cam):
    kind, args, w, h = cam
    st = g.Stage(w, h)
    st.camera = (g.Camera.fit(args, w, h) if kind == "fit"
                 else g.Camera(args[0], args[1], args[2], w, h))
    st.axes()
    nx = ny = 0
    for lb in st._labels:
        val = float(lb.text.replace(",", ""))
        x0, x1, y0, y1 = _label_ink(st, lb)
        assert 0 <= x0 and x1 <= w and 0 <= y0 and y1 <= h          # fully on screen
        if lb.x != 8:                                                # y labels start at x = 8
            nx += 1
            assert (x0 + x1) / 2 == pytest.approx(st.world_to_px(val, 0)[0], abs=2.0), lb.text
        else:                                                        # left strip: a y tick
            ny += 1
            assert x0 < 70
            assert (y0 + y1) / 2 == pytest.approx(st.world_to_px(0, val)[1], abs=2.5), lb.text
    assert nx >= 3 and ny >= 3


@needs_gpu
def test_tick_values_are_round_and_increase_with_the_axis():
    st = g.Stage(480, 360)
    st.camera = g.Camera.fit((0, 4000, 0, 4000), 480, 360)
    st.axes()
    vals = [(float(lb.text.replace(",", "")), lb) for lb in st._labels]
    xv = sorted(v for v, lb in vals if lb.x != 8)
    steps = {round(b - a, 6) for a, b in zip(xv, xv[1:])}
    assert len(steps) == 1                                            # evenly spaced
    step = steps.pop()
    assert step in {1, 2, 5, 10, 20, 50, 100, 200, 250, 500, 1000, 2000} or step % 10 == 0


@needs_gpu
@pytest.mark.parametrize("W,H", [(200, 200), (320, 180), (180, 320)])    # square, wide, tall
def test_a_binned_cell_covers_exactly_its_footprint(W, H):
    st = g.Stage(W, H)
    st.camera = g.Camera.fit((0, 1000, 0, 1000), W, H, margin=1.0)
    # bin 200 whose origin is (200, 200): it owns [200, 400) x [200, 400)
    st.points(np.array([200]), np.array([200]), np.array([5.0]), size=200, cells=True)
    cov = _bg_mask(st.render())
    ys, xs = np.nonzero(cov)
    px0, py_bottom = st.world_to_px(200, 200)
    px1, py_top = st.world_to_px(400, 400)
    assert xs.min() == pytest.approx(px0, abs=1) and xs.max() + 1 == pytest.approx(px1, abs=1)
    assert ys.min() == pytest.approx(py_top, abs=1) and ys.max() + 1 == pytest.approx(py_bottom, abs=1)
    assert cov[int(py_top) + 1:int(py_bottom) - 1, int(px0) + 1:int(px1) - 1].all()   # solid


@needs_gpu
def test_scatter_points_stay_centred_on_their_position():
    W = H = 200
    st = g.Stage(W, H)
    st.camera = g.Camera.fit((0, 1000, 0, 1000), W, H, margin=1.0)
    st.points(np.array([500.0]), np.array([500.0]), np.array([5.0]), size=200)
    ys, xs = np.nonzero(_bg_mask(st.render()))
    cx, cy = st.world_to_px(500, 500)
    assert (xs.min() + xs.max() + 1) / 2 == pytest.approx(cx, abs=1)
    assert (ys.min() + ys.max() + 1) / 2 == pytest.approx(cy, abs=1)


@needs_gpu
def test_show_draws_a_coarse_bin_on_its_own_footprint(tmp_path):
    rng = np.random.default_rng(8)
    cells = rng.choice(100 * 100, size=5000, replace=False)           # 5000 distinct bin-1 spots
    xs, ys = 200 + cells % 100, 600 + cells // 100                    # all inside [200,300) x [600,700)
    path = _mini_gef(tmp_path / "one.gef", xs, ys)
    v = g.open_gef(path)
    st = g.Stage(300, 300, point_budget=16_000)
    ch = st.show(v, label=False)
    assert ch["bin_size"] == 100 and ch["points"] == 1                # a single occupied bin
    ys_, xs_ = np.nonzero(_bg_mask(st.render()))
    px0, py_bottom = st.world_to_px(200, 600)
    px1, py_top = st.world_to_px(300, 700)
    assert xs_.min() == pytest.approx(px0, abs=1) and xs_.max() + 1 == pytest.approx(px1, abs=1)
    assert ys_.min() == pytest.approx(py_top, abs=1) and ys_.max() + 1 == pytest.approx(py_bottom, abs=1)
    v.gef.close()


# ---------------------------------------------------------------------------
# 6. nothing to draw is not an error
# ---------------------------------------------------------------------------

@needs_gpu
def test_stage_points_with_no_points_renders_the_background():
    st = g.Stage(120, 90)
    st.points(np.array([]), np.array([]))                             # TypeError before
    img = st.render()
    assert img.shape == (90, 120, 4) and not _bg_mask(img).any()
    assert st.stats()["points"] == 0 and st.draw_calls == 0
    st.points(np.array([1.0, 2.0]), np.array([1.0, 2.0]))             # and it recovers
    assert st.stats()["points"] == 2
    st.points([], [], [])                                              # lists work too
    st.render()


@needs_gpu
def test_explorer_survives_an_empty_window(tmp_path):
    rng = np.random.default_rng(9)
    cells = rng.choice(300 * 300, size=4000, replace=False)
    path = _mini_gef(tmp_path / "corner.gef", cells % 300, cells // 300)   # data only in the corner
    ex = g.explore(path, width=200, height=160, point_budget=16_000)   # cap: 1000 points
    ex.frame()                                                        # 4000 at bin 1: coarser rung
    assert ex.loaded_points > 0 and ex.loaded_rung > 1
    before = ex.telemetry.reloads
    ex.goto((700, 900, 700, 900))                                     # nothing lives here, and the
    img = ex.frame()                                                  # estimate says bin 1 fits:
    assert ex.telemetry.reloads == before + 1                         # a real (empty) load.
    assert img.shape == (160, 200, 4) and ex.loaded_points == 0       # TypeError before
    ex.handle(g.Event("drag", dx=30, dy=10))                          # panning in the void is fine
    ex.frame()
    ex.goto((0, 300, 0, 300))
    ex.frame()
    assert ex.loaded_points > 0
    ex.view.gef.close()


@needs_gpu
def test_show_hud_and_navigator_over_an_empty_window(tmp_path):
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(10)
    cells = rng.choice(300 * 300, size=4000, replace=False)
    v = g.open_gef(_mini_gef(tmp_path / "e.gef", cells % 300, cells // 300))
    st = g.Stage(300, 240, point_budget=16_000)
    ch = st.show(v, window=(700, 900, 700, 900))
    assert ch["points"] == 0
    st.hud(v)
    assert st.render().shape == (240, 300, 4)
    fig = g.render.navigator(v.select(700, 900, 700, 900), detail=True)
    import matplotlib.pyplot as plt
    plt.close(fig)
    v.gef.close()


@needs_gpu
def test_explorer_record_over_an_empty_region(tmp_path):
    path = _mini_gef(tmp_path / "far.gef", np.arange(50), np.arange(50))
    ex = g.explore(path, width=160, height=120)
    ex.goto((600, 900, 600, 900))                                    # the very first load is empty
    out = ex.record([{"kind": "scroll", "x": 150, "y": 20, "steps": 6},
                     {"kind": "drag", "dx": -40, "dy": 25}])
    assert ex.telemetry.reloads >= 1 and ex.loaded_points == 0
    assert len(out) == 3 and all(f.shape == (120, 160, 4) for _, f in out)
    ex.view.gef.close()


# ---------------------------------------------------------------------------
# 7. stage / explorer / render / spatial leftovers
# ---------------------------------------------------------------------------

def test_aggregate_handles_negative_integer_coordinates():
    x = np.array([-1, -1, 5, 0, 0, -7], dtype=np.int32)
    y = np.array([-3, -3, -3, -1, 0, 4], dtype=np.int32)
    v = np.array([1, 2, 3, 4, 5, 6])
    ax_, ay, av = g.Stage.aggregate(x, y, v)
    got = dict(zip(zip(ax_.tolist(), ay.tolist()), av.tolist()))
    assert got == {(-1, -3): 3.0, (5, -3): 3.0, (0, -1): 4.0, (0, 0): 5.0, (-7, 4): 6.0}
    big = np.array([2 ** 40, 2 ** 40, -2 ** 40], dtype=np.int64)       # beyond a 32-bit key
    bx, by, bv = g.Stage.aggregate(big, np.array([-5, -5, 7], dtype=np.int64), np.array([1, 1, 1]))
    assert sorted(zip(bx.tolist(), by.tolist(), bv.tolist())) == [(-2 ** 40, 7, 1.0), (2 ** 40, -5, 2.0)]
    ux, uy, uv = g.Stage.aggregate(np.array([3, 3]), np.array([-2, -2]))   # no values: counts
    assert (ux.tolist(), uy.tolist(), uv.tolist()) == ([3], [-2], [2.0])


@needs_gpu
def test_explore_accepts_a_pathlib_path(chip_gef):
    import pathlib
    ex = g.explore(pathlib.Path(chip_gef), width=160, height=120)
    assert ex.frame().shape == (120, 160, 4)
    ex.view.gef.close()


def test_navigator_draws_a_gef_window_the_right_way_round(chip_gef):
    pytest.importorskip("matplotlib")
    v = g.open_gef(chip_gef, budget=g.Budget(ram="256MB")).select(500, 1500, 2000, 2400)   # wide, short
    fig = g.render.navigator(v, detail=True)                          # TypeError before
    ov = fig.axes[0]
    (rect,) = [p for p in ov.patches]
    assert rect.get_xy() == (500, 2000) and rect.get_width() == 1000 and rect.get_height() == 400
    left, right, bottom, top = ov.images[0].get_extent()
    assert (left, right, bottom, top) == (0, 4000, 0, 4000)
    detail = fig.axes[-1]
    assert detail.collections                                         # the window's points were drawn
    import matplotlib.pyplot as plt
    plt.close(fig)
    v.gef.close()


def test_navigator_still_draws_an_image_window_in_row_column_order():
    pytest.importorskip("matplotlib")
    img = np.arange(60 * 100, dtype=np.uint16).reshape(60, 100)
    v = g.ImageView(img).select(10, 40, 20, 90)                       # rows 10-40, columns 20-90
    fig = g.render.navigator(v, detail=False)
    (rect,) = fig.axes[0].patches
    assert rect.get_xy() == (20, 10) and rect.get_width() == 70 and rect.get_height() == 30
    import matplotlib.pyplot as plt
    plt.close(fig)


def _mini_xenium(root, n=3000, seed=0):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = np.random.default_rng(seed)
    root = str(root)
    os.makedirs(root, exist_ok=True)
    x = np.sort(rng.uniform(0, 1000, n))
    t = pa.table({"feature_name": rng.choice(np.array(["A", "B", "C"]), n),
                  "x_location": x, "y_location": rng.uniform(0, 1000, n),
                  "cell_id": np.arange(n), "qv": rng.uniform(10, 40, n)})
    pq.write_table(t, os.path.join(root, "transcripts.parquet"), row_group_size=250)
    with open(os.path.join(root, "experiment.xenium"), "w") as fh:
        json.dump({"panel_name": "mini", "num_cells": n}, fh)
    return root, t


def test_xenium_window_adds_the_coordinate_columns_it_filters_on(tmp_path):
    root, t = _mini_xenium(tmp_path / "xen")
    xb = g.open_xenium(root)
    x, y = t.column("x_location").to_numpy(), t.column("y_location").to_numpy()
    want = int(((x >= 100) & (x < 300) & (y >= 200) & (y < 600)).sum())
    assert want > 0
    full = xb.window(100, 300, 200, 600)
    assert full.num_rows == want
    for cols in (["feature_name"], ["cell_id", "qv"], ["x_location"], ("feature_name", "y_location")):
        got = xb.window(100, 300, 200, 600, columns=cols)        # KeyError before
        assert got.num_rows == want and got.column_names == list(cols)   # exactly what was asked
    none = xb.window(2000, 3000, 0, 10, columns=["feature_name"])
    assert none.num_rows == 0 and none.column_names == ["feature_name"]


def test_bundle_lookup_survives_glob_metacharacters_in_the_path(tmp_path):
    root, _ = _mini_xenium(tmp_path / "slide[1]" / "xen")
    xb = g.open_xenium(root)
    assert xb.transcripts and xb.experiment                    # was: not found, "neither ... nor"
    assert xb.window(0, 100, 0, 1000, columns=["feature_name"]).num_rows > 0
    assert g.sniff_bundle(root)["kind"] == "xenium"


@need_visium
def test_visium_bundle_in_a_folder_with_brackets(tmp_path):
    dst = tmp_path / "slide[1]" / "sample?*"
    shutil.copytree(VISIUM, dst)
    vb = g.open_visium(str(dst))
    assert vb.describe()["n_spots"] == 1500 and vb.hires
    vb.close()


def _cyx(tmp_path, name, data, **kw):
    import tifffile
    p = str(tmp_path / name)
    tifffile.imwrite(p, data, ome=True,
                     metadata={"axes": "CYX", "Channel": {"Name": [f"m{i}" for i in range(data.shape[0])]}},
                     **kw)
    return p


def _traced_peak(fn):
    import tracemalloc
    tracemalloc.start()
    try:
        out = fn()
        return out, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_multichannel_window_decodes_tiles_of_one_channel_only(tmp_path):
    rng = np.random.default_rng(10)
    data = rng.integers(0, 60000, (6, 1536, 1536), dtype=np.uint16)          # a 28 MB stack
    mc = g.MultiChannelTiff(_cyx(tmp_path, "tiled.ome.tif", data, tile=(256, 256)))
    assert mc.tiled
    win, peak = _traced_peak(lambda: mc.channel(2, 600, 700, 500, 600))
    assert np.array_equal(win, data[2, 600:700, 500:600])
    assert peak < 3 * 1024 ** 2, f"peaked at {peak / 1e6:.1f} MB for a 20 KB window"
    c = mc.cost(2, (600, 700, 500, 600))
    assert c["strategy"] == "tiles" and c["tiles_touched"] == 2
    assert c["decode_bytes"] == 2 * 256 * 256 * 2 and c["resident_bytes"] < 1024 ** 2
    assert np.array_equal(mc.channel("m5", 0, 10, 1500, 1536), data[5, 0:10, 1500:1536])
    assert np.array_equal(mc.channel(0), data[0])                           # the whole plane
    mc.close()


def test_multichannel_striped_file_reads_only_the_strips_it_needs(tmp_path):
    rng = np.random.default_rng(11)
    data = rng.integers(0, 60000, (4, 1024, 1024), dtype=np.uint16)
    mc = g.MultiChannelTiff(_cyx(tmp_path, "strips.ome.tif", data, rowsperstrip=32))
    assert not mc.tiled
    win, peak = _traced_peak(lambda: mc.channel(1, 100, 140, 10, 90))
    assert np.array_equal(win, data[1, 100:140, 10:90])
    assert peak < 1024 ** 2
    c = mc.cost(1, (100, 140, 10, 90))
    assert c["strategy"] == "strips" and c["decode_bytes"] == 2 * 32 * 1024 * 2   # strips 3 and 4
    mc.close()


def test_multichannel_full_plane_read_is_leased_before_it_happens(tmp_path):
    rng = np.random.default_rng(12)
    data = rng.integers(0, 60000, (4, 1024, 1024), dtype=np.uint16)
    p = _cyx(tmp_path, "oneshot.ome.tif", data, rowsperstrip=1024)          # one strip per plane
    plane = 1024 * 1024 * 2
    small = g.Budget(ram="1MB", timeout_s=1.0)
    mc = g.MultiChannelTiff(p, budget=small)
    c = mc.cost(1, (0, 10, 0, 10))
    assert c["strategy"] == "full-plane" and c["decode_bytes"] == plane
    assert "full plane" in c["note"] and c["fits"] is False
    with pytest.raises(MemoryError):
        mc.channel(1, 0, 10, 0, 10)                                         # denied up front
    assert small.stats()["ram"]["denials"] >= 1 and small.stats()["ram"]["used"] == 0
    mc.close()
    big = g.Budget(ram="16MB", timeout_s=1.0)
    mc = g.MultiChannelTiff(p, budget=big)
    win, peak = _traced_peak(lambda: mc.channel(3, 5, 25, 7, 27))
    assert np.array_equal(win, data[3, 5:25, 7:27])
    assert peak < 2 * plane + 1024 ** 2                                     # one plane, not four
    st = big.stats()["ram"]
    assert st["peak"] >= plane and st["used"] == 0                          # leased, then released
    assert mc.cost(3, (5, 25, 7, 27))["fits"] is True
    mc.close()


def test_multichannel_interleaved_samples_select_the_channel(tmp_path):
    import tifffile
    rng = np.random.default_rng(13)
    data = rng.integers(0, 60000, (300, 400, 3), dtype=np.uint16)
    p = str(tmp_path / "yxs.tif")
    tifffile.imwrite(p, data, photometric="rgb", tile=(64, 64))
    mc = g.MultiChannelTiff(p)
    assert mc.n_channels == 3 and mc.axes == "YXS"
    assert np.array_equal(mc.channel(1, 70, 140, 30, 130), data[70:140, 30:130, 1])
    mc.close()


def test_multichannel_describe_and_by_name_lookup_still_work():
    cod = os.path.join(FIX, "glioma", "codex_6plex.ome.tif")
    if not os.path.exists(cod):
        pytest.skip("no CODEX fixture")
    mc = g.MultiChannelTiff(cod)
    gf = mc.channel("GFAP", 600, 1000, 500, 900)
    assert gf.shape == (400, 400) and gf.dtype == np.uint16
    assert np.array_equal(mc.channel(3, 0, 10, 0, 10), mc.channel("GFAP", 0, 10, 0, 10))
    assert mc.describe()["tiled"] is True
    mc.close()


def test_multichannel_unusual_layout_is_a_leased_full_read(tmp_path, monkeypatch):
    rng = np.random.default_rng(14)
    data = rng.integers(0, 60000, (3, 256, 256), dtype=np.uint16)
    p = _cyx(tmp_path, "odd.ome.tif", data, tile=(64, 64))
    stack = data.nbytes
    small = g.Budget(ram="256KB", timeout_s=1.0)
    mc = g.MultiChannelTiff(p, budget=small)
    monkeypatch.setattr(mc, "_channel_pages", lambda c: None)      # pages do not map onto the series
    c = mc.cost(1, (0, 8, 0, 8))
    assert c["strategy"] == "full-stack" and c["decode_bytes"] == stack and "stack" in c["note"]
    with pytest.raises(MemoryError):
        mc.channel(1, 0, 8, 0, 8)                                  # the whole stack is leased first
    big = g.Budget(ram="8MB", timeout_s=1.0)
    mc2 = g.MultiChannelTiff(p, budget=big)
    monkeypatch.setattr(mc2, "_channel_pages", lambda c: None)
    assert np.array_equal(mc2.channel(1, 10, 20, 30, 40), data[1, 10:20, 30:40])
    assert big.stats()["ram"]["peak"] >= stack and big.stats()["ram"]["used"] == 0


def test_multichannel_extra_axes_stay_in_front(tmp_path):
    rng = np.random.default_rng(15)
    data = rng.integers(0, 255, (2, 3, 70, 90), dtype=np.uint8)               # Z, C, Y, X
    mc = g.MultiChannelTiff(_cyx_axes(tmp_path, "zcyx.ome.tif", data, "ZCYX", tile=(32, 32)))
    assert mc.axes == "ZCYX" and mc.n_channels == 3
    assert np.array_equal(mc.channel(2, 5, 40, 10, 80), data[:, 2, 5:40, 10:80])   # (Z, h, w)
    assert mc.cost(2, (5, 40, 10, 80))["planes"] == 2
    mc.close()


def _cyx_axes(tmp_path, name, data, axes, **kw):
    import tifffile
    p = str(tmp_path / name)
    tifffile.imwrite(p, data, ome=True, metadata={"axes": axes}, **kw)
    return p


def test_camera_world_to_px_round_trips_and_the_stage_agrees():
    cam = g.Camera.fit((0, 4000, 0, 3000), 480, 360)
    for wx, wy in ((0, 0), (4000, 3000), (1234.5, 2222.0), (-700, 5100)):
        px, py = cam.world_to_px(wx, wy)
        assert cam.px_to_world(px, py) == pytest.approx((wx, wy))
    px, py = cam.world_to_px(2000, 1500)
    assert (px, py) == pytest.approx((240, 180))                   # the camera centre is the middle
    if g.stage_available():
        st = g.Stage(480, 360)
        st.camera = cam
        assert st.world_to_px(1234.5, 2222.0) == pytest.approx(cam.world_to_px(1234.5, 2222.0))


def test_dicom_affine_places_voxels_in_ras(tmp_path):
    import pydicom
    root = _dicom_series(tmp_path / "aff", np.zeros((4, 5, 6), np.uint16), step=3.6,
                         spacing=(0.9, 0.8))
    for f in os.listdir(root):                                      # move the origin off zero
        ds = pydicom.dcmread(os.path.join(root, f))
        z = float(ds.ImagePositionPatient[2])
        ds.ImagePositionPatient = [10.0, 20.0, 30.0 + z]
        ds.save_as(os.path.join(root, f), enforce_file_format=True)
    d = g.DicomSeries(root)
    want = np.array([[-0.8, 0, 0, -10], [0, -0.9, 0, -20], [0, 0, 3.6, 30], [0, 0, 0, 1.0]])
    assert np.allclose(d.affine, want)                              # (column, row, slice) -> RAS
    import nibabel as nib
    # nibabel reads the same affine as (column, row, slice) = (L, P, S): the array-order
    # axcodes reversed. Two routes to the same answer, for an axial and for an oblique series.
    assert nib.aff2axcodes(d.affine) == ("L", "P", "S")
    assert d.axcodes == tuple(reversed(nib.aff2axcodes(d.affine)))
    sag = g.DicomSeries(_dicom_series(tmp_path / "sag2", np.zeros((4, 6, 5), np.uint16),
                                      iop=(0, 1, 0, 0, 0, -1), step=2.0))
    assert sag.axcodes == tuple(reversed(nib.aff2axcodes(sag.affine))) == ("R", "I", "P")
    sink = g.MaskSink3D.like(d, str(tmp_path / "m.zarr"))
    assert sink.z.shape == d.shape and np.allclose(sink.affine, d.affine)


def test_cohort_accepts_custom_regex_and_function_layouts(tmp_path):
    import re
    _touch(str(tmp_path), "alice/scan_a/T1w.nii.gz", "alice/scan_b/T1w.nii.gz", "bob/scan_a/FLAIR.nii.gz")
    rx = g.Cohort(str(tmp_path), layouts=[re.compile(r"^(?P<sub>[a-z]+)/scan_(?P<ses>[a-z])/")])
    assert set(rx.subjects) == {"alice", "bob"} and rx["alice"].timepoints == ["a", "b"]
    fn = g.Cohort(str(tmp_path), layouts=[lambda rel: (rel.split("/")[0], rel.split("/")[1][-1])
                                          if rel.startswith("alice") else (None, None)])
    assert set(fn.subjects) == {"alice"} and fn.unassigned == ["bob/scan_a/FLAIR.nii.gz"]
