"""MRI volumes: lazy NIfTI/DICOM, 3-D navigation, cohort grouping, mask writing — and the
budget accounting that every View now does."""
import os
import numpy as np
import pytest
import geminga as g

FIX = os.environ.get("GEMINGA_FIXTURES", "fixtures")
MS = os.path.join(FIX, "ms")
NII = os.path.join(MS, "train", "P01", "T1", "P01_T1_FLAIR.nii")
NIIGZ = os.path.join(MS, "train", "P01", "T2", "P01_T2_FLAIR.nii.gz")
MASK = os.path.join(MS, "train", "P01", "T2", "P01_T2_MASK.nii.gz")
DCM = os.path.join(MS, "dicom", "P04_FLAIR")
need = pytest.mark.skipif(not os.path.exists(NII), reason="run examples/make_ms_fixtures.py first")


@pytest.mark.parametrize("name,seq", [
    ("P01_T1_T2w.nii.gz", "T2w"), ("P01_T2_T1w.nii.gz", "T1w"),
    ("sub-01_ses-02_FLAIR.nii", "FLAIR"), ("FLAIR_lesion_mask.nii.gz", "MASK"),
    ("AX FLAIR", "FLAIR"), ("t1_mprage_sag", "T1w"), ("training01_02_pd.nii", "PDw"),
    ("swi_mag.nii.gz", "SWI"), ("nothing_here.nii", "unknown"),
    ("MSLesSeg_P1_T1_FLAIR.nii.gz", "FLAIR"), ("P1_T1_FLAIR_seg.nii.gz", "MASK"),
])
def test_sequence_guess_prefers_the_last_token(name, seq):
    assert g.guess_sequence(name) == seq


@need
def test_nifti_header_now_voxels_later():
    v = g.NiftiVolume(NII)
    assert v.shape == (48, 96, 96) and v.dtype == np.int16
    # voxel_mm follows the array axes (z, y, x), like shape: 3 mm slices of 1 x 1 mm pixels
    assert v.voxel_mm == (3.0, 1.0, 1.0) and v.sequence == "FLAIR" and not v.gzipped
    assert v.axcodes == ("S", "A", "R")                 # the fixture's affine is RAS+, 3 mm in z
    s = v.slab(10, 14, 20, 60, 30, 70)
    assert s.shape == (4, 40, 40) and s.dtype == np.int16
    full = v.slab(0, 48)
    assert np.array_equal(s, full[10:14, 20:60, 30:70])
    v.close()


@need
def test_gzip_cost_is_honest_about_linear_reads():
    plain = g.view(NII, budget=g.Budget(ram="64MB")).select(30, 40, 0, 96, 0, 96).cost()
    gz = g.view(NIIGZ, budget=g.Budget(ram="64MB")).select(30, 40, 0, 96, 0, 96).cost()
    assert plain["total_bytes"] == gz["total_bytes"]          # same data kept
    assert gz["read_bytes"] > plain["read_bytes"]              # more read to get it
    assert "gzip" in gz["seek"] and plain["seek"] == "random"


@need
def test_3d_zoom_holds_cost_flat_and_select_frac_broadcasts():
    v = g.view(NII, budget=g.Budget(ram="64MB")).reset().set_zoom(2)
    first = v.cost()["total_bytes"]
    for _ in range(2):
        v.zoom_in(2)
        assert v.cost()["total_bytes"] == pytest.approx(first, rel=0.15)
    v.select_frac(0.25, 0.75)
    z0, z1, y0, y1, x0, x1 = v.window
    assert (z0, z1) == (12, 36) and (y0, y1) == (24, 72) and (x0, x1) == (24, 72)


@need
def test_dicom_series_orders_by_position_and_reads_only_asked_slices():
    d = g.DicomSeries(DCM)
    assert d.shape == (48, 96, 96) and d.modality == "MR" and d.sequence == "FLAIR"
    pos = [s[0] for s in d.slices]
    assert pos == sorted(pos)
    s = d.slab(20, 23, 10, 50, 10, 50)
    assert s.shape == (3, 40, 40)
    ref = g.NiftiVolume(NII).slab(20, 23, 10, 50, 10, 50)
    # same anatomy, independent noise draw: structure must agree, bytes need not
    r = np.corrcoef(s.ravel().astype(float), ref.ravel().astype(float))[0, 1]
    assert r > 0.95


@need
def test_every_view_accounts_its_load_against_the_budget():
    b = g.Budget(ram="64MB")
    v = g.view(NIIGZ, budget=b).select(10, 30, 0, 96, 0, 96)
    a = v.load()
    st = b.stats()["ram"]
    assert st["used"] == a.nbytes and st["peak"] == a.nbytes
    v.release()
    assert b.stats()["ram"]["used"] == 0
    with v:                                                     # context manager releases
        v.load()
        assert b.stats()["ram"]["used"] > 0
    assert b.stats()["ram"]["used"] == 0


@need
def test_chunks_hold_one_chunk_at_a_time():
    b = g.Budget(ram="2MB")
    v = g.view(NII, budget=b).reset()
    seen = 0
    for slab, _win in v.chunks():
        assert b.stats()["ram"]["used"] == slab.nbytes
        seen += slab.shape[0]
    assert seen == 48 and b.stats()["ram"]["used"] == 0


@need
def test_cohort_groups_subjects_sessions_sequences():
    co = g.Cohort(os.path.join(MS, "train"))
    assert len(co) == 3 and not co.unassigned
    assert {s.id for s in co.longitudinal()} == {"P01", "P02"}
    assert {s.id for s in co.complete()} == {"P01", "P02", "P03"}
    assert all("MASK" in s.sequences() for s in co)
    assert co["P01"].sessions["1"]["T2w"].endswith("P01_T1_T2w.nii")     # the uncompressed one
    assert co["P01"].sessions["2"]["T2w"].endswith("P01_T2_T2w.nii.gz")


@need
def test_longitudinal_window_shows_lesion_growth():
    co = g.Cohort(os.path.join(MS, "train"))
    views = co.paired_windows("P01", "MASK", budget=g.Budget(ram="64MB"))
    counts = {ses: int((vv.select(14, 34, 30, 70, 20, 70).load() > 0).sum())
              for ses, vv in views.items()}
    assert counts["2"] > counts["1"]                            # a new lesion appeared


@need
def test_mask_sink_3d_roundtrip(tmp_path):
    v = g.view(NIIGZ, budget=g.Budget(ram="8MB")).reset()
    sink = g.MaskSink3D(str(tmp_path / "pred.zarr"), shape=v.vol.shape, affine=v.vol.affine)
    for slab, win in v.chunks():
        sink.write((slab > 1100).astype("uint8"), win)
    truth = g.NiftiVolume(MASK).slab(0, 48) > 0
    pred = np.asarray(sink.z[:]) > 0
    dice = 2 * (pred & truth).sum() / (pred.sum() + truth.sum())
    assert dice > 0.95
    out = sink.to_nifti(str(tmp_path / "pred.nii.gz"))
    assert np.array_equal(g.NiftiVolume(out).slab(0, 48) > 0, pred)


@need
def test_profile_recognises_mri_and_masks():
    p = g.profile(NII, fetch_page=False)
    assert p.format == "NIfTI" and p.best.label.startswith("MRI") and p.confident
    assert p.metadata["sequence"] == "FLAIR"
    m = g.profile(MASK, fetch_page=False)
    assert m.metadata.get("is_label_volume") is True
    d = g.profile(DCM, fetch_page=False)
    assert d.format == "DICOM series" and d.metadata["modality"] == "MR"


def test_neuro_hosts_classify_and_route():
    assert g.classify("doi:10.6084/m9.figshare.27919209") == "figshare_doi"
    assert g.classify("10.5281/zenodo.7051658") == "zenodo_doi"
    assert g.classify("ds004199") == "openneuro"
    assert g.classify("https://www.kaggle.com/datasets/a/b") == "kaggle"
    assert g.classify("https://www.nitrc.org/projects/x") == "nitrc"
    k = g.acquisition_plan("https://www.kaggle.com/datasets/a/b", offline=True).routes[0]
    assert k.needs_auth and k.tool == "kaggle"
    o = g.acquisition_plan("ds004199", offline=True).routes
    assert any(r.kind == "aws-odp" and "--no-sign-request" in " ".join(g.command_for(r))
               for r in o)
