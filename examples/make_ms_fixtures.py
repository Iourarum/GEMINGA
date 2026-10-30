"""A miniature MS lesion cohort: longitudinal NIfTI (T1w/T2w/FLAIR + expert mask, 2 timepoints
for some subjects, one .nii.gz and one uncompressed), plus one DICOM series. Shaped like
MSLesSeg / ISBI-2015 layouts, small enough to ship."""
import os
import sys
import numpy as np
import nibabel as nib

out = sys.argv[1] if len(sys.argv) > 1 else "fixtures/ms"
Z, Y, X = 48, 96, 96
rng = np.random.default_rng(3)
zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]

def brain(tp: int):
    c = (Z/2, Y/2, X/2); r = (Z*0.42, Y*0.42, X*0.42)
    ell = (((zz-c[0])/r[0])**2 + ((yy-c[1])/r[1])**2 + ((xx-c[2])/r[2])**2) < 1
    wm = ell & ((((zz-c[0])/(r[0]*0.7))**2 + ((yy-c[1])/(r[1]*0.7))**2 + ((xx-c[2])/(r[2]*0.7))**2) < 1)
    return ell, wm

lesion_seeds = [(24, 40, 38, 3), (20, 55, 60, 2), (30, 48, 52, 4), (18, 60, 30, 2)]
def lesions(tp: int):
    m = np.zeros((Z, Y, X), np.uint8)
    for i, (cz, cy, cx, r) in enumerate(lesion_seeds[: 3 + tp]):     # a new lesion at tp2
        rr = r + 0.6 * tp                                             # and growth over time
        m |= ((((zz-cz))**2 + ((yy-cy))**2 + ((xx-cx))**2) < rr*rr).astype(np.uint8)
    return m

def volume(seq: str, tp: int):
    ell, wm = brain(tp)
    les = lesions(tp).astype(bool)
    base = {"T1w": (900, 600, 450), "T2w": (400, 700, 1100), "FLAIR": (300, 500, 1400)}[seq]
    v = np.full((Z, Y, X), 60.0)
    v[ell] = base[0]; v[wm] = base[1]; v[les] = base[2]
    v += rng.normal(0, 25, v.shape)
    return v.clip(0, 4095).astype(np.int16)

aff = np.diag([1.0, 1.0, 3.0, 1.0])
def save(arr_zyx, path):
    nib.save(nib.Nifti1Image(np.transpose(arr_zyx, (2, 1, 0)), aff), path)

for sub, tps in [("P01", [1, 2]), ("P02", [1, 2]), ("P03", [1])]:
    for tp in tps:
        d = os.path.join(out, "train", sub, f"T{tp}"); os.makedirs(d, exist_ok=True)
        for seq in ("T1w", "T2w", "FLAIR"):
            gz = not (sub == "P01" and tp == 1)                # one uncompressed, for seek tests
            save(volume(seq, tp - 1), os.path.join(d, f"{sub}_T{tp}_{seq}.nii" + (".gz" if gz else "")))
        save(lesions(tp - 1), os.path.join(d, f"{sub}_T{tp}_MASK.nii.gz"))
    print(f"  {sub}: {len(tps)} timepoint(s)")

# one DICOM series, single-slice files, out of order on purpose
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
dd = os.path.join(out, "dicom", "P04_FLAIR"); os.makedirs(dd, exist_ok=True)
vol = volume("FLAIR", 0)
suid = generate_uid()
for z in rng.permutation(Z):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientID = "P04"; ds.Modality = "MR"; ds.SeriesDescription = "AX FLAIR"
    ds.SeriesInstanceUID = suid; ds.StudyDate = "20260101"; ds.Manufacturer = "Synthetic"
    ds.MagneticFieldStrength = 1.5
    ds.Rows, ds.Columns = Y, X; ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15
    ds.PixelRepresentation = 1; ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelSpacing = [1.0, 1.0]; ds.SliceThickness = 3.0
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0.0, 0.0, float(z) * 3.0]
    ds.InstanceNumber = int(z) + 1
    ds.PixelData = np.ascontiguousarray(vol[z]).tobytes()
    ds.save_as(os.path.join(dd, f"slice_{int(rng.integers(1000,9999))}_{z:03d}.dcm"),
               write_like_original=False)
print(f"  P04: DICOM series, {Z} files")
tot = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs)
print(f"fixtures in {out}: {tot:,} bytes")
