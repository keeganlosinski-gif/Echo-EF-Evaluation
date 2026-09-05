"""Simpson's biplane (method of disks) EF from CAMUS ground-truth masks.

PARITY GATE. Before training on either the masks or the EF labels, check that they agree.
The dataset ships BOTH a per-patient EF and expert LV contours. EF is computable from the
contours by the same method the ultrasound cart uses, so the two are redundant -- and any
disagreement means one of them is not what we think it is.

Method (ASE/EACVI biplane Simpson):
  1. LV cavity = label 1. Mitral annulus = the basal edge of the cavity (borders LA, label 3).
  2. Long axis = apex -> midpoint of the annulus.
  3. Rotate so the long axis is vertical, split it into N=20 disks.
  4. Each disk is an ellipse with diameters a (from A2C) and b (from A4C):
        V = (pi/4) * h * SUM(a_i * b_i),  h = L/N
  5. EF = (EDV - ESV) / EDV * 100

NOTE ON SCALE: EF is a ratio, so an isotropic scaling of both volumes cancels. Absolute
volumes in mL do NOT cancel, so real pixel spacing is read from the NIfTI header per view.
"""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import os, numpy as np, nibabel as nib
from scipy.ndimage import rotate as ndrotate

LV, MYO, LA = 1, 2, 3
NDISK = 20


def _annulus_and_apex(m):
    """Basal (mitral) edge midpoint and the apex of the LV cavity mask."""
    ys, xs = np.nonzero(m)
    if len(ys) < 50: return None
    # apical acquisition: apex is near-field (small y), annulus is far-field (large y)
    ybase = ys.max()
    band = ys >= ybase - 2                       # the basal 3 rows of cavity
    xb = xs[band]
    if len(xb) < 2: return None
    p1 = np.array([xb.min(), ybase], float)
    p2 = np.array([xb.max(), ybase], float)
    mid = (p1 + p2) / 2.0
    d = np.hypot(xs - mid[0], ys - mid[1])       # apex = farthest cavity pixel from annulus mid
    k = int(np.argmax(d))
    apex = np.array([xs[k], ys[k]], float)
    return mid, apex


def disk_diameters(m, sx, sy, ndisk=NDISK):
    """Return (disk diameters in mm, apex->base long-axis length in mm)."""
    r = _annulus_and_apex(m)
    if r is None: return None, None
    mid, apex = r
    v = mid - apex
    ang = np.degrees(np.arctan2(v[0], v[1]))     # angle to rotate long axis to vertical
    rm = ndrotate(m.astype(float), -ang, reshape=True, order=0) > 0.5
    r2 = _annulus_and_apex(rm)
    if r2 is None: return None, None
    mid2, apex2 = r2
    y0, y1 = apex2[1], mid2[1]
    if y1 <= y0 + ndisk: return None, None
    L = y1 - y0
    edges = np.linspace(y0, y1, ndisk + 1)
    dia = []
    for i in range(ndisk):
        lo, hi = edges[i], edges[i + 1]
        band = rm[int(round(lo)):max(int(round(hi)), int(round(lo)) + 1), :]
        if band.size == 0 or not band.any():
            dia.append(0.0); continue
        w = band.sum(axis=1).max()               # widest row inside the disk
        dia.append(float(w) * sx)
    return np.array(dia), float(L) * sy


def volume_biplane(m2, m4, sp2, sp4):
    """Biplane disk-summation volume in mL. m2/m4 = A2C/A4C cavity masks."""
    d2, L2 = disk_diameters(m2, sp2[0], sp2[1])
    d4, L4 = disk_diameters(m4, sp4[0], sp4[1])
    if d2 is None or d4 is None: return None
    L = (L2 + L4) / 2.0                          # ASE: use the shorter/averaged long axis
    h = L / NDISK
    v_mm3 = (np.pi / 4.0) * h * float(np.sum(d2 * d4))
    return v_mm3 / 1000.0                        # mm^3 -> mL


def patient_ef(pdir, pid):
    def load(view, phase):
        p = os.path.join(pdir, '%s_%s_%s_gt.nii.gz' % (pid, view, phase))
        im = nib.load(p)
        z = im.header.get_zooms()
        return (im.get_fdata() == LV), (float(z[0]), float(z[1]))
    m2e, s2 = load('2CH', 'ED'); m4e, s4 = load('4CH', 'ED')
    m2s, _ = load('2CH', 'ES');  m4s, _ = load('4CH', 'ES')
    edv = volume_biplane(m2e, m4e, s2, s4)
    esv = volume_biplane(m2s, m4s, s2, s4)
    if edv is None or esv is None or edv <= 0: return None
    return edv, esv, (edv - esv) / edv * 100.0
