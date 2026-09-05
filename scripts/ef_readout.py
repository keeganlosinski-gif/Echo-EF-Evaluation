"""STAGE 5: EF read-out from segmentation masks. The last pipeline stage, built first because it
is validatable on GROUND-TRUTH masks before any model exists.

Estimator: AREA-LENGTH  V = 8*A^2 / (3*pi*L), L = PCA-extent long axis.
Chosen over disk-Simpson for the mask->EF step because it needs no rotation, no annulus band,
no disk binning -- and it was validated 08-24 on CAMUS (agrees with disk-Simpson to 0.8pp; both
independently expose the cfg-EF provenance gap).

EchoNet (A4C only): monoplane area-length per frame -> EF from the traced ED/ES pair.
CAMUS (2CH+4CH):    biplane average of per-view volumes.
"""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import numpy as np

def vol_area_length(mask, sx=1.0, sy=1.0):
    """Volume (mL if spacings in mm) from a binary LV cavity mask. None if mask too small."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 50: return None
    pts = np.stack([xs * sx, ys * sy], 1).astype(np.float64)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    proj = (pts - c) @ vt[0]
    L = proj.max() - proj.min()
    if L <= 0: return None
    A = mask.sum() * sx * sy
    return 8.0 * A * A / (3.0 * np.pi * L) / 1000.0

def ef_from_pair(mask_ed, mask_es, sx=1.0, sy=1.0):
    """EF (percent) from an ED/ES mask pair (monoplane)."""
    ved = vol_area_length(mask_ed, sx, sy); ves = vol_area_length(mask_es, sx, sy)
    if ved is None or ves is None or ved <= 0: return None
    return (ved - ves) / ved * 100.0

def ef_biplane(m2_ed, m2_es, m4_ed, m4_es, s2=(1,1), s4=(1,1)):
    v_ed = [vol_area_length(m2_ed,*s2), vol_area_length(m4_ed,*s4)]
    v_es = [vol_area_length(m2_es,*s2), vol_area_length(m4_es,*s4)]
    if any(v is None for v in v_ed+v_es): return None
    ed = 0.5*(v_ed[0]+v_ed[1]); es = 0.5*(v_es[0]+v_es[1])
    return (ed-es)/ed*100.0 if ed>0 else None
