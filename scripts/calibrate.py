"""Stage 2: multi-anchor intensity calibration.

WHY MORE THAN TWO ANCHORS (measured, not assumed):
  The affine-invariant shape statistic s=(myo-cav)/(rim-cav) has CV 0.298 across CAMUS studies
  against ~0.02-0.05 expected from noise. Studies therefore differ in transfer-curve SHAPE, not
  only level. After a PERFECT two-point affine fit on cavity+rim, the myocardium still lands
  anywhere in s = 0.194..0.551 -- a residual mid-tone error of 36% of the range. A line cannot
  fix a curve; three or more knots are required.

WHY THESE ANCHORS:
  A0 outside-sector  exactly 0.0 in all 500 studies -> a free, exact origin
  A1 LV cavity       Spearman vs EF -0.021 (p=0.65)  clean
  A2 myocardium      Spearman vs EF +0.030 (p=0.50)  clean
  A3 bright rim      pericardial side, obtained by dilating the myocardium mask
  LA is EXCLUDED: Spearman vs EF +0.187 (p<0.0001), monotonic 75->101 across EF bands. It
  tracks atrial dilation, which tracks filling pressure, which tracks the target.

ROBUSTNESS: each anchor is a MEDIAN over a region (thousands of pixels), not a landmark point,
so speckle and a locally abnormal patch cannot move it. With 4 knots the map survives one
corrupted anchor via the monotonicity guard below.
"""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import numpy as np
from scipy import ndimage

BG, LV, MYO, LA = 0, 1, 2, 3
# canonical targets = the measured CAMUS population medians (raw 8-bit), scaled to [0,1]
CANON = np.array([0.0, 60.0, 98.0, 161.5], dtype=np.float64) / 255.0


def extract_anchors(im, mask, sector_thresh=8.0, rim_iters=4, rim_pct=85):
    """Return the 4 anchor levels in raw units, or None if the mask is unusable."""
    sec = im > sector_thresh
    cav = mask == LV; myo = mask == MYO
    if cav.sum() < 100 or myo.sum() < 100: return None
    rim = ndimage.binary_dilation(myo, iterations=rim_iters) & (~myo) & (~cav) & (mask == BG) & sec
    if rim.sum() < 100: return None
    a0 = float(np.median(im[~sec])) if (~sec).sum() > 200 else 0.0
    a1 = float(np.median(im[cav])); a2 = float(np.median(im[myo]))
    a3 = float(np.percentile(im[rim], rim_pct))
    return np.array([a0, a1, a2, a3], dtype=np.float64)


def _monotone(a, min_gap=2.0):
    """Enforce strictly increasing knots. A non-monotone anchor means a bad segmentation or a
    genuinely odd study; we repair rather than divide by zero (the failure that produced a
    2.8-million-x spread when the two-point map was tried)."""
    a = a.astype(np.float64).copy()
    for i in range(1, len(a)):
        if a[i] <= a[i - 1] + min_gap: a[i] = a[i - 1] + min_gap
    return a


def calibrate(im, anchors, canon=CANON, clip=(0.0, 1.5)):
    """Piecewise-linear monotone map anchors -> canon, linear extrapolation past the ends."""
    a = _monotone(np.asarray(anchors, np.float64))
    x = im.astype(np.float64)
    y = np.interp(x, a, canon * 255.0)
    # linear extrapolation beyond the outer knots so bright specular targets are not flattened
    hi_s = (canon[-1] - canon[-2]) * 255.0 / max(a[-1] - a[-2], 1e-6)
    m = x > a[-1]
    if m.any(): y[m] = canon[-1] * 255.0 + (x[m] - a[-1]) * hi_s
    y = y / 255.0
    return np.clip(y, clip[0], clip[1]).astype(np.float32)


def shape_stat(anchors):
    """s = (myo - cav)/(rim - cav). Invariant to any affine change; moves only if the CURVE differs."""
    a = np.asarray(anchors, np.float64)
    d = a[3] - a[1]
    return float('nan') if abs(d) < 1e-6 else float((a[2] - a[1]) / d)
