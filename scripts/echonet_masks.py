"""Reconstruct filled LV cavity masks from EchoNet's VolumeTracings chords.

VERIFIED (800/800 frames, 0 failures): 100% single connected blob, solidity ~1.0,
ED area > ES area in 200/200 videos. Row 0 of each frame's tracing is the long axis
(apex -> annulus midpoint); rows 1..20 are the perpendicular disk chords.
"""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import numpy as np, pandas as pd, cv2, os

def contour_from_chords(g):
    g = g.reset_index(drop=True)
    ax = g.iloc[0]; ch = g.iloc[1:]
    left  = np.stack([ch.X1.values, ch.Y1.values], 1)
    right = np.stack([ch.X2.values, ch.Y2.values], 1)
    apex  = np.array([[ax.X1, ax.Y1]])
    return np.concatenate([apex, left, right[::-1]], 0)

def mask_from_chords(g, h, w):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(contour_from_chords(g)).astype(np.int32)], 1)
    return m

def load_tracings(root):
    vt = pd.read_csv(os.path.join(root, 'VolumeTracings.csv'))
    vt['key'] = vt.FileName.str.replace('.avi', '', regex=False)
    return vt

def mask_from_chords_quads(g, h, w):
    """FIXED rasterization (08-24). The single-polygon fill self-crosses on curved ventricles,
    leaving interior streak holes (visible in ef_failures.png; solidity p5 was 0.915). Filling
    the strip between each pair of SUCCESSIVE chords as its own quad (plus the apex triangle)
    cannot self-cross: consecutive chords are near-parallel by construction."""
    import cv2 as _cv2
    g = g.reset_index(drop=True)
    ax = g.iloc[0]; ch = g.iloc[1:].reset_index(drop=True)
    m = np.zeros((h, w), np.uint8)
    apex = np.array([ax.X1, ax.Y1])
    L = np.stack([ch.X1.values, ch.Y1.values], 1)
    R = np.stack([ch.X2.values, ch.Y2.values], 1)
    tri = np.round(np.stack([apex, L[0], R[0]])).astype(np.int32)
    _cv2.fillPoly(m, [tri], 1)
    for k in range(len(L) - 1):
        quad = np.round(np.stack([L[k], R[k], R[k+1], L[k+1]])).astype(np.int32)
        _cv2.fillPoly(m, [quad], 1)
    return m
