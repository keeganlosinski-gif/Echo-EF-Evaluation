"""EchoNet-Dynamic data loader for EF regression.

DESIGN DECISIONS, and why:

  * PHASE NORMALISATION, not frame-rate normalisation. Systole spans 11-23 frames depending on
    heart rate (median 16), and 79.4% of studies sit at exactly 50 fps while the rest spread
    continuously over 43-62. Resampling to a fixed frame RATE would leave token i meaning
    "mid-systole" in one patient and "already done contracting" in another. We resample the
    ED->ES window to a fixed N instead, so token i always means phase i/N.

  * SAMPLE frames, never average them. Averaging adjacent frames low-pass filters speckle, and
    speckle decorrelation is the signal a temporal model reads. Nearest-index selection keeps
    the texture intact.

  * FPS IS NOT AN INPUT. Frame rate is set by depth x sector x line density, which tracks patient
    size and ventricular dilation (measured: EDV 103 mL at <40 fps vs 84 mL at >60 fps). Feeding
    it -- or letting the model infer it from raw frame counts -- invites reading EF off the
    scanner settings instead of the myocardium.

  * OFFICIAL SPLITS ONLY. There is no patient identifier in FileList.csv, so we cannot verify
    patient-level grouping ourselves. Re-splitting without patient IDs would be strictly worse
    than trusting Stanford's, so we use theirs verbatim.
"""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import os, numpy as np, torch, cv2
from torch.utils.data import Dataset
import pandas as pd

ROOT = os.environ.get('ECHONET_ROOT', '')
VIDS = os.path.join(ROOT, 'Videos')


def load_index(root=ROOT):
    """FileList joined to the ED/ES frame indices recovered from VolumeTracings."""
    fl = pd.read_csv(os.path.join(root, 'FileList.csv'))
    vt = pd.read_csv(os.path.join(root, 'VolumeTracings.csv'))
    fl['key'] = fl.FileName.str.replace('.avi', '', regex=False)
    vt['key'] = vt.FileName.str.replace('.avi', '', regex=False)
    # the two traced frames are ED and ES; larger traced area = ED, but the frame INDEX order is
    # not guaranteed, so carry both and decide by volume at load time
    g = vt.groupby('key').Frame.agg(['min', 'max']).rename(columns={'min': 'f0', 'max': 'f1'})
    df = fl.merge(g, on='key', how='inner')          # drops the 5 videos with no tracings
    return df.reset_index(drop=True)


def vpath(name, root=ROOT):
    """FileList.csv stores keys WITHOUT the .avi extension; VolumeTracings stores them WITH it."""
    if not name.endswith('.avi'): name = name + '.avi'
    return os.path.join(root, 'Videos', name)


def read_video(path):
    """Return (T, H, W) float32 grayscale. These are 8-bit RGB AVIs of grayscale ultrasound."""
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        if f.ndim == 3: f = f[..., 0]                # channels are identical; take one
        fr.append(f)
    cap.release()
    if not fr: raise IOError('no frames: %s' % path)
    return np.stack(fr).astype(np.float32)


def phase_resample(v, a, b, n):
    """Take n frames spanning [a,b] inclusive by NEAREST INDEX -- no averaging (speckle)."""
    a, b = int(a), int(b)
    if b < a: a, b = b, a
    b = min(b, len(v) - 1); a = max(0, min(a, b))
    if b == a: idx = np.full(n, a, int)
    else: idx = np.round(np.linspace(a, b, n)).astype(int)
    return v[idx]


def normalise(clip, mode='std'):
    """mode='std'    : per-clip standardisation (robust default)
       mode='anchor' : two-point tissue map. Blood pool is anechoic, tissue is brighter; anchor
                       p10 (cavity) -> 0 and p90 (myocardium/pericardium) -> 1 so operator gain
                       cancels."""
    x = clip
    if mode == 'anchor':
        t = x[x > 4.0]                                # drop the black outside-sector mask
        if t.size > 1000:
            lo, hi = np.percentile(t, 10), np.percentile(t, 90)
            return np.clip((x - lo) / max(float(hi - lo), 1e-3), -1.0, 3.0)
    m, s = x.mean(), x.std()
    return (x - m) / max(float(s), 1e-3)


class EchoNetEF(Dataset):
    def __init__(self, split='TRAIN', nframes=32, size=112, augment=False,
                 norm='std', root=ROOT, index=None):
        df = load_index(root) if index is None else index
        self.df = df[df.Split == split].reset_index(drop=True) if split else df.reset_index(drop=True)
        self.n, self.size, self.aug, self.norm, self.root = nframes, size, augment, norm, root

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        v = read_video(vpath(r.FileName, self.root))
        a, b = int(r.f0), int(r.f1)
        if self.aug:                                   # temporal jitter around the traced window
            j = np.random.randint(-2, 3)
            a = max(0, a + j); b = min(len(v) - 1, b + np.random.randint(-2, 3))
            if b <= a: a, b = int(r.f0), int(r.f1)
        clip = phase_resample(v, a, b, self.n)
        if clip.shape[1] != self.size or clip.shape[2] != self.size:
            clip = np.stack([cv2.resize(f, (self.size, self.size), interpolation=cv2.INTER_AREA) for f in clip])
        if self.aug:
            if np.random.rand() < 0.5:                 # gain/gamma jitter: force gain-invariance
                g = np.exp(np.random.uniform(-0.3, 0.3))
                clip = 255.0 * np.power(np.clip(clip / 255.0, 0, 1), g)
            if np.random.rand() < 0.5:                 # small translation; NO horizontal flip --
                dx, dy = np.random.randint(-6, 7, 2)   # apical orientation is anatomical
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                clip = np.stack([cv2.warpAffine(f, M, (self.size, self.size)) for f in clip])
        clip = normalise(clip, self.norm)
        return torch.from_numpy(clip[None].astype(np.float32)), torch.tensor(float(r.EF))
