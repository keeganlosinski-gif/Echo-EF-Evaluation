"""Is the gamma sweep a REAL test, or does per-image standardisation absorb it too?

Standardisation cancels affine changes exactly, so the gain sweep proved nothing. Gamma is
nonlinear so it CANNOT be cancelled in principle -- but standardisation re-centres and
re-scales afterwards, which may absorb most of it in practice.

Measure what the NETWORK actually sees: standardise the clean image and the gamma-distorted
image, then compare. If the residual is negligible, the gamma sweep was also a non-test.
Reference point: the gain sweep, which must come out at exactly zero.
"""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import os, numpy as np, nibabel as nib
D=CAMUS_ROOT
pats=sorted(x for x in os.listdir(D) if x.startswith('patient'))[:120]

def std(im):
    t=im[im>4]
    m,s=(t.mean(),t.std()) if t.size>500 else (im.mean(),im.std())
    return (im-m)/max(float(s),1e-3)

def distort(im,gain=1.0,gamma=1.0,off=0.0):
    x=np.clip(im*gain+off,0,255)
    if gamma!=1.0: x=255.0*np.power(x/255.0,gamma)
    return x

rows=[]
for p in pats:
    im=nib.load(os.path.join(D,p,'%s_4CH_ED.nii.gz'%p)).get_fdata().astype(np.float32)
    base=std(im); sec=im>4
    r={}
    for nm,kw in [('gain 0.7',dict(gain=0.7)),('gain 1.5',dict(gain=1.5)),
                  ('offset +20',dict(off=20.0)),
                  ('gamma 0.5',dict(gamma=0.5)),('gamma 0.7',dict(gamma=0.7)),
                  ('gamma 1.5',dict(gamma=1.5)),('gamma 2.0',dict(gamma=2.0))]:
        d=std(distort(im,**kw))
        diff=np.abs(d[sec]-base[sec])
        r[nm]=(diff.mean(), np.corrcoef(d[sec],base[sec])[0,1])
    rows.append(r)

print("What the NETWORK sees after standardisation, vs the clean standardised image.")
print("(input has sd = 1.0 by construction, so MAD is in units of one standard deviation)\n")
print("%-12s %-16s %-12s"%("distortion","mean |diff| (sd)","corr"))
keys=list(rows[0].keys())
for k in keys:
    md=np.mean([r[k][0] for r in rows]); cr=np.mean([r[k][1] for r in rows])
    tag=""
    if md<0.01: tag="   <-- ABSORBED: non-test"
    elif md<0.05: tag="   <-- mostly absorbed"
    else: tag="   <-- REAL distortion survives"
    print("%-12s %-16.4f %-12.6f%s"%(k,md,cr,tag))
print()
print("Interpretation: gain/offset must be ~0 (affine, cancelled by construction).")
print("If gamma rows are also ~0, the compression sweep proved nothing either and the")
print("no-augmentation control is the ONLY evidence that the augmentation mattered.")
