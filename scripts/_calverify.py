"""Does the calibration map actually make studies comparable?

NAIVE TEST IS CIRCULAR: calibrating TO the anchors and then re-measuring those anchors returns
the canonical values by construction. So we HOLD OUT a knot.

Fit the map using only {outside-sector, cavity, rim} and then ask where the MYOCARDIUM lands.
If the map genuinely linearises the transfer curve, myocardium lands in the same place in every
study. If studies differ in curve shape in a way the map cannot capture, it scatters.

Compared against the 2-point affine baseline (cavity+rim only), which we predicted leaves ~36%
residual. Ground-truth masks are used so this measures the CALIBRATION alone, with segmentation
error excluded -- that comes later.
"""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import os, sys, numpy as np, nibabel as nib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import extract_anchors, calibrate, shape_stat, CANON
from scipy import ndimage, stats

D=CAMUS_ROOT
rows=[]
for p in sorted(x for x in os.listdir(D) if x.startswith('patient')):
    pd_=os.path.join(D,p)
    cfg=dict(l.split(':',1) for l in open(pd_+'/Info_4CH.cfg').read().splitlines() if ':' in l)
    cfg={k.strip():v.strip() for k,v in cfg.items()}
    im=nib.load(os.path.join(pd_,'%s_4CH_ED.nii.gz'%p)).get_fdata().astype(np.float32)
    gt=nib.load(os.path.join(pd_,'%s_4CH_ED_gt.nii.gz'%p)).get_fdata().astype(np.uint8)
    a=extract_anchors(im,gt)
    if a is None: continue
    myo_mask=(gt==2)

    # --- 2-POINT affine on cavity+rim (the naive baseline) ---
    lo,hi=a[1],a[3]
    im2=(im-lo)/max(hi-lo,1e-6)
    myo_2pt=float(np.median(im2[myo_mask]))

    # --- 3-KNOT monotone map, MYOCARDIUM HELD OUT ---
    a3=np.array([a[0],a[1],a[3]]); c3=np.array([CANON[0],CANON[1],CANON[3]])
    im3=np.interp(im.astype(np.float64),np.sort(a3),c3*255.0)/255.0
    myo_3knot=float(np.median(im3[myo_mask]))

    rows.append(dict(ef=float(cfg['EF']),q=cfg['ImageQuality'],
                     raw=float(np.median(im[myo_mask])), two=myo_2pt, three=myo_3knot,
                     s=shape_stat(a)))
import pandas as pd
d=pd.DataFrame(rows); print("patients:",len(d)); print()

def rep(nm,v):
    v=np.asarray(v,float); v=v[np.isfinite(v)]
    cv=v.std()/max(abs(v.mean()),1e-9)
    print("  %-34s median %7.3f   p5-p95 %7.3f - %7.3f   CV %.4f"%(nm,np.median(v),np.percentile(v,5),np.percentile(v,95),cv))
    return cv

print("=== WHERE DOES THE HELD-OUT MYOCARDIUM LAND? (lower CV = more comparable) ===")
c_raw = rep("raw (no calibration)",       d.raw)
c_two = rep("2-point affine (cav,rim)",   d.two)
c_thr = rep("3-knot monotone (0,cav,rim)",d.three)
print()
print("  CV reduction vs raw:   2-point %.1f%%   3-knot %.1f%%"%(100*(1-c_two/c_raw),100*(1-c_thr/c_raw)))
print("  3-knot vs 2-point:     %+.1f%% further reduction"%(100*(1-c_thr/c_two)))
print()
print("=== does the residual still leak EF? ===")
for nm,c in [('raw',d.raw),('2-point',d.two),('3-knot',d.three)]:
    r,pv=stats.spearmanr(c,d.ef)
    print("  spearman(%-8s, EF) = %+.4f  p=%.4f %s"%(nm,r,pv,"  <-- leaks" if pv<0.05 else "  clean"))
print()
print("=== residual by image quality (3-knot) ===")
for q in ['Good','Medium','Poor']:
    g=d[d.q==q]
    if len(g): print("  %-7s n=%3d  myo lands at %.3f  sd %.3f"%(q,len(g),g.three.median(),g.three.std()))
print()
print("NOTE: a perfect map would put myocardium at the canonical %.3f in every study."%(CANON[2]))
print("      residual scatter = curve variation the anchors cannot express.")
