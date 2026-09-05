"""TRACING-vs-IMAGE agreement -- the non-circular automated check (principle: use the data
before using the human). A correct endocardial tracing sits on a dark->bright edge: pixels just
INSIDE the contour (blood) darker than just OUTSIDE (myocardium).
METRIC per frame: polarity = fraction of contour where mean(outside strip) > mean(inside strip).
NULL CONTROL: the same contour eroded/dilated 4px must score WORSE, else the metric is vacuous
(the day's standing rule: a check must be able to fail).
Frames scoring below the null band get flagged -- ONLY those go to human adjudication."""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import numpy as np, pandas as pd, cv2, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
meta=pd.read_csv(os.path.join(WORK, 'en_meta.csv'))
MSK=np.load(os.path.join(WORK, 'en_msk.npy'),mmap_mode='r'); IM=np.load(os.path.join(WORK, 'en_img.npy'),mmap_mode='r')
K=np.ones((3,3),np.uint8)
def polarity(im,m):
    m8=m.astype(np.uint8)
    inner=(m8-cv2.erode(m8,K,iterations=2)).astype(bool)      # 1-2px strip inside
    outer=(cv2.dilate(m8,K,iterations=2)-m8).astype(bool)     # 1-2px strip outside
    if inner.sum()<20 or outer.sum()<20: return np.nan
    # local comparison: blur strips to compare locally, not global means
    return float((im[outer].mean()-im[inner].mean()))
ok=meta[meta.qc_video==True]
true_s=[]; ero_s=[]; dil_s=[]
idx=list(ok.index)
for i in idx:
    im=np.asarray(IM[i]).astype(np.float32); m=np.asarray(MSK[i])>0
    true_s.append(polarity(im,m))
    m8=m.astype(np.uint8)
    ero_s.append(polarity(im,cv2.erode(m8,K,iterations=4)>0))
    dil_s.append(polarity(im,cv2.dilate(m8,K,iterations=4)>0))
t,e,d=map(lambda a:np.array(a,float),(true_s,ero_s,dil_s))
v=np.isfinite(t)&np.isfinite(e)&np.isfinite(d)
t,e,d=t[v],e[v],d[v]; idx=np.array(idx)[v]
print("frames scored: %d"%len(t))
print("edge contrast (out-in, higher=better):")
print("  TRUE contour   median %+6.2f  p10 %+6.2f"%(np.median(t),np.percentile(t,10)))
print("  eroded 4px     median %+6.2f   <- null: should be WORSE"%np.median(e))
print("  dilated 4px    median %+6.2f   <- null: should be WORSE"%np.median(d))
beats=np.mean((t>e)&(t>d))
print("  true beats BOTH nulls in %.1f%% of frames  (metric validity)"%(100*beats))
thr=np.percentile(np.concatenate([e,d]),75)
flag=t<thr
print()
print("FLAGGED (true contour no better than shifted-null band): %d frames (%.2f%%)"%(flag.sum(),100*flag.mean()))
pd.DataFrame({'idx':idx[flag],'score':t[flag]}).to_csv(os.path.join(WORK, 'edge_flags.csv'),index=False)
print("-> only THESE need human eyes; the rest are verified tracing-on-edge computationally.")
