"""AUDIT THE GATE before excluding a large slice of a benchmark (the first pass looked like ~27%; the
final ledger excluded 4.4%).
Suspect 1: G4 'non-monotone chord order' -- if some tracings list chords base->apex (reversed),
my forward-monotone test fails VALID data. Test: are G4 failures monotone in REVERSE?
Suspect 2: G6 solidity>=0.95 -- a diseased LV at ES is genuinely non-convex; 0.95 may reject
real anatomy. Look at the distribution and where failures sit relative to it."""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import numpy as np, pandas as pd, sys, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from echonet_masks import load_tracings
import echonet_data as E
meta=pd.read_csv(os.path.join(WORK, 'en_meta.csv'))
MSK=np.load(os.path.join(WORK, 'en_msk.npy'),mmap_mode='r')
vt=load_tracings(E.ROOT); by=dict(list(vt.groupby('key')))
rev_mono=0; true_jumble=0; checked=0
sols=[]
for i,r in meta.iterrows():
    g=by[r.file]; gg=g[g.Frame==r.frame].reset_index(drop=True)
    if len(gg)!=21: continue
    ax=gg.iloc[0]; a0=np.array([ax.X1,ax.Y1]); a1=np.array([ax.X2,ax.Y2])
    u=(a1-a0); u=u/max(np.linalg.norm(u),1e-9)
    ch=gg.iloc[1:]; mid=(ch[['X1','Y1']].values+ch[['X2','Y2']].values)/2.0
    t=(mid-a0)@u
    fwd=np.all(np.diff(t)>-0.5); rev=np.all(np.diff(t)<0.5)
    if not fwd:
        checked+=1
        if rev: rev_mono+=1
        else:
            d=np.diff(t); frac_bad=np.mean(d<=-0.5)
            if frac_bad<0.15: rev_mono+=0  # mostly fine, minor wobble
            true_jumble+=1 if frac_bad>=0.15 else 0
    m=np.asarray(MSK[i])>0
    cnts,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if cnts:
        sols.append(m.sum()/max(cv2.contourArea(cv2.convexHull(np.vstack(cnts))),1))
sols=np.array(sols)
print("G4 audit: of %d forward-monotone failures: %d are PERFECTLY monotone in REVERSE"%(checked,rev_mono))
print("          (reversed storage order = valid data, gate bug) | truly jumbled: %d"%true_jumble)
print()
print("G6 audit: solidity distribution  p1 %.3f  p5 %.3f  p10 %.3f  p25 %.3f  median %.3f"
      %tuple(np.percentile(sols,[1,5,10,25,50])))
print("          threshold 0.95 rejects %.1f%% | 0.90 rejects %.1f%% | 0.85 rejects %.1f%%"
      %(100*(sols<0.95).mean(),100*(sols<0.90).mean(),100*(sols<0.85).mean()))
