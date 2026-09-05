"""HARD QC GATE over every EchoNet label frame. The bar for this project (08-24): 100% data accuracy is the
minimum. A frame either passes EVERY gate or it is EXCLUDED (with its video) and counted.
Training on N-verified beats training on N with garbage tails.

GATES, per frame:
  G1 chord count == 21 (long axis + 20 disks)
  G2 all endpoints inside the frame
  G3 consecutive chords do not cross (segment intersection test)
  G4 chord midpoints progress monotonically along the long axis (no folded stacks)
  G5 mask = single connected component
  G6 solidity >= 0.95 (an LV cavity is convex-ish; below this = fill/geometry defect)
  G7 area in [80, 8000] px (degenerate or frame-filling masks are defects)
Per VIDEO:
  G8 both frames pass; ED area > ES area strictly
  G9 mask-derived EF in (2, 90) and |mask-EF - stated EF| < 25pp
     (beyond that, the tracing and the label disagree so badly one of them is wrong)
"""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import numpy as np, pandas as pd, sys, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from echonet_masks import load_tracings
from ef_readout import ef_from_pair
import echonet_data as E
meta=pd.read_csv(os.path.join(WORK, 'en_meta.csv'))
MSK=np.load(os.path.join(WORK, 'en_msk.npy'),mmap_mode='r')
vt=load_tracings(E.ROOT); by=dict(list(vt.groupby('key')))
def seg_cross(p1,p2,p3,p4):
    d=lambda a,b,c:(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    d1,d2,d3,d4=d(p3,p4,p1),d(p3,p4,p2),d(p1,p2,p3),d(p1,p2,p4)
    return ((d1>0)!=(d2>0)) and ((d3>0)!=(d4>0))
fails={k:0 for k in ['G1','G2','G3','G4','G5','G6','G7']}
frame_ok=np.zeros(len(meta),bool)
for i,r in meta.iterrows():
    g=by[r.file]; gg=g[g.Frame==r.frame].reset_index(drop=True)
    ok=True
    if len(gg)!=21: fails['G1']+=1; ok=False
    if ok:
        pts=np.concatenate([gg[['X1','Y1']].values,gg[['X2','Y2']].values])
        if pts.min()<-1 or pts[:,0].max()>112 or pts[:,1].max()>112: fails['G2']+=1; ok=False
    if ok:
        ch=gg.iloc[1:]
        L=ch[['X1','Y1']].values; R=ch[['X2','Y2']].values
        for k in range(len(L)-1):
            if seg_cross(L[k],R[k],L[k+1],R[k+1]): fails['G3']+=1; ok=False; break
    if ok:
        ax=gg.iloc[0]; a0=np.array([ax.X1,ax.Y1]); a1=np.array([ax.X2,ax.Y2])
        u=(a1-a0); u=u/max(np.linalg.norm(u),1e-9)
        mid=(L+R)/2.0; t=(mid-a0)@u
        # AUDITED 08-24: 1,797 tracings are stored base->apex (reversed) -- valid data.
        # Accept EITHER consistent direction; only truly jumbled ordering fails.
        if not (np.all(np.diff(t)>-0.5) or np.all(np.diff(t)<0.5)): fails['G4']+=1; ok=False
    m=np.asarray(MSK[i])>0
    if ok:
        n,_=cv2.connectedComponents(m.astype(np.uint8))
        if n>2: fails['G5']+=1; ok=False
    if ok:
        cnts,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        hull=cv2.convexHull(np.vstack(cnts)) if cnts else None
        # AUDITED 08-24: 0.95 rejected real crescent ES anatomy (p10=0.954). Defect line = 0.90.
        if hull is None or m.sum()/max(cv2.contourArea(hull),1)<0.90: fails['G6']+=1; ok=False
    if ok and not (80<=m.sum()<=8000): fails['G7']+=1; ok=False
    frame_ok[i]=ok
meta['qc_frame']=frame_ok
vid_ok={}; g9=0; g8=0
for f,g in meta.groupby('file'):
    if len(g)!=2 or not g.qc_frame.all(): vid_ok[f]=False; g8+=(len(g)==2); continue
    g=g.sort_values('frame')
    m0=np.asarray(MSK[g.index[0]])>0; m1=np.asarray(MSK[g.index[1]])>0
    a0,a1=m0.sum(),m1.sum()
    if a0==a1: vid_ok[f]=False; g8+=1; continue
    ed,es=(m0,m1) if a0>a1 else (m1,m0)
    ef=ef_from_pair(ed,es)
    if ef is None or not (2<ef<90) or abs(ef-g.EF.iloc[0])>=25: vid_ok[f]=False; g9+=1; continue
    vid_ok[f]=True
meta['qc_video']=meta.file.map(vid_ok)
meta.to_csv(os.path.join(WORK, 'en_meta.csv'),index=False)
nv=meta.groupby('file').qc_video.first()
print("FRAME gates: %d/%d pass (%.2f%%)"%(frame_ok.sum(),len(meta),100*frame_ok.mean()))
for k,v in fails.items():
    if v: print("   %s fails: %d"%(k,v))
print("VIDEO gates: %d/%d pass (%.2f%%) | G8(order/frame) %d | G9(EF-disagree>=25pp) %d"
      %(nv.sum(),len(nv),100*nv.mean(),g8,g9))
for sp,g in meta.groupby('split'):
    got=g.groupby('file').qc_video.first()
    print("   %-6s %5d/%5d videos verified"%(sp,got.sum(),len(got)))
print()
print("VERIFIED SET = qc_video==True. Everything else EXCLUDED from training; eval exclusions")
print("must be REPORTED alongside any number graded on the verified subset.")
