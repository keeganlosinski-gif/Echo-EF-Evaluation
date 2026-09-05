"""DECISIVE FORK for the +8.5pp CAMUS parity gap.
Area-length method: V = 8*A^2/(3*pi*L) per view, biplane-averaged. NO rotation, NO disk slicing,
NO annulus-band heuristic -- A = mask area, L = PCA-extent long axis. Geometrically independent
of the Simpson implementation.
  area-length ALSO ~+8pp high  -> gap is CAMUS-side (their EF provenance) -> documentable finding
  area-length ~0 bias          -> my disk/rotation geometry is the bug     -> fix simpson.py
"""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import os, numpy as np, nibabel as nib
D=CAMUS_ROOT
def vol_al(m, sx, sy):
    ys,xs=np.nonzero(m)
    if len(ys)<50: return None
    pts=np.stack([xs*sx, ys*sy],1).astype(np.float64)
    c=pts.mean(0); u,s,vt=np.linalg.svd(pts-c, full_matrices=False)
    proj=(pts-c)@vt[0]
    L=proj.max()-proj.min()                       # long-axis extent, mm
    A=m.sum()*sx*sy                               # area, mm^2
    return 8.0*A*A/(3.0*np.pi*L)/1000.0           # mm^3 -> mL
rows=[]
for p in sorted(x for x in os.listdir(D) if x.startswith('patient')):
    pd_=os.path.join(D,p)
    cfg=dict(l.split(':',1) for l in open(pd_+'/Info_4CH.cfg').read().splitlines() if ':' in l)
    ef_st=float({k.strip():v.strip() for k,v in cfg.items()}['EF'])
    ok=True; V={}
    for ph in ['ED','ES']:
        vs=[]
        for view in ['2CH','4CH']:
            im=nib.load(os.path.join(pd_,'%s_%s_%s_gt.nii.gz'%(p,view,ph)))
            z=im.header.get_zooms(); m=(im.get_fdata()==1)
            v=vol_al(m,float(z[0]),float(z[1]))
            if v is None: ok=False; break
            vs.append(v)
        if not ok: break
        V[ph]=0.5*(vs[0]+vs[1])                   # biplane average
    if not ok: continue
    if V['ED']<=0: continue
    rows.append((ef_st, (V['ED']-V['ES'])/V['ED']*100.0))
a=np.array(rows)
d=a[:,1]-a[:,0]
print("n=%d"%len(a))
print("stated EF      mean %.1f"%a[:,0].mean())
print("area-length EF mean %.1f"%a[:,1].mean())
print("bias %+.2f pp   MAE %.2f   sd %.2f   corr %.4f"%(d.mean(),np.abs(d).mean(),d.std(),np.corrcoef(a[:,0],a[:,1])[0,1]))
print()
print("Simpson-from-masks (prior): bias +8.45, corr 0.974")
print("VERDICT: %s"%("GAP IS CAMUS-SIDE (both independent methods read high -> their EF provenance differs)"
      if d.mean()>4 else ("MY DISK GEOMETRY IS THE BUG (area-length agrees with stated EF)" if abs(d.mean())<3
      else "AMBIGUOUS -- methods disagree with each other; geometry sensitivity dominates")))
