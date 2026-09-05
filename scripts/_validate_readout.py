"""VALIDATE stage 5 on GROUND-TRUTH masks before any model touches it.
EchoNet: chord-reconstructed GT masks (cached en_msk.npy) -> monoplane area-length EF vs the
dataset's stated EF. This measures the READOUT's intrinsic error floor: whatever bias/spread
appears here is the best the full pipeline can ever do, and any model only adds error on top."""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import numpy as np, pandas as pd, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ef_readout import ef_from_pair
IM=np.load(os.path.join(WORK, 'en_msk.npy'),mmap_mode='r')
meta=pd.read_csv(os.path.join(WORK, 'en_meta.csv'))
rows=[]
for f,g in meta.groupby('file'):
    if len(g)!=2: continue
    g=g.sort_values('frame')
    m0=np.asarray(IM[g.index[0]])>0; m1=np.asarray(IM[g.index[1]])>0
    a0,a1=m0.sum(),m1.sum()
    ed,es=(m0,m1) if a0>=a1 else (m1,m0)          # ED = larger cavity
    ef=ef_from_pair(ed,es)
    if ef is None: continue
    rows.append((f,float(g.EF.iloc[0]),ef,g.split.iloc[0]))
d=pd.DataFrame(rows,columns=['f','ef_st','ef_al','split'])
print("videos: %d"%len(d))
for sp,g in [('ALL',d)]+list(d.groupby('split')):
    dd=g.ef_al-g.ef_st
    print("%-6s n=%5d  bias %+6.2f  MAE %5.2f  sd %5.2f  corr %.4f"%(
        sp,len(g),dd.mean(),dd.abs().mean(),dd.std(),np.corrcoef(g.ef_st,g.ef_al)[0,1]))
print()
print("Reference: monoplane disk-Simpson-from-chords gave bias -0.65 / MAE 1.05 / corr 0.971.")
print("If area-length is close to that, stage 5 is validated; a constant bias is calibratable.")
d.to_csv(os.path.join(WORK, 'readout_val.csv'),index=False)
