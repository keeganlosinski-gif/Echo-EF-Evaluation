"""HUMAN ADJUDICATION PANEL (reviewer: the author, RDCS): the verification a script cannot do.
60 frames, stratified: 30 random gate-passers, 15 solidity-borderline (0.90-0.95),
15 lowest-solidity passers. Each tile: raw frame LEFT, chord-mask overlay RIGHT, numbered.
Review protocol: mark any tile where the green does NOT follow the endocardial border
(papillary handling per EchoNet convention: papillary muscles INCLUDED in cavity).
The reviewer records the bad tile numbers (or 'all good'). Decision rule, fixed before viewing:
0/60 bad -> label error <5% at 95% CI. The tally must be written into the log to be quotable."""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import numpy as np, pandas as pd, cv2, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
meta=pd.read_csv(os.path.join(WORK, 'en_meta.csv'))
MSK=np.load(os.path.join(WORK, 'en_msk.npy'),mmap_mode='r')
IM=np.load(os.path.join(WORK, 'en_img.npy'),mmap_mode='r')
ok=meta[meta.qc_video==True].copy()
# solidity per frame for stratification
sols=[]
for i in ok.index:
    m=(np.asarray(MSK[i])>0).astype(np.uint8)
    c,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    sols.append(m.sum()/max(cv2.contourArea(cv2.convexHull(np.vstack(c))),1) if c else 0)
ok['sol']=sols
rng=np.random.RandomState(7)
rand=ok.sample(30,random_state=7)
border=ok[(ok.sol>=0.90)&(ok.sol<0.95)].sample(min(15,((ok.sol>=0.90)&(ok.sol<0.95)).sum()),random_state=7)
lowest=ok.nsmallest(15,'sol')
panel=pd.concat([rand,border,lowest]).drop_duplicates().head(60).reset_index()
R=180
tiles=[]
for k,row in panel.iterrows():
    im=np.asarray(IM[row['index']]); m=np.asarray(MSK[row['index']])>0
    g=cv2.cvtColor(np.clip(im,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    ov=g.copy(); ov[m]=(np.array([0,200,0])*0.4+g[m]*0.6).astype(np.uint8)
    cnt,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ov,cnt,-1,(0,255,255),1)
    t=np.hstack([cv2.resize(g,(R,R),interpolation=cv2.INTER_NEAREST),cv2.resize(ov,(R,R),interpolation=cv2.INTER_NEAREST)])
    t=cv2.copyMakeBorder(t,20,2,2,2,cv2.BORDER_CONSTANT,value=(30,30,30))
    cv2.putText(t,'#%02d  sol %.2f'%(k+1,row['sol']),(6,15),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)
    tiles.append(t)
rows=[np.hstack(tiles[i:i+5]) for i in range(0,len(tiles),5)]
w=max(r.shape[1] for r in rows)
rows=[cv2.copyMakeBorder(r,0,0,0,w-r.shape[1],cv2.BORDER_CONSTANT,value=(0,0,0)) for r in rows]
out=np.vstack(rows)
cv2.imwrite(os.path.join(WORK, 'adjudication_panel.png'),out)
panel[['index','file','frame','sol']].to_csv(os.path.join(WORK, 'adjudication_key.csv'),index=False)
print("panel: %d tiles (30 random / %d borderline / 15 lowest-solidity passers)"%(len(panel),len(border)))
