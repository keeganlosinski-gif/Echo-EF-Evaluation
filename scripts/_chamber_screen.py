"""WRONG-CHAMBER SCREEN (an RV tracing was spotted among the excluded). A wrong-chamber trace
over a DARK cavity would pass all existing gates. Screen every verified frame by SPATIAL
position: in EchoNet's A4C display the LV occupies a consistent region; a tracing centered where
the RV/RA/LA live is a laterality outlier. Metric: mask centroid (x,y) in frame coords ->
Mahalanobis distance from the population. Render the worst for RDCS eyes."""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import numpy as np, pandas as pd, cv2
meta=pd.read_csv(os.path.join(WORK, 'en_meta.csv'))
MSK=np.load(os.path.join(WORK, 'en_msk.npy'),mmap_mode='r'); IM=np.load(os.path.join(WORK, 'en_img.npy'),mmap_mode='r')
ok=meta[meta.qc_video==True]
cent=np.zeros((len(ok),2)); idxs=np.array(ok.index)
for k,i in enumerate(idxs):
    ys,xs=np.nonzero(np.asarray(MSK[i])>0)
    cent[k]=[xs.mean(),ys.mean()]
mu=cent.mean(0); C=np.cov(cent.T); Ci=np.linalg.inv(C)
d=np.sqrt(np.einsum('ij,jk,ik->i',cent-mu,Ci,cent-mu))
print("centroid population: mean (%.1f,%.1f)  sd (%.1f,%.1f)"%(mu[0],mu[1],cent[:,0].std(),cent[:,1].std()))
print("Mahalanobis tails: p50 %.2f  p99 %.2f  p99.9 %.2f  max %.2f"%tuple(np.percentile(d,[50,99,99.9,100])))
worst=np.argsort(d)[::-1][:12]
print("outliers >4 sd: %d frames"%(d>4).sum())
R=170; tiles=[]
for k in worst:
    i=idxs[k]
    im=np.asarray(IM[i]); m=(np.asarray(MSK[i])>0).astype(np.uint8)
    g=cv2.cvtColor(np.clip(im,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    cnt,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(g,cnt,-1,(0,255,0),1)
    t=cv2.resize(g,(R,R),interpolation=cv2.INTER_NEAREST)
    t=cv2.copyMakeBorder(t,18,2,2,2,cv2.BORDER_CONSTANT,value=(25,25,25))
    cv2.putText(t,'d=%.1f'%d[k],(4,13),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,cv2.LINE_AA)
    tiles.append(t)
rows=[np.hstack(tiles[i:i+6]) for i in range(0,len(tiles),6)]
cv2.imwrite(os.path.join(WORK, 'chamber_screen.png'),np.vstack(rows))
print("wrote chamber_screen.png (12 most spatially anomalous verified tracings)")
