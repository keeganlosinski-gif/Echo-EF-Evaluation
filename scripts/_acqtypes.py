"""Identify the exact videos behind the d=9.1/8.8/5.0/4.7/4.5 chamber-outlier tiles,
classify acquisition type (split-screen / diamond / normal sector), and validate the
644-seam census by rendering the top suspects."""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import numpy as np, pandas as pd, cv2, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import echonet_data as E

meta=pd.read_csv(os.path.join(WORK, 'en_meta.csv'))
IM=np.load(os.path.join(WORK, 'en_img.npy'),mmap_mode='r')
ok=meta[meta.qc_video==True]

# --- recompute Mahalanobis chamber screen (same as _chamber_screen.py) to recover IDs ---
feats=[]
for i in ok.index:
    im=np.asarray(IM[i]).astype(np.float32); lit=im>8
    ys,xs=np.nonzero(lit)
    m=np.load(os.path.join(WORK, 'en_msk.npy'),mmap_mode='r')[i]>0
    if m.sum()<20: feats.append((i,np.nan,np.nan,np.nan,np.nan)); continue
    my,mx=np.nonzero(m)
    feats.append((i,mx.mean()/112,my.mean()/112,m.sum()/lit.sum(),(my.max()-my.min())/112))
F=pd.DataFrame(feats,columns=['i','cx','cy','frac','h']).dropna()
X=F[['cx','cy','frac','h']].values
mu=X.mean(0); C=np.cov(X.T); Ci=np.linalg.inv(C)
d=np.sqrt(np.einsum('ij,jk,ik->i',X-mu,Ci,X-mu))
F['d']=d
top=F.sort_values('d',ascending=False).head(12)
top=top.merge(meta[['file']],left_on='i',right_index=True)
print("chamber outliers (matches the panel d= labels):")
for _,r in top.iterrows(): print("  d=%.1f  %s  frame_idx=%d"%(r.d,r.file,r.i))

# --- acquisition-type metrics per outlier + top seam suspects ---
susp=pd.read_csv(os.path.join(WORK, 'composite_suspects.csv')).sort_values('seam',ascending=False)
first=dict((f,g.index[0]) for f,g in ok.groupby('file'))
def acq(im):
    lit=(im>8).astype(np.uint8)
    ys,xs=np.nonzero(lit)
    pts=np.column_stack([xs,ys]).astype(np.float32)
    (cx,cy),(w,h),ang=cv2.minAreaRect(pts)
    rectfill=lit.sum()/max(w*h,1)                    # fill of the min-area ROTATED rect
    diag = rectfill>0.85 and 25<abs(ang)%90<65       # diamond = rotated square, well filled
    colmean=lit.mean(0)
    seam=(colmean[30:45].mean()+colmean[67:82].mean())/2-colmean[52:60].mean()
    return rectfill,ang,seam,diag
print("\nacq-type of the 12 chamber outliers:")
for _,r in top.iterrows():
    im=np.asarray(IM[int(r.i)]).astype(np.float32)
    rf,ang,seam,diag=acq(im)
    kind='DIAMOND' if diag else ('SPLIT' if seam>0.25 else 'sector')
    print("  d=%.1f %s  rectfill=%.2f ang=%5.1f seam=%+.2f -> %s"%(r.d,r.file,rf,ang,seam,kind))

# --- validate the 644: render top-16 seam suspects ---
tiles=[]
for f in susp.f.head(16):
    i=first.get(f)
    if i is None: continue
    im=np.asarray(IM[i]).astype(np.float32)
    gg=cv2.cvtColor(np.clip(cv2.resize(im,(224,224),interpolation=cv2.INTER_NEAREST),0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    gg=cv2.copyMakeBorder(gg,18,2,2,2,cv2.BORDER_CONSTANT,value=(25,25,25))
    cv2.putText(gg,f[:18],(4,13),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,255,255),1,cv2.LINE_AA)
    tiles.append(gg)
rows=[np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]
cv2.imwrite(os.path.join(WORK, 'seam_suspects.png'),np.vstack(rows))
print("\nseam_suspects.png written (top16 of 644 by seam score)")

# --- full-dataset diamond census ---
dia=[]
for f,i in first.items():
    im=np.asarray(IM[i]).astype(np.float32)
    rf,ang,seam,d_=acq(im)
    if d_: dia.append((f,rf,ang))
print("DIAMOND/rotated-acquisition videos in verified set: %d"%len(dia))
for f,rf,ang in dia[:10]: print("   %s rectfill=%.2f ang=%.1f"%(f,rf,ang))
pd.DataFrame(dia,columns=['f','rectfill','ang']).to_csv(os.path.join(WORK, 'diamond_videos.csv'),index=False)
