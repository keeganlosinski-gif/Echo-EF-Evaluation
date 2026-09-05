"""PRE-STAGE-4 FULL AUDIT (08-24: verify everything, median + BOTH tails, before GPU).
Every check prints PASS/FAIL; any FAIL blocks the build."""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import numpy as np, pandas as pd, sys, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FAILS=[]
def chk(name,cond,detail=""):
    print("  [%s] %-52s %s"%("PASS" if cond else "FAIL",name,detail))
    if not cond: FAILS.append(name)
def tails(name,a,lo=None,hi=None,fmt="%.2f"):
    a=np.asarray(a,float); a=a[np.isfinite(a)]
    q=np.percentile(a,[0,1,50,99,100])
    s="min "+fmt%q[0]+"  p1 "+fmt%q[1]+"  med "+fmt%q[2]+"  p99 "+fmt%q[3]+"  max "+fmt%q[4]
    ok=True
    if lo is not None: ok&=q[0]>=lo
    if hi is not None: ok&=q[4]<=hi
    chk("tails: "+name, ok, s)
print("=== 1. META INTEGRITY ===")
meta=pd.read_csv(os.path.join(WORK, 'en_meta.csv'))
chk("meta rows == 20048", len(meta)==20048, str(len(meta)))
for c in ['qc_video','noise_score','edge_flag','file','frame','EF','split']:
    chk("column present: "+c, c in meta.columns)
ok=meta[meta.qc_video==True]
per=ok.groupby('file').size()
chk("every verified video has exactly 2 frames", (per==2).all(), "%d videos"%len(per))
sp=ok.groupby('file').split.nunique()
chk("no video straddles splits", (sp==1).all())
cnt=ok.groupby('file').split.first().value_counts().to_dict()
chk("verified counts TRAIN/VAL/TEST", cnt.get('TRAIN')==7150 and cnt.get('VAL')==1220 and cnt.get('TEST')==1219, str(cnt))
print("=== 2. MASKS (quad-fixed) ===")
MSK=np.load(os.path.join(WORK, 'en_msk.npy'),mmap_mode='r')
chk("mask array shape", MSK.shape==(20048,112,112), str(MSK.shape))
from echonet_masks import mask_from_chords_quads, load_tracings
import echonet_data as E
vt=load_tracings(E.ROOT); by=dict(list(vt.groupby('key')))
rng=np.random.RandomState(3); idxs=rng.choice(ok.index,25,replace=False)
mismatch=0
for i in idxs:
    r=meta.loc[i]; gg=by[r.file]; gg=gg[gg.Frame==r.frame]
    if not np.array_equal(mask_from_chords_quads(gg,112,112).astype(bool),np.asarray(MSK[i])>0): mismatch+=1
chk("cached masks == quad rasterizer (25 random)", mismatch==0, "%d mismatches"%mismatch)
areas=np.array([(np.asarray(MSK[i])>0).sum() for i in ok.index[:4000]])
tails("mask area px (verified sample)",areas,lo=80,hi=8000,fmt="%.0f")
print("=== 3. LABELS / EF ===")
efs=ok.groupby('file').EF.first()
tails("stated EF",efs,lo=2,hi=100)
from ef_readout import ef_from_pair
diffs=[]; edes_bad=0
for f,g in list(ok.groupby('file'))[:1500]:
    g=g.sort_values('frame')
    m0=np.asarray(MSK[g.index[0]])>0; m1=np.asarray(MSK[g.index[1]])>0
    if m0.sum()==m1.sum(): edes_bad+=1; continue
    ed,es=(m0,m1) if m0.sum()>m1.sum() else (m1,m0)
    ef=ef_from_pair(ed,es)
    if ef is None: edes_bad+=1; continue
    diffs.append(ef-g.EF.iloc[0])
chk("ED>ES + readout valid on all sampled", edes_bad==0, "%d bad"%edes_bad)
tails("mask-EF minus stated (verified, n=%d)"%len(diffs),diffs,lo=-25,hi=25,fmt="%+.1f")
print("=== 4. QUALITY SIGNALS ===")
ns=ok.noise_score
chk("noise_score finite on verified", ns.notna().all(), "%d NaN"%ns.isna().sum())
tails("noise_score",ns,lo=-5,hi=300,fmt="%.1f")
print("=== 5. VIDEOS READABLE (tail sample: lowest/highest noise + random) ===")
vids=ok.groupby('file').noise_score.mean().sort_values()
test_files=list(vids.index[:4])+list(vids.index[-4:])+list(rng.choice(vids.index,6,replace=False))
bad=0
fl=pd.read_csv(E.ROOT+'/FileList.csv'); fl['key']=fl.FileName.str.replace('.avi','',regex=False)
nfr=dict(zip(fl.key,fl.NumberOfFrames))
for f in test_files:
    try:
        v=E.read_video(E.vpath(f))
        if len(v)!=nfr.get(f,-1): bad+=1
        fr=ok[ok.file==f].frame.max()
        if fr>=len(v): bad+=1
    except Exception: bad+=1
chk("videos decode + frame counts + traced idx in range (14 tail/random)", bad==0, "%d bad"%bad)
print("=== 6. SEGMENTER v3 CHECKPOINT (post-crash integrity) ===")
try:
    from unet import UNet
    ck=torch.load(os.path.join(WORK, 'seg_combined_v3.pt'),map_location='cuda',weights_only=False)
    net=UNet(4,ck['enc'],False,1).cuda(); net.load_state_dict(ck['sd']); net.eval()
    IM=np.load(os.path.join(WORK, 'en_img.npy'),mmap_mode='r')
    sub=ok[ok.split=='VAL'].index[:200]
    inter=0; union=0
    for i in sub:
        im=np.asarray(IM[i]).astype(np.float32)
        x=cv2.resize(im,(ck['size'],ck['size']),interpolation=cv2.INTER_CUBIC)
        t=x[x>4]; mu,sd=(t.mean(),t.std()) if t.size>500 else (x.mean(),x.std())
        xx=torch.from_numpy(((x-mu)/max(float(sd),1e-3)).astype(np.float32))[None,None].cuda()
        with torch.no_grad(), torch.amp.autocast('cuda',dtype=torch.bfloat16):
            p=net(xx).argmax(1)[0].cpu().numpy().astype(np.uint8)
        p=cv2.resize(p,(112,112),interpolation=cv2.INTER_NEAREST)==1
        g=np.asarray(MSK[i])>0
        inter+=(p&g).sum(); union+=p.sum()+g.sum()
    dice=2*inter/max(union,1)
    chk("v3 pooled Dice on 200 VAL frames ~0.92", 0.90<dice<0.96, "%.4f"%dice)
except Exception as e:
    chk("v3 checkpoint loads+infers", False, str(e)[:70])
print()
print("="*68)
print("RESULT:", "ALL CHECKS PASSED - CLEAR TO BUILD" if not FAILS else "BLOCKED: %s"%FAILS)
