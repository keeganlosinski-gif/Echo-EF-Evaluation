"""WAS THE v4 TEST A GOOD TEST? (a) feature scales, (b) flow sanity vs dArea/dt,
(c) input-knockout on the trained v4 model (banked ablation-audit law)."""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import numpy as np, torch, os, sys, glob
os.environ['TOK_DIR']=os.path.join(WORK, 'tokens_v4'); os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_tok as TT

print("== (a) FEATURE-GROUP SCALES (sd per group; healthy training wants comparable) ==")
X=[]
for k in list(TT.data.keys())[:300]:
    X.append(TT.data[k][0].astype(np.float32))
X=np.concatenate(X,0)
groups={'deep 0:2048':X[:,:2048],'mid 2048:2304':X[:,2048:2304],'radial 2304:2336':X[:,2304:2336],'flow 2336:2340':X[:,2336:2340]}
for g,v in groups.items():
    print("  %-16s mean %+8.3f  sd %8.3f  |max| %9.1f"%(g,v.mean(),v.std(),np.abs(v).max()))

print("\n== (b) FLOW SANITY: corr(radial-flow, -dArea/dt) per video (motion should track area change) ==")
cs=[]
for k in list(TT.data.keys())[:200]:
    f,a=TT.data[k][0].astype(np.float32),TT.data[k][1].astype(np.float32)
    rad=f[:,2337]              # mean radial velocity in band (+out/-in)
    dA=np.gradient(a[:,0])     # LV area derivative
    if dA.std()>1 and rad.std()>1e-6:
        cs.append(np.corrcoef(rad[1:],dA[1:])[0,1])
cs=np.array(cs); cs=cs[np.isfinite(cs)]
print("  corr(radial_out, dArea): median %+.3f  (physical signal => strongly POSITIVE; noise => ~0)"%np.median(cs))

print("\n== (c) INPUT KNOCKOUT on trained v4 (does the model even USE the new features?) ==")
net=TT.TokAttn(); net.load_state_dict(torch.load(os.path.join(WORK, 'tok_v4.pt'),map_location='cpu')); net.eval()
ds=TT.DS('TEST',False)
def preds(mask_slice=None):
    out=np.zeros(300)
    with torch.no_grad():
        for i in range(300):
            f,a,ef,_,ns,f0,f1=TT.data[ds.keys[i]]; T=f.shape[0]
            idx=np.linspace(0,T-1,TT.NF).round().astype(int)
            fx=f[idx].astype(np.float32).copy()
            if mask_slice: fx[:,mask_slice[0]:mask_slice[1]]=0
            x=np.concatenate([fx,a[idx].astype(np.float32),(idx/max(T-1,1))[:,None].astype(np.float32)],1)
            mu,_=net(torch.from_numpy(x)[None],torch.tensor([ns],dtype=torch.float32))
            out[i]=float(mu)
    return out
base=preds()
efs=np.array([TT.data[ds.keys[i]][2] for i in range(300)])
print("  baseline MAE (300 vids): %.3f"%np.abs(base-efs).mean())
for name,sl in [('deep',(0,2048)),('mid',(2048,2304)),('radial',(2304,2336)),('flow',(2336,2340))]:
    p=preds(sl)
    print("  knockout %-7s mean|dPred| %6.3f EF-pts   MAE -> %.3f"%(name,np.abs(p-base).mean(),np.abs(p-efs).mean()))
