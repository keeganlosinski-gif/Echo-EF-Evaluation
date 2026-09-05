import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import numpy as np, torch, os, sys
os.environ['TOK_DIR']=os.path.join(WORK, 'tokens_v4'); os.environ['TOK_STD']='1'; os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_tok as TT
net=TT.TokAttn(); net.load_state_dict(torch.load(os.path.join(WORK, 'tok_v4std.pt'),map_location='cpu')); net.eval()
ds=TT.DS('TEST',False)
def preds(sl=None):
    out=np.zeros(300)
    with torch.no_grad():
        for i in range(300):
            f,a,ef,_,ns,f0,f1=TT.data[ds.keys[i]]; T=f.shape[0]
            idx=np.linspace(0,T-1,TT.NF).round().astype(int)
            fx=(f[idx].astype(np.float32)-TT.MU)/TT.SDV
            if sl: fx[:,sl[0]:sl[1]]=0
            av=(a[idx].astype(np.float32)-TT.AMU)/TT.ASD
            x=np.concatenate([fx,av,(idx/max(T-1,1))[:,None].astype(np.float32)],1)
            mu,_=net(torch.from_numpy(x)[None],torch.tensor([ns],dtype=torch.float32))
            out[i]=float(mu)
    return out
base=preds(); efs=np.array([TT.data[ds.keys[i]][2] for i in range(300)])
print("v4std baseline MAE(300): %.3f"%np.abs(base-efs).mean())
for name,sl in [('deep',(0,2048)),('mid',(2048,2304)),('radial',(2304,2336)),('flow',(2336,2340))]:
    p=preds(sl)
    print("  knockout %-7s mean|dPred| %6.3f   MAE -> %.3f"%(name,np.abs(p-base).mean(),np.abs(p-efs).mean()))
