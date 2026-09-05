"""PREREG discipline: LDS judged on EF bands + decision rates, not pooled MAE. CPU."""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import numpy as np, torch, os, sys
os.environ['TOK_DIR']=os.path.join(WORK, 'tokens_v4'); os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_tok as TT
def ens(tags):
    nets=[]
    for t in tags:
        n=TT.TokAttn(); n.load_state_dict(torch.load(os.path.join(WORK, 'tok_%s.pt')%t,map_location='cpu')); n.eval(); nets.append(n)
    ds=TT.DS('TEST',False); mus=np.zeros(len(ds)); efs=np.zeros(len(ds))
    with torch.no_grad():
        for i in range(len(ds)):
            f,a,ef,_,ns,f0,f1=TT.data[ds.keys[i]]; T=f.shape[0]; pr=[]
            idx=np.linspace(0,T-1,TT.NF).round().astype(int)
            x=np.concatenate([f[idx].astype(np.float32),a[idx].astype(np.float32),(idx/max(T-1,1))[:,None].astype(np.float32)],1)
            xt=torch.from_numpy(x)[None]; nst=torch.tensor([ns],dtype=torch.float32)
            for n_ in nets: mu,_=n_(xt,nst); pr.append(float(mu))
            mus[i]=np.mean(pr); efs[i]=ef
    return mus,efs
for name,tags in [('v4   ',['v4','v4_s1','v4_s2']),('v4lds',['v4lds','v4lds_s1','v4lds_s2'])]:
    mu,ef=ens(tags); ae=np.abs(mu-ef)
    print("%s pooled %.3f |"%(name,ae.mean()),end='')
    for lo,hi in [(0,35),(35,45),(45,55),(55,101)]:
        m=(ef>=lo)&(ef<hi)
        print(" %d-%d: MAE %.2f bias %+.2f |"%(lo,hi,ae[m].mean(),(mu-ef)[m].mean()),end='')
    ct=np.digitize(ef,[35,53]); cp=np.digitize(mu,[35,53])
    print(" 3cls %.1f%% | mid-band correct %.1f%% | EF<35 correct %.1f%%"%(
        100*(ct==cp).mean(),100*(cp[ct==1]==1).mean(),100*(cp[ct==0]==0).mean()))
