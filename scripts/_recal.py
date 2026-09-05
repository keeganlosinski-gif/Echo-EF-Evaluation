"""VAL-fitted monotone recalibration of the spatial ensemble (held-out recalibration layer).
Fit isotonic mu->ef on VAL; apply once to TEST. No TEST information enters the fit."""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import numpy as np, torch, os, sys
os.environ['TOK_DIR']=os.path.join(WORK, 'tokens_sp'); os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_tok as TT
from sklearn.isotonic import IsotonicRegression
nets=[]
for tag in ['sp','sp_s1','sp_s2']:
    n=TT.TokAttn(); n.load_state_dict(torch.load(os.path.join(WORK, 'tok_%s.pt')%tag,map_location='cpu')); n.eval(); nets.append(n)
def preds(split):
    ds=TT.DS(split,False)
    mus=np.zeros(len(ds)); efs=np.zeros(len(ds))
    with torch.no_grad():
        for i in range(len(ds)):
            f,a,ef,_,ns,f0,f1=TT.data[ds.keys[i]]; T=f.shape[0]
            pr=[]
            for off in (-1,0,1):
                idx=np.clip(np.linspace(0,T-1,TT.NF).round().astype(int)+off,0,T-1)
                x=np.concatenate([f[idx].astype(np.float32),a[idx].astype(np.float32),
                                  (idx/max(T-1,1))[:,None].astype(np.float32)],1)
                for n_ in nets:
                    mu,_=n_(torch.from_numpy(x)[None],torch.tensor([ns],dtype=torch.float32)); pr.append(float(mu))
            mus[i]=np.mean(pr); efs[i]=ef
    return mus,efs
vm,ve=preds('VAL')
print("VAL raw MAE %.3f"%np.abs(vm-ve).mean())
iso=IsotonicRegression(out_of_bounds='clip').fit(vm,ve)
print("VAL recal MAE %.3f (in-sample on VAL, sanity only)"%np.abs(iso.predict(vm)-ve).mean())
z=np.load(os.path.join(WORK, 'sp_ens_test_preds.npz'))
tm,te=z['mu'],z['ef']
tc=iso.predict(tm)
ae0,ae1=np.abs(tm-te),np.abs(tc-te)
print("\nTEST raw   MAE %.3f  corr %.3f"%(ae0.mean(),np.corrcoef(tm,te)[0,1]))
print("TEST recal MAE %.3f  corr %.3f"%(ae1.mean(),np.corrcoef(tc,te)[0,1]))
print("\nband audit after recal:")
for lo,hi in [(0,35),(35,45),(45,55),(55,65),(65,101)]:
    m=(te>=lo)&(te<hi)
    print("  EF %3d-%3d: MAE %.2f->%.2f  bias %+.2f->%+.2f"%(lo,hi,ae0[m].mean(),ae1[m].mean(),(tm-te)[m].mean(),(tc-te)[m].mean()))
def cat(v): return np.digitize(v,[35,53])
print("3-class agreement %.1f%% -> %.1f%%"%(100*(cat(tm)==cat(te)).mean(),100*(cat(tc)==cat(te)).mean()))
np.savez(os.path.join(WORK, 'sp_ens_test_recal.npz'),mu=tc,ef=te)
