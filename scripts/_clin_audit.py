"""Structural audits on the spatial ensemble (CPU only):
EF-band structural exclusion, clinical decision-flips (35/53), sigma reliability."""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import numpy as np, torch, os, sys
os.environ['TOK_DIR']=os.path.join(WORK, 'tokens_sp'); os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_tok as TT
nets=[]
for tag in ['sp','sp_s1','sp_s2']:
    n=TT.TokAttn(); n.load_state_dict(torch.load(os.path.join(WORK, 'tok_%s.pt')%tag,map_location='cpu')); n.eval(); nets.append(n)
ds=TT.DS('TEST',False)
mus=np.zeros(len(ds)); efs=np.zeros(len(ds)); sgs=np.zeros(len(ds))
with torch.no_grad():
    for i in range(len(ds)):
        f,a,ef,_,ns,f0,f1=TT.data[ds.keys[i]]; T=f.shape[0]
        pr=[]; sg=[]
        for off in (-1,0,1):
            idx=np.clip(np.linspace(0,T-1,TT.NF).round().astype(int)+off,0,T-1)
            x=np.concatenate([f[idx].astype(np.float32),a[idx].astype(np.float32),
                              (idx/max(T-1,1))[:,None].astype(np.float32)],1)
            xt=torch.from_numpy(x)[None]; nst=torch.tensor([ns],dtype=torch.float32)
            for n_ in nets:
                mu,ls=n_(xt,nst); pr.append(float(mu)); sg.append(float(ls.exp()*10))
        mus[i]=np.mean(pr); efs[i]=ef; sgs[i]=np.mean(sg)
np.savez(os.path.join(WORK, 'sp_ens_test_preds.npz'),mu=mus,ef=efs,sg=sgs,keys=np.array(ds.keys))
ae=np.abs(mus-efs)
print("pooled MAE %.3f (should match 4.733)"%ae.mean())
print("\n== EF-BAND STRUCTURAL AUDIT (does pooled MAE hide band structure?) ==")
for lo,hi in [(0,35),(35,45),(45,55),(55,65),(65,101)]:
    m=(efs>=lo)&(efs<hi)
    if m.sum()==0: continue
    print("  EF %3d-%3d: n=%4d  MAE %.2f  bias %+.2f"%(lo,hi,m.sum(),ae[m].mean(),(mus-efs)[m].mean()))
print("\n== CLINICAL DECISION AUDIT (grade the decision, not the proxy) ==")
def cat(v): return np.digitize(v,[35,53])   # 0:<35 ICD-territory, 1:35-53 reduced, 2:>53 normal
ct=cat(efs); cp=cat(mus)
agree=(ct==cp).mean()
print("  3-class agreement %.1f%%"%(100*agree))
for c,name in [(0,'EF<35'),(1,'35-53'),(2,'>53')]:
    m=ct==c
    print("  true %-6s n=%4d  correctly categorized %.1f%%"%(name,m.sum(),100*(cp[m]==c).mean()))
sev=((ct==0)&(cp==2))|((ct==2)&(cp==0))
print("  severe flips (<35 <-> >53): %d (%.2f%%)"%(sev.sum(),100*sev.mean()))
print("\n== SIGMA RELIABILITY (the recal-layer question) ==")
q=np.quantile(sgs,np.linspace(0,1,6))
for i in range(5):
    m=(sgs>=q[i])&(sgs<=q[i+1])
    print("  sigma decile %d: pred sigma %.2f  vs  empirical MAE %.2f  (|err|>2sigma: %.1f%%)"%(
        i,sgs[m].mean(),ae[m].mean(),100*(ae[m]>2*sgs[m]).mean()))
