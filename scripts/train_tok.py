"""STAGE 4 baseline: temporal attention over frozen v3 tokens -> heteroscedastic (EF, sigma).
Labels are TEMPORAL objects (FINDINGS 8): this model accesses the label's information channel.
Targets: beat pooled 7.18 / crisp 5.5; read-out floor is 1.71 on the quad-fill masks (2.00 on the original masks). Deterministic; minutes per run."""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import numpy as np, pandas as pd, glob, os, time, math, random
import torch, torch.nn as nn
_S=int(os.environ.get('TOK_SEED','0'))
torch.manual_seed(_S); np.random.seed(_S); random.seed(_S)
torch.backends.cudnn.deterministic=True
import torch as _t0
DEV='cuda' if (_t0.cuda.is_available() and os.environ.get('TOK_CPU','0')!='1') else 'cpu'
NF   = int(os.environ.get('TOK_NF','64'))      # frames sampled per video
DD   = int(os.environ.get('TOK_DD','192'))
NL   = int(os.environ.get('TOK_NL','4'))
EP   = int(os.environ.get('TOK_EP','12'))
BS   = int(os.environ.get('TOK_BS','64'))
LR   = float(os.environ.get('TOK_LR','3e-4'))
TAG  = os.environ.get('TOK_TAG','base')
NOAUX= os.environ.get('TOK_NOAUX','0')=='1'   # ablation: zero the 7 aux features
NONS = os.environ.get('TOK_NONS','0')=='1'    # ablation: zero the noise covariate
LAP  = os.environ.get('TOK_LAP','0')=='1'     # Laplace NLL (L1-natured) instead of Gaussian
TOKDIR=os.environ.get('TOK_DIR',os.path.join(WORK, 'tokens'))
DENSE= os.environ.get('TOK_DENSE','0')=='1'   # dense clip (stride 2) instead of whole-loop comb
LDS  = os.environ.get('TOK_LDS','0')=='1'     # inverse-EF-density loss weighting (band-bias fix)
STD  = os.environ.get('TOK_STD','0')=='1'     # per-dim z-score feats+aux from TRAIN stats (scale fix)
SEED = int(os.environ.get('TOK_SEED','0'))

print('loading token cache into RAM...', flush=True)
t0=time.time(); data={}
for p in sorted(glob.glob(TOKDIR+'/*.npz')):
    z=np.load(p)
    data[p.split('/')[-1][:-4]]=(z['feats'],z['aux'].astype(np.float16),float(z['ef']),
                                 str(z['split']),float(z['ns']),int(z['f0']),int(z['f1']))
print('loaded %d videos in %.0fs'%(len(data),time.time()-t0), flush=True)

FD=next(iter(data.values()))[0].shape[1]      # feature dim (512 pooled / 2048 spatial)
print('feature dim %d'%FD,flush=True)
MU=None; SDV=None; AMU=None; ASD=None
if STD:
    _fs=[]; _as=[]
    for k,v in list(data.items())[::7]:
        if v[3]=='TRAIN': _fs.append(v[0][::5].astype(np.float32)); _as.append(v[1][::5].astype(np.float32))
    _F=np.concatenate(_fs,0); _A=np.concatenate(_as,0)
    MU,SDV=_F.mean(0),np.maximum(_F.std(0),1e-4)
    AMU,ASD=_A.mean(0),np.maximum(_A.std(0),1e-4)
    print('STD on: feat/aux z-scored from TRAIN stats',flush=True)
WMAP=None
if LDS:
    _efs=np.array([v[2] for v in data.values() if v[3]=='TRAIN'])
    _h,_=np.histogram(_efs,bins=20,range=(0,100))
    WMAP=np.clip((1.0/np.maximum(_h,3))**0.5,None,None)
    WMAP=np.clip(WMAP/ (WMAP[np.clip((_efs//5).astype(int),0,19)].mean()),0.2,5.0)
    print('LDS weights per 5pp band:',np.round(WMAP,2),flush=True)
def sample_idx(T,train=False):
    if DENSE:
        L=NF*2
        if T<=L: return np.clip(np.arange(NF)*2,0,T-1)
        st=np.random.randint(0,T-L+1) if train else (T-L)//2
        return st+np.arange(NF)*2
    return np.linspace(0,T-1,NF).round().astype(int)

class DS(torch.utils.data.Dataset):
    def __init__(self,split,train):
        self.keys=[k for k,v in data.items() if v[3]==split]; self.train=train
    def __len__(self): return len(self.keys)
    def __getitem__(self,i):
        f,a,ef,_,ns,f0,f1=data[self.keys[i]]
        T=f.shape[0]; idx=sample_idx(T,self.train)
        if self.train and T>NF and not DENSE:         # temporal jitter: shift the comb
            off=np.random.randint(-1,2); idx=np.clip(idx+off,0,T-1)
        av=a[idx].astype(np.float32)
        fv=f[idx].astype(np.float32)
        if STD: fv=(fv-MU)/SDV; av=(av-AMU)/ASD
        if NOAUX: av=av*0
        x=np.concatenate([fv,av,
                          (idx/max(T-1,1))[:,None].astype(np.float32)],1)   # 512+7+1
        w=1.0 if WMAP is None else float(WMAP[min(int(ef//5),19)])
        return x, np.float32(ef), np.float32(0.0 if NONS else ns), np.float32(w)

class TokAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp=nn.Linear(FD+8,DD)
        enc=nn.TransformerEncoderLayer(DD,8,DD*4,dropout=0.1,batch_first=True,norm_first=True)
        self.tr=nn.TransformerEncoder(enc,NL)
        self.q=nn.Parameter(torch.randn(1,1,DD)*0.02)
        self.pool=nn.MultiheadAttention(DD,8,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(DD+1),nn.Linear(DD+1,DD),nn.GELU(),nn.Linear(DD,2))
    def forward(self,x,ns):
        h=self.tr(self.inp(x))
        p,_=self.pool(self.q.expand(x.shape[0],-1,-1),h,h)
        z=torch.cat([p[:,0],ns[:,None]],1)
        o=self.head(z)
        return o[:,0]*10+55, o[:,1].clamp(-3,3)      # mu (EF units), log sigma (scaled units)

def run():
    dtr=torch.utils.data.DataLoader(DS('TRAIN',True),batch_size=BS,shuffle=True,num_workers=0,drop_last=True)
    dva=torch.utils.data.DataLoader(DS('VAL',False),batch_size=256,num_workers=0)
    dte=torch.utils.data.DataLoader(DS('TEST',False),batch_size=256,num_workers=0)
    net=TokAttn().to(DEV)
    print('%.2fM params | %d train vids'%(sum(p.numel() for p in net.parameters())/1e6,len(dtr.dataset)),flush=True)
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=1e-2)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,total_steps=EP*len(dtr),pct_start=0.1)
    best=1e9
    for ep in range(EP):
        net.train()
        for x,ef,ns,w in dtr:
            x,ef,ns,w=x.to(DEV),ef.to(DEV),ns.to(DEV),w.to(DEV)
            mu,ls=net(x,ns)
            z=(ef-mu)/10.0                            # NLL in scaled units
            if LAP: loss=((z.abs()/ls.exp()+ls)*w).mean()
            else:   loss=((0.5*(z/ls.exp())**2+ls)*w).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        net.eval(); errs=[]
        with torch.no_grad():
            for x,ef,ns,_w in dva:
                mu,_=net(x.to(DEV),ns.to(DEV))
                errs.append((mu.cpu()-ef).abs())
        mae=torch.cat(errs).mean().item()
        print('EP%d VAL MAE %.3f'%(ep,mae),flush=True)
        if mae<best: best=mae; torch.save(net.state_dict(),os.path.join(WORK, 'tok_%s.pt')%TAG)
    # honest TEST read, once, best checkpoint, noise-stratified
    net.load_state_dict(torch.load(os.path.join(WORK, 'tok_%s.pt')%TAG)); net.eval()
    mus=[]; efs=[]; nss=[]; sgs=[]
    with torch.no_grad():
        for x,ef,ns,_w in dte:
            mu,ls=net(x.to(DEV),ns.to(DEV))
            mus.append(mu.cpu()); efs.append(ef); nss.append(ns); sgs.append(ls.exp().cpu()*10)
    mu=torch.cat(mus); ef=torch.cat(efs); ns=torch.cat(nss); sg=torch.cat(sgs)
    ae=(mu-ef).abs()
    print('TEST MAE %.3f  corr %.3f  n=%d'%(ae.mean(),np.corrcoef(mu,ef)[0,1],len(ae)),flush=True)
    q=torch.quantile(ns,torch.tensor([0.25,0.5,0.75]))
    for i,(lo,hi) in enumerate([(-9,q[0]),(q[0],q[1]),(q[1],q[2]),(q[2],9e9)]):
        m=(ns>=lo)&(ns<hi)
        print('  noiseQ%d MAE %.3f  blowup>15 %.1f%%  n=%d'%(i,ae[m].mean(),(ae[m]>15).float().mean()*100,m.sum()),flush=True)
    # sigma sanity: does predicted sigma rank error?
    k=sg.argsort()
    lo_s,hi_s=ae[k[:len(k)//4]].mean(),ae[k[-len(k)//4:]].mean()
    print('  sigma-head: MAE crispest-quartile %.3f vs mushiest %.3f'%(lo_s,hi_s),flush=True)
if __name__=='__main__': run()
