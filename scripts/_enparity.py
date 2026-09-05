"""DECISIVE TEST of my Simpson implementation.

EchoNet's ESV/EDV were derived FROM the VolumeTracings chords. So recomputing volume from
those chords must reproduce the stated volumes. If it does, my method-of-disks arithmetic is
correct and the +8.5pp CAMUS gap is a property of CAMUS, not my code. If it fails here too,
my code is wrong and the CAMUS result means nothing.

EchoNet is A4C only -> MONOPLANE Simpson: each disk is a circle, so a=b=d:
    V = (pi/4) * h * SUM(d_i^2)
"""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import pandas as pd, numpy as np
P=ECHONET_ROOT + '/'
fl=pd.read_csv(P+'FileList.csv'); vt=pd.read_csv(P+'VolumeTracings.csv')
fl['key']=fl.FileName.str.replace('.avi','',regex=False)
vt['key']=vt.FileName.str.replace('.avi','',regex=False)

def vol_from_chords(g):
    """First row is the long axis; the rest are perpendicular disk chords."""
    a=g.iloc[0]
    L=np.hypot(a.X2-a.X1, a.Y2-a.Y1)
    d=np.hypot(g.X2.values[1:]-g.X1.values[1:], g.Y2.values[1:]-g.Y1.values[1:])
    n=len(d)
    if n<5: return np.nan
    h=L/n
    return (np.pi/4.0)*h*float(np.sum(d**2))

rows=[]
for key,g in vt.groupby('key'):
    frames=sorted(g.Frame.unique())
    if len(frames)!=2: continue
    v=[vol_from_chords(g[g.Frame==f].reset_index(drop=True)) for f in frames]
    if not all(np.isfinite(v)): continue
    rows.append((key, max(v), min(v)))          # EDV = larger, ESV = smaller
df=pd.DataFrame(rows, columns=['key','edv_c','esv_c']).merge(fl[['key','EF','ESV','EDV']], on='key')
df['ef_c']=(df.edv_c-df.esv_c)/df.edv_c*100

print("videos compared:", len(df))
print()
print("=== VOLUMES: computed-from-chords vs stated (raw pixel^3 units, scale unknown) ===")
for a,b in [('edv_c','EDV'),('esv_c','ESV')]:
    r=np.corrcoef(df[a],df[b])[0,1]
    ratio=(df[a]/df[b]).median()
    print("  %-6s corr %.5f   median ratio computed/stated = %.4f" % (b, r, ratio))
print()
print("=== EF (scale-invariant, so this is the real test) ===")
d=df.ef_c-df.EF
print("  stated EF   mean %.2f" % df.EF.mean())
print("  computed EF mean %.2f" % df.ef_c.mean())
print("  bias  %+.3f pp    MAE %.3f pp    sd %.3f" % (d.mean(), d.abs().mean(), d.std()))
print("  correlation %.5f" % np.corrcoef(df.EF,df.ef_c)[0,1])
print("  within 1pp: %.1f%%   within 2pp: %.1f%%" % (100*(d.abs()<=1).mean(), 100*(d.abs()<=2).mean()))
print()
print("VERDICT: if bias ~0 and MAE <1pp, the Simpson arithmetic is CORRECT")
print("         and the CAMUS +8.5pp is a CAMUS property, not my bug.")
