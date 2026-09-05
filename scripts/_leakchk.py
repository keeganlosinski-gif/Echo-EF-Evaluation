"""Can a patient appear in both TRAIN and TEST? That is the only leak that would matter."""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import pandas as pd, numpy as np
fl=pd.read_csv(os.path.join(ECHONET_ROOT, 'FileList.csv'))
print("columns:", list(fl.columns))
print("-> is there a patient identifier?", "YES" if any('patient' in c.lower() or 'mrn' in c.lower() or 'subject' in c.lower() for c in fl.columns) else "NO")
print()
print("unique FileName:", fl.FileName.nunique(), "of", len(fl))
# exact-duplicate label tuples would hint at the same study appearing twice
dup=fl.duplicated(subset=['EF','ESV','EDV','NumberOfFrames','FPS'], keep=False)
print("rows sharing an identical (EF,ESV,EDV,NumberOfFrames,FPS) tuple:", dup.sum())
if dup.sum():
    d=fl[dup].sort_values(['EF','ESV'])
    cross=d.groupby(['EF','ESV','EDV']).Split.nunique()
    print("  ...of those, groups spanning MORE THAN ONE split:", (cross>1).sum())
    print(d.head(6)[['FileName','EF','ESV','EDV','NumberOfFrames','Split']].to_string(index=False))
print()
print("split sizes:", dict(fl.Split.value_counts()))
for s,g in fl.groupby('Split'):
    print("  %-6s n=%4d  EF mean %.1f  %%EF<50 %.1f" % (s,len(g),g.EF.mean(),100*(g.EF<50).mean()))
