"""What does a Dice of 0.9376 actually MEAN? Convert it to millimetres of boundary error.

Dice is dominated by the BOUNDARY -- the interior always matches. So for a fixed boundary
error, Dice depends on the shape's perimeter-to-area ratio. A thin ring (myocardium) is
penalised far harder than a solid blob (LV cavity, LA) for the SAME tracing accuracy.

For a region of area A and perimeter P, a boundary displaced outward by d gives roughly:
    Dice ~ 2A / (2A + P*d)   ->   d ~ (2A/P) * (1/Dice - 1)
We measure A and P from the actual ground-truth masks, with real pixel spacing.
"""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import os, numpy as np, nibabel as nib, cv2
D=CAMUS_ROOT
names={1:'LV cavity',2:'myocardium',3:'left atrium'}
DICE={1:0.9376,2:0.8764,3:0.9204}       # our measured CAMUS test Dice
acc={c:{'A':[],'P':[],'sp':[]} for c in names}
for p in sorted(x for x in os.listdir(D) if x.startswith('patient')):
    for ph in ['ED','ES']:
        f=os.path.join(D,p,'%s_4CH_%s_gt.nii.gz'%(p,ph))
        im=nib.load(f); gt=im.get_fdata().astype(np.uint8); sp=float(im.header.get_zooms()[0])
        for c in names:
            m=(gt==c).astype(np.uint8)
            if m.sum()<50: continue
            cnts,_=cv2.findContours(m,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
            P=sum(cv2.arcLength(k,True) for k in cnts)
            acc[c]['A'].append(m.sum()); acc[c]['P'].append(P); acc[c]['sp'].append(sp)
print("CAMUS ground truth geometry (pixel spacing median %.3f mm)\n"%np.median(acc[1]['sp']))
print("%-14s %10s %10s %10s   %8s   %12s"%("class","area px","perim px","2A/P px","Dice","boundary err"))
for c in names:
    A=np.median(acc[c]['A']); P=np.median(acc[c]['P']); sp=np.median(acc[c]['sp'])
    t=2*A/P
    d_px=t*(1.0/DICE[c]-1.0); d_mm=d_px*sp
    print("%-14s %10.0f %10.0f %10.1f   %8.4f   %6.2f px = %.2f mm"%(names[c],A,P,t,DICE[c],d_px,d_mm))
print()
print("=== the point: same boundary error, different Dice ===")
print("If EVERY class were traced to exactly 1.0 mm accuracy, Dice would read:")
for c in names:
    A=np.median(acc[c]['A']); P=np.median(acc[c]['P']); sp=np.median(acc[c]['sp'])
    d_px=1.0/sp; t=2*A/P
    print("   %-14s Dice %.4f   (thickness-limited: 2A/P = %.1f px = %.1f mm)"%(names[c],t/(t+d_px),t,t*sp))
print()
print("=== why ES scores lower than ED ===")
for ph in ['ED','ES']:
    A=[];P=[]
    for p in sorted(x for x in os.listdir(D) if x.startswith('patient')):
        f=os.path.join(D,p,'%s_4CH_%s_gt.nii.gz'%(p,ph))
        gt=nib.load(f).get_fdata().astype(np.uint8); m=(gt==1).astype(np.uint8)
        if m.sum()<50: continue
        cnts,_=cv2.findContours(m,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        A.append(m.sum()); P.append(sum(cv2.arcLength(k,True) for k in cnts))
    A=np.median(A);P=np.median(P); t=2*A/P; sp=np.median(acc[1]['sp'])
    d_px=1.0/sp
    print("   LV at %s: area %5.0f px  2A/P %5.1f px  -> Dice at 1mm accuracy = %.4f"%(ph,A,t,t/(t+d_px)))
