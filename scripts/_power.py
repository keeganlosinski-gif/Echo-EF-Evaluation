import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import numpy as np
from math import sqrt

def hanley_se(auc, n1, n0):
    """Hanley-McNeil SE of AUC. n1=positives, n0=negatives."""
    q1 = auc/(2-auc); q2 = 2*auc*auc/(1+auc)
    v = (auc*(1-auc) + (n1-1)*(q1-auc**2) + (n0-1)*(q2-auc**2))/(n1*n0)
    return sqrt(v)

N = 84; prev = 0.654
n1 = round(N*prev); n0 = N-n1
print("EXTERNAL CHALLENGE TEST SET: n=%d  positives=%d  negatives=%d  (prev %.1f%%)" % (N,n1,n0,prev*100))
print()
se0 = hanley_se(0.5, n1, n0)
print("SE(AUC) at the null (AUC=0.5): %.4f" % se0)
print("  -> 95%% CI half-width = %.3f  => any point estimate below %.3f cannot clear 0.5" % (1.96*se0, 0.5+1.96*se0))
print()
print("Organisers' reported baseline: AUC 0.541, CI 0.390-0.692  (half-width %.3f)" % ((0.692-0.390)/2))
print()
print("MINIMUM DETECTABLE AUC (80%% power, two-sided a=0.05):")
for target in [0.60,0.65,0.70,0.75,0.80]:
    se = hanley_se(target, n1, n0)
    z = (target-0.5)/se
    # power = P(reject | true auc = target)
    from math import erf
    Phi = lambda x: 0.5*(1+erf(x/sqrt(2)))
    power = 1-Phi(1.96-z)+Phi(-1.96-z)
    print("  true AUC %.2f -> SE %.4f, z=%.2f, POWER = %.0f%%" % (target,se,z,power*100))
print()
# how big must a true effect be for 80% power
from scipy.optimize import brentq
def pw(a):
    from math import erf
    Phi = lambda x: 0.5*(1+erf(x/sqrt(2)))
    se=hanley_se(a,n1,n0); z=(a-0.5)/se
    return 1-Phi(1.96-z)+Phi(-1.96-z)-0.80
try:
    a80 = brentq(pw, 0.51, 0.99)
    print("=> Need a TRUE AUC of %.3f for 80%% power at n=84." % a80)
except Exception as e:
    print("solve failed:", e)
print()
print("SAMPLE SIZE needed to detect a true AUC of 0.65 at 80%% power:")
for n in [84,150,250,400,600,1000]:
    p1=round(n*prev); p0=n-p1
    se=hanley_se(0.65,p1,p0); z=(0.65-0.5)/se
    from math import erf
    Phi = lambda x: 0.5*(1+erf(x/sqrt(2)))
    power=1-Phi(1.96-z)+Phi(-1.96-z)
    print("  n=%4d -> power %.0f%%" % (n,power*100))
