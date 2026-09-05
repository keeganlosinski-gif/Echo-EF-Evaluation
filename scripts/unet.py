"""U-Net with a timm encoder.

The CNN's job here is narrow and deliberate: DENSE SPATIAL LABELLING. It is a filter, not the
model. Convolutions are used because local spatial priors are exactly what they are good at --
and because we measured that this task is irreducibly spatial (brightness alone gives corr
+0.014 on the whole sector, vs +0.942 once the LV is isolated).

nnU-Net tops the CAMUS leaderboard with a PLAIN U-Net, winning on training recipe rather than
architecture, so nothing exotic is warranted here.
"""
import os  # all paths come from the environment; no dataset content lives in this repository
ECHONET_ROOT = os.environ.get('ECHONET_ROOT', '')   # EchoNet-Dynamic root: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT = os.environ.get('CAMUS_ROOT', '')       # CAMUS root: database_nifti/
WORK = os.environ.get('WORK_DIR', '.')              # scratch directory for caches, checkpoints and outputs
import torch, torch.nn as nn, torch.nn.functional as F, timm


class ConvBlock(nn.Module):
    def __init__(s, cin, cout):
        super().__init__()
        s.b = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    def forward(s, x): return s.b(x)


class UNet(nn.Module):
    def __init__(s, ncls=4, enc='resnet34', pretrained=True, cin=1):
        super().__init__()
        s.enc = timm.create_model(enc, pretrained=pretrained, features_only=True, in_chans=cin)
        ch = s.enc.feature_info.channels()                      # e.g. [64,64,128,256,512]
        s.ch = ch
        dec = []
        rev = ch[::-1]
        for i in range(len(rev) - 1):
            dec.append(ConvBlock(rev[i] + rev[i + 1], rev[i + 1]))
        s.dec = nn.ModuleList(dec)
        s.head = nn.Conv2d(rev[-1], ncls, 1)

    def forward(s, x):
        H, W = x.shape[-2:]
        f = s.enc(x)                                            # coarse..fine reversed below
        f = f[::-1]
        y = f[0]
        for i, d in enumerate(s.dec):
            y = F.interpolate(y, size=f[i + 1].shape[-2:], mode='nearest')
            y = d(torch.cat([y, f[i + 1]], 1))
        y = s.head(y)
        return F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)


def dice_loss(logits, target, ncls=4, eps=1.0, ignore_bg=True):
    p = torch.softmax(logits, 1)
    t = F.one_hot(target, ncls).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (p * t).sum(dims); den = p.sum(dims) + t.sum(dims)
    d = 1 - (2 * inter + eps) / (den + eps)
    return d[1:].mean() if ignore_bg else d.mean()


@torch.no_grad()
def dice_score(pred, target, ncls=4, eps=1e-6):
    """Per-class Dice on hard predictions. Returns [ncls] with nan where the class is absent."""
    out = []
    for c in range(ncls):
        pc = (pred == c); tc = (target == c)
        den = pc.sum() + tc.sum()
        out.append(float('nan') if tc.sum() == 0 else (2.0 * (pc & tc).sum() / (den + eps)).item())
    return out
