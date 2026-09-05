# Case 2: Two ground truths that disagree

CAMUS ships expert masks and a stated EF for every patient. EF computed from the masks came out about 8 points higher than the stated EF. Whose fault?

## My code first

The first hypothesis was that my disk summation geometry was off: a rotation step, an erosion asymmetry, the annulus band heuristic. Any of those could bias a volume. I wrote the own code suspects down before I let myself think about the dataset. <!-- src: AUDIT_2026-08-24.md lines 54-58, 124-130 -->

Two things killed them. The same Simpson implementation had just reproduced EchoNet's stated EF to 1.05 points on 10,024 videos (Case 1), and an 8 point arithmetic bug would have shown up there. And I built a second, geometrically independent estimator: area length (V = 8A² / 3πL, L from the PCA extent of the mask), which uses no rotation, no disks and no annulus band. The fork was written into the script before it ran (`scripts/_arealen.py`): if area length agreed with the stated EF, my disk geometry was the bug. If it also read 8 high, the gap was in the dataset. <!-- src: _arealen.py docstring -->

Disk Simpson +8.45, area length +7.67. Two independent methods within 0.8 points of each other, corr about 0.97, and flat across CAMUS's image quality strata. Gap is in the data. <!-- src: FINDINGS.md sec 1 lines 17-24; AUDIT_2026-08-24.md RESOLUTION -->

## What it means

The stated EF wasn't derived from the shipped contours. Both are legitimate measurements of the same patient by different procedures, and they're different targets. Any CAMUS EF result has to say which one it's scored against, because "MAE 3.9" and "MAE 8.6" for the same model is entirely a choice of target. <!-- src: FINDINGS.md sec 4 -->

A trained model reproduced it later: on the CAMUS test split (n=52), biplane EF from predicted masks scored 3.94 against mask derived EF and 8.56 with bias +8.08 against stated EF. The spatial ensemble did it again (4.043 vs 7.689). Four independent reads of the same offset. <!-- src: FINDINGS.md sec 4 lines 64-72; camuseval_sp.log -->

## The number I nearly published

Early on I had "CAMUS MAE 3.08, beats state of the art." It was circular. The ground truth masks I was scoring against are the same contours the EF was computed from, so the test couldn't tell a model from a copy of the label. Retracted the same session, kept in the log with the reason. <!-- src: AUDIT_2026-08-24.md lines 44-50 -->

## What I'd do again

When two ground truths disagree, don't pick one. Rule out your own code with an independent estimator, then report every number against both targets and say which. And when something looks like it beats the field, check whether the ruler is made out of the label.

Scripts: `simpson.py`, `_arealen.py`, `ef_readout.py`.
