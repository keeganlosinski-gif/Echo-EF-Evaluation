# Case 6: Internal vs external, one model per row

The published EchoNet-Dynamic model reported MAE 4.05 on its own test set and 6.0 on an external site, and the field mostly stopped reporting the external number after that. What happens to mine off site, and how do I report it without fooling myself? <!-- src: FINDINGS.md intro lines 3-7; AUDIT_2026-08-24.md lines 133-134 -->

## The rule I set

Every model gets two numbers in the same row, internal (EchoNet verified test) and external (CAMUS test, a different hospital, vendor and acquisition protocol). On CAMUS every EF number is graded against both of its targets (Case 2) and the table says which. Nothing gets reported without the log file it came from. <!-- src: _camuseval_tok.py docstring; PROJECT_LOG.md sec 2 -->

## The table

| Model | Internal | External (CAMUS n=52, vs mask derived EF) | Change |
|---|---|---|---|
| Published EchoNet-Dynamic model (Ouyang et al. 2020), as reported | 4.05 (n=1,276) | 6.0 (their external set) | +48% |
| Spatial ensemble, 3 seeds x 3 TTA | 4.733 (n=1,219) | 4.043, corr 0.914, bias -0.67 | none (external lower, -15%, n=52 so wide interval) |
| Standardised feature ensemble | 4.488 (n=1,219) | 6.909 | +54% |

<!-- src: PROJECT_LOG.md sec 2A corrected table lines 44-58; ens_sp.log; camuseval_sp.log; ens_v4std.log; camuseval_v4std.log -->

The features that bought internal accuracy (the standardised descriptors from Case 5) were the least transferable, and the plainer model held its accuracy off site. Naive fine tuning of the encoder on EF went further the wrong way, 4.637 internal and a CAMUS collapse to 11.225 with bias -10.39. The encoder had become vendor specific. <!-- src: FINDINGS.md sec 14 lines 295-302; WRITEUP_DRAFT.md sec 2 -->

## Caveats that travel with the table

- The external sets are different (CAMUS here, a different site in the published work), so this isn't a head to head on identical data. The defensible claim is about degradation under distribution shift, none for the spatial ensemble vs +48 percent for the published model, not a direct MAE comparison. I don't write "beats the published model externally." <!-- src: WRITEUP_DRAFT.md lines 74-79 -->
- Internal n differs. My temporal model numbers are on the 1,219 verified test videos (57 official test videos excluded by the gates for defective ground truth). The published number is on all 1,276 and so is my own frame based baseline (7.18). <!-- src: _qc_masks.py; RESULTS_0824.log eval note -->
- CAMUS n=52 is small. The external column has a wide interval and I haven't pretended otherwise.

## The receipt defect I caught in my own draft

An earlier version of the writeup put 4.49 internal and 4.04 external in one row. Those come from two different models. It read like the best of both and it was a cross model composite. I caught it re reading against the logs, fixed it to one model per row, and that's where the rule came from: every table states which model produced every number, and no number appears without a named log on disk. <!-- src: PROJECT_LOG.md lines 40-58, 67-68 -->

Two numbers failed that rule and are listed as open defects instead of printed: a sigma quartile pair I'd attributed to the ensemble that actually belongs to the single base model (the ensemble's own pair is 3.530 vs 6.863), and a "5.47" external figure with no log on disk. <!-- src: PROJECT_LOG.md lines 60-66 -->

## What the pooled number hides (`scripts/_clin_audit.py`)

A pooled MAE of 4.733 is hiding an inverted error structure. By true EF band: bias +5.8 below EF 35, +5.0 at 35 to 45 with a band MAE of 9.28, -4.9 above 65, and 3.30 in the 55 to 65 mode that holds half the test set. Regression to the mean. A clinical 3 class read with cuts at 35 and 53 gives 82.3 percent agreement pooled but only 48.6 percent correct inside the 35 to 53 band, which is the surveillance population where the number actually changes management. <!-- src: FINDINGS.md sec 11 lines 225-231 -->

A VAL fitted isotonic recalibration failed on TEST (4.733 to 4.798, band biases unchanged) because the bias is conditional on true EF, not predicted EF, and an output remap can't create information. Inverse density loss weighting, judged on bands as pre registered, traded 35 to 45 band MAE from 9.44 to 8.62 for +0.33 pooled and -2.4 points of 3 class agreement. Recorded as what they are, a null with a mechanism and a trade. <!-- src: FINDINGS.md sec 11 lines 233-239, sec 13 lines 260-262; _recal.py; _lds_bands.py -->

## What I'd do again

One model per row and every number names its log. Report the external number even when it's worse, and report degradation instead of a level when the external sets differ. Put the band structure next to the pooled metric and the decision level agreement next to the regression metric. State every exclusion that changes n.

Scripts: `_clin_audit.py`, `_recal.py`, `_lds_bands.py`, `train_tok.py`.
