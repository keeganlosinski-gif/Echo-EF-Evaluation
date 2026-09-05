# Case 4: Audit the gate before you apply it

A quality gate looked like it was about to exclude a big slice of a public benchmark. How do I know the gate is right?

## The gate

Eleven gates, per frame and per video (`scripts/_qc_masks.py`, definitions in [METHODS.md](../METHODS.md) section 3). The bar was set before any results: a frame passes every gate or its video is excluded and counted, training on N verified beats training on N with garbage in the tails, and every exclusion gets reported next to any number graded on the verified subset. <!-- src: _qc_masks.py docstring and final print -->

First pass, the geometry gates G1 to G9 excluded 433 videos. <!-- src: FINDINGS.md sec 9.3 -->

## Auditing it (`scripts/_qc_audit.py`)

Before the fixes below the gate looked like it would throw out roughly a quarter of the benchmark. That figure only exists in the audit script's docstring, never in a log, so I quote it only as the reason the audit happened. A gate that excludes that much is itself a claim, so I tested its two most suspicious rules for false positives before accepting any exclusion, and both were wrong.

- **G4, monotone chord order.** The test required chords to run apex to base. 1,797 tracings are stored base to apex, which is perfectly monotone in reverse, ie valid data with a different storage convention. Fixed to accept either consistent direction and only fail actually jumbled ordering. <!-- src: FINDINGS.md sec 7 lines 103-105; _qc_masks.py inline comment -->
- **G6, solidity at least 0.95.** A diseased ventricle at end systole is genuinely non convex. The 10th percentile of solidity across all masks is 0.954, so a 0.95 line was rejecting real crescent shaped anatomy. Moved to 0.90. <!-- src: FINDINGS.md sec 7 lines 103-105; _qc_audit.py -->

After the fixes: G1 to G9 433 videos, G10 (the wrong chamber tracing from Case 3 plus one degenerate sliver) 2, G11 (acquisition type) 2. 437 of 10,024 excluded, 4.4 percent. Verified set 9,587 videos, 19,174 frames. One loose end I'm leaving visible: the per split counts the preflight asserts (7,150 / 1,220 / 1,219) sum to 9,589, because they were fixed before the two G11 exclusions and I didn't rerun the preflight. The logs don't say which splits those two videos were in. The temporal model logs report test n=1,219 and that's the number I quote. <!-- src: FINDINGS.md sec 9.3 lines 177-179; _preflight.py line 28 -->

## Checking the labels the gate passed, without a human (`scripts/_edgeagree.py`)

Geometry gates say a tracing is well formed. They don't say it sits on the endocardium. For that I used the image: a correct tracing lies on a dark to bright edge (blood inside, myocardium outside), so the contrast between a thin strip just inside the contour and a thin strip just outside should be positive.

What makes it a test instead of a number is the null control. The same contour eroded and dilated by four pixels has to score worse than the true contour or the metric is measuring nothing. Frames whose true contour did no better than the shifted null band got flagged. <!-- src: _edgeagree.py docstring -->

Over the 19,155 frames that had passed the gate as it stood that evening (the final ledger, written later, counts 19,174, and I never reconciled the two snapshots): 95.9 percent verified tracing on edge, 3.5 percent image limited (no usable edge in the image itself), and 31 frames (0.16 percent) true suspects, ie good image, off edge tracing. Only those 31 went to a human. <!-- src: FINDINGS.md sec 7 lines 105-109 -->

## The human step, protocol written first (`scripts/_adjudicate.py`)

A 60 tile panel: 30 random gate passers, 15 solidity borderline, 15 lowest solidity passers, each tile the raw frame and the overlay side by side. Protocol fixed before viewing: mark any tile where the traced contour doesn't follow the endocardial border, papillary muscles included in the cavity per the dataset's convention. Decision rule fixed before viewing too: 0 of 60 bad would put the label error rate under 5 percent at 95 percent confidence.

I reviewed the panel and didn't write the tally into the log, so I'm not claiming a bound from it. The rule stays here because it was written before I looked, which was the point, and the missing tally is the one number in this project I can't quote. What the log does have is the 31 true suspects: mixed, some defensible, some genuinely mistraced. Residual tracer noise is real and it's part of the readout floor from Case 1. <!-- src: FINDINGS.md sec 7 lines 107-109; the 60-tile tally has no receipt -->

I was the adjudicator, as a credentialed sonographer. One reader, not a panel, and the calls are reported that way.

## The preflight (`scripts/_preflight.py`)

Before the GPU stage one script rechecks everything the gate and the caches claim: row counts, exactly two frames per verified video, no video straddling splits, expected verified counts per split, cached masks equal to the rasteriser on random samples, label tails at both extremes, quality signals finite, videos decoding with the right frame counts, checkpoint integrity. Every line prints PASS or FAIL and any FAIL blocks the build. <!-- src: _preflight.py -->

## What I'd do again

Audit the gate before applying it. Two false positive modes in nine gates, both of which would have thrown away valid data. Use the data before the human: the null controlled check cleared 95.9 percent and I looked at 31 frames plus a panel. Write the protocol and the decision rule before looking, and then actually write the result down afterwards. And report exclusions next to every number graded on the subset.

Scripts: `_qc_masks.py`, `_qc_audit.py`, `_edgeagree.py`, `_adjudicate.py`, `_preflight.py`.
