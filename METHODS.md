# Methods

What was built, in the order it was validated, and the rules that came out of it. Numbers live in [FINDINGS.md](FINDINGS.md), this file is procedure.

## 1. Data

- **EchoNet-Dynamic**: 10,030 apical four-chamber (A4C) videos, 112x112 pixels, each with a stated EF and, for all but six videos, two traced frames (end-diastole and end-systole) given as a long axis plus 20 perpendicular chords. Official TRAIN/VAL/TEST splits of 7,460 / 1,288 / 1,276. <!-- src: FINDINGS.md sec 1; AUDIT_2026-08-24.md sec 2.1 -->
- **CAMUS**: 500 patients, two-chamber and four-chamber views, expert masks (LV cavity, myocardium, left atrium) at ED and ES, a per-patient stated EF, and an image-quality label. <!-- src: FINDINGS.md intro -->
- Official EchoNet splits are used verbatim. `FileList.csv` carries no patient identifier, so re-splitting could not be checked for patient leakage and would be strictly worse than trusting the distributor's split. <!-- src: echonet_data.py docstring; _leakchk.py -->
- No dataset content (frames, masks, videos, label tables, cached arrays, per-video results, study identifiers) is in this repository.

## 2. Validation order

The rule was that nothing gets used until it's been checked against something it couldn't have produced itself.

1. **Label parity** (`_enparity.py`, `simpson.py`). EchoNet's stated volumes were derived from the shipped chords, so recomputing volume from the chords by Simpson's method of disks must reproduce the stated EF. It does (MAE 1.05 points). This validated both the labels and my Simpson arithmetic in one step, which mattered for step 5. <!-- src: FINDINGS.md sec 1 -->
2. **Split and duplicate check** (`_leakchk.py`). Splits disjoint (7,460 / 1,288 / 1,276, recorded in the audit); a duplicate-label-tuple check across splits was run by the same script but its output was not retained; no patient identifier is available. <!-- src: AUDIT_2026-08-24.md sec 2.1 -->
3. **Chord-to-mask reconstruction** (`echonet_masks.py`). Verified on 800/800 frames: single connected component, ED area greater than ES area in 200/200 videos. A first rasteriser self-crossed on curved ventricles and left interior holes; it was replaced by a per-chord quad fill that cannot self-cross. <!-- src: echonet_masks.py docstring -->
4. **EF read-out floor** (`ef_readout.py`, `_validate_readout.py`). The last stage of the pipeline (mask to EF, area-length method) was built first and run on ground-truth masks before any model existed. Its error against the stated EF is the floor any model through this read-out can reach. <!-- src: FINDINGS.md sec 4 -->
5. **Cross-dataset parity fork** (`_arealen.py`). When CAMUS mask-derived EF disagreed with its stated EF by about 8 points, the first suspect was my code. Steps 1 and 4 had already validated the arithmetic on EchoNet; a second, geometrically independent estimator (no rotation, no disks, no annulus heuristic) gave the same offset, which located the gap on the dataset side. <!-- src: AUDIT_2026-08-24.md RESOLUTION -->
6. **Orientation.** CAMUS volumes are stored apex-left, EchoNet apex-top. Found by rendering, after a cross-dataset Dice of 0.056 whose median was exactly zero. <!-- src: FINDINGS.md sec 1; AUDIT_2026-08-24.md sec 2.3 -->
7. **Only then, models.**

## 3. Quality-control gate ledger (`_qc_masks.py`)

Per traced frame:

| Gate | Test |
|---|---|
| G1 | chord count is 21 (long axis plus 20 disks) |
| G2 | all endpoints inside the frame |
| G3 | consecutive chords do not cross |
| G4 | chord midpoints progress monotonically along the long axis, in either consistent direction |
| G5 | reconstructed mask is a single connected component |
| G6 | solidity at least 0.90 |
| G7 | mask area within 80 to 8,000 pixels |

Per video:

| Gate | Test |
|---|---|
| G8 | both frames pass; ED area strictly greater than ES area |
| G9 | mask-derived EF in (2, 90) and within 25 points of the stated EF |
| G10 | wrong-chamber tracing (adjudicated) and a degenerate sliver |
| G11 | acquisition type outside protocol (split-screen composite, rotated scan region) |

A frame either passes every gate or its video is excluded and counted. Exclusions are reported alongside any number graded on the verified subset. <!-- src: _qc_masks.py; FINDINGS.md sec 9.3 -->

### 3.1 The gate was audited before it was applied (`_qc_audit.py`)

The first pass of the gate looked as though it would exclude roughly a quarter of the benchmark (an estimate recorded only in the audit script's docstring, never logged), so before accepting any exclusion I tested the two most suspicious gates for false positives. After the fixes the ledger excluded 437 of 10,024 videos (4.4 percent). Two were found and fixed: G4 originally required forward chord order and rejected 1,797 tracings stored base-to-apex, which are valid data; G6 originally sat at 0.95 and rejected genuine crescent-shaped end-systolic anatomy (the 10th percentile of solidity is 0.954), so the defect line was moved to 0.90. <!-- src: FINDINGS.md sec 7; _qc_masks.py inline comments 'AUDITED 08-24' -->

## 4. Non-circular label check with a null control (`_edgeagree.py`)

A correct endocardial tracing sits on a dark-to-bright edge: pixels just inside the contour (blood) are darker than pixels just outside (myocardium). The metric is the contrast between a thin inside strip and a thin outside strip along the contour. The null control is the same contour eroded or dilated by four pixels: it must score worse, or the metric measures nothing. Frames whose true contour scored no better than the shifted-null band were flagged; only those went to a human. <!-- src: _edgeagree.py docstring; FINDINGS.md sec 7 -->

## 5. Human adjudication protocol (`_adjudicate.py`)

- A 60-tile panel: 30 random gate-passers, 15 solidity-borderline (0.90 to 0.95), 15 lowest-solidity passers. Each tile shows the raw frame and the overlay side by side.
- Written protocol, fixed before viewing: mark any tile where the traced contour does not follow the endocardial border, with papillary muscles included in the cavity per the dataset's convention.
- Outcome rule, fixed before viewing: 0 of 60 bad implies a label-error rate below 5 percent at 95 percent confidence. The panel was reviewed, but its tally was not written into the log, so FINDINGS.md claims no bound from it.
- Adjudicator: me, as a credentialed sonographer. One reader, not a panel.
- The same reader reviewed the 31 true suspects from section 4 and the 12 most confident model failures. Those reviews are reported as one reader's calls. <!-- src: _adjudicate.py docstring; FINDINGS.md sec 7-8 -->

## 6. Anomaly screens and the detector-validation rule

- **Wrong-chamber screen** (`_chamber_screen.py`): Mahalanobis distance of each mask centroid from the population. The worst 12 were rendered for review. The screen's top hits turned out to be acquisition-type anomalies rather than label errors, so the same screen doubles as an out-of-protocol acquisition detector. <!-- src: FINDINGS.md sec 9 -->
- **Acquisition-type census** (`_acqtypes.py`): a rotated-rectangle fill test for rotated exports and a column-brightness divider test for split-screen composites.
- **Rule**: validate any anomaly detector on a known positive before quoting its census. The first seam metric claimed 644 suspects; rendering its top 16 falsified it. The replacement was checked on the one known composite before it was run over the dataset. <!-- src: FINDINGS.md sec 9.2 -->

## 7. Pre-build preflight (`_preflight.py`)

Before any GPU stage, a script checks metadata integrity (row counts, required columns, exactly two frames per verified video, no video straddling splits, expected verified counts per split), that cached masks equal the rasteriser on random samples, label tails (median and both extremes, not just the middle), that quality signals are finite, that videos decode with the expected frame counts, and checkpoint integrity. Every check prints PASS or FAIL; any FAIL blocks the build. <!-- src: _preflight.py -->

## 8. Models (brief)

- **Segmenter**: U-Net with a ResNet34 encoder, four classes, trained on CAMUS, then on CAMUS plus EchoNet with a masked partial-label loss (four classes where CAMUS provides them, LV-only where EchoNet's chords provide it), orientation normalised, resolution jittered. <!-- src: FINDINGS.md sec 3; AUDIT_2026-08-24.md sec 2.2 -->
- **Frame-based EF**: segmenter masks on the two traced frames, largest connected component, area-length read-out. No learned read-out. <!-- src: FINDINGS.md sec 4 -->
- **Temporal model**: per-frame tokens from the frozen segmenter encoder, a four-layer transformer over the cardiac cycle, a heteroscedastic head predicting (EF, sigma) under a Gaussian or Laplace negative log-likelihood. Deterministic seed, minutes per run. <!-- src: train_tok.py docstring; FINDINGS.md sec 10 -->
- **Quality signals built because the dataset ships none**: a per-frame noise score (achievable cavity/myocardium contrast) and softmax entropy along the predicted border. <!-- src: FINDINGS.md sec 7 -->
- **Excluded input**: frame rate. 79 percent of EchoNet sits at exactly 50 fps and fps tracks heart size through depth and sector (EDV 103 mL below 40 fps vs 84 mL above 60 fps), so clips are resampled by cardiac phase and frames are sampled, never averaged. <!-- src: FINDINGS.md sec 2; echonet_data.py docstring -->

## 9. Evaluation and reporting rules

1. **Selection on validation, test read once.** Arms are chosen on VAL; TEST ordering is checked against VAL ordering after the fact. <!-- src: FINDINGS.md sec 11 -->
2. **Report exclusions with every number.** Temporal-model numbers are on the 1,219 verified test videos; frame-based numbers and the published reference are on all 1,276. The difference is stated wherever they appear. <!-- src: _qc_masks.py; RESULTS_0824.log eval note -->
3. **One model per table row; every number names its log.** Adopted after I caught a cross-model composite in my own draft. Numbers without a log on disk are listed as open defects, not printed. <!-- src: PROJECT_LOG.md sec 2 -->
4. **Cross-dataset EF is graded against both targets.** On CAMUS, every EF number is reported against the mask-derived EF (self-consistent) and against the stated EF (known offset of about 8 points), and the write-up says which. <!-- src: _camuseval_tok.py docstring; FINDINGS.md sec 11 -->
5. **Band audits beside pooled metrics.** MAE and bias per EF band and 3-class clinical agreement (cuts at 35 and 53) are reported next to pooled MAE. <!-- src: _clin_audit.py -->
6. **A robustness test must be able to fail.** Checked by measuring what the network actually sees after standardisation (`_gammaresid.py`). <!-- src: FINDINGS.md sec 6 -->
7. **A check that checks nothing must not PASS.** A gate that reported PASS on zero items was treated as a bug. <!-- src: FINDINGS.md sec 6 -->
8. **Retractions stay in the log**, labelled, with the reason.
9. **Calibration is verified on a held-out knot**, never in-sample (`_calverify.py`). <!-- src: FINDINGS.md sec 2 -->
10. **Dice is converted to millimetres** before it is compared across classes, because Dice depends on perimeter-to-area ratio and a thin ring is penalised harder than a blob for the same tracing accuracy (`_dicemm.py`). <!-- src: FINDINGS.md sec 3 -->
11. **Completion markers are not trusted without the process's own output.** A chain log once recorded a completion marker for a stage whose trainer had been killed; it is annotated as phantom in the log and the stage was re-run. <!-- src: _stage4_results.log 'E2E_DONE above is PHANTOM' note -->


