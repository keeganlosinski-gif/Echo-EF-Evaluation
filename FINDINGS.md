# Findings

This is the project's findings log, cleaned up. Every number names the log or script it came from in a hidden comment (view the raw file to see them). Retracted results are kept at the end with the reason, and anything I couldn't tie to a log on disk is in section 16 instead of being printed as a result. All numbers date from 2026-08-24 to 2026-08-28 unless stated; a 2026-08-31 re-run is listed in section 13.

## 1. Label audits ran before any modelling

- **EchoNet labels are self-consistent.** Simpson's method of disks recomputed from the shipped VolumeTracings chords reproduces the stated EF to MAE 1.05 points (corr 0.971, n=10,024). Six of 10,030 videos lack tracings and are excluded. Splits are disjoint at 7,460 / 1,288 / 1,276; a duplicate-label-tuple check across splits was run by `_leakchk.py` but its output was not retained. <!-- src: FINDINGS.md sec 1 lines 13-15; AUDIT_2026-08-24.md sec 2.1 -->
- **CAMUS's two ground truths disagree.** EF from CAMUS's expert masks sits a systematic +8 points above the stated per-patient EF: disk-Simpson +8.45, area-length +7.67 (two geometrically independent estimators agreeing to 0.8 points, corr about 0.97, flat across image-quality strata). Conclusion: the stated EF was not derived from the shipped contours; mask-derived EF and stated EF are different targets. <!-- src: FINDINGS.md sec 1 lines 17-24; AUDIT_2026-08-24.md RESOLUTION lines 62-71 -->
- **Storage orientation trap.** CAMUS volumes are stored apex-left, EchoNet apex-top. A cross-dataset Dice of 0.056 with median exactly 0.000 became 0.693 after one rotation. Found by rendering, not statistics. <!-- src: FINDINGS.md sec 1 lines 26-27 -->

## 2. Acquisition physics, measured

- **Frame rate is a shortcut and was excluded as an input.** 79 percent of EchoNet is pinned at 50 fps; fps tracks heart size through depth and sector (EDV 103 mL below 40 fps vs 84 mL above 60 fps). Clips are resampled by cardiac phase; frames are sampled, never averaged. <!-- src: FINDINGS.md sec 2 lines 31-34; _whyfps.py -->
- **Calibration verified on a held-out knot.** Calibrating to anchors and re-measuring the anchors is circular, so the myocardium knot is held out: it lands at 0.381 against a canonical 0.384; a 3-knot monotone map cuts inter-study spread by 47.5 percent; a two-point affine map is worse than no calibration (CV 0.298 vs 0.217) because transfer curves differ in shape, not just level. <!-- src: FINDINGS.md sec 2 lines 35-41; AUDIT_2026-08-24.md sec 2.4; _calverify.py -->

## 3. Segmentation

- U-Net (ResNet34 encoder, 24.5M parameters), four classes, CAMUS test Dice LV 0.9376 / myocardium 0.8764 / left atrium 0.9204, in 11 minutes of training. Converted to boundary error by `_dicemm.py` this is about 1.55 mm, which is in the range reported for human intra-observer variability on CAMUS (the project's notes carry about 1.53 mm for that; the specific reference was not recorded, so the comparison is indicative, not a citation). The Dice differences between classes are geometry (perimeter-to-area), not accuracy: the thin myocardial ring at 0.876 is the same tracing quality as the cavity at 0.938. <!-- src: FINDINGS.md sec 3 lines 45-50; AUDIT_2026-08-24.md sec 2.2; _dicemm.py -->
- **Robustness test design.** Gamma robustness is real (flat under a 4x curve swing). The gain sweep was vacuous because per-image standardisation absorbs affine changes exactly; the test could not fail and was discarded. A one-variable augmentation-versus-none control put augmentation at +0.008 Dice in-domain. <!-- src: FINDINGS.md sec 3 lines 51-53; AUDIT_2026-08-24.md sec 2.5-2.6; _gammaresid.py -->
- **Cross-dataset segmentation.** CAMUS-trained to EchoNet LV Dice 0.693 vs 0.937 in-domain (about -26 percent). Combined training with a masked partial-label loss, orientation normalisation and resolution jitter closed it: cross-dataset Dice 0.693 to 0.9215 while CAMUS improved to 0.9399, in 10.7 minutes. <!-- src: FINDINGS.md sec 3 lines 54-56; RESULTS_0824.log 'TEST CAMUS LV 0.9399 ... TEST EchoNet LV 0.9215 ... DONE 10.7 min' -->

## 4. EF read-out: validated before use, then evaluated end-to-end

- **Floor measured on ground truth before any model.** Area-length read-out on chord-reconstructed ground-truth masks, EchoNet official test (n=1,276): MAE 2.00 points, bias -0.28, corr 0.954 on the original polygon-rasterised masks; 1.71 on the quad-fill masks that replaced them (METHODS section 2, step 3); 1.73 on the 2026-08-31 re-run of the same chain. The floor has no training randomness, so the 2.00 to 1.71 change is the rasteriser fix, not run-to-run noise. This is the best any model through this read-out can do. The working log also carried this floor with n=10,024 (all videos), but no receipt for that line survives on disk, so the receipted n=1,276 is quoted. <!-- src: RESULTS_0824.log line 63 'GT-mask EF vs stated (readout floor) n=1276 ... MAE 2.00', lines 78 and 99 'floor(fixed masks) 1.71', 08-31 block line 133 '1.73'; _validate_readout.py -->
- **Frame-based EF, zero learned read-out, EchoNet official test (n=1,276):** first segmenter (v1, trained on the original masks): raw masks MAE 8.78 / corr 0.662; with largest-connected-component MAE 7.18 / corr 0.742 (catastrophic errors 100 to 68); a naive volume-curve p98/p2 read gives MAE 9.12 with extreme-value bias +8.04. Two later retrains on the same 1,276 did not beat it: v2 (fixed quad-fill masks) largest-CC 9.59 / corr 0.675, v3 (verified labels) largest-CC 7.41 / corr 0.733. The v2 regression is not explained in the logs. 7.18 is the frame-based number the temporal-model comparison in section 9 uses, because it was the working baseline at the time; the other two are reported here so the choice is visible. <!-- src: RESULTS_0824.log lines 61-63 (v1), 77-78 (v2), 98-99 (v3) -->
- **CAMUS test, biplane (n=52):** vs mask-derived EF MAE 3.94 / corr 0.923; vs stated EF MAE 8.56 with bias +8.08, the provenance gap reproduced end-to-end by a model. <!-- src: FINDINGS.md sec 4 lines 64-72 -->
- **Failure analysis by looking.** The worst EchoNet cases are stray disconnected blobs on ES frames inflating ESV (one case: EF -8.9 vs stated 56.5). The cavity is perceived correctly; the loss is mask-to-measurement compounding. The general point, a proxy metric (Dice) can improve while the endpoint (EF) does not, is what the failure analysis shows directly; the working notes also recorded a published nnU-Net-on-CAMUS parallel, but the reference was not kept, so it is not quoted here. <!-- src: FINDINGS.md sec 4 lines 74-78 -->

## 5. Rules that came out of this (nothing echo specific about them)

1. Parity-gate labels against an analytic identity before training.
2. Verify calibration by held-out knot, never in-sample.
3. A robustness test must be able to fail.
4. Look at the images: two of the largest bugs were invisible to aggregate metrics.
5. A check that checked nothing must not PASS (the 0/0 alignment-gate bug). <!-- src: FINDINGS.md sec 6 lines 92-99 -->

## 6. Label-noise verdict, the gate audit and the quality signals

- **Gate ledger.** Of 10,024 videos with tracings, G1-G9 geometry gates excluded 433; G10 (a wrong-chamber tracing plus a degenerate sliver) 2; G11 (acquisition type) 2. Final verified set: 9,587 videos / 19,174 traced frames (two per video). The per-split counts asserted by the preflight, 7,150 / 1,220 / 1,219, sum to 9,589, which is the pre-G11 set (the acquisition-type census in section 8 also ran over 9,589); the two G11 exclusions came after those counts were fixed, and the preflight was not re-run to update them. The logs do not record which splits the two G11 videos fall in; the temporal-model logs report test n=1,219, which is the number quoted. <!-- src: FINDINGS.md sec 9.3 lines 177-179 and sec 9.2 line 170; _preflight.py line 28 -->
- **The gate was audited.** Two false-positive modes found and fixed before the exclusion stood: 1,797 tracings stored base-to-apex are valid (G4 now accepts either consistent direction); solidity 0.95 rejected real crescent-shaped ES anatomy (p10 = 0.954), so the defect line moved to 0.90. <!-- src: FINDINGS.md sec 7 lines 103-105; _qc_audit.py -->
- **Null-controlled edge agreement.** Run on the 19,155 frames that survived the gate as it stood on the evening of 2026-08-24 (the final ledger above, written later, counts 19,174; the two snapshots were not reconciled in the working log): 95.9 percent verified tracing-on-edge, 3.5 percent image-limited, 31 frames (0.16 percent) true suspects (good image, off-edge tracing). Only the flagged frames went to a human. <!-- src: FINDINGS.md sec 7 lines 105-109; _edgeagree.py -->
- **Human adjudication.** A 60-tile stratified panel was built and reviewed under a written protocol whose decision rule was fixed before viewing (0 of 60 bad would bound label error below 5 percent at 95 percent confidence). The panel's tally was not written into the log, so no bound is claimed from it. What the log does record is the review of the 31 true suspects: mixed, some defensible, some genuinely mistraced. Residual tracer noise is real and is a component of the read-out floor. One reader (me, RDCS). <!-- src: _adjudicate.py docstring (panel and rule); FINDINGS.md sec 7 lines 107-109 (the 31-suspect verdict); no log carries the 60-tile tally -->
- **Quality signals built because the dataset ships none.** A per-frame noise score (the image's own achievable cavity/myocardium contrast) tracks EF error monotonically: frame-model MAE 8.07 to 5.80 and blow-up rate 13.4 to 6.2 percent from worst to best quartile. Softmax entropy along the predicted border stratifies MAE 5.52 to 8.29 and correlates with the noise score at only -0.35, so they are two different signals. <!-- src: FINDINGS.md sec 7 lines 111-128 -->

## 7. Expert adjudication of the confident failures

- 12 confident-disaster ES frames reviewed: all model-side, 11 of 12 underestimates. Two mechanisms: the model under-extends the apex where near-field artifact hides it (a sonographer extrapolates through the artifact); the tracer pulls the ES border into the cavity where no endocardium is visible, plausibly borrowed from adjacent frames.
- The reviewer's hedge ("could have been visualised temporally") was tested: the tracer's contour fits its own frame best in only 17 percent of videos (n=300). EchoNet labels are temporal objects rendered onto single frames. That set the requirement for the next model. <!-- src: FINDINGS.md sec 8 lines 130-148; _temporaltrace.py; _crispfails.py -->

## 8. Acquisition and wrong-chamber anomalies

- **Wrong-chamber tracing.** One video's shipped chords sit on the right ventricle in both traced frames of a normal apex-up A4C. My adjudication: definitively an RV; the dextrocardia alternative rejected as an implausible inclusion. Intentionality ruled out from the dataset's own documentation (tracings are defined as LV endocardium only; there is no chamber field and no errata). Provenance proven by drawing the raw CSV chords on the video with no rasteriser in the loop. Rate: 1 confirmed in 10,024. A count, not an extrapolation. <!-- src: FINDINGS.md sec 9.1 lines 154-163; _rv_evidence.py -->
- **Split-screen dual-view composite.** Two sectors side by side with a bright divider. Even if its left-half tracing is a correct LV, all geometry computes at quarter scale on a composite, so it is excluded. A divider-line detector was validated on this known positive (line score 0.79 vs neighbour 0.02) and then run over all 9,589 videos: it is the only one, with one false positive (a bright septum). <!-- src: FINDINGS.md sec 9.2 lines 166-172 -->
- **Scan region rotated about 45 degrees.** A rotated export, not a different probe; the only one in the dataset; it broke the orientation detector's assumptions; excluded. <!-- src: FINDINGS.md sec 9.2 lines 173-175; _find_diamond.py (not included: embeds a study id) -->
- **All three were found by human eyes on rendered panels, then censused programmatically.** The dataset ships no flag for any of them. Each is one instance in 10,030: invisible to aggregate metrics, fatal to the individual study. <!-- src: FINDINGS.md sec 9 lines 150-152 -->
- **Rule: validate any anomaly detector on a known positive before quoting its census.** A first-pass seam metric claimed 644 suspects; rendering its top hits showed normal sectors; the metric was discarded. <!-- src: FINDINGS.md sec 9.2 lines 170-172, 181-183 -->
- **Rule: a screen for one thing may be a detector for another.** The Mahalanobis centroid screen built for wrong-chamber labels surfaced acquisition-type anomalies instead. <!-- src: FINDINGS.md sec 9 lines 181-183 -->

## 9. Temporal model (stage 4)

- Frozen per-frame tokens into a four-layer temporal transformer (1.8M parameters), heteroscedastic (EF, sigma) head, about 10 minutes, deterministic seed, trained on the 9,587 verified videos. EchoNet verified test: MAE 4.852, corr 0.845, n=1,219, one read. Against the 7.18 frame-based number that is about -32 percent, with the caveat that 7.18 is on all 1,276. <!-- src: FINDINGS.md sec 10 lines 187-198; tok_base.log 'TEST MAE 4.852 corr 0.845 n=1219' -->
- **Sigma head as a deployable flag.** Ranking the test set by predicted sigma, crispest quartile vs most difficult quartile: 3.725 vs 7.003 (single model); 3.530 vs 6.863 (3-seed spatial ensemble); 3.499 vs 6.198 (standardised-feature ensemble). The predicted Laplace scale matches empirical MAE in every decile without further calibration. The clinical reading: crisp, report EF; difficult, flag as technically limited. <!-- src: tok_base.log; ens_sp.log; ens_v4std.log; FINDINGS.md sec 10-11 lines 200-202, 241-243 -->
- **Ablation sequence with pre-registered selection** (arms selected on VAL, TEST ordering matched VAL ordering): Laplace NLL 4.798 beats Gaussian 4.852; denser clip sampling, 2x frames and extra capacity are null; the auxiliary features are worth 0.54 (5.391 without them); seeds replicate to about plus or minus 0.03; a 3-seed x 3-TTA spatial ensemble reaches 4.733 / corr 0.851. <!-- src: FINDINGS.md sec 11 lines 215-218; _stage4_results.log -->

## 10. Internal versus external (one model per row)

| Model | Internal (EchoNet test) | External (CAMUS, n=52, vs mask-derived EF) | Change |
|---|---|---|---|
| Published reference (Ouyang et al. 2020), as reported | 4.05 (n=1,276) | 6.0 (their external set) | +48% |
| Spatial ensemble (3 seeds x 3 TTA) | 4.733 (n=1,219) | 4.043, corr 0.914, bias -0.67 | none (external lower, -15%; n=52, wide interval) |
| Standardised-feature ensemble | 4.488 (n=1,219) | 6.909 | +54% |

<!-- src: PROJECT_LOG.md sec 2A corrected table lines 44-58; ens_sp.log; camuseval_sp.log; ens_v4std.log; camuseval_v4std.log; WRITEUP_DRAFT.md lines 46-62 -->

- Against CAMUS's stated EF instead of its mask-derived EF, the spatial ensemble reads 7.689: the +8 point provenance offset reproduced a fourth time. <!-- src: camuseval_sp.log; FINDINGS.md sec 11 lines 220-223 -->
- **Caveat that goes with the table.** The external datasets differ from the published work's, so this is not a head-to-head on identical data. The defensible claim is about degradation under distribution shift (none for the spatial ensemble vs +48 percent for the published reference), not a direct MAE comparison. The internal n also differs (1,219 verified vs 1,276 official). <!-- src: WRITEUP_DRAFT.md lines 74-79 -->
- **Reading.** The features that bought internal accuracy (standardised descriptors) were the least transferable. <!-- src: WRITEUP_DRAFT.md sec 2 -->

## 11. Structure the pooled number hides

- **EF-band audit** (spatial ensemble): bias +5.8 below EF 35; +5.0 at 35-45 with band MAE 9.28; -4.9 above 65; the population mode 55-65 (half the test set) sits at 3.30. Regression to the mean. <!-- src: FINDINGS.md sec 11 lines 225-231; _clin_audit.py -->
- **Clinical 3-class read** (cuts at 35 and 53): 82.3 percent agreement pooled but only 48.6 percent correct inside the 35-53 band, which is the surveillance population that matters most. <!-- src: FINDINGS.md sec 11 lines 225-231 -->
- **Recalibration null, with mechanism.** A VAL-fitted isotonic remap failed on TEST (4.733 to 4.798, band biases unchanged) because the bias is conditional on true EF, not predicted EF; an output remap cannot create information. <!-- src: FINDINGS.md sec 11 lines 233-239; _recal.py -->
- **Inverse-density weighting**, judged on bands per the pre-registration: 35-45 band 9.44 to 8.62, at a cost of +0.33 pooled and -2.4 points of 3-class agreement. A trade, not information. <!-- src: FINDINGS.md sec 13 lines 260-262; _lds_bands.py -->

## 12. A premise retired, then un-retired by a better test

- A descriptor-feature arm was pre-registered, measured and read as null (correlation cap unchanged). A knockout audit then showed the test was bad: the new features had entered at 10-30x smaller scale and carried 0.002-0.007 EF points of influence. Per-dimension standardisation from TRAIN statistics unlocked them (knockout influence radial 6.7 / deep 5.4 / mid 3.7 / flow 0.8 EF points) and the correlation cap that had held at 0.845-0.852 across about 15 arms broke: MAE 4.488, corr 0.865. The earlier "encoder exhaustion" conclusion was retracted in place as premature. <!-- src: FINDINGS.md sec 12-13 lines 251-282; _v4_testaudit.py; _v4std_knockout.py -->
- **Interference result, recorded.** Fine-tuning the encoder with engineered features read 4.637 internal (worse than 4.488) and collapsed on CAMUS to 11.225 (bias -10.39): naive EF fine-tuning made the encoder vendor-specific. Anchor-loss fine-tune arms read 6.694 / 6.177 / 4.794 against a 4.768 control; none beat the frozen champion. <!-- src: FINDINGS.md sec 14 lines 295-302; ens_v5std.log; camuseval_v5.log; _stage4_results.log E2E sections -->

## 13. Run-to-run variation (2026-08-31 re-run)

Re-running the combined segmentation chain gave CAMUS LV 0.9391 / EchoNet LV 0.9255 (vs 0.9399 / 0.9215 on 08-24), raw frame-based EF MAE 8.05 (vs 8.78 for the v1 chain on 08-24), CAMUS biplane 3.64 (vs 3.94). I quote the 08-24 values because they are the ones the named logs and the findings log carry; the spread is listed here rather than picking the better run. The read-out floor is not part of this spread: it is computed on ground-truth masks with no training randomness, and its values (2.00 original masks, 1.71 quad-fill masks, 1.73 on the 08-31 re-run of the quad-fill chain) are stated in section 4. <!-- src: RESULTS_0824.log '=== ECHO CHAIN Mon Aug 31' block vs 08-24 block -->

## 14. Power analysis of an external benchmark (measure the ruler)

For a public challenge task with n=84 test cases at 65.4 percent prevalence, any observed AUC below 0.631 cannot clear 0.5; the organisers' own baseline of 0.541 has a CI of 0.390-0.692 crossing 0.5. Before entering, know whether the test set can resolve anything. <!-- src: AUDIT_2026-08-24.md sec 2.7 lines 41-42; _power.py -->

## 15. Retracted, and not to be quoted

- "CAMUS MAE 3.08 beats state of the art": circular. The ground-truth masks are the expert contour the EF came from. Retracted in-session. <!-- src: AUDIT_2026-08-24.md lines 44-50 -->
- "Brightness is null": the test was self-referential. <!-- src: AUDIT_2026-08-24.md lines 44-50 -->
- Rows 2-3 of the whole-sector / LV-box / GT-area sequence: not clean; only row 1 stands. <!-- src: AUDIT_2026-08-24.md lines 44-50 -->
- "Encoder exhaustion": retracted after the knockout audit (section 12). <!-- src: FINDINGS.md sec 13 -->

## 16. Numbers with no log on disk (listed, not printed as results)

- The widely quoted sigma-quartile pair 3.73 vs 7.00 belongs to the single base model; the ensemble's values are 3.530 vs 6.863. Section 9 uses the receipted pair for each model. <!-- src: PROJECT_LOG.md lines 60-66 -->
- A "5.47 with domain-adaptive self-statistics" CAMUS figure in the original log has no log file on disk. The only logged external number for that model is 6.909, which is what section 10 shows. <!-- src: PROJECT_LOG.md lines 60-66; FINDINGS.md lines 284-286 -->

## 17. Self audit of the first night

The audit I wrote after the first night isn't flattering and I'm keeping it: the pipeline stopped at stage 1 of 5 with nothing outputting an EF; the +8 point gap sat undiagnosed for hours while a minutes-cheap test was available; one circular number briefly stood before it was caught. The same audit's read of the negative results was that most were good tests correctly killing bad ideas. <!-- src: AUDIT_2026-08-24.md lines 73-80, 144-147 -->


