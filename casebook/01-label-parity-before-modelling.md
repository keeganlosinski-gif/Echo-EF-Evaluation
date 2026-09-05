# Case 1: Label parity before modelling

The question I started with: before training anything on 10,030 labelled videos, how do I know the labels mean what I think they mean?

## What EchoNet gives you

Per video, a stated EF and two traced frames (ED and ES), each stored as a long axis plus 20 perpendicular chords across the LV cavity. The stated EF was produced from those same chords by Simpson's method of disks, which makes the two redundant, and redundancy is a free test. If I recompute EF from the chords and it doesn't match the stated EF, either the labels aren't what they claim or my arithmetic is wrong, and either way I shouldn't be training yet. <!-- src: _enparity.py docstring; FINDINGS.md sec 1 -->

## What I did

Reimplemented monoplane Simpson from the chord geometry (`scripts/_enparity.py`, biplane version for CAMUS in `scripts/simpson.py`), ran it over every video with two traced frames, compared to the stated EF.

MAE 1.05 points, corr 0.971, n=10,024. Six videos have no tracings. <!-- src: FINDINGS.md sec 1 lines 13-15 -->

Then the cheap checks people skip (`scripts/_leakchk.py`). The file list carries no patient identifier, so I can't verify there's no patient leakage across splits, and re-splitting without patient IDs would only make that worse, so the official 7,460 / 1,288 / 1,276 splits are used as shipped. The same script checks for duplicate label tuples across splits. I didn't keep that output, so I'm not claiming a result from it. <!-- src: AUDIT_2026-08-24.md sec 2.1; echonet_data.py docstring -->

## Measuring the floor before the model

The last stage of the pipeline is a geometric readout from a mask to an EF (`scripts/ef_readout.py`). I built that first and ran it on the ground truth masks (`scripts/_validate_readout.py`) before any segmenter existed. On the official test set (n=1,276): MAE 2.00, bias -0.28, corr 0.954 with the first rasteriser's masks, 1.71 after the rasteriser was replaced. That's the floor. A model that segments perfectly still lands around 1.7 through this readout, so every later score has a ceiling to be read against, and anything that came in under the floor would itself be a red flag. <!-- src: RESULTS_0824.log line 63 (2.00, n=1276) and lines 78, 99 ('floor(fixed masks) 1.71') -->

The chord to mask rasteriser was checked on its own: 800 of 800 frames reconstructed as a single connected blob, ED area greater than ES area in 200 of 200 videos. The first version self crossed on curved ventricles and left holes, so it got replaced with a per chord quad fill that can't. <!-- src: echonet_masks.py docstring -->

## Why this paid off

The next day CAMUS's mask derived EF came in 8 points above its stated EF (Case 2) and my first suspect was my own Simpson code. Because the parity check had already validated that arithmetic on 10,024 independent cases, I could spend the time building a second estimator instead of re-debugging the first.

## What I'd do again

Before training on a label, find something the label has to satisfy and test it. If the dataset ships two things that should agree, make them agree before using either. And measure the pipeline's floor on ground truth before the first model, so later scores get read against a ceiling instead of against each other.

Scripts: `_enparity.py`, `simpson.py`, `_leakchk.py`, `ef_readout.py`, `_validate_readout.py`, `echonet_masks.py`.
