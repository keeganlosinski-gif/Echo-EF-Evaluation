# Case 5: A test that can't fail isn't a test

Four small episodes from the same week that all came down to the same question: could this result have come out the other way?

## The robustness sweep that proved nothing (`scripts/_gammaresid.py`)

I swept the segmenter over gain changes and it was flat. Good news, until I asked what the network actually sees. Inputs are standardised per image, and standardisation cancels an affine change exactly, so the gain sweep couldn't have failed. It was measuring the identity. Gamma is different: a nonlinear curve survives standardisation in principle, so I measured the residual the network sees after standardisation for each distortion. Gain and offset came out at zero, as they have to, and gamma survived as a real distortion, so the gamma result (flat under a 4x curve swing) stands and the gain result was dropped. A separate one variable control, augmentation on vs off, put the augmentation's in domain value at +0.008 Dice, mostly overfitting control. <!-- src: FINDINGS.md sec 3 lines 51-53, sec 6 line 96; _gammaresid.py docstring -->

## The gate that passed on nothing

An alignment gate reported PASS. It had examined zero items: an empty cache made the check 0 of 0 and 0 of 0 evaluated as success. Obvious once seen. The lesson I took is that PASS has to be earned by a count and a check with no denominator is a bug in the check. <!-- src: FINDINGS.md sec 6 line 99 -->

## The null that was a bad test (`scripts/_v4_testaudit.py`, `scripts/_v4std_knockout.py`)

A pre registered arm added engineered descriptor features to the temporal model. It came back null, the correlation cap (0.845 to 0.852 across about 15 arms) didn't move, and I wrote down "encoder exhaustion" as the explanation.

Then I asked whether the test had been a good test. A knockout audit zeroed each feature group in turn on the trained model and measured how far the prediction moved. The new features had entered at 10 to 30x smaller scale than the existing ones and carried 0.002 to 0.007 EF points of influence. The model had never been able to use them. So the arm hadn't tested the hypothesis, it had tested whether an unscaled feature can matter, and the answer to that is no.

Per dimension standardisation from TRAIN statistics fixed the scale. Knockout influence became radial 6.7 / deep 5.4 / mid 3.7 / flow 0.8 EF points and the cap broke: MAE 4.488, corr 0.865. "Encoder exhaustion" was retracted in place, labelled premature, with the audit that overturned it next to it. <!-- src: FINDINGS.md sec 12-13 lines 251-282; _v4_testaudit.py docstring -->

Those same standardised features later turned out to be the least transferable (Case 6), which is its own lesson about what an internal gain is worth.

## Measuring the ruler of an external benchmark (`scripts/_power.py`)

Before considering a public challenge task I ran a power analysis on its test set: n=84 at 65.4 percent prevalence. At that n any observed AUC below 0.631 can't be distinguished from 0.5, and the organisers' own baseline of 0.541 has a confidence interval (0.390 to 0.692) that crosses chance. Whatever a model scored there, the test set couldn't have told anyone whether it worked. <!-- src: AUDIT_2026-08-24.md sec 2.7 lines 41-42; _power.py -->

## What I'd do again

Before reading a result, ask whether the test could have come out the other way. If the input transform cancels the perturbation, if the denominator is zero, if the feature enters at a scale the model can't see, or if the sample can't resolve the effect, the result isn't evidence either way. Retire the test, not the hypothesis, and say so in the log.

Scripts: `_gammaresid.py`, `_v4_testaudit.py`, `_v4std_knockout.py`, `_power.py`.
