[Echo-EF-Evaluation-README.md](https://github.com/user-attachments/files/31869501/Echo-EF-Evaluation-README.md)
# Echo EF evaluation: check the ruler before you trust the score

Keegan Losinski, RDCS (registered diagnostic cardiac sonographer).
Six days of work, Aug 23 to 28 2026, one rerun Aug 31. Research project, not a clinical tool, not peer reviewed, not clinically validated. Methods, findings, a casebook and the sanitised scripts are here. No dataset content is (see CITATIONS.md).

## What this is

I built a left ventricular ejection fraction pipeline on two public echo datasets, EchoNet-Dynamic (10,030 apical four chamber videos with a stated EF and volume tracings) and CAMUS (500 patients, two views, expert masks at ED and ES). Segmenter, geometric EF readout, small temporal model. The models are ordinary and I don't think they're the interesting part.

What I'd want someone to look at is the order I did things in. Nothing got used until it had been checked against something it couldn't have produced itself. The labels were checked against an analytic identity before any model saw them. The EF readout was run on ground truth masks before a model existed, so I knew its error floor going in. The quality gate got audited for false positives before I let it throw anything away. Every anomaly detector was tested on a known positive before I quoted a count from it. Every model number in the writeup names the log it came from, and the two I couldn't tie to a log are listed as defects instead of printed.

## What came out of doing it in that order

**CAMUS's two ground truths disagree.** EF computed from its own expert masks sits about 8 points above the EF it states for the same patients. Two geometrically independent estimators agree on it (+8.45 disk Simpson, +7.67 area length) and a trained model later reproduced it end to end (+8.08). The stated EF wasn't derived from the shipped contours, so any EF number on CAMUS has to say which of the two it's scored against.

**Three anomalies the dataset doesn't flag,** each exactly one instance in 10,030: a tracing that sits on the right ventricle, a split screen dual view composite, and a scan sector rotated about 45 degrees. All three found by looking at rendered frames, then counted programmatically. Invisible to any average, fatal to the one study they belong to.

**A detector I trusted was wrong.** My first split screen metric said 644 suspects. I rendered its top 16 before quoting it and every one was a normal sector. The replacement was checked on the one known composite first (line score 0.79 vs 0.02 for its neighbour), then run over 9,589 videos: one true, one false positive.

## The models, briefly

Segmenter at about 1.55 mm boundary error on CAMUS. Temporal EF model at MAE 4.85 on the verified EchoNet test set (n=1,219), and its uncertainty head separates a crisp quartile at 3.7 from a difficult quartile at 7.0, which is the part I'd actually deploy: report the crisp ones, flag the rest as technically limited. The 3 seed ensemble reads 4.73 internal and 4.04 on CAMUS, where the published EchoNet-Dynamic model went from 4.05 on its own test set to 6.0 on its external set. The variant that scored best internally (4.49) transferred worst (6.91). One model per row, every caveat, in FINDINGS.md.

## What I'd carry into evaluating anything else

Audit the labels and measure the readout's floor before touching a model. This sounds obvious but most pipelines I've seen do it the other way around and then discover the floor after the model is already built.

A test has to be able to fail. A gain sweep that couldn't fail (per image standardisation cancels it) got dropped. A gate that passed on 0 of 0 items got treated as a bug. If you can't describe what failure looks like before you run it, the result doesn't mean anything.

Validate a detector on a known positive before quoting its count. This is the 644 to 1 lesson. I would have published a false number if I hadn't rendered the top hits first.

Send a human only what a script can't decide, and write the protocol before looking. A null controlled edge check cleared 95.9 percent of frames. 31 went to me. Writing the adjudication criteria before opening the images keeps you from rationalizing what you're seeing.

Pick arms on validation, read test once, and when a null result comes in ask whether the test could have detected the effect. One of mine couldn't, and the conclusion I'd written down got retracted. Pooled numbers hide structure too. A pooled MAE of 4.7 was hiding a 9.3 in the 35 to 45 EF band, which is the band where the number changes what a cardiologist does.

Retractions stay in the log with the reason.

This isn't a trust and safety project and I don't present it as one. It's the closest thing I can show in full to how I'd run an evaluation of anything.

## Limitations

Six days, one person. Every anatomical call (the RV tracing, the 31 suspect frames, the 12 confident failures) is one credentialed reader, not a panel.

Temporal model numbers are on the 1,219 verified test videos (57 of the official 1,276 excluded by the gates for defective ground truth). Frame based numbers and the published figure are on all 1,276. Every comparison says which.

The external comparison uses a different external set than the published work did, so the claim is about degradation under distribution shift, not a head to head MAE.

The anomaly screens were for acquisition type and chamber identity. I didn't screen for foreshortened or off axis apical views, which are the commonest real world quality problem in A4C echo and would need a sonographer read per video.

An Aug 31 rerun moved headline numbers by hundredths of Dice and tenths of an EF point. I quote the Aug 24 values the named logs carry and list the rerun in FINDINGS.md rather than pick the better one.

## Map

| Path | What it is |
| --- | --- |
| METHODS.md | Pipeline stages, validation order, gate definitions, adjudication protocol, reporting rules |
| FINDINGS.md | Every result with its receipt, the retractions, the open defects |
| casebook/ | Six short cases, each ending in the rule it produced |
| scripts/ | The analysis scripts, sanitised (paths via environment variables), indexed in scripts/README.md |
| figures/ | Intentionally empty, see figures/README.md |
| CITATIONS.md | Dataset citations and licence notes |
| LICENSE | MIT for the code and text here. The datasets keep their own terms |

## Casebook

1. Label parity before modelling
2. Two ground truths that disagree
3. Anomalies the dataset never flagged
4. Audit the gate before you apply it
5. A test that can't fail isn't a test
6. Internal vs external, one model per row

## Reproducing

The scripts expect three environment variables: `ECHONET_ROOT` (the EchoNet-Dynamic folder with `FileList.csv`, `VolumeTracings.csv`, `Videos/`), `CAMUS_ROOT` (the CAMUS `database_nifti` folder) and `WORK_DIR` (scratch for caches and outputs). Both datasets have to come from their distributors under their own terms. Study identifiers for the anomalies and the verified subset list are derived annotations on the dataset and aren't in this repo; anyone holding the dataset can regenerate both by running the gate scripts.

The code in this repo was written with Claude Code. The design decisions, the validation order, the clinical reads and the writeup are mine.
