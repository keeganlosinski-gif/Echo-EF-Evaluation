[Echo-EF-Evaluation-README (1).md](https://github.com/user-attachments/files/34098856/Echo-EF-Evaluation-README.1.md)
# Echo EF evaluation: dataset audit and error analysis

Keegan Losinski, RDCS (registered diagnostic cardiac sonographer).
Six days of work, Aug 23 to 28 2026, with one rerun on Aug 31. Research project, not a clinical tool, not peer reviewed, not clinically validated. Methods, findings, a casebook and the cleaned-up scripts are here. The datasets themselves are not (see CITATIONS.md).

## What this is

Ejection fraction (EF) is the share of blood the left ventricle pushes out with each beat. It's the number on an echo report that most often changes treatment. I built a pipeline that estimates it from ultrasound video, using two public datasets: Stanford's EchoNet-Dynamic (10,030 apical four-chamber videos, each with a stated EF and expert tracings) and CAMUS (500 patients, two views, expert outlines of the ventricle at the start and end of the beat). The pipeline outlines the ventricle, measures EF from that outline, and uses a small model that looks across the whole heartbeat. The models are ordinary and I don't think they're the interesting part.

What I'd want someone to look at is the order I did things in. Nothing got used until it had been checked against something it couldn't have produced itself.

- The labels were checked before any model saw them. EchoNet's stated EF is calculated from the same tracings it ships, so I recalculated it and compared (average difference 1.05 points across 10,024 videos).
- The EF measurement was run on the experts' own outlines before a model existed, so I knew the best error it could ever reach.
- The quality filter was checked for good videos it was wrongly throwing out before I let it remove anything. It was about to reject 1,797 valid tracings because they were stored in the opposite order.
- Every anomaly detector was tested on a known example before I quoted a count from it.
- Every model number in the writeup names the log it came from. The two I couldn't tie to a log are listed as defects instead of printed.

## What came out of doing it in that order

**CAMUS's two ground truths disagree.** EF calculated from its own expert outlines sits about 8 points above the EF it states for the same patients. Two independent measurement methods agree on the gap (+8.45 and +7.67), and a trained model later reproduced it end to end (+8.08). The stated EF wasn't derived from the outlines that ship with it, so any EF result on CAMUS has to say which of the two it was scored against.

**Three videos the dataset doesn't flag,** one of each in 10,030: a tracing drawn on the right ventricle instead of the left, a split-screen image with two views side by side, and a scan rotated about 45 degrees. I found each by looking at the frames, then counted them programmatically. None of them moves an average, but each one makes its own study wrong.

**A detector I trusted was wrong.** My first split-screen check flagged 644 videos. Before quoting that number I looked at the top 16, and every one was a normal scan. I rebuilt the check, confirmed it on the one real split-screen video first (score 0.79 against 0.02 for a normal one), then ran it over 9,589 videos. It found one true case and one false alarm.

## The models

The outlining model's boundaries land about 1.55 mm from the experts' on CAMUS. The EF model is off by 4.85 points on average across the 1,219 EchoNet test videos that passed the quality checks.

It also estimates its own uncertainty, and that estimate is useful. On the quarter of cases it's most confident about, the error is 3.7. On the quarter it's least confident about, it's 7.0. That's the part I'd actually deploy: report the confident ones and flag the rest as technically limited.

On a second dataset the picture changes. An average of three training runs scores 4.73 on EchoNet and 4.04 on CAMUS. For reference, the published EchoNet-Dynamic model went from 4.05 on its own test set to 6.0 on its outside set. The version of mine that scored best on EchoNet (4.49) did worst on CAMUS (6.91). CAMUS was also part of my training data, so that 4.04 is not an out-of-sample result. Every model gets its own row, with every caveat, in FINDINGS.md.

## What I'd carry into evaluating anything else

Check the labels and measure the best possible error before building a model. It's easy to do it the other way around and only find the ceiling after the model is built.

A test has to be able to fail. I dropped one experiment because a preprocessing step cancelled out the effect it was meant to test, and I treated a check that passed on zero items as a bug. If you can't say what failure would look like before you run something, the result doesn't tell you anything.

Confirm a detector on a known example before you quote its count. The split-screen check would have put a false number in this writeup if I hadn't looked at its top hits first.

Send a person only what a script can't decide, and write the rules for deciding before you look. An automated edge check cleared 95.9 percent of frames, and 31 came to me. I wrote the criteria for those 31 before opening them, so I couldn't talk myself into whatever I saw.

Choose between versions on the validation set and look at the test set once. When a result comes back null, ask whether the test could have detected the effect at all. One of mine couldn't, and I retracted the conclusion I'd written. The retraction stays in the log with the reason. Averages hide things too. An overall error of 4.7 was hiding a 9.3 in the 35 to 45 EF range, which is the range where the number changes what a cardiologist does.

## Limitations

Six days, one person. Every anatomical call (the right ventricle tracing, the 31 suspect frames, the 12 confident failures) is one credentialed reader, not a panel.

EF model numbers are on the 1,219 test videos that passed the checks (57 of the official 1,276 were excluded for defective ground truth). Frame-based numbers and the published figure are on all 1,276. Every comparison says which.

The outside-dataset comparison uses a different outside set than the published work did, so the claim is about how much each model degrades on new data, not a head-to-head error comparison.

The anomaly checks looked for the wrong type of image and the wrong chamber. I didn't check for foreshortened or off-axis views, which are the most common real-world quality problem in four-chamber echo and would need a sonographer to read each video.

The Aug 31 rerun moved the headline numbers by hundredths of a Dice point and tenths of an EF point. I quote the Aug 24 values that the named logs carry and list the rerun in FINDINGS.md rather than pick the better one.

## Map

| Path | What it is |
| --- | --- |
| METHODS.md | Pipeline stages, validation order, quality-check definitions, review protocol, reporting rules |
| FINDINGS.md | Every result with its source log, the retractions, the open defects |
| casebook/ | Six short cases, each ending in the rule it produced |
| scripts/ | The analysis scripts, cleaned up (paths via environment variables), indexed in scripts/README.md |
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

The scripts expect three environment variables: `ECHONET_ROOT` (the EchoNet-Dynamic folder with `FileList.csv`, `VolumeTracings.csv`, `Videos/`), `CAMUS_ROOT` (the CAMUS `database_nifti` folder) and `WORK_DIR` (scratch space for caches and outputs). Both datasets have to come from their distributors under their own terms. The study identifiers for the anomalies and the list of videos that passed the checks are derived from the dataset and aren't in this repo. Anyone with the dataset can regenerate both by running the quality-check scripts.

The code in this repo was written with Claude Code. The design decisions, the validation order, the clinical reads and the writeup are mine.
