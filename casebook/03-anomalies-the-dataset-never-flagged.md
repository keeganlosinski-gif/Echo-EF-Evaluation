# Case 3: Anomalies the dataset never flagged

What's in a 10,030 video benchmark that no aggregate metric will ever show you? EchoNet-Dynamic ships no quality or protocol flag of any kind. All three of these were found by rendering frames and looking, then counted programmatically. <!-- src: FINDINGS.md sec 9 lines 150-152 -->

## Three cases, each unique in the dataset

**A tracing on the wrong chamber.** One video's shipped chords sit on the right ventricle in both traced frames of a normal apex up A4C. I read it as a sonographer and it's definitively an RV. I considered dextrocardia (mirrored heart) and rejected it as an implausible inclusion for this dataset. I ruled out intent from the dataset's own documentation: tracings are defined as LV endocardium only, there's no chamber field, there's no errata. To prove the anomaly was in the data and not in my rasteriser I drew the raw CSV chords straight onto the video with no reconstruction step in the loop. Rate: one confirmed in 10,024. That's a count, not an extrapolation. <!-- src: FINDINGS.md sec 9.1 lines 154-163; _rv_evidence.py -->

**A split screen composite.** Two sectors side by side with a bright divider, the kind of dual view export some scanners produce. I saw it on a rendered panel. Even if the left half tracing were a correct LV, everything geometric computes at quarter scale on a composite, so the video can't be used. Excluded. <!-- src: FINDINGS.md sec 9.2 lines 166-172 -->

**A scan region rotated about 45 degrees.** A rotated export, not a different probe. It broke the orientation detector's assumptions and it's the only one in the dataset. Excluded. <!-- src: FINDINGS.md sec 9.2 lines 173-175 -->

## The detector that was wrong

Finding the composite made me want a census, so I wrote a column brightness seam metric and it returned 644 suspects. That would have been a headline. Before quoting it I rendered its top 16 hits and all of them were normal sectors. It was measuring something, but not split screens. Discarded. <!-- src: FINDINGS.md sec 9.2 lines 170-172; _acqtypes.py docstring -->

The replacement is a divider line detector, and this time I checked it on the one known positive before running it: line score 0.79 on the composite vs 0.02 on its nearest neighbour. Then over all 9,589 candidate videos: it found the known composite and one false positive (a bright septum). One. <!-- src: FINDINGS.md sec 9.2 lines 166-172 -->

## A screen for one thing found another

The wrong chamber screen (`scripts/_chamber_screen.py`) takes the Mahalanobis distance of each mask centroid from the population and renders the worst. Its top hits weren't label errors, they were the composite and the rotated export. A screen I built for laterality turned out to be an out of protocol acquisition detector. <!-- src: FINDINGS.md sec 9 lines 181-183 -->

## Are they in the data, or in my copy of it?

Before saying any of this out loud I wanted to rule out my own copy, my decoder and my rasteriser, because a transfer glitch or a bad decode would be a much more boring explanation. Four checks, all of which had to agree:

1. **Inside the pipeline.** The container header and the dataset's own frame counts agree for all three videos, two different decoders produce the same frames, and for the RV case the raw CSV chords drawn straight onto the decoded frames land on the right ventricle with no reconstruction step involved. Control videos run through the same path look normal.
2. **Against an independent copy.** My working copy is the archive I downloaded from the Stanford AIMI portal on Aug 23. A public re-upload of the same archive publishes a SHA-256 for every large file, and those match my portal copy's three videos and VolumeTracings.csv exactly, without downloading anything.
3. **Fresh bytes.** I then pulled the three videos, VolumeTracings.csv and FileList.csv from that independent copy and compared MD5 and size. All five byte identical to the portal copy.
4. **Render from the fresh bytes.** Same three anomalies.

So every hypothesis that puts the defect in my copy, my decode or my rasteriser is dead. The portal archive on my disk and an independent mirror agree byte for byte on all three videos and both label tables. The RV tracing, the split screen composite and the rotated export are in the files as distributed.

The caveat that goes with that: the distributor publishes no checksums, so the cross-check is against a third party's re-upload rather than Stanford's own server. Two independently obtained copies agreeing byte for byte is as strong as this gets without an official checksum list. The hashes aren't in this repo, since together with the dataset they'd identify the three videos as precisely as the study IDs would, but they're available to the maintainers on request, and the maintainers can compare against their own archive.

## What I'd do again

Validate any anomaly detector on a known positive before quoting its count, because a detector that has never seen a positive has never been tested. Render the top hits before believing a number (644 to 1 cost ten minutes). Report rare anomalies as counts, not rates. And look at the images, because two of the biggest bugs in this project (this one and the orientation trap in Case 1) were invisible in the metrics and obvious on a rendered panel.

Study identifiers for the three cases are derived annotations on the dataset and aren't in this repo. Anyone holding the dataset can regenerate them with the screens above.

Scripts: `_chamber_screen.py`, `_acqtypes.py`, `_qc_masks.py` (G10 and G11).
