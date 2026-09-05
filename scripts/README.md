# Scripts

These are the project's actual analysis scripts, sanitised: absolute paths were replaced by three environment variables; a few docstring and print strings were reworded (references to unrelated work removed, one challenge name genericised); and one loader default was changed. A diff against the originals is available. They are research scripts, not a package. They are runnable in principle against the datasets obtained from their distributors; they have not been re-run from this repository.

```
ECHONET_ROOT   EchoNet-Dynamic folder: FileList.csv, VolumeTracings.csv, Videos/
CAMUS_ROOT     CAMUS folder containing database_nifti/ (the scripts point at database_nifti itself)
WORK_DIR       scratch directory for caches (*.npy, *.npz, *.csv), checkpoints (*.pt) and rendered panels
```

Dependencies: numpy, pandas, scipy, opencv-python, nibabel, torch, timm, scikit-learn. Several scripts read caches produced by the project's build scripts (`en_meta.csv`, `en_img.npy`, `en_msk.npy`, token caches under `WORK_DIR/tokens*`); those builders are not included because they are ordinary data plumbing, and the caches themselves are dataset content.

Scripts that render dataset frames (`_adjudicate.py`, `_chamber_screen.py`, `_acqtypes.py`) write PNG panels to `WORK_DIR`. Those panels are dataset content and are not in this repository.

## Library modules

| File | What it does |
|---|---|
| `ef_readout.py` | Stage 5, built first: area-length volume and EF from a cavity mask (monoplane and biplane). Validated on ground-truth masks before any model. |
| `simpson.py` | Biplane Simpson's method of disks from CAMUS masks with real pixel spacing. The parity gate. |
| `echonet_masks.py` | Chord-to-mask reconstruction, including the per-chord quad fill that replaced a self-crossing polygon fill. |
| `echonet_data.py` | EchoNet loader. The docstring records the design decisions: phase normalisation rather than frame-rate normalisation, sample not average, frame rate excluded as an input, official splits only. |
| `calibrate.py` | Multi-anchor intensity calibration with a monotonicity guard; the docstring records why two anchors were not enough. |
| `unet.py` | U-Net with a timm encoder; Dice loss and per-class Dice. |
| `train_tok.py` | Stage 4 temporal model: attention over frozen per-frame tokens, heteroscedastic (EF, sigma) head, Gaussian or Laplace NLL, ablation switches via environment variables. |

## Validation before modelling

| File | What it does |
|---|---|
| `_enparity.py` | Recompute EF from EchoNet's chords by monoplane Simpson and compare with the stated EF. Validates the labels and the arithmetic in one step. |
| `_leakchk.py` | Is there a patient identifier? Duplicate label tuples across splits? Split sizes. |
| `_validate_readout.py` | Run the read-out on ground-truth masks: the floor any model can reach. |
| `_arealen.py` | The decisive fork for the CAMUS gap: a geometrically independent estimator, with the verdict logic written before it ran. |
| `_calverify.py` | Held-out-knot verification of the calibration map, against the circular in-sample test. |

## Gates, label checks and adjudication

| File | What it does |
|---|---|
| `_qc_masks.py` | The G1-G9 gate ledger, with the two audited fixes marked inline. |
| `_qc_audit.py` | Audit of the gate itself: reversed chord order, solidity threshold. |
| `_edgeagree.py` | Tracing-versus-image edge agreement with an erode/dilate null control; flags only frames that fail the null band. |
| `_adjudicate.py` | Builds the 60-tile stratified human panel and writes the protocol. |
| `_chamber_screen.py` | Mahalanobis centroid screen for wrong-chamber tracings (which found acquisition anomalies). |
| `_acqtypes.py` | Acquisition-type classification (rotated export, split-screen) and the rendering that falsified the 644-suspect census. Expects a `composite_suspects.csv` from the earlier seam metric in `WORK_DIR`. |
| `_preflight.py` | PASS/FAIL preflight before any GPU stage; any FAIL blocks the build. |

## Tests of tests, and measurement

| File | What it does |
|---|---|
| `_gammaresid.py` | What the network sees after standardisation under gain, offset and gamma: which robustness sweeps can fail. |
| `_dicemm.py` | Convert Dice to millimetres of boundary error using each class's perimeter-to-area ratio. |
| `_power.py` | Hanley-McNeil power analysis for an external benchmark's test set. |
| `_v4_testaudit.py` | Was the descriptor-feature test a good test: feature scales, physical sanity, input knockout. |
| `_v4std_knockout.py` | The same knockout after per-dimension standardisation. |

## Evaluation structure

| File | What it does |
|---|---|
| `_clin_audit.py` | EF-band MAE and bias, 3-class clinical agreement, sigma reliability by decile. |
| `_recal.py` | VAL-fitted isotonic recalibration applied once to TEST. |
| `_lds_bands.py` | Inverse-density weighting judged on bands, as pre-registered. |

## Not included, and why

- `_find_diamond.py`, `_rv_evidence.py`, `_rend3.py`, `_ids12.py`: embed dataset study identifiers.
- Cache builders, token-cache builders, the training launchers and the per-arm evaluation scripts: data plumbing with no methodological content beyond what the included scripts show.
- `adjudication_key.csv`, `en_meta.csv`, all `.npy` / `.npz` / `.pt` artefacts: dataset content or derived from it.
