# Citations and data terms

**Nothing from either dataset is redistributed in this repository**: no videos, frames, masks, tracings, label tables, cached arrays, per-study results, or study identifiers. The scripts read the datasets from locations you supply via environment variables after obtaining the data from the distributors under their own terms.

## EchoNet-Dynamic

- Distributor: Stanford University School of Medicine, Center for Artificial Intelligence in Medicine and Imaging (AIMI).
- Access: by registration with the distributor, for research use under the distributor's data use terms. Those terms do not permit redistribution, and none is made here. The data used in this project were downloaded from the official distribution point, the Stanford AIMI shared datasets portal, after sign-in and acceptance of the research use terms (Aug 23 2026). A copy from a public mirror was used once, for a hash check on three videos and the label tables (see casebook Case 3), never for any result.
- Citation: Ouyang D, He B, Ghorbani A, Yuan N, Ebinger J, Langlotz CP, Heidenreich PA, Harrington RA, Liang DH, Ashley EA, Zou JY. Video-based AI for beat-to-beat assessment of cardiac function. *Nature* 580, 252-256 (2020).
- The published internal (4.05) and external (6.0) MAE figures quoted in this repository are as reported in that paper, not re-measured here.

## CAMUS

- Distributor: CREATIS laboratory (Lyon, France); the acquisitions come from the University Hospital of St-Etienne.
- Access: by registration on the CAMUS challenge site, for research use under the distributor's terms. No redistribution is made here.
- Citation: Leclerc S, Smistad E, Pedrosa J, Ostvik A, Cervenansky F, Espinosa F, Espeland T, Berg EAR, Jodoin PM, Grenier T, Lartizien C, D'hooge J, Lovstakken L, Bernard O. Deep Learning for Segmentation Using an Open Large-Scale Dataset in 2D Echocardiography. *IEEE Transactions on Medical Imaging* 38(9), 2198-2210 (2019).
- The human intra-observer comparison in FINDINGS.md section 3 is indicative only: the project's notes carry a figure of about 1.53 mm for human intra-observer boundary variability on CAMUS, but the reference it was taken from was not recorded, so no citation is claimed for it.

## Other references used in the project

- Hanley JA, McNeil BJ. The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology* 143(1), 29-36 (1982). Used in `scripts/_power.py`.

## Study identifiers

The identifiers of the three anomalous videos and the list of the 9,587 verified videos are derived annotations on the dataset and are not published here. Holders of the EchoNet-Dynamic dataset can regenerate both by running `scripts/_qc_masks.py`, `scripts/_chamber_screen.py` and `scripts/_acqtypes.py` against their own copy.


