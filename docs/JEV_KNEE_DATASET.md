# JEV-Knee dataset audit

Verified and downloaded on 2026-10-09 using the existing Kaggle CLI and credentials.
No new dependency was required. No training pipeline was changed.

- Canonical URL: https://www.kaggle.com/datasets/riadmohamed42/jev-knee-labels
- Dataset identifier: `riadmohamed42/jev-knee-labels`
- Publisher confirmed by the Kaggle dataset API: RIAD MOHAMED
- Kaggle dataset ID: 12114723; downloaded version: 5, updated 2026-09-21.
- Declared license: CC0-1.0. This does not supersede terms for underlying
  competition data or the two source label datasets.
- Local root: `C:/Users/User/Documents/ChatGPT/RSNA Knee Abnormality Detection/data/external/jev_knee/`

Original extracted files and the downloaded archive are retained under `original/`.
Kaggle metadata is under `metadata/`. SHA-256 hashes, exact dimensions, every column,
per-column missing values, per-label ranges and ID overlap are recorded in
`validation_report.json`. Every original file was verified byte-for-byte against
the retained ZIP. The entire `data/` directory is already ignored by Git.

## Original files

| Filename | Bytes | Rows x columns | Findings |
| --- | ---: | ---: | --- |
| HYBRID_labels_scores.csv | 1,328,515 | 4,407 x 13 | Soft scores; 3 missing cells on one official gold study |
| FINAL2_labels_scores.csv | 1,389,334 | 4,407 x 13 | JEV-only soft scores; no missing cells |
| HYBRID_labels_binary.csv | 392,354 | 4,407 x 13 | Binary labels; no missing cells |
| FINAL2_labels_with_confidence.csv | 1,105,959 | 4,407 x 37 | Binary label columns plus agreement/tier columns |
| gold74_eval_set.csv | 12,121 | 74 x 15 | Evaluation annotations; 18 missing label cells |
| FINAL2_confidence_report.json | 1,330 | — | Publisher calibration/interval report |
| HYBRID_thresholds.json | 1,211 | — | Thresholds and blend information |
| metrics.json | 1,239 | — | Publisher metrics |
| README.md | 9,643 | — | Publisher method and license |
| cover.png | 55,940 | — | Dataset image |
| slop_audit.py | 4,871 | — | Publisher writing-audit script; not executed |
| slop_phrase_prob_adjustments.json | 19,010 | — | Writing-audit resource |

The ZIP is 1,471,337 bytes; the 12 extracted original files total 4,321,527 bytes.

## Schema, validity and coverage

The shared columns, in order, are:

`StudyInstanceUID`, `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`,
`Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`,
`Contusion`, `Fracture`.

The confidence table adds `<finding>_agree` and `<finding>_tier` for each of
these twelve findings, ordered as in the shared schema. The evaluation table
adds `source` and `note`. The JSON audit lists all columns explicitly.

All four full-size label CSVs contain exactly 4,407 unique study identifiers,
with zero missing IDs, duplicate IDs, unknown competition IDs or out-of-range/
non-finite probabilities. They each match all 4,407 IDs in the local official
`audit/training/labels/train.csv`; none is limited to 74 studies.

- HYBRID soft scores: observed global range 0.0047515572460509 to
  0.9992882361060444. ACL, MCL and Fracture each have one NaN, all on the same
  official gold study. Missing values were preserved.
- FINAL2 soft scores: observed global range 0.0000004521441544497127 to
  0.99094961023733; all 52,884 label cells are present.
- Both binary tables have only 0/1 label values. The confidence file's label
  columns are binary despite its README describing scores; they must not be
  substituted for FINAL2 soft scores.
- The evaluation file has 74 unique official training IDs: 58 tagged
  `original_gold58`, 9 `manual_verified`, and 7 `manual_verified_r2`.
  All 696 original-gold annotations agree exactly with local official metadata.
  Missing evaluation labels: Medial Meniscus 1, Medial OA 2, Lateral OA 3,
  PF OA 1, Synovitis 1, Baker's 7, Contusion 1, Fracture 2.
  Its 58 missing `note` cells are separate from the missing labels.

## Training exports

The validator generated two separate 4,333 x 13 CSVs:

- `prepared/hybrid_soft_train.csv`, from the original HYBRID soft scores.
- `prepared/final2_soft_train.csv`, from the original JEV-only soft scores.

Both exclude every study in `gold74_eval_set.csv` and every official fully
annotated study (the official 58 are a subset of the 74). Neither contains
missing targets, gold overrides, duplicate IDs or invalid probabilities.
No labels were filled with zero. The hybrid's missing row was excluded because
it is an evaluation study, not repaired. Identifiers use the official column
name and string type; the twelve columns keep competition names and order.

These exports are structurally ready for soft-target training after joining to
your MRI corpus by `StudyInstanceUID`. Any additional validation split in your
pipeline must also be excluded. This task did not inspect MRI availability or
integrate the files into the existing loader.

## Provenance and evaluation limitations

The shipped hybrid is 95% a per-finding selection of Steven Lee Hans
`llm_labels_v4_blend.csv` (9 findings) and Pilkwang `report_labels_v2.csv`
(ACL/MCL/Fracture), plus 5% JEV text ensemble. According to the publisher,
the JEV component combines six OpenJev report-reading configurations.

Gold labels were used to select the public source per finding, choose ensemble
weights, tune thresholds, and compare blends. Excluding the 74 evaluation IDs
from gradient updates does not remove this prior supervision or make Gold58
an independent test set. The publisher describes a seven-study holdout, but
also says its results informed rejection of a competing blend. It is too small
and selection-exposed to establish an unbiased gain. Sixteen additional
evaluation studies were labelled from reports, not independently annotated MRI.

An additional local check reproduced the advertised 0.9211643 HYBRID macro AUC
using per-target available labels across the 74-study evaluation file. Different
findings therefore use different subsets. The complete-case metric uses only
63 studies and is 0.91466205. For the original official Gold58, per-target
available HYBRID macro AUC is 0.90365270 (57 complete cases: 0.90445025).
FINAL2 alone scores 0.87447804 on all official 58. These describe label-to-gold
agreement, not the AUC of a trained MRI model. The publisher's headline should
not be interpreted as a 0.9212 score on your official Gold58.

## Reproducing the advertised AUC

```powershell
python scripts/reproduce_jev_knee_auc.py
```

The publisher's verification notebook was retrieved from
https://www.kaggle.com/code/riadmohamed42/jev-knee-labels-verification and
inspected without executing it. It joins HYBRID soft scores to the evaluation
CSV by UID, masks missing truth/score pairs separately for each finding, and
averages twelve ROC AUCs. Counts per finding are 73, 73, 73, 74, 72, 71, 73,
74, 73, 67, 73, 71, in competition target order. A single global count of 71
in `metrics.json` is not the denominator for every finding.

The local script verifies all twelve four-decimal values in `metrics.json`
with zero differences and the unrounded macro against `HYBRID_thresholds.json`
within 1e-12: **0.9211642973687014**. The publisher notebook averages rounded
per-finding values first, giving **0.9211666666666667**; both display as 0.9212.
No thresholding, missing-value replacement or MRI training is needed.
Input hashes, per-finding counts and the Gold58 comparison are saved in
`results/jev_knee/auc_reproduction.json` and `hybrid_per_finding_auc.csv`.

## Reusable validation

Run from the project root:

```powershell
python scripts/validate_jev_knee.py --prepare
```

Without `--prepare`, the script writes `validation_only_report.json` and does
not create training CSVs. It supports `--data-dir` and `--official-train` for
other locations. Existing generated outputs are reused only if byte-identical;
different existing outputs cause an error rather than being overwritten.
Original dataset files are read only. Missing retained targets or malformed
probabilities/IDs cause training preparation to fail instead of silently filling
or removing values.
