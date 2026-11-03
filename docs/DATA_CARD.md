# Data card

## External sources

The research uses the [RSNA Knee Abnormality Detection competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection), public pretrained assets, and [JEV knee labels](https://www.kaggle.com/datasets/riadmohamed42/jev-knee-labels). Access, permitted uses, and redistribution are governed by the respective sources. This repository does not distribute access rights.

The recorded project inventory contains 4,407 studies: 4,333 non-Gold studies and a 74-study Gold evaluation subset. Earlier baseline diagnostics used the 58 studies with all structured labels; that Gold58 analysis is a different evaluation from Gold74. HYBRID soft targets, HYBRID binary targets, and structured Gold labels must not be conflated.

## Preprocessing

Preprocessing selects series/slots, orders acquired slices, applies the documented spatial/intensity recipe, and builds 384 × 384 three-slice windows. The raw series-aware loader, six-slot export, series-safe export, and minimum-overlap export have different construction rules. Recipe identities and manifests are used to reject mismatched or incomplete shards.

The recorded large-scale exports use 32 shards. Real reproduction requires the correct recipe and version, complete manifests, and accessible tensor files. Source code can be reviewed and synthetic tests run without these data.

## Included in this publication

- Source code, notebooks with execution output cleared, configuration, and historical dependency/snapshot manifests.
- Fold assignments needed by the recorded CV workflow.
- Selected aggregate evaluation tables, plots, and narrative reports.

## Excluded from the current publication tree

MRI/DICOM files, clinical reports, external label tables, preprocessed study tensors, pretrained/fine-tuned weights, per-study predictions, error-review exports, credentials, and cloud runtime state. Historical repository commits and branches retain earlier research artifacts; the portfolio tree is the curated entry point.

Use synthetic data when contributing tests or demonstrating submission formatting. Do not upload participant data to an issue or portfolio gallery.
