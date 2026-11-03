# A short code review route

| Area | Files worth reading | What to look for |
| --- | --- | --- |
| Modern fusion model | [ResNet50 architecture](../experiments/resnet50_fusion/models/raptor_v5_resnet50.py) | Gated residual feature fusion, zero initialization, strict checkpoint loading, and optimizer groups. |
| Training reliability | [Cloud trainer](../experiments/resnet50_fusion/scripts/train_resnet50_cloud.py) | EMA state, fixed BatchNorm statistics, atomic writes, study split, and resumable state. |
| Cross-validation | [CV trainer](../experiments/resnet50_fusion/scripts/train_resnet50_cv5.py) | Fold provenance, selection on validation, and checkpoint bookkeeping. |
| Medical-image preprocessing | [Minimum-overlap preprocessing](../scripts/preprocess_min_overlap_384.py) | Sampling capacity, within-series windows, recipe metadata, and shard generation. |
| Variable-length studies | [Series triplets](../raptor_data/series_triplets.py) | Series allocation, shortage behavior, ordering, and audit records. |
| Evaluation | [Recorded-run analysis](../experiments/resnet50_fusion/scripts/analyze_training_evidence.py) | Masked labels, paired bootstrap, calibration, and evidence checks. |
| Simplification/migration | [CoAtNet model](../experiments/coatnet_only/models/raptor_v5.py), [checkpoint export](../experiments/coatnet_only/scripts/strip_raptor_fusion.py) | Removing fusion while retaining the parent head and strict tensor loading. |
| Contracts | [Submission validator](../scripts/validate_submission.py), [data-free demo](../scripts/portfolio_demo.py) | Column/row ordering, IDs, numeric/finite scores, and probability range checks. |

The `experiments` directories are complete source snapshots. Run each from its own directory; package names deliberately match the original code. Historical operational docs and launchers need your own runtime paths and assets.

Source provenance: [SOURCE_SNAPSHOT.json](../manifests/SOURCE_SNAPSHOT.json). Portfolio preparation did not retrain models, recompute the recorded medical-image results, or alter the original research worktrees.
