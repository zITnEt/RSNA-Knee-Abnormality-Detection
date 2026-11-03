# Published evaluation evidence

This directory includes selected **aggregate** reports, metric tables, and figures from the existing research. It does not include the model checkpoints, source label tables, MRI data, per-study predictions, error queues, or private inference caches needed to recompute every report.

- `resnet50_20261013/`: gated ImageNet ResNet50 experiment; the portfolio's principal evidence. Selected raw/EMA checkpoints were chosen by validation, and uncertainty against the parent is reported explicitly.
- `training_analysis/`: earlier component diagnostics for the credited public baseline. Gold58 component metrics do not validate the complete submitted ensemble and are separate from Gold74.

Historical report text may refer to research files that are intentionally excluded from this publication tree. The aggregate tables and figures accompanying each report are retained for review. No new model performance measurements were produced while preparing this portfolio.

Read the [model card](../docs/MODEL_CARD.md), [data card](../docs/DATA_CARD.md), and [publication checks](../docs/VALIDATION.md) for context.
