# Gated ResNet50 + CoAtNet experiment

This directory is a complete source snapshot of the latest gated ImageNet ResNet50 fusion worktree, including its uncommitted cloud training, CV, prediction, and submission tools. Run commands from this directory to use its own `models` and `scripts` packages.

The model in [models/raptor_v5_resnet50.py](models/raptor_v5_resnet50.py) combines CoAtNet with ImageNet ResNet50 through zero-initialized gated residual bottlenecks at two stages, retaining the per-finding attention-MIL head. It differs from the root's earlier RadImageNet MLP-fusion experiment.

Useful entry points:

- [Architecture details](docs/RAPTOR_V5_RESNET50.md)
- [Single-split trainer](scripts/train_resnet50_cloud.py)
- [Five-fold trainer](scripts/train_resnet50_cv5.py)
- [Full non-Gold training](scripts/train_resnet50_full_gold4.py)
- [Inference](scripts/infer_cv5_submission.py)
- [Recorded-run statistical analysis](scripts/analyze_training_evidence.py)
- [Recorded aggregate results](../../results/resnet50_20261013/REPORT.md)

Install the repository's root `requirements.txt` into your environment. See the [shared reproduction guide](../../docs/REPRODUCIBILITY.md) and [model card](../../docs/MODEL_CARD.md) for input requirements and interpretation. Historical launchers need your own runtime configuration and authorized assets.

The selected EMA checkpoint's recorded Gold74 macro ROC-AUC is 0.925106. The paired interval against its parent crosses zero; no reliable fusion gain is established.
