# RSNA Knee — DINOsaur V5 baseline reproduction

This repository tracks an exact, inference-only reproduction of Roman Tamrazov's public Kaggle notebook **RSNA Knee | DINOsaur V5? 🦖**, historical V10 (Kaggle script version `351856876`).

The baseline reproduction uses raw RSNA competition data and model assets on Kaggle. The CoAtNet-only model below loads attached or local Raptor checkpoints offline.

Key files:

- `kaggle/dinosaur_v5_reproduction.ipynb` — exact downloaded V10 notebook source.
- `manifests/DINOSAUR_V5_MANIFEST.md` — dependency and reproducibility audit.
- `scripts/validate_submission.py` — strict submission validator.
- `scripts/check_environment.py` — safe environment/authentication check.
- `results/dinosaur_v5_baseline/` — small run artifacts only.

The published V10 run's optional ConvNeXt arm failed closed because its manifest was absent. Its scored `submission.csv` therefore remained the exact 0.943 Speedy Raptors anchor. No model substitution or retraining is part of BASELINE-001.

## CoAtNet-only Raptor v5

The `codex/raptor-v5-coatnet-only` child branch removes the ResNet encoder and
both fusion blocks. `models/raptor_v5.py` keeps the existing CoAtNet backbone,
original per-finding attention-MIL head, 12-label order and preprocessed
`[B,K,3,384,384]` input contract. Existing raw or EMA fusion checkpoints can
supply their trained CoAtNet/head weights, with all fusion and ResNet tensors
discarded. Optimizers now contain only CoAtNet and head groups.

See [the model guide](docs/RAPTOR_V5_COATNET.md), the self-contained
`kaggle/raptor_v5_coatnet.ipynb`, and `scripts/strip_raptor_fusion.py` for
loading, training-step examples and offline checkpoint export. The historical
fusion report remains under `results/resnet50_20261013`.
