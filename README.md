# RSNA Knee — DINOsaur V5 baseline reproduction

This repository tracks an exact, inference-only reproduction of Roman Tamrazov's public Kaggle notebook **RSNA Knee | DINOsaur V5? 🦖**, historical V10 (Kaggle script version `351856876`).

The raw RSNA competition data and all model assets remain on Kaggle. Nothing in this repository downloads DICOMs or checkpoints locally.

Key files:

- `kaggle/dinosaur_v5_reproduction.ipynb` — exact downloaded V10 notebook source.
- `manifests/DINOSAUR_V5_MANIFEST.md` — dependency and reproducibility audit.
- `scripts/validate_submission.py` — strict submission validator.
- `scripts/check_environment.py` — safe environment/authentication check.
- `results/dinosaur_v5_baseline/` — small run artifacts only.

The published V10 run's optional ConvNeXt arm failed closed because its manifest was absent. Its scored `submission.csv` therefore remained the exact 0.943 Speedy Raptors anchor. No model substitution or retraining is part of BASELINE-001.

## Raptor v5 + RadImageNet experiment

The `codex/raptor-v5-radimagenet-fusion` branch adds a trainable model in
`models/raptor_v5_radimagenet.py` with residual concatenation-MLP fusion of
RadImageNet conv3_x/conv5_x into CoAtNet Stage2/Stage4. Both backbones receive
the same 384×384 triplet and remain unfrozen. `raptor_data/series_triplets.py`
keeps every triplet within one MRI series while preserving the dense triplet
count, sampling extra slices when available. See
[the model guide](docs/RAPTOR_V5_RADIMAGENET.md) and the self-contained
`kaggle/raptor_v5_radimagenet_fusion.ipynb` for checkpoint loading and usage.
