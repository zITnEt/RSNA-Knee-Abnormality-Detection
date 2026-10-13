# RSNA Knee — DINOsaur V5 baseline reproduction

This repository tracks an exact, inference-only reproduction of Roman Tamrazov's public Kaggle notebook **RSNA Knee | DINOsaur V5? 🦖**, historical V10 (Kaggle script version `351856876`).

The baseline reproduction uses raw RSNA competition data and model assets on Kaggle. The ResNet50 experiment below also supports torchvision's pretrained weight cache; its Kaggle notebook loads attached weights offline.

Key files:

- `kaggle/dinosaur_v5_reproduction.ipynb` — exact downloaded V10 notebook source.
- `manifests/DINOSAUR_V5_MANIFEST.md` — dependency and reproducibility audit.
- `scripts/validate_submission.py` — strict submission validator.
- `scripts/check_environment.py` — safe environment/authentication check.
- `results/dinosaur_v5_baseline/` — small run artifacts only.

The published V10 run's optional ConvNeXt arm failed closed because its manifest was absent. Its scored `submission.csv` therefore remained the exact 0.943 Speedy Raptors anchor. No model substitution or retraining is part of BASELINE-001.

## Raptor v5 + ImageNet ResNet50 experiment

The `codex/raptor-v5-resnet50-fusion` branch adds a trainable model in
`models/raptor_v5_resnet50.py` with residual bottleneck fusion (separate
LayerNorms, 1×1 down, depthwise 3×3, GELU, 1×1 up; ≈1.2M parameters) of
ImageNet-1K V2 ResNet50 conv3_x/conv5_x into CoAtNet Stage2/Stage4. This child
branch of `codex/raptor-v5-radimagenet-fusion` replaces the RadImageNet weights
and swaps the parent's ≈14.7M-parameter three-layer MLP fusion for the lighter
bottleneck, keeping the same fusion points and training settings. Both backbones
receive the same complete three-channel 384×384 triplet and remain unfrozen;
ResNet50 does not select or repeat the middle slice. `raptor_data/series_triplets.py`
keeps every triplet within one MRI series while preserving the dense triplet
count, sampling extra slices when available. See
[the model guide](docs/RAPTOR_V5_RESNET50.md) and the self-contained
`kaggle/raptor_v5_resnet50_fusion.ipynb` for checkpoint loading and usage.
