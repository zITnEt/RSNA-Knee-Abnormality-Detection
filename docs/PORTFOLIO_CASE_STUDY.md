# Building a reproducible knee MRI machine learning pipeline

**Author:** Zafar Nematov  
**Role:** Machine Learning Engineer / Computer Vision Engineer  
**Project type:** Independent research portfolio using a public Kaggle competition

## The problem

A knee MRI study contains multiple series and a variable number of slices. A useful research pipeline must turn these inputs into a consistent representation, aggregate evidence across the study, and predict 12 findings. Model quality alone is insufficient: data provenance, reproducible preprocessing, interrupted GPU sessions, label uncertainty, and reliable evaluation all affect the result.

## The work delivered

The project began with an audited reproduction of a credited public baseline, then developed three model variants and the tooling around them:

- DICOM preprocessing with physical slice ordering, series ownership checks, documented crop/normalization rules, and alternative six-slot, series-safe, and minimum-overlap recipes.
- Study-bag data loaders and attention-based multiple-instance learning for variable-length inputs.
- An early RadImageNet fusion model, a gated ImageNet ResNet50 fusion model, and a CoAtNet-only simplification with strict checkpoint migration.
- Mixed-precision training, DDP experiments, gradient checkpointing, BatchNorm handling, EMA evaluation, atomic resumable checkpoints, and deterministic split bookkeeping.
- Kaggle notebook generation, sharded preprocessing/publication tooling, cloud launch scripts, inference export, and submission validation.
- Statistical reporting with masked labels, paired study bootstrap intervals, calibration, per-finding analysis, ablations, and checkpoint identity checks.

## Technical decisions

**Preserve series provenance.** The series-aware loaders construct windows inside their source series and record sampling assumptions. The historical six-slot export is documented separately because its original windowing behavior differs.

**Aggregate at study level.** The model encodes slice triplets and applies a per-finding attention-MIL head. Training can sample study windows while evaluation encodes complete bags in manageable chunks.

**Start fusion from the parent.** Zero-initialized residual output projections preserve the loaded parent at initialization. Tests check routing, gradients, optimizer coverage, checkpoint loading, and the original attention-head contract.

**Treat training operations as part of the model.** Small-batch BatchNorm behavior, recomputation, raw versus EMA buffers, and interrupted sessions can change results. The project implements explicit handling and regression checks for these cases.

**Audit results rather than selecting attractive numbers.** The evaluation distinguishes validation-selected checkpoints from an epoch that happened to peak on Gold74. It compares models using paired resamples and reports uncertainty and provenance limits.

## Outcomes

The publication brings together the source from all three development worktrees, runnable synthetic checks, model/data documentation, and selected aggregate evidence. The recorded single-split ResNet training used 3,900 training studies, 433 validation studies, and a separate 74-study Gold evaluation set.

The validation-selected EMA checkpoint scored **0.925106 macro ROC-AUC** on Gold74 (95% study-bootstrap interval 0.900750–0.946106). Its paired difference from the parent was +0.002494 (−0.009237 to +0.012958), so a reliable gain is not established. The practical result is an inspectable pipeline and an evidence-based experiment audit, with the limits visible.

The baseline's historical 0.943 leaderboard anchor is upstream work, credited to its authors; it is not a custom-model performance claim. No competition rank, client outcome, or clinical benefit is asserted.

## What a client can assess

This repository provides concrete examples of computer vision implementation, medical-image data engineering, GPU training workflows, reproducibility, and technical reporting. These skills transfer to projects involving custom PyTorch models, large image datasets, model evaluation, and repeatable research pipelines.

Review the [code guide](REPOSITORY_GUIDE.md), [reproduction instructions](REPRODUCIBILITY.md), and [statistical report](../results/resnet50_20261013/REPORT.md).
