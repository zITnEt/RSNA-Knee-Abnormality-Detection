# 5-fold CV fine-tune (2026-10-10)

* Model: Raptor V5 CoAtNet + RadImageNet ResNet-50 residual fusion, initialised from the parent
  (zero-init fusion, so step 0 = parent Raptor). Same recipe as `fusion_ddp_run1` except the items below.
* Codex 1: checkpoints hold CPU copies of the weights; `fusion_best.pt` is re-read and compared with
  the selected (raw or EMA) weights. Codex 6: `rad_input='pm1_middle'` (RadImageNet trunk gets the
  middle slice in [-1, 1], replicated to 3 channels).
* BatchNorm statistics frozen; 1 study/GPU x 2 GPUs (DDP), effective batch 16, K=12, fp16, 5 epochs.
* Data: the 4,333 training studies only. Gold74 is a separate preprocessing split and is never trained on.
  Folds: `raptor_data/cv_folds.py` (iterative multilabel stratification on HYBRID binary labels,
  seed 20261010, study-level: no patient IDs available). Assignment: `configs/cv5_folds.csv`
  (866/866/867/867/867 studies; positives per finding balanced within +-1).
* Each epoch, raw and EMA weights: gold74 macro AUC (missing labels masked), left-out-fold macro AUC
  (HYBRID binary), validation accuracy = mean of the two. Stored in `fold{k}/cv_metrics.csv|json`
  (also `history.json`, `training_events.jsonl`, `predictions/`). `fusion_best.pt` = best validation accuracy.
* Caveat: the parent Raptor was most likely trained on these 4,333 studies, so left-out-fold AUCs are
  probably optimistic; gold74 is held out from both the parent and this fine-tune.
* Kernels: `rsna-fusion-cv5-f01` (folds 0,1), `rsna-fusion-cv5-f23` (folds 2,3), `rsna-fusion-cv5-f4`
  (fold 4); built by `scripts/build_cv5_notebooks.py`, configs in `configs/cv5_kernels.json`.
  No step-0 evaluation (saves ~12 min per fold).
