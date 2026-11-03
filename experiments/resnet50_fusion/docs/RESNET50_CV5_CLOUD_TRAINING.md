# Raptor V5 + ResNet50 fusion: five-fold cloud fine-tuning

Run on `YOUR_GPU_VM` in `us-central1-a` with its NVIDIA L4.
The pinned `configs/cv5_folds.csv` assigns each of the 4,333 non-Gold V2
studies to one of five study-level, multi-label-balanced folds (866–867
validation studies per fold). Gold74 is excluded from every training and fold
validation split. Every fold starts independently from the same pinned Raptor
V5 CoAtNet and ImageNet ResNet50 parent weights; it never starts from either
of the previous fused runs.

Keep the earlier minimum-overlap 384 V2 study tensors and the same K=24
slot-balanced sampling, intensity augmentation, soft-label training targets,
model, AdamW optimizer groups, BF16 where supported, frozen BatchNorm running
statistics, effective batch of 16 studies, gradient clipping, and float32 EMA
rule (`decay=.998`). Train each fold for five epochs. The fold's 217 optimizer
steps per epoch replay the corresponding first five epochs of the original
ten-epoch learning-rate curve, interpolating to preserve each epoch's starting
and ending factor. The optimizer group base rates are `1e-5` for backbones and
`1e-4` for fusion and head.

After every epoch, evaluate raw and EMA weights on that fold's held-out
studies using `HYBRID_labels_binary.csv`, and on Gold74 using
`gold74_eval_set.csv` with missing labels masked. Save accuracy at sigmoid
threshold 0.5, macro and per-finding AUROC, BCE, and per-study logits. Select
the best raw, best EMA, and overall snapshot for each fold using **only** fold
validation macro AUROC. Gold74 is a separate report and cannot select a
checkpoint or epoch. Fold validation labels are the existing hybrid pseudo
labels, so their scores should be interpreted accordingly.

Each fold retains its best raw and best EMA snapshots by held-out-fold AUROC,
along with a rolling full-state checkpoint for recovery. The rolling state
includes optimizer, scheduler, scaler, RNG, cursor, and complete EMA state.
Obsolete periodic and intermediate snapshots are removed after their
replacements are complete. `run_cv5/fold_0` through `run_cv5/fold_4` live on
the VM's existing disk; the previous ten- and four-epoch runs are untouched.
`rsna-resnet50-cv5.service` restarts on failure, and the launcher skips
completed folds and resumes incomplete folds. Dataset and parent asset
manifests are verified at every service start.

After fold 4 completes, the launcher writes `cv5_summary.json` and out-of-fold
prediction files. The summary includes per-fold-best and fixed-epoch-5
out-of-fold scores, plus Gold74 five-model probability ensembles. Scores from
best-per-fold epochs are optimistic because those epochs were chosen on the
same fold; fixed-epoch-5 out-of-fold scores avoid that particular selection.

The Raptor V5 parent was already fine-tuned; its prior training exposure to
these studies has not been fully ruled out. Fold metrics assess this fusion
fine-tuning campaign and should not be described as leakage-free validation
of the full pipeline. Gold74 remains the separate external check.
