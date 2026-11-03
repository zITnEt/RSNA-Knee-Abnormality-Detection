# ResNet50 fusion: four epochs on all non-Gold V2 studies

Target the existing `YOUR_GPU_VM` L4 VM. This campaign is a fresh
initialization from the same pinned CoAtNet and ImageNet ResNet50 assets as the
completed ten-epoch run. It uses the same minimum-overlap 384 V2 shards,
soft-label targets, K=24 slot-balanced study sampling, augmentation, model,
AdamW optimizer groups, BF16, frozen BatchNorm statistics, effective study
batch of 16, gradient clipping, and EMA rule.

All 4,333 non-Gold studies train for exactly four epochs. There is no internal
validation split. Gold74 remains outside training and is the only evaluation
set. Raw and EMA weights are evaluated on Gold74 after every epoch, including
masked accuracy, macro/per-finding AUROC, and BCE. Gold74 macro AUROC selects
the best raw and EMA snapshots.

The old run had 244 optimizer steps per epoch and used a 100-step warmup
followed by cosine decay over 2,440 scheduled steps. This full-data run has 271
steps per epoch. Each new step uses an exact LR factor from the matching old
epoch, indexed by `floor(local_step * 244 / 271)`. Thus the first and last
learning rates and the complete set of LR values in each of epochs 1–4 match
the prior run; 27 extra updates repeat factors. The optimizer group base rates
remain 1e-5 for both backbones and 1e-4 for fusion and head.

The isolated output directory is `run_full4333_gold74_4ep/`. It retains the
initial checkpoint, every-100-update checkpoint, pre-evaluation checkpoint,
raw and EMA epoch checkpoints, and best Gold74 raw/EMA snapshots. Raw epoch
checkpoints include optimizer, scheduler, scaler, RNG, cursor, and complete
EMA accumulator for faithful restart. No pruning runs. `events.jsonl`,
`history.json`, `progress.json`, and per-study Gold74 logits are saved there.
The previous ten-epoch `run/` is untouched.

The separate `rsna-resnet50-full-gold4.service` starts the launcher on the VM,
restarts a failed attempt up to three times in ten minutes, and resumes the
latest complete checkpoint. The launcher rechecks pinned V2 manifests before
each start.
