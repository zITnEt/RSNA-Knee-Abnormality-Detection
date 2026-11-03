# ResNet50 fusion: single 90/10 split on Google Cloud

Target: `YOUR_GPU_VM`, `us-central1-a`, project
`YOUR_GCP_PROJECT`. The existing VM has one NVIDIA L4 and a
400 GB persistent boot disk. Training uses the gated bottleneck model from
`d49b345`, format `raptor_v5_resnet50_v3`.

Run exactly 10 epochs, with no cross-validation or early stopping.
Seed 20261013 partitions the 4,333 non-Gold studies into 3,900 training and
433 validation studies. Gold74 is a separate evaluation set and never enters
training or checkpoint selection. The persisted study IDs define the split.

Inputs are the minimum-overlap 384 V2 export, recipe
`aad96c9225fa13edd3d2dff0a105f2e7de6b3efc91a0ad45a8229bd15acfd35c`.
Stage the 32 private Kaggle datasets at version 2. Some versioned archive
endpoints return HTTP 404 despite listing all V2 files. For those shards,
the staging script downloads the corresponding publication notebook output
files directly. It compares publication manifests and verifies every study
file's SHA256 before training. It refuses to delete existing data if disk
space is insufficient.

Training draws 24 windows balanced across populated series slots, without
replacement. Validation and Gold use every exported window. Saved images are
already normalized. Training alone applies the original mild 0.9–1.1 intensity
factor in unnormalized space. No further crop or resize occurs.

Both backbones, the gated fusion blocks, and the original attention MIL head
are trainable. Pretrained BatchNorm running statistics stay fixed, avoiding
small-batch and checkpoint-recomputation contamination; affine BN parameters
remain trainable. Backbone LR is 1e-5; fusion and head LR are 1e-4. AdamW weight
decay is 0.01. Use warmup plus cosine LR decay, effective study batch 16,
gradient clipping at norm 3, and BF16 on L4. Encode six windows at a time with
activation checkpointing; attention still pools the complete sampled study.

After every epoch, evaluate four combinations independently: raw/validation,
raw/Gold74, EMA/validation, and EMA/Gold74. Report masked binary accuracy at
sigmoid >= 0.5, macro/per-finding AUROC, and BCE. Missing Gold labels are masked;
single-class findings are excluded from macro AUROC and reported with null AUC.
Do not average validation and Gold metrics into an invented accuracy score.

EMA includes all named parameters and floating buffers in float32, with
integer buffers copied exactly. It updates once per successful optimizer step.
Decay warms up as min(0.998, (1+updates)/(10+updates)). EMA evaluation swaps and
restores the entire state, including buffers. EMA checkpoints contain actual
EMA model tensors, while raw resumable checkpoints also retain the complete
EMA accumulator and update count.

Retain initial, every-100-update, pre-evaluation, raw epoch, and EMA epoch
checkpoints. No pruning runs. All 10 raw epoch checkpoints retain optimizer,
LR scheduler, AMP scaler, RNG, cursor, split, config, history and EMA state.
Best raw and EMA snapshots are selected using validation macro AUROC only.
Per-study logits and labels are retained for all four evaluations each epoch.
Atomic writes preserve the preceding complete checkpoint if interrupted.

The desktop background worker `scripts/retry_resnet50_cloud.ps1` retries only
the specified VM, waiting five minutes between capacity failures or temporary
Google API connection failures. Permanent permission/configuration errors still
stop with a needs-attention status. After a
successful start it uses SSH to copy the verified bundle and a separate,
restricted Kaggle configuration into the isolated campaign directory
`~/rsna-resnet50-20261013/`. No credentials enter Cloud Storage. A detached
bootstrap checks the CUDA environment, creates an isolated Python environment,
stages the pinned dataset and starts training. The computer running the retry
worker must remain awake until deployment; training subsequently runs on the VM.
The worker holds a file lock, records `results/resnet50_cloud/retry_state.json`
on the desktop, and reports setup failures rather than claiming training began.

The Linux launcher stages the data, detects the latest resumable checkpoint,
and runs training. Invoke it from an environment containing a compatible
CUDA PyTorch/torchvision pair, timm==1.0.22, numpy, pandas, scikit-learn, and
the authenticated Kaggle CLI. Credentials are never included in the bundle.

```bash
nohup bash scripts/launch_resnet50_cloud.sh > launcher.log 2>&1 < /dev/null &
```

Inspect `run/progress.json`, `run/history.json`, `run/events.jsonl`,
`run/predictions/`, and `run/checkpoints/`. The launcher uses a file lock to
prevent concurrent duplicate runs. A real CUDA forward/backward preflight
must pass before the first optimizer update.
