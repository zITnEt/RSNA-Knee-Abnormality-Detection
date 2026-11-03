# Raptor v5 + RadImageNet ResNet-50

Branch: `codex/raptor-v5-radimagenet-fusion`.

## Kaggle training with the completed tensors

The generated `kaggle/raptor_v5_radimagenet_fusion.ipynb` now runs training via
`scripts/train_fusion_ready.py`, using `ReadyTripletDataset` instead of the raw
DICOM loader described below. Load the saved `[K,3,384,384]` float16 images
directly: no decoding, crop, resize, `/255`, or second normalization occurs.
Both backbones receive exactly the same saved tensor. Sampling K=12 is bag
selection only in the generic loader. This run sets K=None and uses all saved
triplets. All exported windows, including documented series-boundary
windows, are preserved to follow the requested six-slot dataset recipe.

Attach all 32 completed preprocessing notebook outputs, the two pretrained
parent datasets, and `riadmohamed42/jev-knee-labels`.
`scripts/build_fusion_notebook.py` generates the notebook and
private upload metadata in `kaggle/raptor_v5_radimagenet_training/`.
The launch uses Kaggle's `NvidiaTeslaT4` machine shape (T4 ×2), per the
[official metadata documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md).

Defaults: 50 epochs, `HYBRID_labels_scores.csv` soft targets, deterministic seed 20261009,
3,466 gradient-training studies / 867 study-level validation studies (80/20), and
all 74 gold studies excluded. This validation monitors soft-target BCE;
neither source-label calibration nor the pretrained parent's prior training
is undone by the new split. Patient-level disjointness has not been established.
Training batch size is 2 with 8 gradient-accumulation steps. Both backbones
remain unfrozen at LR 1e-5; fusion/head LR 1e-4, AdamW weight decay 0.01,
cosine schedule, AMP and gradient clipping. Full variable-length study bags
use chunks of four triplets with checkpointing around the entire backbone.
All chunk features feed a single study attention head, and all triplets receive
gradients. Ragged batches put one complete study on each GPU, without padding.
BatchNorm running buffers are restored after recomputation to avoid counting
each chunk twice; nested CoAtNet/fusion checkpointing is disabled.
Evaluation encodes the complete saved bag in chunks before a single MIL head.

`fusion_last.pt` stores model, optimizer, scaler, scheduler, split, settings,
epoch/batch position and RNG states. It is saved after the first optimizer
update, every 100 updates in version 2 / 25 updates in continuation sessions,
and each epoch. `fusion_best.pt` stores best soft-BCE
validation weights. A 10.5-hour guard saves a resumable checkpoint and ends
cleanly. The revised initial version 2 uses the original 10.5-hour guard;
its replacement with a shorter guard was blocked by the two-session GPU cap.
Continuation sessions use a guard up to 4.5 hours, reduced to fit reported
remaining quota, and request a six-hour timeout. Override `RUN_CONFIG` in
the notebook for another run or an attached resume checkpoint. The read-only
`scripts/monitor_kaggle_training.py` checks status every 60 seconds for up to
13 hours and saves small progress receipts under `results/fusion_training`.
The new `scripts/monitor_fusion_campaign.py` monitors the HYBRID campaign every
minute and creates private continuation notebooks after a clean checkpoint
pause. It waits for quota without enabling paid scaling, stops on unexplained
failures, and records state under `results/fusion_training_hybrid50`. It is
bounded to 14 days; rerun the script to continue the saved campaign after that
monitor expires. GPU quota can prevent completing all 50 epochs in that time.
The HYBRID file has three missing cells on an excluded gold study; no training
or validation target is missing or filled in. The split excludes gold studies
before target validation. New continuation checkpoints also pin the label file
hash and batch size, and copy the previous best weights forward when present.
Continuation AMP dtype is set explicitly in each GPU worker: native BF16
where supported, otherwise FP16 with a gradient scaler. This avoids relying
on the worker thread's default autocast dtype.

The raw DICOM loader below remains available for future raw inputs; it is
not called by this Kaggle training notebook.

The new model changes CoAtNet inside Raptor v5. Both unfrozen encoders see the
**same preprocessed adjacent-slice triplet**, at 384×384. The dataset should
return `[B, K, 3, 384, 384]`, where K is the number of triplets in each study.
Flattening B and K happens once before passing the same tensor to both trunks.
No branch-specific normalization, resizing, or detachment occurs in the model.

## Fusion and continuation

The stem is CoAtNet S0. Its four entries in `stages` are S1 through S4.

| Symbol | Feature source | Shape per triplet |
| --- | --- | --- |
| X | RadImageNet conv3_x = torchvision layer2 | 512 × 48 × 48 |
| Z | CoAtNet S2 = stages[1], convolution | 256 × 48 × 48 |
| Y | RadImageNet conv5_x = torchvision layer4 | 2048 × 12 × 12 |
| V | CoAtNet S4 = stages[3], transformer | 1024 × 12 × 12 |

```text
shared 3×384×384 triplet
  ├─ ResNet-50 stem → conv2_x → conv3_x X → conv4_x → conv5_x Y
  └─ CoAtNet S0 → S1 → S2 Z
                         ↓
               Z' = Z + F1(concat[X,Z])
                         ↓
                    S3 → S4 V
                         ↓
               V' = V + F2(concat[V,Y])
                         ↓
                CoAtNet final norm → average pooling
                         ↓
               original Raptor per-finding attention-MIL head → 12 logits
```

Both MLPs concatenate raw feature channels first, then apply per-position
LayerNorm → Linear → GELU → Dropout → Linear → GELU → Dropout → Linear.
F1 uses `768 → 1024 → 1024 → 256`; F2 uses
`3072 → 2048 → 2048 → 1024`. Hidden sizes/dropout are configurable.
The position-wise MLP preserves spatial information; neither feature map is
globally pooled before fusion. Matching strides make interpolation unnecessary.

Only the last linear's weight and bias start at zero. Therefore both residual
corrections are exactly zero initially, and the loaded Raptor predictions are
preserved for the **same 384 input** in evaluation mode. Earlier MLP layers use
normal random initialization, allowing the final projection to learn on the
first backward pass. The zero output projections also mean RadImageNet and the
earlier fusion layers receive zero loss gradients on the first backward pass;
their gradients open after the projections change. This is initialization
behavior, not freezing. CoAtNet retains its original gradient path throughout.

## Load the pretrained parents

Install `requirements-model.txt` in the training runtime, using a compatible
PyTorch/torchvision pair for its GPU. timm is pinned to 1.0.22, as in the
baseline. Model construction never downloads pretrained weights.

```python
import torch
from models.raptor_v5_radimagenet import (
    FusionConfig, build_raptor_v5_radimagenet, optimizer_groups,
    fused_checkpoint, load_fused_checkpoint,
)

model = build_raptor_v5_radimagenet(
    raptor_checkpoint="/kaggle/input/datasets/dreaddevelopment/raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt",
    radimagenet_checkpoint="/kaggle/input/datasets/marwanmath/resnet-50-radimagenet-marwan/ResNet50.pt",
    config=FusionConfig(),
).cuda().train()
optimizer = torch.optim.AdamW(optimizer_groups(model), weight_decay=0.01)

# Use SeriesTripletDataset from raptor_data.series_triplets (see below).
# triplets: float tensor [B,K,3,384,384]; targets: float tensor [B,12].
optimizer.zero_grad(set_to_none=True)
logits = model(triplets.cuda())
loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.cuda())
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
optimizer.step()

torch.save(fused_checkpoint(model), "raptor_v5_radimagenet.pt")
resumed = load_fused_checkpoint("raptor_v5_radimagenet.pt").cuda()
```

The factory first strict-loads the complete Raptor checkpoint into the original
architecture, preserving all CoAtNet and MIL-head parameters. It strict-loads
RadImageNet using the baseline encoder's `backbone.0` through `backbone.7` names.
Then it wraps CoAtNet with fusion. Missing or incompatible weights fail loading;
there is no random-weight fallback for either pretrained parent.

All model parameters have `requires_grad=True`. The optimizer helper includes
every parameter exactly once: CoAtNet and RadImageNet at `1e-5`, fusion and MIL
head at `1e-4`, with configurable rates. Calling `.train()` also trains both
backbones' BatchNorm statistics. There is no frozen feature cache.

`FusionConfig(gradient_checkpointing=True)` uses timm's CoAtNet block
checkpointing and checkpoints the fusion MLPs. RadImageNet remains a normal
trainable forward pass. Use a small bag/batch size appropriate for GPU memory.

The self-contained `kaggle/raptor_v5_radimagenet_fusion.ipynb` embeds the same
model and series-aware loader modules, and verifies both parent files by the
baseline manifest's Rad SHA-256 and the expected Raptor path. It builds the
model and optimizer; use `make_series_loader` with existing split IDs/labels
to start an experiment. No training run or new
accuracy result is claimed by this branch.

## Series-aware triplets

The parent dense inference path in baseline cell 22 selects up to 96 unique
source slices across five series, concatenates them, and slides a three-slice
window over the concatenation. It produces `min(96, total_capacity)-2`
triplets: usually 94. For five nonempty series of sufficient length, two
windows at each of the four boundaries mix series (eight windows in total).
The earlier nominal 64-slice/62-triplet arm constants are superseded by this
dense preparation function. The published training script has a separate K:
12 by default, and an older evaluation default of 24; K is not built into
the classifier. The new dataset keeps the supplied training K unchanged.

`raptor_data/series_triplets.py` retains the five-slot selection, 2–98% source
band, series intensity reference, 140mm center crop, and shared normalization,
but allocates **triplets per series before sampling slices**. A series assigned
q triplets supplies q+2 unique ordered slices. All triplets are constructed
locally in that series. For 94 unique triplets across five used series, this
requires 104 sampled slices; with S used series, it requires 94+2S. Allocations
redistribute to series with available capacity.

If there are not enough acquired slices, it repeats complete valid triplets
to preserve the parent's per-study count and records the repetition count in
the audit. Each repeated triplet still has three distinct acquired slices
from one series. `shortage="error"` forbids repetition. A study with no series
containing at least three acquired slices raises an error. No cross-series
or synthetic padding triplets are permitted.

```python
from raptor_data.series_triplets import SeriesTripletDataset

# series_by_study: study UID -> records from train_series.csv/test_series.csv
train_dataset = SeriesTripletDataset(
    train_ids, series_by_study, series_root, labels,
    train=True, k=12, norm="imagenet",
)
eval_dataset = SeriesTripletDataset(
    val_ids, series_by_study, series_root, labels,
    train=False, eval_k=None, norm="imagenet",
)
triplets, audit = eval_dataset.load(0)
# audit includes every triplet's series UID and three source file paths.
```

Training selects K unique safe candidate triplets without replacement when
available. Evaluation defaults to the parent's per-study count; use batch size
one for variable-length bags. Triplets use adjacent **sampled** slices in
physical order, as in Raptor, rather than necessarily consecutive native
DICOM slice numbers. The two encoders receive the same resulting tensor.
The existing anonymous `all_vols.npy` corpus does not provide series identities
to its loader; use raw DICOM series or rebuild a cache retaining per-slice
SeriesInstanceUID and physical-order provenance. Do not guess those boundaries.

## Validation

```powershell
python -m unittest discover -s tests -p test_raptor_v5_radimagenet.py -v
python -m unittest discover -s tests -p test_raptor_real_backbones.py -v
python -m unittest discover -s tests -p test_series_triplets.py -v
python -m unittest discover -s tests -p test_series_dicom_reader.py -v
```

The first suite checks ordered concatenation, zero residuals, shared input,
updated-stage routing, gradients after gate opening, complete optimizer
coverage, checkpointed MLP gradients, and input errors. The second uses the
actual timm CoAtNet and torchvision ResNet-50 with random validation weights:
real 384×384 map sizes, exact initial parent-logit equality, equivalence to the
MIL class extracted from the baseline notebook, both backbone gradients, and
strict fused checkpoint roundtrip. It does not download medical images or
pretrained checkpoint files.

Sampler checks include 100 randomized capacity layouts, five-series
104-slice/94-triplet cardinality, sparse-study count preservation, strict
shortage handling, source file ownership for every channel, and training K.
Generated DICOM files verify physical ordering, modality rescaling, rejection
of foreign series headers, and duplicate acquired slice detection.

Source references: the preserved baseline notebook's cell 22 (`_KE_SRC`,
`RaptorClassifier`) and `_RadEncoder`; [timm 1.0.22 CoAtNet implementation](https://github.com/huggingface/pytorch-image-models/blob/v1.0.22/timm/models/maxxvit.py);
[RadImageNet project](https://github.com/BMEII-AI/RadImageNet).
