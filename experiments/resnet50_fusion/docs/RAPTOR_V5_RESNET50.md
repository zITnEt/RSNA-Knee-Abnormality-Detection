# Raptor v5 + ImageNet ResNet-50

Branch: `codex/raptor-v5-resnet50-fusion`, based on `codex/raptor-v5-radimagenet-fusion` at `d42bb2b`.

The new model changes CoAtNet inside Raptor v5. Both unfrozen encoders see the
**same preprocessed adjacent-slice triplet**, at 384×384. The dataset should
return `[B, K, 3, 384, 384]`, where K is the number of triplets in each study.
Flattening B and K happens once before passing the same tensor to both trunks.
All three acquired-slice channels reach both trunks unchanged. ResNet50 does not
select or repeat the middle slice. No branch-specific normalization, resizing,
or detachment occurs in the model.

## Fusion and continuation

The stem is CoAtNet S0. Its four entries in `stages` are S1 through S4.

| Symbol | Feature source | Shape per triplet |
| --- | --- | --- |
| X | ResNet50 conv3_x = torchvision layer2 | 512 × 48 × 48 |
| Z | CoAtNet S2 = stages[1], convolution | 256 × 48 × 48 |
| Y | ResNet50 conv5_x = torchvision layer4 | 2048 × 12 × 12 |
| V | CoAtNet S4 = stages[3], transformer | 1024 × 12 × 12 |

```text
shared 3×384×384 triplet
  ├─ ResNet-50 stem → conv2_x → conv3_x X → conv4_x → conv5_x Y
  └─ CoAtNet S0 → S1 → S2 Z
                         ↓
               Z' = Z + g1 ⊙ F1(X, Z)
                         ↓
                    S3 → S4 V
                         ↓
               V' = V + g2 ⊙ F2(V, Y)
                         ↓
                CoAtNet final norm → average pooling
                         ↓
               original Raptor per-finding attention-MIL head → 12 logits
```

Both fusion blocks are a Mix-FFN style bottleneck (`BottleneckFusion`):
each input gets its own LayerNorm over channels, the two are concatenated,
then 1×1 down-projection → depthwise 3×3 → GELU → Dropout → 1×1 up-projection.
F1 uses `768 → 128 → 256` (≈0.13M parameters); F2 uses `3072 → 256 → 1024`
(≈1.06M), ≈1.2M in total, versus ≈14.7M for the earlier three-layer MLP.
Hidden sizes and dropout are configurable.

Separate norms keep ResNet's post-ReLU features and CoAtNet's features on a
comparable scale before mixing; one LayerNorm over the concatenation would let
the larger-magnitude input dominate the shared statistics. The depthwise 3×3
lets each output position use the fused features of its 3×3 neighbourhood,
tolerating small misalignment between the two trunks' receptive fields.
Neither feature map is globally pooled before fusion, and matching strides make
interpolation unnecessary.

Each residual is gated per channel: `Z' = Z + g1 ⊙ F1(X, Z)` and
`V' = V + g2 ⊙ F2(V, Y)`. The gate is SE-style,
`g = sigmoid(W2 · GELU(W1 · p))`, where `p` is the spatial mean of the two
separately normalized, concatenated inputs. It gives one value in (0, 1) per
window and output channel, shared across positions, so each image decides how
much of each channel's correction to admit. Gate MLPs are `768 → 64 → 256`
(≈66k) and `3072 → 128 → 1024` (≈525k), so the gates add ≈0.59M and the two
fusion blocks total ≈1.78M parameters. `stage2_gate_hidden=0` or
`stage4_gate_hidden=0` disables a gate (g = 1) for ablations. The gate's logit
bias starts at zero, so gates start near 0.5.

Only the up-projection's weight and bias start at zero. Therefore both residual
corrections are exactly zero initially, and the loaded Raptor predictions are
preserved for the **same 384 input** in evaluation mode. The down-projection
and depthwise conv use normal random initialization, allowing the up-projection
to learn on the first backward pass. The gate multiplies that zero residual, so
it cannot disturb the zero start. The zero up-projections also mean ResNet50,
the earlier fusion layers and the gates receive zero loss gradients on the first backward pass;
their gradients open after the projections change. This is initialization
behavior, not freezing. CoAtNet retains its original gradient path throughout.

## Load the pretrained parents

Install `requirements-model.txt` in the training runtime, using a compatible
PyTorch/torchvision pair for its GPU. timm is pinned to 1.0.22, as in the
baseline. The standard factory uses torchvision `IMAGENET1K_V2` ResNet50
weights, downloading them into the torch cache if absent. For offline use,
provide the official full torchvision state dict, as below; this does not download
weights. Do not apply the torchvision weight transforms: the existing shared
384×384 triplet preprocessing stays unchanged.

```python
import torch
from models.raptor_v5_resnet50 import (
    FusionConfig, build_raptor_v5_resnet50, optimizer_groups,
    fused_checkpoint, load_fused_checkpoint,
)

model = build_raptor_v5_resnet50(
    raptor_checkpoint="/kaggle/input/datasets/dreaddevelopment/raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt",
    resnet50_checkpoint="/kaggle/input/your-resnet50-weights/resnet50-11ad3fa6.pth",
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

torch.save(fused_checkpoint(model), "raptor_v5_resnet50.pt")
resumed = load_fused_checkpoint("raptor_v5_resnet50.pt").cuda()
```

The factory first strict-loads the complete Raptor checkpoint into the original
architecture, preserving all CoAtNet and MIL-head parameters. ResNet50 uses
standard ImageNet-1K V2 weights in place of the parent branch's RadImageNet
weights. Local checkpoints use the full torchvision ResNet50 state dict,
including `fc.weight` and `fc.bias`; loading is strict before the average pool
and classification head are removed. Missing or incompatible weights fail
loading. `resnet50_weights=None` explicitly requests random weights for
architecture tests only. Fused checkpoints have a separate
`raptor_v5_resnet50_v3` format and `backbone.resnet50.*` keys; restoration loads
the saved weights without downloading either parent. Fused `v1` (three-layer
MLP fusion) and `v2` (ungated bottleneck) checkpoints are rejected.

All model parameters have `requires_grad=True`. The optimizer helper includes
every parameter exactly once: CoAtNet and ResNet50 at `1e-5`, fusion and MIL
head at `1e-4`, with configurable rates. Calling `.train()` also trains both
backbones' BatchNorm statistics. There is no frozen feature cache.

`FusionConfig(gradient_checkpointing=True)` uses timm's CoAtNet block
checkpointing and checkpoints the fusion blocks. ResNet50 remains a normal
trainable forward pass. Use a small bag/batch size appropriate for GPU memory.

The self-contained `kaggle/raptor_v5_resnet50_fusion.ipynb` embeds the same
model and series-aware loader modules, finds the attached Raptor checkpoint,
and verifies the official `resnet50-11ad3fa6.pth` SHA-256 filename prefix.
Attach this file as a Kaggle dataset for offline execution. It builds the
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
    train=True, k=24, norm="imagenet",
)
eval_dataset = SeriesTripletDataset(
    val_ids, series_by_study, series_root, labels,
    train=False, eval_k=None, norm="imagenet",
)
triplets, audit = eval_dataset.load(0)
# audit includes every triplet's series UID and three source file paths.
```

Training selects K = 24 triplets, balanced over series slots: with n
populated slots, each slot gets floor(K/n) or ceil(K/n) windows (the K mod n
extras go to randomly chosen slots each step), drawn uniformly at random and
without replacement inside the slot. A slot with fewer unique windows than its
share passes the deficit evenly to slots with spare windows. Windows repeat
only when the whole study has fewer than K unique candidates; those repeats
go to the smallest slots first, keeping the split as close to K/n as possible.
The audit records `slot_quotas` and each window's slot in `window_slots`.

Evaluation defaults to the parent's per-study count; use batch size one for
variable-length bags. Triplets use adjacent **sampled** slices in
physical order, as in Raptor, rather than necessarily consecutive native
DICOM slice numbers. The two encoders receive the same resulting tensor.
The existing anonymous `all_vols.npy` corpus does not provide series identities
to its loader; use raw DICOM series or rebuild a cache retaining per-slice
SeriesInstanceUID and physical-order provenance. Do not guess those boundaries.

## Validation

```powershell
python -m unittest discover -s tests -p test_raptor_v5_resnet50.py -v
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
strict local ImageNet checkpoint loading, all-three-channel shared input, and
strict fused checkpoint roundtrip. It does not download medical images or
pretrained checkpoint files.

Sampler checks include 100 randomized capacity layouts, five-series
104-slice/94-triplet cardinality, sparse-study count preservation, strict
shortage handling, source file ownership for every channel, and training K.
Generated DICOM files verify physical ordering, modality rescaling, rejection
of foreign series headers, and duplicate acquired slice detection.

Source references: the preserved baseline notebook's cell 22 (`_KE_SRC`,
`RaptorClassifier`) and `_RadEncoder`; [timm 1.0.22 CoAtNet implementation](https://github.com/huggingface/pytorch-image-models/blob/v1.0.22/timm/models/maxxvit.py);
[torchvision ResNet50 weights and architecture](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html).
