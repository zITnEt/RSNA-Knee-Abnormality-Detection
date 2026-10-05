# Raptor v5 + RadImageNet ResNet-50

Branch: `codex/raptor-v5-radimagenet-fusion`.

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

# Use the existing Raptor loader, with its output triplets set to 384×384.
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
model module and verifies both parent files by the baseline manifest's Rad
SHA-256 and the expected Raptor path. It builds the model and optimizer; supply
the existing training loader to start an experiment. No training run or new
accuracy result is claimed by this branch.

## Validation

```powershell
python -m unittest discover -s tests -p test_raptor_v5_radimagenet.py -v
python -m unittest discover -s tests -p test_raptor_real_backbones.py -v
```

The first suite checks ordered concatenation, zero residuals, shared input,
updated-stage routing, gradients after gate opening, complete optimizer
coverage, checkpointed MLP gradients, and input errors. The second uses the
actual timm CoAtNet and torchvision ResNet-50 with random validation weights:
real 384×384 map sizes, exact initial parent-logit equality, equivalence to the
MIL class extracted from the baseline notebook, both backbone gradients, and
strict fused checkpoint roundtrip. It does not download medical images or
pretrained checkpoint files.

Source references: the preserved baseline notebook's cell 22 (`_KE_SRC`,
`RaptorClassifier`) and `_RadEncoder`; [timm 1.0.22 CoAtNet implementation](https://github.com/huggingface/pytorch-image-models/blob/v1.0.22/timm/models/maxxvit.py);
[RadImageNet project](https://github.com/BMEII-AI/RadImageNet).
