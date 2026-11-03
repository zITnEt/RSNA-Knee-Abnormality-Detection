# CoAtNet-only Raptor v5

Branch `codex/raptor-v5-coatnet-only` starts from the fusion branch at `4dd2104`.
The model now contains the existing CoAtNet RMLP-2 backbone and the original
per-finding attention-MIL head. The ResNet encoder, both fusion blocks and both
fusion gates are removed, including their parameters and optimizer groups.

```text
existing preprocessed [B,K,3,384,384] triplet bag
    -> CoAtNet stem -> S1 -> S2 -> S3 -> S4
    -> final norm and spatial average pooling
    -> original per-finding attention across K windows
    -> original 12-label classifier
```

All three acquired-slice channels still reach CoAtNet. The model performs no
resizing, normalization, slice selection or repetition. Continue using the
existing prepared bags, series boundaries, labels and study split; this change
does not regenerate inputs. In the analyzed cloud run, training sampled 24
windows and evaluation used all available saved windows (normally 60).

## Load existing weights

```python
from models.raptor_v5 import load_raptor_checkpoint, optimizer_groups, raptor_checkpoint
import torch

model = load_raptor_checkpoint("epoch_03_ema.pt")  # standalone EMA snapshot
optimizer = torch.optim.AdamW(optimizer_groups(model), weight_decay=0.01)
# Supply your existing [B,K,3,384,384] preprocessed triplets.
logits = model(triplets)
probabilities = logits.sigmoid()
torch.save(raptor_checkpoint(model), "raptor_v5_coatnet.pt")
```

Accepted source layouts:

| Source | Selection | What is retained |
|---|---|---|
| Original Raptor v5 parent | `weights="raw"` (default) | Strict-load the original `model` state |
| Trained v3 fusion checkpoint | `weights="raw"` | Its trained CoAtNet tensors, renamed to `backbone.*`, and unchanged head tensors |
| Trainer checkpoint with embedded EMA | `weights="ema"` | CoAtNet/head from `ema.shadow`; fails if EMA is absent |
| Standalone EMA snapshot | `weights="raw"` (default) | Its EMA weights are already stored in `model` |
| New CoAtNet-only checkpoint | `weights="raw"` | Strict-load `raptor_v5_coatnet_v1` |

Loading never constructs a ResNet, downloads weights, or initializes a new head
in place of existing trained head weights. Unknown formats, wrong input/label
contracts, missing retained tensors and shape mismatches fail explicitly.

## Export a smaller checkpoint

From this worktree, with the existing torch and timm 1.0.22 environment:

```powershell
python scripts/strip_raptor_fusion.py path/to/best_ema.pt path/to/raptor_v5_coatnet.pt
```

For an embedded EMA, add `--weights ema`. The command requires a new output
path. It retains only model weights/config and source-hash provenance; it drops
the old optimizer, scheduler, scaler and training cursor. Start a new optimizer
when fine-tuning: the two groups are now `coatnet` and `head`.

Removing trained fusion changes predictions. The exported model matches the
original CoAtNet path with both fusion corrections removed; it is not generally
identical to the old full fusion model or to its pre-fine-tuning parent.
The previous report's selected-EMA no-fusion result was 0.925198 macro-AUC
[0.900625, 0.945829]; this is an existing ablation, not a new training result.

## Notebook and validation

`kaggle/raptor_v5_coatnet.ipynb` embeds the same model source and illustrates
chunked encoding with the unchanged whole-study MIL head. It consumes the
existing preprocessed bags. Regenerate and test with:

```powershell
python scripts/build_raptor_notebook.py
python -m unittest discover -s tests -p "test_raptor*.py" -v
```

Tests cover strict raw/EMA migration, exact preservation of retained tensors,
the original head, unchanged triplet channels, optimizer coverage, gradients,
checkpoint round trips, export provenance and real timm CoAtNet inference.
Synthetic/random test weights are not trained model evaluation.

The historical ResNet50 analysis under `results/resnet50_20261013` describes
the parent branch. Its CPU cached-evidence analysis remains available; full
historical fusion re-inference requires that branch's original model/trainer.
