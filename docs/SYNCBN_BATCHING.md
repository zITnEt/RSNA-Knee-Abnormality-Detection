# Two studies per GPU with SyncBatchNorm

Configuration: `configs/fusion_ddp_syncbn_b2.json`. The separate generated training
and smoke notebooks are under `kaggle/rsna-fusion-ddp-syncbn-b2/` and
`kaggle/rsna-fusion-ddp-syncbn-b2-smoke/`. These artifacts are prepared locally;
generating them does not launch or change a Kaggle run.

| Quantity | Two-GPU setting |
| --- | --- |
| Studies per GPU per forward/backward | 2 |
| Triplets per study | 12 |
| Encoder inputs per GPU per forward | 24 triplets |
| SyncBatchNorm batch across GPUs | 4 studies / 48 triplets, plus spatial positions per feature channel |
| Gradient accumulation | 8 microsteps |
| Effective optimizer batch | 32 studies / 384 triplets |
| Microsteps per epoch with 3,900 studies | 975 per GPU, running concurrently |
| Optimizer updates per epoch | 122 (121 batches of 32 studies, then 28 studies) |

Unfrozen BatchNorm uses current forward-pass statistics during training and updates
its running estimates every microstep. It does not wait for an optimizer update.
The running estimates retain information over time, but are not the exact mean and
variance of one accumulated 32-study batch. At validation/inference, saved running
estimates are used and no statistics are synchronized or updated.

The trainer uses timm's SyncBatchNorm converter to preserve CoAtNet's fused
BatchNorm activation/dropout behavior. Both ranks start from the same buffers and
execute equal training microsteps. Gradient `no_sync()` during accumulation does
not turn off SyncBatchNorm communication. EMA averages parameters; its evaluation
and saved checkpoints use the current synchronized running statistics.

All backbone sections use activation checkpointing for this configuration. The
recomputation context restores running buffers afterward, so a backward pass does
not count as a second statistics update. The preflight runs a complete two-study
forward/backward on each GPU before SyncBatchNorm conversion, restores its BN
buffers, and checks the maximum measured allocation plus a 2 GiB optimizer/DDP
reserve against a 13.5 GiB limit. This estimate is not a substitute for a real
two-GPU smoke run; SyncBatchNorm communication and optimizer allocations must also
fit. No GPU memory fit or score improvement is claimed by the local tests.

This is a fresh experiment from the configured pretrained parents. It inherits
the current trainer's other defaults, including `rad_input='pm1_middle'`. Existing
one-study/frozen-BN runs and notebooks keep their configuration. Resume checks reject
changing studies per GPU, effective batch, or BatchNorm mode partway through a saved
run; the microbatch cursor and scheduler depend on those settings.

Rebuild the training notebook from the repository root with:

```python
import json
from pathlib import Path
from scripts.build_fusion_v2_notebook import build

cfg = json.loads(Path('configs/fusion_ddp_syncbn_b2.json').read_text())
slug = 'rsna-fusion-ddp-syncbn-b2'
notebook, metadata = build(slug, False, cfg, launcher='ddp')
out = Path('kaggle') / slug
out.mkdir(parents=True, exist_ok=True)
(out / f'{slug}.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
(out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
```

Use `smoke=True` and the smoke slug to prepare the 25-update verification run.
The training and inference preprocessing, K, loss, and learning rates are unchanged
relative to current source defaults. Doubling the optimizer batch halves the update
count per epoch approximately; compare experiments by both epochs and update counts.
