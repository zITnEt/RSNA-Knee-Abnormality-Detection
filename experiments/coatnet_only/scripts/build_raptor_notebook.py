"""Generate the self-contained, offline CoAtNet-only Raptor model notebook."""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    cells = [cell("markdown", """# Raptor v5: CoAtNet only

The existing CoAtNet RMLP-2 trunk and original 12-label attention-MIL head.
There is no ResNet branch, fusion block, or fusion gate in the model.
Input remains an existing preprocessed `[B,K,3,384,384]` adjacent-slice bag.
Continue using the same study splits, labels, sampling and normalization as
your current data pipeline. This notebook does not replace preprocessing.

Attach an original Raptor v5 checkpoint, a trained v3 fusion checkpoint, or
an exported CoAtNet-only checkpoint. A fusion checkpoint contributes only its
CoAtNet/head tensors. No model weights are downloaded. Use timm 1.0.22.
"""), cell("code", """import importlib.metadata
import subprocess
import sys
from pathlib import Path

try:
    timm_version = importlib.metadata.version('timm')
except importlib.metadata.PackageNotFoundError:
    timm_version = None
if timm_version != '1.0.22':
    wheels = sorted(Path('/kaggle/input').rglob('timm-1.0.22-py3-none-any.whl'))
    if len(wheels) != 1:
        raise RuntimeError('Attach exactly one timm 1.0.22 wheel for offline use')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           '--no-index', '--no-deps', str(wheels[0])])
"""), cell("code", (ROOT / "models/raptor_v5.py").read_text(encoding="utf-8")),
        cell("code", """# Choose the checkpoint whose CoAtNet/head weights you want to retain.
checkpoint_path = Path('/kaggle/input/raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt')
# 'ema' selects embedded ema.shadow; standalone epoch_*_ema.pt files use 'raw'.
source_weights = 'raw'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_raptor_checkpoint(checkpoint_path, weights=source_weights).to(device)
optimizer = torch.optim.AdamW(optimizer_groups(model), weight_decay=0.01)
print('Parameters:', sum(p.numel() for p in model.parameters()))
print('Optimizer groups:', [(g['name'], g['lr']) for g in optimizer.param_groups])
"""), cell("code", """# Reuse your existing loader's preprocessed images and targets.
# Keep batch_size=1 for variable-length studies; evaluate all saved windows.
from torch.utils.checkpoint import checkpoint

def bag_logits(triplets, chunk=12):
    # Chunk only the encoder; the unchanged head sees the entire study bag.
    if (triplets.ndim != 5 or triplets.shape[0] != 1 or triplets.shape[1] == 0
            or tuple(triplets.shape[2:]) != (3, 384, 384)):
        raise ValueError('Expected [1,K,3,384,384]')
    if chunk < 1:
        raise ValueError('chunk must be positive')
    features = []
    for windows in triplets[0].split(chunk):
        if model.training and torch.is_grad_enabled():
            features.append(checkpoint(model.backbone, windows, use_reentrant=False))
        else:
            features.append(model.backbone(windows))
    return model.head(torch.cat(features)[None])

use_amp = device.type == 'cuda'
amp_dtype = (torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported()
             else torch.float16)
scaler = torch.amp.GradScaler(device.type, enabled=use_amp and amp_dtype == torch.float16)

def train_step(triplets, targets):
    model.train()
    # Preserve the completed run's frozen BatchNorm running statistics.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
        logits = bag_logits(triplets.to(device))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets.to(device=device, dtype=torch.float32))
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach())

# Start a new optimizer: the old fusion optimizer has different groups.
# saved = raptor_checkpoint(model)
# saved.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), epoch=epoch)
# torch.save(saved, '/kaggle/working/raptor_v5_coatnet.pt')
""")]
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }, "nbformat": 4, "nbformat_minor": 5}
    destination = ROOT / "kaggle/raptor_v5_coatnet.ipynb"
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
