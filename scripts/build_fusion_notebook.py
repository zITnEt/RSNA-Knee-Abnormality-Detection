"""Regenerate the self-contained Kaggle model notebook from its Python module."""

from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    cells = [cell("markdown", """# Raptor v5 + ResNet50 residual fusion

Both unfrozen backbones receive the same preprocessed 384×384 adjacent-slice
triplet. CoAtNet S2 uses `Z + F1(X,Z)`; S4 uses `V + F2(V,Y)`.
Each fusion block is a small bottleneck (separate LayerNorms, 1×1 down,
depthwise 3×3, GELU, 1×1 up; ≈1.2M parameters in total), and both
corrections initialize to exactly zero. The original Raptor MIL head
continues after CoAtNet pooling. This notebook builds the architecture and
optimizer and a series-aware loader. Use this loader to train it; anonymous
concatenated slice caches cannot enforce MRI series boundaries.

Attach `dreaddevelopment/raptor-knee-maxspan`, the official torchvision
ImageNet-1K V2 checkpoint `resnet50-11ad3fa6.pth`, and a `timm-1.0.22` wheel if
your runtime does not already have that version. Attach the existing training
competition series and labels for training. No checkpoint or competition data
downloads occur. ResNet50 receives all three slice channels, exactly as CoAtNet
does. Dense eval preserves the parent's per-study triplet count
(usually 94), using up to 104 sampled slices across five series.
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
    if not wheels:
        raise RuntimeError('Attach the timm 1.0.22 wheel before running offline')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           '--no-index', '--no-deps', str(wheels[0])])
"""), cell("code", (ROOT / "models/raptor_v5_resnet50.py").read_text(encoding="utf-8")),
        cell("code", (ROOT / "raptor_data/series_triplets.py").read_text(encoding="utf-8")),
        cell("code", """def attached_asset(relative_paths):
    existing = [Path(p) for p in relative_paths if Path(p).is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(f'Expected one attached parent asset: {relative_paths}; found {existing}')
    return existing[0]

raptor_path = attached_asset([
    '/kaggle/input/datasets/dreaddevelopment/raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt',
    '/kaggle/input/raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt',
])
resnet50_paths = sorted(Path('/kaggle/input').rglob('resnet50-11ad3fa6.pth'))
resnet50_path = attached_asset(resnet50_paths)
# Check the official torchvision SHA-256 filename prefix before strict loading.
import hashlib
digest = hashlib.sha256()
with resnet50_path.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
if not digest.hexdigest().startswith('11ad3fa6'):
    raise RuntimeError('ResNet50 checkpoint differs from torchvision ImageNet-1K V2')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = FusionConfig()
model = build_raptor_v5_resnet50(raptor_path, resnet50_path, config).to(device).train()
optimizer = torch.optim.AdamW(optimizer_groups(model), weight_decay=0.01)
print('Parameters:', sum(p.numel() for p in model.parameters()))
print('Trainable:', sum(p.numel() for p in model.parameters() if p.requires_grad))
print('Optimizer groups:', [(g['name'], g['lr']) for g in optimizer.param_groups])
"""), cell("code", """import pandas as pd

def make_series_loader(series_csv, series_root, ids, labels=None, *, train=False, k=24):
    # Preserve your existing train/validation study split and label mapping.
    frame = pd.read_csv(series_csv, dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    series = {uid: group.to_dict('records') for uid, group in frame.groupby('StudyInstanceUID')}
    dataset = SeriesTripletDataset(ids, series, series_root, labels,
                                  train=train, k=k, eval_k=None, norm='imagenet')
    # Eval K is the historical per-study count and can vary for sparse studies.
    # batch_size=1 handles that and bounds memory for two unfrozen backbones.
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=train, num_workers=0)

# Example with your existing split IDs and label dict (12 targets per study):
# train_loader = make_series_loader(COMP_ROOT/'train_series.csv', COMP_ROOT/'train_images',
#                                   train_ids, labels, train=True, k=24)
# Training K=24 is split evenly over the populated series slots (~K/n each).
# val_loader = make_series_loader(COMP_ROOT/'train_series.csv', COMP_ROOT/'train_images',
#                                 val_ids, labels, train=False)
# Each channel is a distinct acquired slice from one SeriesInstanceUID.
"""), cell("code", """# Use the series-aware loader above; it yields [B,K,3,384,384].
# Shared preprocessing/normalization is applied ONCE by that loader.
use_amp = device.type == 'cuda'
amp_dtype = (torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported()
             else torch.float16)
scaler = torch.amp.GradScaler(device.type, enabled=use_amp and amp_dtype == torch.float16)

def train_step(triplets, targets):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
        logits = model(triplets.to(device))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets.to(device=device, dtype=torch.float32))
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach())

# After training, preserve both trunks, both fusion blocks, and the original head:
# checkpoint_data = fused_checkpoint(model)
# checkpoint_data.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), epoch=epoch)
# torch.save(checkpoint_data, '/kaggle/working/raptor_v5_resnet50.pt')
""")]
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }, "nbformat": 4, "nbformat_minor": 5}
    destination = ROOT / "kaggle/raptor_v5_resnet50_fusion.ipynb"
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
