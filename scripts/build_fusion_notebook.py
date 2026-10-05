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
    cells = [cell("markdown", """# Raptor v5 + RadImageNet residual fusion

Both unfrozen backbones receive the same preprocessed 384×384 adjacent-slice
triplet. CoAtNet S2 uses `Z + F1(X,Z)`; S4 uses `V + F2(V,Y)`.
Both MLP corrections initialize to exactly zero. The original Raptor MIL head
continues after CoAtNet pooling. This notebook builds the architecture and
optimizer; use the existing Raptor training loader to train it.

Attach `dreaddevelopment/raptor-knee-maxspan`,
`marwanmath/resnet-50-radimagenet-marwan`, and a `timm-1.0.22` wheel if your
runtime does not already have that version. Attach the existing training
corpus/labels for training. No checkpoint or competition data downloads occur.
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
"""), cell("code", (ROOT / "models/raptor_v5_radimagenet.py").read_text(encoding="utf-8")),
        cell("code", """import hashlib

def attached_asset(relative_paths):
    existing = [Path(p) for p in relative_paths if Path(p).is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(f'Expected one attached parent asset: {relative_paths}; found {existing}')
    return existing[0]

raptor_path = attached_asset([
    '/kaggle/input/datasets/dreaddevelopment/raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt',
    '/kaggle/input/raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt',
])
rad_path = attached_asset([
    '/kaggle/input/datasets/marwanmath/resnet-50-radimagenet-marwan/ResNet50.pt',
    '/kaggle/input/resnet-50-radimagenet-marwan/ResNet50.pt',
])
digest = hashlib.sha256()
with rad_path.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
if digest.hexdigest() != '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734':
    raise RuntimeError('RadImageNet checkpoint differs from the baseline manifest')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = FusionConfig()
model = build_raptor_v5_radimagenet(raptor_path, rad_path, config).to(device).train()
optimizer = torch.optim.AdamW(optimizer_groups(model), weight_decay=0.01)
print('Parameters:', sum(p.numel() for p in model.parameters()))
print('Trainable:', sum(p.numel() for p in model.parameters() if p.requires_grad))
print('Optimizer groups:', [(g['name'], g['lr']) for g in optimizer.param_groups])
"""), cell("code", """# Connect this step to your existing loader; it must yield [B,K,3,384,384].
# Apply the shared Raptor preprocessing/normalization ONCE in that loader.
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

# After training, preserve both trunks, both fusion MLPs, and the original head:
# checkpoint_data = fused_checkpoint(model)
# checkpoint_data.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), epoch=epoch)
# torch.save(checkpoint_data, '/kaggle/working/raptor_v5_radimagenet.pt')
""")]
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }, "nbformat": 4, "nbformat_minor": 5}
    destination = ROOT / "kaggle/raptor_v5_radimagenet_fusion.ipynb"
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
