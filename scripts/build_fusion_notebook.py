"""Build a self-contained Kaggle training notebook using completed tensor shards."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source, index):
    result = dict(cell_type=kind, metadata={}, id=f'fusion-{index}', source=source.splitlines(keepends=True))
    if kind == 'code':
        ast.parse(source)
        result.update(execution_count=None, outputs=[])
    return result


def main():
    trainer = (ROOT/'scripts/train_fusion_ready.py').read_text(encoding='utf-8')
    trainer = '\n'.join(line for line in trainer.splitlines()
                        if not line.startswith(('from models.raptor_v5_radimagenet import ',
                                                'from raptor_data.ready_tensors import ')))
    sources = [
        ('markdown', '''# Raptor V5 + RadImageNet fusion training

Training from `codex/raptor-v5-radimagenet-fusion`, using **already preprocessed**
six-slot tensors. There is no DICOM decoding, cropping, resizing, intensity
scaling, division by 255, or image normalization in this notebook.

All 32 completed preprocessing outputs are attached. Saved float16 triplets go
directly into both unfrozen backbones. All exported windows are preserved,
including the series-boundary windows documented in the dataset recipe.

HYBRID_labels_scores.csv soft targets: 3,466 training studies, 867 study-level validation
studies, and 74 gold studies excluded from this run. The validation BCE monitors
fit to soft targets; it is not an independent clinical performance estimate.
The source labels and pretrained parent have their own prior training history.

Defaults: 50 epochs, ALL saved triplets per study (no sampling), T4 ×2, batch size 4,
4 accumulation steps (effective 16 studies), AdamW backbone LR 1e-5 and
fusion/head LR 1e-4. Full-backbone checkpointed chunks of eight and AMP bound memory.
Each GPU receives two full variable-length bags; all chunk features contribute
to one study-level attention head and loss. BN running buffers are preserved
during checkpoint recomputation. Validation
uses all saved triplets, with chunked encoding and one study attention head.

`fusion_last.pt` saves model, optimizer, scaler, scheduler and resume position
after the first update, every 25 updates, and each epoch. `fusion_best.pt`
contains the best validation model. A 4.5-hour guard saves progress and exits
before the requested six-hour session limit. Set `resume_path` to an attached
`fusion_last.pt` to continue, preserving the split and settings.
'''),
        ('code', '''import importlib.metadata
import subprocess
import sys
try:
    installed = importlib.metadata.version('timm')
except importlib.metadata.PackageNotFoundError:
    installed = None
if installed != '1.0.22':
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', 'timm==1.0.22'])
assert importlib.metadata.version('timm') == '1.0.22'
print('Pinned timm:', importlib.metadata.version('timm'), flush=True)
'''),
        ('code', (ROOT/'models/raptor_v5_radimagenet.py').read_text(encoding='utf-8')),
        ('code', (ROOT/'raptor_data/ready_tensors.py').read_text(encoding='utf-8')),
        ('code', "RUN_CONFIG = {}  # Override defaults here; images remain preprocessed.\n"),
        ('code', trainer),
    ]
    notebook = dict(nbformat=4, nbformat_minor=5, metadata={
        'kernelspec':dict(display_name='Python 3', language='python', name='python3')},
        cells=[cell(kind, source, i) for i, (kind, source) in enumerate(sources)])
    destination = ROOT/'kaggle/raptor_v5_radimagenet_fusion.ipynb'
    destination.write_text(json.dumps(notebook, indent=1)+'\n', encoding='utf-8')
    upload = ROOT/'kaggle/raptor_v5_radimagenet_training'
    upload.mkdir(exist_ok=True)
    (upload/'training.ipynb').write_text(destination.read_text(encoding='utf-8'), encoding='utf-8')
    metadata = dict(id='zafarnematov/rsna-raptor-v5-radimagenet-fusion-train',
        title='RSNA Raptor V5 RadImageNet Fusion Train', code_file='training.ipynb',
        language='python', kernel_type='notebook', is_private=True, enable_gpu=True,
        machine_shape='NvidiaTeslaT4', enable_internet=True,
        dataset_sources=['dreaddevelopment/raptor-knee-maxspan', 'marwanmath/resnet-50-radimagenet-marwan',
                         'riadmohamed42/jev-knee-labels'],
        kernel_sources=[f'zafarnematov/rsna-six-slot-raptor-v5-preprocessing-{i:02d}' for i in range(32)],
        competition_sources=[], model_sources=[])
    (upload/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    main()
