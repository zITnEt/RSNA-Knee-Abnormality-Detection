"""Package the five selected EMA checkpoints as a private Kaggle dataset."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'run_cv5'
OUT = ROOT / 'cv5_submission_weights'
RECIPE_SHA = 'aad96c9225fa13edd3d2dff0a105f2e7de6b3efc91a0ad45a8229bd15acfd35c'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    summary = json.loads((RUN / 'cv5_summary.json').read_text())
    chosen = summary['variants']['selected']['fold_models']
    if len(chosen) != 5 or [item['weight_kind'] for item in chosen] != ['ema'] * 5:
        raise ValueError('Expected five EMA-selected folds')
    OUT.mkdir(exist_ok=True)
    checkpoints = []
    for fold, item in enumerate(chosen):
        if item['fold'] != fold or item['epoch'] not in range(1, 6):
            raise ValueError(f'Invalid selection for fold {fold}')
        directory = RUN / f'fold_{fold}'
        if not (directory / 'complete.json').is_file():
            raise FileNotFoundError(f'Incomplete fold {fold}')
        source = directory / 'checkpoints' / 'best_validation_ema.pt'
        target = OUT / f'fold_{fold}_ema.pt'
        if not target.exists():
            os.link(source, target)
        if not os.path.samefile(source, target):
            raise ValueError(f'Packaged fold {fold} is not the selected checkpoint')
        checkpoint = torch.load(source, map_location='cpu', weights_only=True)
        if (checkpoint['fold'] != fold or checkpoint['epoch'] != item['epoch']
                or checkpoint['weight_kind'] != 'ema'
                or checkpoint['training_config']['recipe_sha256'] != RECIPE_SHA):
            raise ValueError(f'Unexpected checkpoint provenance for fold {fold}')
        checkpoints.append(dict(fold=fold, epoch=item['epoch'], file=target.name,
                                bytes=target.stat().st_size, sha256=sha256(target),
                                validation_macro_auc=checkpoint['metrics']['validation']['macro_auc']))
        del checkpoint
    manifest = dict(recipe_sha256=RECIPE_SHA, checkpoints=checkpoints,
                    selection='Best per-fold validation macro AUROC, Gold74 excluded',
                    oof_macro_auc=summary['variants']['selected']['oof']['macro_auc'],
                    gold74_ensemble_macro_auc=summary['variants']['selected']['gold74_ensemble']['macro_auc'])
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    metadata = dict(id='zafarnematov/rsna-raptor-resnet50-cv5-ema',
                    title='RSNA Raptor ResNet50 Five Fold EMA',
                    subtitle='Selected five-fold EMA checkpoints for minimum-overlap 384 V2 fusion',
                    description='Private float32 model checkpoints selected by held-out-fold validation AUROC.',
                    licenses=[dict(name='unknown')],
                    resources=[dict(path=item['file'], description=f"Fold {item['fold']} selected EMA epoch {item['epoch']}")
                               for item in checkpoints] + [dict(path='manifest.json', description='SHA256 and provenance')])
    (OUT / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
