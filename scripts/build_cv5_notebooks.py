"""Build the 5-fold CV Kaggle notebooks (frozen BN, 5 epochs, parent Raptor V5 + RadImageNet init).

Kernels run folds back to back (two Kaggle GPU sessions at a time):
    rsna-fusion-cv5-f01 -> folds 0, 1;  rsna-fusion-cv5-f23 -> folds 2, 3;  rsna-fusion-cv5-f4 -> fold 4
    rsna-fusion-cv5-smoke -> fold 0, 4 optimizer updates, 24 fold studies

    python scripts/build_cv5_notebooks.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_fusion_v2_notebook import build  # noqa: E402

BASE = dict(epochs=5, validation_eval='every', eval_at_start=False, k=12, studies_per_gpu=1, effective_batch=16,
            freeze_bn_stats=True, sync_batchnorm=False, rad_input='pm1_middle', cv_folds=5, cv_seed=20261010,
            max_hours=7.0, resume_path=None)
KERNELS = {
    'rsna-fusion-cv5-smoke': dict(BASE, run_folds=[0], smoke=True, smoke_updates=4, smoke_validation_studies=24),
    'rsna-fusion-cv5-f01': dict(BASE, run_folds=[0, 1]),
    'rsna-fusion-cv5-f23': dict(BASE, run_folds=[2, 3]),
    'rsna-fusion-cv5-f4': dict(BASE, run_folds=[4]),
}

for slug, config in KERNELS.items():
    notebook, metadata = build(slug, config.get('smoke', False), config, 'ddp')
    out = ROOT/'kaggle'/slug
    out.mkdir(parents=True, exist_ok=True)
    (out/f'{slug}.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    (out/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(out)
(ROOT/'configs/cv5_kernels.json').write_text(json.dumps(KERNELS, indent=2), encoding='utf-8')
