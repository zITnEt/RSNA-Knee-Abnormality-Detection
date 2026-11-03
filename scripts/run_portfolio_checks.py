"""Run existing CPU regression tests in isolated processes for each model variant."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    '.': [
        'test_cv_folds.py', 'test_series_triplets.py', 'test_series_dicom_reader.py',
        'test_six_slot_preprocessing.py', 'test_series_safe_384.py', 'test_min_overlap_384.py',
        'test_raptor_v5_radimagenet.py', 'test_checkpoint_batchnorm.py',
        'test_fusion_ready_training.py', 'test_train_fusion_ddp.py', 'test_train_fusion_v2.py',
        'test_codex_1_6.py', 'test_syncbn_batching.py',
    ],
    'experiments/resnet50_fusion': [
        'test_raptor_v5_resnet50.py', 'test_series_triplets.py', 'test_series_dicom_reader.py',
        'test_analyze_resnet50_run.py', 'test_train_resnet50_cloud.py',
        'test_train_resnet50_full_gold4.py', 'test_train_resnet50_cv5.py',
        'test_summarize_resnet50_cv5.py',
    ],
    'experiments/coatnet_only': [
        'test_raptor_v5.py', 'test_series_triplets.py', 'test_series_dicom_reader.py',
        'test_analyze_resnet50_run.py',
    ],
}
WORKER = """
import os,sys,unittest
import torch
torch.set_num_threads(2)
loader=unittest.TestLoader()
suite=unittest.TestSuite(loader.discover('tests',pattern=p) for p in sys.argv[1:])
if os.environ.get('PORTFOLIO_DISTRIBUTED') != '1' and 'test_train_fusion_ddp' in sys.modules:
    unittest.skip('Enable --distributed to run the local-socket Gloo integration test')(
        sys.modules['test_train_fusion_ddp'].DDPLockstep)
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distributed', action='store_true',
                        help='Also run the two-process Gloo test; requires local socket access')
    args = parser.parse_args()
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
    env['PORTFOLIO_DISTRIBUTED'] = '1' if args.distributed else '0'
    failures = []
    for variant, patterns in SUITES.items():
        print(f'Running CPU checks: {variant}', flush=True)
        completed = subprocess.run([sys.executable, '-c', WORKER, *patterns],
                                   cwd=ROOT / variant, env=env, check=False, timeout=300)
        if completed.returncode:
            failures.append(variant)
    if failures:
        print('FAILED: ' + ', '.join(failures))
        return 1
    print('PASS: all three variant suites. Tests needing the private label package may be skipped.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
