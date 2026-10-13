"""Build the internet-off Kaggle submission notebook for the 5-fold CV fusion ensemble.

    python scripts/build_cv5_submission_notebook.py                         # all 5 folds
    python scripts/build_cv5_submission_notebook.py --folds 0,1,2,3 --slug rsna-fusion-cv5-submit-smoke
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = 'zafarnematov'
SOURCES = {
    'models/__init__.py': ROOT/'models/__init__.py',
    'models/raptor_v5_radimagenet.py': ROOT/'models/raptor_v5_radimagenet.py',
    'preprocess_six_slot_raptor.py': ROOT/'scripts/preprocess_six_slot_raptor.py',
    'infer_fusion_cv5_submission.py': ROOT/'scripts/infer_fusion_cv5_submission.py',
}
FOLD_KERNELS = {0: 'rsna-fusion-cv5-f01', 1: 'rsna-fusion-cv5-f01', 2: 'rsna-fusion-cv5-f23',
                3: 'rsna-fusion-cv5-f23', 4: 'rsna-fusion-cv5-f4'}
DOCKER = 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'


def cell(kind, text):
    base = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True))
    if kind == 'code':
        base.update(execution_count=None, outputs=[])
    return base


def build(slug, folds):
    texts = {name: path.read_text(encoding='utf-8') for name, path in SOURCES.items()}
    digests = {name: hashlib.sha256(t.encode()).hexdigest()[:16] for name, t in texts.items()}
    kernels = sorted({f'{OWNER}/{FOLD_KERNELS[k]}' for k in folds})
    intro = f"""# RSNA Knee - 5-fold CV fusion ensemble submission

Raptor v5 CoAtNet + RadImageNet ResNet-50 residual fusion. Folds {folds}: `fusion_best.pt` of each fold
(selected by validation accuracy = mean of gold74 and left-out-fold macro AUC). Prediction = mean of the
fold models' sigmoid probabilities; no TTA. Same six-slot preprocessing and same-series windows as training.

Embedded source digests: `{json.dumps(digests)}`
"""
    write_sources = ('from pathlib import Path\nimport sys\n'
                     f'SOURCES = {texts!r}\n'
                     "src = Path('/kaggle/working/fusion_src')\n"
                     'for name, text in SOURCES.items():\n'
                     '    path = src/name\n'
                     '    path.parent.mkdir(parents=True, exist_ok=True)\n'
                     "    path.write_text(text, encoding='utf-8')\n"
                     "print('sources written:', sorted(SOURCES), flush=True)\n")
    run = ('import os, subprocess, sys\n'
           f"env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONPATH=str(src), OMP_NUM_THREADS='1', "
           f"CV5_FOLDS={','.join(map(str, folds))!r})\n"
           "proc = subprocess.Popen([sys.executable, str(src/'infer_fusion_cv5_submission.py')], stdout=subprocess.PIPE,\n"
           "                        stderr=subprocess.STDOUT, text=True, env=env, cwd='/kaggle/working', bufsize=1)\n"
           'for line in proc.stdout:\n'
           "    print(line, end='', flush=True)\n"
           'if proc.wait() != 0:\n'
           "    raise SystemExit(f'Inference failed with exit code {proc.returncode}')\n"
           "import pandas as pd\n"
           "sub = pd.read_csv('/kaggle/working/submission.csv')\n"
           "print(sub.shape)\n"
           "print(sub.head())\n")
    notebook = dict(cells=[cell('markdown', intro), cell('code', write_sources), cell('code', run)],
                    metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                  language_info=dict(name='python')),
                    nbformat=4, nbformat_minor=5)
    metadata = dict(id=f'{OWNER}/{slug}', title=slug.replace('-', ' ').title().replace('Rsna', 'RSNA'),
                    code_file=f'{slug}.ipynb', language='python', kernel_type='notebook', is_private=True,
                    enable_gpu=True, enable_tpu=False, enable_internet=False, keywords=[],
                    dataset_sources=['mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2'],
                    kernel_sources=kernels, competition_sources=['rsna-knee-abnormality-detection'],
                    model_sources=[], docker_image=DOCKER, machine_shape='NvidiaTeslaT4')
    return notebook, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slug', default='rsna-fusion-cv5-submit')
    parser.add_argument('--folds', default='0,1,2,3,4')
    args = parser.parse_args()
    folds = [int(k) for k in args.folds.split(',')]
    out = ROOT/'kaggle'/args.slug
    out.mkdir(parents=True, exist_ok=True)
    notebook, metadata = build(args.slug, folds)
    (out/f'{args.slug}.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    (out/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(out, metadata['kernel_sources'])


if __name__ == '__main__':
    main()
