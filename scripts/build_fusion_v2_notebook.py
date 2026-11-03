"""Build the self-contained Kaggle notebook + metadata for the v2 fusion fine-tune.

The notebook embeds the exact repository sources of the model, the dataset and the
training script, writes them under /kaggle/working/fusion_src, and calls main(RUN_CONFIG).

    python scripts/build_fusion_v2_notebook.py --slug rsna-fusion-v2-smoke --smoke
    python scripts/build_fusion_v2_notebook.py --slug rsna-fusion-v2-train
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
    'raptor_data/__init__.py': ROOT/'raptor_data/__init__.py',
    'raptor_data/ready_tensors.py': ROOT/'raptor_data/ready_tensors.py',
    'raptor_data/cv_folds.py': ROOT/'raptor_data/cv_folds.py',
    'train_fusion_v2.py': ROOT/'scripts/train_fusion_v2.py',
    'train_fusion_ddp.py': ROOT/'scripts/train_fusion_ddp.py',
}
DATASETS = ['riadmohamed42/jev-knee-labels', 'dreaddevelopment/raptor-knee-maxspan',
            'marwanmath/resnet-50-radimagenet-marwan']
PREPROCESSING = [f'{OWNER}/rsna-six-slot-raptor-v5-preprocessing-{i:02d}' for i in range(32)]
DOCKER = 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'


def cell(kind, text):
    lines = text.splitlines(keepends=True)
    base = dict(cell_type=kind, metadata={}, source=lines)
    if kind == 'code':
        base.update(execution_count=None, outputs=[])
    return base


def build(slug, smoke, run_config, launcher='single', extra_kernels=()):
    texts = {name: path.read_text(encoding='utf-8') for name, path in SOURCES.items()}
    digests = {name: hashlib.sha256(t.encode()).hexdigest()[:16] for name, t in texts.items()}
    config = dict(run_config)
    if smoke:
        config.setdefault('smoke', True)
    title = 'Fusion v2 smoke test' if smoke else 'Fusion v2 fine-tune'
    if config.get('sync_batchnorm', False):
        bn_description = 'SyncBatchNorm updates statistics across GPUs on each training forward; inference uses saved running statistics.'
    elif config.get('freeze_bn_stats', True):
        bn_description = 'BatchNorm statistics frozen.'
    else:
        bn_description = 'BatchNorm running statistics update during training.'
    batch_description = (f"{config.get('studies_per_gpu', 1)} studies per GPU per microstep; "
                         f"effective optimizer batch {config.get('effective_batch', 16)} studies.") if launcher == 'ddp' else 'Single-process launcher.'
    intro = f"""# Raptor V5 + RadImageNet — {title}

Branch `codex/raptor-v5-radimagenet-fusion`, training script `scripts/train_fusion_v2.py`.

* fp16 autocast + GradScaler (T4 has no bf16 tensor cores). {bn_description}
* {batch_description} Gradient accumulation does not increase the BatchNorm batch.
* K={config.get('k', 12)} bags sampled evenly over present slots; series-boundary windows excluded.
* GPU augmentation: rotation/scale/shift, contrast/brightness/gamma, triplet-order reversal; no horizontal flip.
* Warmup + cosine LR (backbones 1e-5, fusion 1e-4, head 3e-5), EMA 0.998.
* Activation checkpointing follows RUN_CONFIG; the DDP memory probe uses the configured studies per GPU.
* Gold74 macro AUC (missing labels masked) and validation AUC/soft-BCE at step 0 (= parent Raptor)
  and after every epoch, for raw and EMA weights. With `fold` set, the validation set is that
  left-out fold and `fusion_best.pt` is selected by validation accuracy = mean of the gold74 and
  fold macro AUCs (`cv_metrics.csv`); otherwise by gold74 AUC.

Embedded source digests: `{json.dumps(digests)}`
"""
    setup = """import importlib.metadata, subprocess, sys
try:
    installed = importlib.metadata.version('timm')
except importlib.metadata.PackageNotFoundError:
    installed = None
if installed != '1.0.22':
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'timm==1.0.22'])
assert importlib.metadata.version('timm') == '1.0.22'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda,
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
"""
    write_sources = ('from pathlib import Path\nimport sys\n'
                     f'SOURCES = {texts!r}\n'
                     "src = Path('/kaggle/working/fusion_src')\n"
                     'for name, text in SOURCES.items():\n'
                     '    path = src/name\n'
                     '    path.parent.mkdir(parents=True, exist_ok=True)\n'
                     "    path.write_text(text, encoding='utf-8')\n"
                     'sys.path.insert(0, str(src))\n'
                     "print('sources written:', sorted(SOURCES))\n")
    run = f'RUN_CONFIG = {config!r}\n'
    if launcher == 'ddp':
        # One process per GPU via torchrun; output streamed into the notebook log.
        launch = ('import json, os, subprocess, sys, torch\n'
                  'from pathlib import Path as _P\n'
                  'def _find(name):\n'
                  "    hits = sorted(p for base in ('/kaggle/input/notebooks', '/kaggle/input') for p in _P(base).glob(f'*/*/{name}') if 'rsna-fusion-ddp-train' in str(p))\n"
                  "    hits = hits or sorted(p for p in _P('/kaggle/input').glob(f'*/{name}') if 'rsna-fusion-ddp-train' in str(p))\n"
                  "    assert len(hits) == 1, (name, hits)\n"
                  '    return str(hits[0])\n'
                  "if RUN_CONFIG.get('resume_path') == 'auto':\n"
                  "    RUN_CONFIG['resume_path'] = _find('fusion_last.pt')\n"
                  "for key, value in list((RUN_CONFIG.get('extra_checkpoints') or {}).items()):\n"
                  "    if value.startswith('auto:'):\n"
                  "        RUN_CONFIG['extra_checkpoints'][key] = _find(value[5:])\n"
                  "print('RUN_CONFIG', RUN_CONFIG, flush=True)\n"
                  "run_folds = RUN_CONFIG.pop('run_folds', None)\n"
                  "runs = [(None, '/kaggle/working')] if run_folds is None else [(k, f'/kaggle/working/fold{k}') for k in run_folds]\n"
                  "env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',\n"
                  "           PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONPATH=str(src))\n"
                  "for fold, workdir in runs:\n"
                  "    cfg = dict(RUN_CONFIG) if fold is None else {**RUN_CONFIG, 'fold': fold}\n"
                  "    os.makedirs(workdir, exist_ok=True)\n"
                  "    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',\n"
                  "           f'--nproc_per_node={torch.cuda.device_count()}', str(src/'train_fusion_ddp.py'),\n"
                  "           '--config', json.dumps(cfg)]\n"
                  "    print(' '.join(cmd), flush=True)\n"
                  "    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,\n"
                  "                            env=env, cwd=workdir, bufsize=1)\n"
                  '    for line in proc.stdout:\n'
                  "        print(line, end='', flush=True)\n"
                  '    if proc.wait() != 0:\n'
                  "        raise SystemExit(f'DDP training (fold {fold}) exited with code {proc.returncode}')\n")
    else:
        launch = 'import train_fusion_v2\ntrain_fusion_v2.main(RUN_CONFIG)\n'
    notebook = dict(cells=[cell('markdown', intro), cell('code', setup), cell('code', write_sources),
                           cell('code', run), cell('code', launch)],
                    metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                  language_info=dict(name='python')),
                    nbformat=4, nbformat_minor=5)
    metadata = dict(id=f'{OWNER}/{slug}', title=slug.replace('-', ' ').title().replace('Rsna', 'RSNA'),
                    code_file=f'{slug}.ipynb', language='python', kernel_type='notebook', is_private=True,
                    enable_gpu=True, enable_tpu=False, enable_internet=True, keywords=[],
                    dataset_sources=DATASETS, kernel_sources=PREPROCESSING + list(extra_kernels), competition_sources=[],
                    model_sources=[], docker_image=DOCKER, machine_shape='NvidiaTeslaT4')
    return notebook, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slug', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--config', default='{}', help='JSON overrides for RUN_CONFIG')
    parser.add_argument('--out', default=None)
    parser.add_argument('--launcher', choices=('single', 'ddp'), default='single')
    parser.add_argument('--extra-kernel', action='append', default=[])
    args = parser.parse_args()
    out = Path(args.out or ROOT/'kaggle'/args.slug)
    out.mkdir(parents=True, exist_ok=True)
    notebook, metadata = build(args.slug, args.smoke, json.loads(args.config), args.launcher, args.extra_kernel)
    (out/f'{args.slug}.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    (out/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
