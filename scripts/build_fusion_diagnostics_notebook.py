"""Build the Kaggle notebook that runs scripts/fusion_diagnostics.py on 2 GPUs (one process per GPU).

    python scripts/build_fusion_diagnostics_notebook.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_fusion_v2_notebook import DATASETS, DOCKER, OWNER, PREPROCESSING, cell  # noqa: E402

SLUG = 'rsna-fusion-diagnostics'
SOURCES = {
    'models/__init__.py': ROOT/'models/__init__.py',
    'models/raptor_v5_radimagenet.py': ROOT/'models/raptor_v5_radimagenet.py',
    'raptor_data/__init__.py': ROOT/'raptor_data/__init__.py',
    'raptor_data/ready_tensors.py': ROOT/'raptor_data/ready_tensors.py',
    'raptor_data/cv_folds.py': ROOT/'raptor_data/cv_folds.py',
    'train_fusion_v2.py': ROOT/'scripts/train_fusion_v2.py',
    'fusion_diagnostics.py': ROOT/'scripts/fusion_diagnostics.py',
}
WEIGHT_KERNELS = [f'{OWNER}/rsna-fusion-cv5-f01', f'{OWNER}/rsna-fusion-cv5-f23', f'{OWNER}/rsna-fusion-cv5-f4',
                  f'{OWNER}/rsna-fusion-ddp-train', f'{OWNER}/rsna-fusion-ddp-resume']


def main():
    texts = {name: path.read_text(encoding='utf-8') for name, path in SOURCES.items()}
    digests = {name: hashlib.sha256(t.encode()).hexdigest()[:16] for name, t in texts.items()}
    intro = f"""# Fusion diagnostics (no training)

Residual magnitudes, fusion-off and RadImageNet-mean ablations for the 5 CV fold models and the 20-epoch run,
on gold74 and 150 left-out studies per fold; parent Raptor on the same studies. Outputs in `diag/`.

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
print('torch', torch.__version__, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
"""
    write_sources = ('from pathlib import Path\n'
                     f'SOURCES = {texts!r}\n'
                     "src = Path('/kaggle/working/fusion_src')\n"
                     'for name, text in SOURCES.items():\n'
                     '    path = src/name\n'
                     '    path.parent.mkdir(parents=True, exist_ok=True)\n'
                     "    path.write_text(text, encoding='utf-8')\n"
                     "print('sources written:', sorted(SOURCES))\n")
    launch = ('import os, subprocess, sys, time\n'
              "procs = []\n"
              "for rank in range(2):\n"
              "    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', PYTHONPATH=str(src),\n"
              "               CUDA_VISIBLE_DEVICES=str(rank), DIAG_RANK=str(rank))\n"
              "    log = open(f'/kaggle/working/diag_rank{rank}.log', 'w')\n"
              "    procs.append((subprocess.Popen([sys.executable, str(src/'fusion_diagnostics.py')], stdout=log,\n"
              "                                   stderr=subprocess.STDOUT, env=env, cwd='/kaggle/working'), log))\n"
              "while any(p.poll() is None for p, _ in procs):\n"
              "    time.sleep(60)\n"
              "    for rank in range(2):\n"
              "        lines = open(f'/kaggle/working/diag_rank{rank}.log').read().splitlines()\n"
              "        print(rank, lines[-1][:300] if lines else '', flush=True)\n"
              "for rank, (p, log) in enumerate(procs):\n"
              "    log.close()\n"
              "    print(f'--- rank {rank} exit {p.returncode}')\n"
              "    print(open(f'/kaggle/working/diag_rank{rank}.log').read()[-4000:])\n"
              "if any(p.returncode for p, _ in procs):\n"
              "    raise SystemExit('diagnostics failed')\n")
    notebook = dict(cells=[cell('markdown', intro), cell('code', setup), cell('code', write_sources),
                           cell('code', launch)],
                    metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                  language_info=dict(name='python')),
                    nbformat=4, nbformat_minor=5)
    metadata = dict(id=f'{OWNER}/{SLUG}', title='RSNA Fusion Diagnostics', code_file=f'{SLUG}.ipynb',
                    language='python', kernel_type='notebook', is_private=True, enable_gpu=True, enable_tpu=False,
                    enable_internet=True, keywords=[], dataset_sources=DATASETS,
                    kernel_sources=PREPROCESSING + WEIGHT_KERNELS, competition_sources=[], model_sources=[],
                    docker_image=DOCKER, machine_shape='NvidiaTeslaT4')
    out = ROOT/'kaggle'/SLUG
    out.mkdir(parents=True, exist_ok=True)
    (out/f'{SLUG}.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    (out/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(out, len(metadata['kernel_sources']), 'kernel sources')


if __name__ == '__main__':
    main()
