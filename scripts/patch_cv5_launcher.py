"""DDP notebook launcher: RUN_CONFIG['run_folds'] = [a, b, ...] trains those CV folds one after
another in the same Kaggle session, each in /kaggle/working/fold{k} with cfg['fold'] = k.

    python scripts/patch_cv5_launcher.py
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/'scripts/build_fusion_v2_notebook.py'
text = path.read_text(encoding='utf-8').replace('\r\n', '\n')
OLD = '''                  "cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',\\n"
                  "       f'--nproc_per_node={torch.cuda.device_count()}', str(src/'train_fusion_ddp.py'),\\n"
                  "       '--config', json.dumps(RUN_CONFIG)]\\n"
                  "env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',\\n"
                  "           PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONPATH=str(src))\\n"
                  "print(' '.join(cmd), flush=True)\\n"
                  "proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,\\n"
                  "                        env=env, cwd='/kaggle/working', bufsize=1)\\n"
                  'for line in proc.stdout:\\n'
                  "    print(line, end='', flush=True)\\n"
                  'if proc.wait() != 0:\\n'
                  "    raise SystemExit(f'DDP training exited with code {proc.returncode}')\\n")'''
NEW = '''                  "run_folds = RUN_CONFIG.pop('run_folds', None)\\n"
                  "runs = [(None, '/kaggle/working')] if run_folds is None else [(k, f'/kaggle/working/fold{k}') for k in run_folds]\\n"
                  "env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',\\n"
                  "           PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONPATH=str(src))\\n"
                  "for fold, workdir in runs:\\n"
                  "    cfg = dict(RUN_CONFIG) if fold is None else {**RUN_CONFIG, 'fold': fold}\\n"
                  "    os.makedirs(workdir, exist_ok=True)\\n"
                  "    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',\\n"
                  "           f'--nproc_per_node={torch.cuda.device_count()}', str(src/'train_fusion_ddp.py'),\\n"
                  "           '--config', json.dumps(cfg)]\\n"
                  "    print(' '.join(cmd), flush=True)\\n"
                  "    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,\\n"
                  "                            env=env, cwd=workdir, bufsize=1)\\n"
                  '    for line in proc.stdout:\\n'
                  "        print(line, end='', flush=True)\\n"
                  '    if proc.wait() != 0:\\n'
                  "        raise SystemExit(f'DDP training (fold {fold}) exited with code {proc.returncode}')\\n")'''
if "run_folds" in text:
    print('already patched')
else:
    if text.count(OLD) != 1:
        raise SystemExit(f'expected one launcher block, found {text.count(OLD)}')
    text = text.replace(OLD, NEW)
    path.write_bytes(text.encode('utf-8'))
print(path.name, hashlib.sha256(text.encode()).hexdigest())
