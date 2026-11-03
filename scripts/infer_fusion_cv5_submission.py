"""5-fold CV ensemble Kaggle submission: Raptor v5 + RadImageNet fusion.

Internet off. Loads fusion_best.pt of every CV fold (selected per fold by validation accuracy, the mean
of gold74 and left-out-fold macro AUC) from the attached rsna-fusion-cv5-* kernel outputs. For every
study in sample_submission.csv: DICOM, six-slot preprocessing (exact training recipe), same-series
windows, each fold model (fp16, full bag; models alternate across GPUs), sigmoid, mean over folds.
No test-time augmentation.
"""
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
TIMM_SHA = '888981753e65cbaacfc07494370138b1700a27b1f0af587f4f9b47bc024161d0'
CHECKPOINT_NAME = 'fusion_best.pt'   # one per fold
EVAL_CHUNK = 16
INPUT = Path(os.environ.get('RSNA_INPUT', '/kaggle/input'))


def log(kind, **data):
    print(json.dumps(dict(event=kind, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **data)), flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def attached_roots():
    """Attached datasets/notebook outputs, never the competition DICOM tree (too many files to walk)."""
    roots = []
    for base in (INPUT/'datasets', INPUT/'notebooks'):
        if base.is_dir():
            roots += [p for owner in base.iterdir() if owner.is_dir() for p in owner.iterdir() if p.is_dir()]
    roots += [p for p in INPUT.iterdir() if p.is_dir() and p.name not in
              ('datasets', 'notebooks', 'competitions', 'models', 'rsna-knee-abnormality-detection')]
    return roots


def find_one(name, slug):
    hits = sorted({p for r in attached_roots() if slug in str(r) for p in r.rglob(name)})
    if len(hits) != 1:
        raise FileNotFoundError(f'Expected exactly one {name} under {slug}; found {hits}')
    return hits[0]


def ensure_timm():
    try:
        version = importlib.metadata.version('timm')
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version != '1.0.22' and not os.environ.get('RSNA_SKIP_TIMM_INSTALL'):
        wheel = find_one('timm-1.0.22-py3-none-any.whl', 'rsna-knee-coatnet-d4-depthzone-swa3-b2')
        if sha256(wheel) != TIMM_SHA:
            raise RuntimeError('timm wheel hash mismatch')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--no-index', str(wheel)])
        importlib.invalidate_caches()
    for mod in [m for m in sys.modules if m == 'timm' or m.startswith('timm.')]:
        del sys.modules[mod]
    import timm
    if timm.__version__ != '1.0.22' and not os.environ.get('RSNA_SKIP_TIMM_INSTALL'):
        raise RuntimeError(f'timm {timm.__version__} loaded; 1.0.22 required')
    return timm.__version__


def competition_root():
    for p in (INPUT/'competitions'/'rsna-knee-abnormality-detection', INPUT/'rsna-knee-abnormality-detection'):
        if (p/'sample_submission.csv').is_file() and (p/'test_series.csv').is_file():
            return p
    raise FileNotFoundError('Attach the rsna-knee-abnormality-detection competition data')


# ----------------------------------------------------------------------------- preprocessing workers
_GROUPS, _SERIES_ROOT = None, None


def _init_worker(groups, series_root):
    global _GROUPS, _SERIES_ROOT
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    _GROUPS, _SERIES_ROOT = groups, Path(series_root)


def _prepare(uid):
    """Returns (uid, float16 [K,3,384,384] same-series windows, n_all_windows, error)."""
    import numpy as np
    from preprocess_six_slot_raptor import prepare
    started = time.time()
    try:
        rows = _GROUPS.get(uid)
        if not rows:
            raise ValueError('No test_series rows for study')
        data, audit = prepare(uid, rows, _SERIES_ROOT)
        images, same = data['images'], np.asarray(data['same_series'], bool)
        if same.any():              # training/eval convention: windows='pure' with fallback
            images = images[same]
        if images.dtype != np.float16 or images.ndim != 4 or images.shape[1:] != (3, 384, 384) or not len(images):
            raise ValueError(f'Bad prepared tensor {images.shape} {images.dtype}')
        return uid, np.ascontiguousarray(images), int(audit['windows']), None, round(time.time()-started, 2)
    except Exception as error:   # noqa: BLE001 - recorded and handled by the caller
        return uid, None, 0, f'{type(error).__name__}: {str(error)[:300]}', round(time.time()-started, 2)


# ----------------------------------------------------------------------------- checkpoints
def fold_checkpoints(folds):
    """fusion_best.pt of each CV fold, from the attached rsna-fusion-cv5-* kernel outputs."""
    hits = sorted({p for r in attached_roots() if 'rsna-fusion-cv5-f' in str(r) for p in r.rglob(CHECKPOINT_NAME)})
    found = {}
    for p in hits:
        tag = [part for part in p.parts if part.startswith('fold') and part[4:].isdigit()]
        if len(tag) == 1:
            found.setdefault(int(tag[0][4:]), []).append(p)
    missing = [k for k in folds if len(found.get(k, [])) != 1]
    if missing:
        raise FileNotFoundError(f'Need exactly one {CHECKPOINT_NAME} per fold; problems for folds {missing}: {found}')
    return [found[k][0] for k in folds]


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    timm_version = ensure_timm()
    import numpy as np
    import pandas as pd
    import torch
    from models.raptor_v5_radimagenet import load_fused_checkpoint

    folds = [int(k) for k in os.environ.get('CV5_FOLDS', '0,1,2,3,4').split(',')]
    budget_h = float(os.environ.get('CV5_BUDGET_HOURS', '7.5'))
    root = competition_root()
    sample = pd.read_csv(root/'sample_submission.csv', dtype={'StudyInstanceUID': str})
    if list(sample.columns) != ['StudyInstanceUID', *LABELS]:
        raise ValueError(f'Unexpected submission columns: {list(sample.columns)}')
    ids = list(sample.StudyInstanceUID)
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate study IDs in sample_submission.csv')
    series = pd.read_csv(root/'test_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    groups = {uid: g.to_dict('records') for uid, g in series.groupby('StudyInstanceUID', sort=False)}
    series_root = next((root/p for p in ('test_series', 'test_images', 'test') if (root/p).is_dir()), None)
    if series_root is None:
        raise FileNotFoundError('No test DICOM directory')
    workers = max(1, (os.cpu_count() or 2) - 1)
    log('inputs', studies=len(ids), studies_with_series=sum(u in groups for u in ids), series_root=str(series_root),
        workers=workers, timm=timm_version, torch=torch.__version__, gpus=torch.cuda.device_count(), folds=folds)

    # Start CPU workers before CUDA is initialised in this process.
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(groups, str(series_root)))

    cuda = torch.cuda.is_available()
    n_gpu = max(1, torch.cuda.device_count()) if cuda else 1
    models, info = [], []
    for k, path in zip(folds, fold_checkpoints(folds)):
        ck = torch.load(path, map_location='cpu', weights_only=True)
        cfg = ck.get('training_config', {})
        if cfg.get('fold') != k or ck.get('fusion_config', {}).get('rad_input') != 'pm1_middle':
            raise ValueError(f'{path}: fold {cfg.get("fold")} / rad_input {ck.get("fusion_config", {}).get("rad_input")}')
        device = torch.device('cuda', len(models) % n_gpu) if cuda else torch.device('cpu')
        model = load_fused_checkpoint(ck).to(device).eval()
        info.append(dict(fold=k, checkpoint=str(path), bytes=path.stat().st_size, selection=ck.get('selection'),
                         device=str(device)))
        models.append((model, device))
        del ck
    torch.backends.cudnn.benchmark = True
    log('models_loaded', models=info, parameters_each=sum(p.numel() for p in models[0][0].parameters()))

    def predict(images, use):
        """Mean sigmoid probability over models. Work for all GPUs is queued before any result is read,
        so models on different GPUs run concurrently."""
        x = torch.from_numpy(images)
        n = len(x)
        padded = -(-n//EVAL_CHUNK)*EVAL_CHUNK     # fixed chunk shape, as in training evaluation
        if padded > n:
            x = torch.cat([x, x[-1:].expand(padded-n, *x.shape[1:])])
        pinned = x.pin_memory() if cuda else x
        logits = []
        for model, device in use:
            amp = dict(device_type='cuda', dtype=torch.float16) if cuda else dict(device_type='cpu', enabled=False)
            feats = []
            for chunk in pinned.split(EVAL_CHUNK):
                with torch.autocast(**amp):
                    feats.append(model.backbone(chunk.to(device, non_blocking=True).float()).float())
            with torch.autocast(**amp):
                logits.append(model.head(torch.cat(feats, 0)[:n][None]).float())
        probs = np.stack([torch.sigmoid(l)[0].cpu().numpy() for l in logits])
        return probs.mean(0), n

    preds, failures, windows, prep_seconds, reduced = {}, {}, {}, [], []
    done = 0
    queue, remaining, ahead = deque(), iter(ids), 2*workers + 2

    def submit_next():
        uid = next(remaining, None)
        if uid is not None:
            queue.append(pool.submit(_prepare, uid))

    for _ in range(ahead):
        submit_next()
    with torch.inference_mode():
        while queue:
            uid, images, n_all, error, seconds = queue.popleft().result()
            submit_next()
            done += 1
            prep_seconds.append(seconds)
            if error is not None:
                failures[uid] = error
                log('study_failed', study=uid, error=error)
                continue
            # Time guard: past the budget, use one model per GPU for the remaining studies.
            use = models if (time.time()-t0)/3600 < budget_h else models[:n_gpu]
            if len(use) < len(models):
                reduced.append(uid)
            p, n = predict(images, use)
            if not np.isfinite(p).all():
                failures[uid] = 'non-finite prediction'
                continue
            preds[uid] = p
            windows[uid] = (n, n_all)
            if done <= 3 or done % 100 == 0 or done == len(ids):
                log('progress', done=done, total=len(ids), failed=len(failures), windows_used=n, windows_all=n_all,
                    models_used=len(use), elapsed_min=round((time.time()-t0)/60, 2))
    pool.shutdown()

    if not preds:
        raise RuntimeError('No study could be predicted')
    # Studies that could not be processed get the mean predicted probability (rank-neutral fallback).
    fallback = np.mean(np.stack(list(preds.values())), axis=0)
    rows = [[uid, *(preds.get(uid, fallback)).tolist()] for uid in ids]
    sub = pd.DataFrame(rows, columns=['StudyInstanceUID', *LABELS])
    values = sub[LABELS].to_numpy(float)
    if sub.shape != sample.shape or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise RuntimeError('Invalid submission')
    sub.to_csv('submission.csv', index=False)
    receipt = dict(model=f'Raptor v5 + RadImageNet fusion, {len(models)}-fold CV ensemble (mean of sigmoid '
                         'probabilities, fusion_best.pt per fold, no TTA)',
                   folds=info, studies=len(ids), predicted=len(preds), failed=failures,
                   reduced_ensemble_studies=len(reduced),
                   mean_windows_used=float(np.mean([w[0] for w in windows.values()])),
                   prep_seconds_median=float(np.median(prep_seconds)), minutes=round((time.time()-t0)/60, 2))
    Path('inference_receipt.json').write_text(json.dumps(receipt, indent=2))
    log('submission_written', **{k: v for k, v in receipt.items() if k not in ('failed', 'folds')},
        failed=len(failures))


if __name__ == '__main__':
    main()
