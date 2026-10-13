"""Single-model Kaggle submission: Raptor v5 + RadImageNet fusion, fusion_best.pt (epoch-2 EMA).

Internet off. For every study in sample_submission.csv:
  DICOM -> six-slot preprocessing (preprocess_six_slot_raptor.prepare, the exact training recipe)
  -> same-series windows (all windows if a study has none) -> fused model, fp16, full bag
  -> sigmoid -> submission.csv.
No ensembling, no test-time augmentation, no other models.
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
CHECKPOINT_NAME = 'fusion_best.pt'
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


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    timm_version = ensure_timm()
    import numpy as np
    import pandas as pd
    import torch
    from models.raptor_v5_radimagenet import load_fused_checkpoint

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
        workers=workers, timm=timm_version, torch=torch.__version__, gpus=torch.cuda.device_count())

    # Start CPU workers before CUDA is initialised in this process.
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(groups, str(series_root)))

    ck_path = find_one(CHECKPOINT_NAME, 'rsna-fusion-ddp-train')
    ck = torch.load(ck_path, map_location='cpu', weights_only=True)
    selection = ck.get('selection')
    model = load_fused_checkpoint(ck)
    del ck
    cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if cuda else torch.device('cpu')
    amp = dict(device_type='cuda', dtype=torch.float16) if cuda else dict(device_type='cpu', enabled=False)
    model = model.to(device).eval()
    torch.backends.cudnn.benchmark = True
    log('model_loaded', checkpoint=str(ck_path), bytes=ck_path.stat().st_size, selection=selection,
        parameters=sum(p.numel() for p in model.parameters()))

    preds, failures, windows, prep_seconds = {}, {}, {}, []
    done = 0
    # Bounded look-ahead: at most `ahead` prepared studies (~40 MB each) wait in memory.
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
            x = torch.from_numpy(images)
            n = len(x)
            padded = -(-n//EVAL_CHUNK)*EVAL_CHUNK     # fixed chunk shape, as in training evaluation
            if padded > n:
                x = torch.cat([x, x[-1:].expand(padded-n, *x.shape[1:])])
            feats = []
            for chunk in x.split(EVAL_CHUNK):
                with torch.autocast(**amp):
                    feats.append(model.backbone(chunk.to(device, non_blocking=True).float()).float())
            with torch.autocast(**amp):
                logits = model.head(torch.cat(feats, 0)[:n][None])
            p = torch.sigmoid(logits.float())[0].cpu().numpy()
            if not np.isfinite(p).all():
                failures[uid] = 'non-finite prediction'
                continue
            preds[uid] = p
            windows[uid] = (n, n_all)
            if done <= 3 or done % 100 == 0 or done == len(ids):
                log('progress', done=done, total=len(ids), failed=len(failures), windows_used=n, windows_all=n_all,
                    elapsed_min=round((time.time()-t0)/60, 2))
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
    receipt = dict(model='Raptor v5 + RadImageNet fusion (single model, no ensemble, no TTA)',
                   checkpoint=str(ck_path), selection=selection, studies=len(ids), predicted=len(preds),
                   failed=failures, mean_windows_used=float(np.mean([w[0] for w in windows.values()])),
                   prep_seconds_median=float(np.median(prep_seconds)), minutes=round((time.time()-t0)/60, 2))
    Path('inference_receipt.json').write_text(json.dumps(receipt, indent=2))
    log('submission_written', **{k: v for k, v in receipt.items() if k != 'failed'}, failed=len(failures))


if __name__ == '__main__':
    main()
