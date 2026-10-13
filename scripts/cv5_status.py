"""Status of the 5-fold CV kernels: Kaggle status, last progress event, per-epoch metrics, errors, GPU quota.

    python scripts/cv5_status.py [slug ...]      # default: rsna-fusion-cv5-f01 rsna-fusion-cv5-f23 rsna-fusion-cv5-f4
"""
import json
import sys
import threading
from datetime import datetime, timezone

from kaggle.api.kaggle_api_extended import KaggleApi

OWNER = 'zafarnematov'
SLUGS = sys.argv[1:] or ['rsna-fusion-cv5-f01', 'rsna-fusion-cv5-f23', 'rsna-fusion-cv5-f4']
api = KaggleApi(); api.authenticate()
logs = {}


def grab(slug):
    lines = logs.setdefault(slug, [])
    try:
        for ev in api.kernels_logs_stream(f'{OWNER}/{slug}'):
            lines.append(ev.get('data', ''))
    except Exception as exc:  # noqa: BLE001
        lines.append(f'STREAM_ERROR {exc!r}'[:200])


print('now', datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ'))
statuses = {}
for slug in SLUGS:
    try:
        st = api.kernels_status(f'{OWNER}/{slug}')
        statuses[slug] = (str(st.status).split('.')[-1], st.failure_message)
    except Exception as exc:  # noqa: BLE001
        statuses[slug] = ('UNKNOWN', str(exc)[:120])
active = [s for s in SLUGS if statuses[s][0] in ('RUNNING', 'COMPLETE', 'ERROR', 'QUEUED')]
threads = [threading.Thread(target=grab, args=(s,), daemon=True) for s in active]
[t.start() for t in threads]
[t.join(35) for t in threads]
for slug in SLUGS:
    status, failure = statuses[slug]
    print(f'\n== {slug}: {status}' + (f' ({failure})' if failure else ''))
    fold = None
    for raw in logs.get(slug, []):
        for line in raw.splitlines():
            line = line.strip()
            if 'torch.distributed.run' in line and '"fold"' in line:
                fold = json.loads(line.split('--config ', 1)[1]).get('fold')
            if 'Traceback' in line or 'Error' in line and not line.startswith('{'):
                print('  !!', line[:240])
            if not line.startswith('{"event"'):
                continue
            e = json.loads(line)
            kind = e['event']
            if kind == 'inputs_verified':
                print(f"  fold {fold}: started {e['utc']} train={e['training']} fold_val={e['validation']}")
            elif kind == 'epoch_complete':
                va, auc = e.get('validation_accuracy', {}), e['macro_auc']
                print(f"  fold {fold} epoch {e['epoch']} ({e['utc']}, {e['hours']} h) loss {e['training_loss']:.4f}"
                      + ''.join(f" | {k}: gold74 {auc[k]['gold74']:.4f} fold {auc[k].get('validation', float('nan')):.4f}"
                                f" VA {va.get(k, float('nan')):.4f}" for k in ('raw', 'ema')))
            elif kind in ('amp_update_skipped', 'paused_time_budget', 'training_complete'):
                print('  ', json.dumps(e)[:300])
            last = e if kind == 'training_progress' else locals().get('last')
    p = locals().get('last')
    if p and p is not None and slug in logs:
        print(f"  last progress {p['utc']}: epoch {p['epoch']} batch {p['batch']}/{p['batches']} step {p['optimizer_steps']}"
              f" loss {p['loss']:.4f} s/mb {p['sec_per_microbatch']} nonfinite {p['nonfinite_losses']} peak {p['peak_gpu_gib']} GiB")
    last = None
with api.build_kaggle_client() as client:
    q = client.kernels.kernels_api_client.get_accelerator_quota_statistics()
    print('\nGPU quota (API): used', q.gpu_quota.time_used, 'reserved', q.gpu_quota.time_reserved,
          'refresh', q.quota_refresh_time)
