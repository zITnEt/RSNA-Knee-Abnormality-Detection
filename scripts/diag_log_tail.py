"""Print the tail of the rsna-fusion-diagnostics log (30 s timeout)."""
import threading

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()
out = []


def grab():
    try:
        for chunk in api.kernels_logs_stream('zafarnematov/rsna-fusion-diagnostics'):
            out.append(chunk if isinstance(chunk, str) else str(chunk))
    except Exception as e:  # noqa: BLE001
        out.append(f'ERR {e!r}')


t = threading.Thread(target=grab, daemon=True); t.start(); t.join(30)
text = ''.join(out)
lines = [l for l in text.splitlines() if l.strip()]
keys = [l for l in lines if any(k in l for k in ('evaluated', 'Traceback', 'Error', 'job', 'done', 'rank'))]
print('\n'.join(keys[-25:])[-4000:])
print('---- tail ----')
print('\n'.join(lines[-8:])[-2000:])
