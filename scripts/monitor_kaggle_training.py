"""Read-only, bounded monitor for the private fusion run; saves small receipts."""
import json
import os
import subprocess
import time
from pathlib import Path

from kaggle_monitor_runtime import awake_on_ac

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'results/fusion_training'
REF = 'zafarnematov/rsna-raptor-v5-radimagenet-fusion-train'


def command(args):
    result = subprocess.run(['kaggle', *args], cwd=ROOT, capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=90,
                            env={**os.environ, 'PYTHONIOENCODING':'utf-8'})
    return result.returncode, result.stdout+result.stderr


def main(hours=13):
    STATE.mkdir(parents=True, exist_ok=True)
    Path(STATE/'monitor_pid.txt').write_text(str(os.getpid()))
    stream = (STATE/'live.log').open('a', encoding='utf-8')
    watcher = None
    end = time.time()+hours*3600
    try:
        while time.time() < end:
            if watcher is None or watcher.poll() is not None:
                watcher = subprocess.Popen(['kaggle', 'kernels', 'logs', REF, '--follow'], cwd=ROOT,
                    stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, 'PYTHONUNBUFFERED':'1', 'PYTHONIOENCODING':'utf-8'})
            record = dict(ref=REF, checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            try:
                code, output = command(['kernels', 'status', REF])
                record.update(status=output.strip(), status_returncode=code)
                lines = (STATE/'live.log').read_text(encoding='utf-8', errors='replace').splitlines()
                for line in reversed(lines):
                    if line.startswith('{') and '"event"' in line:
                        try:
                            record['last_progress'] = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            pass
                terminal = code==0 and any(s in output.upper() for s in ('COMPLETE', 'ERROR', 'CANCEL'))
                if terminal:
                    # Avoid downloading gigabyte checkpoints. They stay available on Kaggle.
                    receipt_code, receipt_out = command(['kernels', 'output', REF, '-p', str(STATE/'receipts'),
                        '--file-pattern', r'(progress|training_config|training_split|history)\.json$|training_events\.jsonl$',
                        '--page-size', '200', '-q', '-o'])
                    record['receipt_status'] = receipt_code
                    log_code, log_out = command(['kernels', 'logs', REF])
                    if log_code==0:
                        (STATE/'completed_logs.json').write_text(log_out, encoding='utf-8')
                    record['monitor_finished'] = True
            except Exception as exc:
                record['monitor_error'] = type(exc).__name__
                terminal = False
            temporary = STATE/'monitor_state.tmp'
            temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
            temporary.replace(STATE/'monitor_state.json')
            if terminal:
                return
            time.sleep(60)
    finally:
        if watcher is not None and watcher.poll() is None:
            watcher.terminate()
        stream.close()


if __name__ == '__main__':
    with awake_on_ac():
        main()
