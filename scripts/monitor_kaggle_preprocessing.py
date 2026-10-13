"""Bounded, resumable Kaggle queue monitor. Never fetches DICOMs or secrets.

Checks every 60s, launches up to five CPU notebooks, saves small validation
receipts, and stops on repeated/unknown failures instead of overwriting output.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from kaggle_monitor_runtime import awake_on_ac

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT/"results/six_slot_preprocessing"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE = STATE_DIR/"queue_state.json"


def command(args):
    result = subprocess.run(["kaggle", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
    return result.returncode, result.stdout+result.stderr


def write_state(state):
    state['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2), encoding='utf-8')
    temp.replace(STATE)


def log(event, **data):
    entry = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'event': event, **data}
    with (STATE_DIR/'monitor_events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry)+'\n')
    print(json.dumps(entry), flush=True)


def main(hours=12):
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        'format':'six_slot_raptor_v5_queue_v1','jobs':[
            {'shard': i, 'ref':f'zafarnematov/rsna-six-slot-raptor-v5-preprocessing-{i:02d}',
             'state':'running' if i < 3 else 'pending', 'push_attempts':1 if i < 3 else 0}
            for i in range(32)]}
    end = time.time()+hours*3600
    watchers = {}
    log('monitor_started', hours=hours, interval_seconds=60, concurrency=5)
    while time.time() < end:
        warm_active = 0
        for warm in state.get('warmups', []):
            if warm.get('state') == 'complete':
                continue
            try:
                code, output = command(['kernels', 'status', warm['ref']])
                if code == 0 and ('COMPLETE' in output.upper() or 'ERROR' in output.upper()):
                    warm['state'] = 'complete'
                else:
                    warm_active += 1
            except Exception:
                warm_active += 1
        for job in state['jobs']:
            if job['state'] not in ('running','queued','unknown'):
                continue
            key = job['shard']
            logfile = STATE_DIR/f'shard_{key:02d}_live.log'
            if key not in watchers:
                handle = logfile.open('w', encoding='utf-8')
                env = dict(os.environ, PYTHONUNBUFFERED='1')
                proc = subprocess.Popen(['kaggle', 'kernels', 'logs', job['ref'], '--follow'],
                                        cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env)
                watchers[key] = (proc, handle)
            if logfile.exists():
                for line in reversed(logfile.read_text(encoding='utf-8', errors='replace').splitlines()):
                    if line.startswith('{') and 'completed_studies' in line:
                        try:
                            progress = json.loads(line)
                            if progress.get('shard') == key:
                                job['live_progress'] = progress
                                break
                        except json.JSONDecodeError:
                            pass
            try:
                code, output = command(['kernels','status',job['ref']])
                job['last_status'] = output.strip()[:1200]
                if code:
                    job['status_errors'] = job.get('status_errors',0)+1
                    log('status_error', shard=job['shard'], message=output[:800])
                    continue
                upper = output.upper()
                if 'COMPLETE' in upper:
                    job['state'] = 'complete'
                    receipt = STATE_DIR/f"shard_{job['shard']:02d}"
                    manifests=[]
                    for attempt in range(4):
                        receipt_code, receipt_output=command(['kernels','output',job['ref'],'-p',str(receipt),
                             '--file-pattern',r'(manifest|recipe|progress)\.json$|index\.jsonl$|failures\.jsonl$',
                             '--page-size','200','-q','-o'])
                        manifests = list(receipt.rglob('manifest.json'))
                        if receipt_code==0 and manifests and list(receipt.rglob('recipe.json')):
                            break
                        log('receipt_retry',shard=key,attempt=attempt+1,exit_code=receipt_code)
                        time.sleep(15)
                    if not manifests:
                        job['state']='receipt_missing'
                    else:
                        manifest=json.loads(manifests[0].read_text())
                        job['validated_studies']=len(manifest.get('completed',[]))
                        job['output_bytes']=sum(r['bytes'] for r in manifest.get('completed',[]))
                        recipe = json.loads(next(receipt.rglob('recipe.json')).read_text())
                        if not manifest.get('complete') or recipe.get('source_budget') != 64 or 'dicom_weight_type' not in recipe.get('selector',''):
                            job['state']='incomplete'
                    log('job_finished', shard=job['shard'], state=job['state'], studies=job.get('validated_studies',0))
                    watcher = watchers.pop(key, None)
                    if watcher:
                        watcher[0].terminate() if watcher[0].poll() is None else None
                        watcher[1].close()
                elif 'ERROR' in upper or 'CANCEL' in upper or 'FAIL' in upper:
                    job['state']='error'
                    _, logs=command(['kernels','logs',job['ref']])
                    (STATE_DIR/f"shard_{job['shard']:02d}_error.log").write_text(logs,encoding='utf-8')
                    log('job_failed', shard=job['shard'], message=output[:800])
                elif 'RUNNING' in upper:
                    job['state']='running'
                elif 'QUEUED' in upper:
                    job['state']='queued'
                else:
                    job['state']='unknown'
            except Exception as exc:
                log('monitor_exception', shard=job['shard'], message=str(exc))
        # A deterministic recipe failure needs inspection before additional runs.
        errors = [j for j in state['jobs'] if j['state'] in ('error','incomplete','receipt_missing')]
        if errors:
            state['needs_attention']=True
            write_state(state)
            log('queue_paused_on_failure', shards=[j['shard'] for j in errors])
            return 2
        active = warm_active + sum(j['state'] in ('running','queued','unknown') for j in state['jobs'])
        for job in state['jobs']:
            if active >= 5:
                break
            if job['state'] != 'pending':
                continue
            try:
                code, output = command(['kernels','push','-p',str(ROOT/f"kaggle/six_slot_preprocessing/shard_{job['shard']:02d}"),'--timeout','10800'])
                job['push_attempts']+=1
                job['last_push']=output[:1200]
                if code == 0 and 'successfully pushed' in output:
                    job['state']='queued'
                    active+=1
                    log('job_launched', shard=job['shard'], ref=job['ref'])
                else:
                    log('push_deferred', shard=job['shard'], message=output[:1000])
                    if 'Maximum batch CPU session count' in output:
                        job['capacity_deferrals'] = job.get('capacity_deferrals',0)+1
                        job['push_attempts'] -= 1
                        break
                    if job['push_attempts']>=3:
                        job['state']='push_failed'
                        state['needs_attention']=True
                        write_state(state)
                        return 2
                    break
            except Exception as exc:
                log('push_exception', shard=job['shard'], message=str(exc))
                break
        state['completed_shards']=sum(j['state']=='complete' for j in state['jobs'])
        state['validated_studies']=sum(j.get('validated_studies',0) for j in state['jobs'])
        write_state(state)
        if state['completed_shards']==32:
            manifests=[json.loads(next((STATE_DIR/f"shard_{i:02d}").rglob('manifest.json')).read_text()) for i in range(32)]
            actual=[row['study_uid'] for manifest in manifests for row in manifest['completed']]
            expected=[uid for manifest in manifests for uid in manifest['expected_uids']]
            if len(actual)!=4407 or len(set(actual))!=4407 or set(actual)!=set(expected) or len({m['recipe_sha256'] for m in manifests})!=1:
                state['needs_attention']=True
                write_state(state)
                log('coverage_failed', studies=len(actual), unique_studies=len(set(actual)))
                return 2
            json_write_value={'complete':True,'studies':4407,'shards':32,
                              'recipe_sha256':manifests[0]['recipe_sha256'],
                              'notebooks':[j['ref'] for j in state['jobs']]}
            (STATE_DIR/'complete_corpus.json').write_text(json.dumps(json_write_value,indent=2),encoding='utf-8')
            state['complete']=True
            write_state(state)
            log('all_shards_complete', studies=state['validated_studies'])
            # Keep the temporary AC-power execution request alive until the
            # separate publisher finishes, without submitting more CPU jobs.
            publication=STATE_DIR/'publication_state.json'
            while time.time()<end and publication.exists():
                published=json.loads(publication.read_text())
                if published.get('complete'):
                    log('publication_complete',studies=published.get('ready_studies'))
                    break
                if published.get('needs_attention'):
                    log('publication_needs_attention',error=published.get('error'))
                    break
                state['waiting_for_publication']=True
                state['published_shards']=published.get('ready_shards',0)
                write_state(state)
                time.sleep(60)
            return 0
        time.sleep(60)
    state['monitor_expired']=True
    write_state(state)
    log('monitor_time_limit')
    return 3


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--hours',type=float,default=12)
    with awake_on_ac():
        raise SystemExit(main(**vars(p.parse_args())))
