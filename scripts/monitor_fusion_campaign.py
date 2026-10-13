"""Monitor/resume the authorized HYBRID full-bag 50-epoch run on available GPUs.

Uses private notebook outputs directly; never downloads images or checkpoints.
Pauses on unknown training failures or absent checkpoints, and never buys quota.
Run again to continue from persisted campaign_state.json after monitor expiry.
"""
import json
import os
import subprocess
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
from kaggle_monitor_runtime import awake_on_ac

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'results/fusion_training_hybrid50'
UPLOAD = ROOT/'kaggle/raptor_v5_radimagenet_training'
BASE_REF = 'zafarnematov/rsna-raptor-v5-radimagenet-fusion-train'


def write_state(state):
    state['checked_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    temporary = STATE/'campaign_state.tmp'
    temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
    temporary.replace(STATE/'campaign_state.json')


def command(args):
    result = subprocess.run(['kaggle', *args], cwd=ROOT, capture_output=True,
        text=True, encoding='utf-8', errors='replace', timeout=90,
        env={**os.environ, 'PYTHONIOENCODING':'utf-8'})
    return result.returncode, result.stdout+result.stderr


def make_continuation(previous, segment, budget):
    metadata = json.loads((UPLOAD/'kernel-metadata.json').read_text())
    notebook = json.loads((UPLOAD/'training.ipynb').read_text())
    slug = f'rsna-raptor-v5-radimagenet-fusion-train-s{segment:03d}'
    metadata.update(id=f'zafarnematov/{slug}', title=f'RSNA Raptor V5 RadImageNet Fusion Train S{segment:03d}')
    metadata['kernel_sources'].append(previous)
    previous_slug = previous.split('/')[-1]
    source = f'''from pathlib import Path
resume_files = [p for p in Path('/kaggle/input').rglob('fusion_last.pt') if {previous_slug!r} in str(p)]
if len(resume_files) != 1:
    raise RuntimeError(f'Expected one previous checkpoint; found {{resume_files}}')
RUN_CONFIG = {{'resume_path':str(resume_files[0]), 'max_hours':{budget!r}}}
'''
    notebook['cells'][4]['source'] = source.splitlines(keepends=True)
    folder = ROOT/'kaggle/fusion_continuations'/f'segment_{segment:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (folder/'training.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    return folder, metadata['id']


def main(days=14):
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE/'monitor_pid.txt').write_text(str(os.getpid()))
    path = STATE/'campaign_state.json'
    state = json.loads(path.read_text()) if path.exists() else dict(
        current_ref=BASE_REF, segment=1, phase='running', target_epochs=50, sessions=[])
    if state['phase'] in ('complete','needs_attention'):
        return
    api = KaggleApi(); api.authenticate()
    end = time.time()+days*24*3600
    watcher, stream = None, None
    next_launch_attempt = 0
    try:
        while time.time()<end:
            ref = state['current_ref']
            try:
                if state['phase']=='pending_continuation':
                    if time.time() >= next_launch_attempt:
                        with api.build_kaggle_client() as client:
                            quota = client.kernels.kernels_api_client.get_accelerator_quota_statistics()
                        gpu = quota.gpu_quota
                        remaining = (gpu.total_time_allowed-gpu.time_used-gpu.time_reserved).total_seconds()/3600
                        state['quota_remaining_hours'] = remaining
                        state['quota_refresh_utc'] = quota.quota_refresh_time.isoformat()
                        if remaining >= .75:
                            budget = min(4.5, remaining-.25)
                            folder, new_ref = make_continuation(ref,state['segment']+1,budget)
                            response = api.kernels_push(str(folder), timeout='21600', acc='NvidiaTeslaT4')
                            if response.error:
                                state['launch_error'] = response.error
                                if not any(s in response.error.lower() for s in ('quota','session count','maximum batch','capacity')):
                                    state['phase']='needs_attention'
                            elif response.invalid_kernel_sources or response.invalid_dataset_sources:
                                state['phase']='needs_attention'; state['launch_error']='Invalid attached sources'
                            else:
                                state['sessions'].append(dict(ref=new_ref,version=response.version_number,resume_from=ref))
                                state.update(current_ref=new_ref,segment=state['segment']+1,phase='running')
                                state.pop('launch_error',None)
                                if watcher is not None and watcher.poll() is None: watcher.terminate()
                                watcher=None
                                if stream is not None: stream.close()
                                stream=None
                        next_launch_attempt = time.time()+300
                elif state['phase']=='running':
                    log = STATE/f"segment_{state['segment']:03d}_live.log"
                    if watcher is None or watcher.poll() is not None:
                        if stream is None: stream=log.open('a',encoding='utf-8')
                        watcher=subprocess.Popen(['kaggle','kernels','logs',ref,'--follow'],cwd=ROOT,
                            stdout=stream,stderr=subprocess.STDOUT,
                            env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONIOENCODING':'utf-8'})
                    status = api.kernels_status(ref)
                    state['status']=status.status.name
                    state['failure_message']=status.failure_message
                    for line in reversed(log.read_text(encoding='utf-8',errors='replace').splitlines()):
                        if line.startswith('{') and '"event"' in line:
                            try: state['last_progress']=json.loads(line); break
                            except json.JSONDecodeError: pass
                    if state['status'] in ('COMPLETE','ERROR','CANCELLED','CANCELED'):
                        receipt = STATE/f"segment_{state['segment']:03d}_receipts"
                        code,out=command(['kernels','output',ref,'-p',str(receipt),'--file-pattern',
                            r'(progress|history|training_config)\.json$|training_events\.jsonl$',
                            '--page-size','200','-q','-o'])
                        files=list(receipt.rglob('progress.json'))
                        if code or not files:
                            state['receipt_waiting']=True
                            state['receipt_attempts']=state.get('receipt_attempts',0)+1
                            if state['receipt_attempts'] >= 5:state['phase']='needs_attention'
                        else:
                            progress=json.loads(files[0].read_text()); state['last_progress']=progress
                            if progress['event']=='training_complete':
                                state['phase']='complete'
                            elif progress['event']=='paused_time_budget':
                                state['phase']='pending_continuation'
                            elif state['status']=='ERROR' and any(s in status.failure_message.lower() for s in ('quota','time limit','timeout')):
                                # Only resume after a known resource limit with a published checkpoint.
                                request=ApiListKernelSessionOutputRequest()
                                request.user_name,request.kernel_slug=ref.split('/',1)
                                with api.build_kaggle_client() as client:
                                    outputs=client.kernels.kernels_api_client.list_kernel_session_output(request)
                                if any(p.file_name.endswith('fusion_last.pt') for p in outputs.files):
                                    state['phase']='pending_continuation'
                                else: state['phase']='needs_attention'
                            else: state['phase']='needs_attention'
                write_state(state)
                if state['phase'] in ('complete','needs_attention'):return
            except Exception as exc:
                state['monitor_error']=type(exc).__name__
                write_state(state)
            time.sleep(60)
        state['monitor_expired']=True
        write_state(state)
    finally:
        if watcher is not None and watcher.poll() is None:watcher.terminate()
        if stream is not None:stream.close()


if __name__=='__main__':
    with awake_on_ac():
        main()
