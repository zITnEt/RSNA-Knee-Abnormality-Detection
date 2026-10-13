"""Bounded CPU preprocessing queue, with pilot gate and small validation receipts."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from kaggle_monitor_runtime import awake_on_ac

ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'results/series_safe_384'
STATE=REPORT/'queue_state.json'


def command(args):
    try:
        result=subprocess.run(['kaggle',*args],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',
                              timeout=120,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    except subprocess.TimeoutExpired:
        return 124,'Temporary Kaggle API timeout; retry on the next poll'
    return result.returncode,result.stdout+result.stderr


def persist(state):
    state['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    temporary=STATE.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(STATE)


def receipts(ref,directory):
    code,output=command(['kernels','output',ref,'-p',str(directory),'--file-pattern',r'(manifest|recipe|progress)\.json$|index\.jsonl$|failures\.jsonl$',
                         '--page-size','200','-q','-o'])
    manifests=list(directory.rglob('manifest.json'))
    if code or len(manifests)!=1:return None
    manifest=json.loads(manifests[0].read_text())
    if not manifest['complete'] or manifest['failures'] or any(r['cross_series_windows'] for r in manifest['completed']):
        raise ValueError('Incomplete or cross-series preprocessing output')
    return manifest


def main(hours=24):
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'monitor_pid.txt').write_text(str(os.getpid()))
    state=json.loads(STATE.read_text()) if STATE.exists() else dict(pilot='running',jobs=[
        dict(shard=i,ref=f'zafarnematov/rsna-series-safe-384-preprocessing-{i:02d}',state='pending') for i in range(32)])
    state.pop('needs_attention',None);state.pop('error',None)
    end=time.time()+hours*3600
    while time.time()<end:
        try:
            if state['pilot']!='complete':
                code,status=command(['kernels','status','zafarnematov/rsna-series-safe-384-preprocessing-pilot'])
                state['pilot_status']=status.strip()
                if code==0 and 'ERROR' in status:raise RuntimeError('Kaggle pilot failed')
                if code==0 and 'COMPLETE' in status:
                    manifest=receipts('zafarnematov/rsna-series-safe-384-preprocessing-pilot',REPORT/'pilot')
                    if manifest:
                        state['pilot']='complete';state['recipe_sha256']=manifest['recipe_sha256']
                        state['pilot_bytes']=sum(r['bytes'] for r in manifest['completed'])
                        print('PILOT_VALIDATED '+json.dumps(dict(studies=len(manifest['completed']),bytes=state['pilot_bytes'])),flush=True)
                persist(state)
                if state['pilot']!='complete':time.sleep(60);continue
            for job in state['jobs']:
                if job['state'] not in ('queued','running','receipt_wait'):continue
                code,status=command(['kernels','status',job['ref']]);job['last_status']=status.strip()
                if code:continue
                if 'ERROR' in status:raise RuntimeError(f"Shard {job['shard']} failed")
                if 'COMPLETE' in status:
                    manifest=receipts(job['ref'],REPORT/f"shard_{job['shard']:02d}")
                    job['state']='receipt_wait'
                    if manifest:
                        if manifest['recipe_sha256']!=state['recipe_sha256']:raise ValueError('Mixed recipes')
                        job.update(state='complete',validated_studies=len(manifest['completed']),
                                   output_bytes=sum(r['bytes'] for r in manifest['completed']))
                        print('SHARD_VALIDATED '+json.dumps(job),flush=True)
                else:job['state']='running' if 'RUNNING' in status else 'queued'
            active=sum(j['state'] in ('queued','running','receipt_wait') for j in state['jobs'])
            for job in state['jobs']:
                if active>=5:break
                if job['state']!='pending':continue
                code,out=command(['kernels','push','-p',str(ROOT/f"kaggle/series_safe_384_preprocessing/shard_{job['shard']:02d}"),'-t','10800'])
                job['last_push']=out.strip()
                if code==0 and 'successfully pushed' in out:
                    job['state']='queued';active+=1;print('SHARD_LAUNCHED '+job['ref'],flush=True)
                else:
                    if code==124 or any(s in out.lower() for s in ('nameresolutionerror','connectionerror','timed out')):
                        break
                    if not any(s in out.lower() for s in ('session count','capacity')):raise RuntimeError('Kaggle push failed: '+out[:1000])
                    break
                persist(state)
            state['completed_shards']=sum(j['state']=='complete' for j in state['jobs'])
            state['validated_studies']=sum(j.get('validated_studies',0) for j in state['jobs'])
            persist(state)
            if state['completed_shards']==32:
                manifests=[json.loads(next((REPORT/f'shard_{i:02d}').rglob('manifest.json')).read_text()) for i in range(32)]
                rows=[r for m in manifests for r in m['completed']];uids=[r['study_uid'] for r in rows]
                if len(uids)!=4407 or len(set(uids))!=4407 or sum(r['split']=='train' for r in rows)!=4333 or sum(r['split']=='held_out_gold' for r in rows)!=74:
                    raise ValueError('Full corpus coverage or Gold74 exclusion mismatch')
                summary=dict(complete=True,studies=4407,training=4333,heldout_gold=74,shards=32,
                    source_slices=sum(r['source_slices'] for r in rows),triplets=sum(r['windows'] for r in rows),
                    cross_series_triplets=0,bytes=sum(r['bytes'] for r in rows),recipe_sha256=state['recipe_sha256'],
                    notebooks=[j['ref'] for j in state['jobs']])
                (REPORT/'complete_corpus.json').write_text(json.dumps(summary,indent=2))
                state['complete']=True;persist(state);print('ALL_SHARDS_VALIDATED '+json.dumps(summary),flush=True);return
        except Exception as exc:
            state['needs_attention']=True;state['error']=str(exc);persist(state);print('MONITOR_STOP '+str(exc),flush=True);return
        time.sleep(60)
    state['monitor_expired']=True;persist(state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--hours',type=float,default=24)
    with awake_on_ac():main(**vars(parser.parse_args()))
