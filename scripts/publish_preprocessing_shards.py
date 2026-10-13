"""Publish verified notebook outputs as private Kaggle dataset shards.

Only preprocessed tensors are temporarily staged locally; raw DICOMs stay on
Kaggle. Bounded disk use: one shard at a time. Credentials are never copied.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
from kaggle_monitor_runtime import awake_on_ac

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'results/six_slot_preprocessing'
STAGING = ROOT/'data/preprocessed_kaggle_staging'
STATE = REPORT/'publication_state.json'


def log(event, **kwargs):
    value={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'event':event,**kwargs}
    with (REPORT/'publication_events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)


def safe_error(message):
    message=re.sub(r'with url: \S+', 'with url: <redacted>', str(message))
    return re.sub(r'https?://\S+', '<url redacted>', message)


def command(args, timeout=10800):
    result=subprocess.run(['kaggle',*args],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stdout[-1500:]+result.stderr[-1500:])
    return result.stdout


def persist(state):
    temp=STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state,indent=2),encoding='utf-8')
    temp.replace(STATE)


def main(hours=12):
    REPORT.mkdir(parents=True,exist_ok=True)
    STAGING.mkdir(parents=True,exist_ok=True)
    state=json.loads(STATE.read_text()) if STATE.exists() else {'shards':{},'complete':False}
    end=time.time()+hours*3600
    log('publisher_started',hours=hours)
    while time.time()<end:
        queue_file=REPORT/'queue_state.json'
        if not queue_file.exists():
            time.sleep(60)
            continue
        queue=json.loads(queue_file.read_text())
        for job in queue['jobs']:
            key=str(job['shard'])
            if job['state']!='complete' or state['shards'].get(key,{}).get('state')=='ready':
                continue
            shard=job['shard']
            dataset=f'zafarnematov/rsna-six-slot-raptor-v5-shard-{shard:02d}'
            folder=STAGING/f'shard_{shard:02d}'
            try:
                if shutil.disk_usage(STAGING).free < job.get('output_bytes',0)+2*1024**3:
                    raise OSError('Insufficient disk space to stage one preprocessed shard')
                folder.mkdir(parents=True,exist_ok=True)
                cached=list(folder.rglob('manifest.json'))
                cache_complete=False
                if len(cached)==1:
                    prior_manifest=json.loads(cached[0].read_text())
                    cache_complete=bool(prior_manifest.get('complete')) and all(
                        (cached[0].parent/r['file']).is_file() and (cached[0].parent/r['file']).stat().st_size==r['bytes']
                        for r in prior_manifest.get('completed',[]))
                if not cache_complete:
                    log('download_started',shard=shard,bytes=job.get('output_bytes'))
                    for attempt in range(4):
                        try:
                            command(['kernels','output',job['ref'],'-p',str(folder),'--file-pattern',r'\.npz$|\.csv$|\.json$|\.jsonl$','-q'])
                            break
                        except Exception as exc:
                            log('download_retry',shard=shard,attempt=attempt+1,message=safe_error(exc))
                            if attempt==3:
                                raise
                            time.sleep(30)
                else:
                    log('verified_cache_reused',shard=shard)
                manifests=list(folder.rglob('manifest.json'))
                if len(manifests)!=1:
                    raise ValueError('Expected exactly one shard manifest')
                data_root=manifests[0].parent
                manifest=json.loads(manifests[0].read_text())
                recipe=json.loads((data_root/'recipe.json').read_text())
                if not manifest.get('complete') or manifest['shard']!=shard or recipe.get('source_budget')!=64 or 'dicom_weight_type' not in recipe.get('selector',''):
                    raise ValueError('Incomplete or wrong shard')
                for row in manifest['completed']:
                    file=data_root/row['file']
                    digest=hashlib.sha256()
                    with file.open('rb') as f:
                        for block in iter(lambda:f.read(8*1024**2),b''):
                            digest.update(block)
                    if file.stat().st_size!=row['bytes'] or digest.hexdigest()!=row['sha256']:
                        raise ValueError(f"Checksum mismatch: {row['file']}")
                sample=data_root/manifest['completed'][0]['file']
                with np.load(sample,allow_pickle=False) as saved:
                    x=saved['images']
                    if x.dtype!=np.float16 or x.shape[1:]!=(3,384,384) or not np.isfinite(x).all():
                        raise ValueError('Invalid saved training tensor')
                metadata={'id':dataset,'title':f'RSNA Six Slot Raptor V5 Shard {shard:02d}',
                          'licenses':[{'name':'other'}],
                          'description':'Private preprocessed RSNA competition training tensors. Underlying data remains subject to RSNA competition terms; no new rights are granted. Six MRI categories; Raptor V5 pixel math: 140mm center crop, 2–98% source span, per-series p2/p98, 336 uint8 cache, bilinear 384, ImageNet normalization. Float16 [K,3,384,384]. Gold74 excluded from train split. Raw labels remain separate. See recipe.json and manifest.json for exact provenance.'}
                (data_root/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
                shutil.copyfile(ROOT/'raptor_data/ready_tensors.py',data_root/'ready_tensors.py')
                shutil.copyfile(ROOT/'docs/SIX_SLOT_RAPTOR_PREPROCESSING.md',data_root/'README.md')
                shutil.copyfile(ROOT/'kaggle/six_slot_training_inputs.ipynb',data_root/'training_inputs.ipynb')
                gold_file=ROOT/'data/external/jev_knee/original/gold74_eval_set.csv'
                if gold_file.is_file():
                    gold=pd.read_csv(gold_file,dtype={'StudyInstanceUID':str})
                    gold[gold.StudyInstanceUID.isin(manifest['expected_uids'])].to_csv(data_root/'labels_gold74.csv',index=False)
                prior=state['shards'].get(key,{})
                if not prior.get('created'):
                    api=KaggleApi()
                    api.authenticate()
                    datasets=[]
                    for page in range(1,100):
                        items=api.dataset_list(mine=True,page=page) or []
                        datasets.extend(items)
                        if len(items)<20:
                            break
                    used=sum(d.total_bytes for d in datasets if d.is_private)
                    required=sum(r['bytes'] for r in manifest['completed'])
                    if used+required>199*1000**3:
                        raise OSError('Private dataset quota would be exceeded; tensors remain available as private notebook outputs')
                    log('upload_started',shard=shard,dataset=dataset)
                    response=command(['datasets','create','-p',str(data_root),'-t','-q'])
                    if 'creation error' in response.lower():
                        raise RuntimeError(response.strip())
                    state['shards'][key]={'dataset':dataset,'state':'processing','created':True,'create_response':response[-1200:],
                                           'studies':len(manifest['completed']),'bytes':sum(r['bytes'] for r in manifest['completed'])}
                    persist(state)
                ready=False
                for _ in range(60):
                    try:
                        status=command(['datasets','status',dataset,'--format','json'],timeout=90)
                    except Exception as exc:
                        # A new large dataset may exist before its first
                        # version's status/permissions are visible to this API.
                        log('dataset_status_retry',shard=shard,message=safe_error(exc))
                        time.sleep(30)
                        continue
                    state['shards'][key]['last_status']=status.strip()
                    persist(state)
                    if 'ready' in status.lower():
                        ready=True
                        break
                    if 'error' in status.lower() or 'fail' in status.lower():
                        raise RuntimeError(status)
                    time.sleep(30)
                if not ready:
                    raise RuntimeError('Dataset processing did not become ready within 30 minutes')
                api=KaggleApi()
                api.authenticate()
                with api.build_kaggle_client() as client:
                    request=ApiGetDatasetRequest()
                    request.owner_slug,request.dataset_slug=dataset.split('/')
                    if not client.datasets.dataset_api_client.get_dataset(request).is_private:
                        raise ValueError('Dataset is unexpectedly public; stop publication')
                state['shards'][key]['state']='ready'
                state['ready_shards']=sum(s.get('state')=='ready' for s in state['shards'].values())
                state['ready_studies']=sum(s.get('studies',0) for s in state['shards'].values() if s.get('state')=='ready')
                persist(state)
                log('dataset_ready',shard=shard,dataset=dataset,studies=len(manifest['completed']))
                # Delete only this task's newly downloaded, successfully uploaded staging folder.
                resolved=folder.resolve()
                if resolved.parent!=STAGING.resolve() or resolved.name!=f'shard_{shard:02d}':
                    raise ValueError('Unexpected staging path; refusing cleanup')
                shutil.rmtree(resolved)
            except Exception as exc:
                state['needs_attention']=True
                state['error']={'shard':shard,'message':safe_error(exc)}
                persist(state)
                log('publication_paused',shard=shard,message=safe_error(exc))
                return 2
        state['ready_shards']=sum(s.get('state')=='ready' for s in state['shards'].values())
        state['ready_studies']=sum(s.get('studies',0) for s in state['shards'].values() if s.get('state')=='ready')
        state['complete']=state['ready_shards']==32
        persist(state)
        if state['complete']:
            log('all_datasets_ready',studies=state['ready_studies'])
            return 0
        time.sleep(60)
    state['publisher_expired']=True
    persist(state)
    return 3


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--hours',type=float,default=12)
    with awake_on_ac():
        raise SystemExit(main(**vars(p.parse_args())))
