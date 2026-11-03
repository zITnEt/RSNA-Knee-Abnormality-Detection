"""Publish only validated direct-384 outputs as private Kaggle datasets.

Stages one preprocessed shard at a time. Never downloads source DICOMs.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.runtime_preprocess'))
import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
from kaggle_monitor_runtime import awake_on_ac
from preprocess_series_safe_384 import RECIPE_SHA, LABELS, validate_arrays

REPORT=ROOT/'results/series_safe_384'
STAGING=ROOT/'data/series_safe_384_staging'
STATE=REPORT/'publication_state.json'


def command(args,timeout=10800):
    result=subprocess.run(['kaggle',*args],cwd=ROOT,capture_output=True,text=True,
        encoding='utf-8',errors='replace',timeout=timeout,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    output=result.stdout+result.stderr
    if result.returncode or 'creation error' in output.lower():raise RuntimeError(output[-1800:])
    return output


def persist(state):
    state['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(STATE)


def log(event,**kwargs):
    row=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=event,**kwargs)
    with (REPORT/'publication_events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def private_dataset(dataset):
    api=KaggleApi();api.authenticate()
    with api.build_kaggle_client() as client:
        request=ApiGetDatasetRequest();request.owner_slug,request.dataset_slug=dataset.split('/')
        obj=client.datasets.dataset_api_client.get_dataset(request)
    if not obj.is_private:raise ValueError('Unexpected public dataset; stopping publication')
    return obj


def stage(key,ref,expected_bytes):
    folder=STAGING/('pilot' if key=='preview' else f'shard_{int(key):02d}')
    if shutil.disk_usage(STAGING).free<expected_bytes+2*1024**3:raise OSError('Insufficient staging disk space')
    folder.mkdir(parents=True,exist_ok=True)
    manifests=list(folder.rglob('manifest.json'))
    cached=False
    if len(manifests)==1:
        old=json.loads(manifests[0].read_text());base=manifests[0].parent
        cached=old.get('complete') and all((base/r['file']).is_file() and (base/r['file']).stat().st_size==r['bytes'] for r in old['completed'])
    if not cached:
        log('download_started',key=key,bytes=expected_bytes)
        command(['kernels','output',ref,'-p',str(folder),'--file-pattern',r'\.npz$|\.csv$|\.json$|\.jsonl$',
                 '--page-size','200','-q'])
    manifests=list(folder.rglob('manifest.json'))
    if len(manifests)!=1:raise ValueError('Expected one manifest')
    base=manifests[0].parent;manifest=json.loads(manifests[0].read_text());recipe=json.loads((base/'recipe.json').read_text())
    if not manifest['complete'] or manifest['failures'] or manifest['recipe_sha256']!=RECIPE_SHA or recipe['sha256']!=RECIPE_SHA:
        raise ValueError('Incomplete or incorrect preprocessing recipe')
    rows=manifest['completed'];uids=[r['study_uid'] for r in rows]
    if len(set(uids))!=len(uids) or set(uids)!=set(manifest['expected_uids']):raise ValueError('Study coverage mismatch')
    if key!='preview' and (manifest['shard']!=int(key) or manifest['shards']!=32):raise ValueError('Wrong shard')
    split=pd.read_csv(base/'splits.csv',dtype={'StudyInstanceUID':str})
    hybrid=pd.read_csv(base/'labels_hybrid_train.csv',dtype={'StudyInstanceUID':str})
    gold=pd.read_csv(base/'labels_gold74.csv',dtype={'StudyInstanceUID':str})
    training=set(split.loc[split['split']=='train','StudyInstanceUID'])
    heldout=set(split.loc[split['split']=='held_out_gold','StudyInstanceUID'])
    if training & heldout or training|heldout!=set(uids) or set(hybrid.StudyInstanceUID)!=training or set(gold.StudyInstanceUID)!=heldout:
        raise ValueError('Gold exclusion or split coverage mismatch')
    if hybrid.StudyInstanceUID.duplicated().any() or not np.isfinite(hybrid[LABELS].to_numpy()).all():raise ValueError('Invalid HYBRID targets')
    for row in rows:
        file=base/row['file']
        if file.resolve().parent not in (base.resolve(),(base/'train').resolve(),(base/'held_out_gold').resolve()):raise ValueError('Unexpected NPZ path')
        with file.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if file.stat().st_size!=row['bytes'] or digest!=row['sha256']:raise ValueError('NPZ checksum mismatch')
        with np.load(file,allow_pickle=False) as saved:
            validate_arrays(saved)
            audit=json.loads(str(saved['audit_json']))
            if audit['recipe_sha256']!=RECIPE_SHA or audit['study_uid']!=row['study_uid'] or len(saved['images'])!=row['windows']:
                raise ValueError('NPZ identity or recipe mismatch')
        if row['cross_series_windows'] or row['source_slices']>72 or row['windows']>60:raise ValueError('Invalid study allocation')
        target=base/file.name
        if file!=target:
            if target.exists():raise ValueError('Duplicate flattened study path')
            file.replace(target)
        row['file']=target.name
    manifests[0].write_text(json.dumps(manifest,indent=2))
    (base/'index.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    # Kaggle skips directories by default. All study files are explicitly flattened;
    # the split CSV, manifest and loader continue to exclude Gold74 from training.
    npzs=list(base.rglob('*.npz'))
    if len(npzs)!=len(rows) or any(p.parent!=base for p in npzs):raise ValueError('Unlisted or nested study files')
    log('all_arrays_validated',key=key,studies=len(rows),slices=sum(r['source_slices'] for r in rows),
        triplets=sum(r['windows'] for r in rows),cross_series=0)
    return folder,base,manifest


def publish(key,job,state,end):
    preview=key=='preview';suffix='preview' if preview else f'shard-{int(key):02d}'
    dataset='zafarnematov/rsna-series-safe-384-'+suffix
    prior=state['datasets'].get(key,{})
    if prior.get('state')=='ready':return
    folder=None
    if not prior.get('created'):
        folder,base,manifest=stage(key,job['ref'],job.get('output_bytes',0))
        required=sum(r['bytes'] for r in manifest['completed'])
        api=KaggleApi();api.authenticate();items=[]
        for page in range(1,100):
            batch=api.dataset_list(mine=True,page=page) or [];items.extend(batch)
            if len(batch)<20:break
        existing=next((d for d in items if d.ref==dataset),None)
        if existing is not None:
            private_dataset(dataset)
        else:
            used=sum(d.total_bytes for d in items if d.is_private)
            if used+required>199*1000**3:raise OSError('Private dataset storage quota exceeded; notebook outputs retained')
            metadata=dict(id=dataset,title='RSNA Series Safe 384 '+('Preview' if preview else f'Shard {int(key):02d}'),
                licenses=[{'name':'other'}],description='Private RSNA MRI preprocessing: six slots 18/14/12/8/12/8, up to 72 unique slices and 60 strictly single-series triplets. Physical DICOM order, modality rescaling, MONOCHROME1 inversion, per-series p2/p98 reference scaling, 140mm center crop, direct 384x384 uint8. Ready float16 ImageNet-normalized images included. Gold74 held out from training. HYBRID training labels separate. Use SeriesSafe384Dataset; do not preprocess again. Subject to original competition data terms. '+('Eight-study pipeline preview; not the complete corpus.' if preview else 'Part of a 32-shard corpus.'))
            (base/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2))
            shutil.copyfile(ROOT/'raptor_data/series_safe_ready.py',base/'series_safe_ready.py')
            shutil.copyfile(ROOT/'docs/SERIES_SAFE_384_DATASET.md',base/'README.md')
            shutil.copyfile(ROOT/'kaggle/series_safe_384_training_inputs.ipynb',base/'training_inputs.ipynb')
            log('upload_started',key=key,dataset=dataset,bytes=required)
            response=command(['datasets','create','-p',str(base),'-t','-q'])
            log('upload_accepted',key=key,response=response[-400:])
        state['datasets'][key]=dict(dataset=dataset,state='processing',created=True,studies=len(manifest['completed']),
            training=sum(r['split']=='train' for r in manifest['completed']),gold=sum(r['split']=='held_out_gold' for r in manifest['completed']),
            bytes=required,recipe_sha256=RECIPE_SHA)
        persist(state)
    for attempt in range(120):
        if time.time()>end:return
        try:
            status=command(['datasets','status',dataset,'--format','json'],timeout=120)
        except (RuntimeError,subprocess.TimeoutExpired) as exc:
            log('status_retry',key=key,attempt=attempt+1);time.sleep(30);continue
        state['datasets'][key]['last_status']=status.strip();persist(state)
        if 'ready' in status.lower():
            private_dataset(dataset);state['datasets'][key]['state']='ready';persist(state)
            log('dataset_ready',key=key,dataset=dataset,studies=state['datasets'][key]['studies'])
            folder=folder or STAGING/('pilot' if preview else f'shard_{int(key):02d}')
            resolved=folder.resolve()
            if resolved.parent!=STAGING.resolve() or resolved.name!=('pilot' if preview else f'shard_{int(key):02d}'):
                raise ValueError('Unexpected cleanup path')
            if resolved.exists():shutil.rmtree(resolved)
            return
        if 'error' in status.lower() or 'fail' in status.lower():raise ValueError('Kaggle dataset processing failed: '+status)
        time.sleep(30)
    raise TimeoutError('Dataset processing has not finished after one hour')


def main(hours=24):
    REPORT.mkdir(parents=True,exist_ok=True);STAGING.mkdir(parents=True,exist_ok=True)
    (REPORT/'publisher_pid.txt').write_text(str(os.getpid()))
    state=json.loads(STATE.read_text()) if STATE.exists() else dict(datasets={},complete=False)
    state.pop('needs_attention',None);state.pop('error',None)
    end=time.time()+hours*3600;failures=0
    while time.time()<end:
        try:
            queue=json.loads((REPORT/'queue_state.json').read_text())
            if queue.get('pilot')=='complete':
                publish('preview',dict(ref='zafarnematov/rsna-series-safe-384-preprocessing-pilot',output_bytes=queue['pilot_bytes']),state,end)
            for job in queue['jobs']:
                if job['state']=='complete':publish(str(job['shard']),job,state,end)
            ready=[d for k,d in state['datasets'].items() if k!='preview' and d['state']=='ready']
            state.update(ready_shards=len(ready),ready_studies=sum(d['studies'] for d in ready),complete=len(ready)==32)
            persist(state);failures=0
            if state['complete']:
                if state['ready_studies']!=4407 or sum(d['training'] for d in ready)!=4333 or sum(d['gold'] for d in ready)!=74:
                    raise ValueError('Full published coverage mismatch')
                log('all_datasets_ready',studies=4407,training=4333,gold=74,bytes=sum(d['bytes'] for d in ready));return
        except Exception as exc:
            failures+=1
            message=re.sub(r'https?://\S+','<url redacted>',str(exc))[-1500:]
            state['last_retry']=dict(message=message,consecutive=failures);persist(state)
            log('publication_retry',message=message,consecutive=failures)
            transient=isinstance(exc,(subprocess.TimeoutExpired,TimeoutError)) or any(w in message.lower() for w in
                ('connection','nameresolution','timed out','temporary','503','502','504','429','500','403'))
            if not transient or failures>=10:
                state['needs_attention']=True;state['error']=message;persist(state);return
        time.sleep(60)
    state['publisher_expired']=True;persist(state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--hours',type=float,default=24)
    with awake_on_ac():main(**vars(parser.parse_args()))
