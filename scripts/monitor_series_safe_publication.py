"""Verify private datasets copied directly from preprocessing notebook outputs."""
import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
from kaggle_monitor_runtime import awake_on_ac

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'results/series_safe_384';STATE=REPORT/'publication_state.json'
PREFIX='zafarnematov/rsna-series-safe-384-shard-'


def persist(state):
    state['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(STATE)


def verify(api,shard,manifest):
    dataset=PREFIX+f'{shard:02d}';token=None;files=[]
    with api.build_kaggle_client() as client:
        request=ApiGetDatasetRequest();request.owner_slug,request.dataset_slug=dataset.split('/')
        obj=client.datasets.dataset_api_client.get_dataset(request)
    if not obj.is_private:raise ValueError('Unexpected public dataset')
    while True:
        result=api.dataset_list_files(dataset,page_token=token,page_size=200)
        if result.error_message:raise RuntimeError(result.error_message)
        files.extend(result.files);token=result.next_page_token
        if not token:break
    actual={f.name:f.total_bytes for f in files if f.name.endswith('.npz')}
    prefix=f'series_safe_384_shard_{shard:02d}/'
    expected={prefix+r['file']:r['bytes'] for r in manifest['completed']}
    if actual!=expected:raise ValueError('Remote NPZ paths/sizes differ from validated notebook manifest')
    receipt=REPORT/f'published_{shard:02d}';receipt.mkdir(exist_ok=True)
    source_manifest=next((REPORT/f'shard_{shard:02d}').rglob('manifest.json'))
    all_files={f.name:f.total_bytes for f in files}
    for name in ('recipe.json','manifest.json'):
        source=source_manifest.parent/name
        if all_files.get(prefix+name)!=source.stat().st_size:raise ValueError('Remote metadata file size mismatch')
        shutil.copyfile(source,receipt/name)
    recipe=json.loads((receipt/'recipe.json').read_text())
    if recipe['sha256']!=manifest['recipe_sha256']:raise ValueError('Recipe mismatch')
    metadata=dict(id=dataset,title=f'RSNA Series Safe 384 Shard {shard:02d}',
        subtitle='384x384 MRI slices and normalized single-series triplets; Gold74 held out',
        isPrivate=True,licenses=[{'name':'other'}],
        description=(ROOT/'docs/SERIES_SAFE_384_DATASET.md').read_text(encoding='utf-8')+
            f'\n\nThis shard contains {len(expected)} studies. '
            'Use the nested recipe directory as a SeriesSafe384Dataset root. '
            'Ready loader notebook: https://www.kaggle.com/code/zafarnematov/rsna-series-safe-384-training-inputs\n')
    (receipt/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('UPDATE_DESCRIPTION '+str(shard),flush=True)
    api.dataset_metadata_update(dataset,str(receipt))
    with api.build_kaggle_client() as client:
        if not client.datasets.dataset_api_client.get_dataset(request).is_private:
            raise ValueError('Dataset privacy changed unexpectedly')
    return dict(dataset=dataset,state='ready',private=True,remote_file_sizes_verified=True,
        manifest_file_size_verified=True,final_kaggle_validation_pending=True,studies=len(expected),bytes=sum(expected.values()),
        training=sum(r['split']=='train' for r in manifest['completed']),
        gold=sum(r['split']=='held_out_gold' for r in manifest['completed']),recipe_sha256=recipe['sha256'])


def main(hours=24):
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'publication_monitor_pid.txt').write_text(str(os.getpid()))
    state=json.loads(STATE.read_text()) if STATE.exists() else dict(datasets={},complete=False,method='Kaggle direct output copy')
    end=time.time()+hours*3600
    while time.time()<end:
        try:
            api=KaggleApi();api.authenticate()
            ledger=REPORT/'created_shards.json'
            existing={PREFIX+f'{i:02d}' for i in json.loads(ledger.read_text())} if ledger.exists() else set()
            queue=json.loads((REPORT/'queue_state.json').read_text())
            pending=[]
            for job in queue['jobs']:
                key=str(job['shard']);dataset=PREFIX+f"{job['shard']:02d}"
                if job['state']!='complete' or dataset not in existing or state['datasets'].get(key,{}).get('state')=='ready':continue
                pending.append(job)
            def one(job):
                worker_api=KaggleApi();worker_api.authenticate()
                manifest=json.loads(next((REPORT/f"shard_{job['shard']:02d}").rglob('manifest.json')).read_text())
                return verify(worker_api,job['shard'],manifest)
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures={pool.submit(one,job):job for job in pending}
                for future in as_completed(futures):
                    job=futures[future];key=str(job['shard']);dataset=PREFIX+f"{job['shard']:02d}"
                    try:
                        state['datasets'][key]=future.result();persist(state)
                        print('PRIVATE_DATASET_VERIFIED '+json.dumps(state['datasets'][key]),flush=True)
                    except Exception as exc:
                        state['datasets'][key]=dict(dataset=dataset,state='processing',last_retry=str(exc)[-500:]);persist(state)
                        print('VERIFY_RETRY '+json.dumps(dict(shard=job['shard'],error=type(exc).__name__)),flush=True)
            ready=[d for d in state['datasets'].values() if d['state']=='ready']
            state.update(ready_shards=len(ready),ready_studies=sum(d['studies'] for d in ready),complete=len(ready)==32)
            persist(state)
            if state['complete']:
                if state['ready_studies']!=4407 or sum(d['gold'] for d in ready)!=74 or sum(d['training'] for d in ready)!=4333:
                    raise ValueError('Published corpus coverage mismatch')
                print('PUBLICATION_COMPLETE '+json.dumps(dict(studies=4407,training=4333,gold=74,bytes=sum(d['bytes'] for d in ready))),flush=True)
                return
        except Exception as exc:
            state['last_retry']=type(exc).__name__;persist(state);print('MONITOR_RETRY '+type(exc).__name__,flush=True)
        time.sleep(60)
    state['monitor_expired']=True;persist(state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--hours',type=float,default=24)
    with awake_on_ac():main(**vars(parser.parse_args()))
