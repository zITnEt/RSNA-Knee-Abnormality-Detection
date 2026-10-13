"""Verify minimum-overlap dataset V2 and preserved original dataset V1."""
import argparse,json,os,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
from kaggle_monitor_runtime import awake_on_ac
from task_state_io import atomic_json_write
ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'results/min_overlap_384'
STATE=REPORT/'publication_state.json'
PREFIX='zafarnematov/rsna-series-safe-384-shard-'


def persist(state):
    state['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    atomic_json_write(STATE,state)


def files(api,ref):
    out=[];token=None
    while True:
        response=api.dataset_list_files(ref,page_token=token,page_size=200)
        if response.error_message:raise RuntimeError(response.error_message)
        out.extend(response.files);token=response.next_page_token
        if not token:return {f.name:f.total_bytes for f in out}


def verify(shard):
    api=KaggleApi();api.authenticate();dataset=PREFIX+f'{shard:02d}'
    with api.build_kaggle_client() as client:
        request=ApiGetDatasetRequest();request.owner_slug,request.dataset_slug=dataset.split('/')
        if not client.datasets.dataset_api_client.get_dataset(request).is_private:raise ValueError('Unexpected public dataset')
    source=next((REPORT/f'bridge_{shard:02d}').rglob('manifest.json'))
    manifest=json.loads(source.read_text());actual=files(api,dataset+'/2')
    prefix=f'min_overlap_384_shard_{shard:02d}/'
    expected={prefix+r['file']:r['bytes'] for r in manifest['completed']}
    if {name:size for name,size in actual.items() if name.endswith('.npz')}!=expected:
        raise ValueError('V2 files differ from publication bridge')
    for name in ('manifest.json','recipe.json','publication_validation.json'):
        if actual.get(prefix+name)!=(source.parent/name).stat().st_size:raise ValueError('V2 metadata size mismatch')
    old_source=next((ROOT/f'results/series_safe_384/shard_{shard:02d}').rglob('manifest.json'))
    old_manifest=json.loads(old_source.read_text());old_actual=files(api,dataset+'/1')
    old_prefix=f'series_safe_384_shard_{shard:02d}/'
    old_expected={old_prefix+r['file']:r['bytes'] for r in old_manifest['completed']}
    if {name:size for name,size in old_actual.items() if name.endswith('.npz')}!=old_expected:
        raise ValueError('Original V1 is not intact')
    metadata=dict(id=dataset,title=f'RSNA Series Safe 384 Shard {shard:02d}',
        subtitle='V2: up to 60 triplets with minimum slice reuse; original V1 preserved',
        isPrivate=True,licenses=[{'name':'other'}],description=(ROOT/'docs/MIN_OVERLAP_384_DATASET.md').read_text()+
        '\n\nThis is dataset V2. Original series-safe arrays remain available in V1. '+
        'Ready loader: https://www.kaggle.com/code/zafarnematov/rsna-min-overlap-384-training-inputs\n')
    folder=REPORT/f'published_{shard:02d}';folder.mkdir(exist_ok=True)
    (folder/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2))
    api.dataset_metadata_update(dataset,str(folder))
    with api.build_kaggle_client() as client:
        assert client.datasets.dataset_api_client.get_dataset(request).is_private
    return dict(dataset=dataset,version=2,state='ready',private=True,original_v1_preserved=True,
        remote_file_sizes_verified=True,studies=len(expected),bytes=sum(expected.values()),
        training=sum(r['split']=='train' for r in manifest['completed']),
        gold=sum(r['split']=='held_out_gold' for r in manifest['completed']),recipe_sha256=manifest['recipe_sha256'])


def main(hours=24):
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'dataset_monitor_pid.txt').write_text(str(os.getpid()))
    state=json.loads(STATE.read_text()) if STATE.exists() else dict(datasets={},complete=False)
    end=time.time()+hours*3600
    while time.time()<end:
        try:
            ledger=REPORT/'created_versions.json';created=json.loads(ledger.read_text()) if ledger.exists() else []
            bridge=REPORT/'bridge_queue_state.json';jobs=json.loads(bridge.read_text())['jobs'] if bridge.exists() else []
            pending=[j['shard'] for j in jobs if j['state']=='complete' and j['shard'] in created and state['datasets'].get(str(j['shard']),{}).get('state')!='ready']
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures={pool.submit(verify,i):i for i in pending}
                for future in as_completed(futures):
                    shard=futures[future]
                    try:
                        state['datasets'][str(shard)]=future.result();persist(state)
                        print('PRIVATE_V2_VERIFIED '+json.dumps(state['datasets'][str(shard)]),flush=True)
                    except Exception as exc:
                        state['datasets'][str(shard)]=dict(state='processing',last_retry=str(exc)[-500:]);persist(state)
                        print('VERIFY_RETRY '+str(shard)+' '+type(exc).__name__,flush=True)
            ready=[d for d in state['datasets'].values() if d['state']=='ready']
            state.update(ready_shards=len(ready),ready_studies=sum(d['studies'] for d in ready),complete=len(ready)==32)
            persist(state)
            if state['complete']:
                assert state['ready_studies']==4407 and sum(d['gold'] for d in ready)==74
                state['bytes']=sum(d['bytes'] for d in ready);persist(state)
                print('ALL_DATASET_V2_READY '+str(state['bytes']),flush=True);return
        except Exception as exc:
            state['last_retry']=type(exc).__name__;persist(state);print('DATASET_MONITOR_RETRY '+type(exc).__name__,flush=True)
        time.sleep(60)
    state['monitor_expired']=True;persist(state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--hours',type=float,default=24)
    with awake_on_ac():main(**vars(parser.parse_args()))
