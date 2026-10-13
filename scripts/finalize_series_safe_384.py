"""Refresh private dataset cards with the completed corpus's exact recipe."""
import json
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'results/series_safe_384'
state=json.loads((REPORT/'publication_state.json').read_text())
assert state['complete'] and state['ready_shards']==32 and state['ready_studies']==4407
description=(ROOT/'docs/SERIES_SAFE_384_DATASET.md').read_text(encoding='utf-8')


def update(shard):
    folder=REPORT/f'published_{shard:02d}'
    path=folder/'dataset-metadata.json';metadata=json.loads(path.read_text())
    assert metadata['isPrivate'] is True
    metadata['description']=description+'\n\nReady-input notebook (all 32 shards already attached): https://www.kaggle.com/code/zafarnematov/rsna-series-safe-384-training-inputs\n'
    path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    api=KaggleApi();api.authenticate();api.dataset_metadata_update(metadata['id'],str(folder))
    return shard


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(update,i) for i in range(32)]
        completed=[future.result() for future in as_completed(futures)]
    (REPORT/'dataset_cards_finalized.json').write_text(json.dumps(dict(shards=sorted(completed),complete=True),indent=2))
    print('Finalized all 32 private dataset descriptions',flush=True)
