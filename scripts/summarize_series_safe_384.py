"""Summarize the actual exported corpus from all 32 completion manifests."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'results/series_safe_384'
files=[next((REPORT/f'shard_{i:02d}').rglob('manifest.json')) for i in range(32)]
manifests=[json.loads(p.read_text()) for p in files]
assert all(m['complete'] and not m['failures'] for m in manifests)
rows=[row for manifest in manifests for row in manifest['completed']]
assert len(rows)==4407 and len({r['study_uid'] for r in rows})==4407
summary={'recipe_sha256':manifests[0]['recipe_sha256'],'splits':{}}
for split in ['all','train','held_out_gold']:
    selected=rows if split=='all' else [r for r in rows if r['split']==split]
    k=np.asarray([r['windows'] for r in selected]);n=np.asarray([r['source_slices'] for r in selected])
    summary['splits'][split]=dict(studies=len(selected),triplets=int(k.sum()),source_slices=int(n.sum()),
        npz_bytes=sum(r['bytes'] for r in selected),triplets_min=int(k.min()),triplets_max=int(k.max()),
        triplets_median=float(np.median(k)),triplets_mean=float(k.mean()),
        triplet_count_histogram={str(int(value)):int((k==value).sum()) for value in np.unique(k)},
        slice_count_min=int(n.min()),slice_count_max=int(n.max()),slice_count_median=float(np.median(n)),
        slot_present_counts=np.asarray([r['slot_present'] for r in selected]).sum(axis=0).tolist(),
        cross_series_triplets=sum(r['cross_series_windows'] for r in selected))
summary['datasets']=[dict(shard=i,ref=f'zafarnematov/rsna-series-safe-384-shard-{i:02d}',
    url=f'https://www.kaggle.com/datasets/zafarnematov/rsna-series-safe-384-shard-{i:02d}') for i in range(32)]
summary['loader_notebook']='https://www.kaggle.com/code/zafarnematov/rsna-series-safe-384-training-inputs'
(REPORT/'dataset_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({key:{k:v for k,v in value.items() if k not in ('triplet_count_histogram','slot_present_counts')}
    for key,value in summary['splits'].items()},indent=2))
