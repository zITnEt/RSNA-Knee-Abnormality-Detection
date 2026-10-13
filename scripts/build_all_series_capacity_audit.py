"""Read-only competition directory/header audit; produces no new images."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'kaggle/series_safe_384_all_series_capacity';folder.mkdir(exist_ok=True)
helper=(ROOT/'scripts/preprocess_six_slot_raptor.py').read_text().split('if __name__ == "__main__":')[0]
code="""import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
Path('/kaggle/working/preprocess_six_slot_raptor.py').write_text(BASE_HELPER)
sys.path.insert(0,'/kaggle/working')
from preprocess_six_slot_raptor import classify,find_root
SLOTS=[('Sagittal',1,1),('Sagittal',1,0),('Coronal',1,1),('Coronal',0,0),('Axial',1,1),('Sagittal',0,0)]
root=find_root()
series=pd.read_csv(root/'train_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
labels=pd.read_csv(root/'train.csv',dtype={'StudyInstanceUID':str})
series_root=next(root/name for name in ('train_series','train_images','train') if (root/name).is_dir())
jev=next(path for path in (Path('/kaggle/input/datasets/riadmohamed42/jev-knee-labels'),Path('/kaggle/input/jev-knee-labels')) if (path/'gold74_eval_set.csv').is_file())
gold=set(pd.read_csv(jev/'gold74_eval_set.csv',dtype={'StudyInstanceUID':str}).StudyInstanceUID)
groups={uid:g.to_dict('records') for uid,g in series.groupby('StudyInstanceUID',sort=False)}
def one(uid):
    records=[];seen=set();first={}
    for row in groups[uid]:
        series_uid=str(row['SeriesInstanceUID'])
        if series_uid in seen:continue
        seen.add(series_uid)
        selected=classify(row,series_root/uid/series_uid)
        if selected is None:continue
        n=len(selected['files']);lo=int(n*.02);hi=max(int(n*.98)-1,lo);capacity=hi-lo+1
        key=(selected['plane'],selected['fluid'],selected['fs'])
        slot=SLOTS.index(key) if key in SLOTS and (key[1] or selected['weight']=='T1') else None
        record=dict(series_uid=series_uid,slot=slot,weight=selected['weight'],native_slices=n,
            eligible_slices=capacity,disjoint_triplets=capacity//3,description=selected['description'])
        records.append(record)
        if slot is not None and slot not in first:first[slot]=record
    allowed=[r for r in records if r['slot'] is not None]
    return dict(study_uid=uid,split='held_out_gold' if uid in gold else 'train',series=records,
        first_series_max=sum(r['disjoint_triplets'] for r in first.values()),
        all_matching_series_max=sum(r['disjoint_triplets'] for r in allowed),
        all_series_max=sum(r['disjoint_triplets'] for r in records))
rows=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    for row in pool.map(one,sorted(labels.StudyInstanceUID.unique())):
        rows.append(row)
        if len(rows)%250==0:print('Audited',len(rows),'studies',flush=True)
assert len(rows)==4407
summary=dict(studies=len(rows),note='Upper bounds from native file counts and weighting headers; physical geometry/SOP uniqueness not yet validated',splits={})
for split in ('all','train','held_out_gold'):
    group=rows if split=='all' else [r for r in rows if r['split']==split]
    summary['splits'][split]=dict(studies=len(group),
        can_supply_60_first_series=sum(r['first_series_max']>=60 for r in group),
        can_supply_60_all_matching_series=sum(r['all_matching_series_max']>=60 for r in group),
        can_supply_60_all_series=sum(r['all_series_max']>=60 for r in group),
        matching_capacity_min=min(r['all_matching_series_max'] for r in group),
        matching_capacity_median=float(np.median([r['all_matching_series_max'] for r in group])),
        matching_capacity_max=max(r['all_matching_series_max'] for r in group))
Path('/kaggle/working/all_series_capacity.json').write_text(json.dumps(summary,indent=2))
Path('/kaggle/working/all_series_studies.jsonl').write_text(''.join(json.dumps(r)+'\\n' for r in rows))
print(json.dumps(summary,indent=2),flush=True)
"""
source='BASE_HELPER = '+repr(helper)+'\n'+code;ast.parse(source)
cells=[dict(cell_type='markdown',id='scope',metadata={},source=['# Unique-slice feasibility across all matching series\nRead headers and native slice counts only. No images or training data are changed.\n']),
    dict(cell_type='code',id='audit',metadata={},source=source.splitlines(keepends=True),execution_count=None,outputs=[])]
notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata={'kernelspec':dict(name='python3',language='python',display_name='Python 3')})
(folder/'audit.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
metadata=dict(id='zafarnematov/rsna-series-safe-384-all-series-capacity',title='RSNA Series Safe 384 All Series Capacity',code_file='audit.ipynb',
    language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_internet=False,
    competition_sources=['rsna-knee-abnormality-detection'],dataset_sources=['riadmohamed42/jev-knee-labels'],kernel_sources=[],model_sources=[])
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print('Built all-series native capacity audit')
