"""Read audit metadata only; never change images or decode source DICOMs."""
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ALLOCATIONS=[18,14,12,8,12,8]
TARGET_TRIPLETS=[16,12,10,6,10,6]


def audit_one(task):
    root,row=task
    with np.load(root/row['file'],allow_pickle=False) as saved:
        audit=json.loads(str(saved['audit_json']))
    assert audit['study_uid']==row['study_uid']
    groups=defaultdict(list)
    for source in audit['source_files']:groups[source['slot']].append(source)
    capacity=[0]*6;sampled=[0]*6
    for slot,records in groups.items():
        assert len({r['series_uid'] for r in records})==1
        indices=[r['native_index'] for r in records]
        assert len(set(indices))==len(indices)
        # The existing linspace samples include both endpoints of the eligible
        # 2-98% native-index band, so their span recovers its full capacity.
        capacity[slot]=max(indices)-min(indices)+1;sampled[slot]=len(indices)
        assert sampled[slot]==min(ALLOCATIONS[slot],capacity[slot])
    return dict(study_uid=row['study_uid'],split=row['split'],eligible_slices_per_slot=capacity,
        eligible_slices=sum(capacity),max_disjoint_triplets=sum(n//3 for n in capacity),
        supports_60_with_reallocation=sum(n//3 for n in capacity)>=60,
        supports_fixed_60_allocation=all(n>=3*k for n,k in zip(capacity,TARGET_TRIPLETS)),
        current_samples=sum(sampled),current_triplets=row['windows'],
        disjoint_triplets_from_current_samples=sum(n//3 for n in sampled))


def run(roots,output):
    roots=[Path(root) for root in roots];output=Path(output);output.mkdir(parents=True,exist_ok=True)
    tasks=[]
    for root in roots:
        recipe=json.loads((root/'recipe.json').read_text())
        assert recipe['format']=='rsna_series_safe_384_v1'
        manifest=json.loads((root/'manifest.json').read_text())
        assert manifest['complete'] and not manifest['failures']
        tasks.extend((root,row) for row in manifest['completed'])
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(audit_one,tasks):
            rows.append(row)
            if len(rows)%250==0:print('Audited',len(rows),'studies',flush=True)
    assert len({row['study_uid'] for row in rows})==len(rows)
    summary=dict(studies=len(rows),selection='existing one selected series per slot; native 2-98% index band',
                 target_triplets_per_slot=TARGET_TRIPLETS,required_slices_per_slot=[3*n for n in TARGET_TRIPLETS],splits={})
    for split in ('all','train','held_out_gold'):
        selected=rows if split=='all' else [row for row in rows if row['split']==split]
        counts=np.asarray([r['max_disjoint_triplets'] for r in selected])
        if not len(selected):continue
        summary['splits'][split]=dict(studies=len(selected),supports_60_with_reallocation=sum(r['supports_60_with_reallocation'] for r in selected),
            supports_fixed_60_allocation=sum(r['supports_fixed_60_allocation'] for r in selected),
            max_disjoint_min=int(counts.min()),max_disjoint_median=float(np.median(counts)),max_disjoint_max=int(counts.max()),
            current_triplets=sum(r['current_triplets'] for r in selected),
            disjoint_triplets_from_current_samples=sum(r['disjoint_triplets_from_current_samples'] for r in selected))
    (output/'unique_slice_capacity.json').write_text(json.dumps(summary,indent=2))
    (output/'study_capacity.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    print(json.dumps(summary,indent=2),flush=True)
    return summary
