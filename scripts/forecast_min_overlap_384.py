"""Forecast sampling counts from the completed metadata-only capacity audit."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.runtime_preprocess'));sys.path.insert(0,str(ROOT/'scripts'))
from preprocess_min_overlap_384 import allocate_triplets,RECIPE_SHA
rows=[]
for line in (ROOT/'results/series_safe_384/unique_audit_full/study_capacity.jsonl').read_text().splitlines():
    row=json.loads(line);allocation,optimal=allocate_triplets(row['eligible_slices_per_slot'])
    rows.append(dict(study_uid=row['study_uid'],split=row['split'],allocation=allocation.tolist(),**optimal))
summary=dict(studies=len(rows),recipe_sha256=RECIPE_SHA,splits={})
for split in ('all','train','held_out_gold'):
    selected=rows if split=='all' else [r for r in rows if r['split']==split]
    k=[r['target_triplets'] for r in selected]
    summary['splits'][split]=dict(studies=len(selected),triplets=sum(k),triplets_min=min(k),triplets_max=max(k),
        triplets_mean=float(np.mean(k)),triplets_median=float(np.median(k)),short_bags=sum(n<60 for n in k),
        unique_slices=sum(r['optimal_unique_slices'] for r in selected),
        minimum_repeated_slice_uses=sum(r['minimum_repeated_slice_uses'] for r in selected),
        zero_overlap_studies=sum(r['minimum_repeated_slice_uses']==0 for r in selected),
        maximum_disjoint_primary_windows=sum(r['optimal_disjoint_subset'] for r in selected))
report=ROOT/'results/min_overlap_384';report.mkdir(exist_ok=True)
(report/'sampling_forecast.json').write_text(json.dumps(summary,indent=2))
(report/'sampling_forecast_studies.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
print(json.dumps(summary,indent=2))
