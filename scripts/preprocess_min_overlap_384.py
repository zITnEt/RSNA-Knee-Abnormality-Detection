"""Same pixel contract as series-safe 384; globally minimize triplet reuse."""
from __future__ import annotations
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import preprocess_series_safe_384 as parent

SLOTS=parent.SLOTS
PARENT_SHA=parent.RECIPE_SHA
BASE_VALIDATE=parent.validate_arrays
TARGET_WEIGHTS=(16,12,10,6,10,6)
RECIPE={**parent.RECIPE,
    'format':'rsna_min_overlap_384_v1','parent_recipe_sha256':PARENT_SHA,
    'intensity_reference':'UNCHANGED parent reference indices: unique linspace over native 2-98% band, using 18/14/12/8/12/8 per selected series',
    'sampling':'up to 180 unique slices; maximize distinct slices used by up to 60 unique same-series windows',
    'source_budget':180,'triplet_budget':60,'slot_triplet_weights':TARGET_WEIGHTS,
    'allocation':'exact dynamic programming: maximize used unique slices, then maximum mutually disjoint subset, then minimize squared slot-weight deviation',
    'triplets':'three adjacent slices in each series sampled list; disjoint groups first, overlapping windows only as required; no duplicate windows',
    'window_fallback':'keep fewer than 60 if selected series have fewer than 60 unique neighboring three-slice windows; no duplicate padding',
    'selected_series':'UNCHANGED one first matching acquired series per six-slot category; no additional series',
    'missing_slots':'absent; triplet budget can shift to available selected slots to minimize reuse; no category substitution',
    'sampler_priority':'minimize repeated source-slice uses at the attainable triplet count, before matching category weights'}
RECIPE_SHA=hashlib.sha256(json.dumps(RECIPE,sort_keys=True).encode()).hexdigest()


def allocate_triplets(capacities,target=60):
    allocation,optimal=_allocate_cached(tuple(int(c) for c in capacities),int(target))
    return allocation.copy(),dict(optimal)


@lru_cache(maxsize=4096)
def _allocate_cached(capacities,target):
    """Exact optimal allocation for <=6 series and a <=60 triplet bag.

    c native eligible slices support c-2 distinct neighboring windows. With k
    windows, min(c,3*k) unique slices and min(k,c//3) disjoint windows are
    simultaneously attainable by series_windows below.
    """
    capacities=tuple(int(c) for c in capacities)
    if len(capacities)!=6 or any(c<0 for c in capacities):raise ValueError('Invalid six-slot capacity')
    target=min(int(target),sum(max(c-2,0) for c in capacities))
    if target<1:raise ValueError('No eligible three-slice window')
    weights=[w if c>=3 else 0 for w,c in zip(TARGET_WEIGHTS,capacities)]
    denominator=sum(weights)
    # State is (unique used, mutually disjoint subset, negative mix penalty).
    states={0:((0,0,0),())}
    for c,w in zip(capacities,weights):
        following={}
        for total,(score,path) in states.items():
            for k in range(min(max(c-2,0),target-total)+1):
                candidate=(score[0]+min(c,3*k),score[1]+min(k,c//3),
                    score[2]-(k*denominator-target*w)**2)
                old=following.get(total+k)
                if old is None or candidate>old[0] or (candidate==old[0] and path+(k,)>old[1]):
                    following[total+k]=(candidate,path+(k,))
        states=following
    score,allocation=states[target]
    return np.asarray(allocation,np.int32),dict(target_triplets=target,optimal_unique_slices=score[0],
        optimal_disjoint_subset=score[1],minimum_repeated_slice_uses=3*target-score[0])


def series_windows(lo,hi,k):
    """Return native sample indices, local windows, and a disjoint-subset mask."""
    c=int(hi-lo+1);k=int(k)
    if k<1 or c<3 or k>c-2:raise ValueError('Invalid series window count')
    count=min(c,3*k)
    picks=np.linspace(lo,hi,count).round().astype(np.int32)
    if len(np.unique(picks))!=count:raise ValueError('Duplicate native sample')
    q=min(k,count//3)
    baseline=set(range(0,3*q,3));starts=set(baseline)
    usage=np.zeros(count,np.int32)
    for start in starts:usage[start:start+3]+=1
    if count%3 and len(starts)<k:
        tail=count-3;starts.add(tail);usage[tail:tail+3]+=1
    while len(starts)<k:
        candidates=[s for s in range(count-2) if s not in starts]
        # All sources are already covered: minimize concentrated reuse.
        start=min(candidates,key=lambda s:(int((usage[s:s+3]+1).max()),
            int(((usage[s:s+3]+1)**2-usage[s:s+3]**2).sum()),s))
        starts.add(start);usage[start:start+3]+=1
    ordered=sorted(starts)
    windows=np.asarray([[s,s+1,s+2] for s in ordered],np.int32)
    primary=np.asarray([s in baseline for s in ordered],bool)
    assert (usage>0).all() and len(windows)==k and primary.sum()==min(k,c//3)
    return picks,windows,primary


def prepare(uid,rows,series_root):
    candidates=[]
    for row in rows:
        candidate=parent.classify(row,series_root/uid/str(row['SeriesInstanceUID']))
        if candidate is not None:candidates.append(candidate)
    used=set();selected=[];capacities=[]
    for slot,(name,plane,fluid,fs,reference_count) in enumerate(SLOTS):
        matches=[s for s in candidates if s['uid'] not in used and s['plane']==plane and
            s['fluid']==fluid and s['fs']==fs and (fluid or s['weight']=='T1')]
        if not matches:selected.append(None);capacities.append(0);continue
        series=matches[0];used.add(series['uid'])
        files=parent.physical_order(series,uid)
        refs=parent.sample_indices(len(files),reference_count)
        lo,hi=int(refs[0]),int(refs[-1])
        selected.append((series,files,refs,lo,hi));capacities.append(hi-lo+1)
    allocation,optimal=allocate_triplets(capacities)
    selection=[];slices=[];slot_ids=[];series_uids=[];sources=[];triplets=[];primary=[];intensity=[]
    for slot,(entry,k) in enumerate(zip(selected,allocation)):
        name,_,_,_,reference_count=SLOTS[slot]
        if entry is None:
            selection.append(dict(slot=name,present=False,triplets=0));continue
        series,files,refs,lo,hi=entry
        details=dict(slot=name,present=True,series_uid=series['uid'],description=series['description'],
            weight=series['weight'],native_depth=len(files),eligible_lo=lo,eligible_hi=hi,
            eligible_count=hi-lo+1,reference_allocation=reference_count,reference_count=len(refs),triplets=int(k))
        if not k:selection.append({**details,'sampled_count':0});continue
        picks,windows,disjoint=series_windows(lo,hi,int(k))
        arrays={int(i):parent.pixels(files[int(i)]['file']) for i in sorted(set(refs)|set(picks))}
        low,high=np.percentile(np.concatenate([arrays[int(i)].ravel() for i in refs]),[2,98])
        offset=len(slices)
        for native_index in picks:
            record=files[int(native_index)]
            slices.append(parent.crop_plane(arrays[int(native_index)],record['spacing'],low,high))
            slot_ids.append(slot);series_uids.append(series['uid'])
            sources.append(dict(slot=slot,series_uid=series['uid'],file=Path(record['file']).name,
                sop_uid=record['sop_uid'],native_index=int(native_index),position=record['position'],spacing=record['spacing']))
        triplets.extend((windows+offset).tolist());primary.extend(disjoint.tolist())
        selection.append({**details,'sampled_count':len(picks),'disjoint_subset':int(disjoint.sum())})
        intensity.append(dict(slot=slot,p2=float(low),p98=float(high),reference_indices=refs.tolist()))
    slices=np.stack(slices);indices=np.asarray(triplets,np.int32);slot_ids=np.asarray(slot_ids,np.int8)
    series_uids=np.asarray(series_uids);usage=np.bincount(indices.ravel(),minlength=len(slices)).astype(np.int8)
    audit=dict(study_uid=uid,recipe_sha256=RECIPE_SHA,parent_recipe_sha256=PARENT_SHA,
        selection=selection,intensity=intensity,source_files=sources,source_slices=len(slices),windows=len(indices),
        cross_series_windows=0,eligible_capacities=capacities,triplet_allocation=allocation.tolist(),
        repeated_slice_uses=int(3*len(indices)-len(slices)),reused_source_slices=int((usage>1).sum()),
        max_slice_usage=int(usage.max()),disjoint_subset_size=int(sum(primary)),**optimal)
    data=dict(slices_uint8=slices,slice_slot_ids=slot_ids,slice_series_uids=series_uids,
        triplet_indices=indices,triplet_series_uids=series_uids[indices[:,0]],
        images=parent.normalized_triplets(slices,indices),triplet_slot_ids=slot_ids[indices],
        same_series=np.ones(len(indices),bool),slot_present=np.asarray([s['present'] for s in selection],bool),
        slice_usage_counts=usage,primary_disjoint_subset=np.asarray(primary,bool),audit_json=np.asarray(json.dumps(audit,allow_nan=False)))
    validate_arrays(data)
    return data,audit


def validate_arrays(data):
    BASE_VALIDATE(data)
    indices=data['triplet_indices'];audit=json.loads(str(data['audit_json']))
    if len(indices)>60 or not np.all(np.diff(indices,axis=1)==1):raise ValueError('Invalid neighboring triplet')
    if len({tuple(row) for row in indices})!=len(indices):raise ValueError('Duplicate triplet')
    usage=np.bincount(indices.ravel(),minlength=len(data['slices_uint8']))
    if not (usage>0).all() or not np.array_equal(usage,data['slice_usage_counts']):raise ValueError('Invalid source reuse counts')
    if len({r['sop_uid'] for r in audit['source_files']})!=len(usage):raise ValueError('Reused or duplicate source SOP')
    allocation,optimal=allocate_triplets(audit['eligible_capacities'])
    if audit['recipe_sha256']!=RECIPE_SHA or audit['triplet_allocation']!=allocation.tolist():raise ValueError('Wrong sampling recipe')
    if len(indices)!=optimal['target_triplets'] or len(usage)!=optimal['optimal_unique_slices']:
        raise ValueError('Non-optimal unique-slice coverage')
    if 3*len(indices)-len(usage)!=optimal['minimum_repeated_slice_uses']:raise ValueError('Avoidable slice reuse')
    primary=data['primary_disjoint_subset']
    if primary.shape!=(len(indices),) or primary.sum()!=optimal['optimal_disjoint_subset']:
        raise ValueError('Non-optimal disjoint subset')
    members=indices[primary].ravel()
    if len(np.unique(members))!=len(members):raise ValueError('Primary subset overlaps')


def run(shard=0,shards=32,limit=None,root=None,output=None,workers=4):
    # Reuse the exact label, Gold74, save/reopen/checksum and progress machinery.
    # Only this notebook process's imported parent module is configured.
    parent.prepare=prepare;parent.validate_arrays=validate_arrays
    parent.RECIPE=RECIPE;parent.RECIPE_SHA=RECIPE_SHA
    return parent.run(shard=shard,shards=shards,limit=limit,root=root,
        output=output or f'/kaggle/working/min_overlap_384_shard_{shard:02d}',workers=workers)
