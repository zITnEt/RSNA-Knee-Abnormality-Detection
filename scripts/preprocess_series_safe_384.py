"""Six-slot direct-384 exports with provably single-series triplets."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from preprocess_six_slot_raptor import classify, pixels, save_npz, find_root, json_write
import pydicom

SLOTS = [
    ('SAG_FLUID_FS','Sagittal',1,1,18),
    ('SAG_FLUID_NOFS','Sagittal',1,0,14),
    ('COR_FLUID_FS','Coronal',1,1,12),
    ('COR_T1','Coronal',0,0,8),
    ('AX_FLUID_FS','Axial',1,1,12),
    ('SAG_T1','Sagittal',0,0,8),
]
LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
          'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
MEAN = np.asarray([.485,.456,.406],np.float32).reshape(1,3,1,1)
STD = np.asarray([.229,.224,.225],np.float32).reshape(1,3,1,1)
RECIPE = dict(format='rsna_series_safe_384_v1',slots=SLOTS,source_span=[.02,.98],
    depth_indices='lo=floor(N*.02); hi=max(floor(N*.98)-1,lo), inclusive',
    intensity_percentiles=[2,98],intensity_reference='the actual unique sampled slices in each selected series',
    sampling='evenly spaced unique physical-order slices, up to slot allocation; no redistribution or duplication',
    source_budget=72,crop_mm=140,crop='center rectangle using row/column PixelSpacing separately, clamp to acquired field',
    resolution=384,model_resolution=384,cache_resolution=384,resize='direct cv2.INTER_AREA to 384; no intermediate resize',
    quantization='floor(clipped_scaled_resized*255) uint8',slices_dtype='uint8',dtype='float16',normalized=True,
    mean=[.485,.456,.406],std=[.229,.224,.225],triplets='sliding three neighboring sampled slices WITHIN one SeriesInstanceUID',
    missing_slots='absent; never substitute, pad or duplicate slices',selector='dicom_weight_type; positive fat-sat header evidence overrides metadata',
    physical_order='common IOP slice normal dot IPP; fail on missing/inconsistent geometry',
    labels='Gold74 excluded from training split; held_out_gold separate; HYBRID scores separate from pixels')
RECIPE_SHA=hashlib.sha256(json.dumps(RECIPE,sort_keys=True).encode()).hexdigest()


def physical_order(series,study_uid):
    records=[]; sop_uids=set(); normal=None
    for file in series['files']:
        h=pydicom.dcmread(file,stop_before_pixels=True)
        if str(h.SeriesInstanceUID)!=series['uid'] or str(h.StudyInstanceUID)!=study_uid:
            raise ValueError('DICOM series/study ownership mismatch')
        sop=str(h.SOPInstanceUID)
        if sop in sop_uids:raise ValueError('Duplicate acquired SOPInstanceUID')
        sop_uids.add(sop)
        iop=np.asarray(getattr(h,'ImageOrientationPatient',[]),float)
        ipp=np.asarray(getattr(h,'ImagePositionPatient',[]),float)
        spacing=np.asarray(getattr(h,'PixelSpacing',[]),float)
        if iop.shape!=(6,) or ipp.shape!=(3,) or spacing.shape!=(2,) or not np.isfinite(np.r_[iop,ipp,spacing]).all() or (spacing<=0).any():
            raise ValueError('Missing/invalid physical geometry or PixelSpacing')
        candidate=np.cross(iop[:3],iop[3:])
        if np.linalg.norm(candidate)<.99:raise ValueError('Invalid slice normal')
        candidate=candidate/np.linalg.norm(candidate)
        if normal is None:normal=candidate
        if np.dot(normal,candidate)<.999:raise ValueError('Inconsistent slice orientations')
        records.append(dict(file=str(file),spacing=spacing.tolist(),position=float(np.dot(ipp,normal)),sop_uid=sop))
    records.sort(key=lambda r:(r['position'],r['sop_uid']))
    if len({round(r['position'],5) for r in records})!=len(records):
        raise ValueError('Duplicate physical slice positions')
    return records


def sample_indices(count,allocation):
    if count<1:raise ValueError('Empty MRI series')
    lo=int(count*.02); hi=max(int(count*.98)-1,lo)
    chosen=np.linspace(lo,hi,min(allocation,hi-lo+1)).round().astype(np.int32)
    if len(set(chosen.tolist()))!=len(chosen):raise ValueError('Duplicate sampled index')
    return chosen


def crop_plane(array,spacing,low,high):
    scaled=np.clip((array-low)/(high-low+1e-6),0,1)
    height,width=scaled.shape
    ch=min(max(1,int(round(140/spacing[0]))),height)
    cw=min(max(1,int(round(140/spacing[1]))),width)
    top,left=(height-ch)//2,(width-cw)//2
    resized=cv2.resize(scaled[top:top+ch,left:left+cw],(384,384),interpolation=cv2.INTER_AREA)
    return (resized*255).astype(np.uint8)


def normalized_triplets(slices,indices):
    x=slices[indices].astype(np.float32)/np.float32(255.)
    return ((x-MEAN)/STD).astype(np.float16)


def validate_arrays(data):
    slices,indices,x=data['slices_uint8'],data['triplet_indices'],data['images']
    if slices.dtype!=np.uint8 or slices.ndim!=3 or slices.shape[1:]!=(384,384):raise ValueError('Invalid stored slices')
    if not len(indices) or indices.shape!=(len(x),3) or (indices<0).any() or (indices>=len(slices)).any():raise ValueError('Invalid triplet indices')
    owners=data['slice_series_uids'][indices]
    if not np.all(owners==owners[:,:1]):raise ValueError('CROSS-SERIES TRIPLET')
    if not np.all(data['same_series']) or not np.all(data['triplet_slot_ids']==data['slice_slot_ids'][indices]):raise ValueError('Triplet provenance mismatch')
    if not np.all(owners[:,0]==data['triplet_series_uids']):raise ValueError('Triplet series UID mismatch')
    if x.dtype!=np.float16 or x.shape[1:]!=(3,384,384) or not np.isfinite(x).all():raise ValueError('Invalid normalized images')
    if not np.array_equal(normalized_triplets(slices,indices),x):raise ValueError('Normalized images differ from stored slices')


def prepare(uid,rows,series_root):
    candidates=[]
    for row in rows:
        candidate=classify(row,series_root/uid/str(row['SeriesInstanceUID']))
        if candidate is not None:candidates.append(candidate)
    used=set(); selection=[]; slices=[]; sources=[]; slot_ids=[]; series_uids=[]; triplets=[]; intensity=[]
    for slot,(name,plane,fluid,fs,allocation) in enumerate(SLOTS):
        matches=[s for s in candidates if s['uid'] not in used and s['plane']==plane and
                 s['fluid']==fluid and s['fs']==fs and (fluid or s['weight']=='T1')]
        if not matches:
            selection.append(dict(slot=name,present=False));continue
        series=matches[0];used.add(series['uid'])
        files=physical_order(series,uid); picks=sample_indices(len(files),allocation)
        arrays=[pixels(files[int(i)]['file']) for i in picks]
        low,high=np.percentile(np.concatenate([a.ravel() for a in arrays]),[2,98])
        offset=len(slices)
        for native_index,array in zip(picks,arrays):
            record=files[int(native_index)]
            slices.append(crop_plane(array,record['spacing'],low,high))
            slot_ids.append(slot);series_uids.append(series['uid'])
            sources.append(dict(slot=slot,series_uid=series['uid'],file=Path(record['file']).name,
                                sop_uid=record['sop_uid'],native_index=int(native_index),position=record['position'],spacing=record['spacing']))
        triplets.extend([[offset+i,offset+i+1,offset+i+2] for i in range(max(0,len(picks)-2))])
        selection.append(dict(slot=name,present=True,series_uid=series['uid'],description=series['description'],
                              weight=series['weight'],allocation=allocation,sampled_count=len(picks)))
        intensity.append(dict(slot=slot,p2=float(low),p98=float(high),reference_indices=picks.tolist()))
    if not triplets:raise ValueError('No series has three eligible sampled slices')
    slices=np.stack(slices); indices=np.asarray(triplets,np.int32); slot_ids=np.asarray(slot_ids,np.int8)
    series_uids=np.asarray(series_uids)
    x=normalized_triplets(slices,indices)
    audit=dict(study_uid=uid,recipe_sha256=RECIPE_SHA,selection=selection,source_slices=len(slices),windows=len(x),
               cross_series_windows=0,source_files=sources,intensity=intensity)
    data=dict(slices_uint8=slices,slice_slot_ids=slot_ids,slice_series_uids=series_uids,
              triplet_indices=indices,triplet_series_uids=series_uids[indices[:,0]],
              images=x,triplet_slot_ids=slot_ids[indices],same_series=np.ones(len(x),bool),
              slot_present=np.asarray([s['present'] for s in selection],bool),audit_json=np.asarray(json.dumps(audit,allow_nan=False)))
    validate_arrays(data)
    return data,audit


def run(shard=0,shards=32,limit=None,root=None,output=None,workers=4):
    torch.set_num_threads(1);cv2.setNumThreads(1)
    root=Path(root) if root else find_root()
    output=Path(output or f'/kaggle/working/series_safe_384_shard_{shard:02d}')
    output.mkdir(parents=True,exist_ok=True)
    series=pd.read_csv(root/'train_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
    labels=pd.read_csv(root/'train.csv',dtype={'StudyInstanceUID':str})
    jev=next((p for p in [Path('/kaggle/input/datasets/riadmohamed42/jev-knee-labels'),Path('/kaggle/input/jev-knee-labels')]
              if (p/'gold74_eval_set.csv').is_file()),None)
    if jev is None:raise FileNotFoundError('Gold74 exclusion requires riadmohamed42/jev-knee-labels')
    gold_labels=pd.read_csv(jev/'gold74_eval_set.csv',dtype={'StudyInstanceUID':str})
    gold=set(gold_labels.StudyInstanceUID)
    official_gold=set(labels.loc[labels[LABELS].notna().all(axis=1),'StudyInstanceUID'])
    if len(gold)!=74 or not official_gold<=gold:raise ValueError('Gold74 metadata mismatch')
    all_ids=sorted(labels.StudyInstanceUID.unique());ids=all_ids[shard::shards]
    if limit is not None:ids=ids[:limit]
    if not set(gold)<=set(all_ids):raise ValueError('Gold74 IDs outside competition corpus')
    groups={str(uid):group.to_dict('records') for uid,group in series.groupby('StudyInstanceUID',sort=False)}
    series_root=next((root/p for p in ('train_series','train_images','train') if (root/p).is_dir()),None)
    if series_root is None:raise FileNotFoundError('Training DICOM directory not found')
    json_write(output/'recipe.json',{**RECIPE,'sha256':RECIPE_SHA})
    split=pd.DataFrame(dict(StudyInstanceUID=ids,split=['held_out_gold' if uid in gold else 'train' for uid in ids]))
    split.to_csv(output/'splits.csv',index=False)
    labels[labels.StudyInstanceUID.isin(ids)].to_csv(output/'labels_official.csv',index=False)
    gold_labels[gold_labels.StudyInstanceUID.isin(ids)].to_csv(output/'labels_gold74.csv',index=False)
    train_ids=set(ids)-gold
    hybrid=pd.read_csv(jev/'HYBRID_labels_scores.csv',dtype={'StudyInstanceUID':str})
    hybrid=hybrid[hybrid.StudyInstanceUID.isin(train_ids)]
    if set(hybrid.StudyInstanceUID)!=train_ids or hybrid.StudyInstanceUID.duplicated().any() or not np.isfinite(hybrid[LABELS].to_numpy()).all():
        raise ValueError('Missing or invalid non-gold HYBRID targets')
    hybrid.to_csv(output/'labels_hybrid_train.csv',index=False)
    results=[];failures=[];start=time.time()
    def progress(state='running'):
        record=dict(state=state,shard=shard,shards=shards,expected_studies=len(ids),completed_studies=len(results),
                    failed_studies=len(failures),output_bytes=sum(r['bytes'] for r in results),
                    elapsed_seconds=round(time.time()-start,1),recipe_sha256=RECIPE_SHA,
                    updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        json_write(output/'progress.json',record);print(json.dumps(record),flush=True)
    def one(uid):
        if shutil.disk_usage(output).free<512*1024**2:raise OSError('Insufficient disk space')
        data,audit=prepare(uid,groups[uid],series_root)
        split_name='held_out_gold' if uid in gold else 'train'
        folder=output/split_name;folder.mkdir(exist_ok=True)
        file=folder/(uid+'.npz');save_npz(file,data)
        with np.load(file,allow_pickle=False) as saved:
            validate_arrays(saved)
            if not np.array_equal(saved['slices_uint8'],data['slices_uint8']):raise ValueError('Saved uint8 slices mismatch')
        with file.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        return dict(study_uid=uid,file=str(file.relative_to(output)).replace('\\','/'),split=split_name,
                    windows=audit['windows'],source_slices=audit['source_slices'],cross_series_windows=0,
                    slot_present=data['slot_present'].tolist(),bytes=file.stat().st_size,sha256=digest)
    progress()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(one,uid):uid for uid in ids}
        for future in as_completed(futures):
            try:
                row=future.result();results.append(row)
                with (output/'index.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            except Exception as exc:
                failure=dict(study_uid=futures[future],error=f'{type(exc).__name__}: {exc}');failures.append(failure)
                with (output/'failures.jsonl').open('a') as stream:stream.write(json.dumps(failure)+'\n')
                print('STUDY_FAILED '+json.dumps(failure),flush=True)
            progress()
    manifest=dict(recipe_sha256=RECIPE_SHA,shard=shard,shards=shards,expected_uids=ids,completed=results,
                  failures=failures,complete=len(results)==len(ids) and not failures)
    json_write(output/'manifest.json',manifest);progress('complete' if manifest['complete'] else 'incomplete')
    if not manifest['complete']:raise RuntimeError('Incomplete shard; inspect failures.jsonl')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=32)
    parser.add_argument('--limit',type=int);parser.add_argument('--output');args=parser.parse_args();run(**vars(args))
