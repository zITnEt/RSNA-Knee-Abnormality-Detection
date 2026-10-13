"""Load direct-384 preprocessed images without any pixel transformation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MinimumOverlap384Dataset(Dataset):
    def __init__(self,roots,split='train',k=None,labels=True):
        roots=[Path(p) for p in roots]
        recipes=[json.loads((p/'recipe.json').read_text()) for p in roots]
        if not recipes or any(r['format']!='rsna_min_overlap_384_v1' for r in recipes) or len({r['sha256'] for r in recipes})!=1:
            raise ValueError('Attach only the new minimum-overlap direct-384 recipe')
        if k is not None and (not isinstance(k,int) or k<1):raise ValueError('K must be positive or None')
        splits=pd.concat([pd.read_csv(p/'splits.csv',dtype={'StudyInstanceUID':str}) for p in roots],ignore_index=True)
        if splits.StudyInstanceUID.duplicated().any():raise ValueError('Duplicate shards/study IDs')
        wanted=set(splits.loc[splits['split']==split,'StudyInstanceUID'])
        self.files=sorted(p for root in roots for p in root.rglob('*.npz') if p.stem in wanted)
        if {p.stem for p in self.files}!=wanted or len(self.files)!=len(wanted):raise ValueError('Missing or duplicate study files')
        self.k=k;self.labels=None
        if labels:
            if split!='train':raise ValueError('Gold labels remain separate; use labels=False for evaluation images')
            table=pd.concat([pd.read_csv(p/'labels_hybrid_train.csv',dtype={'StudyInstanceUID':str}) for p in roots],ignore_index=True)
            findings=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
            if table.StudyInstanceUID.duplicated().any() or wanted!=set(table.StudyInstanceUID) or not np.isfinite(table[findings].to_numpy()).all():
                raise ValueError('Invalid HYBRID training labels')
            self.labels=table.set_index('StudyInstanceUID')[findings]

    def __len__(self):return len(self.files)

    def __getitem__(self,index):
        file=self.files[index]
        with np.load(file,allow_pickle=False) as saved:
            images=torch.from_numpy(saved['images'])
            series=saved['triplet_series_uids']
            indices=saved['triplet_indices']
            owners=saved['slice_series_uids'][indices]
            if not np.all(owners==owners[:,:1]) or not np.all(owners[:,0]==series) or not np.all(saved['same_series']):
                raise ValueError('Cross-series triplet detected')
            if images.dtype!=torch.float16 or images.ndim!=4 or tuple(images.shape[1:])!=(3,384,384) or not len(images):
                raise ValueError('Invalid preprocessed images')
            slots=torch.from_numpy(saved['triplet_slot_ids'])
            present=torch.from_numpy(saved['slot_present'])
        if self.k is not None:
            chosen=torch.randperm(len(images))[:self.k]
            # A short bag stays short; never duplicate triplets to fill k.
            images,slots=images[chosen],slots[chosen]
        item=dict(images=images,triplet_slot_ids=slots,slot_present=present,study_uid=file.stem)
        if self.labels is not None:item['targets']=torch.tensor(self.labels.loc[file.stem].to_numpy(dtype=np.float32))
        return item
