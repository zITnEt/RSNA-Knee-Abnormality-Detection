"""Build a three-study preview from already verified production tensors."""
import json
import shutil
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data/preprocessed_kaggle_staging/shard_00/six_slot_raptor_shard_00'
folder=ROOT/'data/six_slot_training_preview'
folder.mkdir(parents=True,exist_ok=True)
manifest=json.loads((source/'manifest.json').read_text())
rows=sorted(manifest['completed'],key=lambda r:r['study_uid'])[:3]
uids=[row['study_uid'] for row in rows]
for row in rows:
    shutil.copyfile(source/row['file'],folder/row['file'])
for name in ('labels_official.csv','labels_jev_final2.csv','labels_gold74.csv','splits.csv'):
    table=pd.read_csv(source/name,dtype={'StudyInstanceUID':str})
    table[table.StudyInstanceUID.isin(uids)].to_csv(folder/name,index=False)
for name in ('recipe.json','ready_tensors.py'):
    shutil.copyfile(source/name,folder/name)
(folder/'manifest.json').write_text(json.dumps({'preview':True,'corpus_complete':False,
    'complete':True,'expected_uids':uids,'completed':rows,'source_shard':0,
    'recipe_sha256':manifest['recipe_sha256']},indent=2),encoding='utf-8')
(folder/'README.md').write_text('''# Three-study training preview

This is a three-study preview of the six-slot Raptor V5 preprocessing export.
It is not the complete 4,407-study corpus. The full export is being processed
and published separately. These are the exact production float16 tensors.
Load the images array from any NPZ and pass it directly to your model under
torch.autocast; cropping, resizing, intensity scaling and normalization are done.
All files can be opened with numpy.load(..., allow_pickle=False).

The underlying images remain subject to RSNA competition terms. The Other
license field grants no new rights. This dataset is private.
''',encoding='utf-8')
(folder/'dataset-metadata.json').write_text(json.dumps({
    'id':'zafarnematov/rsna-six-slot-raptor-v5-preview',
    'title':'RSNA Six Slot Raptor V5 Preview','licenses':[{'name':'other'}],
    'description':'Private three-study preview of production model-ready MRI triplets. This is not the complete 4407-study corpus. The underlying RSNA images remain subject to original competition terms; no new rights are granted by the Other license field. Six-category header-aware selection; 140mm crop; 2–98% span and p2/p98 scaling; 336 uint8 cache to bilinear 384; ImageNet normalization; float16 [K,3,384,384].'},indent=2),encoding='utf-8')
print(json.dumps({'preview_studies':3,'tensor_bytes':sum(r['bytes'] for r in rows),'folder':str(folder)}))
