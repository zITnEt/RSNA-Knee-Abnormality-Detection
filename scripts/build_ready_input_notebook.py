"""Create a companion notebook for directly loading exported training tensors."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
loader=(ROOT/'raptor_data/ready_tensors.py').read_text(encoding='utf-8')
setup='''from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

roots = []
for home in [Path('/kaggle/input'), Path('/kaggle/input/datasets/zafarnematov')]:
    roots.extend(home.glob('rsna-six-slot-raptor-v5-shard-*'))
for home in [Path('/kaggle/input'), Path('/kaggle/input/notebooks/zafarnematov')]:
    for notebook in home.glob('rsna-six-slot-raptor-v5-preprocessing-*'):
        roots.extend(notebook.glob('six_slot_raptor_shard_*'))
roots = sorted(set(p for p in roots if (p/'recipe.json').is_file()))
assert roots, 'Attach the completed preprocessing outputs or private tensor shard datasets'
recipes = [json.loads((p/'recipe.json').read_text()) for p in roots]
assert len({r['sha256'] for r in recipes}) == 1, 'Do not mix different preprocessing recipes'
assert all(r['source_budget'] == 64 and 'dicom_weight_type' in r.get('selector', '') for r in recipes)
assert all(json.loads((p/'manifest.json').read_text())['complete'] for p in roots)
print('Attached complete shards:', len(roots), '/ 32')
'''
usage='''LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA',
          'Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
tables = [pd.read_csv(p/'labels_jev_final2.csv', dtype={'StudyInstanceUID':str}) for p in roots]
targets = pd.concat(tables, ignore_index=True)
assert not targets.StudyInstanceUID.duplicated().any()
assert np.isfinite(targets[LABELS].to_numpy()).all()
labels = dict(zip(targets.StudyInstanceUID, targets[LABELS].to_numpy()))
train = ReadyTripletDataset(roots, split='train', k=12, labels=labels)
loader = DataLoader(train, batch_size=2, shuffle=True, num_workers=2,
                    pin_memory=torch.cuda.is_available())
batch = next(iter(loader))
print('Training studies:', len(train))
print('Already normalized model inputs:', batch['images'].shape, batch['images'].dtype)
print('Soft targets:', batch['targets'].shape)
assert batch['images'].shape[1:] == (12,3,384,384)
assert torch.isfinite(batch['images']).all()
# The image tensor can now be passed directly to your Raptor-style model:
# with torch.autocast('cuda', dtype=torch.float16):
#     logits = model(batch['images'].cuda())
#     loss = torch.nn.functional.binary_cross_entropy_with_logits(
#         logits.float(), batch['targets'].cuda().float())
'''
cells=[]
for kind,source in [
    ('markdown', '# Six-slot MRI training inputs\n\nAttach completed tensor shards or notebook outputs. Images already contain the Raptor V5 crop, interpolation, intensity scaling and ImageNet normalization. The six-category extension and float16 storage are explicit in recipe.json.\n\nThis notebook demonstrates loading inputs, without training or changing your model. Gold74 remains held out. JEV FINAL2 supplies the optional soft training targets.\n'),
    ('code',loader),('code',setup),('code',usage),
    ('markdown','Use `batch["images"]` directly in your model. Avoid adding a second resize, `/255`, or normalization transform. `k=12` samples the exported bag; `k=None` loads its full set of windows. The six-element `slot_present` and per-channel `triplet_slot_ids` accompany the images.\n\nAll 32 shards are needed for the complete 4,407-study image corpus; a subset is useful for testing the training loop. Training uses the 4,333 non-gold studies when every shard is attached.\n')]:
    cell={'cell_type':kind,'metadata':{},'id':f'ready-{len(cells)}','source':source.splitlines(keepends=True)}
    if kind=='code':
        cell.update(execution_count=None,outputs=[])
    cells.append(cell)
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
out=ROOT/'kaggle/six_slot_training_inputs.ipynb'
out.write_text(json.dumps(nb,indent=1),encoding='utf-8')
print(str(out))
