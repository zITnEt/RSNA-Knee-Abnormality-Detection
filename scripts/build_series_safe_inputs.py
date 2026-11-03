"""Create a notebook that reads normalized inputs; performs no preprocessing."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
loader=(ROOT/'raptor_data/series_safe_ready.py').read_text()
sources=[('markdown','# Direct training inputs: series-safe 384×384\n\nAttach the 32 private `rsna-series-safe-384-shard-*` datasets. '
    'The preview is a separate eight-study sample and must not be attached alongside production shards. '
    'Images are already normalized float16 tensors; no crop, resize, /255 or normalization is needed. '
    'Gold74 is excluded by the train split; HYBRID targets are provided separately. This notebook only checks and loads data.\n'),
    ('code',loader),('code',"""roots=sorted({p.parent for p in Path('/kaggle/input').rglob('recipe.json')
    if 'rsna-series-safe-384-shard-' in str(p)})
assert roots, 'Attach the new series-safe 384 dataset shards'
train=SeriesSafe384Dataset(roots,split='train',k=None,labels=True)
gold=SeriesSafe384Dataset(roots,split='held_out_gold',k=None,labels=False)
assert not {p.stem for p in train.files}&{p.stem for p in gold.files}
if len(roots)==32:
    assert len(train)==4333 and len(gold)==74
print({'attached_shards':len(roots),'training_studies':len(train),'gold_studies':len(gold)})
sample=train[0]
print(sample['study_uid'],sample['images'].shape,sample['images'].dtype)
print('These images are ready to pass to the model.')
import hashlib
rows=[]
for root in roots:
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['complete'] and not manifest['failures']
    rows.extend(manifest['completed'])
    for row in manifest['completed']:
        file=root/row['file']
        assert file.is_file() and file.stat().st_size==row['bytes']
        assert row['cross_series_windows']==0 and row['source_slices']<=72 and row['windows']<=60
    # Independently exercise one study from each attached shard without any
    # pixel transformation, and verify ownership and the published checksum.
    row=manifest['completed'][0];file=root/row['file']
    with file.open('rb') as stream: assert hashlib.file_digest(stream,'sha256').hexdigest()==row['sha256']
    with np.load(file,allow_pickle=False) as saved:
        owners=saved['slice_series_uids'][saved['triplet_indices']]
        assert np.all(owners==owners[:,:1]) and np.all(saved['same_series'])
        assert np.all(owners[:,0]==saved['triplet_series_uids'])
        assert saved['images'].dtype==np.float16 and saved['images'].shape[1:]==(3,384,384)
        assert np.isfinite(saved['images']).all()
assert len({r['study_uid'] for r in rows})==len(rows)
summary=dict(attached_shards=len(roots),studies=len(rows),training_studies=len(train),gold_studies=len(gold),
    source_slices=sum(r['source_slices'] for r in rows),triplets=sum(r['windows'] for r in rows),
    cross_series_triplets=0,npz_bytes=sum(r['bytes'] for r in rows),
    recipe_sha256=json.loads((roots[0]/'recipe.json').read_text())['sha256'])
Path('/kaggle/working/dataset_validation.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
"""),('markdown','`k=None` returns every triplet. Study bags have variable sizes; use batch size one or a ragged collate. '
    'Convert tensor dtype only if your model requires float32. Preserve whole triplets when batching.\n')]
cells=[]
for index,(kind,source) in enumerate(sources):
    cell=dict(cell_type=kind,id=f'inputs-{index}',metadata={},source=source.splitlines(keepends=True))
    if kind=='code':ast.parse(source);cell.update(execution_count=None,outputs=[])
    cells.append(cell)
notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata={'kernelspec':dict(name='python3',language='python',display_name='Python 3')})
(ROOT/'kaggle/series_safe_384_training_inputs.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
folder=ROOT/'kaggle/series_safe_384_training_inputs';folder.mkdir(exist_ok=True)
(folder/'training_inputs.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
metadata=dict(id='zafarnematov/rsna-series-safe-384-training-inputs',title='RSNA Series Safe 384 Training Inputs',
    code_file='training_inputs.ipynb',language='python',kernel_type='notebook',is_private=True,
    enable_gpu=False,enable_internet=False,dataset_sources=[f'zafarnematov/rsna-series-safe-384-shard-{i:02d}' for i in range(32)],
    competition_sources=[],kernel_sources=[],model_sources=[])
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print('Built ready-input loader notebook')
