"""Build a direct-input notebook for verified minimum-overlap dataset V2."""
import ast,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
loader=(ROOT/'raptor_data/min_overlap_ready.py').read_text()
check="""roots=[]
for recipe_file in Path('/kaggle/input').rglob('recipe.json'):
    if json.loads(recipe_file.read_text()).get('format')=='rsna_min_overlap_384_v1':roots.append(recipe_file.parent)
roots=sorted(set(roots));assert len(roots)==32, 'Attach all 32 minimum-overlap dataset V2 shards'
train=MinimumOverlap384Dataset(roots,split='train',k=None)
gold=MinimumOverlap384Dataset(roots,split='held_out_gold',k=None,labels=False)
assert len(train)==4333 and len(gold)==74
assert not {p.stem for p in train.files}&{p.stem for p in gold.files}
sample=train[0]
print(sample['images'].shape,sample['images'].dtype)
rows=[];audits=[]
import hashlib
for root in roots:
    manifest=json.loads((root/'manifest.json').read_text());assert manifest['complete'] and not manifest['failures']
    rows.extend(manifest['completed'])
    for row in manifest['completed']:
        file=root/row['file'];assert file.is_file() and file.stat().st_size==row['bytes']
        with np.load(file,allow_pickle=False) as saved:
            audit=json.loads(str(saved['audit_json']));indices=saved['triplet_indices']
            owners=saved['slice_series_uids'][indices]
            assert np.all(owners==owners[:,:1]) and np.all(saved['same_series'])
            assert len({tuple(t) for t in indices})==len(indices)
            usage=saved['slice_usage_counts'];assert (usage>0).all() and usage.sum()==3*len(indices)
            assert len(usage)==audit['optimal_unique_slices']
            assert 3*len(indices)-len(usage)==audit['minimum_repeated_slice_uses']
            primary=saved['primary_disjoint_subset'];members=indices[primary].ravel()
            assert len(np.unique(members))==len(members) and primary.sum()==audit['optimal_disjoint_subset']
            assert 0<len(indices)<=60 and audit['recipe_sha256']==manifest['recipe_sha256']
            audits.append(audit)
    file=root/manifest['completed'][0]['file']
    with file.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==manifest['completed'][0]['sha256']
    with np.load(file,allow_pickle=False) as saved:
        assert saved['images'].dtype==np.float16 and saved['images'].shape[1:]==(3,384,384)
        assert np.isfinite(saved['images']).all()
assert len(rows)==4407 and len({r['study_uid'] for r in rows})==4407
summary=dict(attached_shards=32,studies=4407,training_studies=len(train),gold_studies=len(gold),
    triplets=sum(r['windows'] for r in rows),unique_slices=sum(r['source_slices'] for r in rows),
    repeated_slice_uses=sum(a['repeated_slice_uses'] for a in audits),
    zero_overlap_studies=sum(a['repeated_slice_uses']==0 for a in audits),
    short_bags=sum(r['windows']<60 for r in rows),triplets_min=min(r['windows'] for r in rows),
    triplets_max=max(r['windows'] for r in rows),cross_series_triplets=0,
    npz_bytes=sum(r['bytes'] for r in rows),recipe_sha256=audits[0]['recipe_sha256'])
assert summary['triplets']==264323 and summary['unique_slices']==609494
assert summary['zero_overlap_studies']==670 and summary['short_bags']==15
Path('/kaggle/working/dataset_validation.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
"""
cells=[]
for i,(kind,source) in enumerate([('markdown','# Minimum-overlap MRI training inputs\n\nDataset V2 retains the same pixel recipe and labels, with up to 60 triplets and globally minimum repeated slice uses. Original dataset V1 is preserved. Images are ready float16 tensors. Gold74 is held out. This notebook validates and loads data only; it does not train or preprocess pixels.\n'),('code',loader),('code',check)]):
    cell=dict(cell_type=kind,id=f'min-inputs-{i}',metadata={},source=source.splitlines(keepends=True))
    if kind=='code':ast.parse(source);cell.update(execution_count=None,outputs=[])
    cells.append(cell)
notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':dict(name='python3',language='python',display_name='Python 3')},cells=cells)
folder=ROOT/'kaggle/min_overlap_384_training_inputs';folder.mkdir(exist_ok=True)
(folder/'training_inputs.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
metadata=dict(id='zafarnematov/rsna-min-overlap-384-training-inputs',title='RSNA Min Overlap 384 Training Inputs',code_file='training_inputs.ipynb',
    language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_internet=False,
    competition_sources=[],kernel_sources=[],model_sources=[],
    dataset_sources=[f'zafarnematov/rsna-series-safe-384-shard-{i:02d}' for i in range(32)])
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print('Built minimum-overlap ready-input notebook')
