"""Build lossless Kaggle publication bridges; original source versions remain."""
import ast
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
# Shards published byte-identical to the level-1 preprocessing archives. All others are
# losslessly recompressed at level 6 (~2% smaller); 20 and 22-31 were switched to
# recompression so all 32 V2 versions fit the 200 GiB private-dataset quota.
COPY_ONLY_SHARDS={3,16,17,18,19,21}
for shard in range(32):
    name=f'{shard:02d}';folder=ROOT/'kaggle/min_overlap_384_publication'/f'shard_{name}';folder.mkdir(parents=True,exist_ok=True)
    source=f"""import hashlib,io,json,shutil,zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
EXPECTED='aad96c9225fa13edd3d2dff0a105f2e7de6b3efc91a0ad45a8229bd15acfd35c'
COPY_ONLY={shard in COPY_ONLY_SHARDS}
roots=[p.parent for p in Path('/kaggle/input').rglob('recipe.json') if json.loads(p.read_text()).get('sha256')==EXPECTED]
assert len(roots)==1
root=roots[0];manifest=json.loads((root/'manifest.json').read_text())
assert manifest['complete'] and not manifest['failures'] and manifest['shard']=={shard}
output=Path('/kaggle/working/min_overlap_384_shard_{name}');output.mkdir(exist_ok=True)
for name in ('recipe.json','splits.csv','labels_official.csv','labels_gold74.csv','labels_hybrid_train.csv'):
    shutil.copyfile(root/name,output/name)
def pack(row):
    source=root/row['file'];target=output/row['file'];target.parent.mkdir(parents=True,exist_ok=True)
    source_bytes=source.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest()==row['sha256']
    if COPY_ONLY:
        payload=source_bytes
    else:
        with zipfile.ZipFile(io.BytesIO(source_bytes)) as src:
            arrays={{name:src.read(name) for name in src.namelist()}}
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as dst:
            for name,array_bytes in arrays.items():dst.writestr(name,array_bytes)
        payload=buffer.getvalue()
        with zipfile.ZipFile(io.BytesIO(payload)) as dst:
            assert list(arrays)==dst.namelist()
            for name,array_bytes in arrays.items():assert array_bytes==dst.read(name)
    target.write_bytes(payload)
    with target.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(row,bytes=target.stat().st_size,sha256=digest)
rows=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    for row in pool.map(pack,manifest['completed']):
        rows.append(row)
        if len(rows)%25==0:print('Losslessly packed',len(rows),'studies',flush=True)
manifest['completed']=rows
manifest['storage_compression']='original ZIP_DEFLATED level 1; archive byte-identical to preprocessing output' if COPY_ONLY else 'ZIP_DEFLATED level 6; decompressed NPY bytes identical to preprocessing output'
(output/'manifest.json').write_text(json.dumps(manifest,indent=2))
(output/'index.jsonl').write_text(''.join(json.dumps(row)+'\\n' for row in rows))
report=dict(complete=True,shard={shard},studies=len(rows),bytes=sum(r['bytes'] for r in rows),
    source_bytes=sum(r['bytes'] for r in json.loads((root/'manifest.json').read_text())['completed']),
    recipe_sha256=EXPECTED,all_decompressed_array_bytes_identical=True,
    triplets=sum(r['windows'] for r in rows),source_slices=sum(r['source_slices'] for r in rows))
(output/'publication_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
"""
    ast.parse(source)
    cells=[dict(cell_type='markdown',id='purpose',metadata={},source=['# Minimum-overlap dataset publication\n\nLosslessly repack the already preprocessed Kaggle output for dataset V2. All pixel arrays remain identical. The earlier preprocessing run is preserved as notebook version 1. No training or DICOM preprocessing runs in this publication bridge.\n']),
        dict(cell_type='code',id='pack',metadata={},source=source.splitlines(keepends=True),execution_count=None,outputs=[])]
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':dict(name='python3',language='python',display_name='Python 3')},cells=cells)
    (folder/'publish.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    metadata=dict(id=f'zafarnematov/rsna-series-safe-384-preprocessing-{name}',title=f'RSNA Series Safe 384 Preprocessing {name}',
        code_file='publish.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_internet=False,
        competition_sources=[],dataset_sources=[],kernel_sources=[f'zafarnematov/rsna-min-overlap-384-preprocessing-{name}'],model_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print('Built 32 lossless publication bridges')
