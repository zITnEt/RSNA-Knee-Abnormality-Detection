"""Build a private metadata-only feasibility audit for 60 disjoint triplets."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'kaggle/series_safe_384_unique_slice_audit';folder.mkdir(exist_ok=True)
source=(ROOT/'scripts/audit_unique_slice_capacity.py').read_text()
runner="""roots=sorted({p.parent for p in Path('/kaggle/input').rglob('recipe.json')
    if 'rsna-series-safe-384-shard-' in str(p)})
assert len(roots)==32
summary=run(roots,'/kaggle/working')
assert summary['studies']==4407
"""
cells=[]
for i,(kind,code) in enumerate([('markdown','# Feasibility of 60 non-overlapping triplets\n\nRead existing audit metadata to count distinct eligible native slices in each selected series. No image preprocessing or training is performed.\n'),('code',source),('code',runner)]):
    cell=dict(cell_type=kind,id=f'audit-{i}',metadata={},source=code.splitlines(keepends=True))
    if kind=='code':ast.parse(code);cell.update(execution_count=None,outputs=[])
    cells.append(cell)
notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':dict(name='python3',language='python',display_name='Python 3')},cells=cells)
(folder/'audit.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
metadata=dict(id='zafarnematov/rsna-series-safe-384-unique-slice-audit',title='RSNA Series Safe 384 Unique Slice Audit',
    code_file='audit.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_internet=False,
    competition_sources=[],kernel_sources=[],model_sources=[],
    dataset_sources=[f'zafarnematov/rsna-series-safe-384-shard-{i:02d}' for i in range(32)])
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print('Built read-only native-slice capacity audit')
