"""Build an isolated private direct-384 preprocessing campaign."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=(ROOT/'scripts/preprocess_six_slot_raptor.py').read_text(encoding='utf-8').split('if __name__ == "__main__":')[0]
processor=(ROOT/'scripts/preprocess_series_safe_384.py').read_text(encoding='utf-8').split("if __name__=='__main__':")[0]
setup="""from pathlib import Path
import sys
helper = Path('/kaggle/working/preprocess_six_slot_raptor.py')
helper.write_text(BASE_HELPER, encoding='utf-8')
sys.path.insert(0,'/kaggle/working')
"""
for shard in ['pilot',*range(32)]:
    pilot=shard=='pilot';name='pilot' if pilot else f'{shard:02d}'
    folder=ROOT/'kaggle/series_safe_384_preprocessing'/f'shard_{name}';folder.mkdir(parents=True,exist_ok=True)
    cells=[]
    sources=[('markdown', '# Series-safe MRI slices and triplets, direct 384×384\n\n'
              'Six slots: SAG fluid FS 18 / SAG fluid non-FS 14 / COR fluid FS 12 / COR T1 8 / AX fluid FS 12 / SAG T1 8. '
              'Up to 72 unique sampled slices and 60 same-series triplets. Physical ordering, 2–98% depth range, '
              'per-series p2/p98, 140 mm crop, direct INTER_AREA resize, uint8 storage, /255 and ImageNet normalization. '
              'Gold74 is in held_out_gold/, excluded from train/ and HYBRID training labels. Load saved images directly.\n'),
             ('code','BASE_HELPER = '+repr(base)+'\n'+setup),('code',processor),
             ('code',f"manifest = run(shard={0 if pilot else shard}, shards=32, limit={8 if pilot else None}, workers=4)\nassert manifest['complete']\n")]
    for i,(kind,source) in enumerate(sources):
        c=dict(cell_type=kind,id=f'safe-{i}',metadata={},source=source.splitlines(keepends=True))
        if kind=='code':ast.parse(source);c.update(execution_count=None,outputs=[])
        cells.append(c)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':dict(name='python3',language='python',display_name='Python 3')},cells=cells)
    (folder/'preprocess.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    meta=dict(id=f'zafarnematov/rsna-series-safe-384-preprocessing-{name}',title=f'RSNA Series Safe 384 Preprocessing {name.upper()}',
              code_file='preprocess.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_internet=False,
              competition_sources=['rsna-knee-abnormality-detection'],dataset_sources=['riadmohamed42/jev-knee-labels'],kernel_sources=[],model_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print('Built pilot and 32 production notebooks')
