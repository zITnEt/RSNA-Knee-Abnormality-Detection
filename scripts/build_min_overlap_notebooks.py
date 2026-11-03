"""Build a separate private Kaggle-only minimum-overlap campaign."""
import ast
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=(ROOT/'scripts/preprocess_six_slot_raptor.py').read_text().split('if __name__ == "__main__":')[0]
parent=(ROOT/'scripts/preprocess_series_safe_384.py').read_text().split("if __name__=='__main__':")[0]
processor=(ROOT/'scripts/preprocess_min_overlap_384.py').read_text()
setup="""import sys
from pathlib import Path
Path('/kaggle/working/preprocess_six_slot_raptor.py').write_text(BASE_HELPER)
Path('/kaggle/working/preprocess_series_safe_384.py').write_text(PARENT_HELPER)
sys.path.insert(0,'/kaggle/working')
"""
for shard in ['pilot',*range(32)]:
    pilot=shard=='pilot';name='pilot' if pilot else f'{shard:02d}'
    folder=ROOT/'kaggle/min_overlap_384_preprocessing'/f'shard_{name}';folder.mkdir(parents=True,exist_ok=True)
    sources=[('markdown','# Minimum-overlap MRI triplets, 384×384\n\nSame six-slot selection and pixel/label preprocessing as the previous series-safe dataset. Up to 60 unique same-series triplets, up to 180 distinct source slices. Exact allocation minimizes repeated slice uses. Retain original 18/14/12/8/12/8 percentile references. Gold74 held out. No training is performed.\n'),
        ('code','BASE_HELPER = '+repr(base)+'\nPARENT_HELPER = '+repr(parent)+'\n'+setup),('code',processor),
        ('code',f"manifest=run(shard={0 if pilot else shard},shards=32,limit={24 if pilot else None},workers=4)\nassert manifest['complete']\n")]
    cells=[]
    for i,(kind,source) in enumerate(sources):
        cell=dict(cell_type=kind,id=f'min-{i}',metadata={},source=source.splitlines(keepends=True))
        if kind=='code':ast.parse(source);cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':dict(name='python3',language='python',display_name='Python 3')},cells=cells)
    (folder/'preprocess.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    metadata=dict(id=f'zafarnematov/rsna-min-overlap-384-preprocessing-{name}',title=f'RSNA Min Overlap 384 Preprocessing {name.upper()}',
        code_file='preprocess.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_internet=False,
        competition_sources=['rsna-knee-abnormality-detection'],dataset_sources=['riadmohamed42/jev-knee-labels'],kernel_sources=[],model_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print('Built minimum-overlap pilot and 32 production notebooks')
monitor=(ROOT/'scripts/monitor_series_safe_preprocessing.py').read_text()
monitor=monitor.replace('series_safe_384','min_overlap_384').replace('rsna-series-safe-384','rsna-min-overlap-384')
monitor=monitor.replace('from kaggle_monitor_runtime import awake_on_ac','from kaggle_monitor_runtime import awake_on_ac\nfrom task_state_io import atomic_json_write')
monitor=monitor.replace("temporary=STATE.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(STATE)",
    'atomic_json_write(STATE,state)')
(ROOT/'scripts/monitor_min_overlap_preprocessing.py').write_text(monitor)
loader=(ROOT/'raptor_data/series_safe_ready.py').read_text()
loader=loader.replace('SeriesSafe384Dataset','MinimumOverlap384Dataset').replace('rsna_series_safe_384_v1','rsna_min_overlap_384_v1')
loader=loader.replace('new series-safe direct-384','new minimum-overlap direct-384')
loader=loader.replace('if len(chosen)<self.k:chosen=torch.cat([chosen,torch.randint(len(images),(self.k-len(chosen),))])',
    '# A short bag stays short; never duplicate triplets to fill k.')
(ROOT/'raptor_data/min_overlap_ready.py').write_text(loader)
