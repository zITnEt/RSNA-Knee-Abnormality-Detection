"""Build a Kaggle-only probe for browser-free dataset publication auth."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
folder = root / 'kaggle' / 'min_overlap_384_upload_probe'
folder.mkdir(parents=True, exist_ok=True)
source = '''import json
from pathlib import Path
import kagglehub

roots = [p.parent for p in Path('/kaggle/input').rglob('recipe.json')
         if json.loads(p.read_text()).get('sha256') ==
         'aad96c9225fa13edd3d2dff0a105f2e7de6b3efc91a0ad45a8229bd15acfd35c']
assert len(roots) == 1
result = {'source_ready': True, 'kagglehub_version': getattr(kagglehub, '__version__', 'unknown')}
try:
    identity = kagglehub.whoami(verbose=False)
    result['upload_auth_ready'] = identity.get('username') == 'zafarnematov'
except Exception as exc:
    result['upload_auth_ready'] = False
    result['auth_error_type'] = type(exc).__name__
Path('/kaggle/working/upload_probe.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
'''
notebook = {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'name': 'python3', 'language': 'python', 'display_name': 'Python 3'}},
            'cells': [{'cell_type': 'code', 'id': 'auth-probe', 'metadata': {},
                       'source': source.splitlines(keepends=True),
                       'execution_count': None, 'outputs': []}]}
(folder / 'upload_probe.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
metadata = {'id': 'zafarnematov/rsna-min-overlap-384-upload-probe',
            'title': 'RSNA Min Overlap 384 Upload Probe', 'code_file': 'upload_probe.ipynb',
            'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
            'enable_gpu': False, 'enable_internet': False, 'competition_sources': [],
            'dataset_sources': [], 'model_sources': [],
            'kernel_sources': ['zafarnematov/rsna-series-safe-384-preprocessing-22']}
(folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
print(folder)
