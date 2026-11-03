"""Build a Kaggle notebook that exports only V2-preprocessed competition test bags."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parents[1]
SLUG = "rsna-min-overlap-v2-test-bags"
SOURCES = {
    "preprocess_six_slot_raptor.py": MAIN / "scripts/preprocess_six_slot_raptor.py",
    "preprocess_series_safe_384.py": MAIN / "scripts/preprocess_series_safe_384.py",
    "preprocess_min_overlap_384.py": MAIN / "scripts/preprocess_min_overlap_384.py",
}


def cell(kind: str, source: str) -> dict:
    result = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


texts = {name: file.read_text(encoding="utf-8") for name, file in SOURCES.items()}
digests = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in texts.items()}
write_sources = (
    "from pathlib import Path\nimport sys\n"
    f"SOURCES = {texts!r}\n"
    "src = Path('/kaggle/working/preprocess_src')\n"
    "for name, text in SOURCES.items():\n"
    "    path = src/name\n"
    "    path.parent.mkdir(parents=True, exist_ok=True)\n"
    "    path.write_text(text, encoding='utf-8')\n"
    "sys.path.insert(0, str(src))\n"
    "print('sources written:', sorted(SOURCES), flush=True)\n"
)
run = '''import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess_min_overlap_384 import RECIPE_SHA, prepare

root = next((p for p in (Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
                         Path('/kaggle/input/rsna-knee-abnormality-detection'))
             if (p/'sample_submission.csv').is_file() and (p/'test_series.csv').is_file()), None)
if root is None:
    raise FileNotFoundError('Competition test data not attached')
series_root = next((root/p for p in ('test_series', 'test_images', 'test') if (root/p).is_dir()), None)
if series_root is None:
    raise FileNotFoundError('Test DICOM directory not found')
sample = pd.read_csv(root/'sample_submission.csv', dtype={'StudyInstanceUID': str})
series = pd.read_csv(root/'test_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
groups = {uid: g.to_dict('records') for uid, g in series.groupby('StudyInstanceUID', sort=False)}
output = Path('/kaggle/working/test_bags')
output.mkdir(exist_ok=True)
sample.to_csv(output/'sample_submission.csv', index=False)
manifest = {'recipe_sha256': RECIPE_SHA, 'studies': []}
for i, uid in enumerate(sample.StudyInstanceUID, 1):
    start = time.time()
    if uid not in groups:
        raise ValueError(f'Missing test series: {uid}')
    data, audit = prepare(uid, groups[uid], series_root)
    images = data['images']
    if images.dtype != np.float16 or images.ndim != 4 or images.shape[1:] != (3,384,384):
        raise ValueError(f'Bad prepared images: {uid}: {images.shape} {images.dtype}')
    if len(images) < 1 or len(images) > 60 or not np.all(data['same_series']):
        raise ValueError(f'Invalid test bag: {uid}')
    file = output/f'{uid}.npz'
    np.savez_compressed(file, images=images, recipe_sha256=RECIPE_SHA,
                        audit_json=json.dumps(audit, allow_nan=False))
    manifest['studies'].append({'study_uid': uid, 'file': file.name,
                                'bytes': file.stat().st_size, 'windows': len(images)})
    print(json.dumps({'done': i, 'total': len(sample), 'study_uid': uid,
                      'windows': len(images), 'seconds': round(time.time()-start,1)}), flush=True)
(output/'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps({'complete': True, 'studies': len(sample), 'recipe_sha256': RECIPE_SHA}), flush=True)
'''
notebook = dict(cells=[
    cell("markdown", "# RSNA test preprocessing\n\nExport official test studies with the exact minimum-overlap 384 V2 training recipe. No model weights are attached.\n\nSource SHA256: " + json.dumps(digests) + "\n"),
    cell("code", write_sources), cell("code", run)],
    metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                  language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
metadata = dict(id=f"zafarnematov/{SLUG}", title="RSNA Minimum Overlap V2 Test Bags",
                code_file=f"{SLUG}.ipynb", language="python", kernel_type="notebook",
                is_private=True, enable_gpu=False, enable_tpu=False, enable_internet=False,
                keywords=[], dataset_sources=[], kernel_sources=[],
                competition_sources=["rsna-knee-abnormality-detection"], model_sources=[])
out = ROOT / "kaggle" / SLUG
out.mkdir(parents=True, exist_ok=True)
(out / f"{SLUG}.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
(out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(out)
