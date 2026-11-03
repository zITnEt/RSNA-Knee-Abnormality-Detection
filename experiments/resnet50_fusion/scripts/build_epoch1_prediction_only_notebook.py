"""Package only the validated prediction CSV for a notebook-only Kaggle contest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
source = root / "results/epoch1_kaggle_submission/test.csv"
csv_text = source.read_text(encoding="utf-8")
sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
if sha256 != "0829e981e60c354ea0131abdacec52073880fdf040cd916cdff4aca7b5fb33b6":
    raise ValueError("Unexpected epoch-1 test.csv content")

slug = "rsna-raptor-resnet50-epoch1-predictions"
code = f'''import csv
import hashlib
import io
import math
from pathlib import Path

predictions = {csv_text!r}
expected_sha256 = {sha256!r}
assert hashlib.sha256(predictions.encode('utf-8')).hexdigest() == expected_sha256
root = next((p for p in (Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
                         Path('/kaggle/input/rsna-knee-abnormality-detection'))
             if (p/'sample_submission.csv').is_file()), None)
if root is None:
    raise FileNotFoundError('Official competition sample not attached')
with (root/'sample_submission.csv').open(newline='') as stream:
    sample = list(csv.reader(stream))
ours = list(csv.reader(io.StringIO(predictions)))
assert len(ours) == len(sample)
assert ours[0] == sample[0]
assert [row[0] for row in ours[1:]] == [row[0] for row in sample[1:]]
assert all(len(row) == len(ours[0]) and all(math.isfinite(float(x)) and 0 <= float(x) <= 1
    for x in row[1:]) for row in ours[1:])
for name in ('test.csv', 'submission.csv'):
    (Path('/kaggle/working')/name).write_text(predictions, encoding='utf-8')
print('Validated and wrote', len(ours)-1, 'epoch-1 raw test predictions; sha256', expected_sha256)
'''
notebook = {
    "cells": [
        {"cell_type": "markdown", "metadata": {},
         "source": ["# Epoch-1 raw Raptor–ResNet50 predictions\n",
                    "\n", "Validated predictions from the minimum-overlap 384 V2 run. Model weights and preprocessing source are not attached.\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": code.splitlines(keepends=True)},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
metadata = {
    "id": f"zafarnematov/{slug}",
    "title": "RSNA Raptor ResNet50 Epoch1 Predictions",
    "code_file": f"{slug}.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
    "keywords": [], "dataset_sources": [], "kernel_sources": [],
    "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
}
out = root / "kaggle" / slug
out.mkdir(parents=True, exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
(out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(out)
