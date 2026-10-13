"""Embed a reviewed processor in self-contained private Kaggle notebooks."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT/"scripts/preprocess_six_slot_raptor.py").read_text(encoding="utf-8")
SOURCE = SOURCE.split('if __name__ == "__main__":')[0]
OUT = ROOT/"kaggle/six_slot_preprocessing"
for shard in range(32):
    folder = OUT/f"shard_{shard:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Six-slot MRI training tensors — Raptor V5 preprocessing\n",
            "140 mm crop; 2–98% depth span; per-series 2nd/98th intensity percentiles; ",
            "336 uint8 cache → bilinear 384 → ImageNet normalization → float16 tensors.\n",
            "Six categories extend the five-slot parent. 64 source slices produce up to 62 adjacent triplets; cross-series windows are explicitly marked.\n",
            "Load `images` from each NPZ directly. No further image preprocessing is needed.\n",
            "Gold74 is marked held out. Annotation CSVs remain separate; no labels are invented.\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": SOURCE.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [f"manifest = run(shard={shard}, shards=32, workers=4)\n", "assert manifest['complete']\n"]}
    ]}
    for i, cell in enumerate(notebook['cells']):
        cell['id'] = f'preprocess-{i}'
    (folder/"preprocess.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    meta = {"id": f"zafarnematov/rsna-six-slot-raptor-v5-preprocessing-{shard:02d}",
            "title": f"RSNA Six Slot Raptor V5 Preprocessing {shard:02d}",
            "code_file": "preprocess.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": False, "enable_internet": False,
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "dataset_sources": ["riadmohamed42/jev-knee-labels"], "kernel_sources": [], "model_sources": []}
    (folder/"kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps({"notebooks": 32, "directory": str(OUT), "processor_sha256": hashlib.sha256(SOURCE.encode()).hexdigest()}))
