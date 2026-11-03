"""Export the raw epoch-1 model without optimizer/EMA training state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


root = Path(__file__).resolve().parents[1]
source = root / "run/checkpoints/epoch_01_raw.pt"
output_dir = root / "epoch1_submission_weights"
output_dir.mkdir(exist_ok=True)
target = output_dir / "epoch_01_raw_model.pt"

checkpoint = torch.load(source, map_location="cpu", weights_only=True)
assert checkpoint["epoch_index"] == 1
assert checkpoint["next_study"] == 0
assert len(checkpoint["history"]) == 1
assert checkpoint["format"] == "raptor_v5_resnet50_v3"

model_only = {key: checkpoint[key] for key in
              ("format", "arch", "res", "labels", "fusion_config", "model")}
model_only["provenance"] = {
    "source": str(source),
    "epoch": 1,
    "weight_kind": "raw",
    "global_step": checkpoint["global_step"],
    "recipe_sha256": checkpoint["training_config"]["recipe_sha256"],
}
torch.save(model_only, target)

digest = hashlib.sha256()
with target.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
manifest = {"file": target.name, "bytes": target.stat().st_size,
            "sha256": digest.hexdigest(), "provenance": model_only["provenance"]}
(output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(output_dir / "dataset-metadata.json").write_text(json.dumps({
    "id": "zafarnematov/rsna-raptor-resnet50-epoch1-raw",
    "title": "RSNA Raptor ResNet50 Epoch 1 Raw",
    "subtitle": "Raw epoch-1 model weights for the minimum-overlap 384 V2 run",
    "description": "Raw weights only, exported from the epoch-1 resumable checkpoint.",
    "licenses": [{"name": "unknown"}],
    "resources": [{"path": target.name, "description": "Raptor v5 ResNet50 fusion raw epoch-1 state"},
                  {"path": "manifest.json", "description": "Integrity and provenance"}],
}, indent=2) + "\n")
print(json.dumps(manifest))
