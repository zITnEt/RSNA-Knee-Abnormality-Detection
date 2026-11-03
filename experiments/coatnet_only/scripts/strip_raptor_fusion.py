"""Export CoAtNet + the original MIL head from a local raw or EMA checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.raptor_v5 import load_raptor_checkpoint, raptor_checkpoint


def export(source: Path, output: Path, weights: str = "raw") -> dict:
    source, output = source.resolve(), output.resolve()
    if source == output or output.exists():
        raise FileExistsError(f"Choose a new output path: {output}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    source_sha = digest.hexdigest()
    model = load_raptor_checkpoint(source, weights=weights)
    saved = raptor_checkpoint(model)
    saved["conversion"] = {"source_sha256": source_sha, "source_weights": weights,
                           "optimizer_reset_required": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(saved, stream)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"output": str(output), "parameters": sum(p.numel() for p in model.parameters()),
            "source_sha256": source_sha, "source_weights": weights}


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw",
                        help="Use ema for embedded ema.shadow; standalone EMA files use raw")
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output, args.weights), indent=2))


if __name__ == "__main__":
    main()
