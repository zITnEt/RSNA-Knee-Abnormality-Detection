#!/usr/bin/env python3
"""Safe local/Kaggle environment report. Never reads or prints credential contents."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    config = Path.home() / ".kaggle" / "kaggle.json"
    auth = config.is_file() or bool(os.environ.get("KAGGLE_API_TOKEN")) or bool(
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )
    report: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "kaggle_cli": shutil.which("kaggle") is not None,
        "kaggle_auth_configured": auth,
        "competition_data_local": False,
    }
    if shutil.which("nvidia-smi"):
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        report["local_gpus"] = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
    print(json.dumps(report, indent=2))
    return 0 if auth and report["kaggle_cli"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
