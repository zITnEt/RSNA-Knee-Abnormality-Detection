"""Fetch only official competition test files for local VM inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


COMPETITION = "rsna-knee-abnormality-detection"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    api = KaggleApi()
    api.authenticate()
    token = None
    files = []
    pages = 0
    while True:
        try:
            response = api.competition_list_files(COMPETITION, page_token=token, page_size=200)
        except Exception as error:
            if "429" not in str(error):
                raise
            print(json.dumps({"rate_limited_after_pages": pages, "retry_seconds": 30}), flush=True)
            time.sleep(30)
            continue
        pages += 1
        files.extend(file for file in response.files if file.name.startswith("test_series/") or
                     file.name in ("sample_submission.csv", "test_series.csv", "test.csv"))
        token = response.next_page_token
        print(json.dumps({"listed_pages": pages, "test_files": len(files),
                          "last_file": response.files[-1].name if response.files else None}), flush=True)
        # File listing is alphabetical: all test_series entries precede train.csv.
        if not token or any(file.name.startswith("train") for file in response.files):
            break
    dicoms = [file for file in files if file.name.startswith("test_series/")]
    print(json.dumps({"files": len(files), "dicoms": len(dicoms),
                      "bytes": sum(file.total_bytes or 0 for file in files),
                      "studies": sorted({file.name.split("/")[1] for file in dicoms})}), flush=True)
    if args.root is None:
        return
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for i, file in enumerate(files, 1):
        destination = root / file.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size == file.total_bytes:
            continue
        api.competition_download_file(COMPETITION, file.name, path=str(destination.parent), quiet=True)
        if not destination.is_file() or destination.stat().st_size != file.total_bytes:
            raise RuntimeError(f"Download mismatch: {file.name}")
        if i % 25 == 0 or i == len(files):
            print(json.dumps({"downloaded": i, "total": len(files)}), flush=True)


if __name__ == "__main__":
    main()
