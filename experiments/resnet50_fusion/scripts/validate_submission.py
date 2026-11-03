#!/usr/bin/env python3
"""Strict RSNA Knee submission validation against the official sample file."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ID = "StudyInstanceUID"


def validate(submission_path: Path, sample_path: Path) -> str:
    submission = pd.read_csv(submission_path, dtype={ID: str})
    sample = pd.read_csv(sample_path, dtype={ID: str})
    errors: list[str] = []

    if submission.columns.tolist() != sample.columns.tolist():
        errors.append("column names/order differ from sample_submission.csv")
    if len(submission) != len(sample):
        errors.append(f"row count differs: {len(submission)} != {len(sample)}")
    if ID not in submission:
        errors.append(f"missing {ID}")
    else:
        ids = submission[ID]
        if ids.isna().any():
            errors.append("missing study IDs")
        if ids.duplicated().any():
            errors.append("duplicate study IDs")
        if ids.tolist() != sample[ID].tolist():
            errors.append("study IDs/order differ from sample_submission.csv")

    targets = sample.columns.drop(ID).tolist()
    if not errors or all(t in submission for t in targets):
        try:
            values = submission[targets].apply(pd.to_numeric, errors="raise").to_numpy(float)
            if not np.isfinite(values).all():
                errors.append("predictions contain NaN or infinity")
            if ((values < 0) | (values > 1)).any():
                errors.append("predictions fall outside [0, 1]")
        except Exception as exc:  # conversion detail belongs in report
            errors.append(f"non-numeric prediction: {exc}")
            values = None
    else:
        values = None

    sha = hashlib.sha256(submission_path.read_bytes()).hexdigest()
    lines = [
        f"status: {'FAIL' if errors else 'PASS'}",
        f"submission: {submission_path}",
        f"sample: {sample_path}",
        f"sha256: {sha}",
        f"rows: {len(submission)}",
    ]
    if errors:
        lines.extend(f"error: {error}" for error in errors)
    elif values is not None:
        lines.append("target,min,max,mean,std")
        for index, target in enumerate(targets):
            column = values[:, index]
            lines.append(
                f"{target},{column.min():.12g},{column.max():.12g},"
                f"{column.mean():.12g},{column.std(ddof=1):.12g}"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("sample", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate(args.submission, args.sample)
    print(report, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    return 0 if report.startswith("status: PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
