"""Validate original JEV-Knee CSVs; optionally export scores excluding gold/eval IDs.

Requires pandas and numpy. No network calls, report text, or pipeline changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UID = "StudyInstanceUID"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={UID: "string"})
    missing = set([UID, *LABELS]) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    frame[UID] = frame[UID].str.strip()
    return frame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_new_or_identical(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite different output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/external/jev_knee")
    parser.add_argument("--official-train", type=Path, default=ROOT / "audit/training/labels/train.csv")
    parser.add_argument("--prepare", action="store_true", help="Export both soft-score sources; exclude all evaluation/gold IDs.")
    args = parser.parse_args()
    original = args.data_dir / "original"
    tables = {file.name: read_csv(file) for file in sorted(original.glob("*.csv"))}
    if not tables or "gold74_eval_set.csv" not in tables:
        raise ValueError("Original label CSVs and evaluation set are required")
    evaluation = tables["gold74_eval_set.csv"]
    eval_ids = set(evaluation[UID].dropna())
    if evaluation[UID].isna().any() or evaluation[UID].duplicated().any():
        raise ValueError("Evaluation identifiers must be present and unique")
    official = read_csv(args.official_train) if args.official_train.exists() else None
    official_ids = set(official[UID].dropna()) if official is not None else None
    gold_ids = set(official.loc[official[LABELS].notna().all(axis=1), UID]) if official is not None else set()
    inventory = [{"file": f.name, "bytes": f.stat().st_size, "sha256": sha256(f)}
                 for f in sorted(original.iterdir()) if f.is_file()]
    archive = original / "jev-knee-labels.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as zipped:
            for entry in zipped.infolist():
                if entry.is_dir():
                    continue
                target = (original / entry.filename).resolve()
                if not target.is_relative_to(original.resolve()):
                    raise ValueError("Unsafe archive path")
                if hashlib.sha256(zipped.read(entry)).hexdigest() != sha256(target):
                    raise ValueError(f"Original differs from download: {entry.filename}")
    report = {"dataset": "riadmohamed42/jev-knee-labels",
              "url": "https://www.kaggle.com/datasets/riadmohamed42/jev-knee-labels",
              "original_directory": str(original.resolve()), "label_columns": LABELS,
              "inventory": inventory, "archive_originals_match": archive.exists(),
              "official_metadata": str(args.official_train.resolve()) if official is not None else None,
              "official_studies": len(official_ids) if official_ids is not None else None,
              "official_gold_studies": len(gold_ids), "evaluation_studies": len(eval_ids),
              "evaluation_official_gold_overlap": len(eval_ids & gold_ids), "tables": {}, "exports": {}}
    metadata_path = args.data_dir / "metadata/dataset-view.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        report["version"] = metadata.get("currentVersionNumber")
        report["publisher"] = metadata.get("ownerName")
        report["license"] = metadata.get("licenseName")
    for name, frame in tables.items():
        numeric = frame[LABELS].apply(pd.to_numeric, errors="coerce")
        missing = numeric.isna()
        invalid = numeric.notna() & (~np.isfinite(numeric) | (numeric < 0) | (numeric > 1))
        nonnumeric = frame[LABELS].notna() & missing
        valid_values = numeric.to_numpy().ravel()
        valid_values = valid_values[np.isfinite(valid_values)]
        ids = set(frame[UID].dropna())
        report["tables"][name] = {
            "rows": len(frame), "columns_count": len(frame.columns), "columns": list(frame.columns),
            "unique_studies": len(ids), "missing_ids": int((frame[UID].isna() | frame[UID].eq("")).sum()),
            "duplicate_identifier_rows": int(frame[UID].duplicated().sum()),
            "missing_by_column": {c: int(v) for c, v in frame.isna().sum().items()},
            "missing_label_cells": int(missing.sum().sum()), "nonnumeric_label_cells": int(nonnumeric.sum().sum()),
            "invalid_probability_cells": int(invalid.sum().sum()),
            "missing_label_studies": frame.loc[missing.any(axis=1), UID].tolist(),
            "missing_label_studies_are_official_gold": set(frame.loc[missing.any(axis=1), UID]) <= gold_ids,
            "label_type": "binary" if np.isin(valid_values, [0, 1]).all() else "soft",
            "ranges": {c: {"min": float(numeric[c].min()), "max": float(numeric[c].max())} for c in LABELS},
            "official_overlap": len(ids & official_ids) if official_ids is not None else None,
            "not_in_official_metadata": sorted(ids - official_ids) if official_ids is not None else None}
    if official is not None:
        aligned = evaluation.set_index(UID).loc[sorted(eval_ids & gold_ids), LABELS]
        truth = official.set_index(UID).loc[aligned.index, LABELS]
        report["evaluation_vs_official_gold"] = {
            "compared_studies": len(aligned), "missing_label_cells": int(aligned.isna().sum().sum()),
            "disagreements_by_label": (aligned.notna() & truth.notna() & aligned.ne(truth)).sum().astype(int).to_dict()}
    report["caveats"] = [
        "Label-source selection, ensemble weights and blend selection used gold/evaluation labels; reported AUC is selection-biased.",
        "All 74 evaluation IDs and official fully labelled IDs are excluded from training exports.",
        "The 16 added evaluation studies are report-verified, not established independent image annotations.",
        "FINAL2_labels_with_confidence.csv contains binary label columns, not the FINAL2 soft scores.",
        "No missing labels are filled or gold labels copied into training targets.",
        "CC0 is the publisher's declared license; underlying competition/source-label terms remain applicable."]
    if args.prepare:
        if official is None:
            raise ValueError("Official metadata is required for leakage-conscious training exports")
        excluded = eval_ids | gold_ids
        for source, target in [("HYBRID_labels_scores.csv", "hybrid_soft_train.csv"),
                               ("FINAL2_labels_scores.csv", "final2_soft_train.csv")]:
            frame = tables[source]
            stats = report["tables"][source]
            if stats["duplicate_identifier_rows"] or stats["missing_ids"] or stats["invalid_probability_cells"] or stats["nonnumeric_label_cells"]:
                raise ValueError(f"{source}: malformed identifiers or probabilities; cannot export")
            if set(frame[UID]) - official_ids:
                raise ValueError(f"{source}: contains unknown competition IDs")
            cleaned = frame.loc[~frame[UID].isin(excluded), [UID, *LABELS]].copy()
            cleaned[LABELS] = cleaned[LABELS].apply(pd.to_numeric, errors="raise")
            if cleaned[LABELS].isna().any().any():
                raise ValueError(f"{source}: retained training rows have missing labels; use a masked loss or explicit policy")
            assert not (set(cleaned[UID]) & excluded)
            destination = args.data_dir / "prepared" / target
            save_new_or_identical(destination, cleaned.to_csv(index=False).encode("utf-8"))
            report["exports"][target] = {"path": str(destination.resolve()), "source": source,
                "rows": len(cleaned), "columns": len(cleaned.columns), "excluded_studies": len(frame) - len(cleaned),
                "missing_label_cells": 0, "sha256": sha256(destination)}
    destination = args.data_dir / ("validation_report.json" if args.prepare else "validation_only_report.json")
    save_new_or_identical(destination, (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps({"report": str(destination.resolve()), "table_dimensions": {
        name: [s["rows"], s["columns_count"]] for name, s in report["tables"].items()}, "exports": report["exports"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
