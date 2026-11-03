"""Reproduce JEV-Knee's label AUC using per-finding available evaluation pairs.

This evaluates existing weak-label scores, not predictions from an MRI model.
Requires pandas, numpy and scikit-learn. Original inputs are never changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from validate_jev_knee import LABELS, ROOT, UID, read_csv, save_new_or_identical, sha256


def evaluate(truth: pd.DataFrame, scores: pd.DataFrame) -> dict:
    if truth[UID].isna().any() or truth[UID].duplicated().any():
        raise ValueError("Evaluation study identifiers must be present and unique")
    if scores[UID].isna().any() or scores[UID].duplicated().any():
        raise ValueError("Score study identifiers must be present and unique")
    truth = truth.set_index(UID)[LABELS].apply(pd.to_numeric, errors="raise")
    scores = scores.set_index(UID)[LABELS].apply(pd.to_numeric, errors="raise")
    if len(truth.index.difference(scores.index)):
        raise ValueError("Some evaluation studies are missing from the score table")
    scores = scores.reindex(truth.index)
    if (truth.notna() & ~truth.isin([0, 1])).any().any():
        raise ValueError("Evaluation labels must be binary or missing")
    invalid = scores.notna() & (~np.isfinite(scores) | (scores < 0) | (scores > 1))
    if invalid.any().any():
        raise ValueError("Scores must be probabilities or missing")
    rows = []
    for label in LABELS:
        valid = truth[label].notna() & scores[label].notna()
        if truth.loc[valid, label].nunique() != 2:
            raise ValueError(f"{label}: both classes are required to compute AUC")
        auc = float(roc_auc_score(truth.loc[valid, label], scores.loc[valid, label]))
        rows.append({"target": label, "auc": auc, "auc_rounded_4dp": round(auc, 4),
                     "n_scored": int(valid.sum()), "positive": int(truth.loc[valid, label].sum()),
                     "negative": int(valid.sum() - truth.loc[valid, label].sum()),
                     "missing_truth": int(truth[label].isna().sum()),
                     "missing_score": int(scores[label].isna().sum())})
    complete = truth.notna().all(axis=1) & scores.notna().all(axis=1)
    complete_auc = float(roc_auc_score(truth.loc[complete], scores.loc[complete], average="macro"))
    return {"evaluation_studies": len(truth), "per_target": rows,
            "macro_auc": float(np.mean([row["auc"] for row in rows])),
            "publisher_notebook_macro_after_rounding": float(np.mean([row["auc_rounded_4dp"] for row in rows])),
            "complete_case_studies": int(complete.sum()), "complete_case_macro_auc": complete_auc}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/external/jev_knee")
    parser.add_argument("--official-train", type=Path, default=ROOT / "audit/training/labels/train.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/jev_knee")
    args = parser.parse_args()
    original = args.data_dir / "original"
    truth = read_csv(original / "gold74_eval_set.csv")
    hybrid = read_csv(original / "HYBRID_labels_scores.csv")
    final2 = read_csv(original / "FINAL2_labels_scores.csv")
    result = evaluate(truth, hybrid)
    metrics = json.loads((original / "metrics.json").read_text(encoding="utf-8"))
    reference = metrics["HYBRID_0.95fused_0.05ensemble"]
    threshold_reference = json.loads((original / "HYBRID_thresholds.json").read_text(encoding="utf-8"))["macro_auc_gold74"]
    differences = {row["target"]: abs(row["auc_rounded_4dp"] - reference[row["target"]])
                   for row in result["per_target"]}
    report = {"method": "Join by study UID; for each finding exclude only missing truth/score pairs; average 12 ROC AUCs.",
              "scope": "Report-derived label quality; selection-exposed publisher evaluation, not MRI-model performance.",
              "input_hashes": {f.name: sha256(f) for f in [original / "gold74_eval_set.csv",
                  original / "HYBRID_labels_scores.csv", original / "FINAL2_labels_scores.csv",
                  original / "metrics.json", original / "HYBRID_thresholds.json"]},
              "hybrid_publisher_eval": result, "final2_publisher_eval": evaluate(truth, final2),
              "verification": {"max_per_target_rounded_difference": max(differences.values()),
                  "absolute_macro_difference_from_thresholds_json": abs(result["macro_auc"] - threshold_reference)}}
    assert max(differences.values()) < 1e-12, "Per-target metrics disagree with the publisher"
    assert abs(result["macro_auc"] - threshold_reference) < 1e-12, "Macro AUC disagrees with the publisher"
    if args.official_train.exists():
        official = read_csv(args.official_train)
        official = official.loc[official[LABELS].notna().all(axis=1)]
        report["official_train_sha256"] = sha256(args.official_train)
        report["hybrid_official_gold"] = evaluate(official, hybrid)
        report["final2_official_gold"] = evaluate(official, final2)
    notebook = ROOT / "audit/jev_knee_verification/jev-knee-labels-verification.ipynb"
    if notebook.exists():
        report["publisher_notebook"] = {"url": "https://www.kaggle.com/code/riadmohamed42/jev-knee-labels-verification",
                                         "sha256": sha256(notebook)}
    save_new_or_identical(args.output_dir / "auc_reproduction.json", (json.dumps(report, indent=2) + "\n").encode("utf-8"))
    save_new_or_identical(args.output_dir / "hybrid_per_finding_auc.csv", pd.DataFrame(result["per_target"]).to_csv(index=False).encode("utf-8"))
    print(pd.DataFrame(result["per_target"])[["target", "auc_rounded_4dp", "n_scored"]].to_string(index=False))
    print(f"\nHYBRID publisher evaluation macro AUC: {result['macro_auc']:.12f}")
    print(f"Publisher notebook (round findings first): {result['publisher_notebook_macro_after_rounding']:.12f}")
    print(f"Exact published per-finding and macro metrics: VERIFIED")
    if "hybrid_official_gold" in report:
        print(f"HYBRID official Gold58 macro AUC: {report['hybrid_official_gold']['macro_auc']:.12f}")
    print(f"Report: {(args.output_dir / 'auc_reproduction.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
