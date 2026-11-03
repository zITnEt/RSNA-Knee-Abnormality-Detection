# Working on the project

Choose one experiment directory and keep its imports, preprocessing recipe, and checkpoint format together. The root is the RadImageNet experiment; `experiments/resnet50_fusion` and `experiments/coatnet_only` are separate snapshots with different model code.

For a change, explain the behavior affected, run the relevant existing tests, and record validation. `python scripts/run_portfolio_checks.py` runs the lightweight CPU checks. Full architecture checks use `python -m unittest discover -s tests -v` from the selected variant directory and may require more memory and time.

Use synthetic fixtures for new regression tests. Keep credentials, reports, DICOM files, labels, model checkpoints, and per-study prediction exports out of commits. Store new run outputs outside the source tree or under an ignored run directory.

Report metrics with the exact dataset, label mask, checkpoint-selection rule, and uncertainty. Preserve attribution for adapted public code. Do not describe study-level folds as patient-level folds when patient identifiers are unavailable.
