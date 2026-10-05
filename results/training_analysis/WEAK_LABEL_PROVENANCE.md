# Weak-label provenance of the saved ensemble

Audited 2026-10-05. Scope: the saved BASELINE-001 submission, DINOsaur V5 historical V10 (scriptVersionId 351856876), exact 0.943 Speedy Raptors anchor. The optional ConvNeXt arm was skipped. Current public notebooks may have changed; their recipes are not automatically proof of historical checkpoint training.

The exact weak-label dataset cannot be certified for every component from the released evidence. This audit distinguishes matching label values, identified historical files, published recipes, and unresolved provenance.

## Every deployed family

| Family | Label source | Evidence boundary |
| --- | --- | --- |
| DINOv2 | Pilkwang report_labels_v2.csv | Saved binary weak targets verified; exact per-member soft-target recipe not certified |
| A5 | Unresolved | Checkpoint-only release; no training-label manifest found |
| RadImageNet reference/V15 | Mean of report_labels_v2.csv, llm_labels_v2.csv, labels_llm_gpt56sol.csv | Published training recipe recovered; historical checkpoint-to-label-file receipts absent |
| RadImageNet E13 | Same three-teacher recipe in published family training code | Exact historical E13 source unavailable through tested API routes; recipe-level evidence only |
| Raptor V5 | Private LLM soft targets; labels_llm_soft.parquet default | Author documents label type but explicitly withholds original label table |
| Raptor V8 | Private LLM soft-target lineage | Exact version-specific table, reader, prompt, and hashes not published |
| Raptor V10 | Private LLM soft-target lineage | Exact version-specific table, reader, prompt, and hashes not published |
| CoAt Residual | targets_dualsharp_soft.csv + targets_dualsharp_weight.csv | Historical label/weight hashes recovered; actual CSVs and generator not published |
| CoAt Global96 | targets_dualsharp_soft.csv + targets_dualsharp_weight.csv | Historical label/weight hashes recovered; actual CSVs and generator not published |
| CoAt Repair-v1 | targets_dualsharp_soft.csv + targets_dualsharp_weight.csv | Historical label/weight hashes recovered; actual CSVs and generator not published |
| CoAt D4 | targets_dualsharp_soft.csv + targets_dualsharp_weight.csv | Hash-matched parent/adapter/data-builder sources name the pair; actual label-file hashes not in D4 receipt |
| Embedded Rad calibrator | Unresolved fitting targets | Coefficients and inference inputs present; fitting-label provenance not included |

## DINO: direct value comparison

All 4,349 non-gold studies are covered by Pilkwang's table. Thresholding its scores at >0.5 matches all 52,188 saved weak binary labels in dino/oof.npz. Including 57 gold studies with report scores gives 4,406 studies and 52,872 matching values, with zero disagreements. The remaining study is gold and absent from the teacher CSV; its saved y_derived row is all zeros. Do not interpret that row as its official gold annotation or assume a lexicon fallback was stored.

The published baseline recipe loads the soft scores, uses per-cell weight 0.25 + 0.75 * confidence, and overrides annotated studies with gold labels and weight 3.0 when included in a training fold. The 20-member manifest records members, folds, seeds and configuration, but no per-member label-file receipt. The exact binary match therefore proves saved report-target provenance, not every historical optimization target or training exclusion.

Pilkwang CSV score/confidence pairs: 0.08/0.85, 0.28/0.05, 0.68/0.95, 0.82/0.95, 0.94/0.95. The published reader uses YES/NO/UNK and graded severity. Its default hosted-model name is not a run receipt proving which model actually generated this CSV.

## RadImageNet: three-teacher training recipe

The released V15 and V20 training functions explicitly read these three files; the family source also contains the same make_targets function. The following downloaded files match the hashes listed in the public family source:

- Pilkwang: report_labels_v2.csv (4406 rows). Dataset: https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels. SHA-256: `6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920`.
- Steven Lee Hans: llm_labels_v2.csv (4407 rows). Dataset: https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels. SHA-256: `3c082a987939528a55202028f655bd6f9dc5c9e28eff5058bbae7ef088a9f144`.
- lixin73: labels_llm_gpt56sol.csv (4407 rows). Dataset: https://www.kaggle.com/datasets/lixin73/rsna-knee-llm-report-labels-sol56. SHA-256: `a79b05e609336811d51c97f9a48f00ea8004244a1a80fe886504a62bb39867ae`.

For each study and finding:

```text
y = mean of available teacher probabilities (at least two required)
disagreement = mean(abs(teacher_probability - y))
agreement = clip(1 - 2 * disagreement, 0, 1)
certainty = clip(2 * abs(y - 0.5), 0, 1)
weight = 0.15 + 0.85 * (0.65 * agreement + 0.35 * certainty)
On fully annotated studies: y = official gold label; weight = 3.0
```

It reads Steven's v2 file, not the later llm_labels_v4_blend.csv. The published recipe freezes the RadImageNet encoder and trains the prediction heads. The submitted graph loads five reference heads (hash 0f465649799ecfbccaac1767844639e7ced44e1bc9babde6e4bac7c5d9b89eaa) and five E13 heads (hash ad9f19af73bfdf4e49263c0e45060dc3cb239e1195039b26dc8c0a3a6bcd1a8a). The same E13 heads are reused on the E11 slice layout; that pass is not a third trained family. Unused attached alternative Rad head bundles are not counted as deployed members.

The published recipe can be reconstructed from these files, but no recovered receipt binds the exact historical head bundles to all three teacher-file hashes. E13 historical API requests returned 403/404; the latest notebook is V41 and is not substituted for the pinned source. reconstructed_rad_recipe.npz is explicitly a recipe reconstruction, not recovered original training data.

## CoAtNet: DualSharp files

Global96, Residual and Repair histories record the same exact files by hash:

- `targets_dualsharp_soft.csv`: `3b360a6a029ee1864324fba0e895dd84065351495ec6bcff109cd5d2888c888f`.
- `targets_dualsharp_weight.csv`: `cc1f5012fe41e71f6ce3d86358862f5bfa65d3d16864ef84a49eeca9b4fac3cb`.

Hash-matched trainer files read their probabilities and confidence weights directly, without binarizing them. The report-labelled cohort is 4,349 studies. The 58 gold studies are excluded from gradient updates and used for checkpoint/SWA selection. The loss is confidence-weighted BCE plus a rank term with positive targets >0.6 and negative targets <0.4. D4 freezes its parent and trains only the depth adapter using a hash-matched data builder that names the same default CSV pair. D4's receipt does not independently supply the label/weight file hashes.

Neither DualSharp CSV nor its construction script is included in the inspected releases. Consequently, its teacher identities, prompts, sharpening transform and individual values cannot be established. Do not equate it with the Rad three-teacher mean or with Pilkwang merely because the outputs have the same twelve columns.

## Raptor and A5: unresolved exact values

Raptor's public trainer explicitly says the training label table is not provided. It describes LLM-generated probabilities from the reports, reads --labels or the default labels_llm_soft.parquet, excludes Gold58, and uses soft-label BCE. The released trainer describes an earlier 44-slice model and is not a byte-verified V5/V8/V10 run script. The V5 checkpoint serves forward/reverse views; V8 and V10 are separate checkpoints. Their actual probability tables, generator models, prompts and hashes are unavailable.

A5 releases m_f0.pt through m_f4.pt plus a runtime wheel. Its inference notebook validates configuration and label order, but contains no label-construction recipe for those checkpoints. Dataset metadata has no training description. An LLM label input attached to the inference notebook does not prove that A5 used it during training.

The embedded Rad calibrator contains coefficients and feature groups but no fitting-target receipt, so its supervision remains unresolved too.

## Evidence needed to finish an exact audit

1. A5's original trainer and label-file hashes or effective target/weight matrices.
2. Raptor V5/V8/V10 original label parquet files and per-run source/checkpoint receipts.
3. The two DualSharp CSVs with the hashes above, plus their generator and input versions.
4. Historical DINO/Rad training receipts that bind each deployed checkpoint to label hashes, gold overrides and fold membership; calibrator fitting code and target source.

## Verification

The audit checked all three public teacher hashes, six historical trainer/data-builder hashes, DINO target order and binary values, shared CoAt file identities, and the reconstructed Rad recipe's coverage/finite values/gold overrides. No checkpoint was deserialized and no MRI image was downloaded. Run scripts/audit_weak_label_provenance.py to repeat the local checks. weak_label_provenance.json includes all 20 DINO member records and the family inventory.
