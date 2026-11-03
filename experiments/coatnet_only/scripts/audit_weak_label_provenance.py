"""Audit available label evidence without loading checkpoints or training models."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "audit/label_provenance"
OUT = ROOT / "results/training_analysis"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


train = pd.read_csv(ROOT / "audit/training/labels/train.csv")
targets = train.columns.tolist()[2:]
gold = train[targets].notna().all(axis=1).to_numpy()
assert len(train) == 4407 and int(gold.sum()) == 58 and len(targets) == 12
config = read(OUT / "training_configurations.json")
teacher_specs = [
    ("Pilkwang", "pilkwang/report_labels_v2.csv", "pilkwang/rsna-knee-llm-labels",
     "6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920"),
    ("Steven Lee Hans", "stevenleehans/llm_labels_v2.csv", "stevenleehans/rsna-knee-llm-report-labels",
     "3c082a987939528a55202028f655bd6f9dc5c9e28eff5058bbae7ef088a9f144"),
    ("lixin73", "lixin73/labels_llm_gpt56sol.csv", "lixin73/rsna-knee-llm-report-labels-sol56",
     "a79b05e609336811d51c97f9a48f00ea8004244a1a80fe886504a62bb39867ae"),
]
teachers, matrices = [], []
for name, file, slug, expected in teacher_specs:
    path = SRC / file
    assert digest(path) == expected
    frame = pd.read_csv(path)
    assert not frame.StudyInstanceUID.duplicated().any()
    aligned = train[["StudyInstanceUID"]].merge(frame, on="StudyInstanceUID", how="left")
    matrices.append(aligned[targets].to_numpy(float))
    teachers.append(dict(name=name, file=str(path), kaggle_dataset=slug,
                         rows=len(frame), sha256=expected))

with np.load(ROOT / "audit/training/dino/oof.npz", allow_pickle=False) as bundle:
    ids = bundle["ids"].astype(str)
    binary = bundle["y_derived"].copy()
    assert bundle["targets"].tolist() == targets
    by_id = train.set_index("StudyInstanceUID").reindex(ids)
    official_gold = by_id[targets].notna().all(axis=1).to_numpy()
    assert np.array_equal(bundle["gold_mask"].astype(bool), official_gold)
pil = pd.read_csv(SRC / teacher_specs[0][1]).set_index("StudyInstanceUID").reindex(ids)
p = pil[targets].to_numpy(float)
present = np.isfinite(p).all(axis=1)
weak = ~official_gold
assert present[weak].all()
assert np.array_equal((p[present] > .5).astype(int), binary[present])
dino_evidence = dict(
    overlapping_studies=int(present.sum()), overlapping_values=int(present.sum() * 12),
    binary_disagreements=0, weak_studies=int(weak.sum()), weak_label_values=int(weak.sum() * 12),
    missing_teacher_studies=ids[~present].tolist(),
    missing_studies_are_gold=bool(official_gold[~present].all()),
    saved_missing_binary_rows=binary[~present].tolist(),
    boundary="This proves the saved report-derived binary vector matches this teacher. "
             "It does not prove every checkpoint's historical soft scores, confidence weights, "
             "gold override, or fold exclusions.",
)

trainer_specs = [
    ("Global96", "global96/train_coatnet_global96_exact.py", config["Global96"]["trainer_sha256"]),
    ("Repair", "repair/train_coatnet_repairv1_exact.py", config["Repair"]["trainer_sha256"]),
    ("Residual H100", "residual/trainer_h100_epochs07_10.py", config["Residual"]["trainer_sha256"]),
    ("D4 adapter", "d4/executed_train_coatnet_slice_depth_ablation.py", config["D4"]["source_sha256"]["train_coatnet_slice_depth_ablation.py"]),
    ("D4 parent", "d4/executed_train_coatnet_global96_joint_mha_k12_effb24_e24_full_gold_swa.py", config["D4"]["source_sha256"]["train_coatnet_global96_joint_mha_k12_effb24_e24_full_gold_swa.py"]),
    ("D4 data builder", "d4/executed_train_coatnet_adaptive_patch_guard.py", config["D4"]["source_sha256"]["train_coatnet_adaptive_patch_guard.py"]),
]
verified_trainers = []
for name, file, expected in trainer_specs:
    path = SRC / file
    assert digest(path) == expected, name
    verified_trainers.append(dict(name=name, file=str(path), sha256=expected))
shared = {key: config["Global96"][key] for key in ["labels_sha256", "weights_sha256"]}
for family in ["Residual", "Repair"]:
    assert all(config[family][key] == value for key, value in shared.items())

cube = np.stack(matrices)
assert (np.isfinite(cube).sum(0) >= 2).all()
y = np.nanmean(cube, axis=0).astype(np.float32)
disagreement = np.nanmean(np.abs(cube - y[None]), axis=0)
agreement = np.clip(1 - 2 * disagreement, 0, 1)
certainty = np.clip(2 * np.abs(y - .5), 0, 1)
w = (.15 + .85 * (.65 * agreement + .35 * certainty)).astype(np.float32)
y[gold] = train.loc[gold, targets].to_numpy(np.float32)
w[gold] = 3.0
assert np.isfinite(y).all() and np.isfinite(w).all()
assert np.array_equal(y[gold], train.loc[gold, targets].to_numpy(np.float32))
# This is a reconstruction of the published recipe, not a recovered historical receipt.
np.savez_compressed(SRC / "reconstructed_rad_recipe.npz", study_uids=train.StudyInstanceUID.to_numpy(str),
                    targets=np.asarray(targets), target=y, weight=w, gold_mask=gold)

families = [
    dict(family="DINOv2", members=20, labels="Pilkwang report_labels_v2.csv",
         status="Saved binary weak targets verified; exact per-member soft-target recipe not certified"),
    dict(family="A5", members=5, labels="Unresolved", status="Checkpoint-only release; no training-label manifest found"),
    dict(family="RadImageNet reference/V15", members=5,
         labels="Mean of report_labels_v2.csv, llm_labels_v2.csv, labels_llm_gpt56sol.csv",
         status="Published training recipe recovered; historical checkpoint-to-label-file receipts absent"),
    dict(family="RadImageNet E13", members=5,
         labels="Same three-teacher recipe in published family training code",
         status="Exact historical E13 source unavailable through tested API routes; recipe-level evidence only"),
    dict(family="Raptor V5", members=1, views=2, labels="Private LLM soft targets; labels_llm_soft.parquet default",
         status="Author documents label type but explicitly withholds original label table"),
    dict(family="Raptor V8", members=1, labels="Private LLM soft-target lineage",
         status="Exact version-specific table, reader, prompt, and hashes not published"),
    dict(family="Raptor V10", members=1, labels="Private LLM soft-target lineage",
         status="Exact version-specific table, reader, prompt, and hashes not published"),
    dict(family="CoAt Residual", epochs=[4, 6, 8], labels="targets_dualsharp_soft.csv + targets_dualsharp_weight.csv",
         status="Historical label/weight hashes recovered; actual CSVs and generator not published", **shared),
    dict(family="CoAt Global96", epochs=[16, 23, 18], labels="targets_dualsharp_soft.csv + targets_dualsharp_weight.csv",
         status="Historical label/weight hashes recovered; actual CSVs and generator not published", **shared),
    dict(family="CoAt Repair-v1", epochs=[12, 7, 11], labels="targets_dualsharp_soft.csv + targets_dualsharp_weight.csv",
         status="Historical label/weight hashes recovered; actual CSVs and generator not published", **shared),
    dict(family="CoAt D4", parent_swa_epochs=[15, 16], adapter_swa_epochs=[11, 9, 5],
         labels="targets_dualsharp_soft.csv + targets_dualsharp_weight.csv",
         status="Hash-matched parent/adapter/data-builder sources name the pair; actual label-file hashes not in D4 receipt"),
    dict(family="Embedded Rad calibrator", labels="Unresolved fitting targets",
         status="Coefficients and inference inputs present; fitting-label provenance not included"),
]
result = dict(scope="Saved BASELINE-001: historical DINOsaur V5 V10, scriptVersionId 351856876, exact 0.943 anchor",
              scope_boundary="Later local notebook copies are not used as proof of this submitted graph.",
              gold_studies=58, weak_studies=4349, targets=targets, teachers=teachers,
              dino_evidence=dino_evidence, verified_trainers=verified_trainers,
              families=families, optional_convnext="Absent/skipped in the saved submitted run",
              dino_members=read(ROOT / "audit/training/dino/manifest.json")["members"])
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "weak_label_provenance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

report = ["# Weak-label provenance of the saved ensemble", "",
          "Audited 2026-10-05. Scope: the saved BASELINE-001 submission, DINOsaur V5 historical V10 "
          "(scriptVersionId 351856876), exact 0.943 Speedy Raptors anchor. The optional ConvNeXt arm was skipped. "
          "Current public notebooks may have changed; their recipes are not automatically proof of historical checkpoint training.", "",
          "The exact weak-label dataset cannot be certified for every component from the released evidence. "
          "This audit distinguishes matching label values, identified historical files, published recipes, and unresolved provenance.", "",
          "## Every deployed family", "", "| Family | Label source | Evidence boundary |",
          "| --- | --- | --- |"]
for family in families:
    report.append(f"| {family['family']} | {family['labels']} | {family['status']} |")
report += ["", "## DINO: direct value comparison", "",
           "All 4,349 non-gold studies are covered by Pilkwang's table. Thresholding its scores at >0.5 "
           "matches all 52,188 saved weak binary labels in dino/oof.npz. Including 57 gold studies with "
           "report scores gives 4,406 studies and 52,872 matching values, with zero disagreements. "
           "The remaining study is gold and absent from the teacher CSV; its saved y_derived row is all zeros. "
           "Do not interpret that row as its official gold annotation or assume a lexicon fallback was stored.", "",
           "The published baseline recipe loads the soft scores, uses per-cell weight 0.25 + 0.75 * confidence, "
           "and overrides annotated studies with gold labels and weight 3.0 when included in a training fold. "
           "The 20-member manifest records members, folds, seeds and configuration, but no per-member "
           "label-file receipt. The exact binary match therefore proves saved report-target provenance, "
           "not every historical optimization target or training exclusion.", "",
           "Pilkwang CSV score/confidence pairs: 0.08/0.85, 0.28/0.05, 0.68/0.95, 0.82/0.95, 0.94/0.95. "
           "The published reader uses YES/NO/UNK and graded severity. Its default hosted-model name is "
           "not a run receipt proving which model actually generated this CSV.", "",
           "## RadImageNet: three-teacher training recipe", "",
           "The released V15 and V20 training functions explicitly read these three files; the family source "
           "also contains the same make_targets function. The following downloaded files match the hashes "
           "listed in the public family source:", ""]
for teacher in teachers:
    report += [f"- {teacher['name']}: {Path(teacher['file']).name} ({teacher['rows']} rows). "
               f"Dataset: https://www.kaggle.com/datasets/{teacher['kaggle_dataset']}. "
               f"SHA-256: `{teacher['sha256']}`."]
report += ["", "For each study and finding:", "", "```text",
           "y = mean of available teacher probabilities (at least two required)",
           "disagreement = mean(abs(teacher_probability - y))",
           "agreement = clip(1 - 2 * disagreement, 0, 1)",
           "certainty = clip(2 * abs(y - 0.5), 0, 1)",
           "weight = 0.15 + 0.85 * (0.65 * agreement + 0.35 * certainty)",
           "On fully annotated studies: y = official gold label; weight = 3.0", "```", "",
           "It reads Steven's v2 file, not the later llm_labels_v4_blend.csv. The published recipe freezes "
           "the RadImageNet encoder and trains the prediction heads. The submitted graph loads five reference "
           "heads (hash 0f465649799ecfbccaac1767844639e7ced44e1bc9babde6e4bac7c5d9b89eaa) "
           "and five E13 heads (hash ad9f19af73bfdf4e49263c0e45060dc3cb239e1195039b26dc8c0a3a6bcd1a8a). "
           "The same E13 heads are reused on the E11 slice layout; that pass is not a third trained family. "
           "Unused attached alternative Rad head bundles are not counted as deployed members.", "",
           "The published recipe can be reconstructed from these files, but no recovered receipt binds "
           "the exact historical head bundles to all three teacher-file hashes. E13 historical API requests "
           "returned 403/404; the latest notebook is V41 and is not substituted for the pinned source. "
           "reconstructed_rad_recipe.npz is explicitly a recipe reconstruction, not recovered original training data.", "",
           "## CoAtNet: DualSharp files", "",
           "Global96, Residual and Repair histories record the same exact files by hash:", "",
           f"- `targets_dualsharp_soft.csv`: `{shared['labels_sha256']}`.",
           f"- `targets_dualsharp_weight.csv`: `{shared['weights_sha256']}`.", "",
           "Hash-matched trainer files read their probabilities and confidence weights directly, without "
           "binarizing them. The report-labelled cohort is 4,349 studies. The 58 gold studies are excluded "
           "from gradient updates and used for checkpoint/SWA selection. The loss is confidence-weighted BCE "
           "plus a rank term with positive targets >0.6 and negative targets <0.4. D4 freezes its parent "
           "and trains only the depth adapter using a hash-matched data builder that names the same default "
           "CSV pair. D4's receipt does not independently supply the label/weight file hashes.", "",
           "Neither DualSharp CSV nor its construction script is included in the inspected releases. "
           "Consequently, its teacher identities, prompts, sharpening transform and individual values "
           "cannot be established. Do not equate it with the Rad three-teacher mean or with Pilkwang "
           "merely because the outputs have the same twelve columns.", "",
           "## Raptor and A5: unresolved exact values", "",
           "Raptor's public trainer explicitly says the training label table is not provided. It "
           "describes LLM-generated probabilities from the reports, reads --labels or the default "
           "labels_llm_soft.parquet, excludes Gold58, and uses soft-label BCE. The released trainer "
           "describes an earlier 44-slice model and is not a byte-verified V5/V8/V10 run script. "
           "The V5 checkpoint serves forward/reverse views; V8 and V10 are separate checkpoints. "
           "Their actual probability tables, generator models, prompts and hashes are unavailable.", "",
           "A5 releases m_f0.pt through m_f4.pt plus a runtime wheel. Its inference notebook validates "
           "configuration and label order, but contains no label-construction recipe for those checkpoints. "
           "Dataset metadata has no training description. An LLM label input attached to the inference "
           "notebook does not prove that A5 used it during training.", "",
           "The embedded Rad calibrator contains coefficients and feature groups but no fitting-target "
           "receipt, so its supervision remains unresolved too.", "",
           "## Evidence needed to finish an exact audit", "",
           "1. A5's original trainer and label-file hashes or effective target/weight matrices.",
           "2. Raptor V5/V8/V10 original label parquet files and per-run source/checkpoint receipts.",
           "3. The two DualSharp CSVs with the hashes above, plus their generator and input versions.",
           "4. Historical DINO/Rad training receipts that bind each deployed checkpoint to label hashes, "
           "gold overrides and fold membership; calibrator fitting code and target source.", "",
           "## Verification", "",
           "The audit checked all three public teacher hashes, six historical trainer/data-builder hashes, "
           "DINO target order and binary values, shared CoAt file identities, and the reconstructed Rad "
           "recipe's coverage/finite values/gold overrides. No checkpoint was deserialized and no MRI "
           "image was downloaded. Run scripts/audit_weak_label_provenance.py to repeat the local checks. "
           "weak_label_provenance.json includes all 20 DINO member records and the family inventory.", ""]
(OUT / "WEAK_LABEL_PROVENANCE.md").write_text("\n".join(report), encoding="utf-8")
print(json.dumps(dict(dino_weak_values_verified=dino_evidence["weak_label_values"],
                      teachers_hash_verified=len(teachers), historical_source_hashes_verified=len(verified_trainers),
                      families_audited=len(families), exact_all_component_provenance=False), indent=2))
