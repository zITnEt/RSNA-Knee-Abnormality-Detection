# BASELINE-001 reproduction notes

Status: audit complete; exact historical V10 identified; private Version #1 completed successfully; user submitted it to the competition; hidden-test rerun pending.

No modeling changes are permitted. The published V10 run skipped the optional ConvNeXt arm because its manifest was absent and retained the exact 0.943 anchor. The reproduction must do the same.

The private fork is `zafarnematov/rsna-knee-dinosaur-v5`. Kaggle accepted Version #1 on 2026-09-27 with internet off and GPU T4 x2 selected. It later completed in 1131.5 seconds. All fail-closed checkpoint gates passed. The three-row preflight `submission.csv` exactly matches the saved 0.943 anchor and has SHA-256 `8f0c5e7b2b561538dae55439a4d2e9173141096da3defd9bc0a4bac351eef37a`. The user submitted Version #1 as Kaggle submission `56627210` on 2026-09-28; the hidden-test rerun is pending.
