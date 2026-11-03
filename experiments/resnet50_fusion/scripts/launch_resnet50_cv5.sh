#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="$TASK_ROOT/run_cv5"
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/launcher.lock"
flock -n 9 || { echo 'The CV launcher is already active' >&2; exit 1; }

cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export KAGGLE_CONFIG_DIR="$TASK_ROOT/.auth"

# Recheck all 32 published V2 shard manifests and pinned parent assets before
# resuming. Every fold starts afresh from these same parent weights.
"$TASK_ROOT/.venv/bin/python" scripts/stage_resnet50_cloud.py \
    --data "$TASK_ROOT/data" --expected "$TASK_ROOT/expected_manifests" \
    --assets "$TASK_ROOT/assets"

for FOLD in 0 1 2 3 4; do
    FOLD_ROOT="$RUN_ROOT/fold_$FOLD"
    if [[ -f "$FOLD_ROOT/complete.json" ]]; then
        echo "Fold $FOLD already complete"
        continue
    fi
    mkdir -p "$FOLD_ROOT"
    ARGS=(--data "$TASK_ROOT/data" --labels "$TASK_ROOT/labels"
          --folds-csv "$TASK_ROOT/configs/cv5_folds.csv" --fold "$FOLD"
          --raptor "$TASK_ROOT/assets/raptor_ft_coatnet_v5_full_swa.pt"
          --resnet "$TASK_ROOT/assets/resnet50-11ad3fa6.pth"
          --output "$FOLD_ROOT" --epochs 5)
    if [[ -f "$FOLD_ROOT/latest_checkpoint.json" ]]; then
        LATEST="$("$TASK_ROOT/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$FOLD_ROOT/latest_checkpoint.json")"
        ARGS+=(--resume "$FOLD_ROOT/$LATEST")
    elif [[ -f "$FOLD_ROOT/checkpoints/initial.pt" ]]; then
        ARGS+=(--resume "$FOLD_ROOT/checkpoints/initial.pt")
    fi
    echo "Starting fold $FOLD at $(date --utc --iso-8601=seconds)"
    "$TASK_ROOT/.venv/bin/python" -m scripts.train_resnet50_cv5 "${ARGS[@]}"
done
"$TASK_ROOT/.venv/bin/python" -m scripts.summarize_resnet50_cv5 --run-root "$RUN_ROOT"
echo "All five CV folds complete at $(date --utc --iso-8601=seconds)"
