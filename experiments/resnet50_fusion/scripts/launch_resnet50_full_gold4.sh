#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export KAGGLE_CONFIG_DIR="$TASK_ROOT/.auth"
RUN_ROOT="$TASK_ROOT/run_full4333_gold74_4ep"
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/launcher.lock"
flock -n 9 || { echo 'The full-data launcher is already active'; exit 1; }

# Reuse the previously verified V2 shards and pinned pretrained assets. The
# staging command rechecks their publication manifests before every launch.
"$TASK_ROOT/.venv/bin/python" scripts/stage_resnet50_cloud.py \
    --data "$TASK_ROOT/data" --expected "$TASK_ROOT/expected_manifests" \
    --assets "$TASK_ROOT/assets"

ARGS=(--data "$TASK_ROOT/data" --labels "$TASK_ROOT/labels"
      --raptor "$TASK_ROOT/assets/raptor_ft_coatnet_v5_full_swa.pt"
      --resnet "$TASK_ROOT/assets/resnet50-11ad3fa6.pth"
      --output "$RUN_ROOT" --epochs 4)
if [[ -f "$RUN_ROOT/latest_checkpoint.json" ]]; then
    LATEST="$("$TASK_ROOT/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$RUN_ROOT/latest_checkpoint.json")"
    ARGS+=(--resume "$RUN_ROOT/$LATEST")
elif [[ -f "$RUN_ROOT/checkpoints/initial.pt" ]]; then
    ARGS+=(--resume "$RUN_ROOT/checkpoints/initial.pt")
fi

exec "$TASK_ROOT/.venv/bin/python" -m scripts.train_resnet50_full_gold4 "${ARGS[@]}"
