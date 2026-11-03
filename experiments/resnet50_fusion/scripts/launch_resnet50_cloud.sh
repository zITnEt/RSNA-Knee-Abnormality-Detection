#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
RUN_ROOT="${RUN_ROOT:-$TASK_ROOT/run}"
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/launcher.lock"
flock -n 9 || { echo 'A launcher is already active'; exit 1; }
python scripts/stage_resnet50_cloud.py --data "$TASK_ROOT/data" --expected "$TASK_ROOT/expected_manifests" --assets "$TASK_ROOT/assets"
ARGS=(--data "$TASK_ROOT/data" --labels "$TASK_ROOT/labels" --raptor "$TASK_ROOT/assets/raptor_ft_coatnet_v5_full_swa.pt" --resnet "$TASK_ROOT/assets/resnet50-11ad3fa6.pth" --output "$RUN_ROOT" --epochs 10)
if [[ -f "$RUN_ROOT/latest_checkpoint.json" ]]; then
    LATEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$RUN_ROOT/latest_checkpoint.json")"
    ARGS+=(--resume "$RUN_ROOT/$LATEST")
elif [[ -f "$RUN_ROOT/checkpoints/initial.pt" ]]; then
    ARGS+=(--resume "$RUN_ROOT/checkpoints/initial.pt")
fi
python -m scripts.train_resnet50_cloud "${ARGS[@]}"
