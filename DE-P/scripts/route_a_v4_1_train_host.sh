#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
CONFIG="$ROOT/configs/route_a_v4_1_local_goal_training.yaml"
RESUME_CHECKPOINT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME_CHECKPOINT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$ROOT"
[[ -f "$CONFIG" ]] || conda run --no-capture-output -n yopo \
  python tools/prepare_route_a_v4_1_training_config.py
conda run --no-capture-output -n yopo \
  python tools/train_mixed_static_yopo_v1.py \
  --config "$CONFIG" --verify-only

ARGS=(python tools/train_mixed_static_yopo_v1.py --config "$CONFIG" --authorized)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  ARGS+=(--resume "$RESUME_CHECKPOINT")
fi
conda run --no-capture-output -n yopo "${ARGS[@]}"
