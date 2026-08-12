#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
CONFIG="$ROOT/configs/route_a_v4_5_1_relative_kinematic_training.yaml"
AUTHORIZED=0
RESUME_CHECKPOINT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --authorize-long-training) AUTHORIZED=1; shift ;;
    --resume) RESUME_CHECKPOINT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$AUTHORIZED" != "1" ]]; then
  echo "Refusing long V4.5.1 training before shakedown review." >&2
  echo "Run scripts/route_a_v4_5_1_shakedown_host.sh first." >&2
  exit 2
fi

cd "$ROOT"
conda run --no-capture-output -n yopo \
  python tools/prepare_route_a_v4_5_1_training_configs.py
conda run --no-capture-output -n yopo \
  python tools/train_mixed_static_yopo_v1.py --config "$CONFIG" --verify-only
args=(python tools/train_mixed_static_yopo_v1.py --config "$CONFIG" --authorized)
if [[ -n "$RESUME_CHECKPOINT" ]]; then args+=(--resume "$RESUME_CHECKPOINT"); fi
conda run --no-capture-output -n yopo "${args[@]}"
