#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
CONFIG="$ROOT/configs/route_a_v4_2_3_ego_feasibility_training.yaml"
RESUME_CHECKPOINT=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME_CHECKPOINT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$ROOT"
conda run --no-capture-output -n yopo \
  python tools/validate_route_a_v4_2_dataset.py
conda run --no-capture-output -n yopo \
  python tools/prepare_route_a_v4_2_3_ego_feasibility_training_config.py
conda run --no-capture-output -n yopo \
  python tools/train_mixed_static_yopo_v1.py --config "$CONFIG" --verify-only

ARGS=(python tools/train_mixed_static_yopo_v1.py --config "$CONFIG" --authorized)
if [[ "$DRY_RUN" == "1" ]]; then ARGS+=(--dry-run); fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then ARGS+=(--resume "$RESUME_CHECKPOINT"); fi
conda run --no-capture-output -n yopo "${ARGS[@]}"
