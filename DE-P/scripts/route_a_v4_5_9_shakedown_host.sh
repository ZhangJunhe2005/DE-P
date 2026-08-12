#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
CONFIG="$ROOT/configs/route_a_v4_5_9_time_mean_safety_shakedown.yaml"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$ROOT"
conda run --no-capture-output -n yopo python tools/validate_route_a_v4_2_dataset.py
conda run --no-capture-output -n yopo \
  python tools/prepare_route_a_v4_5_9_training_config.py
conda run --no-capture-output -n yopo python -m pytest -q \
  tests/test_static_yopo_parity_v4_5_8.py \
  tests/test_static_yopo_parity_v4_5_9.py \
  tests/test_runtime_safety_v1.py \
  tests/test_runtime_profile_v4_5_7.py \
  tests/test_static_yopo_split_training_v1.py
conda run --no-capture-output -n yopo \
  python tools/train_mixed_static_yopo_v1.py --config "$CONFIG" --verify-only

args=(python tools/train_mixed_static_yopo_v1.py --config "$CONFIG" --authorized)
if [[ "$DRY_RUN" == "1" ]]; then args+=(--dry-run); fi
conda run --no-capture-output -n yopo "${args[@]}"
