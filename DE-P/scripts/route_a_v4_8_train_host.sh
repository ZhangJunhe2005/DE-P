#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
CONFIG="$ROOT/configs/route_a_v4_8_recovery_capacity_shakedown.yaml"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$ROOT"
# V4.8 reuses the frozen V4.7 dataset; it never generates or rewrites data.
conda run --no-capture-output -n yopo \
  python tools/validate_route_a_v4_7_dataset.py
conda run --no-capture-output -n yopo \
  python tools/prepare_route_a_v4_8_training_config.py
conda run --no-capture-output -n yopo python -m pytest -q \
  tests/test_route_a_v4_8_training_entry.py \
  tests/test_static_yopo_recovery_state_v4_8.py \
  tests/test_static_yopo_recovery_coverage_v4_8.py \
  tests/test_route_a_v4_7_dynamic_runtime.py \
  tests/test_runtime_profile_v4_5_10.py \
  tests/test_static_yopo_split_training_v1.py
conda run --no-capture-output -n yopo \
  python tools/train_mixed_static_yopo_v1.py \
    --config "$CONFIG" --verify-only

# Reaching this line means the user explicitly launched this host entry.
args=(python tools/train_mixed_static_yopo_v1.py --config "$CONFIG" --authorized)
if [[ "$DRY_RUN" -eq 1 ]]; then args+=(--dry-run); fi
conda run --no-capture-output -n yopo "${args[@]}"
