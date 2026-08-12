#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
TARGET="all"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --map-type) TARGET="$2"; shift 2 ;;
    --all) TARGET="all"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

MAP_TYPES=(cave forest pillar room wall)
if [[ "$TARGET" != "all" ]]; then
  case "$TARGET" in
    cave|forest|pillar|room|wall) MAP_TYPES=("$TARGET") ;;
    *) echo "Invalid map type: $TARGET" >&2; exit 2 ;;
  esac
fi

cd "$ROOT"
conda run --no-capture-output -n yopo \
  python tools/prepare_route_a_v4_5_training_configs.py
conda run --no-capture-output -n yopo python -m pytest -q \
  tests/test_static_yopo_parity_v4_5.py \
  tests/test_runtime_profile_v4_5.py \
  tests/test_runtime_safety_v1.py

for map_type in "${MAP_TYPES[@]}"; do
  config="$ROOT/configs/route_a_v4_5_probe_${map_type}.yaml"
  conda run --no-capture-output -n yopo \
    python tools/train_mixed_static_yopo_v1.py \
      --config "$config" --verify-only
  args=(python tools/train_mixed_static_yopo_v1.py --config "$config" --authorized)
  if [[ "$DRY_RUN" == "1" ]]; then args+=(--dry-run); fi
  conda run --no-capture-output -n yopo "${args[@]}"
done
