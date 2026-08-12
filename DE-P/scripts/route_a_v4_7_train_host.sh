#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
CONFIG="$ROOT/configs/route_a_v4_7_original_density_finetune.yaml"
DRY_RUN=0
RESUME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --resume)
      [[ $# -ge 2 ]] || { echo "--resume requires a run path" >&2; exit 2; }
      RESUME="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$ROOT"
conda run --no-capture-output -n yopo \
  python tools/validate_route_a_v4_7_dataset.py
conda run --no-capture-output -n yopo \
  python tools/prepare_route_a_v4_7_training_config.py
conda run --no-capture-output -n yopo python -m pytest -q \
  tests/test_route_a_v4_7_training_entry.py \
  tests/test_static_yopo_parity_v4_5_10.py \
  tests/test_runtime_profile_v4_5_10.py \
  tests/test_static_yopo_split_training_v1.py
conda run --no-capture-output -n yopo \
  python tools/train_mixed_static_yopo_v1.py \
    --config "$CONFIG" --verify-only

args=(python tools/train_mixed_static_yopo_v1.py --config "$CONFIG" --authorized)
if [[ "$DRY_RUN" -eq 1 ]]; then args+=(--dry-run); fi
if [[ -n "$RESUME" ]]; then args+=(--resume "$RESUME"); fi
conda run --no-capture-output -n yopo "${args[@]}"
