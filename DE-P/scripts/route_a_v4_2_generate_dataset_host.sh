#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
CONFIG="$ROOT/configs/route_a_v4_2_raw_static_resolved.yaml"
RAW="$ROOT/data/route_a_v4_2_raw_static"
DERIVED="$ROOT/data/route_a_v4_2_static_yopo"
WORKERS=8
DEVICE=0
RESUME=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$ROOT"
conda run --no-capture-output -n yopo \
  python tools/prepare_route_a_v4_2_raw_config.py

AVAILABLE="$(df --output=avail -B1 "$ROOT/data" | tail -n 1 | tr -d ' ')"
REQUIRED=$((30 * 1024 * 1024 * 1024))
if [[ ! -d "$RAW" && "$AVAILABLE" -lt "$REQUIRED" ]]; then
  echo "V4.2 requires at least 30 GiB free before first generation." >&2
  echo "available_bytes=$AVAILABLE" >&2
  exit 1
fi

run_split() {
  local split="$1"
  local args=(--config "$CONFIG" --output "$RAW" --workers "$WORKERS"
              --device "$DEVICE" --fail-fast --split "$split")
  if [[ "$RESUME" -eq 1 || -d "$RAW" ]]; then
    args+=(--resume)
  fi
  conda run --no-capture-output -n yopo \
    python -m authoritative_dataset.generate_v1 "${args[@]}"
}

run_split train
run_split valid

if [[ ! -d "$DERIVED" ]]; then
  conda run --no-capture-output -n yopo \
    python tools/build_route_a_v4_static_dataset.py \
      --authorize-build \
      --source "$RAW" \
      --output "$DERIVED" \
      --raw-dataset-version route_a_v4_2_raw_static_v1 \
      --derived-dataset-version route_a_v4_2_static_yopo \
      --derived-schema-version static_yopo_sample_v4_2_route_a \
      --split-contract-version route_a_v4_2_split_contract_v1 \
      --catalog-version 5
fi

conda run --no-capture-output -n yopo \
  python tools/validate_route_a_v4_2_dataset.py
conda run --no-capture-output -n yopo \
  python tools/prepare_route_a_v4_2_training_config.py
conda run --no-capture-output -n yopo \
  python tools/train_mixed_static_yopo_v1.py \
    --config configs/route_a_v4_2_scene_relative_training.yaml --verify-only

echo '{"status":"PASS","dataset":"route_a_v4_2_static_yopo","training_started":false}'
