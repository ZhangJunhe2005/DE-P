#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
MAPS="$ROOT/data/route_a_v4_6_mixed_maps"
RAW="$ROOT/data/route_a_v4_6_raw_static"
DERIVED="$ROOT/data/route_a_v4_6_static_yopo"
DERIVED_STAGING="$ROOT/data/.route_a_v4_6_static_yopo.staging"
CONFIG="$ROOT/configs/route_a_v4_6_raw_static_resolved.yaml"
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
if [[ ! -x artifacts/mixed_scene_map_generator/yopo_mixed_scene_map_generator ]]; then
  echo "Missing mixed-map generator; run scripts/build_mixed_scene_map_generator.sh" >&2
  exit 1
fi

if [[ ! -e "$MAPS/manifests/map_plan.json" ]]; then
  conda run --no-capture-output -n yopo \
    python tools/prepare_route_a_v4_6_maps.py plan
fi
for split in train valid; do
  if [[ ! -e "$MAPS/manifests/${split}_maps.json" ]]; then
    conda run --no-capture-output -n yopo \
      python tools/prepare_route_a_v4_6_maps.py generate \
      --split "$split" --workers "$WORKERS"
  fi
done
conda run --no-capture-output -n yopo \
  python tools/prepare_route_a_v4_6_maps.py status
conda run --no-capture-output -n yopo \
  python tools/prepare_route_a_v4_6_raw_config.py

AVAILABLE="$(df --output=avail -B1 "$ROOT/data" | tail -n 1 | tr -d ' ')"
REQUIRED=$((30 * 1024 * 1024 * 1024))
if [[ ! -d "$RAW" && "$AVAILABLE" -lt "$REQUIRED" ]]; then
  echo "V4.6 first generation requires at least 30 GiB free." >&2
  echo "available_bytes=$AVAILABLE" >&2
  exit 1
fi

for split in train valid; do
  args=(
    python -m authoritative_dataset.generate_v1
    --config "$CONFIG"
    --output "$RAW"
    --split "$split"
    --workers "$WORKERS"
    --device "$DEVICE"
    --fail-fast
  )
  if [[ "$RESUME" -eq 1 || -d "$RAW" ]]; then
    args+=(--resume)
  fi
  conda run --no-capture-output -n yopo "${args[@]}"
done

if [[ ! -d "$DERIVED" ]]; then
  if [[ -d "$DERIVED_STAGING" ]]; then
    if [[ "$RESUME" -ne 1 ]]; then
      echo "Incomplete V4.6 derived staging exists: $DERIVED_STAGING" >&2
      echo "Re-run this script with --resume to discard only that incomplete staging and rebuild the derived view." >&2
      exit 1
    fi
    echo "Removing incomplete generated staging before deterministic rebuild: $DERIVED_STAGING"
    rm -rf -- "$DERIVED_STAGING"
  fi
  conda run --no-capture-output -n yopo \
    python tools/build_route_a_v4_static_dataset.py \
      --authorize-build \
      --source "$RAW" \
      --output "$DERIVED" \
      --raw-dataset-version route_a_v4_6_raw_static_v1 \
      --derived-dataset-version route_a_v4_6_static_yopo \
      --derived-schema-version static_yopo_sample_v4_6_route_a_four_scene \
      --split-contract-version route_a_v4_6_split_contract_v1 \
      --catalog-version 6
fi

conda run --no-capture-output -n yopo \
  python tools/validate_route_a_v4_6_dataset.py
conda run --no-capture-output -n yopo \
  python tools/prepare_route_a_v4_6_training_config.py
conda run --no-capture-output -n yopo \
  python tools/train_mixed_static_yopo_v1.py \
    --config configs/route_a_v4_6_four_scene_finetune.yaml --verify-only

echo '{"status":"PASS","dataset":"route_a_v4_6_static_yopo","room_samples":0,"training_started":false}'
