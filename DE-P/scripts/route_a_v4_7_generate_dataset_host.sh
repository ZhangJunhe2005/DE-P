#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
MAPS="$ROOT/data/route_a_v4_7_mixed_maps"
RAW="$ROOT/data/route_a_v4_7_raw_static"
DERIVED="$ROOT/data/route_a_v4_7_static_yopo"
DERIVED_STAGING="$ROOT/data/.route_a_v4_7_static_yopo.staging"
CONFIG="$ROOT/configs/route_a_v4_7_raw_static_resolved.yaml"
WORKERS=8
DEVICE=0
RESUME=0
MAPS_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --maps-only) MAPS_ONLY=1; shift ;;
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
    python tools/prepare_route_a_v4_7_maps.py plan
fi
for split in train valid; do
  if [[ ! -e "$MAPS/manifests/${split}_maps.json" ]]; then
    conda run --no-capture-output -n yopo \
      python tools/prepare_route_a_v4_7_maps.py generate \
      --split "$split" --workers "$WORKERS"
  fi
done
conda run --no-capture-output -n yopo \
  python tools/prepare_route_a_v4_7_maps.py status
if [[ "$MAPS_ONLY" -eq 1 ]]; then
  echo '{"status":"PASS","stage":"V4.7_MAPS_ONLY","raw_generation_started":false,"training_started":false}'
  exit 0
fi
conda run --no-capture-output -n yopo \
  python tools/prepare_route_a_v4_7_raw_config.py

AVAILABLE="$(df --output=avail -B1 "$ROOT/data" | tail -n 1 | tr -d ' ')"
REQUIRED=$((30 * 1024 * 1024 * 1024))
if [[ ! -d "$RAW" && "$AVAILABLE" -lt "$REQUIRED" ]]; then
  echo "V4.7 first generation requires at least 30 GiB free." >&2
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
  if [[ -e "$DERIVED_STAGING" ]]; then
    echo "Refusing to overwrite incomplete V4.7 derived staging: $DERIVED_STAGING" >&2
    echo "Move it aside after inspection, then re-run this command." >&2
    exit 1
  fi
  conda run --no-capture-output -n yopo \
    python tools/build_route_a_v4_static_dataset.py \
      --authorize-build \
      --source "$RAW" \
      --output "$DERIVED" \
      --raw-dataset-version route_a_v4_7_raw_static_v1 \
      --derived-dataset-version route_a_v4_7_static_yopo \
      --derived-schema-version static_yopo_sample_v4_7_route_a_four_scene \
      --split-contract-version route_a_v4_7_split_contract_v1 \
      --catalog-version 7
fi

conda run --no-capture-output -n yopo \
  python tools/validate_route_a_v4_7_dataset.py

echo '{"status":"PASS","dataset":"route_a_v4_7_static_yopo","room_samples":0,"training_started":false}'
