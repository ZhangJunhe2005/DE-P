#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
WORKERS=8
DEVICE=0
RESUME=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --resume) RESUME=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$ROOT"
[[ -x artifacts/mixed_scene_map_generator/yopo_mixed_scene_map_generator ]] || {
  echo "Missing mixed-map generator. Run scripts/build_mixed_scene_map_generator.sh first." >&2
  exit 1
}
[[ ! -e data/phase8_dynamic_evidence_formal_v3/.route_a_v4_write_probe ]] || {
  echo "Refusing unexpected V3 probe path" >&2
  exit 1
}

if [[ ! -f data/route_a_v4_mixed_maps/manifests/map_plan.json ]]; then
  conda run --no-capture-output -n yopo \
    python tools/prepare_route_a_v4_maps.py plan
fi
for SPLIT in train valid; do
  if [[ ! -f "data/route_a_v4_mixed_maps/manifests/${SPLIT}_maps.json" ]]; then
    conda run --no-capture-output -n yopo \
      python tools/prepare_route_a_v4_maps.py generate \
      --split "$SPLIT" --workers "$WORKERS"
  fi
done
conda run --no-capture-output -n yopo \
  python tools/prepare_route_a_v4_maps.py status
conda run --no-capture-output -n yopo \
  python tools/prepare_route_a_v4_raw_config.py
conda run --no-capture-output -n yopo \
  python tools/repair_route_a_v4_partial_generation_plan.py --apply

for SPLIT in train valid; do
  ARGS=(
    python -m authoritative_dataset.generate_v1
    --config "$ROOT/configs/route_a_v4_raw_static_resolved.yaml"
    --output "$ROOT/data/route_a_v4_raw_static"
    --split "$SPLIT"
    --workers "$WORKERS"
    --device "$DEVICE"
    --fail-fast
  )
  if [[ "$RESUME" == true || -d data/route_a_v4_raw_static ]]; then
    ARGS+=(--resume)
  fi
  conda run --no-capture-output -n yopo "${ARGS[@]}"
done

if [[ ! -d data/route_a_v4_static_yopo ]]; then
  conda run --no-capture-output -n yopo \
    python tools/build_route_a_v4_static_dataset.py --authorize-build
fi
conda run --no-capture-output -n yopo \
  python tools/repair_route_a_v4_split_contract.py --apply
conda run --no-capture-output -n yopo \
  python tools/validate_route_a_v4_dataset.py
if [[ ! -f configs/route_a_v4_static_yopo_training.yaml ]]; then
  conda run --no-capture-output -n yopo \
    python tools/freeze_route_a_v4_training_config.py
fi
conda run --no-capture-output -n yopo \
  python tools/train_mixed_static_yopo_v1.py \
  --config configs/route_a_v4_static_yopo_training.yaml --verify-only

echo "Route-A V4 dataset and frozen training config are ready."
echo "No training was started."
