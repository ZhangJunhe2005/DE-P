#!/usr/bin/env bash
set -euo pipefail

root=/home/zjh/YOPO/DE-P
sim=/home/zjh/YOPO/Simulator
generator="$sim/devel/lib/sensor_simulator/dataset_generator"
output="$root/data/phase8c_static_maps"
determinism="$root/reports/phase8c_map_determinism"
[[ ! -e "$output" ]] || { echo "refusing existing output: $output" >&2; exit 2; }
[[ ! -e "$determinism" ]] || { echo "refusing existing output: $determinism" >&2; exit 2; }

mkdir -p "$determinism"
for spec in "train 810000 820000 830000 12" "valid 820000 920000 930000 3" "test 830000 1020000 1030000 3"; do
  read -r split map_seed pose_seed actor_seed env_num <<<"$spec"
  for copy in a b; do
    conda run --no-capture-output -n yopo python "$root/tools/run_safe_map_generation.py" \
      --generator "$generator" --output "$determinism/${split}_${copy}" \
      --generator-arg=--map-seed --generator-arg="$map_seed" \
      --generator-arg=--pose-seed --generator-arg="$pose_seed" \
      --generator-arg=--actor-seed --generator-arg="$actor_seed" \
      --generator-arg=--env-num --generator-arg=1 \
      --generator-arg=--image-num --generator-arg=64
  done
  conda run --no-capture-output -n yopo python "$root/tools/verify_phase8c_map_determinism.py" \
    "$determinism/${split}_a" "$determinism/${split}_b" \
    >"$determinism/${split}_comparison.json"
  conda run --no-capture-output -n yopo python "$root/tools/run_safe_map_generation.py" \
    --generator "$generator" --output "$output/$split" \
    --generator-arg=--map-seed --generator-arg="$map_seed" \
    --generator-arg=--pose-seed --generator-arg="$pose_seed" \
    --generator-arg=--actor-seed --generator-arg="$actor_seed" \
    --generator-arg=--env-num --generator-arg="$env_num" \
    --generator-arg=--image-num --generator-arg=64
done

conda run --no-capture-output -n yopo python "$root/tools/finalize_phase8c_static_maps.py" "$output"
echo "PHASE8C_STATIC_MAPS=$output"
