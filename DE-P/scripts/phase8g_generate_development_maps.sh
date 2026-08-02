#!/usr/bin/env bash
set -euo pipefail

root=/home/zjh/YOPO/DE-P
sim=/home/zjh/YOPO/Simulator
generator="$sim/devel/lib/sensor_simulator/dataset_generator"
output="$root/data/phase8g_perception_protocol/static_maps"
[[ ! -e "$output" ]] || { echo "refusing existing output: $output" >&2; exit 2; }

for spec in "development 4 1841000 1842000 1843000" \
            "shadow_validation 2 1851000 1852000 1853000"; do
  read -r split count map_seed pose_seed actor_seed <<<"$spec"
  conda run --no-capture-output -n yopo python "$root/tools/run_safe_map_generation.py" \
    --generator "$generator" --output "$output/$split" \
    --generator-arg=--map-seed --generator-arg="$map_seed" \
    --generator-arg=--pose-seed --generator-arg="$pose_seed" \
    --generator-arg=--actor-seed --generator-arg="$actor_seed" \
    --generator-arg=--env-num --generator-arg="$count" \
    --generator-arg=--image-num --generator-arg=64
done

conda run --no-capture-output -n yopo python \
  "$root/tools/finalize_phase8g_static_maps.py" "$output" \
  --split development 4 1841000 1842000 1843000 \
  --split shadow_validation 2 1851000 1852000 1853000
