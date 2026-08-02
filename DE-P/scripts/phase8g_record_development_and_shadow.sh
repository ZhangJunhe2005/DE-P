#!/usr/bin/env bash
set -euo pipefail

root=/home/zjh/YOPO/DE-P
protocol="$root/data/phase8g_perception_protocol"
catalog="$protocol/static_maps/map_catalog.yaml"

conda run --no-capture-output -n yopo python \
  "$root/tools/run_phase8g_instance_recording.py" \
  --output "$protocol/development" --static-catalog "$catalog" \
  --split development --workers 8 --ros-master-port-base 13100

conda run --no-capture-output -n yopo python \
  "$root/tools/run_phase8g_instance_recording.py" \
  --output "$protocol/shadow_validation" --static-catalog "$catalog" \
  --split shadow_validation --workers 8 --ros-master-port-base 13200
