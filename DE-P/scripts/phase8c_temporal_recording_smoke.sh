#!/usr/bin/env bash
set -euo pipefail
source /opt/ros/noetic/setup.bash
source /home/zjh/YOPO/Simulator/devel/setup.bash
cd /home/zjh/YOPO/DE-P
conda run --no-capture-output -n yopo python tools/run_phase8c_formal_recording.py \
  --output /tmp/phase8c_temporal_recording_smoke_v3 \
  --static-catalog /home/zjh/YOPO/DE-P/data/phase8c_static_maps/map_catalog.yaml \
  --scenario-type temporal_separation \
  --limit 2
