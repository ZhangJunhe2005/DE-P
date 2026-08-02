#!/usr/bin/env bash
set -euo pipefail
source /opt/ros/noetic/setup.bash
source /home/zjh/YOPO/Simulator/devel/setup.bash
cd /home/zjh/YOPO/DE-P
conda run --no-capture-output -n yopo python tools/finalize_phase8c_static_maps.py \
  /home/zjh/YOPO/DE-P/data/phase8c_static_maps
conda run --no-capture-output -n yopo python tools/run_phase8c_formal_recording.py \
  --output /home/zjh/YOPO/DE-P/data/phase8_dynamic_production \
  --static-catalog /home/zjh/YOPO/DE-P/data/phase8c_static_maps/map_catalog.yaml \
  --workers 8 \
  --ros-master-port-base 12100
