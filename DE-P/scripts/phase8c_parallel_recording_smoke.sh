#!/usr/bin/env bash
set -euo pipefail
source /opt/ros/noetic/setup.bash
source /home/zjh/YOPO/Simulator/devel/setup.bash
cd /home/zjh/YOPO/DE-P
conda run --no-capture-output -n yopo python tools/run_phase8c_formal_recording.py \
  --output /tmp/phase8c_parallel_recording_smoke \
  --static-catalog /home/zjh/YOPO/DE-P/data/phase8c_static_maps/map_catalog.yaml \
  --workers 8 \
  --ros-master-port-base 12200 \
  --sequence-id phase8c_train_0027 \
  --sequence-id phase8c_train_0028 \
  --sequence-id phase8c_train_0029 \
  --sequence-id phase8c_train_0030 \
  --sequence-id phase8c_train_0031 \
  --sequence-id phase8c_train_0032 \
  --sequence-id phase8c_train_0033 \
  --sequence-id phase8c_train_0035
