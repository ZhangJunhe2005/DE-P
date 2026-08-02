#!/usr/bin/env bash
set -euo pipefail
source /opt/ros/noetic/setup.bash
cd /home/zjh/YOPO/Simulator
catkin_make --pkg sensor_simulator
cd /home/zjh/YOPO/DE-P
bash scripts/phase8c_generate_static_maps.sh
