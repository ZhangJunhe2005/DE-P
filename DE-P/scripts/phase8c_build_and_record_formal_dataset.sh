#!/usr/bin/env bash
set -euo pipefail
source /opt/ros/noetic/setup.bash
cd /home/zjh/YOPO/Simulator
catkin_make -DCMAKE_BUILD_TYPE=Release
source /home/zjh/YOPO/Simulator/devel/setup.bash
cd /home/zjh/YOPO/DE-P
bash scripts/phase8c_record_formal_dataset.sh
