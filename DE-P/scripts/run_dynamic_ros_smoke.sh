#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/noetic/setup.bash
if [[ -f /home/zjh/YOPO/Controller/devel/setup.bash ]]; then
  source /home/zjh/YOPO/Controller/devel/setup.bash
fi
if [[ -f /home/zjh/YOPO/Simulator/devel/setup.bash ]]; then
  source /home/zjh/YOPO/Simulator/devel/setup.bash
fi
cd "$project_root"

mode="${1:---check}"
if [[ "$mode" == "--check" ]]; then
  conda run --no-capture-output -n yopo python -m py_compile test_dep_ros.py policy/dynamic/ros_bridge.py
  conda run --no-capture-output -n yopo python - <<'PY'
from policy.dynamic.types import DynamicPerceptionConfig
import test_dep_ros

config = DynamicPerceptionConfig.from_global_config()
print("ROS_DYNAMIC_SMOKE_CHECK")
print("enabled_default:", config.enabled)
print("source_default:", config.source)
print("depth_topic:", config.depth_topic)
print("pointcloud_topic:", config.pointcloud_topic)
print("camera_info_topic:", config.camera_info_topic)
print("odom_topic:", config.odom_topic)
print("control_period_seconds: 0.02")
print("import_without_ros_master: PASS")
PY
  if rosnode list >/dev/null 2>&1; then
    echo "ROS Master detected; current relevant topics:"
    rostopic list | grep -E 'depth_image|lidar_points|camera_info|sim/odom|pos_cmd' || true
  else
    echo "ROS Master not running: topic inspection skipped as designed."
  fi
  exit 0
fi

if [[ "$mode" == "--run-depth" ]]; then
  echo "Using the existing ROS Master/simulator; this script does not start or stop external processes."
  exec conda run --no-capture-output -n yopo python test_dep_ros.py --dynamic-enabled 1 --dynamic-source depth
fi

if [[ "$mode" == "--run-pointcloud" ]]; then
  echo "PointCloud2 must be in configured optical-camera coordinates; current simulator /lidar_points is rejected because its frame/content disagree."
  exec conda run --no-capture-output -n yopo python test_dep_ros.py --dynamic-enabled 1 --dynamic-source pointcloud
fi

echo "usage: $0 [--check|--run-depth|--run-pointcloud]" >&2
exit 2
