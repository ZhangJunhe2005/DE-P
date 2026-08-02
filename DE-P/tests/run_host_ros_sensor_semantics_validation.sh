#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="$(mktemp -d /tmp/dep-phase6-ros-XXXXXX)"
process_ids=()

cleanup() {
  for process_id in "${process_ids[@]:-}"; do
    kill "$process_id" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  rm -rf "$runtime_dir"
}
trap cleanup EXIT INT TERM

source /opt/ros/noetic/setup.bash
source /home/zjh/YOPO/Simulator/devel/setup.bash

roscore >"$runtime_dir/roscore.log" 2>&1 &
process_ids+=("$!")
sleep 2
rosrun sensor_simulator sensor_simulator_cuda >"$runtime_dir/simulator.log" 2>&1 &
process_ids+=("$!")
sleep 4

cd "$project_root"
set +e
conda run --no-capture-output -n yopo python tests/run_host_ros_sensor_semantics_validation.py
validation_status=$?
set -e
if [[ $validation_status -ne 0 ]]; then
  echo "Simulator log tail:" >&2
  tail -n 80 "$runtime_dir/simulator.log" >&2 || true
  exit "$validation_status"
fi
