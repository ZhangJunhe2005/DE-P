#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zjh/YOPO/DE-P
CONTROLLER=/home/zjh/YOPO/Controller
PORT="${PHASE8JQ_ROS_MASTER_PORT:-12931}"
WORK="$(mktemp -d /tmp/phase8jq-controller.XXXXXX)"
ROSCORE_PID=
LAUNCH_PID=

cleanup() {
  set +e
  if [[ -n "${LAUNCH_PID}" ]]; then kill -INT "${LAUNCH_PID}" 2>/dev/null; fi
  if [[ -n "${ROSCORE_PID}" ]]; then kill -INT "${ROSCORE_PID}" 2>/dev/null; fi
  wait "${LAUNCH_PID}" 2>/dev/null
  wait "${ROSCORE_PID}" 2>/dev/null
}
trap cleanup EXIT INT TERM

source /opt/ros/noetic/setup.bash
source "${CONTROLLER}/devel/setup.bash"
export ROS_MASTER_URI="http://127.0.0.1:${PORT}"
export ROS_HOME="${WORK}/ros-home"
mkdir -p "${ROS_HOME}"

roscore -p "${PORT}" >"${WORK}/roscore.log" 2>&1 &
ROSCORE_PID=$!
for _ in $(seq 1 50); do
  if rosparam list >/dev/null 2>&1; then break; fi
  sleep 0.1
done
roslaunch so3_quadrotor_simulator simulator_attitude_control.launch \
  >"${WORK}/controller.log" 2>&1 &
LAUNCH_PID=$!
sleep 1
if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
  tail -n 100 "${WORK}/controller.log"
  exit 1
fi
conda run --no-capture-output -n yopo \
  python "${ROOT}/tests/run_phase8jq_controller_identification.py"
