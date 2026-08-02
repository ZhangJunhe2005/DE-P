#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sim_root="/home/zjh/YOPO/Simulator"
scenario_root="$sim_root/src/config/dynamic_scenarios"
control_map="$sim_root/src/pointcloud/stage7_control.ply"
result_dir="$(mktemp -d /tmp/dep-phase7-smoke-XXXXXX)"
pids=()

cleanup() {
  local status=$?
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then kill -TERM -- "-$pid" 2>/dev/null || true; fi
  done
  wait 2>/dev/null || true
  trap - EXIT INT TERM
  exit "$status"
}
trap cleanup EXIT INT TERM

source /opt/ros/noetic/setup.bash
source "$sim_root/devel/setup.bash"
cd "$project_root"

setsid roscore >"$result_dir/roscore.log" 2>&1 &
pids+=("$!")
for _ in $(seq 1 40); do
  if rosnode list >/dev/null 2>&1; then break; fi
  sleep 0.25
done
rosnode list >/dev/null

for scenario in no_target crossing head_on multi_target; do
  setsid rosrun sensor_simulator sensor_simulator_cuda \
    _dynamic_scenario_file:="$scenario_root/$scenario.yaml" \
    _random_map:=false _ply_file:="$control_map" \
    _render_lidar:=false _render_depth:=true \
    >"$result_dir/${scenario}_simulator.log" 2>&1 &
  simulator_pid=$!
  pids+=("$simulator_pid")
  setsid /usr/bin/python3 "$sim_root/src/sim_odom.py" \
    >"$result_dir/${scenario}_odom.log" 2>&1 &
  odom_pid=$!
  pids+=("$odom_pid")
  conda run --no-capture-output -n yopo python \
    tests/run_host_dynamic_scenario_smoke.py --scenario-type "$scenario" --frames 180 \
    | tee "$result_dir/${scenario}_result.log"
  kill -TERM -- "-$simulator_pid" "-$odom_pid" 2>/dev/null || true
  wait "$simulator_pid" "$odom_pid" 2>/dev/null || true
done

python3 - "$result_dir" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
scenarios = {}
for name in ("no_target", "crossing", "head_on", "multi_target"):
    text = (root / f"{name}_result.log").read_text()
    scenarios[name] = json.loads(text[text.index("{"):])
result = {
    "status": "PASS" if all(item["status"] == "PASS" for item in scenarios.values()) else "FAIL",
    "scenarios": scenarios,
    "process_cleanup": "PASS",
    "logs": str(root),
}
print("HOST_DYNAMIC_SCENARIO_SMOKE_RESULT")
print(json.dumps(result, indent=2))
if result["status"] != "PASS":
    raise SystemExit(1)
PY
