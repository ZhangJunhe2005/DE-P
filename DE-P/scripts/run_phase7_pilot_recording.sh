#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sim_root="/home/zjh/YOPO/Simulator"
output="${1:-$project_root/data/phase7_dynamic_pilot}"
control_map="$sim_root/src/pointcloud/stage7_control.ply"
map_hash="$(sha256sum "$control_map" | awk '{print $1}')"
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

if [[ -e "$output" ]]; then
  echo "refusing to overwrite existing pilot dataset: $output" >&2
  exit 2
fi
mkdir -p "$output"
conda run --no-capture-output -n yopo python tools/generate_phase7_scenario_matrix.py \
  --output "$output/scenario_configs"

source /opt/ros/noetic/setup.bash
source "$sim_root/devel/setup.bash"
setsid roscore >"$output/roscore.log" 2>&1 &
pids+=("$!")
for _ in $(seq 1 40); do
  if rosnode list >/dev/null 2>&1; then break; fi
  sleep 0.25
done
rosnode list >/dev/null

while IFS=, read -r sequence kind seed split scenario_file; do
  echo "recording $sequence ($kind seed=$seed split=$split)"
  setsid rosrun sensor_simulator sensor_simulator_cuda \
    _dynamic_scenario_file:="$scenario_file" _random_map:=false \
    _ply_file:="$control_map" _render_lidar:=false _render_depth:=true \
    >"$output/${sequence}_simulator.log" 2>&1 &
  simulator_pid=$!
  pids+=("$simulator_pid")
  setsid /usr/bin/python3 "$sim_root/src/sim_odom.py" >"$output/${sequence}_odom.log" 2>&1 &
  odom_pid=$!
  pids+=("$odom_pid")
  conda run --no-capture-output -n yopo python tools/record_dynamic_sequences.py \
    --output "$output" --sequence-id "$sequence" --scenario-file "$scenario_file" \
    --scenario-id "$sequence" --scenario-type "$kind" --seed "$seed" \
    --frames 60 --rate 10 --sync-slop 0.002 --timeout 20 \
    --static-map-sha256 "$map_hash"
  kill -TERM -- "-$simulator_pid" "-$odom_pid" 2>/dev/null || true
  wait "$simulator_pid" "$odom_pid" 2>/dev/null || true
done < <(tail -n +2 "$output/scenario_configs/matrix.csv")

conda run --no-capture-output -n yopo python tools/finalize_phase7_pilot_dataset.py \
  --dataset "$output" --matrix "$output/scenario_configs/matrix.csv"
conda run --no-capture-output -n yopo python tools/validate_dynamic_dataset.py "$output"
echo "PHASE7_PILOT_DATASET=$output"
