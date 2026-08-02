#!/usr/bin/env bash
set -euo pipefail
root=/home/zjh/YOPO/DE-P
[[ $# -ge 1 ]] || { echo "usage: $0 DYNAMIC_CHECKPOINT [--execute]" >&2; exit 2; }
checkpoint=$(realpath "$1")
execute=${2:-}
echo "Gate E configuration: $root/configs/eval_gate_e_random_maps.yaml"
echo "Dynamic checkpoint: $checkpoint"
echo "Held-out maps/actor seeds, static-vs-dynamic, 6 categories x 3 seeds"
if [[ "$execute" != "--execute" ]]; then
  echo "Preparation check only. Pass --execute to launch the ROS matrix after production data/training."
  exit 0
fi
[[ -f "$root/data/phase8_dynamic_production/dataset_manifest.yaml" ]] || {
  echo "production held-out dataset is not present" >&2; exit 2;
}
echo "ROS Gate E executor remains locked until a production dynamic checkpoint and dataset exist."
exit 2
