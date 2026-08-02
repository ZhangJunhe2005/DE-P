#!/usr/bin/env bash
set -euo pipefail
root=/home/zjh/YOPO/DE-P
config="$root/configs/train_dynamic_production.yaml"
detach=0
yes=0
for arg in "$@"; do
  case "$arg" in --detach) detach=1;; --yes) yes=1;; *) echo "unknown argument: $arg" >&2; exit 2;; esac
done
python_cmd=(conda run --no-capture-output -n yopo python "$root/tools/run_managed_dynamic_training.py" --config "$config")
(( yes )) && python_cmd+=(--yes)
echo "Command: ${python_cmd[*]}"
echo "Plan: 50 epochs; dataset=/home/zjh/YOPO/DE-P/data/phase8_dynamic_production"
echo "Capacity plan: $root/reports/phase8_dataset_capacity_plan.json"
echo "Run root: $root/runs/dynamic (a unique directory will be created)"
echo "Stop: scripts/stop_dynamic_training.sh <run-directory>"
if (( ! yes )); then
  read -r -p "Start production training? [y/N] " answer
  [[ "$answer" == y || "$answer" == Y ]] || exit 1
  python_cmd+=(--yes)
fi
if (( detach )); then
  nohup "${python_cmd[@]}" >"$root/runs/dynamic/launcher-$(date +%Y%m%d-%H%M%S).log" 2>&1 &
  echo "Detached launcher PID: $!"
else
  "${python_cmd[@]}"
fi
