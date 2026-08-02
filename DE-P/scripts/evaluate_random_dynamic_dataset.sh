#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 DYNAMIC_CHECKPOINT [OUTPUT_DIR]" >&2; exit 2; }
root=/home/zjh/YOPO/DE-P
output=${2:-"$root/reports/random_dynamic_eval_$(date +%Y%m%d-%H%M%S)"}
conda run --no-capture-output -n yopo python "$root/tools/evaluate_random_dynamic_dataset.py" \
  --dataset "$root/data/phase8_dynamic_production" \
  --static-checkpoint "$root/saved/DEP_corrected_init/epoch10_converted.pth" \
  --dynamic-checkpoint "$1" --output-dir "$output"
