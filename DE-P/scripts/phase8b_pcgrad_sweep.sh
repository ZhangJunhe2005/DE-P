#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P
for weight in 0.1 0.3 0.6; do
  conda run --no-capture-output -n yopo python tools/run_dynamic_objective_ablation.py \
    --config configs/phase8b_gradient_diagnostics.yaml --only F_full --steps 40 \
    --reference-scale 1.0 --cvar-coefficient 0.0 --dynamic-weight "$weight" \
    --gradient-strategy dynamic_priority_pcgrad \
    --output "reports/phase8b_gradient_diagnostics/pcgrad_weight_${weight}.json"
done
