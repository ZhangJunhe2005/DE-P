#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P
reference_scale=0.1476668417453766
for weight in 0.1 0.3 0.6 1.0; do
  conda run --no-capture-output -n yopo python tools/run_dynamic_objective_ablation.py \
    --config configs/phase8b_gradient_diagnostics.yaml --only F_full --steps 40 \
    --reference-scale "$reference_scale" --cvar-coefficient 0.5 \
    --dynamic-weight "$weight" \
    --output "reports/phase8b_gradient_diagnostics/sweep_weight_${weight}.json"
done
