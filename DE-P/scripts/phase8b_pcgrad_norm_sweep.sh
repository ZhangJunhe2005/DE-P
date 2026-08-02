#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P
for ratio in 0.25 0.5 1.0; do
  conda run --no-capture-output -n yopo python tools/run_dynamic_objective_ablation.py \
    --config configs/phase8b_gradient_diagnostics.yaml --only F_full --steps 40 \
    --reference-scale 1.0 --cvar-coefficient 0.0 --dynamic-weight 0.1 \
    --gradient-strategy dynamic_priority_pcgrad --pcgrad-other-norm-ratio "$ratio" \
    --output "reports/phase8b_gradient_diagnostics/pcgrad_norm_ratio_${ratio}.json"
done
