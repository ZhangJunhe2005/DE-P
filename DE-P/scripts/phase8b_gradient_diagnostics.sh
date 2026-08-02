#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P

test -f diagnostics/fixed_hard_risk.json
test -f diagnostics/fixed_no_target.json
conda run --no-capture-output -n yopo python tools/diagnose_dynamic_candidate_risk.py \
  --config configs/phase8b_gradient_diagnostics.yaml
conda run --no-capture-output -n yopo python tools/diagnose_loss_gradients.py \
  --config configs/phase8b_gradient_diagnostics.yaml
conda run --no-capture-output -n yopo python tools/run_dynamic_objective_ablation.py \
  --config configs/phase8b_gradient_diagnostics.yaml
