#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P
conda run --no-capture-output -n yopo python tools/mine_phase8c_production_risk_sets.py \
  --config configs/train_dynamic_production_v2.yaml \
  --output-dir diagnostics
