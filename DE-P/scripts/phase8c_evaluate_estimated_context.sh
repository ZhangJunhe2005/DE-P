#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P
conda run --no-capture-output -n yopo python tools/evaluate_phase8c_estimated_context.py \
  --dataset data/phase8_dynamic_production \
  --checkpoint saved/DEP_corrected_init/epoch10_converted.pth \
  --config configs/train_dynamic_production_v2.yaml \
  --cache-dir cache/phase8c_estimated_context_quality \
  --output reports/phase8d_estimated_context_quality.json
