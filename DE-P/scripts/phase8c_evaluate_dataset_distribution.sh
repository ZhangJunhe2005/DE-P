#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P
conda run --no-capture-output -n yopo python tools/evaluate_phase8c_dataset_distribution.py \
  --config configs/train_dynamic_production_v2.yaml \
  --output reports/phase8d_formal_dataset_distribution.json
