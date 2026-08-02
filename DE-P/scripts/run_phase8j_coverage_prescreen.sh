#!/usr/bin/env bash
set -euo pipefail

cd /home/zjh/YOPO/DE-P

for variant in C_k1 C_k2 C_k3 D E; do
  conda run --no-capture-output -n yopo \
    python tools/run_phase8j_coverage_training.py \
    --variant "${variant}" --seed 8511 --epochs 1 --num-workers 4
done
