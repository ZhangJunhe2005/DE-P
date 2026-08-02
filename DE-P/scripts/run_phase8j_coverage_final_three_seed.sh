#!/usr/bin/env bash
set -euo pipefail

cd /home/zjh/YOPO/DE-P

for seed in 8511 8512 8513; do
  conda run --no-capture-output -n yopo \
    python tools/run_phase8j_coverage_training.py \
    --variant C_k3 --seed "${seed}" --epochs 3 --num-workers 4
done
