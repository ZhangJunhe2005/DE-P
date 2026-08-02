#!/usr/bin/env bash
set -euo pipefail

cd /home/zjh/YOPO/DE-P
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

PYTHON=/home/zjh/miniconda3/envs/yopo/bin/python
"${PYTHON}" -m unittest tests.test_phase8jqv2_4deoacr1
taskset -c 0-7 "${PYTHON}" tools/run_phase8jqv2_4deoacr1.py prepare
