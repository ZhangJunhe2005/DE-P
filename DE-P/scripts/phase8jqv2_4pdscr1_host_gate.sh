#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
cd "${ROOT}"

conda run --no-capture-output -n yopo \
  python tools/run_phase8jqv2_4pdscr1_host_runtime.py --device 0
conda run --no-capture-output -n yopo \
  python -m unittest tests.test_phase8jqv2_4pdscr1
