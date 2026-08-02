#!/usr/bin/env bash
set -euo pipefail

cd /home/zjh/YOPO/DE-P

conda run --no-capture-output -n yopo \
  python tools/run_phase8jqv2_4pepcr1_review.py --stage development
conda run --no-capture-output -n yopo \
  python tools/run_phase8jqv2_4pepcr1_review.py --stage fresh
conda run --no-capture-output -n yopo \
  python tools/run_phase8jqv2_4pepcr1_host_runtime.py --device 0
conda run --no-capture-output -n yopo \
  python tools/run_phase8jqv2_4pepcr1_review.py --stage finalize
conda run --no-capture-output -n yopo \
  python -m unittest tests.test_phase8jqv2_4pepcr1
