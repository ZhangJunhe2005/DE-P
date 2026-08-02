#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P
conda run --no-capture-output -n yopo python tools/run_phase8jqv2_4demdcr1_review.py --stage prepare
conda run --no-capture-output -n yopo python tools/run_phase8jqv2_4demdcr1_host_forward.py
conda run --no-capture-output -n yopo python tools/run_phase8jqv2_4demdcr1_review.py --stage finalize
conda run --no-capture-output -n yopo python -m unittest tests.test_phase8jqv2_4demdcr1
