#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P
bash scripts/phase8jqv2_4_state_v2_validate.sh
conda run --no-capture-output -n yopo python \
  tools/finalize_phase8jqv2_4_state_v2.py
