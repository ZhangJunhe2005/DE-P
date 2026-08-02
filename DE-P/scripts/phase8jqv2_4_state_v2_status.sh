#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P
conda run --no-capture-output -n yopo python \
  tools/status_phase8jqv2_4_state_v2.py
