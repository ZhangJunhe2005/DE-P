#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P
conda run --no-capture-output -n yopo python \
  tools/validate_phase8jqv2_4m1_occlusion_host.py
