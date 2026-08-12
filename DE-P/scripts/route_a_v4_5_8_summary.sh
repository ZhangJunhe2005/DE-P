#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P
exec conda run --no-capture-output -n yopo \
  python tools/summarize_route_a_v4_5_8_training.py
