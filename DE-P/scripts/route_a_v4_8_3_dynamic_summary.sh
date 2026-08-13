#!/usr/bin/env bash
set -euo pipefail

cd /home/zjh/YOPO/DE-P
conda run --no-capture-output -n yopo \
  python tools/summarize_route_a_v4_8_3_dynamic_runs.py
