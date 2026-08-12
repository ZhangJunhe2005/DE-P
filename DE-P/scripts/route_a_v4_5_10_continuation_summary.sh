#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/zjh/YOPO/DE-P"
cd "$ROOT"
conda run --no-capture-output -n yopo \
  python tools/summarize_route_a_v4_5_10_continuation.py
