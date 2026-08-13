#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
cd "$ROOT"
exec conda run --no-capture-output -n yopo \
  python tools/benchmark_route_a_v4_9_dynamic_safety.py "$@"
