#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
YOPO_PYTHON="/home/zjh/miniconda3/envs/yopo/bin/python"

cd "$ROOT"
exec "$YOPO_PYTHON" tools/run_dep_interactive_demo.py "$@"
