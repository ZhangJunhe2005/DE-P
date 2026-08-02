#!/usr/bin/env bash
set -euo pipefail

cd /home/zjh/YOPO/DE-P
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/dep-host-gpu-mpl \
conda run --no-capture-output -n yopo python tests/run_host_gpu_validation.py
