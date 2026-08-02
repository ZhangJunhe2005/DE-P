#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P
conda run --no-capture-output -n yopo python tests/run_phase8b_map_generator_validation.py
conda run --no-capture-output -n yopo python tests/run_host_phase8b_preflight.py
