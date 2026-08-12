#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P
exec /home/zjh/miniconda3/envs/yopo/bin/python \
  tools/summarize_route_a_v4_5_7_training.py
