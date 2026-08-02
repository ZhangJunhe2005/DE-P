#!/usr/bin/env bash
set -euo pipefail

root=/home/zjh/YOPO/DE-P
source /home/zjh/YOPO/Simulator/devel/setup.bash
exec conda run --no-capture-output -n yopo python \
  "$root/tools/phase8g_blind_protocol.py" "$@"
