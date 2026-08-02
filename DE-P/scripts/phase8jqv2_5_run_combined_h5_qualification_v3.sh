#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
if [[ "${1:-}" != "--dry-run" || "$#" -ne 1 ]]; then
  echo "SMGSS-TR1 permits only: $0 --dry-run" >&2
  exit 2
fi

cd "$ROOT"
exec conda run --no-capture-output -n yopo \
  python tools/run_phase8jqv2_5smgsstr1_combined_h5_dryrun.py --dry-run
