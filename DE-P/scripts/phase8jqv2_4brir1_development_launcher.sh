#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-SHADOW}"

case "${MODE}" in
  SHADOW|DEVELOPMENT_ACTIVE) ;;
  *)
    echo "development launcher mode must be SHADOW or DEVELOPMENT_ACTIVE" >&2
    exit 2
    ;;
esac

cd "${ROOT_DIR}"
exec conda run --no-capture-output -n yopo \
  python tools/run_phase8jqv2_4brir1_evaluation.py \
  --stage development --mode "${MODE}"
