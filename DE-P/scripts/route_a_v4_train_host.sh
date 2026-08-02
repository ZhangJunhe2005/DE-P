#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
CONFIG="$ROOT/configs/route_a_v4_static_yopo_training.yaml"
RESUME_CHECKPOINT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME_CHECKPOINT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$ROOT"
[[ -f "$CONFIG" ]] || {
  echo "Frozen V4 config missing; finish route_a_v4_generate_dataset_host.sh first." >&2
  exit 1
}
conda run --no-capture-output -n yopo \
  python tools/train_mixed_static_yopo_v1.py \
  --config "$CONFIG" --verify-only

# Establish the original YOPO-derived epoch10 selection baseline on exactly the
# new validation view before consuming GPU-hours on V4 training.
conda run --no-capture-output -n yopo \
  python tools/evaluate_route_a_v4_anti_hover.py \
  --checkpoint "$ROOT/saved/DEP_0/epoch10.pth" \
  --output "$ROOT/reports/route_a_v4_epoch10_anti_hover_baseline.json" \
  --report-only

ARGS=(python tools/train_mixed_static_yopo_v1.py --config "$CONFIG" --authorized)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  ARGS+=(--resume "$RESUME_CHECKPOINT")
fi
conda run --no-capture-output -n yopo "${ARGS[@]}"
