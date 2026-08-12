#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
TARGET="${1:-shakedown}"
case "$TARGET" in
  shakedown) RUN_ROOT="$ROOT/runs/route_a_static_yopo_v4_5_2_mixed_shakedown" ;;
  full) RUN_ROOT="$ROOT/runs/route_a_static_yopo_v4_5_2_calibrated" ;;
  *) echo "Usage: $0 [shakedown|full]" >&2; exit 2 ;;
esac
LATEST="$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
if [[ -z "$LATEST" ]]; then echo '{"status":"NOT_STARTED"}'; exit 0; fi
echo "run_dir: $LATEST"
[[ -f "$LATEST/run_state.json" ]] && cat "$LATEST/run_state.json"
if [[ -f "$LATEST/metrics.jsonl" ]]; then
  echo
  echo "latest_epoch_metrics:"
  tail -n 1 "$LATEST/metrics.jsonl"
fi
[[ -f "$LATEST/checkpoints/best.pth" ]] && \
  echo "best_checkpoint: $LATEST/checkpoints/best.pth"
[[ -f "$LATEST/training_complete.json" ]] && {
  echo "training_complete:"
  cat "$LATEST/training_complete.json"
}
