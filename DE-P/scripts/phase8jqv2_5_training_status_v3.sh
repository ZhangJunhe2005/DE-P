#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
RUN_ROOT="$ROOT/runs/phase8_mixed_static_yopo_v3_1_fp32"
RUN_DIR="${1:-}"

if [[ -z "$RUN_DIR" ]]; then
  while read -r _ CANDIDATE; do
    if [[ -f "$CANDIDATE/run_state.json" ]] && \
      grep -q '"long_training_started": true' "$CANDIDATE/run_state.json"; then
      RUN_DIR="$CANDIDATE"
      break
    fi
  done < <(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d \
    -printf '%T@ %p\n' | sort -nr)
fi
[[ -n "$RUN_DIR" && -d "$RUN_DIR" ]] || {
  echo "No training run found." >&2
  exit 2
}

echo "run_dir: $RUN_DIR"
if [[ -f "$RUN_DIR/run_state.json" ]]; then
  echo "run_state:"
  cat "$RUN_DIR/run_state.json"
fi
if [[ -f "$RUN_DIR/metrics.jsonl" ]]; then
  echo "latest_epoch_metrics:"
  tail -n 1 "$RUN_DIR/metrics.jsonl"
  echo "completed_epochs: $(wc -l < "$RUN_DIR/metrics.jsonl")"
fi
CHECKPOINT_COUNT=0
if [[ -d "$RUN_DIR/checkpoints" ]]; then
  CHECKPOINT_COUNT="$(find "$RUN_DIR/checkpoints" -maxdepth 1 \
    -type f -name 'epoch_*.pth' | wc -l)"
fi
echo "epoch_checkpoints: $CHECKPOINT_COUNT"
[[ -f "$RUN_DIR/checkpoints/best.pth" ]] && echo "best_checkpoint: $RUN_DIR/checkpoints/best.pth"
[[ -f "$RUN_DIR/training_complete.json" ]] && {
  echo "training_complete:"
  cat "$RUN_DIR/training_complete.json"
}
[[ -f "$RUN_DIR/failure.json" ]] && {
  echo "failure:"
  cat "$RUN_DIR/failure.json"
}
exit 0
