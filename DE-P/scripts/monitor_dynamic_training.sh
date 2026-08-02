#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 RUN_DIR" >&2; exit 2; }
run_dir=$(realpath "$1")
conda run --no-capture-output -n yopo python /home/zjh/YOPO/DE-P/tools/training_watchdog.py "$run_dir" || true
tail -n 30 "$run_dir/train.log" 2>/dev/null || true
nvidia-smi || true
free -h
ls -lh "$run_dir"/checkpoints/*.pt 2>/dev/null || true
echo "TensorBoard: tensorboard --logdir $run_dir/tensorboard --port 6006"
