#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 RUN_DIR [--yes]" >&2; exit 2; }
root=/home/zjh/YOPO/DE-P
run_dir=$(realpath "$1")
[[ "$run_dir" == "$root"/runs/dynamic/* ]] || { echo "run outside managed root" >&2; exit 2; }
checkpoint="$run_dir/checkpoints/latest.pt"
[[ -f "$checkpoint" && -f "$run_dir/config.yaml" ]] || { echo "missing config/latest checkpoint" >&2; exit 2; }
echo "Resume run: $run_dir"
conda run --no-capture-output -n yopo python - <<PY
import torch
p=torch.load("$checkpoint", map_location="cpu", weights_only=True)
m=p.get("metadata", {})
assert m.get("checkpoint_role")=="dynamic_training", "not a dynamic training checkpoint"
assert m.get("backbone_variant")=="corrected", "architecture mismatch"
print("resume epoch:", p["epoch"], "global_step:", p["global_step"])
PY
conda run --no-capture-output -n yopo python "$root/tools/run_managed_dynamic_training.py" \
  --config "$run_dir/config.yaml" --run-dir "$run_dir" --resume "$checkpoint" --yes
