#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
OUTPUT="$ROOT/data/route_a_v4_2_spatial_smoke"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"

cd "$ROOT"
conda run --no-capture-output -n yopo \
  python tools/prepare_route_a_v4_2_raw_config.py
for SPLIT in train valid; do
  conda run --no-capture-output -n yopo \
    python -m authoritative_dataset.generate_v1 \
      --config configs/route_a_v4_2_raw_static_resolved.yaml \
      --output "$OUTPUT" --split "$SPLIT" --smoke \
      --workers "$WORKERS" --device "$DEVICE" --fail-fast --resume
done
