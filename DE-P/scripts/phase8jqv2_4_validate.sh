#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P
ROOT=data/phase8_authoritative_v1
MARKER="$ROOT/generation_state/completion/FULL_GENERATION_COMPLETE"
[[ -f "$MARKER" ]] || { echo "Refusing validation: missing $MARKER" >&2; exit 2; }
conda run --no-capture-output -n yopo python \
  tools/validate_phase8jqv2_4_dataset.py \
  --root "$ROOT" --config configs/phase8_authoritative_v1_generation.yaml
