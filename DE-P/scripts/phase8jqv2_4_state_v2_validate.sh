#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P

conda run --no-capture-output -n yopo python \
  tools/validate_phase8jqv2_4_dataset.py \
  --root data/phase8_authoritative_v2 \
  --config configs/phase8_authoritative_v2_generation.yaml \
  --output reports/phase8jqv2_4_formal_generation_summary.json

conda run --no-capture-output -n yopo python \
  tools/validate_authoritative_state_semantics_v2.py \
  --root data/phase8_authoritative_v2 \
  --expected-version phase8_authoritative_v2 \
  --output reports/phase8jqv2_4_dataset_semantic_validation.json
