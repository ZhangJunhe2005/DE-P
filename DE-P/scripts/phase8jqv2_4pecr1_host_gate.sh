#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

conda run --no-capture-output -n yopo python - <<'PY'
import torch
assert torch.cuda.is_available(), "host CUDA unavailable"
print({
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
})
PY

conda run --no-capture-output -n yopo \
  python tools/run_phase8jqv2_4pecr1_review.py --stage verify
conda run --no-capture-output -n yopo \
  python tools/run_phase8jqv2_4pecr1_host_runtime.py --device 0
conda run --no-capture-output -n yopo \
  python -m unittest tests.test_phase8jqv2_4pecr1 -v
conda run --no-capture-output -n yopo python -m compileall -q \
  policy/dynamic/provisional_evidence_authorizer_v1.py \
  policy/dynamic/provisional_support_birth_contract_v1.py \
  policy/dynamic/provisional_outcome_mapper_v3.py \
  tools/run_phase8jqv2_4pecr1_review.py \
  tools/run_phase8jqv2_4pecr1_host_runtime.py \
  tests/test_phase8jqv2_4pecr1.py
git diff --check -- \
  configs/provisional_evidence_contract_v1_candidate.yaml \
  policy/dynamic/provisional_evidence_authorizer_v1.py \
  policy/dynamic/provisional_support_birth_contract_v1.py \
  policy/dynamic/provisional_outcome_mapper_v3.py \
  tools/run_phase8jqv2_4pecr1_review.py \
  tools/run_phase8jqv2_4pecr1_host_runtime.py \
  scripts/phase8jqv2_4pecr1_host_gate.sh \
  tests/test_phase8jqv2_4pecr1.py
