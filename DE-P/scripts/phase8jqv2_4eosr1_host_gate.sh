#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

conda run --no-capture-output -n yopo \
  python tools/prepare_phase8jqv2_4eosr1.py
conda run --no-capture-output -n yopo \
  python tools/run_phase8jqv2_4eosr1_boundary_solver.py
conda run --no-capture-output -n yopo \
  python tools/validate_phase8jqv2_4eosr1.py
conda run --no-capture-output -n yopo \
  python tools/finalize_phase8jqv2_4eosr1.py
conda run --no-capture-output -n yopo \
  python -m unittest tests.test_phase8jqv2_4eosr1 -v
conda run --no-capture-output -n yopo \
  python -m compileall -q \
    authoritative_dataset/natural_exact_occlusion_solver_v2.py \
    tools/prepare_phase8jqv2_4eosr1.py \
    tools/run_phase8jqv2_4eosr1_solver.py \
    tools/run_phase8jqv2_4eosr1_boundary_solver.py \
    tools/validate_phase8jqv2_4eosr1.py \
    tools/finalize_phase8jqv2_4eosr1.py
git diff --check
