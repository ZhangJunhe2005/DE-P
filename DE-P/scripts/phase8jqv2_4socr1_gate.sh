#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

conda run --no-capture-output -n yopo \
  python tools/run_phase8jqv2_4socr1_review.py
conda run --no-capture-output -n yopo \
  python tools/finalize_phase8jqv2_4socr1.py
conda run --no-capture-output -n yopo \
  python -m unittest tests.test_phase8jqv2_4socr1 -v
conda run --no-capture-output -n yopo \
  python -m compileall -q \
    tools/run_phase8jqv2_4socr1_review.py \
    tools/finalize_phase8jqv2_4socr1.py \
    tests/test_phase8jqv2_4socr1.py
git diff --check
