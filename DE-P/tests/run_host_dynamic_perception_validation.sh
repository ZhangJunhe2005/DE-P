#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec conda run --no-capture-output -n yopo \
  python tests/run_host_dynamic_perception_validation.py "$@"
