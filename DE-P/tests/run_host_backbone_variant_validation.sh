#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec python tests/run_host_backbone_variant_validation.py "$@"
