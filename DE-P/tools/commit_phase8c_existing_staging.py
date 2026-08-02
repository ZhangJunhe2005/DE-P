#!/usr/bin/env python3
"""Validate and atomically commit a fully recorded Phase 8C staging directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-catalog", type=Path, required=True)
    args = parser.parse_args()
    staging = args.staging.expanduser().resolve()
    output = args.output.expanduser().resolve()
    expected_prefix = f".{output.name}.staging-"
    if not staging.is_dir():
        raise FileNotFoundError(staging)
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    if staging.parent != output.parent or not staging.name.startswith(expected_prefix):
        raise ValueError("staging/output identity check failed")
    matrix = staging / "scenario_configs/matrix.csv"
    subprocess.run([
        sys.executable, str(ROOT / "tools/finalize_phase8c_dynamic_dataset.py"),
        "--dataset", str(staging), "--matrix", str(matrix),
        "--map-catalog", str(args.map_catalog.expanduser().resolve()),
    ], check=True)
    subprocess.run([
        sys.executable, str(ROOT / "tools/validate_dynamic_dataset.py"), str(staging),
    ], check=True)
    incomplete = staging / "INCOMPLETE"
    if incomplete.exists():
        incomplete.unlink()
    os.replace(staging, output)
    print(json.dumps({
        "status": "PASS", "scope": "validated_existing_staging_commit",
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
