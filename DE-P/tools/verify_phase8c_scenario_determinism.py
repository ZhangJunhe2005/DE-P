#!/usr/bin/env python3
"""Generate the formal scenario matrix twice and compare every YAML byte."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def hashes(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*.yaml"))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="phase8c-scenario-determinism-") as temporary:
        root = Path(temporary)
        for name in ("first", "second"):
            subprocess.run([
                sys.executable, str(ROOT / "tools/generate_phase8c_scenario_matrix.py"),
                "--catalog", str(args.catalog), "--output", str(root / name),
            ], check=True, capture_output=True)
        first, second = hashes(root / "first"), hashes(root / "second")
    failures = [name for name in sorted(set(first) | set(second))
                if first.get(name) != second.get(name)]
    payload = {
        "status": "PASS" if not failures else "FAIL", "scenario_yaml_count": len(first),
        "all_scenario_yaml_sha256_identical": not failures,
        "mismatched_files": failures, "hashes": first,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "hashes"}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
