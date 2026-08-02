#!/usr/bin/env python3
"""Snapshot/verify deterministic Phase 8J-P resume outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "reports/phase8jp_parameterization_comparison.json",
    "reports/phase8jp_o5a_midpoint_position.json",
    "reports/phase8jp_o5b_midpoint_velocity.json",
    "reports/phase8jp_o5c_time_split.json",
    "reports/phase8jp_o5d_brake_yield.json",
    "reports/phase8jp_o6_kinodynamic_oracle.json",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def current():
    comparison = json.loads((ROOT / FILES[0]).read_text())
    return {
        "config_sha256": comparison["oracle_config"]["config_sha256"],
        "completion_count": comparison["completion_count"],
        "report_sha256": {
            path: sha256(ROOT / path) for path in FILES
        },
    }


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--snapshot", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    observed = current()
    if args.snapshot:
        atomic_json(args.snapshot, observed)
        print(json.dumps({"status": "SNAPSHOT", **observed}, indent=2))
        return
    expected = json.loads(args.verify.read_text())
    checks = {
        "same_config": (
            observed["config_sha256"] == expected["config_sha256"]
        ),
        "all_284_completions": (
            observed["completion_count"]
            == expected["completion_count"] == 284
        ),
        "report_sha256_identical": (
            observed["report_sha256"] == expected["report_sha256"]
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "before": expected,
        "after": observed,
        "optimizer_recomputation_count": 0,
    }
    atomic_json(ROOT / "reports/phase8jp_resume_validation.json", report)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("Phase 8J-P resume validation failed")


if __name__ == "__main__":
    main()
