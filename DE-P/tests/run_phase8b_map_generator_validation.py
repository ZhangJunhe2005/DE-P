#!/usr/bin/env python3
"""Bounded source/build/fixture Gate for the hardened formal map workflow."""

import json
from pathlib import Path
import subprocess
import sys

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
SIMULATOR = ROOT.parent / "Simulator"


def main():
    source = SIMULATOR / "src/src/dataset_generator.cpp"
    binary = SIMULATOR / "devel/lib/sensor_simulator/dataset_generator"
    text = source.read_text(encoding="utf-8")
    plan = YAML(typ="safe").load(
        (ROOT / "configs/phase8b_production_map_plan.yaml").read_text(encoding="utf-8")
    )
    tests = subprocess.run([
        sys.executable, "-m", "unittest", "tests.test_phase8b_map_generation"
    ], cwd=ROOT, text=True, capture_output=True)
    gates = {
        "explicit_map_pose_actor_seeds": all(token in text for token in ("map_seed", "pose_seed", "actor_seed")),
        "random_device_removed": "std::random_device" not in text,
        "existing_output_refused_by_default": "output exists; pass --overwrite explicitly" in text,
        "overwrite_lists_and_confirms_target": "Existing output target:" in text and "Type OVERWRITE" in text,
        "unique_staging_and_atomic_rename": ".staging-" in text and "fs::rename" in text,
        "compiled_binary_current": binary.is_file() and binary.stat().st_mtime >= source.stat().st_mtime,
        "reachability_and_non_destructive_fixtures": tests.returncode == 0,
        "map_splits_disjoint": not (
            set(plan["map_ranges"]["train"]) & set(plan["map_ranges"]["valid"])
            or set(plan["map_ranges"]["train"]) & set(plan["map_ranges"]["test"])
            or set(plan["map_ranges"]["valid"]) & set(plan["map_ranges"]["test"])
        ),
        "all_six_scenarios_required_per_split": len(plan["required_scenarios_per_split"]) == 6,
        "held_out_test_actor_ranges_declared": bool(plan["held_out_test_actor_ranges"]),
        "automatic_generation_disabled": plan["automatic_generation_allowed"] is False,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    result = {"status": status, "gates": gates, "test_stdout": tests.stdout,
              "test_stderr": tests.stderr,
              "full_dual_dataset_generation_performed": False,
              "production_data_recording_started": False,
              "note": "Determinism is exercised on a bounded reachability fixture; formal map generation remains capacity-gated."}
    output = ROOT / "reports/phase8b_map_generator_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if status == "PASS" else 2)


if __name__ == "__main__":
    main()
