#!/usr/bin/env python3
"""Versioned mathematical controls and immutable-history audit for v2.2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset.occlusion_constructor_v2_2 import (  # noqa: E402
    CONSTRUCTOR_VERSION,
    feasibility_interval,
    legacy_v2_1_detection_lower_bound,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    v21 = ROOT / "authoritative_dataset/occlusion_constructor_v2_1.py"
    v22 = ROOT / "authoritative_dataset/occlusion_constructor_v2_2.py"
    legacy_lower = legacy_v2_1_detection_lower_bound()
    legacy_upper_ratio = 1.8 / (.85 * .355)
    positives = [
        {
            "name": f"connected_patch_gap_{gap}",
            "input": {
                "occluder_distance_m": .355,
                "horizontal_extent_m": .40,
                "vertical_extent_m": .80,
                "gap_frames": gap,
                "speed_mps": 1.40,
            },
        }
        for gap in (1, 2, 3)
    ]
    negatives = [
        {
            "name": "legacy_v2_1_minimum_ratio_detection_bound",
            "expected": "infeasible",
            "observed_center_distance_lower_bound_m": legacy_lower,
            "maximum_detection_center_distance_m": 1.8,
            "passed": legacy_lower > 1.8,
        },
        {
            "name": "single_voxel_patch_gap_3",
            "input": {
                "occluder_distance_m": .355,
                "horizontal_extent_m": .10,
                "vertical_extent_m": .10,
                "gap_frames": 3,
                "speed_mps": 1.40,
            },
        },
        {
            "name": "insufficient_vertical_extent",
            "input": {
                "occluder_distance_m": .355,
                "horizontal_extent_m": .80,
                "vertical_extent_m": .10,
                "gap_frames": 1,
                "speed_mps": 1.40,
            },
        },
    ]
    for row in positives:
        row["result"] = feasibility_interval(**row["input"])
        row["passed"] = row["result"]["feasible"]
    for row in negatives[1:]:
        row["result"] = feasibility_interval(**row["input"])
        row["expected"] = "infeasible"
        row["passed"] = not row["result"]["feasible"]
    report = {
        "status": "PASS" if all(
            row["passed"] for row in positives + negatives
        ) else "FAIL",
        "constructor_version": CONSTRUCTOR_VERSION,
        "implementation_hash": sha256(v22),
        "frozen_v2_1_hash": sha256(v21),
        "contract": {
            "authority_resolution_m": .1,
            "camera_face_clearance_m": .305,
            "voxel_center_distance_m": .355,
            "maximum_detection_center_distance_m": 1.8,
            "tangent_fraction": .85,
        },
        "v2_1_impossibility": {
            "minimum_depth_ratio": 6.4,
            "closest_approach_lower_bound_equation":
                "alpha * r * d = 0.85 * 6.4 * 0.355",
            "closest_approach_lower_bound_m": legacy_lower,
            "detection_limit_m": 1.8,
            "excess_m": legacy_lower - 1.8,
            "maximum_detection_feasible_ratio": legacy_upper_ratio,
            "conclusion": "legacy minimum ratio exceeds detection-feasible maximum",
        },
        "synthetic_positive_controls": positives,
        "synthetic_negative_controls": negatives,
        "compatibility_matrix": [
            {
                "dimension": "motion_contract",
                "v2_1": "authoritative_dynamic_motion_contract_v2_1",
                "v2_2": "same frozen contract",
                "compatible": True,
            },
            {
                "dimension": "output_schema",
                "v2_1": "OcclusionConstruction",
                "v2_2": "OcclusionConstruction",
                "compatible": True,
            },
            {
                "dimension": "authority_input",
                "v2_1": "canonical occupancy",
                "v2_2": "canonical occupancy",
                "compatible": True,
            },
            {
                "dimension": "depth_ratio",
                "v2_1": "fixed 6.4/7.0/7.8/8.5",
                "v2_2": "patch- and detection-derived closed interval",
                "compatible": False,
            },
            {
                "dimension": "patch_model",
                "v2_1": "single/2-voxel nominal width",
                "v2_2": "connected horizontal and vertical extents",
                "compatible": False,
            },
            {
                "dimension": "annex_provenance",
                "v2_1": "none in frozen implementation",
                "v2_2": "none",
                "compatible": True,
            },
        ],
        "formal_generation_started": False,
        "training_executed": False,
        "test_accessed": False,
        "blind_accessed": False,
    }
    atomic_json(
        ROOT / "reports/occlusion_constructor_v2_2_math_validation.json",
        report,
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
