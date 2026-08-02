#!/usr/bin/env python3
"""Audit persisted formal-valid state/frame semantics without reading images."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT/"data/phase8_authoritative_v1"


def main():
    root_manifest = json.loads(
        (DATASET/"manifests/dataset_manifest.json").read_text()
    )
    counts = Counter()
    by_scenario = defaultdict(Counter)
    maximum = Counter()
    goal_distances = []
    for record in root_manifest["sequences"]:
        sequence_manifest = json.loads((DATASET/record["path"]).read_text())
        if sequence_manifest["split"] != "valid":
            continue
        base = (
            DATASET/sequence_manifest["suite"]/"valid"
            /sequence_manifest["sequence_id"]
        )
        frames = [
            json.loads(row)
            for row in (base/"frames.jsonl").read_text().splitlines()
        ]
        counts["sequences"] += 1
        counts["frames"] += len(frames)
        scenario = sequence_manifest["scenario"]
        positions = np.asarray([row["position_world"] for row in frames])
        velocities = np.asarray([row["velocity_world"] for row in frames])
        accelerations = np.asarray([row["acceleration_world"] for row in frames])
        goal = np.asarray([row["goal_world"] for row in frames])
        body_goal = np.asarray([row["goal_body"] for row in frames])
        quaternion = np.asarray([
            row["quaternion_world_from_body"] for row in frames
        ])
        # Formal generator emits yaw-only wxyz quaternions.
        yaw = 2*np.arctan2(quaternion[:, 3], quaternion[:, 0])
        delta = goal-positions
        expected_body = np.stack((
            np.cos(yaw)*delta[:, 0] + np.sin(yaw)*delta[:, 1],
            -np.sin(yaw)*delta[:, 0] + np.cos(yaw)*delta[:, 1],
            delta[:, 2],
        ), axis=1)
        position_motion = np.linalg.norm(
            positions-positions[0], axis=1
        )
        velocity_norm = np.linalg.norm(velocities, axis=1)
        acceleration_norm = np.linalg.norm(accelerations, axis=1)
        mismatch = np.any(np.abs(body_goal-expected_body) > 1e-6, axis=1)
        recoverable = np.asarray([
            bool(row["actionability"]["recoverable"]) for row in frames
        ])
        stress = np.asarray([
            bool(row["actionability"]["stress"]) for row in frames
        ])
        counts["moving_frames"] += int((position_motion > 1e-9).sum())
        counts["nonzero_velocity_frames"] += int((velocity_norm > 1e-9).sum())
        counts["nonzero_acceleration_frames"] += int(
            (acceleration_norm > 1e-9).sum()
        )
        counts["goal_body_frame_mismatch"] += int(mismatch.sum())
        counts["recoverable_frames"] += int(recoverable.sum())
        counts["stress_frames"] += int(stress.sum())
        by_scenario[scenario]["frames"] += len(frames)
        by_scenario[scenario]["moving_frames"] += int(
            (position_motion > 1e-9).sum()
        )
        by_scenario[scenario]["nonzero_velocity_frames"] += int(
            (velocity_norm > 1e-9).sum()
        )
        by_scenario[scenario]["nonzero_acceleration_frames"] += int(
            (acceleration_norm > 1e-9).sum()
        )
        goal_distances.extend(np.linalg.norm(delta, axis=1).tolist())
        maximum["goal_body_abs_error_nano"] = max(
            maximum["goal_body_abs_error_nano"],
            int(round(float(np.max(np.abs(body_goal-expected_body)))*1e9)),
        )
    failures = []
    if counts["frames"] != 100_000:
        failures.append("formal valid frame count is not 100,000")
    if counts["moving_frames"] == 0:
        failures.append("all formal valid camera positions are constant within sequence")
    if counts["nonzero_velocity_frames"] == 0:
        failures.append("all formal valid UAV velocities are zero")
    if counts["nonzero_acceleration_frames"] == 0:
        failures.append("all formal valid UAV accelerations are zero")
    if counts["goal_body_frame_mismatch"]:
        failures.append(
            "goal_body is not R_body_from_world @ (goal_world-position_world)"
        )
    scenario_motion = {
        name: dict(value) for name, value in sorted(by_scenario.items())
    }
    result = {
        "status": "FAIL" if failures else "PASS",
        "audit": "phase8jqv2_5_formal_valid_semantic_contract",
        "counts": dict(counts),
        "goal_distance_m": {
            "minimum": float(np.min(goal_distances)),
            "maximum": float(np.max(goal_distances)),
            "mean": float(np.mean(goal_distances)),
        },
        "goal_body_max_abs_error_m":
            maximum["goal_body_abs_error_nano"]/1e9,
        "scenario_motion": scenario_motion,
        "contract_failures": failures,
        "source_findings": {
            "generate_v1_positions":
                "np.repeat(base[None,:], frame_count, axis=0)",
            "generate_v1_velocity_world": "[0,0,0]",
            "generate_v1_acceleration_world": "[0,0,0]",
            "generate_v1_goal_body": "(goal-base).tolist() without body rotation",
        },
        "dataset_modified": False,
        "production_test_used": False,
        "blind_used": False,
    }
    output = ROOT/"reports/phase8jqv2_5_dataset_semantic_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
