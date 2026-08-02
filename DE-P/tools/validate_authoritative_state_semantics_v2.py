#!/usr/bin/env python3
"""Streaming semantic validator for authoritative train/valid dataset V2."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset import (
    FORMAL_DATASET_VERSION, FORMAL_DATASET_VERSION_V2,
)
from authoritative_dataset.generate_v1 import canonical, sha256
from authoritative_dataset.state_semantics_v2 import (
    DEMONSTRATED_ACCELERATION_MPS2, DEMONSTRATED_SPEED_MPS,
    GOAL_TRANSFORM_TOLERANCE_M, yaw_rotation_world_from_body,
)

STRESS_PROFILE_SCENARIO = "near_boundary_recovery_stress"
MINIMUM_STRESS_PROFILE_FRACTION = .15


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def distribution(chunks, histogram_bins=None):
    values = np.concatenate(chunks) if chunks else np.empty(0)
    if not len(values):
        return {"count": 0}
    result = {
        "count": int(len(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "p05": float(np.quantile(values, .05)),
        "p50": float(np.quantile(values, .50)),
        "p95": float(np.quantile(values, .95)),
        "p99": float(np.quantile(values, .99)),
    }
    if histogram_bins is not None:
        counts, edges = np.histogram(values, bins=histogram_bins)
        result["histogram"] = {
            "edges": edges.tolist(), "counts": counts.tolist()
        }
    return result


def validate(root, expected_version, expected_frames=None):
    root = Path(root).expanduser().resolve()
    manifest_path = root/"manifests/dataset_manifest.json"
    root_hash_before = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    failures = []
    if manifest["dataset_version"] != expected_version:
        failures.append("dataset version mismatch")
    if expected_version != FORMAL_DATASET_VERSION_V2:
        failures.append("validator only authorizes V2 state semantics")
    split_counts = {"train": Counter(), "valid": Counter()}
    scenario_counts = {
        "train": defaultdict(Counter), "valid": defaultdict(Counter)
    }
    values = {
        split: defaultdict(list) for split in ("train", "valid")
    }
    authority_hashes = {"train": set(), "valid": set()}
    sequence_count = Counter()
    for record in manifest["sequences"]:
        path = root/record["path"]
        if sha256(path) != record["sha256"]:
            failures.append(f"sequence manifest hash mismatch: {record['path']}")
            continue
        sequence = json.loads(path.read_text())
        split = sequence["split"]
        if split not in split_counts:
            failures.append(f"forbidden split in manifest: {split}")
            continue
        base = root/sequence["suite"]/split/sequence["sequence_id"]
        frames = [
            json.loads(row)
            for row in (base/"frames.jsonl").read_text().splitlines()
        ]
        if len(frames) != sequence["frame_count"]:
            failures.append(f"frame count mismatch: {sequence['sequence_id']}")
            continue
        sequence_count[split] += 1
        count = split_counts[split]
        count["frames"] += len(frames)
        scenario = sequence["scenario"]
        scenario_counts[split][scenario]["frames"] += len(frames)
        position = np.asarray(
            [row["position_world"] for row in frames], dtype=np.float64
        )
        velocity = np.asarray(
            [row["velocity_world"] for row in frames], dtype=np.float64
        )
        acceleration = np.asarray(
            [row["acceleration_world"] for row in frames], dtype=np.float64
        )
        timestamp = np.asarray(
            [row["timestamp_ns"] for row in frames], dtype=np.int64
        )
        if not (
            np.isfinite(position).all()
            and np.isfinite(velocity).all()
            and np.isfinite(acceleration).all()
        ):
            failures.append(f"non-finite state: {sequence['sequence_id']}")
            continue
        dt = np.diff(timestamp)/1e9
        if len(dt) and np.any(dt <= 0):
            count["timestamp_nonmonotonic"] += 1
        if len(dt):
            values[split]["dt"].append(dt)
        median_dt = float(np.median(dt)) if len(dt) else .1
        edge_order = 2 if len(frames) >= 3 else 1
        velocity_from_position = np.gradient(
            position, median_dt, axis=0, edge_order=edge_order
        )
        acceleration_from_velocity = np.gradient(
            velocity, median_dt, axis=0, edge_order=edge_order
        )
        velocity_error = np.linalg.norm(
            velocity_from_position-velocity, axis=1
        )
        acceleration_error = np.linalg.norm(
            acceleration_from_velocity-acceleration, axis=1
        )
        values[split]["velocity_error"].append(velocity_error)
        values[split]["acceleration_error"].append(acceleration_error)
        speed = np.linalg.norm(velocity, axis=1)
        accel = np.linalg.norm(acceleration, axis=1)
        span = float(np.linalg.norm(np.ptp(position, axis=0)))
        values[split]["speed"].append(speed)
        values[split]["acceleration"].append(accel)
        values[split]["position_span"].append(np.asarray([span]))
        movement = np.r_[False, np.linalg.norm(
            np.diff(position, axis=0), axis=1
        ) > 1e-6]
        count["moving_frames"] += int(movement.sum())
        count["nonzero_velocity_frames"] += int((speed > 1e-6).sum())
        count["nonzero_acceleration_frames"] += int((accel > 1e-6).sum())
        count["stationary_sequences"] += int(span <= 1e-6)
        scenario_counts[split][scenario]["sequences"] += 1
        scenario_counts[split][scenario]["moving_frames"] += int(
            movement.sum()
        )
        goals = []
        directions = []
        transform_errors = []
        quaternion_errors = []
        orthogonality_errors = []
        sequence_stress = 0
        sequence_recoverable = 0
        for index, row in enumerate(frames):
            authority_hashes[split].add(row["authority_manifest_hash"])
            quaternion = np.asarray(
                row["quaternion_world_from_body"], dtype=np.float64
            )
            quaternion_errors.append(abs(np.linalg.norm(quaternion)-1.0))
            yaw = 2*np.arctan2(quaternion[3], quaternion[0])
            rotation = yaw_rotation_world_from_body(yaw)
            orthogonality_errors.append(float(np.linalg.norm(
                rotation.T@rotation-np.eye(3), ord="fro"
            )))
            delta = (
                np.asarray(row["goal_world"], dtype=np.float64)
                - np.asarray(row["position_world"], dtype=np.float64)
            )
            body = np.asarray(row["goal_body"], dtype=np.float64)
            error = float(np.linalg.norm(rotation@body-delta))
            transform_errors.append(error)
            distance = float(np.linalg.norm(delta))
            goals.append(distance)
            if distance > 1e-12:
                directions.append(body/distance)
            action = row["actionability"]
            if action["feasibility_unknown"]:
                count["unknown_frames"] += 1
            if action.get("nominal", False):
                count["nominal_frames"] += 1
            if action["stress"]:
                count["stress_frames"] += 1
                sequence_stress += 1
            if action["recoverable"]:
                count["recoverable_frames"] += 1
                sequence_recoverable += 1
            if action.get("unrecoverable", False):
                count["unrecoverable_frames"] += 1
            if (
                action["stress"] and not action["recoverable"]
                and not action.get("unrecoverable", False)
            ):
                count["stress_non_recovery_frames"] += 1
            expected_stress = bool(
                speed[index] > DEMONSTRATED_SPEED_MPS+1e-9
                or accel[index] > DEMONSTRATED_ACCELERATION_MPS2+1e-9
            )
            returns_to_envelope = bool(np.any(
                (speed[index:] <= DEMONSTRATED_SPEED_MPS+1e-9)
                & (
                    accel[index:]
                    <= DEMONSTRATED_ACCELERATION_MPS2+1e-9
                )
            ))
            expected_recoverable = bool(
                action["initially_safe"]
                and expected_stress
                and returns_to_envelope
            )
            if (
                bool(action["stress"]) != expected_stress
                or bool(action["recoverable"]) != expected_recoverable
            ):
                count["label_state_mismatch_frames"] += 1
        scenario_counts[split][scenario]["stress_frames"] += (
            sequence_stress
        )
        scenario_counts[split][scenario]["recoverable_frames"] += (
            sequence_recoverable
        )
        if scenario == STRESS_PROFILE_SCENARIO:
            count["stress_profile_sequences"] += 1
            minimum = int(np.ceil(
                MINIMUM_STRESS_PROFILE_FRACTION*len(frames)
            ))
            if sequence_stress < minimum:
                count["stress_profile_violation_sequences"] += 1
            if sequence_recoverable < minimum:
                count["recovery_profile_violation_sequences"] += 1
        values[split]["goal_distance"].append(np.asarray(goals))
        values[split]["goal_transform_error"].append(
            np.asarray(transform_errors)
        )
        values[split]["quaternion_error"].append(
            np.asarray(quaternion_errors)
        )
        values[split]["orthogonality_error"].append(
            np.asarray(orthogonality_errors)
        )
        if directions:
            directions = np.asarray(directions)
            values[split]["goal_body_yaw"].append(
                np.arctan2(directions[:, 1], directions[:, 0])
            )
            values[split]["goal_body_pitch"].append(np.arctan2(
                directions[:, 2],
                np.linalg.norm(directions[:, :2], axis=1),
            ))
        diagnostic = json.loads(
            (base/"render_diagnostics.json").read_text()
        )
        expected_pose_hash = hashlib.sha256(canonical({
            "position_world": [
                row["position_world"] for row in frames
            ],
            "quaternion_world_from_body": [
                row["quaternion_world_from_body"] for row in frames
            ],
            "timestamp_ns": [row["timestamp_ns"] for row in frames],
        })).hexdigest()
        if diagnostic.get("camera_pose_semantic_hash") != expected_pose_hash:
            count["camera_pose_hash_mismatch"] += 1
    root_hash_after = sha256(manifest_path)
    summaries = {}
    for split in ("train", "valid"):
        count = split_counts[split]
        total = count["frames"]
        sequence_total = sequence_count[split]
        distributions = {
            "position_span_m": distribution(values[split]["position_span"]),
            "velocity_norm_mps": distribution(values[split]["speed"]),
            "acceleration_norm_mps2":
                distribution(values[split]["acceleration"]),
            "goal_distance_m": distribution(
                values[split]["goal_distance"],
                histogram_bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
            ),
            "goal_body_yaw_rad":
                distribution(values[split]["goal_body_yaw"]),
            "goal_body_pitch_rad":
                distribution(values[split]["goal_body_pitch"]),
            "dt_s": distribution(values[split]["dt"]),
            "velocity_from_position_error_mps":
                distribution(values[split]["velocity_error"]),
            "acceleration_from_velocity_error_mps2":
                distribution(values[split]["acceleration_error"]),
            "goal_body_transform_error_m":
                distribution(values[split]["goal_transform_error"]),
            "orientation_normalization_error":
                distribution(values[split]["quaternion_error"]),
            "rotation_orthogonality_error":
                distribution(values[split]["orthogonality_error"]),
        }
        checks = {
            "frame_count": (
                expected_frames is None
                or total == expected_frames[split]
            ),
            "moving_fraction": (
                total > 0 and count["moving_frames"]/total >= .75
            ),
            "nonzero_velocity_fraction": (
                total > 0
                and count["nonzero_velocity_frames"]/total >= .75
            ),
            "nonzero_acceleration_fraction": (
                total > 0
                and count["nonzero_acceleration_frames"]/total >= .50
            ),
            "stationary_sequence_fraction": (
                sequence_total > 0
                and count["stationary_sequences"]/sequence_total <= .10
            ),
            "timestamp_monotonic": count["timestamp_nonmonotonic"] == 0,
            "dt_consistent": (
                distributions["dt_s"].get("maximum", np.inf)
                - distributions["dt_s"].get("minimum", -np.inf) <= 1e-12
            ),
            "velocity_difference_consistent": (
                distributions["velocity_from_position_error_mps"].get(
                    "maximum", np.inf
                ) <= 1e-9
            ),
            "acceleration_difference_consistent": (
                distributions["acceleration_from_velocity_error_mps2"].get(
                    "maximum", np.inf
                ) <= 1e-9
            ),
            "goal_body_transform": (
                distributions["goal_body_transform_error_m"].get(
                    "maximum", np.inf
                ) <= GOAL_TRANSFORM_TOLERANCE_M
            ),
            "orientation_normalized": (
                distributions["orientation_normalization_error"].get(
                    "maximum", np.inf
                ) <= 1e-12
            ),
            "rotation_orthogonal": (
                distributions["rotation_orthogonality_error"].get(
                    "maximum", np.inf
                ) <= 1e-12
            ),
            "goal_not_near_zero": (
                distributions["goal_distance_m"].get("p05", 0) >= 1.0
                and distributions["goal_distance_m"].get("p95", 0) >= 2.0
            ),
            "goal_direction_not_collapsed": (
                distributions["goal_body_yaw_rad"].get("p95", 0)
                - distributions["goal_body_yaw_rad"].get("p05", 0) >= .25
            ),
            "nominal_coverage": count["nominal_frames"] > 0,
            "stress_coverage": (
                count["stress_frames"] > 0
                and count["stress_profile_sequences"] > 0
                and count["stress_profile_violation_sequences"] == 0
            ),
            "recoverable_coverage": (
                count["recoverable_frames"] > 0
                and count["stress_profile_sequences"] > 0
                and count["recovery_profile_violation_sequences"] == 0
            ),
            "label_state_consistent":
                count["label_state_mismatch_frames"] == 0,
            "label_partition": (
                count["nominal_frames"]+count["recoverable_frames"]
                + count["unrecoverable_frames"]
                + count["stress_non_recovery_frames"] == total
            ),
            "unknown_zero": count["unknown_frames"] == 0,
            "camera_pose_matches_persisted_state":
                count["camera_pose_hash_mismatch"] == 0,
        }
        summaries[split] = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "counts": dict(count),
            "sequence_count": sequence_total,
            "checks": checks,
            "distributions": distributions,
            "scenario_counts": {
                name: dict(value)
                for name, value in sorted(scenario_counts[split].items())
            },
            "authority_manifest_hash_count":
                len(authority_hashes[split]),
        }
    if expected_frames is not None and set(summaries) != set(expected_frames):
        failures.append("split set mismatch")
    if root_hash_before != root_hash_after:
        failures.append("validator modified root manifest")
    all_checks = (
        not failures
        and all(value["status"] == "PASS" for value in summaries.values())
    )
    return {
        "status": "PASS" if all_checks else "FAIL",
        "validator_version": "authoritative_state_semantic_validator_v2",
        "dataset_version": manifest["dataset_version"],
        "root_manifest_hash": root_hash_after,
        "root_manifest_unchanged_during_validation":
            root_hash_before == root_hash_after,
        "splits": summaries,
        "failures": failures,
        "threshold_basis": {
            "movement": "at most one hold scenario of 15; 75% leaves margin",
            "nonzero_acceleration": "smooth non-hold profiles; 50% minimum",
            "stress": (
                "every dedicated stress-profile sequence must have at least "
                "15% state-derived outside-demonstrated frames; this avoids "
                "a split-size rounding artifact in a global 1% ratio"
            ),
            "recoverable": (
                "every dedicated stress-profile sequence must have at least "
                "15% initially-safe frames outside the demonstrated 2 m/s "
                "envelope that later return inside"
            ),
            "label_state_consistency":
                "labels are recomputed from persisted kinematics per frame",
            "goal_distance": "YOPO local goal contract is 2-8 m in V2",
            "difference_tolerance": 1e-9,
            "goal_transform_tolerance_m": GOAL_TRANSFORM_TOLERANCE_M,
            "demonstrated_speed_mps": DEMONSTRATED_SPEED_MPS,
            "demonstrated_acceleration_mps2":
                DEMONSTRATED_ACCELERATION_MPS2,
        },
        "dataset_semantic_audit_complete": True,
        "coverage_training_started": False,
        "score_training_started": False,
        "optimizer_step_executed": False,
        "production_test_used": False,
        "blind_used": False,
        "network_weights_modified": False,
        "dataset_modified_by_validator": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--expected-version", default=FORMAL_DATASET_VERSION_V2
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    expected = (
        {"train": 300, "valid": 300}
        if args.smoke else {"train": 1_000_000, "valid": 100_000}
    )
    result = validate(args.root, args.expected_version, expected)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
