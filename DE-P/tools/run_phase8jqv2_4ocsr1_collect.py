#!/usr/bin/env python3
"""Collect OCSR1 real tracker snapshots and offline GT prediction errors."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics/phase8jqv2_4ocsr1"
CONTROLS = ROOT / "data/phase8_dynamic_perception_controls_v1"
sys.path.insert(0, str(ROOT))

from controller.dynamic_safety_shadow_adapter_v2 import predict_state
from policy.dynamic.types import CameraModel, Pose
from tools.run_phase8jqv2_4tccr1_telemetry import (
    _pose, camera_model, make_perception,
)


SCENARIOS = {
    "crossing", "head_on", "multi_target",
    "temporal_separation", "occluded_but_tracked",
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def split_for(case_id):
    bucket = int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 3
    return "validation" if bucket == 0 else "calibration"


def finite_velocity(positions, timestamps):
    positions = np.asarray(positions, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    velocity = np.zeros_like(positions)
    if len(positions) > 1:
        dt = np.diff(timestamps)
        velocity[1:] = np.diff(positions, axis=0) / dt[:, None, None]
        velocity[0] = velocity[1]
    return velocity


def control_cases():
    manifest = json.loads((CONTROLS / "manifest.json").read_text())
    model = camera_model(manifest["sensor"])
    cases = []
    for control_id in manifest["control_ids"]:
        root = CONTROLS / "controls" / control_id
        control = json.loads((root / "control.json").read_text())
        if control["split"] != "development":
            continue
        runtime = control["runtime_inputs"]
        depths = np.load(root / runtime["depth_file"])
        positions = np.load(root / runtime["camera_positions"])
        yaws = np.load(root / runtime["camera_yaws"])
        timestamps = np.load(root / runtime["timestamps"])
        poses = [
            _pose(position, yaw, timestamp)
            for position, yaw, timestamp in zip(
                positions, yaws, timestamps
            )
        ]
        trajectory = control.get("actor_trajectory")
        if trajectory is None:
            gt = [dict() for _ in timestamps]
        else:
            actor_positions = np.asarray(
                trajectory["positions_world"], dtype=np.float64
            )
            velocities = finite_velocity(actor_positions, timestamps)
            gt = []
            for frame in range(len(timestamps)):
                gt.append({
                    actor: {
                        "position_world":
                            actor_positions[frame, actor].tolist(),
                        "velocity_world":
                            velocities[frame, actor].tolist(),
                        "radius_m":
                            float(trajectory["radii_m"][actor]),
                        "active": True,
                    }
                    for actor in range(actor_positions.shape[1])
                })
        camera_motion = (
            "moving_camera"
            if (
                np.max(np.linalg.norm(positions - positions[0], axis=1)) > 1e-6
                or np.max(np.abs(yaws - yaws[0])) > 1e-6
            ) else "static_camera"
        )
        cases.append({
            "case_id": control_id,
            "category": "physical_controls",
            "scenario": control["semantic_class"],
            "depths": depths,
            "poses": poses,
            "timestamps": np.asarray(timestamps),
            "model": model,
            "gt": gt,
            "camera_motion": camera_motion,
            "negative": trajectory is None,
            "source_root": str(root),
        })
    return cases


def ordinary_case(sequence_id):
    root = ROOT / "data/phase8_dynamic_production/sequences" / sequence_id
    metadata = yaml.safe_load((root / "metadata.yaml").read_text())
    if not sequence_id.startswith("phase8c_train_"):
        raise RuntimeError("OCSR1 only permits existing train development data")
    rotation_body_from_camera = np.asarray(
        metadata["camera_rotation_body_from_camera"], dtype=np.float64
    )
    rows = list(csv.DictReader((root / "frames.csv").open()))
    depths, poses, timestamps, gt, body_rotations = [], [], [], [], []
    camera_positions = []
    for row in rows:
        timestamp = float(row["timestamp"])
        position = np.asarray([
            float(row["camera_x"]), float(row["camera_y"]),
            float(row["camera_z"]),
        ])
        quaternion = np.asarray([
            float(row["camera_qx"]), float(row["camera_qy"]),
            float(row["camera_qz"]), float(row["camera_qw"]),
        ])
        body_rotation = Rotation.from_quat(quaternion).as_matrix()
        poses.append(Pose(
            position, body_rotation @ rotation_body_from_camera, timestamp
        ))
        body_rotations.append(body_rotation)
        camera_positions.append(position)
        depths.append(np.load(root / row["depth_path"]))
        timestamps.append(timestamp)
        objects = json.loads((root / row["dynamic_objects_path"]).read_text())
        gt.append({
            int(item["object_id"]): {
                "position_world": item["position_world"],
                "velocity_world": item["velocity_world"],
                "radius_m": float(item["radius"]),
                "active": bool(item.get("active", True)),
            }
            for item in objects
            if item.get("dynamic", False) and item.get("active", True)
        })
    model = CameraModel(
        width=int(metadata["raw_image_width"]),
        height=int(metadata["raw_image_height"]),
        fx=float(metadata["camera_intrinsics"]["fx"]),
        fy=float(metadata["camera_intrinsics"]["fy"]),
        cx=float(metadata["camera_intrinsics"]["cx"]),
        cy=float(metadata["camera_intrinsics"]["cy"]),
        depth_scale=1.,
        min_depth=float(metadata["camera_intrinsics"]["min_depth"]),
        max_depth=float(metadata["camera_intrinsics"]["max_depth"]),
    )
    camera_positions = np.asarray(camera_positions)
    return {
        "case_id": sequence_id,
        "category": "ordinary_dynamic",
        "scenario": metadata["scenario_type"],
        "depths": np.asarray(depths),
        "poses": poses,
        "body_rotations": body_rotations,
        "timestamps": np.asarray(timestamps),
        "model": model,
        "gt": gt,
        "camera_motion": (
            "moving_camera"
            if np.max(np.linalg.norm(
                camera_positions-camera_positions[0], axis=1
            )) > 1e-6 else "static_camera"
        ),
        "negative": False,
        "source_root": str(root),
    }


def ordinary_cases():
    selected = defaultdict(list)
    for path in sorted((
        ROOT / "data/phase8_dynamic_production/sequences"
    ).glob("phase8c_train_*/metadata.yaml")):
        metadata = yaml.safe_load(path.read_text())
        scenario = metadata["scenario_type"]
        if scenario in SCENARIOS and len(selected[scenario]) < 3:
            selected[scenario].append(path.parent.name)
    return [
        ordinary_case(sequence_id)
        for scenario in sorted(selected)
        for sequence_id in selected[scenario]
    ]


def runtime_track(track, manager, result, frame):
    details = {
        int(row["track_id"]): row
        for row in result.diagnostics["track_manager"]["match_details"]
    }
    match = details.get(track.track_id)
    return {
        "track_id": int(track.track_id),
        "position_world": track.position_world.tolist(),
        "velocity_world": track.velocity_world.tolist(),
        "state_covariance": track.state_covariance.tolist(),
        "age": int(track.age),
        "hit_count": int(track.hit_count),
        "missed_count": int(track.missed_count),
        "is_confirmed": bool(track.is_confirmed),
        "is_dynamic": bool(track.is_dynamic),
        "attention_authorized": bool(track.attention_authorized),
        "confidence": float(track.confidence),
        "dynamic_reason": track.dynamic_reason,
        "birth_frame": int(track.birth_frame),
        "birth_observation_id": int(track.birth_observation_id),
        "last_direct_observation_frame":
            int(track.last_direct_observation_frame),
        "consecutive_direct_hits": int(track.consecutive_direct_hits),
        "prediction_only_age": int(track.prediction_only_age),
        "visibility_state": track.visibility_state,
        "measurement_present":
            int(track.last_direct_observation_frame) == frame,
        "split_merge_suspected": bool(
            match and match["split_merge_suspected"]
        ),
    }


def associate_gt(track_rows, gt):
    if not track_rows or not gt:
        return {}
    gt_ids = sorted(gt)
    track_positions = np.asarray([
        row["position_world"] for row in track_rows
    ])
    gt_positions = np.asarray([
        gt[actor]["position_world"] for actor in gt_ids
    ])
    cost = np.linalg.norm(
        track_positions[:, None, :] - gt_positions[None, :, :], axis=-1
    )
    row_indices, column_indices = linear_sum_assignment(cost)
    return {
        track_rows[row]["track_id"]: gt_ids[column]
        for row, column in zip(row_indices, column_indices)
        if cost[row, column] <= 1.5
    }


def process_case(case):
    sensor = {
        "height": case["model"].height, "width": case["model"].width,
        "intrinsics": [
            case["model"].fx, case["model"].fy,
            case["model"].cx, case["model"].cy,
        ],
        "min_depth_m": case["model"].min_depth,
        "max_depth_m": case["model"].max_depth,
    }
    perception, config = make_perception(sensor)
    frames, errors = [], []
    dynamic_streak = defaultdict(int)
    started = time.perf_counter()
    for frame, timestamp in enumerate(case["timestamps"]):
        result = perception.update_depth(
            np.asarray(case["depths"][frame], dtype=np.float32),
            case["poses"][frame], float(timestamp), case["model"],
        )
        tracks = [
            runtime_track(
                track, perception.track_manager, result, frame
            )
            for track in result.all_tracks
        ]
        direct_dynamic = [
            track for track in tracks
            if (
                track["measurement_present"] and track["is_confirmed"]
                and track["is_dynamic"]
            )
        ]
        assignment = associate_gt(direct_dynamic, case["gt"][frame])
        for track in tracks:
            if (
                track["measurement_present"] and track["is_dynamic"]
                and track["is_confirmed"]
            ):
                dynamic_streak[track["track_id"]] += 1
            elif track["measurement_present"]:
                dynamic_streak[track["track_id"]] = 0
        frames.append({
            "frame": frame,
            "timestamp": float(timestamp),
            "tracks": tracks,
            "offline_gt_assignment": {
                str(key): int(value) for key, value in assignment.items()
            },
            "runtime_gt_used": False,
        })
        for track in direct_dynamic:
            actor_id = assignment.get(track["track_id"])
            if actor_id is None:
                continue
            for horizon in (1, 2, 3):
                future = frame + horizon
                if future >= len(case["timestamps"]):
                    continue
                if actor_id not in case["gt"][future]:
                    continue
                elapsed = float(
                    case["timestamps"][future] - case["timestamps"][frame]
                )
                predicted, covariance = predict_state(
                    track["position_world"], track["velocity_world"],
                    track["state_covariance"], [elapsed],
                    config.process_noise_acceleration,
                )
                truth = case["gt"][future][actor_id]
                error = np.asarray(
                    predicted[0] - truth["position_world"],
                    dtype=np.float64,
                )
                velocity_error = np.asarray(
                    track["velocity_world"], dtype=np.float64
                ) - np.asarray(truth["velocity_world"], dtype=np.float64)
                position_covariance = covariance[0, :3, :3]
                try:
                    mahalanobis_sq = float(
                        error @ np.linalg.solve(position_covariance, error)
                    )
                except np.linalg.LinAlgError:
                    mahalanobis_sq = float("inf")
                errors.append({
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "scenario": case["scenario"],
                    "camera_motion": case["camera_motion"],
                    "single_or_multi_target": (
                        "multi_target"
                        if len(case["gt"][frame]) > 1 else "single_target"
                    ),
                    "track_id": track["track_id"],
                    "actor_id": int(actor_id),
                    "start_frame": frame,
                    "horizon_frames": horizon,
                    "elapsed_s": elapsed,
                    "position_error_vector_m": error.tolist(),
                    "position_error_m": float(np.linalg.norm(error)),
                    "velocity_error_vector_mps": velocity_error.tolist(),
                    "velocity_error_mps":
                        float(np.linalg.norm(velocity_error)),
                    "predicted_covariance": covariance[0].tolist(),
                    "position_std_m": float(np.sqrt(max(
                        np.linalg.eigvalsh(position_covariance).max(), 0.
                    ))),
                    "velocity_std_mps": float(np.sqrt(max(
                        np.linalg.eigvalsh(
                            covariance[0, 3:, 3:]
                        ).max(), 0.
                    ))),
                    "mahalanobis_sq": mahalanobis_sq,
                    "normalized_error":
                        float(np.sqrt(max(mahalanobis_sq, 0.))),
                    "actor_radius_m": float(truth["radius_m"]),
                    "track_age": track["age"],
                    "pre_gap_confidence": track["confidence"],
                    "dynamic_streak":
                        dynamic_streak[track["track_id"]],
                    "runtime_gt_used": False,
                    "offline_gt_used": True,
                })
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "scenario": case["scenario"],
        "camera_motion": case["camera_motion"],
        "negative": case["negative"],
        "source_root": case["source_root"],
        "frames": frames,
        "prediction_errors": errors,
        "runtime_average_ms":
            (time.perf_counter() - started) * 1000 / len(case["timestamps"]),
    }


def summarize(errors):
    result = {}
    for horizon in (1, 2, 3):
        rows = [
            row for row in errors if row["horizon_frames"] == horizon
        ]
        values = np.asarray([
            row["position_error_m"] for row in rows
        ], dtype=np.float64)
        normalized = np.asarray([
            row["normalized_error"] for row in rows
        ], dtype=np.float64)
        position_std = np.asarray([
            row["position_std_m"] for row in rows
        ], dtype=np.float64)
        velocity_std = np.asarray([
            row["velocity_std_mps"] for row in rows
        ], dtype=np.float64)
        if not len(rows):
            result[str(horizon)] = {"sample_count": 0}
            continue
        result[str(horizon)] = {
            "sample_count": len(rows),
            "position_error_p50_m": float(np.percentile(values, 50)),
            "position_error_p90_m": float(np.percentile(values, 90)),
            "position_error_p95_m": float(np.percentile(values, 95)),
            "position_error_p99_m": float(np.percentile(values, 99)),
            "position_error_max_m": float(values.max()),
            "coverage_1sigma": float(np.mean(normalized <= 1.)),
            "coverage_2sigma": float(np.mean(normalized <= 2.)),
            "coverage_3sigma": float(np.mean(normalized <= 3.)),
            "mahalanobis_p50": float(np.percentile(normalized, 50)),
            "mahalanobis_p90": float(np.percentile(normalized, 90)),
            "mahalanobis_p95": float(np.percentile(normalized, 95)),
            "mahalanobis_p99": float(np.percentile(normalized, 99)),
            "maximum_normalized_error": float(normalized.max()),
            "position_std_p99_m": float(np.percentile(position_std, 99)),
            "velocity_std_p99_mps": float(np.percentile(velocity_std, 99)),
            "error_position_std_correlation": float(
                np.corrcoef(values, position_std)[0, 1]
                if len(values) > 1 else 0.
            ),
            "error_velocity_std_correlation": float(
                np.corrcoef(values, velocity_std)[0, 1]
                if len(values) > 1 else 0.
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=("calibration", "validation"), required=True
    )
    arguments = parser.parse_args()
    all_cases = control_cases() + ordinary_cases()
    selected = [
        case for case in all_cases
        if split_for(case["case_id"]) == arguments.split
    ]
    started = time.perf_counter()
    results = []
    for index, case in enumerate(selected, 1):
        result = process_case(case)
        results.append(result)
        print(json.dumps({
            "split": arguments.split,
            "case": case["case_id"],
            "progress": f"{index}/{len(selected)}",
            "prediction_errors": len(result["prediction_errors"]),
        }))
    errors = [
        row for result in results for row in result["prediction_errors"]
    ]
    destination = OUT / f"{arguments.split}_runtime_and_gt.json"
    atomic_json(destination, {
        "split": arguments.split,
        "cases": results,
        "runtime_gt_used": False,
        "offline_gt_used_for_evaluation_only": True,
    })
    summary = {
        "status": "PASS" if errors else "FAIL",
        "split": arguments.split,
        "case_count": len(selected),
        "negative_case_count": sum(case["negative"] for case in selected),
        "prediction_error_count": len(errors),
        "by_horizon": summarize(errors),
        "runtime_average_ms": float(np.mean([
            result["runtime_average_ms"] for result in results
        ])),
        "elapsed_seconds": time.perf_counter() - started,
        "runtime_gt_used": False,
        "holdout_accessed": False,
        "production_test_accessed": False,
        "blind_accessed": False,
        "source": str(destination),
        "source_sha256": hashlib.sha256(
            destination.read_bytes()
        ).hexdigest(),
    }
    atomic_json(
        OUT / f"{arguments.split}_prediction_summary.json", summary
    )
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise RuntimeError("no prediction calibration samples collected")


if __name__ == "__main__":
    main()
