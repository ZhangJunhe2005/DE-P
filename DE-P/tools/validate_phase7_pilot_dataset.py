#!/usr/bin/env python3
"""Deep acceptance checks for the formal Phase-7 pilot dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import cfg
from loss.dynamic_safety_loss import DynamicCollisionLoss
from loss.dynamic_types import DynamicLossConfig
from loss.trajectory_sampler import QuinticTrajectorySampler
from policy.dynamic_collate import dynamic_sequence_collate
from policy.dynamic_sequence_dataset import DynamicSequenceDataset, validate_dataset_splits
from tests.test_trajectory_sampler_regression import coefficient_map


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return YAML(typ="safe").load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    manifest, splits = validate_dataset_splits(root)
    matrix_rows = list(csv.DictReader((root / "scenario_configs/matrix.csv").open()))
    matrix = {row["sequence_id"]: row for row in matrix_rows}
    failures = []
    scenario_stats = defaultdict(lambda: {
        "sequences": 0, "frames": 0, "visible_frames": 0, "collision_frames": 0,
        "active_invisible_frames": 0, "occluded_frames": 0, "behind_camera_frames": 0,
        "maximum_active_objects": 0, "depth_error_max_m": 0.0,
        "stored_center_depth_error_max_m": 0.0,
        "velocity_error_max_mps": 0.0, "duration_min_seconds": float("inf"),
        "duration_max_seconds": 0.0,
    })
    all_seeds = set()
    static_hashes = set()
    for split, sequence_ids in splits.items():
        types = {matrix[item]["scenario_type"] for item in sequence_ids}
        if types != {"no_target", "crossing", "head_on", "multi_target"}:
            failures.append(f"split {split} lacks scenario coverage: {sorted(types)}")
        for sequence_id in sequence_ids:
            directory = root / "sequences" / sequence_id
            metadata = load_yaml(directory / "metadata.yaml")
            kind = matrix[sequence_id]["scenario_type"]
            stats = scenario_stats[kind]
            stats["sequences"] += 1
            if metadata.get("completion_status") != "complete":
                failures.append(f"{sequence_id}: incomplete")
            if metadata["sensor_source"] != "depth" or "lidar" in metadata["source_topic"]:
                failures.append(f"{sequence_id}: forbidden source")
            if (metadata["raw_image_width"], metadata["raw_image_height"]) != (160, 90):
                failures.append(f"{sequence_id}: wrong raw geometry")
            seed = int(metadata["random_seed"])
            if seed in all_seeds:
                failures.append(f"duplicate seed {seed}")
            all_seeds.add(seed)
            static_hashes.add(metadata["static_map_sha256"])
            with (directory / "frames.csv").open(newline="", encoding="utf-8") as stream:
                frames = list(csv.DictReader(stream))
            if len(frames) != 60:
                failures.append(f"{sequence_id}: expected 60 frames")
            stats["frames"] += len(frames)
            timestamps = np.asarray([float(frame["timestamp"]) for frame in frames])
            if not np.all(np.diff(timestamps) > 0):
                failures.append(f"{sequence_id}: non-monotonic timestamps")
            duration = float(timestamps[-1] - timestamps[0])
            stats["duration_min_seconds"] = min(stats["duration_min_seconds"], duration)
            stats["duration_max_seconds"] = max(stats["duration_max_seconds"], duration)
            ids_seen = set()
            histories = defaultdict(list)
            for index, frame in enumerate(frames):
                if int(frame["frame_index"]) != index:
                    failures.append(f"{sequence_id}: discontinuous frame index")
                if frame["scenario_id"] != sequence_id or int(frame["seed"]) != seed:
                    failures.append(f"{sequence_id}: per-frame scenario/seed mismatch")
                offsets = [abs(float(frame[name])) for name in (
                    "depth_odom_offset", "depth_camera_info_offset", "depth_gt_offset"
                )]
                if max(offsets) > float(metadata["sync_slop_seconds"]):
                    failures.append(f"{sequence_id}: synchronization slop exceeded")
                if frame["pointcloud_path"]:
                    failures.append(f"{sequence_id}: pointcloud path is not empty")
                depth = np.load(directory / frame["depth_path"], allow_pickle=False)
                if depth.shape != (90, 160) or depth.dtype != np.float32 or not np.isfinite(depth).all():
                    failures.append(f"{sequence_id}: invalid depth at {index}")
                objects = json.loads((directory / frame["dynamic_objects_path"]).read_text())
                if kind == "no_target" and objects:
                    failures.append(f"{sequence_id}: no-target frame contains GT")
                stats["maximum_active_objects"] = max(stats["maximum_active_objects"], len(objects))
                stats["visible_frames"] += int(any(obj["visibility"] > 0 for obj in objects))
                stats["collision_frames"] += int(any(obj["collision"] for obj in objects))
                stats["active_invisible_frames"] += sum(obj["visibility"] <= 0 for obj in objects)
                stats["occluded_frames"] += sum(bool(obj["occluded"]) for obj in objects)
                camera_position = np.asarray([
                    frame["camera_x"], frame["camera_y"], frame["camera_z"]
                ], dtype=float)
                rotation_world_from_body = Rotation.from_quat([
                    frame["camera_qx"], frame["camera_qy"],
                    frame["camera_qz"], frame["camera_qw"],
                ]).as_matrix()
                rotation_world_from_camera = rotation_world_from_body @ np.asarray(
                    metadata["camera_rotation_body_from_camera"], dtype=float
                )
                for obj in objects:
                    object_id = int(obj["object_id"])
                    ids_seen.add(object_id)
                    histories[object_id].append((
                        float(frame["timestamp"]), np.asarray(obj["position_world"], float),
                        np.asarray(obj["velocity_world"], float),
                    ))
                    position_camera = rotation_world_from_camera.T @ (
                        np.asarray(obj["position_world"], dtype=float) - camera_position
                    )
                    stats["behind_camera_frames"] += int(
                        position_camera[2] <= 0 and obj["visibility"] <= 0
                    )
                    if obj["visibility"] > 0 and obj["inside_image"]:
                        stats["stored_center_depth_error_max_m"] = max(
                            stats["stored_center_depth_error_max_m"], float(obj["depth_error"])
                        )
                        expected = float(obj["expected_surface_depth"])
                        if expected > 0:
                            u = int(round(float(obj["projected_u"])))
                            v = int(round(float(obj["projected_v"])))
                            pixel_radius = max(2, int(np.ceil(
                                80.0 * float(obj["radius"]) /
                                (expected + float(obj["radius"]))
                            )) + 2)
                            patch = depth[
                                max(0, v - pixel_radius):min(90, v + pixel_radius + 1),
                                max(0, u - pixel_radius):min(160, u + pixel_radius + 1),
                            ]
                            neighborhood_error = float(np.min(np.abs(patch - expected)))
                            stats["depth_error_max_m"] = max(
                                stats["depth_error_max_m"], neighborhood_error
                            )
            for history in histories.values():
                for previous, current in zip(history, history[1:]):
                    t0, p0, v0 = previous
                    t1, p1, v1 = current
                    if np.linalg.norm(v1 - v0) < 0.1:
                        error = float(np.linalg.norm((p1 - p0) / (t1 - t0) - v0))
                        stats["velocity_error_max_mps"] = max(stats["velocity_error_max_mps"], error)
            if kind != "no_target" and not ids_seen:
                failures.append(f"{sequence_id}: dynamic scenario has no active actor")

    expected_counts = {kind: 6 for kind in ("no_target", "crossing", "head_on", "multi_target")}
    actual_counts = {kind: value["sequences"] for kind, value in scenario_stats.items()}
    if actual_counts != expected_counts:
        failures.append(f"scenario counts mismatch: {actual_counts}")
    if len(static_hashes) != 1:
        failures.append("static map hash varies across sequences")
    if scenario_stats["crossing"]["visible_frames"] == 0 or scenario_stats["head_on"]["visible_frames"] == 0:
        failures.append("dynamic actors never appear in depth")
    if scenario_stats["multi_target"]["maximum_active_objects"] < 2:
        failures.append("multi-target never has two simultaneous actors")
    if sum(value["behind_camera_frames"] for value in scenario_stats.values()) == 0:
        failures.append("dataset contains no validated active actor behind the camera")
    if sum(value["occluded_frames"] for value in scenario_stats.values()) == 0:
        failures.append("dataset contains no validated occluded actor frame")
    if max(value["depth_error_max_m"] for value in scenario_stats.values()) > 0.15:
        failures.append("GT projected depth neighborhood error exceeds 0.15 m")
    if max(value["collision_frames"] for kind, value in scenario_stats.items() if kind != "no_target") == 0:
        failures.append("no dynamic scene contains collision-risk frames")

    # Load real windows through the formal Phase-6 path and evaluate a straight
    # candidate trajectory against current GT labels.  Network context uses only
    # the current window; future motion is extrapolated inside the loss.
    datasets = {split: DynamicSequenceDataset(root, split=split) for split in splits}
    duration = float(cfg["sgm_time"])
    sampler = QuinticTrajectorySampler(coefficient_map(duration), duration, 30)
    loss = DynamicCollisionLoss(
        sampler, replace(DynamicLossConfig.from_global_config(), enabled=True)
    )
    loss_stats = {kind: {
        "window_count": 0, "positive_window_count": 0, "maximum": 0.0,
        "minimum_positive": None, "dynamic_obstacle_batch_shapes": set(),
        "dynamic_context_shape": None,
    } for kind in ("no_target", "crossing", "head_on", "multi_target")}
    for dataset in datasets.values():
        for index in range(len(dataset)):
            sample = dataset[index]
            kind = matrix[sample["sequence_id"]]["scenario_type"]
            batch = dynamic_sequence_collate([sample])
            position = sample["position_world"]
            fixed = torch.zeros(15, 3, 3)
            predicted = torch.zeros_like(fixed)
            fixed[:, :, 0] = position
            predicted[:, :, 0] = position + torch.tensor([2.0, 0.0, 0.0])
            cost, _ = loss(fixed, predicted, batch["dynamic_obstacles"])
            value = float(cost.max())
            stats = loss_stats[kind]
            stats["window_count"] += 1
            stats["maximum"] = max(stats["maximum"], value)
            stats["dynamic_obstacle_batch_shapes"].add(
                tuple(batch["dynamic_obstacles"].positions_world.shape)
            )
            stats["dynamic_context_shape"] = list(
                batch["dynamic_context"].attention("backbone_output").shape
            )
            if value > 0:
                stats["positive_window_count"] += 1
                stats["minimum_positive"] = value if stats["minimum_positive"] is None else min(
                    stats["minimum_positive"], value
                )
    for stats in loss_stats.values():
        stats["dynamic_obstacle_batch_shapes"] = [
            list(shape) for shape in sorted(stats["dynamic_obstacle_batch_shapes"])
        ]
    if loss_stats["no_target"]["maximum"] != 0.0:
        failures.append("no-target dynamic loss is not exactly zero")
    for kind in ("crossing", "head_on", "multi_target"):
        if loss_stats[kind]["positive_window_count"] == 0:
            failures.append(f"{kind} contains no positive-risk dynamic loss window")

    result = {
        "status": "PASS" if not failures else "FAIL",
        "root": str(root), "manifest": manifest,
        "sequence_count": len(matrix_rows), "frame_count": sum(v["frames"] for v in scenario_stats.values()),
        "split_counts": {name: len(values) for name, values in splits.items()},
        "split_scene_coverage": {
            name: sorted({matrix[item]["scenario_type"] for item in values})
            for name, values in splits.items()
        },
        "unique_seed_count": len(all_seeds), "static_map_sha256": sorted(static_hashes),
        "scenario_stats": dict(scenario_stats), "offline_dynamic_loss": loss_stats,
        "future_information_in_context": False, "future_gt_used_by_loss": "constant_velocity_extrapolation",
        "official_sensor_source": "depth", "formal_pointcloud_training_allowed": False,
        "failures": failures,
    }
    output = json.dumps(result, indent=2)
    print("PHASE7_PILOT_VALIDATION_RESULT")
    print(output)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
