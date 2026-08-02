#!/usr/bin/env python3
"""Commit formal split metadata only after all 216 sequences are complete."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML


FRAME_FIELDS = {
    "sequence_id", "frame_index", "timestamp", "depth_path", "pointcloud_path",
    "camera_x", "camera_y", "camera_z", "camera_qx", "camera_qy", "camera_qz",
    "camera_qw", "velocity_x", "velocity_y", "velocity_z", "acceleration_x",
    "acceleration_y", "acceleration_z", "goal_x", "goal_y", "goal_z", "map_id",
    "dynamic_objects_path", "scenario_id", "seed", "depth_odom_offset",
    "depth_camera_info_offset", "depth_gt_offset",
}
OBJECT_FIELDS = {
    "object_id", "position_world", "velocity_world", "position_covariance",
    "radius", "type", "visibility", "occluded", "dynamic", "active",
    "collision", "inside_image",
}


def load_yaml(path):
    return YAML(typ="safe").load(path)


def validate_frame_identity(sequence, row, frame, index):
    """Validate identities using the dataset-wide map ID, not split-local PLY ID."""
    if (int(frame["frame_index"]) != index or
            frame["sequence_id"] != sequence or
            frame["scenario_id"] != sequence or
            int(frame["seed"]) != int(row["actor_seed"]) or
            int(frame["map_id"]) != int(row["map_id"])):
        raise RuntimeError(f"per-frame identity mismatch: {sequence}/{index}")


def merge_sequence_stats(target, values):
    """Sum counters while retaining maximum_actors as a true maximum."""
    maximum_actors = int(values.get("maximum_actors", 0))
    target.update({
        name: value for name, value in values.items() if name != "maximum_actors"
    })
    target["maximum_actors"] = max(target["maximum_actors"], maximum_actors)


def validate_sequence(root, row, metadata):
    sequence = row["sequence_id"]
    directory = root / "sequences" / sequence
    intrinsics = metadata["camera_intrinsics"]
    if (int(metadata["raw_image_width"]), int(metadata["raw_image_height"])) != (160, 90):
        raise RuntimeError(f"unexpected camera geometry: {sequence}")
    if any(not np.isfinite(float(intrinsics[name])) for name in
           ("fx", "fy", "cx", "cy", "min_depth", "max_depth")):
        raise RuntimeError(f"non-finite CameraInfo: {sequence}")
    with (directory / "frames.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != FRAME_FIELDS:
            raise RuntimeError(f"frames.csv schema mismatch: {sequence}")
        frames = list(reader)
    if len(frames) != 60:
        raise RuntimeError(f"expected 60 frames: {sequence}")
    timestamps = np.asarray([float(frame["timestamp"]) for frame in frames])
    if not np.all(np.diff(timestamps) > 0):
        raise RuntimeError(f"non-monotonic timestamps: {sequence}")
    stats = Counter(frames=60)
    actor_histories = defaultdict(list)
    for index, frame in enumerate(frames):
        validate_frame_identity(sequence, row, frame, index)
        if frame["pointcloud_path"]:
            raise RuntimeError(f"pointcloud entered formal data: {sequence}/{index}")
        offsets = [abs(float(frame[name])) for name in
                   ("depth_odom_offset", "depth_camera_info_offset", "depth_gt_offset")]
        if max(offsets) > float(metadata["sync_slop_seconds"]):
            raise RuntimeError(f"synchronization slop exceeded: {sequence}/{index}")
        depth = np.load(directory / frame["depth_path"], allow_pickle=False)
        if depth.shape != (90, 160) or depth.dtype != np.float32 or not np.isfinite(depth).all():
            raise RuntimeError(f"invalid depth: {sequence}/{index}")
        if float(depth.min()) < 0 or float(depth.max()) > float(intrinsics["max_depth"]) + 1e-5:
            raise RuntimeError(f"depth range invalid: {sequence}/{index}")
        objects = json.loads((directory / frame["dynamic_objects_path"]).read_text())
        if row["scenario_type"] == "no_target" and objects:
            raise RuntimeError(f"no-target GT is not empty: {sequence}/{index}")
        stats["actor_instances"] += len(objects)
        stats["visible_frames"] += int(any(float(obj["visibility"]) > 0 for obj in objects))
        stats["occluded_frames"] += int(any(bool(obj["occluded"]) for obj in objects))
        stats["active_invisible_frames"] += int(any(
            bool(obj["active"]) and float(obj["visibility"]) <= 0 for obj in objects
        ))
        stats["collision_frames"] += int(any(bool(obj["collision"]) for obj in objects))
        stats["maximum_actors"] = max(stats["maximum_actors"], len(objects))
        for obj in objects:
            if not OBJECT_FIELDS.issubset(obj):
                raise RuntimeError(f"dynamic object schema mismatch: {sequence}/{index}")
            numeric = np.asarray(obj["position_world"] + obj["velocity_world"], dtype=float)
            if numeric.shape != (6,) or not np.isfinite(numeric).all() or float(obj["radius"]) <= 0:
                raise RuntimeError(f"invalid actor GT: {sequence}/{index}")
            actor_histories[int(obj["object_id"])].append((float(frame["timestamp"]), numeric[:3]))
    kind = row["scenario_type"]
    if kind != "no_target" and not actor_histories:
        raise RuntimeError(f"dynamic scenario has no recorded actor: {sequence}")
    if kind in {"crossing", "head_on", "multi_target", "temporal_separation",
                "occluded_but_tracked"} and stats["visible_frames"] == 0:
        raise RuntimeError(f"dynamic scenario is never visible: {sequence}")
    if kind == "multi_target" and stats["maximum_actors"] < 2:
        raise RuntimeError(f"multi-target lacks simultaneous actors: {sequence}")
    if kind == "temporal_separation" and stats["collision_frames"]:
        raise RuntimeError(f"temporal-separation scenario collided: {sequence}")
    if kind == "occluded_but_tracked" and stats["active_invisible_frames"] == 0:
        raise RuntimeError(f"occluded-tracked lacks invisible active frames: {sequence}")
    if any(len(history) < 2 for history in actor_histories.values()):
        raise RuntimeError(f"actor future GT is not continuous: {sequence}")
    return dict(stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--map-catalog", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset.expanduser().resolve()
    rows = list(csv.DictReader(args.matrix.open(newline="", encoding="utf-8")))
    if len(rows) != 216:
        raise ValueError(f"expected 216 matrix rows, got {len(rows)}")
    splits = defaultdict(list)
    map_splits = defaultdict(set)
    seeds, sequences = set(), set()
    coverage = defaultdict(Counter)
    validation_stats = defaultdict(Counter)
    for row in rows:
        sequence = row["sequence_id"]
        metadata = load_yaml(root / "sequences" / sequence / "metadata.yaml")
        if metadata.get("completion_status") != "complete" or int(metadata["frame_count"]) != 60:
            raise RuntimeError(f"incomplete sequence: {sequence}")
        seed = int(metadata["random_seed"])
        if seed in seeds or sequence in sequences:
            raise RuntimeError("sequence or actor seed collision")
        if seed != int(row["actor_seed"]):
            raise RuntimeError(f"actor seed mismatch: {sequence}")
        if metadata["static_map_sha256"] != row["static_map_sha256"]:
            raise RuntimeError(f"static map hash mismatch: {sequence}")
        if metadata["sensor_source"] != "depth" or metadata["pointcloud_frame"] != "none":
            raise RuntimeError(f"forbidden sensor source: {sequence}")
        sequence_stats = validate_sequence(root, row, metadata)
        merge_sequence_stats(validation_stats[row["scenario_type"]], sequence_stats)
        seeds.add(seed); sequences.add(sequence)
        split = row["split"]
        splits[split].append(sequence)
        map_splits[split].add(int(row["map_id"]))
        coverage[split][row["scenario_type"]] += 1
    if any(map_splits[left] & map_splits[right]
           for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise RuntimeError("map leakage across splits")
    expected = {"train": 144, "valid": 36, "test": 36}
    if {name: len(value) for name, value in splits.items()} != expected:
        raise RuntimeError("formal split count mismatch")
    for split, counts in coverage.items():
        if set(counts) != {
            "no_target", "crossing", "head_on", "multi_target",
            "temporal_separation", "occluded_but_tracked",
        }:
            raise RuntimeError(f"{split}: incomplete scenario coverage")
    split_dir = root / "splits"
    split_dir.mkdir(exist_ok=False)
    for split, values in splits.items():
        (split_dir / f"{split}.txt").write_text(
            "".join(f"{item}\n" for item in values), encoding="utf-8"
        )
    catalog_hash = hashlib.sha256(args.map_catalog.read_bytes()).hexdigest()
    catalog_target = root / "map_catalog.yaml"
    catalog_target.write_bytes(args.map_catalog.read_bytes())
    manifest = {
        "dataset_version": "dep_dynamic_sequence_v1",
        "generation_scope": "phase8c_formal_production",
        "completion_status": "complete",
        "sensor_source": "depth", "formal_pointcloud_training_allowed": False,
        "time_unit": "second", "distance_unit": "meter",
        "map_catalog": "map_catalog.yaml", "map_catalog_sha256": catalog_hash,
        "splits": {split: f"splits/{split}.txt" for split in splits},
        "map_splits": {split: sorted(values) for split, values in map_splits.items()},
        "sequence_count": len(rows), "unique_actor_seed_count": len(seeds),
        "scenario_coverage": {split: dict(sorted(counts.items()))
                              for split, counts in coverage.items()},
        "test_heldout_actor_ranges": True,
        "precommit_full_frame_validation": True,
    }
    yaml = YAML()
    with (root / "dataset_manifest.yaml").open("w", encoding="utf-8") as stream:
        yaml.dump(manifest, stream)
    (root / "precommit_validation.json").write_text(json.dumps({
        "status": "PASS", "sequence_count": len(rows), "frame_count": 216 * 60,
        "scenario_stats": {kind: dict(values) for kind, values in validation_stats.items()},
        "pointcloud_training_allowed": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", **manifest}, indent=2))


if __name__ == "__main__":
    main()
