#!/usr/bin/env python3
"""Fail-closed integrity/readiness validation for Route-A V4 static data."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_yopo_dataset_v1 import StaticYOPODatasetV1
from policy.static_yopo_wide_state_v1 import StaticYOPOWideStateDatasetV1
from tools.build_route_a_v4_static_dataset import validate_split_contract


RAW = ROOT / "data/route_a_v4_raw_static"
DERIVED = ROOT / "data/route_a_v4_static_yopo"


def main():
    errors = []
    for marker in (
        "TRAIN_SPLIT_GENERATION_COMPLETE",
        "VALID_SPLIT_GENERATION_COMPLETE",
        "FULL_GENERATION_COMPLETE",
    ):
        if not (RAW / f"generation_state/completion/{marker}").is_file():
            errors.append(f"missing raw completion marker: {marker}")
    raw_manifest = json.loads((RAW / "manifests/dataset_manifest.json").read_text())
    derived = json.loads((DERIVED / "manifests/dataset_manifest.json").read_text())
    try:
        split_hash = validate_split_contract(DERIVED, derived)
    except (FileNotFoundError, RuntimeError) as error:
        split_hash = None
        errors.append(str(error))
    authority = json.loads((DERIVED / "manifests/map_authority.json").read_text())
    if raw_manifest["dataset_version"] != "route_a_v4_raw_static_v1":
        errors.append("raw dataset version mismatch")
    if derived["status"] != "COMPLETE_FROZEN":
        errors.append("derived dataset is not frozen")
    if derived["actor_input_used"] or derived["composed_depth_used"]:
        errors.append("static policy dataset contains actor-derived input")
    maps = authority["maps"]
    split_counts = Counter(row["source_split"] for row in maps)
    if split_counts != {"train": 48, "valid": 12}:
        errors.append(f"map count mismatch: {dict(split_counts)}")
    large = [row for row in maps if row["size_class"] == "large"]
    if len(large) != 50:
        errors.append(f"large map count mismatch: {len(large)}")
    train_uuid = {row["map_uuid"] for row in maps if row["source_split"] == "train"}
    valid_uuid = {row["map_uuid"] for row in maps if row["source_split"] == "valid"}
    if train_uuid & valid_uuid:
        errors.append("map UUID leakage between train and validation")
    per_type = {
        split: set(values)
        for split, values in derived["map_type_sample_counts"].items()
    }
    required = {"cave", "pillar", "forest", "room", "wall"}
    if per_type.get("train") != required or per_type.get("validation") != required:
        errors.append(f"five-type sample coverage mismatch: {per_type}")
    observations = {}
    samples = {}
    for split, seed in (("train", 82501), ("validation", 82502)):
        base = StaticYOPODatasetV1(DERIVED, split, mmap_cache_size=2)
        view = StaticYOPOWideStateDatasetV1(base, seed)
        indices = np.linspace(0, len(view) - 1, min(512, len(view)), dtype=int)
        values = np.stack([
            view[int(index)]["observation"].numpy() for index in indices
        ])
        speed = np.linalg.norm(values[:, :3], axis=1)
        acceleration = np.linalg.norm(values[:, 3:6], axis=1)
        goal = np.linalg.norm(values[:, 6:9], axis=1)
        observations[split] = {
            "speed_min": float(speed.min()),
            "speed_max": float(speed.max()),
            "acceleration_min": float(acceleration.min()),
            "acceleration_max": float(acceleration.max()),
            "goal_min": float(goal.min()),
            "goal_max": float(goal.max()),
            "finite": bool(np.isfinite(values).all()),
        }
        samples[split] = len(base)
        if (
            not np.isfinite(values).all()
            or speed.max() > 6.0001
            or acceleration.max() > 6.0001
            or goal.min() < 9.999
            or goal.max() > 40.001
        ):
            errors.append(f"wide-state range failure: {split}")
    result = {
        "status": "PASS" if not errors else "FAIL",
        "raw_dataset_version": raw_manifest["dataset_version"],
        "derived_dataset_version": derived["dataset_version"],
        "split_hash": split_hash,
        "samples": samples,
        "map_counts": dict(split_counts),
        "large_maps": len(large),
        "narrow_maps": len(maps) - len(large),
        "map_type_coverage": {
            key: sorted(value) for key, value in per_type.items()
        },
        "observation_probe": observations,
        "actor_input_used": derived["actor_input_used"],
        "composed_depth_used": derived["composed_depth_used"],
        "errors": errors,
        "training_started": False,
    }
    output = ROOT / "reports/route_a_v4_dataset_validation.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
