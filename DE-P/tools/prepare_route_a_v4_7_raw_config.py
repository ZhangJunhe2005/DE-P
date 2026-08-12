#!/usr/bin/env python3
"""Resolve the actor-free four-scene Route-A V4.7 raw-data contract."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase8jqv2_4m1_mixed_maps import source_hash


PROFILE_VERSION = "mixed_scene_map_profiles_v4_7_route_a"
MAP_ROOT = ROOT / "data/route_a_v4_7_mixed_maps"
OUTPUT_ROOT = ROOT / "data/route_a_v4_7_raw_static"
TARGET = ROOT / "configs/route_a_v4_7_raw_static_resolved.yaml"
BASE = ROOT / "configs/route_a_v4_2_raw_static_resolved.yaml"
EXPECTED_TYPES = {"cave", "pillar", "forest", "wall"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def load_maps():
    manifests, maps = {}, {}
    expected_counts = {
        "train": Counter({name: 12 for name in EXPECTED_TYPES}),
        "valid": Counter({name: 3 for name in EXPECTED_TYPES}),
    }
    for split in ("train", "valid"):
        path = MAP_ROOT / f"manifests/{split}_maps.json"
        document = json.loads(path.read_text())
        if document.get("status") != "PASS" \
                or document.get("room_included") is not False:
            raise RuntimeError(f"V4.7 {split} map manifest is incomplete")
        if document.get("profile_version") != PROFILE_VERSION:
            raise RuntimeError(f"V4.7 {split} profile version mismatch")
        rows = document.get("maps", [])
        types = {row["map_type"] for row in rows}
        if types != EXPECTED_TYPES or any(int(row["maze_type"]) == 6 for row in rows):
            raise RuntimeError(f"V4.7 {split} map-type contract mismatch")
        counts = Counter(row["map_type"] for row in rows)
        if counts != expected_counts[split]:
            raise RuntimeError(f"V4.7 {split} map balance mismatch: {counts}")
        manifests[split] = sha(path)
        maps[split] = [{
            key: row[key] for key in (
                "map_id", "map_uuid", "seed", "maze_type",
                "semantic_name", "profile_name", "profile_hash",
            )
        } for row in rows]
    if len(maps["train"]) != 48 or len(maps["valid"]) != 12:
        raise RuntimeError("V4.7 map counts must be 48 train / 12 valid")
    return manifests, maps


def main():
    manifests, maps = load_maps()
    value = yaml.safe_load(BASE.read_text())
    value["dataset_version"] = "route_a_v4_7_raw_static_v1"
    value["output_root"] = str(OUTPUT_ROOT)
    value["authority_source_dataset"] = "route_a_v4_7_mixed_maps"
    value["authority_source_root"] = str(MAP_ROOT)
    value["authority_source_root_manifest_hash"] = hashlib.sha256(
        (manifests["train"] + manifests["valid"]).encode()
    ).hexdigest()
    value["frozen_hashes"]["source_hash"] = source_hash()
    value["frozen_hashes"]["split_manifest_hash"] = hashlib.sha256(
        canonical(maps)
    ).hexdigest()
    value["formal_splits"]["train"].update({
        "frames": 300000, "maps": maps["train"],
    })
    value["formal_splits"]["valid"].update({
        "frames": 60000, "maps": maps["valid"],
    })
    value["random_seeds"]["sequence_base"] = 858000000
    value["dynamic_scenarios"] = []
    value["actor_distribution"] = {
        "maximum_actors": 0,
        "policy": "route_a_v4_7_static_actor_free_four_scene",
    }
    value["state_sampling_contract"]["semantics_version"] = (
        "route_a_v4_7_four_scene_camera_pose_sampling_v1"
    )
    spatial = value["scene_spatial_sampling_contract"]
    spatial["required_map_types"] = sorted(EXPECTED_TYPES)
    spatial["map_types"] = {
        name: policy for name, policy in spatial["map_types"].items()
        if name in EXPECTED_TYPES
    }
    spatial["map_types"]["wall"].update({
        "minimum_return_fraction": 0.30,
        "minimum_near_obstacle_fraction": 0.14,
    })
    # Preserve obstacle-bearing views while rejecting sequences whose camera
    # is effectively sealed by a local pillar/wall clump.  These are dataset
    # sampling checks, not runtime trajectory Gates.
    spatial["map_types"]["pillar"].update({
        "maximum_near_obstacle_fraction": 0.75,
        "maximum_frame_near_obstacle_fraction": 0.90,
        "far_depth_m": 6.0,
        "minimum_far_fraction": 0.20,
        "minimum_frame_far_fraction": 0.05,
    })
    spatial["map_types"]["wall"].update({
        "maximum_near_obstacle_fraction": 0.80,
        "maximum_frame_near_obstacle_fraction": 0.92,
        "far_depth_m": 6.0,
        "minimum_far_fraction": 0.15,
        "minimum_frame_far_fraction": 0.05,
    })
    spatial.pop("contract_hash", None)
    spatial["contract_hash"] = hashlib.sha256(canonical(spatial)).hexdigest()
    value["smoke_settings"].update({
        "frames_per_split": 800,
        "frames_per_sequence": 20,
        "map_count": 4,
    })
    encoded = yaml.safe_dump(value, sort_keys=False)
    if TARGET.exists():
        if TARGET.read_text() != encoded:
            raise RuntimeError(
                f"refusing to overwrite existing V4.7 config: {TARGET}"
            )
    else:
        TARGET.write_text(encoded)
    print(json.dumps({
        "status": "PASS",
        "config": str(TARGET),
        "config_sha256": sha(TARGET),
        "output": str(OUTPUT_ROOT),
        "frames": {"train": 300000, "valid": 60000},
        "maps": {"train": 48, "valid": 12},
        "map_types": sorted(EXPECTED_TYPES),
        "room_included": False,
        "pillar_profile": {
            "large_obstacle_number": 100,
            "narrow_obstacle_number": 25,
            "width_m": [0.6, 1.5],
            "distribution_contract": (
                "route_a_v4_7_pillar_distribution_v1"
            ),
            "blocked_view_sequences_rejected": True,
        },
        "wall_profile": {
            "large_wall_number": 100,
            "narrow_wall_number": 25,
            "width_m": [0.5, 6.0],
            "thickness_m": 0.5,
            "ceiling": 1,
        },
        "static_only": True,
        "training_started": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
