#!/usr/bin/env python3
"""Resolve a static-only authoritative generation config for Route-A V4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase8jqv2_4m1_mixed_maps import source_hash


MAP_ROOT = ROOT / "data/route_a_v4_mixed_maps"
OUTPUT_ROOT = ROOT / "data/route_a_v4_raw_static"
TARGET = ROOT / "configs/route_a_v4_raw_static_resolved.yaml"
BASE = ROOT / "configs/phase8_authoritative_v3_mixed_generation.yaml"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifests = {}
    maps = {}
    for split in ("train", "valid"):
        path = MAP_ROOT / f"manifests/{split}_maps.json"
        document = json.loads(path.read_text())
        if document["status"] != "PASS":
            raise RuntimeError(f"V4 {split} map manifest is not complete")
        manifests[split] = sha(path)
        maps[split] = [{
            key: row[key] for key in (
                "map_id", "map_uuid", "seed", "maze_type",
                "semantic_name", "profile_name", "profile_hash",
            )
        } for row in document["maps"]]
    value = yaml.safe_load(BASE.read_text())
    value["dataset_version"] = "route_a_v4_raw_static_v1"
    value["output_root"] = str(OUTPUT_ROOT)
    value["authority_source_dataset"] = "route_a_v4_mixed_maps"
    value["authority_source_root"] = str(MAP_ROOT)
    value["authority_source_root_manifest_hash"] = hashlib.sha256(
        (manifests["train"] + manifests["valid"]).encode()
    ).hexdigest()
    value["frozen_hashes"]["source_hash"] = source_hash()
    value["frozen_hashes"]["split_manifest_hash"] = hashlib.sha256(
        json.dumps(
            maps, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    value["formal_splits"]["train"]["frames"] = 300000
    value["formal_splits"]["train"]["maps"] = maps["train"]
    value["formal_splits"]["valid"]["frames"] = 60000
    value["formal_splits"]["valid"]["maps"] = maps["valid"]
    value["dynamic_scenarios"] = []
    value["actor_distribution"] = {
        "maximum_actors": 0,
        "policy": "static_policy_training_actor_free",
    }
    # V4 changes the network-observation authority, but the raw authoritative
    # generator still consumes physical timing fields from the inherited
    # camera-state contract.  Extend that contract instead of replacing it.
    state_sampling_contract = dict(value["state_sampling_contract"])
    state_sampling_contract.update({
        "semantics_version": "route_a_v4_camera_pose_sampling_v1",
        "frame_state_source": "continuous_reference_trajectory",
        "network_observation_source": "derived_route_a_wide_state_v1",
        "network_speed_limit_mps": 6.0,
        "network_acceleration_limit_mps2": 6.0,
        "goal_distance_m": [10.0, 40.0],
        "raw_observation_is_training_authority": False,
    })
    required = {
        "command_latency_s",
        "network_speed_limit_mps",
        "network_acceleration_limit_mps2",
    }
    missing = sorted(required - set(state_sampling_contract))
    if missing:
        raise RuntimeError(
            f"resolved V4 state sampling contract is incomplete: {missing}"
        )
    value["state_sampling_contract"] = state_sampling_contract
    TARGET.write_text(yaml.safe_dump(value, sort_keys=False))
    print(json.dumps({
        "status": "PASS",
        "config": str(TARGET),
        "config_sha256": sha(TARGET),
        "output": str(OUTPUT_ROOT),
        "frames": {"train": 300000, "valid": 60000},
        "static_only": True,
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
