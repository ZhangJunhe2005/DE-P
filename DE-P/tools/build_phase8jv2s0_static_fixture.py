#!/usr/bin/env python3
"""Build the immutable 10,000-window legacy-static validation fixture."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import cfg
from policy.dep_dataset import DEPDataset
from policy.static_v2_fixture import (
    FIXTURE_VERSION, IMPLEMENTATION_VERSION, canonical_hash,
    fixture_semantic_hash, per_index_seed, sample_state_goal,
)


ARTIFACTS = ROOT / "artifacts/phase8jv2s0"
REPORTS = ROOT / "reports"
OUTPUT = ARTIFACTS / "static_valid_fixture_v1.npz"
MANIFEST = ARTIFACTS / "static_valid_fixture_v1_manifest.json"
GLOBAL_SEED = 8172402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    entry = json.loads(
        (REPORTS / "phase8jv2s0_entry_gate.json").read_text()
    )
    if entry["status"] != "PASS":
        raise RuntimeError("S0 entry Gate is not PASS")
    dataset = DEPDataset(
        mode="valid", cache_size=0, global_seed=GLOBAL_SEED
    )
    if len(dataset) != 10000:
        raise RuntimeError(f"expected 10,000 static windows, got {len(dataset)}")
    sampler_fields = {
        name: cfg[name] for name in (
            "vel_max_train", "acc_max_train", "vx_mean_unit",
            "vy_mean_unit", "vz_mean_unit", "vx_std_unit", "vy_std_unit",
            "vz_std_unit", "ax_mean_unit", "ay_mean_unit", "az_mean_unit",
            "ax_std_unit", "ay_std_unit", "az_std_unit", "goal_length",
            "goal_pitch_std", "goal_yaw_std",
        )
    }
    sampler_hash = canonical_hash(sampler_fields)
    catalog = json.loads(json.dumps({}))
    # The catalog is simple YAML; extract the already frozen hashes without
    # loading any PLY into memory.
    from ruamel.yaml import YAML
    catalog = YAML(typ="safe").load(
        ROOT / "configs/static_map_catalog.yaml"
    )
    map_hash = {
        int(row["map_id"]): str(row["static_map_sha256"])
        for row in catalog["maps"]
    }
    data_root = Path(dataset.image_paths[0]).parents[1]
    pose_hash = {
        map_id: sha256(data_root / f"pose-{map_id}.csv")
        for map_id in range(10)
    }

    image_hashes, original_indices, fixture_ids, seeds = [], [], [], []
    position, rotation, velocity_b, acceleration_b, goal_b = [], [], [], [], []
    velocity_w, acceleration_w, goal_w = [], [], []
    for index, (path, map_id, pos, quaternion) in enumerate(zip(
        dataset.image_paths, dataset.map_idx,
        dataset.positions, dataset.quaternions,
    )):
        image_sha = sha256(path)
        original_index = int(Path(path).stem.split("_")[-1])
        seed = per_index_seed(
            GLOBAL_SEED, map_id, original_index, image_sha
        )
        rng = np.random.default_rng(seed)
        vel, acc, goal_level = sample_state_goal(rng)
        q_wxyz = quaternion
        world_from_body = Rotation.from_quat([
            q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]
        ])
        yaw, pitch, roll = world_from_body.as_euler("ZYX")
        world_from_level_body = Rotation.from_euler(
            "ZYX", [0.0, pitch, roll]
        )
        body_goal = world_from_level_body.inv().apply(goal_level)
        fixture_id = canonical_hash({
            "version": FIXTURE_VERSION,
            "map_id": int(map_id),
            "original_dataset_index": original_index,
            "image_sha256": image_sha,
        })
        image_hashes.append(image_sha)
        original_indices.append(original_index)
        fixture_ids.append(fixture_id)
        seeds.append(seed)
        position.append(pos)
        rotation.append(world_from_body.as_matrix())
        velocity_b.append(vel)
        acceleration_b.append(acc)
        goal_b.append(body_goal)
        velocity_w.append(world_from_body.apply(vel))
        acceleration_w.append(world_from_body.apply(acc))
        goal_w.append(pos + world_from_body.apply(body_goal))
        if (index + 1) % 1000 == 0:
            print(f"fixture hashes/states {index + 1}/10000", flush=True)

    arrays = {
        "fixture_version": np.asarray(FIXTURE_VERSION),
        "loader_implementation_version": np.asarray(IMPLEMENTATION_VERSION),
        "fixture_id": np.asarray(fixture_ids, dtype="U64"),
        "original_dataset_index": np.asarray(original_indices, np.int64),
        "image_path": np.asarray(dataset.image_paths, dtype="U512"),
        "image_sha256": np.asarray(image_hashes, dtype="U64"),
        "map_id": np.asarray(dataset.map_idx, np.int64),
        "position_world": np.asarray(position, np.float32),
        "rotation_world_from_body": np.asarray(rotation, np.float32),
        "velocity_body": np.asarray(velocity_b, np.float32),
        "acceleration_body": np.asarray(acceleration_b, np.float32),
        "goal_body": np.asarray(goal_b, np.float32),
        "velocity_world": np.asarray(velocity_w, np.float32),
        "acceleration_world": np.asarray(acceleration_w, np.float32),
        "goal_world": np.asarray(goal_w, np.float32),
        "rng_seed": np.asarray(seeds, np.int64),
        "sampler_config_hash": np.asarray(
            [sampler_hash] * len(dataset), dtype="U64"
        ),
        "pose_csv_hash": np.asarray(
            [pose_hash[int(value)] for value in dataset.map_idx], dtype="U64"
        ),
        "static_map_hash": np.asarray(
            [map_hash[int(value)] for value in dataset.map_idx], dtype="U64"
        ),
    }
    semantic_hash = fixture_semantic_hash(arrays)
    arrays["semantic_hash"] = np.asarray(semantic_hash)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, OUTPUT)
    manifest = {
        "status": "PASS",
        "fixture_version": FIXTURE_VERSION,
        "window_count": len(dataset),
        "global_seed": GLOBAL_SEED,
        "semantic_hash": semantic_hash,
        "fixture_sha256": sha256(OUTPUT),
        "sampler_config_hash": sampler_hash,
        "loader_implementation_version": IMPLEMENTATION_VERSION,
        "loader_implementation_hash":
            sha256(ROOT / "policy/static_v2_fixture.py"),
        "traj_opt_hash": sha256(ROOT / "config/traj_opt.yaml"),
        "static_map_catalog_hash":
            sha256(ROOT / "configs/static_map_catalog.yaml"),
        "pose_csv_hashes": {str(key): value for key, value in pose_hash.items()},
        "static_map_hashes": {str(key): value for key, value in map_hash.items()},
        "depth_or_ply_rebuilt": False,
        "per_index_seed_contract": (
            "sha256(fixture_version,global_seed,map_id,"
            "original_dataset_index,image_sha256)[:63-bit]"
        ),
    }
    atomic_json(MANIFEST, manifest)
    atomic_json(REPORTS / "phase8jv2s0_static_fixture_manifest.json", {
        **manifest,
        "fixture": str(OUTPUT.resolve()),
        "artifact_manifest": str(MANIFEST.resolve()),
    })
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
