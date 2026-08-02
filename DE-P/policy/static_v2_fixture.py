"""Immutable, index-seeded legacy-static validation fixture.

This module is audit-only.  It does not change DEPDataset's training behavior
or rebuild any depth/point-cloud source data.
"""

from __future__ import annotations

from collections import OrderedDict
import cv2
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from config.config import cfg


FIXTURE_VERSION = "phase8jv2s0_static_valid_fixture_v1"
IMPLEMENTATION_VERSION = "immutable_static_v2_fixture_loader_v1"


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def per_index_seed(global_seed, map_id, original_index, image_sha256):
    digest = canonical_hash({
        "fixture_version": FIXTURE_VERSION,
        "global_seed": int(global_seed),
        "map_id": int(map_id),
        "original_dataset_index": int(original_index),
        "image_sha256": str(image_sha256),
    })
    return int(digest[:16], 16) & ((1 << 63) - 1)


def sample_state_goal(rng):
    vel_max = float(cfg["vel_max_train"])
    acc_max = float(cfg["acc_max_train"])
    v_mean = np.asarray([
        cfg["vx_mean_unit"], cfg["vy_mean_unit"], cfg["vz_mean_unit"]
    ], dtype=float)
    v_std = np.asarray([
        cfg["vx_std_unit"], cfg["vy_std_unit"], cfg["vz_std_unit"]
    ], dtype=float)
    a_mean = np.asarray([
        cfg["ax_mean_unit"], cfg["ay_mean_unit"], cfg["az_mean_unit"]
    ], dtype=float)
    a_std = np.asarray([
        cfg["ax_std_unit"], cfg["ay_std_unit"], cfg["az_std_unit"]
    ], dtype=float)
    while True:
        velocity = vel_max * (v_mean + v_std * rng.standard_normal(3))
        forward = -1.0
        while forward < 0:
            forward = vel_max * rng.lognormal(
                mean=np.log(1 - float(cfg["vx_mean_unit"])),
                sigma=np.log(float(cfg["vx_std_unit"])),
            )
            forward = -forward + 1.2 * vel_max
        velocity[0] = forward
        if np.linalg.norm(velocity) < 1.2 * vel_max:
            break
    while True:
        acceleration = acc_max * (
            a_mean + a_std * rng.standard_normal(3)
        )
        if np.linalg.norm(acceleration) < 1.2 * acc_max:
            break
    pitch = np.radians(rng.normal(0.0, float(cfg["goal_pitch_std"])))
    yaw = np.radians(rng.normal(0.0, float(cfg["goal_yaw_std"])))
    direction = np.asarray([
        np.cos(yaw) * np.cos(pitch),
        np.sin(yaw) * np.cos(pitch),
        np.sin(pitch),
    ])
    near = rng.random()
    if near < 0.1:
        direction = near * 10.0 * direction
    goal_level = float(cfg["goal_length"]) * direction
    return velocity, acceleration, goal_level


def fixture_semantic_hash(arrays) -> str:
    digest = hashlib.sha256()
    ordered = (
        "fixture_id", "original_dataset_index", "image_path",
        "image_sha256", "map_id", "position_world",
        "rotation_world_from_body", "velocity_body", "acceleration_body",
        "goal_body", "velocity_world", "acceleration_world", "goal_world",
        "rng_seed", "sampler_config_hash", "pose_csv_hash",
        "static_map_hash",
    )
    for name in ordered:
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


class StaticV2FixtureDataset(Dataset):
    def __init__(self, fixture_path, cache_size=128):
        self.fixture_path = Path(fixture_path)
        loaded = np.load(self.fixture_path, allow_pickle=False)
        self.arrays = {name: loaded[name] for name in loaded.files}
        self.height = int(cfg["image_height"])
        self.width = int(cfg["image_width"])
        self.cache_size = int(cache_size)
        self._depth_cache = OrderedDict()
        if str(self.arrays["fixture_version"]) != FIXTURE_VERSION:
            raise ValueError("unsupported static fixture version")
        observed = fixture_semantic_hash(self.arrays)
        expected = str(self.arrays["semantic_hash"])
        if observed != expected:
            raise ValueError("static fixture semantic hash mismatch")

    def __len__(self):
        return len(self.arrays["fixture_id"])

    @property
    def semantic_hash(self):
        return str(self.arrays["semantic_hash"])

    def metadata(self, index):
        return {
            name: self.arrays[name][index]
            for name in (
                "fixture_id", "original_dataset_index", "image_path",
                "image_sha256", "map_id", "position_world",
                "rotation_world_from_body", "velocity_body",
                "acceleration_body", "goal_body", "velocity_world",
                "acceleration_world", "goal_world", "rng_seed",
                "sampler_config_hash", "pose_csv_hash", "static_map_hash",
            )
        }

    def _load_depth(self, path):
        key = str(path)
        if key in self._depth_cache:
            value = self._depth_cache.pop(key)
            self._depth_cache[key] = value
            return value.copy()
        raw = cv2.imread(key, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"failed to read fixture depth: {key}")
        value = np.expand_dims(
            cv2.resize(
                raw.astype(np.float32), (self.width, self.height),
                interpolation=cv2.INTER_NEAREST,
            ) / 65535.0,
            axis=0,
        ).astype(np.float32, copy=False)
        if not np.isfinite(value).all():
            raise ValueError(f"fixture depth contains NaN/Inf: {key}")
        if self.cache_size:
            self._depth_cache[key] = value.copy()
            while len(self._depth_cache) > self.cache_size:
                self._depth_cache.popitem(last=False)
        return value

    def __getitem__(self, index):
        observation = np.concatenate((
            self.arrays["velocity_body"][index],
            self.arrays["acceleration_body"][index],
            self.arrays["goal_body"][index],
        )).astype(np.float32)
        return (
            self._load_depth(self.arrays["image_path"][index]),
            self.arrays["position_world"][index].astype(np.float32),
            self.arrays["rotation_world_from_body"][index].astype(np.float32),
            observation,
            int(self.arrays["map_id"][index]),
        )


class StaticV2FixtureMetadataDataset(Dataset):
    def __init__(self, fixture_path):
        self.fixture = StaticV2FixtureDataset(fixture_path, cache_size=0)

    def __len__(self):
        return len(self.fixture)

    def __getitem__(self, index):
        row = self.fixture.metadata(index)
        return {
            "fixture_id": str(row["fixture_id"]),
            "image_sha256": str(row["image_sha256"]),
            "map_id": int(row["map_id"]),
            "position_world": np.asarray(row["position_world"]),
            "rotation_world_from_body": np.asarray(
                row["rotation_world_from_body"]
            ),
            "velocity_body": np.asarray(row["velocity_body"]),
            "acceleration_body": np.asarray(row["acceleration_body"]),
            "goal_body": np.asarray(row["goal_body"]),
        }
