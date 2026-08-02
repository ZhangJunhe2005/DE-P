"""Frozen identities and hashing helpers for MixedSceneStaticYOPOV1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SOURCE_V3_MANIFEST_SHA256 = (
    "a1acfa2fee7e8336917316cb3f07ac005c1ce1354a8bda8de10d02f2438354f6"
)
INITIAL_CHECKPOINT_SHA256 = (
    "615c40c638cf4741d19d232687514ee7e33ffb68b414b8d154263caed5a41223"
)
DATASET_VERSION = "phase8_mixed_scene_static_yopo_derived_v1"
SCHEMA_VERSION = "static_yopo_sample_v1"
DATA_ROUTE = "D1_V3_STATIC_ONLY"
REPRESENTATION = "P0_REFERENCE_VIEW"
FORBIDDEN_BATCH_KEYS = frozenset({
    "composed_depth", "depth_composed", "actor_owner", "actor_metadata",
    "actor_velocity", "actor_future", "dynamic_actionability", "dynamic_label",
    "scenario", "scenario_class", "map_type", "maze_type",
})
ALLOWED_BATCH_KEYS = frozenset({
    "depth", "observation", "position_world", "rotation_world_from_body",
    "map_id", "sample_id",
})


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def assert_source_identity(source_root):
    manifest = Path(source_root) / "manifests" / "dataset_manifest.json"
    actual = sha256_file(manifest)
    if actual != SOURCE_V3_MANIFEST_SHA256:
        raise RuntimeError(
            f"frozen V3 manifest mismatch: expected={SOURCE_V3_MANIFEST_SHA256} actual={actual}"
        )
    return actual


def sample_identity(sequence_id, frame_index, map_uuid, depth_hash, observation):
    return canonical_hash({
        "source_manifest": SOURCE_V3_MANIFEST_SHA256,
        "sequence_id": sequence_id,
        "frame_index": int(frame_index),
        "map_uuid": map_uuid,
        "static_depth_hash": depth_hash,
        "observation": [float(value) for value in observation],
    })


def validate_batch_allowlist(batch):
    keys = frozenset(batch)
    forbidden = keys & FORBIDDEN_BATCH_KEYS
    if forbidden:
        raise ValueError(f"forbidden static YOPO batch keys: {sorted(forbidden)}")
    unexpected = keys - ALLOWED_BATCH_KEYS
    if unexpected:
        raise ValueError(f"unexpected static YOPO batch keys: {sorted(unexpected)}")

