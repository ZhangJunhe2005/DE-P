"""Schema and hashing for physical dynamic-perception controls V1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


CONTRACT_VERSION = "dynamic_perception_physical_control_contract_v1"
SUITE_VERSION = "phase8_dynamic_perception_controls_v1"
SEMANTIC_CLASSES = {
    "required_in_contract_positive",
    "required_hard_negative",
    "bounded_latency_positive",
    "diagnostic_out_of_contract",
    "fixture_invalid_historical",
}
RUNTIME_FIELDS = {
    "depth_file", "camera_positions", "camera_yaws", "intrinsics",
    "timestamps",
}
FORBIDDEN_RUNTIME_FIELDS = {
    "actor_id", "actor_mask", "instance_id", "expected_classification",
    "authority_correspondence", "future_depth", "future_actor_state",
}


def canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def canonical_hash(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_control(control):
    required = {
        "control_id", "contract_version", "suite_version",
        "semantic_class", "role", "expected_outcome", "split",
        "authority", "sensor_hash", "renderer_hash",
        "camera_trajectory", "actor_trajectory", "deterministic_seed",
        "generator_hash", "validator_hash", "runtime_inputs",
        "offline_ground_truth", "provenance_expectation",
    }
    missing = required-set(control)
    if missing:
        raise ValueError(f"control missing fields: {sorted(missing)}")
    if control["contract_version"] != CONTRACT_VERSION:
        raise ValueError("control contract version mismatch")
    if control["suite_version"] != SUITE_VERSION:
        raise ValueError("control suite version mismatch")
    if control["semantic_class"] not in SEMANTIC_CLASSES:
        raise ValueError("unknown semantic class")
    runtime = set(control["runtime_inputs"])
    if runtime != RUNTIME_FIELDS:
        raise ValueError(
            f"runtime fields mismatch: {sorted(runtime)}"
        )
    forbidden = runtime & FORBIDDEN_RUNTIME_FIELDS
    if forbidden:
        raise ValueError(f"offline GT leaked to runtime: {sorted(forbidden)}")
    camera = control["camera_trajectory"]
    positions = np.asarray(camera["positions_world"], dtype=np.float64)
    yaws = np.asarray(camera["yaws_rad"], dtype=np.float64)
    timestamps = np.asarray(camera["timestamps"], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("camera positions must be [T,3]")
    if yaws.shape != (len(positions),) or timestamps.shape != (len(positions),):
        raise ValueError("camera trajectory lengths differ")
    if not np.isfinite(positions).all() or not np.isfinite(yaws).all():
        raise ValueError("camera trajectory must be finite")
    actors = control["actor_trajectory"]
    if actors is not None:
        centers = np.asarray(actors["positions_world"], dtype=np.float64)
        if centers.ndim != 3 or centers.shape[0] != len(positions):
            raise ValueError("actor positions must be [T,A,3]")
        if len(actors["radii_m"]) != centers.shape[1]:
            raise ValueError("actor radii do not match actor count")
    return True


def finalize_manifest(manifest):
    value = dict(manifest)
    value.pop("manifest_hash", None)
    value["manifest_hash"] = canonical_hash(value)
    return value


__all__ = [
    "CONTRACT_VERSION", "SUITE_VERSION", "SEMANTIC_CLASSES",
    "RUNTIME_FIELDS", "FORBIDDEN_RUNTIME_FIELDS", "canonical_hash",
    "canonical_json", "file_hash", "validate_control",
    "finalize_manifest",
]
