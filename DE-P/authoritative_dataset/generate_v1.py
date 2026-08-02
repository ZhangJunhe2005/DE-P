#!/usr/bin/env python3
"""Resumable Authoritative Dataset V1 train/validation generator.

Formal generation is deliberately host-operated.  The implementation writes
one immutable sequence package at a time and regards a unit as complete only
after its manifest and every recorded hash have been verified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import time
import uuid

import numpy as np
import yaml

from authoritative_dataset import (
    FORMAL_DATASET_VERSION, FORMAL_DATASET_VERSION_V2,
    FORMAL_DATASET_VERSION_V3, FORMAL_DATASET_VERSION_V3_MIXED,
    PROTOCOL_VERSION,
)
from authoritative_dataset.dynamic_motion_v2 import (
    CONTRACT_VERSION_V2_1, MotionSamplingError, actor_position,
    build_actor_specs_v2,
    load_motion_contract,
)
from authoritative_dataset.occlusion_constructor_v2_1 import (
    build_occluded_actor_specs_v2_1, natural_occlusion_pattern,
    sample_occlusion_uav_sequence_v2_1,
)
from authoritative_dataset.natural_observable_occlusion_proposer_v1 import (
    propose_natural_cameras, propose_observable_actor,
)
from authoritative_dataset.continuous_v1 import (
    ContinuousState, certify_curve,
)
from authoritative_dataset.cuda_renderer_v1 import (
    CudaAuthorityRenderer, RENDERER_VERSION,
)
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.perception_probe_v2 import (
    run_frozen_perception_probe,
)
from authoritative_dataset.state_semantics_v2 import (
    UavSequenceState,
    actionability_from_state,
    goal_body_from_world,
    quaternion_wxyz_from_yaw,
    sample_uav_sequence,
    vector_body_from_world,
)
from geometry_authority.static_v1 import (
    AUTHORITY_VERSION, EMPTY_GAP_M, build_authority_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "authoritative_dataset_schema_v1"
GENERATOR_VERSION = "authoritative_dataset_generator_v1"
ROUTE_A_V4_STATIC_DATASET_VERSION = "route_a_v4_raw_static_v1"
ROUTE_A_V4_2_STATIC_DATASET_VERSION = "route_a_v4_2_raw_static_v1"
ACTOR_SAMPLING_HOTFIX = {
    "id": "phase8jqv2_4_actor_sampling_fallback_v1",
    "scope": "deterministic fallback after legacy 1000-attempt exhaustion",
    "existing_sequence_semantics_changed": False,
}
CUDA_PARALLEL_HOTFIX = {
    "id": "phase8_dynamic_evidence_cuda_spawn_parallel_v1",
    "scope": (
        "sequence-level spawn multiprocessing with one independent CUDA "
        "context per worker"
    ),
    "existing_sequence_semantics_changed": False,
}
MULTI_TARGET_PAIRING_HOTFIX = {
    "id": "phase8_dynamic_evidence_multi_target_pairing_v2_1",
    "scope": (
        "bounded independent candidate construction and exact continuous "
        "pair selection after coupled multi-target lattice exhaustion"
    ),
    "existing_sequence_semantics_changed": False,
}
COMPATIBLE_HOTFIXES = (
    ACTOR_SAMPLING_HOTFIX,
    CUDA_PARALLEL_HOTFIX,
    MULTI_TARGET_PAIRING_HOTFIX,
)
STOP = False


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def append_journal(root, event):
    path = Path(root) / "generation_state/generation_journal.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp_ns": time.time_ns(), **event}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    progress = ROOT / "logs/phase8jqv2_4/progress.jsonl"
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def load_config(path):
    value = yaml.safe_load(Path(path).read_text())
    value["_file_hash"] = sha256(path)
    if value["dataset_protocol_version"] != PROTOCOL_VERSION:
        raise RuntimeError("protocol version mismatch")
    if value["dataset_version"] not in {
        FORMAL_DATASET_VERSION, FORMAL_DATASET_VERSION_V2,
        FORMAL_DATASET_VERSION_V3, FORMAL_DATASET_VERSION_V3_MIXED,
        ROUTE_A_V4_STATIC_DATASET_VERSION,
        ROUTE_A_V4_2_STATIC_DATASET_VERSION,
    }:
        raise RuntimeError("dataset version mismatch")
    if not value["test_disabled"] or not value["blind_disabled"]:
        raise RuntimeError("test and blind must remain disabled")
    spatial = value.get("scene_spatial_sampling_contract")
    if spatial is not None:
        if spatial.get("version") != "route_a_scene_spatial_sampling_v1":
            raise RuntimeError("unsupported scene spatial sampling contract")
        declared_hash = spatial.get("contract_hash")
        payload = {
            key: entry for key, entry in spatial.items()
            if key != "contract_hash"
        }
        actual_hash = hashlib.sha256(canonical(payload)).hexdigest()
        if declared_hash != actual_hash:
            raise RuntimeError("scene spatial sampling contract hash mismatch")
        required_types = {"cave", "pillar", "forest", "room", "wall"}
        if set(spatial.get("map_types", {})) != required_types:
            raise RuntimeError("scene spatial contract must cover five map types")
    contract_path = value.get("dynamic_motion_contract_path")
    if value["dataset_version"] in {
        FORMAL_DATASET_VERSION_V3, FORMAL_DATASET_VERSION_V3_MIXED,
    }:
        if not contract_path:
            raise RuntimeError("Formal V3 requires a dynamic motion contract")
        contract = load_motion_contract(
            ROOT / contract_path if not Path(contract_path).is_absolute()
            else contract_path
        )
        expected = value.get("dynamic_motion_contract_hash")
        if expected != contract["_file_hash"]:
            raise RuntimeError("dynamic motion contract hash mismatch")
        value["_dynamic_motion_contract"] = contract
        eligibility_path = value.get("occlusion_map_eligibility_path")
        eligibility_hash = value.get("occlusion_map_eligibility_hash")
        if contract["contract_version"] == CONTRACT_VERSION_V2_1:
            if not eligibility_path or not eligibility_hash:
                raise RuntimeError("v2.1 requires frozen map eligibility")
            eligibility_path = (
                ROOT/eligibility_path
                if not Path(eligibility_path).is_absolute()
                else Path(eligibility_path)
            )
            if sha256(eligibility_path) != eligibility_hash:
                raise RuntimeError("occlusion map eligibility hash mismatch")
            eligibility = yaml.safe_load(eligibility_path.read_text())
            value["_formal_scenario_map_eligibility"] = eligibility[
                "formal_scenario_map_eligibility"]
            minimum = eligibility["minimum_diversity"]
            rows = value["_formal_scenario_map_eligibility"][
                "occluded_but_tracked"]
            value["_occlusion_eligibility_gate_passed"] = bool(
                len(rows["train"])
                >= int(minimum["train_eligible_maps_min"])
                and len(rows["valid"])
                >= int(minimum["valid_eligible_maps_min"])
            )
    return value


class GenerationLock:
    def __init__(self, root, scope):
        self.path = Path(root) / (
            "generation_state/locks/"
            f"phase8jqv2_4_generation_{scope}.lock"
        )

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(), "host": socket.gethostname(),
            "started_ns": time.time_ns(), "scope": self.path.stem,
        }
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            existing = self.path.read_text(errors="replace")
            raise RuntimeError(f"generation lock exists: {existing}") from error
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        return self

    def __exit__(self, *_):
        self.path.unlink(missing_ok=True)


def verify_manifest(root, manifest_path):
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return False
    value = json.loads(manifest_path.read_text())
    for relative, expected in value["files"].items():
        path = Path(root) / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"complete unit hash mismatch: {relative}")
    return True


def map_points(seed, smoke=False):
    rng = np.random.default_rng(seed)
    count = 1800 if smoke else 50000
    bounds = np.asarray([8., 8., 4.]) if smoke else np.asarray([60., 60., 15.])
    points = rng.uniform([0, 0, 0], bounds, size=(count, 3)).astype("<f4")
    return points, bounds


def ensure_map(root, config, split, row, smoke=False):
    map_root = Path(root) / f"geometry_authority/{split}/{row['map_uuid']}"
    state = Path(root) / (
        f"generation_state/map_state/{split}_{row['map_uuid']}.json"
    )
    if state.is_file():
        value = json.loads(state.read_text())
        if verify_manifest(root, Path(root) / value["unit_manifest"]):
            return map_root
    append_journal(root, {
        "event": "map_start", "split": split,
        "map_uuid": row["map_uuid"], "status": "running",
    })
    authority_source = config.get("authority_source_root")
    if authority_source:
        source_root = (
            Path(authority_source).expanduser().resolve()
            / f"geometry_authority/{split}/{row['map_uuid']}"
        )
        if not source_root.is_dir():
            raise FileNotFoundError(
                f"reused authority map missing: {source_root}"
            )
        if map_root.exists():
            raise RuntimeError(
                f"unverified reused authority map exists: {map_root}"
            )
        shutil.copytree(source_root, map_root, copy_function=shutil.copy2)
        metadata = json.loads(
            (map_root/"occupancy_metadata.json").read_text()
        )
        derived_source = (
            Path(authority_source).expanduser().resolve()
            / f"derived_geometry/{split}/{row['map_uuid']}.json"
        )
        derived = (
            Path(root)/f"derived_geometry/{split}/{row['map_uuid']}.json"
        )
        derived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(derived_source, derived)
        unit = Path(root)/f"manifests/maps/{split}_{row['map_uuid']}.json"
        files = {}
        artifact_names = [
            "raw_cloud.bin", "occupancy.bin",
            "occupancy_metadata.json", "authority_manifest.json",
        ]
        if (map_root/"mixed_scene_provenance.json").is_file():
            artifact_names.append("mixed_scene_provenance.json")
        for name in artifact_names:
            path = map_root/name
            files[str(path.relative_to(root))] = sha256(path)
        files[str(derived.relative_to(root))] = sha256(derived)
        atomic_json(unit, {
            "unit": "map", "split": split,
            "map_uuid": row["map_uuid"],
            "authority_hash": metadata["artifact_manifest_hash"],
            "occupancy_hash": metadata["occupancy_hash"],
            "reused_from_dataset": config.get(
                "authority_source_dataset", FORMAL_DATASET_VERSION),
            "files": files,
        })
        atomic_json(state, {
            "status": "complete",
            "unit_manifest": str(unit.relative_to(root)),
            "authority_reused": True,
        })
        append_journal(root, {
            "event": "map_complete", "split": split,
            "map_uuid": row["map_uuid"], "status": "complete",
            "authority_reused": True,
        })
        return map_root
    points, bounds = map_points(row["seed"], smoke)
    metadata = build_authority_artifact(
        map_root, points, map_uuid=row["map_uuid"],
        generator_seed=row["seed"], map_id=row["map_id"],
        generator_source_hash=config["frozen_hashes"]["source_hash"],
        generator_config_hash=config["_file_hash"],
        parent_git_commit=config["parent_git_commit"],
        map_namespace=row["namespace"],
        bounds_min=[0, 0, 0], bounds_max=bounds.tolist(),
    )
    derived = Path(root) / f"derived_geometry/{split}/{row['map_uuid']}.json"
    atomic_json(derived, {
        "version": "derived_esdf_authority_v1",
        "authoritative": False,
        "exact_verifier_required": True,
        "source_authority_hash": metadata["artifact_manifest_hash"],
        "source_occupancy_hash": metadata["occupancy_hash"],
        "resolution_m": metadata["occupancy_resolution_m"],
        "origin": metadata["occupancy_origin"],
        "dimensions": metadata["occupancy_dimensions"],
        "generator_hash": config["frozen_hashes"]["source_hash"],
        "representation": "deferred_exact_distance_cache",
    })
    unit = Path(root) / (
        f"manifests/maps/{split}_{row['map_uuid']}.json"
    )
    files = {}
    for name in ("raw_cloud.bin", "occupancy.bin",
                 "occupancy_metadata.json", "authority_manifest.json"):
        path = map_root / name
        files[str(path.relative_to(root))] = sha256(path)
    files[str(derived.relative_to(root))] = sha256(derived)
    atomic_json(unit, {
        "unit": "map", "split": split, "map_uuid": row["map_uuid"],
        "authority_hash": metadata["artifact_manifest_hash"],
        "occupancy_hash": metadata["occupancy_hash"], "files": files,
    })
    atomic_json(state, {
        "status": "complete",
        "unit_manifest": str(unit.relative_to(root)),
    })
    append_journal(root, {
        "event": "map_complete", "split": split,
        "map_uuid": row["map_uuid"], "status": "complete",
    })
    return map_root


def _sample_spatial_point(backend, rng, spatial_policy):
    lower = backend.map.bounds_min + .4
    upper = backend.map.bounds_max - .4
    point = rng.uniform(lower, upper)
    if spatial_policy is None:
        return point, None
    bands = spatial_policy["altitude_bands"]
    weights = np.asarray(
        [float(row["weight"]) for row in bands], dtype=np.float64
    )
    weights /= weights.sum()
    band_index = int(rng.choice(len(bands), p=weights))
    band = bands[band_index]
    relative = float(rng.uniform(
        float(band["relative_min"]), float(band["relative_max"])
    ))
    point[2] = lower[2] + relative * (upper[2] - lower[2])
    return point, {
        "band": str(band["name"]),
        "relative_altitude": relative,
    }


def safe_position(backend, rng, spatial_policy=None):
    for _ in range(1000):
        point, spatial_sample = _sample_spatial_point(
            backend, rng, spatial_policy
        )
        query = backend.query_one(point, .3)
        if not query["collision"] and query["minimum_gap_m"] > .05:
            return (
                (point, query)
                if spatial_policy is None
                else (point, query, spatial_sample)
            )
    raise RuntimeError("maximum_attempts_per_window exceeded")


def scene_spatial_policy(config, map_root):
    contract = config.get("scene_spatial_sampling_contract")
    if contract is None:
        return None, None
    provenance = json.loads(
        (Path(map_root) / "mixed_scene_provenance.json").read_text()
    )
    map_type = {
        1: "cave", 2: "pillar", 5: "forest", 6: "room", 7: "wall",
    }.get(int(provenance["maze_type"]))
    if map_type is None:
        raise RuntimeError("scene spatial contract received unknown maze type")
    return map_type, contract["map_types"][map_type]


def validate_scene_observability(depths, sensor, map_type, spatial_policy):
    maximum = float(sensor["max_depth_m"])
    return_fraction = float(np.mean(depths < maximum - 1e-4))
    near_fraction = float(np.mean(
        depths < float(spatial_policy["near_depth_m"])
    ))
    result = {
        "map_type": map_type,
        "return_fraction": return_fraction,
        "near_obstacle_fraction": near_fraction,
        "near_depth_m": float(spatial_policy["near_depth_m"]),
        "minimum_return_fraction": float(
            spatial_policy["minimum_return_fraction"]
        ),
        "minimum_near_obstacle_fraction": float(
            spatial_policy["minimum_near_obstacle_fraction"]
        ),
    }
    result["passed"] = bool(
        return_fraction + 1e-12 >= result["minimum_return_fraction"]
        and near_fraction + 1e-12
        >= result["minimum_near_obstacle_fraction"]
    )
    return result


def depth_image_cpu(backend, position, yaw, sensor):
    """Canonical occupancy raycast; vectorized over pixels."""
    height, width = sensor["height"], sensor["width"]
    fx, fy, cx, cy = sensor["intrinsics"]
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    rays = np.stack((np.ones_like(u), (u-cx)/fx, (v-cy)/fy), -1)
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    c, s = np.cos(yaw), np.sin(yaw)
    rays = rays @ np.asarray([[c, s, 0], [-s, c, 0], [0, 0, 1.]])
    result = np.full((height, width), sensor["max_depth_m"], np.float32)
    active = np.ones((height, width), bool)
    occupancy = backend.map.flat.reshape(tuple(backend.map.dimensions))
    for distance in np.arange(
        sensor["ray_step_m"], sensor["max_depth_m"] + 1e-9,
        sensor["ray_step_m"],
    ):
        points = position + rays * distance
        indices = np.floor(
            (points-backend.map.origin)/backend.map.resolution
        ).astype(np.int64)
        inside = np.all(
            (indices >= 0) & (indices < backend.map.dimensions), axis=-1)
        hit = np.zeros_like(active)
        valid = active & inside
        hit[valid] = occupancy[
            indices[..., 0][valid], indices[..., 1][valid],
            indices[..., 2][valid]]
        result[hit] = distance
        active &= inside & ~hit
        if not active.any():
            break
    return result.astype("<f4")


_CUDA_RENDERER = None


def cuda_renderer(sensor, device):
    global _CUDA_RENDERER
    if _CUDA_RENDERER is None \
            or str(_CUDA_RENDERER.device) != str(device):
        _CUDA_RENDERER = CudaAuthorityRenderer(sensor, device)
    return _CUDA_RENDERER


def build_actor_specs(
    backend, rng, scenario, uav_positions, yaw, frame_times,
    motion_contract=None,
):
    if motion_contract is not None:
        return build_actor_specs_v2(
            backend, rng, scenario, uav_positions, yaw, frame_times,
            motion_contract,
        )
    if scenario == "no_target":
        return []
    count = 2 if scenario == "multi_target" else 1
    future_end = float(frame_times[-1] + 1.7)
    check_times = np.linspace(0, future_end, 40)
    uav_positions = np.asarray(uav_positions, dtype=np.float64)
    camera_path = np.stack([
        np.interp(check_times, frame_times, uav_positions[:, axis],
                  right=uav_positions[-1, axis])
        for axis in range(3)
    ], axis=1)
    uav_path_span = float(np.max(np.linalg.norm(
        uav_positions-uav_positions[0], axis=1
    )))
    actor_forward_max = min(8.0, max(2.8, uav_path_span+1.2))
    for _ in range(1000):
        c, s = np.cos(yaw), np.sin(yaw)
        forward = np.asarray([c, s, 0.])
        lateral_axis = np.asarray([-s, c, 0.])
        actors = []
        valid = True
        for actor_id in range(count):
            lateral = (
                (actor_id-.5)*1.0 if count == 2
                else float(rng.uniform(-.25, .25)))
            start = uav_positions[0] + forward*float(
                rng.uniform(.9, actor_forward_max)
            ) \
                + lateral_axis*lateral
            if scenario == "crossing":
                velocity = lateral_axis*.12
            elif scenario == "head_on":
                velocity = -forward*.08
            elif scenario == "temporal_separation":
                velocity = lateral_axis*.06
            elif scenario == "static_dynamic_joint_constraint":
                velocity = -forward*.04
            else:
                velocity = np.zeros(3)
            path = start[None, :] + check_times[:, None]*velocity
            if (
                any(backend.query_one(point, .2)["collision"]
                    for point in path)
                or np.min(np.linalg.norm(path-camera_path, axis=1)) <= .55
                or any(
                    np.min(np.linalg.norm(
                        path - (
                            row["start"][None, :]
                            + check_times[:, None]*row["velocity"]),
                        axis=1)) <= .4
                    for row in actors)
            ):
                valid = False
                break
            actors.append({
                "actor_id": actor_id, "start": start,
                "velocity": velocity, "radius_m": .2,
            })
        if valid:
            return actors
    # Compatibility-preserving fallback: this is reached only when the legacy
    # random sampler would have raised.  Formal-duration multi-target sequences
    # can leave a narrow feasible set because both actors must remain clear of
    # the entire moving camera path.  Search a deterministic local lattice
    # around the beginning and end of that same path instead of aborting.
    c, s = np.cos(yaw), np.sin(yaw)
    forward = np.asarray([c, s, 0.])
    lateral_axis = np.asarray([-s, c, 0.])

    def velocity_for_scenario():
        if scenario == "crossing":
            return lateral_axis*.12
        if scenario == "head_on":
            return -forward*.08
        if scenario == "temporal_separation":
            return lateral_axis*.06
        if scenario == "static_dynamic_joint_constraint":
            return -forward*.04
        return np.zeros(3)

    candidates = []
    velocity = velocity_for_scenario()
    for anchor in (uav_positions[0], uav_positions[-1]):
        for forward_distance in (.7, 1., 1.4, 1.8, 2.2, 2.8):
            for lateral_distance in (
                0., .2, -.2, .35, -.35, .5, -.5, .7, -.7
            ):
                for vertical_distance in (0., .2, -.2):
                    start = (
                        anchor + forward*forward_distance
                        + lateral_axis*lateral_distance
                        + np.asarray([0., 0., vertical_distance])
                    )
                    path = (
                        start[None, :]
                        + check_times[:, None]*velocity
                    )
                    sightline = np.linspace(
                        anchor, start, 17, dtype=np.float64
                    )[1:-1]
                    if (
                        any(backend.query_one(point, .2)["collision"]
                            for point in path)
                        or any(backend.query_one(point, .05)["collision"]
                               for point in sightline)
                        or np.min(np.linalg.norm(
                            path-camera_path, axis=1
                        )) <= .55
                    ):
                        continue
                    candidates.append({
                        "start": start,
                        "velocity": velocity.copy(),
                        "path": path,
                    })
    for first_index, first in enumerate(candidates):
        selected = [first]
        if count == 2:
            selected.extend(
                candidate for candidate in candidates[first_index+1:]
                if all(np.min(np.linalg.norm(
                    candidate["path"]-existing["path"], axis=1
                )) > .4 for existing in selected)
            )
            selected = selected[:2]
        if len(selected) == count:
            return [{
                "actor_id": actor_id,
                "start": actor["start"],
                "velocity": actor["velocity"],
                "radius_m": .2,
                "sampling_method":
                    "deterministic_local_lattice_fallback_v1",
            } for actor_id, actor in enumerate(selected)]
    raise RuntimeError(
        "continuous actor sampling exhausted legacy and lattice fallback"
    )


def requires_frozen_dynamic_perception_validation(task, config):
    """Whether this sequence is governed by the dynamic-track contract.

    Static suites reuse scenario names such as ``brake`` for UAV motion. A
    dataset-level dynamic contract must not reinterpret those actor-free
    sequences as dynamic tracking samples.
    """
    return (
        task.get("suite") == "dynamic"
        and config.get("_dynamic_motion_contract") is not None
        and not config.get("evidence_authority_outputs", False)
    )


def natural_proposal_uav_state(backend, proposal, frame_times):
    """Convert a certified natural camera proposal into a stationary UAV state."""
    positions = np.asarray(proposal.position_world, dtype=np.float64)
    camera = positions[0].copy()
    certificate = certify_curve(
        lambda _value, point=camera: point,
        backend, 1.7, max_depth=14,
    )
    if certificate.state is not ContinuousState.CERTIFIED_SAFE:
        raise MotionSamplingError(
            "natural camera proposal failed continuous UAV certification")
    zeros = np.zeros_like(positions)
    return UavSequenceState(
        position_world=positions,
        velocity_world=zeros,
        acceleration_world=zeros,
        yaw=np.asarray(proposal.yaw, dtype=np.float64),
        goal_world=np.repeat(camera[None, :], len(frame_times), axis=0),
        reference_certificate=certificate,
        reference_goal_world=camera,
    )


def generate_sequence(root, config, split, task, map_root, device):
    manifest = Path(root) / f"manifests/sequences/{task['sequence_id']}.json"
    state = Path(root) / (
        f"generation_state/sequence_state/{task['sequence_id']}.json"
    )
    if state.is_file() and verify_manifest(root, manifest):
        return "skipped"
    append_journal(root, {
        "event": "sequence_start", "split": split,
        "sequence_id": task["sequence_id"], "suite": task["suite"],
        "status": "running",
    })
    backend = ExactAuthorityBVH(map_root)
    map_type, spatial_policy = scene_spatial_policy(
        config, map_root
    )
    rng = np.random.default_rng(task["seed"])
    frames, certificates = [], []
    sensor = config["sensor_settings"]
    latency = config["state_sampling_contract"]["command_latency_s"]
    frame_times = (
        np.arange(task["frame_count"], dtype=np.float64)
        * sensor["frame_period_ns"]/1e9)
    joint_attempts = (
        int(config.get("_dynamic_motion_contract", {}).get(
            "sampling", {}
        ).get(
            "maximum_motion_joint_attempts",
            config["certificate_settings"].get(
                "maximum_motion_joint_attempts", 16
            ),
        ))
        if task["suite"] == "dynamic"
        and config.get("_dynamic_motion_contract") is not None
        else int(config.get(
            "scene_spatial_sampling_contract", {}
        ).get("maximum_sequence_resamples", 1))
    )
    last_motion_error = None
    renderer_diagnostics = None
    evidence_authority = None
    occlusion_construction = None
    perception_probe = None
    occlusion_camera_proposals = None
    for motion_joint_attempt in range(joint_attempts):
        occlusion_construction = None
        renderer_diagnostics = None
        use_occlusion_uav_constructor = (
            config.get("_dynamic_motion_contract", {}).get(
                "contract_version") == CONTRACT_VERSION_V2_1
            and task["scenario"] == "occluded_but_tracked"
        )
        use_formal_natural_proposer = (
            use_occlusion_uav_constructor
            and config.get("formal_occlusion_constructor")
            == "natural_observable_occlusion_proposer_v1"
        )
        try:
            if use_occlusion_uav_constructor:
                if "formal_occlusion_gap_frames" in config:
                    allowed_gaps = tuple(
                        int(value) for value in
                        config["formal_occlusion_gap_frames"])
                    if not allowed_gaps:
                        raise RuntimeError(
                            "formal occlusion gap contract is empty")
                    requested_gap = allowed_gaps[
                        int(task["seed"]) % len(allowed_gaps)]
                else:
                    requested_gap = 1 + int(task["seed"]) % 3
                if use_formal_natural_proposer:
                    if occlusion_camera_proposals is None:
                        occlusion_camera_proposals = (
                            propose_natural_cameras(
                                backend, frame_times, requested_gap,
                                maximum_proposals=int(config.get(
                                    "formal_occlusion_maximum_proposals",
                                    32,
                                )),
                                maximum_patch_checks=int(config.get(
                                    "formal_occlusion_maximum_patch_checks",
                                    4096,
                                )),
                            )
                        )
                        if occlusion_camera_proposals:
                            offset = int(task["seed"]) % len(
                                occlusion_camera_proposals)
                            occlusion_camera_proposals = (
                                occlusion_camera_proposals[offset:]
                                + occlusion_camera_proposals[:offset]
                            )
                    if motion_joint_attempt >= len(
                        occlusion_camera_proposals
                    ):
                        raise MotionSamplingError(
                            "bounded natural camera proposals exhausted")
                    uav_state = natural_proposal_uav_state(
                        backend,
                        occlusion_camera_proposals[motion_joint_attempt],
                        frame_times,
                    )
                else:
                    uav_state = sample_occlusion_uav_sequence_v2_1(
                        backend, rng, frame_times, requested_gap)
            else:
                position_sampler = (
                    safe_position
                    if spatial_policy is None
                    else lambda backend_value, rng_value: safe_position(
                        backend_value, rng_value, spatial_policy
                    )
                )
                uav_state = sample_uav_sequence(
                    backend, rng, task["scenario"], frame_times,
                    position_sampler,
                    maximum_attempts=config["certificate_settings"][
                        "maximum_attempts_per_window"
                    ],
                )
        except MotionSamplingError as error:
            last_motion_error = error
            continue
        positions = uav_state.position_world
        yaws = uav_state.yaw
        static_certificate = uav_state.reference_certificate
        visibility_attempts = (
            config["certificate_settings"].get(
                "maximum_actor_visibility_attempts", 128)
            if task["suite"] == "dynamic" else 1)
        try:
            for visibility_attempt in range(visibility_attempts):
                if task["suite"] == "dynamic":
                    contract = config.get("_dynamic_motion_contract")
                    is_v2_1_occlusion = (
                        contract is not None
                        and contract["contract_version"]
                        == CONTRACT_VERSION_V2_1
                        and task["scenario"] == "occluded_but_tracked"
                    )
                    if is_v2_1_occlusion:
                        if not str(device).startswith("cuda"):
                            raise MotionSamplingError(
                                "v2.1 occlusion requires canonical CUDA raster")
                        if use_formal_natural_proposer:
                            try:
                                occlusion_construction = (
                                    propose_observable_actor(
                                        backend,
                                        cuda_renderer(sensor, device),
                                        occlusion_camera_proposals[
                                            motion_joint_attempt],
                                        frame_times, contract,
                                        requested_gap,
                                        maximum_raster_candidates=int(
                                            config.get(
                                                "formal_occlusion_actor_"
                                                "candidate_budget",
                                                128,
                                            )
                                        ),
                                    )
                                )
                            except RuntimeError as error:
                                if "bounded observable actor" not in str(
                                    error
                                ):
                                    raise
                                raise MotionSamplingError(str(error)) \
                                    from error
                        else:
                            occlusion_construction = (
                                build_occluded_actor_specs_v2_1(
                                    backend,
                                    cuda_renderer(sensor, device),
                                    rng, positions, yaws, frame_times,
                                    contract, requested_gap,
                                    method=(
                                        "random_seeded"
                                        if int(task["seed"]) % 2
                                        else "deterministic"
                                    ),
                                )
                            )
                        actor_specs = occlusion_construction.actors
                    else:
                        actor_specs = build_actor_specs(
                            backend, rng, task["scenario"], positions,
                            float(yaws[0]), frame_times, contract,
                        )
                else:
                    actor_specs = []
                actor_positions = np.empty(
                    (task["frame_count"], len(actor_specs), 3),
                    dtype=np.float64,
                )
                for actor_index, actor in enumerate(actor_specs):
                    actor_positions[:, actor_index, :] = actor_position(
                        actor, frame_times
                    )
                if str(device).startswith("cuda"):
                    if occlusion_construction is not None:
                        renderer_diagnostics = (
                            occlusion_construction.diagnostics)
                    elif (
                        config.get("_dynamic_motion_contract", {}).get(
                            "contract_version") == CONTRACT_VERSION_V2_1
                    ):
                        renderer_diagnostics = cuda_renderer(
                            sensor, device).render_with_actor_diagnostics(
                                backend, positions, yaws, actor_positions,
                                [row["radius_m"] for row in actor_specs])
                    else:
                        renderer_diagnostics = None
                    if renderer_diagnostics is not None:
                        depths = renderer_diagnostics["composed_depth"]
                        static_depths = renderer_diagnostics["static_depth"]
                        actor_pixel_counts = renderer_diagnostics[
                            "actor_pixel_count"]
                    else:
                        depths, static_depths, actor_pixel_counts = (
                            cuda_renderer(sensor, device).render(
                                backend, positions, yaws, actor_positions,
                                [row["radius_m"] for row in actor_specs])
                        )
                else:
                    static_depths = np.asarray([
                        depth_image_cpu(backend, position, frame_yaw, sensor)
                        for position, frame_yaw in zip(positions, yaws)])
                    depths = static_depths.copy()
                    actor_pixel_counts = np.zeros(
                        task["frame_count"], dtype=np.int64)
                if not actor_specs or np.any(actor_pixel_counts):
                    break
            else:
                raise MotionSamplingError(
                    "contract actor invisible after deterministic resampling"
                )
            if requires_frozen_dynamic_perception_validation(task, config):
                occlusion_required = (
                    task["scenario"] == "occluded_but_tracked"
                )
                visible_mask = (
                    renderer_diagnostics[
                        "per_actor_visible_pixel_count"][:, 0] > 0
                    if renderer_diagnostics is not None and actor_specs
                    else actor_pixel_counts > 0
                )
                if occlusion_required:
                    if renderer_diagnostics is None:
                        raise MotionSamplingError(
                            "occlusion provenance diagnostics unavailable")
                    pattern = natural_occlusion_pattern(
                        renderer_diagnostics, 0,
                        config["_dynamic_motion_contract"])
                    valid_gap = bool(pattern["accepted_gaps"])
                    if not valid_gap:
                        raise MotionSamplingError(
                            "no bounded natural static-occlusion gap"
                        )
                duration = float(config["_dynamic_motion_contract"][
                    "sampling"
                ]["minimum_sustained_dynamic_duration_s"])
                minimum_frames = max(
                    1, int(np.ceil(
                        duration
                        / (sensor["frame_period_ns"]/1e9)
                    ))
                )
                perception_probe = run_frozen_perception_probe(
                    depths, positions, yaws, actor_positions, frame_times,
                    sensor, minimum_sustained_frames=minimum_frames,
                    visibility_mask=visible_mask,
                    require_occlusion_identity=occlusion_required,
                    prediction_position_error_max_m=float(
                        config["_dynamic_motion_contract"][
                            "validation"].get(
                                "prediction_position_error_max_m", 1.0)),
                )
                if perception_probe["status"] != "PASS":
                    raise MotionSamplingError(
                        "frozen perception did not sustain every actor"
                    )
            scene_observability = (
                validate_scene_observability(
                    static_depths, sensor, map_type, spatial_policy
                )
                if spatial_policy is not None else None
            )
            if (
                scene_observability is not None
                and not scene_observability["passed"]
            ):
                raise MotionSamplingError(
                    "scene-relative obstacle observability gate failed: "
                    f"{scene_observability}"
                )
        except MotionSamplingError as error:
            last_motion_error = error
            continue
        if config.get("evidence_authority_outputs", False):
            if not str(device).startswith("cuda"):
                raise RuntimeError("V3 evidence authority outputs require CUDA")
            evidence_authority = cuda_renderer(
                sensor, device).render_with_actor_diagnostics(
                    backend, positions, yaws, actor_positions,
                    [row["radius_m"] for row in actor_specs],
                    return_owner_map=True,
                )
            if not np.array_equal(
                evidence_authority["composed_depth"], depths
            ):
                raise RuntimeError("authority re-render changed composed depth")
        break
    else:
        raise RuntimeError(
            "motion/map/UAV joint sampling exhausted without weakening "
            f"the contract: {last_motion_error}"
        )
    for index in range(task["frame_count"]):
        timestamp = index * sensor["frame_period_ns"]
        actors = []
        for actor in actor_specs:
            current = actor_position(
                actor, [frame_times[index]]
            )[0]
            future_times = np.linspace(0, 1.7, 18)
            future = actor_position(
                {
                    **actor,
                    "start": current,
                },
                future_times,
            )
            actor_index = int(actor["actor_id"])
            projected_pixels = (
                int(renderer_diagnostics[
                    "per_actor_projected_pixel_count"][
                        index, actor_index])
                if renderer_diagnostics is not None else
                int(actor_pixel_counts[index])
            )
            visible_pixels = (
                int(renderer_diagnostics[
                    "per_actor_visible_pixel_count"][
                        index, actor_index])
                if renderer_diagnostics is not None else
                int(actor_pixel_counts[index])
            )
            blocked_pixels = (
                int(renderer_diagnostics[
                    "per_actor_static_blocked_pixel_count"][
                        index, actor_index])
                if renderer_diagnostics is not None else 0
            )
            outside_fov = bool(
                renderer_diagnostics["per_actor_outside_fov"][
                    index, actor_index]
            ) if renderer_diagnostics is not None else False
            behind_camera = bool(
                renderer_diagnostics["per_actor_behind_camera"][
                    index, actor_index]
            ) if renderer_diagnostics is not None else False
            beyond_depth = bool(
                renderer_diagnostics["per_actor_beyond_max_depth"][
                    index, actor_index]
            ) if renderer_diagnostics is not None else False
            natural_occluded = bool(
                projected_pixels > 0 and visible_pixels == 0
                and blocked_pixels == projected_pixels
                and not outside_fov and not behind_camera
                and not beyond_depth
            )
            reason = (
                "static_authority" if natural_occluded
                else "outside_fov" if outside_fov
                else "behind_camera" if behind_camera
                else "beyond_max_depth" if beyond_depth
                else "none"
            )
            constructor = actor.get("occlusion_constructor", {})
            physical_collision = bool(
                backend.query_one(
                    current, actor["radius_m"])["collision"])
            margin = float(config.get(
                "_dynamic_motion_contract", {}).get(
                    "sampling", {}).get(
                        "actor_static_clearance_margin_m", 0.0))
            actors.append({
                    "actor_id": actor["actor_id"], "shape": "sphere",
                    "radius_m": actor["radius_m"],
                    "position_world": current.tolist(),
                    "velocity_world": actor["velocity"].tolist(),
                    "acceleration_world": [0., 0., 0.],
                    "future_world": future.tolist(),
                    "future_timestamps_ns": [
                        timestamp + round(t*1e9)
                        for t in future_times],
                    "spawn_time_ns": 0,
                    "exit_time_ns":
                        task["frame_count"]*sensor["frame_period_ns"],
                    "motion_contract_version":
                        actor.get("motion_contract_version", "legacy_unversioned"),
                    "motion_profile":
                        actor.get("motion_profile", "constant_velocity"),
                    "configured_speed_mps": float(actor.get(
                        "configured_speed_mps",
                        np.linalg.norm(actor["velocity"]),
                    )),
                    "sampling_method":
                        actor.get("sampling_method", "legacy_random_v1"),
                    "occluded": natural_occluded,
                    "visibility": (
                        float(visible_pixels/projected_pixels)
                        if projected_pixels else 0.0),
                    "inside_image": projected_pixels > 0,
                    "occlusion_reason": reason,
                    "projected_pixel_count": projected_pixels,
                    "visible_pixel_count": visible_pixels,
                    "static_blocked_pixel_count": blocked_pixels,
                    "occluder_voxel_index":
                        constructor.get("occluder_voxel_index"),
                    "occluder_authority_hash":
                        constructor.get("occluder_authority_hash"),
                    "occlusion_gap_id": (
                        f"{task['sequence_id']}:{constructor.get('gap_start')}"
                        if natural_occluded else None),
                    "natural_occlusion_verified": natural_occluded,
                    "physical_static_collision": physical_collision,
                    "static_collision": physical_collision,
                    "static_clearance_margin_satisfied": not bool(
                        backend.query_one(
                            current,
                            actor["radius_m"]+margin)["collision"]),
                    "future_static_collision": any(
                        backend.query_one(
                            point, actor["radius_m"])["collision"]
                        for point in future),
                })
        dynamic_gap = min([
            float(np.linalg.norm(
                np.asarray(actor["position_world"])-positions[index])
                - actor["radius_m"]-.3) for actor in actors
        ] or [float("inf")])
        actionability, first_position = actionability_from_state(
            backend, uav_state, index, latency, dynamic_gap
        )
        static_gap = static_certificate.minimum_exact_sampled_gap_m
        empty_mask = static_gap >= EMPTY_GAP_M
        classification = (
            "recovery" if actionability["recoverable"]
            else "stress_test" if actionability["stress"]
            else "generator_guaranteed_preventable"
        )
        certificate = {
            "certificate_version":
                "authoritative_feasibility_certificate_v1",
            "static_state": static_certificate.state.value,
            "static_safety_only_success":
                static_certificate.state is ContinuousState.CERTIFIED_SAFE,
            "static_progress_success":
                static_certificate.state is ContinuousState.CERTIFIED_SAFE,
            "dynamic_joint_safe": dynamic_gap > 0,
            "classification": classification,
            "recovery_certificate": actionability["recoverable"],
            "minimum_static_gap_m":
                None if empty_mask else static_gap,
            "minimum_gap_valid_mask": not empty_mask,
            "minimum_dynamic_gap_m":
                None if not np.isfinite(dynamic_gap) else dynamic_gap,
            "unknown_reason": static_certificate.unknown_reason,
            "query_count": static_certificate.query_count,
            "recursion_depth": static_certificate.recursion_depth,
        }
        frame = {
            "schema_version": SCHEMA_VERSION,
            "dataset_version": config["dataset_version"],
            "protocol_version": PROTOCOL_VERSION,
            "split": split, "suite": task["suite"],
            "scenario_class": task["scenario"],
            "sequence_id": task["sequence_id"], "frame_index": index,
            "timestamp_ns": timestamp, "sensor_timestamp_ns": timestamp,
            "state_timestamp_ns": timestamp,
            "odometry_timestamp_ns": timestamp,
            "command_issue_timestamp_ns": timestamp,
            "first_controllable_timestamp_ns":
                timestamp + round(latency*1e9),
            "map_uuid": backend.map.metadata["map_uuid"],
            "authority_version": AUTHORITY_VERSION,
            "authority_manifest_hash":
                backend.map.metadata["artifact_manifest_hash"],
            "occupancy_hash": backend.map.metadata["occupancy_hash"],
            "render_backend_version": (
                RENDERER_VERSION if str(device).startswith("cuda")
                else "canonical_occupancy_numpy_raycast_v1"),
            "dynamic_depth_composited": bool(actor_specs),
            "actor_depth_pixel_count": int(actor_pixel_counts[index]),
            "actor_visibility_resample_attempts": visibility_attempt,
            "static_depth_frame_hash": hashlib.sha256(
                static_depths[index].tobytes()).hexdigest(),
            "position_world": positions[index].tolist(),
            "quaternion_world_from_body":
                quaternion_wxyz_from_yaw(yaws[index]).tolist(),
            "velocity_world":
                uav_state.velocity_world[index].tolist(),
            "velocity_body": vector_body_from_world(
                uav_state.velocity_world[index], yaws[index]
            ).tolist(),
            "acceleration_world":
                uav_state.acceleration_world[index].tolist(),
            "acceleration_body": vector_body_from_world(
                uav_state.acceleration_world[index], yaws[index]
            ).tolist(),
            "angular_velocity_body": [0., 0., 0.],
            "previous_command": [
                *uav_state.velocity_world[index].tolist(), yaws[index]
            ],
            "measured_command_latency_s": latency,
            "first_controllable_state": {
                "position_world": first_position.tolist(),
                "velocity_world": (
                    uav_state.velocity_world[index]
                    + latency*uav_state.acceleration_world[index]
                ).tolist(),
                "acceleration_world":
                    uav_state.acceleration_world[index].tolist()},
            "goal_world": uav_state.goal_world[index].tolist(),
            "goal_body": goal_body_from_world(
                positions[index], uav_state.goal_world[index],
                yaws[index],
            ).tolist(),
            "goal_type": task["scenario"],
            "minimum_progress_requirement_m": 0.,
            "actor_metadata": actors,
            "actionability": actionability,
            "runtime_random_sampling": False,
            "rng_seed": task["seed"] + index,
        }
        if spatial_policy is not None:
            available_min = float(backend.map.bounds_min[2] + .4)
            available_max = float(backend.map.bounds_max[2] - .4)
            relative_altitude = float(
                (positions[index, 2] - available_min)
                / (available_max - available_min)
            )
            frame["scene_spatial_sampling"] = {
                "contract_version":
                    config["scene_spatial_sampling_contract"]["version"],
                "contract_hash":
                    config["scene_spatial_sampling_contract"][
                        "contract_hash"],
                "map_type": map_type,
                "relative_altitude": relative_altitude,
                "observability": scene_observability,
            }
        frames.append(frame)
        certificates.append(certificate)
    relative_base = Path(f"{task['suite']}/{split}/{task['sequence_id']}")
    staging = Path(root) / (
        f".staging/{task['sequence_id']}-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    np.save(staging/"depth.npy", depths, allow_pickle=False)
    authority_files = []
    if config.get("evidence_authority_outputs", False):
        if evidence_authority is None:
            raise RuntimeError("missing V3 offline evidence authority")
        np.save(
            staging/"static_depth.npy",
            evidence_authority["static_depth"], allow_pickle=False,
        )
        np.save(
            staging/"actor_owner.npy",
            evidence_authority["nearest_actor_owner"], allow_pickle=False,
        )
        authority_files = ["static_depth.npy", "actor_owner.npy"]
    camera_pose_semantic_hash = hashlib.sha256(canonical({
        "position_world": [
            row["position_world"] for row in frames
        ],
        "quaternion_world_from_body": [
            row["quaternion_world_from_body"] for row in frames
        ],
        "timestamp_ns": [row["timestamp_ns"] for row in frames],
    })).hexdigest()
    atomic_json(staging/"render_diagnostics.json", {
        "renderer_version": (
            RENDERER_VERSION if str(device).startswith("cuda")
            else "canonical_occupancy_numpy_raycast_v1"),
        "device": str(device),
        "actor_count": len(actor_specs),
        "actor_depth_pixels_total": int(actor_pixel_counts.sum()),
        "frames_with_actor_depth": int(np.count_nonzero(actor_pixel_counts)),
        "actor_visibility_resample_attempts": visibility_attempt,
        "static_depth_semantic_hash": hashlib.sha256(
            static_depths.tobytes()).hexdigest(),
        "composed_depth_semantic_hash": hashlib.sha256(
            depths.tobytes()).hexdigest(),
        "camera_pose_semantic_hash": camera_pose_semantic_hash,
            "camera_pose_source": "persisted_uav_state_v2",
            "frozen_perception_probe": perception_probe,
            "scene_spatial_observability": scene_observability,
        })
    (staging/"frames.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True)+"\n" for row in frames))
    (staging/"certificates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True)+"\n"
                for row in certificates))
    destination = Path(root) / relative_base
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"unverified sequence exists: {destination}")
    os.replace(staging, destination)
    files = {
        str((relative_base/name)): sha256(destination/name)
        for name in (
            "depth.npy", "frames.jsonl", "certificates.jsonl",
            "render_diagnostics.json", *authority_files)
    }
    atomic_json(manifest, {
        "unit": "sequence", "protocol_version": PROTOCOL_VERSION,
        "dataset_version": config["dataset_version"],
        "split": split, "suite": task["suite"],
        "scenario": task["scenario"],
        "sequence_id": task["sequence_id"],
        "map_uuid": backend.map.metadata["map_uuid"],
        "frame_count": task["frame_count"],
        "renderer_version": (
            RENDERER_VERSION if str(device).startswith("cuda")
            else "canonical_occupancy_numpy_raycast_v1"),
        "scene_spatial_sampling_contract_hash": (
            config.get("scene_spatial_sampling_contract", {}).get(
                "contract_hash"
            )
        ),
        "files": files,
    })
    atomic_json(state, {
        "status": "complete",
        "unit_manifest": str(manifest.relative_to(root)),
    })
    append_journal(root, {
        "event": "sequence_complete", "split": split,
        "sequence_id": task["sequence_id"], "suite": task["suite"],
        "frames": task["frame_count"], "certificates": task["frame_count"],
        "status": "complete",
    })
    return "complete"


def _sequence_worker(arguments):
    return generate_sequence(*arguments)


def _initialize_sequence_worker(runtime_device):
    """Initialize an isolated worker without inheriting a CUDA context."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if str(runtime_device).startswith("cuda"):
        import torch
        device_index = int(str(runtime_device).split(":", 1)[1])
        torch.set_num_threads(1)
        torch.cuda.set_device(device_index)


def _run_process_pool(
    root, split, work, workers, runtime_device, fail_fast,
):
    """Run immutable sequence units concurrently and report parent progress."""
    context = (
        multiprocessing.get_context("spawn")
        if str(runtime_device).startswith("cuda") else None
    )
    options = {
        "max_workers": workers,
        "initializer": _initialize_sequence_worker,
        "initargs": (runtime_device,),
    }
    if context is not None:
        options["mp_context"] = context
    with ProcessPoolExecutor(**options) as executor:
        futures = {
            executor.submit(_sequence_worker, row): row[3]
            for row in work
        }
        completed_count = 0
        last_summary = time.monotonic()
        for future in as_completed(futures):
            task = futures[future]
            if STOP:
                for pending in futures:
                    pending.cancel()
                append_journal(root, {
                    "event": "interrupted", "status": "running",
                    "split": split,
                })
                raise KeyboardInterrupt
            try:
                future.result()
                completed_count += 1
            except Exception as error:
                append_journal(root, {
                    "event": "sequence_failed", "status": "failed",
                    "split": split,
                    "sequence_id": task["sequence_id"],
                    "suite": task["suite"],
                    "scenario": task["scenario"],
                    "map_uuid": task["map_uuid"],
                    "error": repr(error),
                })
                if fail_fast:
                    for pending in futures:
                        pending.cancel()
                    raise
            if time.monotonic() - last_summary >= 30:
                payload = {
                    "status": "RUNNING", "split": split,
                    "sequences_complete_this_run": completed_count,
                    "sequences_this_run": len(work),
                    "workers_active": workers,
                }
                if str(runtime_device).startswith("cuda"):
                    payload.update({
                        "renderer": RENDERER_VERSION,
                        "device": runtime_device,
                        "cuda_execution": "spawn_process_pool",
                    })
                print(json.dumps(payload, sort_keys=True), flush=True)
                last_summary = time.monotonic()


def tasks_for(config, split, smoke=False):
    split_plan = config["formal_splits"][split]
    smoke_settings = config.get("smoke_settings", {})
    frames = int(smoke_settings.get("frames_per_split", 300)) \
        if smoke else split_plan["frames"]
    maps = split_plan["maps"][
        :int(smoke_settings.get("map_count", 3))
    ] if smoke else split_plan["maps"]
    per_sequence = int(
        smoke_settings.get("frames_per_sequence", 20)
    ) if smoke else config["frames_per_sequence"]
    suites = (
        [("static", value) for value in config["static_scenarios"]]
        + [("dynamic", value) for value in config["dynamic_scenarios"]]
    )
    total_sequences = (frames + per_sequence - 1)//per_sequence
    result, remaining, index = [], frames, 0
    scenario_occurrences = {}
    while remaining:
        count = min(per_sequence, remaining)
        suite, scenario = suites[index % len(suites)]
        formal_eligibility = config.get(
            "_formal_scenario_map_eligibility", {}).get(
                scenario, {}).get(split)
        formal_occlusion_types = set(config.get(
            "formal_occlusion_map_semantic_names", ()))
        formal_occlusion_uuids = tuple(config.get(
            "formal_occlusion_map_uuids", {}).get(split, ()))
        formal_dynamic_actor_uuids = tuple(config.get(
            "formal_dynamic_actor_map_uuids", {}).get(split, ()))
        override = (
            smoke_settings.get("scenario_map_indices", {}).get(scenario)
            if smoke and suite == "dynamic" else None
        )
        if (
            suite == "dynamic"
            and scenario == "occluded_but_tracked"
            and formal_occlusion_uuids
        ):
            by_uuid = {row["map_uuid"]: row for row in maps}
            missing = [
                value for value in formal_occlusion_uuids
                if value not in by_uuid
            ]
            if missing:
                raise RuntimeError(
                    f"formal occlusion maps are outside {split}: {missing}")
            occurrence = scenario_occurrences.get(scenario, 0)
            map_row = by_uuid[
                formal_occlusion_uuids[
                    occurrence % len(formal_occlusion_uuids)]]
            scenario_occurrences[scenario] = occurrence + 1
        elif (
            suite == "dynamic"
            and scenario not in {"no_target", "occluded_but_tracked"}
            and formal_dynamic_actor_uuids
        ):
            by_uuid = {row["map_uuid"]: row for row in maps}
            missing = [
                value for value in formal_dynamic_actor_uuids
                if value not in by_uuid
            ]
            if missing:
                raise RuntimeError(
                    f"formal dynamic actor maps are outside {split}: "
                    f"{missing}")
            occurrence = scenario_occurrences.get(scenario, 0)
            map_row = by_uuid[
                formal_dynamic_actor_uuids[
                    occurrence % len(formal_dynamic_actor_uuids)]]
            scenario_occurrences[scenario] = occurrence + 1
        elif (
            suite == "dynamic"
            and scenario == "occluded_but_tracked"
            and formal_occlusion_types
        ):
            eligible_maps = [
                row for row in maps
                if row.get("semantic_name") in formal_occlusion_types
            ]
            if not eligible_maps:
                raise RuntimeError(
                    "no maps satisfy formal occlusion semantic contract")
            occurrence = scenario_occurrences.get(scenario, 0)
            map_row = eligible_maps[occurrence % len(eligible_maps)]
            scenario_occurrences[scenario] = occurrence + 1
        elif override:
            occurrence = scenario_occurrences.get(scenario, 0)
            map_row = maps[int(override[occurrence % len(override)])]
            scenario_occurrences[scenario] = occurrence + 1
        elif formal_eligibility:
            occurrence = scenario_occurrences.get(scenario, 0)
            eligible_uuid = formal_eligibility[
                occurrence % len(formal_eligibility)]
            by_uuid = {row["map_uuid"]: row for row in maps}
            if eligible_uuid not in by_uuid:
                raise RuntimeError(
                    f"eligible {split} map is outside split: "
                    f"{eligible_uuid}")
            map_row = by_uuid[eligible_uuid]
            scenario_occurrences[scenario] = occurrence + 1
        elif config.get("composition_derivation") == (
            "scenario_x_map_type_round_robin_v1"
        ):
            occurrence = scenario_occurrences.get(scenario, 0)
            map_row = maps[occurrence % len(maps)]
            scenario_occurrences[scenario] = occurrence + 1
        else:
            map_row = maps[min(
                index*len(maps)//total_sequences, len(maps)-1)]
        result.append({
            "sequence_id": (
                f"{'smoke' if smoke else 'formal'}_{split}_{index:07d}"),
            "suite": suite, "scenario": scenario,
            "map_uuid": map_row["map_uuid"],
            "frame_count": count,
            "seed": config["random_seeds"]["sequence_base"]
                    + index + (0 if split == "train" else 10_000_000),
        })
        remaining -= count
        index += 1
    return maps, result


def ensure_generation_plan(root, config, renderer_version):
    path = Path(root)/"generation_state/generation_plan.json"
    expected = {
        "dataset_version": config["dataset_version"],
        "protocol_version": PROTOCOL_VERSION,
        "config_hash": config["_file_hash"],
        "source_hash": config["frozen_hashes"]["source_hash"],
        "split_manifest_hash":
            config["frozen_hashes"]["split_manifest_hash"],
        "renderer_version": renderer_version,
        "renderer_required": "cuda",
    }
    if "_dynamic_motion_contract" in config:
        expected.update({
            "dynamic_motion_contract_version":
                config["_dynamic_motion_contract"]["contract_version"],
            "dynamic_motion_contract_hash":
                config["_dynamic_motion_contract"]["_file_hash"],
        })
    if "scene_spatial_sampling_contract" in config:
        expected.update({
            "scene_spatial_sampling_contract_version":
                config["scene_spatial_sampling_contract"]["version"],
            "scene_spatial_sampling_contract_hash":
                config["scene_spatial_sampling_contract"]["contract_hash"],
        })
    if path.is_file():
        actual = json.loads(path.read_text())
        if any(actual.get(key) != value
               for key, value in expected.items()):
            raise RuntimeError(
                "generation plan mismatch; refusing unsafe resume")
        hotfixes = actual.setdefault("compatible_hotfixes", [])
        changed = False
        existing_ids = {row.get("id") for row in hotfixes}
        for hotfix in COMPATIBLE_HOTFIXES:
            if hotfix["id"] not in existing_ids:
                hotfixes.append(hotfix)
                changed = True
        if changed:
            atomic_json(path, actual)
    else:
        journal = Path(root)/"generation_state/generation_journal.jsonl"
        if journal.is_file() and journal.stat().st_size:
            raise RuntimeError(
                "existing journal has no frozen generation plan; "
                "refusing legacy CPU resume")
        expected["compatible_hotfixes"] = list(COMPATIBLE_HOTFIXES)
        atomic_json(path, expected)
    return expected


def root_manifest(root, config):
    manifests = sorted(
        (Path(root)/"manifests/sequences").glob("*.json"))
    maps = sorted((Path(root)/"manifests/maps").glob("*.json"))
    generation_plan_path = (
        Path(root)/"generation_state/generation_plan.json")
    generation_plan = json.loads(generation_plan_path.read_text())
    value = {
        "dataset_version": config["dataset_version"],
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "authority_version": AUTHORITY_VERSION,
        "config_hash": config["_file_hash"],
        "split_manifest_hash":
            config["frozen_hashes"]["split_manifest_hash"],
        "source_hash": config["frozen_hashes"]["source_hash"],
        "parent_git_commit": config["parent_git_commit"],
        "sequences": [
            {"path": str(p.relative_to(root)), "sha256": sha256(p)}
            for p in manifests],
        "maps": [
            {"path": str(p.relative_to(root)), "sha256": sha256(p)}
            for p in maps],
        "test_generated": False, "blind_access_count": 0,
        "derived_esdf_authoritative": False,
        "runtime_random_sampling": False,
        "compatible_hotfixes":
            generation_plan.get("compatible_hotfixes", []),
        "authority_source_dataset":
            config.get("authority_source_dataset"),
        "authority_source_root_manifest_hash":
            config.get("authority_source_root_manifest_hash"),
    }
    if "_dynamic_motion_contract" in config:
        value.update({
            "dynamic_motion_contract_version":
                config["_dynamic_motion_contract"]["contract_version"],
            "dynamic_motion_contract_hash":
                config["_dynamic_motion_contract"]["_file_hash"],
        })
    if "scene_spatial_sampling_contract" in config:
        value.update({
            "scene_spatial_sampling_contract_version":
                config["scene_spatial_sampling_contract"]["version"],
            "scene_spatial_sampling_contract_hash":
                config["scene_spatial_sampling_contract"]["contract_hash"],
        })
    value["dataset_semantic_hash"] = hashlib.sha256(
        canonical(value)).hexdigest()
    atomic_json(Path(root)/"manifests/dataset_manifest.json", value)
    return value


def handle_signal(_signum, _frame):
    global STOP
    STOP = True


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument(
        "--scenario", default="",
        help="smoke-only scenario filter; forbidden for formal generation",
    )
    args = parser.parse_args(argv)
    if args.workers < 1 or args.num_shards < 1:
        parser.error("workers and num-shards must be positive")
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("invalid shard")
    config = load_config(args.config)
    if (
        config.get("_dynamic_motion_contract", {}).get(
            "contract_version") == CONTRACT_VERSION_V2_1
        and not config.get("_occlusion_eligibility_gate_passed", False)
    ):
        raise RuntimeError(
            "formal authority maps failed the frozen occlusion-map "
            "diversity gate; generation is fail-closed")
    root = Path(args.output).resolve()
    forbidden = {
        (ROOT/"data/phase8_authoritative_pilot_v1").resolve(),
        (ROOT/"data/phase8_dynamic_production").resolve(),
    }
    if root in forbidden or "test" in root.name.lower() or "blind" in root.name.lower():
        raise RuntimeError("forbidden output namespace")
    maps, tasks = tasks_for(config, args.split, args.smoke)
    if args.scenario:
        if not args.smoke:
            parser.error("--scenario is restricted to isolated smoke generation")
        tasks = [row for row in tasks if row["scenario"] == args.scenario]
        if not tasks:
            parser.error("scenario filter selected no smoke tasks")
    tasks = [
        row for index, row in enumerate(tasks)
        if index % args.num_shards == args.shard_id]
    print(json.dumps({
        "status": "DRY_RUN" if args.dry_run else "STARTING",
        "split": args.split, "maps": len(maps),
        "sequences": len(tasks),
        "frames": sum(row["frame_count"] for row in tasks),
        "workers_requested": args.workers,
        "device": args.device, "smoke": args.smoke,
    }, indent=2))
    if args.dry_run:
        return
    if args.device == "cpu":
        runtime_device = "cpu"
    else:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "formal CUDA renderer requested but CUDA is unavailable")
        runtime_device = f"cuda:{int(args.device)}"
    if root.exists() and not args.resume:
        raise RuntimeError("existing output requires --resume")
    root.mkdir(parents=True, exist_ok=True)
    for name in (
        "protocol", "geometry_authority", "derived_geometry",
        "static/train", "static/valid", "dynamic/train", "dynamic/valid",
        "certificates", "manifests/maps", "manifests/sequences",
        "generation_state/map_state", "generation_state/sequence_state",
        "generation_state/completion", "diagnostics", "logs", ".staging",
    ):
        (root/name).mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    ensure_generation_plan(
        root, config,
        RENDERER_VERSION if runtime_device.startswith("cuda")
        else "canonical_occupancy_numpy_raycast_v1")
    with GenerationLock(root, f"{args.split}_shard{args.shard_id}"):
        for stale in (root/".staging").glob("*"):
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()
            append_journal(root, {
                "event": "staging_recovered", "status": "complete",
                "path": str(stale.relative_to(root))})
        map_paths = {
            row["map_uuid"]: ensure_map(
                root, config, args.split, row, args.smoke)
            for row in maps}
        work = [
            (root, config, args.split, task,
             map_paths[task["map_uuid"]], runtime_device) for task in tasks]
        if runtime_device.startswith("cuda") and args.workers == 1:
            completed_count = 0
            last_summary = time.monotonic()
            for row in work:
                if STOP:
                    append_journal(root, {
                        "event": "interrupted", "status": "running",
                        "split": args.split})
                    raise KeyboardInterrupt
                try:
                    _sequence_worker(row)
                except Exception as error:
                    task = row[3]
                    append_journal(root, {
                        "event": "sequence_failed", "status": "failed",
                        "split": args.split,
                        "sequence_id": task["sequence_id"],
                        "suite": task["suite"],
                        "scenario": task["scenario"],
                        "map_uuid": task["map_uuid"],
                        "error": repr(error)})
                    raise
                completed_count += 1
                if time.monotonic()-last_summary >= 30:
                    print(json.dumps({
                        "status": "RUNNING", "split": args.split,
                        "sequences_complete_this_run": completed_count,
                        "sequences_this_run": len(work),
                        "renderer": RENDERER_VERSION,
                        "device": runtime_device,
                        "workers_active": 1,
                        "cuda_execution": "serial",
                    }, sort_keys=True), flush=True)
                    last_summary = time.monotonic()
        else:
            _run_process_pool(
                root, args.split, work, args.workers,
                runtime_device, args.fail_fast,
            )
        root_manifest(root, config)
        completion_name = (
            f"{args.split.upper()}_SPLIT_GENERATION_COMPLETE"
            if args.num_shards == 1 else
            f"{args.split.upper()}_SHARD_{args.shard_id}_"
            f"OF_{args.num_shards}_COMPLETE")
        completion = root/f"generation_state/completion/{completion_name}"
        completion.write_text(json.dumps({
            "split": args.split, "completed_ns": time.time_ns(),
            "frames": sum(row["frame_count"] for row in tasks),
        }, sort_keys=True)+"\n")
        train_marker = root/(
            "generation_state/completion/"
            "TRAIN_SPLIT_GENERATION_COMPLETE"
        )
        valid_marker = root/(
            "generation_state/completion/"
            "VALID_SPLIT_GENERATION_COMPLETE"
        )
        if train_marker.is_file() and valid_marker.is_file():
            (root/"generation_state/completion/FULL_GENERATION_COMPLETE"
             ).write_text(json.dumps({
                 "dataset_generation_complete": True,
                 "completed_ns": time.time_ns(),
                 "train_marker": train_marker.name,
                 "valid_marker": valid_marker.name,
             }, sort_keys=True)+"\n")
    print(json.dumps({"status": "PASS", "split": args.split}, indent=2))


if __name__ == "__main__":
    main()
