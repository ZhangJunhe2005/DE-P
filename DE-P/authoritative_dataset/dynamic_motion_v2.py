"""Versioned deterministic actor-motion contract and swept-volume sampler."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


CONTRACT_VERSION = "authoritative_dynamic_motion_contract_v2"
CONTRACT_VERSION_V2_1 = "authoritative_dynamic_motion_contract_v2_1"
SUPPORTED_CONTRACT_VERSIONS = {CONTRACT_VERSION, CONTRACT_VERSION_V2_1}


class MotionSamplingError(RuntimeError):
    """The sampled UAV/map window cannot host this motion contract."""


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def load_motion_contract(path):
    path = Path(path).expanduser().resolve()
    value = yaml.safe_load(path.read_text())
    if value.get("contract_version") not in SUPPORTED_CONTRACT_VERSIONS:
        raise RuntimeError("dynamic motion contract version mismatch")
    frozen = value["frozen_perception"]
    sampling = value["sampling"]
    if frozen["dynamic_enter_speed_mps"] != 0.30:
        raise RuntimeError("frozen dynamic-enter threshold must remain 0.30 m/s")
    if frozen["dynamic_exit_speed_mps"] != 0.15:
        raise RuntimeError("frozen dynamic-exit threshold must remain 0.15 m/s")
    required_minimum = (
        frozen["dynamic_enter_speed_mps"]
        + sampling["minimum_speed_margin_above_enter_mps"]
    )
    for scenario, profile in value["scenarios"].items():
        low, high = map(float, profile["speed_range_mps"])
        if low < required_minimum or high < low:
            raise RuntimeError(
                f"{scenario} speed range violates authoritative margin"
            )
        if int(profile["actor_count"]) < 1:
            raise RuntimeError(f"{scenario} actor_count must be positive")
        if profile["motion_profile"] != "constant_velocity":
            raise RuntimeError("only validated constant-velocity profile is allowed")
    if value["contract_version"] == CONTRACT_VERSION_V2_1:
        occlusion = value["occlusion"]
        forbidden = (
            "artificial_frame_hiding", "metadata_only_occlusion",
            "fov_exit_allowed", "behind_camera_allowed",
            "max_depth_exit_allowed",
        )
        if any(bool(occlusion[key]) for key in forbidden):
            raise RuntimeError("v2.1 forbids artificial/non-static occlusion")
        if (
            occlusion["source"]
            != "canonical_occupancy_sensor_visibility"
            or not occlusion["actor_only_footprint_required"]
            or not occlusion["fully_static_blocked_required"]
            or not occlusion["same_track_id_required"]
            or not occlusion["prediction_track_required_during_gap"]
        ):
            raise RuntimeError("v2.1 natural-occlusion provenance mismatch")
        derived = max(
            int(frozen["confirmation_hits"]),
            int(frozen["dynamic_min_confirmed_hits"]),
            int(frozen["motion_consistency_frames"]) + 1,
        ) + int(occlusion["pre_gap_guard_frames"])
        if int(occlusion["pre_gap_visible_min"]) != derived or derived < 4:
            raise RuntimeError("pre-gap visibility minimum is not derived")
        if int(occlusion["post_gap_visible_min"]) < 2:
            raise RuntimeError("post-gap visibility minimum must be at least 2")
    value["_path"] = str(path)
    value["_file_hash"] = hashlib.sha256(path.read_bytes()).hexdigest()
    value["_semantic_hash"] = canonical_hash({
        key: item for key, item in value.items() if not key.startswith("_")
    })
    return value


def actor_position(actor, times):
    times = np.asarray(times, dtype=np.float64)
    return (
        np.asarray(actor["start"], dtype=np.float64)[None, :]
        + times[:, None] * np.asarray(actor["velocity"], dtype=np.float64)
    )


def _interpolate_path(times, frame_times, positions):
    return np.stack([
        np.interp(times, frame_times, positions[:, axis],
                  right=positions[-1, axis])
        for axis in range(3)
    ], axis=1)


def _dense_times(end_time, maximum_speed, spacing):
    count = max(2, int(np.ceil(end_time * maximum_speed / spacing)) + 1)
    return np.linspace(0.0, end_time, count, dtype=np.float64)


def _direction(scenario, actor_id, forward, lateral):
    if scenario == "crossing":
        vector = -0.72*forward + 0.694*lateral
        return vector / np.linalg.norm(vector)
    if scenario == "head_on":
        return -forward
    if scenario == "temporal_separation":
        vector = -0.78*forward + 0.626*lateral
        return vector / np.linalg.norm(vector)
    if scenario == "static_dynamic_joint_constraint":
        vector = -0.65 * forward + 0.76 * lateral
        return vector / np.linalg.norm(vector)
    if scenario == "multi_target":
        vector = (
            -forward if actor_id == 0
            else -0.95*forward + 0.312*lateral
        )
        return vector / np.linalg.norm(vector)
    if scenario == "occluded_but_tracked":
        vector = -0.72*forward + 0.694*lateral
        return vector / np.linalg.norm(vector)
    raise ValueError(f"unsupported dynamic scenario: {scenario}")


def _trajectory_valid(
    backend, path, camera_path, actors, radius, settings, validation_times
):
    uav_limit = (
        radius + float(settings["uav_radius_m"])
        + float(settings["actor_uav_clearance_margin_m"])
    )
    if float(np.min(np.linalg.norm(path-camera_path, axis=1))) <= uav_limit:
        return False
    entry_time = float(settings["maximum_detection_entry_time_s"])
    detection = validation_times <= entry_time + 1e-12
    detection_prefix = path[detection]
    camera_prefix = camera_path[detection]
    if float(np.min(np.linalg.norm(
        detection_prefix-camera_prefix, axis=1
    ))) > float(settings["maximum_detection_center_distance_m"]):
        return False
    actor_limit = (
        2.0*radius + float(settings["actor_actor_clearance_margin_m"])
    )
    for actor in actors:
        other = actor_position(actor, actor["validation_times"])
        if float(np.min(np.linalg.norm(path-other, axis=1))) <= actor_limit:
            return False
    # Keep the authoritative exact-geometry query last.  The preceding checks
    # are contract-equivalent, inexpensive array operations and reject most
    # infeasible lattice candidates without traversing the BVH.
    static_radius = radius + float(settings["actor_static_clearance_margin_m"])
    if any(backend.query_one(point, static_radius)["collision"] for point in path):
        return False
    return True


def _multi_target_direction_candidates(actor_id, forward, lateral):
    """Deterministic, distinct closing directions for the two actors.

    The original v2 fallback bound each actor to one direction.  A narrow cave
    can contain valid constant-velocity tracks whose heading is aligned with a
    different free corridor, while that single coupled pair is infeasible.
    Every direction below remains closing and the two ordered families retain
    opposite transverse components.
    """
    side = -1.0 if actor_id == 0 else 1.0
    values = (
        -forward,
        -0.95 * forward + side * 0.312 * lateral,
        -0.85 * forward + side * 0.527 * lateral,
        -0.70 * forward + side * 0.714 * lateral,
    )
    return tuple(value / np.linalg.norm(value) for value in values)


def _multi_target_independent_pair_fallback(
    backend, camera_path, uav_positions, yaw, frame_times, profile,
    settings, radius, low, high, validation_times,
):
    """Find two independently feasible tracks, then pair them exactly.

    This is a bounded deterministic feasibility solver, not a contract
    relaxation.  Individual candidates pass the same swept static, UAV and
    detection checks as the random sampler.  A returned pair additionally
    passes the authoritative continuous actor-to-actor clearance check.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    forward = np.asarray([c, s, 0.0], dtype=np.float64)
    lateral = np.asarray([-s, c, 0.0], dtype=np.float64)
    speed_values = (0.5 * (low + high), low, high)
    candidate_limit = 32

    def candidates_for(actor_id):
        candidates = []
        side = -1.0 if actor_id == 0 else 1.0
        directions = _multi_target_direction_candidates(
            actor_id, forward, lateral)
        for event_time in (.55, .75, .95, 1.15):
            camera_event = _interpolate_path(
                np.asarray([event_time]), frame_times, uav_positions
            )[0]
            for lateral_distance in (.60, .82, 1.05, 1.30):
                for forward_distance in (.40, .60, .82):
                    for vertical_distance in (0.0, .28, -.28, .52, -.52):
                        target = (
                            camera_event + forward * forward_distance
                            + lateral * side * lateral_distance
                            + np.asarray(
                                [0.0, 0.0, vertical_distance],
                                dtype=np.float64,
                            )
                        )
                        for direction in directions:
                            for speed in speed_values:
                                actor = {
                                    "actor_id": actor_id,
                                    "start": target-direction*speed*event_time,
                                    "velocity": direction*speed,
                                    "radius_m": radius,
                                    "motion_profile":
                                        profile["motion_profile"],
                                    "configured_speed_mps": float(speed),
                                    "sampling_method": (
                                        "deterministic_independent_pair_"
                                        "lattice_v2_1"
                                    ),
                                    "motion_contract_version":
                                        CONTRACT_VERSION_V2_1,
                                    "validation_times": validation_times,
                                }
                                path = actor_position(
                                    actor, validation_times)
                                if not _trajectory_valid(
                                    backend, path, camera_path, [], radius,
                                    settings, validation_times,
                                ):
                                    continue
                                candidates.append((actor, path))
                                if len(candidates) >= candidate_limit:
                                    return candidates
        return candidates

    first = candidates_for(0)
    second = candidates_for(1)
    actor_limit = (
        2.0*radius + float(settings["actor_actor_clearance_margin_m"])
    )
    for actor_0, path_0 in first:
        for actor_1, path_1 in second:
            if float(np.min(np.linalg.norm(
                path_0-path_1, axis=1
            ))) > actor_limit:
                return [actor_0, actor_1]
    return None


def build_actor_specs_v2(
    backend, rng, scenario, uav_positions, yaw, frame_times, contract
):
    """Sample complete, continuously checked constant-velocity trajectories."""
    if scenario == "no_target":
        return []
    profile = contract["scenarios"][scenario]
    settings = contract["sampling"]
    count = int(profile["actor_count"])
    radius = float(profile.get(
        "actor_radius_m", settings["actor_radius_m"]
    ))
    low, high = map(float, profile["speed_range_mps"])
    end_time = float(frame_times[-1] + 1.7)
    validation_times = _dense_times(
        end_time, high, float(settings["continuous_sample_spacing_m"])
    )
    uav_positions = np.asarray(uav_positions, dtype=np.float64)
    camera_path = _interpolate_path(
        validation_times, frame_times, uav_positions
    )
    c, s = np.cos(yaw), np.sin(yaw)
    forward = np.asarray([c, s, 0.0], dtype=np.float64)
    lateral = np.asarray([-s, c, 0.0], dtype=np.float64)
    def attempt(starts, speeds, method):
        actors = []
        for actor_id in range(count):
            direction = _direction(
                scenario, actor_id, forward, lateral
            )
            velocity = direction * speeds[actor_id]
            actor = {
                "actor_id": actor_id,
                "start": np.asarray(starts[actor_id], dtype=np.float64),
                "velocity": velocity,
                "radius_m": radius,
                "motion_profile": profile["motion_profile"],
                "configured_speed_mps": float(speeds[actor_id]),
                "sampling_method": method,
                "motion_contract_version": CONTRACT_VERSION,
                "validation_times": validation_times,
            }
            path = actor_position(actor, validation_times)
            if not _trajectory_valid(
                backend, path, camera_path, actors, radius, settings,
                validation_times,
            ):
                return None
            actors.append(actor)
        return actors

    maximum_attempts = int(settings["maximum_random_attempts"])
    for _ in range(maximum_attempts):
        starts, speeds = [], []
        event_time = float(rng.uniform(0.65, 1.15))
        camera_event = _interpolate_path(
            np.asarray([event_time]), frame_times, uav_positions
        )[0]
        for actor_id in range(count):
            speed = float(rng.uniform(low, high))
            direction = _direction(
                scenario, actor_id, forward, lateral
            )
            side = -1.0 if actor_id == 0 else 1.0
            if count == 1:
                side = float(rng.choice((-1.0, 1.0)))
            target = (
                camera_event
                + forward*float(rng.uniform(.45, .70))
                + lateral*side*float(rng.uniform(.72, .95))
                + np.asarray([0.0, 0.0, float(rng.uniform(-.15, .15))])
            )
            starts.append(target-direction*speed*event_time)
            speeds.append(speed)
        result = attempt(starts, speeds, "contract_random_swept_volume_v2")
        if result is not None:
            return result

    # Deterministic fallback is a finite feasibility construction. It never
    # lowers speed, injects jitter, or substitutes a stationary actor.
    speed_values = (low, 0.5*(low+high), high)
    for event_time in (.70, .90, 1.10):
        camera_event = _interpolate_path(
            np.asarray([event_time]), frame_times, uav_positions
        )[0]
        for forward_distance in (.45, .60):
            for lateral_distance in (.75, .90, 1.10):
                for vertical_distance in (0.0, .25, -.25):
                    for speed_index, speed in enumerate(speed_values):
                        starts, speeds = [], []
                        for actor_id in range(count):
                            side = -1.0 if actor_id == 0 else 1.0
                            target = (
                                camera_event + forward*forward_distance
                                + lateral*side*lateral_distance
                                + np.asarray([
                                    0.0, 0.0,
                                    vertical_distance
                                    * (-1.0 if actor_id else 1.0),
                                ])
                            )
                            actor_speed = speed_values[
                                (speed_index+actor_id) % len(speed_values)
                            ]
                            direction = _direction(
                                scenario, actor_id, forward, lateral
                            )
                            starts.append(
                                target-direction*actor_speed*event_time
                            )
                            speeds.append(actor_speed)
                        result = attempt(
                            starts, speeds,
                            "deterministic_swept_volume_lattice_v2",
                        )
                        if result is not None:
                            return result
    if scenario == "multi_target":
        result = _multi_target_independent_pair_fallback(
            backend, camera_path, uav_positions, yaw, frame_times, profile,
            settings, radius, low, high, validation_times,
        )
        if result is not None:
            return result
    raise MotionSamplingError(
        f"no contract-valid swept-volume actor trajectory: {scenario}"
    )
