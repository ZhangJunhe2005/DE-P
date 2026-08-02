"""Detection-feasible natural-occlusion constructor.

Version 2.2 intentionally has no map-provenance input.  Proposals are derived
only from canonical occupancy, camera state, the motion contract, and the CUDA
renderer's per-actor visibility diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.spatial import cKDTree

from authoritative_dataset.continuous_v1 import ContinuousState, certify_curve
from authoritative_dataset.dynamic_motion_v2 import (
    CONTRACT_VERSION_V2_1,
    MotionSamplingError,
    _dense_times,
    _interpolate_path,
    actor_position,
)
from authoritative_dataset.occlusion_constructor_v2_1 import (
    OcclusionConstruction,
    _static_path_clear,
    _trajectory_valid_exact,
    natural_occlusion_pattern,
)
from authoritative_dataset.occlusion_geometry_v1 import exact_first_hit
from authoritative_dataset.state_semantics_v2 import UavSequenceState


CONSTRUCTOR_VERSION = "authority_sightline_occlusion_constructor_v2_2"
ACTOR_RADIUS_M = 0.30
CAMERA_FACE_CLEARANCE_M = 0.305
TANGENT_FRACTION = 0.85


@dataclass(frozen=True)
class SurfacePatch:
    voxel_offset: int
    voxel_index: tuple[int, int, int]
    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    tangent: tuple[float, float, float]
    horizontal_extent_m: float
    vertical_extent_m: float


def feasibility_interval(
    occluder_distance_m,
    horizontal_extent_m,
    vertical_extent_m,
    gap_frames,
    speed_mps,
    *,
    actor_radius_m=ACTOR_RADIUS_M,
    detection_distance_m=1.80,
    frame_period_s=0.10,
    tangent_fraction=TANGENT_FRACTION,
    static_clearance_m=0.05,
):
    """Return the closed feasible depth-ratio interval and its derivation.

    For an anchor at range ``r*d`` and velocity whose tangent component is
    ``alpha*v``, the infinite-line closest camera approach is ``alpha*r*d``.
    Full silhouette coverage requires the magnified patch to cover the actor
    diameter and the tangent distance traversed during the requested gap.
    """
    d = float(occluder_distance_m)
    width = float(horizontal_extent_m)
    height = float(vertical_extent_m)
    gap = int(gap_frames)
    speed = float(speed_mps)
    radius = float(actor_radius_m)
    alpha = float(tangent_fraction)
    if min(d, width, height, speed, alpha) <= 0 or gap not in (1, 2, 3):
        raise ValueError("non-positive geometry/speed or unsupported gap")
    tangent_speed = alpha * speed
    gap_duration = gap * float(frame_period_s)
    detection_upper = float(detection_distance_m) / (alpha * d)
    horizontal_lower = (
        2.0 * radius + tangent_speed * gap_duration
    ) / width
    temporal_upper = (
        2.0 * radius
        + tangent_speed * (gap + 1) * float(frame_period_s)
    ) / width
    vertical_lower = 2.0 * radius / height
    static_lower = 1.0 + (
        radius + float(static_clearance_m)
    ) / d
    lower = max(horizontal_lower, vertical_lower, static_lower, 1.0)
    upper = min(detection_upper, temporal_upper)
    return {
        "feasible": bool(lower <= upper),
        "lower_ratio": float(lower),
        "upper_ratio": float(upper),
        "terms": {
            "horizontal_shadow_lower": float(horizontal_lower),
            "vertical_shadow_lower": float(vertical_lower),
            "static_clearance_lower": float(static_lower),
            "detection_upper": float(detection_upper),
            "temporal_quantization_upper": float(temporal_upper),
            "tangent_speed_mps": float(tangent_speed),
            "gap_duration_s": float(gap_duration),
        },
    }


def legacy_v2_1_detection_lower_bound(
    ratio=6.4,
    voxel_resolution_m=0.1,
    camera_face_clearance_m=CAMERA_FACE_CLEARANCE_M,
    tangent_fraction=TANGENT_FRACTION,
):
    distance = float(camera_face_clearance_m) + .5 * float(
        voxel_resolution_m
    )
    return float(tangent_fraction) * float(ratio) * distance


def _contiguous_runs(values):
    values = sorted(set(int(value) for value in values))
    result = {}
    if not values:
        return result
    start = previous = values[0]
    for value in values[1:] + [None]:
        if value is not None and value == previous + 1:
            previous = value
            continue
        length = previous - start + 1
        center = .5 * (start + previous)
        for member in range(start, previous + 1):
            result[member] = (length, center)
        if value is not None:
            start = previous = value
    return result


def canonical_surface_patches(backend):
    """Extract connected exposed vertical patches from canonical occupancy."""
    cached = getattr(backend, "_occlusion_v2_2_surface_patches", None)
    if cached is not None:
        return cached
    occupied = np.asarray(backend.occupied, dtype=np.int64)
    lookup = {tuple(row) for row in occupied.tolist()}
    exposed = []
    for offset, voxel in enumerate(occupied):
        for axis in (0, 1):
            for sign in (-1, 1):
                neighbour = voxel.copy()
                neighbour[axis] += sign
                if tuple(neighbour) not in lookup:
                    exposed.append((offset, tuple(map(int, voxel)), axis, sign))

    horizontal_groups = {}
    vertical_groups = {}
    for _, voxel, axis, sign in exposed:
        tangent_axis = 1 - axis
        horizontal_groups.setdefault(
            (axis, sign, voxel[axis], voxel[2]), []
        ).append(voxel[tangent_axis])
        vertical_groups.setdefault(
            (axis, sign, voxel[axis], voxel[tangent_axis]), []
        ).append(voxel[2])
    horizontal_runs = {
        key: _contiguous_runs(values)
        for key, values in horizontal_groups.items()
    }
    vertical_runs = {
        key: _contiguous_runs(values)
        for key, values in vertical_groups.items()
    }

    resolution = float(backend.map.resolution)
    patches = []
    for offset, voxel_tuple, axis, sign in exposed:
        voxel = np.asarray(voxel_tuple, dtype=np.int64)
        tangent_axis = 1 - axis
        horizontal, horizontal_center = horizontal_runs[
            (axis, sign, voxel_tuple[axis], voxel_tuple[2])
        ][voxel_tuple[tangent_axis]]
        vertical, vertical_center = vertical_runs[
            (axis, sign, voxel_tuple[axis], voxel_tuple[tangent_axis])
        ][voxel_tuple[2]]
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = sign
        tangent = np.zeros(3, dtype=np.float64)
        tangent[tangent_axis] = 1.0
        center = (
            backend.map.origin
            + (voxel.astype(np.float64) + .5) * resolution
        )
        center[tangent_axis] = (
            backend.map.origin[tangent_axis]
            + (horizontal_center + .5) * resolution
        )
        center[2] = (
            backend.map.origin[2] + (vertical_center + .5) * resolution
        )
        patches.append(SurfacePatch(
            voxel_offset=int(offset),
            voxel_index=voxel_tuple,
            center=tuple(map(float, center)),
            normal=tuple(map(float, normal)),
            tangent=tuple(map(float, tangent)),
            horizontal_extent_m=float(horizontal * resolution),
            vertical_extent_m=float(vertical * resolution),
        ))
    unique = {}
    for patch in patches:
        key = (
            tuple(round(value, 8) for value in patch.center),
            patch.normal,
            round(patch.horizontal_extent_m, 8),
            round(patch.vertical_extent_m, 8),
        )
        current = unique.get(key)
        if current is None or patch.voxel_offset < current.voxel_offset:
            unique[key] = patch
    patches = sorted(
        unique.values(), key=lambda patch: patch.voxel_offset
    )
    backend._occlusion_v2_2_surface_patches = patches
    return patches


def _ratios(interval):
    low = interval["lower_ratio"]
    high = interval["upper_ratio"]
    if low > high:
        return ()
    middle = .5 * (low + high)
    return tuple(dict.fromkeys(round(value, 8) for value in (
        low, middle, high
    )))


def _candidate_camera(backend, patch):
    center = np.asarray(patch.center, dtype=np.float64)
    normal = np.asarray(patch.normal, dtype=np.float64)
    face = center + normal * (.5 * float(backend.map.resolution))
    base = face + normal * CAMERA_FACE_CLEARANCE_M
    goal = base + normal * 2.2
    if backend.query_one(base, .30)["collision"]:
        return None
    if backend.query_one(goal, .30)["collision"]:
        return None
    reference = lambda value, b=base, n=normal: (
        b + n * 2.2 * (
            10 * (value / 1.7) ** 3
            - 15 * (value / 1.7) ** 4
            + 6 * (value / 1.7) ** 5
        )
    )
    certificate = certify_curve(reference, backend, 1.7, max_depth=14)
    if certificate.state is not ContinuousState.CERTIFIED_SAFE:
        return None
    return base, goal, normal, certificate


def sample_occlusion_uav_sequence_v2_2(
    backend,
    rng,
    frame_times,
    requested_gap_frames,
    maximum_candidates=4096,
):
    """Select a mathematically feasible natural patch and safe camera pose."""
    frame_times = np.asarray(frame_times, dtype=np.float64)
    gap = int(requested_gap_frames)
    patches = canonical_surface_patches(backend)
    if not patches:
        raise MotionSamplingError("map has no exposed vertical authority patch")
    order = rng.permutation(len(patches))
    gap_center = min(
        10, len(frame_times) - gap - 3
    ) + .5 * (gap - 1)
    center_time = float(np.interp(
        gap_center, np.arange(len(frame_times)), frame_times
    ))
    end_time = float(frame_times[-1] + 1.7)
    validation_times = _dense_times(end_time, 1.70, .05)
    voxel_centers = (
        backend.map.origin[None, :]
        + (backend.occupied.astype(np.float64) + .5)
        * backend.map.resolution
    )
    occupancy_tree = cKDTree(voxel_centers)
    for patch_index in order[:maximum_candidates]:
        patch = patches[int(patch_index)]
        nominal_distance = (
            CAMERA_FACE_CLEARANCE_M
            + .5 * float(backend.map.resolution)
        )
        if not any(feasibility_interval(
            nominal_distance, patch.horizontal_extent_m,
            patch.vertical_extent_m, gap, speed,
        )["feasible"] for speed in (1.40, 1.55, 1.70)):
            continue
        camera = _candidate_camera(backend, patch)
        if camera is None:
            continue
        base, goal, normal, certificate = camera
        distance = float(np.linalg.norm(
            np.asarray(patch.center) - base
        ))
        ray = np.asarray(patch.center) - base
        ray /= np.linalg.norm(ray)
        tangent = np.asarray(patch.tangent, dtype=np.float64)
        tangent -= ray * np.dot(tangent, ray)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm < 1e-8:
            continue
        tangent /= tangent_norm
        viable = False
        for speed in (1.40, 1.55, 1.70):
            interval = feasibility_interval(
                distance, patch.horizontal_extent_m,
                patch.vertical_extent_m, gap, speed,
            )
            for ratio in _ratios(interval):
                anchor = base + ray * (ratio * distance)
                tangent_speed = speed * TANGENT_FRACTION
                radial_speed = np.sqrt(max(
                    0.0, speed * speed - tangent_speed * tangent_speed
                ))
                for side in (-1.0, 1.0):
                    velocity = (
                        ray * radial_speed + side * tangent * tangent_speed
                    )
                    path = (
                        anchor - velocity * center_time
                        + validation_times[:, None] * velocity
                    )
                    camera_path = np.repeat(
                        base[None, :], len(path), axis=0
                    )
                    early = validation_times <= 1.2 + 1e-12
                    if (
                        _static_path_clear(
                            backend, occupancy_tree, path, .35
                        )
                        and float(np.min(np.linalg.norm(
                            path - camera_path, axis=1
                        ))) > .70
                        and float(np.min(np.linalg.norm(
                            path[early] - camera_path[early], axis=1
                        ))) <= 1.80
                    ):
                        viable = True
                        break
                if viable:
                    break
            if viable:
                break
        if not viable:
            continue
        backend._occlusion_v2_2_preferred_patch = patch
        positions = np.repeat(base[None, :], len(frame_times), axis=0)
        zeros = np.zeros_like(positions)
        view = -normal
        yaw = float(np.arctan2(view[1], view[0]))
        return UavSequenceState(
            position_world=positions,
            velocity_world=zeros,
            acceleration_world=zeros,
            yaw=np.full(len(frame_times), yaw, dtype=np.float64),
            goal_world=np.repeat(goal[None, :], len(frame_times), axis=0),
            reference_certificate=certificate,
            reference_goal_world=goal,
        )
    raise MotionSamplingError(
        "no detection-feasible natural authority patch for camera")


def _visible_patch_candidates(backend, camera, yaw, sensor):
    patches = canonical_surface_patches(backend)
    camera = np.asarray(camera, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    forward_axis = np.asarray([c, s, 0.0])
    lateral_axis = np.asarray([-s, c, 0.0])
    fx, fy, cx, cy = sensor["intrinsics"]
    horizontal_limit = max(cx, sensor["width"] - 1 - cx) / fx
    vertical_limit = max(cy, sensor["height"] - 1 - cy) / fy
    selected = []
    for patch in patches:
        delta = np.asarray(patch.center) - camera
        forward = float(delta @ forward_axis)
        lateral = float(delta @ lateral_axis)
        vertical = float(delta[2])
        if (
            forward > .30
            and forward < min(3.0, float(sensor["max_depth_m"]) / 4.0)
            and abs(lateral) < forward * horizontal_limit * .90
            and abs(vertical) < forward * vertical_limit * .90
        ):
            selected.append((float(np.linalg.norm(delta)), patch))
    selected.sort(key=lambda row: (
        row[0], row[1].voxel_offset, row[1].normal
    ))
    result = [row[1] for row in selected]
    preferred = getattr(
        backend, "_occlusion_v2_2_preferred_patch", None
    )
    if preferred is not None:
        result.sort(key=lambda patch: (
            patch != preferred,
            float(np.linalg.norm(np.asarray(patch.center) - camera)),
            patch.voxel_offset,
        ))
    return result


def build_occluded_actor_specs_v2_2(
    backend,
    renderer,
    rng,
    uav_positions,
    yaws,
    frame_times,
    contract,
    requested_gap_frames,
    method="deterministic",
    maximum_candidates=128,
):
    """Construct and independently raster-check a natural occlusion actor."""
    if contract["contract_version"] != CONTRACT_VERSION_V2_1:
        raise RuntimeError("v2.2 requires the frozen v2.1 motion contract")
    gap = int(requested_gap_frames)
    if gap not in (1, 2, 3):
        raise ValueError("requested gap must be 1, 2, or 3 frames")
    if method not in ("deterministic", "random_seeded"):
        raise ValueError("unknown bounded constructor method")
    started = time.perf_counter()
    positions = np.asarray(uav_positions, dtype=np.float64)
    yaws = np.asarray(yaws, dtype=np.float64)
    frame_times = np.asarray(frame_times, dtype=np.float64)
    pre_min = int(contract["occlusion"]["pre_gap_visible_min"])
    post_min = int(contract["occlusion"]["post_gap_visible_min"])
    starts = list(range(pre_min, len(frame_times) - gap - post_min))
    preferred = min(10, len(frame_times) - gap - post_min - 1)
    starts.sort(key=lambda value: (abs(value - preferred), value))
    if method == "random_seeded":
        starts = [starts[index] for index in rng.permutation(len(starts))]
    profile = contract["scenarios"]["occluded_but_tracked"]
    low, high = map(float, profile["speed_range_mps"])
    speeds = (low, .5 * (low + high), high)
    voxel_centers = (
        backend.map.origin[None, :]
        + (backend.occupied.astype(np.float64) + .5)
        * backend.map.resolution
    )
    occupancy_tree = cKDTree(voxel_centers)
    end_time = float(frame_times[-1] + 1.7)
    attempts = 0
    rejected = {
        "trajectory": 0,
        "raster_gap": 0,
        "authority_sightline": 0,
    }
    observed_gap_lengths = {}
    for gap_start in starts:
        gap_center = gap_start + .5 * (gap - 1)
        center_time = float(np.interp(
            gap_center, np.arange(len(frame_times)), frame_times
        ))
        camera = _interpolate_path(
            np.asarray([center_time]), frame_times, positions
        )[0]
        yaw = float(np.interp(
            center_time, frame_times, np.unwrap(yaws)
        ))
        patches = _visible_patch_candidates(
            backend, camera, yaw, renderer.sensor
        )
        if method == "random_seeded" and patches:
            order = rng.permutation(len(patches))
            patches = [patches[index] for index in order]
        for patch in patches[:maximum_candidates]:
            ray = np.asarray(patch.center) - camera
            distance = float(np.linalg.norm(ray))
            if distance <= 0:
                continue
            ray /= distance
            tangent = np.asarray(patch.tangent, dtype=np.float64)
            tangent -= ray * np.dot(tangent, ray)
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm < 1e-8:
                continue
            tangent /= tangent_norm
            for speed in speeds:
                interval = feasibility_interval(
                    distance, patch.horizontal_extent_m,
                    patch.vertical_extent_m, gap, speed,
                    detection_distance_m=float(contract["sampling"][
                        "maximum_detection_center_distance_m"
                    ]),
                )
                for ratio in _ratios(interval):
                    anchor = camera + ray * (ratio * distance)
                    tangent_speed = speed * TANGENT_FRACTION
                    radial_speed = np.sqrt(max(
                        0.0, speed * speed - tangent_speed * tangent_speed
                    ))
                    for side in (-1.0, 1.0):
                        attempts += 1
                        velocity = ray * radial_speed + side * tangent * tangent_speed
                        actor = {
                            "actor_id": 0,
                            "start": anchor - velocity * center_time,
                            "velocity": velocity,
                            "radius_m": ACTOR_RADIUS_M,
                            "motion_profile": "constant_velocity",
                            "configured_speed_mps": float(speed),
                            "sampling_method": f"{CONSTRUCTOR_VERSION}:{method}",
                            "motion_contract_version": CONTRACT_VERSION_V2_1,
                        }
                        validation_times = _dense_times(
                            end_time, speed, float(contract["sampling"][
                                "continuous_sample_spacing_m"
                            ])
                        )
                        actor["validation_times"] = validation_times
                        path = actor_position(actor, validation_times)
                        camera_path = _interpolate_path(
                            validation_times, frame_times, positions
                        )
                        if not _trajectory_valid_exact(
                            backend, occupancy_tree, path, camera_path,
                            ACTOR_RADIUS_M, contract["sampling"],
                            validation_times,
                        ):
                            rejected["trajectory"] += 1
                            continue
                        actor_frames = actor_position(
                            actor, frame_times
                        )[:, None, :]
                        diagnostics = renderer.render_with_actor_diagnostics(
                            backend, positions, yaws, actor_frames,
                            [ACTOR_RADIUS_M], return_owner_map=False,
                            return_actor_near_depth=False,
                        )
                        pattern = natural_occlusion_pattern(
                            diagnostics, 0, contract
                        )
                        for observed_start, observed_end in pattern[
                            "accepted_gaps"
                        ]:
                            observed_length = (
                                int(observed_end) - int(observed_start) + 1
                            )
                            observed_gap_lengths[observed_length] = (
                                observed_gap_lengths.get(observed_length, 0) + 1
                            )
                        match = next((
                            item for item in pattern["accepted_gaps"]
                            if item[1] - item[0] + 1 == gap
                        ), None)
                        if match is None:
                            rejected["raster_gap"] += 1
                            continue
                        evidence = exact_first_hit(
                            backend, positions[match[0]],
                            actor_frames[match[0], 0],
                        )
                        if evidence is None:
                            rejected["authority_sightline"] += 1
                            continue
                        actor["occlusion_constructor"] = {
                            "version": CONSTRUCTOR_VERSION,
                            "requested_gap_frames": gap,
                            "gap_start": int(match[0]),
                            "gap_end": int(match[1]),
                            "occluder_voxel_index": evidence["voxel_index"],
                            "occluder_authority_hash": evidence["authority_hash"],
                            "occluder_distance_m": evidence["distance_m"],
                            "depth_ratio": float(ratio),
                            "horizontal_extent_m":
                                patch.horizontal_extent_m,
                            "vertical_extent_m": patch.vertical_extent_m,
                            "feasibility_interval": interval,
                            "method": method,
                        }
                        return OcclusionConstruction(
                            [actor], diagnostics, evidence, attempts,
                            time.perf_counter() - started, gap,
                            int(match[0]), int(match[1]), method,
                        )
    raise MotionSamplingError(
        "bounded v2.2 natural-occlusion construction exhausted "
        f"after {attempts} raster candidates; rejected={rejected}; "
        f"observed_gap_lengths={observed_gap_lengths}"
    )


__all__ = [
    "ACTOR_RADIUS_M",
    "CAMERA_FACE_CLEARANCE_M",
    "CONSTRUCTOR_VERSION",
    "TANGENT_FRACTION",
    "SurfacePatch",
    "build_occluded_actor_specs_v2_2",
    "canonical_surface_patches",
    "feasibility_interval",
    "legacy_v2_1_detection_lower_bound",
    "sample_occlusion_uav_sequence_v2_2",
]
