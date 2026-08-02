"""Deterministic authority-sightline constructor for natural occlusion."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.spatial import cKDTree

from authoritative_dataset.dynamic_motion_v2 import (
    CONTRACT_VERSION_V2_1, MotionSamplingError, _dense_times,
    _interpolate_path, actor_position,
)
from authoritative_dataset.continuous_v1 import ContinuousState, certify_curve
from authoritative_dataset.occlusion_geometry_v1 import exact_first_hit
from authoritative_dataset.state_semantics_v2 import UavSequenceState


CONSTRUCTOR_VERSION = "authority_sightline_occlusion_constructor_v2_1"


@dataclass
class OcclusionConstruction:
    actors: list[dict]
    diagnostics: dict
    evidence: dict
    attempts: int
    elapsed_seconds: float
    requested_gap_frames: int
    gap_start: int
    gap_end: int
    method: str


def sample_occlusion_uav_sequence_v2_1(
    backend, rng, frame_times, requested_gap_frames,
    maximum_candidates=4096,
):
    """Place a safe camera path from authority geometry, never metadata hiding."""
    occupied_set = {tuple(row) for row in backend.occupied.tolist()}
    candidates = []
    for voxel in backend.occupied:
        center = (
            backend.map.origin
            + (voxel.astype(np.float64)+.5)*backend.map.resolution
        )
        if np.any(center < backend.map.bounds_min+1.0) \
                or np.any(center > backend.map.bounds_max-1.0):
            continue
        for axis in (0, 1):
            for sign in (-1, 1):
                neighbour = voxel.copy()
                neighbour[axis] += sign
                if tuple(neighbour) in occupied_set:
                    continue
                candidates.append((voxel.copy(), axis, sign))
    if not candidates:
        raise MotionSamplingError("map has no horizontal exposed occluder")
    voxel_centers = (
        backend.map.origin[None, :]
        + (backend.occupied.astype(np.float64)+.5)
        * backend.map.resolution
    )
    tree = cKDTree(voxel_centers)
    gap_length = int(requested_gap_frames)
    preferred_start = min(
        max(8, 10),
        len(frame_times)-gap_length-3)
    gap_time = float(frame_times[preferred_start]) \
        + .5*(gap_length-1)*float(np.median(np.diff(frame_times)))
    end_time = float(frame_times[-1]+1.7)
    validation_times = _dense_times(end_time, 1.70, .05)
    order = rng.permutation(len(candidates))
    for candidate_index in order[:maximum_candidates]:
        voxel, axis, sign = candidates[int(candidate_index)]
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = sign
        face = (
            backend.map.origin
            + (voxel.astype(np.float64)+.5)*backend.map.resolution
        )
        face[axis] += sign*backend.map.resolution/2
        base = face+normal*.305
        if backend.query_one(base, .30)["collision"]:
            continue
        goal = base+normal*2.2
        if backend.query_one(goal, .30)["collision"]:
            continue
        reference = lambda value, b=base, n=normal: (
            b+n*2.2*(
                10*(value/1.7)**3
                - 15*(value/1.7)**4
                + 6*(value/1.7)**5
            )
        )
        certificate = certify_curve(
            reference, backend, 1.7, max_depth=14)
        if certificate.state is not ContinuousState.CERTIFIED_SAFE:
            continue
        ray = (
            backend.map.origin
            + (voxel.astype(np.float64)+.5)*backend.map.resolution
            - base
        )
        distance = float(np.linalg.norm(ray))
        ray /= distance
        tangent = np.zeros(3, dtype=np.float64)
        tangent[1-axis] = 1.0
        viable_actor_path = False
        for ratio in {
            1: (6.4, 7.0),
            2: (7.0, 7.8),
            3: (8.5, 7.8),
        }[gap_length]:
            anchor = base+ray*(ratio*distance)
            for speed in (1.40, 1.55, 1.70):
                tangent_speed = speed*.85
                radial_speed = np.sqrt(
                    speed*speed-tangent_speed*tangent_speed)
                for side in (-1.0, 1.0):
                    velocity = (
                        ray*radial_speed+tangent*side*tangent_speed)
                    start = anchor-velocity*gap_time
                    path = (
                        start[None, :]
                        + validation_times[:, None]*velocity)
                    camera_path = np.repeat(
                        base[None, :], len(path), axis=0)
                    if (
                        _static_path_clear(backend, tree, path, .35)
                        and np.min(np.linalg.norm(
                            path-camera_path, axis=1)) > .70
                        and np.min(np.linalg.norm(
                            path[validation_times <= 1.2]
                            - camera_path[validation_times <= 1.2],
                            axis=1)) <= 1.80
                    ):
                        viable_actor_path = True
                        break
                if viable_actor_path:
                    break
            if viable_actor_path:
                break
        if not viable_actor_path:
            continue
        frame_times = np.asarray(frame_times, dtype=np.float64)
        positions = np.repeat(base[None, :], len(frame_times), axis=0)
        zeros = np.zeros_like(positions)
        view = -normal
        yaw = float(np.arctan2(view[1], view[0]))
        yaws = np.full(len(frame_times), yaw, dtype=np.float64)
        goals = np.repeat(goal[None, :], len(frame_times), axis=0)
        return UavSequenceState(
            position_world=positions,
            velocity_world=zeros,
            acceleration_world=zeros,
            yaw=yaws,
            goal_world=goals,
            reference_certificate=certificate,
            reference_goal_world=goal,
        )
    raise MotionSamplingError(
        "bounded occlusion camera-path construction exhausted")


def _static_path_clear(backend, tree, path, radius):
    path = np.asarray(path, dtype=np.float64)
    if np.any(path-radius <= backend.map.bounds_min[None, :]) \
            or np.any(path+radius >= backend.map.bounds_max[None, :]):
        return False
    search = radius + np.sqrt(3.0)*backend.map.resolution/2
    for point, neighbours in zip(
        path, tree.query_ball_point(path, search)
    ):
        if not neighbours:
            continue
        delta = np.maximum(np.maximum(
            backend.minimum[neighbours]-point,
            point-backend.maximum[neighbours]), 0)
        if np.any(np.linalg.norm(delta, axis=1) <= radius+1e-9):
            return False
    return True


def _trajectory_valid_exact(
    backend, tree, path, camera_path, radius, settings, validation_times
):
    static_radius = radius + float(
        settings["actor_static_clearance_margin_m"])
    if not _static_path_clear(backend, tree, path, static_radius):
        return False
    uav_limit = (
        radius + float(settings["uav_radius_m"])
        + float(settings["actor_uav_clearance_margin_m"])
    )
    if float(np.min(np.linalg.norm(path-camera_path, axis=1))) <= uav_limit:
        return False
    detection = (
        validation_times
        <= float(settings["maximum_detection_entry_time_s"])+1e-12
    )
    return float(np.min(np.linalg.norm(
        path[detection]-camera_path[detection], axis=1
    ))) <= float(settings["maximum_detection_center_distance_m"])


def _candidate_voxels(backend, camera, yaw, sensor):
    centers = (
        backend.map.origin[None, :]
        + (backend.occupied.astype(np.float64)+0.5)
        * backend.map.resolution
    )
    delta = centers-np.asarray(camera, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    forward_axis = np.asarray([c, s, 0.0])
    lateral_axis = np.asarray([-s, c, 0.0])
    forward = delta@forward_axis
    lateral = delta@lateral_axis
    vertical = delta[:, 2]
    fx, fy, cx, cy = sensor["intrinsics"]
    horizontal_limit = max(cx, sensor["width"]-1-cx)/fx
    vertical_limit = max(cy, sensor["height"]-1-cy)/fy
    mask = (
        (forward > .30)
        & (forward < min(3.0, float(sensor["max_depth_m"])/7.0))
        & (np.abs(lateral) < forward*horizontal_limit*.90)
        & (np.abs(vertical) < forward*vertical_limit*.90)
    )
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    distance = np.linalg.norm(delta[indices], axis=1)
    order = np.lexsort((backend.linear[indices], distance))
    occupied_lookup = {
        tuple(row): index
        for index, row in enumerate(backend.occupied.tolist())
    }
    connected, isolated, seen = [], [], set()
    for item in order:
        index = int(indices[item])
        voxel = backend.occupied[index]
        for axis in (0, 1):
            for sign in (-1, 1):
                neighbour = voxel.copy()
                neighbour[axis] += sign
                other = occupied_lookup.get(tuple(neighbour))
                if other is None:
                    continue
                pair = tuple(sorted((index, int(other))))
                if pair in seen:
                    continue
                seen.add(pair)
                patch_center = .5*(centers[index]+centers[other])
                tangent = np.zeros(3, dtype=np.float64)
                tangent[axis] = 1.0
                connected.append((
                    index, patch_center, tangent,
                    2.0*backend.map.resolution))
        isolated.append((
            index, centers[index], None, backend.map.resolution))
    return isolated+connected


def natural_occlusion_pattern(diagnostics, actor_index, contract):
    projected = diagnostics[
        "per_actor_projected_pixel_count"][:, actor_index]
    visible = diagnostics[
        "per_actor_visible_pixel_count"][:, actor_index]
    blocked = diagnostics[
        "per_actor_static_blocked_pixel_count"][:, actor_index]
    outside = diagnostics["per_actor_outside_fov"][:, actor_index]
    behind = diagnostics["per_actor_behind_camera"][:, actor_index]
    beyond = diagnostics[
        "per_actor_beyond_max_depth"][:, actor_index]
    full = (
        (projected > 0) & (visible == 0) & (blocked == projected)
        & ~outside & ~behind & ~beyond
    )
    directly_visible = (projected > 0) & (visible > 0)
    pre_min = int(contract["occlusion"]["pre_gap_visible_min"])
    post_min = int(contract["occlusion"]["post_gap_visible_min"])
    accepted = []
    index = 0
    while index < len(full):
        if not full[index]:
            index += 1
            continue
        end = index
        while end+1 < len(full) and full[end+1]:
            end += 1
        pre = index >= pre_min and bool(
            directly_visible[index-pre_min:index].all())
        post = (
            end+post_min < len(full)
            and bool(directly_visible[end+1:end+1+post_min].all())
        )
        if pre and post:
            accepted.append((index, end))
        index = end+1
    return {
        "accepted_gaps": accepted,
        "projected": projected,
        "visible": visible,
        "static_blocked": blocked,
        "outside_fov": outside,
        "behind_camera": behind,
        "beyond_max_depth": beyond,
        "full_static_occlusion": full,
        "directly_visible": directly_visible,
    }


def build_occluded_actor_specs_v2_1(
    backend, renderer, rng, uav_positions, yaws, frame_times,
    contract, requested_gap_frames, method="deterministic",
    maximum_candidates=32,
):
    if contract["contract_version"] != CONTRACT_VERSION_V2_1:
        raise RuntimeError("occlusion constructor requires v2.1 contract")
    gap_length = int(requested_gap_frames)
    if gap_length not in (1, 2, 3):
        raise ValueError("requested gap must be 1, 2, or 3 frames")
    started = time.perf_counter()
    sensor = renderer.sensor
    pre_min = int(contract["occlusion"]["pre_gap_visible_min"])
    post_min = int(contract["occlusion"]["post_gap_visible_min"])
    starts = list(range(
        pre_min, len(frame_times)-gap_length-post_min))
    preferred_start = min(
        10, len(frame_times)-gap_length-post_min-1)
    starts.sort(key=lambda value: (abs(value-preferred_start), value))
    if method == "random_seeded":
        starts = [starts[index] for index in rng.permutation(len(starts))]
    elif method != "deterministic":
        raise ValueError("unknown bounded constructor method")
    profile = contract["scenarios"]["occluded_but_tracked"]
    low, high = map(float, profile["speed_range_mps"])
    speeds = (low, .5*(low+high), high)
    depth_ratios = {
        1: (6.4, 7.0),
        2: (7.0, 7.8),
        3: (8.5, 7.8),
    }[gap_length]
    attempts = 0
    positions = np.asarray(uav_positions, dtype=np.float64)
    yaws = np.asarray(yaws, dtype=np.float64)
    frame_times = np.asarray(frame_times, dtype=np.float64)
    end_time = float(frame_times[-1]+1.7)
    voxel_centers = (
        backend.map.origin[None, :]
        + (backend.occupied.astype(np.float64)+.5)
        * backend.map.resolution
    )
    occupancy_tree = cKDTree(voxel_centers)
    for gap_start in starts:
        gap_center = gap_start + .5*(gap_length-1)
        center_time = float(np.interp(
            gap_center, np.arange(len(frame_times)), frame_times))
        camera = _interpolate_path(
            np.asarray([center_time]), frame_times, positions)[0]
        yaw = float(np.interp(
            center_time, frame_times, np.unwrap(yaws)))
        candidates = _candidate_voxels(
            backend, camera, yaw, sensor)
        if method == "random_seeded" and candidates:
            order = rng.permutation(len(candidates))
            candidates = [candidates[index] for index in order]
        for (
            voxel_offset, voxel_center, patch_tangent,
            occluder_width,
        ) in candidates[:maximum_candidates]:
            ray = voxel_center-camera
            occluder_distance = float(np.linalg.norm(ray))
            if occluder_distance <= 0:
                continue
            ray /= occluder_distance
            horizontal_tangent = (
                np.asarray(patch_tangent, dtype=np.float64)
                if patch_tangent is not None
                else np.asarray([-ray[1], ray[0], 0.0])
            )
            horizontal_tangent = (
                horizontal_tangent
                - ray*np.dot(horizontal_tangent, ray)
            )
            norm = float(np.linalg.norm(horizontal_tangent))
            if norm < 1e-8:
                continue
            horizontal_tangent /= norm
            vertical_tangent = np.cross(ray, horizontal_tangent)
            vertical_tangent /= np.linalg.norm(vertical_tangent)
            for ratio in depth_ratios:
                anchor = camera+ray*(ratio*occluder_distance)
                for speed in speeds:
                    full_shadow_width = max(
                        .01,
                        occluder_width*ratio-2*.30)
                    nominal_tangent_speed = (
                        full_shadow_width/max(gap_length*.1, .1))
                    for tangent_scale in (.85, 1.0, 1.15):
                        tangent_speed = min(
                            speed*.90,
                            max(speed*.85,
                                nominal_tangent_speed*tangent_scale))
                        radial_speed = np.sqrt(max(
                            0.0,
                            speed*speed-tangent_speed*tangent_speed))
                        tangent_angles = (
                            0.0, np.pi,
                        )
                        for tangent_angle in tangent_angles:
                                tangent_direction = (
                                    horizontal_tangent*np.cos(tangent_angle)
                                    + vertical_tangent*np.sin(tangent_angle)
                                )
                                attempts += 1
                                velocity = (
                                    ray*radial_speed
                                    + tangent_direction*tangent_speed
                                )
                                actor = {
                                    "actor_id": 0,
                                    "start": anchor-velocity*center_time,
                                    "velocity": velocity,
                                    "radius_m": .30,
                                    "motion_profile": "constant_velocity",
                                    "configured_speed_mps": float(speed),
                                    "sampling_method":
                                        f"{CONSTRUCTOR_VERSION}:{method}",
                                    "motion_contract_version":
                                        CONTRACT_VERSION_V2_1,
                                }
                                validation_times = _dense_times(
                                    end_time, speed,
                                    float(contract["sampling"][
                                        "continuous_sample_spacing_m"]))
                                actor["validation_times"] = validation_times
                                path = actor_position(
                                    actor, validation_times)
                                camera_path = _interpolate_path(
                                    validation_times, frame_times, positions)
                                if not _trajectory_valid_exact(
                                    backend, occupancy_tree, path, camera_path,
                                    .30, contract["sampling"],
                                    validation_times,
                                ):
                                    continue
                                actor_frames = actor_position(
                                    actor, frame_times)[:, None, :]
                                diagnostics = (
                                    renderer.render_with_actor_diagnostics(
                                        backend, positions, yaws,
                                        actor_frames, [.30],
                                        return_owner_map=False,
                                        return_actor_near_depth=False,
                                    )
                                )
                                pattern = natural_occlusion_pattern(
                                    diagnostics, 0, contract)
                                match = next((
                                    gap for gap in pattern["accepted_gaps"]
                                    if gap[1]-gap[0]+1 == gap_length
                                ), None)
                                if match is None:
                                    continue
                                evidence = exact_first_hit(
                                    backend, positions[match[0]],
                                    actor_frames[match[0], 0])
                                if evidence is None:
                                    continue
                                actor["occlusion_constructor"] = {
                                    "version": CONSTRUCTOR_VERSION,
                                    "requested_gap_frames": gap_length,
                                    "gap_start": int(match[0]),
                                    "gap_end": int(match[1]),
                                    "occluder_voxel_index":
                                        evidence["voxel_index"],
                                    "occluder_authority_hash":
                                        evidence["authority_hash"],
                                    "occluder_distance_m":
                                        evidence["distance_m"],
                                    "depth_ratio": float(ratio),
                                    "method": method,
                                }
                                return OcclusionConstruction(
                                    [actor], diagnostics, evidence,
                                    attempts,
                                    time.perf_counter()-started,
                                    gap_length, int(match[0]),
                                    int(match[1]), method,
                                )
    raise MotionSamplingError(
        "bounded authority-sightline occlusion construction exhausted "
        f"after {attempts} candidates")
