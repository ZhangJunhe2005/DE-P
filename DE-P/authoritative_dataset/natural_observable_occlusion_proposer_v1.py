"""Bounded development-only camera proposals for natural observability."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from authoritative_dataset.occlusion_constructor_v2_2 import (
    ACTOR_RADIUS_M, TANGENT_FRACTION,
    canonical_surface_patches, feasibility_interval,
)
from authoritative_dataset.dynamic_motion_v2 import (
    CONTRACT_VERSION_V2_1, _dense_times, _interpolate_path, actor_position,
)
from authoritative_dataset.occlusion_constructor_v2_1 import (
    OcclusionConstruction, _trajectory_valid_exact, natural_occlusion_pattern,
)
from authoritative_dataset.occlusion_geometry_v1 import exact_first_hit
from scipy.spatial import cKDTree
import time


PROPOSER_VERSION = "natural_observable_occlusion_proposer_v1"
CAMERA_STANDOFFS_M = (0.355, 0.45, 0.60, 0.80, 1.00)


@dataclass(frozen=True)
class NaturalCameraProposal:
    position_world: np.ndarray
    yaw: np.ndarray
    patch_voxel_offset: int
    patch_center: tuple[float, float, float]
    patch_normal: tuple[float, float, float]
    camera_standoff_m: float
    proposal_rank: int
    certificate: dict


def propose_natural_cameras(
    backend,
    frame_times,
    requested_gap,
    *,
    maximum_proposals=32,
    maximum_patch_checks=512,
):
    """Enumerate general patch/standoff poses without map-ID or seed branches."""
    times = np.asarray(frame_times, dtype=np.float64)
    candidates = []
    patches = canonical_surface_patches(backend)
    if len(patches) > maximum_patch_checks:
        indices = np.linspace(
            0, len(patches) - 1, maximum_patch_checks, dtype=np.int64
        )
        patches = [patches[int(index)] for index in indices]
    for patch in patches:
        center = np.asarray(patch.center, dtype=np.float64)
        normal = np.asarray(patch.normal, dtype=np.float64)
        face = center + normal * (.5 * float(backend.map.resolution))
        for standoff in CAMERA_STANDOFFS_M:
            camera = face + normal * standoff
            if backend.query_one(camera, .30)["collision"]:
                continue
            distance = float(np.linalg.norm(center - camera))
            feasible = any(
                feasibility_interval(
                    distance, patch.horizontal_extent_m,
                    patch.vertical_extent_m, requested_gap, speed,
                )["feasible"]
                for speed in (1.40, 1.55, 1.70)
            )
            if not feasible:
                continue
            view = -normal
            yaw_value = float(np.arctan2(view[1], view[0]))
            # Prefer broad surfaces, moderate stand-off, then canonical offset.
            score = (
                -min(patch.horizontal_extent_m, 3.0)
                -min(patch.vertical_extent_m, 2.0)
                + abs(standoff - .75),
                patch.voxel_offset,
                standoff,
            )
            candidates.append((score, camera, yaw_value, patch, standoff))
    candidates.sort(key=lambda row: row[0])
    output = []
    implementation_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for rank, (_, camera, yaw, patch, standoff) in enumerate(
        candidates[:maximum_proposals]
    ):
        output.append(NaturalCameraProposal(
            position_world=np.repeat(camera[None, :], len(times), axis=0),
            yaw=np.full(len(times), yaw, dtype=np.float64),
            patch_voxel_offset=patch.voxel_offset,
            patch_center=patch.center,
            patch_normal=patch.normal,
            camera_standoff_m=float(standoff),
            proposal_rank=rank,
            certificate={
                "version": PROPOSER_VERSION,
                "time_varying_camera": False,
                "map_occupancy_modified": False,
                "annex_used": False,
                "case_id_branch": False,
                "map_uuid_branch": False,
                "seed_branch": False,
                "implementation_hash": implementation_hash,
                "maximum_patch_checks": maximum_patch_checks,
            },
        ))
    return tuple(output)


def _ratios(interval):
    low, high = interval["lower_ratio"], interval["upper_ratio"]
    if low > high:
        return ()
    return tuple(dict.fromkeys(round(value, 8) for value in (
        low, .5 * (low + high), high,
    )))


def propose_observable_actor(
    backend,
    renderer,
    camera_proposal,
    frame_times,
    contract,
    requested_gap,
    *,
    maximum_raster_candidates=768,
):
    """Construct a frozen-speed actor while explicitly auditing radial sign."""
    started = time.perf_counter()
    times = np.asarray(frame_times, dtype=np.float64)
    positions = np.asarray(camera_proposal.position_world, dtype=np.float64)
    yaws = np.asarray(camera_proposal.yaw, dtype=np.float64)
    patch = next(
        value for value in canonical_surface_patches(backend)
        if value.voxel_offset == camera_proposal.patch_voxel_offset
    )
    gap = int(requested_gap)
    pre_min = int(contract["occlusion"]["pre_gap_visible_min"])
    post_min = int(contract["occlusion"]["post_gap_visible_min"])
    starts = list(range(pre_min, len(times) - gap - post_min))
    preferred = min(14, len(times) - gap - post_min - 1)
    starts.sort(key=lambda value: (abs(value - preferred), value))
    camera = positions[0]
    ray = np.asarray(patch.center, dtype=np.float64) - camera
    distance = float(np.linalg.norm(ray))
    ray /= max(distance, 1e-12)
    tangent = np.asarray(patch.tangent, dtype=np.float64)
    tangent -= ray * np.dot(tangent, ray)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
    profile = contract["scenarios"]["occluded_but_tracked"]
    low, high = map(float, profile["speed_range_mps"])
    speeds = (low, .5 * (low + high), high)
    voxel_centers = (
        backend.map.origin[None, :]
        + (backend.occupied.astype(np.float64) + .5)
        * backend.map.resolution
    )
    tree = cKDTree(voxel_centers)
    attempts = 0
    rejected = {"trajectory": 0, "raster_gap": 0, "sightline": 0}
    for gap_start in starts:
        center_time = float(np.interp(
            gap_start + .5 * (gap - 1),
            np.arange(len(times)), times,
        ))
        for speed in speeds:
            interval = feasibility_interval(
                distance, patch.horizontal_extent_m,
                patch.vertical_extent_m, gap, speed,
                detection_distance_m=float(
                    contract["sampling"]["maximum_detection_center_distance_m"]
                ),
            )
            for ratio in _ratios(interval):
                anchor = camera + ray * (ratio * distance)
                tangent_speed = speed * TANGENT_FRACTION
                radial_speed = np.sqrt(max(
                    0.0, speed * speed - tangent_speed * tangent_speed
                ))
                # Inbound first: it supplies causal positive range residuals.
                for radial_sign in (-1.0, 1.0):
                    for tangent_sign in (-1.0, 1.0):
                        attempts += 1
                        if attempts > maximum_raster_candidates:
                            raise RuntimeError(
                                "bounded observable actor candidates exhausted; "
                                f"rejected={rejected}"
                            )
                        velocity = (
                            radial_sign * ray * radial_speed
                            + tangent_sign * tangent * tangent_speed
                        )
                        actor = {
                            "actor_id": 0,
                            "start": anchor - velocity * center_time,
                            "velocity": velocity,
                            "radius_m": ACTOR_RADIUS_M,
                            "motion_profile": "constant_velocity",
                            "configured_speed_mps": float(speed),
                            "sampling_method": PROPOSER_VERSION,
                            "motion_contract_version": CONTRACT_VERSION_V2_1,
                            "radial_direction": (
                                "toward_camera" if radial_sign < 0
                                else "away_from_camera"
                            ),
                        }
                        validation_times = _dense_times(
                            float(times[-1] + 1.7), speed,
                            float(contract["sampling"][
                                "continuous_sample_spacing_m"
                            ]),
                        )
                        path = actor_position(actor, validation_times)
                        camera_path = _interpolate_path(
                            validation_times, times, positions
                        )
                        if not _trajectory_valid_exact(
                            backend, tree, path, camera_path, ACTOR_RADIUS_M,
                            contract["sampling"], validation_times,
                        ):
                            rejected["trajectory"] += 1
                            continue
                        actor_frames = actor_position(actor, times)[:, None, :]
                        diagnostics = renderer.render_with_actor_diagnostics(
                            backend, positions, yaws, actor_frames,
                            [ACTOR_RADIUS_M],
                        )
                        pattern = natural_occlusion_pattern(
                            diagnostics, 0, contract
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
                            rejected["sightline"] += 1
                            continue
                        actor["proposal"] = {
                            "version": PROPOSER_VERSION,
                            "gap_start": int(match[0]),
                            "gap_end": int(match[1]),
                            "depth_ratio": float(ratio),
                            "radial_sign": radial_sign,
                            "tangent_sign": tangent_sign,
                            "camera_standoff_m":
                                camera_proposal.camera_standoff_m,
                            "v2_2_feasibility_interval": interval,
                        }
                        return OcclusionConstruction(
                            [actor], diagnostics, evidence, attempts,
                            time.perf_counter() - started, gap,
                            int(match[0]), int(match[1]),
                            PROPOSER_VERSION,
                        )
    raise RuntimeError(
        f"bounded observable actor construction failed; rejected={rejected}"
    )


__all__ = [
    "CAMERA_STANDOFFS_M", "NaturalCameraProposal", "PROPOSER_VERSION",
    "propose_natural_cameras", "propose_observable_actor",
]
