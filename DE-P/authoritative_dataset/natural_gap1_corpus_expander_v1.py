"""Bounded CE1 proposal wrapper for natural, exact one-frame occlusions.

This module does not alter constructor v2.2.  It only enumerates deterministic
camera motions around geometry-only camera proposals and delegates physical and
visibility decisions to the frozen authority, continuous checker, and CUDA
renderer.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from authoritative_dataset.continuous_v1 import (
    ContinuousState, certify_curve,
)
from authoritative_dataset.dynamic_motion_v2 import (
    CONTRACT_VERSION_V2_1, _dense_times, _interpolate_path, actor_position,
)
from authoritative_dataset.natural_observable_occlusion_proposer_v1 import (
    NaturalCameraProposal, propose_natural_cameras,
    propose_observable_actor,
)
from authoritative_dataset.occlusion_constructor_v2_1 import (
    OcclusionConstruction, _trajectory_valid_exact,
    natural_occlusion_pattern,
)
from authoritative_dataset.occlusion_constructor_v2_2 import (
    ACTOR_RADIUS_M, TANGENT_FRACTION, canonical_surface_patches,
    feasibility_interval,
)
from authoritative_dataset.occlusion_geometry_v1 import exact_first_hit


EXPANDER_VERSION = "natural_gap1_corpus_expander_v1"
CAMERA_MOTIONS = (
    "static",
    "lateral_linear_positive",
    "lateral_linear_negative",
    "retreat_linear",
    "approach_linear",
    "yaw_left",
    "yaw_right",
    "yaw_lateral",
    "yaw_forward",
)


def camera_motion(proposal, variant, frame_times, backend):
    times = np.asarray(frame_times, dtype=np.float64)
    positions = np.asarray(proposal.position_world, dtype=np.float64).copy()
    yaws = np.asarray(proposal.yaw, dtype=np.float64).copy()
    forward = np.array([np.cos(yaws[0]), np.sin(yaws[0]), 0.0])
    lateral = np.array([-forward[1], forward[0], 0.0])
    unit = np.linspace(-1.0, 1.0, len(times))
    if variant == "static":
        pass
    elif variant == "lateral_linear_positive":
        positions += (.035 * unit)[:, None] * lateral
    elif variant == "lateral_linear_negative":
        positions -= (.035 * unit)[:, None] * lateral
    elif variant == "retreat_linear":
        positions -= (.035 * (unit + 1.0))[:, None] * forward
    elif variant == "approach_linear":
        positions += (.035 * (unit + 1.0))[:, None] * forward
    elif variant == "yaw_left":
        yaws += .035 * unit
    elif variant == "yaw_right":
        yaws -= .035 * unit
    elif variant == "yaw_lateral":
        positions += (.025 * unit)[:, None] * lateral
        yaws += .025 * unit
    elif variant == "yaw_forward":
        positions += (.025 * (unit + 1.0))[:, None] * forward
        yaws -= .025 * unit
    else:
        raise ValueError(f"unknown camera motion: {variant}")

    duration = float(times[-1])
    start, end = positions[0].copy(), positions[-1].copy()
    position_at = lambda value: start + (end - start) * (
        np.clip(float(value) / max(duration, 1e-12), 0.0, 1.0)
    )
    certificate = certify_curve(position_at, backend, duration, max_depth=14)
    if certificate.state is not ContinuousState.CERTIFIED_SAFE:
        raise RuntimeError(
            f"camera_path_collision:{certificate.state.value}"
        )
    expanded = replace(
        proposal,
        position_world=positions,
        yaw=yaws,
        certificate={
            **proposal.certificate,
            "expander_version": EXPANDER_VERSION,
            "camera_motion": variant,
            "continuous_certificate": asdict(certificate),
            "implementation_hash": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
    )
    return expanded


def enumerate_camera_candidates(
    backend,
    frame_times,
    *,
    p0_budget=128,
    exact_budget=24,
):
    if p0_budget > 128 or exact_budget > 24:
        raise ValueError("CE1 existing-map budget exceeded")
    proposals = propose_natural_cameras(
        backend, frame_times, 1,
        maximum_proposals=p0_budget,
        maximum_patch_checks=p0_budget,
    )
    candidates = []
    for motion_offset, motion in enumerate(CAMERA_MOTIONS):
        for proposal in proposals:
            candidates.append((
                proposal.proposal_rank,
                motion_offset,
                proposal,
                motion,
            ))
    candidates.sort(key=lambda row: (row[0], row[1]))
    output = []
    for _, _, proposal, motion in candidates:
        try:
            output.append(camera_motion(
                proposal, motion, frame_times, backend
            ))
        except RuntimeError:
            continue
        if len(output) >= exact_budget:
            break
    return tuple(output)


def propose_one(
    backend,
    renderer,
    camera_proposal,
    frame_times,
    contract,
    *,
    actor_candidate_budget=8,
):
    if not 1 <= int(actor_candidate_budget) <= 24:
        raise ValueError("invalid CE1 per-proposal actor budget")
    return propose_observable_actor(
        backend, renderer, camera_proposal, frame_times, contract, 1,
        maximum_raster_candidates=int(actor_candidate_budget),
    )


def propose_at_gap_origins(
    backend,
    renderer,
    camera_proposal,
    frame_times,
    contract,
    *,
    gap_origins=(8, 12, 16, 20),
    maximum_candidates=20,
):
    """Enumerate bounded timing/direction combinations around one patch."""
    if contract["contract_version"] != CONTRACT_VERSION_V2_1:
        raise RuntimeError("motion contract mismatch")
    if not 1 <= int(maximum_candidates) <= 20:
        raise ValueError("CE1 E3 exact budget exceeded")
    started = time.perf_counter()
    times = np.asarray(frame_times, dtype=np.float64)
    positions = np.asarray(
        camera_proposal.position_world, dtype=np.float64
    )
    yaws = np.asarray(camera_proposal.yaw, dtype=np.float64)
    patch = next(
        value for value in canonical_surface_patches(backend)
        if value.voxel_offset == camera_proposal.patch_voxel_offset
    )
    camera = positions[0]
    ray = np.asarray(patch.center, dtype=np.float64) - camera
    distance = float(np.linalg.norm(ray))
    ray /= max(distance, 1e-12)
    tangent = np.asarray(patch.tangent, dtype=np.float64)
    tangent -= ray * float(tangent @ ray)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
    voxel_centers = (
        backend.map.origin[None, :]
        + (backend.occupied.astype(np.float64) + .5)
        * backend.map.resolution
    )
    tree = cKDTree(voxel_centers)
    low, high = map(float, contract["scenarios"][
        "occluded_but_tracked"
    ]["speed_range_mps"])
    speeds = (low, .5 * (low + high), high)
    attempts = 0
    rejected = CounterLike()
    for gap_start in gap_origins:
        if not 4 <= int(gap_start) < len(times) - 3:
            continue
        center_time = float(times[int(gap_start)])
        for speed in speeds:
            interval = feasibility_interval(
                distance, patch.horizontal_extent_m,
                patch.vertical_extent_m, 1, speed,
                detection_distance_m=float(contract["sampling"][
                    "maximum_detection_center_distance_m"
                ]),
            )
            if not interval["feasible"]:
                rejected["ratio_interval"] += 1
                continue
            ratios = tuple(dict.fromkeys((
                interval["lower_ratio"],
                .5 * (interval["lower_ratio"] + interval["upper_ratio"]),
                interval["upper_ratio"],
            )))
            for ratio in ratios:
                anchor = camera + ray * (ratio * distance)
                tangent_speed = speed * TANGENT_FRACTION
                radial_speed = np.sqrt(max(
                    0.0, speed * speed - tangent_speed * tangent_speed
                ))
                for radial_sign in (-1.0, 1.0):
                    for tangent_sign in (-1.0, 1.0):
                        attempts += 1
                        if attempts > maximum_candidates:
                            raise RuntimeError(
                                "bounded_explicit_gap_origin_exhausted:"
                                f"{dict(rejected)}"
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
                            "sampling_method":
                                f"{EXPANDER_VERSION}:explicit_gap_origin",
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
                        actor["validation_times"] = validation_times
                        path = actor_position(actor, validation_times)
                        camera_path = _interpolate_path(
                            validation_times, times, positions
                        )
                        if not _trajectory_valid_exact(
                            backend, tree, path, camera_path,
                            ACTOR_RADIUS_M, contract["sampling"],
                            validation_times,
                        ):
                            rejected["trajectory"] += 1
                            continue
                        actor_frames = actor_position(
                            actor, times
                        )[:, None, :]
                        diagnostics = renderer.render_with_actor_diagnostics(
                            backend, positions, yaws, actor_frames,
                            [ACTOR_RADIUS_M],
                        )
                        pattern = natural_occlusion_pattern(
                            diagnostics, 0, contract
                        )
                        if [int(gap_start), int(gap_start)] not in [
                            list(value) for value in pattern["accepted_gaps"]
                        ]:
                            rejected["raster_gap"] += 1
                            continue
                        evidence = exact_first_hit(
                            backend, positions[int(gap_start)],
                            actor_frames[int(gap_start), 0],
                        )
                        if evidence is None:
                            rejected["sightline"] += 1
                            continue
                        actor["proposal"] = {
                            "version": EXPANDER_VERSION,
                            "gap_start": int(gap_start),
                            "gap_end": int(gap_start),
                            "depth_ratio": float(ratio),
                            "radial_sign": radial_sign,
                            "tangent_sign": tangent_sign,
                            "camera_standoff_m":
                                camera_proposal.camera_standoff_m,
                            "v2_2_feasibility_interval": interval,
                            "explicit_gap_origin": True,
                        }
                        return OcclusionConstruction(
                            [actor], diagnostics, evidence, attempts,
                            time.perf_counter() - started, 1,
                            int(gap_start), int(gap_start),
                            f"{EXPANDER_VERSION}:explicit_gap_origin",
                        )
    raise RuntimeError(
        f"bounded_explicit_gap_origin_failed:{dict(rejected)}"
    )


class CounterLike(dict):
    def __missing__(self, key):
        return 0

    def __setitem__(self, key, value):
        super().__setitem__(key, int(value))


__all__ = [
    "CAMERA_MOTIONS", "EXPANDER_VERSION", "camera_motion",
    "enumerate_camera_candidates", "propose_one",
    "propose_at_gap_origins",
]
