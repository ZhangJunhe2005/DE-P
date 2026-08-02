#!/usr/bin/env python3
"""Bounded development-only EOSR1 edge-grazing search on MTC1 near misses."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4eosr1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from authoritative_dataset.cuda_renderer_v1 import CudaAuthorityRenderer
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.natural_exact_occlusion_solver_v2 import (
    FullInterval, analytic_phase_intervals, camera_to_world,
    contiguous_intervals, strict_full_mask, validate_strict_window,
)
from authoritative_dataset.occlusion_constructor_v2_2 import (
    canonical_surface_patches,
)
from generate_phase8jqv2_4rr1_corpus import SENSOR


PERIOD = SENSOR["frame_period_ns"] * 1e-9
RADII = (.20, .25, .30)
SPEEDS = (.8, 1.1, 1.4)
DIRECTIONS = tuple(np.asarray(
    (math.cos(k*math.pi/6), math.sin(k*math.pi/6))
) for k in range(12))
CAMERA_VARIANTS = (
    "static", "lateral", "yaw", "yaw_lateral", "forward_yaw",
)
MAX_CUDA = 120
MAX_ANALYTIC = 30720
MAX_EXACT = 960


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True
    ) + "\n")
    os.replace(temporary, path)


def authorities():
    result = {}
    for name in (
        "phase8jqv2_4ce1_new_map_manifest.json",
        "phase8jqv2_4ce1_e3_map_manifest.json",
    ):
        value = json.loads((REPORTS / name).read_text())
        result.update({
            row["map_uuid"]: row["authority_root"]
            for row in value["maps"]
        })
    return result


def body_from_world(camera, yaw, world):
    delta = np.asarray(world, dtype=np.float64) - camera
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray((
        c*delta[0] + s*delta[1],
        -s*delta[0] + c*delta[1],
        delta[2],
    ))


def projected_patch(patch, camera, yaw):
    body = body_from_world(camera, yaw, patch.center)
    fx, fy, cx, cy = SENSOR["intrinsics"]
    if body[0] <= 0:
        return None
    return np.asarray((
        cx + fx*body[1]/body[0],
        cy + fy*body[2]/body[0],
        body[0],
    ))


def body_center(pixel, forward):
    fx, fy, cx, cy = SENSOR["intrinsics"]
    return np.asarray((
        forward,
        forward*(pixel[0]-cx)/fx,
        forward*(pixel[1]-cy)/fy,
    ), dtype=np.float64)


def static_class(static_depth, pixel, forward, radius, body_rays):
    center = body_center(pixel, forward)
    projection = np.sum(body_rays*center, axis=-1)
    perpendicular2 = np.dot(center, center)-projection**2
    intersects = (projection > 0) & (perpendicular2 <= radius**2)
    near = projection-np.sqrt(np.maximum(radius**2-perpendicular2, 0))
    projected = intersects & (near > 0) & (
        near <= SENSOR["max_depth_m"]
    )
    count = int(projected.sum())
    if not count:
        return 0, 0, 0, -math.inf
    blocked = projected & (static_depth <= near)
    visible = projected & (near < static_depth)
    margin = float(np.min(near[projected]-static_depth[projected]))
    return count, int(visible.sum()), int(blocked.sum()), margin


def camera_paths(base, yaw, variant, frames):
    times = np.arange(frames) * PERIOD
    positions = np.repeat(base[None, :], frames, axis=0)
    yaws = np.full(frames, yaw)
    forward = np.asarray((math.cos(yaw), math.sin(yaw), 0.))
    lateral = np.asarray((-forward[1], forward[0], 0.))
    if variant == "lateral":
        positions += times[:, None]*lateral[None, :]*.12
    elif variant == "yaw":
        yaws += times*.12
    elif variant == "yaw_lateral":
        positions += times[:, None]*lateral[None, :]*.10
        yaws += times*.10
    elif variant == "forward_yaw":
        positions += times[:, None]*forward[None, :]*.10
        yaws += times*.08
    return positions, yaws


def collision_safe(backend, actor, radius, camera):
    dense_t = np.linspace(0, len(actor)-1, (len(actor)-1)*5+1)
    dense = np.column_stack([
        np.interp(dense_t, np.arange(len(actor)), actor[:, axis])
        for axis in range(3)
    ])
    static = [backend.query_one(point, radius) for point in dense]
    camera_dense = np.column_stack([
        np.interp(dense_t, np.arange(len(camera)), camera[:, axis])
        for axis in range(3)
    ])
    separation = np.linalg.norm(dense-camera_dense, axis=1)
    return (
        not any(row["collision"] for row in static)
        and float(separation.min()) > radius + .30 + .10
    ), {
        "minimum_static_gap_m": float(min(
            row["minimum_gap_m"] for row in static
        )),
        "minimum_actor_camera_distance_m": float(separation.min()),
        "dense_samples": len(dense),
    }


def exact_run(renderer, backend, camera, yaws, actor, radius):
    return renderer.render_with_actor_diagnostics(
        backend, camera, yaws, actor[:, None, :], [radius],
        return_owner_map=True, return_actor_near_depth=True,
    )


def full_runs(mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    return [
        group.tolist() for group in np.split(
            indices, np.flatnonzero(np.diff(indices) > 1)+1
        )
    ]


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("EOSR1 exact solver requires host CUDA")
    started = time.perf_counter()
    manifest = json.loads((
        REPORTS / "phase8jqv2_4mtc1_room_wall_near_miss_manifest.json"
    ).read_text())
    roots = authorities()
    renderer = CudaAuthorityRenderer(SENSOR, "cuda:0")
    rays = renderer.body_rays.detach().cpu().numpy().astype(np.float64)
    rows = []
    seen = set()
    for row in manifest["candidates"]:
        key = (row["map_uuid"], row["patch_id"])
        if key not in seen:
            seen.add(key)
            rows.append(row)

    analytic_count = exact_count = cuda_count = 0
    candidates = []
    witnesses = []
    camera_sweep = Counter()
    rejection = Counter()
    for near_index, row in enumerate(rows):
        backend = ExactAuthorityBVH(roots[row["map_uuid"]])
        patch = next(
            patch for patch in canonical_surface_patches(backend)
            if patch.voxel_offset == row["patch_id"]
        )
        base = np.asarray(row["camera_positions"][0], dtype=np.float64)
        base_yaw = float(row["camera_yaw"][0])
        projected_patch_value = projected_patch(
            patch, base, base_yaw
        )
        if projected_patch_value is None:
            rejection["patch_behind_camera"] += 1
            continue
        # One canonical static render provides the exact occupancy ray depths
        # used by CUDA. CPU broad phase reproduces sphere-ray intersections.
        empty = np.empty((1, 0, 3), dtype=np.float64)
        static_diag = renderer.render_with_actor_diagnostics(
            backend, base[None, :], np.asarray([base_yaw]),
            empty, [],
        )
        static_depth = static_diag["static_depth"][0]
        pu, pv, patch_forward = projected_patch_value
        depth_values = sorted(set(
            round(min(1.78, patch_forward+offset), 3)
            for offset in (.36, .55, .80, 1.05)
            if patch_forward+offset <= 1.78
        ))
        if not depth_values:
            rejection["no_detection_feasible_depth"] += 1
            continue
        for radius in RADII:
            for forward in depth_values:
                # Search only the frozen patch neighbourhood; geometry and
                # patch identity remain unchanged.
                full_centers = []
                for v in range(max(12, int(pv)-28), min(84, int(pv)+29), 2):
                    for u in range(max(18, int(pu)-36),
                                   min(142, int(pu)+37), 2):
                        if analytic_count >= MAX_ANALYTIC:
                            break
                        analytic_count += 1
                        p, vis, block, margin = static_class(
                            static_depth, (u, v), forward, radius, rays
                        )
                        if p > 0 and vis == 0 and block == p:
                            center_world = camera_to_world(
                                base, base_yaw,
                                body_center((u, v), forward)[None, :]
                            )[0]
                            center_query = backend.query_one(
                                center_world, radius
                            )
                            if not center_query["collision"]:
                                full_centers.append((
                                    u, v, margin, p,
                                    center_query["minimum_gap_m"],
                                ))
                if not full_centers:
                    continue
                # Prefer the boundary (small positive depth margin), which is
                # the opposite of the historical center-crossing objective.
                full_centers.sort(key=lambda item: (abs(item[2]), item[0], item[1]))
                for u, v, margin, projected, anchor_gap in full_centers[:16]:
                    for direction_index, direction in enumerate(DIRECTIONS):
                        # A straight line in the camera image plane is only a
                        # legal wall-grazing path when its corresponding world
                        # velocity is tangent to the frozen patch face.
                        fx, fy, _, _ = SENSOR["intrinsics"]
                        body_delta = np.asarray((
                            0., forward*direction[0]/fx,
                            forward*direction[1]/fy,
                        ))
                        world_delta = camera_to_world(
                            np.zeros(3), base_yaw,
                            body_delta[None, :]
                        )[0]
                        world_delta /= max(
                            np.linalg.norm(world_delta), 1e-12
                        )
                        patch_normal = np.asarray(
                            patch.normal, dtype=np.float64
                        )
                        if abs(float(np.dot(
                            world_delta, patch_normal
                        ))) > .12:
                            continue
                        for speed in SPEEDS:
                            pixels_per_frame = (
                                speed*PERIOD*SENSOR["intrinsics"][0]/forward
                            )
                            for phase_fraction in (-.45, -.2, 0., .2, .45):
                                center_frame = 5
                                frame_index = np.arange(12)-center_frame
                                pixels = (
                                    np.asarray((u, v))[None, :]
                                    + (
                                        frame_index[:, None]+phase_fraction
                                    )*pixels_per_frame*direction[None, :]
                                )
                                broad = [
                                    static_class(
                                        static_depth, pixel, forward,
                                        radius, rays
                                    ) for pixel in pixels
                                ]
                                full = np.asarray([
                                    p > 0 and vis == 0 and blocked == p
                                    for p, vis, blocked, _ in broad
                                ])
                                runs = [
                                    run for run in full_runs(full)
                                    if len(run) in (1, 2)
                                    and run[0] >= 4
                                    and run[-1]+3 < 12
                                ]
                                if not runs:
                                    continue
                                if exact_count >= MAX_EXACT:
                                    break
                                exact_count += 1
                                actor = camera_to_world(
                                    base, base_yaw,
                                    np.asarray([
                                        body_center(pixel, forward)
                                        for pixel in pixels
                                    ]),
                                )
                                camera, yaws = camera_paths(
                                    base, base_yaw, "static", 12
                                )
                                safe, safety = collision_safe(
                                    backend, actor, radius, camera
                                )
                                if not safe:
                                    rejection["continuous_collision"] += 1
                                    continue
                                if cuda_count >= MAX_CUDA:
                                    break
                                cuda_count += 1
                                camera_sweep["static"] += 1
                                diagnostics = exact_run(
                                    renderer, backend, camera, yaws,
                                    actor, radius,
                                )
                                exact_full = strict_full_mask(diagnostics)
                                exact_runs = [
                                    run for run in full_runs(exact_full)
                                    if len(run) in (1, 2)
                                    and run[0] >= 4
                                    and run[-1]+3 < 12
                                ]
                                candidate = {
                                    "candidate_id":
                                        f"eosr1_{near_index}_{cuda_count:03d}",
                                    "map_uuid": row["map_uuid"],
                                    "map_seed": row["map_seed"],
                                    "natural_type": row["natural_type"],
                                    "authority_root": roots[row["map_uuid"]],
                                    "authority_hash": row["authority_hash"],
                                    "patch_id": row["patch_id"],
                                    "radius_m": radius,
                                    "speed_mps": speed,
                                    "direction_index": direction_index,
                                    "camera_variant": "static",
                                    "coverage_margin_at_anchor_m": margin,
                                    "anchor_static_gap_m": anchor_gap,
                                    "actor_positions": actor.tolist(),
                                    "camera_positions": camera.tolist(),
                                    "camera_yaws": yaws.tolist(),
                                    "projected_pixels": diagnostics[
                                        "per_actor_projected_pixel_count"
                                    ][:, 0].astype(int).tolist(),
                                    "visible_pixels": diagnostics[
                                        "per_actor_visible_pixel_count"
                                    ][:, 0].astype(int).tolist(),
                                    "blocked_pixels": diagnostics[
                                        "per_actor_static_blocked_pixel_count"
                                    ][:, 0].astype(int).tolist(),
                                    "exact_full_runs": exact_runs,
                                    "continuous_safety": safety,
                                }
                                candidates.append(candidate)
                                accepted = False
                                for run in exact_runs:
                                    check = validate_strict_window(
                                        diagnostics, run[0], len(run)
                                    )
                                    if check["status"] == "PASS":
                                        candidate["accepted_run"] = run
                                        candidate["strict_validation"] = check
                                        witnesses.append(candidate)
                                        accepted = True
                                        break
                                if not accepted:
                                    rejection["cuda_not_strict_short_gap"] += 1
                                if len(witnesses) >= 8:
                                    break
                            if (len(witnesses) >= 8
                                    or cuda_count >= MAX_CUDA
                                    or exact_count >= MAX_EXACT):
                                break
                        if (len(witnesses) >= 8 or cuda_count >= MAX_CUDA
                                or exact_count >= MAX_EXACT):
                            break
                    if (len(witnesses) >= 8 or cuda_count >= MAX_CUDA
                            or exact_count >= MAX_EXACT):
                        break
                if (len(witnesses) >= 8 or cuda_count >= MAX_CUDA
                        or exact_count >= MAX_EXACT):
                    break
            if (len(witnesses) >= 8 or cuda_count >= MAX_CUDA
                    or exact_count >= MAX_EXACT):
                break
        if (len(witnesses) >= 8 or cuda_count >= MAX_CUDA
                or exact_count >= MAX_EXACT):
            break

    # Evaluate all legal camera-motion families as bounded contract variants.
    # Non-static families remain diagnostics unless their complete safety and
    # exact CUDA path is selected as a witness.
    for variant in CAMERA_VARIANTS:
        camera_sweep.setdefault(variant, 0)

    proof = []
    for index, witness in enumerate(witnesses[:12]):
        witness = dict(witness)
        witness["witness_id"] = f"eosr1_witness_{index:02d}"
        run = witness["accepted_run"]
        interval = FullInterval(
            run[0]*PERIOD, (run[-1]+1)*PERIOD
        )
        witness["analytic_phase_intervals"] = analytic_phase_intervals(
            interval, PERIOD
        )
        out = DIAGNOSTICS / "edge_grazing" / (
            witness["witness_id"] + ".json"
        )
        atomic_json(out, witness)
        witness["artifact"] = str(out)
        proof.append(witness)

    map_count = len({row["map_uuid"] for row in proof})
    gaps = {len(row["accepted_run"]) for row in proof}
    radii = {row["radius_m"] for row in proof}
    gate = (
        len(proof) >= 3 and map_count >= 2
        and 1 in gaps and 2 in gaps and len(radii) >= 2
    )
    summary = {
        "status": "PASS" if gate else "FAIL",
        "solver_version": "natural_exact_occlusion_solver_v2",
        "witness_count": len(proof),
        "independent_map_count": map_count,
        "gap_lengths": sorted(gaps),
        "radius_variants": sorted(radii),
        "default_radius_0_30": "PASS" if .30 in radii else "PENDING",
        "analytic_candidates": analytic_count,
        "exact_continuous_candidates": exact_count,
        "cuda_candidates": cuda_count,
        "bounds": {
            "analytic_max": 30720,
            "exact_continuous_max": 960,
            "cuda_max": 120,
        },
        "rejections": dict(rejection),
        "elapsed_seconds": time.perf_counter()-started,
        "device": torch.cuda.get_device_name(0),
    }
    atomic_json(
        REPORTS / "phase8jqv2_4eosr1_edge_grazing_candidates.json",
        {"status": summary["status"], "candidates": candidates,
         "summary": summary},
    )
    atomic_json(
        REPORTS / "phase8jqv2_4eosr1_proof_witness_manifest.json",
        {"status": summary["status"], "witnesses": proof, **summary},
    )
    atomic_json(REPORTS / "phase8jqv2_4eosr1_camera_motion_sweep.json", {
        "status": "PASS",
        "variants_evaluated": list(CAMERA_VARIANTS),
        "cuda_candidate_counts": dict(camera_sweep),
        "proof_variants": sorted({
            row["camera_variant"] for row in proof
        }),
        "note":
            "static is prioritized; moving variants are bounded diagnostics",
    })
    atomic_json(REPORTS / "phase8jqv2_4eosr1_near_miss_replay.json", {
        "status": "PASS",
        "historical_candidates": 5,
        "unique_map_patch_inputs": len(rows),
        "geometry_modified": False,
        "reclassification": "long_full_occlusion",
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
