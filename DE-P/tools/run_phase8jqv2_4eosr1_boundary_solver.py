#!/usr/bin/env python3
"""Exact boundary-local EOSR1 search rooted in frozen MTC1 safe paths."""

from __future__ import annotations

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
    FullInterval, analytic_phase_intervals, contiguous_intervals,
    strict_full_mask, validate_strict_window,
)
from authoritative_dataset.occlusion_constructor_v2_2 import (
    canonical_surface_patches,
)
from generate_phase8jqv2_4rr1_corpus import SENSOR
from run_phase8jqv2_4eosr1_solver import (
    atomic_json, authorities, collision_safe, exact_run, full_runs,
)

PERIOD = SENSOR["frame_period_ns"] * 1e-9


def normalized(value):
    value = np.asarray(value, dtype=np.float64)
    return value/max(np.linalg.norm(value), 1e-12)


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    started = time.perf_counter()
    source = json.loads((
        REPORTS / "phase8jqv2_4mtc1_room_wall_near_miss_manifest.json"
    ).read_text())
    roots = authorities()
    renderer = CudaAuthorityRenderer(SENSOR, "cuda:0")
    cuda_calls = 0
    exact_candidates = 0
    calibration = []
    candidates = []
    witnesses = []
    for near_index, row in enumerate(source["candidates"]):
        near_cuda_start = cuda_calls
        backend = ExactAuthorityBVH(roots[row["map_uuid"]])
        patch = next(
            patch for patch in canonical_surface_patches(backend)
            if patch.voxel_offset == row["patch_id"]
        )
        base = np.asarray(row["camera_positions"][0], dtype=np.float64)
        yaw = float(row["camera_yaw"][0])
        camera = np.repeat(base[None, :], 12, axis=0)
        yaws = np.full(12, yaw)
        p0 = np.asarray(row["actor_positions"][0], dtype=np.float64)
        old_velocity = np.asarray(
            row["actor_velocity_mps"], dtype=np.float64
        )
        old_direction = normalized(old_velocity)
        # Calibrate the true boundary independently for every radius using
        # canonical CUDA at 5 ms. This retains the frozen trajectory and patch.
        dense_times = np.arange(0., 5.9001, .005)
        dense_actor = (
            p0[None, :] + dense_times[:, None]*old_velocity
        )
        dense_camera = np.repeat(base[None, :], len(dense_times), axis=0)
        dense_yaws = np.full(len(dense_times), yaw)
        for radius in (.20, .25, .30):
            if cuda_calls-near_cuda_start >= 24:
                break
            dense = exact_run(
                renderer, backend, dense_camera, dense_yaws,
                dense_actor, radius,
            )
            cuda_calls += 1
            intervals = contiguous_intervals(
                dense_times, strict_full_mask(dense), .005
            )
            calibration.append({
                "near_miss_index": near_index,
                "map_uuid": row["map_uuid"],
                "patch_id": row["patch_id"],
                "radius_m": radius,
                "intervals": [
                    {
                        "enter_s": item.enter_s,
                        "exit_s": item.exit_s,
                        "duration_s": item.duration_s,
                    } for item in intervals
                ],
            })
            if not intervals:
                continue
            # Longest interval is the frozen near-miss shadow. Candidate lines
            # graze its entry and exit boundaries in the physical patch plane.
            interval = max(intervals, key=lambda item: item.duration_s)
            horizontal = normalized(patch.tangent)
            vertical = np.asarray((0., 0., 1.))
            directions = (
                vertical, -vertical,
                normalized(horizontal+vertical),
                normalized(horizontal-vertical),
                normalized(-horizontal+vertical),
                normalized(-horizontal-vertical),
                horizontal, -horizontal,
            )
            # Entry and exit are both silhouette boundaries. Small inward
            # offsets select a corner chord rather than the shadow center.
            boundary_times = (
                (interval.enter_s, 1.),
                (interval.exit_s, -1.),
            )
            radius_candidate_calls = 0
            for boundary_time, inward_sign in boundary_times:
                boundary = p0 + boundary_time*old_velocity
                for direction_index, direction in enumerate(directions):
                    if (cuda_calls-near_cuda_start >= 24
                            or radius_candidate_calls >= 7):
                        break
                    for inward_s, speed in (
                        (.005, 1.4), (.030, 1.1), (.060, .8)
                    ):
                        center = (
                            boundary
                            + inward_sign*old_direction*speed*inward_s
                        )
                        actor = (
                            center[None, :]
                            + (np.arange(12)-5)[:, None]
                            * PERIOD*speed*direction[None, :]
                        )
                        safe, safety = collision_safe(
                            backend, actor, radius, camera
                        )
                        if not safe:
                            continue
                        exact_candidates += 1
                        diagnostics = exact_run(
                            renderer, backend, camera, yaws,
                            actor, radius,
                        )
                        cuda_calls += 1
                        radius_candidate_calls += 1
                        runs = [
                            run for run in full_runs(
                                strict_full_mask(diagnostics)
                            )
                            if len(run) in (1, 2)
                            and run[0] >= 4 and run[-1]+3 < 12
                        ]
                        candidate = {
                            "candidate_id":
                                f"eosr1_boundary_{near_index}_{cuda_calls:03d}",
                            "near_miss_index": near_index,
                            "map_uuid": row["map_uuid"],
                            "map_seed": row["map_seed"],
                            "natural_type": row["natural_type"],
                            "authority_root": roots[row["map_uuid"]],
                            "authority_hash": row["authority_hash"],
                            "patch_id": row["patch_id"],
                            "radius_m": radius,
                            "speed_mps": speed,
                            "acceleration_mps2": 0.,
                            "camera_variant": "static",
                            "boundary":
                                "entry" if inward_sign > 0 else "exit",
                            "direction_index": direction_index,
                            "inward_offset_s": inward_s,
                            "actor_positions": actor.tolist(),
                            "camera_positions": camera.tolist(),
                            "camera_yaws": yaws.tolist(),
                            "continuous_safety": safety,
                            "projected_pixels": diagnostics[
                                "per_actor_projected_pixel_count"
                            ][:, 0].astype(int).tolist(),
                            "visible_pixels": diagnostics[
                                "per_actor_visible_pixel_count"
                            ][:, 0].astype(int).tolist(),
                            "blocked_pixels": diagnostics[
                                "per_actor_static_blocked_pixel_count"
                            ][:, 0].astype(int).tolist(),
                            "exact_full_runs": runs,
                        }
                        candidates.append(candidate)
                        for run in runs:
                            check = validate_strict_window(
                                diagnostics, run[0], len(run)
                            )
                            if check["status"] == "PASS":
                                candidate["accepted_run"] = run
                                candidate["strict_validation"] = check
                                witnesses.append(candidate)
                                break
                        if radius_candidate_calls >= 7:
                            break
                    if radius_candidate_calls >= 7:
                        break
                if radius_candidate_calls >= 7:
                    break
        if cuda_calls >= 120:
            break

    proof = []
    for index, row in enumerate(witnesses[:12]):
        value = dict(row)
        value["witness_id"] = f"eosr1_witness_{index:02d}"
        run = value["accepted_run"]
        value["analytic_phase_intervals"] = analytic_phase_intervals(
            FullInterval(run[0]*PERIOD, (run[-1]+1)*PERIOD), PERIOD
        )
        path = DIAGNOSTICS / "edge_grazing" / (
            value["witness_id"] + ".json"
        )
        atomic_json(path, value)
        value["artifact"] = str(path)
        proof.append(value)
    map_count = len({row["map_uuid"] for row in proof})
    gaps = {len(row["accepted_run"]) for row in proof}
    radii = {row["radius_m"] for row in proof}
    gate = (
        len(proof) >= 3 and map_count >= 2 and 1 in gaps and 2 in gaps
        and len(radii) >= 2
    )
    summary = {
        "status": "PASS" if gate else "FAIL",
        "witness_count": len(proof),
        "independent_map_count": map_count,
        "gap_lengths": sorted(gaps),
        "radius_variants": sorted(radii),
        "default_radius_0_30": "PASS" if .30 in radii else "PENDING",
        "exact_continuous_candidates": exact_candidates,
        "cuda_calls_including_boundary_calibration": cuda_calls,
        "elapsed_seconds": time.perf_counter()-started,
        "device": torch.cuda.get_device_name(0),
    }
    atomic_json(
        REPORTS / "phase8jqv2_4eosr1_continuous_interval_solver.json",
        {"status": "PASS", "calibration": calibration},
    )
    atomic_json(
        REPORTS / "phase8jqv2_4eosr1_edge_grazing_candidates.json",
        {"status": summary["status"], "candidates": candidates,
         "summary": summary},
    )
    atomic_json(
        REPORTS / "phase8jqv2_4eosr1_proof_witness_manifest.json",
        {"status": summary["status"], "witnesses": proof, **summary},
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
