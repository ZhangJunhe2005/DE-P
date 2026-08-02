#!/usr/bin/env python3
"""CE1 E2 bounded Gap-1 search on the frozen new development maps."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from authoritative_dataset.cuda_renderer_v1 import CudaAuthorityRenderer
from authoritative_dataset.dynamic_motion_v2 import (
    actor_position, load_motion_contract,
)
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.natural_gap1_corpus_expander_v1 import (
    CAMERA_MOTIONS, camera_motion, propose_one,
)
from authoritative_dataset.natural_observable_occlusion_proposer_v1 import (
    propose_natural_cameras,
)
from authoritative_dataset.occlusion_constructor_v2_2 import ACTOR_RADIUS_M
from generate_phase8jqv2_4rr1_corpus import SENSOR, validate_case
from run_phase8jqv2_4ce1_existing_search import (
    BudgetedRenderer, atomic_json, save_case,
)


MAP_MANIFEST = ROOT / "reports/phase8jqv2_4ce1_new_map_manifest.json"
STAGING = ROOT / "data/phase8_natural_representation_audit_v2_staging/new_map_cases"
REPORT = ROOT / "reports/phase8jqv2_4ce1_new_map_search.json"
BUDGET_REPORT = ROOT / "reports/phase8jqv2_4ce1_new_map_search_budget.json"
REJECTIONS = ROOT / "diagnostics/phase8jqv2_4ce1/new_maps/search_rejections.jsonl"
TYPE_NAMES = {2: "pillar", 6: "room", 7: "wall"}


def corridor_score(backend, proposal, frame_times):
    """P0 coarse actor-corridor score; never treated as acceptance evidence."""
    camera = np.asarray(proposal.position_world[0], dtype=np.float64)
    patch = np.asarray(proposal.patch_center, dtype=np.float64)
    ray = patch - camera
    distance = float(np.linalg.norm(ray))
    ray /= max(distance, 1e-12)
    normal = np.asarray(proposal.patch_normal, dtype=np.float64)
    tangent = np.asarray([-normal[1], normal[0], 0.0])
    tangent -= ray * float(tangent @ ray)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
    # The v2.2 feasible camera proposer already proved at least one interval.
    # P0 ranks full-horizon free corridors without claiming exact safety.
    center_time = 1.4
    horizon = float(frame_times[-1] + 1.7)
    sample_times = np.linspace(0.0, horizon, 24)
    best = -1
    for ratio in (1.9, 2.2, 2.5, 2.8, 3.1):
        if ratio * distance > 1.8:
            continue
        anchor = camera + ray * ratio * distance
        for radial_sign in (-1.0, 1.0):
            for tangent_sign in (-1.0, 1.0):
                speed = 1.55
                tangent_speed = .85 * speed
                radial_speed = np.sqrt(speed**2 - tangent_speed**2)
                velocity = (
                    radial_sign * ray * radial_speed
                    + tangent_sign * tangent * tangent_speed
                )
                start = anchor - velocity * center_time
                path = start + sample_times[:, None] * velocity
                free = 0
                for point in path:
                    if backend.query_one(point, ACTOR_RADIUS_M)["collision"]:
                        break
                    if float(np.linalg.norm(point - camera)) <= .70:
                        break
                    free += 1
                best = max(best, free)
    return best


def ranked_candidates(backend, frame_times):
    proposals = propose_natural_cameras(
        backend, frame_times, 1,
        maximum_proposals=128, maximum_patch_checks=128,
    )
    scored = sorted(
        ((corridor_score(backend, proposal, frame_times), proposal)
         for proposal in proposals),
        key=lambda item: (
            -item[0], item[1].proposal_rank,
            item[1].patch_voxel_offset,
        ),
    )
    output = []
    # Two independent base patches, with deterministic motion diversity.
    for rank, (_, proposal) in enumerate(scored[:4]):
        motion = CAMERA_MOTIONS[rank % len(CAMERA_MOTIONS)]
        try:
            output.append(camera_motion(
                proposal, motion, frame_times, backend
            ))
        except RuntimeError:
            continue
    return proposals, output, scored


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is required for CE1 E2 search")
    if STAGING.exists():
        raise FileExistsError(f"refusing to overwrite: {STAGING}")
    STAGING.mkdir(parents=True)
    manifest = json.loads(MAP_MANIFEST.read_text())
    if manifest["status"] != "PASS" or manifest["map_count"] != 18:
        raise RuntimeError("new development map manifest is invalid")
    maps = sorted(manifest["maps"], key=lambda row: (
        {6: 0, 7: 1, 2: 2}[int(row["maze_type"])],
        int(row["seed"]),
    ))
    contract = load_motion_contract(
        ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml"
    )
    frame_times = np.arange(60, dtype=np.float64) * .1
    renderer = CudaAuthorityRenderer(SENSOR, "cuda:0")
    accepted, rejection_rows, map_results = [], [], []
    total = Counter()
    started = time.perf_counter()
    for map_index, original in enumerate(maps):
        map_started = time.perf_counter()
        map_row = {
            **original,
            "authority_hash": original["authority_manifest_hash"],
            "source": "ce1_e2_frozen_profile_v2_new_seed",
        }
        backend = ExactAuthorityBVH(map_row["authority_root"])
        budgeted = BudgetedRenderer(renderer, 10)
        proposals, candidates, scored = ranked_candidates(
            backend, frame_times
        )
        exact_used = 0
        evidence_cuda = 0
        accepted_row = None
        for candidate_index, proposal in enumerate(candidates[:2]):
            allocation = min(10, 20 - exact_used)
            if allocation <= 0 or budgeted.calls >= 10:
                break
            trajectory_seed = (
                1_972_000_000 + map_index * 1009
                + candidate_index * 37 + int(map_row["seed"])
            ) % (2**31 - 1)
            accounted = False
            try:
                construction = propose_one(
                    backend, budgeted, proposal, frame_times, contract,
                    actor_candidate_budget=allocation,
                )
                exact_used += int(construction.attempts)
                accounted = True
                actor_frames = actor_position(
                    construction.actors[0], frame_times
                )[:, None, :]
                diagnostics = renderer.render_with_actor_diagnostics(
                    backend, proposal.position_world, proposal.yaw,
                    actor_frames, [ACTOR_RADIUS_M],
                    return_owner_map=True, return_actor_near_depth=True,
                )
                evidence_cuda += 1
                construction.diagnostics.update(diagnostics)
                validation = validate_case(
                    map_row, backend, construction,
                    proposal.position_world, proposal.yaw, actor_frames,
                    frame_times, contract, proposal.certificate,
                )
                gap = validation["gap_start"]
                visible = diagnostics[
                    "per_actor_visible_pixel_count"
                ][:, 0]
                if gap < 4 or gap + 3 >= len(visible):
                    raise RuntimeError("insufficient_expanded_pre_post_window")
                if not np.all(visible[gap-4:gap] > 0):
                    raise RuntimeError("insufficient_pre_visibility")
                if not np.all(visible[gap+1:gap+4] > 0):
                    raise RuntimeError("insufficient_post_visibility")
                case_id = (
                    f"ce1_gap1_{TYPE_NAMES[int(map_row['maze_type'])]}_"
                    f"{map_row['map_uuid'][:8]}_{trajectory_seed}"
                )
                accepted_row = save_case(
                    case_id, map_row, trajectory_seed, construction,
                    proposal, actor_frames, frame_times, validation,
                    staging_root=STAGING,
                )
                accepted_row["source_stage"] = "E2_new_development_map"
                accepted.append(accepted_row)
                break
            except Exception as error:
                if not accounted:
                    exact_used += allocation
                rejection_rows.append({
                    "stage": "E2_exact_continuous_cuda_search",
                    "reason": f"{type(error).__name__}: {error}",
                    "map_uuid": map_row["map_uuid"],
                    "map_seed": int(map_row["seed"]),
                    "maze_type": int(map_row["maze_type"]),
                    "natural_type": TYPE_NAMES[int(map_row["maze_type"])],
                    "profile": map_row["profile_name"],
                    "attempt": candidate_index,
                    "camera_motion": proposal.certificate["camera_motion"],
                    "patch_voxel_offset": proposal.patch_voxel_offset,
                    "p0_corridor_score": next(
                        score for score, value in scored
                        if value.patch_voxel_offset
                        == proposal.patch_voxel_offset
                    ),
                    "hash": map_row["authority_hash"],
                    "cleanup_eligibility": "retain_diagnostic",
                })
        total["p0"] += min(128, len(proposals))
        total["exact"] += exact_used
        total["cuda"] += budgeted.calls
        map_results.append({
            "map_uuid": map_row["map_uuid"],
            "natural_type": TYPE_NAMES[int(map_row["maze_type"])],
            "map_seed": int(map_row["seed"]),
            "p0_used": min(128, len(proposals)),
            "exact_continuous_used": exact_used,
            "cuda_candidates_used": budgeted.calls,
            "accepted_evidence_cuda_used": evidence_cuda,
            "accepted_case_id":
                accepted_row["case_id"] if accepted_row else None,
            "elapsed_seconds": time.perf_counter() - map_started,
        })
        print(json.dumps(map_results[-1]), flush=True)
        combined = json.loads(
            (ROOT / "data/phase8_natural_representation_audit_v1/manifest.json").read_text()
        )["unused_valid_cases"] + accepted
        type_counts = Counter(row["natural_type"] for row in combined)
        hard = (
            len({row["map_uuid"] for row in combined}) >= 6
            and len(type_counts) >= 3
            and len({row["map_seed"] for row in combined}) >= 6
            and any(row["natural_type"] not in ("cave", "forest")
                    for row in combined)
            and max(type_counts.values()) / len({
                row["map_uuid"] for row in combined
            }) <= .5
        )
        recommended = (
            hard
            and len({row["map_uuid"] for row in combined}) >= 8
            and len(type_counts) >= 4
        )
        if recommended:
            break
    REJECTIONS.parent.mkdir(parents=True, exist_ok=True)
    with REJECTIONS.open("w") as handle:
        for row in rejection_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    payload = {
        "status": "PASS",
        "stage": "E2_new_development_map_bounded_search",
        "accepted_case_count": len(accepted),
        "accepted_cases": accepted,
        "maps": map_results,
        "budgets": {
            "per_map": {"p0": 128, "exact_continuous": 20, "cuda": 10},
            "new_map_cuda_hard_total": 260,
            "used": dict(total),
            "within_budget": (
                all(row["p0_used"] <= 128
                    and row["exact_continuous_used"] <= 20
                    and row["cuda_candidates_used"] <= 10
                    for row in map_results)
                and total["cuda"] <= 260
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "declared_runtime_limit_seconds": 2400,
        "peak_cpu_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "detector_output_used_for_selection": False,
        "representation_output_used_for_selection": False,
        "tracker_output_used_for_selection": False,
        "test_accessed": False,
        "blind_accessed": False,
        "formal_generation_started": False,
        "training_started": False,
    }
    atomic_json(REPORT, payload)
    atomic_json(BUDGET_REPORT, {
        "status": "PASS" if payload["budgets"]["within_budget"] else "FAIL",
        **payload["budgets"],
        "elapsed_seconds": payload["elapsed_seconds"],
    })
    print(json.dumps({
        "status": payload["status"],
        "accepted": len(accepted),
        "maps_searched": len(map_results),
        "budgets": payload["budgets"],
    }, indent=2))


if __name__ == "__main__":
    main()
