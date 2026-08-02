#!/usr/bin/env python3
"""CE1 E3 bounded explicit-gap search on profile-variant maps."""

from __future__ import annotations

from collections import Counter
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
    camera_motion, propose_at_gap_origins,
)
from authoritative_dataset.occlusion_constructor_v2_2 import ACTOR_RADIUS_M
from generate_phase8jqv2_4rr1_corpus import SENSOR, validate_case
from run_phase8jqv2_4ce1_existing_search import (
    BudgetedRenderer, atomic_json, save_case,
)
from run_phase8jqv2_4ce1_new_map_search import ranked_candidates


MAP_MANIFEST = ROOT / "reports/phase8jqv2_4ce1_e3_map_manifest.json"
STAGING = ROOT / "data/phase8_natural_representation_audit_v2_staging/profile_variant_cases"
REPORT = ROOT / "reports/phase8jqv2_4ce1_e3_search.json"
BUDGET_REPORT = ROOT / "reports/phase8jqv2_4ce1_e3_search_budget.json"
REJECTIONS = ROOT / "diagnostics/phase8jqv2_4ce1/new_maps/e3_search_rejections.jsonl"
E2_BUDGET = ROOT / "reports/phase8jqv2_4ce1_new_map_search_budget.json"
TYPE_NAMES = {2: "pillar", 6: "room", 7: "wall"}


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is required for CE1 E3 search")
    if STAGING.exists():
        raise FileExistsError(f"refusing to overwrite: {STAGING}")
    STAGING.mkdir(parents=True)
    manifest = json.loads(MAP_MANIFEST.read_text())
    maps = sorted(manifest["maps"], key=lambda row: (
        {6: 0, 7: 1, 2: 2}[int(row["maze_type"])],
        int(row["seed"]),
    ))
    contract = load_motion_contract(
        ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml"
    )
    frame_times = np.arange(60, dtype=np.float64) * .1
    renderer = CudaAuthorityRenderer(SENSOR, "cuda:0")
    accepted, rejected, results = [], [], []
    total = Counter()
    started = time.perf_counter()
    for map_index, original in enumerate(maps):
        map_started = time.perf_counter()
        row = {
            **original,
            "authority_hash": original["authority_manifest_hash"],
            "source": "ce1_e3_original_parameter_profile_variant",
        }
        backend = ExactAuthorityBVH(row["authority_root"])
        budgeted = BudgetedRenderer(renderer, 10)
        proposals, _, scored = ranked_candidates(backend, frame_times)
        selected = [value for _, value in scored[:4]]
        exact_used = 0
        evidence_cuda = 0
        accepted_row = None
        for attempt_index, (proposal, gap_origin) in enumerate(zip(
            selected, (8, 12, 16, 20)
        )):
            allocation = min(5, 20 - exact_used)
            if allocation <= 0 or budgeted.calls >= 10:
                break
            try:
                expanded = camera_motion(
                    proposal, "static", frame_times, backend
                )
            except RuntimeError as error:
                exact_used += allocation
                rejected.append({
                    "stage": "E3_camera_motion",
                    "reason": f"{type(error).__name__}: {error}",
                    "map_uuid": row["map_uuid"],
                    "map_seed": int(row["seed"]),
                    "profile": row["profile_name"],
                    "attempt": attempt_index,
                    "gap_origin": gap_origin,
                    "cleanup_eligibility": "retain_diagnostic",
                })
                continue
            accounted = False
            trajectory_seed = (
                1_980_000_000 + map_index * 1009
                + attempt_index * 41 + int(row["seed"])
            ) % (2**31 - 1)
            try:
                construction = propose_at_gap_origins(
                    backend, budgeted, expanded, frame_times, contract,
                    gap_origins=(gap_origin,),
                    maximum_candidates=allocation,
                )
                exact_used += int(construction.attempts)
                accounted = True
                actor_frames = actor_position(
                    construction.actors[0], frame_times
                )[:, None, :]
                diagnostics = renderer.render_with_actor_diagnostics(
                    backend, expanded.position_world, expanded.yaw,
                    actor_frames, [ACTOR_RADIUS_M],
                    return_owner_map=True, return_actor_near_depth=True,
                )
                evidence_cuda += 1
                construction.diagnostics.update(diagnostics)
                validation = validate_case(
                    row, backend, construction,
                    expanded.position_world, expanded.yaw, actor_frames,
                    frame_times, contract, expanded.certificate,
                )
                gap = validation["gap_start"]
                visible = diagnostics[
                    "per_actor_visible_pixel_count"
                ][:, 0]
                if not (
                    gap >= 4 and gap + 3 < len(visible)
                    and np.all(visible[gap-4:gap] > 0)
                    and np.all(visible[gap+1:gap+4] > 0)
                ):
                    raise RuntimeError(
                        "expanded_pre4_post3_visibility_failed"
                    )
                case_id = (
                    f"ce1_gap1_{TYPE_NAMES[int(row['maze_type'])]}_"
                    f"{row['map_uuid'][:8]}_{trajectory_seed}"
                )
                accepted_row = save_case(
                    case_id, row, trajectory_seed, construction, expanded,
                    actor_frames, frame_times, validation,
                    staging_root=STAGING,
                )
                accepted_row["source_stage"] = "E3_profile_variant_map"
                accepted.append(accepted_row)
                break
            except Exception as error:
                if not accounted:
                    exact_used += allocation
                rejected.append({
                    "stage": "E3_exact_continuous_cuda_search",
                    "reason": f"{type(error).__name__}: {error}",
                    "map_uuid": row["map_uuid"],
                    "map_seed": int(row["seed"]),
                    "maze_type": int(row["maze_type"]),
                    "natural_type": TYPE_NAMES[int(row["maze_type"])],
                    "profile": row["profile_name"],
                    "attempt": attempt_index,
                    "gap_origin": gap_origin,
                    "patch_voxel_offset": expanded.patch_voxel_offset,
                    "hash": row["authority_hash"],
                    "cleanup_eligibility": "retain_diagnostic",
                })
        total["p0"] += min(128, len(proposals))
        total["exact"] += exact_used
        total["cuda"] += budgeted.calls
        results.append({
            "map_uuid": row["map_uuid"],
            "natural_type": TYPE_NAMES[int(row["maze_type"])],
            "map_seed": int(row["seed"]),
            "profile": row["profile_name"],
            "p0_used": min(128, len(proposals)),
            "exact_continuous_used": exact_used,
            "cuda_candidates_used": budgeted.calls,
            "accepted_evidence_cuda_used": evidence_cuda,
            "accepted_case_id":
                accepted_row["case_id"] if accepted_row else None,
            "elapsed_seconds": time.perf_counter() - map_started,
        })
        print(json.dumps(results[-1]), flush=True)
    REJECTIONS.parent.mkdir(parents=True, exist_ok=True)
    with REJECTIONS.open("w") as handle:
        for row in rejected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    e2_cuda = json.loads(E2_BUDGET.read_text())["used"].get("cuda", 0)
    value = {
        "status": "PASS",
        "stage": "E3_bounded_profile_variant_search",
        "accepted_case_count": len(accepted),
        "accepted_cases": accepted,
        "maps": results,
        "budgets": {
            "per_map": {"p0": 128, "exact_continuous": 20, "cuda": 10},
            "used": dict(total),
            "e2_plus_e3_cuda": e2_cuda + total["cuda"],
            "new_map_cuda_hard_total": 260,
            "within_budget": (
                all(row["p0_used"] <= 128
                    and row["exact_continuous_used"] <= 20
                    and row["cuda_candidates_used"] <= 10
                    for row in results)
                and e2_cuda + total["cuda"] <= 260
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "declared_runtime_limit_seconds": 1800,
        "peak_cpu_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "detector_output_used_for_selection": False,
        "representation_output_used_for_selection": False,
        "tracker_output_used_for_selection": False,
        "formal_generation_started": False,
        "test_accessed": False,
        "blind_accessed": False,
        "training_started": False,
    }
    atomic_json(REPORT, value)
    atomic_json(BUDGET_REPORT, {
        "status": "PASS" if value["budgets"]["within_budget"] else "FAIL",
        **value["budgets"],
        "elapsed_seconds": value["elapsed_seconds"],
    })
    print(json.dumps({
        "status": value["status"],
        "accepted": len(accepted),
        "maps_searched": len(results),
        "budgets": value["budgets"],
    }, indent=2))


if __name__ == "__main__":
    main()
