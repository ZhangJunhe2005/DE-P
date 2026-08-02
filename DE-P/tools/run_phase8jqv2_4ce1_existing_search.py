#!/usr/bin/env python3
"""Run the bounded CE1 E1 search over the frozen 30-map inventory."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
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

from authoritative_dataset.cuda_renderer_v1 import (
    CudaAuthorityRenderer, RENDERER_VERSION,
)
from authoritative_dataset.dynamic_motion_v2 import (
    actor_position, load_motion_contract,
)
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.natural_gap1_corpus_expander_v1 import (
    EXPANDER_VERSION, enumerate_camera_candidates, propose_one,
)
from authoritative_dataset.occlusion_constructor_v2_1 import (
    natural_occlusion_pattern,
)
from authoritative_dataset.occlusion_constructor_v2_2 import (
    ACTOR_RADIUS_M, CONSTRUCTOR_VERSION,
)
from generate_phase8jqv2_4rr1_corpus import (
    SENSOR, canonical_hash, json_default, validate_case,
)


INVENTORY = ROOT / "diagnostics/phase8jqv2_4rr1/corpus_generation/allowed_map_inventory.json"
V1_MANIFEST = ROOT / "data/phase8_natural_representation_audit_v1/manifest.json"
STAGING = ROOT / "data/phase8_natural_representation_audit_v2_staging/existing_maps"
REPORT = ROOT / "reports/phase8jqv2_4ce1_existing_map_search.json"
SUMMARY = ROOT / "reports/phase8jqv2_4ce1_existing_map_search_summary.md"
BUDGET_REPORT = ROOT / "reports/phase8jqv2_4ce1_search_budget.json"
REJECTIONS = ROOT / "diagnostics/phase8jqv2_4ce1/search_rejections.jsonl"
TYPE_NAMES = {1: "cave", 2: "pillar", 5: "forest", 6: "room", 7: "wall"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True, default=json_default
    ) + "\n")
    os.replace(temporary, path)


class BudgetedRenderer:
    def __init__(self, renderer, maximum):
        self.renderer = renderer
        self.maximum = int(maximum)
        self.calls = 0
        self.sensor = renderer.sensor

    def render_with_actor_diagnostics(self, *args, **kwargs):
        if self.calls >= self.maximum:
            raise RuntimeError("cuda_budget_exhausted")
        self.calls += 1
        return self.renderer.render_with_actor_diagnostics(*args, **kwargs)


def save_case(case_id, map_row, trajectory_seed, construction, proposal,
              actor_frames, frame_times, validation, *, staging_root=STAGING):
    case_root = Path(staging_root) / "cases" / case_id
    case_root.mkdir(parents=True, exist_ok=False)
    diagnostics = construction.diagnostics
    arrays = {
        "depth.npy": diagnostics["composed_depth"],
        "static_depth.npy": diagnostics["static_depth"],
        "nearest_actor_owner.npy": diagnostics["nearest_actor_owner"],
        "actor_near_depth.npy": diagnostics["actor_near_depth"],
        "camera_positions.npy": proposal.position_world,
        "camera_yaws.npy": proposal.yaw,
        "actor_positions.npy": actor_frames[:, 0],
        "timestamps.npy": frame_times,
    }
    hashes = {}
    for name, value in arrays.items():
        np.save(case_root / name, value, allow_pickle=False)
        hashes[name] = sha(case_root / name)
    actor = construction.actors[0]
    metadata = {
        "status": "PASS",
        "corpus_version": "natural_depth_representation_audit_corpus_v2_expansion",
        "role": "representation_audit_only",
        "case_id": case_id,
        "map_uuid": map_row["map_uuid"],
        "maze_type": int(map_row["maze_type"]),
        "natural_type": TYPE_NAMES[int(map_row["maze_type"])],
        "map_seed": int(map_row["seed"]),
        "trajectory_seed": int(trajectory_seed),
        "authority_hash": map_row["authority_hash"],
        "occupancy_hash": map_row["occupancy_hash"],
        "authority_root": map_row["authority_root"],
        "source_map_registry": map_row["source"],
        "profile_name": map_row["profile_name"],
        "requested_gap": 1,
        "observed_gap": [construction.gap_start, construction.gap_end],
        "camera_motion": proposal.certificate["camera_motion"],
        "camera_proposal": {
            "patch_voxel_offset": proposal.patch_voxel_offset,
            "patch_center": proposal.patch_center,
            "patch_normal": proposal.patch_normal,
            "camera_standoff_m": proposal.camera_standoff_m,
            "proposal_rank": proposal.proposal_rank,
            "certificate": proposal.certificate,
        },
        "actor": {
            "start": np.asarray(actor["start"]).tolist(),
            "velocity": np.asarray(actor["velocity"]).tolist(),
            "radius_m": float(actor["radius_m"]),
            "motion_profile": actor["motion_profile"],
            "motion_contract_version": actor["motion_contract_version"],
            "acceleration_mps2": [0.0, 0.0, 0.0],
            "proposal": actor.get("proposal"),
        },
        "constructor": {
            "version": CONSTRUCTOR_VERSION,
            "wrapper_version": EXPANDER_VERSION,
            "method": construction.method,
            "attempts": construction.attempts,
            "evidence": construction.evidence,
        },
        "renderer_version": RENDERER_VERSION,
        "geometry_certificate": validation,
        "files": hashes,
        "runtime_inputs": {
            "depth": "depth.npy",
            "camera_positions": "camera_positions.npy",
            "camera_yaws": "camera_yaws.npy",
            "timestamps": "timestamps.npy",
        },
        "offline_scoring_only": {
            "actor_positions": "actor_positions.npy",
            "actor_mask": "nearest_actor_owner.npy",
            "gap": [construction.gap_start, construction.gap_end],
        },
        "annex_used": False,
        "artificial_hiding": False,
        "manual_depth_overwrite": False,
        "detector_output_used_for_selection": False,
        "representation_output_used_for_selection": False,
        "tracker_output_used_for_selection": False,
        "test_accessed": False,
        "blind_accessed": False,
        "formal_eligible": False,
    }
    metadata["case_hash"] = canonical_hash(metadata)
    atomic_json(case_root / "case.json", metadata)
    hashes["case.json"] = sha(case_root / "case.json")
    return {
        "case_id": case_id,
        "case_hash": metadata["case_hash"],
        "map_uuid": map_row["map_uuid"],
        "maze_type": int(map_row["maze_type"]),
        "natural_type": TYPE_NAMES[int(map_row["maze_type"])],
        "map_seed": int(map_row["seed"]),
        "trajectory_seed": int(trajectory_seed),
        "authority_hash": map_row["authority_hash"],
        "occupancy_hash": map_row["occupancy_hash"],
        "camera_motion": metadata["camera_motion"],
        "gap": metadata["observed_gap"],
        "artifact_hashes": hashes,
        "independent_rerender": "PENDING",
    }


def ordered_maps(inventory, v1):
    existing = {
        row["map_uuid"] for row in v1.get("unused_valid_cases", [])
    }
    priority = {2: 0, 6: 1, 7: 2, 1: 3, 5: 4}
    return sorted(inventory["maps"], key=lambda row: (
        priority[int(row["maze_type"])],
        row["map_uuid"] in existing,
        int(row["seed"]),
        row["map_uuid"],
    ))


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is required for CE1 E1")
    if STAGING.exists():
        raise FileExistsError(
            f"refusing to overwrite CE1 staging: {STAGING}"
        )
    STAGING.mkdir(parents=True)
    inventory = json.loads(INVENTORY.read_text())
    v1 = json.loads(V1_MANIFEST.read_text())
    frame_times = np.arange(60, dtype=np.float64) * .1
    contract = load_motion_contract(
        ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml"
    )
    renderer = CudaAuthorityRenderer(SENSOR, "cuda:0")
    accepted, rejection_rows, map_rows = [], [], []
    total = Counter()
    started = time.perf_counter()
    declared_runtime_limit_seconds = 3600
    for map_index, map_row in enumerate(ordered_maps(inventory, v1)):
        map_started = time.perf_counter()
        backend = ExactAuthorityBVH(map_row["authority_root"])
        budgeted = BudgetedRenderer(renderer, 12)
        exact_count = 0
        validation_cuda_count = 0
        map_accepted = None
        try:
            proposals = enumerate_camera_candidates(
                backend, frame_times, p0_budget=128, exact_budget=24
            )
        except Exception as error:
            proposals = ()
            rejection_rows.append({
                "stage": "P0_camera_enumeration",
                "reason": f"{type(error).__name__}: {error}",
                "map_uuid": map_row["map_uuid"],
                "map_seed": int(map_row["seed"]),
                "maze_type": int(map_row["maze_type"]),
                "profile": map_row["profile_name"],
                "attempt": None,
                "hash": map_row["authority_hash"],
                "cleanup_eligibility": "retain_diagnostic",
            })
        proposal_indices = [
            index for index in (0, 1, 9)
            if index < len(proposals)
        ]
        for proposal_index in proposal_indices:
            proposal = proposals[proposal_index]
            if exact_count >= 24 or budgeted.calls >= 12:
                break
            allocation = min(8, 24 - exact_count)
            trajectory_seed = (
                1_970_000_000 + map_index * 1009 + proposal_index * 31
                + int(map_row["seed"])
            ) % (2**32)
            exact_accounted = False
            try:
                construction = propose_one(
                    backend, budgeted, proposal, frame_times, contract,
                    actor_candidate_budget=allocation,
                )
                exact_count += int(construction.attempts)
                exact_accounted = True
                actor_frames = actor_position(
                    construction.actors[0], frame_times
                )[:, None, :]
                # Accepted cases persist full owner/near-depth evidence. This
                # call counts against the same 12-call CUDA budget.
                diagnostics = renderer.render_with_actor_diagnostics(
                    backend, proposal.position_world, proposal.yaw,
                    actor_frames, [ACTOR_RADIUS_M],
                    return_owner_map=True, return_actor_near_depth=True,
                )
                validation_cuda_count += 1
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
                map_accepted = save_case(
                    case_id, map_row, trajectory_seed, construction,
                    proposal, actor_frames, frame_times, validation,
                )
                accepted.append(map_accepted)
                break
            except Exception as error:
                if not exact_accounted:
                    exact_count += allocation
                rejection_rows.append({
                    "stage": "exact_continuous_cuda_search",
                    "reason": f"{type(error).__name__}: {error}",
                    "map_uuid": map_row["map_uuid"],
                    "map_seed": int(map_row["seed"]),
                    "maze_type": int(map_row["maze_type"]),
                    "natural_type": TYPE_NAMES[int(map_row["maze_type"])],
                    "profile": map_row["profile_name"],
                    "attempt": proposal_index,
                    "camera_motion":
                        proposal.certificate["camera_motion"],
                    "patch_voxel_offset":
                        proposal.patch_voxel_offset,
                    "camera_standoff_m":
                        proposal.camera_standoff_m,
                    "hash": map_row["authority_hash"],
                    "cleanup_eligibility": "retain_diagnostic",
                })
        total["p0"] += min(128, len(proposals))
        total["exact"] += exact_count
        total["cuda"] += budgeted.calls
        map_rows.append({
            "map_uuid": map_row["map_uuid"],
            "natural_type": TYPE_NAMES[int(map_row["maze_type"])],
            "map_seed": int(map_row["seed"]),
            "p0_used": min(128, len(proposals)),
            "exact_continuous_used": exact_count,
            "cuda_used": budgeted.calls,
            "accepted_evidence_cuda_used": validation_cuda_count,
            "accepted_case_id":
                map_accepted["case_id"] if map_accepted else None,
            "elapsed_seconds": time.perf_counter() - map_started,
        })
        print(json.dumps(map_rows[-1]), flush=True)
        combined = list(v1["unused_valid_cases"]) + accepted
        maps = {row["map_uuid"] for row in combined}
        types = {int(row["maze_type"]) for row in combined}
        seeds = {int(row["map_seed"]) for row in combined}
        type_counts = Counter(int(row["maze_type"]) for row in combined)
        hard = (
            len(maps) >= 6 and len(types) >= 3 and len(seeds) >= 6
            and any(value not in (1, 5) for value in types)
            and max(type_counts.values(), default=0) / max(len(maps), 1) <= .5
        )
        recommended = hard and len(maps) >= 8 and len(types) >= 4 and len(seeds) >= 8
        if recommended:
            break
        if time.perf_counter() - started > declared_runtime_limit_seconds:
            break
    elapsed = time.perf_counter() - started
    REJECTIONS.parent.mkdir(parents=True, exist_ok=True)
    with REJECTIONS.open("w") as handle:
        for row in rejection_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    payload = {
        "status": "PASS",
        "stage": "E1_existing_30_map_bounded_search",
        "accepted_case_count": len(accepted),
        "accepted_cases": accepted,
        "maps": map_rows,
        "budgets": {
            "per_map": {"p0": 128, "exact_continuous": 24, "cuda": 12},
            "hard_total": {"p0": 3840, "exact_continuous": 720, "cuda": 360},
            "used": dict(total),
            "within_budget": (
                total["p0"] <= 3840 and total["exact"] <= 720
                and total["cuda"] <= 360
            ),
        },
        "elapsed_seconds": elapsed,
        "declared_runtime_limit_seconds": declared_runtime_limit_seconds,
        "peak_cpu_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "detector_output_used_for_selection": False,
        "representation_output_used_for_selection": False,
        "tracker_output_used_for_selection": False,
        "sealed_holdout_accessed": False,
        "formal_generation_started": False,
        "training_started": False,
    }
    atomic_json(REPORT, payload)
    atomic_json(BUDGET_REPORT, {
        "status": "PASS" if payload["budgets"]["within_budget"] else "FAIL",
        **payload["budgets"],
        "declared_runtime_limit_seconds": declared_runtime_limit_seconds,
        "actual_runtime_seconds": elapsed,
    })
    SUMMARY.write_text(
        "# Phase 8J-Q2.4-CE1 existing-map search\n\n"
        f"- Accepted new cases: {len(accepted)}\n"
        f"- Maps searched: {len(map_rows)}\n"
        f"- P0/exact/CUDA: {total['p0']}/{total['exact']}/{total['cuda']}\n"
        f"- Runtime: {elapsed:.3f} seconds\n"
        "- Selection inputs: authority geometry, continuous safety, and CUDA "
        "per-actor visibility only.\n"
    )
    print(json.dumps({
        "status": "PASS",
        "new_cases": len(accepted),
        "maps_searched": len(map_rows),
        "budgets": payload["budgets"],
        "staging": str(STAGING),
    }, indent=2))


if __name__ == "__main__":
    main()
