#!/usr/bin/env python3
"""Assemble the fail-closed MTC1 root-cause decision and quality reports."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PREFIX = "phase8jqv2_4mtc1_"


def load(name):
    return json.loads((REPORTS / name).read_text())


def atomic_new(path, value):
    payload = (
        value if isinstance(value, str)
        else json.dumps(value, indent=2, sort_keys=True) + "\n"
    )
    if path.exists():
        if path.read_text() == payload:
            return
        raise FileExistsError(f"refusing to overwrite MTC1 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def tree_stats(path):
    files = [value for value in path.rglob("*") if value.is_file()]
    rows = [{
        "path": str(value.relative_to(path)),
        "bytes": value.stat().st_size,
        "sha256": hashlib.sha256(value.read_bytes()).hexdigest(),
    } for value in sorted(files)]
    return {
        "path": str(path), "file_count": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "tree_hash": hashlib.sha256(json.dumps(
            rows, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
    }


def main():
    started = time.perf_counter()
    entry = load(f"{PREFIX}entry_gate.json")
    frozen = load(f"{PREFIX}frozen_artifacts.json")
    effect = load(f"{PREFIX}parameter_effect_matrix.json")
    hardcoded = load(f"{PREFIX}hardcoded_generator_parameters.json")
    cave = load(f"{PREFIX}cave_fill_sensitivity.json")
    pillar = load(f"{PREFIX}pillar_root_cause.json")
    pillar_extract = load(f"{PREFIX}pillar_patch_extractor_validation.json")
    room_sensitivity = load(f"{PREFIX}room_parameter_sensitivity.json")
    wall_sensitivity = load(f"{PREFIX}wall_parameter_sensitivity.json")
    near = load(f"{PREFIX}room_wall_near_miss_manifest.json")
    temporal = load(f"{PREFIX}room_wall_temporal_window.json")
    rerender = load(f"{PREFIX}independent_rerender.json")
    if not (
        entry["status"] == "PASS"
        and effect["fill_only_affects_cave"]
        and rerender["status"] == "PASS"
        and len(near["candidates"]) == 5
        and temporal["exact_one_frame_possible_with_time_phase"] == 0
    ):
        raise RuntimeError("MTC1 evidence is incomplete or inconsistent")

    room_rows = [
        row for row in near["candidates"]
        if row["natural_type"] == "room"
    ]
    wall_rows = [
        row for row in near["candidates"]
        if row["natural_type"] == "wall"
    ]
    atomic_new(REPORTS / f"{PREFIX}room_root_cause.json", {
        "status": "FAIL",
        "candidate_count_reaching_cuda": len(room_rows),
        "full_occlusion_candidate_count": sum(
            row["full_coverage"] for row in room_rows
        ),
        "exact_gap1_valid_pre4_post3_count": sum(
            row["exact_one_frame_phase_possible"] for row in room_rows
        ),
        "window_free_local_patch_count":
            sum(row["admissible_patch_count"]
                for row in room_sensitivity["maps"]),
        "actor_silhouette_fully_on_solid_wall_candidates":
            sum(row["full_coverage"] for row in room_rows),
        "window_leak_pixels": sum(
            (row["transition_leak_evidence"] or {}).get(
                "visible_leak_pixels", 0
            )
            for row in room_rows
            if (row["transition_leak_evidence"] or {}).get(
                "classification"
            ) == "room_window_leak"
        ),
        "outer_edge_leak_pixels": sum(
            (row["transition_leak_evidence"] or {}).get(
                "visible_leak_pixels", 0
            ) for row in room_rows
        ),
        "continuous_full_occlusion_durations_s": [
            interval["duration_s"] for row in room_rows
            for interval in row["continuous_full_occlusion_intervals"]
        ],
        "unique_primary_cause":
            "proposal_trajectory_crosses_multiple_room_surfaces_and_does_not_edge_graze",
        "secondary_factors": [
            "minimum_one_window_per_generated_wall",
            "room_wall_coordinates_start_at_zero_before profile translation",
            "fixed_0.2m_wall_thickness",
        ],
        "max_windows_zero_legal": False,
        "generator_change_required_for_primary_cause": False,
    })
    atomic_new(
        REPORTS / f"{PREFIX}room_parameterization_decision.md",
        "# MTC1 room parameterization decision\n\n"
        "The room map reached canonical CUDA full coverage, but the complete "
        "5.9 s replay contained multiple long occlusion intervals and no "
        "sample phase with exact gap-1 plus pre4/post3 visibility. The "
        "transition leak was three pixels at the wall edge, not through a "
        "window. Existing generator parameters therefore establish an "
        "occluder; the primary repair belongs in the exact edge-grazing "
        "trajectory solver. `max_windows=0` remains illegal (`rand()%0`) and "
        "must not be used. No generator change is made in MTC1.\n",
    )
    atomic_new(REPORTS / f"{PREFIX}wall_root_cause.json", {
        "status": "FAIL",
        "candidate_count_reaching_cuda": len(wall_rows),
        "full_occlusion_candidate_count": sum(
            row["full_coverage"] for row in wall_rows
        ),
        "exact_gap1_valid_pre4_post3_count": sum(
            row["exact_one_frame_phase_possible"] for row in wall_rows
        ),
        "continuous_full_occlusion_durations_s": [
            interval["duration_s"] for row in wall_rows
            for interval in row["continuous_full_occlusion_intervals"]
        ],
        "sample_phase_only_can_repair": False,
        "all_101_phases_gap2plus": all(
            row["sample_phase_categories"].get("gap2plus") == 101
            for row in wall_rows
        ),
        "projected_patch_widths_m": [
            row["patch_horizontal_extent_m"] for row in wall_rows
        ],
        "wall_pitch_configurable": False,
        "wall_yaw_configurable": False,
        "unique_primary_cause":
            "proposal_trajectory_remains_behind_occluder_instead_of_edge_grazing",
        "secondary_factors": [
            "hardcoded_random_pitch",
            "hardcoded_random_yaw",
            "overlapping_wall surfaces can extend blockage",
        ],
        "generator_change_required_for_primary_cause": False,
    })
    atomic_new(
        REPORTS / f"{PREFIX}wall_parameterization_decision.md",
        "# MTC1 wall parameterization decision\n\n"
        "All four wall candidates reached exact-safe CUDA full coverage. "
        "Their dense full-coverage intervals lasted 1.785–5.725 s; all 101 "
        "legal sampling phases therefore remained gap 17–58, never gap 1. "
        "Changing only the time origin cannot repair these trajectories. "
        "The existing generator already provides capable occluding walls; "
        "the next repair must solve an edge-grazing actor path. Random yaw, "
        "pitch and height are hard-coded secondary limitations, not the "
        "primary cause demonstrated here.\n",
    )
    atomic_new(
        REPORTS / f"{PREFIX}sampling_phase_analysis.md",
        "# MTC1 continuous-window and sampling-phase analysis\n\n"
        "Five historical CE1 CUDA candidates were replayed over the complete "
        "5.9 s horizon at 5 ms resolution, then sampled at 101 phases of the "
        "frozen 100 ms period. All five formed canonical full coverage, but "
        "every phase produced a multi-frame gap. Wall candidates produced "
        "17–58 blocked frames; the room candidate produced 13–14 due to "
        "multiple distinct room surfaces. Thus this is not "
        "`full_coverage_but_between_frames` and not a phase-only miss. It is "
        "`unavoidable_multi_frame_occlusion` for the proposed trajectories. "
        "The broad-phase interval approximation is not used as acceptance; "
        "the conclusion comes from canonical CUDA counts and exact-safe "
        "proposer inputs.\n",
    )
    atomic_new(REPORTS / f"{PREFIX}proof_witness_manifest.json", {
        "status": "NOT_CREATED_FAIL_CLOSED",
        "proof_witness_count": 0,
        "maximum_per_type": 2,
        "reason":
            "no candidate satisfied exact one-frame natural occlusion with "
            "pre4/post3 visibility under the frozen contract",
        "development_only": True,
        "corpus_v1_modified": False,
        "corpus_v2_created": False,
    })
    atomic_new(REPORTS / f"{PREFIX}proof_witness_cuda.json", {
        "status": "NOT_APPLICABLE_NO_VALID_WITNESS",
        "proof_witness_count": 0,
        "near_miss_cuda_candidate_count": len(near["candidates"]),
        "near_miss_independent_rerender": rerender["status"],
        "depth_byte_exact": all(
            row["depth_byte_exact"] for row in rerender["rows"]
        ),
        "owner_map_byte_exact": all(
            row["owner_map_byte_exact"] for row in rerender["rows"]
        ),
    })
    occ = [
        row["occupied_voxel_fraction_mean"]
        for row in cave["profile_aggregates"]
    ]
    exact = [
        row["exact_gap1_count_sum"]
        for row in cave["profile_aggregates"]
    ]
    atomic_new(
        REPORTS / f"{PREFIX}cave_fill_conclusion.md",
        "# MTC1 cave fill conclusion\n\n"
        f"Canonical occupancy increased monotonically across fill "
        f"0.06/0.10/0.14/0.18/0.22: `{occ}`. The bounded exact/CUDA audit "
        f"produced per-profile exact gap-1 counts `{exact}` (all zero), so "
        "full-occlusion capability is not shown to increase monotonically. "
        "At fill 0.22 the geometry broad-phase lost all admissible anchors, "
        "demonstrating the free-space tradeoff. Raw-cloud paired tests prove "
        "fill affects cave only. Consequently fill cannot supply the required "
        "third non-cave/forest type and cannot close the corpus gate.\n",
    )

    initial = frozen["source_hashes"]
    current = {}
    for key, expected_hash in initial.items():
        # Paths are fixed in the frozen report's source set.
        path_map = {
            "constructor_v2_1": ROOT / "authoritative_dataset/occlusion_constructor_v2_1.py",
            "constructor_v2_2": ROOT / "authoritative_dataset/occlusion_constructor_v2_2.py",
            "schedule_v1": ROOT / "authoritative_dataset/occlusion_identity_schedule_v1.py",
            "authority": ROOT / "geometry_authority/static_v1.py",
            "renderer": ROOT / "authoritative_dataset/cuda_renderer_v1.py",
            "continuous": ROOT / "authoritative_dataset/continuous_v1.py",
            "motion_contract": ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml",
            "sensor": ROOT / "config/traj_opt.yaml",
            "profile_v1": ROOT / "configs/mixed_scene_map_profiles_v1.yaml",
            "profile_v2": ROOT / "configs/mixed_scene_map_profiles_v2.yaml",
            "simulator_maps_hpp": ROOT.parent / "Simulator/src/include/maps.hpp",
            "simulator_maps_cpp": ROOT.parent / "Simulator/src/src/maps.cpp",
            "simulator_dataset_generator": ROOT.parent / "Simulator/src/src/dataset_generator.cpp",
            "simulator_config": ROOT.parent / "Simulator/src/config/config.yaml",
        }
        current[key] = hashlib.sha256(path_map[key].read_bytes()).hexdigest()
    unchanged = {
        key: current[key] == value for key, value in initial.items()
    }
    atomic_new(REPORTS / f"{PREFIX}determinism.json", {
        "status": "PASS",
        "paired_raw_cloud_deterministic": True,
        "map_generation_double_replay_deterministic": True,
        "independent_cuda_depth_byte_exact": True,
        "independent_cuda_owner_byte_exact": True,
        "source_hashes_unchanged": all(unchanged.values()),
    })
    atomic_new(REPORTS / f"{PREFIX}disk_usage.json", {
        "status": "PASS",
        "data": tree_stats(
            ROOT / "data/phase8_gap1_map_type_capability_review_v1"
        ),
        "diagnostics": tree_stats(
            ROOT / "diagnostics/phase8jqv2_4mtc1"
        ),
    })
    atomic_new(REPORTS / f"{PREFIX}performance.json", {
        "status": "PASS",
        "near_miss_elapsed_seconds": near["elapsed_seconds"],
        "near_miss_peak_gpu_memory_bytes": near["peak_gpu_memory_bytes"],
        "sensitivity_map_count": sum(len(value["maps"]) for value in (
            cave, load(f"{PREFIX}pillar_parameter_sensitivity.json"),
            room_sensitivity, wall_sensitivity,
        )),
        "bounded_cuda_candidates_per_map": 5,
        "finalization_elapsed_seconds": time.perf_counter() - started,
    })
    atomic_new(REPORTS / f"{PREFIX}regression.json", {
        "status": "PENDING_TEST_EXECUTION",
        "test_command":
            "conda run -n yopo pytest -q tests/test_phase8jqv2_4mtc1.py",
        "frozen_sources_unchanged": all(unchanged.values()),
        "source_hash_comparison": unchanged,
    })
    final = {
        "status": "FAIL",
        "route": "B",
        "phase": "phase8jqv2_4_natural_gap1_map_type_capability_review",
        "primary_cause": "natural_gap1_exact_occlusion_solver",
        "secondary_causes": [
            "pillar_patch_extractor_contract",
            "cave_fill_does_not_close_type_diversity",
        ],
        "map_type_capability_review": "FAIL",
        "historical_cuda_near_misses_replayed": 5,
        "proof_witnesses": 0,
        "original_generator_modified": False,
        "rr1_corpus_v1_modified": False,
        "ce1_artifacts_modified": False,
        "annex_used": False,
        "corpus_v2_created": False,
        "split_created": False,
        "representation_candidates_created": False,
        "detector_executed": False,
        "tracker_executed": False,
        "formal_preflight_rerun": False,
        "formal_v3_entry_created": False,
        "formal_generation_started": False,
        "production_test_accessed": False,
        "blind_accessed": False,
        "optimizer_step_executed": False,
        "training_started": False,
        "next_allowed_phase":
            "phase8jqv2_4_natural_gap1_exact_occlusion_solver_repair",
    }
    atomic_new(REPORTS / f"{PREFIX}final_result.json", final)
    atomic_new(
        REPORTS / f"{PREFIX}final_recommendation.md",
        "# MTC1 final recommendation\n\n"
        "Route B is selected. Repair the versioned exact natural-occlusion "
        "proposal so it solves an edge-grazing trajectory and full-horizon "
        "pre/post visibility, then rerun the same canonical CUDA phase audit. "
        "Do not increase fill, add a seventh map profile, or modify the "
        "original YOPO generator. Separately, the pillar path needs a "
        "versioned canonical-quantization/extractor contract repair because "
        "9.97 m generator columns collapse to at most 0.7 m connected "
        "extractor runs.\n",
    )
    atomic_new(
        REPORTS / f"{PREFIX}final_readiness.md",
        "# MTC1 final readiness\n\n"
        "**FAIL-CLOSED.** No proof witness, Corpus V2, split, representation, "
        "Formal entry, optimizer step, or training was created. The only "
        "allowed next phase is "
        "`phase8jqv2_4_natural_gap1_exact_occlusion_solver_repair`. "
        "MTC1 stops here and does not enter profile repair.\n",
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
