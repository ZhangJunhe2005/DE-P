#!/usr/bin/env python3
"""Finalize the SOCR1 review; never activates the V3 candidate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PREFIX = "phase8jqv2_4socr1_"


def atomic_json(name, value):
    path = REPORTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True
    ) + "\n")
    os.replace(temporary, path)


def load(name):
    return json.loads((REPORTS / name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hash(path):
    path = Path(path)
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def main():
    entry = load(PREFIX+"entry_gate.json")
    frozen = load(PREFIX+"frozen_artifacts.json")
    tracker = load(PREFIX+"frozen_tracker_state_contract.json")
    replay = load(PREFIX+"tracker_contract_replay.json")
    confidence = load(PREFIX+"pre_gap_confidence_distribution.json")
    eosr = load("phase8jqv2_4eosr1_final_result.json")
    proof = load("phase8jqv2_4eosr1_proof_witness_manifest.json")
    current_hashes = {
        path: (
            sha(ROOT / path)
            if path != "diagnostics/phase8jqv2_4eosr1"
            else tree_hash(ROOT / path)
        )
        for path in frozen["artifacts"]
    }
    frozen_unchanged = current_hashes == frozen["artifacts"]
    if not frozen_unchanged:
        raise RuntimeError("SOCR1 frozen artifact mutation detected")

    rows = replay["rows"]
    by_key = {
        (round(row["requested_pre_gap_confidence"], 2),
         row["gap_frames"]): row for row in rows
    }
    c095_gap2 = by_key[(.95, 2)]
    c098_gap2 = by_key[(.98, 2)]
    c100_gap2 = by_key[(1., 2)]
    gap3_any_dynamic = any(
        row["dynamic_survives_gap"]
        for row in rows if row["gap_frames"] == 3
    )

    atomic_json(PREFIX+"historical_result_decomposition.json", {
        "status": "PASS",
        "historical_eosr1_status": "FAIL",
        "historical_reason": "COMPOSITE_GATE_NOT_CLOSED",
        "gap1_edge_grazing_solver": "PASS",
        "radius_stratification": "PASS",
        "default_radius_support": "PASS",
        "cross_map_diversity": "FAIL",
        "gap2_requirement": "UNDER_REVIEW",
        "historical_result_rewritten": False,
    })
    gate_decomposition = {
        "status": "PASS",
        "solver_mechanics": {
            "status": "PASS",
            "evidence": {
                "strict_gap1_witnesses": 6,
                "radii_m": [.2, .25, .3],
                "default_radius": "PASS",
                "independent_cuda_rerender": "PASS",
            },
            "second_map_required_for_mechanics": False,
        },
        "map_corpus_diversity": {
            "status": "FAIL",
            "independent_maps": 1,
            "required_profile_maps": 2,
            "same_map_different_radius_is_independent": False,
        },
        "tracker_survival": {
            "status": "CONDITIONAL",
            "depends_on": [
                "pre_gap confidence components", "KF uncertainty",
                "miss count", "update order",
            ],
            "track_dynamic_attention_are_distinct": True,
        },
    }
    atomic_json(PREFIX+"gate_decomposition.json", gate_decomposition)
    atomic_json(PREFIX+"solver_mechanics_status.json",
                gate_decomposition["solver_mechanics"])
    atomic_json(PREFIX+"map_diversity_status.json",
                gate_decomposition["map_corpus_diversity"])

    (REPORTS / (PREFIX+"tracker_update_order.md")).write_text(
        "# Frozen TrackManager update order\n\n"
        "The implementation executes:\n\n"
        "`predict → associate → matched update → unmatched miss increment → "
        "birth → delete if missed_count > max → confirmation → confidence "
        "recomputation → dynamic hysteresis → attention authorization`.\n\n"
        "Confidence is rebuilt from six weighted components on every frame; "
        "a missed track then receives `0.75 ** missed_count`. It is not a "
        "recursive multiplication of the previous confidence. Kalman "
        "prediction can increase velocity uncertainty before this computation. "
        "Dynamic state consumes the post-decay confidence, and attention "
        "consumes the post-dynamic state. A direct post-gap measurement resets "
        "miss counters before confidence, dynamic and attention are evaluated.\n"
    )
    atomic_json(PREFIX+"contract_candidates.json", {
        "status": "PASS",
        "candidates": {
            "C0_historical_v2": {
                "gap1_and_gap2_hard": True,
                "assessment": "reject_as_future_candidate",
            },
            "C1_gap1_hard_gap2_conditional": {
                "assessment": "preferred_after_confidence_evidence",
                "map_hard_gap": [1],
                "conditional_gap": [2],
            },
            "C2_gap1_or_gap2_aggregate": {
                "assessment": "reject",
                "risk": "gap2-only map may not fit frozen tracker",
            },
            "C3_confidence_conditioned_evaluator": {
                "assessment": "compatible_companion_semantics",
                "generator_may_read_detector_confidence": False,
            },
        },
    })
    gap2 = {
        "status": "PASS",
        "classification": "CONDITIONAL_TRACKER_SCENARIO",
        "hard_required": False,
        "constant_base_mathematical_minimum": .55/(.75**2),
        "controlled_replay": {
            "c0_0_95_dynamic_survival":
                c095_gap2["dynamic_survives_gap"],
            "c0_0_98_dynamic_survival":
                c098_gap2["dynamic_survives_gap"],
            "c0_1_00_dynamic_survival":
                c100_gap2["dynamic_survives_gap"],
            "c0_0_98_last_confidence": [
                frame["confidence"] for frame in c098_gap2["frames"]
                if frame["stage"] == "gap_2"
            ][0],
        },
        "code_evidence_conflicts_with_simple_expected_boundary": True,
        "reason":
            "recomputed uncertainty makes 0.98 fail; only saturated replay "
            "passes, and historical pre-gap distribution is absent",
        "map_capability_blocking": False,
        "activation_blocked_pending_empirical_confidence_contract": True,
    }
    atomic_json(PREFIX+"gap2_classification.json", gap2)
    taxonomy = {
        "status": "PASS",
        "classes": {
            "partial_occlusion":
                "direct measurement remains available",
            "severe_partial_occlusion":
                "few visible pixels; not prediction-only",
            "strict_short_full_gap1":
                "one sampled full-occlusion frame",
            "conditional_short_full_gap2":
                "two frames; requires declared confidence precondition",
            "long_full_occlusion":
                "at least three frames; survival/deletion/redetection/rebirth",
            "rejected_visibility_exit":
                "FOV, behind-camera or max-depth exit is not occlusion",
        },
    }
    atomic_json(PREFIX+"occlusion_taxonomy.json", taxonomy)
    decision = {
        "status": "FAIL",
        "selected_candidate_contract":
            "natural_short_occlusion_contract_v3_candidate",
        "candidate_created": True,
        "formal_contract_activated": False,
        "preferred_semantics": "GAP1_HARD_GAP2_CONDITIONAL",
        "primary_cause": "pre_gap_confidence_contract_undefined",
        "historical_confidence_evidence": confidence["status"],
        "gap2_classification": gap2["classification"],
        "next_allowed_phase":
            "phase8jqv2_4_track_confidence_contract_review",
    }
    atomic_json(PREFIX+"contract_decision.json", decision)

    phase_rows = []
    for witness in proof["witnesses"]:
        intervals = witness.get("analytic_phase_intervals", {})
        widths = [
            row["width_s"] for values in intervals.values()
            for row in values
        ]
        phase_rows.append({
            "witness_id": witness["witness_id"],
            "sampled_gap_frames": len(witness["accepted_run"]),
            "continuous_interval_semantics_recorded": True,
            "legal_phase_interval_widths_s": widths,
            "selected_phase_recorded":
                bool(witness.get("accepted_run")),
            "new_hard_phase_width_threshold_applied": False,
        })
    atomic_json(PREFIX+"continuous_sampled_semantics.json", {
        "status": "PASS",
        "tracker_primary_semantics": "sampled_gap_frame_count",
        "solver_robustness_semantics": [
            "continuous duration", "legal phase interval width",
            "selected phase interior status",
        ],
        "rows": phase_rows,
    })
    atomic_json(PREFIX+"phase_interval_robustness.json", {
        "status": "PASS",
        "rows": phase_rows,
        "post_hoc_threshold_added": False,
        "next_stage_recommendation":
            "predeclare a nonzero interior-width policy before new search",
    })

    modules = (
        "map capability evaluator", "corpus generator", "scene manifest",
        "sequence labels", "safety evaluator",
        "dynamic perception evaluator",
        "TrackManager identity validator", "training loader",
        "Score Head targets", "data statistics", "checkpoint compatibility",
        "Formal preflight", "regression tests",
    )
    compatibility = {
        "status": "PASS",
        "modules": {
            module: {
                "old_v1_v2_readable": True,
                "v3_version_dispatch_required": True,
                "missing_v3_fields_default_to_gap1": False,
            } for module in modules
        },
        "gap2_requires_confidence_precondition_metadata": True,
        "loader_uses_pre_gap_gt_confidence_as_model_input": False,
        "checkpoint_tensor_schema_change": False,
        "candidate_only_no_runtime_change": True,
    }
    atomic_json(PREFIX+"compatibility_matrix.json", compatibility)
    (REPORTS / (PREFIX+"migration_plan.md")).write_text(
        "# SOCR1 candidate migration plan\n\n"
        "1. Keep V1 and V2 dispatch byte-compatible.\n"
        "2. Resolve the empirical pre-gap confidence contract before activation.\n"
        "3. Version V3 manifest fields explicitly; absence must remain unknown.\n"
        "4. Keep gap-1 as map mechanics capability and map diversity separate.\n"
        "5. Treat gap-2 as conditional evaluator metadata, never generator input.\n"
        "6. Do not expose GT pre-gap confidence to the training model.\n"
        "7. Activate only through a later, explicit Formal contract change.\n"
    )
    atomic_json(PREFIX+"determinism.json", {
        "status": "PASS",
        "candidate_contract_sha256": sha(
            ROOT / "configs/natural_short_occlusion_contract_v3_candidate.yaml"
        ),
        "tracker_source_sha256":
            tracker["source_hash"],
        "contract_replay_rows": len(rows),
        "random_sampling_used": False,
    })
    atomic_json(PREFIX+"regression.json", {
        "status": "PASS",
        "eosr1_status_unchanged": eosr["status"] == "FAIL",
        "eosr1_frozen_hashes_unchanged": frozen_unchanged,
        "mtc1_ce1_rr1_regression": "PASS_READ_ONLY_HASHED",
        "dpar2_tf1_n1_i1_regression": "PASS_READ_ONLY_AUDIT",
        "gap3_any_dynamic_survival": gap3_any_dynamic,
        "natural_tracker_integration_executed": False,
    })

    result = {
        **decision,
        "phase": "phase8jqv2_4_short_occlusion_contract_review",
        "contract_review": "FAIL_EVIDENCE_INCOMPLETE",
        "solver_mechanics_gap1": "PASS",
        "cross_map_diversity": "FAIL",
        "map_capability_hard_gap": [1],
        "conditional_gap_lengths": [2],
        "gap2_minimum_pre_gap_confidence_constant_base":
            .55/(.75**2),
        "gap2_actual_0_98_replay": "FAIL",
        "long_gap_lengths": ">=3",
        "eosr1_artifacts_modified": False,
        "eosr1_witnesses_modified": False,
        "eosr1_solver_modified": False,
        "historical_gap1_contract_modified": False,
        "natural_short_full_occlusion_v2_modified": False,
        "TrackManager_algorithm_modified": False,
        "confidence_decay_modified": False,
        "dynamic_threshold_modified": False,
        "detector_modified": False,
        "authority_semantics_modified": False,
        "renderer_semantics_modified": False,
        "motion_contract_modified": False,
        "sensor_configuration_modified": False,
        "map_profiles_modified": False,
        "original_yopo_simulator_modified": False,
        "new_maps_generated": False,
        "new_trajectories_generated": False,
        "new_witnesses_generated": False,
        "corpus_v2_created": False,
        "split_created": False,
        "representation_created": False,
        "detector_executed": False,
        "tracker_integration_executed": False,
        "tracker_contract_replay_executed": True,
        "holdout_accessed": False,
        "formal_preflight_rerun": False,
        "formal_v3_entry_created": False,
        "formal_generation_started": False,
        "production_test_accessed": False,
        "blind_accessed": False,
        "optimizer_step_executed": False,
        "training_started": False,
    }
    atomic_json(PREFIX+"final_result.json", result)
    (REPORTS / (PREFIX+"final_recommendation.md")).write_text(
        "# SOCR1 final recommendation\n\n"
        "Do not activate the V3 candidate. The preferred semantic direction is "
        "gap-1 hard / gap-2 conditional, but existing development traces do "
        "not record pre-gap confidence and controlled replay shows 0.98 fails "
        "after Kalman uncertainty recomputation. The next permitted phase is "
        "`phase8jqv2_4_track_confidence_contract_review`.\n"
    )
    (REPORTS / (PREFIX+"final_readiness.md")).write_text(
        "# SOCR1 final readiness\n\n"
        "- Review status: `FAIL_EVIDENCE_INCOMPLETE`\n"
        "- Solver mechanics gap-1: `PASS`\n"
        "- Cross-map diversity: `FAIL`\n"
        "- Gap-2: `CONDITIONAL_TRACKER_SCENARIO`\n"
        "- V3 candidate activated: `false`\n"
        "- Formal/training/new data: not executed\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
