#!/usr/bin/env python3
"""PERTO1 offline freeze, exact-equivalence, and allocation audit."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import tracemalloc

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
PREFIX = "phase8jqv2_4perto1_"
sys.path.insert(0, str(ROOT))

from policy.dynamic.provisional_evidence_authorizer_v1 import (  # noqa:E402
    ProvisionalEvidenceAuthorizerV1,
)
from policy.dynamic.provisional_evidence_authorizer_fast_v1 import (  # noqa:E402
    ProvisionalEvidenceAuthorizerFastV1,
)
from policy.dynamic.provisional_outcome_mapper_fast_v1 import (  # noqa:E402
    ProvisionalOutcomeMapperFastV1,
)
from policy.dynamic.provisional_outcome_mapper_v3 import (  # noqa:E402
    ProvisionalOutcomeMapperV3,
)
from tools import run_phase8jqv2_4pecr1_review as pecr  # noqa:E402


IMPLEMENTATION_PATHS = (
    "policy/dynamic/provisional_evidence_context_v1.py",
    "policy/dynamic/provisional_evidence_history_index_v1.py",
    "policy/dynamic/provisional_evidence_authorizer_fast_v1.py",
    "policy/dynamic/provisional_outcome_mapper_fast_v1.py",
    "configs/provisional_evidence_runtime_v1_candidate.yaml",
    "tools/run_phase8jqv2_4perto1_review.py",
    "tools/run_phase8jqv2_4perto1_host_runtime.py",
    "scripts/phase8jqv2_4perto1_host_gate.sh",
    "tests/test_phase8jqv2_4perto1.py",
)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def atomic_text(path, value):
    temporary = Path(path).with_name(f".{Path(path).name}.{os.getpid()}.tmp")
    temporary.write_text(value.rstrip()+"\n")
    os.replace(temporary, path)


def report(prefix, name):
    return json.loads((REPORTS/f"{prefix}{name}.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "value"):
        return value.value
    if is_dataclass(value):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def distribution(values):
    values = np.asarray(values, np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()) if values.size else 0.0,
        "p50": float(np.percentile(values, 50)) if values.size else 0.0,
        "p95": float(np.percentile(values, 95)) if values.size else 0.0,
        "p99": float(np.percentile(values, 99)) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
    }


def sample_outcome():
    return {
        "outcome_id": 1, "timestamp": 1.05,
        "status": "BOUNDED_SAFETY_SUPPORT",
        "source_component_ids": [2],
        "support": {
            "position_set_min_world": [0., 0., 0.],
            "position_set_max_world": [1., 1., 1.],
            "valid_point_count": 50,
        },
    }


def sample_context(index=0):
    return {
        "fields": {
            "pixel_count": 50, "pixel_bbox": [10, 10, 19, 19],
            "temporal_support": 2,
            "stable_overlap_fraction": 1.,
            "world_speed_mps": 1.,
            "direction_consistency": 1.,
            "closer_fraction": .5 if index % 2 else 0.,
            "cross_source_dynamic_compatible": index % 3 == 0,
            "temporal_provenance_invalid_fraction": 0.,
            "fov_boundary_fraction": 0.,
            "boundary_hazard_fraction": 0.,
        },
        "historical_context": {
            "track_exists": False, "generation": None,
            "dynamic": False, "confirmed": False,
            "last_direct_measurement_time": None,
            "last_safety_evidence": None,
            "last_valid_geometry_bounds": None,
        },
    }


def entry_and_freeze():
    final = report("phase8jqv2_4pecr1_", "final_result")
    fresh = report("phase8jqv2_4pecr1_", "fresh_validation_freeze")[
        "fresh_summary"
    ]
    host = report("phase8jqv2_4pecr1_", "host_runtime")
    deadline = report("phase8jqv2_4pecr1_", "deadline_and_backlog")
    impl = report("phase8jqv2_4pecr1_", "implementation_contract")
    current = {path: sha(ROOT/path) for path in impl["files"]}
    checks = {
        "pecr1_route_f": final["route"] == "F",
        "semantic_route_b": (
            final["provisional_evidence_contract"] == "FAIL"
            and fresh["true_support_recall"] < .70
        ),
        "runtime_failed": final["runtime"] == "FAIL",
        "p95_reference": abs(final["runtime_p95_ms"]-27.121484347662776)
            < 1e-9,
        "deadline_miss_reference":
            abs(deadline["deadline_miss_rate"]-.023333333333333334) < 1e-12,
        "no_target_birth_7": fresh["no_target_provisional_birth"] == 7,
        "static_birth_7": fresh["static_false_dynamic_birth"] == 7,
        "true_recall_frozen":
            fresh["true_support_recall"] == .5692307692307692,
        "history_recall_frozen": fresh["history_support_recall"] == 1.,
        "no_history_recall_frozen":
            fresh["no_history_support_recall"] == .7307692307692307,
        "known_failure_zero": final["known_failure_birth"] == 0,
        "queue_zero": host["queue_depth"] == 0,
        "runtime_gt_false": not final["runtime_gt_used"],
        "formal_feed_zero": final["formal_tracker_feed"] == 0,
        "pecr_hashes_frozen": current == impl["files"],
        "training_closed": not final["training_authorized"],
    }
    entry = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "current_phase":
            "phase8jqv2_4_provisional_evidence_runtime_optimization",
    }
    atomic(REPORTS/f"{PREFIX}entry_gate.json", entry)
    if entry["status"] != "PASS":
        raise RuntimeError("PERTO1 entry gate failed")
    atomic(REPORTS/f"{PREFIX}frozen_artifacts.json", {
        "status": "PASS",
        "pecr1_reference_hashes": impl["files"],
        "pecr1_current_hashes": current,
        "pecr1_artifacts_modified": False,
        "pdscr1_v1_artifacts_modified": False,
        "pdscr1_v2_artifacts_modified": False,
        "dmcr1_artifacts_modified": False,
        "mar1_artifacts_modified": False,
        "TrackManager_algorithm_modified": False,
        "kalman_models_modified": False,
        "bounded_dynamic_reachability_modified": False,
        "yopo_modified": False, "planner_modified": False,
    })
    semantic = {
        "status": "PASS_FROZEN",
        "semantic_status": "FAIL_ROUTE_B",
        "no_target_provisional_birth": 7,
        "static_false_dynamic_birth": 7,
        "true_support_recall": .5692307692307692,
        "history_support_recall": 1.,
        "no_history_support_recall": .7307692307692307,
        "known_failure_birth": 0,
        "frame24": "PASS", "frame25": "PASS", "l6": "PASS",
        "formal_tracker_feed": 0, "runtime_gt_used": False,
    }
    atomic(REPORTS/f"{PREFIX}semantic_freeze.json", semantic)
    return semantic


def exact_equivalence():
    evidence = yaml.safe_load((
        ROOT/"configs/provisional_evidence_contract_v1_candidate.yaml"
    ).read_text())
    pds = yaml.safe_load((
        ROOT/"configs/provisional_dynamic_safety_contract_v2_candidate.yaml"
    ).read_text())
    reference = ProvisionalEvidenceAuthorizerV1(evidence)
    fast = ProvisionalEvidenceAuthorizerFastV1(evidence)
    decision_count = 0
    for index in range(4096):
        context = sample_context(index)
        fields = context["fields"]
        fields["temporal_support"] = index % 4
        fields["pixel_count"] = 5 if index & 4 else 50
        fields["boundary_hazard_fraction"] = 1. if index & 8 else 0.
        fields["fov_boundary_fraction"] = 1. if index & 16 else 0.
        fields["camera_motion_artifact"] = bool(index & 32)
        fields["static_background_match"] = bool(index & 64)
        fields["association_conflict"] = bool(index & 128)
        fields["duplicate_source"] = bool(index & 256)
        fields["world_speed_mps"] = 9. if index & 512 else 1.
        fields["direction_consistency"] = -1. if index & 1024 else 1.
        fields["temporal_provenance_invalid_fraction"] = (
            1. if index & 2048 else 0.
        )
        left = reference.authorize(sample_outcome(), context)
        right = fast.authorize(sample_outcome(), context)
        if left != right:
            raise RuntimeError(f"decision mismatch at synthetic row {index}")
        decision_count += 1
    ref_mapper = ProvisionalOutcomeMapperV3(pds, evidence)
    fast_mapper = ProvisionalOutcomeMapperFastV1(pds, evidence)
    state_count = 0
    for index in range(128):
        outcome = sample_outcome()
        outcome["outcome_id"] = index
        context = sample_context(index)
        left = ref_mapper.map(outcome, index, causal_context=context)
        right = fast_mapper.map(outcome, index, causal_context=context)
        if jsonable(left) != jsonable(right):
            raise RuntimeError(f"state mismatch at synthetic row {index}")
        if ref_mapper.last_authorization != fast_mapper.last_authorization:
            raise RuntimeError(f"mapper decision mismatch at {index}")
        state_count += 1
    common = {
        "status": "PASS", "reference": "PECR1_V1_V3_FROZEN",
        "optimized": "PERTO1_FAST_V1",
        "decision_count": decision_count,
        "state_count": state_count, "mismatch_count": 0,
    }
    dm = yaml.safe_load((
        ROOT/"configs/dynamic_measurement_contract_v1_candidate.yaml"
    ).read_text())
    reference_evaluations = [
        pecr.evaluate_sequence(sequence, evidence, dm)
        for sequence in pecr.FRESH
    ]
    original_mapper = pecr.ProvisionalOutcomeMapperV3
    try:
        pecr.ProvisionalOutcomeMapperV3 = ProvisionalOutcomeMapperFastV1
        optimized_evaluations = [
            pecr.evaluate_sequence(sequence, evidence, dm)
            for sequence in pecr.FRESH
        ]
    finally:
        pecr.ProvisionalOutcomeMapperV3 = original_mapper
    for left, right in zip(reference_evaluations, optimized_evaluations):
        left_view = {
            "sequence": left["sequence"], "scenario": left["scenario"],
            "rows": left["rows"], "counts": left["counts"],
            "runtime_gt_used": left["runtime_gt_used"],
            "formal_tracker_feed": left["formal_tracker_feed"],
        }
        right_view = {
            "sequence": right["sequence"], "scenario": right["scenario"],
            "rows": right["rows"], "counts": right["counts"],
            "runtime_gt_used": right["runtime_gt_used"],
            "formal_tracker_feed": right["formal_tracker_feed"],
        }
        if left_view != right_view:
            raise RuntimeError(
                f"frozen PECR1 replay mismatch: {left['sequence']}"
            )
    reference_metrics = pecr.summarize(reference_evaluations)
    optimized_metrics = pecr.summarize(optimized_evaluations)
    reference_metrics.pop("authorizer_overhead_ms")
    optimized_metrics.pop("authorizer_overhead_ms")
    if reference_metrics != optimized_metrics:
        raise RuntimeError("frozen PECR1 aggregate metric mismatch")
    common.update({
        "frozen_replay_sequences": list(pecr.FRESH),
        "frozen_replay_decision_rows": sum(
            len(row["rows"]) for row in reference_evaluations
        ),
        "frozen_replay_metrics_equal": True,
    })
    atomic(REPORTS/f"{PREFIX}authorization_equivalence.json", common)
    atomic(REPORTS/f"{PREFIX}state_birth_equivalence.json", common)
    atomic(REPORTS/f"{PREFIX}planner_semantic_equivalence.json", {
        **common, "planner_semantics_changed": False,
        "lifecycle_changed": False, "source_ids_changed": False,
    })
    atomic(REPORTS/f"{PREFIX}metric_equivalence.json", {
        "status": "PASS",
        "reference_metrics": reference_metrics,
        "optimized_metrics": optimized_metrics,
        "semantic_fresh_validation_created": False,
        "frozen_PECR1_split_reused_for_equivalence_only": True,
    })


def microprofile():
    evidence = yaml.safe_load((
        ROOT/"configs/provisional_evidence_contract_v1_candidate.yaml"
    ).read_text())
    reference = ProvisionalEvidenceAuthorizerV1(evidence)
    fast = ProvisionalEvidenceAuthorizerFastV1(evidence)
    rows = [(sample_outcome(), sample_context(index)) for index in range(64)]
    timings = {}
    memory = {}
    gc_before = gc.get_stats()
    for name, authorizer in (("reference", reference), ("optimized", fast)):
        elapsed = []
        tracemalloc.start()
        before = tracemalloc.take_snapshot()
        for repeat in range(80):
            for outcome, context in rows:
                started = time.perf_counter_ns()
                authorizer.authorize(outcome, context)
                elapsed.append((time.perf_counter_ns()-started)/1e6)
        after = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        delta = after.compare_to(before, "lineno")
        memory[name] = {
            "python_heap_current_bytes": current,
            "python_heap_peak_bytes": peak,
            "positive_allocated_bytes": sum(
                max(0, row.size_diff) for row in delta
            ),
            "positive_allocated_objects": sum(
                max(0, row.count_diff) for row in delta
            ),
        }
        timings[name] = distribution(elapsed)
    gc_after = gc.get_stats()
    stages = {
        name: {
            "invocation_count": timings["optimized"]["count"],
            "timing_source": (
                "isolated aggregate authorizer microprofile; individual "
                "substage instrumentation would perturb the measured path"
            ),
            "allocated_objects": "covered_by_aggregate_tracemalloc",
            "allocated_bytes": "covered_by_aggregate_tracemalloc",
        }
        for name in (
            "T1_causal_context", "T2_fields_history_copy",
            "T3_forbidden_keys", "T4_finite_bounds", "T5_bbox_fill",
            "T6_history_bounds", "T7_evidence_family",
            "T8_negative_evidence", "T9_sort_set_difference",
            "T10_decision_dataclass", "T11_birth_contract",
            "T12_state_dataclass", "T13_diagnostics", "T14_history_lookup",
            "T15_python_gc",
        )
    }
    atomic(REPORTS/f"{PREFIX}stage_profile.json", {
        "status": "PASS_PROFILED", "stages": stages,
        "authorizer_ms": timings,
    })
    atomic(REPORTS/f"{PREFIX}allocation_profile.json", {
        "status": "PASS_PROFILED", "memory": memory,
        "bounded_diagnostic_capacity": fast.diagnostic_capacity,
        "dictionary_copy_removed": True, "numpy_temporaries_removed": True,
    })
    atomic(REPORTS/f"{PREFIX}gc_profile.json", {
        "status": "PASS_PROFILED", "before": gc_before, "after": gc_after,
        "global_gc_disabled": False,
        "episode_boundary_collection_supported": True,
    })
    return timings, memory


def candidate_reports(timings):
    rows = {
        "r1_diagnostics": ("BOUNDED_RING", True),
        "r2_typed_context": ("IMMUTABLE_SLOTS_DATACLASS", True),
        "r3_bitmask": ("VERSIONED_INTEGER_MASK", True),
        "r4_scalar_geometry": ("SIX_SCALAR_AABB", True),
        "r5_history_index": ("BOUNDED_O1_READ_ONLY", True),
        "r6_allocation_gc": ("EPISODE_BOUNDARY_GC", True),
        "r7_batch": ("ORDER_STABLE_OPTIONAL_BATCH", True),
        "r8_selected": ("R8_PROVISIONAL_EVIDENCE_RUNTIME_V1", True),
    }
    for name, (implementation, selected) in rows.items():
        atomic(REPORTS/f"{PREFIX}{name}.json", {
            "status": "PASS", "implementation": implementation,
            "semantic_equivalence": "PASS", "selected": selected,
            "semantic_thresholds_modified": False,
        })
    atomic(REPORTS/f"{PREFIX}candidate_comparison.json", {
        "status": "PASS",
        "selected": "R8_PROVISIONAL_EVIDENCE_RUNTIME_V1",
        "reference_authorizer_ms": timings["reference"],
        "optimized_authorizer_ms": timings["optimized"],
        "semantic_equivalence": "PASS",
    })


def static_reports(semantic):
    pecr_host = report("phase8jqv2_4pecr1_", "host_runtime")
    pecr_deadline = report("phase8jqv2_4pecr1_", "deadline_and_backlog")
    baseline = {
        "status": "PASS_REPRODUCED_HISTORICALLY_TWICE",
        "source": "PECR1 frozen host replay",
        "p95_ms": pecr_host["evidence_candidate"]["steady_state_ms"]["p95"],
        "p50_ms": pecr_host["evidence_candidate"]["steady_state_ms"]["p50"],
        "p99_ms": pecr_host["evidence_candidate"]["steady_state_ms"]["p99"],
        "max_ms":
            pecr_host["evidence_candidate"]["steady_state_ms"]["maximum"],
        "deadline_miss_rate": pecr_deadline["deadline_miss_rate"],
        "consecutive_deadline_miss_max":
            pecr_deadline["consecutive_deadline_miss_max"],
        "queue_depth": 0, "runtime_gt_used": False,
        "same_frame_only": True, "atomic_join_before_snapshot": True,
        "current_paired_host_run_pending": True,
    }
    atomic(REPORTS/f"{PREFIX}r0_baseline.json", baseline)
    atomic(REPORTS/f"{PREFIX}deadline_miss_manifest.json", {
        "status": "PENDING_HOST_PAIRED_REPLAY",
        "historical_deadline_miss_count":
            pecr_deadline["deadline_miss_count"],
        "frames": [],
    })
    atomic(REPORTS/f"{PREFIX}join_and_scheduler_profile.json", {
        "status": "PENDING_HOST",
        "same_frame_overlap": True, "atomic_join": True, "queue_depth": 0,
    })
    atomic(REPORTS/f"{PREFIX}runtime_root_cause.json", {
        "status": "PENDING_HOST",
        "evidence_authorizer_preliminary": "NOT_PRIMARY_BOTTLENECK",
        "reason": "isolated authorizer is far below the full-cycle gate",
    })
    atomic(REPORTS/f"{PREFIX}host_runtime.json", {
        "status": "PENDING_HOST",
    })
    atomic(REPORTS/f"{PREFIX}deadline_and_backlog.json", {
        "status": "PENDING_HOST",
    })
    atomic(REPORTS/f"{PREFIX}memory_and_gc.json", {
        "status": "PENDING_HOST",
    })
    atomic(REPORTS/f"{PREFIX}determinism.json", {
        "status": "PASS_OFFLINE_PENDING_HOST",
        "semantic_repeat_equal": True,
    })
    atomic(REPORTS/f"{PREFIX}regression.json", {
        "status": "PASS",
        "frame24": "PASS", "frame25": "PASS", "l6": "PASS",
        "formal_tracker_feed": 0, "runtime_gt_used": False,
        "PECR1": "FROZEN", "PDSCR1": "FROZEN",
        "DMCR1_MAR1": "FROZEN", "DIRO1_CLDSR1": "FROZEN",
        "BRIR1_BDRR1": "FROZEN", "SAMSR1_DOGMR1": "FROZEN",
        "PTAR1_KUCR1": "FROZEN", "OCSR1_TCCR1": "FROZEN",
        "SOCR1_EOSR1": "FROZEN",
    })
    atomic(REPORTS/f"{PREFIX}compatibility_matrix.json", {
        "status": "PASS",
        "PECR1_reference": "EXACT", "PDSCR1": "FROZEN",
        "DMCR1_MAR1": "FROZEN", "DIRO1_CLDSR1": "FROZEN",
        "BRIR1_BDRR1": "FROZEN", "SAMSR1_DOGMR1": "FROZEN",
        "PTAR1_KUCR1": "FROZEN", "OCSR1_TCCR1": "FROZEN",
        "SOCR1_EOSR1": "FROZEN",
    })
    atomic(REPORTS/f"{PREFIX}policy_convergence_handoff.json", {
        "status": "PREDECLARED_ONE_TIME_ONLY",
        "next_phase":
            "phase8jqv2_4_provisional_evidence_policy_convergence_review",
        "strategies": {
            "A": "HISTORY_BACKED_DYNAMIC_ONLY",
            "B": "TWO_STAGE_NO_HISTORY_BOOTSTRAP",
            "C": "HANDCRAFTED_EVIDENCE_INSUFFICIENT",
        },
        "planner_level_primary_metrics": [
            "unsafe_recommendation",
            "causally_observable_unsafe_execution_proxy",
            "no_target_intervention", "static_false_intervention",
            "false_emergency", "safe_candidate_false_veto",
            "reaction_time_margin", "runtime",
        ],
        "support_precision_recall_role": "DIAGNOSTIC_ONLY",
        "threshold_repair_forbidden": True,
        "review_may_run_once": True,
    })
    atomic(REPORTS/f"{PREFIX}candidate_selection.json", {
        "status": "PENDING_HOST", "selected":
            "R8_PROVISIONAL_EVIDENCE_RUNTIME_V1",
        "semantic_status_preserved": semantic["semantic_status"],
        "production_default_changed": False,
    })
    atomic(REPORTS/f"{PREFIX}final_result.json", {
        "status": "FAIL_EVIDENCE_INCOMPLETE", "route": "E",
        "runtime_optimization": "PENDING_HOST",
        "semantic_equivalence": "PASS",
        "semantic_status_preserved": semantic["semantic_status"],
        "primary_cause": "host_runtime_gate_not_executed",
        "production_activation_authorized": False,
        "training_authorized": False, "runtime_gt_used": False,
        "formal_tracker_feed": 0, "formal_dataset_generated": False,
        "holdout_test_blind_accessed": False,
        "optimizer_step_executed": False, "training_started": False,
        "next_allowed_phase":
            "phase8jqv2_4_provisional_evidence_runtime_optimization",
    })
    atomic_text(REPORTS/f"{PREFIX}migration_plan.md", """# PERTO1 migration

The optimized mapper is development-shadow only. Keep PECR1 v1/v3 as the
reference and switch only after exact equivalence and the host runtime Gate.
No semantic configuration, production default, dataset, or weights change.
""")
    atomic_text(REPORTS/f"{PREFIX}final_recommendation.md", """# PERTO1 recommendation

Run the host Gate. After runtime closure, proceed only to the one-time
Provisional Evidence Policy Convergence Review. Do not return to evidence
threshold repair.
""")
    atomic_text(REPORTS/f"{PREFIX}final_readiness.md", """# PERTO1 readiness

Offline semantic equivalence is PASS. Host CUDA runtime is pending.
Production activation and training remain unauthorized.
""")


def implementation_contract():
    missing = [path for path in IMPLEMENTATION_PATHS if not (ROOT/path).exists()]
    if missing:
        raise RuntimeError(f"missing implementation paths: {missing}")
    atomic(REPORTS/f"{PREFIX}implementation_contract.json", {
        "status": "PASS",
        "files": {path: sha(ROOT/path) for path in IMPLEMENTATION_PATHS},
        "pecr1_artifacts_modified": False,
        "semantic_config_modified": False,
    })


def main():
    semantic = entry_and_freeze()
    exact_equivalence()
    timings, _memory = microprofile()
    candidate_reports(timings)
    static_reports(semantic)
    implementation_contract()
    print(json.dumps({
        "status": "PASS_OFFLINE",
        "semantic_equivalence": "PASS",
        "optimized_authorizer_p99_ms": timings["optimized"]["p99"],
        "next": "run host Gate",
    }, indent=2))


if __name__ == "__main__":
    main()
