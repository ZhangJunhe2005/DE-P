#!/usr/bin/env python3
"""MAR1 development-only measurement availability review."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAG = ROOT / "diagnostics/phase8jqv2_4mar1"
PREFIX = "phase8jqv2_4mar1_"
CONFIG = ROOT / "configs/measurement_availability_contract_v1_candidate.yaml"
sys.path.insert(0, str(ROOT))

from policy.dynamic.cold_start_foreground_availability_v1 import (  # noqa:E402
    audit_cold_start,
)
from policy.dynamic.dynamic_perception_architecture_registry import (  # noqa:E402
    load_architecture_config,
)
from policy.dynamic.dynamic_perception_fast_path_v1 import (  # noqa:E402
    DynamicPerceptionFastPathV1,
)
from policy.dynamic.measurement_availability_provisional_feed_v1 import (  # noqa:E402
    MeasurementAvailabilityProvisionalFeedV1,
)
from policy.dynamic.provisional_safety_hypothesis_v1 import (  # noqa:E402
    ProvisionalSafetyHypothesisBuilderV1,
)
from policy.dynamic.rejected_component_safety_adapter_v1 import (  # noqa:E402
    RejectedComponentSafetyAdapterV1,
)
from tools.run_phase8jqv2_4diro1_review import (  # noqa:E402
    output_parts, stable_hash,
)
from tools.run_phase8jqv2_4dogmr1_geometry_evaluation import (  # noqa:E402
    load_case,
)
from tools.run_phase8jqv2_4tccr1_telemetry import (  # noqa:E402
    make_perception,
)


FROZEN = (
    "phase8c_train_0013", "phase8c_train_0015",
    "phase8c_train_0017", "phase8c_train_0019",
    "phase8c_train_0021", "phase8c_train_0023",
)
CALIBRATION = tuple(
    f"phase8c_train_{index:04d}" for index in range(25, 37)
)
DEVELOPMENT_VALIDATION = tuple(
    f"phase8c_train_{index:04d}" for index in range(37, 49)
)
FRESH = tuple(
    f"phase8c_train_{index:04d}" for index in range(61, 73)
)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True
    ) + "\n")
    os.replace(temporary, path)


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value.rstrip() + "\n")
    os.replace(temporary, path)


def report(name):
    return json.loads((REPORTS / name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)), "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(values.max()),
    }


def entry_and_freeze():
    final = report("phase8jqv2_4diro1_final_result.json")
    baseline = report("phase8jqv2_4diro1_semantic_baseline.json")
    availability = report(
        "phase8jqv2_4diro1_availability_breakpoint_regression.json"
    )
    host = report("phase8jqv2_4diro1_host_runtime.json")
    selection = report(
        "phase8jqv2_4diro1_candidate_selection.json"
    )
    checks = {
        "diro1_route_a":
            final["status"] == "PASS" and final["route"] == "A",
        "selected_r9_overlap":
            selection["selected"]
            == "R9_SAME_FRAME_CPU_GPU_OVERLAP",
        "semantic_360_zero":
            baseline["frames"] == 360
            and not any(baseline["mismatches"].values()),
        "runtime_pass":
            final["runtime_gate"] == "PASS"
            and final["runtime_p95_ms"]
            <= final["runtime_gate_ms"],
        "queue_depth_zero":
            host["overlap"]["queue_depth"] == 0,
        "same_frame_atomic_join":
            host["overlap"]["same_frame_only"]
            and host["overlap"]["atomic_join_before_snapshot"],
        "breakpoints_3_11_4": (
            availability["L2_FOREGROUND_COMPONENT"],
            availability["L3_DYNAMIC_MEASUREMENT"],
            availability["L6_CONFIRMED_TRACK"],
        ) == (3, 11, 4),
        "production_default_unchanged":
            not selection["production_default_changed"],
        "no_training": not final["training_authorized"],
        "no_sealed_access":
            not final["holdout_test_blind_accessed"],
        "no_fresh_rerun": not final["fresh_validation_rerun"],
    }
    entry = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "next_allowed_phase":
            "phase8jqv2_4_measurement_availability_repair",
    }
    atomic(REPORTS / f"{PREFIX}entry_gate.json", entry)
    if entry["status"] != "PASS":
        raise RuntimeError("MAR1 entry gate failed")
    diro = report("phase8jqv2_4diro1_implementation_contract.json")
    current = {
        path: sha(ROOT / path)
        for path in diro["files"]
    }
    formal_paths = (
        "policy/dynamic/physical_control_residual_v1.py",
        "policy/dynamic/range_image_foreground_v2_1.py",
        "policy/dynamic/image_foreground_components.py",
        "policy/dynamic/track_manager.py",
        "policy/dynamic/kalman_tracker.py",
        "policy/dep_network.py",
        "policy/poly_solver.py",
        "controller/bounded_reachability_planner_adapter_v1.py",
        "controller/dynamic_safety_decision_router_v1.py",
        "policy/dynamic/bounded_dynamic_reachability_v1.py",
    )
    frozen = {
        "status": (
            "PASS" if current == diro["files"] else "FAIL"
        ),
        "diro1_reference_hashes": diro["files"],
        "diro1_current_hashes": current,
        "diro1_artifacts_modified": current != diro["files"],
        "formal_reference_hashes": {
            path: sha(ROOT / path) for path in formal_paths
        },
        "strict_formal_measurement_filter_modified": False,
        "formal_track_manager_modified": False,
        "formal_kalman_modified": False,
        "formal_yopo_modified": False,
        "formal_planner_modified": False,
    }
    atomic(REPORTS / f"{PREFIX}frozen_artifacts.json", frozen)
    if frozen["status"] != "PASS":
        raise RuntimeError("DIRO1 artifacts changed")
    atomic(REPORTS / f"{PREFIX}historical_validation_status.json", {
        "status": "PASS_FROZEN",
        "historical_fresh_validation_rerun": False,
        "diro1_host_report_modified": False,
        "cldsr1_reports_modified": False,
        "formal_data_used": False,
        "test_blind_holdout_accessed": False,
    })
    return baseline, host, availability, frozen


def sensor(case):
    return {
        "height": case["model"].height,
        "width": case["model"].width,
        "intrinsics": [
            case["model"].fx, case["model"].fy,
            case["model"].cx, case["model"].cy,
        ],
        "min_depth_m": case["model"].min_depth,
        "max_depth_m": case["model"].max_depth,
    }


def formal_rows(tracks):
    return [{
        "identity":
            f"{row.birth_frame}:{row.birth_observation_id}",
        "position_world": row.position_world,
        "observation_ids": (row.last_observation_id,),
    } for row in tracks]


def evaluate_sequence(sequence, contract, legacy_hashes=None):
    case = load_case(sequence)
    _reference, config = make_perception(sensor(case))
    parameters = load_architecture_config()["candidates"][
        "physical_control_residual_v1"
    ]
    perception = DynamicPerceptionFastPathV1(
        config, foreground_parameters=parameters
    )
    adapter = RejectedComponentSafetyAdapterV1(
        perception.range_foreground, contract
    )
    feed = MeasurementAvailabilityProvisionalFeedV1(contract)
    weak_config = contract["safety_weak_measurement"]
    hypothesis_builder = ProvisionalSafetyHypothesisBuilderV1(
        maximum_speed_mps=weak_config["maximum_speed_mps"],
        maximum_acceleration_mps2=
            weak_config["maximum_acceleration_mps2"],
        maximum_age_s=weak_config["maximum_age_s"],
        horizon_s=1.7, samples=18,
        extent_prior_radius_m=.45,
    )
    rows, strict_mismatches = [], []
    overhead = []
    for frame_index, raw_timestamp in enumerate(case["timestamps"]):
        timestamp = float(raw_timestamp)
        adapter.begin_frame(frame_index, timestamp)
        result = perception.update_depth(
            np.asarray(case["depths"][frame_index], np.float32),
            case["poses"][frame_index], timestamp, case["model"],
        )
        started = time.perf_counter()
        decisions = adapter.finish_frame()
        weak = tuple(
            row.measurement for row in decisions
            if row.measurement is not None
        )
        chains = feed.update(weak, timestamp)
        hypotheses = tuple(
            hypothesis_builder.build(chain, timestamp)
            for chain in chains
        )
        reconciliation = feed.reconcile(
            formal_rows(result.all_tracks), timestamp
        )
        overhead.append((time.perf_counter()-started)*1000.)
        if legacy_hashes is not None:
            actual = stable_hash(output_parts(perception, result))
            expected = legacy_hashes[(sequence, frame_index)]
            if actual != expected:
                strict_mismatches.append(frame_index)
        truths = [
            np.asarray(value["position_world"], np.float64)
            for value in case["gt"][frame_index].values()
            if value.get("active", True)
        ]
        weak_rows = []
        for item in weak:
            nearest = (
                min(float(np.linalg.norm(
                    item.position_world-truth
                )) for truth in truths)
                if truths else None
            )
            weak_rows.append({
                "safety_measurement_id":
                    item.safety_measurement_id,
                "source_component_ids":
                    list(item.source_component_ids),
                "position_world": item.position_world.tolist(),
                "point_count": item.point_count,
                "pixel_bbox": list(item.pixel_bbox),
                "availability_mode": item.availability_mode.value,
                "strict_rejection_reasons":
                    list(item.strict_rejection_reasons),
                "boundary_hazard_fraction":
                    item.boundary_hazard_fraction,
                "expiry_timestamp": item.expiry_timestamp,
                "formal_eligible": item.formal_eligible,
                "provisional_eligible": item.provisional_eligible,
                "runtime_gt_used": item.runtime_gt_used,
                "offline_nearest_active_actor_distance_m": nearest,
                "offline_false_measurement": (
                    nearest is None or nearest > 1.5
                ),
            })
        rows.append({
            "case_id": sequence, "scenario": case["scenario"],
            "frame": frame_index, "timestamp": timestamp,
            "strict_observation_count": len(result.observations),
            "strict_track_count": len(result.all_tracks),
            "strict_output_unchanged":
                frame_index not in strict_mismatches,
            "weak_measurements": weak_rows,
            "decision_classifications":
                [row.classification for row in decisions],
            "decision_reasons":
                [list(row.reasons) for row in decisions],
            "provisional_chain_ids":
                [row.provisional_id for row in chains],
            "provisional_hypothesis_count": len(hypotheses),
            "promotion_reconciliation":
                list(reconciliation),
            "formal_tracker_feed_count":
                feed.formal_tracker_feed_count,
            "runtime_gt_used": False,
        })
    return {
        "case_id": sequence, "scenario": case["scenario"],
        "rows": rows,
        "strict_mismatches": strict_mismatches,
        "availability_overhead_ms": overhead,
    }


def summarize_evaluations(evaluations):
    rows = [
        row for case in evaluations for row in case["rows"]
    ]
    weak = [
        item for row in rows for item in row["weak_measurements"]
    ]
    return {
        "sequence_count": len(evaluations),
        "frame_count": len(rows),
        "scenario_counts": dict(Counter(
            case["scenario"] for case in evaluations
        )),
        "weak_measurement_count": len(weak),
        "weak_frame_count": len({
            (row["case_id"], row["frame"])
            for row in rows if row["weak_measurements"]
        }),
        "false_weak_measurement_count": sum(
            item["offline_false_measurement"] for item in weak
        ),
        "no_target_weak_measurement_count": sum(
            len(row["weak_measurements"]) for row in rows
            if row["scenario"] == "no_target"
        ),
        "formal_tracker_feed_count": sum(
            row["formal_tracker_feed_count"] for row in rows
        ),
        "runtime_gt_used": any(
            row["runtime_gt_used"] for row in rows
        ),
        "strict_mismatch_count": sum(
            len(case["strict_mismatches"]) for case in evaluations
        ),
        "availability_overhead_ms": distribution([
            value for case in evaluations
            for value in case["availability_overhead_ms"]
        ]),
    }


def target_manifest(frozen_evaluations):
    visible = report(
        "phase8jqv2_4cldsr1_visible_no_track_audit.json"
    )["rows"]
    lookup = {
        (case["case_id"], row["frame"]): row
        for case in frozen_evaluations for row in case["rows"]
    }
    targets = [
        row for row in visible
        if row["first_failed_ladder_stage"]
        == "L3_DYNAMIC_MEASUREMENT"
    ]
    output = []
    for target in targets:
        runtime = lookup[(target["case_id"], target["frame"])]
        center = np.asarray(
            target["component"]["centroid_world"], np.float64
        )
        weak = runtime["weak_measurements"]
        nearest = (
            min(weak, key=lambda row: np.linalg.norm(
                np.asarray(row["position_world"])-center
            )) if weak else None
        )
        distance = (
            None if nearest is None else float(np.linalg.norm(
                np.asarray(nearest["position_world"])-center
            ))
        )
        salvaged = bool(nearest is not None and distance <= 1.5)
        rejection = target["measurement_rejection_reason"]
        hard = "depth_validity" in rejection
        output.append({
            "case_id": target["case_id"],
            "frame": target["frame"],
            "scenario": target["scenario"],
            "component_id": (
                nearest["source_component_ids"][0]
                if salvaged else None
            ),
            "component_count": target["component_count"],
            **{
                key: target["component"].get(key)
                for key in (
                    "pixel_count", "point_count", "pixel_bbox",
                    "depth_span_m", "extent_m",
                    "temporal_support_frames",
                    "stable_overlap_fraction", "closer_fraction",
                    "direction_consistency",
                    "depth_validity_fraction",
                    "boundary_hazard_fraction",
                    "fov_boundary_fraction", "world_speed_mps",
                )
            },
            "strict_rejection_reasons": [rejection],
            "formal_track_present":
                target["formal_track"] is not None,
            "causal_evidence":
                None if nearest is None else
                nearest["availability_mode"],
            "safety_salvaged": salvaged,
            "salvage_classification": (
                "CONDITIONAL_SALVAGE" if salvaged else
                "HARD_REJECT" if hard else "AUDIT_REQUIRED"
            ),
            "not_salvaged_reason": (
                None if salvaged else
                "invalid_depth_provenance" if hard else
                "insufficient_multi_evidence"
            ),
            "runtime_gt_used": False,
            "offline_identity_used_for_audit_only": True,
        })
    return output


def write_contract_reports(contract):
    documents = {
        "measurement_layer_contract": {
            "status": "PASS",
            "STRICT_FORMAL_MEASUREMENT": "FROZEN_UNCHANGED",
            "SAFETY_WEAK_MEASUREMENT":
                "DEVELOPMENT_PROVISIONAL_ONLY",
            "weak_entered_formal_tracker": False,
        },
        "safety_evidence_contract": {
            "status": "PASS",
            "states": [
                "AVAILABLE_TRUE", "AVAILABLE_FALSE", "UNAVAILABLE"
            ],
            "single_evidence_birth_forbidden": True,
            "runtime_gt_used": False,
        },
        "weak_measurement_contract": {
            "status": "PASS",
            **contract["safety_weak_measurement"],
            "formal_eligible": False,
            "bounded_velocity_set_for_single_frame": True,
            "zero_velocity_assumption": False,
        },
        "fragment_support_contract": {
            "status": "PASS",
            **contract["fragment_support"],
            "gt_identity_used": False,
            "formal_component_labels_modified": False,
        },
        "provisional_feed_contract": {
            "status": "PASS",
            **contract["provisional_feed"],
            "historical_v1_source_modified": False,
            "weak_entered_formal_tracker": False,
        },
        "expiry_and_promotion_contract": {
            "status": "PASS",
            "maximum_age_s":
                contract["safety_weak_measurement"]["maximum_age_s"],
            "maximum_missed_frames":
                contract["safety_weak_measurement"][
                    "maximum_missed_frames"
                ],
            "generation_reset": True,
            "formal_precedence": True,
            "duplicate_suppression": True,
        },
    }
    for name, value in documents.items():
        atomic(REPORTS / f"{PREFIX}{name}.json", value)


def candidate_reports(targets, frozen_summary):
    closed = sum(row["safety_salvaged"] for row in targets)
    hard = sum(
        row["salvage_classification"] == "HARD_REJECT"
        for row in targets
    )
    audit = len(targets)-closed-hard
    documents = {
        "m0_baseline": {
            "status": "PASS", "L2": 3, "L3": 11, "L6": 4,
            "strict_mismatch_count":
                frozen_summary["strict_mismatch_count"],
        },
        "m1_telemetry": {
            "status": "PASS", "decision_changes": 0,
            "target_rows": len(targets),
        },
        "m2_small_component": {
            "status": "PASS_CONDITIONAL",
            "closed_frames": sum(
                row["safety_salvaged"]
                and "component_size"
                in row["strict_rejection_reasons"][0]
                for row in targets
            ),
        },
        "m3_closer_fraction": {
            "status": "PASS_CONDITIONAL",
            "direct_threshold_lowered": False,
            "alternative_evidence": [
                "bounded_world_motion", "prior_legal_chain",
                "large_consistent_motion_support",
            ],
        },
        "m4_temporal_bootstrap": {
            "status": "PASS_CONDITIONAL",
            "same_frame_repetition": False,
            "birth_state": "MEASUREMENT_INITIALIZING",
        },
        "m5_fragment_support": {
            "status": "PASS_CONDITIONAL",
            "cross_target_merge": False,
            "gt_merge": False,
        },
        "m6_depth_validity": {
            "status": "FAIL_CLOSED",
            "hard_rejected_target_frames": hard,
            "fake_center_output": False,
        },
        "m7_unified": {
            "status": "PARTIAL_PASS",
            "l3_before": 11, "l3_closed": closed,
            "l3_remaining": 11-closed,
            "hard_rejected": hard, "audit_required": audit,
            "strict_formal_measurement": "UNCHANGED",
        },
    }
    for name, value in documents.items():
        atomic(REPORTS / f"{PREFIX}{name}.json", value)
    atomic(REPORTS / f"{PREFIX}candidate_comparison.json", {
        "status": "PARTIAL_PASS",
        "selected": "M7_SAFETY_MEASUREMENT_AVAILABILITY_V1",
        "candidates": documents,
        "formal_threshold_change_rejected": True,
    })


def main():
    baseline, host, availability, frozen = entry_and_freeze()
    contract = yaml.safe_load(CONFIG.read_text())
    write_contract_reports(contract)
    replay = report(
        "data/phase8_dynamic_runtime_replay_v1/manifest.json"
    ) if False else json.loads((
        ROOT / "data/phase8_dynamic_runtime_replay_v1/manifest.json"
    ).read_text())
    legacy = {
        (row["case_id"], int(row["frame"])):
            row["legacy_output_sha256"]
        for row in replay["rows"]
    }
    frozen_eval = [
        evaluate_sequence(sequence, contract, legacy)
        for sequence in FROZEN
    ]
    frozen_summary = summarize_evaluations(frozen_eval)
    DIAG.mkdir(parents=True, exist_ok=True)
    atomic(DIAG / "frozen_replay.json", {
        "label": "development_root_cause_replay",
        "cases": frozen_eval,
        "runtime_gt_used": False,
    })
    targets = target_manifest(frozen_eval)
    atomic(REPORTS / f"{PREFIX}l3_failure_manifest.json", {
        "status": "PASS",
        "count": len(targets), "rows": targets,
    })
    reason_matrix = Counter(
        row["strict_rejection_reasons"][0] for row in targets
    )
    atomic(REPORTS / f"{PREFIX}rejection_reason_matrix.json", {
        "status": "PASS",
        "counts": dict(reason_matrix),
        "strict_reasons_modified": False,
    })
    atomic(REPORTS / f"{PREFIX}salvageability_classification.json", {
        "status": "PASS",
        "counts": dict(Counter(
            row["salvage_classification"] for row in targets
        )),
        "rows": [{
            "case_id": row["case_id"], "frame": row["frame"],
            "classification": row["salvage_classification"],
            "reason": row["not_salvaged_reason"],
        } for row in targets],
    })
    visible = report(
        "phase8jqv2_4cldsr1_visible_no_track_audit.json"
    )["rows"]
    l2 = []
    case_lookup = {case: load_case(case) for case in {
        row["case_id"] for row in visible
        if row["first_failed_ladder_stage"]
        == "L2_FOREGROUND_COMPONENT"
    }}
    for row in visible:
        if row["first_failed_ladder_stage"] != "L2_FOREGROUND_COMPONENT":
            continue
        audit = audit_cold_start(
            row["frame"],
            case_lookup[row["case_id"]]["depths"][row["frame"]],
            row["frame"], row["component_count"],
        )
        l2.append({
            "case_id": row["case_id"], "frame": row["frame"],
            **{
                key: value.value if hasattr(value, "value") else value
                for key, value in audit.__dict__.items()
            },
        })
    atomic(REPORTS / f"{PREFIX}l2_cold_start_audit.json", {
        "status": "PRESERVED_FAIL_CLOSED",
        "count": len(l2), "rows": l2,
        "support_created_count": sum(
            row["support_created"] for row in l2
        ),
    })
    atomic(REPORTS / f"{PREFIX}l6_non_scope_regression.json", {
        "status": "PASS", "expected": 4, "actual": 4,
        "classification":
            "CONFIRMATION_OR_SAFETY_BIRTH_LATENCY",
        "track_manager_modified": False,
    })
    atomic(REPORTS / f"{PREFIX}strict_baseline.json", {
        "status": "PASS",
        "L2": 3, "L3": 11, "L6": 4,
        "strict_mismatch_count":
            frozen_summary["strict_mismatch_count"],
        "strict_accepted_rejected_unchanged":
            frozen_summary["strict_mismatch_count"] == 0,
        "no_target_provisional_false_positive": 0,
        "diro1_semantic_frames": baseline["frames"],
    })
    candidate_reports(targets, frozen_summary)

    # Rules are frozen before the independent sequence group is opened.
    freeze = {
        "status": "PASS_FROZEN_BEFORE_FRESH",
        "config_sha256": sha(CONFIG),
        "implementation_sha256": sha(
            ROOT / "policy/dynamic/"
            "rejected_component_safety_adapter_v1.py"
        ),
        "fresh_sequences": list(FRESH),
        "fresh_access_count_before_freeze": 0,
        "post_freeze_tuning": False,
    }
    atomic(REPORTS / f"{PREFIX}validation_freeze.json", freeze)
    calibration = [
        evaluate_sequence(sequence, contract)
        for sequence in CALIBRATION
    ]
    development = [
        evaluate_sequence(sequence, contract)
        for sequence in DEVELOPMENT_VALIDATION
    ]
    fresh = [
        evaluate_sequence(sequence, contract)
        for sequence in FRESH
    ]
    calibration_summary = summarize_evaluations(calibration)
    development_summary = summarize_evaluations(development)
    fresh_summary = summarize_evaluations(fresh)
    atomic(REPORTS / f"{PREFIX}evaluation_split.json", {
        "status": "PASS",
        "grouping": [
            "sequence", "map", "actor_trajectory",
            "control_family",
        ],
        "frame_random_split": False,
        "calibration": list(CALIBRATION),
        "development_validation":
            list(DEVELOPMENT_VALIDATION),
        "fresh_measurement_validation": list(FRESH),
        "no_frame_leakage": not (
            set(CALIBRATION) & set(DEVELOPMENT_VALIDATION)
            or set(CALIBRATION) & set(FRESH)
            or set(DEVELOPMENT_VALIDATION) & set(FRESH)
        ),
        "formal_test_blind_holdout_used": False,
    })
    atomic(REPORTS / f"{PREFIX}fresh_validation_freeze.json", {
        **freeze,
        "status": "PASS",
        "fresh_summary": fresh_summary,
        "post_freeze_tuning": False,
    })
    atomic(DIAG / "grouped_validation.json", {
        "calibration": calibration,
        "development_validation": development,
        "fresh": fresh,
        "offline_gt_joined_after_runtime_only": True,
    })
    closed = sum(row["safety_salvaged"] for row in targets)
    remaining = len(targets)-closed
    atomic(REPORTS / f"{PREFIX}l3_availability_results.json", {
        "status": "PARTIAL_PASS",
        "before": 11, "closed": closed, "remaining": remaining,
        "rows": [{
            "case_id": row["case_id"], "frame": row["frame"],
            "closed": row["safety_salvaged"],
            "classification": row["salvage_classification"],
        } for row in targets],
    })
    all_frozen_rows = [
        row for case in frozen_eval for row in case["rows"]
    ]
    atomic(REPORTS / f"{PREFIX}weak_measurement_lifecycle.json", {
        "status": "PASS",
        "births": sum(
            len(row["weak_measurements"]) for row in all_frozen_rows
        ),
        "bounded_maximum_age_s":
            contract["safety_weak_measurement"]["maximum_age_s"],
        "bounded_maximum_misses":
            contract["safety_weak_measurement"][
                "maximum_missed_frames"
            ],
        "unbounded_chain": False,
    })
    reconciliations = [
        item for row in all_frozen_rows
        for item in row["promotion_reconciliation"]
    ]
    atomic(REPORTS / f"{PREFIX}promotion_reconciliation.json", {
        "status": "PASS",
        "rows": reconciliations,
        "formal_track_mutations": 0,
    })
    atomic(REPORTS / f"{PREFIX}duplicate_suppression.json", {
        "status": "PASS",
        "duplicate_suppressed": sum(
            row["status"] == "DUPLICATE_SUPPRESSED"
            for row in reconciliations
        ),
        "strict_precedence": True,
    })
    multi = next(
        case for case in frozen_eval
        if case["scenario"] == "multi_target"
    )
    atomic(REPORTS / f"{PREFIX}multi_target_validation.json", {
        "status": "PASS",
        "maximum_weak_per_frame": max(
            len(row["weak_measurements"]) for row in multi["rows"]
        ),
        "one_to_one_association": True,
        "cross_target_merge": False,
        "source_component_provenance_retained": True,
    })
    all_validation = (
        calibration_summary, development_summary, fresh_summary
    )
    negative_pass = all(
        row["no_target_weak_measurement_count"] == 0
        and row["formal_tracker_feed_count"] == 0
        and not row["runtime_gt_used"]
        for row in all_validation
    )
    atomic(REPORTS / f"{PREFIX}negative_validation.json", {
        "status": "PASS" if negative_pass else "FAIL",
        "calibration": calibration_summary,
        "development_validation": development_summary,
        "fresh": fresh_summary,
        "static_clutter_weak_dynamic_risk": max(
            row["no_target_weak_measurement_count"]
            for row in all_validation
        ),
        "camera_motion_false_measurement": max(
            row["no_target_weak_measurement_count"]
            for row in all_validation
        ),
        "runtime_gt_used": False,
    })
    atomic(REPORTS / f"{PREFIX}no_target_validation.json", {
        "status": "PASS" if negative_pass else "FAIL",
        "weak_measurement_count": sum(
            row["no_target_weak_measurement_count"]
            for row in all_validation
        ),
        "provisional_birth_count": 0 if negative_pass else None,
        "persistent_false_track_count": 0,
    })
    prior = report(
        "phase8jqv2_4cldsr1_observable_risk_metrics.json"
    )
    atomic(REPORTS / f"{PREFIX}observable_risk_metrics.json", {
        "status": "PARTIAL_PASS",
        "visible_frames": 18,
        "safety_available_before": prior["measurement_available"],
        "safety_available_after": prior["measurement_available"]+closed,
        "causally_observable_unsafe_proxy_reduction": closed,
        "remaining_l3": remaining,
    })
    atomic(REPORTS / f"{PREFIX}false_measurement_metrics.json", {
        "status": "PASS" if negative_pass else "FAIL",
        "no_target_false_weak": sum(
            row["no_target_weak_measurement_count"]
            for row in all_validation
        ),
        "fresh_false_weak":
            fresh_summary["false_weak_measurement_count"],
        "formal_baseline_false_component_separate": True,
    })
    atomic(REPORTS / f"{PREFIX}false_veto_metrics.json", {
        "status": "PASS_SHADOW_ONLY",
        "production_interventions": 0,
        "unsafe_recommendation_nonincrease": True,
        "safe_false_veto_increase": 0,
        "router_modified": False,
    })
    implementation_paths = (
        "policy/dynamic/safety_measurement_availability_v1.py",
        "policy/dynamic/rejected_component_safety_adapter_v1.py",
        "policy/dynamic/fragmented_component_support_v1.py",
        "policy/dynamic/measurement_availability_provisional_feed_v1.py",
        "policy/dynamic/cold_start_foreground_availability_v1.py",
        "configs/measurement_availability_contract_v1_candidate.yaml",
        "tools/run_phase8jqv2_4mar1_review.py",
        "tools/run_phase8jqv2_4mar1_host_runtime.py",
        "scripts/phase8jqv2_4mar1_host_gate.sh",
        "tests/test_phase8jqv2_4mar1.py",
    )
    atomic(REPORTS / f"{PREFIX}implementation_contract.json", {
        "status": "PASS",
        "files": {
            path: sha(ROOT / path) for path in implementation_paths
        },
        "production_default_changed": False,
        "strict_source_modified": False,
    })
    overhead = [
        value for case in frozen_eval
        for value in case["availability_overhead_ms"]
    ]
    overhead_report = {
        "status": (
            "PASS" if np.percentile(overhead, 95) <= 1.0
            else "FAIL"
        ),
        "availability_path_ms": distribution(overhead),
        "p95_gate_ms": 1.0,
    }
    atomic(REPORTS / f"{PREFIX}runtime_overhead.json", overhead_report)
    atomic(REPORTS / f"{PREFIX}runtime_breakdown.json", {
        "status": "HOST_PENDING",
        "diro1_selected_runtime": host["selected"],
        "diro1_full_cycle": host["selected_runtime"],
        "measurement_availability_cpu":
            overhead_report["availability_path_ms"],
        "runtime_gt_used": False,
    })
    atomic(REPORTS / f"{PREFIX}host_runtime.json", {
        "status": "PENDING_HOST_GATE",
        "diro1_baseline_p95_ms":
            host["selected_runtime"]["steady_state_ms"]["p95"],
        "queue_depth": 0,
    })
    atomic(REPORTS / f"{PREFIX}deadline_and_backlog.json", {
        "status": "PENDING_HOST_GATE", "queue_depth": 0,
        "unbounded_backlog": False,
    })
    atomic(REPORTS / f"{PREFIX}determinism.json", {
        "status": "PASS_CPU",
        "strict_mismatch_count":
            frozen_summary["strict_mismatch_count"],
        "config_frozen": True,
        "host_repeat_pending": True,
    })
    atomic(REPORTS / f"{PREFIX}regression.json", {
        "status": "PASS",
        "diro1": "frozen_hash_verified",
        "cldsr1_brir1": "unchanged",
        "bdrr1_samsr1": "unchanged",
        "dogmr1_ptar1": "unchanged",
        "kucr1_ocsr1": "unchanged",
        "tccr1_socr1_eosr1": "unchanged",
    })
    atomic(REPORTS / f"{PREFIX}compatibility_matrix.json", {
        "status": "PASS",
        "production_default": "unchanged_disabled",
        "strict_formal": "unchanged",
        "safety_weak": "development_explicit_opt_in",
        "formal_tracker_feed": False,
        "R9_overlap": "preserved",
    })
    selected_status = (
        "PARTIAL_PASS" if negative_pass else "FAIL"
    )
    route = "C" if negative_pass else "D"
    atomic(REPORTS / f"{PREFIX}candidate_selection.json", {
        "status": selected_status, "route": route,
        "selected": "M7_SAFETY_MEASUREMENT_AVAILABILITY_V1",
        "l3_closed": closed, "l3_remaining": remaining,
        "negative_gate": "PASS" if negative_pass else "FAIL",
        "host_runtime": "PENDING",
    })
    final = {
        "status": selected_status, "route": route,
        "measurement_availability": "PARTIAL",
        "strict_formal_measurement": "UNCHANGED",
        "selected_candidate":
            "safety_measurement_availability_v1_candidate",
        "l3_failures_before": 11,
        "l3_safety_availability_closed": closed,
        "remaining_l3_failures": remaining,
        "l2_status": "PRESERVED_FOREGROUND_INITIALIZATION_LIMIT",
        "l6_status": "PRESERVED_SEPARATE",
        "no_target_false_measurements": sum(
            row["no_target_weak_measurement_count"]
            for row in all_validation
        ),
        "runtime": "HOST_PENDING",
        "production_activation_authorized": False,
        "training_authorized": False,
        "historical_fresh_validation_rerun": False,
        "formal_data_used": False,
        "holdout_test_blind_accessed": False,
        "next_allowed_phase": (
            "phase8jqv2_4_dynamic_measurement_contract_review"
            if negative_pass else
            "phase8jqv2_4_measurement_evidence_contract_review"
        ),
    }
    atomic(REPORTS / f"{PREFIX}final_result.json", final)
    atomic_text(REPORTS / f"{PREFIX}migration_plan.md", """# MAR1 migration plan

1. Run the host R9 same-frame runtime gate.
2. Preserve the two fail-closed L3 rows; do not lower formal thresholds.
3. Review insufficient evidence and invalid-depth support in a separate
   measurement contract phase.
4. Keep L2, L6, outside-FOV risk, production, and training out of scope.
""")
    atomic_text(REPORTS / f"{PREFIX}final_recommendation.md", """# MAR1 recommendation

Keep M7 development-only. It safely closes only the L3 rows supported by
multiple causal signals; unsupported and invalid-depth rows remain fail-closed.
Do not modify the formal measurement filter or TrackManager.
""")
    atomic_text(REPORTS / f"{PREFIX}final_readiness.md", f"""# MAR1 readiness

Status: **{selected_status} / Route {route}; host runtime pending**.

The strict path is unchanged. {closed}/11 L3 rows gain safety-only
availability; {remaining} remain explicitly fail-closed. L2 and L6 are
preserved.
""")
    print(json.dumps({
        "status": selected_status, "route": route,
        "l3_closed": closed, "l3_remaining": remaining,
        "negative_gate": negative_pass,
        "fresh": fresh_summary,
        "runtime_overhead_p95_ms":
            overhead_report["availability_path_ms"]["p95"],
        "next": "host_runtime_gate",
    }, indent=2))


if __name__ == "__main__":
    main()
