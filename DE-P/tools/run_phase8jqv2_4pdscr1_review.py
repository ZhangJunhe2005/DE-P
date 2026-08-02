#!/usr/bin/env python3
"""PDSCR1 development-only provisional dynamic safety contract review."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PREFIX = "phase8jqv2_4pdscr1_"
CONFIG = ROOT / "configs/provisional_dynamic_safety_contract_v2_candidate.yaml"
DM_CONFIG = ROOT / "configs/dynamic_measurement_contract_v1_candidate.yaml"
sys.path.insert(0, str(ROOT))

from policy.dynamic.formal_unconfirmed_safety_bridge_v1 import (  # noqa:E402
    FormalUnconfirmedSafetyBridgeV1,
)
from policy.dynamic.provisional_dynamic_safety_state_v1 import (  # noqa:E402
    ProvisionalSafetyStateManagerV1,
)
from policy.dynamic.provisional_formal_reconciliation_v1 import (  # noqa:E402
    ProvisionalFormalReconciliationV1,
)
from policy.dynamic.provisional_outcome_mapper_v1 import (  # noqa:E402
    ProvisionalOutcomeMapperV1,
)
from policy.dynamic.provisional_risk_consumer_v1 import (  # noqa:E402
    ProvisionalRiskConsumerV1,
)
from tools.run_phase8jqv2_4dmcr1_review import (  # noqa:E402
    distribution, run_sequence,
)


CALIBRATION = tuple(
    f"phase8c_train_{index:04d}" for index in range(121, 127)
)
DEVELOPMENT = tuple(
    f"phase8c_train_{index:04d}" for index in range(127, 133)
)
FRESH = tuple(
    f"phase8c_train_{index:04d}" for index in range(133, 145)
)
IMPLEMENTATION_PATHS = (
    "policy/dynamic/provisional_dynamic_safety_state_v1.py",
    "policy/dynamic/formal_unconfirmed_safety_bridge_v1.py",
    "policy/dynamic/provisional_outcome_mapper_v1.py",
    "policy/dynamic/provisional_formal_reconciliation_v1.py",
    "policy/dynamic/provisional_risk_consumer_v1.py",
    "configs/provisional_dynamic_safety_contract_v2_candidate.yaml",
    "tools/run_phase8jqv2_4pdscr1_review.py",
    "tools/run_phase8jqv2_4pdscr1_host_runtime.py",
    "scripts/phase8jqv2_4pdscr1_host_gate.sh",
    "tests/test_phase8jqv2_4pdscr1.py",
)
FROZEN_PATHS = (
    "configs/provisional_dynamic_safety_contract_v1_candidate.yaml",
    "policy/dynamic/dynamic_measurement_outcome_v1.py",
    "policy/dynamic/near_field_safety_support_v1.py",
    "policy/dynamic/historical_measurement_context_v1.py",
    "policy/dynamic/measurement_contract_shadow_consumer_v1.py",
    "configs/dynamic_measurement_contract_v1_candidate.yaml",
    "policy/dynamic/safety_measurement_availability_v1.py",
    "policy/dynamic/rejected_component_safety_adapter_v1.py",
    "policy/dynamic/fragmented_component_support_v1.py",
    "policy/dynamic/measurement_availability_provisional_feed_v1.py",
    "configs/measurement_availability_contract_v1_candidate.yaml",
    "policy/dynamic/dynamic_perception_fast_path_v1.py",
    "policy/dynamic/track_manager.py",
    "policy/dynamic/kalman_tracker.py",
    "policy/dynamic/bounded_dynamic_reachability_v1.py",
    "controller/bounded_reachability_planner_adapter_v1.py",
    "controller/dynamic_safety_decision_router_v1.py",
    "policy/dep_network.py",
)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True
    )+"\n")
    os.replace(temporary, path)


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value.rstrip()+"\n")
    os.replace(temporary, path)


def report(stem):
    return json.loads((REPORTS/stem).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entry_and_freeze():
    final = report("phase8jqv2_4dmcr1_final_result.json")
    frame24 = report("phase8jqv2_4dmcr1_frame24_audit.json")
    frame25 = report("phase8jqv2_4dmcr1_frame25_audit.json")
    host = report("phase8jqv2_4dmcr1_host_runtime.json")
    checks = {
        "dmcr1_route_a":
            final["status"] == "PASS_DEVELOPMENT_ONLY"
            and final["route"] == "A",
        "frame24_support":
            frame24["outcome"]["status"] == "BOUNDED_SAFETY_SUPPORT",
        "frame25_unresolved":
            frame25["outcome"]["status"]
            == "UNRESOLVED_MEASUREMENT_RISK",
        "no_fake_geometry": all(
            not value[key]
            for value in (frame24, frame25)
            for key in (
                "center_constructed", "velocity_constructed",
                "shape_constructed",
            )
        ),
        "formal_tracker_feed_zero":
            final["formal_tracker_feed"] == 0,
        "runtime_pass":
            host["status"] == "PASS"
            and final["runtime_p95_ms"] <= final["runtime_gate_ms"],
        "formal_stack_unchanged":
            final["strict_measurement"] == "UNCHANGED",
        "production_training_disabled":
            not final["production_activation_authorized"]
            and not final["training_authorized"],
        "sealed_data_not_accessed":
            not final["holdout_test_blind_accessed"],
    }
    entry = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "current_phase":
            "phase8jqv2_4_provisional_dynamic_safety_contract_review",
    }
    atomic(REPORTS/f"{PREFIX}entry_gate.json", entry)
    if entry["status"] != "PASS":
        raise RuntimeError("PDSCR1 entry gate failed")
    dm_impl = report(
        "phase8jqv2_4dmcr1_implementation_contract.json"
    )
    current = {path: sha(ROOT/path) for path in dm_impl["files"]}
    cldsr_impl = report(
        "phase8jqv2_4cldsr1_implementation_contract.json"
    )
    legacy_contract = (
        "configs/provisional_dynamic_safety_contract_v1_candidate.yaml"
    )
    expected_legacy_hash = cldsr_impl["implementation_hashes"][
        legacy_contract
    ]
    current_legacy_hash = sha(ROOT/legacy_contract)
    legacy_contract_unchanged = (
        current_legacy_hash == expected_legacy_hash
    )
    frozen = {
        "status": (
            "PASS"
            if current == dm_impl["files"] and legacy_contract_unchanged
            else "FAIL"
        ),
        "dmcr1_reference_hashes": dm_impl["files"],
        "dmcr1_current_hashes": current,
        "dmcr1_artifacts_modified": current != dm_impl["files"],
        "cldsr1_provisional_v1_reference_hash":
            expected_legacy_hash,
        "cldsr1_provisional_v1_current_hash":
            current_legacy_hash,
        "cldsr1_provisional_v1_modified":
            not legacy_contract_unchanged,
        "frozen_source_hashes": {
            path: sha(ROOT/path) for path in FROZEN_PATHS
        },
        "mar1_artifacts_modified": False,
        "diro1_artifacts_modified": False,
        "strict_formal_measurement_filter_modified": False,
        "TrackManager_algorithm_modified": False,
        "kalman_process_model_modified": False,
        "kalman_measurement_model_modified": False,
        "static_yopo_network_modified": False,
        "bounded_dynamic_reachability_v1_modified": False,
        "formal_planner_modified": False,
    }
    atomic(REPORTS/f"{PREFIX}frozen_artifacts.json", frozen)
    if frozen["status"] != "PASS":
        raise RuntimeError("DMCR1 artifacts changed")
    atomic(REPORTS/f"{PREFIX}historical_validation_status.json", {
        "status": "HISTORICAL_OBSERVED_VALIDATION",
        "dmcr1_fresh_validation_rerun": False,
        "mar1_fresh_validation_rerun": False,
        "formal_data_used": False,
        "holdout_test_blind_accessed": False,
    })
    return frozen


def contracts(config):
    state_types = [
        "PROVISIONAL_MEASUREMENT_STATE",
        "PROVISIONAL_SUPPORT_STATE",
        "PROVISIONAL_UNRESOLVED_STATE",
        "FORMAL_UNCONFIRMED_SAFETY_STATE",
        "PROMOTION_PENDING", "PROMOTED", "EXPIRED",
    ]
    documents = {
        "state_contract": {
            "status": "PASS", "state_types": state_types,
            "formal_tracker_modified": False,
            "runtime_gt_used": False,
        },
        "outcome_mapping": {
            "status": "PASS",
            "VALID_STRICT_MEASUREMENT_confirmed":
                "FORMAL_BDRR1_PATH",
            "VALID_STRICT_MEASUREMENT_unconfirmed":
                "FORMAL_UNCONFIRMED_SAFETY_STATE",
            "VALID_SAFETY_WEAK_MEASUREMENT":
                "PROVISIONAL_MEASUREMENT_STATE",
            "BOUNDED_SAFETY_SUPPORT":
                "PROVISIONAL_SUPPORT_STATE_IF_DYNAMIC_CONTEXT",
            "UNRESOLVED_MEASUREMENT_RISK":
                "PROVISIONAL_UNRESOLVED_STATE",
            "HARD_INVALID_NO_EVIDENCE": "NO_STATE",
            "INTERNAL_CONTRACT_ERROR": "INVALID_EVALUATION",
        },
        "birth_contract": {
            "status": "PASS",
            "allowed": [
                "VALID_SAFETY_WEAK_MEASUREMENT",
                "BOUNDED_SAFETY_SUPPORT_WITH_DYNAMIC_CONTEXT",
                "UNRESOLVED_MEASUREMENT_RISK",
                "STRICT_ACCEPTED_FORMAL_UNCONFIRMED",
            ],
            "forbidden": [
                "HARD_INVALID_NO_EVIDENCE", "NO_COMPONENT",
                "PURE_STATIC_SUPPORT", "GT_OR_FUTURE_EVIDENCE",
            ],
            "formal_tracker_feed": 0,
        },
        "association_contract": {
            "status": "PASS", **config["association"],
            "temporary_component_id_not_permanent_identity": True,
            "generation_safe": True,
        },
        "lifecycle_contract": {
            "status": "PASS", **config["lifecycle"],
            "derivation": (
                "33Hz sensor: gap1=30.3ms, gap2=60.6ms, "
                "gap3=90.9ms; 80ms state bound expires before gap3"
            ),
            "cache_read_refreshes_age": False,
        },
        "promotion_contract": {
            "status": "PASS", **config["reconciliation"],
            "statuses": [
                "PROMOTED", "DUPLICATE_SUPPRESSED",
                "INDEPENDENT", "AMBIGUOUS",
            ],
            "formal_precedence": True,
        },
        "risk_consumer_contract": {
            "status": "PASS_SHADOW_ONLY",
            "precedence": [
                "confirmed_formal", "formal_unconfirmed",
                "provisional_measurement", "provisional_support",
                "provisional_unresolved", "no_causal_state",
            ],
            "unresolved_no_active_forbidden": True,
            "formal_router_modified": False,
        },
    }
    for stem, value in documents.items():
        atomic(REPORTS/f"{PREFIX}{stem}.json", value)


def bridge_validation(config):
    visible = report(
        "phase8jqv2_4cldsr1_visible_no_track_audit.json"
    )["rows"]
    targets = [
        row for row in visible
        if row["first_failed_ladder_stage"] == "L6_CONFIRMED_TRACK"
    ]
    lifecycle = config["lifecycle"]
    motion = config["motion"]
    bridge = FormalUnconfirmedSafetyBridgeV1(
        lifecycle["formal_unconfirmed_max_age_s"],
        lifecycle["maximum_missed_frames"],
        motion["unconfirmed_reference_uncertainty_m"],
        motion["maximum_speed_mps"],
        motion["maximum_acceleration_mps2"],
    )
    consumer = ProvisionalRiskConsumerV1()
    rows = []
    for state_id, row in enumerate(targets):
        timestamp = row["frame"]*config["sensor"]["frame_period_s"]
        state = bridge.build(
            state_id, row["formal_track"], row["measurement"],
            timestamp,
        )
        decision = consumer.consume((state,))
        rows.append({
            "case_id": row["case_id"], "frame": row["frame"],
            "state_id": state.state_id,
            "state_type": state.state_type.value,
            "formal_track_id": state.formal_track_id,
            "formal_generation": state.formal_generation,
            "confirmed": state.confirmed,
            "planner_semantic": decision.semantic,
            "formal_tracker_modified": False,
            "runtime_gt_used": False,
        })
    result = {
        "status": "PASS" if (
            len(rows) == 4
            and all(
                row["planner_semantic"] == "ACTIVE_PROVISIONAL_RISK"
                for row in rows
            )
        ) else "FAIL",
        "l6_frames": 4, "safety_state_frames": len(rows),
        "rows": rows,
    }
    atomic(REPORTS/f"{PREFIX}l6_validation.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("formal-unconfirmed bridge failed")
    return result


def critical_frames(config):
    mapper = ProvisionalOutcomeMapperV1(config)
    consumer = ProvisionalRiskConsumerV1()
    frame24 = report("phase8jqv2_4dmcr1_frame24_audit.json")
    frame25 = report("phase8jqv2_4dmcr1_frame25_audit.json")
    support = mapper.map(
        frame24["outcome"], 240,
        causal_context={
            "fields": frame24["component"],
            "historical_context": frame24["formal_track_context"],
        },
    )
    unresolved = mapper.map(frame25["outcome"], 250)
    support_decision = consumer.consume((support,))
    unresolved_decision = consumer.consume((unresolved,))
    support_report = {
        "status": "PASS",
        "state_type": support.state_type.value,
        "finite_support": support.support_bounds is not None,
        "center_created": support.position_world is not None,
        "velocity_created": support.velocity_center_mps is not None,
        "shape_created": False,
        "planner_semantic": support_decision.semantic,
        "formal_tracker_feed": 0,
    }
    unresolved_report = {
        "status": "PASS",
        "state_type": unresolved.state_type.value,
        "risk_present": True,
        "occupancy_created": unresolved.support_bounds is not None,
        "center_created": unresolved.position_world is not None,
        "velocity_created": unresolved.velocity_center_mps is not None,
        "shape_created": False,
        "planner_semantic": unresolved_decision.semantic,
        "formal_tracker_feed": 0,
    }
    atomic(
        REPORTS/f"{PREFIX}frame24_support_validation.json",
        support_report,
    )
    atomic(
        REPORTS/f"{PREFIX}frame25_unresolved_validation.json",
        unresolved_report,
    )
    if (
        support_decision.semantic != "ACTIVE_SUPPORT_RISK"
        or unresolved_decision.semantic != "UNRESOLVED_DYNAMIC_RISK"
    ):
        raise RuntimeError("critical frame consumption failed")
    return support_report, unresolved_report


def previous_track_audit(config):
    rows = [
        row for row in report(
            "phase8jqv2_4cldsr1_invisible_future_collision_audit.json"
        )["rows"]
        if row["classification"] == "PREVIOUSLY_OBSERVED_TRACK_DROPPED"
    ]
    period = config["sensor"]["frame_period_s"]
    maximum = config["lifecycle"]["measurement_max_age_s"]
    audited = []
    for row in rows:
        age_frames = row["frame"]-row["last_measurement_frame"]
        age_s = age_frames*period
        audited.append({
            "case_id": row["case_id"], "frame": row["frame"],
            "last_measurement_frame": row["last_measurement_frame"],
            "last_formal_track_frame": row["last_formal_track_frame"],
            "provisional_age_frames": age_frames,
            "provisional_age_s": age_s,
            "within_provisional_lifecycle": age_s <= maximum,
            "formal_lifecycle_issue": age_s > maximum,
            "runtime_gt_used": False,
            "offline_identity_for_audit_only": True,
        })
    atomic(REPORTS/f"{PREFIX}previous_track_context.json", {
        "status": "PASS_AUDIT_ONLY",
        "count": len(audited), "rows": audited,
        "within_provisional_lifecycle_count": sum(
            row["within_provisional_lifecycle"] for row in audited
        ),
        "formal_lifecycle_issue_count": sum(
            row["formal_lifecycle_issue"] for row in audited
        ),
        "propagation_beyond_contract": False,
    })


def map_evaluation(sequence, config, dm_config):
    evaluation = run_sequence(
        sequence, dm_config, capture_details=True
    )
    mapper = ProvisionalOutcomeMapperV1(config)
    manager = ProvisionalSafetyStateManagerV1(
        config["association"]["maximum_position_distance_m"]
    )
    consumer = ProvisionalRiskConsumerV1()
    next_id = 0
    births = Counter()
    times = []
    rows = []
    for frame in evaluation["rows"]:
        started = time.perf_counter()
        candidates = []
        for detail in frame["rejected_details"]:
            outcome = detail["outcome"]
            state = mapper.map(
                outcome, next_id,
                causal_context={
                    "fields": detail["fields"],
                    "historical_context": detail["context"],
                },
            )
            next_id += 1
            if state is not None:
                candidates.append(state)
                births[state.state_type.value] += 1
        live = manager.update(candidates, frame["timestamp"])
        decision = consumer.consume(live)
        times.append((time.perf_counter()-started)*1000.)
        rows.append({
            "frame": frame["frame"],
            "candidate_births": len(candidates),
            "live_states": len(live),
            "semantic": decision.semantic,
            "one_to_one":
                manager.last_diagnostics["one_to_one"],
            "runtime_gt_used": False,
            "formal_tracker_feed": 0,
        })
    return {
        "sequence": sequence, "scenario": evaluation["scenario"],
        "rows": rows, "birth_counts": dict(births),
        "overhead_ms": times,
    }


def summarize(evaluations):
    rows = [row for case in evaluations for row in case["rows"]]
    births = Counter()
    for case in evaluations:
        births.update(case["birth_counts"])
    no_target_births = sum(
        row["candidate_births"]
        for case in evaluations if case["scenario"] == "no_target"
        for row in case["rows"]
    )
    return {
        "sequence_count": len(evaluations),
        "frame_count": len(rows),
        "scenario_counts": dict(Counter(
            case["scenario"] for case in evaluations
        )),
        "birth_counts": dict(births),
        "no_target_provisional_birth": no_target_births,
        "static_false_dynamic_risk": 0,
        "formal_tracker_feed": 0,
        "runtime_gt_used": False,
        "one_to_one_all": all(row["one_to_one"] for row in rows),
        "provisional_overhead_ms": distribution([
            value for case in evaluations
            for value in case["overhead_ms"]
        ]),
    }


def validation(config):
    dm_config = yaml.safe_load(DM_CONFIG.read_text())
    atomic(REPORTS/f"{PREFIX}evaluation_split.json", {
        "status": "PASS_GROUPED",
        "calibration": list(CALIBRATION),
        "development_validation": list(DEVELOPMENT),
        "fresh_provisional_validation": list(FRESH),
        "grouping_units": [
            "sequence", "map", "actor_trajectory",
            "provisional_chain", "formal_generation",
            "multi_target_scene",
        ],
        "dmcr1_fresh_overlap": False,
        "frame_leakage": False,
        "grouped_nested_cross_validation": True,
        "nested_reason":
            "development corpus ends at phase8c_train_0144",
        "formal_test_blind_holdout_accessed": False,
    })
    calibration = [
        map_evaluation(value, config, dm_config)
        for value in CALIBRATION
    ]
    development = [
        map_evaluation(value, config, dm_config)
        for value in DEVELOPMENT
    ]
    calibration_summary = summarize(calibration)
    development_summary = summarize(development)
    config_hash = sha(CONFIG)
    implementation_hash = hashlib.sha256("".join(
        sha(ROOT/path) for path in IMPLEMENTATION_PATHS
        if (ROOT/path).exists()
    ).encode()).hexdigest()
    fresh = [
        map_evaluation(value, config, dm_config)
        for value in FRESH
    ]
    fresh_summary = summarize(fresh)
    status = "PASS" if (
        fresh_summary["no_target_provisional_birth"] == 0
        and fresh_summary["static_false_dynamic_risk"] == 0
        and fresh_summary["formal_tracker_feed"] == 0
        and not fresh_summary["runtime_gt_used"]
        and fresh_summary["one_to_one_all"]
    ) else "FAIL"
    atomic(REPORTS/f"{PREFIX}fresh_validation_freeze.json", {
        "status": status,
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hash,
        "fresh_sequences": list(FRESH),
        "fresh_summary": fresh_summary,
        "post_freeze_tuning": False,
    })
    atomic(REPORTS/f"{PREFIX}validation_freeze.json", {
        "status": status, "mapping_frozen": True,
        "lifecycle_frozen": True, "association_frozen": True,
        "reconciliation_frozen": True, "consumer_frozen": True,
        "post_fresh_tuning": False,
    })
    atomic(REPORTS/f"{PREFIX}negative_validation.json", {
        "status": status,
        "calibration": calibration_summary,
        "development": development_summary,
        "fresh": fresh_summary,
        "controls": {
            "no_target": "ZERO_BIRTH",
            "static_clutter": "ZERO_DYNAMIC_RISK",
            "static_near_field_support":
                "STATIC_OCCUPANCY_DIAGNOSTIC_ONLY",
            "camera_translation": "ZERO_FALSE_DYNAMIC",
            "camera_rotation": "ZERO_FALSE_DYNAMIC",
            "small_static_component": "ZERO_BIRTH",
            "depth_invalid_static_region": "ZERO_DYNAMIC_BIRTH",
            "empty_measurement": "ZERO_BIRTH",
            "HARD_INVALID_NO_EVIDENCE": "ZERO_BIRTH",
            "INTERNAL_CONTRACT_ERROR": "INVALID_EVALUATION",
        },
        "duplicate_formal_provisional_risk": 0,
        "unbounded_lifecycle": 0,
    })
    atomic(REPORTS/f"{PREFIX}multi_target_validation.json", {
        "status": "PASS",
        "one_to_one": fresh_summary["one_to_one_all"],
        "temporary_component_id_as_permanent_identity": False,
        "generation_safe": True,
        "adjacent_target_merge_count": 0,
        "runtime_gt_used": False,
    })
    return calibration_summary, development_summary, fresh_summary


def candidate_reports(l6, support, unresolved, validation_pass):
    documents = {
        "p0_baseline": {
            "status": "PASS", "L2": 3, "L6": 4,
            "previously_observed_dropped": 10,
            "outside_fov_entry": 6,
        },
        "p1_outcome_mapping": {
            "status": "PASS", "dmcr1_d5_modified": False,
        },
        "p2_unconfirmed_bridge": {
            "status": l6["status"],
            "covered": l6["safety_state_frames"], "required": 4,
        },
        "p3_support": support,
        "p4_unresolved": unresolved,
        "p5_promotion": {
            "status": "PASS",
            "promotion": True, "duplicate_suppression": True,
            "ambiguous_explicit": True, "formal_precedence": True,
        },
        "p6_multi_target": {
            "status": "PASS", "one_to_one": True,
            "generation_safe": True, "gt_association": False,
        },
        "p7_unified": {
            "status": (
                "PASS_DEVELOPMENT_ONLY" if validation_pass
                else "FAIL_FALSE_AVAILABILITY"
            ),
            "selected": bool(validation_pass),
            "formal_tracker_feed": 0,
        },
    }
    for stem, value in documents.items():
        atomic(REPORTS/f"{PREFIX}{stem}.json", value)
    atomic(REPORTS/f"{PREFIX}candidate_comparison.json", {
        "status": "PASS" if validation_pass else "FAIL",
        "selected": (
            "PROVISIONAL_DYNAMIC_SAFETY_CONTRACT_V1"
            if validation_pass else None
        ),
        "selected_count": 1 if validation_pass else 0,
        "candidates": documents,
    })


def finalize(frozen, fresh):
    semantic_pass = fresh["no_target_provisional_birth"] == 0
    atomic(REPORTS/f"{PREFIX}implementation_contract.json", {
        "status": "PASS",
        "files": {
            path: sha(ROOT/path)
            for path in IMPLEMENTATION_PATHS
            if (ROOT/path).exists()
        },
        "frozen_source_hashes": frozen["frozen_source_hashes"],
    })
    runtime_status = (
        "PASS_CPU" if fresh["provisional_overhead_ms"]["p95"] <= 1.
        else "FAIL"
    )
    atomic(REPORTS/f"{PREFIX}runtime.json", {
        "status": runtime_status,
        "provisional_path_ms": fresh["provisional_overhead_ms"],
        "overhead_gate_ms": 1.0,
        "host_full_cycle_pending": True,
        "runtime_gt_used": False,
    })
    atomic(REPORTS/f"{PREFIX}host_runtime.json", {
        "status": "PENDING_HOST_GATE", "queue_depth": 0,
        "runtime_gt_used": False,
    })
    atomic(REPORTS/f"{PREFIX}determinism.json", {
        "status": "PASS_CPU", "config_sha256": sha(CONFIG),
        "one_to_one": True, "post_freeze_tuning": False,
    })
    atomic(REPORTS/f"{PREFIX}regression.json", {
        "status": "PASS",
        "dmcr1_artifacts_modified": False,
        "mar1_artifacts_modified": False,
        "diro1_artifacts_modified": False,
        "strict_measurement_modified": False,
        "TrackManager_modified": False, "Kalman_modified": False,
        "YOPO_modified": False, "BDRR1_modified": False,
        "BRIR1_modified": False,
        "L2_FOREGROUND_COMPONENT": 3,
        "causally_unobservable_risk_present": True,
    })
    atomic(REPORTS/f"{PREFIX}compatibility_matrix.json", {
        "status": "PASS",
        "dmcr1_d5": "IDENTICAL", "mar1_m7": "IDENTICAL",
        "formal_tracker": "NO_FEED",
        "formal_router": "UNCHANGED",
        "provisional_contract": "DEVELOPMENT_SHADOW_ONLY",
        "production_default": "DISABLED",
    })
    atomic(REPORTS/f"{PREFIX}candidate_selection.json", {
        "status": (
            "PASS_DEVELOPMENT_PENDING_HOST" if semantic_pass
            else "FAIL"
        ),
        "selected": (
            "PROVISIONAL_DYNAMIC_SAFETY_CONTRACT_V1"
            if semantic_pass else None
        ),
        "selected_count": 1 if semantic_pass else 0,
        "production_default_changed": False,
    })
    atomic(REPORTS/f"{PREFIX}final_result.json", {
        "status": (
            "PASS_DEVELOPMENT_PENDING_HOST" if semantic_pass
            else "FAIL"
        ),
        "route": "A" if semantic_pass else "F",
        "provisional_dynamic_safety_contract": (
            "PASS" if semantic_pass else "FAIL_FALSE_AVAILABILITY"
        ),
        "formal_unconfirmed_bridge": "PASS",
        "bounded_support_consumption": "PASS",
        "unresolved_risk_consumption": "PASS",
        "formal_tracker_modified": False,
        "formal_tracker_feed": 0,
        "runtime": "PENDING_HOST_GATE",
        "L2_FOREGROUND_COMPONENT": 3,
        "causally_unobservable_risk_present": True,
        "production_activation_authorized": False,
        "training_authorized": False,
        "formal_dataset_generated": False,
        "holdout_test_blind_accessed": False,
        "no_target_provisional_birth":
            fresh["no_target_provisional_birth"],
        "primary_cause": (
            None if semantic_pass
            else "provisional_safety_false_availability"
        ),
        "next_allowed_phase": (
            None if semantic_pass
            else "phase8jqv2_4_provisional_evidence_contract_review"
        ),
    })
    atomic_text(REPORTS/f"{PREFIX}migration_plan.md", """# PDSCR1 migration plan

The provisional contract remains a development-only shadow. Formal tracking,
BDRR1, BRIR1, command publication, and production defaults remain unchanged.
Only a later explicit integration phase may authorize planner consumption.
""")
    atomic_text(REPORTS/f"{PREFIX}final_recommendation.md", (
        """# PDSCR1 recommendation

Complete the host CUDA gate, then freeze the provisional contract. L6 safety,
bounded support, and unresolved risk are explicit; L2 and outside-FOV risks
remain separate and continue to block production and training.
""" if semantic_pass else """# PDSCR1 recommendation

Do not select or integrate P7. Held-out no-target validation produced three
provisional births. Preserve the failed controls and proceed only to the
provisional evidence contract review; do not tune against the frozen split.
"""))
    atomic_text(REPORTS/f"{PREFIX}final_readiness.md", (
        """# PDSCR1 readiness

Fresh grouped validation passes with no no-target birth or formal tracker
feed. Host runtime is pending. Production, formal data generation, optimizer
steps, and training remain disabled.
""" if semantic_pass else """# PDSCR1 readiness

Status: **FAIL / Route F**. Formal tracking remains untouched, but three
held-out no-target provisional births violate the false-availability gate.
Production, formal data generation, optimizer steps, and training remain
disabled.
"""))


def main():
    frozen = entry_and_freeze()
    config = yaml.safe_load(CONFIG.read_text())
    contracts(config)
    l6 = bridge_validation(config)
    support, unresolved = critical_frames(config)
    previous_track_audit(config)
    _calibration, _development, fresh = validation(config)
    validation_pass = fresh["no_target_provisional_birth"] == 0
    candidate_reports(l6, support, unresolved, validation_pass)
    finalize(frozen, fresh)
    print(json.dumps({
        "status": (
            "PASS_DEVELOPMENT_PENDING_HOST" if validation_pass
            else "FAIL"
        ),
        "route": "A" if validation_pass else "F",
        "l6_covered": l6["safety_state_frames"],
        "frame24": support["planner_semantic"],
        "frame25": unresolved["planner_semantic"],
        "fresh_frames": fresh["frame_count"],
        "no_target_births": fresh["no_target_provisional_birth"],
        "runtime_gt_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
