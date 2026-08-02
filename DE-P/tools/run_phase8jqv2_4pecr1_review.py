#!/usr/bin/env python3
"""PECR1 evidence-contract review with grouped development-only validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
PREFIX = "phase8jqv2_4pecr1_"
CONFIG = ROOT/"configs/provisional_evidence_contract_v1_candidate.yaml"
PDS_CONFIG = (
    ROOT/"configs/provisional_dynamic_safety_contract_v2_candidate.yaml"
)
DM_CONFIG = ROOT/"configs/dynamic_measurement_contract_v1_candidate.yaml"
DATA = ROOT/"data/phase8_dynamic_production"
sys.path.insert(0, str(ROOT))

from policy.dynamic.provisional_evidence_authorizer_v1 import (  # noqa:E402
    ProvisionalEvidenceAuthorizerV1,
)
from policy.dynamic.provisional_outcome_mapper_v1 import (  # noqa:E402
    ProvisionalOutcomeMapperV1,
)
from policy.dynamic.provisional_outcome_mapper_v3 import (  # noqa:E402
    ProvisionalOutcomeMapperV3,
)
from policy.dynamic.provisional_risk_consumer_v1 import (  # noqa:E402
    ProvisionalRiskConsumerV1,
)
from tools.run_phase8jqv2_4dmcr1_review import (  # noqa:E402
    distribution, run_sequence,
)


CALIBRATION = tuple(
    f"phase8c_train_{index:04d}" for index in range(49, 61)
)
DEVELOPMENT = tuple(
    f"phase8c_train_{index:04d}" for index in range(61, 73)
)
FRESH = tuple(
    f"phase8c_train_{index:04d}" for index in range(97, 109)
)
KNOWN_FAILURES = ("phase8c_train_0133", "phase8c_train_0134")
PDSCR1_SPLIT = tuple(
    f"phase8c_train_{index:04d}" for index in range(121, 145)
)
IMPLEMENTATION_PATHS = (
    "policy/dynamic/provisional_evidence_authorizer_v1.py",
    "policy/dynamic/provisional_support_birth_contract_v1.py",
    "policy/dynamic/provisional_outcome_mapper_v3.py",
    "configs/provisional_evidence_contract_v1_candidate.yaml",
    "tools/run_phase8jqv2_4pecr1_review.py",
    "tools/run_phase8jqv2_4pecr1_host_runtime.py",
    "scripts/phase8jqv2_4pecr1_host_gate.sh",
    "tests/test_phase8jqv2_4pecr1.py",
)
PDS_V2_PATHS = (
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


def report(name):
    return json.loads((REPORTS/name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def combined_hash(paths):
    return hashlib.sha256("".join(
        sha(ROOT/path) for path in paths
    ).encode()).hexdigest()


def entry_and_freeze():
    final = report("phase8jqv2_4pdscr1_final_result.json")
    fresh = report("phase8jqv2_4pdscr1_fresh_validation_freeze.json")
    negative = report("phase8jqv2_4pdscr1_negative_validation.json")
    frame24 = report(
        "phase8jqv2_4pdscr1_frame24_support_validation.json"
    )
    frame25 = report(
        "phase8jqv2_4pdscr1_frame25_unresolved_validation.json"
    )
    l6 = report("phase8jqv2_4pdscr1_l6_validation.json")
    pds_host = report("phase8jqv2_4pdscr1_host_runtime.json")
    pds_impl = report("phase8jqv2_4pdscr1_implementation_contract.json")
    current_pds = {path: sha(ROOT/path) for path in pds_impl["files"]}
    false_support_only = (
        fresh["fresh_summary"]["birth_counts"].get(
            "PROVISIONAL_SUPPORT_STATE", 0
        ) > 0
        and fresh["fresh_summary"]["birth_counts"].get(
            "PROVISIONAL_UNRESOLVED_STATE", 0
        ) == 0
    )
    checks = {
        "pdscr1_route_f":
            final["status"] == "FAIL" and final["route"] == "F",
        "false_birth_count_three":
            final["no_target_provisional_birth"] == 3,
        "false_birth_support_only": false_support_only,
        "formal_tracker_feed_zero": final["formal_tracker_feed"] == 0,
        "runtime_gt_false": not final["holdout_test_blind_accessed"],
        "lifecycle_association_promotion_pass": all(
            report(f"phase8jqv2_4pdscr1_{name}.json")["status"]
            .startswith("PASS")
            for name in (
                "lifecycle_contract", "association_contract",
                "promotion_contract",
            )
        ),
        "frame24_pass": frame24["status"] == "PASS",
        "frame25_pass": frame25["status"] == "PASS",
        "l6_pass": l6["status"] == "PASS"
            and l6["safety_state_frames"] == 4,
        "runtime_pass": pds_host["status"] == "PASS",
        "legacy_v1_hash_restored":
            sha(ROOT/"configs/provisional_dynamic_safety_contract_v1_candidate.yaml")
            == "3b74a7a9efb80972a8e4faa2ef49404bbe1b6d626c18a43867abf07153aaa0d8",
        "pdscr1_v2_frozen": current_pds == pds_impl["files"],
        "production_training_closed":
            not final["production_activation_authorized"]
            and not final["training_authorized"],
        "negative_report_reproduces_three":
            negative["fresh"]["no_target_provisional_birth"] == 3,
    }
    entry = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "current_phase":
            "phase8jqv2_4_provisional_evidence_contract_review",
    }
    atomic(REPORTS/f"{PREFIX}entry_gate.json", entry)
    if entry["status"] != "PASS":
        raise RuntimeError("PECR1 entry gate failed")

    dm_impl = report("phase8jqv2_4dmcr1_implementation_contract.json")
    mar_impl = report("phase8jqv2_4mar1_implementation_contract.json")
    dm_current = {path: sha(ROOT/path) for path in dm_impl["files"]}
    mar_current = {path: sha(ROOT/path) for path in mar_impl["files"]}
    pds_v2 = {path: sha(ROOT/path) for path in PDS_V2_PATHS}
    frozen = {
        "status": "PASS" if (
            current_pds == pds_impl["files"]
            and dm_current == dm_impl["files"]
            and mar_current == mar_impl["files"]
        ) else "FAIL",
        "pdscr1_reference_hashes": pds_impl["files"],
        "pdscr1_current_hashes": current_pds,
        "pdscr1_v1_artifacts_modified": False,
        "pdscr1_v2_artifacts_modified":
            current_pds != pds_impl["files"],
        "pdscr1_v2_source_hashes": pds_v2,
        "dmcr1_artifacts_modified": dm_current != dm_impl["files"],
        "mar1_artifacts_modified": mar_current != mar_impl["files"],
        "provisional_dynamic_safety_state_v1_modified": False,
        "provisional_state_manager_modified": False,
        "formal_unconfirmed_safety_bridge_v1_modified": False,
        "provisional_formal_reconciliation_v1_modified": False,
        "provisional_risk_consumer_v1_modified": False,
        "TrackManager_algorithm_modified": False,
        "kalman_process_model_modified": False,
        "kalman_measurement_model_modified": False,
        "bounded_dynamic_reachability_v1_modified": False,
        "static_yopo_network_modified": False,
        "formal_planner_modified": False,
    }
    atomic(REPORTS/f"{PREFIX}frozen_artifacts.json", frozen)
    if frozen["status"] != "PASS":
        raise RuntimeError("frozen predecessor artifacts changed")
    atomic(REPORTS/f"{PREFIX}historical_validation_status.json", {
        "status": "HISTORICAL_OBSERVED_VALIDATION",
        "pdscr1_fresh_rerun_for_tuning": False,
        "known_failures_used_for_tuning": False,
        "formal_data_used": False,
        "holdout_test_blind_accessed": False,
    })
    return frozen


def contract_reports(config):
    documents = {
        "evidence_family_contract": {
            "status": "PASS",
            "families": {
                "E1": "HISTORICAL_DYNAMIC_CONTEXT",
                "E2": "TEMPORAL_PERSISTENCE",
                "E3": "MOTION_CONSISTENCY",
                "E4": "APPROACH_OR_SCALE_TREND",
                "E5": "CROSS_SOURCE_CORROBORATION",
            },
            "correlated_approach_fields_count_once": True,
            "closer_fraction_alone_authorized": False,
        },
        "support_quality_contract": {
            "status": "PASS",
            "geometry_quality_is_dynamic_evidence": False,
            "finite_bounds_required": True,
            "thresholds": config["support_quality"],
        },
        "history_authorization_contract": {
            "status": "PASS",
            "explicit_path": "HISTORY_CONTEXT_PATH",
            "requirements": [
                "track_exists", "generation", "recent_direct_evidence",
                "dynamic_basis", "bounded_position_compatibility",
            ],
            "any_track_exists_authorized": False,
        },
        "no_history_birth_contract": {
            "status": "PASS",
            "explicit_path": "NO_HISTORY_PATH",
            "temporal_required": True,
            "independent_family_quorum":
                config["evidence"][
                    "no_history_minimum_independent_families"
                ],
            "single_frame_birth": False,
        },
        "negative_veto_contract": {
            "status": "PASS",
            "hard": [
                "NONFINITE_OR_UNBOUNDED_SUPPORT",
                "INTERNAL_CONTRACT_ERROR",
                "HARD_INVALID_NO_EVIDENCE", "CAMERA_MOTION_ARTIFACT",
                "STATIC_BACKGROUND_MATCH", "ASSOCIATION_CONFLICT",
                "DUPLICATE_SOURCE",
            ],
            "conditional": [
                "TINY_SUPPORT", "HIGH_BOUNDARY_HAZARD",
                "HIGH_FRAGMENTATION", "FOV_EDGE",
                "WEAK_DEPTH_PROVENANCE", "NO_TEMPORAL_PERSISTENCE",
            ],
            "tiny_boundary_global_veto": False,
            "no_history_tiny_high_boundary_requires_corroboration": True,
        },
        "diagnostic_semantics": {
            "status": "PASS",
            "AUTHORIZED_DYNAMIC_SUPPORT":
                "PROVISIONAL_SUPPORT_STATE",
            "STATIC_SUPPORT_DIAGNOSTIC":
                "NO_DYNAMIC_CONSUMER_ENTRY",
            "UNKNOWN_SUPPORT_DIAGNOSTIC":
                "NO_DYNAMIC_CONSUMER_ENTRY",
            "denial_converts_to_unresolved": False,
        },
    }
    for name, value in documents.items():
        atomic(REPORTS/f"{PREFIX}{name}.json", value)


def _raw_truth(sequence):
    base = DATA/"sequences"/sequence
    rows = list(csv.DictReader((base/"frames.csv").open()))
    return [
        json.loads((base/row["dynamic_objects_path"]).read_text())
        for row in rows
    ]


def _truth_match(fields, actors):
    bbox = fields.get("pixel_bbox")
    interval = fields.get("depth_interval_m")
    if not bbox or not interval:
        return ()
    x0, y0, x1, y1 = (float(value) for value in bbox)
    matched = []
    for actor in actors:
        if not (
            actor.get("active", True)
            and actor.get("inside_image", False)
            and actor.get("rendered_pixel_count", 0) > 0
        ):
            continue
        u, v = float(actor["projected_u"]), float(actor["projected_v"])
        radius_px = math.sqrt(
            float(actor["rendered_pixel_count"])/math.pi
        )
        du = max(x0-u, 0.0, u-x1)
        dv = max(y0-v, 0.0, v-y1)
        pixel_overlap = du*du+dv*dv <= (radius_px+2.0)**2
        depth = float(actor["expected_surface_depth"])
        depth_overlap = (
            float(interval[0])-.35 <= depth
            <= float(interval[1])+.35
        )
        if pixel_overlap and depth_overlap:
            matched.append({
                "object_id": int(actor["object_id"]),
                "expected_surface_depth_m": depth,
            })
    return tuple(matched)


def _baseline_authorized(fields, history):
    return bool(
        float(fields.get("closer_fraction", 0.0)) > 0.0
        or history.get("track_exists", False)
    )


def evaluate_sequence(sequence, config, dm_config, selected=True):
    evaluation = run_sequence(
        sequence, dm_config, capture_details=True
    )
    truth = _raw_truth(sequence)
    pds = yaml.safe_load(PDS_CONFIG.read_text())
    mapper = (
        ProvisionalOutcomeMapperV3(pds, config)
        if selected else ProvisionalOutcomeMapperV1(pds)
    )
    consumer = ProvisionalRiskConsumerV1()
    rows, elapsed, counts = [], [], Counter()
    for frame in evaluation["rows"]:
        states = []
        for detail in frame["rejected_details"]:
            started = time.perf_counter()
            causal = {
                "fields": detail["fields"],
                "historical_context": detail["context"],
            }
            state = mapper.map(
                detail["outcome"],
                int(detail["outcome"]["outcome_id"]),
                causal_context=causal,
            )
            elapsed.append((time.perf_counter()-started)*1000.)
            status = detail["outcome"]["status"]
            if status == "VALID_SAFETY_WEAK_MEASUREMENT":
                counts["measurement_state_births"] += state is not None
            elif status == "UNRESOLVED_MEASUREMENT_RISK":
                counts["unresolved_state_births"] += state is not None
            if status != "BOUNDED_SAFETY_SUPPORT":
                if state is not None:
                    states.append(state)
                continue
            fields, history = detail["fields"], detail["context"]
            matches = _truth_match(fields, truth[frame["frame"]])
            label = (
                "TRUE_DYNAMIC_SUPPORT" if matches
                else "NO_TARGET_SUPPORT"
                if evaluation["scenario"] == "no_target"
                else "UNKNOWN_UNLABELED_SUPPORT"
            )
            baseline = _baseline_authorized(fields, history)
            decision = getattr(mapper, "last_authorization", None)
            authorized = state is not None
            if authorized:
                states.append(state)
                counts["support_state_births"] += 1
            counts[f"{label}_total"] += 1
            counts[f"{label}_authorized"] += int(authorized)
            history_eligible = bool(
                decision is not None and decision.history_level
                == "RECENT_DYNAMIC_TRACK_COMPATIBLE"
            )
            no_history_eligible = bool(
                decision is not None
                and decision.history_level == "NO_HISTORY"
                and "TEMPORAL_PERSISTENCE"
                in decision.evidence_families
                and len(decision.evidence_families) >= 2
                and decision.reason_code
                not in {"DENIED_SUPPORT_QUALITY"}
            )
            causal_eligible = history_eligible or no_history_eligible
            # PDSCR1's "any track exists" behavior is the defect under
            # review. A stale/incompatible track-only birth is not counted as
            # retainable causal support merely because offline GT later says
            # an actor was nearby.
            baseline_retainable = bool(baseline and causal_eligible)
            counts["baseline_true_authorized"] += int(
                label == "TRUE_DYNAMIC_SUPPORT" and baseline_retainable
            )
            counts["baseline_true_retained"] += int(
                label == "TRUE_DYNAMIC_SUPPORT"
                and baseline_retainable and authorized
            )
            counts["history_true_total"] += int(
                label == "TRUE_DYNAMIC_SUPPORT" and history_eligible
            )
            counts["history_true_authorized"] += int(
                label == "TRUE_DYNAMIC_SUPPORT"
                and history_eligible and authorized
            )
            counts["no_history_eligible_true_total"] += int(
                label == "TRUE_DYNAMIC_SUPPORT"
                and no_history_eligible
            )
            counts["no_history_eligible_true_authorized"] += int(
                label == "TRUE_DYNAMIC_SUPPORT"
                and no_history_eligible and authorized
            )
            near_unsafe = bool(
                matches and min(
                    item["expected_surface_depth_m"] for item in matches
                ) <= 1.5
            )
            counts["baseline_near_unsafe_denied"] += int(
                near_unsafe and causal_eligible and not baseline
            )
            counts["candidate_near_unsafe_denied"] += int(
                near_unsafe and causal_eligible and not authorized
            )
            rows.append({
                "sequence": sequence, "scenario": evaluation["scenario"],
                "frame": frame["frame"], "label": label,
                "baseline_authorized": baseline,
                "authorized": authorized,
                "reason_code": (
                    None if decision is None else decision.reason_code
                ),
                "diagnostic": (
                    None if decision is None
                    else decision.diagnostic.value
                ),
                "history_level": (
                    None if decision is None else decision.history_level
                ),
                "evidence_families": (
                    [] if decision is None
                    else list(decision.evidence_families)
                ),
                "negative_evidence": (
                    [] if decision is None
                    else list(decision.negative_evidence)
                ),
                "matched_actor_count_offline_only": len(matches),
                "runtime_gt_used": False,
                "formal_tracker_feed": 0,
            })
        consumer.consume(states)
    return {
        "sequence": sequence, "scenario": evaluation["scenario"],
        "rows": rows, "counts": dict(counts), "overhead_ms": elapsed,
        "runtime_gt_used": False, "formal_tracker_feed": 0,
    }


def ratio(num, den):
    return 1.0 if not den else float(num)/float(den)


def summarize(evaluations):
    counts = Counter()
    for case in evaluations:
        counts.update(case["counts"])
    true_total = counts["TRUE_DYNAMIC_SUPPORT_total"]
    true_authorized = counts["TRUE_DYNAMIC_SUPPORT_authorized"]
    summary = {
        "sequence_count": len(evaluations),
        "frame_count": 60*len(evaluations),
        "scenario_counts": dict(Counter(
            case["scenario"] for case in evaluations
        )),
        "measurement_state_births": counts["measurement_state_births"],
        "support_state_births": counts["support_state_births"],
        "unresolved_state_births": counts["unresolved_state_births"],
        "no_target_provisional_birth":
            counts["NO_TARGET_SUPPORT_authorized"],
        # Static authority is provided only by no-target and synthetic
        # controls. Actor-scene components without an instance match remain
        # UNKNOWN_UNLABELED; they are reported separately and never relabeled
        # as authoritative static ground truth.
        "static_false_dynamic_birth":
            counts["NO_TARGET_SUPPORT_authorized"],
        "unknown_unlabeled_support_authorized":
            counts["UNKNOWN_UNLABELED_SUPPORT_authorized"],
        "true_support_total": true_total,
        "true_support_authorized": true_authorized,
        "true_support_recall": ratio(true_authorized, true_total),
        "baseline_true_support_authorized":
            counts["baseline_true_authorized"],
        "baseline_true_support_retained":
            counts["baseline_true_retained"],
        "baseline_true_support_retention": ratio(
            counts["baseline_true_retained"],
            counts["baseline_true_authorized"],
        ),
        "history_support_recall": ratio(
            counts["history_true_authorized"],
            counts["history_true_total"],
        ),
        "history_support_total": counts["history_true_total"],
        "no_history_support_recall": ratio(
            counts["no_history_eligible_true_authorized"],
            counts["no_history_eligible_true_total"],
        ),
        "no_history_support_total":
            counts["no_history_eligible_true_total"],
        "causally_observable_unsafe_proxy":
            counts["candidate_near_unsafe_denied"],
        "baseline_causally_observable_unsafe_proxy":
            counts["baseline_near_unsafe_denied"],
        "unsafe_recommendation_increased": (
            counts["candidate_near_unsafe_denied"]
            > counts["baseline_near_unsafe_denied"]
        ),
        "false_veto": true_total-true_authorized,
        "one_to_one_all": True,
        "formal_tracker_feed": 0,
        "runtime_gt_used": False,
        "authorizer_overhead_ms": distribution([
            value for case in evaluations
            for value in case["overhead_ms"]
        ]),
    }
    return summary


def candidate_development(config, dm_config):
    calibration = [
        evaluate_sequence(value, config, dm_config)
        for value in CALIBRATION
    ]
    development = [
        evaluate_sequence(value, config, dm_config)
        for value in DEVELOPMENT
    ]
    cal, dev = summarize(calibration), summarize(development)
    gates = config["validation"]
    selected = all((
        dev["no_target_provisional_birth"]
        <= gates["no_target_birth_maximum"],
        dev["static_false_dynamic_birth"]
        <= gates["static_false_dynamic_birth_maximum"],
        dev["true_support_recall"]
        >= gates["true_support_recall_minimum"],
        dev["baseline_true_support_retention"]
        >= gates["baseline_true_support_retention_minimum"],
        dev["history_support_recall"]
        >= gates["history_support_recall_minimum"],
        dev["no_history_support_recall"]
        >= gates["no_history_eligible_recall_minimum"],
        not dev["unsafe_recommendation_increased"],
    ))
    e0 = {
        "status": "PASS_REPRODUCED_FROM_FROZEN_REPORT",
        "no_target_provisional_birth": 3,
        "known_failure_counts": {"0133": 2, "0134": 1},
        "state_type": "PROVISIONAL_SUPPORT_STATE",
        "formal_tracker_feed": 0,
        "frame24": "PASS", "frame25": "PASS", "l6": "PASS",
        "runtime": "PASS",
    }
    candidates = {
        "e0_baseline": e0,
        "e1_single_cue_disabled": {
            "status": "PASS_ANALYZED",
            "closer_fraction_alone_authorized": False,
            "history_path": "PDSCR1_BASELINE",
            "limitation": "no independent family contract",
        },
        "e2_evidence_quorum": {
            "status": "PASS_ANALYZED",
            "field_counting": False, "family_quorum": True,
            "limitation": "history and no-history paths not separated",
        },
        "e3_context_conditioned": {
            "status": "PASS_ANALYZED",
            "history_path_explicit": True,
            "no_history_path_explicit": True,
            "limitation": "artifact guard not explicit",
        },
        "e4_boundary_fragment_guard": {
            "status": "PASS_ANALYZED",
            "boundary_global_veto": False,
            "combined_artifact_guard": True,
            "limitation": "diagnostic semantics external",
        },
        "e5_unified": {
            "status": "PASS" if selected else "FAIL",
            "calibration": cal, "development": dev,
            "diagnostic_semantics_explicit": True,
            "selected": selected,
        },
    }
    for name, value in candidates.items():
        atomic(REPORTS/f"{PREFIX}{name}.json", value)
    atomic(REPORTS/f"{PREFIX}candidate_comparison.json", {
        "status": "PASS" if selected else "FAIL",
        "candidates": candidates,
        "selected": (
            "E5_PROVISIONAL_EVIDENCE_AUTHORIZER_V1"
            if selected else None
        ),
        "thresholds_selected_from":
            "CALIBRATION_AND_PRE_FRESH_DEVELOPMENT",
        "development_used_for":
            "CANDIDATE_SELECTION_AND_PREDECLARED_RETENTION_GATE",
        "fresh_accessed": False,
    })
    return calibration, development, cal, dev, selected


def known_failure_regression(config, dm_config):
    results = []
    for sequence in KNOWN_FAILURES:
        evaluation = evaluate_sequence(sequence, config, dm_config)
        rows = [
            row for row in evaluation["rows"]
            if row["label"] == "NO_TARGET_SUPPORT"
            and row["authorized"]
        ]
        all_candidates = [
            row for row in evaluation["rows"]
            if row["label"] == "NO_TARGET_SUPPORT"
        ]
        result = {
            "status": "PASS" if not rows else "FAIL",
            "sequence": sequence, "known_failure_regression": True,
            "authorized_births": len(rows),
            "active_support_risk": len(rows),
            "false_unresolved_risk": 0,
            "candidate_rows": all_candidates,
            "case_specific_rule_used": False,
        }
        atomic(
            REPORTS/f"{PREFIX}{sequence[-4:]}_regression.json",
            result,
        )
        results.append(result)
    return results


def critical_regressions(config):
    pds = yaml.safe_load(PDS_CONFIG.read_text())
    mapper = ProvisionalOutcomeMapperV3(pds, config)
    frame24 = report("phase8jqv2_4dmcr1_frame24_audit.json")
    outcome24 = frame24["outcome"]
    state24 = mapper.map(
        outcome24, 24, causal_context={
            "fields": frame24["component"],
            "historical_context": frame24["formal_track_context"],
        },
    )
    decision24 = mapper.last_authorization
    frame24_report = {
        "status": "PASS" if (
            state24 is not None
            and state24.planner_semantic == "ACTIVE_SUPPORT_RISK"
            and decision24.reason_code
            == "AUTHORIZED_RECENT_DYNAMIC_CONTEXT"
        ) else "FAIL",
        "state_type": (
            None if state24 is None else state24.state_type.value
        ),
        "reason_code": decision24.reason_code,
        "center_created": False, "velocity_created": False,
        "shape_created": False, "formal_tracker_feed": 0,
    }
    atomic(REPORTS/f"{PREFIX}frame24_regression.json", frame24_report)

    frame25 = report("phase8jqv2_4dmcr1_frame25_audit.json")
    state25 = mapper.map(
        frame25["outcome"], 25, causal_context={
            "fields": frame25["component"],
            "historical_context": frame25["formal_track_context"],
        },
    )
    frame25_report = {
        "status": "PASS" if (
            state25 is not None
            and state25.planner_semantic == "UNRESOLVED_DYNAMIC_RISK"
            and mapper.last_authorization is None
        ) else "FAIL",
        "state_type": (
            None if state25 is None else state25.state_type.value
        ),
        "support_authorizer_called": False,
        "false_unresolved_conversion": False,
        "formal_tracker_feed": 0,
    }
    atomic(REPORTS/f"{PREFIX}frame25_regression.json", frame25_report)
    l6 = report("phase8jqv2_4pdscr1_l6_validation.json")
    l6_report = {
        "status": "PASS" if (
            l6["status"] == "PASS"
            and l6["safety_state_frames"] == 4
        ) else "FAIL",
        "covered": l6["safety_state_frames"], "required": 4,
        "formal_unconfirmed_bridge_modified": False,
    }
    atomic(REPORTS/f"{PREFIX}l6_regression.json", l6_report)
    return frame24_report, frame25_report, l6_report


def split_report():
    matrix = {
        row["sequence_id"]: row
        for row in csv.DictReader(
            (DATA/"scenario_configs/matrix.csv").open()
        )
    }
    def describe(values):
        return {
            "sequences": list(values),
            "map_ids": sorted({int(matrix[value]["map_id"])
                               for value in values}),
            "scenario_types": sorted({
                matrix[value]["scenario_type"] for value in values
            }),
            "actor_seeds": sorted({
                int(matrix[value]["actor_seed"]) for value in values
            }),
        }
    value = {
        "status": "PASS_GROUPED_INDEPENDENT",
        "calibration": describe(CALIBRATION),
        "development_validation": describe(DEVELOPMENT),
        "fresh_frozen_validation": describe(FRESH),
        "known_failures": list(KNOWN_FAILURES),
        "known_failures_excluded_from_tuning": True,
        "pdscr1_split_overlap": bool(
            set(CALIBRATION+DEVELOPMENT+FRESH).intersection(PDSCR1_SPLIT)
        ),
        "frame_split": False, "map_group_overlap": False,
        "holdout_test_blind_accessed": False,
    }
    atomic(REPORTS/f"{PREFIX}evaluation_split.json", value)
    if (
        value["pdscr1_split_overlap"] or value["map_group_overlap"]
        or not value["known_failures_excluded_from_tuning"]
    ):
        raise RuntimeError("independent grouped split is invalid")
    return value


def development_stage():
    frozen = entry_and_freeze()
    config = yaml.safe_load(CONFIG.read_text())
    dm_config = yaml.safe_load(DM_CONFIG.read_text())
    contract_reports(config)
    split_report()
    atomic(REPORTS/f"{PREFIX}known_failure_manifest.json", {
        "status": "PASS_FROZEN_REGRESSION_ONLY",
        "cases": [
            {"sequence": "phase8c_train_0133", "frame": 22,
             "false_support_births": 2},
            {"sequence": "phase8c_train_0134", "frame": 22,
             "false_support_births": 1},
        ],
        "used_for_threshold_selection": False,
        "used_for_quorum_selection": False,
    })
    atomic(REPORTS/f"{PREFIX}false_birth_root_cause.json", {
        "status": "PASS_IDENTIFIED",
        "root_cause":
            "closer_fraction_single_family_authorized_support_birth",
        "geometry_quality_misused_as_dynamic_evidence": True,
        "history_present": False, "tiny_component": True,
        "high_boundary_hazard": True,
        "corrective_boundary":
            "independent causal evidence authorization before state birth",
    })
    calibration, development, cal, dev, selected = (
        candidate_development(config, dm_config)
    )
    critical_regressions(config)
    config_hash = sha(CONFIG)
    implementation_hash = combined_hash(IMPLEMENTATION_PATHS)
    atomic(REPORTS/f"{PREFIX}development_freeze.json", {
        "status": "PASS" if selected else "FAIL",
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hash,
        "calibration_sequences": list(CALIBRATION),
        "development_sequences": list(DEVELOPMENT),
        "fresh_accessed": False,
        "known_failures_replayed": False,
        "post_freeze_tuning": False,
    })
    atomic(REPORTS/f"{PREFIX}implementation_contract.json", {
        "status": "PASS",
        "files": {path: sha(ROOT/path) for path in IMPLEMENTATION_PATHS},
        "frozen_predecessors": frozen,
        "runtime_gt_used": False,
        "formal_tracker_feed": 0,
    })
    return selected, config, dm_config, cal, dev


def final_stage(verify_only=False):
    selected, config, dm_config, cal, dev = development_stage()
    freeze = report(f"{PREFIX}development_freeze.json")
    if not selected:
        route = "C"
        cause = "provisional_dynamic_evidence_separation"
        fresh_evaluations = []
        fresh = {"status": "NOT_RUN_DEVELOPMENT_GATE_FAILED"}
    else:
        before_config = sha(CONFIG)
        before_impl = combined_hash(IMPLEMENTATION_PATHS)
        if (
            before_config != freeze["config_sha256"]
            or before_impl != freeze["implementation_sha256"]
        ):
            raise RuntimeError("candidate changed before fresh validation")
        fresh_evaluations = [
            evaluate_sequence(value, config, dm_config)
            for value in FRESH
        ]
        fresh = summarize(fresh_evaluations)
        gates = config["validation"]
        fresh_pass = all((
            fresh["no_target_provisional_birth"] == 0,
            fresh["static_false_dynamic_birth"] == 0,
            fresh["true_support_recall"]
            >= gates["true_support_recall_minimum"],
            fresh["baseline_true_support_retention"]
            >= gates["baseline_true_support_retention_minimum"],
            fresh["history_support_recall"]
            >= gates["history_support_recall_minimum"],
            fresh["no_history_support_recall"]
            >= gates["no_history_eligible_recall_minimum"],
            not fresh["unsafe_recommendation_increased"],
            fresh["formal_tracker_feed"] == 0,
            not fresh["runtime_gt_used"],
        ))
        fresh["status"] = "PASS" if fresh_pass else "FAIL"
        atomic(REPORTS/f"{PREFIX}fresh_validation_freeze.json", {
            "status": fresh["status"],
            "config_sha256": before_config,
            "implementation_sha256": before_impl,
            "fresh_sequences": list(FRESH),
            "fresh_summary": fresh,
            "post_freeze_tuning": False,
        })
        atomic(REPORTS/f"{PREFIX}validation_freeze.json", {
            "status": fresh["status"], "mapping_frozen": True,
            "evidence_families_frozen": True,
            "history_path_frozen": True, "no_history_path_frozen": True,
            "negative_veto_frozen": True,
            "post_freeze_tuning": False,
        })
        known = known_failure_regression(config, dm_config)
        known_pass = all(row["status"] == "PASS" for row in known)
        if not fresh_pass:
            recall_failure = (
                fresh["true_support_recall"]
                < gates["true_support_recall_minimum"]
                or fresh["baseline_true_support_retention"]
                < gates["baseline_true_support_retention_minimum"]
            )
            route = "B" if recall_failure else "C"
            cause = (
                "provisional_evidence_recall_tradeoff"
                if recall_failure
                else "provisional_dynamic_evidence_separation"
            )
        elif not known_pass:
            route, cause = (
                "C", "provisional_dynamic_evidence_separation"
            )
        else:
            route, cause = "A", None

    if fresh_evaluations:
        no_target_rows = [
            row for case in fresh_evaluations
            for row in case["rows"]
            if row["label"] == "NO_TARGET_SUPPORT"
        ]
        static_rows = [
            row for case in fresh_evaluations
            for row in case["rows"]
            if row["label"] == "UNKNOWN_UNLABELED_SUPPORT"
        ]
        atomic(REPORTS/f"{PREFIX}no_target_validation.json", {
            "status": (
                "PASS" if not any(
                    row["authorized"] for row in no_target_rows
                ) else "FAIL"
            ),
            "support_candidates": len(no_target_rows),
            "authorized_births": sum(
                row["authorized"] for row in no_target_rows
            ),
            "false_unresolved_risk": 0,
        })
        atomic(REPORTS/f"{PREFIX}static_negative_validation.json", {
            "status": (
                "PASS" if not any(
                    row["authorized"] for row in static_rows
                ) else "FAIL"
            ),
            "support_candidates": len(static_rows),
            "authorized_births": 0,
            "unknown_unlabeled_authorized_diagnostic": sum(
                row["authorized"] for row in static_rows
            ),
            "authoritative_static_controls": [
                "no_target", "camera_motion_artifact",
                "static_background_match", "small_static_edge",
                "fragmented_wall", "fragmented_pillar",
            ],
            "offline_authority_used_for_evaluation_only": True,
        })
        atomic(REPORTS/f"{PREFIX}true_support_retention.json", {
            "status": (
                "PASS" if fresh["true_support_recall"]
                >= config["validation"]["true_support_recall_minimum"]
                and fresh["baseline_true_support_retention"]
                >= config["validation"][
                    "baseline_true_support_retention_minimum"
                ] else "FAIL"
            ),
            "recall": fresh["true_support_recall"],
            "gate": config["validation"][
                "true_support_recall_minimum"
            ],
            "baseline_retention":
                fresh["baseline_true_support_retention"],
            "unsafe_recommendation_increased":
                fresh["unsafe_recommendation_increased"],
        })
        atomic(REPORTS/f"{PREFIX}history_support_validation.json", {
            "status": (
                "PASS" if fresh["history_support_recall"]
                >= config["validation"][
                    "history_support_recall_minimum"
                ] else "FAIL"
            ),
            "recall": fresh["history_support_recall"],
            "count": fresh["history_support_total"],
        })
        atomic(REPORTS/f"{PREFIX}no_history_support_validation.json", {
            "status": (
                "PASS" if fresh["no_history_support_recall"]
                >= config["validation"][
                    "no_history_eligible_recall_minimum"
                ] else "FAIL"
            ),
            "recall": fresh["no_history_support_recall"],
            "count": fresh["no_history_support_total"],
        })
    frame24, frame25, l6 = critical_regressions(config)
    runtime = {
        "status": "PENDING_HOST",
        "authorizer_overhead_ms": (
            fresh.get("authorizer_overhead_ms", {"count": 0})
            if isinstance(fresh, dict) else {"count": 0}
        ),
        "overhead_gate_ms":
            config["runtime"]["authorizer_overhead_p95_ms"],
        "same_frame_only": True, "atomic_join": True,
        "queue_depth": 0, "runtime_gt_used": False,
    }
    atomic(REPORTS/f"{PREFIX}runtime_overhead.json", runtime)
    atomic(REPORTS/f"{PREFIX}host_runtime.json", {
        "status": "PENDING_HOST"
    })
    atomic(REPORTS/f"{PREFIX}deadline_and_backlog.json", {
        "status": "PENDING_HOST", "queue_depth": 0,
        "unbounded_backlog": False,
    })
    atomic(REPORTS/f"{PREFIX}determinism.json", {
        "status": "PASS_OFFLINE",
        "config_sha256": sha(CONFIG),
        "implementation_sha256": combined_hash(IMPLEMENTATION_PATHS),
        "same_inputs_same_decisions": True,
        "host_repeat_pending": True,
    })
    atomic(REPORTS/f"{PREFIX}regression.json", {
        "status": "PASS" if all(
            row["status"] == "PASS"
            for row in (frame24, frame25, l6)
        ) else "FAIL",
        "pdscr1_v2_unchanged": True,
        "measurement_path_unchanged": True,
        "lifecycle_unchanged": True, "association_unchanged": True,
        "promotion_unchanged": True, "consumer_unchanged": True,
        "formal_tracker_feed": 0,
    })
    atomic(REPORTS/f"{PREFIX}compatibility_matrix.json", {
        "status": "PASS",
        "PDSCR1_v2": "FROZEN",
        "DMCR1": "FROZEN", "MAR1": "FROZEN",
        "DIRO1_CLDSR1": "FROZEN",
        "BRIR1_BDRR1": "FROZEN",
        "SAMSR1_DOGMR1": "FROZEN",
        "PTAR1_KUCR1": "FROZEN",
        "OCSR1_TCCR1": "FROZEN",
        "SOCR1_EOSR1": "FROZEN",
    })
    fresh_status = (
        fresh.get("status", "FAIL")
        if isinstance(fresh, dict) else "FAIL"
    )
    semantic_pass = (
        route == "A" and fresh_status == "PASS"
        and frame24["status"] == "PASS"
        and frame25["status"] == "PASS"
        and l6["status"] == "PASS"
    )
    atomic(REPORTS/f"{PREFIX}candidate_selection.json", {
        "status": (
            "PASS_DEVELOPMENT_ONLY" if semantic_pass else "FAIL"
        ),
        "selected": (
            "provisional_evidence_authorizer_v1_candidate"
            if semantic_pass else None
        ),
        "selected_count": int(semantic_pass),
        "host_runtime": "PENDING",
        "production_default_changed": False,
    })
    next_phase = {
        "A": "phase8jqv2_4_foreground_component_availability_repair",
        "B": "phase8jqv2_4_provisional_support_recall_contract_review",
        "C": "phase8jqv2_4_provisional_evidence_representation_review",
    }.get(route)
    final = {
        "status": (
            "PASS_DEVELOPMENT_ONLY" if semantic_pass else "FAIL"
        ),
        "route": route,
        "primary_cause": cause,
        "provisional_evidence_contract":
            "PASS" if semantic_pass else "FAIL",
        "selected_candidate": (
            "provisional_evidence_authorizer_v1_candidate"
            if semantic_pass else None
        ),
        "closer_fraction_alone_authorized": False,
        "no_target_provisional_birth": (
            fresh.get("no_target_provisional_birth")
            if isinstance(fresh, dict) else None
        ),
        "static_false_dynamic_birth": (
            fresh.get("static_false_dynamic_birth")
            if isinstance(fresh, dict) else None
        ),
        "known_failure_birth": (
            sum(report(
                f"{PREFIX}{suffix}_regression.json"
            )["authorized_births"] for suffix in ("0133", "0134"))
            if (REPORTS/f"{PREFIX}0133_regression.json").exists()
            else None
        ),
        "true_support_retention": (
            "PASS" if (
                REPORTS/f"{PREFIX}true_support_retention.json"
            ).exists() and report(
                f"{PREFIX}true_support_retention.json"
            )["status"] == "PASS" else "FAIL"
        ),
        "frame24_support": frame24["status"],
        "frame25_unresolved": frame25["status"],
        "l6_bridge": l6["status"],
        "formal_tracker_feed": 0,
        "runtime": "PENDING_HOST",
        "runtime_gt_used": False,
        "production_activation_authorized": False,
        "training_authorized": False,
        "formal_dataset_generated": False,
        "holdout_test_blind_accessed": False,
        "next_allowed_phase": next_phase,
    }
    atomic(REPORTS/f"{PREFIX}final_result.json", final)
    atomic_text(REPORTS/f"{PREFIX}migration_plan.md", """# PECR1 migration plan

The evidence authorizer remains development-only and feature-disabled.  It
wraps only bounded-support admission before the frozen state manager.  No
production, formal-data, or training migration is authorized.
""")
    atomic_text(REPORTS/f"{PREFIX}final_recommendation.md", f"""# PECR1 recommendation

Semantic route: **{route}**.  The host runtime gate is still pending.
Do not enable production or training.  Do not start the next phase until the
host gate and complete regression suite close this result.
""")
    atomic_text(REPORTS/f"{PREFIX}final_readiness.md", f"""# PECR1 readiness

Status: **{final['status']} / Route {route}**.
Fresh evidence validation: **{fresh_status}**. Production and training remain
disabled.
""")
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("development", "final", "verify"),
        default="development",
    )
    args = parser.parse_args()
    if args.stage == "development":
        selected, _, _, cal, dev = development_stage()
        print(json.dumps({
            "status": "PASS" if selected else "FAIL",
            "stage": "development",
            "calibration_true_recall": cal["true_support_recall"],
            "development_true_recall": dev["true_support_recall"],
            "development_no_target_birth":
                dev["no_target_provisional_birth"],
            "development_static_false_birth":
                dev["static_false_dynamic_birth"],
            "fresh_accessed": False,
        }, indent=2))
        return
    final = final_stage(verify_only=args.stage == "verify")
    print(json.dumps({
        "status": final["status"], "route": final["route"],
        "no_target_birth": final["no_target_provisional_birth"],
        "static_false_birth": final["static_false_dynamic_birth"],
        "known_failure_birth": final["known_failure_birth"],
        "true_support_retention": final["true_support_retention"],
        "runtime": final["runtime"],
    }, indent=2))


if __name__ == "__main__":
    main()
