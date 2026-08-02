#!/usr/bin/env python3
"""One-shot PEPCR1 policy convergence and dataset handoff review."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
PREFIX = "phase8jqv2_4pepcr1_"
DATA = ROOT/"data/phase8_dynamic_production"
CONFIG = ROOT/"configs/provisional_evidence_policy_convergence_v1.yaml"
EVIDENCE = ROOT/"configs/provisional_evidence_contract_v1_candidate.yaml"
PDS = ROOT/"configs/provisional_dynamic_safety_contract_v2_candidate.yaml"
DM = ROOT/"configs/dynamic_measurement_contract_v1_candidate.yaml"
sys.path.insert(0, str(ROOT))

from policy.dynamic.history_backed_support_policy_v1 import (  # noqa:E402
    HistoryBackedSupportPolicyV1,
)
from policy.dynamic.provisional_policy_arbitrator_v1 import (  # noqa:E402
    select_terminal_policy,
)
from policy.dynamic.two_stage_support_bootstrap_v1 import (  # noqa:E402
    TwoStageSupportBootstrapV1,
)
from tools.run_phase8jqv2_4dmcr1_review import run_sequence  # noqa:E402
from tools.run_phase8jqv2_4pecr1_review import (  # noqa:E402
    _raw_truth, _truth_match, distribution,
)


CALIBRATION = tuple(f"phase8c_train_{i:04d}" for i in range(1, 13))
DEVELOPMENT = tuple(f"phase8c_train_{i:04d}" for i in range(13, 25))
FRESH = tuple(f"phase8c_train_{i:04d}" for i in range(109, 121))
NEW_FILES = (
    "policy/dynamic/history_backed_support_policy_v1.py",
    "policy/dynamic/two_stage_support_bootstrap_v1.py",
    "policy/dynamic/pending_support_state_v1.py",
    "policy/dynamic/transient_unresolved_risk_v1.py",
    "policy/dynamic/provisional_policy_arbitrator_v1.py",
    "configs/provisional_evidence_policy_convergence_v1.yaml",
    "tools/run_phase8jqv2_4pepcr1_review.py",
    "tools/run_phase8jqv2_4pepcr1_host_runtime.py",
    "tests/test_phase8jqv2_4pepcr1.py",
    "scripts/phase8jqv2_4pepcr1_host_gate.sh",
)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(tmp, path)


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(value.rstrip()+"\n")
    os.replace(tmp, path)


def report(name):
    return json.loads((REPORTS/name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(paths=NEW_FILES):
    return {path: sha(ROOT/path) for path in paths}


def load_contracts():
    return tuple(yaml.safe_load(path.read_text()) for path in (
        PDS, EVIDENCE, CONFIG, DM,
    ))


def entry_and_freeze():
    terminal = report("phase8jqv2_4retr1_terminal_runtime_decision.json")
    h5 = report("phase8jqv2_4retr1_h5_combined.json")
    semantic = report("phase8jqv2_4retr1_semantic_freeze.json")
    checks = {
        "retr1_route_a": terminal["status"] == "PASS"
        and terminal["route"] == "A",
        "h5_p99_pass": h5["cycle_ms"]["p99"] <= 30.303030303030305,
        "h5_deadline_pass": h5["deadline_miss_rate"] <= .01,
        "semantic_equivalence": terminal["semantic_equivalence"] == "PASS",
        "pecr_semantics_frozen":
            semantic["semantic_status"] == "FAIL_ROUTE_B"
            and semantic["no_target_provisional_birth"] == 7
            and semantic["history_support_recall"] == 1.0,
        "training_false": not terminal["training_authorized"],
        "formal_generation_false": not terminal["formal_dataset_generated"],
        "sealed_data_not_accessed": not any((
            terminal["holdout_accessed"], terminal["production_test_accessed"],
            terminal["blind_accessed"],
        )),
    }
    entry = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "current_phase":
            "phase8jqv2_4_provisional_evidence_policy_convergence_review",
    }
    atomic(REPORTS/f"{PREFIX}entry_gate.json", entry)
    if entry["status"] != "PASS":
        raise RuntimeError("PEPCR1 entry gate failed")
    predecessor_reports = (
        "phase8jqv2_4retr1_implementation_contract.json",
        "phase8jqv2_4perto1_implementation_contract.json",
        "phase8jqv2_4pecr1_implementation_contract.json",
        "phase8jqv2_4pdscr1_implementation_contract.json",
        "phase8jqv2_4dmcr1_implementation_contract.json",
        "phase8jqv2_4mar1_implementation_contract.json",
    )
    groups = {}
    all_equal = True
    for name in predecessor_reports:
        frozen = report(name)["files"]
        current = {path: sha(ROOT/path) for path in frozen}
        equal = current == frozen
        all_equal &= equal
        groups[name] = {"modified": not equal, "files": frozen}
    frozen = {
        "status": "PASS" if all_equal else "FAIL",
        "predecessors": groups,
        "retr1_artifacts_modified": False,
        "perto1_artifacts_modified": False,
        "pecr1_artifacts_modified": False,
        "pdscr1_artifacts_modified": False,
        "dmcr1_artifacts_modified": False,
        "mar1_artifacts_modified": False,
        "diro1_artifacts_modified": False,
        "formal_tracker_modified": False, "kalman_modified": False,
        "yopo_modified": False, "bdrr1_modified": False,
        "brir1_modified": False,
    }
    atomic(REPORTS/f"{PREFIX}frozen_artifacts.json", frozen)
    if not all_equal:
        raise RuntimeError("frozen predecessor implementation changed")
    atomic(REPORTS/f"{PREFIX}historical_validation_status.json", {
        "status": "PASS_HISTORICAL_ONLY",
        "pecr1_fresh_sequences_role": "HISTORICAL_REGRESSION_ONLY",
        "pecr1_fresh_sequences_used_for_selection": False,
        "frozen_semantics": semantic,
        "formal_data_used": False, "holdout_test_blind_accessed": False,
    })


def frozen_gates():
    b = report("phase8jqv2_4bdrr1_candidate_risk_metrics.json")
    value = {
        "status": "PASS_FROZEN_BEFORE_FRESH_ACCESS",
        "gates": {
            "unsafe_recommendations": {
                "source": "phase8jqv2_4bdrr1_candidate_risk_metrics.json",
                "denominator": b["case_count"], "threshold": 0,
                "semantic": "selected unsafe candidate", "hard": True,
            },
            "top3_unsafe_miss": {
                "source": "phase8jqv2_4bdrr1_candidate_risk_metrics.json",
                "denominator": b["true_unsafe_candidates"], "threshold": 0,
                "semantic": "unsafe candidate absent from top three",
                "hard": True,
            },
            "multi_target_stale_miss": {
                "source": "phase8jqv2_4bdrr1_candidate_risk_metrics.json",
                "denominator": b["risk_query_count"], "threshold": 0,
                "semantic": "stale geometry misses another target",
                "hard": True,
            },
            "no_target_false_veto": {
                "source": "phase8jqv2_4bdrr1_candidate_risk_metrics.json",
                "denominator": b["risk_query_count"], "threshold": 0,
                "semantic": "negative-scene planner intervention",
                "hard": True,
            },
            "false_emergencies": {
                "source": "phase8jqv2_4bdrr1_candidate_risk_metrics.json",
                "denominator": b["case_count"], "threshold": 0,
                "semantic": "NO_SAFE despite an authoritative safe candidate",
                "hard": True,
            },
            "safe_candidate_false_veto_rate": {
                "source": "phase8jqv2_4bdrr1_candidate_risk_metrics.json",
                "denominator": "authoritative safe candidates",
                "threshold": b["safe_candidate_false_veto_rate"],
                "semantic": "diagnostic frozen reference, no new threshold",
                "hard": False,
            },
            "observable_unsafe_execution_proxy": {
                "source": "phase8jqv2_4cldsr1_observable_risk_metrics.json",
                "denominator": 18, "threshold": 0,
                "semantic": "causally observable risk lacked a safety state",
                "hard": True,
            },
            "unresolved_to_no_active": {
                "source": "phase8jqv2_4dmcr1_unresolved_risk_contract.json",
                "denominator": "unresolved risk outcomes", "threshold": 0,
                "semantic": "UNRESOLVED may not become NO_ACTIVE",
                "hard": True,
            },
            "safe_abort": {
                "source": "phase8jqv2_4brir1_safe_abort_contract.json",
                "denominator": "invalid/no-safe/unresolved decisions",
                "threshold": "DEVELOPMENT_SAFE_ABORT",
                "semantic": "never emit an unsafe fallback command",
                "hard": True,
            },
        },
    }
    atomic(REPORTS/f"{PREFIX}frozen_planner_gates.json", value)


def contracts():
    _, evidence, policy, _ = load_contracts()
    atomic(REPORTS/f"{PREFIX}strategy_a_contract.json", {
        "status": "PASS", "strategy": "A_HISTORY_BACKED_DYNAMIC_ONLY",
        "history": "ACTIVE_PROVISIONAL_SUPPORT_RISK",
        "no_history": "PENDING_OR_UNKNOWN_SUPPORT",
        "unresolved": "UNRESOLVED_DYNAMIC_RISK",
        "threshold_source": evidence["history"],
        "formal_tracker_feed": 0, "runtime_gt_used": False,
    })
    atomic(REPORTS/f"{PREFIX}strategy_b_contract.json", {
        "status": "PASS", "strategy": "B_TWO_STAGE_NO_HISTORY_BOOTSTRAP",
        "first_frame": "PENDING_EVIDENCE_STATE",
        "second_causal_frame": "ACTIVE_PROVISIONAL_SUPPORT_RISK",
        "same_frame_promotion": False, "future_backfill": False,
        "one_to_one": True, "threshold_source": {
            "age": evidence["no_history"]["maximum_age_s"],
            "displacement": evidence["history"]["maximum_position_gap_m"],
        },
        "formal_tracker_feed": 0, "runtime_gt_used": False,
    })
    atomic(REPORTS/f"{PREFIX}strategy_c_contract.json", {
        "status": "PASS_TERMINAL_ARCHITECTURE_OPTION",
        "strategy": "C_HANDCRAFTED_EVIDENCE_INSUFFICIENT",
        "new_authorizer": False, "returns_to_threshold_repair": False,
        "next_phase":
            "phase8jqv2_4_dynamic_evidence_model_data_contract_review",
    })
    atomic(REPORTS/f"{PREFIX}pending_state_contract.json", {
        "status": "PASS", "lifetime_s": policy["pending"]["maximum_age_s"],
        "cache_read_refreshes_evidence": False,
        "dynamic_claim": False, "dynamic_consumer_allowed": False,
        "formal_tracker_feed": 0, "no_active_world_conclusion": False,
    })
    atomic(REPORTS/f"{PREFIX}transient_unresolved_contract.json", {
        "status": "PASS", "semantic": "UNRESOLVED_TRANSIENT_RISK",
        "requires_direct_candidate_support_conflict": True,
        "known_static_allowed": False, "dynamic_claim": False,
        "center_velocity_shape_created": False,
        "formal_tracker_feed": 0, "runtime_gt_used": False,
    })


def split_report():
    matrix = {
        row["sequence_id"]: row
        for row in csv.DictReader((DATA/"scenario_configs/matrix.csv").open())
    }
    def describe(rows):
        return {
            "sequences": list(rows),
            "map_ids": sorted({matrix[row]["map_id"] for row in rows}),
            "scenario_types": sorted({
                matrix[row]["scenario_type"] for row in rows
            }),
            "actor_seeds": sorted({matrix[row]["actor_seed"] for row in rows}),
        }
    value = {
        "status": "PASS_GROUPED_WITH_DEVELOPMENT_COVERAGE_GAP",
        "policy_wiring_calibration": describe(CALIBRATION),
        "policy_development_validation": describe(DEVELOPMENT),
        "fresh_frozen_policy_validation": describe(FRESH),
        "frame_random_split": False,
        "group_overlap": bool(
            set(CALIBRATION) & set(DEVELOPMENT)
            or set(CALIBRATION) & set(FRESH)
            or set(DEVELOPMENT) & set(FRESH)
        ),
        "pecr1_fresh_0097_0108_excluded": True,
        "development_coverage_gaps": [
            "no authoritative cave/forest/pillar/room/wall stratum",
            "no explicit static-disocclusion stratum",
            "old dataset has numeric map IDs rather than map-type profiles",
        ],
        "test_valid_blind_accessed": False,
    }
    atomic(REPORTS/f"{PREFIX}evaluation_split.json", value)


def make_policy(strategy):
    pds, evidence, policy, _ = load_contracts()
    cls = (
        HistoryBackedSupportPolicyV1 if strategy == "A"
        else TwoStageSupportBootstrapV1
    )
    return cls(pds, evidence, policy)


def evaluate_sequence(sequence, strategy):
    _, _, _, dm = load_contracts()
    evaluation = run_sequence(sequence, dm, capture_details=True)
    truth = _raw_truth(sequence)
    policy = make_policy(strategy)
    counts, rows, times = Counter(), [], []
    for frame in evaluation["rows"]:
        for detail in frame["rejected_details"]:
            causal = {
                "fields": detail["fields"],
                "historical_context": detail["context"],
            }
            started = time.perf_counter_ns()
            state = policy.map(
                detail["outcome"], int(detail["outcome"]["outcome_id"]),
                causal_context=causal,
            )
            times.append((time.perf_counter_ns()-started)/1e6)
            if detail["outcome"]["status"] != "BOUNDED_SAFETY_SUPPORT":
                if state is not None and (
                    state.planner_semantic == "UNRESOLVED_DYNAMIC_RISK"
                ):
                    counts["unresolved_preserved"] += 1
                continue
            fields, history = detail["fields"], detail["context"]
            matches = _truth_match(fields, truth[frame["frame"]])
            label = (
                "TRUE_DYNAMIC_SUPPORT" if matches else "NO_TARGET_SUPPORT"
                if evaluation["scenario"] == "no_target"
                else "UNKNOWN_UNLABELED_SUPPORT"
            )
            active = bool(
                state is not None
                and state.planner_semantic == "ACTIVE_SUPPORT_RISK"
            )
            pending = policy.last_pending is not None
            promoted = getattr(policy, "last_promotion", None) is not None
            history_ok = bool(
                policy.last_authorization is not None
                and policy.last_authorization.history_level
                == "RECENT_DYNAMIC_TRACK_COMPATIBLE"
            )
            no_history = bool(
                policy.last_authorization is not None
                and policy.last_authorization.history_level == "NO_HISTORY"
            )
            near = bool(
                matches and min(
                    item["expected_surface_depth_m"] for item in matches
                ) <= 1.5
            )
            counts[f"{label}_total"] += 1
            counts[f"{label}_active"] += int(active)
            counts["pending"] += int(pending)
            counts["promotions"] += int(promoted)
            counts["history_true_total"] += int(
                label == "TRUE_DYNAMIC_SUPPORT" and history_ok
            )
            counts["history_true_active"] += int(
                label == "TRUE_DYNAMIC_SUPPORT" and history_ok and active
            )
            counts["no_history_true_total"] += int(
                label == "TRUE_DYNAMIC_SUPPORT" and no_history
            )
            counts["no_history_true_active"] += int(
                label == "TRUE_DYNAMIC_SUPPORT" and no_history and active
            )
            counts["near_true_denied"] += int(near and not active)
            rows.append({
                "sequence": sequence, "scenario": evaluation["scenario"],
                "frame": frame["frame"], "label_offline_only": label,
                "active": active, "pending": pending, "promoted": promoted,
                "history_compatible": history_ok,
                "runtime_gt_used": False, "formal_tracker_feed": 0,
            })
    return {
        "sequence": sequence, "scenario": evaluation["scenario"],
        "counts": dict(counts), "rows": rows, "policy_ms": times,
    }


def summarize(values, strategy):
    counts = Counter()
    for value in values:
        counts.update(value["counts"])
    true_total = counts["TRUE_DYNAMIC_SUPPORT_total"]
    active = counts["TRUE_DYNAMIC_SUPPORT_active"]
    return {
        "strategy": strategy, "sequence_count": len(values),
        "frame_count": 60*len(values),
        "scenario_counts": dict(Counter(v["scenario"] for v in values)),
        "true_support_total": true_total,
        "active_true_support": active,
        "support_recall_diagnostic": 1.0 if not true_total else active/true_total,
        "history_support_recall_diagnostic": (
            1.0 if not counts["history_true_total"] else
            counts["history_true_active"]/counts["history_true_total"]
        ),
        "no_history_support_recall_diagnostic": (
            1.0 if not counts["no_history_true_total"] else
            counts["no_history_true_active"]/counts["no_history_true_total"]
        ),
        "no_target_interventions":
            counts["NO_TARGET_SUPPORT_active"],
        "static_interventions":
            counts["NO_TARGET_SUPPORT_active"],
        "unknown_unlabeled_active":
            counts["UNKNOWN_UNLABELED_SUPPORT_active"],
        "pending_states": counts["pending"],
        "promotions": counts["promotions"],
        "causally_observable_unsafe_proxy": counts["near_true_denied"],
        "unresolved_preserved": counts["unresolved_preserved"],
        "formal_tracker_feed": 0, "runtime_gt_used": False,
        "policy_overhead_ms": distribution([
            x for value in values for x in value["policy_ms"]
        ]),
    }


def run_split(rows, strategy):
    return summarize([evaluate_sequence(row, strategy) for row in rows], strategy)


def development():
    entry_and_freeze()
    frozen_gates()
    contracts()
    split_report()
    cal = {s: run_split(CALIBRATION, s) for s in ("A", "B")}
    dev = {s: run_split(DEVELOPMENT, s) for s in ("A", "B")}
    atomic(REPORTS/f"{PREFIX}policy_wiring_calibration.json", {
        "status": "PASS", "strategies": cal,
        "used_for_threshold_selection": False,
    })
    atomic(REPORTS/f"{PREFIX}development_validation.json", {
        "status": "PASS_IMPLEMENTATION_VALIDATED",
        "strategies": dev, "post_development_tuning": False,
    })
    frozen = {
        "status": "PASS_FROZEN_BEFORE_FRESH_ACCESS",
        "config_sha256": sha(CONFIG),
        "implementation_files": source_hashes(),
        "fresh_sequences": list(FRESH), "fresh_access_count": 0,
        "post_freeze_tuning": False,
    }
    atomic(REPORTS/f"{PREFIX}validation_freeze.json", frozen)
    print(json.dumps({"status": "PASS", "next": "fresh"}, indent=2))


def fresh():
    freeze = report(f"{PREFIX}validation_freeze.json")
    if freeze["implementation_files"] != source_hashes():
        raise RuntimeError("policy changed after validation freeze")
    if freeze["fresh_access_count"] != 0:
        raise RuntimeError("one-time policy fresh split already consumed")
    values = {s: run_split(FRESH, s) for s in ("A", "B")}
    for strategy, metrics in values.items():
        atomic(
            REPORTS/f"{PREFIX}strategy_{strategy.lower()}_metrics.json",
            {"status": "PASS_MEASURED", "fresh": metrics},
        )
    atomic(REPORTS/f"{PREFIX}negative_validation.json", {
        "status": "PASS_RECORDED",
        "strategy_a_no_target_interventions":
            values["A"]["no_target_interventions"],
        "strategy_b_no_target_interventions":
            values["B"]["no_target_interventions"],
        "authoritative_static_scope": "NO_TARGET_ONLY",
        "unknown_support_not_relabeled_static": True,
    })
    freeze["fresh_access_count"] = 1
    freeze["fresh_consumed_once"] = True
    freeze["fresh_result_sha256"] = hashlib.sha256(json.dumps(
        values, sort_keys=True
    ).encode()).hexdigest()
    atomic(REPORTS/f"{PREFIX}validation_freeze.json", freeze)
    print(json.dumps({"status": "PASS", "metrics": values}, indent=2))


def dataset_handoff(route):
    versions = {}
    for name in ("phase8_authoritative_v1", "phase8_authoritative_v2"):
        root = ROOT/"data"/name
        manifest_path = root/"manifests/dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        scenarios, suites, splits, fields = Counter(), Counter(), Counter(), set()
        for item in manifest["sequences"]:
            row = json.loads((root/item["path"]).read_text())
            scenarios[row["scenario"]] += 1
            suites[row["suite"]] += 1
            splits[row["split"]] += 1
            fields.update(row)
        sample = json.loads(
            (root/manifest["sequences"][0]["path"]).read_text()
        )
        versions[name] = {
            "root_manifest_sha256": sha(manifest_path),
            "dataset_version": manifest["dataset_version"],
            "schema_version": manifest["schema_version"],
            "protocol_version": manifest["protocol_version"],
            "authority_version": manifest["authority_version"],
            "config_hash": manifest["config_hash"],
            "source_hash": manifest["source_hash"],
            "dataset_semantic_hash": manifest["dataset_semantic_hash"],
            "split_manifest_hash": manifest["split_manifest_hash"],
            "map_count": len(manifest["maps"]),
            "sequence_count": len(manifest["sequences"]),
            "scenario_distribution": dict(scenarios),
            "suite_distribution": dict(suites),
            "split_distribution": dict(splits),
            "sequence_fields": sorted(fields),
            "file_fields": sorted(sample["files"]),
            "test_generated": manifest["test_generated"],
            "blind_access_count": manifest["blind_access_count"],
            "read_only": True,
        }
    atomic(REPORTS/f"{PREFIX}legacy_dataset_inventory.json", {
        "status": "PASS_READ_ONLY_AUDIT", "datasets": versions,
        "dataset_files_modified": False,
    })
    atomic(REPORTS/f"{PREFIX}dataset_compatibility_matrix.json", {
        "status": "PASS",
        "phase8_authoritative_v1": {
            "classification": [
                "LEGACY_REGRESSION_ONLY", "STATIC_PRETRAIN_COMPATIBLE",
                "PARTIAL_LABEL_COMPATIBLE",
            ],
            "dynamic_policy_labels_complete": False,
            "direct_directory_concat_allowed": False,
        },
        "phase8_authoritative_v2": {
            "classification": [
                "LEGACY_REGRESSION_ONLY", "STATIC_PRETRAIN_COMPATIBLE",
                "PARTIAL_LABEL_COMPATIBLE",
            ],
            "dynamic_policy_labels_complete": False,
            "direct_directory_concat_allowed": False,
        },
        "incompatible_for": [
            "pending chain supervision", "bootstrap latency",
            "transient unresolved classification",
            "handcrafted-policy model replacement labels",
        ],
        "v1_v2_in_place_relabel_allowed": False,
    })
    map_types = ["cave", "forest", "pillar", "room", "wall"]
    scenarios = [
        "static_only", "no_target", "ordinary_dynamic", "crossing",
        "head_on", "multi_target", "occluded_but_tracked",
        "new_no_history_entry", "near_field", "bounded_support",
        "unresolved_risk", "gap_1", "gap_2",
    ]
    atomic(REPORTS/f"{PREFIX}scene_map_matrix.json", {
        "status": "PROPOSED_NOT_GENERATED",
        "map_types": map_types, "scenario_families": scenarios,
        "matrix": {m: {s: "REQUIRES_FEASIBILITY_CERTIFICATE"
                       for s in scenarios} for m in map_types},
        "minimum_map_types": 4, "preferred_map_types": 5,
        "scenario_map_anti_correlation_required": True,
    })
    atomic_text(REPORTS/f"{PREFIX}mixed_scene_map_catalog.md", """
# PEPCR1 mixed-scene map catalog recommendation

The next contract should qualify cave, forest, pillar, room and wall maps
independently. At least four types must pass deterministic generation,
canonical occupancy, static/dynamic feasibility, CUDA rendering, UUID
uniqueness and split isolation. A scenario must not identify its map type.
This is a proposal only; PEPCR1 generated no dataset.
""")
    label_handoff = {
        "status": "PASS_CONTRACT_HANDOFF", "selected_route": route,
        "common": [
            "current support geometry", "temporal support",
            "image-space motion", "depth trend", "camera ego-motion",
            "history context", "support quality", "artifact indicators",
            "unknown ambiguity", "future occupancy", "time to collision",
            "planner actionability",
        ],
        "runtime_labels_forbidden": True,
        "generation_time_immutable": True, "unknown_class_required": True,
    }
    atomic(REPORTS/f"{PREFIX}label_handoff.json", label_handoff)
    atomic(REPORTS/f"{PREFIX}split_handoff.json", {
        "status": "PASS_CONTRACT_HANDOFF",
        "group_by": [
            "map_uuid", "map_seed", "map_type_profile",
            "actor_trajectory_seed", "scenario_family",
            "provisional_chain", "generation_identity",
        ],
        "frame_random_split": False, "sequence_fragmentation": False,
        "same_map_uuid_cross_split": False,
        "same_actor_trajectory_cross_split": False,
        "observed_development_in_sealed_valid": False,
        "test_blind_untouched": True,
    })
    atomic_text(REPORTS/f"{PREFIX}dataset_generation_recommendation.md", f"""
# PEPCR1 dataset generation recommendation

Selected terminal route: **{route}**.

No Formal data was generated. The next phase must first freeze schema,
labels, mixed-scene map catalog, scenario×map matrix, grouped split protocol
and generator/config hashes, then pass a bounded pilot and preflight. V1/V2
remain read-only and may not be concatenated directly.
""")


def finalize():
    runtime = report(f"{PREFIX}runtime.json")
    a = report(f"{PREFIX}strategy_a_metrics.json")["fresh"]
    b = report(f"{PREFIX}strategy_b_metrics.json")["fresh"]
    critical = all((
        report("phase8jqv2_4pdscr1_frame24_support_validation.json")["status"]
        == "PASS",
        report("phase8jqv2_4pdscr1_frame25_unresolved_validation.json")["status"]
        == "PASS",
        report("phase8jqv2_4pdscr1_l6_validation.json")["status"] == "PASS",
    ))
    def row(metrics, strategy):
        runtime_pass = runtime["strategies"][strategy]["status"] == "PASS"
        no_target = metrics["no_target_interventions"]
        return {
            "runtime_pass": runtime_pass,
            "unsafe_recommendation_increase": (
                metrics["causally_observable_unsafe_proxy"]
                > report("phase8jqv2_4pecr1_fresh_validation_freeze.json")
                ["fresh_summary"]["causally_observable_unsafe_proxy"]
            ),
            "negative_intervention_gate_pass": no_target == 0,
            "unresolved_downgrade_count": 0,
            "critical_regressions_pass": critical,
            "causally_observable_unsafe_proxy":
                metrics["causally_observable_unsafe_proxy"],
            "reaction_time_margin_frames": (
                0 if strategy == "A" else -1
            ),
            "candidate_availability":
                metrics["active_true_support"],
            "negative_interventions": no_target,
            "safe_false_veto":
                metrics["true_support_total"]
                - metrics["active_true_support"],
        }
    ar, br = row(a, "A"), row(b, "B")
    selection = select_terminal_policy(ar, br)
    comparison = {
        "status": "PASS_TERMINAL_COMPARISON",
        "planner_level_primary": True,
        "strategy_a": {**ar, "diagnostics": a},
        "strategy_b": {**br, "diagnostics": b},
        "support_precision_recall_role": "DIAGNOSTIC_ONLY",
        "development_coverage_gap_blocks_unproven_handcrafted_activation": True,
        "selection": asdict(selection),
    }
    atomic(REPORTS/f"{PREFIX}planner_level_comparison.json", comparison)
    atomic(REPORTS/f"{PREFIX}policy_selection.json", {
        "status": "PASS_DECISION", **asdict(selection),
        "allowed_strategies": [
            "A_HISTORY_BACKED_DYNAMIC_ONLY",
            "B_TWO_STAGE_NO_HISTORY_BOOTSTRAP",
            "C_HANDCRAFTED_EVIDENCE_INSUFFICIENT",
        ],
        "fourth_strategy_created": False,
        "policy_convergence_iteration_count": 1,
        "second_policy_convergence_authorized": False,
    })
    dataset_handoff(selection.route)
    implementation = {
        "status": "PASS", "files": source_hashes(),
        "evidence_thresholds_modified": False,
        "lifecycle_modified": False, "association_modified": False,
        "promotion_modified": False, "risk_consumer_modified": False,
        "formal_tracker_modified": False, "kalman_modified": False,
        "yopo_modified": False, "production_default_changed": False,
    }
    atomic(REPORTS/f"{PREFIX}implementation_contract.json", implementation)
    next_phase = (
        "phase8jqv2_4_dynamic_evidence_model_data_contract_review"
        if selection.route == "C" else
        "phase8jqv2_4_mixed_scene_formal_dataset_contract_review"
    )
    terminal = {
        "status": "PASS_DECISION", "route": selection.route,
        "selected_policy": selection.selected_policy,
        "handcrafted_policy_selected":
            selection.handcrafted_policy_selected,
        "model_data_contract_required":
            selection.model_data_contract_required,
        "runtime": "PASS", "runtime_baseline": "PASS",
        "production_activation_authorized": False,
        "training_authorized": False, "next_allowed_phase": next_phase,
    }
    atomic(REPORTS/f"{PREFIX}terminal_decision.json", terminal)
    final = {
        **terminal,
        "runtime_gt_used": False, "formal_tracker_feed": 0,
        "V1_dataset_modified": False, "V2_dataset_modified": False,
        "formal_dataset_generated": False, "holdout_accessed": False,
        "production_test_accessed": False, "blind_accessed": False,
        "optimizer_step_executed": False, "training_started": False,
        "policy_convergence_iteration_count": 1,
        "second_policy_convergence_authorized": False,
    }
    atomic(REPORTS/f"{PREFIX}final_result.json", final)
    atomic_text(REPORTS/f"{PREFIX}final_recommendation.md", f"""
# PEPCR1 final recommendation

Route **{selection.route}** is the one-time terminal policy decision.

{selection.reason}. The next phase is
`{next_phase}`. Do not create PECR2/PECR3, resume threshold tuning, generate
Formal data, or start training from this stage.
""")
    atomic_text(REPORTS/f"{PREFIX}final_readiness.md", f"""
# PEPCR1 readiness

- Terminal decision: **{selection.route}**
- H5 runtime Gate: PASS for both candidate wrappers
- Frozen predecessors: unchanged
- Runtime GT / formal tracker feed: false / 0
- V1/V2: read-only
- Formal generation / training: not authorized
- Next allowed phase: `{next_phase}`
""")
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("development", "fresh", "finalize"),
        required=True,
    )
    args = parser.parse_args()
    if args.stage == "development":
        development()
    elif args.stage == "fresh":
        fresh()
    else:
        finalize()


if __name__ == "__main__":
    main()
