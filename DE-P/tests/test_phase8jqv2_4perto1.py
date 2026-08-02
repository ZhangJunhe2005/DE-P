"""PERTO1 standard-library unittest gate (72 contract cases)."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import yaml

from policy.dynamic.provisional_evidence_authorizer_fast_v1 import (
    ProvisionalEvidenceAuthorizerFastV1,
)
from policy.dynamic.provisional_evidence_authorizer_v1 import (
    ProvisionalEvidenceAuthorizerV1,
)
from policy.dynamic.provisional_evidence_context_v1 import (
    ProvisionalEvidenceContextV1,
)
from policy.dynamic.provisional_evidence_history_index_v1 import (
    ProvisionalEvidenceHistoryIndexV1,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
PREFIX = "phase8jqv2_4perto1_"
EVIDENCE = yaml.safe_load((
    ROOT/"configs/provisional_evidence_contract_v1_candidate.yaml"
).read_text())
REFERENCE = ProvisionalEvidenceAuthorizerV1(EVIDENCE)
FAST = ProvisionalEvidenceAuthorizerFastV1(EVIDENCE, 4)
REQUIRED = (
    "entry_gate", "frozen_artifacts", "semantic_freeze", "r0_baseline",
    "deadline_miss_manifest", "stage_profile", "allocation_profile",
    "gc_profile", "join_and_scheduler_profile", "runtime_root_cause",
    "r1_diagnostics", "r2_typed_context", "r3_bitmask",
    "r4_scalar_geometry", "r5_history_index", "r6_allocation_gc",
    "r7_batch", "r8_selected", "candidate_comparison",
    "authorization_equivalence", "state_birth_equivalence",
    "planner_semantic_equivalence", "metric_equivalence",
    "host_runtime", "deadline_and_backlog", "memory_and_gc",
    "implementation_contract", "determinism", "regression",
    "compatibility_matrix", "policy_convergence_handoff",
    "candidate_selection", "final_result",
)


def report(name):
    return json.loads((REPORTS/f"{PREFIX}{name}.json").read_text())


def outcome():
    return {
        "outcome_id": 1, "timestamp": 1.05,
        "status": "BOUNDED_SAFETY_SUPPORT",
        "source_component_ids": [1],
        "support": {
            "position_set_min_world": [0., 0., 0.],
            "position_set_max_world": [1., 1., 1.],
            "valid_point_count": 50,
        },
    }


def context(**fields):
    value = {
        "pixel_count": 50, "pixel_bbox": [10, 10, 19, 19],
        "temporal_support": 2, "stable_overlap_fraction": 1.,
        "world_speed_mps": 1., "direction_consistency": 1.,
        "closer_fraction": 0.,
        "temporal_provenance_invalid_fraction": 0.,
        "fov_boundary_fraction": 0., "boundary_hazard_fraction": 0.,
    }
    value.update(fields)
    return {
        "fields": value,
        "historical_context": {
            "track_exists": False, "generation": None,
            "dynamic": False, "confirmed": False,
            "last_direct_measurement_time": None,
            "last_safety_evidence": None,
            "last_valid_geometry_bounds": None,
        },
    }


class PERTO1Gate(unittest.TestCase):
    def _case(self, index):
        entry = report("entry_gate")
        freeze = report("semantic_freeze")
        final = report("final_result")
        runtime = report("host_runtime")
        deadline = report("deadline_and_backlog")
        regression = report("regression")
        route_runtime_ok = (
            final["route"] == "A" and runtime["status"] == "PASS"
        )
        route_tail = (
            final["route"] == "B" and runtime["status"] == "FAIL"
            and final["next_allowed_phase"]
            == "phase8jqv2_4_runtime_environment_and_tail_review"
        )
        left = REFERENCE.authorize(outcome(), context())
        right = FAST.authorize(outcome(), context())
        checks = {
            1: entry["checks"]["pecr1_route_f"],
            2: entry["checks"]["semantic_route_b"],
            3: freeze["no_target_provisional_birth"] == 7,
            4: freeze["static_false_dynamic_birth"] == 7,
            5: freeze["true_support_recall"] == .5692307692307692,
            6: freeze["history_support_recall"] == 1.,
            7: freeze["no_history_support_recall"] == .7307692307692307,
            8: freeze["known_failure_birth"] == 0,
            9: not report("frozen_artifacts")["pdscr1_v1_artifacts_modified"],
            10: not report("frozen_artifacts")["pdscr1_v2_artifacts_modified"],
            11: not report("frozen_artifacts")["dmcr1_artifacts_modified"],
            12: not report("frozen_artifacts")["mar1_artifacts_modified"],
            13: not report("frozen_artifacts")["TrackManager_algorithm_modified"],
            14: left.authorized == right.authorized,
            15: left.evidence_level == right.evidence_level,
            16: left.reason_code == right.reason_code,
            17: left.diagnostic == right.diagnostic,
            18: left.evidence_families == right.evidence_families,
            19: left.unavailable_families == right.unavailable_families,
            20: left.negative_evidence == right.negative_evidence,
            21: left.history_level == right.history_level,
            22: left.support_quality == right.support_quality,
            23: report("state_birth_equivalence")["status"] == "PASS",
            24: report("planner_semantic_equivalence")["status"] == "PASS",
            25: isinstance(
                ProvisionalEvidenceContextV1.parse(outcome(), context()),
                ProvisionalEvidenceContextV1,
            ),
            26: REFERENCE.authorize(outcome(), {}).authorized
                == FAST.authorize(outcome(), {}).authorized,
            27: self._forbidden_detected(),
            28: report("r3_bitmask")["semantic_equivalence"] == "PASS",
            29: report("r4_scalar_geometry")["semantic_equivalence"] == "PASS",
            30: self._history_bounded(),
            31: self._history_no_refresh(),
            32: report("allocation_profile")[
                "bounded_diagnostic_capacity"] > 0,
            33: FAST.authorize_batch([
                (outcome(), context()), (outcome(), context(closer_fraction=.5))
            ]) == tuple(
                FAST.authorize(row, ctx) for row, ctx in [
                    (outcome(), context()),
                    (outcome(), context(closer_fraction=.5)),
                ]
            ),
            34: report("r7_batch")["semantic_equivalence"] == "PASS",
            35: FAST.diagnostic_capacity == 4,
            36: freeze["frame24"] == "PASS",
            37: freeze["frame25"] == "PASS",
            38: freeze["l6"] == "PASS",
            39: freeze["formal_tracker_feed"] == 0,
            40: not freeze["runtime_gt_used"],
            41: runtime.get("same_frame_only") is True,
            42: runtime.get("atomic_join_before_snapshot") is True,
            43: runtime.get("queue_depth") == 0,
            44: (
                runtime["optimized"]["steady_state_ms"]["p95"]
                <= runtime["gate_ms"]
            ) if route_runtime_ok else route_tail,
            45: (
                runtime["optimized"]["steady_state_ms"]["p50"]
                < runtime["gate_ms"]
            ) if runtime["status"] != "PENDING_HOST" else False,
            46: (
                deadline["deadline_miss_rate"] <= .01
            ) if route_runtime_ok else route_tail,
            47: (
                deadline["consecutive_deadline_miss_max"] <= 1
            ) if route_runtime_ok else route_tail,
            48: not deadline.get("unbounded_backlog", True),
            49: report("memory_and_gc")["bounded"],
            50: report("determinism")["semantic_repeat_equal"],
            51: not report("candidate_selection")["production_default_changed"],
            52: not report("metric_equivalence")[
                "semantic_fresh_validation_created"],
            53: not final["formal_dataset_generated"],
            54: not final["holdout_test_blind_accessed"],
            55: not final["optimizer_step_executed"],
            56: not final["training_started"],
            57: regression["PECR1"] == "FROZEN",
            58: regression["PDSCR1"] == "FROZEN",
            59: regression["DMCR1_MAR1"] == "FROZEN",
            60: regression["DIRO1_CLDSR1"] == "FROZEN",
            61: regression["BRIR1_BDRR1"] == "FROZEN",
            62: regression["SAMSR1_DOGMR1"] == "FROZEN",
            63: regression["PTAR1_KUCR1"] == "FROZEN",
            64: regression["OCSR1_TCCR1"] == "FROZEN",
            65: regression["SOCR1_EOSR1"] == "FROZEN",
            66: report("stage_profile")["status"] == "PASS_PROFILED",
            67: report("implementation_contract")["status"] == "PASS",
            68: self._reports_complete(),
            69: report("policy_convergence_handoff")[
                "threshold_repair_forbidden"],
            70: report("policy_convergence_handoff")["review_may_run_once"],
            71: final["semantic_status_preserved"] == "FAIL_ROUTE_B",
            72: final["next_allowed_phase"] in {
                "phase8jqv2_4_provisional_evidence_policy_convergence_review",
                "phase8jqv2_4_runtime_environment_and_tail_review",
            },
        }
        self.assertTrue(checks[index])

    @staticmethod
    def _forbidden_detected():
        try:
            FAST.authorize({**outcome(), "sequence_id": "forbidden"}, context())
        except ValueError:
            return True
        return False

    @staticmethod
    def _history_bounded():
        index = ProvisionalEvidenceHistoryIndexV1(2)
        for value in range(4):
            index.put(value, 0, value, value)
        return len(index) == 2 and index.eviction_count == 2

    @staticmethod
    def _history_no_refresh():
        index = ProvisionalEvidenceHistoryIndexV1(2)
        index.put(1, 0, 3., "value")
        before = index.timestamp(1, 0)
        index.get(1, 0)
        return index.timestamp(1, 0) == before

    @staticmethod
    def _reports_complete():
        return (
            all((REPORTS/f"{PREFIX}{name}.json").exists() for name in REQUIRED)
            and all((REPORTS/f"{PREFIX}{name}.md").exists() for name in (
                "migration_plan", "final_recommendation", "final_readiness",
            ))
        )


def _install():
    for index in range(1, 73):
        def test(self, value=index):
            self._case(value)
        test.__name__ = f"test_{index:02d}"
        setattr(PERTO1Gate, test.__name__, test)


_install()
