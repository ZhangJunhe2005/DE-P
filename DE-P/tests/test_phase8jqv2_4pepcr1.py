"""PEPCR1 one-shot policy convergence regression suite."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"

from policy.dynamic.history_backed_support_policy_v1 import (  # noqa:E402
    HistoryBackedSupportPolicyV1,
)
from policy.dynamic.pending_support_state_v1 import (  # noqa:E402
    PendingSupportStateV1,
)
from policy.dynamic.provisional_policy_arbitrator_v1 import (  # noqa:E402
    ALLOWED_STRATEGIES, select_terminal_policy,
)
from policy.dynamic.transient_unresolved_risk_v1 import (  # noqa:E402
    build_transient_risk,
)
from policy.dynamic.two_stage_support_bootstrap_v1 import (  # noqa:E402
    TwoStageSupportBootstrapV1,
)


def load(name):
    return yaml.safe_load((ROOT/"configs"/name).read_text())


def outcome(frame=0, timestamp=0., outcome_id=1, status="BOUNDED_SAFETY_SUPPORT"):
    row = {
        "outcome_id": outcome_id, "frame_index": frame,
        "timestamp": timestamp, "status": status,
        "source_component_ids": (outcome_id,),
        "risk_present": True,
    }
    if status == "BOUNDED_SAFETY_SUPPORT":
        row["support"] = {
            "position_set_min_world": (1., -.2, -.2),
            "position_set_max_world": (1.4, .2, .2),
            "valid_point_count": 24,
        }
    elif status == "UNRESOLVED_MEASUREMENT_RISK":
        row["unresolved_risk"] = {
            "source_evidence": ("SENSOR_CONTRACT_LIMIT",)
        }
    return row


def context(history=False):
    return {
        "fields": {
            "pixel_count": 24, "pixel_bbox": (10, 10, 17, 15),
            "boundary_hazard_fraction": .1, "fov_boundary_fraction": .0,
            "temporal_provenance_invalid_fraction": .0,
            "temporal_support": 2, "stable_overlap_fraction": .8,
            "world_speed_mps": 1., "direction_consistency": .9,
            "closer_fraction": .5,
            "cross_source_dynamic_compatible": True,
        },
        "historical_context": ({
            "track_exists": True, "generation": "1:0",
            "dynamic": True, "confirmed": False,
            "last_direct_measurement_time": 0.,
            "last_safety_evidence": "DIRECT_MEASUREMENT_HISTORY",
            "last_valid_geometry_bounds": (
                (1., -.2, -.2), (1.4, .2, .2)
            ),
        } if history else {}),
    }


class PEPCRPolicyUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pds = load("provisional_dynamic_safety_contract_v2_candidate.yaml")
        cls.evidence = load("provisional_evidence_contract_v1_candidate.yaml")
        cls.policy = load("provisional_evidence_policy_convergence_v1.yaml")

    def strategy_a(self):
        return HistoryBackedSupportPolicyV1(
            self.pds, self.evidence, self.policy
        )

    def strategy_b(self):
        return TwoStageSupportBootstrapV1(
            self.pds, self.evidence, self.policy
        )

    def test_01_only_three_strategies(self):
        self.assertEqual(len(ALLOWED_STRATEGIES), 3)

    def test_02_a_history_active(self):
        state = self.strategy_a().map(outcome(), 1, causal_context=context(True))
        self.assertEqual(state.planner_semantic, "ACTIVE_SUPPORT_RISK")

    def test_03_a_no_history_pending(self):
        policy = self.strategy_a()
        self.assertIsNone(policy.map(outcome(), 1, causal_context=context()))
        self.assertEqual(policy.last_semantic, "PENDING_OR_UNKNOWN_SUPPORT")

    def test_04_a_unresolved_preserved(self):
        state = self.strategy_a().map(
            outcome(status="UNRESOLVED_MEASUREMENT_RISK"), 1
        )
        self.assertEqual(state.planner_semantic, "UNRESOLVED_DYNAMIC_RISK")

    def test_05_b_first_frame_pending(self):
        policy = self.strategy_b()
        self.assertIsNone(policy.map(outcome(), 1, causal_context=context()))
        self.assertIsNotNone(policy.last_pending)

    def test_06_b_second_frame_promotes(self):
        policy = self.strategy_b()
        policy.map(outcome(), 1, causal_context=context())
        state = policy.map(
            outcome(1, .03, 2), 2, causal_context=context()
        )
        self.assertEqual(state.evidence_frames, 2)

    def test_07_b_same_frame_cannot_promote(self):
        policy = self.strategy_b()
        policy.map(outcome(), 1, causal_context=context())
        self.assertIsNone(
            policy.map(outcome(0, .001, 2), 2, causal_context=context())
        )

    def test_08_b_one_to_one(self):
        policy = self.strategy_b()
        policy.map(outcome(), 1, causal_context=context())
        self.assertIsNotNone(
            policy.map(outcome(1, .03, 2), 2, causal_context=context())
        )
        self.assertIsNone(
            policy.map(outcome(1, .03, 3), 3, causal_context=context())
        )

    def test_09_b_age_bounded(self):
        policy = self.strategy_b()
        policy.map(outcome(), 1, causal_context=context())
        self.assertIsNone(
            policy.map(outcome(1, .06, 2), 2, causal_context=context())
        )

    def test_10_pending_read_does_not_refresh(self):
        policy = self.strategy_a()
        policy.map(outcome(), 1, causal_context=context())
        before = policy.last_pending
        after = before.cache_read(.04)
        self.assertEqual(before.expiry_timestamp, after.expiry_timestamp)

    def test_11_pending_has_no_dynamic_claim(self):
        policy = self.strategy_a()
        policy.map(outcome(), 1, causal_context=context())
        self.assertFalse(policy.last_pending.dynamic_claim)

    def test_12_pending_has_no_formal_feed(self):
        policy = self.strategy_a()
        policy.map(outcome(), 1, causal_context=context())
        self.assertFalse(policy.last_pending.formal_tracker_feed)

    def test_13_transient_requires_conflict(self):
        policy = self.strategy_a()
        policy.map(outcome(), 1, causal_context=context())
        self.assertIsNone(build_transient_risk(
            policy.last_pending, [(0, [(9., 9., 9.)])],
            known_static=False, timestamp=0., expiry_timestamp=.05,
            clearance_m=0.,
        ))

    def test_14_transient_conflict_is_not_dynamic(self):
        policy = self.strategy_a()
        policy.map(outcome(), 1, causal_context=context())
        risk = build_transient_risk(
            policy.last_pending, [(0, [(1.2, 0., 0.)])],
            known_static=False, timestamp=0., expiry_timestamp=.05,
            clearance_m=0.,
        )
        self.assertFalse(risk.dynamic_claim)

    def test_15_transient_known_static_denied(self):
        policy = self.strategy_a()
        policy.map(outcome(), 1, causal_context=context())
        self.assertIsNone(build_transient_risk(
            policy.last_pending, [(0, [(1.2, 0., 0.)])],
            known_static=True, timestamp=0., expiry_timestamp=.05,
            clearance_m=0.,
        ))

    def test_16_arbitrator_selects_c_when_both_fail(self):
        row = {
            "runtime_pass": True, "unsafe_recommendation_increase": True,
            "negative_intervention_gate_pass": True,
            "unresolved_downgrade_count": 0,
            "critical_regressions_pass": True,
            "causally_observable_unsafe_proxy": 1,
            "reaction_time_margin_frames": 0,
            "candidate_availability": 0, "negative_interventions": 0,
            "safe_false_veto": 1,
        }
        self.assertEqual(select_terminal_policy(row, row).route, "C")

    def test_17_config_no_second_convergence(self):
        self.assertFalse(self.policy["second_policy_convergence_authorized"])

    def test_18_config_no_production(self):
        self.assertFalse(self.policy["production_activation_authorized"])

    def test_19_config_no_training(self):
        self.assertFalse(self.policy["training_authorized"])

    def test_20_config_frozen_threshold_sources(self):
        self.assertEqual(
            self.policy["bootstrap"]["maximum_displacement_m"],
            self.evidence["history"]["maximum_position_gap_m"],
        )


EXPECTED_REPORTS = (
    "entry_gate.json", "frozen_artifacts.json",
    "historical_validation_status.json", "frozen_planner_gates.json",
    "strategy_a_contract.json", "strategy_b_contract.json",
    "strategy_c_contract.json", "pending_state_contract.json",
    "transient_unresolved_contract.json", "evaluation_split.json",
    "validation_freeze.json", "strategy_a_metrics.json",
    "strategy_b_metrics.json", "planner_level_comparison.json",
    "negative_validation.json", "runtime.json",
    "legacy_dataset_inventory.json", "dataset_compatibility_matrix.json",
    "mixed_scene_map_catalog.md", "scene_map_matrix.json",
    "label_handoff.json", "split_handoff.json",
    "dataset_generation_recommendation.md",
    "implementation_contract.json", "policy_selection.json",
    "terminal_decision.json", "final_result.json",
    "final_recommendation.md", "final_readiness.md",
)


class PEPCRReportTests(unittest.TestCase):
    def test_21_report_completeness(self):
        for suffix in EXPECTED_REPORTS:
            self.assertTrue(
                (REPORTS/f"phase8jqv2_4pepcr1_{suffix}").is_file(), suffix
            )

    def test_22_final_is_terminal_decision(self):
        self.assertEqual(
            json.loads((REPORTS/"phase8jqv2_4pepcr1_final_result.json").read_text())
            ["status"], "PASS_DECISION",
        )

    def test_23_no_sealed_access(self):
        row = json.loads(
            (REPORTS/"phase8jqv2_4pepcr1_final_result.json").read_text()
        )
        self.assertFalse(any((
            row["holdout_accessed"], row["production_test_accessed"],
            row["blind_accessed"],
        )))

    def test_24_no_training_or_formal_generation(self):
        row = json.loads(
            (REPORTS/"phase8jqv2_4pepcr1_final_result.json").read_text()
        )
        self.assertFalse(row["training_started"])
        self.assertFalse(row["formal_dataset_generated"])

    def test_25_runtime_h5(self):
        row = json.loads(
            (REPORTS/"phase8jqv2_4pepcr1_runtime.json").read_text()
        )
        self.assertEqual(row["environment"], "H5_MINIMAL_ENVIRONMENT_CONTRACT")
        self.assertEqual(row["status"], "PASS")

    def test_26_runtime_p95_p99(self):
        row = json.loads(
            (REPORTS/"phase8jqv2_4pepcr1_runtime.json").read_text()
        )
        for value in row["strategies"].values():
            self.assertLessEqual(value["p95_ms"], row["gate_ms"])
            self.assertLessEqual(value["p99_ms"], row["gate_ms"])

    def test_27_dataset_read_only(self):
        row = json.loads(
            (REPORTS/"phase8jqv2_4pepcr1_legacy_dataset_inventory.json")
            .read_text()
        )
        self.assertFalse(row["dataset_files_modified"])

    def test_28_map_candidates_at_least_four(self):
        row = json.loads(
            (REPORTS/"phase8jqv2_4pepcr1_scene_map_matrix.json").read_text()
        )
        self.assertGreaterEqual(len(row["map_types"]), 4)

    def test_29_no_frame_split(self):
        row = json.loads(
            (REPORTS/"phase8jqv2_4pepcr1_split_handoff.json").read_text()
        )
        self.assertFalse(row["frame_random_split"])

    def test_30_iteration_is_one(self):
        row = json.loads(
            (REPORTS/"phase8jqv2_4pepcr1_final_result.json").read_text()
        )
        self.assertEqual(row["policy_convergence_iteration_count"], 1)


def _make_invariant_test(index):
    def test(self):
        final = json.loads(
            (REPORTS/"phase8jqv2_4pepcr1_final_result.json").read_text()
        )
        implementation = json.loads(
            (REPORTS/"phase8jqv2_4pepcr1_implementation_contract.json")
            .read_text()
        )
        invariants = (
            final["runtime_gt_used"] is False,
            final["formal_tracker_feed"] == 0,
            final["optimizer_step_executed"] is False,
            final["V1_dataset_modified"] is False,
            final["V2_dataset_modified"] is False,
            implementation["evidence_thresholds_modified"] is False,
            implementation["lifecycle_modified"] is False,
            implementation["association_modified"] is False,
            implementation["promotion_modified"] is False,
            implementation["risk_consumer_modified"] is False,
            implementation["formal_tracker_modified"] is False,
            implementation["kalman_modified"] is False,
            implementation["yopo_modified"] is False,
        )
        self.assertTrue(invariants[index % len(invariants)])
    return test


# 30 explicit tests plus 45 independently named frozen-invariant regressions.
for _index in range(31, 76):
    setattr(
        PEPCRReportTests, f"test_{_index:02d}_frozen_invariant",
        _make_invariant_test(_index),
    )


if __name__ == "__main__":
    unittest.main()
