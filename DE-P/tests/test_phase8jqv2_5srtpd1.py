"""SRTPD1 contract tests. No holdout sample is opened by this module."""

import json
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
P = "phase8jqv2_5srtpd1_"


def report(name):
    return json.loads((REPORTS / f"{P}{name}.json").read_text())


class SRTPD1ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entry = report("entry_gate")
        cls.freeze = report("deoacr1_terminal_freeze")
        cls.archive = report("learned_branch_archive")
        cls.inventory = report("reusable_asset_inventory")
        cls.runtime = report("runtime_component_matrix")
        cls.odd = report("dynamic_odd_contract")
        cls.out = report("out_of_odd_cases")
        cls.fail_closed = report("fail_closed_contract")
        cls.task = report("static_training_task")
        cls.compat = report("legacy_dataset_compatibility")
        cls.v3 = report("v3_static_view_contract")
        cls.plan = report("mixed_map_training_plan")
        cls.pilot = report("static_dataset_pilot")
        cls.h5 = report("h5_integration_plan")
        cls.routes = report("route_comparison")
        cls.final = report("final_result")

    def test_01_deoacr1_route_d(self):
        self.assertEqual(self.freeze["deoacr1_terminal"]["route"], "D")

    def test_02_original_next_null(self):
        self.assertIsNone(
            self.freeze["deoacr1_terminal"]["next_allowed_phase"]
        )

    def test_03_explicit_rebaseline(self):
        self.assertTrue(self.entry["explicit_project_rebaseline_authorized"])

    def test_04_archive_read_only(self):
        self.assertTrue(all(not x["modified"] for x in self.archive["checkpoints"]))

    def test_05_non_formal_checkpoints(self):
        self.assertEqual(self.archive["checkpoint_count"], 15)
        self.assertTrue(all(x["original_non_formal"] for x in self.archive["checkpoints"]))
        self.assertTrue(all("NOT_FOR_PRODUCTION" in x["archive_labels"]
                            for x in self.archive["checkpoints"]))

    def test_06_no_s1_s2_restart(self):
        self.assertFalse(self.final["S1_training_restarted"])
        self.assertFalse(self.final["S2_training_restarted"])

    def test_07_no_s3_s4(self):
        self.assertFalse(self.final["S3_S4_created"])

    def test_08_actionability_diagnostic_only(self):
        self.assertIn("planner_counterfactual_actionability_v2",
                      self.archive["retained_diagnostics"])

    def test_09_v1_read_only(self):
        self.assertTrue(self.compat["sources"]["V1"]["read_only"])

    def test_10_v2_read_only(self):
        self.assertTrue(self.compat["sources"]["V2"]["read_only"])

    def test_11_v3_read_only(self):
        self.assertTrue(self.compat["sources"]["V3_STATIC_VIEW"]["read_only"])

    def test_12_static_depth_exists(self):
        self.assertTrue(self.pilot["checks"]["static_depth_exists"])

    def test_13_static_depth_no_actor(self):
        self.assertTrue(
            self.pilot["checks"]["static_suite_has_no_actor_contribution"]
        )

    def test_14_composed_static_boundary(self):
        self.assertTrue(
            self.pilot["checks"]["dynamic_composed_static_boundary"]
        )

    def test_15_static_training_not_composed(self):
        self.assertFalse(
            self.pilot["checks"]["composed_depth_used_for_static_supervision"]
        )

    def test_16_no_owner_input(self):
        self.assertFalse(self.final["actor_owner_used_as_training_input"])

    def test_17_no_future_input(self):
        self.assertFalse(self.final["actor_future_used_as_training_input"])

    def test_18_no_dynamic_label_input(self):
        self.assertNotIn("dynamic label/actionability", self.task["inputs"])

    def test_19_cave(self):
        self.assertGreater(self.plan["map_coverage"]["all"]["cave"], 0)

    def test_20_forest(self):
        self.assertGreater(self.plan["map_coverage"]["all"]["forest"], 0)

    def test_21_pillar(self):
        self.assertGreater(self.plan["map_coverage"]["all"]["pillar"], 0)

    def test_22_room(self):
        self.assertGreater(self.plan["map_coverage"]["all"]["room"], 0)

    def test_23_wall(self):
        self.assertGreater(self.plan["map_coverage"]["all"]["wall"], 0)

    def test_24_v1_matrix(self):
        self.assertEqual(
            self.compat["sources"]["V1"]["classification"], "PRETRAIN_ONLY"
        )

    def test_25_v2_matrix(self):
        self.assertEqual(
            self.compat["sources"]["V2"]["classification"], "PRETRAIN_ONLY"
        )

    def test_26_v3_matrix(self):
        self.assertEqual(
            self.compat["sources"]["V3_STATIC_VIEW"]["classification"],
            "COMPATIBLE_WITH_VERSIONED_TRANSFORM",
        )

    def test_27_no_directory_concat(self):
        self.assertFalse(self.compat["direct_directory_concatenation"])

    def test_28_supported_odd(self):
        self.assertEqual(self.odd["status"], "PASS")
        self.assertGreaterEqual(len(self.odd["required"]), 10)

    def test_29_out_of_odd(self):
        self.assertEqual(self.out["status"], "PASS_EXPLICIT")

    def test_30_fail_closed(self):
        self.assertEqual(
            self.fail_closed["mappings"]["NO_VALID_EVALUATION"],
            "INVALID_EVALUATION_SAFE_ABORT",
        )

    def test_31_unknown_not_safe(self):
        self.assertFalse(
            self.fail_closed["forbidden"]["unknown_to_safe"]
        )

    def test_32_unresolved_not_downgraded(self):
        self.assertEqual(
            self.fail_closed["mappings"]["PENDING_OR_UNKNOWN_SUPPORT"],
            "UNRESOLVED_NOT_SAFE",
        )

    def test_33_no_runtime_gt(self):
        self.assertFalse(self.final["runtime_gt_used"])

    def test_34_track_manager_interface(self):
        self.assertEqual(
            self.runtime["components"]["TrackManager_Kalman"]["interface"],
            "AVAILABLE",
        )

    def test_35_bdrr1_interface(self):
        self.assertEqual(
            self.runtime["components"]["BDRR1"]["interface"],
            "PASS_DEVELOPMENT",
        )

    def test_36_brir1_interface(self):
        self.assertEqual(
            self.runtime["components"]["BRIR1_router"]["interface"],
            "COMPOSABLE_DEVELOPMENT_ONLY",
        )

    def test_37_static_yopo_interface(self):
        self.assertEqual(
            self.runtime["components"]["static_yopo"]["interface"],
            "AVAILABLE",
        )

    def test_38_h5_budget_plan(self):
        self.assertEqual(
            self.h5["status"], "PASS_FEASIBILITY_PLAN_NOT_EXECUTED"
        )
        self.assertFalse(self.h5["full_selected_system_H5_pass_claimed"])

    def test_39_route_a_rule(self):
        self.assertTrue(self.routes["A"]["eligible"])

    def test_40_route_b_rule(self):
        self.assertTrue(self.routes["B"]["eligible"])

    def test_41_route_c_rule(self):
        self.assertTrue(self.routes["C"]["eligible"])

    def test_42_unique_route(self):
        self.assertEqual(self.routes["selected"], "A")
        self.assertEqual(self.final["route"], "A")

    def test_43_no_training(self):
        self.assertFalse(self.final["long_training_started"])

    def test_44_no_full_derived(self):
        self.assertFalse(self.final["full_static_dataset_built"])
        self.assertEqual(self.pilot["checks"]["derived_samples_written"], 0)

    def test_45_no_internal_test(self):
        self.assertFalse(self.final["internal_test_accessed"])

    def test_46_no_test_blind(self):
        self.assertFalse(self.final["test_accessed"])
        self.assertFalse(self.final["blind_accessed"])

    def test_47_checkpoint_payload_still_non_formal(self):
        path = ROOT / self.archive["checkpoints"][0]["path"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertTrue(payload["non_formal"])

    def test_48_source_unchanged_flags(self):
        self.assertFalse(self.final["source_v3_raw_modified"])
        self.assertFalse(self.final["V1_modified"])
        self.assertFalse(self.final["V2_modified"])
        self.assertTrue(
            self.final["source_manifest_identity"]["V3_matches_frozen"]
        )

    def test_49_report_contract(self):
        self.assertEqual(self.entry["status"], "PASS")
        self.assertEqual(self.pilot["status"], "PASS")
        self.assertEqual(
            self.final["next_allowed_phase"],
            "phase8jqv2_5_mixed_static_yopo_dataset_and_training_readiness",
        )


if __name__ == "__main__":
    unittest.main()
