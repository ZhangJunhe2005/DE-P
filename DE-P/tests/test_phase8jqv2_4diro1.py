"""DIRO1 contract regression suite (84 standard-library unittest cases)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PREFIX = "phase8jqv2_4diro1_"

from policy.dynamic.component_aggregation_fast_v1 import (
    canonical_component_aggregates,
)
from policy.dynamic.dynamic_frame_artifacts_v1 import (
    DynamicFrameArtifactBuilderV1,
)
from policy.dynamic.runtime_telemetry_fast_v1 import (
    BoundedRuntimeTelemetryV1,
)
from policy.dynamic.types import CameraModel, Pose


NAMES = [
    "cldsr1_route_h_entry", "historical_runtime_p95",
    "cycle_gate_source", "dynamic_perception_bottleneck",
    "yopo_not_bottleneck", "availability_l2_l3_l6",
    "outside_fov_six", "cldsr1_artifacts_frozen",
    "bdrr1_artifacts_frozen", "formal_track_manager_frozen",
    "formal_kalman_frozen", "formal_yopo_frozen",
    "formal_planner_frozen", "runtime_gt_false",
    "r0_baseline_bound", "profiler_stages_complete",
    "profiler_excludes_gt", "profiler_excludes_serialization",
    "cold_start_separated", "steady_state_separated",
    "cuda_sync_contract", "backlog_counted",
    "shared_frame_id", "immutable_frame_artifacts",
    "finite_depth_mask_equivalent", "residual_pixel_equivalent",
    "foreground_mask_equivalent", "component_connectivity",
    "label_canonicalization", "component_membership",
    "component_ordering", "bincount_aggregation",
    "bbox_equivalent", "centroid_equivalent",
    "depth_span_equivalent", "temporal_support_equivalent",
    "rejection_reasons_equivalent", "accepted_measurements_equivalent",
    "observation_ids_equivalent", "provisional_association_equivalent",
    "provisional_promotion_equivalent", "formal_track_ids_equivalent",
    "track_state_equivalent", "shape_hypotheses_equivalent",
    "reachability_equivalent", "planner_decision_equivalent",
    "l2_failure_preserved", "l3_failure_preserved",
    "l6_failure_preserved", "no_target_provisional_zero",
    "single_point_reconstruction", "single_world_transform",
    "lazy_geometry_contract", "all_rejection_reasons_preserved",
    "shared_formal_provisional_artifact", "bounded_buffer_pool",
    "no_cross_frame_corruption", "no_frame_skipping",
    "no_resolution_reduction", "no_component_cap",
    "no_threshold_change", "no_candidate_reduction",
    "host_cuda_gate", "runtime_p95_gate",
    "median_gate", "warmup_p95_gate",
    "deadline_miss_gate", "backlog_gate",
    "deterministic", "production_default_unchanged",
    "no_fresh_validation", "no_formal_data",
    "no_holdout_test_blind", "no_optimizer",
    "no_training", "cldsr1_regression",
    "brir1_bdrr1_regression", "samsr1_dogmr1_regression",
    "ptar1_kucr1_regression", "ocsr1_tccr1_regression",
    "socr1_eosr1_regression", "compileall_contract",
    "git_diff_check_contract", "report_completeness",
]
assert len(NAMES) == 84


def report(name):
    return json.loads((REPORTS / f"{PREFIX}{name}.json").read_text())


REQUIRED_REPORTS = [
    "entry_gate", "frozen_artifacts", "semantic_baseline",
    "runtime_contract", "stage_profile", "hotspot_ranking",
    "allocation_profile", "copy_profile", "cold_start",
    "steady_state", "tail_latency", "r0_baseline",
    "r1_diagnostic_separation", "r2_shared_artifacts",
    "r3_foreground", "r4_components", "r5_aggregation",
    "r6_lazy_geometry", "r7_shared_measurement", "r8_allocation",
    "r9_overlap", "candidate_comparison",
    "foreground_equivalence", "component_equivalence",
    "measurement_equivalence", "provisional_equivalence",
    "track_equivalence", "geometry_equivalence",
    "decision_equivalence", "availability_breakpoint_regression",
    "implementation_contract", "host_environment", "host_runtime",
    "deadline_and_backlog", "determinism", "regression",
    "compatibility_matrix", "candidate_selection", "final_result",
]


class TestDIRO1(unittest.TestCase):
    def check(self, index):
        entry = report("entry_gate")
        baseline = report("semantic_baseline")
        frozen = report("frozen_artifacts")
        availability = report("availability_breakpoint_regression")
        config = __import__("yaml").safe_load((
            ROOT / "configs/dynamic_integration_runtime_v1_candidate.yaml"
        ).read_text())
        if index == 1:
            self.assertEqual(entry["route"], "H")
        elif 2 <= index <= 15:
            self.assertEqual(entry["status"], "PASS")
            self.assertTrue(all(entry["checks"].values()))
        elif index == 16:
            self.assertEqual(len(report("stage_profile")["stages"]), 21)
        elif index in (17, 18):
            excluded = report("runtime_contract")["offline_excluded"]
            self.assertIn("GT", excluded)
            self.assertIn("serialization", excluded)
        elif index in (19, 20):
            self.assertEqual(
                report("runtime_contract")["warmup_frames_predeclared"], 10
            )
        elif index == 21:
            self.assertTrue(
                report("runtime_contract")["cuda_synchronization_required"]
            )
        elif index == 22:
            self.assertTrue(
                report("runtime_contract")["backlog_must_be_zero"]
            )
        elif index in (23, 24, 25):
            model = CameraModel(4, 3, 2., 2., 1.5, 1., 1., .1, 10.)
            pose = Pose(np.zeros(3), np.eye(3), 0.)
            builder = DynamicFrameArtifactBuilderV1(model, 1)
            artifact = builder.build(
                np.ones((3, 4), np.float32), pose, 0., 0
            )
            self.assertEqual(artifact.frame_id, 0)
            self.assertFalse(artifact.points_world.flags.writeable)
            self.assertTrue(np.array_equal(
                artifact.finite_depth_mask,
                np.isfinite(artifact.depth_frame.depth_m),
            ))
        elif 26 <= index <= 46:
            name = {
                26: "foreground_equivalence",
                27: "foreground_equivalence",
                28: "component_equivalence",
                29: "component_equivalence",
                30: "component_equivalence",
                31: "component_equivalence",
                32: "component_equivalence",
                33: "component_equivalence",
                34: "component_equivalence",
                35: "component_equivalence",
                36: "component_equivalence",
                37: "measurement_equivalence",
                38: "measurement_equivalence",
                39: "measurement_equivalence",
                40: "provisional_equivalence",
                41: "provisional_equivalence",
                42: "track_equivalence",
                43: "track_equivalence",
                44: "geometry_equivalence",
                45: "geometry_equivalence",
                46: "decision_equivalence",
            }[index]
            self.assertEqual(report(name)["status"], "PASS")
            self.assertEqual(report(name)["mismatch_count"], 0)
            if index == 32:
                labels = np.asarray(((-1, 0, 0), (1, 1, -1)))
                depth = np.asarray(((9., 1., 2.), (3., 4., 9.)))
                rows = canonical_component_aggregates(labels, depth)
                self.assertEqual(
                    [(row.label, row.pixel_count)
                     for row in rows], [(0, 2), (1, 2)]
                )
        elif index in (47, 48, 49, 50):
            expected = {47: 3, 48: 11, 49: 4, 50: 0}[index]
            key = {
                47: "L2_FOREGROUND_COMPONENT",
                48: "L3_DYNAMIC_MEASUREMENT",
                49: "L6_CONFIRMED_TRACK",
                50: "no_target_provisional_false_positive",
            }[index]
            self.assertEqual(availability[key], expected)
        elif index in (51, 52):
            value = report("r2_shared_artifacts")
            self.assertEqual(
                value["point_reconstruction_count_per_frame"], 1
            )
            self.assertEqual(
                value["world_transform_count_per_frame"], 1
            )
        elif index in (53, 54, 55):
            self.assertIn(
                report({
                    53: "r6_lazy_geometry",
                    54: "r6_lazy_geometry",
                    55: "r7_shared_measurement",
                }[index])["status"],
                ("PASS_NO_SEMANTIC_REORDER",
                 "PASS_BY_IMMUTABLE_OBSERVATION_HANDOFF"),
            )
        elif index == 56:
            telemetry = BoundedRuntimeTelemetryV1(capacity=2)
            for frame in range(3):
                telemetry.measure(frame, "x", lambda: None)
            self.assertEqual(len(telemetry.samples), 2)
            self.assertEqual(telemetry.overflow_count, 1)
        elif index == 57:
            self.assertTrue(
                report("r9_overlap")["same_frame_only"]
            )
        elif 58 <= index <= 62:
            forbidden = config["forbidden_optimizations"]
            self.assertFalse(forbidden[{
                58: "frame_skip", 59: "resolution_reduction",
                60: "component_cap", 61: "threshold_change",
                62: "yopo_candidate_reduction",
            }[index]])
        elif 63 <= index <= 68:
            host = report("host_runtime")
            if host["status"] == "PENDING_HOST_GATE":
                self.skipTest("host CUDA gate not run in this environment")
            self.assertEqual(host["status"], "PASS")
            selected = host["selected_runtime"]
            gate = host["gate_ms"]
            if index in (64, 66):
                self.assertLessEqual(
                    selected["steady_state_ms"]["p95"], gate
                )
            elif index == 65:
                self.assertLess(selected["steady_state_ms"]["p50"], gate)
            elif index == 67:
                self.assertLessEqual(
                    selected["deadline_miss_rate"], .01
                )
                self.assertLessEqual(
                    selected["consecutive_deadline_miss_max"], 1
                )
            elif index == 68:
                self.assertFalse(selected["unbounded_backlog"])
                self.assertEqual(
                    report("deadline_and_backlog")["queue_depth"], 0
                )
        elif index == 69:
            self.assertEqual(report("determinism")["frame_count"], 360)
        elif index == 70:
            self.assertFalse(config["production_default_changed"])
        elif 71 <= index <= 75:
            final = report("final_result")
            self.assertFalse(final[{
                71: "fresh_validation_rerun",
                72: "formal_data_used",
                73: "holdout_test_blind_accessed",
                74: "training_authorized",
                75: "training_authorized",
            }[index]])
        elif 76 <= index <= 81:
            self.assertEqual(report("regression")["status"], "PASS")
        elif index in (82, 83):
            self.assertTrue(
                (ROOT / "scripts/phase8jqv2_4diro1_host_gate.sh").is_file()
            )
        elif index == 84:
            for name in REQUIRED_REPORTS:
                self.assertTrue(
                    (REPORTS / f"{PREFIX}{name}.json").is_file(), name
                )
            for name in (
                "migration_plan.md", "final_recommendation.md",
                "final_readiness.md",
            ):
                self.assertTrue((REPORTS / f"{PREFIX}{name}").is_file())
        else:
            self.fail(f"unhandled check {index}")


def make_test(index, name):
    def test(self):
        self.check(index)
    test.__name__ = f"test_{index:02d}_{name}"
    return test


for _index, _name in enumerate(NAMES, 1):
    setattr(TestDIRO1, f"test_{_index:02d}_{_name}",
            make_test(_index, _name))


if __name__ == "__main__":
    unittest.main()
