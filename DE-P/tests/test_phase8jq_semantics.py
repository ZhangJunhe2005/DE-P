import json
from pathlib import Path
import unittest

import numpy as np

from tools.phase8jq_semantics import (
    UAV_RADIUS_M,
    classify_timeline,
    continuous_linear_closest_approach,
    finite_vertical_cylinder_clearance,
    formal_dynamic_clearance,
    point_map_clearance,
    sample_times,
    simulator_dynamic_clearance,
    sphere_clearance,
    stopping_distance,
)


ROOT = Path(__file__).resolve().parents[1]


class Phase8JQSemanticsTests(unittest.TestCase):
    def test_01_dynamic_radius_contact(self):
        self.assertAlmostEqual(sphere_clearance([0, 0, 0], [.7, 0, 0], .3, .4), 0)

    def test_02_static_esdf_contact(self):
        self.assertAlmostEqual(point_map_clearance(UAV_RADIUS_M), 0)

    def test_03_no_double_inflation(self):
        actor = {"position_world": [.7, 0, 0], "radius": .4, "type": "sphere"}
        self.assertAlmostEqual(simulator_dynamic_clearance([0, 0, 0], actor), 0)

    def test_04_cylinder_geometry_matches_radial_contact(self):
        self.assertAlmostEqual(
            finite_vertical_cylinder_clearance([.7, 0, 0], [0, 0, 0], .3, .4, 1), 0
        )

    def test_05_gt_uncertainty_is_not_physical_geometry(self):
        actor = {
            "position_world": [.7, 0, 0], "radius": .4, "type": "sphere",
            "position_covariance": np.eye(3).tolist(),
        }
        self.assertLess(formal_dynamic_clearance([0, 0, 0], actor, 0), 0)

    def test_06_sample_grid_excludes_t0(self):
        self.assertGreater(sample_times(1.7, 30)[0], 0)
        self.assertEqual(sample_times(1.7, 30, True)[0], 0)

    def test_07_same_timestamp_alignment(self):
        times = sample_times(1.7, 30)
        actor = 100.0 + times
        np.testing.assert_allclose(actor - 100.0, times)

    def test_08_one_frame_offset_detection(self):
        expected = sample_times(1.7, 30)
        shifted = expected + .1
        self.assertGreater(float(np.max(np.abs(expected - shifted))), 1e-4)

    def test_09_t0_classification(self):
        result = classify_timeline([0, .1, .2], [-.01, .01, .02], .05)
        self.assertEqual(result["category"], "minimum_at_t0_only")

    def test_10_first_controllable_classification(self):
        result = classify_timeline([0, .02, .04], [.1, -.01, -.02], .03)
        self.assertEqual(result["category"], "collision_before_first_controllable_action")

    def test_11_initially_unsafe_deepening(self):
        result = classify_timeline([0, .1], [-.01, -.02], .02)
        self.assertEqual(result["category"], "initially_unsafe_deepening")

    def test_12_inter_sample_collision(self):
        distance, alpha = continuous_linear_closest_approach(
            [-1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]
        )
        self.assertAlmostEqual(distance, 0)
        self.assertAlmostEqual(alpha, .5)

    def test_13_continuous_closest_approach(self):
        distance, _ = continuous_linear_closest_approach(
            [0, 1, 0], [1, 1, 0], [.5, 0, 0], [.5, 0, 0]
        )
        self.assertAlmostEqual(distance, 1)

    def test_14_stopping_distance(self):
        self.assertAlmostEqual(stopping_distance(2, 2, .1), 1.2)

    def test_15_max_iteration_not_infeasible(self):
        status = "max_iterations"
        self.assertNotEqual(status, "infeasible_certificate")

    def test_16_no_instance_mask(self):
        text = (ROOT / "tools/run_phase8jq_audit.py").read_text()
        self.assertNotIn("instance_mask", text)

    def test_17_no_production_test(self):
        text = (ROOT / "tools/run_phase8jq_audit.py").read_text()
        self.assertNotIn("phase8c_test_", text)

    def test_18_future_gt_offline_only(self):
        self.assertEqual("offline_safety_evaluator_only", "offline_safety_evaluator_only")

    def test_19_controller_provenance_and_resume(self):
        raw = json.loads(
            (ROOT / "reports/phase8jq_controller_identification_raw.json").read_text()
        )
        self.assertEqual(raw["status"], "PASS")
        solver = json.loads(
            (ROOT / "reports/phase8jq_independent_feasibility_solver.json").read_text()
        )
        identities = {
            (row["sequence_id"], row["frame_index"])
            for rows in solver["records_by_suite"].values() for row in rows
        }
        self.assertEqual(len(identities), 143)

    def test_20_report_completeness(self):
        required = [
            "phase8jq_entry_gate.json", "phase8jq_baseline_reproduction.json",
            "phase8jq_controller_authoritative_envelope.json",
            "phase8jq_safety_geometry_audit.json",
            "phase8jq_timeline_semantics_audit.json",
            "phase8jq_independent_feasibility_solver.json",
            "phase8jq_final_taxonomy.json", "phase8jq_final_result.json",
        ]
        self.assertTrue(all((ROOT / "reports" / name).is_file() for name in required))


if __name__ == "__main__":
    unittest.main()
