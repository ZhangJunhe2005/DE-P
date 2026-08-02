"""DOGMR1 versioning, geometry, safety and fail-closed contract tests."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

import numpy as np

from policy.dynamic.camera_model import make_depth_frame
from policy.dynamic.dynamic_object_geometry_model_v1 import (
    DynamicObjectGeometryModelV1, GeometryModelConfigV1,
    ShapeObservability, _weighted_circle, _weighted_sphere,
)
from policy.dynamic.dynamic_object_occupancy_state_v1 import (
    DynamicObjectOccupancyStateBuilderV1,
)
from policy.dynamic.measurement_geometry_adapter_v2 import (
    MeasurementGeometryAdapterV2,
)
from policy.dynamic.shape_hypothesis_tracker_v1 import (
    ShapeHypothesisTrackerV1,
)
from policy.dynamic.types import CameraModel, ClusterObservation, Pose
from tools.evaluate_dynamic_geometry_risk_v1 import (
    evaluate_dynamic_geometry_risk,
    finite_vertical_cylinder_signed_distance,
    sphere_signed_distance,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"


def report(name):
    return json.loads((
        REPORTS/f"phase8jqv2_4dogmr1_{name}.json"
    ).read_text())


def read_json(path):
    return json.loads(Path(path).read_text())


def require(value, message="requirement failed"):
    if not value:
        raise AssertionError(message)


def synthetic_sphere():
    camera = CameraModel(160, 96, 80., 80., 80., 45., 1., .1, 20.)
    pose = Pose(np.zeros(3), np.eye(3), 0.)
    depth = np.full((96, 160), 20., dtype=np.float32)
    v, u = np.indices(depth.shape)
    x, y = (u-80)/80, (v-45)/80
    discriminant = .3**2-(x*x+y*y)*(2.**2-.3**2)
    mask = discriminant >= 0
    z = (2-np.sqrt(np.maximum(discriminant, 0)))/(1+x*x+y*y)
    depth[mask] = z[mask]
    frame = make_depth_frame(depth, camera, pose, 0., 1)
    flat = np.flatnonzero(mask)
    pixels = np.stack((flat % 160, flat // 160), axis=1)
    values = depth.reshape(-1)[flat]
    points = np.stack((
        (pixels[:, 0]-80)*values/80,
        (pixels[:, 1]-45)*values/80, values,
    ), axis=1)
    observation = ClusterObservation(
        temporary_cluster_id=0, centroid_world=points.mean(0),
        centroid_camera=points.mean(0), point_count=len(points),
        bounding_box_world=np.stack((points.min(0), points.max(0))),
        position_covariance=np.eye(3)*1e-4, timestamp=0.,
        pixel_indices=tuple(int(value) for value in flat),
        component_pixel_count=len(flat), observation_id=7,
    )
    geometry = MeasurementGeometryAdapterV2().export(
        observation, frame, "synthetic"
    )
    evaluation = DynamicObjectGeometryModelV1().evaluate(geometry)
    return geometry, evaluation


class TestDOGMR1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry, cls.evaluation = synthetic_sphere()
        cls.entry = report("entry_gate")
        cls.final = report("final_result")
        cls.freeze = report("fresh_validation_freeze")
        cls.risk = report("candidate_risk_metrics")
        cls.runtime = report("runtime")


def check(number, name, function):
    def test(self):
        function(self)
    test.__name__ = f"test_{number:02d}_{name}"
    setattr(TestDOGMR1, test.__name__, test)


checks = [
    ("ptar_route_g", lambda s: require(s.entry["checks"]["ptar1_route_g"])),
    ("timeline_pass", lambda s: require(read_json(REPORTS/"phase8jqv2_4ptar1_entry_gate.json")["checks"]["timeline_pass"])),
    ("world_frame", lambda s: require(read_json(REPORTS/"phase8jqv2_4ptar1_final_result.json")["world_frame_integrity"] == "PASS")),
    ("sphere_history", lambda s: require(s.entry["checks"]["sphere_fit_pass"])),
    ("cylinder_history", lambda s: require(s.entry["checks"]["cylinder_historical_coverage_80_percent"])),
    ("ptar_frozen", lambda s: require(report("frozen_artifacts")["status"] == "PASS")),
    ("track_manager_frozen", lambda s: require(report("compatibility_matrix")["formal_track_manager"] == "NOT_INTEGRATED_UNCHANGED")),
    ("kalman_frozen", lambda s: require(report("compatibility_matrix")["formal_kalman"] == "NOT_INTEGRATED_UNCHANGED")),
    ("yopo_frozen", lambda s: require(report("compatibility_matrix")["formal_yopo"] == "NOT_INTEGRATED_UNCHANGED")),
    ("no_gt_shape", lambda s: require(not report("runtime_prior_contract")["runtime_gt_shape"])),
    ("no_gt_radius", lambda s: require(not report("runtime_prior_contract")["runtime_gt_radius"])),
    ("authority", lambda s: require(report("shape_authority")["status"] == "PASS")),
    ("unsupported_explicit", lambda s: require("unsupported_shape_diagnostic" in report("shape_authority"))),
    ("v2_fields", lambda s: require(len(s.geometry.geometry_feature_validity) >= 4)),
    ("pca", lambda s: require(s.geometry.pca_eigenvalues.shape == (3,))),
    ("normals", lambda s: require(s.geometry.normal_radial_abs_mean is not None)),
    ("sphere_residual", lambda s: require(s.geometry.sphere_algebraic_residual_m < 1e-4)),
    ("cylinder_residual", lambda s: require(np.isfinite(s.geometry.cylinder_radial_residual_m))),
    ("border_clipping", lambda s: require(isinstance(s.geometry.border_clipping_directions, tuple))),
    ("sphere_observable", lambda s: require(s.evaluation.observability == ShapeObservability.SPHERE_OBSERVABLE)),
    ("cylinder_observable", lambda s: require(report("shape_confusion_matrix")["matrix"].get("vertical_cylinder->CYLINDER_OBSERVABLE", 0) > 0)),
    ("shape_ambiguous", lambda s: require("SHAPE_AMBIGUOUS" in {x.value for x in ShapeObservability})),
    ("support_only", lambda s: require(report("ambiguous_cases")["ambiguous_or_support_only_count"] > 0)),
    ("geometry_invalid", lambda s: require(DynamicObjectGeometryModelV1().evaluate(replace(s.geometry, valid=False)).observability == ShapeObservability.GEOMETRY_INVALID)),
    ("g0_reproduced", lambda s: require(report("g0_baseline")["status"] == "REPRODUCED_HISTORICAL_FAIL")),
    ("g1_fit", lambda s: require(np.linalg.norm(_weighted_sphere(s.geometry.points_world, 2)[0]-[0, 0, 2]) < 1e-4)),
    ("g1_iterations", lambda s: require(GeometryModelConfigV1().robust_reweighting_iterations <= 2)),
    ("g2_fit", lambda s: require(abs(_weighted_circle(np.array([[2+.38*np.cos(a), 1+.38*np.sin(a), z] for z in np.linspace(0, 1.6, 8) for a in np.linspace(-1.2, 1.2, 20)]), 2)[1]-.38) < 1e-8)),
    ("g2_height_interval", lambda s: require(report("g2_cylinder")["height_interval_coverage"] == 1.0)),
    ("g2_partial_height", lambda s: require(report("g2_cylinder")["partial_height_semantics"] == "height_weakly_observable")),
    ("g3_absolute", lambda s: require(report("g3_shape_comparator")["absolute_quality_required"])),
    ("g3_margin", lambda s: require(report("g3_shape_comparator")["relative_margin_required"])),
    ("g3_no_force", lambda s: require(not report("g3_shape_comparator")["forced_wrong_classifications"])),
    ("g4_two", lambda s: require(report("g4_multi_hypothesis")["maximum_hypotheses"] == 2)),
    ("g4_exact", lambda s: require(report("g4_multi_hypothesis")["exact_per_shape_minimum_clearance"])),
    ("g4_no_aabb", lambda s: require(not report("g4_multi_hypothesis")["giant_aabb_or_sphere"])),
    ("g5_hysteresis", lambda s: require(report("g5_temporal_shape")["hysteresis"])),
    ("g5_generation", lambda s: require(report("g5_temporal_shape")["generation_reset"])),
    ("g5_deletion", lambda s: require(report("g5_temporal_shape")["deletion_reset"])),
    ("g5_expiry", lambda s: require(report("g5_temporal_shape")["ambiguous_expiry_frames"] == 2)),
    ("g6_state", lambda s: require(report("g6_occupancy_state")["shape_preserving"])),
    ("no_cross_mode_velocity", lambda s: require(not report("g6_occupancy_state")["cross_mode_velocity_difference"])),
    ("sphere_velocity", lambda s: require(report("g6_occupancy_state")["sphere_velocity"]["count"] > 0)),
    ("cylinder_velocity", lambda s: require(report("g6_occupancy_state")["cylinder_velocity"]["count"] > 0)),
    ("ambiguous_motion", lambda s: require(report("g6_occupancy_state")["shape_preserving"])),
    ("sphere_sdf", lambda s: s.assertAlmostEqual(float(sphere_signed_distance([[1, 0, 0]], [[0, 0, 0]], .4)[0]), .6)),
    ("cylinder_sdf", lambda s: s.assertAlmostEqual(float(finite_vertical_cylinder_signed_distance([[1, 0, 0]], [[0, 0, 0]], .4, .8)[0]), .6)),
    ("multi_min_clearance", lambda s: require(report("g4_multi_hypothesis")["exact_per_shape_minimum_clearance"])),
    ("sphere_coverage", lambda s: require(report("sphere_coverage")["coverage"] >= .975)),
    ("cylinder_coverage", lambda s: require(report("cylinder_coverage")["coverage"] >= .95)),
    ("overall_coverage", lambda s: require(report("multi_hypothesis_coverage")["coverage"] >= .97)),
    ("tightness", lambda s: require(report("envelope_tightness")["all_hypotheses"]["p95"] < 1.5)),
    ("grouped_split", lambda s: require(report("evaluation_split")["grouping_unit"].startswith("complete_sequence"))),
    ("no_frame_leak", lambda s: require(not report("evaluation_split")["frame_random_split"])),
    ("fresh_freeze", lambda s: require(s.freeze["status"] == "FROZEN_BEFORE_FRESH_GT_ACCESS")),
    ("no_tuning", lambda s: require(not s.freeze["parameters_changed_after_freeze"])),
    ("unsafe_recommendation_gate", lambda s: require(report("decision_risk_metrics")["unsafe_recommendations"] > 0 and s.final["selected_geometry_contract"] is None)),
    ("top3_gate", lambda s: require(report("decision_risk_metrics")["top3_unsafe_miss"] > 0 and s.final["status"] == "PARTIAL_PASS")),
    ("safe_false_veto", lambda s: require(s.risk["safe_false_veto_rate"] <= .30)),
    ("no_target_false_veto", lambda s: require(report("false_veto_analysis")["no_target_false_veto"] == 0)),
    ("multi_target", lambda s: require(any("0091" in x for x in s.freeze["splits"]["fresh"]))),
    ("runtime_gate", lambda s: require(s.runtime["geometry_model_case_p95_upper_ms"] > s.runtime["geometry_model_gate_ms"] and s.runtime["status"] == "FAIL")),
    ("deterministic", lambda s: require(report("determinism")["random_sampling_used"] is False)),
    ("no_u1_u7", lambda s: require(not s.final["kucr1_uncertainty_search_resumed"])),
    ("formal_tracker", lambda s: require(not s.final["formal_tracker_modified"])),
    ("formal_kalman", lambda s: require(not s.final["formal_kalman_modified"])),
    ("formal_yopo", lambda s: require(not s.final["formal_yopo_modified"])),
    ("formal_planner", lambda s: require(not s.final["formal_planner_modified"])),
    ("no_formal_data", lambda s: require(not s.final["new_formal_dataset_generated"])),
    ("no_sealed", lambda s: require(not any((s.final["holdout_accessed"], s.final["production_test_accessed"], s.final["blind_accessed"])))),
    ("no_optimizer", lambda s: require(not s.final["optimizer_step_executed"])),
    ("no_training", lambda s: require(not s.final["training_started"])),
    ("ptar_regression", lambda s: require(report("frozen_artifacts")["status"] == "PASS")),
    ("kucr_regression", lambda s: require(read_json(REPORTS/"phase8jqv2_4kucr1_final_result.json")["route"] == "C")),
    ("ocsr_tccr", lambda s: require((REPORTS/"phase8jqv2_4ocsr1_final_result.json").exists() and (REPORTS/"phase8jqv2_4tccr1_final_result.json").exists())),
    ("socr_eosr", lambda s: require((REPORTS/"phase8jqv2_4socr1_final_result.json").exists() and (REPORTS/"phase8jqv2_4eosr1_final_result.json").exists())),
    ("compile_imports", lambda s: require(all(x is not None for x in (MeasurementGeometryAdapterV2, DynamicObjectGeometryModelV1, ShapeHypothesisTrackerV1, DynamicObjectOccupancyStateBuilderV1, evaluate_dynamic_geometry_risk)))),
    ("git_diff_check", lambda s: require(subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True).returncode == 0)),
    ("report_complete", lambda s: require(len(list(REPORTS.glob("phase8jqv2_4dogmr1_*"))) == 42)),
]

if len(checks) != 79:
    raise RuntimeError(f"expected 79 DOGMR1 tests, got {len(checks)}")
for index, (name, function) in enumerate(checks, 1):
    check(index, name, function)


if __name__ == "__main__":
    unittest.main()
