"""PTAR1 reference-origin alignment and fail-closed contract tests."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from policy.dynamic.camera_model import make_depth_frame
from policy.dynamic.measurement_geometry_adapter_v1 import (
    DynamicMeasurementGeometryV1, FORBIDDEN_RUNTIME_FIELDS,
    MeasurementGeometryAdapterV1,
)
from policy.dynamic.reference_aligned_center_tracker_v1 import (
    ReferenceAlignedCenterTrackerV1,
)
from policy.dynamic.tracking_collision_reference_bridge_v1 import (
    ReferenceObservability, TrackingCollisionReferenceBridgeV1,
    fixed_radial_shift,
)
from policy.dynamic.types import CameraModel, ClusterObservation, Pose


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"


def load(name):
    return json.loads((
        REPORTS/f"phase8jqv2_4ptar1_{name}.json"
    ).read_text())


def require(value, message="requirement failed"):
    if not value:
        raise AssertionError(message)


def synthetic():
    model = CameraModel(160, 96, 80., 80., 80., 45., 1., .1, 20.)
    pose = Pose(np.zeros(3), np.eye(3), 0.)
    depth = np.full((96, 160), 20., dtype=np.float32)
    v, u = np.indices(depth.shape)
    x, y = (u-80)/80, (v-45)/80
    discriminant = .3**2-(x*x+y*y)*(2.**2-.3**2)
    mask = discriminant >= 0
    z = (2-np.sqrt(np.maximum(discriminant, 0)))/(1+x*x+y*y)
    depth[mask] = z[mask]
    frame = make_depth_frame(depth, model, pose, 0., 1)
    flat = np.flatnonzero(mask)
    pixels = np.stack((flat % 160, flat // 160), axis=1)
    values = depth.reshape(-1)[flat]
    points = np.stack((
        (pixels[:, 0]-80)*values/80,
        (pixels[:, 1]-45)*values/80, values,
    ), axis=1)
    observation = ClusterObservation(
        temporary_cluster_id=0,
        centroid_world=points.mean(0),
        centroid_camera=points.mean(0),
        point_count=len(points),
        bounding_box_world=np.stack((points.min(0), points.max(0))),
        position_covariance=np.eye(3)*1e-4,
        timestamp=0.,
        pixel_indices=tuple(int(value) for value in flat),
        component_pixel_count=len(flat),
        observation_id=7,
    )
    geometry = MeasurementGeometryAdapterV1().export(observation, frame)
    evidence = TrackingCollisionReferenceBridgeV1().evaluate_geometry(geometry)
    return frame, observation, geometry, evidence


class TestPhase8JPTAR1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.observation, cls.geometry, cls.evidence = synthetic()
        cls.entry = load("entry_gate")
        cls.final = load("final_result")


def check(number, name):
    def decorate(function):
        setattr(
            TestPhase8JPTAR1,
            f"test_{number:02d}_{name}",
            function,
        )
        return function
    return decorate


@check(1, "kucr1_route_c_entry")
def _(self): require(self.entry["checks"]["kucr1_route_c"])


@check(2, "timeline_pass")
def _(self): require(self.entry["checks"]["timeline_pass"])


@check(3, "no_double_prediction")
def _(self): require(self.entry["checks"]["no_double_prediction"])


@check(4, "world_frame_pass")
def _(self): require(self.entry["checks"]["world_frame_pass"])


@check(5, "reference_mismatch_entry")
def _(self): require(self.entry["checks"]["reference_point_fail"])


@check(6, "kucr1_artifacts_frozen")
def _(self): require(not self.final["kucr1_artifacts_modified"])


@check(7, "ocsr1_adapter_v2_frozen")
def _(self): require(not self.final["ocsr1_adapter_v2_modified"])


@check(8, "track_manager_frozen")
def _(self): require(not self.final["TrackManager_algorithm_modified"])


@check(9, "kalman_qr_frozen")
def _(self):
    require(not self.final["kalman_process_model_modified"])
    require(not self.final["kalman_measurement_model_modified"])


@check(10, "yopo_frozen")
def _(self):
    require(not self.final["static_yopo_network_modified"])
    require(not self.final["static_yopo_checkpoint_modified"])


@check(11, "radius_provenance")
def _(self):
    report = load("radius_provenance")
    require(report["candidate_source"] == "fixed_contract_prior")
    require(report["legacy_adapter_source"] == "implicit_default")


@check(12, "no_runtime_gt_radius")
def _(self): require(not self.final["runtime_gt_radius_used"])


@check(13, "actor_shape_contract")
def _(self):
    report = load("actor_shape_contract")
    require("vertical_cylinder" in report["development_shapes"]["phase8c_train"])
    require(not report["sphere_center_fit_universal"])


@check(14, "measurement_geometry_fields")
def _(self):
    names = {item.name for item in fields(DynamicMeasurementGeometryV1)}
    require({"points_camera", "points_world", "pixels_uv", "depth_values_m"} <= names)
    require(self.geometry.valid and self.geometry.point_count > 0)


@check(15, "no_gt_measurement_field")
def _(self):
    names = {item.name for item in fields(DynamicMeasurementGeometryV1)}
    require(not names & FORBIDDEN_RUNTIME_FIELDS)


@check(16, "observation_model_version")
def _(self):
    text = (REPORTS/"phase8jqv2_4ptar1_observation_model.md").read_text()
    require("z_surface" in text and "Sigma_center" in text)


@check(17, "r0_reproduction")
def _(self):
    report = load("r0_surface_raw")
    require(abs(report["historical_kucr1"]["median"]-.3191826030835568) < 1e-12)


@check(18, "r1_aggregate_diagnostic")
def _(self): require(load("r1_fixed_radial")["status"] == "DIAGNOSTIC_NOT_SELECTED")


@check(19, "r1_edge_case_detection")
def _(self): require(load("edge_case_analysis")["edge"]["R0"]["count"] > 0)


@check(20, "fixed_shift_not_selected")
def _(self): require(not load("candidate_selection")["R1_selected"])


@check(21, "r2_sphere_fit")
def _(self):
    require(self.evidence.observability == ReferenceObservability.CENTER_OBSERVABLE)
    require(np.linalg.norm(self.evidence.center_estimate_world-[0, 0, 2]) < 1e-5)


@check(22, "r2_robust_residual")
def _(self): require(self.evidence.fit_residual_m < 1e-5)


@check(23, "r2_condition_number")
def _(self): require(self.evidence.fit_condition_number < 2e6)


@check(24, "r2_observability_rejection")
def _(self):
    invalid = MeasurementGeometryAdapterV1()._invalid(
        self.observation, self.frame, "test"
    )
    evidence = TrackingCollisionReferenceBridgeV1().evaluate_geometry(invalid)
    require(evidence.observability == ReferenceObservability.REFERENCE_UNOBSERVABLE)


@check(25, "r3_center_interval")
def _(self): require(self.evidence.center_interval_world.shape == (2, 3))


@check(26, "r3_interval_contains_valid_center")
def _(self):
    endpoints = self.evidence.center_interval_world
    unit = (self.geometry.cluster_centroid_world
            - self.geometry.camera_position_world)
    unit /= np.linalg.norm(unit)
    projection = (np.asarray([0, 0, 2])-endpoints[0])@unit
    require(-1e-6 <= projection <= np.linalg.norm(endpoints[1]-endpoints[0])+1e-6)


@check(27, "r3_no_gt_runtime")
def _(self): require(not self.evidence.runtime_gt_used)


@check(28, "r4_support_envelope")
def _(self): require(self.evidence.occupancy_model != "none")


@check(29, "r4_actor_coverage")
def _(self):
    report = load("r4_support_envelope")
    require(0 < report["actor_geometry_coverage"] < 1)


@check(30, "r4_bounded_extent")
def _(self):
    require(np.isfinite(self.evidence.support_half_extent_world).all())
    require(np.max(self.evidence.support_half_extent_world) < 3)


@check(31, "r5_velocity_not_surface_velocity")
def _(self):
    tracker = ReferenceAlignedCenterTrackerV1()
    state = tracker.update(1, "0:0", self.evidence)
    require(state.center_velocity_source == "uninitialized_zero")


@check(32, "r5_generation_reset")
def _(self):
    tracker = ReferenceAlignedCenterTrackerV1()
    tracker.update(1, "0:0", self.evidence)
    state = tracker.update(1, "1:1", self.evidence)
    require(state.generation == "1:1" and np.allclose(state.velocity_world, 0))


@check(33, "r5_deletion_reset")
def _(self):
    tracker = ReferenceAlignedCenterTrackerV1()
    tracker.update(1, "0:0", self.evidence)
    require(tracker.delete_missing([]) == (1,))
    require(tracker.predict(1, "0:0", 1.) is None)


@check(34, "r6_fallback_order")
def _(self):
    require(load("r6_hybrid")["fallback_order"] == [
        "geometry_center_fit", "weak_center_interval",
        "surface_support_envelope", "INVALID",
    ])


@check(35, "center_observable")
def _(self): require("CENTER_OBSERVABLE" in load("reference_observability")["classes"])


@check(36, "center_weakly_observable")
def _(self): require("CENTER_WEAKLY_OBSERVABLE" in load("reference_observability")["classes"])


@check(37, "support_only")
def _(self): require("SUPPORT_ONLY" in load("reference_observability")["classes"])


@check(38, "reference_unobservable")
def _(self): require("REFERENCE_UNOBSERVABLE" in load("reference_observability")["classes"])


@check(39, "position_transform")
def _(self):
    shifted = fixed_radial_shift(self.geometry, .3, 1.)
    require(np.linalg.norm(shifted-self.geometry.cluster_centroid_world) > .29)


@check(40, "velocity_transform")
def _(self): require(load("center_velocity_error")["surface_velocity_reused"] is False)


@check(41, "covariance_transform")
def _(self):
    require(np.all(np.linalg.eigvalsh(self.evidence.center_covariance_world) > 0))
    require(not np.array_equal(
        self.evidence.center_covariance_world,
        self.evidence.surface_covariance_world,
    ))


@check(42, "model_uncertainty")
def _(self): require(np.trace(self.evidence.reference_transform_covariance_world) > 0)


@check(43, "radius_uncertainty")
def _(self): require(self.evidence.radius_interval_m == (.18, .42))


@check(44, "edge_left_right")
def _(self): require(load("edge_case_analysis")["edge"]["R0"]["count"] >= 2)


@check(45, "upper_lower")
def _(self):
    report = load("edge_case_analysis")
    require(report["upper_lower"]["scope"] == "historical_physical_control_diagnostic_only")
    require(not report["upper_lower"]["fresh_claimed"])


@check(46, "moving_camera")
def _(self): require(load("edge_case_analysis")["moving_camera"]["R6"]["count"] > 0)


@check(47, "partial_visibility")
def _(self): require("occluded_but_tracked" in load("edge_case_analysis")["partial_visibility"])


@check(48, "multi_target")
def _(self): require("multi_target" in load("scenario_conditioned_error")["fresh"])


@check(49, "grouped_split")
def _(self): require(load("evaluation_split")["grouping_unit"] == "complete_sequence")


@check(50, "no_frame_leakage")
def _(self): require(not load("evaluation_split")["frame_leakage"])


@check(51, "fresh_validation_freeze")
def _(self):
    require(load("fresh_validation_freeze")["status"] == "FROZEN_BEFORE_FRESH_GT_ACCESS")


@check(52, "no_tuning_after_freeze")
def _(self): require(not load("validation_freeze")["fresh_parameters_changed_after_freeze"])


@check(53, "center_median_gate")
def _(self): require(load("center_error")["all_reference_modes"]["p50"] <= .15)


@check(54, "center_p90_gate")
def _(self): require(load("center_error")["all_reference_modes"]["p90"] > .25)


@check(55, "scenario_no_regression")
def _(self):
    require(all(
        value <= .10
        for value in load("scenario_conditioned_error")["scenario_p90_regression"].values()
    ))


@check(56, "occupancy_coverage")
def _(self): require(load("support_coverage")["status"] == "FAIL")


@check(57, "aligned_covariance_reassessment")
def _(self): require(load("aligned_covariance_reassessment")["status"] == "NOT_RUN_REFERENCE_GATE_FAIL")


@check(58, "no_u1_u7_before_alignment")
def _(self): require(not load("uncertainty_next_step")["kucr1_uncertainty_search_may_resume"])


@check(59, "negative_false_veto_zero")
def _(self): require(load("false_veto_analysis")["negative_false_veto"] == 0)


@check(60, "unsafe_recommendation_no_increase_gate")
def _(self): require(not load("unsafe_recommendation_analysis")["hard_requirement_no_increase"])


@check(61, "runtime_gt_false")
def _(self): require(not self.final["runtime_gt_used"])


@check(62, "runtime_bound")
def _(self):
    report = load("runtime")
    require(np.isfinite(report["reference_bridge_case_p95_upper_ms"]))
    require(report["reference_bridge_limit_ms"] == 6.)


@check(63, "deterministic")
def _(self): require(load("determinism")["status"] == "PASS")


@check(64, "formal_tracker_unchanged")
def _(self): require(not self.final["formal_tracker_modified"])


@check(65, "formal_kalman_unchanged")
def _(self): require(not self.final["formal_kalman_modified"])


@check(66, "formal_planner_unchanged")
def _(self): require(not self.final["formal_planner_modified"])


@check(67, "no_new_maps")
def _(self): require(not self.final["new_maps_generated"])


@check(68, "no_formal_dataset")
def _(self): require(not self.final["new_formal_dataset_generated"])


@check(69, "no_holdout_test_blind")
def _(self):
    require(not any((
        self.final["holdout_accessed"],
        self.final["production_test_accessed"],
        self.final["blind_accessed"],
    )))


@check(70, "no_optimizer")
def _(self): require(not self.final["optimizer_step_executed"])


@check(71, "no_training")
def _(self): require(not self.final["training_started"])


@check(72, "natural_eosr1_failure_preserved")
def _(self): require(self.final["natural_eosr1_tracker_gate"] == "FAIL_SEPARATE")


@check(73, "kucr1_regression")
def _(self): require(not load("regression")["kucr1_artifacts_modified"])


@check(74, "ocsr1_tccr1_regression")
def _(self):
    require(not load("regression")["ocsr1_adapter_v2_modified"])
    require(load("historical_artifact_integrity")["status"] == "PASS")


@check(75, "socr1_eosr1_regression")
def _(self): require(self.final["natural_eosr1_tracker_gate"] == "FAIL_SEPARATE")


@check(76, "compileall")
def _(self):
    for path in (
        ROOT/"policy/dynamic/measurement_geometry_adapter_v1.py",
        ROOT/"policy/dynamic/tracking_collision_reference_bridge_v1.py",
        ROOT/"policy/dynamic/reference_aligned_center_tracker_v1.py",
    ):
        compile(path.read_text(), str(path), "exec")


@check(77, "git_diff_check_contract")
def _(self):
    for path in (
        ROOT/"policy/dynamic/measurement_geometry_adapter_v1.py",
        ROOT/"policy/dynamic/tracking_collision_reference_bridge_v1.py",
        ROOT/"policy/dynamic/reference_aligned_center_tracker_v1.py",
    ):
        require("\r" not in path.read_text())


@check(78, "report_completeness")
def _(self):
    required = {
        "entry_gate", "frozen_artifacts", "historical_artifact_integrity",
        "reference_contract", "radius_provenance", "actor_shape_contract",
        "measurement_geometry_availability", "evaluation_split",
        "fresh_validation_freeze", "validation_freeze", "r0_surface_raw",
        "r1_fixed_radial", "r2_geometry_center", "r3_center_interval",
        "r4_support_envelope", "r5_temporal_center", "r6_hybrid",
        "candidate_comparison", "reference_observability", "center_error",
        "center_velocity_error", "support_coverage",
        "scenario_conditioned_error", "edge_case_analysis",
        "aligned_covariance_reassessment", "aligned_normalized_error",
        "uncertainty_next_step", "candidate_risk_regression",
        "false_veto_analysis", "unsafe_recommendation_analysis",
        "implementation_contract", "runtime", "determinism", "regression",
        "candidate_selection", "compatibility_matrix", "final_result",
    }
    require(all(
        (REPORTS/f"phase8jqv2_4ptar1_{name}.json").is_file()
        for name in required
    ))
    require(all((
        (REPORTS/"phase8jqv2_4ptar1_observation_model.md").is_file(),
        (REPORTS/"phase8jqv2_4ptar1_migration_plan.md").is_file(),
        (REPORTS/"phase8jqv2_4ptar1_final_recommendation.md").is_file(),
        (REPORTS/"phase8jqv2_4ptar1_final_readiness.md").is_file(),
    )))


if __name__ == "__main__":
    unittest.main()
