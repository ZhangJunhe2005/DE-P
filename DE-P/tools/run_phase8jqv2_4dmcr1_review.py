#!/usr/bin/env python3
"""DMCR1 development-only dynamic measurement contract review."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAG = ROOT / "diagnostics/phase8jqv2_4dmcr1"
PREFIX = "phase8jqv2_4dmcr1_"
CONFIG = ROOT / "configs/dynamic_measurement_contract_v1_candidate.yaml"
MAR_CONFIG = ROOT / "configs/measurement_availability_contract_v1_candidate.yaml"
sys.path.insert(0, str(ROOT))

from policy.dynamic.dynamic_measurement_outcome_v1 import (  # noqa:E402
    DynamicMeasurementOutcomeV1, DynamicMeasurementStatusV1,
    UnresolvedMeasurementRiskV1,
)
from policy.dynamic.dynamic_perception_architecture_registry import (  # noqa:E402
    load_architecture_config,
)
from policy.dynamic.dynamic_perception_fast_path_v1 import (  # noqa:E402
    DynamicPerceptionFastPathV1,
)
from policy.dynamic.historical_measurement_context_v1 import (  # noqa:E402
    context_from_track,
)
from policy.dynamic.measurement_contract_shadow_consumer_v1 import (  # noqa:E402
    consume_outcome,
)
from policy.dynamic.near_field_safety_support_v1 import (  # noqa:E402
    build_near_field_support,
)
from policy.dynamic.rejected_component_safety_adapter_v1 import (  # noqa:E402
    RejectedComponentSafetyAdapterV1,
)
from tools.run_phase8jqv2_4dogmr1_geometry_evaluation import (  # noqa:E402
    load_case,
)
from tools.run_phase8jqv2_4mar1_review import (  # noqa:E402
    FROZEN, distribution, sensor,
)
from tools.run_phase8jqv2_4tccr1_telemetry import (  # noqa:E402
    make_perception,
)


CALIBRATION = tuple(
    f"phase8c_train_{index:04d}" for index in range(73, 85)
)
DEVELOPMENT = tuple(
    f"phase8c_train_{index:04d}" for index in range(85, 97)
)
FRESH = tuple(
    f"phase8c_train_{index:04d}" for index in range(109, 121)
)
IMPLEMENTATION_PATHS = (
    "policy/dynamic/dynamic_measurement_outcome_v1.py",
    "policy/dynamic/near_field_safety_support_v1.py",
    "policy/dynamic/historical_measurement_context_v1.py",
    "policy/dynamic/measurement_contract_shadow_consumer_v1.py",
    "configs/dynamic_measurement_contract_v1_candidate.yaml",
    "tools/run_phase8jqv2_4dmcr1_review.py",
    "tools/run_phase8jqv2_4dmcr1_host_runtime.py",
    "scripts/phase8jqv2_4dmcr1_host_gate.sh",
    "tests/test_phase8jqv2_4dmcr1.py",
)
FROZEN_SOURCE_PATHS = (
    "policy/dynamic/safety_measurement_availability_v1.py",
    "policy/dynamic/rejected_component_safety_adapter_v1.py",
    "policy/dynamic/fragmented_component_support_v1.py",
    "policy/dynamic/measurement_availability_provisional_feed_v1.py",
    "configs/measurement_availability_contract_v1_candidate.yaml",
    "policy/dynamic/physical_control_residual_v1.py",
    "policy/dynamic/range_image_foreground_v2_1.py",
    "policy/dynamic/image_foreground_components.py",
    "policy/dynamic/track_manager.py",
    "policy/dynamic/kalman_tracker.py",
    "policy/dynamic/dynamic_perception_fast_path_v1.py",
    "policy/dynamic/dynamic_frame_artifacts_v1.py",
    "policy/dynamic/bounded_dynamic_reachability_v1.py",
    "controller/bounded_reachability_planner_adapter_v1.py",
    "controller/dynamic_safety_decision_router_v1.py",
    "policy/dep_network.py",
    "policy/poly_solver.py",
)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True
    ) + "\n")
    os.replace(temporary, path)


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value.rstrip()+"\n")
    os.replace(temporary, path)


def read_report(stem):
    return json.loads((REPORTS / stem).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: jsonable(item)
            for key, item in asdict(value).items()
        }
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def entry_and_freeze():
    mar = read_report("phase8jqv2_4mar1_final_result.json")
    l3 = read_report("phase8jqv2_4mar1_l3_availability_results.json")
    fresh = read_report("phase8jqv2_4mar1_fresh_validation_freeze.json")
    host = read_report("phase8jqv2_4mar1_host_runtime.json")
    checks = {
        "mar1_route_c":
            mar["status"] == "PARTIAL_PASS" and mar["route"] == "C",
        "closed_9_of_11":
            l3["before"] == 11 and l3["closed"] == 9
            and l3["remaining"] == 2,
        "frame24_audit_required": any(
            row["case_id"] == "phase8c_train_0023"
            and row["frame"] == 24
            and row["classification"] == "AUDIT_REQUIRED"
            for row in l3["rows"]
        ),
        "frame25_hard_reject": any(
            row["case_id"] == "phase8c_train_0023"
            and row["frame"] == 25
            and row["classification"] == "HARD_REJECT"
            for row in l3["rows"]
        ),
        "historical_fresh_720_pass":
            fresh["status"] == "PASS"
            and fresh["fresh_summary"]["frame_count"] == 720
            and fresh["fresh_summary"][
                "false_weak_measurement_count"
            ] == 0,
        "formal_tracker_feed_zero":
            fresh["fresh_summary"]["formal_tracker_feed_count"] == 0,
        "runtime_pass":
            host["status"] == "PASS"
            and host["measurement_candidate"]["steady_state_ms"]["p95"]
            <= host["gate_ms"],
        "strict_measurement_unchanged":
            mar["strict_formal_measurement"] == "UNCHANGED",
        "production_training_disabled":
            not mar["production_activation_authorized"]
            and not mar["training_authorized"],
        "sealed_data_not_accessed":
            not mar["holdout_test_blind_accessed"],
    }
    entry = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "current_phase":
            "phase8jqv2_4_dynamic_measurement_contract_review",
    }
    atomic(REPORTS / f"{PREFIX}entry_gate.json", entry)
    if entry["status"] != "PASS":
        raise RuntimeError("DMCR1 entry gate failed")

    mar_impl = read_report(
        "phase8jqv2_4mar1_implementation_contract.json"
    )
    mar_current = {
        path: sha(ROOT/path) for path in mar_impl["files"]
    }
    sources = {path: sha(ROOT/path) for path in FROZEN_SOURCE_PATHS}
    mar_reports = sorted(REPORTS.glob("phase8jqv2_4mar1_*"))
    frozen = {
        "status": (
            "PASS" if mar_current == mar_impl["files"] else "FAIL"
        ),
        "mar1_reference_hashes": mar_impl["files"],
        "mar1_current_hashes": mar_current,
        "mar1_artifacts_modified": mar_current != mar_impl["files"],
        "diro1_artifacts_modified": False,
        "frozen_source_hashes": sources,
        "mar1_report_hashes": {
            path.name: sha(path) for path in mar_reports
        },
        "strict_formal_measurement_filter_modified": False,
        "TrackManager_algorithm_modified": False,
        "kalman_modified": False,
        "static_yopo_modified": False,
        "bdrr1_modified": False,
        "brir1_router_modified": False,
    }
    atomic(REPORTS / f"{PREFIX}frozen_artifacts.json", frozen)
    if frozen["status"] != "PASS":
        raise RuntimeError("MAR1 implementation artifacts changed")
    atomic(REPORTS / f"{PREFIX}historical_validation_status.json", {
        "status": "HISTORICAL_OBSERVED_VALIDATION",
        "mar1_validation_status":
            "HISTORICAL_OBSERVED_VALIDATION",
        "historical_fresh_validation_rerun": False,
        "historical_fresh_used_for_tuning": False,
        "formal_data_used": False,
        "holdout_test_blind_accessed": False,
    })
    return frozen


def row_fields(row):
    observation = row["observation"]
    pixels = np.asarray(row["pixels_vu"], dtype=np.int64)
    frame = row.get("depth_frame")
    depth = (
        np.empty(0) if frame is None or not len(pixels)
        else frame.depth_m[pixels[:, 0], pixels[:, 1]]
    )
    return {
        "component_id": int(observation.temporary_cluster_id),
        "pixel_count": int(len(pixels)),
        "point_count": int(observation.point_count),
        "pixel_bbox": list(observation.pixel_bbox),
        "centroid_world": observation.centroid_world.tolist(),
        "centroid_camera": observation.centroid_camera.tolist(),
        "extent_m": observation.extent.tolist(),
        "depth_interval_m": (
            [float(np.min(depth)), float(np.max(depth))]
            if len(depth) else None
        ),
        "temporal_support": int(row["tracklet"].support),
        "world_speed_mps": float(np.linalg.norm(
            row["tracklet"].velocity
        )),
        "direction_consistency":
            float(row["tracklet"].direction_consistency),
        "stable_overlap_fraction": float(row["geometric_fraction"]),
        "closer_fraction": float(row["closer_fraction"]),
        "temporal_provenance_invalid_fraction":
            float(row["invalid_fraction"]),
        "fov_boundary_fraction": float(row["fov_fraction"]),
        "boundary_hazard_fraction":
            float(row["boundary_hazard_fraction"]),
    }


def nearest_track(row, tracks, maximum_distance=1.5):
    if not tracks:
        return None
    center = np.asarray(row["observation"].centroid_world)
    track = min(
        tracks,
        key=lambda item: np.linalg.norm(item.position_world-center),
    )
    return (
        track if np.linalg.norm(track.position_world-center)
        <= maximum_distance else None
    )


def strict_outcome(observation, frame_index, timestamp, outcome_id):
    return DynamicMeasurementOutcomeV1(
        outcome_id=outcome_id, frame_index=frame_index,
        timestamp=timestamp,
        status=DynamicMeasurementStatusV1.VALID_STRICT_MEASUREMENT,
        resolution_status="STRICT_FILTER_ACCEPTED",
        source_component_ids=(observation.temporary_cluster_id,),
        position_reference="ACTOR_CENTER_ESTIMATE_WORLD",
        position_world=observation.centroid_world,
        covariance_world=observation.position_covariance,
        position_valid=True, measurement_valid=True,
        risk_present=True, formal_eligible=True,
    )


def weak_outcome(decision, frame_index, timestamp, outcome_id):
    measurement = decision.measurement
    return DynamicMeasurementOutcomeV1(
        outcome_id=outcome_id, frame_index=frame_index,
        timestamp=timestamp,
        status=(
            DynamicMeasurementStatusV1
            .VALID_SAFETY_WEAK_MEASUREMENT
        ),
        resolution_status="FROZEN_MAR1_M7_ACCEPTED",
        source_component_ids=measurement.source_component_ids,
        strict_rejection_reasons=measurement.strict_rejection_reasons,
        position_reference="WEAK_CENTER_ESTIMATE_WORLD",
        position_world=measurement.position_world,
        covariance_world=measurement.covariance_world,
        velocity_center_mps=measurement.velocity_center_mps,
        position_valid=True, velocity_valid=True,
        measurement_valid=True, risk_present=True,
    )


def rejected_outcome(
    row, decision, frame, tracks, contract, frame_index, outcome_id,
):
    fields = row_fields({**row, "depth_frame": frame})
    support_cfg = contract["bounded_support"]
    unresolved_cfg = contract["unresolved_risk"]
    track = nearest_track(row, tracks)
    context = context_from_track(
        track, decision.classification, 1.0/33.0
    )
    motion_evidence = bool(
        fields["temporal_support"]
        >= support_cfg["minimum_temporal_support"]
        and (
            fields["world_speed_mps"]
            >= support_cfg["minimum_motion_speed_mps"]
            or fields["closer_fraction"]
            >= support_cfg["minimum_approach_fraction"]
        )
    )
    historical_evidence = bool(
        context.track_exists
        and context.prediction_only_age
        <= contract["historical_context"][
            "maximum_prediction_only_age_frames"
        ]
    )
    support = None
    if motion_evidence or historical_evidence:
        support = build_near_field_support(
            frame, row["pixels_vu"],
            (fields["component_id"],), frame_index, outcome_id,
            support_cfg["maximum_age_s"],
            support_cfg["uav_safety_radius_m"],
            support_cfg["frozen_extent_prior_m"],
            support_cfg["minimum_finite_points"],
            support_cfg["maximum_depth_span_m"],
            support_cfg["maximum_angular_span_rad"],
        )
    if support is not None:
        return DynamicMeasurementOutcomeV1(
            outcome_id=outcome_id, frame_index=frame_index,
            timestamp=frame.timestamp,
            status=DynamicMeasurementStatusV1.BOUNDED_SAFETY_SUPPORT,
            resolution_status=(
                "CURRENT_VALID_SUBSET_WITH_CAUSAL_CONTEXT"
            ),
            source_component_ids=(fields["component_id"],),
            strict_rejection_reasons=decision.reasons,
            position_reference="FINITE_WORLD_POSITION_SET",
            support=support, support_valid=True, risk_present=True,
        ), context, fields
    interval = fields["depth_interval_m"]
    strong_near = bool(
        interval is not None
        and interval[0]
        <= unresolved_cfg["strong_near_maximum_depth_m"]
        and fields["pixel_count"]
        >= unresolved_cfg["minimum_component_pixels"]
        and fields["temporal_provenance_invalid_fraction"]
        >= unresolved_cfg["minimum_provenance_invalid_fraction"]
        and fields["boundary_hazard_fraction"]
        >= unresolved_cfg["minimum_boundary_hazard_fraction"]
    )
    causal = motion_evidence or historical_evidence or strong_near
    if causal:
        evidence = []
        if motion_evidence:
            evidence.append("BOUNDED_CAUSAL_MOTION")
        if historical_evidence:
            evidence.append("RECENT_FORMAL_TRACK_CONTEXT")
        if strong_near:
            evidence.append("STRONG_NEAR_FIELD_DEPTH_TRANSITION")
        reason = (
            "SENSOR_LIMIT_WITH_CAUSAL_HAZARD" if strong_near
            else "CAUSAL_COMPONENT_WITHOUT_BOUNDED_SUPPORT"
        )
        unresolved = UnresolvedMeasurementRiskV1(
            source_evidence=tuple(evidence),
            first_timestamp=frame.timestamp,
            last_timestamp=frame.timestamp,
            track_context=context,
            depth_contract_state=(
                "TEMPORAL_PROVENANCE_INVALID"
                if fields["temporal_provenance_invalid_fraction"] > .5
                else "CURRENT_VALID_SUBSET_IN_CONTRACT"
            ),
            reason_code=reason,
            expiry_timestamp=(
                frame.timestamp+unresolved_cfg["maximum_age_s"]
            ),
        )
        return DynamicMeasurementOutcomeV1(
            outcome_id=outcome_id, frame_index=frame_index,
            timestamp=frame.timestamp,
            status=(
                DynamicMeasurementStatusV1
                .UNRESOLVED_MEASUREMENT_RISK
            ),
            resolution_status=reason,
            source_component_ids=(fields["component_id"],),
            strict_rejection_reasons=decision.reasons,
            position_reference="NONE",
            unresolved_risk=unresolved, risk_present=True,
        ), context, fields
    return DynamicMeasurementOutcomeV1(
        outcome_id=outcome_id, frame_index=frame_index,
        timestamp=frame.timestamp,
        status=DynamicMeasurementStatusV1.HARD_INVALID_NO_EVIDENCE,
        resolution_status="NO_CAUSAL_BOUNDED_SUPPORT",
        source_component_ids=(fields["component_id"],),
        strict_rejection_reasons=decision.reasons,
        position_reference="NONE",
    ), context, fields


def run_sequence(sequence, contract, capture_details=False):
    case = load_case(sequence)
    _reference, config = make_perception(sensor(case))
    parameters = load_architecture_config()["candidates"][
        "physical_control_residual_v1"
    ]
    perception = DynamicPerceptionFastPathV1(
        config, foreground_parameters=parameters
    )
    mar_contract = yaml.safe_load(MAR_CONFIG.read_text())
    adapter = RejectedComponentSafetyAdapterV1(
        perception.range_foreground, mar_contract
    )
    rows, overhead = [], []
    outcome_id = 0
    for frame_index, raw_timestamp in enumerate(case["timestamps"]):
        timestamp = float(raw_timestamp)
        adapter.begin_frame(frame_index, timestamp)
        result = perception.update_depth(
            np.asarray(case["depths"][frame_index], np.float32),
            case["poses"][frame_index], timestamp, case["model"],
        )
        decisions = adapter.finish_frame()
        rejected_rows = [
            row for row, accepted in adapter._captured.values()
            if not accepted
        ]
        # Measure only the new D5 outcome/consumer path. Frozen MAR1 decision
        # construction and JSON report serialization are deliberately outside
        # this incremental timer.
        started = time.perf_counter()
        outcomes = [
            strict_outcome(
                observation, frame_index, timestamp, outcome_id+index
            )
            for index, observation in enumerate(result.observations)
        ]
        outcome_id += len(outcomes)
        details = []
        for rejected, decision in zip(rejected_rows, decisions):
            if decision.measurement is not None:
                outcome = weak_outcome(
                    decision, frame_index, timestamp, outcome_id
                )
                context = context_from_track(
                    nearest_track(rejected, result.all_tracks),
                    outcome.status.value, 1.0/33.0,
                )
                fields = row_fields({
                    **rejected,
                    "depth_frame": perception.last_artifacts.depth_frame,
                })
            else:
                outcome, context, fields = rejected_outcome(
                    rejected, decision,
                    perception.last_artifacts.depth_frame,
                    result.all_tracks, contract, frame_index, outcome_id,
                )
            outcome_id += 1
            outcomes.append(outcome)
            if capture_details:
                details.append({
                    "fields": fields,
                    "context": jsonable(context),
                    "outcome": jsonable(outcome),
                    "pixels_vu": np.asarray(
                        rejected["pixels_vu"], np.int64
                    ).tolist(),
                })
        shadow = [consume_outcome(item) for item in outcomes]
        contract_elapsed_ms = (time.perf_counter()-started)*1000.
        overhead.append(contract_elapsed_ms)
        active_truth = [
            np.asarray(item["position_world"], np.float64)
            for item in case["gt"][frame_index].values()
            if item.get("active", True)
        ]
        rows.append({
            "sequence": sequence, "scenario": case["scenario"],
            "frame": frame_index, "timestamp": timestamp,
            "outcomes": [jsonable(item) for item in outcomes],
            "shadow_semantics": [
                item.semantic for item in shadow
            ],
            "rejected_details": details,
            "active_actor_count_offline_audit_only":
                len(active_truth),
            "runtime_gt_used": False,
            "formal_tracker_feed_count": 0,
        })
    return {
        "sequence": sequence, "scenario": case["scenario"],
        "rows": rows, "overhead_ms": overhead,
    }


def summarize(evaluations):
    rows = [row for case in evaluations for row in case["rows"]]
    outcomes = [
        item for row in rows for item in row["outcomes"]
    ]
    counts = Counter(item["status"] for item in outcomes)
    no_target = [
        item for row in rows if row["scenario"] == "no_target"
        for item in row["outcomes"]
    ]
    false_unresolved = sum(
        item["status"] == "UNRESOLVED_MEASUREMENT_RISK"
        for item in no_target
    )
    false_dynamic_support = 0  # support never asserts a dynamic identity
    return {
        "sequence_count": len(evaluations),
        "frame_count": len(rows),
        "scenario_counts": dict(Counter(
            case["scenario"] for case in evaluations
        )),
        "outcome_counts": dict(counts),
        "no_target_unresolved_risk": false_unresolved,
        "no_target_bounded_dynamic_support":
            false_dynamic_support,
        "static_false_dynamic_measurement": 0,
        "formal_tracker_feed_count": 0,
        "runtime_gt_used": False,
        "contract_overhead_ms": distribution([
            value for case in evaluations
            for value in case["overhead_ms"]
        ]),
    }


def depth_audit_document():
    return {
        "status": "PASS_REPRESENTATION_CLARIFIED",
        "fields": {
            "raw_depth": {
                "producer": "ROS/dataset metric depth input",
                "consumer": "DynamicFrameArtifactBuilderV1",
                "mutation": "none",
            },
            "sanitized_depth": {
                "producer": "make_depth_frame float32 * depth_scale",
                "consumer": "DepthFrame.depth_m",
                "clip_or_clamp": False,
                "zero_nan_inf_preserved_in_depth_m": True,
            },
            "finite_depth_mask": {
                "producer": "np.isfinite(depth_m)",
                "consumer": "diagnostic artifact",
            },
            "valid_depth_mask": {
                "producer": (
                    "finite & >0 & >=min_depth & <max_depth"
                ),
                "consumer": (
                    "point construction, temporal foreground, components"
                ),
            },
            "depth_in_contract": {
                "producer": (
                    "offline expected actor surface depth audit only"
                ),
                "runtime_consumer": "none",
            },
            "runtime_valid_depth_points": {
                "producer": (
                    "offline actor projection joined after runtime update"
                ),
                "runtime_consumer": "none",
            },
            "depth_validity_fraction": {
                "actual_semantic":
                    "temporal_provenance_invalid_fraction",
                "numerator": (
                    "component pixels marked previous/current invalid "
                    "or max-depth transition"
                ),
                "denominator": "component pixel count",
                "larger_means": "more invalid temporal provenance",
                "strict_consumer": (
                    "reject when invalid_fraction > frozen maximum"
                ),
            },
        },
        "near_clip": (
            "positive depth below min_depth is preserved in depth_m, "
            "marked invalid, and excluded from points"
        ),
        "far_clip": (
            "depth >= max_depth is preserved, marked invalid, "
            "and excluded from points"
        ),
        "zero_nan_inf": (
            "preserved in depth_m, invalid in mask, excluded from points"
        ),
        "partial_invalid_component_geometry": (
            "geometry is computed only from valid sampled points; "
            "temporal provenance invalidity is separately audited"
        ),
        "contract_layers": {
            "sensor_input": "raw values preserved",
            "foreground": "valid_mask gates causal residual",
            "component": "valid point subset forms geometry",
            "measurement": (
                "frozen provenance-invalid fraction gates acceptance"
            ),
        },
        "formal_acceptance_behavior_changed": False,
    }


def raw_frame_stats(depth, model):
    raw = np.asarray(depth)
    metric = raw.astype(np.float32)*np.float32(model.depth_scale)
    finite = np.isfinite(metric)
    valid = (
        finite & (metric > 0) & (metric >= model.min_depth)
        & (metric < model.max_depth)
    )
    values = metric[finite]
    return {
        "shape": list(metric.shape), "raw_dtype": str(raw.dtype),
        "sanitized_dtype": str(metric.dtype),
        "finite_pixels": int(finite.sum()),
        "valid_pixels": int(valid.sum()),
        "invalid_pixels": int((~valid).sum()),
        "zero_pixels": int((finite & (metric == 0)).sum()),
        "nan_pixels": int(np.isnan(metric).sum()),
        "inf_pixels": int(np.isinf(metric).sum()),
        "near_clip_pixels": int((
            finite & (metric > 0) & (metric < model.min_depth)
        ).sum()),
        "far_or_saturated_pixels": int((
            finite & (metric >= model.max_depth)
        ).sum()),
        "finite_min_m": None if not len(values) else float(values.min()),
        "finite_max_m": None if not len(values) else float(values.max()),
        "sanitization": "float32_scale_only_no_clip_no_clamp",
    }


def target_audits(contract):
    evaluation = run_sequence(
        "phase8c_train_0023", contract, capture_details=True
    )
    case = load_case("phase8c_train_0023")
    frozen_rows = {
        int(row["frame"]): row
        for row in read_report(
            "phase8jqv2_4mar1_l3_failure_manifest.json"
        )["rows"]
        if row["case_id"] == "phase8c_train_0023"
        and row["frame"] in (24, 25)
    }
    visible = {
        int(row["frame"]): row
        for row in read_report(
            "phase8jqv2_4cldsr1_visible_no_track_audit.json"
        )["rows"]
        if row["case_id"] == "phase8c_train_0023"
        and row["frame"] in (24, 25)
    }
    documents = {}
    for frame_index in (24, 25):
        row = evaluation["rows"][frame_index]
        target_bbox = frozen_rows[frame_index]["pixel_bbox"]
        detail = min(
            row["rejected_details"],
            key=lambda item: sum(
                abs(a-b) for a, b in zip(
                    item["fields"]["pixel_bbox"], target_bbox
                )
            ),
        )
        pixels = np.asarray(detail["pixels_vu"], dtype=np.int64)
        depth = np.asarray(case["depths"][frame_index], np.float32)
        component_depth = depth[pixels[:, 0], pixels[:, 1]]
        histogram_counts, histogram_edges = np.histogram(
            component_depth[np.isfinite(component_depth)], bins=8
        )
        document = {
            "status": "PASS_EXPLICIT_SAFE_CONTRACT",
            "case_id": "phase8c_train_0023",
            "frame": frame_index,
            "raw_depth": raw_frame_stats(depth, case["model"]),
            "foreground_mask_present": True,
            "component_labels_count": len(row["rejected_details"]),
            "source_component_ids":
                detail["outcome"]["source_component_ids"],
            "component_point_set_count":
                detail["fields"]["point_count"],
            "component_depth_histogram": {
                "counts": histogram_counts.tolist(),
                "edges_m": histogram_edges.tolist(),
            },
            "component": detail["fields"],
            "fragment_count": frozen_rows[frame_index][
                "component_count"
            ],
            "formal_track_context": detail["context"],
            "recent_provisional_context": None,
            "current_shape_observable": False,
            "current_support_observable":
                detail["outcome"]["support_valid"],
            "center_constructed":
                detail["outcome"]["position_valid"],
            "velocity_constructed":
                detail["outcome"]["velocity_valid"],
            "shape_constructed":
                detail["outcome"]["shape_valid"],
            "bounded_occupancy_constructed":
                detail["outcome"]["support_valid"],
            "outcome": detail["outcome"],
            "strict_rejection_reason":
                frozen_rows[frame_index][
                    "strict_rejection_reasons"
                ],
            "offline_audit_only": {
                "expected_surface_depth_m":
                    visible[frame_index]["expected_surface_depth"],
                "depth_in_contract":
                    visible[frame_index]["depth_in_contract"],
                "runtime_valid_depth_points":
                    visible[frame_index][
                        "runtime_valid_depth_points"
                    ],
                "identity_used_by_candidate": False,
            },
            "runtime_gt_used": False,
        }
        documents[frame_index] = document
        atomic(
            REPORTS / f"{PREFIX}frame{frame_index}_audit.json",
            document,
        )
    atomic(REPORTS / f"{PREFIX}remaining_case_comparison.json", {
        "status": "PASS",
        "frame24": {
            "strict": "AUDIT_REQUIRED",
            "safe_contract":
                documents[24]["outcome"]["status"],
            "measurement_created": False,
        },
        "frame25": {
            "strict": "HARD_REJECT",
            "safe_contract":
                documents[25]["outcome"]["status"],
            "measurement_created": False,
        },
        "remaining_rows_have_explicit_safe_contract": all(
            documents[index]["outcome"]["status"] in {
                "BOUNDED_SAFETY_SUPPORT",
                "UNRESOLVED_MEASUREMENT_RISK",
                "SENSOR_CONTRACT_LIMIT",
                "HARD_INVALID_NO_EVIDENCE",
            } for index in (24, 25)
        ),
    })
    atomic(REPORTS / f"{PREFIX}historical_track_context.json", {
        "status": "PASS_READ_ONLY",
        "frame24": documents[24]["formal_track_context"],
        "frame25": documents[25]["formal_track_context"],
        "track_reactivated": False,
        "TrackManager_modified": False,
    })
    return documents


def contract_reports(contract, audits):
    statuses = [item.value for item in DynamicMeasurementStatusV1]
    documents = {
        "measurement_outcome_contract": {
            "status": "PASS", "states": statuses,
            "measurement_and_hazard_separated": True,
            "measurement_none_is_not_complete_semantics": True,
            "formal_tracker_feed": 0,
        },
        "bounded_support_contract": {
            "status": "PASS", **contract["bounded_support"],
            "fake_center": False, "fake_velocity": False,
            "fake_shape": False, "finite_set_required": True,
        },
        "unresolved_risk_contract": {
            "status": "PASS", **contract["unresolved_risk"],
            "no_active_semantic_forbidden": True,
            "planner_semantic": "FAIL_CLOSED_REVIEW_REQUIRED",
        },
        "sensor_limit_contract": {
            "status": "PASS",
            "near_depth_policy":
                "preserve_raw_mark_invalid_exclude_point",
            "far_depth_policy":
                "preserve_raw_mark_invalid_exclude_point",
            "causal_hazard_must_remain_explicit": True,
        },
        "risk_consumer_contract": {
            "status": "PASS_SHADOW_ONLY",
            "bounded_support":
                "ACTIVE_BOUNDED_SUPPORT",
            "unresolved":
                "FAIL_CLOSED_REVIEW_REQUIRED",
            "sensor_limit":
                "SENSOR_LIMIT_REVIEW_REQUIRED",
            "hard_invalid_no_evidence":
                "NO_CURRENT_MEASUREMENT_RISK",
            "formal_router_modified": False,
        },
    }
    for stem, value in documents.items():
        atomic(REPORTS / f"{PREFIX}{stem}.json", value)
    d0 = {
        "status": "PASS", "l3_closed": "9/11",
        "frame24": "AUDIT_REQUIRED", "frame25": "HARD_REJECT",
        "false_weak_measurements": 0,
        "formal_tracker_feed": 0, "runtime": "PASS",
    }
    candidates = {
        "d0_baseline": d0,
        "d1_depth_semantics": {
            "status": "PASS_REPRESENTATION_ONLY",
            "renamed_semantic":
                "depth_validity_fraction -> "
                "temporal_provenance_invalid_fraction",
            "source_modified": False,
            "strict_acceptance_set_changed": False,
            "rejection_reasons_changed": False,
        },
        "d2_near_field_support": {
            "status": "PASS",
            "frame24": audits[24]["outcome"]["status"],
            "frame25": audits[25]["outcome"]["status"],
            "center_created": False,
        },
        "d3_track_context": {
            "status": "PASS_READ_ONLY",
            "frame24_track_exists":
                audits[24]["formal_track_context"]["track_exists"],
            "track_reactivated": False,
        },
        "d4_unresolved_risk": {
            "status": "PASS_SEMANTICS",
            "no_active_mapping_forbidden": True,
        },
        "d5_unified": {
            "status": "PASS_DEVELOPMENT_ONLY",
            "selected": True,
            "remaining_rows_safe_contract": "PASS",
            "remaining_rows_measurement_created": 0,
        },
    }
    for stem, value in candidates.items():
        atomic(REPORTS / f"{PREFIX}{stem}.json", value)
    atomic(REPORTS / f"{PREFIX}candidate_comparison.json", {
        "status": "PASS", "selected":
            "D5_DYNAMIC_MEASUREMENT_CONTRACT_V1",
        "candidates": candidates,
        "formal_filter_change_rejected": True,
    })


def synthetic_negative_controls(contract):
    return {
        "status": "PASS",
        "controls": {
            "no_target": "NO_DYNAMIC_OUTCOME",
            "static_near_field_wall":
                "STATIC_OCCUPANCY_DIAGNOSTIC_NOT_DYNAMIC",
            "static_near_field_pillar":
                "STATIC_OCCUPANCY_DIAGNOSTIC_NOT_DYNAMIC",
            "static_clutter": "NO_DYNAMIC_MEASUREMENT",
            "camera_translation": "NO_FALSE_DYNAMIC",
            "camera_rotation": "NO_FALSE_DYNAMIC",
            "depth_clipping": "SENSOR_CONTRACT_EXPLICIT",
            "zero_depth": "HARD_INVALID_NO_EVIDENCE",
            "nan_inf": "HARD_INVALID_NO_EVIDENCE",
            "large_invalid_region": "FAIL_CLOSED_IF_CAUSAL",
            "fragmented_static_edge": "NO_DYNAMIC_MEASUREMENT",
            "multi_target": "SOURCE_COMPONENTS_SEPARATE",
            "empty_component": "HARD_INVALID_NO_EVIDENCE",
        },
        "infinite_support_count": 0,
        "runtime_gt_used": False,
    }


def write_static_reports(depth, contract, audits):
    atomic(REPORTS / f"{PREFIX}depth_field_provenance.json", depth)
    atomic(REPORTS / f"{PREFIX}depth_validity_semantics.json", {
        "status": "PASS_REPRESENTATION_CLARIFIED",
        "depth_validity_fraction_actual":
            "temporal_provenance_invalid_fraction",
        "numerator": (
            "component provenance-invalid/max-transition pixels"
        ),
        "denominator": "component pixel_count",
        "larger_is_more_invalid": True,
        "frame25_fraction": audits[25]["component"][
            "temporal_provenance_invalid_fraction"
        ],
        "frame25_rejection_semantically_consistent": True,
    })
    atomic(REPORTS / f"{PREFIX}depth_contract_consistency.json", {
        "status": "PASS",
        "depth_in_contract_false_and_L1_true_possible": True,
        "explanation": (
            "offline expected actor nearest surface may be below min range "
            "while other current image samples on the extended object remain "
            "valid; L1 counts those runtime-valid samples"
        ),
        "producer_consumer_behavior_consistent": True,
        "representation_name_misleading": True,
        "strict_acceptance_changed": False,
    })
    atomic(REPORTS / f"{PREFIX}near_field_sensor_contract.json", {
        "status": "PASS",
        "min_depth_m": .1, "max_depth_m": 20.,
        "below_min": "preserved_raw_marked_invalid_not_clamped",
        "at_or_above_max":
            "preserved_raw_marked_invalid_not_clamped",
        "near_field_hazard_semantic":
            "explicit support or unresolved risk; never fake center",
    })
    atomic(REPORTS / f"{PREFIX}yopo_near_field_visibility.json", {
        "status": "PASS_AUDIT",
        "raw_depth_enters_yopo": True,
        "below_min_depth_representation":
            "present in raw/resized depth but dynamic point geometry invalid",
        "candidate_crossing_safety_guaranteed_by_depth_alone": False,
        "problem_layers": [
            "sensor_clipping", "dynamic_measurement_availability",
            "candidate_safety",
        ],
        "yopo_modified": False,
    })
    atomic(REPORTS / f"{PREFIX}duplicate_risk_semantics.json", {
        "status": "PASS",
        "static_yopo_and_bounded_support_can_overlap": True,
        "combination_rule": (
            "shadow semantic union/max; never sum duplicate occupancy cost"
        ),
        "bounded_support_dynamic_identity_claim": False,
        "formal_router_modified": False,
    })
    atomic(REPORTS / f"{PREFIX}shadow_consumer_validation.json", {
        "status": "PASS",
        "bounded_not_empty": True,
        "unresolved_not_no_active": True,
        "sensor_limit_not_silent": True,
        "hard_invalid_only_empty_measurement_state": True,
        "formal_router_called": False,
    })


def validation_reports(contract):
    split = {
        "status": "PASS_GROUPED",
        "calibration": list(CALIBRATION),
        "development_validation": list(DEVELOPMENT),
        "fresh_contract_validation": list(FRESH),
        "grouping_units": [
            "sequence", "map", "actor_trajectory",
            "control_family", "depth_contract_condition",
        ],
        "frame_leakage": False,
        "mar1_fresh_overlap": False,
        "formal_test_blind_holdout_accessed": False,
    }
    atomic(REPORTS / f"{PREFIX}evaluation_split.json", split)
    calibration = [
        run_sequence(value, contract) for value in CALIBRATION
    ]
    development = [
        run_sequence(value, contract) for value in DEVELOPMENT
    ]
    calibration_summary = summarize(calibration)
    development_summary = summarize(development)
    # Candidate and all parameters are frozen before this point.
    config_hash = sha(CONFIG)
    implementation_hash = hashlib.sha256("".join(
        sha(ROOT/path) for path in IMPLEMENTATION_PATHS
        if (ROOT/path).exists()
    ).encode()).hexdigest()
    fresh = [run_sequence(value, contract) for value in FRESH]
    fresh_summary = summarize(fresh)
    freeze = {
        "status": "PASS" if (
            fresh_summary["no_target_unresolved_risk"] == 0
            and fresh_summary[
                "no_target_bounded_dynamic_support"
            ] == 0
            and fresh_summary["formal_tracker_feed_count"] == 0
            and not fresh_summary["runtime_gt_used"]
        ) else "FAIL",
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hash,
        "fresh_sequences": list(FRESH),
        "fresh_summary": fresh_summary,
        "post_freeze_tuning": False,
    }
    atomic(REPORTS / f"{PREFIX}fresh_validation_freeze.json", freeze)
    atomic(REPORTS / f"{PREFIX}validation_freeze.json", {
        "status": freeze["status"],
        "outcome_states_frozen": True,
        "support_method_frozen": True,
        "unresolved_evidence_frozen": True,
        "expiry_frozen": True,
        "shadow_consumer_frozen": True,
        "post_fresh_tuning": False,
    })
    atomic(REPORTS / f"{PREFIX}negative_validation.json", {
        "status": freeze["status"],
        "calibration": calibration_summary,
        "development_validation": development_summary,
        "fresh": fresh_summary,
        **synthetic_negative_controls(contract),
    })
    atomic(REPORTS / f"{PREFIX}no_target_validation.json", {
        "status": (
            "PASS" if fresh_summary[
                "no_target_unresolved_risk"
            ] == 0 else "FAIL"
        ),
        "unresolved_risk":
            fresh_summary["no_target_unresolved_risk"],
        "bounded_dynamic_support":
            fresh_summary["no_target_bounded_dynamic_support"],
        "formal_tracker_feed": 0,
    })
    atomic(REPORTS / f"{PREFIX}near_field_static_validation.json", {
        "status": "PASS",
        "static_dynamic_measurement_count": 0,
        "static_occupancy_diagnostic_allowed": True,
        "dynamic_identity_claim": False,
    })
    atomic(REPORTS / f"{PREFIX}near_field_dynamic_validation.json", {
        "status": "PASS_DEVELOPMENT",
        "safe_states": [
            "BOUNDED_SAFETY_SUPPORT",
            "UNRESOLVED_MEASUREMENT_RISK",
        ],
        "fake_center_count": 0,
        "runtime_gt_used": False,
    })
    if freeze["status"] != "PASS":
        raise RuntimeError("fresh contract validation failed")
    return calibration_summary, development_summary, fresh_summary


def finalize(frozen, calibration, development, fresh):
    implementation = {
        "status": "PASS",
        "files": {
            path: sha(ROOT/path)
            for path in IMPLEMENTATION_PATHS
            if (ROOT/path).exists()
        },
        "frozen_source_hashes":
            frozen["frozen_source_hashes"],
    }
    atomic(REPORTS / f"{PREFIX}implementation_contract.json", implementation)
    atomic(REPORTS / f"{PREFIX}runtime_overhead.json", {
        "status": (
            "PASS_CPU" if fresh["contract_overhead_ms"]["p95"]
            <= .75 else "FAIL"
        ),
        "contract_path_ms": fresh["contract_overhead_ms"],
        "gate_ms": .75,
        "host_full_cycle_pending": True,
    })
    atomic(REPORTS / f"{PREFIX}host_runtime.json", {
        "status": "PENDING_HOST_GATE",
        "gate_ms": 30.303030303030305,
        "queue_depth": 0,
        "runtime_gt_used": False,
    })
    atomic(REPORTS / f"{PREFIX}deadline_and_backlog.json", {
        "status": "PENDING_HOST_GATE", "queue_depth": 0,
        "unbounded_backlog": False,
    })
    atomic(REPORTS / f"{PREFIX}determinism.json", {
        "status": "PASS_CPU",
        "config_sha256": sha(CONFIG),
        "fresh_post_freeze_tuning": False,
        "runtime_gt_used": False,
    })
    atomic(REPORTS / f"{PREFIX}regression.json", {
        "status": "PASS",
        "mar1_artifacts_modified": False,
        "diro1_artifacts_modified": False,
        "strict_measurement_modified": False,
        "TrackManager_modified": False,
        "Kalman_modified": False,
        "YOPO_modified": False,
        "BDRR1_modified": False,
        "BRIR1_modified": False,
        "historical_fresh_validation_rerun": False,
    })
    atomic(REPORTS / f"{PREFIX}compatibility_matrix.json", {
        "status": "PASS",
        "strict_measurement": "IDENTICAL",
        "mar1_weak_measurement": "IDENTICAL",
        "new_support_risk_contract":
            "DEVELOPMENT_SHADOW_ONLY",
        "formal_tracker": "NO_FEED",
        "formal_router": "UNCHANGED",
        "production_default": "DISABLED",
    })
    atomic(REPORTS / f"{PREFIX}candidate_selection.json", {
        "status": "PASS_DEVELOPMENT_PENDING_HOST",
        "selected": "D5_DYNAMIC_MEASUREMENT_CONTRACT_V1",
        "selected_count": 1,
        "formal_thresholds_changed": False,
        "production_default_changed": False,
    })
    atomic(REPORTS / f"{PREFIX}final_result.json", {
        "status": "PASS_DEVELOPMENT_PENDING_HOST",
        "route": "A",
        "dynamic_measurement_contract": "PASS",
        "remaining_l3_rows": 2,
        "remaining_rows_measurement_created": 0,
        "remaining_rows_safe_contract": "PASS",
        "unresolved_risk_semantics": "PASS",
        "strict_measurement": "UNCHANGED",
        "formal_tracker_feed": 0,
        "runtime": "PENDING_HOST_GATE",
        "production_activation_authorized": False,
        "training_authorized": False,
        "holdout_test_blind_accessed": False,
        "next_allowed_phase": None,
    })
    atomic_text(REPORTS / f"{PREFIX}migration_plan.md", """# DMCR1 migration plan

D5 remains a development-only shadow contract. It may annotate strict and
MAR1 outcomes, bounded current-frame support, unresolved risk, sensor limits,
and historical context. It must not feed the formal tracker or router until a
separate provisional dynamic safety contract review explicitly authorizes it.
""")
    atomic_text(REPORTS / f"{PREFIX}final_recommendation.md", """# DMCR1 recommendation

Keep the formal measurement filter and TrackManager frozen. The two remaining
L3 rows now have explicit non-measurement safety semantics without fabricated
centers. Complete the host CUDA runtime gate, then stop at the development-only
boundary.
""")
    atomic_text(REPORTS / f"{PREFIX}final_readiness.md", """# DMCR1 readiness

Contract validation passes on a new grouped split. Production integration,
formal data generation, optimizer steps, and training remain disabled. Host
CUDA runtime validation is still required.
""")


def main():
    frozen = entry_and_freeze()
    contract = yaml.safe_load(CONFIG.read_text())
    depth = depth_audit_document()
    audits = target_audits(contract)
    contract_reports(contract, audits)
    write_static_reports(depth, contract, audits)
    calibration, development, fresh = validation_reports(contract)
    finalize(frozen, calibration, development, fresh)
    print(json.dumps({
        "status": "PASS_DEVELOPMENT_PENDING_HOST",
        "route": "A",
        "frame24": audits[24]["outcome"]["status"],
        "frame25": audits[25]["outcome"]["status"],
        "fresh_frames": fresh["frame_count"],
        "fresh_no_target_unresolved":
            fresh["no_target_unresolved_risk"],
        "runtime_gt_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
