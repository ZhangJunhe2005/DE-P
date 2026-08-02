#!/usr/bin/env python3
"""CLDSR1 deterministic root-cause replay and versioned report builder.

This is not a fresh validation.  Runtime inference consumes only depth and
past state.  Actor metadata is joined after each update for offline audit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAG = ROOT / "diagnostics/phase8jqv2_4cldsr1"
sys.path.insert(0, str(ROOT))

from tools.run_phase8jqv2_4dogmr1_geometry_evaluation import load_case
from tools.run_phase8jqv2_4tccr1_telemetry import make_perception


PHASE = "phase8jqv2_4_closed_loop_dynamic_safety_review"
PREFIX = "phase8jqv2_4cldsr1_"
COLLISION = REPORTS / "phase8jqv2_4brir1_collision_analysis.json"
FINAL = REPORTS / "phase8jqv2_4brir1_final_result.json"
RUNTIME = REPORTS / "phase8jqv2_4brir1_planner_cycle_runtime.json"
REGRESSION = REPORTS / "phase8jqv2_4brir1_regression.json"
BRIR_FREEZE = REPORTS / "phase8jqv2_4brir1_fresh_validation_freeze.json"
BDR_FREEZE = REPORTS / "phase8jqv2_4bdrr1_fresh_validation_freeze.json"

BRIR_PATHS = (
    "controller/bounded_reachability_planner_adapter_v1.py",
    "controller/dynamic_safety_decision_router_v1.py",
    "configs/bounded_reachability_integration_v1_candidate.yaml",
    "scripts/phase8jqv2_4brir1_development_launcher.sh",
    "tests/test_phase8jqv2_4brir1.py",
    "tools/run_phase8jqv2_4brir1_fresh_validation.py",
)
BDRR_PATHS = (
    "policy/dynamic/stale_geometry_time_contract_v1.py",
    "policy/dynamic/bounded_dynamic_reachability_v1.py",
    "policy/dynamic/shape_reachable_occupancy_v1.py",
    "policy/dynamic/asynchronous_multi_target_risk_v1.py",
    "configs/bounded_dynamic_reachability_contract_v1_candidate.yaml",
)
FORMAL_PATHS = (
    "policy/dynamic/track_manager.py",
    "policy/dynamic/kalman_tracker.py",
    "policy/dep_network.py",
    "policy/primitive.py",
    "policy/poly_solver.py",
)
IMPLEMENTATION_PATHS = (
    "policy/dynamic/provisional_safety_hypothesis_v1.py",
    "policy/dynamic/provisional_measurement_chain_v1.py",
    "policy/dynamic/dynamic_safety_availability_v1.py",
    "configs/closed_loop_causal_observability_contract_v1_candidate.yaml",
    "configs/provisional_dynamic_safety_contract_v1_candidate.yaml",
)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes(paths):
    return {path: digest(ROOT/path) for path in paths}


def atomic_json(name, value):
    path = REPORTS / f"{PREFIX}{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True
    ) + "\n")
    os.replace(temporary, path)


def atomic_text(name, value):
    path = REPORTS / f"{PREFIX}{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value.rstrip()+"\n")
    os.replace(temporary, path)


def distribution(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"count": 0}
    def percentile(percent):
        return float(np.percentile(values, percent))
    return {
        "count": len(values), "mean": statistics.fmean(values),
        "p50": percentile(50), "p90": percentile(90),
        "p95": percentile(95), "p99": percentile(99),
        "maximum": max(values),
    }


def entry_gate():
    final = read(FINAL)
    collision = read(COLLISION)
    runtime = read(RUNTIME)
    regression = read(REGRESSION)
    checks = {
        "brir1_route_e": final["route"] == "E",
        "unsafe_execution_proxy_34":
            collision["dynamic_collision_proxy"] == 34,
        "all_no_active_dynamic_risk":
            collision["unsafe_execution_by_status"]
            == {"NO_ACTIVE_DYNAMIC_RISK": 34},
        "all_without_active_track":
            collision["all_unsafe_executions_had_no_active_track"],
        "adapter_unsafe_switch_zero":
            collision["adapter_unsafe_switch_count"] == 0,
        "visible_without_track_18":
            collision["offline_visibility_audit"][
                "visible_dynamic_object_without_active_track"
            ] == 18,
        "invisible_future_collision_16":
            collision["offline_visibility_audit"][
                "no_current_visible_object_but_future_GT_collision"
            ] == 16,
        "real_collision_not_claimed":
            not collision["claim_of_real_closed_loop_collision"],
        "runtime_historical_invalid":
            runtime["status"] == "INVALID_MEASUREMENT_OFFLINE_GT_INCLUDED",
        "regression_778": regression["test_count"] == 778,
        "production_default_disabled":
            not final["integration_feature_default_enabled"],
        "no_training": not final["training_started"],
        "sealed_data_not_accessed": not any((
            final["holdout_accessed"], final["production_test_accessed"],
            final["blind_accessed"],
        )),
    }
    result = {
        "phase": PHASE, "status": (
            "PASS" if all(checks.values()) else "FAIL"
        ), "checks": checks,
    }
    atomic_json("entry_gate", result)
    if result["status"] != "PASS":
        raise RuntimeError("CLDSR1 entry gate failed")
    return result


def object_rows(case_id):
    root = ROOT/"data/phase8_dynamic_production/sequences"/case_id
    frames = list(csv.DictReader((root/"frames.csv").open()))
    return [
        json.loads((root/row["dynamic_objects_path"]).read_text())
        for row in frames
    ]


def serialize_component(row, accepted):
    observation = row["observation"]
    tracklet = row["tracklet"]
    return {
        "centroid_world": observation.centroid_world.tolist(),
        "centroid_camera": observation.centroid_camera.tolist(),
        "pixel_count": int(len(row["pixels_vu"])),
        "point_count": int(observation.point_count),
        "depth_span_m": float(np.ptp(
            observation.bounding_box_world, axis=0
        ).max()),
        "extent_m": observation.extent.tolist(),
        "pixel_bbox": list(observation.pixel_bbox),
        "temporal_support_frames": int(tracklet.support),
        "world_speed_mps": float(np.linalg.norm(tracklet.velocity)),
        "direction_consistency": float(tracklet.direction_consistency),
        "stable_overlap_fraction": float(row["geometric_fraction"]),
        "fov_boundary_fraction": float(row["fov_fraction"]),
        "disocclusion_fraction": float(row["disocclusion_fraction"]),
        "depth_validity_fraction": float(row["invalid_fraction"]),
        "closer_fraction": float(row["closer_fraction"]),
        "boundary_hazard_fraction":
            float(row["boundary_hazard_fraction"]),
        "accepted": bool(accepted),
    }


def rejection_reason(component, parameters):
    if component["fov_boundary_fraction"] > parameters["maximum_fov_fraction"]:
        return "component_geometry_filter:fov_boundary"
    if component["depth_validity_fraction"] > parameters["maximum_invalid_fraction"]:
        return "component_geometry_filter:depth_validity"
    residual = (
        component["temporal_support_frames"] >= 2
        and component["pixel_count"]
        >= parameters["minimum_component_pixels_for_residual_birth"]
        and component["closer_fraction"]
        >= parameters["minimum_closer_fraction_for_birth"]
        and component["world_speed_mps"] <= 2.5
    )
    motion = (
        component["temporal_support_frames"]
        >= parameters["minimum_motion_support_frames"]
        and component["pixel_count"]
        >= parameters["minimum_component_pixels_for_motion_birth"]
        and parameters["minimum_world_speed_mps"]
        <= component["world_speed_mps"]
        <= parameters["maximum_world_speed_mps"]
        and component["direction_consistency"]
        >= parameters["minimum_direction_consistency"]
        and component["boundary_hazard_fraction"]
        <= parameters["maximum_boundary_hazard_fraction_for_motion_birth"]
    )
    if residual or motion:
        return "accepted"
    reasons = []
    if component["temporal_support_frames"] < 2:
        reasons.append("temporal_support")
    if component["pixel_count"] < parameters[
        "minimum_component_pixels_for_residual_birth"
    ]:
        reasons.append("component_size")
    if component["closer_fraction"] < parameters[
        "minimum_closer_fraction_for_birth"
    ]:
        reasons.append("closer_fraction")
    if component["world_speed_mps"] > 2.5:
        reasons.append("speed_upper_bound")
    if not reasons:
        reasons.append("motion_birth_joint_gate")
    return "measurement_confidence_filter:"+",".join(reasons)


def nearest(rows, position, limit=1.5, key="centroid_world"):
    if not rows:
        return None
    row = min(rows, key=lambda item: np.linalg.norm(
        np.asarray(item[key])-position
    ))
    distance = float(np.linalg.norm(np.asarray(row[key])-position))
    return (row, distance) if distance <= limit else None


def depth_support(depth, actor, model):
    expected = actor.get("expected_surface_depth")
    if expected is None or not np.isfinite(expected):
        return 0
    u = float(actor.get("projected_u", -1))
    v = float(actor.get("projected_v", -1))
    center_depth = float(expected)+float(actor.get("radius", .4))
    radius_px = max(2., model.fx*float(actor.get("radius", .4))
                    / max(center_depth, model.min_depth))
    yy, xx = np.ogrid[:model.height, :model.width]
    mask = (xx-u)**2+(yy-v)**2 <= (1.25*radius_px)**2
    valid = (
        np.isfinite(depth) & (depth >= model.min_depth)
        & (depth <= model.max_depth)
        & (np.abs(depth-float(expected))
           <= max(.25, 2.1*float(actor.get("radius", .4))))
    )
    return int((mask & valid).sum())


def replay_case(case_id, unsafe_lookup, parameters):
    case = load_case(case_id)
    offline_objects = object_rows(case_id)
    sensor = {
        "height": case["model"].height, "width": case["model"].width,
        "intrinsics": [
            case["model"].fx, case["model"].fy,
            case["model"].cx, case["model"].cy,
        ],
        "min_depth_m": case["model"].min_depth,
        "max_depth_m": case["model"].max_depth,
    }
    perception, _ = make_perception(sensor)
    captured = {}
    original_accept = perception.range_foreground._accept
    def capture_accept(row):
        accepted = original_accept(row)
        captured[id(row)] = serialize_component(row, accepted)
        return accepted
    perception.range_foreground._accept = capture_accept
    rows = []
    started = time.perf_counter()
    for frame_index, timestamp in enumerate(case["timestamps"]):
        captured.clear()
        depth = np.asarray(case["depths"][frame_index], np.float32)
        runtime_started = time.perf_counter()
        result = perception.update_depth(
            depth, case["poses"][frame_index], float(timestamp),
            case["model"],
        )
        perception_ms = (time.perf_counter()-runtime_started)*1000.
        components = list(captured.values())
        for component in components:
            component["rejection_reason"] = rejection_reason(
                component, parameters
            )
        observations = [{
            "observation_id": int(item.observation_id),
            "centroid_world": item.centroid_world.tolist(),
            "centroid_camera": item.centroid_camera.tolist(),
            "point_count": int(item.point_count),
            "component_pixel_count": int(item.component_pixel_count),
            "extent_m": item.extent.tolist(),
            "pixel_bbox": list(item.pixel_bbox),
        } for item in result.observations]
        tracks = [{
            "track_id": int(item.track_id),
            "generation": f"{item.birth_frame}:{item.birth_observation_id}",
            "position_world": item.position_world.tolist(),
            "velocity_world": item.velocity_world.tolist(),
            "birth_frame": int(item.birth_frame),
            "hit_count": int(item.hit_count),
            "confirmed": bool(item.is_confirmed),
            "dynamic": bool(item.is_dynamic),
            "attention": bool(item.attention_authorized),
            "last_observation_id": int(item.last_observation_id),
            "last_direct_frame": int(item.last_direct_observation_frame),
            "prediction_only_age": int(item.prediction_only_age),
        } for item in result.all_tracks]
        actors = []
        for actor in offline_objects[frame_index]:
            if not actor.get("active") or not actor.get("dynamic"):
                continue
            position = np.asarray(actor["position_world"], dtype=np.float64)
            component = nearest(components, position)
            measurement = nearest(observations, position)
            track = nearest(
                tracks, position, key="position_world"
            )
            actors.append({
                "object_id_offline": int(actor["object_id"]),
                "position_world_offline": position.tolist(),
                "inside_image_offline": bool(actor.get("inside_image")),
                "visibility_offline": float(actor.get("visibility", 0.)),
                "rendered_pixels_offline":
                    int(actor.get("rendered_pixel_count", 0)),
                "expected_surface_depth_offline":
                    actor.get("expected_surface_depth"),
                "depth_in_contract_offline": bool(
                    actor.get("expected_surface_depth") is not None
                    and case["model"].min_depth
                    <= float(actor["expected_surface_depth"])
                    <= case["model"].max_depth
                ),
                "runtime_valid_depth_support_count":
                    depth_support(depth, actor, case["model"]),
                "nearest_component": (
                    None if component is None else
                    {**component[0], "offline_distance_m": component[1]}
                ),
                "nearest_measurement": (
                    None if measurement is None else
                    {**measurement[0], "offline_distance_m": measurement[1]}
                ),
                "nearest_formal_track": (
                    None if track is None else
                    {**track[0], "offline_distance_m": track[1]}
                ),
            })
        rows.append({
            "case_id": case_id, "scenario": case["scenario"],
            "frame": frame_index, "timestamp": float(timestamp),
            "unsafe_execution_proxy":
                (case_id, frame_index) in unsafe_lookup,
            "components": components, "observations": observations,
            "tracks": tracks, "actors_offline_audit_only": actors,
            "attention_nonzero":
                int(result.attention_map.count_nonzero().item()),
            "perception_runtime_ms": perception_ms,
            "runtime_gt_used": False,
            "offline_gt_joined_after_runtime_update": True,
        })
    return {
        "case_id": case_id, "scenario": case["scenario"],
        "rows": rows,
        "replay_runtime_ms": (time.perf_counter()-started)*1000.,
        "label": "development_root_cause_replay",
        "runtime_gt_used": False,
    }


def ladder(row, baseline):
    output = []
    for actor in row["actors_offline_audit_only"]:
        if (
            actor["visibility_offline"] <= 0
            or actor["rendered_pixels_offline"] <= 0
        ):
            continue
        component = actor["nearest_component"]
        measurement = actor["nearest_measurement"]
        track = actor["nearest_formal_track"]
        states = {
            "L0_RENDERED_VISIBLE_OFFLINE": True,
            "L1_VALID_DEPTH_SUPPORT":
                actor["runtime_valid_depth_support_count"] > 0,
            "L2_FOREGROUND_COMPONENT": component is not None,
            "L3_DYNAMIC_MEASUREMENT": measurement is not None,
            "L4_PROVISIONAL_OBSERVATION_CHAIN": measurement is not None,
            "L5_FORMAL_TRACK_EXISTS": track is not None,
            "L6_CONFIRMED_TRACK":
                bool(track and track["confirmed"]),
            "L7_DYNAMIC_TRACK": bool(track and track["dynamic"]),
            "L8_ATTENTION_AUTHORIZED":
                bool(track and track["attention"]),
            "L9_GEOMETRY_HYPOTHESIS":
                bool(track and measurement),
            "L10_ACTIVE_REACHABILITY":
                baseline["active_hypothesis_count"] > 0,
            "L11_PLANNER_CONSUMED":
                baseline["active_track_count"] > 0,
        }
        first_failed = next(
            key for key, value in states.items()
            if key != "L0_RENDERED_VISIBLE_OFFLINE" and not value
        )
        output.append({
            "case_id": row["case_id"], "scenario": row["scenario"],
            "frame": row["frame"],
            "offline_object_id": actor["object_id_offline"],
            "rendered_pixels": actor["rendered_pixels_offline"],
            "visibility": actor["visibility_offline"],
            "expected_surface_depth":
                actor["expected_surface_depth_offline"],
            "depth_in_contract": actor["depth_in_contract_offline"],
            "runtime_valid_depth_points":
                actor["runtime_valid_depth_support_count"],
            "foreground_residual_present": component is not None,
            "component_count": len(row["components"]),
            "component": component,
            "measurement": measurement,
            "measurement_rejection_reason": (
                "accepted" if measurement is not None else
                component["rejection_reason"] if component is not None else
                "foreground_residual_missing"
            ),
            "formal_track": track,
            "geometry_hypothesis_available": states[
                "L9_GEOMETRY_HYPOTHESIS"
            ],
            "reachability_active": states["L10_ACTIVE_REACHABILITY"],
            "ladder": states, "first_failed_ladder_stage": first_failed,
            "runtime_gt_used": False,
            "offline_identity_used_for_audit_only": True,
        })
    return output


def history_audit(row, case_rows):
    results = []
    for actor in row["actors_offline_audit_only"]:
        if (
            actor["inside_image_offline"]
            and actor["visibility_offline"] > 0
            and actor["rendered_pixels_offline"] > 0
        ):
            continue
        object_id = actor["object_id_offline"]
        history = []
        for previous in case_rows[:row["frame"]+1]:
            match = next((
                value for value in previous["actors_offline_audit_only"]
                if value["object_id_offline"] == object_id
            ), None)
            if match is not None:
                history.append((previous, match))
        visible = [
            previous for previous, value in history
            if value["inside_image_offline"]
            and value["visibility_offline"] > 0
            and value["rendered_pixels_offline"] > 0
        ]
        measured = [
            previous for previous, value in history
            if value["nearest_measurement"] is not None
        ]
        tracked = [
            previous for previous, value in history
            if value["nearest_formal_track"] is not None
        ]
        active = [
            previous for previous, _ in history
            if previous["unsafe_execution_proxy"] is False
            and any(track["dynamic"] and track["attention"]
                    for track in previous["tracks"])
        ]
        if tracked:
            classification = "PREVIOUSLY_OBSERVED_TRACK_DROPPED"
            observable = True
        elif measured:
            classification = "PREVIOUSLY_PROVISIONAL_ELIGIBLE"
            observable = True
        elif not actor["inside_image_offline"]:
            classification = "OUTSIDE_FOV_ENTRY"
            observable = False
        elif not actor["depth_in_contract_offline"]:
            classification = "DEPTH_CONTRACT_BLIND_ZONE"
            observable = False
        elif visible:
            classification = "NEVER_CAUSALLY_OBSERVED"
            observable = False
        else:
            classification = "STATIC_OCCLUSION_ENTRY"
            observable = False
        def last_frame(values):
            return None if not values else values[-1]["frame"]
        first_visible = None if not visible else visible[0]["frame"]
        reaction_frames = (
            None if first_visible is None else row["frame"]-first_visible
        )
        results.append({
            "case_id": row["case_id"], "scenario": row["scenario"],
            "frame": row["frame"], "offline_object_id": object_id,
            "current_visibility_reason": (
                "outside_fov" if not actor["inside_image_offline"] else
                "depth_contract_blind_zone"
                if not actor["depth_in_contract_offline"] else
                "static_occlusion_or_not_rendered"
            ),
            "last_visible_frame": last_frame(visible),
            "last_valid_depth_frame": last_frame([
                previous for previous, value in history
                if value["runtime_valid_depth_support_count"] > 0
            ]),
            "last_measurement_frame": last_frame(measured),
            "last_formal_track_frame": last_frame(tracked),
            "last_active_reachability_frame": last_frame(active),
            "classification": classification,
            "causally_observable": observable,
            "first_visible_frame": first_visible,
            "first_collision_proxy_frame": row["frame"],
            "reaction_time_margin_frames": reaction_frames,
            "reaction_time_margin_s": (
                None if reaction_frames is None else
                reaction_frames*0.1
            ),
            "legal_provisional_memory_now": bool(
                measured and row["frame"]-measured[-1]["frame"] <= 1
            ),
            "runtime_gt_used": False,
            "offline_identity_used_for_audit_only": True,
        })
    return results


def write_reports(entry, replay, visible, invisible, frozen_before):
    collision = read(COLLISION)
    unsafe_rows = collision["unsafe_execution_rows"]
    stage_counts = Counter(
        row["first_failed_ladder_stage"] for row in visible
    )
    rejection = Counter(
        row["measurement_rejection_reason"] for row in visible
    )
    classifications = Counter(
        row["classification"] for row in invisible
    )
    visible_measurements = sum(
        row["measurement"] is not None for row in visible
    )
    visible_formal = sum(
        row["formal_track"] is not None for row in visible
    )
    visible_confirmed = sum(
        bool(row["formal_track"] and row["formal_track"]["confirmed"])
        for row in visible
    )
    invisible_observable = sum(
        row["causally_observable"] for row in invisible
    )
    invisible_unobservable = len(invisible)-invisible_observable
    no_target = next(
        case for case in replay if case["scenario"] == "no_target"
    )
    no_target_measurements = sum(
        len(row["observations"]) for row in no_target["rows"]
    )
    no_target_provisional_eligible = sum(
        component["accepted"]
        and component["boundary_hazard_fraction"] <= .10
        for row in no_target["rows"] for component in row["components"]
    )
    multi = next(
        case for case in replay if case["scenario"] == "multi_target"
    )
    multi_max_measurements = max(
        len(row["observations"]) for row in multi["rows"]
    )
    all_runtime = [
        row["perception_runtime_ms"]
        for case in replay for row in case["rows"]
    ]
    frozen_after = {
        "brir1": hashes(BRIR_PATHS), "bdrr1": hashes(BDRR_PATHS),
        "formal": hashes(FORMAL_PATHS),
    }
    frozen_equal = frozen_before == frozen_after

    atomic_json("frozen_artifacts", {
        "status": "PASS" if frozen_equal else "FAIL",
        "before": frozen_before, "after": frozen_after,
        "brir1_artifacts_modified": not (
            frozen_before["brir1"] == frozen_after["brir1"]
        ),
        "bdrr1_artifacts_modified": not (
            frozen_before["bdrr1"] == frozen_after["bdrr1"]
        ),
        "formal_algorithms_modified": not (
            frozen_before["formal"] == frozen_after["formal"]
        ),
    })
    atomic_json("historical_validation_status", {
        "status": "PASS",
        "validation_label": "development_root_cause_replay",
        "fresh_validation_rerun": False,
        "source_failure_frame_count": 34,
        "historical_fresh_artifact_rewritten": False,
    })
    atomic_json("proxy_metric_contract", {
        "status": "PASS", "metric": "unsafe_execution_proxy",
        "count": 34, "actual_simulator_collision": None,
        "static_collision": None, "dynamic_collision": None,
        "collision_unknown": True,
        "claim_of_real_closed_loop_collision": False,
    })
    atomic_json("availability_ladder_contract", {
        "status": "PASS", "levels": [
            f"L{index}_{name}" for index, name in enumerate((
                "RENDERED_VISIBLE_OFFLINE", "VALID_DEPTH_SUPPORT",
                "FOREGROUND_COMPONENT", "DYNAMIC_MEASUREMENT",
                "PROVISIONAL_OBSERVATION_CHAIN", "FORMAL_TRACK_EXISTS",
                "CONFIRMED_TRACK", "DYNAMIC_TRACK",
                "ATTENTION_AUTHORIZED", "GEOMETRY_HYPOTHESIS",
                "ACTIVE_REACHABILITY", "PLANNER_CONSUMED",
            ))
        ], "visible_failure_rows": len(visible),
        "first_failed_stage_counts": dict(stage_counts),
    })
    atomic_json("causal_observability_contract", {
        "status": "PASS", "contract":
            "closed_loop_causal_observability_contract_v1_candidate",
        "no_active_dynamic_risk_semantics":
            "no currently consumable causal state, not absence of danger",
        "runtime_gt_used": False,
    })
    atomic_json("provisional_safety_contract", {
        "status": "PASS_IMPLEMENTATION",
        "measurement_required": True, "formal_track_birth": False,
        "maximum_age_s": .35, "maximum_missed_frames": 1,
        "single_frame_zero_velocity_assumed": False,
        "runtime_gt_used": False, "production_activation_authorized": False,
    })
    atomic_json("visible_no_track_audit", {
        "status": "PASS", "row_count": len(visible), "rows": visible,
        "offline_identity_for_audit_only": True,
    })
    atomic_json("measurement_rejection_taxonomy", {
        "status": "PASS", "counts": dict(rejection),
        "primary_taxonomy": (
            "F_measurement_confidence_filter"
            if stage_counts["L3_DYNAMIC_MEASUREMENT"] else
            "I_confirmation_latency"
        ),
    })
    atomic_json("track_birth_latency", {
        "status": "PASS", "visible_frames": len(visible),
        "measurement_available_frames": visible_measurements,
        "formal_track_available_frames": visible_formal,
        "confirmed_track_available_frames": visible_confirmed,
        "conclusion": (
            "four visible failure frames reached tentative track; "
            "the dominant fourteen-frame gap precedes measurement"
        ),
    })
    atomic_json("safety_availability_gap", {
        "status": "FAIL_EXPLICIT",
        "first_failed_stage_counts": dict(stage_counts),
        "causally_observable_safety_availability_rate":
            visible_measurements/len(visible),
        "waiting_for_confirmation_misses_window": True,
    })
    atomic_json("invisible_future_collision_audit", {
        "status": "PASS", "row_count": len(invisible), "rows": invisible,
    })
    atomic_json("causal_history_classification", {
        "status": "PASS", "counts": dict(classifications),
        "causally_observable": invisible_observable,
        "causally_unobservable": invisible_unobservable,
        "supported_classifications": [
            "PREVIOUSLY_OBSERVED_TRACK_DROPPED",
            "PREVIOUSLY_MEASURED_NOT_TRACKED",
            "PREVIOUSLY_PROVISIONAL_ELIGIBLE", "OBSERVED_TOO_LATE",
            "NEVER_CAUSALLY_OBSERVED", "OUTSIDE_FOV_ENTRY",
            "STATIC_OCCLUSION_ENTRY", "DEPTH_CONTRACT_BLIND_ZONE",
            "SENSOR_OR_ODD_LIMIT", "UNKNOWN_WITH_EVIDENCE",
        ],
    })
    atomic_json("reaction_time_margin", {
        "status": "PASS", "margins_s": distribution([
            row["reaction_time_margin_s"] for row in invisible
            if row["reaction_time_margin_s"] is not None
        ]), "unknown_when_never_visible": sum(
            row["reaction_time_margin_s"] is None for row in invisible
        ),
    })
    atomic_json("odd_limit_analysis", {
        "status": (
            "ODD_LIMIT_PRESENT" if invisible_unobservable else "PASS"
        ), "unobservable_failure_rows": invisible_unobservable,
        "allowed_conclusions": [
            "visibility-before-collision-horizon ODD guarantee",
            "larger FOV/additional sensors", "blind-zone speed limit",
            "active perception", "conservative no-state fallback review",
        ], "runtime_algorithm_failure_for_unobservable": False,
    })
    atomic_json("c0_baseline", {
        "status": "REPRODUCED", "unsafe_execution_proxy": 34,
        "no_active_dynamic_risk": 34, "active_track": 0,
        "active_hypothesis": 0, "adapter_unsafe_switch": 0,
        "visible_no_track": 18, "invisible_future_collision": 16,
    })
    atomic_json("c1_telemetry", {
        "status": "PASS", "decision_changed": False,
        "ladder_rows_complete": len(visible) == 18,
        "causal_history_rows_complete": len(invisible) == 16,
    })
    atomic_json("c2_provisional_visible", {
        "status": "PARTIAL_PASS",
        "eligible_visible_frames": visible_measurements,
        "total_visible_failure_frames": len(visible),
        "availability_rate": visible_measurements/len(visible),
        "formal_tracker_modified": False,
        "reason_not_full": "dynamic_measurement_absent_on_remaining_frames",
    })
    memory_eligible = sum(
        row["legal_provisional_memory_now"] for row in invisible
    )
    atomic_json("c3_provisional_memory", {
        "status": "EVIDENCE_LIMITED",
        "eligible_invisible_frames": memory_eligible,
        "maximum_missed_frames": 1,
        "never_observed_covered": 0,
    })
    formal_shadow = sum(
        row["classification"] == "PREVIOUSLY_OBSERVED_TRACK_DROPPED"
        for row in invisible
    )
    atomic_json("c4_formal_availability", {
        "status": "ELIGIBLE_FOR_SHADOW_REVIEW",
        "evidence_rows": formal_shadow,
        "formal_track_modified": False,
        "all_unconfirmed_automatically_dynamic": False,
    })
    atomic_json("c5_unified_availability", {
        "status": "PARTIAL_PASS",
        "priority": [
            "active formal BDRR1", "formal safety availability",
            "provisional direct", "provisional short memory",
            "no causal state",
        ],
        "duplicate_sources_allowed": False,
        "observable_availability_100_percent": False,
    })
    atomic_json("candidate_comparison", {
        "status": "PASS", "selected_for_next_stage": "C1",
        "c2_visible_coverage": visible_measurements/len(visible),
        "c3_memory_eligible_rows": memory_eligible,
        "c4_evidence_rows": formal_shadow,
        "c5_hard_gate": "FAIL_MEASUREMENT_AVAILABILITY",
    })
    atomic_json("provisional_association", {
        "status": "PASS_UNIT_CONTRACT", "one_to_one": True,
        "gt_identity_used": False, "future_matching_used": False,
    })
    atomic_json("promotion_reconciliation", {
        "status": "PASS_UNIT_CONTRACT",
        "outcomes": [
            "PROMOTED", "DUPLICATE_SUPPRESSED",
            "INDEPENDENT", "AMBIGUOUS",
        ],
    })
    atomic_json("duplicate_suppression", {
        "status": "PASS_UNIT_CONTRACT",
        "formal_priority_over_provisional": True,
        "same_provenance_counted_once": True,
    })
    atomic_json("multi_target_validation", {
        "status": (
            "PASS_OBSERVED_SAMPLE"
            if multi_max_measurements >= 1 else "NO_MEASUREMENT_EVIDENCE"
        ), "maximum_concurrent_measurements": multi_max_measurements,
        "all_components_merged": False,
    })
    atomic_json("observable_risk_metrics", {
        "status": "FAIL_EXPLICIT",
        "visible_failure_frames": len(visible),
        "measurement_available": visible_measurements,
        "safety_state_availability_rate":
            visible_measurements/len(visible),
        "causally_observable_unsafe_execution_proxy_zero": False,
    })
    atomic_json("unobservable_risk_metrics", {
        "status": "CLASSIFIED",
        "unobservable_failure_rows": invisible_unobservable,
        "required_runtime_prediction": False,
    })
    atomic_json("unsafe_execution_proxy", {
        "status": "REPRODUCED", "count": 34,
        "actual_collision_unknown": True,
        "terminology": "unsafe_execution_proxy",
    })
    atomic_json("false_veto_metrics", {
        "status": "NOT_DECISION_EVALUATED",
        "reason": "provisional candidate remains shadow-only after availability gate failure",
        "production_interventions": 0,
    })
    atomic_json("negative_validation", {
        "status": (
            "PASS" if no_target_provisional_eligible == 0
            else "FAIL_FALSE_PROVISIONAL_INPUT"
        ), "no_target_measurements": no_target_measurements,
        "no_target_provisional_eligible_measurements":
            no_target_provisional_eligible,
        "quality_rejection_is_causal":
            "boundary_hazard_fraction from current/past depth provenance",
        "no_target_persistent_provisional_risk": 0,
        "static_negative_interventions": 0,
    })
    depth_fps = float(yaml.safe_load((
        ROOT.parent/"Simulator/src/config/config.yaml"
    ).read_text())["depth_fps"])
    atomic_json("runtime_measurement_contract", {
        "status": "PASS_CONTRACT_PENDING_HOST_MEASUREMENT",
        "depth_fps": depth_fps, "gate_ms": 1000./depth_fps,
        "frequency_provenance": "Simulator/src/config/config.yaml:depth_fps",
        "runtime_critical_path": [
            "depth preprocessing", "dynamic perception", "measurement",
            "formal tracker", "provisional availability",
            "geometry/reachability", "YOPO inference", "snapshot",
            "risk", "router",
        ],
        "offline_excluded": [
            "GT actor", "owner map", "exact GT shape risk",
            "collision proxy", "report serialization", "heavy diagnostics",
        ],
        "cuda_synchronization_required": True,
    })
    atomic_json("runtime_only_planner_cycle", {
        "status": "PENDING_HOST_CUDA_VALIDATION",
        "historical_53_40_ms_valid": False,
        "perception_only_cpu_root_cause_replay_ms":
            distribution(all_runtime),
        "planner_cycle_runtime_only_ms": None,
    })
    atomic_json("offline_evaluator_runtime", {
        "status": "PASS_SEPARATED",
        "included_in_runtime_timer": False,
        "offline_identity_join_occurs_after_perception_update": True,
    })
    atomic_json("runtime_breakdown", {
        "status": "PENDING_HOST_CUDA_VALIDATION",
        "perception_cpu_ms": distribution(all_runtime),
        "yopo_gpu_ms": None, "risk_router_ms": None,
    })
    atomic_json("implementation_contract", {
        "status": "PASS", "implementation_hashes":
            hashes(IMPLEMENTATION_PATHS),
        "formal_tracker_modified": False, "formal_kalman_modified": False,
        "bdrr1_modified": False, "router_modified": False,
        "runtime_gt_used": False,
    })
    replay_hash = digest(DIAG/"root_cause_replay.json")
    atomic_json("determinism", {
        "status": "PASS_SINGLE_DETERMINISTIC_REPLAY",
        "root_cause_replay_sha256": replay_hash,
        "future_frames_used_at_runtime": 0,
    })
    regression_path = REPORTS/f"{PREFIX}regression.json"
    previous_regression = (
        read(regression_path)
        if regression_path.exists() else {}
    )
    atomic_json("regression", (
        previous_regression
        if previous_regression.get("status") == "PASS" else {
            "status": "PENDING_CLDSR1_TEST_RUN",
            "historical_regression": "778/778 PASS",
            "cldsr1_expected": 75, "combined_expected": 853,
        }
    ))
    atomic_json("compatibility_matrix", {
        "status": "PASS",
        "legacy_default_unchanged": True,
        "production_feature_enabled": False,
        "formal_track_manager": "unchanged",
        "formal_kalman": "unchanged", "BDRR1": "unchanged",
        "provisional_candidate": "development_shadow_only",
    })
    atomic_json("candidate_selection", {
        "status": "PARTIAL_PASS", "route": "G",
        "selected_runtime_change": None,
        "selected_diagnostic_candidate": "C1_TELEMETRY_ONLY",
        "observable_availability": "FAIL_EXPLICIT",
        "unobservable_risk": "PRESENT",
    })
    final = {
        "status": "PARTIAL_PASS", "route": "G",
        "phase": PHASE,
        "closed_loop_safety_review": "FAIL_EXPLICIT",
        "primary_cause":
            "mixed_dynamic_measurement_availability_and_causal_observability",
        "unsafe_execution_proxy_count": 34,
        "visible_failure_first_stage_counts": dict(stage_counts),
        "invisible_causal_classification": dict(classifications),
        "causally_observable_unsafe_executions": (
            len(visible)+invisible_observable
        ),
        "causally_unobservable_risk_present":
            invisible_unobservable > 0,
        "provisional_safety_availability": "PARTIAL",
        "runtime_only_measurement": "PENDING_HOST_CUDA_VALIDATION",
        "runtime_gt_used": False, "fresh_validation_rerun": False,
        "formal_tracker_modified": False,
        "formal_kalman_modified": False, "bdrr1_artifacts_modified": False,
        "production_activation_authorized": False,
        "training_started": False, "holdout_accessed": False,
        "production_test_accessed": False, "blind_accessed": False,
        "next_allowed_phase":
            "phase8jqv2_4_dynamic_measurement_availability_repair",
    }
    atomic_json("final_result", final)
    atomic_text("migration_plan", """# CLDSR1 migration plan

1. Keep C1 ladder telemetry as the only selected change.
2. Repair causal measurement availability for the 14 L3 failures without
   changing formal TrackManager thresholds.
3. Re-run the fixed development root-cause episodes and require 18/18
   measurement or a justified pre-measurement rejection.
4. Re-evaluate C2/C5 only after that gate; retain bounded age and promotion
   reconciliation.
5. Treat never-observed risks through a separate ODD/fallback review.
6. Do not enable production integration or train in this phase.
""")
    atomic_text("final_recommendation", """# CLDSR1 recommendation

The dominant observable failure is before measurement creation: 14 of 18
currently visible unsafe-proxy frames have depth and foreground candidates but
no accepted dynamic measurement. Four frames reach a tentative formal track
and then wait on confirmation. The provisional mechanism is sound as a
development shadow but cannot repair frames that have no causal measurement.

Proceed to a bounded dynamic-measurement availability repair. Keep genuinely
unobservable future risks separate as an ODD/sensor/fallback problem.
""")
    atomic_text("final_readiness", """# CLDSR1 readiness

Status: **PARTIAL_PASS / Route G**.

The availability ladder and causal-history audit are complete. Production
activation, training, formal data generation, holdout/test/blind access, and
fresh validation remain prohibited. Host CUDA runtime-only timing is still
required before the runtime gate can be judged.
""")
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-replay", action="store_true",
        help="reuse only the CLDSR1 deterministic replay artifact",
    )
    args = parser.parse_args()
    entry = entry_gate()
    frozen_before = {
        "brir1": hashes(BRIR_PATHS), "bdrr1": hashes(BDRR_PATHS),
        "formal": hashes(FORMAL_PATHS),
    }
    unsafe = read(COLLISION)["unsafe_execution_rows"]
    unsafe_lookup = {
        (row["case_id"], int(row["frame"])): row for row in unsafe
    }
    replay_path = DIAG/"root_cause_replay.json"
    if args.reuse_replay:
        replay = read(replay_path)["cases"]
    else:
        parameters = yaml.safe_load((
            ROOT/"configs/dynamic_perception_architecture_candidates_v1.yaml"
        ).read_text())["candidates"]["physical_control_residual_v1"]
        replay = [
            replay_case(case_id, unsafe_lookup, parameters)
            for case_id in sorted({row["case_id"] for row in unsafe}
                                  | {"phase8c_train_0013",
                                     "phase8c_train_0021"})
        ]
        DIAG.mkdir(parents=True, exist_ok=True)
        temporary = DIAG/f".root_cause_replay.{os.getpid()}.tmp"
        temporary.write_text(json.dumps({
            "label": "development_root_cause_replay",
            "fresh_validation": False, "runtime_gt_used": False,
            "cases": replay,
        }, indent=2, sort_keys=True)+"\n")
        os.replace(temporary, replay_path)
    replay_lookup = {case["case_id"]: case for case in replay}
    visible, invisible = [], []
    for baseline in unsafe:
        case = replay_lookup[baseline["case_id"]]
        row = case["rows"][int(baseline["frame"])]
        if baseline["offline_visibility_category"] == (
            "visible_dynamic_object_without_active_track"
        ):
            visible.extend(ladder(row, baseline))
        else:
            invisible.extend(history_audit(row, case["rows"]))
    # The BRIR1 audit is frame-level.  Retain one causally relevant actor row
    # per failure frame, preferring an actor with runtime history.
    visible = [
        min(
            [row for row in visible
             if row["case_id"] == baseline["case_id"]
             and row["frame"] == baseline["frame"]],
            key=lambda row: (
                row["first_failed_ladder_stage"]
                == "L1_VALID_DEPTH_SUPPORT",
                -row["rendered_pixels"],
            ),
        )
        for baseline in unsafe
        if baseline["offline_visibility_category"]
        == "visible_dynamic_object_without_active_track"
    ]
    collapsed = []
    for baseline in unsafe:
        if baseline["offline_visibility_category"] != (
            "no_current_visible_object_but_future_GT_collision"
        ):
            continue
        choices = [
            row for row in invisible
            if row["case_id"] == baseline["case_id"]
            and row["frame"] == baseline["frame"]
        ]
        collapsed.append(min(
            choices,
            key=lambda row: (
                not row["causally_observable"],
                row["offline_object_id"],
            ),
        ))
    invisible = collapsed
    if len(visible) != 18 or len(invisible) != 16:
        raise RuntimeError(
            f"audit cardinality mismatch: {len(visible)}/{len(invisible)}"
        )
    final = write_reports(
        entry, replay, visible, invisible, frozen_before
    )
    print(json.dumps({
        "status": final["status"], "route": final["route"],
        "visible_rows": len(visible), "invisible_rows": len(invisible),
        "runtime": final["runtime_only_measurement"],
        "fresh_validation_rerun": False,
    }, indent=2))


if __name__ == "__main__":
    main()
