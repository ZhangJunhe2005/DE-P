#!/usr/bin/env python3
"""Frozen fresh PTAR1 shadow-risk comparison using host CUDA for YOPO."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controller.dynamic_safety_shadow_adapter_v2 import (
    BoundedCoastingSafetyAdapter,
)
from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_network import DepNetwork
from policy.dynamic.camera_model import make_depth_frame
from policy.dynamic.measurement_geometry_adapter_v1 import (
    MeasurementGeometryAdapterV1,
)
from policy.dynamic.reference_aligned_center_tracker_v1 import (
    ReferenceAlignedCenterTrackerV1,
)
from policy.dynamic.tracking_collision_reference_bridge_v1 import (
    ReferenceBridgeConfigV1, ReferenceObservability,
    TrackingCollisionReferenceBridgeV1,
)
from tools.evaluate_yopo_dynamic_candidate_risk_v1 import (
    compare_shadow_with_gt, evaluate_gt_candidate_risk,
)
from tools.run_phase8jqv2_4ocsr1_risk import (
    candidates_for_frame, gt_at_times, selected_config, yopo_auxiliary,
)
from tools.run_phase8jqv2_4ptar1_reference_evaluation import load_case
from tools.run_phase8jqv2_4tccr1_telemetry import make_perception


OUT = ROOT/"diagnostics/phase8jqv2_4ptar1/fresh_risk_regression.json"
FREEZE = ROOT/"reports/phase8jqv2_4ptar1_fresh_validation_freeze.json"
CONFIG = ROOT/"configs/tracking_collision_reference_contract_v1_candidate.yaml"


def runtime_track(track, frame):
    return {
        "track_id": int(track.track_id),
        "state_generation":
            f"{track.birth_frame}:{track.birth_observation_id}",
        "position_world": track.position_world,
        "velocity_world": track.velocity_world,
        "state_covariance": track.state_covariance,
        "is_confirmed": track.is_confirmed,
        "is_dynamic": track.is_dynamic,
        "attention_authorized": track.attention_authorized,
        "missed_count": track.missed_count,
        "measurement_present":
            track.last_direct_observation_frame == frame,
        "consecutive_direct_hits": track.consecutive_direct_hits,
        "birth_frame": track.birth_frame,
        "birth_observation_id": track.birth_observation_id,
        "split_merge_suspected": False,
    }


def process_case(case, model, bridge_config, safety_config):
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
    geometry_adapter = MeasurementGeometryAdapterV1()
    bridge = TrackingCollisionReferenceBridgeV1(bridge_config)
    center_tracker = ReferenceAlignedCenterTrackerV1()
    raw_adapter = BoundedCoastingSafetyAdapter(safety_config)
    aligned_adapter = BoundedCoastingSafetyAdapter(safety_config)
    velocity, acceleration, goal, rotation = yopo_auxiliary(case)
    evaluations = []
    for frame_index, timestamp in enumerate(case["timestamps"]):
        frame = make_depth_frame(
            np.asarray(case["depths"][frame_index], dtype=np.float32),
            case["model"], case["poses"][frame_index], float(timestamp),
            perception.config.depth_stride,
        )
        result = perception.update_depth(
            np.asarray(case["depths"][frame_index], dtype=np.float32),
            case["poses"][frame_index], float(timestamp), case["model"],
        )
        center_tracker.delete_missing(
            track.track_id for track in result.all_tracks
        )
        observations = {
            item.observation_id: item for item in result.observations
        }
        evidence_by_track = {}
        for track in result.all_tracks:
            observation = observations.get(track.last_observation_id)
            if (
                observation is None
                or track.last_direct_observation_frame != frame_index
            ):
                continue
            geometry = geometry_adapter.export(observation, frame)
            evidence = bridge.evaluate_geometry(geometry)
            generation = f"{track.birth_frame}:{track.birth_observation_id}"
            state = center_tracker.update(
                track.track_id, generation, evidence
            )
            evidence_by_track[track.track_id] = (evidence, state)
        raw_tracks, aligned_tracks = [], []
        for track in result.all_tracks:
            raw = runtime_track(track, frame_index)
            raw_tracks.append(raw)
            generation = raw["state_generation"]
            evidence_state = evidence_by_track.get(track.track_id)
            if evidence_state is None:
                state = center_tracker.predict(
                    track.track_id, generation, float(timestamp)
                )
                evidence = None
            else:
                evidence, state = evidence_state
            if state is None:
                continue
            aligned = dict(raw)
            aligned.update({
                "position_world": state.position_world,
                "velocity_world": state.velocity_world,
                "state_covariance": state.state_covariance,
                "radius_m": (
                    bridge_config.radius_prior_max_m
                    if evidence is not None and evidence.observability
                    == ReferenceObservability.CENTER_OBSERVABLE
                    else bridge_config.vertical_half_extent_upper_m
                ),
                "reference_mode": (
                    "prediction_only"
                    if evidence is None else evidence.reference_mode
                ),
            })
            aligned_tracks.append(aligned)
        should_evaluate = bool(case["negative"]) or any(
            track.is_confirmed and track.is_dynamic
            for track in result.all_tracks
        )
        if not should_evaluate:
            raw_adapter.update_safety_states(
                raw_tracks, float(timestamp), frame_index
            )
            aligned_adapter.update_safety_states(
                aligned_tracks, float(timestamp), frame_index
            )
            continue
        candidates, times, scores = candidates_for_frame(
            model, case, frame_index, velocity, acceleration, goal, rotation
        )
        original = int(np.argmin(scores))
        raw_result = raw_adapter.evaluate(
            candidates, times, raw_tracks, timestamp=float(timestamp),
            frame_index=frame_index, original_candidate_id=original,
            candidate_scores=scores,
        )
        aligned_result = aligned_adapter.evaluate(
            candidates, times, aligned_tracks, timestamp=float(timestamp),
            frame_index=frame_index, original_candidate_id=original,
            candidate_scores=scores,
        )
        actors, radii, valid = gt_at_times(case, frame_index, times)
        gt = evaluate_gt_candidate_risk(
            candidates, times, actors, radii,
            uav_radius_m=safety_config.uav_radius_m,
            required_margin_m=safety_config.clearance_margin_m,
            actor_valid=valid,
        )
        if gt["unknown_candidate_ids"]:
            continue
        evaluations.append({
            "case_id": case["case_id"],
            "scenario": case["scenario"],
            "negative": case["negative"],
            "frame": frame_index,
            "raw": compare_shadow_with_gt(raw_result, gt),
            "aligned": compare_shadow_with_gt(aligned_result, gt),
            "raw_decision": raw_result["decision_status"],
            "aligned_decision": aligned_result["decision_status"],
            "runtime_gt_used": False,
        })
    return evaluations


def aggregate(rows, key):
    metrics = [row[key] for row in rows]
    safe_total = sum(
        row["false_vetoed_safe"]+row["retained_safe"] for row in metrics
    )
    return {
        "evaluation_count": len(metrics),
        "true_unsafe_candidates": sum(
            row["true_unsafe_candidates"] for row in metrics
        ),
        "missed_unsafe": sum(row["missed_unsafe"] for row in metrics),
        "false_vetoed_safe": sum(
            row["false_vetoed_safe"] for row in metrics
        ),
        "safe_candidate_false_veto_rate": (
            sum(row["false_vetoed_safe"] for row in metrics)/safe_total
            if safe_total else 0.
        ),
        "unsafe_recommendations": sum(
            row["unsafe_recommendation"] for row in metrics
        ),
        "false_emergencies": sum(
            row["false_emergency"] for row in metrics
        ),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is required for frozen YOPO candidate replay")
    freeze = json.loads(FREEZE.read_text())
    document = yaml.safe_load(CONFIG.read_text())
    bridge_config = ReferenceBridgeConfigV1.from_mapping(document["bridge"])
    safety_config = selected_config()
    model = DepNetwork(backbone_variant="legacy").cuda().eval()
    load_dep_checkpoint(model, ROOT/"saved/DEP_0/epoch10.pth", "legacy")
    rows = []
    started = time.perf_counter()
    for index, sequence_id in enumerate(freeze["splits"]["fresh"], 1):
        case_rows = process_case(
            load_case(sequence_id), model, bridge_config, safety_config
        )
        rows.extend(case_rows)
        print(json.dumps({
            "progress": f"{index}/{len(freeze['splits']['fresh'])}",
            "case_id": sequence_id, "evaluations": len(case_rows),
        }))
    result = {
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "runtime_seconds": time.perf_counter()-started,
        "raw_surface": aggregate(rows, "raw"),
        "aligned_hybrid": aggregate(rows, "aligned"),
        "negative_false_veto": sum(
            row["aligned"]["false_vetoed_safe"]
            for row in rows if row["negative"]
        ),
        "runtime_gt_used": False,
        "offline_gt_used_for_scoring_only": True,
        "fresh_parameters_changed_after_freeze": False,
        "evaluations": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({key: result[key] for key in (
        "status", "device", "runtime_seconds", "raw_surface",
        "aligned_hybrid", "negative_false_veto",
    )}, indent=2))


if __name__ == "__main__":
    main()
