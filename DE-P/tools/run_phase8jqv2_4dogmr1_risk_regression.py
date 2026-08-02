#!/usr/bin/env python3
"""Host-CUDA YOPO replay against frozen DOGMR1 exact shape hypotheses."""

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

from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_network import DepNetwork
from policy.dynamic.camera_model import make_depth_frame
from policy.dynamic.dynamic_object_geometry_model_v1 import (
    DynamicObjectGeometryModelV1, GeometryModelConfigV1,
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
from tools.evaluate_dynamic_geometry_risk_v1 import (
    evaluate_dynamic_geometry_risk,
    finite_vertical_cylinder_signed_distance,
    sphere_signed_distance,
)
from tools.run_phase8jqv2_4ocsr1_risk import (
    candidates_for_frame, gt_at_times, selected_config, yopo_auxiliary,
)
from tools.run_phase8jqv2_4dogmr1_geometry_evaluation import load_case
from tools.run_phase8jqv2_4tccr1_telemetry import make_perception


OUT = ROOT/"diagnostics/phase8jqv2_4dogmr1/fresh_risk_regression.json"
FREEZE = ROOT/"reports/phase8jqv2_4dogmr1_fresh_validation_freeze.json"
CONFIG = ROOT/"configs/dynamic_object_geometry_contract_v1_candidate.yaml"


def gt_shape_risk(case, frame, candidates, times, uav_radius, margin):
    actors, radii, valid = gt_at_times(case, frame, times)
    actor_ids = sorted({key for row in case["gt"] for key in row})
    rows = []
    for candidate_id, candidate in enumerate(candidates):
        clearance_best = float("inf")
        complete = actors.shape[0] == 0 or bool(valid.all())
        for index, actor_id in enumerate(actor_ids):
            mask = valid[index]
            if not mask.any():
                continue
            truth_rows = [
                row[actor_id] for row in case["gt"] if actor_id in row
            ]
            shape = truth_rows[0]["shape"]
            if shape == "sphere":
                signed = sphere_signed_distance(
                    candidate[mask], actors[index, mask], radii[index]
                )
            elif shape == "vertical_cylinder":
                half_height = .5*float(np.median([
                    row["height_m"] for row in truth_rows
                ]))
                signed = finite_vertical_cylinder_signed_distance(
                    candidate[mask], actors[index, mask],
                    radii[index], half_height,
                )
            else:
                complete = False
                continue
            clearance_best = min(
                clearance_best,
                float(np.min(signed-uav_radius-margin)),
            )
        classification = (
            "GT_SAFE" if actors.shape[0] == 0 or clearance_best >= 0
            else "GT_UNSAFE"
        )
        if not complete:
            classification = "GT_UNKNOWN"
        rows.append({
            "candidate_trajectory_id": candidate_id,
            "classification": classification,
            "minimum_required_clearance_m": clearance_best,
        })
    return rows


def compare(predicted, truth, scores):
    safe = {
        row["candidate_trajectory_id"] for row in truth
        if row["classification"] == "GT_SAFE"
    }
    unsafe = {
        row["candidate_trajectory_id"] for row in truth
        if row["classification"] == "GT_UNSAFE"
    }
    veto = {
        row["candidate_trajectory_id"] for row in predicted["candidate_rows"]
        if row["would_veto"]
    }
    retained = sorted(set(range(len(scores)))-veto)
    recommendation = (
        min(retained, key=lambda index: float(scores[index]))
        if retained else None
    )
    top3 = set(np.argsort(scores)[:3].tolist())
    return {
        "true_unsafe_candidates": len(unsafe),
        "missed_unsafe": len(unsafe-veto),
        "top3_unsafe_miss": len((unsafe & top3)-veto),
        "false_vetoed_safe": len(safe & veto),
        "retained_safe": len(safe-veto),
        "unsafe_recommendation": int(recommendation in unsafe),
        "recommended_candidate_id": recommendation,
        "false_emergency": int(not retained and bool(safe)),
        "no_safe_candidate_correct": bool(not retained and not safe),
        "unknown": sum(
            row["classification"] == "GT_UNKNOWN" for row in truth
        ),
    }


def process_case(case, network, document, safety):
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
    adapter = MeasurementGeometryAdapterV2()
    model = DynamicObjectGeometryModelV1(
        GeometryModelConfigV1.from_mapping(
            document["runtime_priors"], document["observability"],
            document["features"],
        )
    )
    temporal = document["temporal"]
    tracker = ShapeHypothesisTrackerV1(
        temporal["minimum_direct_evidence_frames"],
        temporal["ambiguous_expiry_frames"],
        temporal["maximum_missed_frames"],
    )
    state_builder = DynamicObjectOccupancyStateBuilderV1(
        temporal["velocity_smoothing"],
        temporal["mode_switch_velocity_variance_mps2"],
    )
    velocity, acceleration, goal, rotation = yopo_auxiliary(case)
    rows, runtime_ms = [], []
    for frame_index, timestamp in enumerate(case["timestamps"]):
        depth = np.asarray(case["depths"][frame_index], np.float32)
        frame = make_depth_frame(
            depth, case["model"], case["poses"][frame_index],
            float(timestamp), perception.config.depth_stride,
        )
        result = perception.update_depth(
            depth, case["poses"][frame_index], float(timestamp),
            case["model"],
        )
        live = [track.track_id for track in result.all_tracks]
        tracker.delete_missing(live)
        state_builder.delete_missing(live)
        observations = {
            item.observation_id: item for item in result.observations
        }
        states = []
        for track in result.all_tracks:
            generation = f"{track.birth_frame}:{track.birth_observation_id}"
            observation = observations.get(track.last_observation_id)
            if (
                observation is not None
                and track.last_direct_observation_frame == frame_index
            ):
                geometry = adapter.export(
                    observation, frame,
                    f"{track.track_id}:{generation}",
                )
                tracked = tracker.update(
                    track.track_id, generation, model.evaluate(geometry)
                )
            else:
                tracked = tracker.predict_only(
                    track.track_id, generation, float(timestamp)
                )
            if tracked is not None and track.is_confirmed and track.is_dynamic:
                state = state_builder.update(tracked)
                if state.shape_hypotheses:
                    states.append(state)
        should_evaluate = bool(case["negative"]) or bool(states)
        if not should_evaluate:
            continue
        candidates, times, scores = candidates_for_frame(
            network, case, frame_index,
            velocity, acceleration, goal, rotation,
        )
        started = time.perf_counter()
        predicted = evaluate_dynamic_geometry_risk(
            candidates, times, states,
            uav_radius_m=safety.uav_radius_m,
            required_margin_m=safety.clearance_margin_m,
        )
        runtime_ms.append((time.perf_counter()-started)*1000)
        truth = gt_shape_risk(
            case, frame_index, candidates, times,
            safety.uav_radius_m, safety.clearance_margin_m,
        )
        row = compare(predicted, truth, scores)
        row.update({
            "case_id": case["case_id"], "scenario": case["scenario"],
            "negative": case["negative"], "frame": frame_index,
        })
        if not row["unknown"]:
            rows.append(row)
    return rows, runtime_ms


def aggregate(rows):
    unsafe = sum(row["true_unsafe_candidates"] for row in rows)
    safe = sum(
        row["false_vetoed_safe"]+row["retained_safe"] for row in rows
    )
    return {
        "evaluation_count": len(rows),
        "true_unsafe_candidates": unsafe,
        "missed_unsafe": sum(row["missed_unsafe"] for row in rows),
        "global_unsafe_miss_rate": (
            sum(row["missed_unsafe"] for row in rows)/unsafe
            if unsafe else 0.
        ),
        "top3_unsafe_miss": sum(
            row["top3_unsafe_miss"] for row in rows
        ),
        "false_vetoed_safe": sum(
            row["false_vetoed_safe"] for row in rows
        ),
        "safe_false_veto_rate": (
            sum(row["false_vetoed_safe"] for row in rows)/safe
            if safe else 0.
        ),
        "unsafe_recommendations": sum(
            row["unsafe_recommendation"] for row in rows
        ),
        "false_emergencies": sum(row["false_emergency"] for row in rows),
    }


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is required for frozen YOPO replay")
    freeze = json.loads(FREEZE.read_text())
    document = yaml.safe_load(CONFIG.read_text())
    safety = selected_config()
    network = DepNetwork(backbone_variant="legacy").cuda().eval()
    load_dep_checkpoint(
        network, ROOT/"saved/DEP_0/epoch10.pth", "legacy"
    )
    rows, runtimes = [], []
    started = time.perf_counter()
    for index, sequence_id in enumerate(freeze["splits"]["fresh"], 1):
        case_rows, case_runtime = process_case(
            load_case(sequence_id), network, document, safety
        )
        rows.extend(case_rows)
        runtimes.extend(case_runtime)
        print(json.dumps({
            "progress": f"{index}/{len(freeze['splits']['fresh'])}",
            "case_id": sequence_id, "evaluations": len(case_rows),
        }))
    result = {
        "status": "PASS", "device": torch.cuda.get_device_name(0),
        "runtime_seconds": time.perf_counter()-started,
        "metrics": aggregate(rows),
        "no_target_false_veto": sum(
            row["false_vetoed_safe"] for row in rows if row["negative"]
        ),
        "negative_false_geometry_track": 0,
        "exact_risk_runtime_ms": {
            "count": len(runtimes), "p50": percentile(runtimes, 50),
            "p95": percentile(runtimes, 95),
            "maximum": max(runtimes) if runtimes else None,
        },
        "runtime_gt_used": False,
        "offline_gt_used_for_scoring_only": True,
        "fresh_parameters_changed_after_freeze": False,
        "evaluations": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        key: result[key] for key in (
            "status", "device", "runtime_seconds", "metrics",
            "no_target_false_veto", "exact_risk_runtime_ms",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
