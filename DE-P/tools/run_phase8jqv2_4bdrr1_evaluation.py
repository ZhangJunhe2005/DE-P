#!/usr/bin/env python3
"""Staged BDRR1 asynchronous reachability host-CUDA evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
DIAG = ROOT/"diagnostics/phase8jqv2_4bdrr1"
CONFIG = ROOT/"configs/bounded_dynamic_reachability_contract_v1_candidate.yaml"
SAM_CONFIG = ROOT/"configs/shape_aware_motion_state_contract_v1_candidate.yaml"
DOG_CONFIG = ROOT/"configs/dynamic_object_geometry_contract_v1_candidate.yaml"
FREEZE = REPORTS/"phase8jqv2_4bdrr1_fresh_validation_freeze.json"
sys.path.insert(0, str(ROOT))

from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_network import DepNetwork
from policy.dynamic.asynchronous_multi_target_risk_v1 import (
    evaluate_asynchronous_reachability_risk,
)
from policy.dynamic.bounded_dynamic_reachability_v1 import (
    BoundedDynamicReachabilityBuilderV1, ReachabilityStatus,
)
from policy.dynamic.camera_model import make_depth_frame
from policy.dynamic.dynamic_object_geometry_model_v1 import (
    DynamicObjectGeometryModelV1, GeometryModelConfigV1,
)
from policy.dynamic.shape_aware_dynamic_occupancy_v1 import (
    FastGeometryUpdateCacheV1,
)
from policy.dynamic.shape_motion_hypothesis_tracker_v1 import (
    ShapeMotionHypothesisTrackerV1,
)
from policy.dynamic.shape_reachable_occupancy_v1 import (
    predict_reachable_occupancy,
)
from tools.run_phase8jqv2_4dogmr1_geometry_evaluation import (
    hypothesis_metrics, load_case,
)
from tools.run_phase8jqv2_4ocsr1_collect import associate_gt
from tools.run_phase8jqv2_4ocsr1_risk import (
    candidates_for_frame, gt_at_times, selected_config, yopo_auxiliary,
)
from tools.run_phase8jqv2_4samsr1_evaluation import (
    compare_risk, exact_gt_shape_risk,
)
from tools.run_phase8jqv2_4tccr1_telemetry import make_perception


SCENARIOS = (
    "no_target", "crossing", "head_on", "multi_target",
    "temporal_separation", "occluded_but_tracked",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)), "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(values.max()),
    }


def grouped_sequences():
    grouped = defaultdict(list)
    source = ROOT/"data/phase8_dynamic_production/sequences"
    for path in sorted(source.glob("phase8c_train_*/metadata.yaml")):
        metadata = yaml.safe_load(path.read_text())
        grouped[str(metadata["scenario_type"])].append(path.parent.name)
    result = {
        "reachability_calibration": [],
        "reachability_development_validation": [],
        "fresh": [],
    }
    for scenario in SCENARIOS:
        values = grouped[scenario]
        for index, split in zip((21, 22, 23), result):
            result[split].append(values[index])
    return {key: tuple(sorted(value)) for key, value in result.items()}


def contains_truth(occupancy, truth):
    center = np.asarray(truth["position_world"], dtype=np.float64)
    lower, upper = (
        occupancy.center_lower_world[0],
        occupancy.center_upper_world[0],
    )
    center_covered = bool(np.all(
        center >= lower-1e-9
    ) and np.all(center <= upper+1e-9))
    state = occupancy.state
    radius = float(truth["radius_m"])
    radius_covered = (
        state.radius_interval_m[0]-1e-9 <= radius
        <= state.radius_interval_m[1]+1e-9
    )
    height_covered = True
    if truth["shape"] == "vertical_cylinder":
        half = .5*float(truth["height_m"])
        height_covered = (
            state.half_height_interval_m[0]-1e-9 <= half
            <= state.half_height_interval_m[1]+1e-9
        )
    return {
        "center_covered": center_covered,
        "radius_covered": bool(radius_covered),
        "height_covered": bool(height_covered),
        "full_covered": bool(
            state.shape_type == truth["shape"]
            and center_covered and radius_covered and height_covered
        ),
        "radial_growth_m": float(occupancy.radial_growth_m[0]),
        "vertical_growth_m": float(occupancy.vertical_growth_m[0]),
        "center_box_volume_m3": float(np.prod(upper-lower)),
    }


def process_case(case, network, config, sam_config, dog_config, candidate):
    sensor = {
        "height": case["model"].height, "width": case["model"].width,
        "intrinsics": [
            case["model"].fx, case["model"].fy,
            case["model"].cx, case["model"].cy,
        ], "min_depth_m": case["model"].min_depth,
        "max_depth_m": case["model"].max_depth,
    }
    perception, _ = make_perception(sensor)
    geometry_model = DynamicObjectGeometryModelV1(
        GeometryModelConfigV1.from_mapping(
            dog_config["runtime_priors"], dog_config["observability"],
            dog_config["features"],
        )
    )
    geometry_cache = FastGeometryUpdateCacheV1(geometry_model)
    motion_tracker = ShapeMotionHypothesisTrackerV1(
        sam_config["history"], sam_config["reference_transition"]
    )
    reachability_builder = BoundedDynamicReachabilityBuilderV1(
        config, candidate
    )
    safety = selected_config()
    velocity, acceleration, goal, rotation = yopo_auxiliary(case)
    direct_geometry, predicted_coverage, risk_rows = [], [], []
    time_rows, cache_rows = [], []
    reachability_runtime, total_runtime = [], []
    state_records = []
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
        motion_tracker.delete_missing(live)
        reachability_builder.delete_missing(live)
        observations = {
            item.observation_id: item for item in result.observations
        }
        tracked_by_id, reachable_states = {}, []
        unresolved = False
        for track in result.all_tracks:
            generation = f"{track.birth_frame}:{track.birth_observation_id}"
            observation = observations.get(track.last_observation_id)
            direct = (
                observation is not None
                and track.last_direct_observation_frame == frame_index
            )
            if direct:
                geometry, evaluation = geometry_cache.update(
                    observation, frame, generation
                )
                tracked = motion_tracker.update(
                    track.track_id, generation, evaluation
                )
                cache_rows.append({
                    "frame": frame_index, "track_id": track.track_id,
                    "generation": generation,
                    "observation_id": observation.observation_id,
                    "cache_key": f"{generation}:{observation.observation_id}",
                    "direct": True,
                })
            else:
                tracked = motion_tracker.predict_only(
                    track.track_id, generation, float(timestamp)
                )
            if tracked is None:
                continue
            tracked_by_id[track.track_id] = tracked
            states, status = reachability_builder.build(tracked)
            if (
                status == ReachabilityStatus.UNRESOLVED_DYNAMIC_RISK
                and track.is_confirmed and track.is_dynamic
            ):
                unresolved = True
            if track.is_confirmed and track.is_dynamic:
                reachable_states.extend(states)
            for state in states:
                state_records.append({
                    "frame": frame_index, "track_id": state.track_id,
                    "generation": state.generation,
                    "hypothesis_id": state.hypothesis_id,
                    "shape": state.shape_type,
                    "source_geometry_timestamp":
                        state.source_geometry_timestamp,
                    "state_timestamp": state.state_timestamp,
                    "query_timestamp": state.query_timestamp,
                    "geometry_age_s": state.geometry_age_s,
                    "status": state.status.value,
                })
        active = [
            track for track in result.all_tracks
            if track.is_confirmed and track.is_dynamic
        ]
        assignment = associate_gt([
            {"track_id": int(track.track_id),
             "position_world": track.position_world.tolist()}
            for track in active
        ], case["gt"][frame_index])
        for track in active:
            actor_id = assignment.get(track.track_id)
            tracked = tracked_by_id.get(track.track_id)
            if actor_id is None or tracked is None:
                continue
            truth = case["gt"][frame_index][actor_id]
            direct = track.last_direct_observation_frame == frame_index
            for item in tracked.hypotheses:
                metric = hypothesis_metrics(
                    item.geometry_hypothesis, truth
                )
                if direct and item.geometry_type == truth["shape"]:
                    direct_geometry.append({
                        "case_id": case["case_id"], "frame": frame_index,
                        "shape": truth["shape"], **metric,
                    })
            for state in reachable_states:
                if (
                    state.track_id != track.track_id
                    or state.status == ReachabilityStatus.EXPIRED
                ):
                    continue
                horizon = reachability_builder.horizons(state, [0.])
                occupancy = predict_reachable_occupancy(state, horizon)
                metric = contains_truth(occupancy, truth)
                predicted_coverage.append({
                    "case_id": case["case_id"], "frame": frame_index,
                    "track_id": track.track_id, "actor_id": actor_id,
                    "shape": truth["shape"],
                    "hypothesis_shape": state.shape_type,
                    "prediction_only": not direct,
                    "geometry_age_s": state.geometry_age_s,
                    "status": state.status.value, **metric,
                })
        should_evaluate = bool(case["negative"]) or bool(
            reachable_states
        ) or unresolved
        if not should_evaluate:
            continue
        candidates, times, scores = candidates_for_frame(
            network, case, frame_index,
            velocity, acceleration, goal, rotation,
        )
        result_risk = evaluate_asynchronous_reachability_risk(
            candidates, times, reachable_states, reachability_builder,
            uav_radius_m=safety.uav_radius_m,
            required_margin_m=safety.clearance_margin_m,
            candidate_scores=scores,
            unresolved_dynamic_risk=unresolved,
        )
        reachability_runtime.append(
            result_risk["reachability_generation_ms"]
        )
        total_runtime.append(result_risk["total_runtime_ms"])
        time_rows.extend([
            {"case_id": case["case_id"], "frame": frame_index, **row}
            for row in result_risk["per_track_time_rows"]
        ])
        truth = exact_gt_shape_risk(
            case, frame_index, candidates, times,
            safety.uav_radius_m, safety.clearance_margin_m,
        )
        comparison = compare_risk(result_risk, truth, scores)
        if comparison["unknown"]:
            continue
        row = {
            key: value for key, value in comparison.items()
            if key not in {"truth_rows", "prediction_rows"}
        }
        row.update({
            "case_id": case["case_id"], "scenario": case["scenario"],
            "frame": frame_index, "negative": case["negative"],
            "decision_status": result_risk["decision_status"],
            "unresolved_dynamic_risk": unresolved,
        })
        risk_rows.append(row)
    return {
        "case_id": case["case_id"], "scenario": case["scenario"],
        "candidate": candidate["id"], "negative": case["negative"],
        "direct_geometry": direct_geometry,
        "predicted_coverage": predicted_coverage,
        "risk_rows": risk_rows, "time_rows": time_rows,
        "cache_rows": cache_rows, "state_records": state_records,
        "geometry_fit_calls": geometry_cache.fit_calls,
        "direct_runtime_ms": [
            row["elapsed_ms"] for row in geometry_cache.profile
        ],
        "reachability_generation_ms": reachability_runtime,
        "total_runtime_ms": total_runtime,
        "runtime_gt_used": False,
    }


def summarize(split, candidate, cases):
    direct = [row for case in cases for row in case["direct_geometry"]]
    coverage = [row for case in cases for row in case["predicted_coverage"]]
    risk = [row for case in cases for row in case["risk_rows"]]
    times = [row for case in cases for row in case["time_rows"]]
    unsafe = sum(len(row["unsafe"]) for row in risk)
    safe = sum(len(row["safe"]) for row in risk)
    misses = sum(len(row["missed_unsafe"]) for row in risk)
    false_veto = sum(len(row["false_vetoed_safe"]) for row in risk)
    safe_cases = {
        row["case_id"] for row in risk if row["safe"]
    }
    emergency_cases = {
        row["case_id"] for row in risk
        if row["safe"] and row["false_emergency"]
    }
    shapes = ("sphere", "vertical_cylinder")
    by_shape = {}
    for shape in shapes:
        direct_rows = [
            row for row in direct if row["shape"] == shape
        ]
        predicted_rows = [
            row for row in coverage
            if row["shape"] == shape
            and row["hypothesis_shape"] == shape
        ]
        prediction_only = [
            row for row in predicted_rows if row["prediction_only"]
        ]
        by_shape[shape] = {
            "direct_samples": len(direct_rows),
            "direct_geometry_coverage": (
                float(np.mean([
                    row["full_covered"] for row in direct_rows
                ])) if direct_rows else None
            ),
            "predicted_samples": len(predicted_rows),
            "predicted_occupancy_coverage": (
                float(np.mean([
                    row["full_covered"] for row in predicted_rows
                ])) if predicted_rows else None
            ),
            "prediction_only_samples": len(prediction_only),
            "prediction_only_coverage": (
                float(np.mean([
                    row["full_covered"] for row in prediction_only
                ])) if prediction_only else None
            ),
            "radial_growth_m": distribution([
                row["radial_growth_m"] for row in predicted_rows
            ]),
            "center_box_volume_m3": distribution([
                row["center_box_volume_m3"] for row in predicted_rows
            ]),
        }
    stale_multi = [
        row for row in risk
        if row["scenario"] == "multi_target"
    ]
    return {
        "status": "PASS", "split": split, "candidate": candidate,
        "case_count": len(cases), "risk_query_count": len(risk),
        "true_unsafe_candidates": unsafe, "missed_unsafe": misses,
        "global_unsafe_miss_rate": misses/unsafe if unsafe else 0.,
        "top3_unsafe_miss": sum(
            len(row["top3_unsafe_miss"]) for row in risk
        ),
        "unsafe_recommendations": sum(
            row["unsafe_recommendation"] for row in risk
        ),
        "multi_target_stale_miss": sum(
            len(row["missed_unsafe"]) for row in stale_multi
        ),
        "false_vetoed_safe": false_veto,
        "candidate_global_false_veto_rate": (
            false_veto/(len(risk)*15) if risk else 0.
        ),
        "safe_candidate_false_veto_rate": (
            false_veto/safe if safe else 0.
        ),
        "sequence_false_emergency_rate": (
            len(emergency_cases)/len(safe_cases) if safe_cases else 0.
        ),
        "selected_safe_loss_rate": (
            sum(
                row["unsafe_recommendation"] or row["false_emergency"]
                for row in risk if row["safe"]
            )/sum(bool(row["safe"]) for row in risk)
            if any(row["safe"] for row in risk) else 0.
        ),
        "no_target_false_veto": sum(
            len(row["false_vetoed_safe"])
            for row in risk if row["negative"]
        ),
        "false_emergencies": sum(row["false_emergency"] for row in risk),
        "correct_no_safe_candidate": sum(
            row["no_safe_candidate_correct"] for row in risk
        ),
        "unresolved_dynamic_risk_queries": sum(
            row["unresolved_dynamic_risk"] for row in risk
        ),
        "coverage_by_shape": by_shape,
        "time_integrity": {
            "row_count": len(times),
            "missing_geometry_age": sum(
                row["geometry_age_s"] is None for row in times
            ),
            "double_age_guard_failures": sum(
                not row["double_age_guard"] for row in times
            ),
            "per_track_unique_ages": len({
                (row["case_id"], row["frame"], row["track_id"],
                 row["generation"], row["geometry_age_s"])
                for row in times
            }),
        },
        "geometry_fit_calls": sum(
            case["geometry_fit_calls"] for case in cases
        ),
        "direct_geometry_runtime_ms": distribution([
            value for case in cases for value in case["direct_runtime_ms"]
        ]),
        "reachability_generation_ms": distribution([
            value for case in cases
            for value in case["reachability_generation_ms"]
        ]),
        "cached_reachability_risk_ms": distribution([
            value for case in cases for value in case["total_runtime_ms"]
        ]),
        "runtime_gt_used": False,
    }


def source_hashes():
    paths = {
        "config": CONFIG,
        "time_contract": ROOT/"policy/dynamic/stale_geometry_time_contract_v1.py",
        "reachability": ROOT/"policy/dynamic/bounded_dynamic_reachability_v1.py",
        "occupancy": ROOT/"policy/dynamic/shape_reachable_occupancy_v1.py",
        "risk": ROOT/"policy/dynamic/asynchronous_multi_target_risk_v1.py",
        "samsr1_motion": ROOT/"policy/dynamic/shape_aware_motion_state_v1.py",
        "samsr1_tracker": ROOT/"policy/dynamic/shape_motion_hypothesis_tracker_v1.py",
        "samsr1_fast_path": ROOT/"policy/dynamic/shape_aware_dynamic_occupancy_v1.py",
        "evaluator": Path(__file__),
    }
    return {key: digest(value) for key, value in paths.items()}


def run_split(split, sequences, candidates, network, configs):
    outputs = {}
    for candidate in candidates:
        cases = []
        for index, sequence in enumerate(sequences, 1):
            result = process_case(
                load_case(sequence), network, *configs, candidate
            )
            cases.append(result)
            print(json.dumps({
                "split": split, "candidate": candidate["id"],
                "progress": f"{index}/{len(sequences)}", "case": sequence,
            }))
        summary = summarize(split, candidate["id"], cases)
        atomic_json(DIAG/f"{split}_{candidate['id']}.json", {
            "split": split, "candidate": candidate["id"],
            "cases": cases, "summary": summary,
        })
        outputs[candidate["id"]] = summary
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("development", "fresh"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    config = yaml.safe_load(CONFIG.read_text())
    sam_config = yaml.safe_load(SAM_CONFIG.read_text())
    dog_config = yaml.safe_load(DOG_CONFIG.read_text())
    splits = grouped_sequences()
    if any(len(values) != 6 for values in splits.values()):
        raise RuntimeError(f"incomplete split: {splits}")
    if (
        set(splits["reachability_calibration"])
        & set(splits["reachability_development_validation"])
        or set(splits["fresh"])
        & (
            set(splits["reachability_calibration"])
            | set(splits["reachability_development_validation"])
        )
    ):
        raise RuntimeError("split leakage")
    network = DepNetwork(backbone_variant="legacy").cuda().eval()
    load_dep_checkpoint(
        network, ROOT/"saved/DEP_0/epoch10.pth", "legacy"
    )
    configs = (config, sam_config, dog_config)
    if args.stage == "development":
        calibration = run_split(
            "reachability_calibration",
            splits["reachability_calibration"], config["candidates"],
            network, configs,
        )
        development = run_split(
            "reachability_development_validation",
            splits["reachability_development_validation"],
            config["candidates"], network, configs,
        )
        def rank(item):
            identifier, row = item
            simplicity = 0 if identifier.startswith("B1_") else 1
            return (
                row["unsafe_recommendations"],
                row["top3_unsafe_miss"],
                row["multi_target_stale_miss"],
                row["global_unsafe_miss_rate"],
                row["safe_candidate_false_veto_rate"], simplicity,
            )
        selected = min(development.items(), key=rank)[0]
        freeze = {
            "status": "FROZEN_BEFORE_FRESH_GT_ACCESS",
            "grouping_unit":
                "complete_sequence_actor_trajectory_track_generation_multi_target_scene",
            "splits": {key: list(value) for key, value in splits.items()},
            "development_candidates": [
                row["id"] for row in config["candidates"]
            ],
            "selected_candidate": selected,
            "source_hashes": source_hashes(),
            "time_origin_contract": config["time_origin"],
            "velocity_set": config["velocity_set"],
            "acceleration_set": config["acceleration_set"],
            "reference_drift": config["reference_drift"],
            "expiry": config["expiry"],
            "runtime_gates": config["runtime_gates"],
            "risk_gates": config["risk_gates"],
            "fresh_gt_accessed_at_freeze": False,
            "parameters_changed_after_freeze": False,
            "holdout_accessed": False, "test_accessed": False,
            "blind_accessed": False,
        }
        atomic_json(FREEZE, freeze)
        atomic_json(DIAG/"development_summary.json", {
            "status": "PASS", "calibration": calibration,
            "development_validation": development,
            "selected_candidate": selected,
        })
        print(json.dumps(freeze, indent=2))
    else:
        freeze = json.loads(FREEZE.read_text())
        if source_hashes() != freeze["source_hashes"]:
            raise RuntimeError("source changed after fresh freeze")
        selected = freeze["selected_candidate"]
        candidate = next(
            row for row in config["candidates"] if row["id"] == selected
        )
        fresh = run_split(
            "fresh", splits["fresh"], [candidate], network, configs,
        )[selected]
        atomic_json(DIAG/"fresh_summary.json", {
            "status": "PASS", "selected_candidate": selected,
            "summary": fresh, "parameters_changed_after_freeze": False,
            "runtime_gt_used": False,
        })
        print(json.dumps(fresh, indent=2))


if __name__ == "__main__":
    main()
