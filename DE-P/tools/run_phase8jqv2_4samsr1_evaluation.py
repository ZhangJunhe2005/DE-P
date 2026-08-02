#!/usr/bin/env python3
"""Staged SAMSR1 development/fresh host-CUDA shadow evaluation."""

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
DIAG = ROOT/"diagnostics/phase8jqv2_4samsr1"
CONFIG = ROOT/"configs/shape_aware_motion_state_contract_v1_candidate.yaml"
DOG_CONFIG = ROOT/"configs/dynamic_object_geometry_contract_v1_candidate.yaml"
FREEZE = REPORTS/"phase8jqv2_4samsr1_fresh_validation_freeze.json"
sys.path.insert(0, str(ROOT))

from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_network import DepNetwork
from policy.dynamic.camera_model import make_depth_frame
from policy.dynamic.dynamic_object_geometry_model_v1 import (
    DynamicObjectGeometryModelV1, GeometryModelConfigV1,
)
from policy.dynamic.shape_aware_dynamic_occupancy_v1 import (
    FastGeometryUpdateCacheV1, ReachabilityConfigV1,
    evaluate_shape_aware_risk,
)
from policy.dynamic.shape_motion_hypothesis_tracker_v1 import (
    ShapeMotionHypothesisTrackerV1,
)
from tools.run_phase8jqv2_4dogmr1_geometry_evaluation import (
    hypothesis_metrics, load_case,
)
from tools.run_phase8jqv2_4ocsr1_collect import associate_gt
from tools.run_phase8jqv2_4ocsr1_risk import (
    candidates_for_frame, gt_at_times, selected_config, yopo_auxiliary,
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
        "motion_calibration": [], "motion_development_validation": [],
        "fresh": [],
    }
    for scenario in SCENARIOS:
        values = grouped[scenario]
        for offset, split in zip((15, 17, 19), result):
            result[split].extend(values[offset:offset+2])
    return {key: tuple(sorted(value)) for key, value in result.items()}


def exact_gt_shape_risk(case, frame, candidates, times, uav_radius, margin):
    actors, radii, valid = gt_at_times(case, frame, times)
    actor_ids = sorted({key for row in case["gt"] for key in row})
    rows = []
    for candidate_id, candidate in enumerate(candidates):
        best = {
            "clearance": float("inf"), "actor_id": None,
            "shape": None, "time": None,
        }
        complete = actors.shape[0] == 0 or bool(valid.all())
        for index, actor_id in enumerate(actor_ids):
            mask = valid[index]
            if not mask.any():
                continue
            source = [row[actor_id] for row in case["gt"] if actor_id in row]
            shape = source[0]["shape"]
            if shape == "sphere":
                signed = (
                    np.linalg.norm(
                        candidate[mask]-actors[index, mask], axis=1
                    )-radii[index]
                )
            elif shape == "vertical_cylinder":
                half = .5*float(np.median([
                    row["height_m"] for row in source
                ]))
                delta = candidate[mask]-actors[index, mask]
                radial = np.linalg.norm(delta[:, :2], axis=1)-radii[index]
                vertical = np.abs(delta[:, 2])-half
                signed = (
                    np.sqrt(
                        np.maximum(radial, 0.)**2
                        + np.maximum(vertical, 0.)**2
                    )+np.minimum(np.maximum(radial, vertical), 0.)
                )
            else:
                complete = False
                continue
            clearance = signed-uav_radius-margin
            local = int(np.argmin(clearance))
            indices = np.flatnonzero(mask)
            sample = int(indices[local])
            if clearance[local] < best["clearance"]:
                best = {
                    "clearance": float(clearance[local]),
                    "actor_id": int(actor_id), "shape": shape,
                    "time": float(times[sample]),
                }
        classification = (
            "GT_SAFE" if actors.shape[0] == 0 or best["clearance"] >= 0
            else "GT_UNSAFE"
        )
        if not complete:
            classification = "GT_UNKNOWN"
        rows.append({
            "candidate_trajectory_id": candidate_id,
            "classification": classification,
            "minimum_required_clearance_m": best["clearance"],
            "limiting_actor_id": best["actor_id"],
            "limiting_shape": best["shape"],
            "minimum_clearance_time_s": best["time"],
        })
    return rows


def compare_risk(predicted, truth, scores):
    prediction = {
        row["candidate_trajectory_id"]: row
        for row in predicted["candidate_rows"]
    }
    safe = {
        row["candidate_trajectory_id"] for row in truth
        if row["classification"] == "GT_SAFE"
    }
    unsafe = {
        row["candidate_trajectory_id"] for row in truth
        if row["classification"] == "GT_UNSAFE"
    }
    unknown = {
        row["candidate_trajectory_id"] for row in truth
        if row["classification"] == "GT_UNKNOWN"
    }
    veto = {
        key for key, row in prediction.items() if row["would_veto"]
    }
    retained = sorted(set(range(len(scores)))-veto)
    recommended = (
        min(retained, key=lambda index: float(scores[index]))
        if retained else None
    )
    original = int(np.argmin(scores))
    top3 = set(np.argsort(scores)[:3].tolist())
    return {
        "safe": sorted(safe), "unsafe": sorted(unsafe),
        "unknown": sorted(unknown), "veto": sorted(veto),
        "missed_unsafe": sorted(unsafe-veto),
        "top3_unsafe_miss": sorted((unsafe & top3)-veto),
        "false_vetoed_safe": sorted(safe & veto),
        "retained_safe": sorted(safe-veto),
        "recommended_candidate_id": recommended,
        "original_candidate_id": original,
        "unsafe_recommendation": bool(recommended in unsafe),
        "false_emergency": bool(not retained and safe),
        "no_safe_candidate_correct": bool(not retained and not safe and unsafe),
        "truth_rows": truth, "prediction_rows": predicted["candidate_rows"],
    }


def taxonomy_rows(case, frame_index, comparison, scores, states, motion_rows):
    truth = {
        row["candidate_trajectory_id"]: row
        for row in comparison["truth_rows"]
    }
    predicted = {
        row["candidate_trajectory_id"]: row
        for row in comparison["prediction_rows"]
    }
    rows = []
    for candidate_id in comparison["missed_unsafe"]:
        target = truth[candidate_id]
        direct_states = [state for state in states if not state.prediction_only]
        if not states:
            category = "hypothesis_expired_too_early"
        elif all(state.prediction_only for state in states):
            category = "stale_geometry_after_miss"
        elif any(
            item.motion.observability.value == "MOTION_INITIALIZING"
            for state in states for item in state.hypotheses
        ):
            category = "correct_center_wrong_velocity"
        else:
            category = "velocity_magnitude_error"
        actor_motion = [
            row for row in motion_rows
            if row.get("actor_id") == target["limiting_actor_id"]
        ]
        rows.append({
            "case_id": case["case_id"], "frame": frame_index,
            "candidate_id": candidate_id,
            "candidate_rank": int(np.where(
                np.argsort(scores) == candidate_id
            )[0][0])+1,
            "true_collision_actor": target["limiting_actor_id"],
            "shape_authority": target["limiting_shape"],
            "gt_collision_time_s": target["minimum_clearance_time_s"],
            "predicted_collision_time_s": (
                predicted[candidate_id]["limiting"]["time"]
                if predicted[candidate_id]["limiting"] else None
            ),
            "runtime_shape_states": [
                {
                    "track_id": state.track_id,
                    "generation": state.generation,
                    "prediction_only": state.prediction_only,
                    "missed_frames": state.missed_frames,
                    "observability": state.observability.value,
                    "hypotheses": [
                        {
                            "shape": item.geometry_type,
                            "reference_mode": item.motion.reference_mode,
                            "velocity_world": item.motion.velocity_world.tolist(),
                            "effective_history":
                                item.motion.effective_history,
                            "last_direct_geometry_time":
                                item.motion.last_direct_timestamp,
                        } for item in state.hypotheses
                    ],
                } for state in states
            ],
            "motion_evidence": actor_motion,
            "direct_state_count": len(direct_states),
            "category": category,
            "runtime_gt_used": False,
            "offline_gt_used_for_taxonomy_only": True,
        })
    return rows


def process_case(case, network, config, dog_config, candidate):
    sensor = {
        "height": case["model"].height, "width": case["model"].width,
        "intrinsics": [
            case["model"].fx, case["model"].fy,
            case["model"].cx, case["model"].cy,
        ], "min_depth_m": case["model"].min_depth,
        "max_depth_m": case["model"].max_depth,
    }
    perception, _ = make_perception(sensor)
    model = DynamicObjectGeometryModelV1(
        GeometryModelConfigV1.from_mapping(
            dog_config["runtime_priors"], dog_config["observability"],
            dog_config["features"],
        )
    )
    geometry_cache = FastGeometryUpdateCacheV1(model)
    motion_tracker = ShapeMotionHypothesisTrackerV1(
        config["history"], config["reference_transition"]
    )
    reachability = ReachabilityConfigV1(
        float(candidate["acceleration_bound_mps2"]),
        float(candidate["reference_drift_rate_mps"]),
    )
    safety = selected_config()
    velocity, acceleration, goal, rotation = yopo_auxiliary(case)
    motion_records, risk_records, taxonomy = [], [], []
    query_runtime = []
    prediction_frame_runtime = []
    state_by_track = {}
    for frame_index, timestamp in enumerate(case["timestamps"]):
        frame_started = time.perf_counter()
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
        observations = {
            item.observation_id: item for item in result.observations
        }
        states = []
        state_by_track.clear()
        for track in result.all_tracks:
            generation = f"{track.birth_frame}:{track.birth_observation_id}"
            observation = observations.get(track.last_observation_id)
            if (
                observation is not None
                and track.last_direct_observation_frame == frame_index
            ):
                geometry, evaluation = geometry_cache.update(
                    observation, frame, generation
                )
                state = motion_tracker.update(
                    track.track_id, generation, evaluation
                )
            else:
                state = motion_tracker.predict_only(
                    track.track_id, generation, float(timestamp)
                )
            if state is not None:
                state_by_track[track.track_id] = state
            if (
                state is not None and state.hypotheses
                and track.is_confirmed and track.is_dynamic
            ):
                states.append(state)
        if not any(
            track.last_direct_observation_frame == frame_index
            for track in result.all_tracks
        ):
            prediction_frame_runtime.append(
                (time.perf_counter()-frame_started)*1000
            )
        active = [
            track for track in result.all_tracks
            if track.is_confirmed and track.is_dynamic
            and track.last_direct_observation_frame == frame_index
        ]
        assignment = associate_gt([
            {"track_id": int(track.track_id),
             "position_world": track.position_world.tolist()}
            for track in active
        ], case["gt"][frame_index])
        frame_motion = []
        for track in active:
            actor_id = assignment.get(track.track_id)
            state = state_by_track.get(track.track_id)
            if actor_id is None or state is None:
                continue
            truth = case["gt"][frame_index][actor_id]
            for item in state.hypotheses:
                metric = hypothesis_metrics(
                    item.geometry_hypothesis, truth
                )
                velocity_error = float(np.linalg.norm(
                    item.motion.velocity_world
                    - np.asarray(truth["velocity_world"])
                ))
                row = {
                    "case_id": case["case_id"], "frame": frame_index,
                    "actor_id": int(actor_id),
                    "track_id": int(track.track_id),
                    "generation": state.generation,
                    "shape_authority": truth["shape"],
                    "hypothesis_shape": item.geometry_type,
                    "shape_matches": item.geometry_type == truth["shape"],
                    "full_geometry_coverage": metric["full_covered"],
                    "motion_observability":
                        item.motion.observability.value,
                    "velocity_error_mps": velocity_error,
                    "velocity_world": item.motion.velocity_world.tolist(),
                    "gt_velocity_world":
                        np.asarray(truth["velocity_world"]).tolist(),
                    "reference_mode": item.motion.reference_mode,
                    "effective_history": item.motion.effective_history,
                    "prediction_only": state.prediction_only,
                }
                frame_motion.append(row)
                motion_records.append(row)
        should_evaluate = bool(case["negative"]) or bool(states)
        if not should_evaluate:
            continue
        candidates, times, scores = candidates_for_frame(
            network, case, frame_index,
            velocity, acceleration, goal, rotation,
        )
        started = time.perf_counter()
        predicted = evaluate_shape_aware_risk(
            candidates, times, states, reachability,
            uav_radius_m=safety.uav_radius_m,
            required_margin_m=safety.clearance_margin_m,
        )
        query_runtime.append((time.perf_counter()-started)*1000)
        truth = exact_gt_shape_risk(
            case, frame_index, candidates, times,
            safety.uav_radius_m, safety.clearance_margin_m,
        )
        comparison = compare_risk(predicted, truth, scores)
        if comparison["unknown"]:
            continue
        record = {
            key: value for key, value in comparison.items()
            if key not in {"truth_rows", "prediction_rows"}
        }
        record.update({
            "case_id": case["case_id"], "scenario": case["scenario"],
            "frame": frame_index, "negative": case["negative"],
        })
        risk_records.append(record)
        taxonomy.extend(taxonomy_rows(
            case, frame_index, comparison, scores, states, frame_motion
        ))
    return {
        "case_id": case["case_id"], "scenario": case["scenario"],
        "negative": case["negative"], "candidate": candidate["id"],
        "motion_records": motion_records, "risk_records": risk_records,
        "unsafe_miss_taxonomy": taxonomy,
        "direct_profile": geometry_cache.profile,
        "geometry_fit_calls": geometry_cache.fit_calls,
        "cache_hits": geometry_cache.cache_hits,
        "prediction_frame_runtime_ms": prediction_frame_runtime,
        "risk_query_runtime_ms": query_runtime,
        "runtime_gt_used": False,
    }


def summarize(split, candidate, cases):
    motion = [row for case in cases for row in case["motion_records"]]
    risk = [row for case in cases for row in case["risk_records"]]
    taxonomy = [
        row for case in cases for row in case["unsafe_miss_taxonomy"]
    ]
    unsafe = sum(len(row["unsafe"]) for row in risk)
    safe = sum(len(row["safe"]) for row in risk)
    false_veto = sum(len(row["false_vetoed_safe"]) for row in risk)
    missed = sum(len(row["missed_unsafe"]) for row in risk)
    queries = len(risk)
    case_safe = defaultdict(bool)
    case_false_emergency = defaultdict(bool)
    for row in risk:
        case_safe[row["case_id"]] |= bool(row["safe"])
        case_false_emergency[row["case_id"]] |= row["false_emergency"]
    safe_cases = sum(case_safe.values())
    shape_rows = sorted({row["shape_authority"] for row in motion})
    direct = [row["elapsed_ms"] for case in cases for row in case["direct_profile"]]
    query_runtime = [
        value for case in cases for value in case["risk_query_runtime_ms"]
    ]
    prediction_runtime = [
        value for case in cases
        for value in case["prediction_frame_runtime_ms"]
    ]
    return {
        "status": "PASS", "split": split, "candidate": candidate,
        "case_count": len(cases), "risk_query_count": queries,
        "true_unsafe_candidates": unsafe, "missed_unsafe": missed,
        "global_unsafe_miss_rate": missed/unsafe if unsafe else 0.,
        "top3_unsafe_miss": sum(
            len(row["top3_unsafe_miss"]) for row in risk
        ),
        "unsafe_recommendations": sum(
            row["unsafe_recommendation"] for row in risk
        ),
        "false_vetoed_safe": false_veto,
        "candidate_global_false_veto_rate": (
            false_veto/(queries*15) if queries else 0.
        ),
        "safe_candidate_false_veto_rate": (
            false_veto/safe if safe else 0.
        ),
        "sequence_false_emergency_rate": (
            sum(case_false_emergency.values())/safe_cases
            if safe_cases else 0.
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
        "no_safe_candidate_correct": sum(
            row["no_safe_candidate_correct"] for row in risk
        ),
        "motion_by_shape": {
            shape: {
                "sample_count": sum(
                    row["shape_authority"] == shape for row in motion
                ),
                "matching_hypothesis_count": sum(
                    row["shape_authority"] == shape
                    and row["shape_matches"] for row in motion
                ),
                "velocity_error_mps": distribution([
                    row["velocity_error_mps"] for row in motion
                    if row["shape_authority"] == shape
                    and row["shape_matches"]
                    and row["motion_observability"]
                        != "MOTION_INITIALIZING"
                ]),
                "geometry_coverage": (
                    float(np.mean([
                        row["full_geometry_coverage"] for row in motion
                        if row["shape_authority"] == shape
                        and row["shape_matches"]
                    ])) if any(
                        row["shape_authority"] == shape
                        and row["shape_matches"] for row in motion
                    ) else 0.
                ),
            } for shape in shape_rows
        },
        "motion_observability": dict(Counter(
            row["motion_observability"] for row in motion
        )),
        "taxonomy": dict(Counter(
            row["category"] for row in taxonomy
        )),
        "direct_geometry_update_ms": distribution(direct),
        "prediction_frame_ms": distribution(prediction_runtime),
        "risk_query_ms": distribution(query_runtime),
        "geometry_fit_calls": sum(
            case["geometry_fit_calls"] for case in cases
        ),
        "cache_hits": sum(case["cache_hits"] for case in cases),
        "runtime_gt_used": False,
    }


def source_hashes():
    paths = {
        "config": CONFIG,
        "geometry_model": ROOT/"policy/dynamic/dynamic_object_geometry_model_v1.py",
        "motion_state": ROOT/"policy/dynamic/shape_aware_motion_state_v1.py",
        "motion_tracker": ROOT/"policy/dynamic/shape_motion_hypothesis_tracker_v1.py",
        "support_occupancy": ROOT/"policy/dynamic/support_reachable_occupancy_v1.py",
        "dynamic_occupancy": ROOT/"policy/dynamic/shape_aware_dynamic_occupancy_v1.py",
        "risk_evaluator": ROOT/"tools/evaluate_dynamic_geometry_risk_v1.py",
        "evaluator": Path(__file__),
    }
    return {key: digest(value) for key, value in paths.items()}


def run_split(split, sequences, candidates, network, config, dog_config):
    outputs = {}
    for candidate in candidates:
        cases = []
        for index, sequence in enumerate(sequences, 1):
            case_result = process_case(
                load_case(sequence), network, config, dog_config, candidate
            )
            cases.append(case_result)
            print(json.dumps({
                "split": split, "candidate": candidate["id"],
                "progress": f"{index}/{len(sequences)}",
                "case": sequence,
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
    parser.add_argument("--stage", choices=("development", "fresh"), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required for frozen YOPO replay")
    config = yaml.safe_load(CONFIG.read_text())
    dog_config = yaml.safe_load(DOG_CONFIG.read_text())
    splits = grouped_sequences()
    if any(len(values) != 12 for values in splits.values()):
        raise RuntimeError(f"incomplete grouped split: {splits}")
    sets = {key: set(values) for key, values in splits.items()}
    if any(
        sets[left] & sets[right] for left, right in (
            ("motion_calibration", "motion_development_validation"),
            ("motion_calibration", "fresh"),
            ("motion_development_validation", "fresh"),
        )
    ):
        raise RuntimeError("split leakage")
    network = DepNetwork(backbone_variant="legacy").cuda().eval()
    load_dep_checkpoint(
        network, ROOT/"saved/DEP_0/epoch10.pth", "legacy"
    )
    if args.stage == "development":
        candidates = config["reachable_candidates"]
        calibration = run_split(
            "motion_calibration", splits["motion_calibration"],
            candidates, network, config, dog_config,
        )
        development = run_split(
            "motion_development_validation",
            splits["motion_development_validation"],
            candidates, network, config, dog_config,
        )
        def rank(item):
            _, row = item
            return (
                row["unsafe_recommendations"],
                row["top3_unsafe_miss"], row["global_unsafe_miss_rate"],
                row["safe_candidate_false_veto_rate"],
            )
        selected = min(development.items(), key=rank)[0]
        freeze = {
            "status": "FROZEN_BEFORE_FRESH_GT_ACCESS",
            "grouping_unit":
                "complete_sequence_actor_trajectory_track_generation",
            "splits": {key: list(value) for key, value in splits.items()},
            "development_candidates": [
                row["id"] for row in candidates
            ],
            "selected_candidate": selected,
            "source_hashes": source_hashes(),
            "runtime_gates": config["runtime_gates"],
            "risk_gates": config["risk_gates"],
            "motion_error_budget": config["motion_error_budget"],
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
        current = source_hashes()
        if current != freeze["source_hashes"]:
            raise RuntimeError("candidate source changed after fresh freeze")
        selected = freeze["selected_candidate"]
        candidate = next(
            row for row in config["reachable_candidates"]
            if row["id"] == selected
        )
        fresh = run_split(
            "fresh", splits["fresh"], [candidate],
            network, config, dog_config,
        )
        atomic_json(DIAG/"fresh_summary.json", {
            "status": "PASS", "selected_candidate": selected,
            "summary": fresh[selected],
            "parameters_changed_after_freeze": False,
            "runtime_gt_used": False,
        })
        print(json.dumps(fresh[selected], indent=2))


if __name__ == "__main__":
    main()
