#!/usr/bin/env python3
"""Grouped DOGMR1 geometry evaluation with a pre-access fresh freeze.

The runtime path receives only causal depth components and fixed contract
priors.  Actor metadata is joined after inference solely for offline metrics.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
DIAG = ROOT/"diagnostics/phase8jqv2_4dogmr1"
CONFIG = ROOT/"configs/dynamic_object_geometry_contract_v1_candidate.yaml"
sys.path.insert(0, str(ROOT))

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
from tools.run_phase8jqv2_4ocsr1_collect import associate_gt, ordinary_case
from tools.run_phase8jqv2_4tccr1_telemetry import make_perception


DYNAMIC_SCENARIOS = (
    "crossing", "head_on", "multi_target", "temporal_separation",
    "occluded_but_tracked",
)
ALL_SCENARIOS = ("no_target",)+DYNAMIC_SCENARIOS


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
        "count": int(len(values)),
        "mean": float(values.mean()),
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
    # PTAR1 consumed no-target [0:6] and dynamic [3:9].  DOGMR1 uses only
    # later complete sequences and never frame-randomizes.
    result = {"calibration": [], "development_validation": [], "fresh": []}
    for scenario in ALL_SCENARIOS:
        start = 6 if scenario == "no_target" else 9
        values = grouped[scenario]
        for offset, split in zip((0, 2, 4), result):
            result[split].extend(values[start+offset:start+offset+2])
    return {key: tuple(sorted(value)) for key, value in result.items()}


def load_case(sequence_id):
    case = ordinary_case(sequence_id)
    rows = list(csv.DictReader(
        (Path(case["source_root"])/"frames.csv").open()
    ))
    for index, row in enumerate(rows):
        objects = json.loads((
            Path(case["source_root"])/row["dynamic_objects_path"]
        ).read_text())
        lookup = {int(item["object_id"]): item for item in objects}
        for actor_id, truth in case["gt"][index].items():
            source = lookup[actor_id]
            truth["shape"] = str(source.get("type", "unknown"))
            truth["height_m"] = float(
                source.get("height", 2*truth["radius_m"])
            )
    case["negative"] = bool(
        case["scenario"] == "no_target" or not any(case["gt"])
    )
    return case


def hypothesis_metrics(hypothesis, truth):
    center = np.asarray(hypothesis.center_world)
    target = np.asarray(truth["position_world"])
    radius = float(truth["radius_m"])
    radius_covered = bool(
        hypothesis.radius_interval_m[0]-1e-9 <= radius
        <= hypothesis.radius_interval_m[1]+1e-9
    )
    if hypothesis.geometry_type == "sphere":
        center_error = float(np.linalg.norm(center-target))
        center_bound = max(
            .02, 2*float(np.sqrt(
                np.linalg.eigvalsh(
                    hypothesis.model_covariance_world
                ).max()
            ))
        )
        center_covered = center_error <= center_bound
        height_covered = True
        volume = 4*np.pi*hypothesis.radius_interval_m[1]**3/3
        vertical_overcoverage = 0.
    else:
        center_error = float(np.linalg.norm(center[:2]-target[:2]))
        center_bound = max(
            .02, 2*float(np.sqrt(
                np.linalg.eigvalsh(
                    hypothesis.model_covariance_world[:2, :2]
                ).max()
            ))
        )
        center_covered = center_error <= center_bound
        half_height = .5*float(truth["height_m"])
        height_covered = bool(
            hypothesis.center_z_interval_m[0]-1e-9 <= target[2]
            <= hypothesis.center_z_interval_m[1]+1e-9
            and hypothesis.half_height_interval_m[0]-1e-9
            <= half_height
            <= hypothesis.half_height_interval_m[1]+1e-9
        )
        volume = (
            np.pi*hypothesis.radius_interval_m[1]**2
            * 2*hypothesis.half_height_interval_m[1]
        )
        vertical_overcoverage = max(
            0., 2*hypothesis.half_height_interval_m[1]
            - float(truth["height_m"])
        )
    gt_volume = (
        4*np.pi*radius**3/3 if truth["shape"] == "sphere"
        else np.pi*radius**2*float(truth["height_m"])
    )
    return {
        "geometry_type": hypothesis.geometry_type,
        "center_error_m": center_error,
        "center_covered": center_covered,
        "radius_covered": radius_covered,
        "height_covered": height_covered,
        "full_covered": bool(
            hypothesis.geometry_type == truth["shape"]
            and center_covered and radius_covered and height_covered
        ),
        "volume_proxy_m3": float(volume),
        "volume_ratio_to_gt": float(volume/max(gt_volume, 1e-12)),
        "radial_overcoverage_m": max(
            0., hypothesis.radius_interval_m[1]-radius
        ),
        "vertical_overcoverage_m": vertical_overcoverage,
    }


def process_case(case, document):
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
    shape_tracker = ShapeHypothesisTrackerV1(
        temporal["minimum_direct_evidence_frames"],
        temporal["ambiguous_expiry_frames"],
        temporal["maximum_missed_frames"],
    )
    occupancy_builder = DynamicObjectOccupancyStateBuilderV1(
        temporal["velocity_smoothing"],
        temporal["mode_switch_velocity_variance_mps2"],
    )
    records, velocities, mode_rows = [], [], []
    feature_availability = Counter()
    runtimes = []
    false_geometry_tracks = 0
    previous_modes = {}
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
        live_ids = [track.track_id for track in result.all_tracks]
        shape_tracker.delete_missing(live_ids)
        occupancy_builder.delete_missing(live_ids)
        observations = {
            item.observation_id: item for item in result.observations
        }
        runtime_by_track = {}
        for track in result.all_tracks:
            generation = f"{track.birth_frame}:{track.birth_observation_id}"
            observation = observations.get(track.last_observation_id)
            direct = (
                observation is not None
                and track.last_direct_observation_frame == frame_index
            )
            started = time.perf_counter()
            if direct:
                geometry = adapter.export(
                    observation, frame,
                    f"{track.track_id}:{generation}",
                )
                evaluation = model.evaluate(geometry)
                tracked = shape_tracker.update(
                    track.track_id, generation, evaluation
                )
                for status in geometry.geometry_feature_validity.values():
                    feature_availability[status] += 1
            else:
                evaluation = None
                tracked = shape_tracker.predict_only(
                    track.track_id, generation, float(timestamp)
                )
            occupancy = (
                None if tracked is None
                else occupancy_builder.update(tracked)
            )
            runtimes.append((time.perf_counter()-started)*1000)
            runtime_by_track[track.track_id] = (
                evaluation, tracked, occupancy
            )
        active = [
            track for track in result.all_tracks
            if track.is_confirmed and track.is_dynamic
            and track.last_direct_observation_frame == frame_index
        ]
        if case["negative"]:
            false_geometry_tracks += len(active)
        assignment = associate_gt([
            {
                "track_id": int(track.track_id),
                "position_world": track.position_world.tolist(),
            } for track in active
        ], case["gt"][frame_index])
        for track in active:
            actor_id = assignment.get(track.track_id)
            runtime = runtime_by_track.get(track.track_id)
            if actor_id is None or runtime is None or runtime[0] is None:
                continue
            evaluation, tracked, occupancy = runtime
            truth = case["gt"][frame_index][actor_id]
            hypotheses = [
                hypothesis_metrics(item, truth)
                for item in evaluation.hypotheses
            ]
            key = (int(track.track_id), tracked.generation)
            previous_mode = previous_modes.get(key)
            if previous_mode is not None and previous_mode != tracked.mode:
                mode_rows.append({
                    "case_id": case["case_id"], "frame": frame_index,
                    "track_id": int(track.track_id),
                    "from": previous_mode, "to": tracked.mode,
                })
            previous_modes[key] = tracked.mode
            velocity_error = []
            if occupancy is not None:
                for item in occupancy.shape_hypotheses:
                    if item.geometry_type == truth["shape"]:
                        velocity_error.append(float(np.linalg.norm(
                            item.velocity_world
                            - np.asarray(truth["velocity_world"])
                        )))
                        velocities.append({
                            "case_id": case["case_id"],
                            "shape": truth["shape"],
                            "source": item.velocity_source,
                            "error_mps": velocity_error[-1],
                        })
            records.append({
                "case_id": case["case_id"],
                "scenario": case["scenario"],
                "camera_motion": case["camera_motion"],
                "frame": frame_index,
                "actor_id": int(actor_id),
                "track_id": int(track.track_id),
                "shape": truth["shape"],
                "observability": evaluation.observability.value,
                "hypothesis_count": len(evaluation.hypotheses),
                "hypotheses": hypotheses,
                "at_least_one_full_coverage": any(
                    item["full_covered"] for item in hypotheses
                ),
                "mode": tracked.mode,
                "velocity_error_mps": velocity_error,
                "runtime_gt_used": False,
            })
    return {
        "case_id": case["case_id"], "scenario": case["scenario"],
        "negative": case["negative"], "records": records,
        "velocities": velocities, "mode_switches": mode_rows,
        "negative_false_geometry_tracks": false_geometry_tracks,
        "feature_availability": dict(feature_availability),
        "runtime_ms": distribution(runtimes),
        "runtime_gt_used": False,
        "offline_gt_used_for_metrics_only": True,
    }


def summarize(split, cases):
    records = [row for case in cases for row in case["records"]]
    hypotheses = [
        item for row in records for item in row["hypotheses"]
    ]
    velocities = [row for case in cases for row in case["velocities"]]
    modes = [row for case in cases for row in case["mode_switches"]]
    shapes = sorted({row["shape"] for row in records})
    by_shape = {}
    for shape in shapes:
        rows = [row for row in records if row["shape"] == shape]
        matching = [
            item for row in rows for item in row["hypotheses"]
            if item["geometry_type"] == shape
        ]
        by_shape[shape] = {
            "sample_count": len(rows),
            "at_least_one_hypothesis_coverage": float(np.mean([
                row["at_least_one_full_coverage"] for row in rows
            ])) if rows else None,
            "center_coverage": float(np.mean([
                item["center_covered"] for item in matching
            ])) if matching else 0.,
            "radius_coverage": float(np.mean([
                item["radius_covered"] for item in matching
            ])) if matching else 0.,
            "height_coverage": float(np.mean([
                item["height_covered"] for item in matching
            ])) if matching else None,
            "full_geometry_coverage": float(np.mean([
                item["full_covered"] for item in matching
            ])) if matching else 0.,
            "volume_ratio_to_gt": distribution([
                item["volume_ratio_to_gt"] for item in matching
            ]),
            "radial_overcoverage_m": distribution([
                item["radial_overcoverage_m"] for item in matching
            ]),
            "velocity_error_mps": distribution([
                item["error_mps"] for item in velocities
                if item["shape"] == shape
                and item["source"] != "same_mode_uninitialized"
            ]),
        }
    confusion = Counter(
        f"{row['shape']}->{row['observability']}" for row in records
    )
    ambiguous = [
        row for row in records
        if row["observability"] in {"SHAPE_AMBIGUOUS", "SUPPORT_ONLY"}
    ]
    runtimes = [
        case["runtime_ms"]["p95"] for case in cases
        if case["runtime_ms"]["count"]
    ]
    return {
        "status": "PASS", "split": split,
        "case_count": len(cases), "record_count": len(records),
        "by_shape": by_shape,
        "overall_geometry_coverage": float(np.mean([
            row["at_least_one_full_coverage"] for row in records
        ])) if records else None,
        "observability": dict(Counter(
            row["observability"] for row in records
        )),
        "shape_confusion": dict(confusion),
        "ambiguous_record_count": len(ambiguous),
        "multi_hypothesis_record_count": sum(
            row["hypothesis_count"] == 2 for row in records
        ),
        "mode_switch_count": len(modes),
        "negative_false_geometry_tracks": sum(
            case["negative_false_geometry_tracks"] for case in cases
        ),
        "feature_availability": dict(sum(
            (Counter(case["feature_availability"]) for case in cases),
            Counter(),
        )),
        "geometry_model_case_p95_ms": distribution(runtimes),
        "all_hypothesis_volume_ratio": distribution([
            item["volume_ratio_to_gt"] for item in hypotheses
        ]),
        "runtime_gt_used": False,
    }


def main():
    document = yaml.safe_load(CONFIG.read_text())
    splits = grouped_sequences()
    if any(len(values) != 12 for values in splits.values()):
        raise RuntimeError(f"incomplete grouped split: {splits}")
    sets = {key: set(value) for key, value in splits.items()}
    if any(
        sets[left] & sets[right]
        for left, right in (
            ("calibration", "development_validation"),
            ("calibration", "fresh"),
            ("development_validation", "fresh"),
        )
    ):
        raise RuntimeError("sequence leakage")
    freeze = {
        "status": "FROZEN_BEFORE_FRESH_GT_ACCESS",
        "grouping_unit": "complete_sequence_actor_trajectory_track_generation",
        "splits": {key: list(value) for key, value in splits.items()},
        "source_hashes": {
            "config": digest(CONFIG),
            "adapter_v2": digest(ROOT/"policy/dynamic/measurement_geometry_adapter_v2.py"),
            "model_v1": digest(ROOT/"policy/dynamic/dynamic_object_geometry_model_v1.py"),
            "shape_tracker_v1": digest(ROOT/"policy/dynamic/shape_hypothesis_tracker_v1.py"),
            "occupancy_state_v1": digest(ROOT/"policy/dynamic/dynamic_object_occupancy_state_v1.py"),
            "risk_evaluator_v1": digest(ROOT/"tools/evaluate_dynamic_geometry_risk_v1.py"),
            "geometry_evaluator": digest(Path(__file__)),
        },
        "fresh_gt_accessed_at_freeze": False,
        "parameters_changed_after_freeze": False,
        "holdout_accessed": False, "test_accessed": False,
        "blind_accessed": False,
    }
    atomic_json(
        REPORTS/"phase8jqv2_4dogmr1_fresh_validation_freeze.json",
        freeze,
    )
    summaries = {}
    for split in ("calibration", "development_validation", "fresh"):
        results = []
        for index, sequence_id in enumerate(splits[split], 1):
            result = process_case(load_case(sequence_id), document)
            results.append(result)
            print(json.dumps({
                "split": split, "progress": f"{index}/{len(splits[split])}",
                "sequence": sequence_id, "records": len(result["records"]),
            }))
        summary = summarize(split, results)
        atomic_json(DIAG/f"{split}_geometry_evaluation.json", {
            "split": split, "cases": results, "summary": summary,
        })
        summaries[split] = summary
    atomic_json(DIAG/"geometry_evaluation_summary.json", {
        "status": "PASS", "splits": summaries,
        "fresh_parameters_changed_after_freeze": False,
        "runtime_gt_used": False,
    })
    print(json.dumps(summaries["fresh"], indent=2))


if __name__ == "__main__":
    main()
