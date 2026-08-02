#!/usr/bin/env python3
"""Run frozen PTAR1 reference-alignment development/fresh evaluation.

Only existing ``phase8c_train`` development sequences are read.  The split is
grouped by complete sequence and the fresh manifest is frozen before any fresh
depth or actor ground truth is loaded.
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
DIAG = ROOT / "diagnostics/phase8jqv2_4ptar1"
REPORTS = ROOT / "reports"
CONFIG = ROOT / "configs/tracking_collision_reference_contract_v1_candidate.yaml"
sys.path.insert(0, str(ROOT))

from policy.dynamic.camera_model import make_depth_frame
from policy.dynamic.measurement_geometry_adapter_v1 import (
    MeasurementGeometryAdapterV1,
)
from policy.dynamic.reference_aligned_center_tracker_v1 import (
    ReferenceAlignedCenterTrackerV1,
)
from policy.dynamic.tracking_collision_reference_bridge_v1 import (
    ReferenceBridgeConfigV1, TrackingCollisionReferenceBridgeV1,
    fixed_radial_shift,
)
from tools.run_phase8jqv2_4ocsr1_collect import (
    associate_gt, ordinary_case,
)
from tools.run_phase8jqv2_4tccr1_telemetry import make_perception


DYNAMIC_SCENARIOS = (
    "crossing", "head_on", "multi_target", "temporal_separation",
    "occluded_but_tracked",
)
ALL_SCENARIOS = ("no_target",) + DYNAMIC_SCENARIOS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(values.max()),
    }


def grouped_sequences():
    grouped = defaultdict(list)
    root = ROOT / "data/phase8_dynamic_production/sequences"
    for path in sorted(root.glob("phase8c_train_*/metadata.yaml")):
        metadata = yaml.safe_load(path.read_text())
        grouped[metadata["scenario_type"]].append(path.parent.name)
    result = {"calibration": [], "development_validation": [], "fresh": []}
    for scenario in ALL_SCENARIOS:
        values = grouped[scenario]
        start = 0 if scenario == "no_target" else 3
        result["calibration"].extend(values[start:start+2])
        result["development_validation"].extend(values[start+2:start+4])
        result["fresh"].extend(values[start+4:start+6])
    return {key: tuple(sorted(value)) for key, value in result.items()}


def enrich_geometry(case):
    rows = list(csv.DictReader((Path(case["source_root"])/"frames.csv").open()))
    for frame, row in enumerate(rows):
        objects = json.loads((
            Path(case["source_root"])/row["dynamic_objects_path"]
        ).read_text())
        lookup = {int(value["object_id"]): value for value in objects}
        for actor_id, gt in case["gt"][frame].items():
            source = lookup[actor_id]
            gt["shape"] = str(source.get("type", "unknown"))
            gt["height_m"] = float(
                source.get("height", 2*gt["radius_m"])
            )
    return case


def load_case(sequence_id):
    case = enrich_geometry(ordinary_case(sequence_id))
    case["negative"] = bool(
        case["scenario"] == "no_target"
        or not any(case["gt"])
    )
    return case


def track_row(track):
    return {
        "track_id": int(track.track_id),
        "position_world": track.position_world.tolist(),
    }


def actor_half_extent(gt):
    radius = float(gt["radius_m"])
    if gt.get("shape") == "vertical_cylinder":
        return np.asarray([radius, radius, .5*float(gt["height_m"])])
    return np.asarray([radius, radius, radius])


def center_record(
    *, case, frame_index, horizon, track, actor_id, truth, predicted,
    covariance, source, evidence=None, state=None,
):
    error = np.asarray(predicted, dtype=np.float64)-np.asarray(
        truth["position_world"], dtype=np.float64
    )
    position_covariance = np.asarray(covariance, dtype=np.float64)[:3, :3]
    try:
        normalized = float(np.sqrt(max(
            error@np.linalg.solve(position_covariance, error), 0.
        )))
    except np.linalg.LinAlgError:
        normalized = float("inf")
    camera = case["poses"][frame_index+horizon].position_world
    toward = camera-np.asarray(truth["position_world"])
    cosine = float(
        error@toward/max(np.linalg.norm(error)*np.linalg.norm(toward), 1e-12)
    )
    return {
        "case_id": case["case_id"],
        "scenario": case["scenario"],
        "camera_motion": case["camera_motion"],
        "frame": frame_index,
        "horizon_frames": horizon,
        "actor_id": int(actor_id),
        "track_id": int(track.track_id),
        "source": source,
        "position_error_vector_m": error.tolist(),
        "position_error_m": float(np.linalg.norm(error)),
        "error_toward_camera_cosine": cosine,
        "error_radius_ratio":
            float(np.linalg.norm(error)/truth["radius_m"]),
        "normalized_error": normalized,
        "position_std_m": float(np.sqrt(max(
            np.linalg.eigvalsh(position_covariance).max(), 0.
        ))),
        "observability": (
            None if evidence is None else evidence.observability.value
        ),
        "reference_mode": (
            None if evidence is None else evidence.reference_mode
        ),
        "image_border_clipped": (
            None if evidence is None else
            evidence.fallback_reason == "border_or_radius_identifiability"
        ),
        "center_velocity_source": (
            None if state is None else state.center_velocity_source
        ),
        "runtime_gt_used": False,
        "offline_gt_used": True,
    }


def process_case(case, bridge_config):
    sensor = {
        "height": case["model"].height, "width": case["model"].width,
        "intrinsics": [
            case["model"].fx, case["model"].fy,
            case["model"].cx, case["model"].cy,
        ],
        "min_depth_m": case["model"].min_depth,
        "max_depth_m": case["model"].max_depth,
    }
    perception, config = make_perception(sensor)
    geometry_adapter = MeasurementGeometryAdapterV1()
    bridge = TrackingCollisionReferenceBridgeV1(bridge_config)
    center_tracker = ReferenceAlignedCenterTrackerV1()
    errors, velocity_errors, direct, envelope = [], [], [], []
    runtimes = {"geometry_export_ms": [], "reference_bridge_ms": []}
    observability = Counter()
    false_reference_tracks = 0
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
            observation.observation_id: observation
            for observation in result.observations
        }
        evidence_by_track, state_by_track = {}, {}
        for track in result.all_tracks:
            observation = observations.get(track.last_observation_id)
            direct_measurement = (
                observation is not None
                and track.last_direct_observation_frame == frame_index
            )
            if not direct_measurement:
                continue
            started = time.perf_counter()
            geometry = geometry_adapter.export(observation, frame)
            runtimes["geometry_export_ms"].append(
                (time.perf_counter()-started)*1000
            )
            evidence = bridge.evaluate_geometry(geometry)
            runtimes["reference_bridge_ms"].append(bridge.last_runtime_ms)
            observability[evidence.observability.value] += 1
            generation = f"{track.birth_frame}:{track.birth_observation_id}"
            state = center_tracker.update(
                track.track_id, generation, evidence
            )
            evidence_by_track[track.track_id] = (evidence, geometry)
            if state is not None:
                state_by_track[track.track_id] = state
        active = [
            track for track in result.all_tracks
            if (
                track.is_confirmed and track.is_dynamic
                and track.last_direct_observation_frame == frame_index
            )
        ]
        assignment = associate_gt(
            [track_row(track) for track in active], case["gt"][frame_index]
        )
        if case["negative"]:
            false_reference_tracks += len(active)
        for track in active:
            actor_id = assignment.get(track.track_id)
            pair = evidence_by_track.get(track.track_id)
            state = state_by_track.get(track.track_id)
            if actor_id is None or pair is None or state is None:
                continue
            evidence, geometry = pair
            truth_now = case["gt"][frame_index][actor_id]
            r1 = {
                str(scale): fixed_radial_shift(
                    geometry, bridge_config.radius_prior_max_m, scale
                ) for scale in bridge_config.fixed_shift_scales
            }
            direct.append({
                "case_id": case["case_id"],
                "scenario": case["scenario"],
                "frame": frame_index,
                "actor_id": int(actor_id),
                "track_id": int(track.track_id),
                "r0_surface_error_m": float(np.linalg.norm(
                    track.position_world-truth_now["position_world"]
                )),
                "r1_error_m": {
                    scale: float(np.linalg.norm(
                        value-truth_now["position_world"]
                    )) for scale, value in r1.items()
                },
                "r6_center_error_m": float(np.linalg.norm(
                    state.position_world-truth_now["position_world"]
                )),
                "reference_mode": evidence.reference_mode,
                "observability": evidence.observability.value,
                "fit_residual_m": evidence.fit_residual_m,
                "fit_condition_number": evidence.fit_condition_number,
                "image_border_clipped": geometry.image_border_clipped,
                "camera_motion": case["camera_motion"],
                "shape": truth_now["shape"],
                "radius_m": truth_now["radius_m"],
                "runtime_gt_used": False,
            })
            half = actor_half_extent(truth_now)
            center_delta = np.abs(
                state.position_world-np.asarray(truth_now["position_world"])
            )
            covered = bool(np.all(
                center_delta+half <= evidence.support_half_extent_world+1e-9
            ))
            envelope.append({
                "case_id": case["case_id"],
                "scenario": case["scenario"],
                "shape": truth_now["shape"],
                "covered": covered,
                "radial_size_m": float(
                    np.max(evidence.support_half_extent_world)
                ),
                "volume_proxy_m3": float(
                    8*np.prod(evidence.support_half_extent_world)
                ),
                "observability": evidence.observability.value,
            })
            velocity_error = (
                state.velocity_world-np.asarray(truth_now["velocity_world"])
            )
            velocity_errors.append({
                "case_id": case["case_id"],
                "scenario": case["scenario"],
                "track_id": int(track.track_id),
                "frame": frame_index,
                "velocity_error_vector_mps": velocity_error.tolist(),
                "velocity_error_mps": float(np.linalg.norm(velocity_error)),
                "source": state.center_velocity_source,
            })
            for horizon in (1, 2, 3):
                future = frame_index+horizon
                if (
                    future >= len(case["timestamps"])
                    or actor_id not in case["gt"][future]
                ):
                    continue
                elapsed = float(
                    case["timestamps"][future]-case["timestamps"][frame_index]
                )
                raw_predicted = track.position_world+elapsed*track.velocity_world
                raw_transition = np.eye(6)
                raw_transition[:3, 3:] = np.eye(3)*elapsed
                raw_covariance = (
                    raw_transition@track.state_covariance@raw_transition.T
                )
                center_predicted = (
                    state.position_world+elapsed*state.velocity_world
                )
                center_transition = np.eye(6)
                center_transition[:3, 3:] = np.eye(3)*elapsed
                center_covariance = (
                    center_transition@state.state_covariance
                    @ center_transition.T
                )
                truth = case["gt"][future][actor_id]
                errors.append(center_record(
                    case=case, frame_index=frame_index, horizon=horizon,
                    track=track, actor_id=actor_id, truth=truth,
                    predicted=raw_predicted, covariance=raw_covariance,
                    source="R0_SURFACE_RAW",
                ))
                errors.append(center_record(
                    case=case, frame_index=frame_index, horizon=horizon,
                    track=track, actor_id=actor_id, truth=truth,
                    predicted=center_predicted, covariance=center_covariance,
                    source="R6_HYBRID_REFERENCE_BRIDGE",
                    evidence=evidence, state=state,
                ))
    return {
        "case_id": case["case_id"],
        "scenario": case["scenario"],
        "camera_motion": case["camera_motion"],
        "negative": case["negative"],
        "prediction_errors": errors,
        "direct_reference_errors": direct,
        "velocity_errors": velocity_errors,
        "support_envelope": envelope,
        "observability": dict(observability),
        "negative_false_reference_tracks": false_reference_tracks,
        "runtime": {
            key: distribution(value) for key, value in runtimes.items()
        },
        "runtime_gt_used": False,
    }


def summarize(split, cases):
    errors = [
        row for case in cases for row in case["prediction_errors"]
    ]
    direct = [
        row for case in cases for row in case["direct_reference_errors"]
    ]
    velocity = [row for case in cases for row in case["velocity_errors"]]
    envelope = [row for case in cases for row in case["support_envelope"]]
    by_source = {}
    for source in ("R0_SURFACE_RAW", "R6_HYBRID_REFERENCE_BRIDGE"):
        rows = [row for row in errors if row["source"] == source]
        by_source[source] = {
            "overall": distribution([row["position_error_m"] for row in rows]),
            "by_horizon": {
                str(horizon): distribution([
                    row["position_error_m"] for row in rows
                    if row["horizon_frames"] == horizon
                ]) for horizon in (1, 2, 3)
            },
            "normalized": distribution([
                row["normalized_error"] for row in rows
                if np.isfinite(row["normalized_error"])
            ]),
            "coverage": {
                str(sigma): float(np.mean([
                    row["normalized_error"] <= sigma for row in rows
                ])) if rows else None for sigma in (1, 2, 3)
            },
        }
    scenarios = sorted({row["scenario"] for row in direct})
    return {
        "status": "PASS",
        "split": split,
        "case_count": len(cases),
        "negative_case_count": sum(case["negative"] for case in cases),
        "prediction": by_source,
        "direct": {
            "R0": distribution([
                row["r0_surface_error_m"] for row in direct
            ]),
            "R1": {
                scale: distribution([
                    row["r1_error_m"][scale] for row in direct
                ]) for scale in ("0.5", "0.75", "1.0")
            },
            "R6": distribution([
                row["r6_center_error_m"] for row in direct
            ]),
            "by_scenario": {
                scenario: {
                    "R0": distribution([
                        row["r0_surface_error_m"] for row in direct
                        if row["scenario"] == scenario
                    ]),
                    "R6": distribution([
                        row["r6_center_error_m"] for row in direct
                        if row["scenario"] == scenario
                    ]),
                } for scenario in scenarios
            },
            "edge": {
                "R0": distribution([
                    row["r0_surface_error_m"] for row in direct
                    if row["image_border_clipped"]
                ]),
                "R6": distribution([
                    row["r6_center_error_m"] for row in direct
                    if row["image_border_clipped"]
                ]),
            },
            "moving_camera": {
                "R0": distribution([
                    row["r0_surface_error_m"] for row in direct
                    if row["camera_motion"] == "moving_camera"
                ]),
                "R6": distribution([
                    row["r6_center_error_m"] for row in direct
                    if row["camera_motion"] == "moving_camera"
                ]),
            },
        },
        "velocity": distribution([
            row["velocity_error_mps"] for row in velocity
            if row["source"] != "uninitialized_zero"
        ]),
        "support": {
            "sample_count": len(envelope),
            "actor_geometry_coverage": float(np.mean([
                row["covered"] for row in envelope
            ])) if envelope else None,
            "radial_size_m": distribution([
                row["radial_size_m"] for row in envelope
            ]),
            "volume_proxy_m3": distribution([
                row["volume_proxy_m3"] for row in envelope
            ]),
            "by_shape": {
                shape: {
                    "count": sum(row["shape"] == shape for row in envelope),
                    "coverage": float(np.mean([
                        row["covered"] for row in envelope
                        if row["shape"] == shape
                    ])),
                } for shape in sorted({
                    row["shape"] for row in envelope
                })
            },
        },
        "observability": dict(sum(
            (Counter(case["observability"]) for case in cases), Counter()
        )),
        "negative_false_reference_tracks": sum(
            case["negative_false_reference_tracks"] for case in cases
        ),
        "runtime_gt_used": False,
    }


def main():
    splits = grouped_sequences()
    if set(splits["calibration"]) & set(splits["development_validation"]):
        raise RuntimeError("calibration/development validation leakage")
    if (
        set(splits["fresh"])
        & (set(splits["calibration"]) | set(splits["development_validation"]))
    ):
        raise RuntimeError("fresh sequence leakage")
    config_document = yaml.safe_load(CONFIG.read_text())
    bridge_config = ReferenceBridgeConfigV1.from_mapping(
        config_document["bridge"]
    )
    manifest = {
        "status": "FROZEN_BEFORE_FRESH_GT_ACCESS",
        "grouping_unit": "complete_sequence",
        "same_actor_trajectory_cross_split": False,
        "splits": {key: list(value) for key, value in splits.items()},
        "config_hash": digest(CONFIG),
        "measurement_adapter_hash": digest(
            ROOT/"policy/dynamic/measurement_geometry_adapter_v1.py"
        ),
        "reference_bridge_hash": digest(
            ROOT/"policy/dynamic/tracking_collision_reference_bridge_v1.py"
        ),
        "center_tracker_hash": digest(
            ROOT/"policy/dynamic/reference_aligned_center_tracker_v1.py"
        ),
        "reference_evaluator_hash": digest(
            ROOT/"tools/run_phase8jqv2_4ptar1_reference_evaluation.py"
        ),
        "risk_evaluator_hash": digest(
            ROOT/"tools/run_phase8jqv2_4ptar1_risk_regression.py"
        ),
        "fresh_gt_accessed_at_freeze": False,
        "holdout_accessed": False,
        "test_accessed": False,
        "blind_accessed": False,
    }
    atomic_json(
        REPORTS/"phase8jqv2_4ptar1_fresh_validation_freeze.json",
        manifest,
    )
    all_results = {}
    for split in ("calibration", "development_validation", "fresh"):
        case_results = []
        for index, sequence_id in enumerate(splits[split], 1):
            result = process_case(load_case(sequence_id), bridge_config)
            case_results.append(result)
            print(json.dumps({
                "split": split, "progress": f"{index}/{len(splits[split])}",
                "case_id": sequence_id,
                "direct_records": len(result["direct_reference_errors"]),
            }))
        document = {
            "split": split,
            "cases": case_results,
            "summary": summarize(split, case_results),
            "runtime_gt_used": False,
            "offline_gt_used_for_evaluation_only": True,
        }
        atomic_json(DIAG/f"{split}_reference_evaluation.json", document)
        all_results[split] = document["summary"]
    atomic_json(DIAG/"reference_evaluation_summary.json", {
        "status": "PASS",
        "splits": all_results,
        "fresh_parameters_changed_after_freeze": False,
        "holdout_accessed": False,
        "test_accessed": False,
        "blind_accessed": False,
    })
    print(json.dumps(all_results["fresh"], indent=2))


if __name__ == "__main__":
    main()
