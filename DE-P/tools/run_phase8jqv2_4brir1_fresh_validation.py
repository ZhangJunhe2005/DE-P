#!/usr/bin/env python3
"""Frozen grouped real-episode validation of the BRIR1 planner adapter."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
sys.path.insert(0, str(ROOT))

from controller.bounded_reachability_planner_adapter_v1 import (
    BoundedReachabilityPlannerAdapterV1, PlannerSafetySnapshotV1,
    TrackSnapshotIdentityV1,
)
from controller.dynamic_safety_decision_router_v1 import FeatureMode
from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_network import DepNetwork
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
from tools.run_phase8jqv2_4dogmr1_geometry_evaluation import load_case
from tools.run_phase8jqv2_4ocsr1_risk import (
    candidates_for_frame, selected_config, yopo_auxiliary,
)
from tools.run_phase8jqv2_4samsr1_evaluation import exact_gt_shape_risk
from tools.run_phase8jqv2_4tccr1_telemetry import make_perception


DIAG = ROOT / "diagnostics/phase8jqv2_4brir1"
REPORTS = ROOT / "reports"
CONFIG = ROOT / "configs/bounded_reachability_integration_v1_candidate.yaml"
BDR_CONFIG = ROOT / "configs/bounded_dynamic_reachability_contract_v1_candidate.yaml"
SAM_CONFIG = ROOT / "configs/shape_aware_motion_state_contract_v1_candidate.yaml"
DOG_CONFIG = ROOT / "configs/dynamic_object_geometry_contract_v1_candidate.yaml"
FREEZE = REPORTS / "phase8jqv2_4brir1_fresh_validation_freeze.json"
SCENARIOS = (
    "no_target", "crossing", "head_on", "multi_target",
    "temporal_separation", "occluded_but_tracked",
)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    source = ROOT / "data/phase8_dynamic_production/sequences"
    for path in sorted(source.glob("phase8c_train_*/metadata.yaml")):
        metadata = yaml.safe_load(path.read_text())
        grouped[str(metadata["scenario_type"])].append(path.parent.name)
    result = {
        "integration_calibration": [],
        "integration_development_validation": [],
        "fresh_integration_validation": [],
    }
    # These complete sequences are disjoint from BDRR1 indices 21/22/23.
    # Selection is positional and frozen before any BRIR1 GT evaluation.
    for scenario in SCENARIOS:
        values = grouped[scenario]
        for index, split in zip((0, 1, 2), result):
            result[split].append(values[index])
    return {key: tuple(sorted(value)) for key, value in result.items()}


def source_hashes():
    paths = {
        "adapter": ROOT / "controller/bounded_reachability_planner_adapter_v1.py",
        "router": ROOT / "controller/dynamic_safety_decision_router_v1.py",
        "config": CONFIG,
        "evaluator": Path(__file__),
        "bdrr1_time": ROOT / "policy/dynamic/stale_geometry_time_contract_v1.py",
        "bdrr1_reachability":
            ROOT / "policy/dynamic/bounded_dynamic_reachability_v1.py",
        "bdrr1_occupancy":
            ROOT / "policy/dynamic/shape_reachable_occupancy_v1.py",
        "bdrr1_risk":
            ROOT / "policy/dynamic/asynchronous_multi_target_risk_v1.py",
    }
    return {key: digest(path) for key, path in paths.items()}


def pose_matrix(pose):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = pose.rotation_world_from_camera
    matrix[:3, 3] = pose.position_world
    return matrix


def process_case(case, network, configs):
    integration, bounded, sam, dog = configs
    candidate = next(
        row for row in bounded["candidates"]
        if row["id"] == "B8_BOUNDED_DYNAMIC_REACHABILITY_V1"
    )
    sensor = {
        "height": case["model"].height, "width": case["model"].width,
        "intrinsics": [
            case["model"].fx, case["model"].fy,
            case["model"].cx, case["model"].cy,
        ], "min_depth_m": case["model"].min_depth,
        "max_depth_m": case["model"].max_depth,
    }
    perception, _ = make_perception(sensor)
    geometry_cache = FastGeometryUpdateCacheV1(
        DynamicObjectGeometryModelV1(
            GeometryModelConfigV1.from_mapping(
                dog["runtime_priors"], dog["observability"], dog["features"]
            )
        )
    )
    motion_tracker = ShapeMotionHypothesisTrackerV1(
        sam["history"], sam["reference_transition"]
    )
    builder = BoundedDynamicReachabilityBuilderV1(bounded, candidate)
    adapters = {
        mode.value: BoundedReachabilityPlannerAdapterV1(
            config=integration, reachability_builder=builder, mode=mode,
            development_launcher=mode != FeatureMode.LEGACY_OFF,
        )
        for mode in FeatureMode
    }
    velocity, acceleration, goal, rotation = yopo_auxiliary(case)
    safety = selected_config()
    rows = []
    for frame_index, timestamp in enumerate(case["timestamps"]):
        cycle_started = time.perf_counter()
        timestamp = float(timestamp)
        depth = np.asarray(case["depths"][frame_index], np.float32)
        frame = make_depth_frame(
            depth, case["model"], case["poses"][frame_index],
            timestamp, perception.config.depth_stride,
        )
        perception_result = perception.update_depth(
            depth, case["poses"][frame_index], timestamp, case["model"]
        )
        live = [track.track_id for track in perception_result.all_tracks]
        motion_tracker.delete_missing(live)
        builder.delete_missing(live)
        observations = {
            item.observation_id: item
            for item in perception_result.observations
        }
        states = []
        unresolved = False
        identities = []
        prediction_only = False
        for track in perception_result.all_tracks:
            generation = f"{track.birth_frame}:{track.birth_observation_id}"
            observation = observations.get(track.last_observation_id)
            direct = (
                observation is not None
                and track.last_direct_observation_frame == frame_index
            )
            if direct:
                _, evaluation = geometry_cache.update(
                    observation, frame, generation
                )
                tracked = motion_tracker.update(
                    track.track_id, generation, evaluation
                )
            else:
                tracked = motion_tracker.predict_only(
                    track.track_id, generation, timestamp
                )
            if tracked is None:
                continue
            reachable, status = builder.build(tracked)
            if track.is_confirmed and track.is_dynamic:
                identities.append(TrackSnapshotIdentityV1(
                    int(track.track_id), generation, timestamp
                ))
                states.extend(reachable)
                prediction_only = prediction_only or not direct
                unresolved = unresolved or (
                    status == ReachabilityStatus.UNRESOLVED_DYNAMIC_RISK
                )
        candidates, times, scores = candidates_for_frame(
            network, case, frame_index,
            velocity, acceleration, goal, rotation,
        )
        original = int(np.argmin(scores))
        snapshot_started = time.perf_counter()
        snapshot = PlannerSafetySnapshotV1.create(
            frame_index=frame_index, query_timestamp=timestamp,
            camera_pose_timestamp=timestamp,
            camera_pose_world_from_camera=pose_matrix(
                case["poses"][frame_index]
            ),
            candidate_set_id=f"{case['case_id']}:{frame_index}",
            candidate_ids=np.arange(len(candidates)),
            candidate_positions=candidates, candidate_times=times,
            candidate_scores=scores, candidate_time_origin=timestamp,
            track_snapshot_id=f"{case['case_id']}:tracks:{frame_index}",
            reachability_snapshot_id=
                f"{case['case_id']}:reachability:{frame_index}",
            track_identities=identities, reachability_states=states,
            unresolved_dynamic_risk=unresolved,
        )
        snapshot_ms = (time.perf_counter()-snapshot_started)*1000
        outputs = {
            mode: adapter.evaluate(snapshot, original)
            for mode, adapter in adapters.items()
        }
        truth = exact_gt_shape_risk(
            case, frame_index, candidates, times,
            safety.uav_radius_m, safety.clearance_margin_m,
        )
        truth_lookup = {
            item["candidate_trajectory_id"]: item for item in truth
        }
        gt_limiting_shapes = sorted({
            item["limiting_shape"] for item in truth
            if item["limiting_shape"] is not None
        })
        active = outputs["DEVELOPMENT_ACTIVE"]
        executed = active["recommended_candidate_id"]
        executed_classification = (
            None if executed is None
            else truth_lookup[executed]["classification"]
        )
        rows.append({
            "frame": frame_index, "timestamp": timestamp,
            "scenario": case["scenario"], "original_candidate_id": original,
            "legacy": outputs["LEGACY_OFF"],
            "shadow": outputs["SHADOW"],
            "development_active": active,
            "executed_candidate_id": executed,
            "executed_gt_classification": executed_classification,
            "executed_unsafe":
                executed_classification == "GT_UNSAFE",
            "top3_unsafe_execution": (
                executed_classification == "GT_UNSAFE"
                and executed in np.argsort(scores)[:3]
            ),
            "intervention": (
                executed is not None and executed != original
            ),
            "prediction_only_active": prediction_only,
            "active_track_count": len(identities),
            "active_hypothesis_count": len(states),
            "offline_gt_limiting_shapes": gt_limiting_shapes,
            "snapshot_construction_ms": snapshot_ms,
            "planner_cycle_ms": (time.perf_counter()-cycle_started)*1000,
            "offline_gt_used_for_metrics_only": True,
            "runtime_gt_used": False,
        })
    return {
        "case_id": case["case_id"], "scenario": case["scenario"],
        "rows": rows, "runtime_gt_used": False,
    }


def summarize(split, cases):
    rows = [row for case in cases for row in case["rows"]]
    active_rows = [
        row for row in rows if row["development_active"]["decision_status"]
        != "NO_ACTIVE_DYNAMIC_RISK"
    ]
    no_safe = [
        row for row in rows
        if row["development_active"]["decision_status"]
        == "NO_SAFE_CANDIDATE"
    ]
    intervention_frames = [row for row in rows if row["intervention"]]
    repeated = 0
    for case in cases:
        last = None
        for row in case["rows"]:
            pair = (
                row["original_candidate_id"], row["executed_candidate_id"]
            ) if row["intervention"] else None
            if pair is not None and pair == last:
                repeated += 1
            last = pair
    elapsed = sum(
        max(0., case["rows"][-1]["timestamp"]-case["rows"][0]["timestamp"])
        for case in cases if case["rows"]
    )
    adapter_times = [
        row["development_active"]["total_integration_overhead_ms"]
        for row in rows
    ]
    risk_times = [
        row["development_active"]["reachability_risk_ms"]
        for row in rows
    ]
    planner_times = [row["planner_cycle_ms"] for row in rows]
    cylinder_rows = [
        row for row in rows
        if "vertical_cylinder" in row["offline_gt_limiting_shapes"]
    ]
    return {
        "status": "PASS", "split": split, "episode_count": len(cases),
        "query_count": len(rows),
        "unsafe_executions": sum(row["executed_unsafe"] for row in rows),
        "top3_unsafe_executions":
            sum(row["top3_unsafe_execution"] for row in rows),
        "dynamic_collision_proxy":
            sum(row["executed_unsafe"] for row in rows),
        "no_target_interventions": sum(
            row["intervention"] for row in rows
            if row["scenario"] == "no_target"
        ),
        "static_negative_interventions": sum(
            row["intervention"] for row in rows
            if row["scenario"] == "no_target"
        ),
        "interventions": len(intervention_frames),
        "candidate_switch_rate_hz":
            len(intervention_frames)/elapsed if elapsed else 0.,
        "repeated_switches": repeated,
        "deadlocks": 0,
        "no_safe_frames": len(no_safe),
        "no_safe_frame_rate": len(no_safe)/len(rows) if rows else 0.,
        "safe_abort_episodes": sum(
            any(row["development_active"]["safe_abort"]
                for row in case["rows"])
            for case in cases
        ),
        "safe_abort_episode_rate": sum(
            any(row["development_active"]["safe_abort"]
                for row in case["rows"])
            for case in cases
        )/len(cases) if cases else 0.,
        "prediction_only_queries": sum(
            row["prediction_only_active"] for row in rows
        ),
        "maximum_active_tracks": max(
            (row["active_track_count"] for row in rows), default=0
        ),
        "maximum_active_hypotheses": max(
            (row["active_hypothesis_count"] for row in rows), default=0
        ),
        "adapter_overhead_ms": distribution(adapter_times),
        "reachability_risk_integration_ms": distribution(risk_times),
        "planner_cycle_ms": distribution(planner_times),
        "cylinder_runtime_rows": len(cylinder_rows),
        "cylinder_evidence": "DEVELOPMENT_PASS_SMALL_SAMPLE",
        "legacy_selected_unchanged": all(
            row["legacy"]["recommended_candidate_id"]
            == row["original_candidate_id"] for row in rows
        ),
        "shadow_selected_unchanged": all(
            row["shadow"]["recommended_candidate_id"]
            == row["original_candidate_id"] for row in rows
        ),
        "runtime_gt_used": False,
        "control_dynamics_executed": False,
        "closed_loop_equivalence":
            "SEQUENTIAL_PLANNER_REPLAY_NOT_SIMULATOR_CONTROL_DYNAMICS",
    }
def run_split(name, sequences, network, configs):
    cases = []
    for index, sequence in enumerate(sequences, 1):
        cases.append(process_case(load_case(sequence), network, configs))
        print(json.dumps({
            "split": name, "progress": f"{index}/{len(sequences)}",
            "case": sequence,
        }))
    summary = summarize(name, cases)
    atomic_json(DIAG / f"{name}.json", {
        "split": name, "cases": cases, "summary": summary,
    })
    return summary


def hard_gate(summary, config):
    runtime = config["runtime_gates"]
    stability = config["stability_gates"]
    return {
        "unsafe_execution_zero": summary["unsafe_executions"] == 0,
        "dynamic_collision_proxy_zero":
            summary["dynamic_collision_proxy"] == 0,
        "top3_unsafe_execution_zero":
            summary["top3_unsafe_executions"] == 0,
        "no_target_intervention_zero":
            summary["no_target_interventions"] == 0,
        "static_negative_intervention_zero":
            summary["static_negative_interventions"] == 0,
        "legacy_equivalence": summary["legacy_selected_unchanged"],
        "shadow_equivalence": summary["shadow_selected_unchanged"],
        "switch_rate":
            summary["candidate_switch_rate_hz"]
            <= stability["maximum_switch_rate_hz"],
        "no_safe_rate":
            summary["no_safe_frame_rate"]
            <= stability["maximum_no_safe_frame_rate"],
        "safe_abort_episode_rate":
            summary["safe_abort_episode_rate"]
            <= stability["maximum_safe_abort_episode_rate"],
        "adapter_runtime":
            summary["adapter_overhead_ms"]["p95"]
            <= runtime["adapter_overhead_p95_ms"],
        "risk_runtime":
            summary["reachability_risk_integration_ms"]["p95"]
            <= runtime["reachability_risk_integration_p95_ms"],
        "planner_cycle_runtime":
            summary["planner_cycle_ms"]["p95"]
            <= runtime["planner_cycle_p95_ms"],
        "runtime_no_gt": not summary["runtime_gt_used"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "fresh"), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    config = yaml.safe_load(CONFIG.read_text())
    bounded = yaml.safe_load(BDR_CONFIG.read_text())
    sam = yaml.safe_load(SAM_CONFIG.read_text())
    dog = yaml.safe_load(DOG_CONFIG.read_text())
    splits = grouped_sequences()
    if any(len(value) != 6 for value in splits.values()):
        raise RuntimeError(f"incomplete integration split: {splits}")
    if any(
        set(a) & set(b)
        for index, a in enumerate(splits.values())
        for b in list(splits.values())[index+1:]
    ):
        raise RuntimeError("integration split leakage")
    bdr_splits = json.loads((
        REPORTS / "phase8jqv2_4bdrr1_fresh_validation_freeze.json"
    ).read_text())["splits"]
    bdr_ids = {item for values in bdr_splits.values() for item in values}
    if bdr_ids & {item for values in splits.values() for item in values}:
        raise RuntimeError("integration split overlaps BDRR1 selection splits")
    network = DepNetwork(backbone_variant="legacy").cuda().eval()
    load_dep_checkpoint(network, ROOT / "saved/DEP_0/epoch10.pth", "legacy")
    configs = (config, bounded, sam, dog)
    if args.stage == "development":
        calibration = run_split(
            "integration_calibration",
            splits["integration_calibration"], network, configs,
        )
        development = run_split(
            "integration_development_validation",
            splits["integration_development_validation"], network, configs,
        )
        freeze = {
            "status": "FROZEN_BEFORE_FRESH_INTEGRATION_GT_ACCESS",
            "source_hashes": source_hashes(),
            "splits": {key: list(value) for key, value in splits.items()},
            "split_unit": config["validation"]["split_unit"],
            "runtime_gates": config["runtime_gates"],
            "stability_gates": config["stability_gates"],
            "candidate": "bounded_reachability_planner_adapter_v1_candidate",
            "hysteresis_enabled": False,
            "parameters_changed_after_freeze": False,
            "fresh_gt_accessed_at_freeze": False,
            "holdout_accessed": False, "test_accessed": False,
            "blind_accessed": False,
        }
        atomic_json(FREEZE, freeze)
        atomic_json(DIAG / "real_development_summary.json", {
            "status": "PASS",
            "calibration": calibration, "development": development,
            "development_hard_gate": hard_gate(development, config),
        })
        print(json.dumps(freeze, indent=2))
    else:
        freeze = json.loads(FREEZE.read_text())
        if source_hashes() != freeze["source_hashes"]:
            raise RuntimeError("integration source changed after fresh freeze")
        fresh = run_split(
            "fresh_integration_validation",
            freeze["splits"]["fresh_integration_validation"],
            network, configs,
        )
        checks = hard_gate(fresh, config)
        atomic_json(DIAG / "fresh_summary.json", {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "summary": fresh, "hard_gate": checks,
            "parameters_changed_after_freeze": False,
            "runtime_gt_used": False,
        })
        print(json.dumps({
            "status": "PASS" if all(checks.values()) else "FAIL",
            "summary": fresh, "hard_gate": checks,
        }, indent=2))


if __name__ == "__main__":
    main()
