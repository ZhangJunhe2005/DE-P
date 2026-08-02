#!/usr/bin/env python3
"""Run frozen static-YOPO shadow risk evaluation for an OCSR1 split."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics/phase8jqv2_4ocsr1"
sys.path.insert(0, str(ROOT))

from config.config import cfg
from controller.dynamic_safety_shadow_adapter_v1 import (
    evaluate_shadow as evaluate_v1,
)
from controller.dynamic_safety_shadow_adapter_v2 import (
    BoundedCoastingSafetyAdapter, CoastingSafetyConfig,
)
from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_network import DepNetwork
from policy.poly_solver import Poly5Solver
from tools.evaluate_yopo_dynamic_candidate_risk_v1 import (
    compare_shadow_with_gt, evaluate_gt_candidate_risk,
)
from tools.run_phase8jqv2_4ocsr1_collect import (
    control_cases, ordinary_cases, split_for,
)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def selected_config():
    document = yaml.safe_load((
        ROOT / "configs/occlusion_coasting_safety_contract_v1_candidate.yaml"
    ).read_text())
    selected = document["selected_parameters"]
    if not selected["frozen"]:
        raise RuntimeError("OCSR1 parameters must be frozen")
    return CoastingSafetyConfig(
        maximum_coasting_misses=int(
            selected["maximum_coasting_misses"]
        ),
        maximum_coasting_time_s=float(
            selected["maximum_coasting_time_s"]
        ),
        maximum_position_std_m=float(
            selected["maximum_position_std_m"]
        ),
        maximum_velocity_std_mps=float(
            selected["maximum_velocity_std_mps"]
        ),
        covariance_sigma=float(selected["covariance_sigma"]),
        process_noise_acceleration=float(
            selected["process_noise_acceleration"]
        ),
        uav_radius_m=float(selected["uav_radius_m"]),
        default_actor_radius_m=float(
            selected["default_actor_radius_m"]
        ),
        clearance_margin_m=float(selected["clearance_margin_m"]),
        maximum_reacquired_frames=int(
            selected["maximum_reacquired_frames"]
        ),
    )


def yopo_auxiliary(case):
    if case["category"] == "ordinary_dynamic":
        rows = list(csv.DictReader(
            (Path(case["source_root"]) / "frames.csv").open()
        ))
        velocity = np.asarray([[
            float(row["velocity_x"]), float(row["velocity_y"]),
            float(row["velocity_z"]),
        ] for row in rows])
        acceleration = np.asarray([[
            float(row["acceleration_x"]), float(row["acceleration_y"]),
            float(row["acceleration_z"]),
        ] for row in rows])
        goal = np.asarray([[
            float(row["goal_x"]), float(row["goal_y"]),
            float(row["goal_z"]),
        ] for row in rows])
        body_rotation = [
            Rotation.from_quat([
                float(row["camera_qx"]), float(row["camera_qy"]),
                float(row["camera_qz"]), float(row["camera_qw"]),
            ]).as_matrix()
            for row in rows
        ]
    else:
        count = len(case["timestamps"])
        velocity = np.zeros((count, 3))
        acceleration = np.zeros((count, 3))
        goal = np.asarray(
            [case["poses"][index].position_world + [10., 0., 0.]
             for index in range(count)]
        )
        body_from_camera = np.asarray(
            cfg["dynamic_perception"]["camera_rotation_body_from_camera"],
            dtype=np.float64,
        )
        body_rotation = [
            pose.rotation_world_from_camera @ body_from_camera.T
            for pose in case["poses"]
        ]
    return velocity, acceleration, goal, body_rotation


def normalize_depth(depth, maximum):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != (96, 160):
        depth = cv2.resize(
            depth, (160, 96), interpolation=cv2.INTER_NEAREST
        )
    depth = np.minimum(depth, maximum) / maximum
    invalid = np.isnan(depth) | (depth < .1 / maximum)
    repaired = cv2.inpaint(
        np.uint8(np.nan_to_num(depth) * 255),
        np.uint8(invalid), 1, cv2.INPAINT_NS,
    )
    return repaired.astype(np.float32) / 255.


def candidates_for_frame(
    model, case, frame, velocity, acceleration, goal, body_rotation
):
    depth = normalize_depth(
        case["depths"][frame], case["model"].max_depth
    )
    rotation = body_rotation[frame]
    current_position = case["poses"][frame].position_world
    observation = np.concatenate((
        rotation.T @ velocity[frame],
        rotation.T @ acceleration[frame],
        rotation.T @ (goal[frame] - current_position),
    )).astype(np.float32)
    with torch.inference_mode():
        endstate, score = model.inference(
            torch.from_numpy(depth[None, None]).cuda(),
            torch.from_numpy(observation[None]).cuda(),
        )
    states = (
        endstate[0].permute(1, 2, 0).reshape(15, 9)
        .detach().cpu().numpy()
    )
    score = score.reshape(-1).detach().cpu().numpy()
    duration = float(cfg["sgm_time"])
    times = np.linspace(duration / 30., duration, 30)
    trajectories = np.empty((15, len(times), 3), dtype=np.float64)
    for candidate, state in enumerate(states):
        end_position = current_position + rotation @ state[:3]
        end_velocity = rotation @ state[3:6]
        end_acceleration = rotation @ state[6:9]
        for axis in range(3):
            solver = Poly5Solver(
                current_position[axis], velocity[frame, axis],
                acceleration[frame, axis], end_position[axis],
                end_velocity[axis], end_acceleration[axis], duration,
            )
            trajectories[candidate, :, axis] = [
                solver.get_position(value) for value in times
            ]
    return trajectories, times, score


def gt_at_times(case, frame, relative_times):
    start = float(case["timestamps"][frame])
    targets = start + np.asarray(relative_times)
    actor_ids = sorted({
        actor_id for gt in case["gt"] for actor_id in gt
    })
    actors, radii, valid = [], [], []
    for actor_id in actor_ids:
        source_times, source_positions, source_radii = [], [], []
        for timestamp, gt in zip(case["timestamps"], case["gt"]):
            if actor_id in gt:
                source_times.append(float(timestamp))
                source_positions.append(gt[actor_id]["position_world"])
                source_radii.append(gt[actor_id]["radius_m"])
        source_times = np.asarray(source_times)
        source_positions = np.asarray(source_positions)
        mask = (
            (targets >= source_times.min() - 1e-9)
            & (targets <= source_times.max() + 1e-9)
        )
        positions = np.full((len(targets), 3), np.nan)
        for axis in range(3):
            positions[mask, axis] = np.interp(
                targets[mask], source_times, source_positions[:, axis]
            )
        actors.append(positions)
        radii.append(float(np.median(source_radii)))
        valid.append(mask)
    if not actors:
        return np.empty((0, len(targets), 3)), np.empty(0), np.empty(
            (0, len(targets)), dtype=bool
        )
    return np.asarray(actors), np.asarray(radii), np.asarray(valid)


def v1_track(track, previously_dynamic):
    return {
        "track_id": track["track_id"],
        "position_world": track["position_world"],
        "velocity_world": track["velocity_world"],
        "state_covariance": track["state_covariance"],
        "is_confirmed": track["is_confirmed"],
        "is_dynamic": track["is_dynamic"],
        "attention_authorized": track["attention_authorized"],
        "confidence": track["confidence"],
        "missed_count": track["missed_count"],
        "previously_dynamic": previously_dynamic,
    }


def should_evaluate(frame, tracks, negative, last_frame):
    if negative:
        return frame == max(last_frame + 17, 0)
    if frame > last_frame:
        return False
    return any(
        (
            track["measurement_present"] and track["is_confirmed"]
            and track["is_dynamic"]
        ) or track["missed_count"] in (1, 2, 3)
        for track in tracks
    )


def process_case(case, runtime_case, model, config):
    velocity, acceleration, goal, rotation = yopo_auxiliary(case)
    adapter = BoundedCoastingSafetyAdapter(config)
    ever_dynamic = set()
    evaluations = []
    gap_counts = defaultdict(int)
    last_authoritative = len(case["timestamps"]) - 18
    for runtime_frame in runtime_case["frames"]:
        frame = runtime_frame["frame"]
        tracks = runtime_frame["tracks"]
        for track in tracks:
            if (
                track["measurement_present"] and track["is_dynamic"]
                and track["is_confirmed"]
            ):
                ever_dynamic.add(track["track_id"])
            if (
                track["track_id"] in ever_dynamic
                and track["missed_count"] in (1, 2, 3)
            ):
                gap_counts[str(track["missed_count"])] += 1
        evaluate = should_evaluate(
            frame, tracks, case["negative"], last_authoritative
        )
        if not evaluate:
            adapter.update_safety_states(
                tracks, runtime_frame["timestamp"], frame
            )
            continue
        candidates, times, scores = candidates_for_frame(
            model, case, frame, velocity, acceleration, goal, rotation
        )
        original = int(np.argmin(scores))
        started = time.perf_counter()
        bounded = adapter.evaluate(
            candidates, times, tracks,
            timestamp=runtime_frame["timestamp"], frame_index=frame,
            original_candidate_id=original, candidate_scores=scores,
        )
        adapter_elapsed = (time.perf_counter() - started) * 1000.
        historical = [
            v1_track(
                track, track["track_id"] in ever_dynamic
            ) for track in tracks
        ]
        comparisons = {}
        for contract in (
            "C0_current", "C1_track_survival",
            "C2_recent_dynamic_coasting",
        ):
            comparisons[contract] = evaluate_v1(
                candidates, times, historical, contract=contract,
                original_candidate_id=original,
            )
        actor_positions, actor_radii, actor_valid = gt_at_times(
            case, frame, times
        )
        gt_started = time.perf_counter()
        ground_truth = evaluate_gt_candidate_risk(
            candidates, times, actor_positions, actor_radii,
            uav_radius_m=config.uav_radius_m,
            required_margin_m=config.clearance_margin_m,
            actor_valid=actor_valid,
        )
        gt_elapsed = (time.perf_counter() - gt_started) * 1000.
        known = not ground_truth["unknown_candidate_ids"]
        metrics = (
            compare_shadow_with_gt(bounded, ground_truth)
            if known else None
        )
        comparison_metrics = {}
        if known:
            for contract, result in comparisons.items():
                safe = [
                    row["candidate_trajectory_id"]
                    for row in result["candidate_rows"]
                    if not row["would_veto"]
                ]
                normalized = {
                    **result,
                    "recommended_candidate_id": (
                        original if original in safe else
                        safe[0] if safe else None
                    ),
                    "decision_status": (
                        "NO_ACTIVE_DYNAMIC_RISK"
                        if result["active_track_count"] == 0 else
                        "KEEP_ORIGINAL" if original in safe else
                        "SWITCH_TO_SAFE_CANDIDATE" if safe else
                        "NO_SAFE_CANDIDATE"
                    ),
                }
                comparison_metrics[contract] = compare_shadow_with_gt(
                    normalized, ground_truth
                )
            comparison_metrics["C2_bounded_v2"] = metrics
        evaluations.append({
            "case_id": case["case_id"],
            "category": case["category"],
            "scenario": case["scenario"],
            "negative": case["negative"],
            "frame": frame,
            "timestamp": runtime_frame["timestamp"],
            "track_count": len(tracks),
            "ground_truth_actor_count": len(actor_radii),
            "ground_truth": ground_truth,
            "bounded_v2": bounded,
            "comparison_contracts": comparisons,
            "comparison_metrics": comparison_metrics,
            "metrics": metrics,
            "adapter_runtime_ms": adapter_elapsed,
            "gt_evaluator_runtime_ms": gt_elapsed,
            "runtime_gt_used": False,
        })
    return {
        "case_id": case["case_id"],
        "negative": case["negative"],
        "scenario": case["scenario"],
        "gap_counts": dict(gap_counts),
        "evaluations": evaluations,
        "final_memory": adapter.memory,
    }


def aggregate(results):
    evaluations = [
        row for result in results for row in result["evaluations"]
    ]
    known = [row for row in evaluations if row["metrics"] is not None]
    candidate = [row["metrics"] for row in known]
    runtime = [row["adapter_runtime_ms"] for row in evaluations]
    negative = [
        row for row in evaluations if row["negative"]
    ]
    return {
        "evaluation_count": len(evaluations),
        "known_gt_evaluation_count": len(known),
        "unknown_gt_evaluation_count": len(evaluations) - len(known),
        "true_unsafe_candidates": sum(
            row["true_unsafe_candidates"] for row in candidate
        ),
        "correctly_vetoed_unsafe": sum(
            row["correctly_vetoed_unsafe"] for row in candidate
        ),
        "missed_unsafe": sum(row["missed_unsafe"] for row in candidate),
        "false_vetoed_safe": sum(
            row["false_vetoed_safe"] for row in candidate
        ),
        "retained_safe": sum(row["retained_safe"] for row in candidate),
        "unsafe_recommendations": sum(
            row["unsafe_recommendation"] for row in candidate
        ),
        "false_emergencies": sum(
            row["false_emergency"] for row in candidate
        ),
        "no_safe_candidate_truth_count": sum(
            row["no_safe_candidate_truth"] for row in candidate
        ),
        "no_safe_candidate_correct_count": sum(
            row["no_safe_candidate_correct"] for row in candidate
        ),
        "all_veto_count": sum(
            row["all_candidates_vetoed"] for row in candidate
        ),
        "negative_false_active_safety_tracks": sum(
            row["bounded_v2"]["active_safety_track_count"] > 0
            for row in negative
        ),
        "negative_false_veto": sum(
            candidate["would_veto"]
            for row in negative
            for candidate in row["bounded_v2"]["candidate_rows"]
        ),
        "negative_false_no_safe_candidate": sum(
            row["bounded_v2"]["decision_status"] == "NO_SAFE_CANDIDATE"
            for row in negative
        ),
        "runtime_average_ms": float(np.mean(runtime)) if runtime else 0.,
        "runtime_p95_ms":
            float(np.percentile(runtime, 95)) if runtime else 0.,
        "runtime_max_ms": max(runtime, default=0.),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=("calibration", "validation"), required=True
    )
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("OCSR1 YOPO candidate evaluation requires host CUDA")
    freeze = json.loads((
        ROOT / "reports/phase8jqv2_4ocsr1_validation_freeze.json"
    ).read_text())
    config_path = (
        ROOT / "configs/occlusion_coasting_safety_contract_v1_candidate.yaml"
    )
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != freeze["config_hash"]:
        raise RuntimeError("frozen OCSR1 config hash changed")
    runtime = json.loads((
        OUT / f"{arguments.split}_runtime_and_gt.json"
    ).read_text())
    runtime_by_id = {
        case["case_id"]: case for case in runtime["cases"]
    }
    cases = {
        case["case_id"]: case
        for case in control_cases() + ordinary_cases()
        if split_for(case["case_id"]) == arguments.split
    }
    model = DepNetwork(backbone_variant="legacy").cuda().eval()
    load_dep_checkpoint(
        model, ROOT / "saved/DEP_0/epoch10.pth", "legacy"
    )
    config = selected_config()
    results = []
    for index, case_id in enumerate(sorted(cases), 1):
        result = process_case(
            cases[case_id], runtime_by_id[case_id], model, config
        )
        results.append(result)
        print(json.dumps({
            "split": arguments.split, "case": case_id,
            "progress": f"{index}/{len(cases)}",
            "evaluations": len(result["evaluations"]),
        }))
    summary = aggregate(results)
    destination = OUT / f"{arguments.split}_candidate_risk.json"
    atomic_json(destination, {
        "split": arguments.split,
        "config": config.__dict__,
        "cases": results,
        "summary": summary,
        "runtime_gt_used": False,
        "formal_control_modified": False,
    })
    atomic_json(
        OUT / f"{arguments.split}_candidate_risk_summary.json",
        {
            "status": "PASS",
            "split": arguments.split,
            **summary,
            "source": str(destination),
            "source_sha256":
                hashlib.sha256(destination.read_bytes()).hexdigest(),
            "runtime_gt_used": False,
            "formal_control_modified": False,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
