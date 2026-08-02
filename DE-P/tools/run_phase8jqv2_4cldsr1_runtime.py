#!/usr/bin/env python3
"""Host-CUDA runtime-only replay for CLDSR1.

Offline GT evaluation starts only after the synchronized runtime timer stops.
This reuses fixed historical episodes and is not a fresh validation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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
DIAG = ROOT/"diagnostics/phase8jqv2_4cldsr1"
sys.path.insert(0, str(ROOT))

from controller.bounded_reachability_planner_adapter_v1 import (
    BoundedReachabilityPlannerAdapterV1, PlannerSafetySnapshotV1,
    TrackSnapshotIdentityV1,
)
from controller.dynamic_safety_decision_router_v1 import FeatureMode
from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_network import DepNetwork
from policy.poly_solver import Poly5Solver
from config.config import cfg
from policy.dynamic.bounded_dynamic_reachability_v1 import (
    BoundedDynamicReachabilityBuilderV1, ReachabilityStatus,
)
from policy.dynamic.camera_model import make_depth_frame
from policy.dynamic.dynamic_object_geometry_model_v1 import (
    DynamicObjectGeometryModelV1, GeometryModelConfigV1,
)
from policy.dynamic.dynamic_safety_availability_v1 import (
    RuntimePlannerMeasurementV1,
)
from policy.dynamic.provisional_measurement_chain_v1 import (
    ProvisionalMeasurementChainManagerV1, ProvisionalObservationV1,
)
from policy.dynamic.provisional_safety_hypothesis_v1 import (
    ProvisionalSafetyHypothesisBuilderV1,
)
from policy.dynamic.shape_aware_dynamic_occupancy_v1 import (
    FastGeometryUpdateCacheV1,
)
from policy.dynamic.shape_motion_hypothesis_tracker_v1 import (
    ShapeMotionHypothesisTrackerV1,
)
from tools.run_phase8jqv2_4dogmr1_geometry_evaluation import load_case
from tools.run_phase8jqv2_4ocsr1_risk import (
    normalize_depth, selected_config, yopo_auxiliary,
)
from tools.run_phase8jqv2_4samsr1_evaluation import exact_gt_shape_risk
from tools.run_phase8jqv2_4tccr1_telemetry import make_perception


PREFIX = "phase8jqv2_4cldsr1_"
SEQUENCES = (
    "phase8c_train_0013", "phase8c_train_0015",
    "phase8c_train_0017", "phase8c_train_0019",
    "phase8c_train_0021", "phase8c_train_0023",
)


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True
    )+"\n")
    os.replace(temporary, path)


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value.rstrip()+"\n")
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
        "p99": float(np.percentile(values, 99)),
        "maximum": float(values.max()),
    }


def pose_matrix(pose):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = pose.rotation_world_from_camera
    matrix[:3, 3] = pose.position_world
    return matrix


def observation(row, timestamp):
    extent = float(max(np.asarray(row.extent).max(), .2))/2.
    u0, v0, u1, v1 = row.pixel_bbox
    boundary = float(
        u0 <= 1 or v0 <= 1 or u1 >= 158 or v1 >= 94
    )
    return ProvisionalObservationV1(
        observation_id=int(row.observation_id),
        timestamp=float(timestamp),
        position_world=row.centroid_world,
        covariance_world=row.position_covariance,
        support_radius_m=max(.2, extent),
        pixel_bbox=row.pixel_bbox, point_count=row.point_count,
        boundary_hazard_fraction=boundary,
    )


def candidates_profiled(
    model, case, frame, velocity, acceleration, goal, body_rotation,
):
    depth = normalize_depth(
        case["depths"][frame], case["model"].max_depth
    )
    rotation = body_rotation[frame]
    current_position = case["poses"][frame].position_world
    state_input = np.concatenate((
        rotation.T@velocity[frame],
        rotation.T@acceleration[frame],
        rotation.T@(goal[frame]-current_position),
    )).astype(np.float32)
    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        gpu_start.record()
        endstate, score = model.inference(
            torch.from_numpy(depth[None, None]).cuda(),
            torch.from_numpy(state_input[None]).cuda(),
        )
        gpu_end.record()
    torch.cuda.synchronize()
    gpu_ms = float(gpu_start.elapsed_time(gpu_end))
    states = (
        endstate[0].permute(1, 2, 0).reshape(15, 9)
        .detach().cpu().numpy()
    )
    score = score.reshape(-1).detach().cpu().numpy()
    duration = float(cfg["sgm_time"])
    times = np.linspace(duration/30., duration, 30)
    trajectories = np.empty((15, len(times), 3), dtype=np.float64)
    for candidate, state in enumerate(states):
        end_position = current_position+rotation@state[:3]
        end_velocity = rotation@state[3:6]
        end_acceleration = rotation@state[6:9]
        for axis in range(3):
            solver = Poly5Solver(
                current_position[axis], velocity[frame, axis],
                acceleration[frame, axis], end_position[axis],
                end_velocity[axis], end_acceleration[axis], duration,
            )
            trajectories[candidate, :, axis] = [
                solver.get_position(value) for value in times
            ]
    return trajectories, times, score, gpu_ms


def run_case(case, network, documents):
    integration, bounded, sam, dog, provisional = documents
    candidate = next(
        row for row in bounded["candidates"]
        if row["id"] == "B8_BOUNDED_DYNAMIC_REACHABILITY_V1"
    )
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
    adapter = BoundedReachabilityPlannerAdapterV1(
        config=integration, reachability_builder=builder,
        mode=FeatureMode.DEVELOPMENT_ACTIVE, development_launcher=True,
    )
    chain_manager = ProvisionalMeasurementChainManagerV1(
        association_distance_m=provisional["association"]["distance_m"],
        maximum_age_s=provisional["lifecycle"]["maximum_age_s"],
        maximum_missed_frames=
            provisional["lifecycle"]["maximum_missed_frames"],
        maximum_observations=
            provisional["association"]["maximum_observations"],
        maximum_boundary_hazard_fraction=provisional[
            "measurement_quality"
        ]["maximum_boundary_hazard_fraction"],
    )
    hypothesis_builder = ProvisionalSafetyHypothesisBuilderV1(
        maximum_speed_mps=provisional["motion"]["maximum_speed_mps"],
        maximum_acceleration_mps2=
            provisional["motion"]["maximum_acceleration_mps2"],
        maximum_age_s=provisional["lifecycle"]["maximum_age_s"],
        horizon_s=provisional["motion"]["horizon_s"],
        samples=provisional["motion"]["samples"],
        extent_prior_radius_m=provisional["geometry"][
            "frozen_extent_prior_radius_m"
        ],
    )
    profiler = RuntimePlannerMeasurementV1(torch.cuda.synchronize)
    velocity, acceleration, goal, rotation = yopo_auxiliary(case)
    safety = selected_config()
    offline_times, rows, gpu_times = [], [], []
    for frame_index, raw_timestamp in enumerate(case["timestamps"]):
        timestamp = float(raw_timestamp)
        metadata = {
            "case_id": case["case_id"], "scenario": case["scenario"],
            "frame": frame_index, "target_count_offline": None,
        }
        profiler.start(metadata)
        def preprocess():
            value = np.asarray(
                case["depths"][frame_index], np.float32
            )
            value_frame = make_depth_frame(
                value, case["model"], case["poses"][frame_index],
                timestamp, perception.config.depth_stride,
            )
            return value, value_frame
        depth, frame = profiler.measure("depth_preprocessing", preprocess)
        result = profiler.measure(
            "dynamic_perception",
            lambda: perception.update_depth(
                depth, case["poses"][frame_index],
                timestamp, case["model"],
            ),
        )
        observations = profiler.measure(
            "measurement", lambda: {
                row.observation_id: row for row in result.observations
            },
        )
        tracks = profiler.measure(
            "formal_tracker", lambda: tuple(result.all_tracks)
        )
        def provisional_step():
            chains = chain_manager.update(
                [observation(row, timestamp)
                 for row in result.observations],
                timestamp,
            )
            return tuple(
                hypothesis_builder.build(chain, timestamp)
                for chain in chains
            )
        provisional_states = profiler.measure(
            "provisional_availability", provisional_step
        )
        def geometry_step():
            live = [track.track_id for track in tracks]
            motion_tracker.delete_missing(live)
            builder.delete_missing(live)
            states, identities = [], []
            unresolved = False
            prediction_only = False
            for track in tracks:
                generation = (
                    f"{track.birth_frame}:{track.birth_observation_id}"
                )
                direct_observation = observations.get(
                    track.last_observation_id
                )
                direct = (
                    direct_observation is not None
                    and track.last_direct_observation_frame == frame_index
                )
                if direct:
                    _, evaluation = geometry_cache.update(
                        direct_observation, frame, generation
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
            return tuple(states), tuple(identities), unresolved, prediction_only
        states, identities, unresolved, prediction_only = profiler.measure(
            "geometry_reachability", geometry_step
        )
        def yopo_step():
            return candidates_profiled(
                network, case, frame_index,
                velocity, acceleration, goal, rotation,
            )
        candidates, times, scores, gpu_ms = profiler.measure(
            "yopo_inference", yopo_step,
        )
        gpu_times.append(gpu_ms)
        original = int(np.argmin(scores))
        snapshot = profiler.measure(
            "snapshot",
            lambda: PlannerSafetySnapshotV1.create(
                frame_index=frame_index, query_timestamp=timestamp,
                camera_pose_timestamp=timestamp,
                camera_pose_world_from_camera=pose_matrix(
                    case["poses"][frame_index]
                ),
                candidate_set_id=f"{case['case_id']}:{frame_index}",
                candidate_ids=np.arange(len(candidates)),
                candidate_positions=candidates, candidate_times=times,
                candidate_scores=scores, candidate_time_origin=timestamp,
                track_snapshot_id=
                    f"{case['case_id']}:tracks:{frame_index}",
                reachability_snapshot_id=
                    f"{case['case_id']}:reachability:{frame_index}",
                track_identities=identities,
                reachability_states=states,
                unresolved_dynamic_risk=unresolved,
            ),
        )
        output = profiler.measure(
            "risk", lambda: adapter.evaluate(snapshot, original)
        )
        profiler.measure("router", lambda: output["decision_status"])
        runtime = profiler.stop()

        offline_started = time.perf_counter()
        truth = exact_gt_shape_risk(
            case, frame_index, candidates, times,
            safety.uav_radius_m, safety.clearance_margin_m,
        )
        offline_times.append((time.perf_counter()-offline_started)*1000.)
        rows.append({
            "frame": frame_index,
            "runtime": runtime,
            "provisional_hypothesis_count": len(provisional_states),
            "formal_reachability_count": len(states),
            "prediction_only_active": prediction_only,
            "runtime_dynamic_state_count":
                len(states)+len(provisional_states),
            "gpu_yopo_critical_ms": gpu_times[-1],
            "offline_truth_row_count": len(truth),
            "runtime_gt_used": False,
        })
    return rows, offline_times, gpu_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is required; do not interpret sandbox CPU")
    if str(args.device) != "0":
        raise ValueError("CLDSR1 frozen host target is cuda:0")
    integration = yaml.safe_load((
        ROOT/"configs/bounded_reachability_integration_v1_candidate.yaml"
    ).read_text())
    bounded = yaml.safe_load((
        ROOT/"configs/bounded_dynamic_reachability_contract_v1_candidate.yaml"
    ).read_text())
    sam = yaml.safe_load((
        ROOT/"configs/shape_aware_motion_state_contract_v1_candidate.yaml"
    ).read_text())
    dog = yaml.safe_load((
        ROOT/"configs/dynamic_object_geometry_contract_v1_candidate.yaml"
    ).read_text())
    provisional = yaml.safe_load((
        ROOT/"configs/provisional_dynamic_safety_contract_v1_candidate.yaml"
    ).read_text())
    network = DepNetwork(backbone_variant="legacy").cuda().eval()
    load_dep_checkpoint(
        network, ROOT/"saved/DEP_0/epoch10.pth", "legacy"
    )
    all_rows, offline, gpu_critical = [], [], []
    for sequence in SEQUENCES:
        case = load_case(sequence)
        rows, times, gpu_times = run_case(
            case, network, (integration, bounded, sam, dog, provisional)
        )
        all_rows.extend(rows)
        offline.extend(times)
        gpu_critical.extend(gpu_times)
        print(json.dumps({
            "label": "development_root_cause_replay",
            "case": sequence, "frames": len(rows),
            "fresh_validation": False,
        }))
    runtime_values = [
        row["runtime"]["cpu_wall_ms"] for row in all_rows
    ]
    per_module = defaultdict(list)
    for row in all_rows:
        for key, value in row["runtime"]["per_module_ms"].items():
            per_module[key].append(value)
    depth_fps = float(yaml.safe_load((
        ROOT.parent/"Simulator/src/config/config.yaml"
    ).read_text())["depth_fps"])
    gate = 1000./depth_fps
    overall = distribution(runtime_values)
    status = "PASS" if overall["p95"] <= gate else "FAIL"
    detail = {
        "label": "development_root_cause_replay",
        "fresh_validation": False, "runtime_gt_used": False,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "rows": all_rows,
    }
    DIAG.mkdir(parents=True, exist_ok=True)
    atomic(DIAG/"runtime_only_host.json", detail)
    atomic(REPORTS/f"{PREFIX}runtime_only_planner_cycle.json", {
        "status": status, "measurement_valid": True,
        "depth_fps": depth_fps, "gate_ms": gate,
        "planner_cycle_runtime_only_ms": overall,
        "gpu_yopo_critical_ms": distribution(gpu_critical),
        "direct_state_cycles_ms": distribution([
            row["runtime"]["cpu_wall_ms"] for row in all_rows
            if row["runtime_dynamic_state_count"] > 0
            and not row["prediction_only_active"]
        ]),
        "prediction_only_cycles_ms": distribution([
            row["runtime"]["cpu_wall_ms"] for row in all_rows
            if row["prediction_only_active"]
        ]),
        "single_target_state_cycles_ms": distribution([
            row["runtime"]["cpu_wall_ms"] for row in all_rows
            if row["runtime_dynamic_state_count"] == 1
        ]),
        "multi_target_state_cycles_ms": distribution([
            row["runtime"]["cpu_wall_ms"] for row in all_rows
            if row["runtime_dynamic_state_count"] > 1
        ]),
        "historical_53_40_ms_valid": False,
        "cuda_synchronized": True, "runtime_gt_used": False,
        "offline_evaluation_included": False,
    })
    atomic(REPORTS/f"{PREFIX}offline_evaluator_runtime.json", {
        "status": "PASS_SEPARATED",
        "included_in_runtime_timer": False,
        "offline_exact_gt_shape_risk_ms": distribution(offline),
        "runtime_gt_used": False,
    })
    atomic(REPORTS/f"{PREFIX}runtime_breakdown.json", {
        "status": status,
        "planner_cycle_runtime_only_ms": overall,
        "per_module_ms": {
            key: distribution(values)
            for key, values in sorted(per_module.items())
        },
        "gpu_yopo_critical_ms": distribution(gpu_critical),
        "dynamic_perception_contains_frozen_foreground_and_tracker": True,
        "measurement_and_formal_tracker_rows_are_read_only_extraction_cost":
            True,
        "cuda_synchronized": True,
    })
    contract_path = REPORTS/f"{PREFIX}runtime_measurement_contract.json"
    contract = json.loads(contract_path.read_text())
    contract.update({
        "status": "PASS", "host_measurement_complete": True,
        "cuda_synchronized": True, "measurement_valid": True,
    })
    atomic(contract_path, contract)
    final_path = REPORTS/f"{PREFIX}final_result.json"
    final = json.loads(final_path.read_text())
    final["runtime_only_measurement"] = status
    final["runtime_only_p95_ms"] = overall["p95"]
    final["runtime_gate_ms"] = gate
    if status == "FAIL":
        final.update({
            "status": "FAIL", "route": "H",
            "secondary_root_causes": [
                final["primary_cause"],
            ],
            "primary_cause": "closed_loop_runtime_budget",
            "next_allowed_phase":
                "phase8jqv2_4_dynamic_integration_runtime_optimization",
        })
    atomic(final_path, final)
    selection_path = REPORTS/f"{PREFIX}candidate_selection.json"
    selection = json.loads(selection_path.read_text())
    selection["runtime_gate"] = status
    selection["runtime_p95_ms"] = overall["p95"]
    selection["runtime_gate_ms"] = gate
    if status == "FAIL":
        selection.update({
            "status": "FAIL", "route": "H",
            "selected_for_next_stage":
                "RUNTIME_CRITICAL_PATH_OPTIMIZATION",
        })
    atomic(selection_path, selection)
    atomic_text(
        REPORTS/f"{PREFIX}final_readiness.md",
        f"""# CLDSR1 readiness

Status: **{final['status']} / Route {final['route']}**.

The runtime-only host measurement is valid and CUDA-synchronized:
p95={overall['p95']:.3f} ms, gate={gate:.3f} ms. Offline GT evaluation was
excluded from the timer. Production activation, training, formal generation,
and sealed-data access remain prohibited.
""",
    )
    atomic_text(
        REPORTS/f"{PREFIX}final_recommendation.md",
        (
            "# CLDSR1 recommendation\n\n"
            + (
                "Optimize the runtime critical path before any integration "
                "activation. Preserve the separately established measurement "
                "availability and causal-observability findings.\n"
                if status == "FAIL" else
                "Proceed to the bounded dynamic-measurement availability "
                "repair; retain provisional safety as development shadow.\n"
            )
        ),
    )
    atomic_text(
        REPORTS/f"{PREFIX}migration_plan.md",
        (
            "# CLDSR1 migration plan\n\n"
            "1. "
            + (
                "Bring the valid runtime-only p95 below the depth period.\n"
                if status == "FAIL" else
                "Preserve the passing runtime-only measurement contract.\n"
            )
            + "2. Repair the 3 L2 and 11 L3 observable availability gaps.\n"
            "3. Re-evaluate C2/C5 only after measurement availability passes.\n"
            "4. Review outside-FOV risks under the separate ODD/fallback "
            "contract.\n"
            "5. Keep production activation and training disabled.\n"
        ),
    )
    print(json.dumps({
        "status": status, "p95_ms": overall["p95"],
        "gate_ms": gate, "frames": len(all_rows),
        "fresh_validation": False,
    }, indent=2))


if __name__ == "__main__":
    main()
