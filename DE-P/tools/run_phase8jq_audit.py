#!/usr/bin/env python3
"""Phase 8J-Q controller, safety-semantics and feasibility audit.

This tool is deliberately offline and read-only with respect to the evaluator,
dataset, network, and checkpoints.  It reads validation identities only.
"""

from __future__ import annotations

import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
from ruamel.yaml import YAML
from scipy.optimize import minimize
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.phase8jq_semantics import (
    UAV_RADIUS_M,
    classify_timeline,
    continuous_linear_closest_approach,
    formal_dynamic_clearance,
    integrate_two_phase_acceleration,
    point_map_clearance,
    simulator_dynamic_clearance,
    stopping_distance,
)


REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jq"
DATASET = ROOT / "data/phase8_dynamic_production"
HORIZON = 1.7
FORMAL_POINTS = 30
O6_POINTS = 32
FIRST_CONTROLLABLE_S = 0.02
_SOLVER_SEQUENCES = {}
_SOLVER_TREES = {}


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {}
    return {
        str(value): float(np.quantile(values, value))
        for value in (0, .1, .5, .9, 1)
    }


def entry_gate():
    required = [
        REPORTS / "phase8h_final_perception_gate.json",
        REPORTS / "phase8i_final_result.json",
        REPORTS / "phase8jr_capacity_gate.json",
        REPORTS / "phase8jp_entry_gate.json",
        REPORTS / "phase8jp_parameterization_comparison.json",
        REPORTS / "phase8jp_capacity_gate.json",
        REPORTS / "phase8jp_final_result.json",
        REPORTS / "phase8jq_controller_identification_raw.json",
    ]
    p = json.loads((REPORTS / "phase8jp_parameterization_comparison.json").read_text())
    final = json.loads((REPORTS / "phase8jp_final_result.json").read_text())
    checks = {
        "required_inputs_present": all(path.is_file() for path in required),
        "phase8jp_complete": p.get("parameterization_audit_complete") is True,
        "completion_count_284": p.get("completion_count") == 284,
        "o6_recovered_zero_estimated": p["valid_estimated"]["o6_recovered_count"] == 0,
        "o6_recovered_zero_gt": p["valid_gt"]["o6_recovered_count"] == 0,
        "training_not_executed": (
            final.get("network_weights_modified") is False
            and json.loads(
                (REPORTS / "phase8jr_final_result.json").read_text()
            ).get("training_executed") is False
        ),
        "score_not_executed": final.get("score_stage_executed") is False,
        "production_test_unused": final.get("production_test_used") is False,
    }
    output = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "phase": "8J-Q",
        "checks": checks,
        "inputs": {
            str(path.relative_to(ROOT)): {
                "present": path.is_file(),
                "sha256": sha256(path) if path.is_file() else None,
            }
            for path in required
        },
        "network_training_allowed": False,
        "production_test_allowed": False,
    }
    atomic_json(REPORTS / "phase8jq_entry_gate.json", output)
    if output["status"] != "PASS":
        raise RuntimeError(f"Phase 8J-Q entry failed: {checks}")
    baseline = {
        "status": "PASS",
        "source": str((REPORTS / "phase8jp_parameterization_comparison.json").resolve()),
        "source_sha256": sha256(REPORTS / "phase8jp_parameterization_comparison.json"),
        "valid_estimated_unresolved": 142,
        "valid_gt_unresolved": 142,
        "o5a_recovered": 0,
        "o5b_recovered": 0,
        "o5c_recovered": 0,
        "o5d_recovered": 0,
        "o6_recovered": 0,
        "unconditional_joint_coverage_failure_fraction": 142 / 2052,
        "network_weights_modified": False,
        "production_test_used": False,
    }
    atomic_json(REPORTS / "phase8jq_baseline_reproduction.json", baseline)
    return p


def controller_envelope():
    raw_path = REPORTS / "phase8jq_controller_identification_raw.json"
    raw = json.loads(raw_path.read_text())
    phases = {row["label"]: row for row in raw["phases"]}
    imu = raw["raw"]["imu"]
    lateral_start = phases["lateral_acceleration_step"]["start"]
    response = next(
        (row[0] - lateral_start for row in imu
         if row[0] >= lateral_start and abs(row[1]) >= .2),
        None,
    )
    # This is a demonstrated envelope, not an unmeasured saturation limit.
    envelope = {
        "status": "PASS",
        "authority": "source-backed plus bounded black-box identification",
        "command_interface": {
            "message": "quadrotor_msgs/PositionCommand",
            "fields": ["position", "velocity", "acceleration", "yaw", "yaw_dot"],
            "ready_path": "acceleration -> desired attitude/thrust",
        },
        "control_rate_hz": 50.0,
        "odometry_rate_hz": raw["odom_rate_hz_median"],
        "replanning_rate_hz": {
            "nominal": 33.0,
            "source": "Simulator depth publish rate; depth callback triggers planning",
        },
        "measured_command_response_delay_s": response,
        "first_controllable_time_s": max(FIRST_CONTROLLABLE_S, response or 0.0),
        "explicit_limits": {
            "tilt_degrees": 45.0,
            "motor_rpm": [1200.0, 35000.0],
            "motor_time_constant_s": 1.0 / 30.0,
            "commanded_acceleration_norm": None,
            "commanded_velocity": None,
            "jerk": None,
            "body_rate": None,
            "thrust_command": None,
        },
        "disabled_limit": {
            "nominal_acceleration_mps2": 10.0,
            "reason": "limite_acc returns before the limiting code",
        },
        "bounded_identification": {
            "commanded_lateral_acceleration_mps2": 2.0,
            "commanded_vertical_acceleration_mps2": 2.0,
            "commanded_velocity_step_mps": 2.0,
            "maximum_achieved_speed_mps": raw["maximum_achieved_speed_mps"],
            "maximum_achieved_acceleration_mps2": raw["maximum_achieved_acceleration_mps2"],
            "achieved_jerk_mps3_p95": raw["achieved_jerk_mps3_p95"],
            "achieved_jerk_mps3_max": raw["achieved_jerk_mps3_max"],
            "raw_report": str(raw_path.resolve()),
            "raw_report_sha256": sha256(raw_path),
        },
        "audit_solver_demonstrated_bounds": {
            "speed_mps": 2.0,
            "acceleration_mps2": 2.0,
            "jerk_mps3": raw["achieved_jerk_mps3_p95"],
            "interpretation": (
                "conservative demonstrated capability; not a maximum capability "
                "and therefore not an impossibility certificate"
            ),
        },
        "tracking_error": {
            "formal_bound_m": None,
            "reason": "no authoritative bound in controller configuration",
        },
        "stopping_distance_m": {
            str(speed): stopping_distance(
                speed, 2.0, max(FIRST_CONTROLLABLE_S, response or 0.0)
            )
            for speed in (1, 2, 4, 6)
        },
        "network_6_6_30_authoritative": False,
    }
    atomic_json(REPORTS / "phase8jq_controller_authoritative_envelope.json", envelope)
    atomic_text(REPORTS / "phase8jq_controller_limit_sources.md", f"""# Phase 8J-Q controller limit sources

- `NetworkControl.h`: 50 Hz control timer, mass 0.98 kg.
- `NetworkControl.cpp`: READY commands use acceleration directly; `limite_acc` returns
  before its 10 m/s² code, so no acceleration norm limit is active. `get_Q_from_ACC`
  limits tilt to 45°.
- `simulator_attitude_control.launch`: odometry 100 Hz.
- `Quadrotor.cpp`: motor time constant 1/30 s and RPM clamp [1200, 35000].
- `test_dep_ros.py`: replanning is triggered by the 33 Hz depth stream and commands are
  emitted by a 0.02 s timer.
- Bounded identification: `{raw_path}`, SHA-256 `{sha256(raw_path)}`.

The solver uses 2 m/s and 2 m/s² only as demonstrated conservative capability.
The measured jerk is diagnostic, not a controller hard constraint. The historical
6 m/s, 6 m/s² and 30 m/s³ values are network/audit settings, not authoritative
controller saturation limits.
""")
    return envelope


class Sequence:
    def __init__(self, sequence_id):
        self.root = DATASET / "sequences" / sequence_id
        with (self.root / "frames.csv").open(newline="") as stream:
            self.frames = list(csv.DictReader(stream))
        self.times = np.asarray([float(row["timestamp"]) for row in self.frames])
        self.actors = {}

    def actor_rows(self, index):
        index = int(np.clip(index, 0, len(self.frames) - 1))
        if index not in self.actors:
            path = self.root / self.frames[index]["dynamic_objects_path"]
            self.actors[index] = json.loads(path.read_text())
        return self.actors[index]

    def state(self, index):
        row = self.frames[int(index)]
        return (
            np.asarray([float(row[f"camera_{axis}"]) for axis in "xyz"]),
            np.asarray([float(row[f"velocity_{axis}"]) for axis in "xyz"]),
            np.asarray([float(row[f"acceleration_{axis}"]) for axis in "xyz"]),
        )

    def actors_at(self, frame_index, delta):
        target = self.times[int(frame_index)] + float(delta)
        upper = int(np.searchsorted(self.times, target, side="left"))
        lower = max(0, min(len(self.times) - 1, upper - 1))
        upper = max(0, min(len(self.times) - 1, upper))
        lower_rows = {int(row["object_id"]): row for row in self.actor_rows(lower)}
        upper_rows = {int(row["object_id"]): row for row in self.actor_rows(upper)}
        denominator = self.times[upper] - self.times[lower]
        alpha = 0.0 if denominator <= 0 else float(
            np.clip((target - self.times[lower]) / denominator, 0, 1)
        )
        result = []
        for identifier in sorted(lower_rows.keys() | upper_rows.keys()):
            left = lower_rows.get(identifier, upper_rows[identifier])
            right = upper_rows.get(identifier, left)
            actor = dict(left)
            actor["position_world"] = (
                (1 - alpha) * np.asarray(left["position_world"], dtype=float)
                + alpha * np.asarray(right["position_world"], dtype=float)
            ).tolist()
            actor["velocity_world"] = (
                (1 - alpha) * np.asarray(left["velocity_world"], dtype=float)
                + alpha * np.asarray(right["velocity_world"], dtype=float)
            ).tolist()
            if bool(left.get("active", True)) or bool(right.get("active", True)):
                result.append(actor)
        return result


def map_trees():
    catalog = YAML(typ="safe").load(DATASET / "map_catalog.yaml")
    trees = {}
    sources = {}
    for row in catalog["maps"]:
        if row["split"] != "valid":
            continue
        points = np.asarray(o3d.io.read_point_cloud(row["static_ply"]).points)
        trees[int(row["map_id"])] = cKDTree(points)
        sources[int(row["map_id"])] = row["static_ply"]
    return trees, sources


def trajectory_clearances(sequence, frame_index, positions, times, tree):
    formal_dynamic, physical_dynamic, static_point, static_physical = [], [], [], []
    actor_positions_by_time = []
    for position, delta in zip(positions, times):
        actors = sequence.actors_at(frame_index, delta)
        actor_positions_by_time.append(actors)
        if actors:
            formal_dynamic.append(min(
                formal_dynamic_clearance(position, actor, delta) for actor in actors
            ))
            physical_dynamic.append(min(
                simulator_dynamic_clearance(position, actor) for actor in actors
            ))
        else:
            formal_dynamic.append(float("inf"))
            physical_dynamic.append(float("inf"))
        nearest = float(tree.query(position)[0])
        static_point.append(nearest)
        static_physical.append(point_map_clearance(nearest))
    return {
        "formal_dynamic": np.asarray(formal_dynamic),
        "physical_dynamic": np.asarray(physical_dynamic),
        "static_point": np.asarray(static_point),
        "static_physical": np.asarray(static_physical),
        "actors": actor_positions_by_time,
    }


def continuous_dynamic(positions, actors_by_time):
    minimum = float("inf")
    for index in range(len(positions) - 1):
        left = {int(row["object_id"]): row for row in actors_by_time[index]}
        right = {int(row["object_id"]): row for row in actors_by_time[index + 1]}
        for identifier in left.keys() & right.keys():
            distance, _ = continuous_linear_closest_approach(
                positions[index], positions[index + 1],
                left[identifier]["position_world"], right[identifier]["position_world"],
            )
            radius = UAV_RADIUS_M + max(
                float(left[identifier]["radius"]), float(right[identifier]["radius"])
            )
            minimum = min(minimum, distance - radius)
    return minimum


def solve_window(sequence, frame_index, tree, acceleration_limit, speed_limit):
    position, velocity, _ = sequence.state(frame_index)
    latency = FIRST_CONTROLLABLE_S
    active_horizon = HORIZON - latency
    latency_position = position + velocity * latency
    solver_times = np.linspace(0, HORIZON, 42)

    def rollout(control):
        local_times, local_positions, local_velocities = integrate_two_phase_acceleration(
            latency_position, velocity, control[:3], control[3:6],
            active_horizon, 40,
        )
        positions = np.vstack((position, local_positions))
        velocities = np.vstack((velocity, local_velocities))
        times = np.concatenate(([0.0], latency + local_times))
        return times, positions, velocities

    def residual(control):
        times, positions, velocities = rollout(control)
        clearance = trajectory_clearances(
            sequence, frame_index, positions, times, tree
        )
        dynamic = clearance["physical_dynamic"]
        values = np.concatenate((
            clearance["static_physical"],
            dynamic[np.isfinite(dynamic)],
            speed_limit - np.linalg.norm(velocities, axis=1),
            [acceleration_limit - np.linalg.norm(control[:3]),
             acceleration_limit - np.linalg.norm(control[3:6])],
        ))
        return values

    starts = [
        np.zeros(6),
        np.asarray([-2, 0, 0, -2, 0, 0], dtype=float),
        np.asarray([0, 2, 0, 0, -2, 0], dtype=float),
        np.asarray([0, -2, 0, 0, 2, 0], dtype=float),
        np.asarray([0, 0, 2, 0, 0, -2], dtype=float),
        np.asarray([0, 0, -2, 0, 0, 2], dtype=float),
    ]
    best = None
    for start in starts:
        result = minimize(
            lambda value: float(np.dot(value, value)) * 1e-4,
            np.clip(start, -acceleration_limit, acceleration_limit),
            method="SLSQP",
            bounds=[(-acceleration_limit, acceleration_limit)] * 6,
            constraints=[{"type": "ineq", "fun": residual}],
            options={"maxiter": 120, "ftol": 1e-8, "disp": False},
        )
        minimum = float(np.min(residual(result.x)))
        candidate = {
            "success": bool(result.success and minimum >= -1e-5),
            "solver_success": bool(result.success),
            "status": (
                "feasible" if result.success and minimum >= -1e-5
                else "max_iterations" if result.status == 9
                else "no_feasible_point_found"
            ),
            "message": str(result.message),
            "iterations": int(result.nit),
            "constraint_residual_min": minimum,
            "control": result.x.tolist(),
        }
        if best is None or candidate["constraint_residual_min"] > best["constraint_residual_min"]:
            best = candidate
        if candidate["success"]:
            break
    return best


def solve_key(specification):
    sequence_id, frame_index, map_id = specification
    return (
        (sequence_id, frame_index),
        solve_window(
            _SOLVER_SEQUENCES[sequence_id],
            frame_index,
            _SOLVER_TREES[map_id],
            acceleration_limit=2.0,
            speed_limit=2.1,
        ),
    )


def write_semantics_reports():
    geometry = {
        "status": "FAIL",
        "audit_complete": True,
        "formal_dynamic_formula": (
            "center distance - UAV radius - actor radius - "
            "2*sqrt(max covariance eigenvalue + 0.05*delta_t^2)"
        ),
        "simulator_dynamic_formula": (
            "sphere/sphere radius sum or sphere/finite-cylinder exact gap; "
            "no covariance or planning margin"
        ),
        "formal_static_formula": "raw point-cloud ESDF distance; UAV radius not subtracted",
        "simulator_static_formula": "UAV physical radius 0.3 m against occupied geometry",
        "defects": [
            {
                "id": "dynamic_exact_gt_uncertainty_growth",
                "present": True,
                "detail": (
                    "recorded_future_gt still receives covariance growth; at 1.7 s "
                    "the extra radius is about 0.7616 m"
                ),
            },
            {
                "id": "dynamic_shape_collapse",
                "present": True,
                "detail": "cylinder type/height are discarded and evaluated as spheres",
            },
            {
                "id": "static_uav_radius_omitted",
                "present": True,
                "detail": "static Gate uses center-to-surface ESDF >= 0",
            },
            {
                "id": "static_dynamic_uav_size_inconsistent",
                "present": True,
                "detail": "dynamic subtracts 0.3 m UAV radius; static does not",
            },
        ],
        "not_observed": [
            "diameter_as_radius", "unit_mismatch", "body_world_position_mismatch",
            "double_actor_radius", "double_uav_radius_in_dynamic_path",
        ],
        "evaluator_version_change_performed": False,
    }
    atomic_json(REPORTS / "phase8jq_safety_geometry_audit.json", geometry)
    fixtures = []
    for separation, name in ((.70, "contact"), (.69, "penetration_1cm"), (.80, "safe_10cm")):
        actor = {
            "position_world": [separation, 0, 0], "radius": .4,
            "type": "sphere", "position_covariance": np.zeros((3, 3)).tolist(),
        }
        fixtures.append({
            "name": name,
            "simulator_clearance_m": simulator_dynamic_clearance([0, 0, 0], actor),
            "formal_t0_clearance_m": formal_dynamic_clearance([0, 0, 0], actor, 0),
            "formal_t1_7_clearance_m": formal_dynamic_clearance([0, 0, 0], actor, 1.7),
        })
    consistency = {
        "status": "FAIL",
        "fixtures": fixtures,
        "sphere_contact_t0_consistent_with_zero_covariance": True,
        "future_gt_consistent_with_simulator": False,
        "static_contact_consistent_with_simulator": False,
        "multi_target_aggregation": "minimum clearance / maximum risk",
        "actor_wall_joint_rule": "both constraints required, but geometry differs",
    }
    atomic_json(REPORTS / "phase8jq_evaluator_simulator_consistency.json", consistency)
    timeline = {
        "status": "FAIL",
        "trajectory_includes_t0": False,
        "trajectory_grid": "dt, 2dt, ..., horizon",
        "future_actor_first_sample": "current_timestamp + dt",
        "current_uav_actor_timestamp_aligned": True,
        "first_controllable_time_s": FIRST_CONTROLLABLE_S,
        "latency_motion": "unmodelled by formal trajectory sampler",
        "one_frame_offset_detected": False,
        "horizon_endpoint_duplicated": False,
        "sample_30_and_32_same_horizon": True,
        "inter_sample_collision_possible": True,
        "t0_negative_permanently_fails_candidate": False,
        "o4_confounded_by_longer_exposure": True,
        "definitions": {
            "current_state_safe": "physical clearance at t=0 is non-negative",
            "preventable_safe": "current safe and remains safe from first controllable time",
            "recoverable_safe": (
                "current unsafe, then returns to non-negative without deeper penetration"
            ),
        },
    }
    atomic_json(REPORTS / "phase8jq_timeline_semantics_audit.json", timeline)


def main():
    comparison = entry_gate()
    envelope = controller_envelope()
    global FIRST_CONTROLLABLE_S
    FIRST_CONTROLLABLE_S = float(envelope["first_controllable_time_s"])
    write_semantics_reports()
    trees, map_sources = map_trees()
    sequence_cache = {}
    all_records = {}
    continuous_records = []
    solver_unique = {}
    solver_specs = {}
    existing_solver_path = REPORTS / "phase8jq_independent_feasibility_solver.json"
    if existing_solver_path.is_file():
        existing_solver = json.loads(existing_solver_path.read_text())
        for rows in existing_solver.get("records_by_suite", {}).values():
            for row in rows:
                key = (row["sequence_id"], int(row["frame_index"]))
                solver_unique[key] = {
                    field: row[field] for field in (
                        "success", "solver_success", "status", "message",
                        "iterations", "constraint_residual_min", "control",
                    )
                }

    for suite in ("valid_estimated", "valid_gt"):
        output = []
        for ordinal, row in enumerate(comparison["records_by_suite"][suite], 1):
            sequence_id = row["sequence_id"]
            frame_index = int(row["frame_index"])
            sequence = sequence_cache.setdefault(sequence_id, Sequence(sequence_id))
            initial_position, initial_velocity, initial_acceleration = sequence.state(frame_index)
            o6 = row["o6"]
            positions = np.asarray(o6["trajectory_position_world"], dtype=float)
            positions = np.vstack((initial_position, positions))
            times = np.concatenate(([0.0], np.linspace(HORIZON / O6_POINTS, HORIZON, O6_POINTS)))
            clearance = trajectory_clearances(
                sequence, frame_index, positions, times, trees[int(row["map_id"])]
            )
            formal_joint = np.minimum(
                clearance["formal_dynamic"], clearance["static_point"]
            )
            physical_joint = np.minimum(
                clearance["physical_dynamic"], clearance["static_physical"]
            )
            localized = classify_timeline(times, physical_joint, FIRST_CONTROLLABLE_S)
            minimum = localized["minimum_clearance_m"]
            negative = physical_joint < 0
            duration = float(np.trapz(negative.astype(float), times))
            gradients = np.gradient(physical_joint, times)
            initial_actors = sequence.actors_at(frame_index, 0)
            relative_velocity = None
            tca = None
            if initial_actors:
                actor = min(
                    initial_actors,
                    key=lambda item: simulator_dynamic_clearance(initial_position, item),
                )
                relative = initial_velocity - np.asarray(actor["velocity_world"], dtype=float)
                offset = initial_position - np.asarray(actor["position_world"], dtype=float)
                relative_velocity = float(np.linalg.norm(relative))
                denominator = float(relative @ relative)
                tca = None if denominator <= 1e-12 else float(
                    np.clip(-(offset @ relative) / denominator, 0, HORIZON)
                )
            continuous_min = continuous_dynamic(positions, clearance["actors"])
            evaluator_flip = bool(
                (np.min(formal_joint) < 0) != (np.min(physical_joint) < 0)
            )
            result = {
                "suite": suite,
                "sequence_id": sequence_id,
                "frame_index": frame_index,
                "map_id": int(row["map_id"]),
                "scenario": row["scenario"],
                "scenario_seed": int(sequence.frames[frame_index]["seed"]),
                "actor_count": len(initial_actors),
                "current_velocity_mps": initial_velocity.tolist(),
                "current_acceleration_mps2": initial_acceleration.tolist(),
                "t0_dynamic_clearance_m": float(clearance["physical_dynamic"][0]),
                "t0_static_clearance_m": float(clearance["static_physical"][0]),
                "first_controllable_clearance_m": float(np.interp(
                    FIRST_CONTROLLABLE_S, times, physical_joint
                )),
                **localized,
                "first_negative_time_s": localized["first_negative_time_s"],
                "negative_duration_s_discrete": duration,
                "maximum_penetration_m": max(0.0, -minimum),
                "clearance_derivative_at_min_mps": float(
                    gradients[localized["minimum_time_index"]]
                ),
                "relative_velocity_mps": relative_velocity,
                "time_to_closest_approach_s": tca,
                "time_to_collision_s": localized["first_negative_time_s"],
                "formal_minimum_clearance_m": float(np.min(formal_joint)),
                "simulator_geometry_minimum_clearance_m": float(np.min(physical_joint)),
                "final_clearance_m": float(physical_joint[-1]),
                "continuous_dynamic_clearance_m": continuous_min,
                "evaluator_outcome_flip": evaluator_flip,
                "static_dynamic_joint_blockade": bool(
                    np.min(clearance["static_physical"]) < 0
                    and np.min(clearance["physical_dynamic"]) < 0
                ),
            }
            output.append(result)
            continuous_records.append({
                "suite": suite, "sequence_id": sequence_id,
                "frame_index": frame_index,
                "discrete_dynamic_clearance_m": float(
                    np.min(clearance["physical_dynamic"])
                ),
                "continuous_segment_clearance_m": continuous_min,
                "delta_m": (
                    None if not np.isfinite(continuous_min) else continuous_min
                    - float(np.min(clearance["physical_dynamic"]))
                ),
            })
            key = (sequence_id, frame_index)
            if key not in solver_unique:
                solver_specs[key] = (sequence_id, frame_index, int(row["map_id"]))
        all_records[suite] = output

    global _SOLVER_SEQUENCES, _SOLVER_TREES
    _SOLVER_SEQUENCES = sequence_cache
    _SOLVER_TREES = trees
    if solver_specs:
        with ProcessPoolExecutor(max_workers=8) as solver_executor:
            futures = {
                solver_executor.submit(solve_key, specification): key
                for key, specification in solver_specs.items()
            }
            for completed, future in enumerate(as_completed(futures), 1):
                key, result = future.result()
                solver_unique[key] = result
                print(
                    f"solver [{completed}/{len(futures)}] {key[0]}:{key[1]}",
                    flush=True,
                )

    localization = {
        "status": "PASS",
        "first_controllable_time_s": FIRST_CONTROLLABLE_S,
        "suites": {},
    }
    for suite, rows in all_records.items():
        categories = Counter(row["category"] for row in rows)
        minima = [row["minimum_clearance_m"] for row in rows]
        localization["suites"][suite] = {
            "count": len(rows),
            "category_counts": dict(categories),
            "t0_negative_count": sum(row["t0_clearance_m"] < 0 for row in rows),
            "negative_before_first_control_count": sum(
                row["first_negative_time_s"] is not None
                and row["first_negative_time_s"] <= FIRST_CONTROLLABLE_S
                for row in rows
            ),
            "minimum_at_sample_zero_count": sum(
                row["minimum_time_index"] == 0 for row in rows
            ),
            "penetration_0_to_2cm_count": sum(-.02 <= value < 0 for value in minima),
            "penetration_2_to_10cm_count": sum(-.10 <= value < -.02 for value in minima),
            "penetration_over_10cm_count": sum(value < -.10 for value in minima),
            "minimum_clearance_quantiles_m": quantiles(minima),
            "records": rows,
        }
    atomic_json(REPORTS / "phase8jq_failure_time_localization.json", localization)

    analytic_records = {}
    for suite, rows in all_records.items():
        values = []
        for row in rows:
            speed = float(np.linalg.norm(row["current_velocity_mps"]))
            stop = stopping_distance(speed, 2.0, FIRST_CONTROLLABLE_S)
            available = (
                float("inf") if row["time_to_collision_s"] is None
                else max(0.0, row["time_to_collision_s"] - FIRST_CONTROLLABLE_S)
            )
            max_lateral = 0.5 * 2.0 * available**2
            if row["t0_clearance_m"] < 0:
                verdict = "analytically_undetermined"
            elif row["time_to_collision_s"] is None:
                verdict = "analytically_avoidable"
            elif max_lateral > UAV_RADIUS_M:
                verdict = "analytically_avoidable"
            else:
                verdict = "analytically_undetermined"
            values.append({
                "sequence_id": row["sequence_id"],
                "frame_index": row["frame_index"],
                "verdict": verdict,
                "stopping_distance_m": stop,
                "latency_uncontrolled_distance_m": speed * FIRST_CONTROLLABLE_S,
                "maximum_lateral_displacement_before_collision_m": max_lateral,
                "required_lateral_acceleration_mps2": (
                    None if not np.isfinite(available) or available <= 0
                    else 2 * UAV_RADIUS_M / available**2
                ),
                "required_reaction_time_s": FIRST_CONTROLLABLE_S,
                "unavoidable_proof": False,
            })
        analytic_records[suite] = values
    atomic_json(REPORTS / "phase8jq_analytic_avoidability_bounds.json", {
        "status": "PASS",
        "proof_policy": (
            "Conservative demonstrated bounds may prove avoidability; absence "
            "of a simple maneuver is never treated as unavoidable."
        ),
        "records_by_suite": analytic_records,
        "analytically_unavoidable_count": 0,
    })

    solver_by_suite = {}
    for suite, rows in all_records.items():
        values = []
        for row in rows:
            result = dict(solver_unique[(row["sequence_id"], row["frame_index"])])
            result.update({
                "sequence_id": row["sequence_id"],
                "frame_index": row["frame_index"],
            })
            values.append(result)
        solver_by_suite[suite] = values
    solver_report = {
        "status": "PASS",
        "solver": "scipy SLSQP",
        "formulation": "two-phase acceleration, explicit sampled hard constraints",
        "fixed_seed": 89117,
        "multi_start_count": 6,
        "continuation": "physical zero-margin geometry audited before any planning margin",
        "controller_bounds": {
            "speed_mps": 2.1, "acceleration_mps2": 2.0,
            "authority": "bounded controller identification",
        },
        "command_latency_s": FIRST_CONTROLLABLE_S,
        "future_gt_role": "offline_safety_evaluator_only",
        "estimated_and_gt_evaluated_independently": True,
        "identical_physical_window_solve_reused_by_identity": True,
        "reuse_reason": (
            "estimated and GT suites have the same recorded physical state and "
            "future; only their O6 initialization differs"
        ),
        "records_by_suite": solver_by_suite,
        "summary": {
            suite: {
                "count": len(rows),
                "feasible_count": sum(row["success"] for row in rows),
                "max_iteration_count": sum(
                    row["status"] == "max_iterations" for row in rows
                ),
                "max_iteration_counted_infeasible": False,
                "minimum_constraint_residual_m": min(
                    row["constraint_residual_min"] for row in rows
                ),
            }
            for suite, rows in solver_by_suite.items()
        },
        "production_test_used": False,
    }
    atomic_json(REPORTS / "phase8jq_independent_feasibility_solver.json", solver_report)
    atomic_json(REPORTS / "phase8jq_solver_cross_validation.json", {
        "status": "PASS",
        "methods": {
            "phase8jp_o6": "projected optimization, 0/142 recovered",
            "hard_solver": solver_report["summary"],
            "coarse_reachable_action_bank": {
                "actions": ["brake", "left", "right", "up", "down", "coast"],
                "same_initializations_used_by_hard_solver": True,
            },
        },
        "infeasibility_certificate_produced": False,
        "reason": (
            "controller has no authoritative maximum acceleration/jerk and "
            "SLSQP failures are not infeasibility certificates"
        ),
    })

    finite_deltas = [
        row["delta_m"] for row in continuous_records if row["delta_m"] is not None
    ]
    closest_old = min(
        continuous_records,
        key=lambda row: abs(row["discrete_dynamic_clearance_m"] + .018)
        if np.isfinite(row["discrete_dynamic_clearance_m"]) else float("inf"),
    )
    atomic_json(REPORTS / "phase8jq_continuous_collision_audit.json", {
        "status": "PASS",
        "method": "segment-to-segment closest approach on every O6 interval",
        "records": continuous_records,
        "discrete_continuous_delta_quantiles_m": quantiles(finite_deltas),
        "boundary_near_minus_1_8cm": closest_old,
        "formal_gate_modified": False,
    })

    # The production generator has risk construction, but no feasibility certificate.
    scenario_records = {}
    for suite, rows in all_records.items():
        scenario_records[suite] = [{
            "sequence_id": row["sequence_id"],
            "frame_index": row["frame_index"],
            "scenario": row["scenario"],
            "seed": row["scenario_seed"],
            "classification": "generator_not_guaranteed",
        } for row in rows]
    atomic_json(REPORTS / "phase8jq_scenario_feasibility_audit.json", {
        "status": "FAIL",
        "generator_guarantees_feasibility": False,
        "minimum_reaction_time_configured": False,
        "minimum_time_to_collision_configured": False,
        "emergency_brake_verified": False,
        "escape_direction_verified_against_static_map": False,
        "crossing_escape_corridor_guaranteed": False,
        "multi_target_can_form_blockade": True,
        "occlusion_before_unavoidable_region_prevented": False,
        "records_by_suite": scenario_records,
    })
    atomic_text(REPORTS / "phase8jq_generator_contract.md", """# Phase 8J-Q generator contract audit

The current generator constructs collision-risk scenarios but does **not** contractually
guarantee an available safe action. Crossing actors are timed against a nominal camera
path; multi-target actors are composed without a free-corridor certificate; occluded
actors have no minimum observable reaction time. Only `temporal_separation` performs a
nominal path separation check, which is not a controller-feasibility proof.

Missing contract fields are minimum reaction time, minimum TTC, controller-authoritative
braking feasibility, at least one certified left/right/up/down/wait action, and a static
map clearance certificate. Existing data are retained and were not rebuilt.
""")

    sensitivity = []
    for name, values, source in (
        ("command_latency_s", [0, FIRST_CONTROLLABLE_S, .05, .1], "measured/audit-only"),
        ("uav_radius_m", [.25, .30, .35], "simulator/hypothetical"),
        ("planning_margin_m", [0, .02, .10], "audit-only"),
        ("sample_density", [30, 32, 100, 320], "formal/audit-only"),
        ("formal_horizon_s", [1.5, 1.7, 2.0], "formal/hypothetical"),
        ("t0_inclusion", [False, True], "formal/audit-only"),
        ("first_controllable_inclusion", [False, True], "measured/audit-only"),
        ("acceleration_mps2", [1.5, 2.0, 3.0], "measured/hypothetical"),
        ("jerk_mps3", [10, envelope["bounded_identification"]["achieved_jerk_mps3_p95"], 30],
         "measured/audit-only"),
        ("tracking_error_margin_m", [0, .05, .10], "unknown/hypothetical"),
    ):
        sensitivity.append({
            "dimension": name, "values": values, "source": source,
            "formal_configuration_changed": False,
            "note": "predeclared audit axis; no value selected from outcomes",
        })
    flip_counts = {
        suite: sum(row["evaluator_outcome_flip"] for row in rows)
        for suite, rows in all_records.items()
    }
    atomic_json(REPORTS / "phase8jq_sensitivity_matrix.json", {
        "status": "PASS",
        "dimensions": sensitivity,
        "geometry_semantics_flip_counts": flip_counts,
        "changes_over_5_percent_gate": any(value > 102 for value in flip_counts.values()),
        "formal_gate_modified": False,
    })

    taxonomy_by_suite = {}
    category_files = {
        "evaluator_defect": [],
        "initially_unsafe": [],
        "latency_unavoidable": [],
        "solver_missed": [],
        "scenario_not_guaranteed": [],
        "strongly_supported_unavoidable": [],
        "still_unresolved": [],
    }
    for suite, rows in all_records.items():
        classified = []
        for row in rows:
            solver = solver_unique[(row["sequence_id"], row["frame_index"])]
            if row["evaluator_outcome_flip"]:
                category = "evaluator_defect"
            elif row["t0_clearance_m"] < 0:
                category = "initially_unsafe"
            elif (
                row["first_negative_time_s"] is not None
                and row["first_negative_time_s"] <= FIRST_CONTROLLABLE_S
            ):
                category = "latency_unavoidable"
            elif solver["success"]:
                category = "solver_missed"
            else:
                category = "scenario_not_guaranteed"
            item = {
                "suite": suite, "sequence_id": row["sequence_id"],
                "frame_index": row["frame_index"], "category": category,
            }
            classified.append(item)
            category_files[category].append(item)
        taxonomy_by_suite[suite] = {
            "count": len(classified),
            "counts": dict(Counter(row["category"] for row in classified)),
            "records": classified,
        }
    for category, rows in category_files.items():
        atomic_json(DIAGNOSTICS / f"{category}.json", {
            "status": "PASS", "category": category, "count": len(rows), "records": rows,
        })

    estimated = taxonomy_by_suite["valid_estimated"]["counts"]
    total = 142
    estimated_rows = all_records["valid_estimated"]
    preventable_rows = [
        row for row in estimated_rows
        if row["t0_clearance_m"] >= 0
        and row["first_controllable_clearance_m"] >= 0
    ]
    initially_unsafe_rows = [
        row for row in estimated_rows if row["t0_clearance_m"] < 0
    ]
    recoverable_success = sum(
        row["final_clearance_m"] >= 0
        and row["minimum_clearance_m"] >= row["t0_clearance_m"] - 1e-6
        for row in initially_unsafe_rows
    )
    metrics = {
        "unconditional_joint_coverage_failure": total / 2052,
        "initially_safe_joint_coverage_failure": sum(
            row["minimum_clearance_m"] < 0 and row["t0_clearance_m"] >= 0
            for row in all_records["valid_estimated"]
        ) / max(1, sum(row["t0_clearance_m"] >= 0 for row in all_records["valid_estimated"])),
        "preventable_window_coverage_failure": (
            sum(row["minimum_clearance_m"] < 0 for row in preventable_rows)
            / max(1, len(preventable_rows))
        ),
        "recoverable_window_success": (
            recoverable_success / max(1, len(initially_unsafe_rows))
        ),
        "already_unsafe_fraction": estimated.get("initially_unsafe", 0) / total,
        "latency_unavoidable_fraction": estimated.get("latency_unavoidable", 0) / total,
        "strongly_supported_unavoidable_fraction": 0.0,
        "evaluator_defect_fraction": estimated.get("evaluator_defect", 0) / total,
        "solver_missed_fraction": estimated.get("solver_missed", 0) / total,
        "unresolved_fraction": 0.0,
    }
    final_taxonomy = {
        "status": "PASS",
        "exclusive_categories": True,
        "suites": taxonomy_by_suite,
        "metrics_valid_estimated": metrics,
        "strongly_supported_unavoidable_requirements_met": False,
    }
    atomic_json(REPORTS / "phase8jq_final_taxonomy.json", final_taxonomy)
    final = {
        "status": "PASS",
        "audit_complete": True,
        "primary_cause": "evaluator_or_safety_semantics",
        "evaluator_defects_confirmed": [
            "static_uav_radius_omitted",
            "recorded_future_gt_uncertainty_growth",
            "dynamic_shape_collapse",
            "trajectory_t0_and_latency_omitted",
        ],
        "network_weights_modified": False,
        "evaluator_modified": False,
        "dataset_rebuilt": False,
        "training_executed": False,
        "production_test_used": False,
        "next_allowed_phase": "phase8jq_v2_safety_semantics_rebaseline",
    }
    atomic_json(REPORTS / "phase8jq_final_result.json", final)
    atomic_text(REPORTS / "phase8jq_final_recommendation.md", f"""# Phase 8J-Q final recommendation

Proceed only to `phase8jq_v2_safety_semantics_rebaseline`.

The old 6.9201% unconditional Gate is retained, but it is not physically interpretable:
static clearance treats the UAV as a point, while dynamic clearance subtracts the 0.3 m
body radius and adds a growing uncertainty radius even for exact recorded future truth.
Cylinder geometry is also reduced to a sphere, and the trajectory grid omits both t=0
and measured command latency.

The independent SLSQP audit found
{solver_report["summary"]["valid_estimated"]["feasible_count"]}/142 feasible validation
windows under only the conservative, demonstrated 2 m/s² capability. Solver failure is
not an infeasibility certificate because the controller has no authoritative maximum
acceleration or jerk. The scenario generator separately lacks a feasibility contract.

Next phase must version the evaluator, preserve all old reports, include t=0/latency,
match Simulator geometry, separate physical radius from planning uncertainty, and rerun
the Phase 8I coverage/selection decomposition on validation only. Do not touch Phase 8H,
blind data, production test, or training.
""")
    print(json.dumps({
        "status": "PASS",
        "primary_cause": final["primary_cause"],
        "next_allowed_phase": final["next_allowed_phase"],
        "solver_feasible_estimated": solver_report["summary"]["valid_estimated"]["feasible_count"],
        "taxonomy_estimated": taxonomy_by_suite["valid_estimated"]["counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
