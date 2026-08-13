#!/usr/bin/env python3
"""Launch one reproducible DE-P scene for interactive RViz evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from policy.runtime_profile_v4_4 import (
    PROFILE_NAME as V44_RUNTIME_PROFILE,
    RUNTIME_BEHAVIOR_VERSION as V44_RUNTIME_BEHAVIOR_VERSION,
)
from policy.runtime_profile_v4_5 import (
    PROFILE_NAME as V45_RUNTIME_PROFILE,
    RUNTIME_BEHAVIOR_VERSION as V45_RUNTIME_BEHAVIOR_VERSION,
)
from policy.runtime_profile_v4_5_7 import (
    PROFILE_NAME as V457_RUNTIME_PROFILE,
    RUNTIME_BEHAVIOR_VERSION as V457_RUNTIME_BEHAVIOR_VERSION,
)
from policy.runtime_profile_v4_5_10 import (
    PROFILE_NAME as V4510_RUNTIME_PROFILE,
    RUNTIME_BEHAVIOR_VERSION as V4510_RUNTIME_BEHAVIOR_VERSION,
)
from policy.runtime_profile_v4_7 import (
    PROFILE_NAME as V47_RUNTIME_PROFILE,
    RUNTIME_BEHAVIOR_VERSION as V47_RUNTIME_BEHAVIOR_VERSION,
)
from policy.runtime_profile_v4_8 import (
    PROFILE_NAME as V48_RUNTIME_PROFILE,
    RUNTIME_BEHAVIOR_VERSION as V48_RUNTIME_BEHAVIOR_VERSION,
)
from policy.runtime_profile_v4_8_1 import (
    PROFILE_NAME as V481_RUNTIME_PROFILE,
    RUNTIME_BEHAVIOR_VERSION as V481_RUNTIME_BEHAVIOR_VERSION,
)
from policy.runtime_profile_v4_8_2 import (
    PROFILE_NAME as V482_RUNTIME_PROFILE,
    RUNTIME_BEHAVIOR_VERSION as V482_RUNTIME_BEHAVIOR_VERSION,
)
from policy.runtime_profile_v4_8_5 import (
    PROFILE_NAME as V485_RUNTIME_PROFILE,
    RUNTIME_BEHAVIOR_VERSION as V485_RUNTIME_BEHAVIOR_VERSION,
)
from policy.runtime_profile_v4_9 import (
    PROFILE_NAME as V49_RUNTIME_PROFILE,
    RUNTIME_BEHAVIOR_VERSION as V49_RUNTIME_BEHAVIOR_VERSION,
)
DEFAULT_SCENES = ROOT / "configs" / "dep_interactive_demo_scenes_v4.json"
DEFAULT_RUN = (
    ROOT / "runs" / "phase8_mixed_static_yopo_v3_2_low_lr_adamw"
    / "20260730T074415Z-260761"
)
CONTROLLER = ROOT.parent / "Controller"
SIMULATOR = ROOT.parent / "Simulator"
YOPO_PYTHON = Path("/home/zjh/miniconda3/envs/yopo/bin/python")
UNGATED_STATIC_TRAINING_CONTRACTS = {
    "route_a_static_yopo_training_v4_3_single_cost_v1",
    "route_a_static_yopo_training_v4_4_static_parity_v1",
    "route_a_static_yopo_training_v4_5_bounded_danger_v1",
    "route_a_static_yopo_training_v4_5_1_relative_kinematic_v1",
    "route_a_static_yopo_training_v4_5_1_mixed_shakedown_v1",
    "route_a_static_yopo_training_v4_5_2_calibrated_v1",
    "route_a_static_yopo_training_v4_5_2_mixed_shakedown_v1",
    "route_a_static_yopo_training_v4_5_3_two_stage_shakedown_v1",
    "route_a_static_yopo_training_v4_5_4_independent_score_shakedown_v1",
    "route_a_static_yopo_training_v4_5_5_dense_static_esdf_shakedown_v1",
    "route_a_static_yopo_training_v4_5_6_continuous_score_safety_v1",
    "route_a_static_yopo_training_v4_5_7_high_safety_retimed_v1",
    "route_a_static_yopo_training_v4_5_8_localized_safety_retimed_v1",
    "route_a_static_yopo_training_v4_5_9_time_mean_localized_safety_v1",
    "route_a_static_yopo_training_v4_5_10_tail_aware_safety_v1",
    "route_a_static_yopo_training_v4_5_10_controlled_continuation_v1",
    "route_a_static_yopo_training_v4_6_four_scene_finetune_v1",
    "route_a_static_yopo_training_v4_7_original_density_finetune_v1",
    "route_a_static_yopo_training_v4_8_recovery_capacity_shakedown_v1",
    "route_a_static_yopo_training_v4_8_3_candidate_only_shakedown_v1",
    "route_a_static_yopo_training_v4_8_4_score_adaptation_v1",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch Controller, CUDA Simulator, DE-P and RViz as one demo."
    )
    parser.add_argument("--scene", required=True,
                        choices=("cave", "forest", "pillar", "room", "wall"))
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--checkpoint", type=Path)
    model.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--epoch", default="best",
        help="'best', 'last', an integer such as 14, or epoch_014.pth",
    )
    parser.add_argument(
        "--actors", choices=("none", "crossing", "head_on", "multi_target"),
        default="crossing",
    )
    parser.add_argument(
        "--actor-count", type=int, default=None,
        help="number of moving actors (1-64); defaults: none=0, crossing/head_on=1, multi_target=3",
    )
    parser.add_argument(
        "--actor-layout", choices=(
            "corridor", "map_wide", "hybrid", "route_encounters",
            "uniform_3d",
        ),
        default="hybrid",
        help=(
            "hybrid disperses actors map-wide while retaining some route "
            "encounters; route_encounters builds a deterministic multi-family "
            "stress route; uniform_3d uses deterministic seed-stratified XY/Z "
            "placement and random three-dimensional motion across the map"
        ),
    )
    parser.add_argument(
        "--actor-seed", type=int, default=8801,
        help="reproducible random seed for map-wide actor placement and motion",
    )
    parser.add_argument(
        "--actor-vertical-span", type=float, default=2.0,
        help="total vertical distribution/motion span in metres (0-6)",
    )
    parser.add_argument(
        "--dynamic-mode", choices=(
            "static_reactive", "dynamic_safety", "dynamic_attention"
        ),
        default="static_reactive",
        help=(
            "dynamic_safety enables causal actor prediction only in the hard "
            "safety layer; dynamic_attention also feeds attention into the CNN"
        ),
    )
    parser.add_argument("--arrival-radius", type=float, default=5.0)
    parser.add_argument(
        "--goal-mode", choices=("interactive", "fixed-ab"),
        default="interactive",
        help=(
            "interactive permits changing 2D Nav Goal during flight; "
            "fixed-ab automatically publishes the scene suggested goal once"
        ),
    )
    parser.add_argument(
        "--align-goal-before-planning", type=int, choices=(0, 1), default=1,
        help="align camera yaw before handing each newly accepted goal to YOPO",
    )
    parser.add_argument(
        "--runtime-safety", type=int, choices=(0, 1), default=1,
        help="enable the hard trajectory safety shield (default: 1)",
    )
    parser.add_argument(
        "--runtime-profile", choices=(
            "strict", "v4_3_minimal", V44_RUNTIME_PROFILE,
            V45_RUNTIME_PROFILE, V457_RUNTIME_PROFILE,
            V4510_RUNTIME_PROFILE, V47_RUNTIME_PROFILE,
            V48_RUNTIME_PROFILE, V481_RUNTIME_PROFILE,
            V482_RUNTIME_PROFILE, V485_RUNTIME_PROFILE,
            V49_RUNTIME_PROFILE,
        ),
        default="strict",
        help=(
            "V4.3/V4.4/V4.5 keep only the minimal physical and dynamic "
            "candidate filters"
        ),
    )
    parser.add_argument(
        "--planning-speed", type=float, default=None,
        help="YOPO lattice speed in m/s (V4.3 default wrapper uses 4.0)",
    )
    parser.add_argument(
        "--dynamic-foreground-mode",
        choices=("temporal_voxel", "range_image_hybrid"),
        default="range_image_hybrid",
        help=(
            "causal foreground extractor; range_image_hybrid is the frozen "
            "Phase-8H validated runtime default"
        ),
    )
    parser.add_argument(
        "--deadlock-recovery", type=int, choices=(0, 1), default=1,
        help="enable deterministic brake/scan/breadcrumb recovery (default: 1)",
    )
    parser.add_argument(
        "--deadlock-recovery-profile",
        choices=("legacy_v2", "bounded_scan_v3"),
        default="legacy_v2",
        help="legacy recovery or translation-free bounded observation scan",
    )
    parser.add_argument("--ros-master-port", type=int, default=11311)
    parser.add_argument("--scenes-config", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--no-rviz", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def resolve_checkpoint(args):
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
    else:
        run_dir = args.run_dir.expanduser().resolve()
        checkpoints = run_dir / "checkpoints"
        token = str(args.epoch).strip()
        if token == "best":
            checkpoint = checkpoints / "best.pth"
        elif token == "last":
            candidates = sorted(checkpoints.glob("epoch_*.pth"))
            if not candidates:
                raise FileNotFoundError(f"No epoch checkpoints in {checkpoints}")
            checkpoint = candidates[-1]
        else:
            name = token if token.endswith(".pth") else f"epoch_{int(token):03d}.pth"
            checkpoint = checkpoints / name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    return checkpoint


def load_scene(config_path, name):
    config_path = config_path.expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("contract_version") not in {
        "dep_interactive_demo_scenes_v1",
        "dep_interactive_demo_scenes_v2",
        "dep_interactive_demo_scenes_v3",
    }:
        raise ValueError(f"Unsupported scene contract: {payload.get('contract_version')}")
    scene = dict(payload["scenes"][name])
    scene["pointcloud"] = (ROOT / scene["pointcloud"]).resolve()
    if not scene["pointcloud"].is_file():
        raise FileNotFoundError(f"Scene point cloud does not exist: {scene['pointcloud']}")
    if len(scene["start"]) != 3 or len(scene["suggested_goal"]) != 3:
        raise ValueError(f"Scene {name} start/goal must be XYZ vectors")
    if "authority_root" in scene:
        scene["authority_root"] = (ROOT / scene["authority_root"]).resolve()
        if not (scene["authority_root"] / "occupancy.bin").is_file():
            raise FileNotFoundError(
                f"Scene authority does not exist: {scene['authority_root']}"
            )
        metadata = json.loads(
            (scene["authority_root"] / "occupancy_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        scene["flight_bounds_min"] = metadata["bounds_min"]
        scene["flight_bounds_max"] = metadata["bounds_max"]
    return scene


class CanonicalOccupancy:
    """Local exact collision queries against the versioned authority grid."""

    def __init__(self, root, expected_map_uuid):
        self.root = Path(root)
        metadata = json.loads(
            (self.root / "occupancy_metadata.json").read_text(encoding="utf-8")
        )
        if metadata["map_uuid"] != expected_map_uuid:
            raise ValueError("scene map_uuid does not match canonical occupancy")
        if metadata["bit_packing"] != "lsb_first":
            raise ValueError("unsupported canonical occupancy bit packing")
        self.origin = np.asarray(metadata["occupancy_origin"], dtype=np.float64)
        self.dimensions = np.asarray(
            metadata["occupancy_dimensions"], dtype=np.int64
        )
        self.resolution = float(metadata["occupancy_resolution_m"])
        packed = np.frombuffer(
            (self.root / "occupancy.bin").read_bytes(), dtype=np.uint8
        )
        count = int(np.prod(self.dimensions))
        self.grid = np.unpackbits(
            packed, bitorder="little", count=count
        ).reshape(tuple(self.dimensions)).astype(bool)
        self.bounds_min = self.origin
        self.bounds_max = self.origin + self.dimensions * self.resolution

    def _local_occupied(self, lower, upper):
        if np.any(lower < self.bounds_min) or np.any(upper > self.bounds_max):
            return None
        lo = np.floor((lower - self.origin) / self.resolution).astype(int)
        hi = np.floor((upper - self.origin) / self.resolution).astype(int)
        lo = np.clip(lo, 0, self.dimensions - 1)
        hi = np.clip(hi, 0, self.dimensions - 1)
        local = np.argwhere(
            self.grid[
                lo[0]:hi[0] + 1,
                lo[1]:hi[1] + 1,
                lo[2]:hi[2] + 1,
            ]
        )
        return local + lo if len(local) else local

    def sphere_collides(self, center, radius):
        center = np.asarray(center, dtype=np.float64)
        radius = float(radius)
        occupied = self._local_occupied(center - radius, center + radius)
        if occupied is None:
            return True
        if not len(occupied):
            return False
        box_min = self.origin + occupied * self.resolution
        box_max = box_min + self.resolution
        delta = np.maximum(np.maximum(box_min - center, center - box_max), 0.0)
        return bool(np.any(np.einsum("ij,ij->i", delta, delta) <= radius ** 2))

    def swept_sphere_collides(self, start, end, radius, max_step=None):
        """Conservatively check the complete motion between two odometry samples.

        The interactive controller can move farther than one occupancy voxel
        between callbacks.  Testing only the two reported centres can therefore
        miss a thin wall or tree trunk.  Sampling at no more than half a voxel
        makes the swept-sphere monitor independent of the odometry frequency.
        """
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        if start.shape != (3,) or end.shape != (3,):
            raise ValueError("swept sphere endpoints must be XYZ vectors")
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
            return True
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("swept sphere radius must be positive")
        step = 0.5 * self.resolution if max_step is None else float(max_step)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("swept sphere max_step must be finite and positive")
        distance = float(np.linalg.norm(end - start))
        sample_count = max(1, int(math.ceil(distance / step)))
        for amount in np.linspace(0.0, 1.0, sample_count + 1):
            if self.sphere_collides(start + amount * (end - start), radius):
                return True
        return False

    def cylinder_collides(self, center, radius, height):
        center = np.asarray(center, dtype=np.float64)
        half_height = 0.5 * float(height)
        lower = center + [-radius, -radius, -half_height]
        upper = center + [radius, radius, half_height]
        occupied = self._local_occupied(lower, upper)
        if occupied is None:
            return True
        if not len(occupied):
            return False
        box_min = self.origin + occupied * self.resolution
        box_max = box_min + self.resolution
        delta_xy = np.maximum(
            np.maximum(box_min[:, :2] - center[:2], center[:2] - box_max[:, :2]),
            0.0,
        )
        vertical_overlap = (
            (box_min[:, 2] <= center[2] + half_height)
            & (box_max[:, 2] >= center[2] - half_height)
        )
        radial_collision = (
            np.einsum("ij,ij->i", delta_xy, delta_xy) <= float(radius) ** 2
        )
        return bool(np.any(vertical_overlap & radial_collision))

    def actor_path_is_clear(self, actor, spacing=0.2):
        first, second = [
            np.asarray(value, dtype=np.float64)
            for value in actor["trajectory"]["waypoints_world"]
        ]
        samples = max(2, int(math.ceil(np.linalg.norm(second - first) / spacing)) + 1)
        for amount in np.linspace(0.0, 1.0, samples):
            center = first + amount * (second - first)
            if actor["shape"] == "sphere":
                collision = self.sphere_collides(center, actor["radius"])
            else:
                collision = self.cylinder_collides(
                    center, actor["radius"], actor["height"]
                )
            if collision:
                return False
        return True


def _point_segment_distance(point, first, second):
    point = np.asarray(point, dtype=np.float64)
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    segment = second - first
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-12:
        return float(np.linalg.norm(point - first))
    amount = float(np.dot(point - first, segment) / denominator)
    amount = min(1.0, max(0.0, amount))
    return float(np.linalg.norm(point - (first + amount * segment)))


def _segment_segment_closest(first_a, second_a, first_b, second_b):
    """Return the exact closest points on two finite 3-D line segments."""
    first_a = np.asarray(first_a, dtype=np.float64)
    second_a = np.asarray(second_a, dtype=np.float64)
    first_b = np.asarray(first_b, dtype=np.float64)
    second_b = np.asarray(second_b, dtype=np.float64)
    vector_a = second_a - first_a
    vector_b = second_b - first_b
    relative = first_a - first_b
    aa = float(np.dot(vector_a, vector_a))
    bb = float(np.dot(vector_b, vector_b))
    ab = float(np.dot(vector_a, vector_b))
    ar = float(np.dot(vector_a, relative))
    br = float(np.dot(vector_b, relative))
    denominator = aa * bb - ab * ab
    if aa <= 1e-12 and bb <= 1e-12:
        amount_a = amount_b = 0.0
    elif aa <= 1e-12:
        amount_a = 0.0
        amount_b = min(1.0, max(0.0, br / bb))
    elif bb <= 1e-12:
        amount_b = 0.0
        amount_a = min(1.0, max(0.0, -ar / aa))
    else:
        amount_a = (
            min(1.0, max(0.0, (ab * br - bb * ar) / denominator))
            if denominator > 1e-12 else 0.0
        )
        amount_b = (ab * amount_a + br) / bb
        if amount_b < 0.0:
            amount_b = 0.0
            amount_a = min(1.0, max(0.0, -ar / aa))
        elif amount_b > 1.0:
            amount_b = 1.0
            amount_a = min(1.0, max(0.0, (ab - ar) / aa))
    point_a = first_a + amount_a * vector_a
    point_b = first_b + amount_b * vector_b
    return point_a, point_b, float(np.linalg.norm(point_a - point_b))


def _route_encounter_evidence(actor, candidate=None):
    """Recompute a route encounter from geometry and time, never metadata.

    ``first appearance`` has an explicit simulator meaning: the actor becomes
    active at ``trajectory.start_time``.  It is compared with a nominal UAV
    flying from route start to route goal at 3 m/s with its camera aligned to
    the route.  An actor therefore cannot first appear in the rear blind
    half-space merely because a manifest calls it an encounter.

    Temporal proximity is solved exactly for every outbound/return leg of the
    ping-pong actor while the nominal UAV traverses the route.  This makes the
    certificate survive path shortening and translation during canonical-map
    clearance, rather than trusting the pre-relocation anchor label.
    """
    contract = actor.get("route_encounter_contract")
    if contract is None:
        return None
    trajectory = actor["trajectory"]
    waypoints = (
        candidate if candidate is not None else trajectory["waypoints_world"]
    )
    actor_first, actor_second = (
        np.asarray(value, dtype=np.float64) for value in waypoints
    )
    route_first = np.asarray(contract["route_start_world"], dtype=np.float64)
    route_second = np.asarray(contract["route_goal_world"], dtype=np.float64)
    anchor = np.asarray(contract["anchor_world"], dtype=np.float64)
    route_vector = route_second - route_first
    route_length = float(np.linalg.norm(route_vector))
    actor_vector = actor_second - actor_first
    actor_path_length = float(np.linalg.norm(actor_vector))
    nominal_speed = float(contract.get("nominal_uav_speed_mps", 3.0))
    actor_speed = float(trajectory["speed"])
    start_time = float(trajectory["start_time"])
    end_time = float(trajectory["end_time"])
    failures = []
    if route_length <= 1e-9 or actor_path_length <= 1e-9:
        return {
            "contract_version": "route_encounter_evidence_v1",
            "passed": False,
            "failures": ["degenerate_route_or_actor_path"],
        }
    route_direction = route_vector / route_length
    actor_direction = actor_vector / actor_path_length
    actor_velocity = actor_speed * actor_direction
    route_duration = route_length / nominal_speed
    first_uav_position = route_first + route_direction * min(
        route_length, nominal_speed * start_time
    )
    first_relative = actor_first - first_uav_position
    first_longitudinal = float(np.dot(first_relative, route_direction))
    first_lateral_vector = first_relative - first_longitudinal * route_direction
    first_lateral = float(np.linalg.norm(first_lateral_vector))
    first_bearing = math.degrees(math.atan2(
        first_lateral, max(0.0, first_longitudinal)
    ))
    direction_cosine = float(np.dot(actor_direction, route_direction))
    _, _, route_distance = _segment_segment_closest(
        actor_first, actor_second, route_first, route_second
    )
    anchor_distance = _point_segment_distance(
        anchor, actor_first, actor_second
    )

    minimum_longitudinal = float(
        contract.get("minimum_first_appearance_longitudinal_m", 1.0)
    )
    maximum_bearing = float(
        contract.get("maximum_first_appearance_bearing_deg", 80.0)
    )
    if first_longitudinal < minimum_longitudinal:
        failures.append("first_appearance_not_ahead")
    if first_bearing > maximum_bearing:
        failures.append("first_appearance_outside_forward_cone")
    if anchor_distance > float(contract["maximum_anchor_distance_m"]):
        failures.append("actor_path_misses_declared_anchor")
    if route_distance > float(contract["maximum_route_distance_m"]):
        failures.append("actor_path_misses_route")

    family = actor.get("encounter_family")
    if family in {"crossing", "staggered_crossing"}:
        if abs(direction_cosine) > float(
            contract.get("maximum_crossing_direction_cosine", 0.25)
        ):
            failures.append("crossing_velocity_not_transverse")
        # A crossing must actually traverse the route centreline; merely
        # travelling parallel within the distance tolerance is insufficient.
        lateral_axis = np.cross(route_direction, [0.0, 0.0, 1.0])
        lateral_norm = float(np.linalg.norm(lateral_axis))
        if lateral_norm > 1e-9:
            lateral_axis /= lateral_norm
            signed_first = float(np.dot(actor_first - route_first, lateral_axis))
            signed_second = float(np.dot(actor_second - route_first, lateral_axis))
            if signed_first * signed_second > 1e-9:
                failures.append("crossing_does_not_span_route")
        if family == "staggered_crossing" and start_time <= 0.0:
            failures.append("staggered_actor_not_delayed")
    elif family == "head_on":
        if direction_cosine > float(
            contract.get("maximum_head_on_direction_cosine", -0.8)
        ):
            failures.append("head_on_velocity_not_opposing_route")
    elif family == "same_direction_slow":
        if direction_cosine < float(
            contract.get("minimum_same_direction_cosine", 0.8)
        ):
            failures.append("same_direction_velocity_not_forward")
        if actor_speed > float(
            contract.get("maximum_same_direction_speed_mps", 0.65)
        ) or actor_speed >= nominal_speed:
            failures.append("same_direction_actor_not_slow")
    else:
        failures.append("unknown_encounter_family")

    best = None
    leg_duration = actor_path_length / actor_speed
    active_end = min(end_time, route_duration)
    leg_index = 0
    leg_start_time = start_time
    while leg_start_time < active_end - 1e-12:
        leg_end_time = min(active_end, leg_start_time + leg_duration)
        if leg_index % 2 == 0:
            leg_first, leg_velocity = actor_first, actor_velocity
        else:
            leg_first, leg_velocity = actor_second, -actor_velocity
        constant = (
            leg_first - leg_velocity * leg_start_time - route_first
        )
        relative_velocity = leg_velocity - nominal_speed * route_direction
        denominator = float(np.dot(relative_velocity, relative_velocity))
        if denominator <= 1e-12:
            encounter_time = leg_start_time
        else:
            encounter_time = -float(
                np.dot(constant, relative_velocity)
            ) / denominator
            encounter_time = min(
                leg_end_time, max(leg_start_time, encounter_time)
            )
        actor_position = (
            leg_first + leg_velocity * (encounter_time - leg_start_time)
        )
        uav_position = (
            route_first
            + nominal_speed * encounter_time * route_direction
        )
        separation = float(np.linalg.norm(actor_position - uav_position))
        record = (
            separation, encounter_time, leg_index, actor_position, uav_position
        )
        if best is None or record[0] < best[0]:
            best = record
        leg_index += 1
        leg_start_time += leg_duration
    if best is None:
        failures.append("no_active_overlap_with_nominal_route")
        best = (
            float("inf"), float("nan"), -1,
            np.full(3, np.nan), np.full(3, np.nan),
        )
    elif best[0] > float(
        contract.get("maximum_nominal_encounter_distance_m", 2.5)
    ):
        failures.append("nominal_route_encounter_too_distant")

    return {
        "contract_version": "route_encounter_evidence_v1",
        "passed": not failures,
        "failures": failures,
        "first_appearance_definition": (
            "actor active at start_time relative to a route-aligned nominal "
            "3mps UAV; forward cone excludes the rear blind half-space"
        ),
        "first_appearance_relative_longitudinal_m": first_longitudinal,
        "first_appearance_relative_lateral_m": first_lateral,
        "first_appearance_bearing_deg": first_bearing,
        "outbound_velocity_route_direction_cosine": direction_cosine,
        "actor_speed_mps": actor_speed,
        "actor_path_to_route_distance_m": route_distance,
        "actor_path_to_anchor_distance_m": anchor_distance,
        "nominal_encounter_distance_m": best[0],
        "nominal_encounter_time_s": best[1],
        "nominal_encounter_actor_leg_index": best[2],
        "nominal_encounter_actor_position_world": best[3].tolist(),
        "nominal_encounter_uav_position_world": best[4].tolist(),
        "nominal_encounter_route_fraction": (
            min(1.0, max(0.0, nominal_speed * best[1] / route_length))
            if np.isfinite(best[1]) else None
        ),
    }


def _route_contract_is_preserved(actor, candidate):
    """Validate actual post-relocation geometry and timing, not actor labels."""
    evidence = _route_encounter_evidence(actor, candidate)
    return evidence is None or evidence["passed"]


def enforce_canonical_actor_clearance(scene, actors, direction, perpendicular):
    if not actors or "authority_root" not in scene:
        route_actors = [
            actor for actor in actors if "route_encounter_contract" in actor
        ]
        route_contract_preserved = 0
        for actor in route_actors:
            evidence = _route_encounter_evidence(actor)
            actor["route_encounter_evidence"] = evidence
            route_contract_preserved += int(evidence["passed"])
        if route_contract_preserved != len(route_actors):
            raise ValueError("route encounter geometry/time contract failed")
        return {
            "checked": False,
            "relocated_actor_count": 0,
            "route_contract_actor_count": len(route_actors),
            "route_contract_preserved_actor_count": route_contract_preserved,
        }
    authority = CanonicalOccupancy(scene["authority_root"], scene["map_uuid"])
    for label, position in (
        ("start", scene["start"]),
        ("suggested_goal", scene["suggested_goal"]),
    ):
        if authority.sphere_collides(position, 0.3):
            raise ValueError(f"forest {label} is not UAV-clear in canonical occupancy")

    relocated = 0
    z_min, z_max = map(float, scene["actor_z_bounds"])
    for actor in actors:
        original = [
            np.asarray(value, dtype=np.float64)
            for value in actor["trajectory"]["waypoints_world"]
        ]
        midpoint = 0.5 * (original[0] + original[1])
        accepted = None
        for path_scale in (1.0, 0.75, 0.5, 0.25):
            scaled = [
                midpoint + path_scale * (value - midpoint) for value in original
            ]
            for z_shift in (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2):
                for along in (
                    0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5,
                    2.0, -2.0, 3.0, -3.0,
                ):
                    for lateral in (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2):
                        shift = np.asarray([
                            along * direction[0] + lateral * perpendicular[0],
                            along * direction[1] + lateral * perpendicular[1],
                            z_shift,
                        ])
                        candidate = [value + shift for value in scaled]
                        if any(
                            not z_min <= value[2] <= z_max for value in candidate
                        ):
                            continue
                        actor["initial_position_world"] = candidate[0].tolist()
                        actor["trajectory"]["waypoints_world"] = [
                            value.tolist() for value in candidate
                        ]
                        if (
                            _route_contract_is_preserved(actor, candidate)
                            and authority.actor_path_is_clear(actor)
                        ):
                            accepted = candidate
                            actor["canonical_path_scale"] = path_scale
                            break
                    if accepted is not None:
                        break
                if accepted is not None:
                    break
            if accepted is not None:
                break
        if accepted is None and "actor_xy_bounds" in scene:
            # Dense maps can have no valid solution near the initially assigned
            # stratum. Re-sample the whole authority extent without weakening
            # the exact path-clearance contract.
            actor_rng = np.random.default_rng(7919 * int(actor["id"]))
            (x_min, x_max), (y_min, y_max) = scene["actor_xy_bounds"]
            for _ in range(512):
                path_scale = float(actor_rng.choice((0.75, 0.5, 0.25)))
                scaled = [
                    midpoint + path_scale * (value - midpoint)
                    for value in original
                ]
                scaled_midpoint = 0.5 * (scaled[0] + scaled[1])
                target = np.asarray([
                    actor_rng.uniform(x_min, x_max),
                    actor_rng.uniform(y_min, y_max),
                    actor_rng.uniform(z_min, z_max),
                ])
                candidate = [
                    value + target - scaled_midpoint for value in scaled
                ]
                if any(not z_min <= value[2] <= z_max for value in candidate):
                    continue
                actor["initial_position_world"] = candidate[0].tolist()
                actor["trajectory"]["waypoints_world"] = [
                    value.tolist() for value in candidate
                ]
                if (
                    _route_contract_is_preserved(actor, candidate)
                    and authority.actor_path_is_clear(actor)
                ):
                    accepted = candidate
                    actor["canonical_path_scale"] = path_scale
                    actor["canonical_global_resample"] = True
                    break
        if accepted is None:
            raise ValueError(
                f"cannot place actor {actor['id']} clear of canonical forest geometry"
            )
        if any(not np.allclose(a, b) for a, b in zip(original, accepted)):
            relocated += 1
    route_actors = [
        actor for actor in actors if "route_encounter_contract" in actor
    ]
    route_contract_preserved = 0
    for actor in route_actors:
        evidence = _route_encounter_evidence(actor)
        actor["route_encounter_evidence"] = evidence
        route_contract_preserved += int(evidence["passed"])
    if route_contract_preserved != len(route_actors):
        raise ValueError("canonical relocation degraded a route encounter contract")
    return {
        "checked": True,
        "relocated_actor_count": relocated,
        "route_contract_actor_count": len(route_actors),
        "route_contract_preserved_actor_count": route_contract_preserved,
    }


def _vector_geometry(scene):
    start = [float(value) for value in scene["start"]]
    goal = [float(value) for value in scene["suggested_goal"]]
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 10.0:
        raise ValueError("Interactive route must be at least 10 m long")
    direction = [dx / length, dy / length]
    perpendicular = [-direction[1], direction[0]]

    def point(fraction, offset=0.0, z=None):
        return [
            start[0] + fraction * dx + offset * perpendicular[0],
            start[1] + fraction * dy + offset * perpendicular[1],
            float(
                start[2] + fraction * (goal[2] - start[2])
                if z is None else z
            ),
        ]

    return start, goal, direction, perpendicular, point


def resolve_actor_count(actor_mode, actor_count):
    defaults = {"none": 0, "crossing": 1, "head_on": 1, "multi_target": 3}
    count = defaults[actor_mode] if actor_count is None else int(actor_count)
    if actor_mode == "none":
        if count != 0:
            raise ValueError("--actors none requires --actor-count 0")
    elif not 1 <= count <= 64:
        raise ValueError("--actor-count must be within [1, 64]")
    return count


def _build_uniform_3d_actor_layout(
    scene, count, actor_seed, vertical_span,
):
    """Build a deterministic stratified full-map 3-D actor fixture.

    This is deliberately different from ``route_encounters``.  XY is divided
    into near-square strata and Z uses four independently shuffled balanced
    layers.  Each accepted actor remains in its assigned XY/Z stratum;
    dense-map rejection sampling is therefore unable to silently collapse the
    fixture back onto the nominal UAV route.
    """
    if "actor_xy_bounds" not in scene:
        raise ValueError("uniform_3d actor layout requires scene actor_xy_bounds")
    if float(vertical_span) <= 0.0:
        raise ValueError(
            "uniform_3d actor layout requires positive actor vertical span"
        )

    (x_min, x_max), (y_min, y_max) = [
        tuple(map(float, bounds)) for bounds in scene["actor_xy_bounds"]
    ]
    if "flight_bounds_min" in scene and "flight_bounds_max" in scene:
        z_min = float(scene["flight_bounds_min"][2])
        z_max = float(scene["flight_bounds_max"][2])
        z_bounds_source = "canonical_flight_bounds"
    else:
        z_min, z_max = map(float, scene["actor_z_bounds"])
        z_bounds_source = "legacy_actor_z_bounds_fallback"
    if not (x_min < x_max and y_min < y_max and z_min < z_max):
        raise ValueError("uniform_3d actor bounds must be increasing")

    rng = np.random.default_rng(int(actor_seed))
    columns = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / columns))
    cell_width = (x_max - x_min) / columns
    cell_height = (y_max - y_min) / rows
    cells = [
        (row, column)
        for row in range(rows)
        for column in range(columns)
    ]
    rng.shuffle(cells)
    cells = cells[:count]
    z_stratum_count = min(4, count)
    z_strata = [index % z_stratum_count for index in range(count)]
    rng.shuffle(z_strata)

    authority = None
    if "authority_root" in scene:
        authority = CanonicalOccupancy(scene["authority_root"], scene["map_uuid"])
        for label, position in (
            ("start", scene["start"]),
            ("suggested_goal", scene["suggested_goal"]),
        ):
            if authority.sphere_collides(position, 0.3):
                raise ValueError(
                    f"{label} is not UAV-clear in canonical occupancy"
                )

    actors = []
    accepted_initial_positions = []
    rejection_counts = {
        "insufficient_motion_extent": 0,
        "launch_or_goal_proximity": 0,
        "initial_actor_overlap": 0,
        "canonical_static_collision": 0,
    }
    maximum_attempts = 2048
    # A common centre interval makes the Z strata comparable even though the
    # emitted spheres have slightly different radii.
    largest_radius = 0.44
    z_center_min = z_min + largest_radius
    z_center_max = z_max - largest_radius
    if z_center_max - z_center_min < 1.0:
        raise ValueError("uniform_3d effective actor Z band is too small")

    for index, ((row, column), z_stratum) in enumerate(zip(cells, z_strata)):
        actor_id = 101 + index
        radius = 0.35 + 0.03 * (index % 4)
        cell_x_min = x_min + column * cell_width
        cell_x_max = cell_x_min + cell_width
        cell_y_min = y_min + row * cell_height
        cell_y_max = cell_y_min + cell_height
        horizontal_margin = radius + 0.15
        if (
            cell_x_max - cell_x_min <= 2.0 * horizontal_margin
            or cell_y_max - cell_y_min <= 2.0 * horizontal_margin
        ):
            raise ValueError("uniform_3d XY stratum is too small for actor")
        raw_z_layer_min = z_min + (
            z_stratum * (z_max - z_min) / z_stratum_count
        )
        raw_z_layer_max = z_min + (
            (z_stratum + 1) * (z_max - z_min) / z_stratum_count
        )
        safe_z_layer_min = max(raw_z_layer_min, z_min + radius)
        safe_z_layer_max = min(raw_z_layer_max, z_max - radius)
        if safe_z_layer_max - safe_z_layer_min < 0.2:
            raise ValueError("uniform_3d Z stratum is too small for actor")

        accepted = None
        actor_rejections = {key: 0 for key in rejection_counts}
        for attempt in range(1, maximum_attempts + 1):
            center = np.asarray([
                rng.uniform(
                    cell_x_min + horizontal_margin,
                    cell_x_max - horizontal_margin,
                ),
                rng.uniform(
                    cell_y_min + horizontal_margin,
                    cell_y_max - horizontal_margin,
                ),
                rng.uniform(
                    safe_z_layer_min + 0.1 * (
                        safe_z_layer_max - safe_z_layer_min
                    ),
                    safe_z_layer_max - 0.1 * (
                        safe_z_layer_max - safe_z_layer_min
                    ),
                ),
            ], dtype=np.float64)

            azimuth = float(rng.uniform(-math.pi, math.pi))
            vertical_component = float(rng.uniform(0.16, 0.48))
            if rng.integers(0, 2) == 0:
                vertical_component *= -1.0
            horizontal_component = math.sqrt(1.0 - vertical_component ** 2)
            motion_direction = np.asarray([
                horizontal_component * math.cos(azimuth),
                horizontal_component * math.sin(azimuth),
                vertical_component,
            ], dtype=np.float64)

            lower = np.asarray([
                cell_x_min + horizontal_margin,
                cell_y_min + horizontal_margin,
                safe_z_layer_min,
            ])
            upper = np.asarray([
                cell_x_max - horizontal_margin,
                cell_y_max - horizontal_margin,
                safe_z_layer_max,
            ])
            maximum_half_extent = float("inf")
            for axis in range(3):
                component = abs(float(motion_direction[axis]))
                if component > 1e-9:
                    maximum_half_extent = min(
                        maximum_half_extent,
                        (center[axis] - lower[axis]) / component,
                        (upper[axis] - center[axis]) / component,
                    )
            maximum_half_extent = min(
                maximum_half_extent,
                0.5 * float(vertical_span) / abs(vertical_component),
                3.5,
            )
            if maximum_half_extent < 0.45:
                actor_rejections["insufficient_motion_extent"] += 1
                continue
            half_extent = float(rng.uniform(
                0.42,
                max(0.420001, min(3.2, 0.92 * maximum_half_extent)),
            ))
            first = center - half_extent * motion_direction
            second = center + half_extent * motion_direction
            if rng.integers(0, 2) == 0:
                first, second = second, first

            # Avoid turning a deterministic validation launch into an
            # immediate spawn collision.  This does not bias actors toward the
            # route; it only excludes a small ball around both endpoints.
            if (
                _point_segment_distance(scene["start"], first, second) < 1.5
                or _point_segment_distance(
                    scene["suggested_goal"], first, second
                ) < 1.0
            ):
                actor_rejections["launch_or_goal_proximity"] += 1
                continue
            if any(
                np.linalg.norm(first - previous_position)
                < radius + previous_radius + 0.6
                for previous_position, previous_radius in accepted_initial_positions
            ):
                actor_rejections["initial_actor_overlap"] += 1
                continue

            actor = {
                "id": actor_id,
                "enabled": True,
                "shape": "sphere",
                "radius": radius,
                "initial_position_world": first.tolist(),
                "trajectory": {
                    "type": "waypoint_ping_pong",
                    "waypoints_world": [first.tolist(), second.tolist()],
                    "speed": float(rng.uniform(0.65, 1.25)),
                    "start_time": 0.0,
                    "end_time": 3600.0,
                },
            }
            if authority is not None and not authority.actor_path_is_clear(actor):
                actor_rejections["canonical_static_collision"] += 1
                continue

            for key, value in actor_rejections.items():
                rejection_counts[key] += value
            actual_center = 0.5 * (first + second)
            actual_direction = second - first
            actual_direction /= np.linalg.norm(actual_direction)
            actor["uniform_3d_sampling_evidence"] = {
                "contract_version": "uniform_3d_actor_sample_v1",
                "seed": int(actor_seed),
                "xy_cell": [int(row), int(column)],
                "xy_grid_shape": [int(rows), int(columns)],
                "z_stratum": int(z_stratum),
                "z_stratum_count": int(z_stratum_count),
                "uniform_3d_z_bounds": [z_min, z_max],
                "z_bounds_source": z_bounds_source,
                "accepted_attempt": int(attempt),
                "rejections_before_acceptance": actor_rejections,
                "sampled_center_world": actual_center.tolist(),
                "motion_unit_direction": actual_direction.tolist(),
                "motion_path_length_m": float(np.linalg.norm(second - first)),
                "vertical_displacement_m": float(abs(second[2] - first[2])),
                "canonical_path_clear": authority is not None,
            }
            accepted = actor
            break
        if accepted is None:
            raise ValueError(
                "cannot place uniform_3d actor "
                f"{actor_id} in XY cell {(row, column)} and Z stratum "
                f"{z_stratum} after {maximum_attempts} attempts"
            )
        actors.append(accepted)
        accepted_initial_positions.append((
            np.asarray(accepted["initial_position_world"], dtype=np.float64),
            float(accepted["radius"]),
        ))

    centers = np.asarray([
        np.mean(np.asarray(actor["trajectory"]["waypoints_world"]), axis=0)
        for actor in actors
    ])
    normalized_centers = np.column_stack((
        (centers[:, 0] - x_min) / (x_max - x_min),
        (centers[:, 1] - y_min) / (y_max - y_min),
        (centers[:, 2] - z_center_min) / (z_center_max - z_center_min),
    ))
    motion_octants = set()
    vertical_direction_signs = set()
    route_proximity_threshold_m = 3.0
    route_proximity_actor_count = 0
    route_start = np.asarray(scene["start"], dtype=np.float64)
    route_goal = np.asarray(scene["suggested_goal"], dtype=np.float64)
    for actor in actors:
        first, second = [
            np.asarray(value, dtype=np.float64)
            for value in actor["trajectory"]["waypoints_world"]
        ]
        direction = second - first
        horizontal_angle = math.atan2(float(direction[1]), float(direction[0]))
        motion_octants.add(int(math.floor(
            ((horizontal_angle + math.pi) % (2.0 * math.pi))
            * 4.0 / math.pi
        )))
        vertical_direction_signs.add(1 if direction[2] >= 0.0 else -1)
        _, _, distance_to_route = _segment_segment_closest(
            first, second, route_start, route_goal
        )
        if distance_to_route <= route_proximity_threshold_m:
            route_proximity_actor_count += 1
    evidence = {
        "contract_version": "uniform_3d_layout_v1",
        "sampling_method": (
            "seeded_balanced_xyz_stratification_with_canonical_rejection_v1"
        ),
        "seed": int(actor_seed),
        "requested_actor_count": int(count),
        "accepted_actor_count": len(actors),
        "xy_bounds": [[x_min, x_max], [y_min, y_max]],
        "actor_z_bounds": [z_min, z_max],
        "uniform_3d_z_bounds": [z_min, z_max],
        "z_bounds_source": z_bounds_source,
        "effective_z_center_bounds": [z_center_min, z_center_max],
        "xy_grid_shape": [rows, columns],
        "occupied_xy_cell_count": len({
            tuple(actor["uniform_3d_sampling_evidence"]["xy_cell"])
            for actor in actors
        }),
        "occupied_z_stratum_count": len({
            actor["uniform_3d_sampling_evidence"]["z_stratum"]
            for actor in actors
        }),
        "z_stratum_count": int(z_stratum_count),
        "normalized_center_min": normalized_centers.min(axis=0).tolist(),
        "normalized_center_max": normalized_centers.max(axis=0).tolist(),
        "normalized_center_span": np.ptp(
            normalized_centers, axis=0
        ).tolist(),
        "all_trajectories_have_3d_motion": all(
            actor["uniform_3d_sampling_evidence"]["vertical_displacement_m"]
            > 1e-6 for actor in actors
        ),
        "canonical_occupancy_checked": authority is not None,
        "canonical_static_clearance_rejections": rejection_counts[
            "canonical_static_collision"
        ],
        "motion_direction_octants_occupied": len(motion_octants),
        "motion_direction_octants": [
            int(octant) for octant in sorted(motion_octants)
        ],
        "vertical_direction_signs": sorted(vertical_direction_signs),
        "route_proximity_threshold_m": route_proximity_threshold_m,
        "route_proximity_actor_count": route_proximity_actor_count,
        "route_encounter_guaranteed": False,
        "planner_ground_truth_exposed": False,
        "rejection_counts": rejection_counts,
    }
    evidence["passed"] = bool(
        len(actors) == count
        and evidence["occupied_xy_cell_count"] == count
        and evidence["occupied_z_stratum_count"] == z_stratum_count
        and evidence["all_trajectories_have_3d_motion"]
        and evidence["motion_direction_octants_occupied"] >= min(4, count)
        and vertical_direction_signs == {-1, 1}
        and (authority is None or all(
            actor["uniform_3d_sampling_evidence"]["canonical_path_clear"]
            for actor in actors
        ))
    )
    if not evidence["passed"]:
        raise ValueError("uniform_3d actor layout evidence did not pass")
    clearance = {
        "checked": authority is not None,
        "relocated_actor_count": 0,
        "route_contract_actor_count": 0,
        "route_contract_preserved_actor_count": 0,
    }
    return actors, evidence, clearance


def build_actor_scenario(
    scene_name, scene, actor_mode, actor_count=None, vertical_span=2.0,
    actor_layout="hybrid", actor_seed=8801,
):
    count = resolve_actor_count(actor_mode, actor_count)
    vertical_span = float(vertical_span)
    if not 0.0 <= vertical_span <= 6.0:
        raise ValueError("--actor-vertical-span must be within [0, 6] m")
    if actor_layout not in {
        "corridor", "map_wide", "hybrid", "route_encounters", "uniform_3d",
    }:
        raise ValueError("unsupported actor layout")
    if (
        actor_layout in {"map_wide", "hybrid", "uniform_3d"}
        and "actor_xy_bounds" not in scene
    ):
        raise ValueError("map-wide actor layout requires scene actor_xy_bounds")
    if actor_layout == "route_encounters":
        if actor_mode != "multi_target":
            raise ValueError(
                "route_encounters requires --actors multi_target"
            )
        if count < 8:
            raise ValueError(
                "route_encounters requires --actor-count at least 8"
            )
    start, goal, direction, perpendicular, point = _vector_geometry(scene)
    rng = np.random.default_rng(int(actor_seed))
    actors = []
    z_bounds = scene.get("actor_z_bounds", [0.8, 14.0])
    z_min, z_max = map(float, z_bounds)
    if not z_min < z_max:
        raise ValueError("scene actor_z_bounds must be increasing")

    def ping_pong(
        actor_id, shape, radius, first, second, speed, height=None,
        start_time=0.0, encounter_family=None, route_fraction=None,
    ):
        actor = {
            "id": actor_id,
            "enabled": True,
            "shape": shape,
            "radius": radius,
            "initial_position_world": first,
            "trajectory": {
                "type": "waypoint_ping_pong",
                "waypoints_world": [first, second],
                "speed": speed,
                "start_time": float(start_time),
                "end_time": 3600.0,
            },
        }
        if height is not None:
            actor["height"] = height
        if encounter_family is not None:
            anchor = point(float(route_fraction))
            actor.update({
                "encounter_family": str(encounter_family),
                "route_fraction": float(route_fraction),
                "route_encounter_contract": {
                    "contract_version": "route_encounter_actor_v2",
                    "anchor_world": anchor,
                    "route_start_world": list(start),
                    "route_goal_world": list(goal),
                    "nominal_uav_speed_mps": 3.0,
                    "maximum_anchor_distance_m": 3.5,
                    "maximum_route_distance_m": 1.25,
                    "maximum_nominal_encounter_distance_m": 2.5,
                    "minimum_first_appearance_longitudinal_m": 1.0,
                    "maximum_first_appearance_bearing_deg": 80.0,
                    "maximum_crossing_direction_cosine": 0.25,
                    "maximum_head_on_direction_cosine": -0.8,
                    "minimum_same_direction_cosine": 0.8,
                    "maximum_same_direction_speed_mps": 0.65,
                    "first_appearance": (
                        "active_at_start_time_inside_route_aligned_forward_cone"
                    ),
                },
            })
        actors.append(actor)

    def bounded_z(value):
        return min(z_max, max(z_min, float(value)))

    if actor_layout == "uniform_3d":
        actors, uniform_evidence, clearance = _build_uniform_3d_actor_layout(
            scene, count, actor_seed, vertical_span
        )
        scenario = {
            "enabled": bool(actors),
            "scenario_id": (
                f"dep_interactive_{scene_name}_{actor_mode}_{actor_layout}_"
                f"{count}_actors"
            ),
            "seed": int(actor_seed),
            "requested_actor_count": count,
            "actor_layout": actor_layout,
            "corridor_actor_count": 0,
            "map_wide_actor_count": count,
            "vertical_span_m": vertical_span,
            "canonical_occupancy_checked": clearance["checked"],
            "relocated_actor_count": clearance["relocated_actor_count"],
            "uniform_3d_contract_version": "uniform_3d_layout_v1",
            "uniform_3d_layout_contract_version": "uniform_3d_layout_v1",
            "xy_strata_shape": uniform_evidence["xy_grid_shape"],
            "xy_strata_occupied": uniform_evidence[
                "occupied_xy_cell_count"
            ],
            "z_strata_count": uniform_evidence["z_stratum_count"],
            "z_strata_occupied": uniform_evidence[
                "occupied_z_stratum_count"
            ],
            "uniform_3d_z_bounds": uniform_evidence[
                "uniform_3d_z_bounds"
            ],
            "z_bounds_source": uniform_evidence["z_bounds_source"],
            "motion_direction_octants_occupied": uniform_evidence[
                "motion_direction_octants_occupied"
            ],
            "vertical_direction_signs": uniform_evidence[
                "vertical_direction_signs"
            ],
            "route_proximity_actor_count": uniform_evidence[
                "route_proximity_actor_count"
            ],
            "route_proximity_threshold_m": uniform_evidence[
                "route_proximity_threshold_m"
            ],
            "route_encounter_guaranteed": False,
            "route_encounter_actor_count": 0,
            "planner_ground_truth_exposed": False,
            "uniform_3d_layout_evidence": uniform_evidence,
            "actors": actors,
        }
        return {"dynamic_scenario": scenario}

    if actor_layout == "corridor":
        corridor_count = count
    elif actor_layout == "map_wide":
        corridor_count = 0
    elif actor_layout == "route_encounters":
        corridor_count = count
    else:
        corridor_count = min(count, max(1, int(math.ceil(count * 0.3))))
    map_wide_count = count - corridor_count
    map_centers = []
    if map_wide_count:
        (x_min, x_max), (y_min, y_max) = scene["actor_xy_bounds"]
        columns = int(math.ceil(math.sqrt(map_wide_count)))
        rows = int(math.ceil(map_wide_count / columns))
        cells = [(row, column) for row in range(rows) for column in range(columns)]
        rng.shuffle(cells)
        for row, column in cells[:map_wide_count]:
            x = x_min + (column + rng.uniform(0.2, 0.8)) * (
                (x_max - x_min) / columns
            )
            y = y_min + (row + rng.uniform(0.2, 0.8)) * (
                (y_max - y_min) / rows
            )
            map_centers.append((float(x), float(y)))

    for index in range(count):
        fraction = 0.18 + 0.64 * (index + 1) / (count + 1)
        centered_layer = ((index % 5) - 2) / 2.0
        layer_offset = centered_layer * vertical_span * 0.5
        vertical_motion = (
            vertical_span * 0.25 * (1.0 if index % 2 == 0 else -1.0)
        )
        base_z = point(fraction)[2] + layer_offset
        speed = 0.65 + 0.12 * (index % 5)
        if actor_layout == "route_encounters":
            encounter_family = (
                "crossing", "head_on", "same_direction_slow",
                "staggered_crossing",
            )[index % 4]
            family_ordinal = index // 4
            route_z = point(fraction)[2]
            route_vertical_motion = min(0.4, vertical_span * 0.2)
            route_vertical_motion *= 1.0 if family_ordinal % 2 == 0 else -1.0
            if encounter_family in {"crossing", "staggered_crossing"}:
                lateral_extent = 2.6 + 0.3 * (family_ordinal % 3)
                first = point(
                    fraction, -lateral_extent,
                    z=bounded_z(route_z - route_vertical_motion),
                )
                second = point(
                    fraction, lateral_extent,
                    z=bounded_z(route_z + route_vertical_motion),
                )
                if family_ordinal % 2:
                    first, second = second, first
                ping_pong(
                    101 + index, "sphere", 0.38 + 0.03 * (index % 3),
                    first, second, 0.75 + 0.08 * (family_ordinal % 3),
                    start_time=(
                        1.5 + 1.25 * family_ordinal
                        if encounter_family == "staggered_crossing" else 0.0
                    ),
                    encounter_family=encounter_family,
                    route_fraction=fraction,
                )
            elif encounter_family == "head_on":
                lane_offset = 0.35 if family_ordinal % 2 == 0 else -0.35
                first = point(
                    min(0.9, fraction + 0.12), lane_offset,
                    z=bounded_z(route_z + route_vertical_motion),
                )
                second = point(
                    max(0.1, fraction - 0.10), lane_offset,
                    z=bounded_z(route_z - route_vertical_motion),
                )
                ping_pong(
                    101 + index, "vertical_cylinder",
                    0.38 + 0.03 * (index % 3), first, second,
                    0.8 + 0.08 * (family_ordinal % 3),
                    height=1.5 + 0.15 * (family_ordinal % 3),
                    encounter_family=encounter_family,
                    route_fraction=fraction,
                )
            else:
                lane_offset = -0.45 if family_ordinal % 2 == 0 else 0.45
                first = point(
                    max(0.1, fraction - 0.08), lane_offset,
                    z=bounded_z(route_z - route_vertical_motion),
                )
                second = point(
                    min(0.9, fraction + 0.16), lane_offset,
                    z=bounded_z(route_z + route_vertical_motion),
                )
                ping_pong(
                    101 + index, "sphere", 0.40,
                    first, second, 0.42 + 0.04 * (family_ordinal % 3),
                    encounter_family=encounter_family,
                    route_fraction=fraction,
                )
            continue
        selected_mode = (
            ("crossing" if index % 2 == 0 else "head_on")
            if actor_mode == "multi_target" else actor_mode
        )
        if index >= corridor_count:
            center_x, center_y = map_centers[index - corridor_count]
            angle = float(rng.uniform(-math.pi, math.pi))
            extent = float(rng.uniform(2.0, 4.5))
            center_z = bounded_z(
                rng.uniform(
                    z_min + min(0.8, 0.2 * (z_max - z_min)),
                    z_max - min(0.8, 0.2 * (z_max - z_min)),
                )
            )
            dz = vertical_motion
            first = [
                center_x - extent * math.cos(angle),
                center_y - extent * math.sin(angle),
                bounded_z(center_z - dz),
            ]
            second = [
                center_x + extent * math.cos(angle),
                center_y + extent * math.sin(angle),
                bounded_z(center_z + dz),
            ]
            if index % 2:
                first, second = second, first
            if selected_mode == "crossing":
                ping_pong(
                    101 + index, "sphere", 0.38 + 0.04 * (index % 4),
                    first, second, speed,
                )
            else:
                ping_pong(
                    101 + index, "vertical_cylinder",
                    0.38 + 0.03 * (index % 3), first, second, speed,
                    height=1.5 + 0.15 * (index % 4),
                )
            continue
        if selected_mode == "crossing":
            lateral_extent = 3.0 + 0.5 * (index % 4)
            first = point(
                fraction, -lateral_extent,
                z=bounded_z(base_z - vertical_motion),
            )
            second = point(
                fraction, lateral_extent,
                z=bounded_z(base_z + vertical_motion),
            )
            if index % 2:
                first, second = second, first
            ping_pong(
                101 + index, "sphere", 0.38 + 0.04 * (index % 4),
                first, second, speed,
            )
        elif selected_mode == "head_on":
            lane_offset = ((index % 5) - 2) * 0.7
            if "authority_root" in scene:
                # A long straight actor trajectory is not physically viable
                # between dense trunks. Keep a local head-on encounter instead.
                first_fraction = min(0.88, fraction + 0.08)
                second_fraction = max(0.12, fraction - 0.08)
            else:
                first_fraction = min(0.88, fraction + 0.22)
                second_fraction = max(0.12, fraction - 0.28)
            first = point(
                first_fraction, lane_offset,
                z=bounded_z(base_z + vertical_motion),
            )
            second = point(
                second_fraction, lane_offset,
                z=bounded_z(base_z - vertical_motion),
            )
            ping_pong(
                101 + index, "vertical_cylinder",
                0.38 + 0.03 * (index % 3), first, second, speed,
                height=1.5 + 0.15 * (index % 4),
            )
    clearance = enforce_canonical_actor_clearance(
        scene, actors, direction, perpendicular
    )
    encounter_family_counts = {}
    for actor in actors:
        family = actor.get("encounter_family")
        if family is not None:
            encounter_family_counts[family] = (
                encounter_family_counts.get(family, 0) + 1
            )
    scenario = {
        "enabled": bool(actors),
        "scenario_id": (
            f"dep_interactive_{scene_name}_{actor_mode}_{actor_layout}_"
            f"{count}_actors"
        ),
        "seed": int(actor_seed),
        "requested_actor_count": count,
        "actor_layout": actor_layout,
        "corridor_actor_count": corridor_count,
        "map_wide_actor_count": map_wide_count,
        "vertical_span_m": vertical_span,
        "canonical_occupancy_checked": clearance["checked"],
        "relocated_actor_count": clearance["relocated_actor_count"],
        "actors": actors,
    }
    if actor_layout == "route_encounters":
        scenario.update({
            "route_encounter_contract_version": "route_encounter_layout_v2",
            "route_encounter_evidence_version": "route_encounter_evidence_v1",
            "nominal_uav_route_speed_mps": 3.0,
            "first_appearance_definition": (
                "actor active at trajectory.start_time, relative to a nominal "
                "3mps UAV whose camera is aligned with start-to-goal; actor "
                "must be at least 1m ahead and within an 80deg forward cone"
            ),
            "route_encounter_actor_count": sum(
                encounter_family_counts.values()
            ),
            "encounter_family_counts": encounter_family_counts,
            "route_contract_actor_count": clearance[
                "route_contract_actor_count"
            ],
            "route_contract_preserved_actor_count": clearance[
                "route_contract_preserved_actor_count"
            ],
        })
    return {"dynamic_scenario": scenario}


def checkpoint_preflight(checkpoint):
    from policy.checkpoint_utils import (
        detect_checkpoint_variant,
        detect_head_variant,
        load_checkpoint_payload,
        unpack_checkpoint,
        load_dep_checkpoint,
    )
    from policy.dep_network import DepNetwork

    payload = load_checkpoint_payload(checkpoint)
    state, metadata = unpack_checkpoint(payload)
    backbone_variant = detect_checkpoint_variant(payload)
    head_variant = detect_head_variant(payload)
    model = DepNetwork(
        backbone_variant=backbone_variant, head_variant=head_variant
    )
    load_result = load_dep_checkpoint(model, checkpoint, backbone_variant)
    result = {
        "checkpoint": str(checkpoint),
        "backbone_variant": backbone_variant,
        "head_variant": head_variant,
        "state_tensors": len(state),
        "metadata": metadata,
        "strict_load": load_result["strict"],
    }
    validation_metric = metadata.get("validation_metric")
    identities = metadata.get("identities") or {}
    training_contract = identities.get("training_contract_version")
    if training_contract in UNGATED_STATIC_TRAINING_CONTRACTS:
        result["training_gate_qualified"] = None
        result["checkpoint_readiness"] = (
            "validation_loss_selected_pending_closed_loop"
        )
    else:
        result["training_gate_qualified"] = not (
            validation_metric is not None
            and float(validation_metric) >= 1_000_000.0
        )
        result["checkpoint_readiness"] = (
            "legacy_training_gate_passed"
            if result["training_gate_qualified"] else "legacy_training_gate_failed"
        )
    return result


def wait_for_port(port, process, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"roscore exited with status {process.returncode}")
        with socket.socket() as connection:
            connection.settimeout(0.2)
            if connection.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"ROS master did not open port {port}")


def wait_for_topics(environment, processes, timeout=60.0):
    required = {
        "/sim/odom",
        "/depth_image",
        "/mock_map",
        "/dynamic_objects/markers",
        "/so3_control/pos_cmd",
        "/dep_demo/any_collision",
    }
    deadline = time.monotonic() + timeout
    observed = set()
    while time.monotonic() < deadline:
        for label, process in processes:
            status = process.poll()
            if status is not None:
                raise RuntimeError(f"{label} exited during ROS readiness with status {status}")
        result = subprocess.run(
            ["/opt/ros/noetic/bin/rostopic", "list"],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            observed = set(result.stdout.splitlines())
            if required <= observed:
                return sorted(required)
        time.sleep(0.25)
    missing = sorted(required - observed)
    raise TimeoutError(f"ROS runtime readiness timed out; missing topics: {missing}")


def publish_fixed_ab_goal(environment, goal):
    """Publish the one versioned suggested goal used by an A/B rollout."""
    x, y, z = (float(value) for value in goal)
    message = json.dumps({
        "header": {"frame_id": "world"},
        "pose": {
            "position": {"x": x, "y": y, "z": z},
            "orientation": {"w": 1.0},
        },
    })
    result = subprocess.run(
        [
            "/opt/ros/noetic/bin/rostopic", "pub", "-1",
            "/move_base_simple/goal", "geometry_msgs/PoseStamped", message,
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15.0,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to publish fixed A/B goal: "
            + (result.stderr.strip() or result.stdout.strip())
        )


def shell_command(command):
    controller_python = CONTROLLER / "devel" / "lib" / "python3" / "dist-packages"
    simulator_python = SIMULATOR / "devel" / "lib" / "python3" / "dist-packages"
    setup = (
        "source /opt/ros/noetic/setup.bash && "
        f"source {SIMULATOR}/devel/setup.bash && "
        f"export ROS_PACKAGE_PATH={quoted(CONTROLLER / 'src')}:{quoted(SIMULATOR / 'src')}:"
        '"$ROS_PACKAGE_PATH" && '
        f"export CMAKE_PREFIX_PATH={quoted(CONTROLLER / 'devel')}:"
        f"{quoted(SIMULATOR / 'devel')}:\"$CMAKE_PREFIX_PATH\" && "
        f"export LD_LIBRARY_PATH={quoted(CONTROLLER / 'devel' / 'lib')}:"
        f"{quoted(SIMULATOR / 'devel' / 'lib')}:\"$LD_LIBRARY_PATH\" && "
        f"export PYTHONPATH={quoted(controller_python)}:{quoted(simulator_python)}:"
        '"$PYTHONPATH" && '
    )
    return ["/bin/bash", "-lc", setup + "exec " + command]


def quoted(value):
    return shlex.quote(str(value))


def _request_shutdown(_signum, _frame):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGTERM, _request_shutdown)
    args = parse_args()
    if not (0.5 <= args.arrival_radius <= 10.0):
        raise ValueError("--arrival-radius must be within [0.5, 10.0] m")
    scene = load_scene(args.scenes_config, args.scene)
    actor_scenario = build_actor_scenario(
        args.scene, scene, args.actors, args.actor_count,
        args.actor_vertical_span, args.actor_layout, args.actor_seed,
    )
    actor_count = len(actor_scenario["dynamic_scenario"]["actors"])
    if args.dynamic_mode == "dynamic_attention" and actor_count > 16:
        raise ValueError(
            "dynamic_attention currently supports at most 16 tracked obstacles; "
            "use static_reactive or reduce --actor-count"
        )
    actor_contract = actor_scenario["dynamic_scenario"]
    checkpoint = resolve_checkpoint(args)
    preflight = checkpoint_preflight(checkpoint)
    summary = {
        "status": "PREFLIGHT_PASS",
        "scene": args.scene,
        "map_uuid": scene["map_uuid"],
        "pointcloud": str(scene["pointcloud"]),
        "start": scene["start"],
        "suggested_goal": scene["suggested_goal"],
        "flight_bounds_min": scene.get("flight_bounds_min"),
        "flight_bounds_max": scene.get("flight_bounds_max"),
        "goal_z": scene["suggested_goal"][2],
        "arrival_radius": args.arrival_radius,
        "goal_mode": args.goal_mode,
        "midflight_goal_replacement": args.goal_mode == "interactive",
        "align_goal_before_planning": bool(args.align_goal_before_planning),
        "actors": args.actors,
        "actor_count": actor_count,
        "actor_vertical_span": args.actor_vertical_span,
        "actor_layout": args.actor_layout,
        "actor_seed": args.actor_seed,
        "actor_z_bounds": scene.get("actor_z_bounds"),
        "canonical_occupancy_checked": actor_contract[
            "canonical_occupancy_checked"
        ],
        "relocated_actor_count": actor_contract["relocated_actor_count"],
        "route_encounter_actor_count": actor_contract.get(
            "route_encounter_actor_count", 0
        ),
        "route_encounter_family_counts": actor_contract.get(
            "encounter_family_counts", {}
        ),
        "route_encounter_contract_version": actor_contract.get(
            "route_encounter_contract_version"
        ),
        "route_encounter_evidence_version": actor_contract.get(
            "route_encounter_evidence_version"
        ),
        "route_first_appearance_definition": actor_contract.get(
            "first_appearance_definition"
        ),
        "route_contract_preserved_actor_count": actor_contract.get(
            "route_contract_preserved_actor_count", 0
        ),
        "uniform_3d_contract_version": actor_contract.get(
            "uniform_3d_contract_version"
        ),
        "uniform_3d_z_bounds": actor_contract.get("uniform_3d_z_bounds"),
        "z_bounds_source": actor_contract.get("z_bounds_source"),
        "xy_strata_shape": actor_contract.get("xy_strata_shape"),
        "xy_strata_occupied": actor_contract.get("xy_strata_occupied"),
        "z_strata_count": actor_contract.get("z_strata_count"),
        "z_strata_occupied": actor_contract.get("z_strata_occupied"),
        "motion_direction_octants_occupied": actor_contract.get(
            "motion_direction_octants_occupied"
        ),
        "vertical_direction_signs": actor_contract.get(
            "vertical_direction_signs"
        ),
        "route_proximity_actor_count": actor_contract.get(
            "route_proximity_actor_count"
        ),
        "route_proximity_threshold_m": actor_contract.get(
            "route_proximity_threshold_m"
        ),
        "route_encounter_guaranteed": actor_contract.get(
            "route_encounter_guaranteed"
        ),
        "planner_ground_truth_exposed": actor_contract.get(
            "planner_ground_truth_exposed"
        ),
        "dynamic_mode": args.dynamic_mode,
        "runtime_safety_enabled": bool(args.runtime_safety),
        "runtime_profile": args.runtime_profile,
        "runtime_behavior_version": {
            V44_RUNTIME_PROFILE: V44_RUNTIME_BEHAVIOR_VERSION,
            V45_RUNTIME_PROFILE: V45_RUNTIME_BEHAVIOR_VERSION,
            V457_RUNTIME_PROFILE: V457_RUNTIME_BEHAVIOR_VERSION,
            V4510_RUNTIME_PROFILE: V4510_RUNTIME_BEHAVIOR_VERSION,
            V47_RUNTIME_PROFILE: V47_RUNTIME_BEHAVIOR_VERSION,
            V48_RUNTIME_PROFILE: V48_RUNTIME_BEHAVIOR_VERSION,
            V481_RUNTIME_PROFILE: V481_RUNTIME_BEHAVIOR_VERSION,
            V482_RUNTIME_PROFILE: V482_RUNTIME_BEHAVIOR_VERSION,
            V485_RUNTIME_PROFILE: V485_RUNTIME_BEHAVIOR_VERSION,
            V49_RUNTIME_PROFILE: V49_RUNTIME_BEHAVIOR_VERSION,
        }.get(args.runtime_profile, args.runtime_profile),
        "dynamic_foreground_mode": args.dynamic_foreground_mode,
        "planning_speed_mps": args.planning_speed,
        "deadlock_recovery_enabled": bool(args.deadlock_recovery),
        "deadlock_recovery_profile": args.deadlock_recovery_profile,
        **preflight,
    }
    print(json.dumps(summary, indent=2, default=str))
    if preflight["training_gate_qualified"] is False:
        print(
            "WARNING: checkpoint never passed its training selection Gate; "
            "runtime safety projection remains mandatory and this checkpoint "
            "must not be treated as production-qualified.",
            flush=True,
        )
    if args.preflight_only:
        return

    for required in (
        YOPO_PYTHON,
        CONTROLLER / "devel" / "setup.bash",
        SIMULATOR / "devel" / "setup.bash",
        ROOT / "dep_interactive_demo.rviz",
    ):
        if not required.exists():
            raise FileNotFoundError(f"Required runtime asset is missing: {required}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runtime = ROOT / "runs" / "dep_interactive_demo" / f"{stamp}-{args.scene}"
    logs = runtime / "logs"
    logs.mkdir(parents=True, exist_ok=False)
    scenario_path = runtime / "dynamic_scenario.yaml"
    scenario_path.write_text(
        json.dumps(actor_scenario, indent=2),
        encoding="utf-8",
    )
    (runtime / "manifest.json").write_text(
        json.dumps({**summary, "runtime": str(runtime)}, indent=2, default=str),
        encoding="utf-8",
    )

    ros_uri = f"http://127.0.0.1:{args.ros_master_port}"
    environment = dict(os.environ, ROS_MASTER_URI=ros_uri, ROS_HOSTNAME="127.0.0.1")
    processes = []
    log_handles = []

    def start(label, command, visible=False):
        output = None
        if not visible:
            handle = (logs / f"{label}.log").open("w", encoding="utf-8")
            log_handles.append(handle)
            output = handle
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        processes.append((label, process))
        return process

    def stop_all():
        for _, process in reversed(processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 5.0
        for _, process in reversed(processes):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for handle in log_handles:
            handle.close()

    try:
        master = start(
            "roscore", ["/opt/ros/noetic/bin/roscore", "-p", str(args.ros_master_port)]
        )
        wait_for_port(args.ros_master_port, master)
        start(
            "controller",
            shell_command(
                f"roslaunch {quoted(ROOT / 'dep_interactive_controller.launch')} "
                f"init_x:={scene['start'][0]} init_y:={scene['start'][1]} "
                f"init_z:={scene['start'][2]}"
            ),
        )
        start(
            "simulator",
            shell_command(
                "rosrun sensor_simulator sensor_simulator_cuda "
                f"_random_map:=false _ply_file:={quoted(scene['pointcloud'])} "
                f"_dynamic_scenario_file:={quoted(scenario_path)} "
                "_render_depth:=true _render_lidar:=false"
            ),
        )
        start(
            "collision_monitor",
            shell_command(
                f"{quoted(YOPO_PYTHON)} "
                f"{quoted(ROOT / 'tools' / 'monitor_dep_interactive_collisions.py')} "
                f"--authority-root {quoted(scene['authority_root'])} "
                f"--map-uuid {quoted(scene['map_uuid'])} "
                f"--report {quoted(runtime / 'collision_report.json')} "
                f"--goal-z {scene['suggested_goal'][2]} "
                f"--arrival-radius {args.arrival_radius}"
            ),
            visible=True,
        )
        dynamic_enabled = int(args.dynamic_mode in {
            "dynamic_safety", "dynamic_attention"
        })
        dynamic_network_attention = int(
            args.dynamic_mode == "dynamic_attention"
        )
        flight_bounds_args = ""
        if "flight_bounds_min" in scene:
            flight_bounds_args = (
                "--flight-bounds-min "
                + " ".join(map(str, scene["flight_bounds_min"]))
                + " --flight-bounds-max "
                + " ".join(map(str, scene["flight_bounds_max"]))
                + " "
            )
        start(
            "planner",
            shell_command(
                f"{quoted(YOPO_PYTHON)} {quoted(ROOT / 'test_dep_ros.py')} "
                f"--checkpoint {quoted(checkpoint)} "
                f"--backbone-variant {preflight['backbone_variant']} "
                f"--head-variant {preflight['head_variant']} "
                f"--goal-z {scene['suggested_goal'][2]} "
                f"--arrival-radius {args.arrival_radius} "
                "--wait-for-goal 1 --hold-on-arrival 1 "
                f"--goal-policy {args.goal_mode.replace('-', '_')} "
                f"--align-goal-before-planning "
                f"{args.align_goal_before_planning} "
                f"--dynamic-enabled {dynamic_enabled} "
                f"--dynamic-foreground-mode "
                f"{args.dynamic_foreground_mode} "
                f"--dynamic-network-attention-enabled "
                f"{dynamic_network_attention} "
                f"--runtime-safety-enabled {args.runtime_safety} "
                f"--runtime-profile {args.runtime_profile} "
                + (
                    "" if args.planning_speed is None
                    else f"--planning-speed {args.planning_speed} "
                )
                +
                f"--deadlock-recovery-enabled {args.deadlock_recovery} "
                f"--deadlock-recovery-profile "
                f"{args.deadlock_recovery_profile} "
                + flight_bounds_args
                +
                f"--safety-telemetry {quoted(runtime / 'safety_decisions.jsonl')}"
            ),
            visible=True,
        )
        ready_topics = wait_for_topics(environment, processes)
        print(json.dumps({
            "status": "RUNTIME_READY",
            "topics": ready_topics,
            "ros_master_uri": ros_uri,
        }, indent=2))
        if args.goal_mode == "fixed-ab":
            publish_fixed_ab_goal(environment, scene["suggested_goal"])
            print(json.dumps({
                "status": "FIXED_AB_GOAL_PUBLISHED",
                "goal": scene["suggested_goal"],
                "additional_goals_allowed": False,
            }, indent=2))
        if not args.no_rviz:
            start(
                "rviz",
                shell_command(f"rviz -d {quoted(ROOT / 'dep_interactive_demo.rviz')}"),
                visible=True,
            )
        if args.goal_mode == "interactive":
            goal_instructions = (
                "In RViz choose '2D Nav Goal'; you may replace the goal at "
                "any time during flight.\n"
            )
        else:
            goal_instructions = (
                "Fixed A/B mode published the suggested goal automatically; "
                "later RViz goals are intentionally rejected.\n"
            )
        print(
            "\nInteractive demo is running.\n"
            + goal_instructions
            + f"Suggested XY goal: ({scene['suggested_goal'][0]:.2f}, "
            f"{scene['suggested_goal'][1]:.2f}); fixed Z={scene['suggested_goal'][2]:.2f} m.\n"
            f"Logs and manifest: {runtime}\n"
            "Press Ctrl-C to stop every process cleanly."
        )
        while True:
            for label, process in processes:
                status = process.poll()
                if status is not None:
                    raise RuntimeError(
                        f"{label} exited with status {status}; inspect {logs / (label + '.log')}"
                    )
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping interactive demo...")
    finally:
        stop_all()


if __name__ == "__main__":
    main()
