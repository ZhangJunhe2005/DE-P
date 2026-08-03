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
DEFAULT_SCENES = ROOT / "configs" / "dep_interactive_demo_scenes_v4.json"
DEFAULT_RUN = (
    ROOT / "runs" / "phase8_mixed_static_yopo_v3_2_low_lr_adamw"
    / "20260730T074415Z-260761"
)
CONTROLLER = ROOT.parent / "Controller"
SIMULATOR = ROOT.parent / "Simulator"
YOPO_PYTHON = Path("/home/zjh/miniconda3/envs/yopo/bin/python")


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
        "--actor-layout", choices=("corridor", "map_wide", "hybrid"),
        default="hybrid",
        help="hybrid disperses actors map-wide while retaining some route encounters",
    )
    parser.add_argument(
        "--actor-seed", type=int, default=8801,
        help="reproducible random seed for map-wide actor placement",
    )
    parser.add_argument(
        "--actor-vertical-span", type=float, default=2.0,
        help="total vertical distribution/motion span in metres (0-6)",
    )
    parser.add_argument(
        "--dynamic-mode", choices=("static_reactive", "dynamic_attention"),
        default="static_reactive",
        help="static_reactive is the valid default for statically trained checkpoints",
    )
    parser.add_argument("--arrival-radius", type=float, default=5.0)
    parser.add_argument(
        "--runtime-safety", type=int, choices=(0, 1), default=1,
        help="enable the hard trajectory safety shield (default: 1)",
    )
    parser.add_argument(
        "--deadlock-recovery", type=int, choices=(0, 1), default=1,
        help="enable deterministic brake/scan/breadcrumb recovery (default: 1)",
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


def enforce_canonical_actor_clearance(scene, actors, direction, perpendicular):
    if not actors or "authority_root" not in scene:
        return {"checked": False, "relocated_actor_count": 0}
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
                        if authority.actor_path_is_clear(actor):
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
                if authority.actor_path_is_clear(actor):
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
    return {"checked": True, "relocated_actor_count": relocated}


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


def build_actor_scenario(
    scene_name, scene, actor_mode, actor_count=None, vertical_span=2.0,
    actor_layout="hybrid", actor_seed=8801,
):
    count = resolve_actor_count(actor_mode, actor_count)
    vertical_span = float(vertical_span)
    if not 0.0 <= vertical_span <= 6.0:
        raise ValueError("--actor-vertical-span must be within [0, 6] m")
    if actor_layout not in {"corridor", "map_wide", "hybrid"}:
        raise ValueError("unsupported actor layout")
    if actor_layout != "corridor" and "actor_xy_bounds" not in scene:
        raise ValueError("map-wide actor layout requires scene actor_xy_bounds")
    _, _, direction, perpendicular, point = _vector_geometry(scene)
    rng = np.random.default_rng(int(actor_seed))
    actors = []
    z_bounds = scene.get("actor_z_bounds", [0.8, 14.0])
    z_min, z_max = map(float, z_bounds)
    if not z_min < z_max:
        raise ValueError("scene actor_z_bounds must be increasing")

    def ping_pong(actor_id, shape, radius, first, second, speed, height=None):
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
                "start_time": 0.0,
                "end_time": 3600.0,
            },
        }
        if height is not None:
            actor["height"] = height
        actors.append(actor)

    def bounded_z(value):
        return min(z_max, max(z_min, float(value)))

    if actor_layout == "corridor":
        corridor_count = count
    elif actor_layout == "map_wide":
        corridor_count = 0
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
    return {
        "dynamic_scenario": {
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
    }


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
    model = DepNetwork(backbone_variant=backbone_variant)
    load_result = load_dep_checkpoint(model, checkpoint, backbone_variant)
    result = {
        "checkpoint": str(checkpoint),
        "backbone_variant": backbone_variant,
        "head_variant": detect_head_variant(payload),
        "state_tensors": len(state),
        "metadata": metadata,
        "strict_load": load_result["strict"],
    }
    validation_metric = metadata.get("validation_metric")
    result["training_gate_qualified"] = not (
        validation_metric is not None and float(validation_metric) >= 1_000_000.0
    )
    if result["head_variant"] != "unified":
        raise ValueError("Interactive ROS node currently requires the unified DEP head")
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
        "goal_z": scene["suggested_goal"][2],
        "arrival_radius": args.arrival_radius,
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
        "dynamic_mode": args.dynamic_mode,
        "runtime_safety_enabled": bool(args.runtime_safety),
        "deadlock_recovery_enabled": bool(args.deadlock_recovery),
        **preflight,
    }
    print(json.dumps(summary, indent=2, default=str))
    if not preflight["training_gate_qualified"]:
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
                f"--report {quoted(runtime / 'collision_report.json')}"
            ),
            visible=True,
        )
        dynamic_enabled = 1 if args.dynamic_mode == "dynamic_attention" else 0
        start(
            "planner",
            shell_command(
                f"{quoted(YOPO_PYTHON)} {quoted(ROOT / 'test_dep_ros.py')} "
                f"--checkpoint {quoted(checkpoint)} "
                f"--backbone-variant {preflight['backbone_variant']} "
                f"--goal-z {scene['suggested_goal'][2]} "
                f"--arrival-radius {args.arrival_radius} "
                "--wait-for-goal 1 --hold-on-arrival 1 "
                f"--dynamic-enabled {dynamic_enabled} "
                f"--runtime-safety-enabled {args.runtime_safety} "
                f"--deadlock-recovery-enabled {args.deadlock_recovery} "
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
        if not args.no_rviz:
            start(
                "rviz",
                shell_command(f"rviz -d {quoted(ROOT / 'dep_interactive_demo.rviz')}"),
                visible=True,
            )
        print(
            "\nInteractive demo is running.\n"
            "In RViz choose '2D Nav Goal', then click the map and drag for heading.\n"
            f"Suggested XY goal: ({scene['suggested_goal'][0]:.2f}, "
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
