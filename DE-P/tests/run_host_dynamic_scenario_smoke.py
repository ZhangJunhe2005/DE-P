#!/usr/bin/env python3
"""Collect and validate one finite Simulator dynamic scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading

import message_filters
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from sensor_simulator.msg import DynamicObjectStateArray

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.dynamic.camera_model import depth_to_pointcloud
from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.ros_bridge import camera_model_from_info, camera_pose_from_odometry
from policy.dynamic.types import DynamicPerceptionConfig


class Collector:
    def __init__(self, target_frames: int, scenario_type: str):
        self.target_frames = target_frames
        self.scenario_type = scenario_type
        self.samples = []
        self.event = threading.Event()
        sources = [
            message_filters.Subscriber("/depth_image", Image),
            message_filters.Subscriber("/camera_info", CameraInfo),
            message_filters.Subscriber("/sim/odom", Odometry),
            message_filters.Subscriber("/dynamic_objects/ground_truth", DynamicObjectStateArray),
        ]
        self.sync = message_filters.ApproximateTimeSynchronizer(
            sources, queue_size=80, slop=0.002, allow_headerless=False
        )
        self.sync.registerCallback(self.callback)

    def callback(self, depth, camera_info, odometry, ground_truth):
        if len(self.samples) >= self.target_frames:
            return
        self.samples.append((depth, camera_info, odometry, ground_truth))
        if len(self.samples) >= self.target_frames:
            self.event.set()

    def run(self, timeout: float):
        if not self.event.wait(timeout):
            raise RuntimeError(
                f"timed out: received {len(self.samples)}/{self.target_frames} synchronized frames"
            )
        return self.evaluate()

    def evaluate(self):
        samples = self.samples[: self.target_frames]
        stamps = []
        all_ids = set()
        per_actor = {}
        visible_frames = 0
        collision_frames = 0
        active_counts = []
        sync_errors = []
        depth_errors = []
        scenario_ids = set()
        seeds = set()
        config = DynamicPerceptionConfig.from_global_config()
        perception = DynamicPerception(config, feature_shape=(3, 5))
        max_dynamic_tracks = 0
        max_confirmed_tracks = 0

        for sample_index, (depth_msg, info, odom, gt) in enumerate(samples):
            messages = (depth_msg, info, odom, gt)
            times = np.asarray([message.header.stamp.to_sec() for message in messages])
            sync_errors.append(float(times.max() - times.min()))
            stamps.append(times[0])
            scenario_ids.add(gt.scenario_id)
            seeds.add(int(gt.seed))
            active = [obj for obj in gt.objects if obj.active]
            active_counts.append(len(active))
            visible_frames += int(any(obj.visible for obj in active))
            collision_frames += int(gt.uav_collision)
            for obj in gt.objects:
                all_ids.add(int(obj.object_id))
                per_actor.setdefault(int(obj.object_id), []).append((
                    times[0], bool(obj.active),
                    np.asarray([obj.position_world.x, obj.position_world.y, obj.position_world.z]),
                    np.asarray([obj.velocity_world.x, obj.velocity_world.y, obj.velocity_world.z]),
                ))
                if obj.active and obj.visible:
                    if obj.rendered_pixel_count <= 0:
                        raise RuntimeError("visible actor lacks image support")
                    if obj.inside_image:
                        depth_errors.append(float(obj.depth_error))

            if depth_msg.encoding != "32FC1" or depth_msg.step < depth_msg.width * 4:
                raise RuntimeError(
                    f"expected packed/padded 32FC1 depth, got {depth_msg.encoding!r} step={depth_msg.step}"
                )
            dtype = np.dtype(">f4" if depth_msg.is_bigendian else "<f4")
            raw = np.ndarray(
                (depth_msg.height, depth_msg.width), dtype=dtype,
                buffer=depth_msg.data, strides=(depth_msg.step, 4),
            ).astype(np.float32, copy=True)
            if raw.shape != (90, 160) or raw.dtype != np.float32 or not np.isfinite(raw).all():
                raise RuntimeError(f"invalid raw depth: shape={raw.shape}, dtype={raw.dtype}")
            # Processing every third sample bounds smoke runtime while retaining temporal motion.
            if sample_index % 3 == 0:
                camera = camera_model_from_info(info, config)
                points = depth_to_pointcloud(raw, camera, stride=config.depth_stride)
                pose = camera_pose_from_odometry(odom, config)
                result = perception.update(points, pose, times[0], camera)
                max_dynamic_tracks = max(max_dynamic_tracks, len(result.dynamic_tracks))
                max_confirmed_tracks = max(max_confirmed_tracks, len(result.confirmed_tracks))

        if not np.all(np.diff(stamps) > 0):
            raise RuntimeError("depth timestamps are not strictly increasing")
        velocity_errors = []
        for observations in per_actor.values():
            for previous, current in zip(observations, observations[1:]):
                t0, active0, p0, v0 = previous
                t1, active1, p1, v1 = current
                dt = t1 - t0
                # A ping-pong waypoint has continuous position and an
                # intentional instantaneous velocity reversal.  Validate the
                # constant-velocity intervals on either side, not the single
                # interval containing that discontinuity.
                if active0 and active1 and dt > 0 and np.linalg.norm(v1 - v0) < 0.1:
                    velocity_errors.append(float(np.linalg.norm((p1 - p0) / dt - v0)))

        no_target = self.scenario_type == "no_target"
        checks = {
            "single_scenario_id": len(scenario_ids) == 1,
            "single_seed": len(seeds) == 1,
            "strict_time": bool(np.all(np.diff(stamps) > 0)),
            "sync_below_2ms": max(sync_errors, default=0.0) <= 0.002,
            "no_target_empty_gt": (not no_target) or max(active_counts, default=0) == 0,
            "dynamic_has_active_gt": no_target or max(active_counts, default=0) > 0,
            "dynamic_visible_in_depth": no_target or visible_frames > 0,
            "stable_object_ids": all(len({entry[0] for entry in values}) == len(values)
                                     for values in per_actor.values()),
            "velocity_consistency": no_target or max(velocity_errors, default=0.0) < 0.08,
            "projected_depth_error": no_target or max(depth_errors, default=999.0) < 0.15,
            "no_target_false_dynamic_zero": (not no_target) or max_dynamic_tracks == 0,
        }
        failures = [name for name, passed in checks.items() if not passed]
        return {
            "status": "PASS" if not failures else "FAIL",
            "scenario_type": self.scenario_type,
            "scenario_ids": sorted(scenario_ids),
            "seeds": sorted(seeds),
            "frames": len(samples),
            "object_ids": sorted(all_ids),
            "maximum_active_objects": max(active_counts, default=0),
            "visible_depth_frames": visible_frames,
            "collision_risk_frames": collision_frames,
            "sync_error_max_seconds": max(sync_errors, default=0.0),
            "projected_depth_error_max_m": max(depth_errors, default=0.0),
            "velocity_error_max_mps": max(velocity_errors, default=0.0),
            "dynamic_perception": {
                "max_confirmed_tracks": max_confirmed_tracks,
                "max_dynamic_tracks": max_dynamic_tracks,
                "false_dynamics": max_dynamic_tracks if no_target else 0,
            },
            "checks": checks,
            "failures": failures,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-type", required=True,
                        choices=("no_target", "crossing", "head_on", "multi_target"))
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    rospy.init_node(f"phase7_{args.scenario_type}_collector", anonymous=True)
    result = Collector(args.frames, args.scenario_type).run(args.timeout)
    print("HOST_DYNAMIC_SCENARIO_SAMPLE_RESULT")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
