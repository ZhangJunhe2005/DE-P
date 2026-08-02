#!/usr/bin/env python3
"""Finite live ROS collector; instance IDs are evaluator-only."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import threading
import time

import message_filters
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from sensor_simulator.msg import DynamicObjectStateArray

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import cfg
from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.instance_evaluation import InstanceEvaluationState
from policy.dynamic.ros_bridge import camera_model_from_info, camera_pose_from_odometry
from policy.dynamic.types import DynamicPerceptionConfig
from tools.record_dynamic_sequences import raw_depth, raw_instance


class Collector:
    def __init__(self, frames, scenario):
        self.frames, self.scenario = frames, scenario
        self.records, self.error = [], None
        self.last_stamp = None
        self.event = threading.Event()
        self.perception = DynamicPerception(
            replace(DynamicPerceptionConfig.from_global_config(),
                    foreground_mode="range_image_hybrid"),
            (cfg["vertical_num"], cfg["horizon_num"]),
        )
        sources = [
            message_filters.Subscriber("/depth_image", Image),
            message_filters.Subscriber("/camera_info", CameraInfo),
            message_filters.Subscriber("/sim/odom", Odometry),
            message_filters.Subscriber("/dynamic_objects/ground_truth",
                                       DynamicObjectStateArray),
            message_filters.Subscriber("/dynamic_actor_instance", Image),
        ]
        self.sync = message_filters.ApproximateTimeSynchronizer(
            sources, queue_size=80, slop=0.002, allow_headerless=False
        )
        self.sync.registerCallback(self.callback)

    def callback(self, depth_msg, info, odom, gt, instance_msg):
        if self.event.is_set():
            return
        try:
            stamp = depth_msg.header.stamp.to_sec()
            if self.last_stamp is not None and stamp - self.last_stamp < 0.095:
                return
            depth = raw_depth(depth_msg)
            started = time.perf_counter()
            # Deliberately pass only depth/camera/pose to runtime.
            result = self.perception.update_depth(
                depth,
                camera_pose_from_odometry(odom, self.perception.config),
                stamp,
                camera_model_from_info(info, self.perception.config),
            )
            latency = (time.perf_counter() - started) * 1000.0
            instance = raw_instance(instance_msg)
            if depth_msg.header != instance_msg.header:
                raise ValueError("depth/instance header mismatch")
            objects = [{
                "object_id": int(obj.object_id),
                "position_world": [
                    obj.position_world.x, obj.position_world.y, obj.position_world.z
                ],
                "velocity_world": [
                    obj.velocity_world.x, obj.velocity_world.y, obj.velocity_world.z
                ],
                "radius": float(obj.radius), "active": bool(obj.active),
                "visible": bool(obj.visible),
                "rendered_pixel_count": int(obj.rendered_pixel_count),
            } for obj in gt.objects]
            self.records.append((objects, instance, result, latency))
            self.last_stamp = stamp
            if len(self.records) >= self.frames:
                self.event.set()
        except Exception as error:
            self.error = error
            self.event.set()

    def run(self, timeout):
        if not self.event.wait(timeout):
            raise TimeoutError(f"received {len(self.records)}/{self.frames} frames")
        if self.error:
            raise self.error
        ever_visible = {
            int(value) for _objects, instance, _result, _latency in self.records
            for value in np.unique(instance) if int(value)
        }
        all_actor_ids = {
            int(obj["object_id"]) for objects, _instance, _result, _latency
            in self.records for obj in objects
        }
        never = all_actor_ids - ever_visible
        state = InstanceEvaluationState()
        dynamic_frames = 0
        for frame, (objects, instance, result, _latency) in enumerate(self.records):
            state.update(self.scenario, frame, objects, instance, result, never)
            dynamic_frames += int(bool(result.dynamic_tracks))
        summary = state.summary()
        failures = []
        for name in (
            "never_observed_illegal_observation_count",
            "never_observed_ungrounded_track_count",
            "never_observed_attention_event_count",
            "prediction_only_confirmed_track_count",
            "prediction_only_attention_count",
        ):
            if summary[name]:
                failures.append(f"{name}={summary[name]}")
        if self.scenario in {"no_target", "always_outside_fov",
                             "static_wall_occluded"} and dynamic_frames:
            failures.append(f"unexpected_dynamic_frames={dynamic_frames}")
        if self.scenario in {"crossing", "occluded_but_tracked"} and not dynamic_frames:
            failures.append("visible_dynamic_actor_was_never_detected")
        return {
            "status": "PASS" if not failures else "FAIL",
            "scenario": self.scenario, "frames": len(self.records),
            "visible_actor_ids": sorted(ever_visible),
            "never_observed_actor_ids": sorted(never),
            "dynamic_track_frames": dynamic_frames,
            "perception_latency_ms": {
                "mean": float(np.mean([x[3] for x in self.records])),
                "p95": float(np.percentile([x[3] for x in self.records], 95)),
            },
            "instance_runtime_input": False,
            "instance_mask_depth_alignment_error": 0,
            "never_observed_proximity_event_count":
                summary["never_observed_proximity_event_count"],
            "never_observed_attention_event_count":
                summary["never_observed_attention_event_count"],
            "prediction_only_attention_count":
                summary["prediction_only_attention_count"],
            "failures": failures,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    rospy.init_node(f"phase8g_live_{args.scenario}", anonymous=True)
    result = Collector(args.frames, args.scenario).run(args.timeout)
    print("PHASE8G_LIVE_COLLECTOR_RESULT")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
