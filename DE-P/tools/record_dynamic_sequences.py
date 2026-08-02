#!/usr/bin/env python3
"""Finite, synchronized ROS recorder for formal depth dynamic sequences."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

import message_filters
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from sensor_simulator.msg import DynamicObjectStateArray
from ruamel.yaml import YAML


FRAME_FIELDS = [
    "sequence_id", "frame_index", "timestamp", "depth_path", "pointcloud_path",
    "camera_x", "camera_y", "camera_z", "camera_qx", "camera_qy", "camera_qz",
    "camera_qw", "velocity_x", "velocity_y", "velocity_z", "acceleration_x",
    "acceleration_y", "acceleration_z", "goal_x", "goal_y", "goal_z", "map_id",
    "dynamic_objects_path", "scenario_id", "seed", "depth_odom_offset",
    "depth_camera_info_offset", "depth_gt_offset",
]
INSTANCE_FIELDS = ["instance_path", "depth_instance_offset"]


def git_revision(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def dump_yaml(path, data):
    yaml = YAML()
    yaml.default_flow_style = False
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.dump(data, stream)


def raw_depth(message):
    if message.encoding != "32FC1" or message.step < message.width * 4:
        raise ValueError(f"formal recorder requires 32FC1, got {message.encoding}")
    dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
    return np.ndarray(
        (message.height, message.width), dtype=dtype, buffer=message.data,
        strides=(message.step, 4),
    ).astype(np.float32, copy=True)


def raw_instance(message):
    if message.encoding != "32SC1" or message.step < message.width * 4:
        raise ValueError(f"instance recorder requires 32SC1, got {message.encoding}")
    dtype = np.dtype(">i4" if message.is_bigendian else "<i4")
    return np.ndarray(
        (message.height, message.width), dtype=dtype, buffer=message.data,
        strides=(message.step, 4),
    ).astype(np.int32, copy=True)


class Recorder:
    def __init__(self, args):
        self.args = args
        self.directory = args.output / "sequences" / args.sequence_id
        if self.directory.exists():
            raise FileExistsError(f"refusing to overwrite sequence: {self.directory}")
        (self.directory / "depth").mkdir(parents=True)
        (self.directory / "dynamic_objects").mkdir()
        if args.instance_topic:
            (self.directory / "instance_ids").mkdir()
        self.rows = []
        self.sync_offsets = []
        self.previous_velocity = None
        self.previous_timestamp = None
        self.last_recorded_timestamp = None
        self.event = threading.Event()
        self.lock = threading.Lock()
        self.camera_info = None
        self.scenario_id = None
        self.seed = None
        self.error = None
        self.write_metadata("incomplete")
        sources = [
            message_filters.Subscriber(args.depth_topic, Image),
            message_filters.Subscriber(args.camera_info_topic, CameraInfo),
            message_filters.Subscriber(args.odom_topic, Odometry),
            message_filters.Subscriber(args.gt_topic, DynamicObjectStateArray),
        ]
        if args.instance_topic:
            sources.append(message_filters.Subscriber(args.instance_topic, Image))
        self.sync = message_filters.ApproximateTimeSynchronizer(
            sources, queue_size=80, slop=args.sync_slop, allow_headerless=False
        )
        self.sync.registerCallback(self.callback)

    def write_metadata(self, status):
        info = self.camera_info
        configuration = self.args.scenario_file.read_bytes()
        offsets = np.asarray(self.sync_offsets, dtype=float)
        metadata = {
            "dataset_version": "dep_dynamic_sequence_v1",
            "sequence_id": self.args.sequence_id,
            "sensor_source": "depth",
            "source_topic": self.args.depth_topic,
            "depth_encoding": "32FC1",
            "depth_scale": 1.0,
            "raw_image_width": int(info.width) if info else 160,
            "raw_image_height": int(info.height) if info else 90,
            "network_image_width": 160,
            "network_image_height": 96,
            "camera_intrinsics": {
                "fx": float(info.K[0]) if info else 80.0,
                "fy": float(info.K[4]) if info else 80.0,
                "cx": float(info.K[2]) if info else 80.0,
                "cy": float(info.K[5]) if info else 45.0,
                "min_depth": 0.1, "max_depth": 20.0,
            },
            "camera_position_body": [0.0, 0.0, 0.0],
            "camera_rotation_body_from_camera": [
                [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]
            ],
            "world_frame": "world", "body_frame": "quadrotor",
            "camera_optical_frame": "camera_optical", "pointcloud_frame": "none",
            "simulator_commit": git_revision("/home/zjh/YOPO/Simulator"),
            "dep_commit": git_revision(Path(__file__).resolve().parents[1]),
            "configuration_hash": hashlib.sha256(configuration).hexdigest(),
            "random_seed": int(self.seed or self.args.seed),
            "time_unit": "second", "distance_unit": "meter",
            "acceleration_method": "finite_difference_odometry_twist",
            "completion_status": status,
            "scenario_id": self.scenario_id or self.args.scenario_id,
            "scenario_type": self.args.scenario_type,
            "frame_count": len(self.rows),
            "record_rate_hz": self.args.rate,
            "sync_slop_seconds": self.args.sync_slop,
            "sync_error_max_seconds": float(offsets.max()) if offsets.size else 0.0,
            "sync_error_mean_seconds": float(offsets.mean()) if offsets.size else 0.0,
            "static_map_sha256": self.args.static_map_sha256,
        }
        if self.args.instance_topic:
            metadata.update({
                "dataset_version": "dep_dynamic_instance_sequence_v2",
                "instance_topic": self.args.instance_topic,
                "instance_encoding": "32SC1",
                "instance_runtime_input": False,
            })
        dump_yaml(self.directory / "metadata.yaml", metadata)

    def callback(self, depth, info, odom, gt, instance_message=None):
        with self.lock:
            if self.event.is_set():
                return
            try:
                timestamp = depth.header.stamp.to_sec()
                if self.last_recorded_timestamp is not None:
                    if timestamp <= self.last_recorded_timestamp:
                        raise ValueError("timestamps must be strictly increasing")
                    if timestamp - self.last_recorded_timestamp < 1.0 / self.args.rate - 1e-4:
                        return
                times = [info.header.stamp.to_sec(), odom.header.stamp.to_sec(),
                         gt.header.stamp.to_sec()]
                if instance_message is not None:
                    times.append(instance_message.header.stamp.to_sec())
                offsets = [abs(timestamp - value) for value in times]
                if max(offsets) > self.args.sync_slop:
                    return
                array = raw_depth(depth)
                if array.shape != (90, 160) or not np.isfinite(array).all():
                    raise ValueError("raw depth must be finite 160x90")
                frame_index = len(self.rows)
                depth_path = Path("depth") / f"{frame_index:06d}.npy"
                object_path = Path("dynamic_objects") / f"{frame_index:06d}.json"
                np.save(self.directory / depth_path, array, allow_pickle=False)
                instance_path = ""
                if instance_message is not None:
                    instance = raw_instance(instance_message)
                    if instance.shape != array.shape:
                        raise ValueError("instance image must exactly match raw depth shape")
                    instance_path = Path("instance_ids") / f"{frame_index:06d}.npy"
                    np.save(self.directory / instance_path, instance, allow_pickle=False)
                objects = []
                for obj in gt.objects:
                    objects.append({
                        "object_id": int(obj.object_id),
                        "position_world": [obj.position_world.x, obj.position_world.y, obj.position_world.z],
                        "velocity_world": [obj.velocity_world.x, obj.velocity_world.y, obj.velocity_world.z],
                        "position_covariance": [[1e-4, 0, 0], [0, 1e-4, 0], [0, 0, 1e-4]],
                        "radius": float(obj.radius), "type": obj.object_type,
                        "visibility": 1.0 if obj.visible else 0.0,
                        "occluded": bool(obj.occluded), "dynamic": True,
                        "active": bool(obj.active), "height": float(obj.height),
                        "collision": bool(obj.collision), "inside_image": bool(obj.inside_image),
                        "projected_u": float(obj.projected_u), "projected_v": float(obj.projected_v),
                        "expected_surface_depth": float(obj.expected_surface_depth),
                        "observed_depth": float(obj.observed_depth),
                        "depth_error": float(obj.depth_error),
                        "rendered_pixel_count": int(obj.rendered_pixel_count),
                    })
                (self.directory / object_path).write_text(
                    json.dumps(objects, indent=2), encoding="utf-8"
                )
                velocity = np.asarray([
                    odom.twist.twist.linear.x, odom.twist.twist.linear.y,
                    odom.twist.twist.linear.z,
                ], dtype=float)
                if self.previous_velocity is None:
                    acceleration = np.zeros(3)
                else:
                    acceleration = (velocity - self.previous_velocity) / (timestamp - self.previous_timestamp)
                pose = odom.pose.pose
                row = {
                    "sequence_id": self.args.sequence_id, "frame_index": frame_index,
                    "timestamp": f"{timestamp:.9f}", "depth_path": str(depth_path),
                    "pointcloud_path": "", "camera_x": pose.position.x,
                    "camera_y": pose.position.y, "camera_z": pose.position.z,
                    "camera_qx": pose.orientation.x, "camera_qy": pose.orientation.y,
                    "camera_qz": pose.orientation.z, "camera_qw": pose.orientation.w,
                    "velocity_x": velocity[0], "velocity_y": velocity[1], "velocity_z": velocity[2],
                    "acceleration_x": acceleration[0], "acceleration_y": acceleration[1],
                    "acceleration_z": acceleration[2], "goal_x": self.args.goal[0],
                    "goal_y": self.args.goal[1], "goal_z": self.args.goal[2],
                    "map_id": self.args.map_id, "dynamic_objects_path": str(object_path),
                    "scenario_id": gt.scenario_id, "seed": int(gt.seed),
                    "depth_camera_info_offset": timestamp - times[0],
                    "depth_odom_offset": timestamp - times[1], "depth_gt_offset": timestamp - times[2],
                }
                if instance_message is not None:
                    row.update({
                        "instance_path": str(instance_path),
                        "depth_instance_offset": timestamp - times[3],
                    })
                self.rows.append(row)
                self.sync_offsets.extend(offsets)
                self.previous_velocity = velocity
                self.previous_timestamp = timestamp
                self.last_recorded_timestamp = timestamp
                self.camera_info = info
                self.scenario_id = gt.scenario_id
                self.seed = int(gt.seed)
                if len(self.rows) >= self.args.frames:
                    self.finish()
            except Exception as error:  # callback exceptions otherwise disappear into rospy logs
                self.error = error
                self.event.set()

    def finish(self):
        with (self.directory / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=FRAME_FIELDS + (
                    INSTANCE_FIELDS if self.args.instance_topic else []
                ),
            )
            writer.writeheader()
            writer.writerows(self.rows)
        self.write_metadata("complete")
        self.event.set()

    def run(self):
        if not self.event.wait(self.args.timeout):
            raise TimeoutError(f"recording timed out at {len(self.rows)}/{self.args.frames} frames")
        if self.error:
            raise self.error
        if len(self.rows) != self.args.frames:
            raise RuntimeError("recorder stopped before requested frame count")
        return {
            "status": "PASS", "sequence_id": self.args.sequence_id,
            "scenario_id": self.scenario_id, "seed": self.seed,
            "frames": len(self.rows), "sensor_source": "depth",
            "sync_error_max_seconds": max(self.sync_offsets, default=0.0),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--scenario-file", type=Path, required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--scenario-type", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--sync-slop", type=float, default=0.002)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--goal", type=float, nargs=3, default=[20.0, 2.0, 1.6])
    parser.add_argument("--map-id", type=int, default=700)
    parser.add_argument("--static-map-sha256", required=True)
    parser.add_argument("--depth-topic", default="/depth_image")
    parser.add_argument("--camera-info-topic", default="/camera_info")
    parser.add_argument("--odom-topic", default="/sim/odom")
    parser.add_argument("--gt-topic", default="/dynamic_objects/ground_truth")
    parser.add_argument("--instance-topic", default="")
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    args.scenario_file = args.scenario_file.expanduser().resolve()
    rospy.init_node(f"record_{args.sequence_id}", anonymous=True)
    result = Recorder(args).run()
    print("DYNAMIC_SEQUENCE_RECORD_RESULT")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
