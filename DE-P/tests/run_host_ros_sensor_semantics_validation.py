#!/usr/bin/env python3
"""Validate live Simulator topic/frame semantics against deterministic odometry."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time

import message_filters
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from scipy.spatial import cKDTree
import tf2_ros

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.camera_model import depth_to_pointcloud
from policy.dynamic.ros_bridge import (
    body_points_to_optical,
    body_points_to_world,
    camera_model_from_info,
    camera_pose_from_odometry,
)
from policy.dynamic.types import DynamicPerceptionConfig


def cloud_array(message):
    return np.asarray(list(point_cloud2.read_points(
        message, field_names=("x", "y", "z"), skip_nans=True
    )), dtype=np.float32).reshape(-1, 3)


class Validation:
    def __init__(self):
        self.config = DynamicPerceptionConfig.from_global_config()
        self.lock = threading.Lock()
        self.sensor_event = threading.Event()
        self.depth_event = threading.Event()
        self.samples = []
        self.depth_sample = None
        self.depth_samples = []
        self.legacy = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.publisher = rospy.Publisher(self.config.odom_topic, Odometry, queue_size=20)
        self.legacy_sub = rospy.Subscriber("/lidar_points", PointCloud2, self.legacy_callback)
        inputs = [
            message_filters.Subscriber(self.config.pointcloud_body_topic, PointCloud2),
            message_filters.Subscriber(self.config.pointcloud_optical_topic, PointCloud2),
            message_filters.Subscriber(self.config.pointcloud_world_topic, PointCloud2),
            message_filters.Subscriber(self.config.odom_topic, Odometry),
        ]
        self.cloud_sync = message_filters.ApproximateTimeSynchronizer(
            inputs, queue_size=30, slop=0.01
        )
        self.cloud_sync.registerCallback(self.cloud_callback)
        depth_inputs = [
            message_filters.Subscriber(self.config.depth_topic, Image),
            message_filters.Subscriber(self.config.camera_info_topic, CameraInfo),
            message_filters.Subscriber(self.config.odom_topic, Odometry),
        ]
        self.depth_sync = message_filters.ApproximateTimeSynchronizer(
            depth_inputs, queue_size=10, slop=0.001
        )
        self.depth_sync.registerCallback(self.depth_callback)

    def legacy_callback(self, message):
        self.legacy = message

    def cloud_callback(self, body, optical, world, odometry):
        with self.lock:
            self.samples.append((body, optical, world, odometry))
            if len(self.samples) >= 12:
                self.sensor_event.set()

    def depth_callback(self, depth, info, odometry):
        self.depth_sample = (depth, info)
        with self.lock:
            self.depth_samples.append((depth, info, odometry))
            if len(self.depth_samples) >= 12:
                self.depth_event.set()

    def publish_motion(self, duration=5.0):
        rate = rospy.Rate(30)
        start = time.monotonic()
        while not rospy.is_shutdown() and time.monotonic() - start < duration:
            elapsed = time.monotonic() - start
            message = Odometry()
            message.header.stamp = rospy.Time.now()
            message.header.frame_id = self.config.world_frame_id
            message.child_frame_id = "/" + self.config.body_frame_id
            message.pose.pose.position.x = 0.05 * elapsed
            message.pose.pose.position.y = 0.0
            message.pose.pose.position.z = 2.0
            message.pose.pose.orientation.w = 1.0
            message.twist.twist.linear.x = 0.05
            self.publisher.publish(message)
            rate.sleep()

    def run(self):
        motion = threading.Thread(target=self.publish_motion, daemon=True)
        motion.start()
        if not self.depth_event.wait(15.0) or not self.sensor_event.wait(15.0):
            raise RuntimeError("timed out waiting for synchronized Simulator topics")
        motion.join(timeout=6.0)
        with self.lock:
            samples = self.samples[:12]
            depth_samples = self.depth_samples[:12]
        if self.legacy is None:
            raise RuntimeError("legacy /lidar_points was not published")

        body_msg, optical_msg, world_msg, odom_msg = samples[-1]
        body = cloud_array(body_msg)
        optical = cloud_array(optical_msg)
        world = cloud_array(world_msg)
        expected_optical = body_points_to_optical(body, self.config)
        expected_optical = expected_optical[expected_optical[:, 2] > 0]
        optical_error = cKDTree(optical).query(expected_optical, k=1)[0]
        expected_world = body_points_to_world(body, odom_msg, self.config)
        world_error = cKDTree(world).query(expected_world, k=1)[0]

        depth, camera_info = self.depth_sample
        if depth.header.stamp != camera_info.header.stamp:
            raise RuntimeError("depth and CameraInfo timestamps differ")
        camera = camera_model_from_info(camera_info, self.config)

        transform = self.tf_buffer.lookup_transform(
            self.config.world_frame_id, self.config.camera_frame_id,
            rospy.Time(0), rospy.Duration(2.0)
        )

        pointcloud_perception = DynamicPerception(self.config, feature_shape=(3, 5))
        pointcloud_max_dynamic_tracks = 0
        pointcloud_maximum_speed = 0.0
        for _, optical_message, _, odometry in samples:
            points = cloud_array(optical_message)
            pose = camera_pose_from_odometry(odometry, self.config)
            timestamp = optical_message.header.stamp.to_sec()
            result = pointcloud_perception.update(points, pose, timestamp, camera)
            pointcloud_max_dynamic_tracks = max(
                pointcloud_max_dynamic_tracks, len(result.dynamic_tracks)
            )
            for track in result.confirmed_tracks:
                pointcloud_maximum_speed = max(
                    pointcloud_maximum_speed, float(np.linalg.norm(track.velocity_world))
                )

        depth_perception = DynamicPerception(self.config, feature_shape=(3, 5))
        depth_max_dynamic_tracks = 0
        depth_maximum_speed = 0.0
        for depth_message, info_message, odometry in depth_samples:
            raw_depth = np.ndarray(
                shape=(depth_message.height, depth_message.width), dtype=np.float32,
                buffer=depth_message.data,
                strides=(depth_message.step, 4),
            ).copy()
            depth_camera = camera_model_from_info(info_message, self.config)
            points = depth_to_pointcloud(
                raw_depth, depth_camera, stride=self.config.depth_stride
            )
            pose = camera_pose_from_odometry(odometry, self.config)
            timestamp = depth_message.header.stamp.to_sec()
            result = depth_perception.update(points, pose, timestamp, depth_camera)
            depth_max_dynamic_tracks = max(depth_max_dynamic_tracks, len(result.dynamic_tracks))
            for track in result.confirmed_tracks:
                depth_maximum_speed = max(
                    depth_maximum_speed, float(np.linalg.norm(track.velocity_world))
                )

        first_world = cloud_array(samples[0][2])
        last_world = cloud_array(samples[-1][2])
        stability = cKDTree(first_world).query(last_world, k=1)[0]
        errors = []
        checks = {
            "legacy_mismatch_retained": self.legacy.header.frame_id == "odom",
            "body_frame": body_msg.header.frame_id == self.config.body_frame_id,
            "optical_frame": optical_msg.header.frame_id == self.config.camera_frame_id,
            "world_frame": world_msg.header.frame_id == self.config.world_frame_id,
            "optical_z_positive": bool(len(optical) and np.all(optical[:, 2] > 0)),
            "body_optical_median_error_below_1mm": float(np.median(optical_error)) < 1e-3,
            "body_world_median_error_below_1mm": float(np.median(world_error)) < 1e-3,
            "world_static_median_drift_below_0_25m": float(np.median(stability)) < 0.25,
            "camera_info_geometry": (camera.width, camera.height, camera.fx, camera.fy,
                                     camera.cx, camera.cy) == (160, 90, 80.0, 80.0, 80.0, 45.0),
            "tf_chain": transform.child_frame_id == self.config.camera_frame_id,
            "depth_ego_motion_no_dynamic_tracks": depth_max_dynamic_tracks == 0,
        }
        errors.extend(name for name, passed in checks.items() if not passed)
        report = {
            "status": "PASS" if not errors else "FAIL",
            "world_frame": self.config.world_frame_id,
            "body_frame": self.config.body_frame_id,
            "camera_optical_frame": self.config.camera_frame_id,
            "depth": {"frame": depth.header.frame_id, "width": depth.width, "height": depth.height},
            "camera_info": {"frame": camera_info.header.frame_id, "width": camera.width,
                            "height": camera.height, "K": list(camera_info.K)},
            "pointcloud_topics": {
                "legacy": {"topic": "/lidar_points", "frame": self.legacy.header.frame_id},
                "body": {"topic": self.config.pointcloud_body_topic, "frame": body_msg.header.frame_id,
                         "points": len(body)},
                "optical": {"topic": self.config.pointcloud_optical_topic,
                            "frame": optical_msg.header.frame_id, "points": len(optical)},
                "world": {"topic": self.config.pointcloud_world_topic,
                          "frame": world_msg.header.frame_id, "points": len(world)},
            },
            "tf_chain": f"{self.config.world_frame_id}->{self.config.body_frame_id}->{self.config.camera_frame_id}",
            "static_scene_velocity": {
                "depth": {"max_confirmed_track_speed": depth_maximum_speed,
                          "max_dynamic_track_count": depth_max_dynamic_tracks},
                "pointcloud": {"max_confirmed_track_speed": pointcloud_maximum_speed,
                               "max_dynamic_track_count": pointcloud_max_dynamic_tracks},
            },
            "official_dynamic_sensor_source": "depth",
            "pointcloud_training_allowed": pointcloud_max_dynamic_tracks == 0,
            "geometry_errors": {
                "body_optical_median_m": float(np.median(optical_error)),
                "body_world_median_m": float(np.median(world_error)),
                "world_static_median_drift_m": float(np.median(stability)),
            },
            "checks": checks,
            "errors": errors,
        }
        print("HOST_ROS_SENSOR_SEMANTICS_RESULT")
        print(json.dumps(report, indent=2))
        if errors:
            raise RuntimeError("sensor semantics checks failed: " + ", ".join(errors))


if __name__ == "__main__":
    rospy.init_node("dep_sensor_semantics_validation", anonymous=True)
    Validation().run()
