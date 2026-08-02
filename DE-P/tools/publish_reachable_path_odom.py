#!/usr/bin/env python3
"""Publish deterministic odometry along a reachability-validated map path."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def sample_polyline(points, distance):
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    distance = float(np.clip(distance, 0.0, cumulative[-1]))
    index = min(int(np.searchsorted(cumulative, distance, side="right") - 1), len(segments) - 1)
    fraction = (distance - cumulative[index]) / max(segments[index], 1e-9)
    position = points[index] * (1.0 - fraction) + points[index + 1] * fraction
    direction = points[index + 1] - points[index]
    direction /= max(np.linalg.norm(direction), 1e-9)
    return position, direction, cumulative[-1]


def main():
    import rospy
    from geometry_msgs.msg import Point, Quaternion
    from nav_msgs.msg import Odometry
    from tf.transformations import quaternion_from_euler

    parser = argparse.ArgumentParser()
    parser.add_argument("--reachability", type=Path, required=True)
    parser.add_argument("--map-id", type=int, required=True)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--rate", type=float, default=30.0)
    args = parser.parse_args()
    payload = json.loads(args.reachability.read_text(encoding="utf-8"))
    entry = next(item for item in payload["maps"] if int(item["map_id"]) == args.map_id)
    points = np.asarray(entry["path_waypoints_world"], dtype=float)
    if points.shape[0] < 2 or args.speed <= 0 or args.rate <= 0:
        raise ValueError("validated path, positive speed and rate are required")
    total_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    rospy.init_node("reachable_path_odom", anonymous=True)
    publisher = rospy.Publisher("/sim/odom", Odometry, queue_size=10)
    rate = rospy.Rate(args.rate)
    start = rospy.Time.now()
    while not rospy.is_shutdown():
        stamp = rospy.Time.now()
        elapsed = (stamp - start).to_sec()
        position, direction, length = sample_polyline(
            points, min(elapsed * args.speed, total_length)
        )
        yaw = math.atan2(direction[1], direction[0])
        horizontal = math.hypot(direction[0], direction[1])
        pitch = math.atan2(-direction[2], max(horizontal, 1e-9))
        quaternion = quaternion_from_euler(0.0, pitch, yaw)
        message = Odometry()
        message.header.stamp = stamp
        message.header.frame_id = "world"
        message.child_frame_id = "quadrotor"
        message.pose.pose.position = Point(*position)
        message.pose.pose.orientation = Quaternion(*quaternion)
        moving = elapsed * args.speed < length
        velocity = direction * args.speed if moving else np.zeros(3)
        message.twist.twist.linear.x = float(velocity[0])
        message.twist.twist.linear.y = float(velocity[1])
        message.twist.twist.linear.z = float(velocity[2])
        publisher.publish(message)
        rate.sleep()


if __name__ == "__main__":
    main()
