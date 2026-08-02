#!/usr/bin/env python3
"""Publish and record authoritative collisions for an interactive DE-P run."""

from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import rospy
from nav_msgs.msg import Odometry
from sensor_simulator.msg import DynamicObjectStateArray
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker

from run_dep_interactive_demo import CanonicalOccupancy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--map-uuid", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--uav-radius", type=float, default=0.3)
    parser.add_argument("--odom-topic", default="/sim/odom")
    parser.add_argument(
        "--dynamic-ground-truth-topic",
        default="/dynamic_objects/ground_truth",
    )
    return parser.parse_args()


class CollisionMonitor:
    def __init__(self, args):
        self.args = args
        self.authority = CanonicalOccupancy(
            args.authority_root, args.map_uuid
        )
        self.lock = threading.Lock()
        self.static_collision = False
        self.dynamic_collision = False
        self.dynamic_actor_ids = []
        self.last_position = None
        self.static_collision_events = 0
        self.dynamic_collision_events = 0
        self.any_collision_events = 0
        self.odom_samples = 0
        self.dynamic_samples = 0
        self.first_collision = None
        self.started_at = datetime.now(timezone.utc)

        self.static_pub = rospy.Publisher(
            "/dep_demo/static_collision", Bool, queue_size=1, latch=True
        )
        self.dynamic_pub = rospy.Publisher(
            "/dep_demo/dynamic_collision", Bool, queue_size=1, latch=True
        )
        self.any_pub = rospy.Publisher(
            "/dep_demo/any_collision", Bool, queue_size=1, latch=True
        )
        self.event_pub = rospy.Publisher(
            "/dep_demo/collision_event", String, queue_size=10, latch=True
        )
        self.marker_pub = rospy.Publisher(
            "/dep_demo/collision_marker", Marker, queue_size=1
        )
        self.previous_static = False
        self.previous_dynamic = False
        self.previous_any = False
        rospy.Subscriber(
            args.odom_topic, Odometry, self.on_odom,
            queue_size=1, tcp_nodelay=True,
        )
        rospy.Subscriber(
            args.dynamic_ground_truth_topic,
            DynamicObjectStateArray,
            self.on_dynamic,
            queue_size=1,
            tcp_nodelay=True,
        )
        rospy.on_shutdown(self.write_report)

    @staticmethod
    def now_iso():
        return datetime.now(timezone.utc).isoformat()

    def publish_state(self, stamp):
        # Subscriber callbacks may already be queued when ROS closes latched
        # publishers during Ctrl-C shutdown. The final report is written by the
        # shutdown hook; publishing at that point is both unnecessary and can
        # raise "publish() to a closed topic".
        if rospy.is_shutdown():
            return
        any_collision = self.static_collision or self.dynamic_collision
        self.static_pub.publish(Bool(data=self.static_collision))
        self.dynamic_pub.publish(Bool(data=self.dynamic_collision))
        self.any_pub.publish(Bool(data=any_collision))

        static_rising = self.static_collision and not self.previous_static
        dynamic_rising = self.dynamic_collision and not self.previous_dynamic
        any_rising = any_collision and not self.previous_any
        self.static_collision_events += int(static_rising)
        self.dynamic_collision_events += int(dynamic_rising)
        self.any_collision_events += int(any_rising)
        if any_rising:
            event = {
                "event": "COLLISION",
                "timestamp": self.now_iso(),
                "static_collision": self.static_collision,
                "dynamic_collision": self.dynamic_collision,
                "actor_ids": self.dynamic_actor_ids,
                "position_world": self.last_position,
            }
            if self.first_collision is None:
                self.first_collision = event
            encoded = json.dumps(event, separators=(",", ":"))
            self.event_pub.publish(String(data=encoded))
            rospy.logerr("DE-P COLLISION %s", encoded)
        if any_collision and self.last_position is not None:
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = "world"
            marker.ns = "dep_collision"
            marker.id = 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = self.last_position[0]
            marker.pose.position.y = self.last_position[1]
            marker.pose.position.z = self.last_position[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.9
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.95
            marker.lifetime = rospy.Duration(0.5)
            self.marker_pub.publish(marker)
        self.previous_static = self.static_collision
        self.previous_dynamic = self.dynamic_collision
        self.previous_any = any_collision

    def on_odom(self, message):
        position = message.pose.pose.position
        center = [position.x, position.y, position.z]
        collision = self.authority.sphere_collides(
            center, self.args.uav_radius
        )
        with self.lock:
            self.odom_samples += 1
            self.last_position = center
            self.static_collision = collision
            self.publish_state(message.header.stamp)

    def on_dynamic(self, message):
        colliding = [
            int(actor.object_id) for actor in message.objects
            if actor.active and actor.collision
        ]
        with self.lock:
            self.dynamic_samples += 1
            self.dynamic_collision = bool(message.uav_collision or colliding)
            self.dynamic_actor_ids = colliding
            self.publish_state(message.header.stamp)

    def write_report(self):
        with self.lock:
            report = {
                "status": "COLLISION_DETECTED" if self.first_collision else "NO_COLLISION",
                "map_uuid": self.args.map_uuid,
                "authority_root": str(self.args.authority_root.resolve()),
                "uav_radius_m": self.args.uav_radius,
                "started_at": self.started_at.isoformat(),
                "finished_at": self.now_iso(),
                "odom_samples": self.odom_samples,
                "dynamic_samples": self.dynamic_samples,
                "static_collision_events": self.static_collision_events,
                "dynamic_collision_events": self.dynamic_collision_events,
                "any_collision_events": self.any_collision_events,
                "first_collision": self.first_collision,
            }
            self.args.report.parent.mkdir(parents=True, exist_ok=True)
            self.args.report.write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            print("COLLISION_MONITOR_RESULT")
            print(json.dumps(report, indent=2))


def main():
    args = parse_args()
    if args.uav_radius <= 0:
        raise ValueError("--uav-radius must be positive")
    rospy.init_node("dep_interactive_collision_monitor", anonymous=False)
    CollisionMonitor(args)
    rospy.loginfo(
        "DE-P collision monitor ready: static canonical occupancy + actor GT"
    )
    rospy.spin()


if __name__ == "__main__":
    main()
