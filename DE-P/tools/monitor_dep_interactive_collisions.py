#!/usr/bin/env python3
"""Publish and record authoritative collisions for an interactive DE-P run."""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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
    parser.add_argument(
        "--near-margin", type=float, default=0.15,
        help="extra radius used only for a clearly labelled near-contact warning",
    )
    parser.add_argument(
        "--active-alarm-period", type=float, default=1.0,
        help="seconds between repeated terminal alarms while contact remains active",
    )
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
        self.last_odom_stamp = None
        self.previous_dynamic_uav_position = None
        self.previous_actor_states = {}
        self.static_detection_source = None
        self.dynamic_detection_sources = []
        self.near_static = False
        self.near_dynamic = False
        self.static_collision_events = 0
        self.dynamic_collision_events = 0
        self.any_collision_events = 0
        self.swept_static_collision_events = 0
        self.independent_dynamic_collision_events = 0
        self.near_contact_events = 0
        self.odom_samples = 0
        self.dynamic_samples = 0
        self.first_collision = None
        self.events_path = args.report.with_name("collision_events.jsonl")
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.write_text("", encoding="utf-8")
        self.started_at = datetime.now(timezone.utc)
        self.last_active_alarm = -math.inf

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
        self.previous_near = False
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

    def record_event(self, event):
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, separators=(",", ":")) + "\n")

    @staticmethod
    def actor_collides(center, actor, uav_radius):
        actor_center = np.asarray(actor["position"], dtype=np.float64)
        offset = actor_center - np.asarray(center, dtype=np.float64)
        if actor["shape"] == "sphere":
            return float(np.linalg.norm(offset)) <= actor["radius"] + uav_radius
        radial_gap = max(0.0, float(np.linalg.norm(offset[:2])) - actor["radius"])
        vertical_gap = max(0.0, abs(float(offset[2])) - 0.5 * actor["height"])
        return math.hypot(radial_gap, vertical_gap) <= uav_radius

    @classmethod
    def swept_actor_collides(cls, uav_start, uav_end, actor_start, actor_end,
                             uav_radius, max_step=0.05):
        relative_distance = float(np.linalg.norm(
            (np.asarray(actor_end["position"]) - np.asarray(uav_end))
            - (np.asarray(actor_start["position"]) - np.asarray(uav_start))
        ))
        samples = max(1, int(math.ceil(relative_distance / max_step)))
        for amount in np.linspace(0.0, 1.0, samples + 1):
            uav = np.asarray(uav_start) + amount * (
                np.asarray(uav_end) - np.asarray(uav_start)
            )
            actor = dict(actor_end)
            actor["position"] = (
                np.asarray(actor_start["position"]) + amount * (
                    np.asarray(actor_end["position"])
                    - np.asarray(actor_start["position"])
                )
            )
            if cls.actor_collides(uav, actor, uav_radius):
                return True
        return False

    @staticmethod
    def actor_state(message):
        return {
            "shape": str(message.object_type),
            "position": np.asarray([
                message.position_world.x,
                message.position_world.y,
                message.position_world.z,
            ], dtype=np.float64),
            "radius": float(message.radius),
            "height": float(message.height),
        }

    def terminal_alarm(self, event, repeated=False):
        label = "COLLISION STILL ACTIVE" if repeated else "COLLISION DETECTED"
        encoded = json.dumps(event, separators=(",", ":"))
        banner = "!" * 88
        print(
            f"\n{banner}\nDE-P {label}\n{encoded}\n{banner}\n",
            file=sys.stderr, flush=True,
        )

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
        any_falling = not any_collision and self.previous_any
        near_collision = self.near_static or self.near_dynamic
        near_rising = near_collision and not self.previous_near
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
                "static_detection_source": self.static_detection_source,
                "dynamic_detection_sources": self.dynamic_detection_sources,
            }
            if self.first_collision is None:
                self.first_collision = event
            encoded = json.dumps(event, separators=(",", ":"))
            self.event_pub.publish(String(data=encoded))
            rospy.logerr("DE-P COLLISION %s", encoded)
            self.record_event(event)
            self.terminal_alarm(event)
            self.last_active_alarm = time.monotonic()
        elif any_collision and (
            time.monotonic() - self.last_active_alarm
            >= self.args.active_alarm_period
        ):
            active = {
                "event": "COLLISION_ACTIVE",
                "timestamp": self.now_iso(),
                "static_collision": self.static_collision,
                "dynamic_collision": self.dynamic_collision,
                "actor_ids": self.dynamic_actor_ids,
                "position_world": self.last_position,
            }
            self.terminal_alarm(active, repeated=True)
            self.last_active_alarm = time.monotonic()
        if any_falling:
            cleared = {
                "event": "COLLISION_CLEARED",
                "timestamp": self.now_iso(),
                "position_world": self.last_position,
            }
            self.record_event(cleared)
            print(
                "\n---------------- DE-P COLLISION CLEARED ----------------\n",
                file=sys.stderr, flush=True,
            )
        if near_rising and not any_collision:
            self.near_contact_events += 1
            warning = {
                "event": "NEAR_CONTACT",
                "timestamp": self.now_iso(),
                "near_margin_m": self.args.near_margin,
                "position_world": self.last_position,
            }
            self.record_event(warning)
            rospy.logwarn("DE-P NEAR CONTACT %s", json.dumps(warning))
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
            marker.lifetime = rospy.Duration(3.0)
            self.marker_pub.publish(marker)
        self.previous_static = self.static_collision
        self.previous_dynamic = self.dynamic_collision
        self.previous_any = any_collision
        self.previous_near = near_collision

    def on_odom(self, message):
        position = message.pose.pose.position
        center = [position.x, position.y, position.z]
        instantaneous = self.authority.sphere_collides(
            center, self.args.uav_radius
        )
        with self.lock:
            swept = False
            stamp = message.header.stamp.to_sec()
            if (
                self.last_position is not None
                and self.last_odom_stamp is not None
                and stamp > self.last_odom_stamp
                and stamp - self.last_odom_stamp <= 0.25
            ):
                swept = self.authority.swept_sphere_collides(
                    self.last_position, center, self.args.uav_radius
                )
            self.odom_samples += 1
            self.last_position = center
            self.last_odom_stamp = stamp
            self.static_collision = instantaneous or swept
            self.static_detection_source = (
                "instantaneous" if instantaneous
                else "swept_between_odometry_samples" if swept else None
            )
            self.near_static = (
                not self.static_collision
                and self.authority.sphere_collides(
                    center, self.args.uav_radius + self.args.near_margin
                )
            )
            if swept and not self.previous_static:
                self.swept_static_collision_events += 1
            self.publish_state(message.header.stamp)

    def on_dynamic(self, message):
        with self.lock:
            current_uav = None if self.last_position is None else np.asarray(
                self.last_position, dtype=np.float64
            )
            current_states = {
                int(actor.object_id): self.actor_state(actor)
                for actor in message.objects if actor.active
            }
            gt_ids = {
                int(actor.object_id) for actor in message.objects
                if actor.active and actor.collision
            }
            independent_ids = set()
            swept_ids = set()
            near_ids = set()
            if current_uav is not None:
                for actor_id, actor in current_states.items():
                    if self.actor_collides(
                        current_uav, actor, self.args.uav_radius
                    ):
                        independent_ids.add(actor_id)
                    elif self.actor_collides(
                        current_uav, actor,
                        self.args.uav_radius + self.args.near_margin,
                    ):
                        near_ids.add(actor_id)
                    previous = self.previous_actor_states.get(actor_id)
                    if (
                        previous is not None
                        and self.previous_dynamic_uav_position is not None
                        and self.swept_actor_collides(
                            self.previous_dynamic_uav_position, current_uav,
                            previous, actor, self.args.uav_radius,
                        )
                    ):
                        swept_ids.add(actor_id)
            colliding = sorted(gt_ids | independent_ids | swept_ids)
            self.dynamic_samples += 1
            self.dynamic_collision = bool(
                message.uav_collision or colliding
            )
            self.dynamic_actor_ids = colliding
            sources = []
            if message.uav_collision or gt_ids:
                sources.append("simulator_ground_truth")
            if independent_ids:
                sources.append("independent_instantaneous_geometry")
            if swept_ids:
                sources.append("swept_relative_motion")
            self.dynamic_detection_sources = sources
            self.near_dynamic = bool(near_ids) and not self.dynamic_collision
            if (independent_ids or swept_ids) and not self.previous_dynamic:
                self.independent_dynamic_collision_events += 1
            self.previous_actor_states = current_states
            self.previous_dynamic_uav_position = current_uav
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
                "swept_static_collision_events": self.swept_static_collision_events,
                "independent_dynamic_collision_events": (
                    self.independent_dynamic_collision_events
                ),
                "near_contact_events": self.near_contact_events,
                "events_jsonl": str(self.events_path),
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
    if args.near_margin < 0:
        raise ValueError("--near-margin must be non-negative")
    if args.active_alarm_period <= 0:
        raise ValueError("--active-alarm-period must be positive")
    rospy.init_node("dep_interactive_collision_monitor", anonymous=False)
    CollisionMonitor(args)
    rospy.loginfo(
        "DE-P collision monitor ready: static canonical occupancy + actor GT"
    )
    rospy.spin()


if __name__ == "__main__":
    main()
