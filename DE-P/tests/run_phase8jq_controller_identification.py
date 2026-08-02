#!/usr/bin/env python3
"""Bounded black-box identification of the deployed NetworkControl stack."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import statistics
import time

import rospy
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from sensor_msgs.msg import Imu


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/phase8jq_controller_identification_raw.json"


class Recorder:
    def __init__(self):
        self.odom = []
        self.imu = []
        self.first_command_wall = None
        self.publisher = rospy.Publisher(
            "/so3_control/pos_cmd", PositionCommand, queue_size=1
        )
        rospy.Subscriber("/sim/odom", Odometry, self.on_odom, queue_size=500)
        rospy.Subscriber("/sim/imu", Imu, self.on_imu, queue_size=500)

    def on_odom(self, message):
        stamp = message.header.stamp.to_sec()
        p = message.pose.pose.position
        v = message.twist.twist.linear
        self.odom.append([stamp, p.x, p.y, p.z, v.x, v.y, v.z])

    def on_imu(self, message):
        stamp = message.header.stamp.to_sec()
        if stamp <= 0.0:
            stamp = rospy.Time.now().to_sec()
        a = message.linear_acceleration
        self.imu.append([stamp, a.x, a.y, a.z])

    def command(self, duration, *, acceleration=(0, 0, 0),
                velocity=(0, 0, 0), ready=True, label=""):
        started = rospy.Time.now().to_sec()
        if self.first_command_wall is None:
            self.first_command_wall = started
        rate = rospy.Rate(50)
        while not rospy.is_shutdown() and rospy.Time.now().to_sec() - started < duration:
            message = PositionCommand()
            message.header.stamp = rospy.Time.now()
            if self.odom:
                message.position.x = self.odom[-1][1]
                message.position.y = self.odom[-1][2]
                message.position.z = self.odom[-1][3]
            message.velocity.x, message.velocity.y, message.velocity.z = velocity
            (message.acceleration.x, message.acceleration.y,
             message.acceleration.z) = acceleration
            message.yaw = 0.0
            message.yaw_dot = 0.0
            message.trajectory_flag = (
                PositionCommand.TRAJECTORY_STATUS_READY if ready
                else PositionCommand.TRAJECTORY_STATUS_EMPTY
            )
            self.publisher.publish(message)
            rate.sleep()
        return {"label": label, "start": started, "end": rospy.Time.now().to_sec()}


def vector_norm(row, offset):
    return math.sqrt(sum(float(row[offset + index]) ** 2 for index in range(3)))


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    return values[min(len(values) - 1, round((len(values) - 1) * fraction))]


def main():
    rospy.init_node("phase8jq_controller_identification", anonymous=True)
    recorder = Recorder()
    deadline = time.monotonic() + 12.0
    while len(recorder.odom) < 20 and time.monotonic() < deadline:
        rospy.sleep(0.05)
    if len(recorder.odom) < 20:
        raise RuntimeError("simulator odometry was not observed")

    # NetworkControl's simulation takeoff needs about 5.5 s after construction.
    rospy.sleep(7.0)
    phases = []
    phases.append(recorder.command(0.8, label="zero_acceleration"))
    phases.append(recorder.command(
        1.0, acceleration=(2.0, 0, 0), label="lateral_acceleration_step"
    ))
    phases.append(recorder.command(
        1.0, acceleration=(-2.0, 0, 0), label="lateral_brake"
    ))
    phases.append(recorder.command(
        1.0, acceleration=(0, 0, 2.0), label="vertical_acceleration_step"
    ))
    phases.append(recorder.command(
        1.0, acceleration=(0, 0, -2.0), label="vertical_brake"
    ))
    phases.append(recorder.command(
        1.0, velocity=(2.0, 0, 0), ready=False, label="velocity_step"
    ))
    phases.append(recorder.command(0.8, label="settle"))

    imu = recorder.imu
    odom = recorder.odom
    imu_dt = [
        imu[index][0] - imu[index - 1][0] for index in range(1, len(imu))
        if imu[index][0] > imu[index - 1][0]
    ]
    odom_dt = [
        odom[index][0] - odom[index - 1][0] for index in range(1, len(odom))
        if odom[index][0] > odom[index - 1][0]
    ]
    jerk = []
    for index in range(1, len(imu)):
        dt = imu[index][0] - imu[index - 1][0]
        if dt > 1e-5:
            jerk.append(math.sqrt(sum(
                ((imu[index][axis] - imu[index - 1][axis]) / dt) ** 2
                for axis in range(1, 4)
            )))
    phase_results = []
    for phase in phases:
        samples = [row for row in odom if phase["start"] <= row[0] <= phase["end"]]
        acc_samples = [row for row in imu if phase["start"] <= row[0] <= phase["end"]]
        displacement = None
        if len(samples) >= 2:
            displacement = math.sqrt(sum(
                (samples[-1][axis] - samples[0][axis]) ** 2
                for axis in range(1, 4)
            ))
        phase_results.append({
            **phase,
            "odom_samples": len(samples),
            "imu_samples": len(acc_samples),
            "maximum_speed_mps": max(
                (vector_norm(row, 4) for row in samples), default=None
            ),
            "maximum_acceleration_mps2": max(
                (vector_norm(row, 1) for row in acc_samples), default=None
            ),
            "displacement_m": displacement,
        })

    output = {
        "status": "PASS",
        "measurement_scope": "bounded NetworkControl + SO3 quadrotor simulator",
        "ros_master_uri": os.environ.get("ROS_MASTER_URI"),
        "odom_samples": len(odom),
        "imu_samples": len(imu),
        "odom_rate_hz_median": (
            1.0 / statistics.median(odom_dt) if odom_dt else None
        ),
        "imu_rate_hz_median": (
            1.0 / statistics.median(imu_dt) if imu_dt else None
        ),
        "maximum_achieved_speed_mps": max(
            (vector_norm(row, 4) for row in odom), default=None
        ),
        "maximum_achieved_acceleration_mps2": max(
            (vector_norm(row, 1) for row in imu), default=None
        ),
        "achieved_jerk_mps3_p95": percentile(jerk, 0.95),
        "achieved_jerk_mps3_max": max(jerk, default=None),
        "phases": phase_results,
        "raw": {"odom": odom, "imu": imu},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps({key: value for key, value in output.items() if key != "raw"}, indent=2))


if __name__ == "__main__":
    main()
