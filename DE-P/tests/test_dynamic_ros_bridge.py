from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from policy.dynamic.ros_bridge import (
    body_points_to_optical,
    body_points_to_world,
    camera_model_from_config,
    camera_model_from_info,
    camera_pose_from_odometry,
    optical_points_to_body,
    validate_depth_encoding,
    validate_sensor_frame,
    validate_synchronized_timestamps,
)
from policy.dynamic.types import DynamicPerceptionConfig


def stamp(value):
    return SimpleNamespace(to_sec=lambda: value)


def header(value, frame_id):
    return SimpleNamespace(stamp=stamp(value), frame_id=frame_id)


def odometry(value=1.0, frame_id="world"):
    position = SimpleNamespace(x=1.0, y=2.0, z=3.0)
    orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    return SimpleNamespace(
        header=header(value, frame_id),
        child_frame_id="/quadrotor",
        pose=SimpleNamespace(pose=SimpleNamespace(position=position, orientation=orientation)),
    )


class RosBridgeTests(unittest.TestCase):
    def setUp(self):
        self.config = DynamicPerceptionConfig.from_global_config()

    def test_explicit_camera_model_and_camera_info(self):
        explicit = camera_model_from_config(self.config)
        self.assertEqual((explicit.width, explicit.height), (160, 90))
        info = SimpleNamespace(
            width=320, height=180,
            K=[160.0, 0, 160.0, 0, 161.0, 90.0, 0, 0, 1],
        )
        from_info = camera_model_from_info(info, self.config)
        self.assertEqual((from_info.width, from_info.height), (320, 180))
        self.assertEqual(from_info.fx, 160.0)

    def test_pose_composes_explicit_body_camera_extrinsic(self):
        pose = camera_pose_from_odometry(odometry(), self.config)
        np.testing.assert_allclose(pose.position_world, [1, 2, 3])
        np.testing.assert_allclose(
            pose.rotation_world_from_camera,
            np.asarray(self.config.camera_rotation_body_from_camera),
        )

    def test_mock_synchronization_accepts_and_rejects_offsets(self):
        sensor = SimpleNamespace(header=header(1.0, ""))
        info = SimpleNamespace(header=header(1.01, "camera_optical"))
        time_value, offsets = validate_synchronized_timestamps(
            sensor, odometry(1.02), info, self.config
        )
        self.assertEqual(time_value, 1.0)
        self.assertAlmostEqual(offsets["odometry"], 0.02)
        with self.assertRaisesRegex(ValueError, "max_pose_time_offset"):
            validate_synchronized_timestamps(sensor, odometry(1.2), None, self.config)

    def test_frame_and_encoding_checks(self):
        with self.assertRaisesRegex(ValueError, "frame_id"):
            validate_sensor_frame(SimpleNamespace(header=header(1, "")), self.config)
        compatible = replace(self.config, allow_empty_sensor_frame=True)
        validate_sensor_frame(SimpleNamespace(header=header(1, "")), compatible)
        self.assertEqual(validate_depth_encoding("32FC1"), "32FC1")
        self.assertEqual(validate_depth_encoding("16UC1"), "16UC1")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_depth_encoding("rgb8")

    def test_body_optical_axes_round_trip_translation_and_world(self):
        body = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        optical = body_points_to_optical(body, self.config)
        np.testing.assert_allclose(optical, [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        np.testing.assert_allclose(optical_points_to_body(optical, self.config), body)
        translated = replace(self.config, camera_position_body=[0.2, -0.1, 0.3])
        samples = np.asarray([[0.2, -0.1, 1.3], [1.2, -0.1, 0.3]])
        np.testing.assert_allclose(
            optical_points_to_body(body_points_to_optical(samples, translated), translated), samples
        )
        np.testing.assert_allclose(body_points_to_world(body, odometry(), self.config), body + [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
    optical_points_to_body,
