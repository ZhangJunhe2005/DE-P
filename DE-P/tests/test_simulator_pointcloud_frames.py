from dataclasses import replace
from pathlib import Path
import unittest

import cv2
import numpy as np

from policy.dynamic.camera_model import depth_to_pointcloud
from policy.dynamic.ros_bridge import (
    body_points_to_optical,
    body_points_to_world,
    optical_points_to_body,
)
from policy.dynamic.types import CameraModel, DynamicPerceptionConfig


SIMULATOR = Path("/home/zjh/YOPO/Simulator/src")


class SimulatorFrameSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.config = DynamicPerceptionConfig.from_global_config()

    def test_simulator_declares_legacy_and_three_semantic_topics(self):
        yaml_text = (SIMULATOR / "config/config.yaml").read_text(encoding="utf-8")
        source = (SIMULATOR / "src/test_simulator_cuda.cpp").read_text(encoding="utf-8")
        for topic in ("/lidar_points", "/lidar_points_body", "/lidar_points_optical",
                      "/lidar_points_odom", "/camera_info"):
            self.assertIn(topic, yaml_text)
        self.assertIn('output.header.frame_id = "odom"', source)
        self.assertIn("Deprecated /lidar_points", source)
        self.assertIn("body_output.header.frame_id = body_frame_id", source)
        self.assertIn("optical_output.header.frame_id = camera_optical_frame_id", source)
        self.assertIn("world_output.header.frame_id = world_frame_id", source)

    def test_known_axes_round_trip_and_ninety_degree_rotation(self):
        points = np.asarray([[2, 0, 0], [0, -2, 0], [0, 0, -2]], dtype=np.float64)
        optical = body_points_to_optical(points, self.config)
        np.testing.assert_allclose(optical, [[0, 0, 2], [2, 0, 0], [0, 2, 0]])
        np.testing.assert_allclose(optical_points_to_body(optical, self.config), points)

        rotation_z_90 = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        rotated = replace(
            self.config,
            camera_rotation_body_from_camera=rotation_z_90.tolist(),
        )
        rotated.validate()
        sample = np.asarray([[1.0, 0.0, 0.0]])
        np.testing.assert_allclose(body_points_to_optical(sample, rotated), [[0, -1, 0]])

    def test_static_world_point_remains_fixed_during_camera_motion(self):
        from types import SimpleNamespace

        def odom(position):
            p = SimpleNamespace(x=position[0], y=position[1], z=position[2])
            q = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
            return SimpleNamespace(
                child_frame_id="/quadrotor",
                pose=SimpleNamespace(pose=SimpleNamespace(position=p, orientation=q)),
            )

        fixed_world = np.asarray([[5.0, 1.0, 2.0]])
        for camera_position in ([0, 0, 0], [1, -2, 0.5]):
            body = fixed_world - np.asarray(camera_position)
            reconstructed = body_points_to_world(body, odom(camera_position), self.config)
            np.testing.assert_allclose(reconstructed, fixed_world)

    def test_depth_geometry_and_cnn_resize_are_separate(self):
        camera = CameraModel(
            width=160, height=90, fx=80, fy=80, cx=80, cy=45,
            depth_scale=1.0, min_depth=0.1, max_depth=20.0,
        )
        raw = np.full((90, 160), 4.0, dtype=np.float32)
        raw_copy = raw.copy()
        points = depth_to_pointcloud(raw, camera)
        center_index = 45 * 160 + 80
        np.testing.assert_allclose(points[center_index], [0, 0, 4], atol=1e-6)
        projected_u = camera.fx * points[:, 0] / points[:, 2] + camera.cx
        projected_v = camera.fy * points[:, 1] / points[:, 2] + camera.cy
        self.assertLess(float(np.max(np.abs(projected_u - np.tile(np.arange(160), 90)))), 1e-5)
        self.assertLess(float(np.max(np.abs(projected_v - np.repeat(np.arange(90), 160)))), 1e-5)
        resized = cv2.resize(raw, (160, 96), interpolation=cv2.INTER_NEAREST)
        self.assertEqual(resized.shape, (96, 160))
        np.testing.assert_array_equal(raw, raw_copy)


if __name__ == "__main__":
    unittest.main()
