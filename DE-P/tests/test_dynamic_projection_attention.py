import unittest

import numpy as np
import torch

from policy.dynamic.attention import build_dynamic_attention
from policy.dynamic.projection import image_to_feature_coordinates, project_world_points_to_image
from tests.dynamic_helpers import camera_model, dynamic_track, pose, test_config


class ProjectionTests(unittest.TestCase):
    def test_optical_axis_directions_and_behind_filter(self):
        camera = camera_model()
        points = np.array([
            [0, 0, 5], [-1, 0, 5], [1, 0, 5], [0, -1, 5], [0, 1, 5], [0, 0, -1],
        ], dtype=np.float64)
        pixels, mask, depths = project_world_points_to_image(
            points, pose(), camera, return_mask=True
        )
        self.assertTrue(np.allclose(pixels[0], [camera.cx, camera.cy]))
        self.assertLess(pixels[1, 0], camera.cx)
        self.assertGreater(pixels[2, 0], camera.cx)
        self.assertLess(pixels[3, 1], camera.cy)
        self.assertGreater(pixels[4, 1], camera.cy)
        self.assertFalse(mask[-1])
        self.assertTrue(np.isfinite(pixels).all())
        self.assertTrue(np.all(depths > 0))

    def test_world_camera_transform_and_image_boundaries(self):
        camera = camera_model()
        translated_pose = pose([1, 2, 0])
        center = project_world_points_to_image([[1, 2, 5]], translated_pose, camera)
        self.assertTrue(np.allclose(center[0], [camera.cx, camera.cy]))
        outside = project_world_points_to_image([[20, 0, 5]], pose(), camera)
        self.assertEqual(outside.shape, (0, 2))

    def test_image_to_3x5_feature_coordinates(self):
        camera = camera_model()
        pixels = np.array([[camera.cx, camera.cy], [-0.5, -0.5],
                           [camera.width - 0.5, camera.height - 0.5]])
        feature = image_to_feature_coordinates(
            pixels, (camera.height, camera.width), (3, 5)
        )
        self.assertTrue(np.allclose(feature[0], [2.0, 1.0]))
        self.assertTrue(np.allclose(feature[1], [-0.5, -0.5]))
        self.assertTrue(np.allclose(feature[2], [4.5, 2.5]))


class DynamicAttentionTests(unittest.TestCase):
    def setUp(self):
        self.camera = camera_model()
        self.pose = pose()
        self.config = test_config(attention_max=0.8)

    def attention(self, tracks):
        return build_dynamic_attention(
            tracks, self.pose, self.camera, (3, 5), self.config
        )

    def test_no_dynamic_track_is_zero_and_shape_is_generic(self):
        attention = self.attention([])
        self.assertEqual(tuple(attention.shape), (1, 1, 3, 5))
        self.assertEqual(float(attention.max()), 0.0)
        static = dynamic_track(dynamic=False)
        self.assertEqual(float(self.attention([static]).max()), 0.0)
        generic = build_dynamic_attention([], self.pose, self.camera, (7, 9), self.config)
        self.assertEqual(tuple(generic.shape), (1, 1, 7, 9))

    def test_attention_is_bounded_finite_and_uses_max_aggregation(self):
        tracks = [dynamic_track(track_id=index, position=(0, 0, 5), velocity=(0, 0, -1))
                  for index in range(10)]
        attention = self.attention(tracks)
        self.assertTrue(torch.isfinite(attention).all())
        self.assertGreater(float(attention.max()), 0.0)
        self.assertLessEqual(float(attention.max()), self.config.attention_max + 1e-7)

    def test_near_and_approaching_weight_more_than_far_or_departing(self):
        near_approach = self.attention([
            dynamic_track(position=(0, 0, 3), velocity=(0, 0, -1))
        ])
        far_approach = self.attention([
            dynamic_track(position=(0, 0, 10), velocity=(0, 0, -1))
        ])
        near_depart = self.attention([
            dynamic_track(position=(0, 0, 3), velocity=(0, 0, 1))
        ])
        self.assertGreater(float(near_approach.max()), float(far_approach.max()))
        self.assertGreater(float(near_approach.max()), float(near_depart.max()))


if __name__ == "__main__":
    unittest.main()
