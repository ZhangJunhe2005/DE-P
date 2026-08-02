import unittest
from dataclasses import replace

import numpy as np

from policy.dynamic.camera_model import make_depth_frame
from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.image_foreground_components import grow_seeded_components
from policy.dynamic.range_image_foreground import (
    CausalRangeImageForeground, reproject_history_depth,
)
from policy.dynamic.types import CameraModel, DynamicPerceptionConfig, Pose


class Phase8FRangeImageTests(unittest.TestCase):
    def setUp(self):
        self.camera = CameraModel(32, 24, 20, 20, 16, 12, 1.0, 0.1, 20.0)
        base = DynamicPerceptionConfig.from_global_config()
        self.config = replace(
            base, foreground_mode="range_image_hybrid", depth_stride=1,
            range_history_frames=3, range_min_history_support=1,
            range_min_seed_pixels=2, range_min_seed_fraction=0.01,
            range_abs_residual_threshold=0.15, range_rel_residual_threshold=0.001,
            range_edge_guard_threshold=3.0, range_component_connectivity=4,
            range_growth_abs_depth=0.15, range_growth_rel_depth=0.001,
            range_growth_max_3d_neighbor_distance=0.5,
            range_max_component_pixels=300, range_max_component_depth_span=0.5,
        )

    def frame(self, depth, timestamp, position=(0, 0, 0), stride=1):
        pose = Pose(np.asarray(position, float), np.eye(3), timestamp)
        return make_depth_frame(np.asarray(depth), self.camera, pose, timestamp, stride)

    def plane(self, value=5.0):
        return np.full((24, 32), value, np.float32)

    def test_01_raw_geometry_is_preserved(self):
        self.assertEqual(self.frame(self.plane(), 0).depth_m.shape, (24, 32))

    def test_02_metric_depth_is_float32(self):
        self.assertEqual(self.frame(self.plane(), 0).depth_m.dtype, np.float32)

    def test_03_pixels_and_points_are_one_to_one(self):
        frame = self.frame(self.plane(), 0, stride=2)
        self.assertEqual(len(frame.pixels_uv), len(frame.points_camera))
        np.testing.assert_allclose(frame.points_camera[:, 2], 5.0)

    def test_04_stride_pixels_refer_to_raw_coordinates(self):
        frame = self.frame(self.plane(), 0, stride=2)
        self.assertTrue(np.all(frame.pixels_uv % 2 == 0))

    def test_05_max_depth_saturation_is_invalid(self):
        depth = self.plane(); depth[0, 0] = 20
        self.assertFalse(self.frame(depth, 0).valid_mask[0, 0])

    def test_06_invalid_and_nan_depth_are_removed(self):
        depth = self.plane(); depth[0, :2] = (0, np.nan)
        self.assertEqual(len(self.frame(depth, 0).points_camera), 24 * 32 - 2)

    def test_07_wrong_raw_shape_fails(self):
        with self.assertRaises(ValueError):
            self.frame(np.ones((24, 31), np.float32), 0)

    def test_08_static_camera_static_plane_has_no_seed(self):
        fg = CausalRangeImageForeground(self.config)
        fg.extract(self.frame(self.plane(), 0)); fg.extract(self.frame(self.plane(), .1))
        self.assertEqual(fg.last_diagnostics["range_seed_count"], 0)

    def test_09_moving_camera_static_plane_has_no_seed(self):
        fg = CausalRangeImageForeground(self.config)
        fg.extract(self.frame(self.plane(), 0))
        fg.extract(self.frame(self.plane(), .1, position=(.1, 0, 0)))
        self.assertEqual(fg.last_diagnostics["range_seed_count"], 0)

    def test_10_static_camera_lateral_actor_has_component(self):
        fg = CausalRangeImageForeground(self.config)
        fg.extract(self.frame(self.plane(), 0))
        depth = self.plane(); depth[8:16, 10:16] = 3
        observations, _ = fg.extract(self.frame(depth, .1))
        self.assertEqual(len(observations), 1)

    def test_11_moving_camera_lateral_actor_has_component(self):
        fg = CausalRangeImageForeground(self.config)
        fg.extract(self.frame(self.plane(), 0))
        depth = self.plane(); depth[8:16, 12:18] = 3
        observations, _ = fg.extract(self.frame(depth, .1, position=(.05, 0, 0)))
        self.assertGreaterEqual(len(observations), 1)

    def test_12_head_on_actor_creates_closer_residual(self):
        fg = CausalRangeImageForeground(self.config)
        old = self.plane(); old[8:16, 12:20] = 4
        new = self.plane(); new[8:16, 12:20] = 3
        fg.extract(self.frame(old, 0)); fg.extract(self.frame(new, .1))
        self.assertGreater(fg.last_diagnostics["range_seed_count"], 0)

    def test_13_actor_visible_at_first_frame_needs_motion_evidence(self):
        fg = CausalRangeImageForeground(self.config)
        first = self.plane(); first[8:16, 8:14] = 3
        second = self.plane(); second[8:16, 12:18] = 3
        observations, _ = fg.extract(self.frame(first, 0)); self.assertFalse(observations)
        observations, _ = fg.extract(self.frame(second, .1)); self.assertTrue(observations)

    def test_14_delayed_actor_is_detected(self):
        fg = CausalRangeImageForeground(self.config)
        fg.extract(self.frame(self.plane(), 0)); fg.extract(self.frame(self.plane(), .1))
        depth = self.plane(); depth[8:16, 12:20] = 3
        observations, _ = fg.extract(self.frame(depth, .2)); self.assertTrue(observations)

    def test_15_waypoint_reversal_remains_causal(self):
        fg = CausalRangeImageForeground(self.config); fg.extract(self.frame(self.plane(), 0))
        for index, column in enumerate((8, 12, 8), 1):
            depth = self.plane(); depth[8:16, column:column + 5] = 3
            fg.extract(self.frame(depth, index * .1))
        self.assertEqual(fg.last_diagnostics["future_frames_used"], 0)

    def test_16_disocclusion_is_not_closer_foreground(self):
        fg = CausalRangeImageForeground(self.config)
        old = self.plane(); old[8:16, 12:20] = 3
        fg.extract(self.frame(old, 0)); fg.extract(self.frame(self.plane(), .1))
        self.assertEqual(fg.last_diagnostics["range_seed_count"], 0)

    def test_17_reprojection_handles_image_boundary(self):
        old = self.frame(self.plane(), 0)
        current = self.frame(self.plane(), .1, position=(100, 0, 0))
        prediction = reproject_history_depth(old, current)
        self.assertEqual(prediction.shape, (24, 32))

    def test_18_zbuffer_output_keeps_nearest_depth(self):
        depth = self.plane(); depth[:, :16] = 3
        prediction = reproject_history_depth(self.frame(depth, 0), self.frame(depth, .1))
        self.assertAlmostEqual(float(prediction[12, 8]), 3.0, places=5)

    def test_19_component_does_not_cross_depth_discontinuity(self):
        frame = self.frame(np.hstack((np.full((24, 16), 3, np.float32),
                                      np.full((24, 16), 5, np.float32))), 0)
        seed = np.zeros((24, 32), bool); seed[12, 15] = True; seed[11, 15] = True
        components = grow_seeded_components(frame, seed, seed, seed & False,
                                            seed & False, seed.astype(int), seed.astype(float), self.config)
        self.assertTrue(all(np.all(component.pixels_vu[:, 1] < 16) for component in components))

    def test_20_edge_seed_recovers_connected_target(self):
        frame = self.frame(self.plane(3), 0); seed = np.zeros((24, 32), bool)
        seed[10:12, 10] = True
        blocked = np.ones((24, 32), bool); blocked[8:16, 8:16] = False
        support = (~blocked).astype(int); residual = (~blocked).astype(float)
        components = grow_seeded_components(frame, seed, seed, seed & False,
                                            blocked, support, residual, self.config)
        self.assertEqual(len(components), 1)
        self.assertEqual(len(components[0].pixels_vu), 64)

    def test_21_no_seed_means_no_component(self):
        frame = self.frame(self.plane(), 0); empty = np.zeros((24, 32), bool)
        self.assertFalse(grow_seeded_components(frame, empty, empty, empty, empty,
                                                empty.astype(int), empty.astype(float), self.config))

    def test_22_history_capacity_is_bounded(self):
        fg = CausalRangeImageForeground(self.config)
        for index in range(8): fg.extract(self.frame(self.plane(), index * .1))
        self.assertLessEqual(fg.history_size, self.config.range_history_frames)

    def test_23_time_gap_resets_history(self):
        fg = CausalRangeImageForeground(self.config); fg.extract(self.frame(self.plane(), 0))
        fg.extract(self.frame(self.plane(), 1.0))
        self.assertTrue(fg.last_diagnostics["reset_due_gap"])

    def test_24_pointcloud_update_rejects_range_mode_without_pixels(self):
        frame = self.frame(self.plane(), 0); perception = DynamicPerception(self.config)
        with self.assertRaisesRegex(ValueError, "update_depth"):
            perception.update(frame.points_camera, frame.camera_pose_world, 0, self.camera)

    def test_25_update_depth_is_finite_and_old_update_still_works(self):
        range_result = DynamicPerception(self.config).update_depth(
            self.plane(), Pose(np.zeros(3), np.eye(3), 0), 0, self.camera)
        old_config = replace(self.config, foreground_mode="temporal_voxel")
        frame = self.frame(self.plane(), 0)
        old_result = DynamicPerception(old_config).update(
            frame.points_camera, frame.camera_pose_world, 0, self.camera)
        self.assertTrue(np.isfinite(range_result.attention_map.numpy()).all())
        self.assertEqual(range_result.attention_map.shape, old_result.attention_map.shape)


if __name__ == "__main__":
    unittest.main()
