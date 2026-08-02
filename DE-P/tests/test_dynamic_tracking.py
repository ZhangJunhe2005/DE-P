import unittest

import numpy as np

from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.track_manager import TrackManager
from tests.dynamic_helpers import (
    camera_cloud_from_world,
    camera_model,
    observation,
    pose,
    test_config,
    world_cluster,
)


class TrackManagerTests(unittest.TestCase):
    def manager(self, **overrides):
        defaults = dict(
            association_distance_threshold=2.0,
            association_mahalanobis_threshold=100.0,
            max_missed_frames=2,
            min_confirmed_hits=2,
            dynamic_min_confirmed_hits=3,
            dynamic_max_velocity_std=2.0,
        )
        defaults.update(overrides)
        return TrackManager(test_config(**defaults))

    def test_stable_id_new_target_and_mutual_exclusion(self):
        manager = self.manager()
        tracks = manager.update((observation([0, 0, 5], 0.0),), 0.0)
        self.assertEqual([track.track_id for track in tracks], [0])
        tracks = manager.update((
            observation([0.1, 0, 5], 0.1, temporary_id=8),
            observation([1.0, 0, 5], 0.1, temporary_id=2),
        ), 0.1)
        self.assertEqual([track.track_id for track in tracks], [0, 1])
        matches = manager.last_diagnostics["matches"]
        self.assertEqual(len({track for track, _ in matches}), len(matches))
        self.assertEqual(len({obs for _, obs in matches}), len(matches))

    def test_one_frame_occlusion_keeps_id_and_deletion_is_delayed(self):
        manager = self.manager(max_missed_frames=2)
        manager.update((observation([0, 0, 5], 0.0),), 0.0)
        hidden = manager.update((), 0.1)
        self.assertEqual(hidden[0].track_id, 0)
        self.assertEqual(hidden[0].missed_count, 1)
        recovered = manager.update((observation([0.2, 0, 5], 0.2),), 0.2)
        self.assertEqual(recovered[0].track_id, 0)
        self.assertEqual(recovered[0].missed_count, 0)
        manager.update((), 0.3)
        still_present = manager.update((), 0.4)
        self.assertEqual(len(still_present), 1)
        deleted = manager.update((), 0.5)
        self.assertEqual(deleted, ())
        new_track = manager.update((observation([0.5, 0, 5], 0.6),), 0.6)
        self.assertEqual(new_track[0].track_id, 1)

    def test_two_parallel_and_crossing_targets_keep_distinct_ids(self):
        parallel = self.manager()
        for frame in range(8):
            time = frame * 0.1
            tracks = parallel.update((
                observation([0.2 * frame, -0.5, 5], time, temporary_id=7),
                observation([0.2 * frame, 0.5, 5], time, temporary_id=3),
            ), time)
        self.assertEqual({track.track_id for track in tracks}, {0, 1})
        self.assertEqual(len(tracks), 2)

        crossing = self.manager()
        for frame in range(11):
            time = frame * 0.1
            a = [-1.0 + 0.2 * frame, -0.1, 5]
            b = [1.0 - 0.2 * frame, 0.1, 5]
            observations = ((observation(a, time, temporary_id=frame % 2),
                             observation(b, time, temporary_id=1 - frame % 2)))
            tracks = crossing.update(observations, time)
        by_id = {track.track_id: track for track in tracks}
        self.assertGreater(by_id[0].position_world[0], 0.7)
        self.assertLess(by_id[1].position_world[0], -0.7)

    def test_dynamic_hysteresis_and_timestamp_rejection(self):
        manager = self.manager(
            dynamic_enter_speed=0.3, dynamic_exit_speed=0.15,
            dynamic_max_velocity_std=5.0,
        )
        for frame in range(8):
            time = frame * 0.1
            tracks = manager.update((observation([0.05 * frame, 0, 5], time),), time)
        self.assertTrue(tracks[0].is_dynamic)
        # A speed estimate between enter/exit thresholds holds the previous dynamic state.
        self.assertIn(tracks[0].dynamic_reason, {"speed_above_enter_threshold", "hysteresis_hold_dynamic"})
        stop_reasons = []
        for frame in range(8, 25):
            time = frame * 0.1
            tracks = manager.update((observation([0.35, 0, 5], time),), time)
            stop_reasons.append(tracks[0].dynamic_reason)
        self.assertFalse(tracks[0].is_dynamic)
        self.assertTrue({
            "speed_below_exit_threshold", "motion_not_yet_consistent"
        }.intersection(stop_reasons))
        self.assertIn(tracks[0].dynamic_reason, {
            "speed_below_enter_threshold", "motion_not_yet_consistent"
        })
        last_time = 24 * 0.1
        with self.assertRaisesRegex(ValueError, "duplicate"):
            manager.update((), last_time)
        with self.assertRaisesRegex(ValueError, "out-of-order"):
            manager.update((), 2.0)

    def test_low_hits_and_high_velocity_uncertainty_do_not_mark_dynamic(self):
        manager = self.manager(
            min_confirmed_hits=2,
            dynamic_min_confirmed_hits=6,
            dynamic_max_velocity_std=1e-4,
        )
        for frame in range(4):
            time = frame * 0.1
            tracks = manager.update((observation([0.1 * frame, 0, 5], time),), time)
        self.assertTrue(tracks[0].is_confirmed)
        self.assertFalse(tracks[0].is_dynamic)
        self.assertIn(tracks[0].dynamic_reason, {
            "insufficient_confirmed_hits", "velocity_uncertainty_too_high"
        })


class EgoMotionCompensationTests(unittest.TestCase):
    def run_scene(self, camera_velocity, object_velocity):
        config = test_config(
            foreground_mode="none",
            cluster_eps=0.15,
            cluster_min_samples=5,
            association_distance_threshold=1.0,
            association_mahalanobis_threshold=100.0,
            min_confirmed_hits=3,
            dynamic_min_confirmed_hits=4,
            dynamic_max_velocity_std=2.0,
        )
        perception = DynamicPerception(config=config, feature_shape=(3, 5))
        offsets = world_cluster([0, 0, 0], seed=21, count=35, scale=0.02)
        result = None
        for frame in range(16):
            timestamp = frame * 0.1
            camera_position = np.asarray(camera_velocity) * timestamp
            object_center = np.array([0.0, 0.0, 5.0]) + np.asarray(object_velocity) * timestamp
            points_world = offsets + object_center
            current_pose = pose(camera_position, timestamp)
            cloud_camera = camera_cloud_from_world(points_world, current_pose)
            result = perception.update(cloud_camera, current_pose, timestamp, camera_model())
        self.assertEqual(len(result.all_tracks), 1)
        return result.all_tracks[0]

    def test_static_camera_static_object(self):
        track = self.run_scene([0, 0, 0], [0, 0, 0])
        self.assertLess(np.linalg.norm(track.velocity_world), 0.03)

    def test_moving_camera_static_object(self):
        track = self.run_scene([0.5, 0, 0], [0, 0, 0])
        self.assertLess(np.linalg.norm(track.velocity_world), 0.03)

    def test_static_camera_moving_object(self):
        track = self.run_scene([0, 0, 0], [0.4, 0, 0])
        self.assertTrue(np.allclose(track.velocity_world, [0.4, 0, 0], atol=0.06))

    def test_moving_camera_moving_object(self):
        track = self.run_scene([0.25, 0, 0], [0.4, 0, 0])
        self.assertTrue(np.allclose(track.velocity_world, [0.4, 0, 0], atol=0.06))


if __name__ == "__main__":
    unittest.main()
