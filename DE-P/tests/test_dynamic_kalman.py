import unittest

import numpy as np

from policy.dynamic.kalman_tracker import LinearKalmanTracker
from tests.dynamic_helpers import test_config


class LinearKalmanTrackerTests(unittest.TestCase):
    def setUp(self):
        self.config = test_config(
            min_dt=0.01, max_dt=0.5, process_noise_acceleration=0.3,
            measurement_noise=0.03,
        )

    def tracker(self, position=(0, 0, 0)):
        return LinearKalmanTracker(position, 0.0, self.config)

    def assert_finite_symmetric(self, tracker):
        self.assertTrue(np.isfinite(tracker.state).all())
        self.assertTrue(np.isfinite(tracker.covariance).all())
        self.assertTrue(np.allclose(tracker.covariance, tracker.covariance.T, atol=1e-9))

    def test_stationary_and_constant_velocity(self):
        stationary = self.tracker([2, -1, 5])
        for frame in range(1, 21):
            stationary.predict_to(frame * 0.1)
            stationary.update([2, -1, 5])
        self.assertLess(np.linalg.norm(stationary.state[3:]), 0.03)
        moving = self.tracker()
        velocity = np.array([0.5, -0.2, 0.1])
        for frame in range(1, 31):
            time = frame * 0.1
            moving.predict_to(time)
            moving.update(velocity * time)
        self.assertTrue(np.allclose(moving.state[3:], velocity, atol=0.04))
        self.assert_finite_symmetric(moving)

    def test_noisy_velocity_and_missing_measurements(self):
        rng = np.random.default_rng(5)
        tracker = self.tracker()
        velocity = np.array([0.4, 0.1, 0.0])
        for frame in range(1, 31):
            time = frame * 0.1
            tracker.predict_to(time)
            if frame not in {10, 11, 12}:
                tracker.update(velocity * time + rng.normal(0, 0.02, 3))
            self.assert_finite_symmetric(tracker)
        self.assertTrue(np.allclose(tracker.state[3:], velocity, atol=0.08))

    def test_single_missing_frame_predicts_without_update(self):
        tracker = self.tracker()
        tracker.predict_to(0.1)
        tracker.update([0.05, 0, 0])
        tracker.predict_to(0.2)  # exactly one missing measurement
        predicted = tracker.state.copy()
        tracker.predict_to(0.3)
        tracker.update([0.15, 0, 0])
        self.assertTrue(np.isfinite(predicted).all())
        self.assert_finite_symmetric(tracker)

    def test_dt_clamping_duplicate_and_out_of_order(self):
        tracker = self.tracker()
        tracker.predict_to(0.0001)
        self.assertEqual(tracker.last_effective_dt, self.config.min_dt)
        tracker.predict_to(10.0)
        self.assertEqual(tracker.last_effective_dt, self.config.max_dt)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            tracker.predict_to(10.0)
        with self.assertRaisesRegex(ValueError, "out-of-order"):
            tracker.predict_to(9.0)
        self.assert_finite_symmetric(tracker)

    def test_predict_without_update_increases_uncertainty(self):
        tracker = self.tracker()
        before = np.trace(tracker.covariance)
        tracker.predict_to(0.1)
        tracker.predict_to(0.2)
        self.assertGreater(np.trace(tracker.covariance), before)
        self.assert_finite_symmetric(tracker)


if __name__ == "__main__":
    unittest.main()
