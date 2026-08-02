"""Regression coverage for the Formal V3 multi-target pairing hotfix."""

from __future__ import annotations

import unittest

import numpy as np

from authoritative_dataset.dynamic_motion_v2 import (
    _multi_target_independent_pair_fallback,
    actor_position,
)


class _OpenAuthority:
    def query_one(self, _point, _radius):
        return {"collision": False}


class _BlockedPositiveSideAuthority:
    """Leave only the opposite-side independent pair feasible."""

    def query_one(self, point, _radius):
        point = np.asarray(point)
        return {"collision": bool(point[1] > 1.45)}


def _contract():
    return {
        "actor_static_clearance_margin_m": 0.05,
        "uav_radius_m": 0.30,
        "actor_uav_clearance_margin_m": 0.10,
        "actor_actor_clearance_margin_m": 0.10,
        "maximum_detection_entry_time_s": 1.20,
        "maximum_detection_center_distance_m": 1.80,
    }


class MultiTargetPairingTests(unittest.TestCase):
    def setUp(self):
        self.frame_times = np.arange(60, dtype=np.float64) * 0.1
        self.validation_times = np.linspace(0.0, 7.6, 267)
        self.uav = np.zeros((60, 3), dtype=np.float64)
        self.camera = np.zeros((267, 3), dtype=np.float64)
        self.profile = {"motion_profile": "constant_velocity"}

    def solve(self, backend):
        return _multi_target_independent_pair_fallback(
            backend, self.camera, self.uav, 0.0, self.frame_times,
            self.profile, _contract(), 0.20, 1.40, 1.75,
            self.validation_times,
        )

    def assert_contract(self, actors):
        self.assertEqual(len(actors), 2)
        paths = [
            actor_position(actor, self.validation_times)
            for actor in actors
        ]
        for actor, path in zip(actors, paths):
            self.assertEqual(
                actor["sampling_method"],
                "deterministic_independent_pair_lattice_v2_1",
            )
            self.assertGreaterEqual(np.linalg.norm(actor["velocity"]), 1.40)
            self.assertLessEqual(np.linalg.norm(actor["velocity"]), 1.75)
            detection = self.validation_times <= 1.2 + 1e-12
            self.assertLessEqual(
                np.min(np.linalg.norm(path[detection], axis=1)), 1.80)
            self.assertGreater(np.min(np.linalg.norm(path, axis=1)), 0.60)
        self.assertGreater(
            np.min(np.linalg.norm(paths[0]-paths[1], axis=1)), 0.50)

    def test_open_space_pair_satisfies_frozen_contract(self):
        self.assert_contract(self.solve(_OpenAuthority()))

    def test_independent_solver_handles_asymmetric_free_space(self):
        self.assert_contract(self.solve(_BlockedPositiveSideAuthority()))

    def test_solver_is_deterministic(self):
        first = self.solve(_OpenAuthority())
        second = self.solve(_OpenAuthority())
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left["start"], right["start"])
            np.testing.assert_array_equal(left["velocity"], right["velocity"])


if __name__ == "__main__":
    unittest.main()
