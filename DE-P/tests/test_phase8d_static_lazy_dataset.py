import unittest
from unittest import mock

import cv2
import numpy as np

from policy.dep_dataset import seed_dataset_worker
from tests.baseline_helpers import load_limited_dataset


class Phase8DStaticLazyDatasetTest(unittest.TestCase):
    def test_initialization_does_not_decode_depth(self):
        with mock.patch("policy.dep_dataset.cv2.imread", side_effect=AssertionError("eager read")):
            dataset, _ = load_limited_dataset("train", 10)
        self.assertEqual(dataset.cached_depth_count, 0)
        self.assertTrue(all(isinstance(value, str) for value in dataset.img_list))

    def test_lazy_value_matches_legacy_transform_and_cache_is_bounded(self):
        dataset, _ = load_limited_dataset("train", 10)
        dataset.cache_size = 1
        depth = dataset[0][0]
        raw = cv2.imread(dataset.image_paths[0], cv2.IMREAD_UNCHANGED).astype(np.float32)
        expected = np.expand_dims(cv2.resize(
            raw, (dataset.width, dataset.height), interpolation=cv2.INTER_NEAREST
        ) / 65535.0, axis=0).astype(np.float32)
        np.testing.assert_array_equal(depth, expected)
        _ = dataset[1]
        self.assertLessEqual(dataset.cached_depth_count, 1)

    def test_worker_seed_policy_is_reproducible_and_distinct(self):
        dataset, _ = load_limited_dataset("train", 10)
        dataset.global_seed = 123
        dataset.set_epoch(4)
        info = type("Info", (), {"dataset": dataset})()
        with mock.patch("policy.dep_dataset.get_worker_info", return_value=info):
            seed_dataset_worker(0)
            first = dataset._get_random_goal()
            seed_dataset_worker(0)
            repeated = dataset._get_random_goal()
            seed_dataset_worker(1)
            other = dataset._get_random_goal()
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, other))


if __name__ == "__main__":
    unittest.main()
