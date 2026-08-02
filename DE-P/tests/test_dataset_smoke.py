import unittest

import numpy as np

from tests.baseline_helpers import DATASET_ROOT, load_limited_dataset, seed_everything


class DatasetSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(0)
        cls.dataset, cls.load_seconds = load_limited_dataset(mode="train", image_count=10)

    def test_one_real_sample(self):
        seed_everything(0)
        depth, pos, rot, obs, map_id = self.dataset[0]
        print(
            "BASELINE_DATASET",
            {
                "load_seconds": round(self.load_seconds, 6),
                "depth_shape": tuple(depth.shape),
                "depth_dtype": str(depth.dtype),
                "depth_min": float(depth.min()),
                "depth_max": float(depth.max()),
                "pos_shape": tuple(pos.shape),
                "rot_shape": tuple(rot.shape),
                "obs_shape": tuple(obs.shape),
                "map_id": map_id,
                "map_id_type": type(map_id).__name__,
            },
        )
        self.assertEqual(tuple(depth.shape), (1, 96, 160))
        self.assertEqual(depth.dtype, np.float32)
        self.assertEqual(tuple(pos.shape), (3,))
        self.assertEqual(tuple(rot.shape), (3, 3))
        self.assertEqual(tuple(obs.shape), (9,))
        self.assertIsInstance(map_id, int)
        self.assertGreaterEqual(float(depth.min()), 0.0)
        self.assertLessEqual(float(depth.max()), 1.0)
        for value in (depth, pos, rot, obs):
            self.assertTrue(np.isfinite(value).all())
        self.assertTrue((DATASET_ROOT / f"pointcloud-{map_id}.ply").is_file())
        self.assertTrue((DATASET_ROOT / str(map_id)).is_dir())


if __name__ == "__main__":
    unittest.main()
