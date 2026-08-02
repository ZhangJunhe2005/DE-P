"""Stage-4 compatibility checks for the retired dynamic experiment APIs."""

import inspect
import unittest
import warnings

import numpy as np
import torch


class DynamicCompatibilityTests(unittest.TestCase):
    def test_attention_backbone_no_longer_passes_cluster_size(self):
        from policy.models.backbone import AttentionDepBackbone

        source = inspect.getsource(AttentionDepBackbone.__init__)
        self.assertNotIn("cluster_size=", source)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            module = AttentionDepBackbone(output_dim=64, input_size=(96, 160))
        self.assertTrue(any("unintegrated experimental" in str(item.message) for item in caught))
        with self.assertRaisesRegex(RuntimeError, "intentionally not connected"):
            module(torch.zeros(1, 1, 96, 160), np.empty((0, 3), dtype=np.float32))

    def test_legacy_ekf_name_wraps_working_linear_kf(self):
        from policy.models.EkfDynPercept import EKFTracker

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tracker = EKFTracker(np.array([1.0, 0.0, 5.0]), dt=0.1)
            predicted = tracker.predict()
            updated = tracker.update(np.array([1.1, 0.0, 5.0]))
        self.assertTrue(np.isfinite(predicted).all())
        self.assertTrue(np.isfinite(updated).all())
        self.assertTrue(any("LinearKalmanTracker" in str(item.message) for item in caught))

    def test_misnamed_clustering_is_deprecated_standard_dbscan(self):
        from policy.models.MonteCarloCutting import PointCloudProcessor

        rng = np.random.default_rng(7)
        cloud = np.vstack((rng.normal([0, 0, 5], 0.03, (20, 3)),
                           rng.normal([1, 0, 5], 0.03, (20, 3))))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            processor = PointCloudProcessor()
            first = processor.monte_carlo_clustering(cloud, eps=0.2, min_samples=5)
            second = processor.monte_carlo_clustering(cloud, eps=0.2, min_samples=5)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(set(first), {0, 1})
        self.assertTrue(any("never random" in str(item.message) for item in caught))

    def test_dynamic_core_remains_independent_from_depnetwork(self):
        from policy.dep_network import DepNetwork
        from policy.dynamic.dynamic_perception import DynamicPerception

        self.assertNotIn("point_cloud", inspect.signature(DepNetwork.forward).parameters)
        update_parameters = inspect.signature(DynamicPerception.update).parameters
        self.assertIn("point_cloud_camera", update_parameters)
        self.assertIn("camera_pose_world", update_parameters)
        self.assertIn("timestamp", update_parameters)
        self.assertIn("camera_model", update_parameters)


if __name__ == "__main__":
    unittest.main()
