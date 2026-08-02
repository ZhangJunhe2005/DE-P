import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import torch

from policy.dynamic_collate import dynamic_sequence_collate
from policy.dynamic_sequence_dataset import DynamicSequenceDataset


ROOT = Path(__file__).resolve().parents[1]


class DynamicSequenceDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name) / "dynamic_dataset"
        subprocess.run([
            sys.executable, str(ROOT / "tools/generate_synthetic_dynamic_sequences.py"),
            "--output", str(cls.root), "--frames", "8",
        ], check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_lazy_continuous_window_and_real_state(self):
        dataset = DynamicSequenceDataset(self.root, "train")
        self.assertEqual(dataset.cached_depth_count, 0)
        sample = dataset[0]
        self.assertEqual(tuple(sample["depth_history"].shape), (4, 1, 96, 160))
        self.assertEqual(tuple(sample["current_depth"].shape), (1, 96, 160))
        self.assertEqual(tuple(sample["observation_9d"].shape), (9,))
        self.assertEqual(tuple(sample["attention"].shape), (1, 3, 5))
        self.assertGreater(dataset.cached_depth_count, 0)
        self.assertLessEqual(dataset.cached_depth_count, dataset.training_config.cache_size)
        self.assertTrue(torch.all(torch.diff(sample["timestamps"]) > 0))
        self.assertAlmostEqual(float(sample["observation_9d"][0]), 0.2, places=6)

    def test_collate_has_independent_context_and_padded_obstacles(self):
        dataset = DynamicSequenceDataset(self.root, "train")
        dynamic_sample = dataset[0]
        empty_sample = next(
            dataset[index] for index in range(len(dataset))
            if dataset[index]["sequence_id"] == "sequence_000002"
        )
        batch = dynamic_sequence_collate([dynamic_sample, empty_sample])
        self.assertEqual(tuple(batch["current_depth"].shape), (2, 1, 96, 160))
        self.assertEqual(batch["dynamic_context"].batch_size, 2)
        self.assertEqual(tuple(batch["dynamic_obstacles"].valid_mask.shape), (2, 1))
        self.assertEqual(int(batch["dynamic_obstacles"].valid_mask[1].sum()), 0)
        self.assertFalse(torch.equal(
            batch["dynamic_context"].attention()[0],
            batch["dynamic_context"].attention()[1],
        ))

    def test_context_uses_only_current_and_history(self):
        dataset = DynamicSequenceDataset(self.root, "valid")
        first = dataset[0]
        frame_index = first["frame_index"]
        self.assertEqual(frame_index, dataset.training_config.history_length - 1)
        self.assertAlmostEqual(
            float(first["timestamps"][-1]), first["sample_timestamp"], places=6
        )
        # The loader never opens annotations after the current frame.
        self.assertTrue(first["sequence_mask"].all())

    def test_previously_seen_occluded_gt_is_retained_as_track(self):
        dataset = DynamicSequenceDataset(self.root, "test")
        occluded = next(
            dataset[index] for index in range(len(dataset))
            if dataset[index]["frame_index"] == 5
        )
        self.assertEqual(len(occluded["objects"]), 1)
        self.assertTrue(occluded["objects"][0]["occluded"])
        self.assertEqual(float(occluded["objects"][0]["visibility"]), 0.0)
        self.assertTrue(occluded["objects"][0]["ever_observed_in_history"])
        self.assertTrue(occluded["objects"][0]["observable"])
        self.assertGreater(float(occluded["attention"].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
