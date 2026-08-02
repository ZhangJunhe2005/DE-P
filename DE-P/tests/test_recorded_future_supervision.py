from dataclasses import replace
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import torch

from config.config import cfg
from loss.dynamic_safety_loss import DynamicCollisionLoss
from loss.dynamic_types import DynamicLossConfig, DynamicObstacleBatch
from loss.trajectory_sampler import QuinticTrajectorySampler
from policy.dynamic_collate import dynamic_sequence_collate
from policy.dynamic_sequence_dataset import DynamicSequenceDataset
from policy.dynamic_training_config import DynamicTrainingConfig
from tests.test_trajectory_sampler_regression import coefficient_map


ROOT = Path(__file__).resolve().parents[1]


class RecordedFutureSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name) / "dataset"
        subprocess.run([
            sys.executable, str(ROOT / "tools/generate_synthetic_dynamic_sequences.py"),
            "--output", str(cls.root), "--frames", "24",
        ], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_future_grid_is_aligned_and_masked_at_sequence_end(self):
        dataset = DynamicSequenceDataset(self.root, "train")
        early = dataset[0]
        late = dataset[len(dataset) - 1]
        expected = early["sample_timestamp"] + torch.linspace(
            cfg["sgm_time"] / cfg["dynamic_loss"]["eval_points"], cfg["sgm_time"],
            cfg["dynamic_loss"]["eval_points"], dtype=torch.float64,
        )
        self.assertTrue(torch.allclose(early["future_timestamps"], expected, atol=1e-5))
        self.assertTrue(torch.all(torch.diff(early["future_timestamps"]) > 0))
        self.assertLessEqual(int(late["future_valid_mask"].sum()),
                             int(early["future_valid_mask"].sum()))

    def test_future_gt_is_not_in_context(self):
        dataset = DynamicSequenceDataset(self.root, "train")
        sample = next(dataset[index] for index in range(len(dataset)) if dataset[index]["objects"])
        original_attention = sample["attention"].clone()
        sample["future_positions_world"].add_(1000)
        self.assertTrue(torch.equal(sample["attention"], original_attention))

    def test_recorded_linear_matches_constant_velocity(self):
        dataset = DynamicSequenceDataset(self.root, "train")
        sample = next(dataset[index] for index in range(len(dataset))
                      if dataset[index]["objects"] and dataset[index]["future_valid_mask"].all())
        obstacles = dynamic_sequence_collate([sample])["dynamic_obstacles"]
        duration = float(cfg["sgm_time"])
        sampler = QuinticTrajectorySampler(coefficient_map(duration), duration, 30)
        fixed = torch.zeros(15, 3, 3)
        predicted = torch.zeros_like(fixed)
        predicted[:, 0, 0] = 2.0
        recorded = DynamicCollisionLoss(sampler, replace(
            DynamicLossConfig.from_global_config(), enabled=True,
            target_source="recorded_future_gt",
        ))(fixed, predicted, obstacles)[0]
        extrapolated = DynamicCollisionLoss(sampler, replace(
            DynamicLossConfig.from_global_config(), enabled=True,
            target_source="constant_velocity",
        ))(fixed, predicted, obstacles)[0]
        self.assertTrue(torch.allclose(recorded, extrapolated, atol=2e-3, rtol=2e-3))
        self.assertTrue(torch.isfinite(recorded).all())

    def test_never_observed_actor_is_exactly_excluded(self):
        dataset = DynamicSequenceDataset(self.root, "train")
        empty = next(dataset[index] for index in range(len(dataset))
                     if dataset[index]["sequence_id"] == "sequence_000002")
        batch = dynamic_sequence_collate([empty])["dynamic_obstacles"]
        self.assertEqual(batch.max_obstacles, 0)
        self.assertEqual(int(batch.observable_mask.sum()), 0)

    def test_future_first_actor_does_not_leak(self):
        sequence = self.root / "sequences/sequence_000001/dynamic_objects"
        originals = {}
        try:
            for frame_index in range(5):
                path = sequence / f"{frame_index:06d}.json"
                originals[path] = path.read_text()
                objects = json.loads(originals[path])
                objects[0]["visibility"] = 0.0
                objects[0]["occluded"] = True
                path.write_text(json.dumps(objects))
            dataset = DynamicSequenceDataset(self.root, "train")
            sample = next(dataset[index] for index in range(len(dataset))
                          if dataset[index]["sequence_id"] == "sequence_000001"
                          and dataset[index]["frame_index"] == 3)
            self.assertEqual(sample["objects"], [])
            self.assertTrue(torch.equal(sample["attention"], torch.zeros_like(sample["attention"])))
            self.assertEqual(sample["future_positions_world"].shape[0], 0)
        finally:
            for path, content in originals.items():
                path.write_text(content)

    def test_noisy_gt_is_deterministic_and_does_not_change_future_labels(self):
        base = DynamicTrainingConfig.from_global_config()
        clean = DynamicSequenceDataset(self.root, "train", replace(base, context_source="ground_truth"))[0]
        noisy_dataset = DynamicSequenceDataset(
            self.root, "train", replace(base, context_source="noisy_ground_truth")
        )
        noisy_a, noisy_b = noisy_dataset[0], noisy_dataset[0]
        self.assertTrue(torch.equal(noisy_a["attention"], noisy_b["attention"]))
        self.assertTrue(torch.equal(clean["future_positions_world"], noisy_a["future_positions_world"]))
        self.assertTrue(torch.equal(clean["future_valid_mask"], noisy_a["future_valid_mask"]))

    def test_estimated_context_warmup_and_cache_are_window_local(self):
        cache = Path(self.temp.name) / "estimated_cache"
        config = replace(DynamicTrainingConfig.from_global_config(),
                         context_source="estimated", estimated_cache_dir=str(cache))
        dataset = DynamicSequenceDataset(self.root, "train", config)
        first = dataset[0]
        cached = list(cache.glob("*.pt"))
        second = dataset[0]
        self.assertTrue(cached)
        self.assertTrue(torch.equal(first["attention"], second["attention"]))


if __name__ == "__main__":
    unittest.main()
