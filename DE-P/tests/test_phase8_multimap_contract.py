import unittest
from pathlib import Path

import numpy as np

from policy.dynamic_sequence_dataset import DynamicSequenceDataset, validate_dataset_splits


ROOT = Path(__file__).resolve().parents[1]


class Phase8MultiMapContractTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "data/phase8_preflight_multimap"

    def test_map_splits_are_disjoint(self):
        manifest, _ = validate_dataset_splits(self.root)
        maps = {key: set(value) for key, value in manifest["map_splits"].items()}
        self.assertFalse(maps["train"] & maps["valid"])
        self.assertFalse(maps["train"] & maps["test"])
        self.assertFalse(maps["valid"] & maps["test"])

    def test_waypoint_reversal_differs_from_constant_velocity(self):
        dataset = DynamicSequenceDataset(self.root, "train")
        sample = next(dataset[index] for index in range(len(dataset))
                      if dataset[index]["sequence_id"] == "phase8_train_0005"
                      and dataset[index]["frame_index"] == 10)
        obj = sample["objects"][0]
        cv = (np.asarray(obj["context_position_world"])[None, :]
              + np.asarray(obj["context_velocity_world"])[None, :]
              * (sample["future_timestamps"].numpy() - sample["sample_timestamp"])[:, None])
        valid = sample["future_valid_mask"][0].numpy()
        delta = np.linalg.norm(sample["future_positions_world"][0].numpy()[valid] - cv[valid], axis=1)
        self.assertGreater(float(delta.max()), 0.2)

    def test_delayed_actor_and_lifecycle_mask(self):
        dataset = DynamicSequenceDataset(self.root, "valid")
        before = next(dataset[index] for index in range(len(dataset))
                      if dataset[index]["sequence_id"] == "phase8_valid_0009"
                      and dataset[index]["frame_index"] == 4)
        after = next(dataset[index] for index in range(len(dataset))
                     if dataset[index]["sequence_id"] == "phase8_valid_0009"
                     and dataset[index]["frame_index"] == 12)
        self.assertAlmostEqual(float(np.linalg.norm(before["objects"][0]["velocity_world"])), 0.0)
        self.assertGreater(float(np.linalg.norm(after["objects"][0]["velocity_world"])), 0.0)
        self.assertFalse(bool(after["future_valid_mask"][0, -1]))


if __name__ == "__main__":
    unittest.main()
