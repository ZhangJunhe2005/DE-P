import unittest

from collections import Counter

from tools.finalize_phase8c_dynamic_dataset import (
    merge_sequence_stats,
    validate_frame_identity,
)


class DynamicFinalizerIdentityTest(unittest.TestCase):
    def setUp(self):
        self.row = {
            "sequence_id": "phase8c_valid_0001",
            "actor_seed": "930000",
            "map_id": "12",
            "local_map_id": "0",
        }
        self.frame = {
            "sequence_id": "phase8c_valid_0001",
            "frame_index": "0",
            "scenario_id": "phase8c_valid_0001",
            "seed": "930000",
            "map_id": "12",
        }

    def test_global_map_id_is_valid_for_non_train_split(self):
        validate_frame_identity("phase8c_valid_0001", self.row, self.frame, 0)

    def test_split_local_map_id_is_rejected_in_frame(self):
        self.frame["map_id"] = "0"
        with self.assertRaisesRegex(RuntimeError, "per-frame identity mismatch"):
            validate_frame_identity("phase8c_valid_0001", self.row, self.frame, 0)

    def test_aggregate_actor_count_retains_maximum(self):
        total = Counter()
        merge_sequence_stats(total, {"frames": 60, "maximum_actors": 3})
        merge_sequence_stats(total, {"frames": 60, "maximum_actors": 2})
        self.assertEqual(total["frames"], 120)
        self.assertEqual(total["maximum_actors"], 3)


if __name__ == "__main__":
    unittest.main()
