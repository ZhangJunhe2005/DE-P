import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from policy.dynamic_sequence_dataset import DynamicSequenceDataset, validate_dataset_splits


ROOT = Path(__file__).resolve().parents[1]


class DynamicDatasetSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dataset"
        subprocess.run([
            sys.executable, str(ROOT / "tools/generate_synthetic_dynamic_sequences.py"),
            "--output", str(self.root), "--frames", "6",
        ], check=True, capture_output=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_splits_have_no_sequence_or_seed_leakage(self):
        _, splits = validate_dataset_splits(self.root)
        self.assertFalse(set(splits["train"]) & set(splits["valid"]))
        self.assertFalse(set(splits["train"]) & set(splits["test"]))

    def test_duplicate_sequence_is_rejected(self):
        with (self.root / "splits/valid.txt").open("a") as stream:
            stream.write("sequence_000001\n")
        with self.assertRaisesRegex(ValueError, "split leakage"):
            validate_dataset_splits(self.root)

    def test_missing_frame_and_non_monotonic_time_are_rejected(self):
        frames = self.root / "sequences/sequence_000001/frames.csv"
        lines = frames.read_text().splitlines()
        fields = lines[3].split(",")
        fields[1] = "99"
        lines[3] = ",".join(fields)
        frames.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(ValueError, "missing or duplicate frame"):
            DynamicSequenceDataset(self.root, "train")


if __name__ == "__main__":
    unittest.main()
