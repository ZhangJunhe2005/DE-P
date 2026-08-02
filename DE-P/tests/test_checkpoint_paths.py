import tempfile
import unittest
import json
from pathlib import Path

from test_dep_ros import DepNet, parser as ros_parser, resolve_weight_path
from train_dep import resolve_checkpoint_path, resolve_explicit_checkpoint_path


class CheckpointPathTests(unittest.TestCase):
    def test_existing_training_checkpoint_uses_dep_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "DEP_3" / "epoch7.pth"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            resolved = resolve_checkpoint_path(True, 3, 7, tmp)
            self.assertEqual(resolved, str(checkpoint.resolve()))

    def test_missing_explicit_training_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                resolve_checkpoint_path(True, 3, 7, tmp)

    def test_from_scratch_does_not_resolve_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(resolve_checkpoint_path(False, 99, 99, tmp))

    def test_corrected_training_checkpoint_uses_separate_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "DEP_corrected_3" / "epoch7.pth"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            resolved = resolve_checkpoint_path(True, 3, 7, tmp, "corrected")
            self.assertEqual(resolved, str(checkpoint.resolve()))

    def test_explicit_converted_training_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "DEP_corrected_init/epoch10_converted.pth"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            self.assertEqual(
                resolve_explicit_checkpoint_path(checkpoint),
                str(checkpoint.resolve()),
            )

    def test_ros_defaults_resolve_existing_epoch10(self):
        root = Path(__file__).resolve().parents[1]
        args = ros_parser().parse_args([])
        self.assertEqual((args.trial, args.epoch), (0, 10))
        resolved = resolve_weight_path(args.use_tensorrt, args.trial, args.epoch, root)
        self.assertEqual(Path(resolved), (root / "saved/DEP_0/epoch10.pth").resolve())

    def test_missing_ros_checkpoint_lists_available_weights(self):
        root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(FileNotFoundError, "Available PyTorch checkpoints"):
                resolve_weight_path(False, 999, 999, root)

    def test_ros_corrected_checkpoint_uses_separate_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "saved/DEP_corrected_4/epoch2.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            resolved = resolve_weight_path(False, 4, 2, tmp, "corrected")
            self.assertEqual(resolved, str(checkpoint.resolve()))

    def test_corrected_tensorrt_requires_matching_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "dep_trt_corrected.pth"
            artifact.touch()
            with self.assertRaisesRegex(FileNotFoundError, "requires variant metadata"):
                resolve_weight_path(True, 0, 0, tmp, "corrected")
            metadata = Path(str(artifact) + ".metadata.json")
            metadata.write_text(json.dumps({"backbone_variant": "corrected"}), encoding="utf-8")
            self.assertEqual(
                resolve_weight_path(True, 0, 0, tmp, "corrected"),
                str(artifact.resolve()),
            )

    def test_ros_variant_mismatch_fails_before_ros_initialization(self):
        root = Path(__file__).resolve().parents[1]
        config = {"backbone_variant": "corrected", "use_tensorrt": False}
        with self.assertRaisesRegex(ValueError, "convert_legacy_to_corrected"):
            DepNet(config, root / "saved/DEP_0/epoch10.pth")


if __name__ == "__main__":
    unittest.main()
