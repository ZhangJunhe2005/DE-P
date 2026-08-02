import contextlib
import hashlib
import io
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from policy.checkpoint_utils import (
    checkpoint_payload,
    detect_checkpoint_variant,
    load_dep_checkpoint,
)
from policy.dep_network import DepNetwork
from tools.convert_legacy_to_corrected import STRATEGY, convert_checkpoint


ROOT = Path(__file__).resolve().parents[1]
LEGACY_CHECKPOINT = ROOT / "saved/DEP_0/epoch10.pth"


class CheckpointSafetyTests(unittest.TestCase):
    def test_plain_legacy_and_plain_corrected_are_detected(self):
        legacy = torch.load(LEGACY_CHECKPOINT, map_location="cpu", weights_only=True)
        corrected = DepNetwork(backbone_variant="corrected").state_dict()
        self.assertEqual(detect_checkpoint_variant(legacy), "legacy")
        self.assertEqual(detect_checkpoint_variant(corrected), "corrected")

    def test_wrapped_corrected_checkpoint_is_detected_and_strictly_loaded(self):
        model = DepNetwork(backbone_variant="corrected").cpu()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corrected.pth"
            torch.save(checkpoint_payload(model, "corrected", source="unit-test"), path)
            self.assertEqual(detect_checkpoint_variant(torch.load(path, weights_only=True)), "corrected")
            result = load_dep_checkpoint(model, path, "corrected")
        self.assertTrue(result["strict"])
        self.assertEqual(result["missing_keys"], [])
        self.assertEqual(result["unexpected_keys"], [])

    def test_cross_variant_loads_are_rejected(self):
        corrected = DepNetwork(backbone_variant="corrected").cpu()
        with self.assertRaisesRegex(ValueError, "convert_legacy_to_corrected"):
            load_dep_checkpoint(corrected, LEGACY_CHECKPOINT, "corrected")
        legacy = DepNetwork(backbone_variant="legacy").cpu()
        with tempfile.TemporaryDirectory() as tmp:
            corrected_path = Path(tmp) / "corrected.pth"
            torch.save(checkpoint_payload(corrected, "corrected"), corrected_path)
            with self.assertRaisesRegex(ValueError, "variant mismatch"):
                load_dep_checkpoint(legacy, corrected_path, "legacy")

    def test_unknown_and_inconsistent_metadata_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unable to identify"):
            detect_checkpoint_variant({"arbitrary.weight": torch.zeros(1)})
        legacy = torch.load(LEGACY_CHECKPOINT, map_location="cpu", weights_only=True)
        with self.assertRaisesRegex(ValueError, "metadata declares"):
            detect_checkpoint_variant({
                "state_dict": legacy,
                "metadata": {"backbone_variant": "corrected"},
            })


class LegacyConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp_dir.name) / "converted.pth"
        cls.source_hash = hashlib.sha256(LEGACY_CHECKPOINT.read_bytes()).hexdigest()
        with contextlib.redirect_stdout(io.StringIO()):
            cls.metadata = convert_checkpoint(LEGACY_CHECKPOINT, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_source_unchanged_and_conversion_is_auditable(self):
        self.assertEqual(hashlib.sha256(LEGACY_CHECKPOINT.read_bytes()).hexdigest(), self.source_hash)
        self.assertTrue(self.output.is_file())
        self.assertEqual(self.metadata["conversion_strategy"], STRATEGY)
        self.assertEqual(len(self.metadata["copied_keys"]), 241)
        self.assertEqual(len(self.metadata["specially_mapped_keys"]), 6)
        self.assertEqual(len(self.metadata["discarded_keys"]), 5)
        self.assertEqual(self.metadata["initialized_keys"], [])
        self.assertIn("not functionally or mathematically equivalent", self.metadata["warning"])

    def test_converted_checkpoint_strict_load_and_mapping(self):
        payload = torch.load(self.output, map_location="cpu", weights_only=True)
        source = torch.load(LEGACY_CHECKPOINT, map_location="cpu", weights_only=True)
        target = payload["state_dict"]
        self.assertEqual(detect_checkpoint_variant(payload), "corrected")
        self.assertTrue(torch.equal(
            target["image_backbone.backbone.0.0.0.weight"],
            source["image_backbone.backbone.0.0.0.0.weight"],
        ))
        self.assertTrue(torch.equal(
            target["image_backbone.backbone.0.0.1.running_mean"],
            source["image_backbone.backbone.0.0.0.1.running_mean"],
        ))
        shared_key = "image_backbone.backbone.0.5.block.2.fc1.weight"
        self.assertTrue(torch.equal(target[shared_key], source[shared_key]))
        model = DepNetwork(backbone_variant="corrected").cpu().eval()
        result = load_dep_checkpoint(model, self.output, "corrected")
        self.assertTrue(result["strict"])
        with torch.inference_mode():
            endstate, score = model(
                torch.randn(1, 1, 96, 160), torch.randn(1, 9, 3, 5)
            )
        self.assertTrue(torch.isfinite(endstate).all())
        self.assertTrue(torch.isfinite(score).all())

    def test_conversion_difference_is_measured_not_hidden(self):
        fixture = torch.load(
            ROOT / "tests/fixtures/legacy_forward_reference.pt",
            map_location="cpu",
            weights_only=True,
        )
        legacy = DepNetwork(backbone_variant="legacy").cpu().eval()
        load_dep_checkpoint(legacy, LEGACY_CHECKPOINT, "legacy")
        corrected = DepNetwork(backbone_variant="corrected").cpu().eval()
        load_dep_checkpoint(corrected, self.output, "corrected")
        with torch.inference_mode():
            legacy_end, legacy_score = legacy.inference(fixture["depth"], fixture["obs"].clone())
            corrected_end, corrected_score = corrected.inference(fixture["depth"], fixture["obs"].clone())
        metrics = {
            "endstate_mae": float(torch.mean(torch.abs(legacy_end - corrected_end))),
            "endstate_max_abs_diff": float(torch.max(torch.abs(legacy_end - corrected_end))),
            "endstate_cosine_similarity": float(F.cosine_similarity(
                legacy_end.flatten().unsqueeze(0), corrected_end.flatten().unsqueeze(0)
            )),
            "score_mae": float(torch.mean(torch.abs(legacy_score - corrected_score))),
            "score_rank_positions_changed": int(torch.sum(
                torch.argsort(legacy_score.flatten()) != torch.argsort(corrected_score.flatten())
            )),
            "best_primitive_same": bool(
                torch.argmin(legacy_score).item() == torch.argmin(corrected_score).item()
            ),
        }
        print("LEGACY_CORRECTED_CONVERSION_DIFFERENCE", metrics)
        self.assertTrue(all(torch.isfinite(tensor).all() for tensor in (
            legacy_end, legacy_score, corrected_end, corrected_score
        )))
        self.assertGreater(metrics["endstate_max_abs_diff"], 0.0)


if __name__ == "__main__":
    unittest.main()
