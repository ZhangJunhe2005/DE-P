from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from policy.dynamic_sequence_dataset import (
    _canonical_hash,
    require_index_for_nonempty_estimated_cache,
    validate_estimated_cache_index,
)
from tools.generate_phase8i_estimated_cache import phase8h_perception_config


class Phase8JCacheIndexContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "dataset_manifest.yaml"
        self.manifest.write_text("frozen-manifest\n")
        self.config = phase8h_perception_config()

    def tearDown(self):
        self.temporary.cleanup()

    def index(self):
        value = {
            "cache_format_version": "phase8i_phase8h_context_v1",
            "production_test_generated": False,
            "allowed_splits": ["train", "valid"],
            "dataset_manifest_hash": hashlib.sha256(
                self.manifest.read_bytes()
            ).hexdigest(),
            "perception_runtime_config_hash": _canonical_hash(asdict(self.config)),
            "entries": {},
        }
        value["index_content_hash"] = _canonical_hash(value)
        return value

    def test_unindexed_pt_file_is_never_visible(self):
        torch.save({"attention": torch.ones(1, 3, 5)}, self.root / "stale.pt")
        with self.assertRaisesRegex(ValueError, "unindexed"):
            require_index_for_nonempty_estimated_cache(self.root)

    def test_empty_legacy_cache_remains_available_for_test_generation(self):
        self.assertIsNone(require_index_for_nonempty_estimated_cache(self.root))

    def test_runtime_foreground_hash_mismatch_fails(self):
        index = self.index()
        index["perception_runtime_config_hash"] = "0" * 64
        content = dict(index)
        content.pop("index_content_hash")
        index["index_content_hash"] = _canonical_hash(content)
        with self.assertRaisesRegex(ValueError, "foreground mode"):
            validate_estimated_cache_index(index, self.manifest, self.config, "train")

    def test_index_content_hash_mismatch_fails(self):
        index = self.index()
        index["entries"]["untrusted"] = {}
        with self.assertRaisesRegex(ValueError, "index content hash"):
            validate_estimated_cache_index(index, self.manifest, self.config, "train")

    def test_valid_index_passes(self):
        self.assertIs(
            validate_estimated_cache_index(
                self.index(), self.manifest, self.config, "valid"
            ).__class__,
            dict,
        )


if __name__ == "__main__":
    unittest.main()
