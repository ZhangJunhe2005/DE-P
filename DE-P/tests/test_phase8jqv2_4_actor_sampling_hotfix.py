import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from authoritative_dataset.cuda_renderer_v1 import RENDERER_VERSION
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.generate_v1 import (
    ACTOR_SAMPLING_HOTFIX, build_actor_specs, ensure_generation_plan,
    load_config, safe_position, tasks_for,
)
from authoritative_dataset.state_semantics_v2 import sample_uav_sequence


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/phase8_authoritative_v2_generation.yaml"
OLD_HASH = "a81319d734871c48d49058e22a36a733baead3d85db2ec39206221d3b154f723"


class ActorSamplingHotfixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG_PATH)
        _, tasks = tasks_for(cls.config, "train")
        cls.task = tasks[506]
        cls.backend = ExactAuthorityBVH(
            Path(cls.config["authority_source_root"])
            / "geometry_authority/train" / cls.task["map_uuid"]
        )
        cls.times = (
            np.arange(cls.task["frame_count"], dtype=np.float64)
            * cls.config["sensor_settings"]["frame_period_ns"] / 1e9
        )
        rng = np.random.default_rng(cls.task["seed"])
        cls.state = sample_uav_sequence(
            cls.backend, rng, cls.task["scenario"], cls.times, safe_position,
            maximum_attempts=cls.config["certificate_settings"][
                "maximum_attempts_per_window"
            ],
        )
        cls.actors = build_actor_specs(
            cls.backend, rng, cls.task["scenario"],
            cls.state.position_world, float(cls.state.yaw[0]), cls.times,
        )

    def test_known_formal_failure_uses_fallback(self):
        self.assertEqual(self.task["sequence_id"], "formal_train_0000506")
        self.assertEqual(len(self.actors), 2)
        self.assertTrue(all(
            actor["sampling_method"]
            == "deterministic_local_lattice_fallback_v1"
            for actor in self.actors
        ))

    def test_fallback_actor_paths_are_safe_and_separated(self):
        check_times = np.linspace(0, float(self.times[-1] + 1.7), 40)
        paths = []
        for actor in self.actors:
            path = (
                actor["start"][None, :]
                + check_times[:, None] * actor["velocity"]
            )
            paths.append(path)
            self.assertFalse(any(
                self.backend.query_one(point, .2)["collision"]
                for point in path
            ))
        separation = np.linalg.norm(paths[0] - paths[1], axis=1)
        self.assertGreater(float(separation.min()), .4)

    def test_resume_plan_records_compatible_hotfix(self):
        expected = {
            "dataset_version": self.config["dataset_version"],
            "protocol_version": self.config["dataset_protocol_version"],
            "config_hash": self.config["_file_hash"],
            "source_hash": self.config["frozen_hashes"]["source_hash"],
            "split_manifest_hash":
                self.config["frozen_hashes"]["split_manifest_hash"],
            "renderer_version": RENDERER_VERSION,
            "renderer_required": "cuda",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "generation_state/generation_plan.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(expected))
            ensure_generation_plan(root, self.config, RENDERER_VERSION)
            actual = json.loads(path.read_text())
        self.assertIn(ACTOR_SAMPLING_HOTFIX, actual["compatible_hotfixes"])

    def test_v1_manifest_remains_immutable(self):
        manifest = (
            ROOT / "data/phase8_authoritative_v1"
            / "manifests/dataset_manifest.json"
        )
        self.assertEqual(
            hashlib.sha256(manifest.read_bytes()).hexdigest(), OLD_HASH
        )


if __name__ == "__main__":
    unittest.main()
