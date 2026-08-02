"""Unit coverage for the versioned authoritative actor-motion contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from authoritative_dataset.dynamic_motion_v2 import (
    CONTRACT_VERSION, actor_position, build_actor_specs_v2,
    load_motion_contract,
)
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT/"configs/authoritative_dynamic_motion_contract_v2.yaml"
V1_HASH = "a81319d734871c48d49058e22a36a733baead3d85db2ec39206221d3b154f723"
V2_HASH = "56d7118862afc22d8df8672d747c930bc4fdcfcb1fe0795c19e5ceef47989461"


class DynamicMotionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_motion_contract(CONTRACT_PATH)
        identifiers = {
            "crossing": "formal_train_0000009",
            "head_on": "formal_train_0000010",
            "multi_target": "formal_train_0000041",
            "temporal_separation": "formal_train_0000012",
            "occluded_but_tracked": "formal_train_0000058",
            "static_dynamic_joint_constraint": "formal_train_0000029",
        }
        cls.fixtures = {}
        for scenario, identifier in identifiers.items():
            sequence = (
                ROOT/"data/phase8_authoritative_v2/dynamic/train"/identifier
            )
            frames = [
                json.loads(line)
                for line in (sequence/"frames.jsonl").read_text().splitlines()
            ][:20]
            map_root = (
                ROOT/"data/phase8_authoritative_v1/geometry_authority/train"
                /frames[0]["map_uuid"]
            )
            cls.fixtures[scenario] = (
                ExactAuthorityBVH(map_root),
                np.asarray(
                    [row["position_world"] for row in frames],
                    dtype=np.float64,
                ),
            )
        cls.times = np.arange(20, dtype=np.float64)*0.1

    def test_contract_is_hashable_and_frozen_threshold_is_unchanged(self):
        self.assertEqual(self.contract["contract_version"], CONTRACT_VERSION)
        self.assertEqual(
            self.contract["frozen_perception"]["dynamic_enter_speed_mps"],
            0.30,
        )
        self.assertEqual(
            self.contract["_file_hash"],
            hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
        )

    def test_all_scenarios_have_margin_and_nonzero_motion(self):
        required = 0.30 + 0.50
        for scenario, profile in self.contract["scenarios"].items():
            with self.subTest(scenario=scenario):
                self.assertGreaterEqual(profile["speed_range_mps"][0], required)
                self.assertGreater(profile["actor_count"], 0)

    def test_persisted_finite_difference_matches_configured_speed(self):
        for scenario in self.contract["scenarios"]:
            with self.subTest(scenario=scenario):
                actors = build_actor_specs_v2(
                    self.fixtures[scenario][0], np.random.default_rng(8511),
                    scenario, self.fixtures[scenario][1], 0.0,
                    self.times, self.contract,
                )
                self.assertEqual(
                    len(actors),
                    self.contract["scenarios"][scenario]["actor_count"],
                )
                velocities = []
                for actor in actors:
                    positions = actor_position(actor, self.times)
                    finite = np.diff(positions, axis=0)/0.1
                    speed = np.linalg.norm(finite, axis=1)
                    self.assertTrue(np.all(speed >= 0.8-1e-8))
                    self.assertTrue(np.allclose(
                        finite, actor["velocity"], atol=1e-10
                    ))
                    self.assertAlmostEqual(
                        np.linalg.norm(actor["velocity"]),
                        actor["configured_speed_mps"], places=10,
                    )
                    velocities.append(actor["velocity"])
                if scenario == "multi_target":
                    self.assertFalse(np.allclose(velocities[0], velocities[1]))

    def test_immutable_manifest_hashes(self):
        for version, expected in (
            ("v1", V1_HASH), ("v2", V2_HASH)
        ):
            path = ROOT/f"data/phase8_authoritative_{version}/manifests/dataset_manifest.json"
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), expected
            )


if __name__ == "__main__":
    unittest.main()
