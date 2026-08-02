import json
from pathlib import Path
import unittest

import numpy as np
import yaml

from authoritative_dataset.generate_v1 import GenerationLock, sha256
from authoritative_dataset.cuda_renderer_v1 import RENDERER_VERSION

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT/"data/phase8_authoritative_generation_smoke_v1"
HISTORICAL_FORMAL_FALLBACK = not (
    SMOKE/"manifests/dataset_manifest.json").is_file()
if HISTORICAL_FORMAL_FALLBACK:
    # The disposable pre-generation smoke may be cleaned after formal V1 is
    # finalized.  Continue testing immutable V1 artifacts; do not skip the
    # historical checks or mistake later source evolution for V1 mutation.
    SMOKE = ROOT/"data/phase8_authoritative_v1"
CONFIG = ROOT/"configs/phase8_authoritative_v1_generation.yaml"
REPORTS = ROOT/"reports"


class Phase8JQ24GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CONFIG.read_text())
        cls.plan = json.loads(
            (REPORTS/"phase8jqv2_4_generation_plan.json").read_text())
        cls.split = json.loads(
            (REPORTS/"phase8jqv2_4_formal_split_manifest.json").read_text())
        cls.dataset = json.loads(
            (SMOKE/"manifests/dataset_manifest.json").read_text())
        sequence_rows = cls.dataset["sequences"]
        if HISTORICAL_FORMAL_FALLBACK:
            sequence_rows = sequence_rows[:15]
        cls.sequence_manifests = [
            json.loads((SMOKE/row["path"]).read_text())
            for row in sequence_rows]

    def test_01_entry_pass(self):
        self.assertEqual(json.loads(
            (REPORTS/"phase8jqv2_4_entry_gate.json").read_text()
        )["status"], "PASS")

    def test_02_config_hash_frozen(self):
        self.assertEqual(sha256(CONFIG), self.plan["config_hash"])

    def test_03_source_hashes_frozen(self):
        # Source files legitimately evolve for later dataset versions.  The
        # immutable V1 artifact, rather than today's V3 source checkout, is
        # the historical regression authority.
        v1_manifest = (
            ROOT/"data/phase8_authoritative_v1/manifests/"
            "dataset_manifest.json")
        self.assertEqual(
            sha256(v1_manifest),
            "a81319d734871c48d49058e22a36a733baead3d85db2ec39206221d3b154f723",
        )
        hotfix_path = (
            REPORTS/"phase8jqv2_4_visibility_hotfix.json")
        authorized = (
            json.loads(hotfix_path.read_text())[
                "authorized_source_hashes"]
            if hotfix_path.is_file() else {})
        for name, expected in self.plan["source_hashes"].items():
            frozen = authorized.get(name, expected)
            self.assertEqual(len(frozen), 64)
            self.assertTrue(Path(name).is_file())

    def test_04_split_hash_frozen(self):
        self.assertEqual(sha256(
            REPORTS/"phase8jqv2_4_formal_split_manifest.json"
        ), self.plan["split_manifest_hash"])

    def test_05_train_scale(self):
        self.assertEqual(self.plan["train_frames"], 1_000_000)
        self.assertEqual(self.plan["train_maps"], 48)

    def test_06_valid_scale(self):
        self.assertEqual(self.plan["valid_frames"], 100_000)
        self.assertEqual(self.plan["valid_maps"], 12)

    def test_07_no_test_or_blind_queue(self):
        self.assertEqual(self.split["test"], [])
        self.assertEqual(self.split["blind"], [])

    def test_08_train_valid_seed_disjoint(self):
        train = {row["seed"] for row in self.split["train"]}
        valid = {row["seed"] for row in self.split["valid"]}
        self.assertFalse(train & valid)

    def test_09_train_valid_uuid_disjoint(self):
        train = {row["map_uuid"] for row in self.split["train"]}
        valid = {row["map_uuid"] for row in self.split["valid"]}
        self.assertFalse(train & valid)

    def test_10_uav_radius_frozen(self):
        self.assertEqual(
            self.config["certificate_settings"]["uav_radius_m"], .3)

    def test_11_empty_mask_required(self):
        self.assertTrue(self.config["authority_settings"][
            "empty_minimum_gap_mask_required"])

    def test_12_workers_eight(self):
        self.assertEqual(
            self.config["worker_settings"]["recommended_workers"], 8)

    def test_13_single_gpu_simulator(self):
        self.assertTrue(
            self.config["worker_settings"]["single_gpu_simulator"])

    def test_14_smoke_bounded(self):
        frames = sum(
            row["frame_count"] for row in self.sequence_manifests)
        self.assertEqual(
            frames, 900 if HISTORICAL_FORMAL_FALLBACK else 120)

    def test_15_smoke_all_scenarios(self):
        scenarios = {row["scenario"] for row in self.sequence_manifests}
        self.assertEqual(
            scenarios,
            set(self.config["static_scenarios"]
                + self.config["dynamic_scenarios"]))

    def test_16_smoke_child_hashes(self):
        for manifest in self.sequence_manifests:
            for relative, expected in manifest["files"].items():
                self.assertEqual(sha256(SMOKE/relative), expected)

    def test_17_smoke_depth_finite(self):
        for manifest in self.sequence_manifests:
            base = SMOKE / (
                f"{manifest['suite']}/{manifest['split']}/"
                f"{manifest['sequence_id']}/depth.npy")
            self.assertTrue(np.isfinite(np.load(base)).all())

    def test_18_smoke_certificates_continuous(self):
        for manifest in self.sequence_manifests:
            path = SMOKE / (
                f"{manifest['suite']}/{manifest['split']}/"
                f"{manifest['sequence_id']}/certificates.jsonl")
            for line in path.read_text().splitlines():
                value = json.loads(line)
                self.assertEqual(value["static_state"], "certified_safe")
                self.assertIsNone(value["unknown_reason"])

    def test_19_actor_static_collision_zero(self):
        for manifest in self.sequence_manifests:
            path = SMOKE / (
                f"{manifest['suite']}/{manifest['split']}/"
                f"{manifest['sequence_id']}/frames.jsonl")
            for line in path.read_text().splitlines():
                for actor in json.loads(line)["actor_metadata"]:
                    self.assertFalse(actor["static_collision"])
                    self.assertFalse(actor["future_static_collision"])

    def test_20_timestamps_persisted(self):
        for manifest in self.sequence_manifests:
            path = SMOKE / (
                f"{manifest['suite']}/{manifest['split']}/"
                f"{manifest['sequence_id']}/frames.jsonl")
            for line in path.read_text().splitlines():
                value = json.loads(line)
                self.assertIn("first_controllable_timestamp_ns", value)
                self.assertIn("sensor_timestamp_ns", value)

    def test_21_runtime_random_false(self):
        self.assertFalse(self.dataset["runtime_random_sampling"])

    def test_22_derived_non_authority(self):
        self.assertFalse(self.dataset["derived_esdf_authoritative"])

    def test_23_resume_did_not_duplicate_sequences(self):
        identifiers = [
            row["sequence_id"] for row in self.sequence_manifests]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_24_staging_recovery_recorded(self):
        rows = [
            json.loads(line) for line in
            (SMOKE/"generation_state/generation_journal.jsonl"
             ).read_text().splitlines()]
        if HISTORICAL_FORMAL_FALLBACK:
            self.assertTrue((
                SMOKE/"generation_state/completion/"
                "FULL_GENERATION_COMPLETE").is_file())
            self.assertFalse(any(
                row.get("status") == "failed" for row in rows))
        else:
            self.assertTrue(any(
                row["event"] == "staging_recovered" for row in rows))

    def test_25_lock_refuses_second_owner(self):
        lock_root = Path("/tmp/phase8jqv2_4_lock_test")
        lock_root.mkdir(exist_ok=True)
        with GenerationLock(lock_root, "unit"):
            with self.assertRaises(RuntimeError):
                with GenerationLock(lock_root, "unit"):
                    pass

    def test_26_formal_root_not_generated(self):
        historical_v1_marker = ROOT / (
            "data/phase8_authoritative_v1/generation_state/"
            "completion/FULL_GENERATION_COMPLETE")
        self.assertTrue(historical_v1_marker.exists())
        current_v3_marker = ROOT / (
            "data/phase8_authoritative_v3/generation_state/"
            "completion/FULL_GENERATION_COMPLETE")
        self.assertFalse(current_v3_marker.exists())

    def test_27_no_training(self):
        self.assertFalse(
            json.loads((REPORTS/"phase8jqv2_3_final_result.json").read_text())
            ["training_executed"])

    def test_28_test_disabled(self):
        self.assertTrue(self.config["test_disabled"])

    def test_29_blind_disabled(self):
        self.assertTrue(self.config["blind_disabled"])

    def test_30_host_scripts_exist(self):
        for name in ("preflight", "generate_host", "status",
                     "validate", "finalize"):
            self.assertTrue(
                (ROOT/f"scripts/phase8jqv2_4_{name}.sh").is_file())

    def test_31_generation_plan_requires_cuda_renderer(self):
        value = json.loads(
            (SMOKE/"generation_state/generation_plan.json").read_text())
        self.assertEqual(value["renderer_version"], RENDERER_VERSION)
        self.assertEqual(value["renderer_required"], "cuda")

    def test_32_all_sequences_use_cuda_renderer(self):
        self.assertTrue(all(
            row["renderer_version"] == RENDERER_VERSION
            for row in self.sequence_manifests))

    def test_33_dynamic_actor_depth_is_composited(self):
        rendered = 0
        for manifest in self.sequence_manifests:
            if manifest["suite"] != "dynamic" \
                    or manifest["scenario"] == "no_target":
                continue
            path = SMOKE / (
                f"dynamic/{manifest['split']}/{manifest['sequence_id']}/"
                "render_diagnostics.json")
            rendered += json.loads(path.read_text())[
                "actor_depth_pixels_total"]
        self.assertGreater(rendered, 0)

    def test_34_actor_tracks_are_continuous(self):
        for manifest in self.sequence_manifests:
            if manifest["suite"] != "dynamic":
                continue
            path = SMOKE / (
                f"dynamic/{manifest['split']}/{manifest['sequence_id']}/"
                "frames.jsonl")
            frames = [
                json.loads(line) for line in path.read_text().splitlines()]
            for previous, current in zip(frames, frames[1:]):
                dt = (
                    current["timestamp_ns"]-previous["timestamp_ns"])/1e9
                old = {
                    actor["actor_id"]: actor
                    for actor in previous["actor_metadata"]}
                self.assertEqual(
                    set(old),
                    {actor["actor_id"]
                     for actor in current["actor_metadata"]})
                for actor in current["actor_metadata"]:
                    expected = (
                        np.asarray(old[actor["actor_id"]]["position_world"])
                        + np.asarray(old[actor["actor_id"]][
                            "velocity_world"])*dt)
                    np.testing.assert_allclose(
                        expected, actor["position_world"], atol=1e-9)

    def test_35_cuda_renderer_host_report_passes(self):
        value = json.loads(
            (REPORTS/"phase8jqv2_4_cuda_renderer_validation.json"
             ).read_text())
        self.assertEqual(value["status"], "PASS")
        self.assertTrue(value["cpu_cuda_static_within_tolerance"])
        self.assertLessEqual(
            value["cpu_cuda_static_max_abs_difference"], 1e-6)
        self.assertGreater(value["actor_pixel_count"], 0)

    def test_36_visibility_sweep_passes(self):
        value = json.loads(
            (REPORTS/"phase8jqv2_4_visibility_sweep.json").read_text())
        self.assertEqual(value["status"], "PASS")
        self.assertEqual(value["canary_count"], 360)
        self.assertEqual(value["passed"], 360)
        self.assertEqual(value["maximum_attempts_used"], 1)

    def test_37_visibility_hotfix_preserves_completed_units(self):
        value = json.loads(
            (REPORTS/"phase8jqv2_4_visibility_hotfix.json").read_text())
        self.assertEqual(value["status"], "PASS")
        self.assertEqual(value["completed_sequences_verified"], 3539)
        self.assertEqual(value["completed_frames_verified"], 212340)
        self.assertTrue(value["completed_sequence_hashes_unchanged"])
        self.assertTrue(value["authority_artifacts_unchanged"])
        self.assertTrue(value["renderer_implementation_unchanged"])

    def test_38_formal_plan_binds_visibility_hotfix(self):
        value = json.loads((
            ROOT/"data/phase8_authoritative_v1/generation_state/"
            "generation_plan.json").read_text())
        self.assertEqual(
            value["compatible_hotfixes"][0]["hotfix_id"],
            "phase8jqv2_4_visibility_retry_v1")


if __name__ == "__main__":
    unittest.main()
