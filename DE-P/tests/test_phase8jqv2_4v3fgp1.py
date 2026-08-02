import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from authoritative_dataset.dynamic_evidence_v3 import (
    LABELS, ExclusiveRootLock, atomic_json, find_leakage,
    load_numeric_shard, motion_bucket, sampled_constant_velocity,
    save_numeric_shard, stable_seed,
)
from authoritative_dataset.generate_v1 import (
    ACTOR_SAMPLING_HOTFIX, CUDA_PARALLEL_HOTFIX,
    ensure_generation_plan, requires_frozen_dynamic_perception_validation,
    tasks_for,
)
from tools.generate_phase8_dynamic_evidence_formal_v3 import extract_sequence


class V3FGP1Test(unittest.TestCase):
    def test_cuda_parallel_hotfix_is_semantics_preserving(self):
        self.assertFalse(
            CUDA_PARALLEL_HOTFIX["existing_sequence_semantics_changed"])
        self.assertNotEqual(
            CUDA_PARALLEL_HOTFIX["id"], ACTOR_SAMPLING_HOTFIX["id"])

    def test_generation_plan_adds_parallel_hotfix_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "generation_state/generation_plan.json"
            plan.parent.mkdir(parents=True)
            config = {
                "dataset_version": "fixture",
                "_file_hash": "config",
                "frozen_hashes": {
                    "source_hash": "source",
                    "split_manifest_hash": "split",
                },
            }
            initial = {
                "dataset_version": "fixture",
                "protocol_version": "authoritative_dataset_protocol_v1",
                "config_hash": "config",
                "source_hash": "source",
                "split_manifest_hash": "split",
                "renderer_version": "renderer",
                "renderer_required": "cuda",
                "compatible_hotfixes": [ACTOR_SAMPLING_HOTFIX],
            }
            plan.write_text(json.dumps(initial))
            ensure_generation_plan(root, config, "renderer")
            updated = json.loads(plan.read_text())
            self.assertIn(
                CUDA_PARALLEL_HOTFIX["id"],
                {row["id"] for row in updated["compatible_hotfixes"]},
            )

    def test_static_suite_never_enters_dynamic_perception_gate(self):
        config = {"_dynamic_motion_contract": {"contract_version": "v2_1"}}
        self.assertFalse(requires_frozen_dynamic_perception_validation(
            {"suite": "static", "scenario": "brake"}, config))
        self.assertTrue(requires_frozen_dynamic_perception_validation(
            {"suite": "dynamic", "scenario": "crossing"}, config))
        config["evidence_authority_outputs"] = True
        self.assertFalse(requires_frozen_dynamic_perception_validation(
            {"suite": "dynamic", "scenario": "crossing"}, config))

    def test_formal_occlusion_routes_only_to_declared_map_type(self):
        config = {
            "formal_splits": {
                "train": {
                    "frames": 60,
                    "maps": [
                        {"map_uuid": "cave", "semantic_name": "cave_perlin"},
                        {
                            "map_uuid": "forest",
                            "semantic_name": "forest_tree_asset",
                        },
                    ],
                },
            },
            "frames_per_sequence": 60,
            "static_scenarios": [],
            "dynamic_scenarios": ["occluded_but_tracked"],
            "random_seeds": {"sequence_base": 10},
            "formal_occlusion_map_semantic_names": ["forest_tree_asset"],
            "formal_occlusion_map_uuids": {"train": ["forest"]},
        }
        _, tasks = tasks_for(config, "train")
        self.assertEqual(tasks[0]["map_uuid"], "forest")

    def test_formal_dynamic_actor_avoids_infeasible_room_map(self):
        config = {
            "formal_splits": {
                "train": {
                    "frames": 60,
                    "maps": [
                        {"map_uuid": "cave", "semantic_name": "cave_perlin"},
                        {"map_uuid": "room", "semantic_name": "room_window_grid"},
                    ],
                },
            },
            "frames_per_sequence": 60,
            "static_scenarios": [],
            "dynamic_scenarios": ["crossing"],
            "random_seeds": {"sequence_base": 10},
            "formal_dynamic_actor_map_uuids": {"train": ["cave"]},
        }
        _, tasks = tasks_for(config, "train")
        self.assertEqual(tasks[0]["map_uuid"], "cave")

    def test_motion_buckets(self):
        self.assertEqual([motion_bucket(x) for x in (0,.1,.2,.4,.7)],
                         ["ZERO_MOTION","LOW_SUBTHRESHOLD","NEAR_THRESHOLD_NEGATIVE","ACTIVE_DYNAMIC","HIGH_DYNAMIC"])

    def test_sampled_actual_speed(self):
        _, speed=sampled_constant_velocity([0,0,0],[.3,.4,0],60,.1)
        self.assertTrue(np.allclose(speed,.5))

    def test_seed_stable(self):
        self.assertEqual(stable_seed("a",1),stable_seed("a",1))

    def test_seed_changes(self):
        self.assertNotEqual(stable_seed("a",1),stable_seed("a",2))

    def test_leakage_clean(self):
        base=dict(scene_family="x",map_identity="a",map_seed=1,actor_trajectory_family="t",actor_seed=2,renderer_seed=3)
        self.assertFalse(find_leakage([{**base,"split":"train"},{**base,"map_identity":"b","split":"validation"}]))

    def test_leakage_injected(self):
        base=dict(scene_family="x",map_identity="a",map_seed=1,actor_trajectory_family="t",actor_seed=2,renderer_seed=3)
        self.assertEqual(len(find_leakage([{**base,"split":"train"},{**base,"split":"validation"}])),1)

    def test_numeric_no_pickle(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.npz"; h=save_numeric_shard(p,{"x":np.ones((2,3),np.float32)})
            self.assertTrue(np.array_equal(load_numeric_shard(p,h)["x"],np.ones((2,3))))

    def test_object_forbidden(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TypeError):
                save_numeric_shard(Path(d)/"x.npz",{"x":np.asarray([object()],object)})

    def test_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.npz"; save_numeric_shard(p,{"x":np.ones(1)})
            with self.assertRaises(RuntimeError): load_numeric_shard(p,"0"*64)

    def test_atomic_failure(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.json"
            with self.assertRaises(OSError): atomic_json(p,{"x":1},"before_write")
            self.assertFalse(p.exists())

    def test_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as d:
            a=ExclusiveRootLock(d).acquire()
            try:
                with self.assertRaises(RuntimeError): ExclusiveRootLock(d).acquire()
            finally: a.release()

    def test_four_labels(self): self.assertEqual(len(LABELS),4)

    def test_formal_root_is_empty_or_atomic_partial(self):
        root = Path("data/phase8_dynamic_evidence_formal_v3")
        if not root.exists() or not any(root.iterdir()):
            return
        raw = root / "raw_authoritative"
        self.assertTrue(
            (raw / "generation_state/generation_plan.json").is_file())
        self.assertFalse(
            (root / "generation_state/"
             "FORMAL_V3_GENERATION_COMPLETE.json").exists())

    def test_config_training_disabled(self):
        cfg=json.loads(Path("configs/phase8_dynamic_evidence_formal_v3_generation.json").read_text())
        self.assertFalse(cfg["training_enabled"])

    def test_config_no_blind(self):
        cfg=json.loads(Path("configs/phase8_dynamic_evidence_formal_v3_generation.json").read_text())
        self.assertFalse(cfg["test_blind_production_access"])

    def test_validity_per_feature(self):
        report=json.loads(Path("reports/phase8jqv2_4demdcr1_feature_schema.json").read_text())
        self.assertTrue(all(x["validity_mask_required"] for x in report["ordered_fields"]))

    def test_feature_count(self):
        report=json.loads(Path("reports/phase8jqv2_4demdcr1_feature_schema.json").read_text())
        self.assertEqual(len(report["ordered_fields"]),32)

    def test_future_holdout_placeholder(self):
        cfg=json.loads(Path("configs/phase8_dynamic_evidence_formal_v3_generation.json").read_text())
        self.assertTrue(cfg["formal_splits"]["future_holdout"]["placeholder_only"])

    def test_launch_requires_authorization(self):
        text=Path("scripts/phase8jqv2_4_run_formal_v3_generation.sh").read_text()
        self.assertIn("--authorize-formal-generation",text)

    def test_launch_has_no_training_command(self):
        text=Path("scripts/phase8jqv2_4_run_formal_v3_generation.sh").read_text()
        self.assertNotIn("train_dep.py",text)

    def test_actual_extractor_is_causal_numeric_and_actor_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); raw=root/"raw"; out=root/"out"
            base=raw/"dynamic/train/seq"; base.mkdir(parents=True)
            depth=np.full((4,8,8),5.0,np.float32); static=np.full_like(depth,5.0)
            owner=np.full((4,8,8),-1,np.int32)
            owner[:,2:4,2:4]=0; owner[:,4:6,4:6]=1
            depth[owner>=0]=2.0
            np.save(base/"depth.npy",depth,allow_pickle=False)
            np.save(base/"static_depth.npy",static,allow_pickle=False)
            np.save(base/"actor_owner.npy",owner,allow_pickle=False)
            actors=[{"actor_id":0,"velocity_world":[.4,0,0]},
                    {"actor_id":1,"velocity_world":[.7,0,0]}]
            (base/"frames.jsonl").write_text("".join(json.dumps({"actor_metadata":actors})+"\n" for _ in range(4)))
            manifest=root/"manifest.json"
            manifest.write_text(json.dumps({"sequence_id":"seq","split":"train","map_uuid":"m","suite":"dynamic","scenario":"multi_target"}))
            unit=extract_sequence(raw,out,manifest,{})
            metadata=json.loads(unit.read_text())
            with np.load(out/metadata["shard"],allow_pickle=False) as shard:
                self.assertEqual(shard["features"].shape,(4,4,32))
                self.assertTrue(np.all(shard["labels"]==0))
                self.assertTrue(np.all(shard["authority_codes"]==1))


if __name__ == "__main__": unittest.main()
