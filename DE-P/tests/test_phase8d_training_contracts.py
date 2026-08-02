import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace

from ruamel.yaml import YAML

from loss.safety_loss import SafetyLoss
from policy.dep_trainer import DepTrainer
from policy.training_schedule import (
    EpochSubsetSampler, TrainingScheduleConfig, mixed_batch_kinds,
)


class Phase8DTrainingContractsTest(unittest.TestCase):
    def test_catalog_is_an_exclusive_map_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy"; legacy.mkdir()
            dynamic = root / "dynamic"; dynamic.mkdir()
            (legacy / "pointcloud-0.ply").write_text("legacy")
            dynamic_ply = dynamic / "pointcloud-0.ply"
            dynamic_ply.write_text("dynamic")
            catalog = root / "catalog.yaml"
            with catalog.open("w") as stream:
                YAML().dump({"maps": [{
                    "map_id": 0, "static_ply": str(dynamic_ply),
                }]}, stream)
            loss = SafetyLoss.__new__(SafetyLoss)
            resolved = loss.resolve_map_files(legacy, catalog)
            self.assertEqual(resolved, [(0, str(dynamic_ply.resolve()))])

    def test_mixed_schedule_is_exact_and_does_not_require_recycling(self):
        schedule = TrainingScheduleConfig.from_mapping({
            "steps_per_epoch": 1026,
            "static_batch_ratio": 0.5,
            "dynamic_passes_per_epoch": 1.0,
            "allow_loader_recycling": False,
        })
        schedule.validate_against_dynamic_loader(513)
        kinds = mixed_batch_kinds(1026, 0.5)
        self.assertEqual(kinds.count("static"), 513)
        self.assertEqual(kinds.count("dynamic"), 513)

    def test_schedule_rejects_hidden_dynamic_recycling(self):
        schedule = TrainingScheduleConfig.from_mapping({
            "steps_per_epoch": 11250,
            "static_batch_ratio": 0.5,
            "dynamic_passes_per_epoch": 1.0,
            "allow_loader_recycling": False,
        })
        with self.assertRaisesRegex(ValueError, "recycling is disabled"):
            schedule.validate_against_dynamic_loader(513)

    def test_static_sampler_is_epoch_and_resume_reproducible(self):
        dataset = list(range(100))
        first = EpochSubsetSampler(dataset, 12, seed=81)
        first.set_epoch(3)
        expected = list(first)
        resumed = EpochSubsetSampler(dataset, 12, seed=81)
        resumed.load_state_dict(first.state_dict())
        self.assertEqual(list(resumed), expected)
        resumed.set_epoch(4)
        self.assertNotEqual(list(resumed), expected)

    def test_curriculum_never_mutates_fixed_validation_contexts(self):
        class Dataset:
            def __init__(self):
                self.calls = []
            def set_curriculum(self, *args):
                self.calls.append(args)

        train = Dataset(); valid_gt = Dataset(); valid_estimated = Dataset()
        trainer = DepTrainer.__new__(DepTrainer)
        trainer.dynamic_training_config = SimpleNamespace(curriculum=[{
            "start_epoch": 0, "context_source": "estimated", "ratio": 0.5,
        }])
        trainer.train_loaders = {
            "dynamic": SimpleNamespace(dataset=train, sampler=SimpleNamespace()),
        }
        trainer.val_loaders = {
            "dynamic": SimpleNamespace(dataset=valid_gt),
        }
        trainer.validation_suites = {
            "valid_gt": SimpleNamespace(dataset=valid_gt),
            "valid_estimated": SimpleNamespace(dataset=valid_estimated),
        }
        trainer._apply_curriculum(20)
        self.assertEqual(train.calls, [(20, "estimated", 0.5)])
        self.assertEqual(valid_gt.calls, [])
        self.assertEqual(valid_estimated.calls, [])


if __name__ == "__main__":
    unittest.main()
