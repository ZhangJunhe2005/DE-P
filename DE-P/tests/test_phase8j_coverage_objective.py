import unittest
from pathlib import Path
import tempfile

import torch

from loss.coverage_loss import CandidateCoverageLoss, CoverageObjectiveConfig
from policy.checkpoint_utils import convert_unified_head_state_dict
from policy.models.head import DepHead
from policy.phase8j_coverage_trainer import Phase8JCoverageTrainer


class CoverageObjectiveTests(unittest.TestCase):
    def config(self, **overrides):
        values = dict(
            enabled=True,
            dynamic_cvar_weight=1.0,
            best_safe_weight=1.0,
            safe_count_weight=1.0,
            endpoint_diversity_weight=1.0,
            trajectory_diversity_weight=1.0,
        )
        values.update(overrides)
        return CoverageObjectiveConfig.from_mapping(values)

    def run_loss(self, has_target):
        risk = torch.rand(2, 15, requires_grad=True)
        dynamic_clearance = torch.full((2, 15), 0.2, requires_grad=True)
        static_clearance = torch.full((2, 15), 0.2, requires_grad=True)
        trajectory = torch.randn(2, 15, 30, 3, requires_grad=True)
        output = CandidateCoverageLoss(self.config())(
            risk, dynamic_clearance, static_clearance, trajectory,
            torch.tensor(has_target),
        )
        return output, (risk, dynamic_clearance, static_clearance, trajectory)

    def test_no_target_dynamic_component_is_exact_zero(self):
        output, _ = self.run_loss([False, False])
        self.assertEqual(output.dynamic_cvar.item(), 0.0)

    def test_all_components_are_finite_and_backward_is_finite(self):
        output, inputs = self.run_loss([True, False])
        output.total.backward()
        self.assertTrue(torch.isfinite(output.total))
        for value in inputs:
            if value.grad is not None:
                self.assertTrue(torch.isfinite(value.grad).all())

    def test_repulsion_is_zero_above_margin(self):
        points = torch.zeros(1, 15, 1, 3)
        points[0, :, 0, 0] = torch.arange(15)
        value = CandidateCoverageLoss._pairwise_repulsion(points, 0.1)
        self.assertEqual(value.item(), 0.0)


class SplitHeadMigrationTests(unittest.TestCase):
    def test_explicit_split_is_numerically_exact(self):
        torch.manual_seed(8)
        unified = DepHead(73, 10, variant="unified").eval()
        split = DepHead(73, 10, variant="split").eval()
        migrated = convert_unified_head_state_dict({
            f"dep_head.{key}": value for key, value in unified.state_dict().items()
        })
        split.load_state_dict({
            key.removeprefix("dep_head."): value for key, value in migrated.items()
        }, strict=True)
        fixture = torch.randn(2, 73, 3, 5)
        self.assertTrue(torch.equal(unified(fixture), split(fixture)))

    def test_score_and_candidate_parameters_are_disjoint(self):
        head = DepHead(73, 10, variant="split")
        score = {id(value) for value in head.score_parameters()}
        candidate = {id(value) for value in head.candidate_parameters()}
        self.assertFalse(score & candidate)


class CoverageResumeContractTests(unittest.TestCase):
    def test_resume_rejects_coverage_config_mismatch_before_state_load(self):
        trainer = Phase8JCoverageTrainer.__new__(Phase8JCoverageTrainer)
        trainer.coverage_config = CoverageObjectiveConfig()
        mismatched = CoverageObjectiveConfig(dynamic_cvar_weight=3.0)
        payload = {
            "metadata": {
                "phase": "8J-A",
                "coverage_objective": vars(mismatched),
                "score_branch_frozen": True,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "resume.pt"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "coverage objective mismatch"):
                trainer.resume_training(checkpoint)


if __name__ == "__main__":
    unittest.main()
