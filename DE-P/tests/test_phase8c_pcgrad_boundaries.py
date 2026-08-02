from types import SimpleNamespace
import unittest

import torch

from loss.dynamic_types import DynamicObjectiveConfig
from policy.dep_trainer import DepTrainer


def objective(**overrides):
    values = dict(
        mean_coefficient=1.0,
        cvar_coefficient=0.0,
        cvar_fraction=4 / 15,
        reference_scale=1.0,
        gradient_strategy="dynamic_priority_pcgrad",
        pcgrad_other_norm_ratio=0.25,
        pcgrad_risk_gate=True,
        pcgrad_activation_clearance=0.15,
        pcgrad_dynamic_grad_epsilon=1e-12,
        pcgrad_ratio_switch_step=-1,
        pcgrad_late_other_norm_ratio=0.5,
    )
    values.update(overrides)
    return DynamicObjectiveConfig.from_mapping(values)


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable = torch.nn.Parameter(torch.tensor([1.0, -0.5]))
        self.frozen = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)


def make_trainer(kind, *, target_count=0, clearance=1.0, dynamic_scale=0.0,
                 max_grad_norm=10.0, config=None):
    trainer = DepTrainer.__new__(DepTrainer)
    trainer.policy = TinyPolicy()
    trainer.optimizer = torch.optim.SGD(
        [trainer.policy.trainable], lr=0.1, momentum=0.0
    )
    trainer.loss_weight = [1.0, 1.0]
    trainer.max_grad_norm = max_grad_norm
    trainer.global_step = 0
    trainer.dynamic_objective_config = config or objective()
    parameter = trainer.policy.trainable
    diagnostics = SimpleNamespace(
        dynamic_obstacle_count=torch.tensor([target_count]),
        candidate_min_clearance=torch.tensor([[clearance]]),
    )

    def compute_batch(_kind, _batch):
        smooth = (parameter.square()).sum()
        static = 0.5 * (parameter - 0.25).square().sum()
        guidance = 0.25 * parameter.square().sum()
        score = (parameter.sum() - 0.1).square()
        dynamic = dynamic_scale * parameter[0].square()
        return {
            "trajectory_loss": smooth + static + guidance + dynamic,
            "score_loss": score,
            "smooth_loss": smooth,
            "static_safety_loss": static,
            "guidance_loss": guidance,
            "dynamic_training_objective": dynamic,
            "dynamic_diagnostics": diagnostics if kind == "dynamic" else None,
        }

    trainer.compute_batch = compute_batch
    return trainer


class Phase8CPCGradBoundaryTests(unittest.TestCase):
    def test_static_only_batch_updates_and_never_uses_pcgrad(self):
        trainer = make_trainer("static")
        before = trainer.policy.trainable.detach().clone()
        frozen = trainer.policy.frozen.detach().clone()
        metrics = trainer.optimize_batch("static", None)
        self.assertFalse(torch.equal(before, trainer.policy.trainable.detach()))
        self.assertEqual(float(metrics["pcgrad_activated"]), 0.0)
        self.assertGreater(float(metrics["pcgrad_non_dynamic_gradient_norm"]), 0.0)
        self.assertTrue(torch.equal(frozen, trainer.policy.frozen.detach()))

    def test_no_target_dynamic_batch_uses_full_non_dynamic_update(self):
        trainer = make_trainer("dynamic", target_count=0, dynamic_scale=0.0)
        before = trainer.policy.trainable.detach().clone()
        metrics = trainer.optimize_batch("dynamic", None)
        self.assertFalse(torch.equal(before, trainer.policy.trainable.detach()))
        self.assertEqual(float(metrics["pcgrad_dynamic_gradient_norm"]), 0.0)
        self.assertGreater(float(metrics["pcgrad_non_dynamic_gradient_norm"]), 0.0)
        self.assertEqual(float(metrics["pcgrad_fallback_no_target"]), 1.0)
        self.assertEqual(float(metrics["pcgrad_activated"]), 0.0)
        self.assertEqual(float(metrics["pcgrad_other_scale"]), 1.0)

    def test_safe_low_risk_batch_does_not_freeze(self):
        trainer = make_trainer(
            "dynamic", target_count=1, clearance=0.5, dynamic_scale=1e-16
        )
        before = trainer.policy.trainable.detach().clone()
        metrics = trainer.optimize_batch("dynamic", None)
        self.assertFalse(torch.equal(before, trainer.policy.trainable.detach()))
        self.assertEqual(float(metrics["pcgrad_activated"]), 0.0)
        self.assertEqual(float(metrics["pcgrad_fallback_safe"]), 1.0)
        self.assertEqual(float(metrics["pcgrad_fallback_small_gradient"]), 1.0)
        self.assertGreater(float(metrics["pcgrad_non_dynamic_gradient_norm"]), 0.0)

    def test_risk_batch_activates_pcgrad_and_clips_after_surgery(self):
        trainer = make_trainer(
            "dynamic", target_count=1, clearance=-0.1, dynamic_scale=1.0,
            max_grad_norm=0.05,
        )
        metrics = trainer.optimize_batch("dynamic", None)
        self.assertEqual(float(metrics["pcgrad_activated"]), 1.0)
        self.assertGreater(float(metrics["pcgrad_dynamic_gradient_norm"]), 0.0)
        self.assertGreater(float(metrics["gradient_norm_before_clip"]), 0.05)
        self.assertLessEqual(float(metrics["gradient_norm_after_clip"]), 0.050001)

    def test_gradient_accumulation_adds_without_touching_frozen_parameter(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        dynamic = parameter[0].square()
        other = parameter[1].square()
        DepTrainer._dynamic_priority_pcgrad(
            dynamic, other, [parameter], allow_pcgrad=False, accumulate=False
        )
        first = parameter.grad.clone()
        dynamic = parameter[0].square()
        other = 3.0 * parameter[1].square()
        DepTrainer._dynamic_priority_pcgrad(
            dynamic, other, [parameter], allow_pcgrad=False, accumulate=True
        )
        self.assertTrue(torch.allclose(parameter.grad, first + torch.tensor([0.0, 12.0])))

    def test_schedule_is_deterministic_from_resumed_global_step(self):
        trainer = make_trainer(
            "dynamic", target_count=1, clearance=-0.1, dynamic_scale=1.0,
            config=objective(pcgrad_ratio_switch_step=10),
        )
        trainer.global_step = 9
        early = trainer.optimize_batch("dynamic", None)
        trainer.global_step = 10
        late = trainer.optimize_batch("dynamic", None)
        self.assertEqual(float(early["pcgrad_applied_ratio"]), 0.25)
        self.assertEqual(float(late["pcgrad_applied_ratio"]), 0.5)


if __name__ == "__main__":
    unittest.main()
