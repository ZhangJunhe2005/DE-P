"""Bounded Phase 8J-A trainer with a frozen, independent score branch."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from loss.coverage_loss import CandidateCoverageLoss, CoverageObjectiveConfig
from policy.dep_trainer import DepTrainer, _gradient_norm


class Phase8JCoverageTrainer(DepTrainer):
    def __init__(self, *args, coverage_config=None, **kwargs):
        self.coverage_config = coverage_config or CoverageObjectiveConfig()
        if isinstance(self.coverage_config, dict):
            self.coverage_config = CoverageObjectiveConfig.from_mapping(
                self.coverage_config
            )
        self.coverage_config.validate()
        kwargs.update({
            "head_variant": "split",
            "allow_unified_to_split": True,
            "freeze_score_branch": True,
        })
        super().__init__(*args, **kwargs)
        self.coverage_loss = CandidateCoverageLoss(self.coverage_config).to(self.device)

    def _forward_details(self, *args, **kwargs):
        details = super()._forward_details(*args, **kwargs)
        batch = details["batch_map_id"].shape[0]
        selected_loss = kwargs.get("dep_loss") or self.dep_loss
        fixed = details["start_state_world_expanded"].permute(0, 2, 1)
        predicted = details["end_state_world_expanded"].permute(0, 2, 1)
        trajectories, _ = selected_loss.safety_loss.trajectory_sampler.grouped(
            fixed, predicted, batch
        )
        _, static_distance = selected_loss.safety_loss.get_distance_cost(
            trajectories.reshape(batch, -1, 3), details["batch_map_id"]
        )
        static_clearance = static_distance.reshape(batch, 15, -1).amin(2)
        diagnostics = details["dynamic_diagnostics"]
        if diagnostics is None:
            dynamic_cost = trajectories.sum(dim=(2, 3)) * 0.0
            dynamic_clearance = torch.full_like(
                dynamic_cost, torch.finfo(dynamic_cost.dtype).max
            )
            has_target = torch.zeros(batch, dtype=torch.bool, device=self.device)
        else:
            dynamic_cost = diagnostics.candidate_cost
            dynamic_clearance = diagnostics.candidate_min_clearance
            has_target = diagnostics.dynamic_obstacle_count > 0
        coverage = self.coverage_loss(
            dynamic_cost, dynamic_clearance, static_clearance,
            trajectories, has_target,
        )
        details.update({
            "coverage_loss": coverage.total,
            "coverage_dynamic_cvar": coverage.dynamic_cvar,
            "coverage_best_safe": coverage.best_safe,
            "coverage_safe_count": coverage.safe_count,
            "coverage_endpoint_diversity": coverage.endpoint_diversity,
            "coverage_trajectory_diversity": coverage.trajectory_diversity,
            "coverage_soft_safe_count": coverage.soft_safe_count.mean(),
            "candidate_static_clearance": static_clearance,
            "candidate_trajectories_world": trajectories,
            "candidate_joint_violation": coverage.joint_violation,
        })
        return details

    def optimize_batch(self, kind, batch):
        self.optimizer.zero_grad(set_to_none=True)
        details = self.compute_batch(kind, batch)
        non_dynamic = self.loss_weight[0] * (
            details["smooth_loss"] + details["static_safety_loss"]
            + details["guidance_loss"]
        )
        safety = self.loss_weight[0] * (
            details["dynamic_training_objective"] + details["coverage_loss"]
        )
        total = non_dynamic + safety
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("Phase 8J coverage loss contains NaN/Inf")
        trainable = [parameter for parameter in self.policy.parameters()
                     if parameter.requires_grad]
        diagnostics = details.get("dynamic_diagnostics")
        valid_target = bool(
            kind == "dynamic" and diagnostics is not None
            and (diagnostics.dynamic_obstacle_count > 0).any()
        )
        if valid_target:
            pcgrad = self._dynamic_priority_pcgrad(
                safety, non_dynamic, trainable,
                self.dynamic_objective_config.pcgrad_other_norm_ratio,
                dynamic_grad_epsilon=self.dynamic_objective_config.pcgrad_dynamic_grad_epsilon,
                allow_pcgrad=True,
                return_diagnostics=True,
            )
        else:
            total.backward()
            zero = total.new_tensor(0.0)
            pcgrad = {
                "conflict": zero, "activated": zero, "dynamic_norm": zero,
                "non_dynamic_norm": _gradient_norm(trainable).to(total.device),
                "other_scale": total.new_tensor(1.0),
                "fallback_no_target": total.new_tensor(float(kind == "dynamic")),
                "fallback_safe": zero, "fallback_small_gradient": zero,
            }
        before = _gradient_norm(trainable)
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, self.max_grad_norm)
        after = _gradient_norm(trainable)
        if not bool(torch.isfinite(before)) or not bool(torch.isfinite(after)):
            raise FloatingPointError("Phase 8J gradient norm contains NaN/Inf")
        self.optimizer.step()
        details.update({
            "total_loss": total.detach(),
            "gradient_norm_before_clip": before.detach(),
            "gradient_norm_after_clip": after.detach(),
            "is_static_batch": float(kind == "static"),
            "is_dynamic_batch": float(kind == "dynamic"),
            "pcgrad_conflict": pcgrad["conflict"].detach(),
            "pcgrad_activated": pcgrad["activated"].detach(),
            "pcgrad_dynamic_gradient_norm": pcgrad["dynamic_norm"].detach(),
            "pcgrad_non_dynamic_gradient_norm": pcgrad["non_dynamic_norm"].detach(),
            "pcgrad_other_scale": pcgrad["other_scale"].detach(),
            "pcgrad_applied_ratio": total.new_tensor(
                self.dynamic_objective_config.pcgrad_other_norm_ratio
            ).detach(),
            "pcgrad_fallback_no_target": pcgrad["fallback_no_target"].detach(),
            "pcgrad_fallback_safe": pcgrad["fallback_safe"].detach(),
            "pcgrad_fallback_small_gradient": pcgrad["fallback_small_gradient"].detach(),
        })
        return details

    def _checkpoint_metadata(self, epoch):
        metadata = super()._checkpoint_metadata(epoch)
        metadata.update({
            "phase": "8J-A",
            "coverage_objective": asdict(self.coverage_config),
            "score_branch_frozen": True,
        })
        return metadata

    def resume_training(self, path):
        """Reject a resume before mutating state when the 8J-A contract differs."""
        path = Path(path).expanduser().resolve()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload.get("metadata", {})
        if metadata.get("phase") != "8J-A":
            raise ValueError("resume checkpoint is not a Phase 8J-A checkpoint")
        if metadata.get("coverage_objective") != asdict(self.coverage_config):
            raise ValueError("resume checkpoint coverage objective mismatch")
        if metadata.get("score_branch_frozen") is not True:
            raise ValueError("resume checkpoint did not freeze the score branch")
        return super().resume_training(path)
