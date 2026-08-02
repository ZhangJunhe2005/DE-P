"""Static, dynamic, and mixed DEP training with a shared optimization path."""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
from ruamel.yaml import YAML
from rich.progress import Progress
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter

from config.config import cfg
from loss.dynamic_types import DynamicLossConfig, DynamicObjectiveConfig, RiskMetricsConfig
from loss.loss_function import DEPLoss
from policy.backbone_variant import resolve_backbone_variant
from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_dataset import DEPDataset, seed_dataset_worker
from policy.dep_network import DepNetwork
from policy.dynamic.types import DynamicPerceptionConfig
from policy.dynamic_collate import dynamic_sequence_collate
from policy.dynamic_sequence_dataset import DynamicSequenceDataset
from policy.dynamic_training_config import DynamicTrainingConfig
from policy.state_transform import state_body2world
from policy.training_schedule import (
    EpochSubsetSampler, TrainingScheduleConfig, mixed_batch_kinds,
)


DATASET_MODES = ("static", "dynamic", "mixed")
FREEZE_POLICIES = ("none", "late")


def _git_revision(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gradient_norm(parameters):
    gradients = [p.grad.detach() for p in parameters if p.grad is not None]
    if not gradients:
        return torch.tensor(0.0)
    device = gradients[0].device
    return torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(g.to(device), 2) for g in gradients
    ]), 2)


class DepTrainer:
    def __init__(
        self,
        learning_rate=0.0001,
        batch_size=32,
        loss_weight=None,
        tensorboard_path=None,
        checkpoint_path=None,
        save_on_exit=False,
        backbone_variant=None,
        dataset_mode="static",
        dynamic_data_root=None,
        dynamic_context_source=None,
        dynamic_loss_enabled=None,
        dynamic_loss_weight=None,
        freeze_policy="none",
        max_grad_norm=None,
        resume=None,
        random_seed=0,
        num_workers=1,
        training_config_override=None,
        dynamic_loss_config_override=None,
        dynamic_objective_config_override=None,
        risk_metrics_config_override=None,
        map_catalog_override=None,
        static_map_catalog_override=None,
        dynamic_map_catalog_override=None,
        training_schedule_override=None,
        static_cache_size=128,
        head_variant="unified",
        allow_unified_to_split=False,
        freeze_score_branch=False,
        dynamic_perception_config_override=None,
        estimated_cache_perception_config_override=None,
    ):
        if dataset_mode not in DATASET_MODES:
            raise ValueError(f"dataset_mode must be one of {DATASET_MODES}")
        if freeze_policy not in FREEZE_POLICIES:
            raise ValueError(f"freeze_policy must be one of {FREEZE_POLICIES}")
        if checkpoint_path and resume:
            raise ValueError("checkpoint_path initializes weights; resume restores a run. Use only one")
        self.dataset_mode = dataset_mode
        self.batch_size = int(batch_size)
        self.loss_weight = list(loss_weight or [1.0, 1.0])
        self.random_seed = int(random_seed)
        self.backbone_variant = resolve_backbone_variant(backbone_variant)
        self.head_variant = head_variant
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.start_epoch = 0
        self.scheduler = None
        self.scaler = None
        self.current_curriculum = None
        self.traj_num = cfg["traj_num"]
        self.num_workers = int(num_workers)
        self.training_schedule = training_schedule_override
        if self.training_schedule is not None:
            if isinstance(self.training_schedule, dict):
                self.training_schedule = TrainingScheduleConfig.from_mapping(
                    self.training_schedule
                )
            self.training_schedule.validate()

        training_config = training_config_override or DynamicTrainingConfig.from_global_config()
        if dynamic_context_source is not None:
            training_config = replace(training_config, context_source=dynamic_context_source)
        training_config.validate()
        self.dynamic_training_config = training_config
        self.max_grad_norm = (
            training_config.max_grad_norm if max_grad_norm is None else float(max_grad_norm)
        )
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be finite and non-negative")

        perception_config = (
            dynamic_perception_config_override
            or DynamicPerceptionConfig.from_global_config()
        )
        if dataset_mode != "static":
            perception_config = replace(perception_config, enabled=True, use_attention=True)
        perception_config.validate()
        self.dynamic_perception_config = perception_config
        self.estimated_cache_perception_config = (
            estimated_cache_perception_config_override or perception_config
        )

        loss_config = dynamic_loss_config_override or DynamicLossConfig.from_global_config()
        if dynamic_loss_enabled is not None:
            loss_config = replace(loss_config, enabled=bool(dynamic_loss_enabled))
        elif dataset_mode != "static":
            loss_config = replace(loss_config, enabled=True)
        if dynamic_loss_weight is not None:
            loss_config = replace(loss_config, weight=float(dynamic_loss_weight))
        loss_config.validate()
        self.dynamic_loss_config = loss_config
        self.dynamic_objective_config = (
            dynamic_objective_config_override or DynamicObjectiveConfig.from_global_config()
        )
        self.risk_metrics_config = (
            risk_metrics_config_override or RiskMetricsConfig.from_global_config()
        )
        self.dynamic_objective_config.validate()
        self.risk_metrics_config.validate()
        if dataset_mode != "static" and self.backbone_variant != "corrected":
            raise ValueError("dynamic/mixed training requires the corrected backbone variant")

        self.progress_log = Progress()
        base_log = tensorboard_path or str(Path(__file__).resolve().parents[1] / "saved")
        self.tensorboard_path = self.get_next_log_path(base_log)
        self.tensorboard_log = SummaryWriter(log_dir=self.tensorboard_path)
        self._exit_func = atexit.register(self.save_model) if save_on_exit else None

        print(f"Backbone variant: {self.backbone_variant}")
        print(f"Dataset mode: {self.dataset_mode}; device: {self.device}")
        self.policy = DepNetwork(
            backbone_variant=self.backbone_variant,
            dynamic_config=self.dynamic_perception_config,
            head_variant=self.head_variant,
        ).to(self.device)
        if checkpoint_path:
            load_dep_checkpoint(
                self.policy, checkpoint_path, self.backbone_variant,
                allow_unified_to_split=allow_unified_to_split,
            )
        self._apply_freeze_policy(freeze_policy)
        if freeze_score_branch:
            if self.head_variant != "split":
                raise ValueError("freezing only the score branch requires head_variant='split'")
            for parameter in self.policy.dep_head.score_parameters():
                parameter.requires_grad_(False)
        self.score_branch_frozen = bool(freeze_score_branch)
        self.freeze_policy = freeze_policy

        legacy_catalog = map_catalog_override
        static_catalog = static_map_catalog_override
        dynamic_catalog = dynamic_map_catalog_override or legacy_catalog
        if static_catalog is None and dataset_mode in {"static", "mixed"}:
            static_catalog = Path(__file__).resolve().parents[1] / "configs/static_map_catalog.yaml"
        static_catalog = (Path(static_catalog).expanduser().resolve()
                          if static_catalog is not None else None)
        dynamic_catalog = (Path(dynamic_catalog).expanduser().resolve()
                           if dynamic_catalog is not None else None)
        if dynamic_data_root:
            data_root = Path(dynamic_data_root).expanduser().resolve()
            candidate = data_root / "map_catalog.yaml"
            if dynamic_catalog is not None:
                if not dynamic_catalog.is_file():
                    raise FileNotFoundError(dynamic_catalog)
            elif candidate.is_file():
                dynamic_catalog = candidate
            else:
                manifest_path = data_root / "dataset_manifest.yaml"
                if manifest_path.is_file():
                    manifest_data = YAML(typ="safe").load(manifest_path)
                    configured = manifest_data.get("map_catalog")
                    if configured:
                        configured_path = Path(configured).expanduser()
                        dynamic_catalog = (configured_path if configured_path.is_absolute()
                                           else data_root / configured_path).resolve()
        for label, catalog in (("static", static_catalog), ("dynamic", dynamic_catalog)):
            if catalog is not None and not catalog.is_file():
                raise FileNotFoundError(f"{label} map catalog missing: {catalog}")
        self.static_map_catalog = static_catalog
        self.dynamic_map_catalog = dynamic_catalog
        self.static_map_catalog_hash = (
            _file_sha256(static_catalog) if static_catalog is not None else None
        )
        self.dynamic_map_catalog_hash = (
            _file_sha256(dynamic_catalog) if dynamic_catalog is not None else None
        )
        common_loss = dict(
            dynamic_objective_config=self.dynamic_objective_config,
            risk_metrics_config=self.risk_metrics_config,
        )
        self.static_dep_loss = None
        self.dynamic_dep_loss = None
        if dataset_mode in {"static", "mixed"}:
            self.static_dep_loss = DEPLoss(
                dynamic_loss_config=replace(self.dynamic_loss_config, enabled=False),
                map_catalog=static_catalog, **common_loss,
            )
        if dataset_mode in {"dynamic", "mixed"}:
            if dynamic_catalog is None:
                raise ValueError("dynamic training requires an explicit map catalog")
            self.dynamic_dep_loss = DEPLoss(
                dynamic_loss_config=self.dynamic_loss_config,
                map_catalog=dynamic_catalog, **common_loss,
            )
        self.dep_loss = self.dynamic_dep_loss or self.static_dep_loss
        trainable = [parameter for parameter in self.policy.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("freeze policy leaves no trainable parameters")
        fused = self.device.type == "cuda"
        try:
            self.optimizer = torch.optim.AdamW(trainable, lr=learning_rate, fused=fused)
        except (RuntimeError, TypeError):
            fused = False
            self.optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
        self.optimizer_fused = fused
        print(f"AdamW fused: {self.optimizer_fused}; trainable tensors: {len(trainable)}")

        self.dynamic_data_root = (
            str(Path(dynamic_data_root).expanduser().resolve()) if dynamic_data_root else None
        )
        self.train_loaders = {}
        self.val_loaders = {}
        self.validation_suites = {}
        pin = self.device.type == "cuda"
        if dataset_mode in {"static", "mixed"}:
            static_train_dataset = DEPDataset(
                mode="train", cache_size=static_cache_size, global_seed=self.random_seed
            )
            static_valid_dataset = DEPDataset(
                mode="valid", cache_size=static_cache_size, global_seed=self.random_seed
            )
            self.train_loaders["static"] = DataLoader(
                static_train_dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin,
                worker_init_fn=seed_dataset_worker,
            )
            self.val_loaders["static"] = DataLoader(
                static_valid_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin,
                worker_init_fn=seed_dataset_worker,
            )
            self.validation_suites["valid_static"] = self.val_loaders["static"]
        if dataset_mode in {"dynamic", "mixed"}:
            if not self.dynamic_data_root:
                raise ValueError("--dynamic-data-root is required for dynamic/mixed training")
            collate = partial(
                dynamic_sequence_collate,
                max_obstacles=self.dynamic_training_config.max_obstacles,
            )
            dynamic_train_dataset = DynamicSequenceDataset(
                self.dynamic_data_root, "train", training_config=training_config,
                perception_config=self.estimated_cache_perception_config,
            )
            sample_weights = [training_config.sampler_weights[category]
                              for category in dynamic_train_dataset.sample_categories]
            sampler_generator = torch.Generator().manual_seed(self.random_seed)
            dynamic_sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True,
                generator=sampler_generator,
            )
            self.train_loaders["dynamic"] = DataLoader(
                dynamic_train_dataset,
                batch_size=self.batch_size, shuffle=False, sampler=dynamic_sampler,
                num_workers=num_workers,
                pin_memory=pin, collate_fn=collate,
                worker_init_fn=seed_dataset_worker,
            )
            ground_truth_config = replace(training_config, context_source="ground_truth")
            estimated_config = replace(training_config, context_source="estimated")
            self.val_loaders["dynamic"] = DataLoader(
                DynamicSequenceDataset(
                    self.dynamic_data_root, "valid", training_config=ground_truth_config,
                    perception_config=self.estimated_cache_perception_config,
                ),
                batch_size=self.batch_size, shuffle=False, num_workers=num_workers,
                pin_memory=pin, collate_fn=collate,
                worker_init_fn=seed_dataset_worker,
            )
            self.validation_suites["valid_gt"] = self.val_loaders["dynamic"]
            self.validation_suites["valid_estimated"] = DataLoader(
                DynamicSequenceDataset(
                    self.dynamic_data_root, "valid", training_config=estimated_config,
                    perception_config=self.estimated_cache_perception_config,
                ),
                batch_size=self.batch_size, shuffle=False, num_workers=num_workers,
                pin_memory=pin, collate_fn=collate,
                worker_init_fn=seed_dataset_worker,
            )
        if dataset_mode == "mixed" and self.training_schedule is not None:
            self.training_schedule.validate_against_dynamic_loader(
                len(self.train_loaders["dynamic"])
            )
            counts = self.training_schedule.batch_counts()
            static_dataset = self.train_loaders["static"].dataset
            static_sampler = EpochSubsetSampler(
                static_dataset, counts["static"] * self.batch_size, self.random_seed
            )
            self.train_loaders["static"] = DataLoader(
                static_dataset, batch_size=self.batch_size, sampler=static_sampler,
                num_workers=num_workers, pin_memory=pin, drop_last=True,
                worker_init_fn=seed_dataset_worker,
            )
        # Historical aliases remain available to external static tests/callers.
        self.train_dataloader = next(iter(self.train_loaders.values()))
        self.val_dataloader = next(iter(self.val_loaders.values()))
        if resume:
            self.resume_training(resume)

    def _apply_freeze_policy(self, freeze_policy):
        if freeze_policy == "none":
            return
        for parameter in self.policy.image_backbone.parameters():
            parameter.requires_grad_(False)
        features = self.policy.image_backbone.backbone[0]
        for block in features[-3:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in self.policy.image_backbone.backbone[1].parameters():
            parameter.requires_grad_(True)
        for parameter in self.policy.dep_head.parameters():
            parameter.requires_grad_(True)

    def _mixed_epoch_length(self):
        ratio = self.dynamic_training_config.static_batch_ratio
        if self.dataset_mode != "mixed":
            return len(next(iter(self.train_loaders.values())))
        if self.training_schedule is not None:
            return self.training_schedule.steps_per_epoch
        if ratio <= 0 or ratio >= 1:
            raise ValueError("mixed mode requires static_batch_ratio strictly between 0 and 1")
        return max(
            math.ceil(len(self.train_loaders["static"]) / ratio),
            math.ceil(len(self.train_loaders["dynamic"]) / (1.0 - ratio)),
        )

    def _iter_training_batches(self):
        if self.dataset_mode != "mixed":
            kind, loader = next(iter(self.train_loaders.items()))
            yield from ((kind, batch) for batch in loader)
            return
        ratio = (self.training_schedule.static_batch_ratio
                 if self.training_schedule is not None
                 else self.dynamic_training_config.static_batch_ratio)
        iterators = {kind: iter(loader) for kind, loader in self.train_loaders.items()}
        recycle_counts = {"static": 0, "dynamic": 0}
        def next_batch(kind):
            try:
                return next(iterators[kind])
            except StopIteration:
                allowed = (self.training_schedule is None
                           or self.training_schedule.allow_loader_recycling)
                if not allowed:
                    raise RuntimeError(
                        f"{kind} loader exhausted while recycling is disabled"
                    )
                recycle_counts[kind] += 1
                iterators[kind] = iter(self.train_loaders[kind])
                return next(iterators[kind])
        counts = {"static": 0, "dynamic": 0}
        samples = {"static": 0, "dynamic": 0}
        try:
            for kind in mixed_batch_kinds(self._mixed_epoch_length(), ratio):
                batch = next_batch(kind)
                counts[kind] += 1
                samples[kind] += (len(batch[0]) if kind == "static"
                                  else len(batch["current_depth"]))
                yield kind, batch
        finally:
            # Preserve truthful partial-epoch accounting when a managed run is
            # stopped after the current batch.
            dynamic_samples = max(1, len(self.train_loaders["dynamic"].dataset))
            self.last_epoch_schedule_stats = {
                "static_batch_count": counts["static"],
                "dynamic_batch_count": counts["dynamic"],
                "static_sample_count": samples["static"],
                "dynamic_sample_count": samples["dynamic"],
                "static_loader_recycle_count": recycle_counts["static"],
                "dynamic_loader_recycle_count": recycle_counts["dynamic"],
                "dynamic_equivalent_passes": samples["dynamic"] / dynamic_samples,
                "epoch_complete": counts["static"] + counts["dynamic"] == self._mixed_epoch_length(),
            }

    def train(self, epoch, save_interval=None):
        with self.progress_log:
            total_progress = self.progress_log.add_task("Training", total=epoch)
            start_epoch = getattr(self, "start_epoch", 0)
            for self.epoch_i in range(start_epoch, start_epoch + epoch):
                self._apply_curriculum(self.epoch_i)
                self.set_training_mode()
                self.train_one_epoch(self.epoch_i, total_progress)
                self.eval_one_epoch(self.epoch_i)
                if save_interval and (self.epoch_i + 1) % save_interval == 0:
                    self.save_checkpoint(Path(self.tensorboard_path) / f"epoch{self.epoch_i + 1}.pth")
            self.progress_log.remove_task(total_progress)

    def _apply_curriculum(self, epoch):
        # Historical lightweight tests/callers construct a static trainer via
        # __new__ and intentionally do not install dynamic configuration.
        if not hasattr(self, "dynamic_training_config"):
            return
        stages = self.dynamic_training_config.curriculum
        stage = max((item for item in stages if item["start_epoch"] <= epoch),
                    key=lambda item: item["start_epoch"])
        self.current_curriculum = dict(stage)
        loader = self.train_loaders.get("dynamic")
        if loader is not None:
            loader.dataset.set_curriculum(
                epoch, stage["context_source"], stage["ratio"]
            )
        for loader in self.train_loaders.values():
            if hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(epoch)
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)

    def set_training_mode(self):
        """Train unfrozen layers while keeping frozen BatchNorm statistics fixed."""
        self.policy.train()
        if getattr(self, "freeze_policy", "none") != "late":
            return
        for module in self.policy.image_backbone.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                parameters = tuple(module.parameters(recurse=False))
                if parameters and not any(parameter.requires_grad for parameter in parameters):
                    module.eval()

    def train_one_epoch(self, epoch, total_progress):
        total_batches = self._mixed_epoch_length()
        task = self.progress_log.add_task(f"Epoch: {epoch}", total=total_batches)
        for kind, batch in self._iter_training_batches():
            metrics = self.optimize_batch(kind, batch)
            self._log_metrics("Train", metrics, self.global_step)
            self.global_step += 1
            self.progress_log.update(task, advance=1)
            self.progress_log.update(total_progress, advance=1 / max(total_batches, 1))
        self.progress_log.remove_task(task)

    def optimize_batch(self, kind, batch):
        self.optimizer.zero_grad(set_to_none=True)
        details = self.compute_batch(kind, batch)
        total = self.loss_weight[0] * details["trajectory_loss"] + self.loss_weight[1] * details["score_loss"]
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("training loss contains NaN/Inf")
        trainable = [p for p in self.policy.parameters() if p.requires_grad]
        pcgrad = {
            "conflict": total.new_tensor(0.0),
            "activated": total.new_tensor(0.0),
            "dynamic_norm": total.new_tensor(0.0),
            "non_dynamic_norm": total.new_tensor(0.0),
            "other_scale": total.new_tensor(1.0),
            "fallback_no_target": total.new_tensor(0.0),
            "fallback_safe": total.new_tensor(0.0),
            "fallback_small_gradient": total.new_tensor(0.0),
        }
        if (kind == "dynamic" and
                self.dynamic_objective_config.gradient_strategy == "dynamic_priority_pcgrad"):
            dynamic_task = self.loss_weight[0] * details["dynamic_training_objective"]
            non_dynamic_task = (
                self.loss_weight[0] * (
                    details["smooth_loss"] + details["static_safety_loss"]
                    + details["guidance_loss"]
                ) + self.loss_weight[1] * details["score_loss"]
            )
            diagnostics = details.get("dynamic_diagnostics")
            valid_target = bool(
                diagnostics is not None
                and (diagnostics.dynamic_obstacle_count > 0).any()
            )
            hard_risk = bool(
                valid_target and diagnostics is not None
                and (diagnostics.candidate_min_clearance
                     < self.dynamic_objective_config.pcgrad_activation_clearance).any()
            )
            allow_pcgrad = valid_target and (
                not self.dynamic_objective_config.pcgrad_risk_gate or hard_risk
            )
            switch = self.dynamic_objective_config.pcgrad_ratio_switch_step
            ratio = self.dynamic_objective_config.pcgrad_other_norm_ratio
            if switch >= 0 and self.global_step >= switch:
                ratio = self.dynamic_objective_config.pcgrad_late_other_norm_ratio
            pcgrad = self._dynamic_priority_pcgrad(
                dynamic_task, non_dynamic_task, trainable,
                ratio,
                dynamic_grad_epsilon=(
                    self.dynamic_objective_config.pcgrad_dynamic_grad_epsilon
                ),
                allow_pcgrad=allow_pcgrad,
                fallback_no_target=not valid_target,
                fallback_safe=(
                    valid_target and self.dynamic_objective_config.pcgrad_risk_gate
                    and not hard_risk
                ),
                return_diagnostics=True,
            )
        else:
            total.backward()
            pcgrad["non_dynamic_norm"] = _gradient_norm(trainable).to(total.device)
        before = _gradient_norm(trainable)
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, self.max_grad_norm)
        after = _gradient_norm(trainable)
        if not bool(torch.isfinite(before)) or not bool(torch.isfinite(after)):
            raise FloatingPointError("gradient norm contains NaN/Inf")
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
            "pcgrad_applied_ratio": total.new_tensor(ratio if (
                kind == "dynamic" and
                self.dynamic_objective_config.gradient_strategy == "dynamic_priority_pcgrad"
            ) else 1.0).detach(),
            "pcgrad_fallback_no_target": pcgrad["fallback_no_target"].detach(),
            "pcgrad_fallback_safe": pcgrad["fallback_safe"].detach(),
            "pcgrad_fallback_small_gradient": pcgrad["fallback_small_gradient"].detach(),
        })
        return details

    @staticmethod
    def _dynamic_priority_pcgrad(dynamic_task, non_dynamic_task, parameters,
                                 other_norm_ratio=1.0, dynamic_grad_epsilon=0.0,
                                 allow_pcgrad=True, fallback_no_target=False,
                                 fallback_safe=False, accumulate=False,
                                 return_diagnostics=False):
        """Apply risk-gated PCGrad without freezing zero-risk batches.

        When PCGrad is gated off or the dynamic gradient is numerically empty,
        the ordinary non-dynamic gradient is used without a norm cap.  The
        optional ``accumulate`` path adds the computed gradient to an existing
        micro-batch gradient instead of replacing it.
        """
        dynamic_gradients = torch.autograd.grad(
            dynamic_task, parameters, retain_graph=True, allow_unused=True
        )
        other_gradients = torch.autograd.grad(
            non_dynamic_task, parameters, allow_unused=True
        )
        dot = dynamic_task.new_tensor(0.0)
        norm_squared = dynamic_task.new_tensor(0.0)
        other_norm_squared = dynamic_task.new_tensor(0.0)
        for dynamic_gradient, other_gradient in zip(dynamic_gradients, other_gradients):
            if dynamic_gradient is not None:
                norm_squared = norm_squared + dynamic_gradient.square().sum()
                if other_gradient is not None:
                    dot = dot + (dynamic_gradient * other_gradient).sum()
            if other_gradient is not None:
                other_norm_squared = other_norm_squared + other_gradient.square().sum()
        dynamic_norm = norm_squared.sqrt()
        non_dynamic_norm = other_norm_squared.sqrt()
        small_gradient = bool(dynamic_norm <= float(dynamic_grad_epsilon))
        activated = bool(allow_pcgrad and not small_gradient)
        if activated:
            other_scale = (
                dynamic_norm * float(other_norm_ratio)
                / non_dynamic_norm.clamp_min(torch.finfo(dot.dtype).eps)
            ).clamp_max(1.0)
            scaled_dot = dot * other_scale
            conflict = (scaled_dot < 0).to(dynamic_task.dtype)
            coefficient = torch.where(
                scaled_dot < 0,
                scaled_dot / norm_squared.clamp_min(torch.finfo(dot.dtype).eps),
                scaled_dot.new_tensor(0.0),
            )
        else:
            other_scale = dot.new_tensor(1.0)
            conflict = dot.new_tensor(0.0)
            coefficient = dot.new_tensor(0.0)
        for parameter, dynamic_gradient, other_gradient in zip(
                parameters, dynamic_gradients, other_gradients):
            if dynamic_gradient is None and other_gradient is None:
                parameter.grad = None
                continue
            dynamic_value = torch.zeros_like(parameter) if dynamic_gradient is None else dynamic_gradient
            other_value = torch.zeros_like(parameter) if other_gradient is None else other_gradient
            if activated:
                value = dynamic_value + other_value * other_scale - coefficient * dynamic_value
            else:
                value = other_value
            if accumulate and parameter.grad is not None:
                parameter.grad = parameter.grad + value
            else:
                parameter.grad = value
        result = {
            "conflict": conflict,
            "activated": dot.new_tensor(float(activated)),
            "dynamic_norm": dynamic_norm,
            "non_dynamic_norm": non_dynamic_norm,
            "other_scale": other_scale,
            "fallback_no_target": dot.new_tensor(float(fallback_no_target and not activated)),
            "fallback_safe": dot.new_tensor(float(fallback_safe and not activated)),
            "fallback_small_gradient": dot.new_tensor(float(small_gradient and not activated)),
        }
        return result if return_diagnostics else conflict

    @torch.inference_mode()
    def eval_one_epoch(self, epoch):
        self.policy.eval()
        # Preserve the legacy lightweight test path built via __new__.
        if not hasattr(self, "val_loaders"):
            task = self.progress_log.add_task(f"Eval: {epoch}", total=len(self.val_dataloader))
            trajectories, scores = [], []
            for depth, pos, rot, obs, map_id in self.val_dataloader:
                result = self.forward_and_compute_loss(depth, pos, rot, obs, map_id)
                trajectories.append(result[0].item())
                scores.append(result[1].item())
                self.progress_log.update(task, advance=1)
            self.tensorboard_log.add_scalar("Eval/TrajLoss", np.mean(trajectories), epoch)
            self.tensorboard_log.add_scalar("Eval/ScoreLoss", np.mean(scores), epoch)
            self.progress_log.remove_task(task)
            return
        for kind, loader in self.val_loaders.items():
            for batch in loader:
                details = self.compute_batch(kind, batch)
                details["total_loss"] = (
                    self.loss_weight[0] * details["trajectory_loss"]
                    + self.loss_weight[1] * details["score_loss"]
                )
                self._log_metrics(f"Eval/{kind}", details, self.global_step)

    def compute_batch(self, kind, batch):
        if kind == "static":
            depth, pos, rot, obs, map_id = batch
            dynamic_context = dynamic_obstacles = None
            dep_loss = self.static_dep_loss
        elif kind in {"dynamic", "dynamic_gt", "dynamic_estimated"}:
            depth = batch["current_depth"]
            pos = batch["position_world"]
            rot = batch["rotation_world_from_body"]
            obs = batch["observation_9d"]
            map_id = batch["map_id"]
            dynamic_context = batch["dynamic_context"].to(self.device)
            dynamic_obstacles = batch["dynamic_obstacles"].to(self.device)
            dep_loss = self.dynamic_dep_loss
        else:
            raise ValueError(f"unknown batch kind {kind!r}")
        return self._forward_details(
            depth, pos, rot, obs, map_id, dynamic_context, dynamic_obstacles,
            dep_loss=dep_loss,
        )

    def _forward_details(self, depth, pos, rot, obs, map_id,
                         dynamic_context=None, dynamic_obstacles=None, dep_loss=None):
        depth, pos, rot, obs, map_id = [
            value.to(self.device) for value in (depth, pos, rot, obs, map_id)
        ]
        batch_size = depth.shape[0]
        if obs.shape != (batch_size, 9):
            raise ValueError(f"observation must have shape [B,9], got {tuple(obs.shape)}")
        goal_w, start_vel_w, start_acc_w = state_body2world(
            pos, rot, obs[:, 6:9], obs[:, 0:3], obs[:, 3:6]
        )
        start_state_w = torch.stack([pos, start_vel_w, start_acc_w], dim=1)
        endstate, score = self.policy.inference(
            depth, obs, dynamic_context=dynamic_context
        )
        if endstate.shape != (batch_size, 9, cfg["vertical_num"], cfg["horizon_num"]):
            raise ValueError(f"unexpected endstate shape {tuple(endstate.shape)}")
        endstate_flat = endstate.permute(0, 2, 3, 1).reshape(batch_size * self.traj_num, 9)
        score_flat = score.reshape(batch_size * self.traj_num)
        pos_expanded = pos.repeat_interleave(self.traj_num, 0)
        rot_expanded = rot.repeat_interleave(self.traj_num, 0)
        start_expanded = start_state_w.repeat_interleave(self.traj_num, 0)
        goal_expanded = goal_w.repeat_interleave(self.traj_num, 0)
        end_pos, end_vel, end_acc = state_body2world(
            pos_expanded, rot_expanded, endstate_flat[:, :3],
            endstate_flat[:, 3:6], endstate_flat[:, 6:9],
        )
        end_state_w = torch.stack([end_pos, end_vel, end_acc], dim=1)
        selected_loss = dep_loss or self.dep_loss
        output = selected_loss(
            start_expanded, end_state_w, goal_expanded, map_id,
            dynamic_obstacles=dynamic_obstacles, return_details=True,
        )
        label = output.detached_score_label()
        if label.shape != score_flat.shape:
            raise ValueError(f"score label shape {tuple(label.shape)} != {tuple(score_flat.shape)}")
        score_loss = F.smooth_l1_loss(score_flat, label)
        diagnostics = output.dynamic_diagnostics
        zero = output.dynamic_safety_cost.new_tensor(0.0)
        if diagnostics:
            observed = diagnostics.dynamic_obstacle_count > 0
            min_dynamic_distance = (
                diagnostics.min_dynamic_distance[observed].mean()
                if bool(observed.any()) else zero
            )
        else:
            min_dynamic_distance = zero
        return {
            "trajectory_loss": output.trajectory_training_loss,
            "score_loss": score_loss,
            "smooth_loss": output.smooth_cost.mean(),
            "static_safety_loss": output.static_safety_cost.mean(),
            "dynamic_safety_loss": output.dynamic_safety_cost.mean(),
            "raw_dynamic_safety_loss": output.raw_dynamic_safety_cost.mean(),
            "dynamic_training_objective": output.dynamic_training_objective,
            "guidance_loss": output.guidance_cost.mean(),
            "score_label": label,
            "predicted_score": score_flat,
            "candidate_smooth_cost": output.smooth_cost,
            "candidate_static_cost": output.static_safety_cost,
            "candidate_guidance_cost": output.guidance_cost,
            "candidate_dynamic_cost_raw": output.raw_dynamic_safety_cost,
            "candidate_dynamic_cost_weighted": output.dynamic_safety_cost,
            "dynamic_diagnostics": diagnostics,
            "dynamic_score_label": output.dynamic_safety_cost.detach(),
            "min_dynamic_distance": min_dynamic_distance,
            "risky_trajectory_ratio": (
                diagnostics.risky_trajectory_ratio.mean() if diagnostics else zero
            ),
            "dynamic_obstacle_count": (
                diagnostics.dynamic_obstacle_count.float().mean() if diagnostics else zero
            ),
            "safe_candidate_fraction": (
                diagnostics.candidate_safe_mask.float().mean() if diagnostics else zero
            ),
            "high_risk_candidate_fraction": (
                diagnostics.candidate_high_risk_mask.float().mean() if diagnostics else zero
            ),
            "min_dynamic_clearance": (
                diagnostics.candidate_min_clearance[
                    diagnostics.dynamic_obstacle_count > 0
                ].min() if diagnostics and bool((diagnostics.dynamic_obstacle_count > 0).any())
                else zero
            ),
            "future_label_delta": (
                diagnostics.constant_velocity_cost_delta.mean() if diagnostics else zero
            ),
            "dynamic_fallback_ratio": float(self.policy.last_dynamic_fallback_reason is not None),
            "start_state_world_expanded": start_expanded,
            "end_state_world_expanded": end_state_w,
            "batch_map_id": map_id,
        }

    def forward_and_compute_loss(self, depth, pos, rot, obs, map_id):
        """Historical static API retained for regression tests and callers."""
        details = self._forward_details(
            depth, pos, rot, obs, map_id, dep_loss=self.static_dep_loss or self.dep_loss
        )
        return (
            details["trajectory_loss"], details["score_loss"], details["smooth_loss"],
            details["static_safety_loss"], details["guidance_loss"],
        )

    def _log_metrics(self, prefix, metrics, step):
        names = {
            "smooth_loss": "SmoothLoss", "static_safety_loss": "StaticSafetyLoss",
            "dynamic_safety_loss": "DynamicSafetyLoss", "guidance_loss": "GuidanceLoss",
            "score_loss": "ScoreLoss", "total_loss": "TotalLoss",
            "min_dynamic_distance": "MinDynamicDistance",
            "risky_trajectory_ratio": "RiskyTrajectoryRatio",
            "dynamic_obstacle_count": "DynamicObstacleCount",
            "future_label_delta": "RecordedVsConstantVelocityDelta",
            "dynamic_fallback_ratio": "DynamicFallbackRatio",
            "gradient_norm_before_clip": "GradientNormBeforeClip",
            "gradient_norm_after_clip": "GradientNormAfterClip",
            "pcgrad_activated": "PCGradActivated",
            "pcgrad_dynamic_gradient_norm": "PCGradDynamicGradientNorm",
            "pcgrad_non_dynamic_gradient_norm": "PCGradNonDynamicGradientNorm",
            "pcgrad_applied_ratio": "PCGradAppliedRatio",
            "pcgrad_other_scale": "PCGradOtherScale",
            "pcgrad_fallback_no_target": "PCGradFallbackNoTarget",
            "pcgrad_fallback_safe": "PCGradFallbackSafe",
            "pcgrad_fallback_small_gradient": "PCGradFallbackSmallGradient",
            "is_static_batch": "StaticBatchRatio", "is_dynamic_batch": "DynamicBatchRatio",
        }
        for key, tag in names.items():
            if key not in metrics:
                continue
            value = metrics[key]
            value = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            if math.isfinite(value):
                self.tensorboard_log.add_scalar(f"{prefix}/{tag}", value, step)

    def _checkpoint_metadata(self, epoch):
        root = Path(__file__).resolve().parents[1]
        manifest = Path(self.dynamic_data_root or "") / "dataset_manifest.yaml"
        metadata = {
            "backbone_variant": self.backbone_variant,
            "head_variant": self.head_variant,
            "checkpoint_role": "dynamic_training" if self.dataset_mode != "static" else "static_training",
            "dynamic_perception_enabled": self.dynamic_perception_config.enabled,
            "dynamic_perception_config": asdict(self.dynamic_perception_config),
            "dynamic_perception_implementation_version": "phase8f_causal_range_image_v1",
            "dynamic_loss_config": asdict(self.dynamic_loss_config),
            "dynamic_objective_config": asdict(self.dynamic_objective_config),
            "risk_metrics_config": asdict(self.risk_metrics_config),
            "dynamic_training_config": asdict(self.dynamic_training_config),
            "dataset_mode": self.dataset_mode,
            "dataset_version": self.dynamic_training_config.dataset_version if self.dynamic_data_root else None,
            "dataset_manifest_hash": _file_sha256(manifest) if manifest.is_file() else None,
            "static_map_catalog_hash": self.static_map_catalog_hash,
            "dynamic_map_catalog_hash": self.dynamic_map_catalog_hash,
            "static_map_catalog": (
                str(self.static_map_catalog) if self.static_map_catalog else None
            ),
            "dynamic_map_catalog": (
                str(self.dynamic_map_catalog) if self.dynamic_map_catalog else None
            ),
            "training_schedule": (
                asdict(self.training_schedule) if self.training_schedule else None
            ),
            "worker_seed_policy": "global_seed + epoch*100003 + worker_id",
            "dataset_implementation_version": "dep_static_lazy_v1",
            "sensor_source": self.dynamic_perception_config.source,
            "world_frame": self.dynamic_perception_config.world_frame_id,
            "body_frame": self.dynamic_perception_config.body_frame_id,
            "camera_frame": self.dynamic_perception_config.camera_frame_id,
            "simulator_commit": _git_revision(root.parent / "Simulator"),
            "dep_commit": _git_revision(root),
            "training_epoch": int(epoch),
            "global_step": int(self.global_step),
            "random_seed": self.random_seed,
            "freeze_policy": self.freeze_policy,
            "score_branch_frozen": self.score_branch_frozen,
            "optimizer": type(self.optimizer).__name__,
            "resume_rng_state_version": 1,
            "scheduler": None,
            "curriculum_stage": self.current_curriculum,
        }
        # ruamel.yaml scalar subclasses are not accepted by PyTorch's safe
        # weights-only unpickler. Persist metadata as plain JSON types only.
        return json.loads(json.dumps(metadata))

    def save_checkpoint(self, path, metadata_overrides=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self._checkpoint_metadata(getattr(self, "epoch_i", -1) + 1)
        if metadata_overrides:
            metadata.update(json.loads(json.dumps(metadata_overrides)))
        dynamic_loader = getattr(self, "train_loaders", {}).get("dynamic")
        static_loader = getattr(self, "train_loaders", {}).get("static")
        sampler_generator = getattr(
            getattr(dynamic_loader, "sampler", None), "generator", None
        )
        payload = {
            "state_dict": self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "epoch": int(getattr(self, "epoch_i", -1) + 1),
            "global_step": int(self.global_step),
            "metadata": metadata,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "dynamic_sampler_generator_state": (
                sampler_generator.get_state() if sampler_generator is not None else None
            ),
            "static_sampler_state": (
                static_loader.sampler.state_dict()
                if static_loader is not None
                and hasattr(static_loader.sampler, "state_dict") else None
            ),
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(payload, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        digest_path = path.with_suffix(path.suffix + ".sha256")
        digest_temporary = digest_path.with_name(f".{digest_path.name}.{os.getpid()}.tmp")
        digest_temporary.write_text(_file_sha256(path) + "\n", encoding="ascii")
        os.replace(digest_temporary, digest_path)
        return str(path)

    def resume_training(self, path):
        path = Path(path).expanduser().resolve()
        digest_path = path.with_suffix(path.suffix + ".sha256")
        if digest_path.is_file() and digest_path.read_text().strip() != _file_sha256(path):
            raise ValueError("checkpoint checksum mismatch; file is corrupt")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        required = {"state_dict", "optimizer_state", "epoch", "global_step", "metadata"}
        if not required.issubset(payload):
            raise ValueError("resume requires a full training checkpoint, not a legacy weight file")
        metadata = payload["metadata"]
        if metadata.get("backbone_variant") != self.backbone_variant:
            raise ValueError("resume checkpoint backbone variant mismatch")
        if metadata.get("head_variant", "unified") != self.head_variant:
            raise ValueError("resume checkpoint head variant mismatch")
        if metadata.get("dataset_mode") != self.dataset_mode:
            raise ValueError("resume checkpoint dataset mode mismatch")
        current = self._checkpoint_metadata(metadata.get("training_epoch", 0))
        for key in (
            "dataset_manifest_hash", "dynamic_loss_config", "dynamic_objective_config",
            "risk_metrics_config", "dynamic_training_config",
            "static_map_catalog_hash", "dynamic_map_catalog_hash",
            "training_schedule", "worker_seed_policy", "dataset_implementation_version",
        ):
            if metadata.get(key) != current.get(key):
                raise ValueError(f"resume checkpoint {key} mismatch")
        self.policy.load_state_dict(payload["state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if self.scheduler and payload.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(payload["scheduler_state"])
        self.start_epoch = int(payload["epoch"])
        self.global_step = int(payload["global_step"])
        if payload.get("torch_rng_state") is not None:
            torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        dynamic_loader = getattr(self, "train_loaders", {}).get("dynamic")
        sampler_generator = getattr(
            getattr(dynamic_loader, "sampler", None), "generator", None
        )
        sampler_state = payload.get("dynamic_sampler_generator_state")
        if sampler_generator is not None and sampler_state is not None:
            sampler_generator.set_state(sampler_state)
        static_loader = getattr(self, "train_loaders", {}).get("static")
        static_sampler_state = payload.get("static_sampler_state")
        if (static_loader is not None and static_sampler_state is not None
                and hasattr(static_loader.sampler, "load_state_dict")):
            static_loader.sampler.load_state_dict(static_sampler_state)
        return metadata

    def save_model(self):
        if hasattr(self, "epoch_i"):
            self.save_checkpoint(Path(self.tensorboard_path) / f"epoch{self.epoch_i + 1}.pth")
        if self._exit_func is not None:
            atexit.unregister(self._exit_func)
            self._exit_func = None

    def get_next_log_path(self, base_path):
        base = Path(base_path)
        base.mkdir(parents=True, exist_ok=True)
        if self.dataset_mode == "static":
            prefix = "DEP_" if self.backbone_variant == "legacy" else "DEP_corrected_"
        else:
            prefix = f"DEP_{self.dataset_mode}_{self.backbone_variant}_"
        numbers = [
            int(path.name[len(prefix):]) for path in base.iterdir()
            if path.is_dir() and path.name.startswith(prefix)
            and path.name[len(prefix):].isdigit()
        ]
        path = base / f"{prefix}{max(numbers, default=-1) + 1}"
        path.mkdir(exist_ok=False)
        print("record tensorboard log to", path)
        return str(path)
