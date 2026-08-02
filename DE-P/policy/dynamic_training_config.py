"""Strict configuration for sequence loading and dynamic training."""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class DynamicTrainingConfig:
    dataset_version: str
    history_length: int
    window_stride: int
    cache_size: int
    context_source: str
    loss_target_source: str
    max_obstacles: int
    max_grad_norm: float
    static_batch_ratio: float
    noise_seed: int
    position_noise_std: float
    velocity_noise_std: float
    visibility_noise_std: float
    attention_noise_std: float
    estimated_cache_dir: object
    sampler_weights: object
    curriculum: object

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        fields = set(cls.__dataclass_fields__)
        unknown, missing = set(values) - fields, fields - set(values)
        if unknown or missing:
            raise ValueError(
                f"dynamic_training config unknown={sorted(unknown)}, missing={sorted(missing)}"
            )
        result = cls(**values)
        result.validate()
        return result

    @classmethod
    def from_global_config(cls):
        from config.config import cfg
        return cls.from_mapping(cfg["dynamic_training"])

    def validate(self):
        if not self.dataset_version:
            raise ValueError("dataset_version must be non-empty")
        for name in ("history_length", "window_stride", "cache_size", "max_obstacles"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.context_source not in {"ground_truth", "noisy_ground_truth", "estimated"}:
            raise ValueError("invalid context_source")
        if self.loss_target_source not in {"recorded_future_gt", "constant_velocity"}:
            raise ValueError("invalid loss_target_source")
        if not isinstance(self.noise_seed, int):
            raise ValueError("noise_seed must be an integer")
        for name in ("position_noise_std", "velocity_noise_std",
                     "visibility_noise_std", "attention_noise_std"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.estimated_cache_dir is not None and not isinstance(self.estimated_cache_dir, str):
            raise ValueError("estimated_cache_dir must be null or a path string")
        categories = {"no_target", "visible_low_risk", "visible_high_risk",
                      "temporally_separated", "occluded_but_tracked", "multi_target"}
        if set(self.sampler_weights) != categories:
            raise ValueError("sampler_weights must define every training category")
        if any(not np.isfinite(value) or value <= 0 for value in self.sampler_weights.values()):
            raise ValueError("sampler weights must be finite and positive")
        if not isinstance(self.curriculum, list) or not self.curriculum:
            raise ValueError("curriculum must be a non-empty list")
        starts = []
        for stage in self.curriculum:
            if set(stage) != {"start_epoch", "context_source", "ratio"}:
                raise ValueError("curriculum stage schema is invalid")
            if stage["context_source"] not in {"ground_truth", "noisy_ground_truth", "estimated"}:
                raise ValueError("curriculum context source is invalid")
            if not isinstance(stage["start_epoch"], int) or stage["start_epoch"] < 0:
                raise ValueError("curriculum start_epoch is invalid")
            if not np.isfinite(stage["ratio"]) or not 0 < stage["ratio"] <= 1:
                raise ValueError("curriculum ratio is invalid")
            starts.append(stage["start_epoch"])
        if starts != sorted(set(starts)) or starts[0] != 0:
            raise ValueError("curriculum must start at epoch zero with unique sorted boundaries")
        if not np.isfinite(self.max_grad_norm) or self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be finite and non-negative")
        if not np.isfinite(self.static_batch_ratio) or not 0 <= self.static_batch_ratio <= 1:
            raise ValueError("static_batch_ratio must be in [0,1]")
