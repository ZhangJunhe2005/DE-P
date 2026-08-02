"""Explicit, reproducible mixed-training epoch schedule."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch.utils.data import Sampler


@dataclass(frozen=True)
class TrainingScheduleConfig:
    steps_per_epoch: int
    static_batch_ratio: float
    dynamic_passes_per_epoch: float
    allow_loader_recycling: bool

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        fields = set(cls.__dataclass_fields__)
        unknown, missing = set(values) - fields, fields - set(values)
        if unknown or missing:
            raise ValueError(
                f"training_schedule unknown={sorted(unknown)}, missing={sorted(missing)}"
            )
        result = cls(**values)
        result.validate()
        return result

    def validate(self):
        if not isinstance(self.steps_per_epoch, int) or self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be a positive integer")
        if not 0.0 < float(self.static_batch_ratio) < 1.0:
            raise ValueError("static_batch_ratio must be strictly between 0 and 1")
        if not math.isfinite(self.dynamic_passes_per_epoch) or self.dynamic_passes_per_epoch <= 0:
            raise ValueError("dynamic_passes_per_epoch must be finite and positive")
        if not isinstance(self.allow_loader_recycling, bool):
            raise ValueError("allow_loader_recycling must be boolean")

    def batch_counts(self):
        static = round(self.steps_per_epoch * self.static_batch_ratio)
        return {"static": static, "dynamic": self.steps_per_epoch - static}

    def validate_against_dynamic_loader(self, dynamic_batches):
        expected = int(math.ceil(dynamic_batches * self.dynamic_passes_per_epoch))
        actual = self.batch_counts()["dynamic"]
        if not self.allow_loader_recycling and actual > dynamic_batches:
            raise ValueError(
                f"schedule requests {actual} dynamic batches from a {dynamic_batches}-batch "
                "loader while recycling is disabled"
            )
        if actual != expected:
            raise ValueError(
                f"dynamic schedule mismatch: configured {actual} batches, "
                f"dynamic_passes_per_epoch requires {expected}"
            )


def mixed_batch_kinds(steps_per_epoch, static_batch_ratio):
    static_emitted = 0
    result = []
    for step in range(int(steps_per_epoch)):
        target = round((step + 1) * float(static_batch_ratio))
        kind = "static" if static_emitted < target else "dynamic"
        static_emitted += int(kind == "static")
        result.append(kind)
    return result


class EpochSubsetSampler(Sampler[int]):
    """Fixed-size, without-replacement sample selected from seed and epoch."""

    def __init__(self, data_source, num_samples, seed):
        self.data_source = data_source
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0
        if self.num_samples <= 0 or self.num_samples > len(data_source):
            raise ValueError("static sampler size must be in 1..len(dataset)")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 100003)
        return iter(torch.randperm(len(self.data_source), generator=generator)[:self.num_samples].tolist())

    def __len__(self):
        return self.num_samples

    def state_dict(self):
        return {"epoch": self.epoch, "seed": self.seed, "num_samples": self.num_samples}

    def load_state_dict(self, state):
        if int(state["seed"]) != self.seed or int(state["num_samples"]) != self.num_samples:
            raise ValueError("static sampler state/config mismatch")
        self.epoch = int(state["epoch"])
