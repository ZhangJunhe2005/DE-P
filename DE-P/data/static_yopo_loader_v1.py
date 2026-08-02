"""Deterministic DataLoader construction for StaticYOPODatasetV1."""

from __future__ import annotations

import random
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


class EpochDeterministicSamplerV1(Sampler):
    def __init__(self, dataset, seed=0, shuffle=True):
        self.dataset = dataset
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def state_dict(self):
        return {"seed": self.seed, "epoch": self.epoch, "shuffle": self.shuffle}

    def load_state_dict(self, state):
        if int(state["seed"]) != self.seed or bool(state["shuffle"]) != self.shuffle:
            raise ValueError("sampler contract mismatch")
        self.epoch = int(state["epoch"])

    def __iter__(self):
        if not self.shuffle:
            return iter(range(len(self.dataset)))
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self):
        return len(self.dataset)


class MapTypeBalancedEpochSamplerV1(Sampler):
    """Deterministically give each map type an equal per-epoch sample budget."""

    def __init__(self, dataset, map_types, seed=0):
        if len(map_types) != len(dataset):
            raise ValueError("map type labels must match the dataset length")
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        grouped = defaultdict(list)
        for index, map_type in enumerate(map_types):
            grouped[str(map_type)].append(index)
        if len(grouped) < 2 or any(not values for values in grouped.values()):
            raise ValueError("balanced sampler requires at least two non-empty map types")
        self.groups = {
            key: torch.tensor(values, dtype=torch.long)
            for key, values in sorted(grouped.items())
        }

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def state_dict(self):
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "strategy": "map_type_balanced_v1",
            "group_sizes": {
                key: int(len(values)) for key, values in self.groups.items()
            },
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ("seed", "strategy", "group_sizes"):
            if state.get(key) != expected[key]:
                raise ValueError(f"balanced sampler contract mismatch: {key}")
        self.epoch = int(state["epoch"])

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        group_count = len(self.groups)
        base, remainder = divmod(len(self.dataset), group_count)
        selected = []
        for group_no, (_, indices) in enumerate(self.groups.items()):
            quota = base + int(group_no < remainder)
            chunks = []
            remaining = quota
            while remaining:
                permutation = indices[
                    torch.randperm(len(indices), generator=generator)
                ]
                take = min(remaining, len(permutation))
                chunks.append(permutation[:take])
                remaining -= take
            selected.append(torch.cat(chunks))
        combined = torch.cat(selected)
        combined = combined[
            torch.randperm(len(combined), generator=generator)
        ]
        return iter(combined.tolist())

    def __len__(self):
        return len(self.dataset)


def _seed_worker(worker_id, seed):
    value = (int(seed) + int(worker_id)) % (2**32)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def make_static_yopo_loader_v1(
    dataset, batch_size, seed, num_workers=4, shuffle=True, prefetch_factor=2,
    pin_memory=True, drop_last=False, balance_map_types=None,
):
    if balance_map_types is not None:
        if not shuffle:
            raise ValueError("map-type balancing is only valid for shuffled training")
        sampler = MapTypeBalancedEpochSamplerV1(
            dataset, balance_map_types, seed=seed
        )
    else:
        sampler = EpochDeterministicSamplerV1(
            dataset, seed=seed, shuffle=shuffle
        )
    kwargs = {}
    if num_workers:
        kwargs.update(
            prefetch_factor=int(prefetch_factor),
            persistent_workers=True,
        )
    loader = DataLoader(
        dataset, batch_size=int(batch_size), sampler=sampler,
        num_workers=int(num_workers), pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
        worker_init_fn=partial(_seed_worker, seed=seed),
        **kwargs,
    )
    return loader, sampler
