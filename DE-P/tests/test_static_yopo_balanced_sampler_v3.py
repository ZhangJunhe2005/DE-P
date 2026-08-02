from __future__ import annotations

from collections import Counter

import pytest

from data.static_yopo_loader_v1 import MapTypeBalancedEpochSamplerV1


class TinyDataset:
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count


def test_balanced_sampler_is_deterministic_and_balanced():
    labels = ["forest"] * 7 + ["room"] * 2 + ["wall"]
    sampler = MapTypeBalancedEpochSamplerV1(TinyDataset(len(labels)), labels, seed=9)
    first = list(sampler)
    assert first == list(sampler)
    counts = Counter(labels[index] for index in first)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert len(first) == len(labels)
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_balanced_sampler_resume_contract_is_strict():
    labels = ["a", "a", "b"]
    sampler = MapTypeBalancedEpochSamplerV1(TinyDataset(3), labels, seed=4)
    sampler.set_epoch(7)
    state = sampler.state_dict()
    restored = MapTypeBalancedEpochSamplerV1(TinyDataset(3), labels, seed=4)
    restored.load_state_dict(state)
    assert restored.epoch == 7
    with pytest.raises(ValueError):
        MapTypeBalancedEpochSamplerV1(
            TinyDataset(3), ["a", "b", "b"], seed=4
        ).load_state_dict(state)
