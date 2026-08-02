from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from ruamel.yaml import YAML

from config.config import cfg
from policy.static_yopo_training_v1 import MixedSceneStaticYOPOObjectiveV1
from tools.train_mixed_static_yopo_v1 import (
    append_jsonl, build_optimizer, build_scheduler, finite_details,
    optimizer_learning_rates, training_implementation_hash,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeLossOutput:
    def __init__(self, count, device):
        self.smooth_cost = torch.ones(count, device=device)
        self.static_safety_cost = torch.ones(count, device=device)
        self.guidance_cost = torch.ones(count, device=device)
        self.dynamic_safety_cost = torch.zeros(count, device=device)
        self.dynamic_training_objective = torch.zeros((), device=device)

    @property
    def trajectory_training_loss(self):
        return (
            self.smooth_cost + self.static_safety_cost + self.guidance_cost
        ).mean()

    def detached_score_label(self):
        return (
            self.smooth_cost + self.static_safety_cost
            + self.guidance_cost + self.dynamic_safety_cost
        ).detach()


class FakeDepLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_dtype = None

    def forward(self, start, end, goal, map_id, **kwargs):
        self.seen_dtype = end.dtype
        return FakeLossOutput(end.shape[0], end.device)


class HalfOutputModel(torch.nn.Module):
    def forward(self, depth, observation):
        batch = depth.shape[0]
        end = torch.zeros(batch, 9, 3, 5, dtype=torch.float16)
        score = torch.zeros(batch, 3, 5, dtype=torch.float16)
        return end, score


def make_objective():
    value = MixedSceneStaticYOPOObjectiveV1.__new__(
        MixedSceneStaticYOPOObjectiveV1
    )
    torch.nn.Module.__init__(value)
    value.dep_loss = FakeDepLoss()
    return value


def test_trajectory_loss_boundary_promotes_half_outputs_to_float32():
    objective = make_objective()
    batch = {
        "depth": torch.zeros(2, 1, 96, 160),
        "observation": torch.zeros(2, 9),
        "position_world": torch.zeros(2, 3),
        "rotation_world_from_body": torch.eye(3).repeat(2, 1, 1),
        "map_id": torch.zeros(2, dtype=torch.long),
    }
    result = objective(HalfOutputModel(), batch)
    assert objective.dep_loss.seen_dtype == torch.float32
    assert result["total_loss"].dtype == torch.float32
    assert finite_details(result)["total_loss"]


def test_corrected_numerical_and_logging_contract():
    config = YAML(typ="safe").load(
        ROOT / "configs/phase8jqv2_5_mixed_static_yopo_training_v3_2.yaml"
    )
    assert config["numerics"]["trajectory_loss_precision"] == "float32"
    assert config["numerics"]["nonfinite_policy"] == "fail_fast"
    assert config["numerics"]["max_consecutive_amp_overflows"] == 8
    assert config["logging"]["progress_every_batches"] == 100
    assert config["validation"]["patience"] == 10
    assert config["optimizer"] == {
        "name": "AdamW",
        "learning_rate": 2e-5,
        "backbone_learning_rate": 5e-6,
        "head_learning_rate": 2e-5,
        "weight_decay": 1e-5,
    }


def test_v32_optimizer_has_disjoint_backbone_and_head_groups():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = torch.nn.Module()
            self.network.image_backbone = torch.nn.Linear(2, 2)
            self.network.dep_head = torch.nn.Linear(2, 2)

    model = TinyModel()
    config = YAML(typ="safe").load(
        ROOT / "configs/phase8jqv2_5_mixed_static_yopo_training_v3_2.yaml"
    )
    optimizer = build_optimizer(model, config)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer_learning_rates(optimizer) == {
        "backbone": 5e-6, "head": 2e-5,
    }
    parameters = [
        value for group in optimizer.param_groups for value in group["params"]
    ]
    assert len(parameters) == len({id(value) for value in parameters})
    assert {id(value) for value in parameters} == {
        id(value) for value in model.parameters()
    }
    assert all(group["weight_decay"] == 1e-5 for group in optimizer.param_groups)


def test_v33_balanced_plateau_contract():
    config = YAML(typ="safe").load(
        ROOT / "configs/phase8jqv2_5_mixed_static_yopo_training_v3_3.yaml"
    )
    assert config["training_implementation_hash"] == training_implementation_hash()
    assert config["loader"]["sampling_strategy"] == "map_type_balanced"
    assert config["validation"]["primary_metric"] == (
        "macro_map_type_total_static_loss"
    )
    assert config["optimizer"]["weight_decay"] == 1e-3
    assert config["scheduler"]["name"] == "ReduceLROnPlateau"
    model = torch.nn.Module()
    model.network = torch.nn.Module()
    model.network.image_backbone = torch.nn.Linear(2, 2)
    model.network.dep_head = torch.nn.Linear(2, 2)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_json_event_writer_rejects_nan(tmp_path):
    with pytest.raises(ValueError):
        append_jsonl(tmp_path / "event.jsonl", {"loss": float("nan")})


def test_root_cause_report_preserves_legacy_failure():
    report = json.load(open(
        ROOT / "reports/phase8jqv2_5_training_convergence_root_cause.json"
    ))
    assert report["legacy_first_failure_replay"]["status"] == "REPRODUCED"
    assert report["legacy_curve"]["nonfinite_train_epochs"] == [0, 19, 26, 42, 49]
    assert report["corrected_500_batch_shakedown"]["status"] == "PASS"
    assert report["resume_legacy_checkpoint_under_corrected_config"] is False
