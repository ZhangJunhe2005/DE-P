from pathlib import Path

import torch

from policy.static_yopo_training_v1 import MixedSceneStaticYOPOV1
from tools.train_mixed_static_yopo_v1 import (
    apply_trainable_group_contract,
    apply_score_only_warmup,
    build_optimizer,
)


def test_candidate_only_contract_freezes_backbone_and_score_tower():
    model = MixedSceneStaticYOPOV1(head_variant="independent")
    config = {
        "optimizer": {
            "name": "AdamW", "learning_rate": 5.0e-7,
            "backbone_learning_rate": 0.0,
            "candidate_head_learning_rate": 5.0e-7,
            "score_head_learning_rate": 0.0,
            "weight_decay": 1.0e-5,
        },
        "training": {"trainable_groups": "candidate_head_only"},
    }
    optimizer = build_optimizer(model, config)
    model.train()
    assert apply_trainable_group_contract(optimizer, config, model) == (
        "candidate_head_only"
    )
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert groups["backbone"]["lr"] == 0.0
    assert groups["candidate_head"]["lr"] == 5.0e-7
    assert groups["score_head"]["lr"] == 0.0
    assert all(not p.requires_grad for p in groups["backbone"]["params"])
    assert all(p.requires_grad for p in groups["candidate_head"]["params"])
    assert all(not p.requires_grad for p in groups["score_head"]["params"])
    assert model.network.image_backbone.training is False
    assert model.network.dep_head.score_model.training is False


def test_score_only_contract_freezes_backbone_and_candidate_tower():
    model = MixedSceneStaticYOPOV1(head_variant="independent")
    config = {
        "optimizer": {
            "name": "AdamW", "learning_rate": 5.0e-7,
            "backbone_learning_rate": 0.0,
            "candidate_head_learning_rate": 0.0,
            "score_head_learning_rate": 5.0e-7,
            "weight_decay": 1.0e-5,
        },
        "training": {"trainable_groups": "score_head_only"},
    }
    optimizer = build_optimizer(model, config)
    model.train()
    assert apply_trainable_group_contract(optimizer, config, model) == (
        "score_head_only"
    )
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert groups["backbone"]["lr"] == 0.0
    assert groups["candidate_head"]["lr"] == 0.0
    assert groups["score_head"]["lr"] == 5.0e-7
    assert all(not p.requires_grad for p in groups["backbone"]["params"])
    assert all(not p.requires_grad for p in groups["candidate_head"]["params"])
    assert all(p.requires_grad for p in groups["score_head"]["params"])
    assert model.network.image_backbone.training is False
    assert model.network.dep_head.trajectory_model.training is False
    assert model.network.dep_head.score_model.training is True


def test_persistent_score_only_step_preserves_candidate_outputs():
    torch.manual_seed(84804)
    model = MixedSceneStaticYOPOV1(head_variant="independent")
    config = {
        "optimizer": {
            "name": "AdamW", "learning_rate": 5.0e-7,
            "backbone_learning_rate": 0.0,
            "candidate_head_learning_rate": 0.0,
            "score_head_learning_rate": 5.0e-7,
            "weight_decay": 1.0e-5,
        },
        "training": {"trainable_groups": "score_head_only"},
    }
    optimizer = build_optimizer(model, config)
    model.train()
    apply_trainable_group_contract(optimizer, config, model)
    depth = torch.rand(2, 1, 96, 160)
    observation = torch.randn(2, 9)
    with torch.no_grad():
        proposal_before, score_before = model(depth, observation)
    optimizer.zero_grad(set_to_none=True)
    _, score = model(depth, observation)
    score.square().mean().backward()
    optimizer.step()
    with torch.no_grad():
        proposal_after, score_after = model(depth, observation)
    assert torch.equal(proposal_before, proposal_after)
    assert not torch.equal(score_before, score_after)


def test_formal_model_unified_to_split_migration_is_exact(tmp_path: Path):
    torch.manual_seed(82701)
    unified = MixedSceneStaticYOPOV1(head_variant="unified").eval()
    checkpoint = tmp_path / "unified.pth"
    torch.save(unified.network.state_dict(), checkpoint)
    split = MixedSceneStaticYOPOV1(
        checkpoint, head_variant="split", allow_unified_to_split=True
    ).eval()
    depth = torch.rand(2, 1, 96, 160)
    observation = torch.randn(2, 9)
    with torch.inference_mode():
        expected = unified(depth, observation)
        actual = split(depth, observation)
    assert split.initial_checkpoint_result["head_migrated"]
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1], actual[1])


def test_formal_model_unified_to_independent_migration_is_exact(tmp_path: Path):
    torch.manual_seed(84541)
    unified = MixedSceneStaticYOPOV1(head_variant="unified").eval()
    checkpoint = tmp_path / "unified.pth"
    torch.save(unified.network.state_dict(), checkpoint)
    independent = MixedSceneStaticYOPOV1(
        checkpoint, head_variant="independent", allow_unified_to_split=True
    ).eval()
    depth = torch.rand(2, 1, 96, 160)
    observation = torch.randn(2, 9)
    with torch.inference_mode():
        expected = unified(depth, observation)
        actual = independent(depth, observation)
    assert independent.initial_checkpoint_result["head_migrated"]
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1], actual[1])

    formal_checkpoint = tmp_path / "independent_formal.pth"
    torch.save({
        "checkpoint_version": "mixed_scene_static_yopo_checkpoint_v1",
        "model": independent.state_dict(),
    }, formal_checkpoint)
    restored = MixedSceneStaticYOPOV1(
        formal_checkpoint, head_variant="independent"
    ).eval()
    with torch.inference_mode():
        restored_output = restored(depth, observation)
    assert not restored.initial_checkpoint_result["head_migrated"]
    assert torch.equal(actual[0], restored_output[0])
    assert torch.equal(actual[1], restored_output[1])


def test_split_optimizer_has_disjoint_learning_rate_groups():
    model = MixedSceneStaticYOPOV1(head_variant="split")
    optimizer = build_optimizer(model, {
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 1.0e-5,
            "backbone_learning_rate": 1.0e-7,
            "candidate_head_learning_rate": 3.0e-6,
            "score_head_learning_rate": 1.0e-5,
            "weight_decay": 5.0e-4,
        }
    })
    assert [group["group_name"] for group in optimizer.param_groups] == [
        "backbone", "candidate_head", "score_head"
    ]
    parameter_ids = [
        id(value)
        for group in optimizer.param_groups
        for value in group["params"]
    ]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert set(parameter_ids) == {id(value) for value in model.parameters()}


def test_independent_optimizer_assigns_whole_score_tower_to_score_group():
    model = MixedSceneStaticYOPOV1(head_variant="independent")
    optimizer = build_optimizer(model, {
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 1.0e-5,
            "backbone_learning_rate": 1.0e-7,
            "candidate_head_learning_rate": 3.0e-6,
            "score_head_learning_rate": 5.0e-5,
            "weight_decay": 1.0e-5,
        }
    })
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    score_ids = {id(value) for value in groups["score_head"]["params"]}
    expected_score_ids = {
        id(value) for value in model.network.dep_head.score_parameters()
    }
    assert score_ids == expected_score_ids
    all_ids = [
        id(value) for group in optimizer.param_groups for value in group["params"]
    ]
    assert len(all_ids) == len(set(all_ids))
    assert set(all_ids) == {id(value) for value in model.parameters()}


def test_score_only_warmup_freezes_then_restores_proposal_groups():
    model = MixedSceneStaticYOPOV1(head_variant="split")
    config = {
        "optimizer": {
            "name": "AdamW", "learning_rate": 1.0e-5,
            "backbone_learning_rate": 1.0e-7,
            "candidate_head_learning_rate": 1.0e-6,
            "score_head_learning_rate": 1.0e-5,
            "score_only_learning_rate": 7.0e-5,
            "weight_decay": 2.0e-4,
        },
        "training": {"score_only_warmup_epochs": 3},
    }
    optimizer = build_optimizer(model, config)
    assert apply_score_only_warmup(optimizer, config, 0, 0) \
        == "score_only_warmup_start"
    rates = {group["group_name"]: group["lr"]
             for group in optimizer.param_groups}
    assert rates == {
        "backbone": 0.0, "candidate_head": 0.0, "score_head": 7.0e-5,
    }
    frozen = {
        group["group_name"]: all(
            not value.requires_grad for value in group["params"]
        )
        for group in optimizer.param_groups
    }
    assert frozen == {
        "backbone": True, "candidate_head": True, "score_head": False,
    }
    assert apply_score_only_warmup(optimizer, config, 3, 0) \
        == "joint_finetune_start"
    rates = {group["group_name"]: group["lr"]
             for group in optimizer.param_groups}
    assert rates == {
        "backbone": 1.0e-7,
        "candidate_head": 1.0e-6,
        "score_head": 1.0e-5,
    }
    assert all(
        value.requires_grad
        for group in optimizer.param_groups for value in group["params"]
    )


def test_score_only_resume_reestablishes_parameter_freeze():
    model = MixedSceneStaticYOPOV1(head_variant="split")
    config = {
        "optimizer": {
            "name": "AdamW", "learning_rate": 1.0e-5,
            "backbone_learning_rate": 1.0e-7,
            "candidate_head_learning_rate": 1.0e-6,
            "score_head_learning_rate": 1.0e-5,
            "weight_decay": 2.0e-4,
        },
        "training": {"score_only_warmup_epochs": 8},
    }
    optimizer = build_optimizer(model, config)
    assert apply_score_only_warmup(optimizer, config, 4, 4) \
        == "score_only_warmup_resume"
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert groups["backbone"]["lr"] == 0.0
    assert groups["candidate_head"]["lr"] == 0.0
    assert all(not value.requires_grad for value in groups["backbone"]["params"])
    assert all(not value.requires_grad for value in groups["candidate_head"]["params"])


def test_score_only_optimizer_step_changes_only_score_parameters():
    torch.manual_seed(84531)
    model = MixedSceneStaticYOPOV1(head_variant="split")
    config = {
        "optimizer": {
            "name": "AdamW", "learning_rate": 2.0e-5,
            "backbone_learning_rate": 5.0e-7,
            "candidate_head_learning_rate": 5.0e-6,
            "score_head_learning_rate": 2.0e-5,
            "score_only_learning_rate": 1.0e-4,
            "weight_decay": 1.0e-5,
        },
        "training": {"score_only_warmup_epochs": 3},
    }
    optimizer = build_optimizer(model, config)
    apply_score_only_warmup(optimizer, config, 0, 0)
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    before = {
        id(parameter): parameter.detach().clone()
        for group in optimizer.param_groups for parameter in group["params"]
    }
    depth = torch.rand(2, 1, 96, 160)
    observation = torch.randn(2, 9)
    _, score = model(depth, observation)
    score.square().mean().backward()
    optimizer.step()
    for name in ("backbone", "candidate_head"):
        assert all(
            torch.equal(parameter, before[id(parameter)])
            for parameter in groups[name]["params"]
        )
    assert any(
        not torch.equal(parameter, before[id(parameter)])
        for parameter in groups["score_head"]["params"]
    )


def test_independent_score_only_step_preserves_proposal_and_batchnorm_buffers():
    torch.manual_seed(84542)
    model = MixedSceneStaticYOPOV1(head_variant="independent")
    config = {
        "optimizer": {
            "name": "AdamW", "learning_rate": 5.0e-5,
            "backbone_learning_rate": 5.0e-7,
            "candidate_head_learning_rate": 5.0e-6,
            "score_head_learning_rate": 5.0e-5,
            "score_only_learning_rate": 5.0e-5,
            "weight_decay": 1.0e-5,
        },
        "training": {"score_only_warmup_epochs": 5},
    }
    optimizer = build_optimizer(model, config)
    model.train()
    apply_score_only_warmup(optimizer, config, 0, 0, model=model)
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    frozen_before = {
        id(parameter): parameter.detach().clone()
        for name in ("backbone", "candidate_head")
        for parameter in groups[name]["params"]
    }
    score_before = {
        id(parameter): parameter.detach().clone()
        for parameter in groups["score_head"]["params"]
    }
    bn_before = {
        name: buffer.detach().clone()
        for name, buffer in model.network.image_backbone.named_buffers()
        if "running_mean" in name or "running_var" in name
    }
    depth = torch.rand(2, 1, 96, 160)
    observation = torch.randn(2, 9)
    _, score = model(depth, observation)
    score.square().mean().backward()
    optimizer.step()
    for name in ("backbone", "candidate_head"):
        assert all(
            torch.equal(parameter, frozen_before[id(parameter)])
            for parameter in groups[name]["params"]
        )
    assert any(
        not torch.equal(parameter, score_before[id(parameter)])
        for parameter in groups["score_head"]["params"]
    )
    for name, buffer in model.network.image_backbone.named_buffers():
        if name in bn_before:
            assert torch.equal(buffer, bn_before[name])
