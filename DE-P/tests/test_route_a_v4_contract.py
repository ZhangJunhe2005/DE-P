from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from authoritative_dataset.generate_v1 import load_config, tasks_for
from policy.static_yopo_wide_state_v1 import sample_wide_observation
from tools import build_route_a_v4_static_dataset as builder
from tools.train_mixed_static_yopo_v1 import source_identity_fields
from policy.static_yopo_training_v1 import local_planning_goal


def test_v4_map_profiles_preserve_large_and_narrow_five_type_coverage():
    document = yaml.safe_load(
        open("configs/mixed_scene_map_profiles_v4_route_a.yaml")
    )
    profiles = list(document["profiles"].values())
    for size_class in ("large", "narrow"):
        selected = [row for row in profiles if row["size_class"] == size_class]
        assert {row["maze_type"] for row in selected} == {1, 2, 5, 6, 7}
    for row in profiles:
        if row["size_class"] == "large":
            assert row["x_length"] >= 60
            assert row["y_length"] >= 60
            assert row["z_length"] >= 15


def test_static_and_dynamic_actor_contracts_are_not_conflated():
    contract = yaml.safe_load(open("configs/route_a_v4_training_contract.yaml"))
    assert contract["static_policy_training"]["actor_count"] == 0
    assert contract["dynamic_closed_loop_evaluation"]["actor_count_tiers"] == [
        0, 1, 2, 4, 8
    ]


def test_wide_state_sampler_is_deterministic_and_bounded():
    first = sample_wide_observation("sample-a", "train", 82501)
    second = sample_wide_observation("sample-a", "train", 82501)
    assert np.array_equal(first, second)
    assert np.linalg.norm(first[:3]) <= 6.0 + 1e-5
    assert np.linalg.norm(first[3:6]) <= 6.0 + 1e-5
    assert 10.0 <= np.linalg.norm(first[6:9]) <= 40.0


def test_wide_state_sampler_covers_all_operating_bands():
    values = np.stack([
        sample_wide_observation(f"sample-{index}", "train", 82501)
        for index in range(2000)
    ])
    speed = np.linalg.norm(values[:, :3], axis=1)
    acceleration = np.linalg.norm(values[:, 3:6], axis=1)
    goal = np.linalg.norm(values[:, 6:9], axis=1)
    assert (speed < 0.5).any()
    assert ((speed >= 2.0) & (speed < 4.5)).any()
    assert (speed >= 4.5).any()
    assert (acceleration >= 4.0).any()
    assert goal.min() >= 10.0 - 1e-4
    assert goal.max() <= 40.0 + 1e-4
    assert torch.isfinite(torch.from_numpy(values)).all()


def test_authoritative_generator_accepts_static_only_v4_contract(tmp_path):
    config = yaml.safe_load(
        open("configs/phase8_authoritative_v3_mixed_generation.yaml")
    )
    config["dataset_version"] = "route_a_v4_raw_static_v1"
    config["dynamic_scenarios"] = []
    config["actor_distribution"] = {
        "maximum_actors": 0,
        "policy": "static_policy_training_actor_free",
    }
    config["formal_splits"]["train"]["frames"] = 120
    config["formal_splits"]["train"]["maps"] = [{
        "map_id": "train_map_0000",
        "map_uuid": "00000000-0000-0000-0000-000000000001",
        "seed": 1,
        "maze_type": 1,
        "semantic_name": "cave_perlin",
        "profile_name": "cave_large_v4",
        "profile_hash": "fixture",
    }]
    path = tmp_path / "v4.yaml"
    path.write_text(yaml.safe_dump(config))
    loaded = load_config(path)
    _, tasks = tasks_for(loaded, "train")
    assert tasks
    assert {row["suite"] for row in tasks} == {"static"}


def test_v4_resolved_config_preserves_raw_generator_timing_contract():
    base = yaml.safe_load(
        open("configs/phase8_authoritative_v3_mixed_generation.yaml")
    )
    assert base["state_sampling_contract"]["command_latency_s"] == 0.04
    source = open("tools/prepare_route_a_v4_raw_config.py").read()
    assert 'dict(value["state_sampling_contract"])' in source
    assert '"command_latency_s"' in source


def test_v4_static_builder_accepts_only_proven_actor_free_depth_fallback(
    tmp_path, monkeypatch
):
    source = tmp_path / "raw"
    sequence_id = "formal_train_0000000"
    relative = Path("static/train") / sequence_id
    root = source / relative
    root.mkdir(parents=True)
    depth = np.ones((2, 96, 160), dtype=np.float32)
    np.save(root / "depth.npy", depth, allow_pickle=False)
    semantic_hash = hashlib.sha256(depth.tobytes()).hexdigest()
    (root / "render_diagnostics.json").write_text(json.dumps({
        "actor_count": 0,
        "actor_depth_pixels_total": 0,
        "frames_with_actor_depth": 0,
        "static_depth_semantic_hash": semantic_hash,
        "composed_depth_semantic_hash": semantic_hash,
    }))
    frames = [
        {"dynamic_depth_composited": False, "actor_metadata": []}
        for _ in range(2)
    ]
    (root / "frames.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in frames)
    )
    manifest_dir = source / "manifests/sequences"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / f"{sequence_id}.json").write_text(
        json.dumps({
            "files": {f"{relative}/depth.npy": "fixture"},
        })
    )
    monkeypatch.setattr(builder, "SOURCE", source)
    row = {
        "suite": "static",
        "split": "train",
        "sequence_id": sequence_id,
        "frame_count": 2,
    }
    assert builder.static_depth_path(row) == root / "depth.npy"
    frames[0]["dynamic_depth_composited"] = True
    (root / "frames.jsonl").write_text(
        "".join(json.dumps(value) + "\n" for value in frames)
    )
    with pytest.raises(
        RuntimeError, match="frame provenance is not actor-free"
    ):
        builder.static_depth_path(row)


def test_checkpoint_v1_source_identity_alias_preserves_exact_hash():
    expected = "a" * 64
    identities = source_identity_fields(expected)
    assert identities == {
        "source_dataset_manifest_hash": expected,
        "source_v3_manifest_hash": expected,
    }


def test_route_goal_is_bounded_only_at_local_objective_boundary():
    goals = torch.tensor([
        [3.0, 4.0, 0.0],
        [0.0, 20.0, 0.0],
        [24.0, 0.0, 18.0],
    ])
    local = local_planning_goal(goals, 10.0)
    assert torch.allclose(local.norm(dim=1), torch.tensor([5.0, 10.0, 10.0]))
    assert torch.allclose(
        torch.nn.functional.normalize(local, dim=1),
        torch.nn.functional.normalize(goals, dim=1),
    )
