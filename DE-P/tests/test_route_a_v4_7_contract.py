from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import pytest
import yaml

from authoritative_dataset.generate_v1 import (
    load_config,
    validate_scene_observability,
)
import tools.phase8jqv2_4m1_mixed_maps as map_backend
import tools.prepare_route_a_v4_6_maps as v4_6_maps
import tools.prepare_route_a_v4_7_maps as v4_7_maps
import tools.prepare_route_a_v4_7_raw_config as raw_config


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()


def test_v4_7_profiles_match_official_pillar_and_wall_parameters():
    profiles = v4_7_maps.load_profiles()
    assert {row["map_type"] for row in profiles} == {
        "cave", "pillar", "forest", "wall"
    }
    assert all(int(row["maze_type"]) != 6 for row in profiles)
    by_key = {(row["map_type"], row["size_class"]): row for row in profiles}
    for size, count in (("large", 100), ("narrow", 25)):
        pillar = by_key[("pillar", size)]
        assert pillar["obstacle_number"] == count
        assert pillar["width_min"] == 0.6
        assert pillar["width_max"] == 1.5
        wall = by_key[("wall", size)]
        assert wall["wall_number"] == count
        assert wall["wall_width_min"] == 0.5
        assert wall["wall_width_max"] == 6.0
        assert wall["wall_thick"] == 0.5
        assert wall["wall_ceiling"] == 1


def test_v4_7_common_defaults_cannot_reintroduce_dense_pillars():
    document = yaml.safe_load(v4_7_maps.PROFILE_PATH.read_text())
    assert document["common"]["obstacle_number"] == 100
    assert document["common"]["width_min"] == 0.6
    assert document["common"]["width_max"] == 1.5


def test_v4_7_pillar_distribution_contract_rejects_a_cluster():
    profile = {"x_length": 30.0, "y_length": 30.0}
    distributed = np.asarray([
        [x, y, 0.6]
        for x in (-10.0, 0.0, 10.0)
        for y in (-10.0, 0.0, 10.0)
    ], dtype=np.float64)
    clustered = np.asarray([
        [-14.0 + 0.1 * index, -14.0 + 0.1 * index, 0.6]
        for index in range(50)
    ], dtype=np.float64)
    assert v4_7_maps.pillar_distribution_metrics(
        distributed, profile
    )["passed"] is True
    assert v4_7_maps.pillar_distribution_metrics(
        clustered, profile
    )["passed"] is False


def test_v4_7_view_contract_rejects_a_fully_blocked_frame():
    sensor = {"max_depth_m": 20.0}
    policy = {
        "near_depth_m": 10.0,
        "minimum_return_fraction": 0.2,
        "minimum_near_obstacle_fraction": 0.1,
        "maximum_near_obstacle_fraction": 0.75,
        "maximum_frame_near_obstacle_fraction": 0.90,
        "far_depth_m": 6.0,
        "minimum_far_fraction": 0.20,
        "minimum_frame_far_fraction": 0.05,
    }
    observable = np.full((2, 4, 4), 20.0, dtype=np.float32)
    observable[:, :, :2] = 4.0
    blocked = observable.copy()
    blocked[1] = 2.0
    assert validate_scene_observability(
        observable, sensor, "pillar", policy
    )["passed"] is True
    result = validate_scene_observability(
        blocked, sensor, "pillar", policy
    )
    assert result["passed"] is False
    assert result["observed_minimum_frame_far_fraction"] == 0.0


def test_v4_7_map_plan_is_exactly_four_type_balanced():
    profiles = v4_7_maps.load_profiles()
    for split, expected_total, expected_per_type in (
        ("train", 48, 12), ("valid", 12, 3),
    ):
        rows = v4_7_maps.selected_profiles(split, profiles)
        assert len(rows) == expected_total
        assert Counter(row["map_type"] for row in rows) == Counter({
            "cave": expected_per_type,
            "pillar": expected_per_type,
            "forest": expected_per_type,
            "wall": expected_per_type,
        })


def test_v4_7_pillar_plan_has_bounded_seed_candidates(tmp_path):
    v4_7_maps.make_plan(tmp_path)
    document = json.loads(v4_7_maps.plan_path(tmp_path).read_text())
    assert document["pillar_distribution_contract"]["version"] == (
        "route_a_v4_7_pillar_distribution_v1"
    )
    for rows in document["maps"].values():
        for row in rows:
            expected = 16 if row["map_type"] == "pillar" else 1
            assert len(row["seed_candidates"]) == expected
            assert len(row["map_uuid_candidates"]) == expected


def test_v4_7_identities_are_disjoint_from_v4_6():
    assert v4_7_maps.DEFAULT_OUTPUT != v4_6_maps.DEFAULT_OUTPUT
    assert v4_7_maps.NAMESPACE != v4_6_maps.NAMESPACE
    assert set(v4_7_maps.SEEDS.values()).isdisjoint(v4_6_maps.SEEDS.values())


def test_historical_v4_6_profile_remains_unchanged():
    profiles = v4_6_maps.load_profiles()
    by_key = {(row["map_type"], row["size_class"]): row for row in profiles}
    assert by_key[("pillar", "large")]["obstacle_number"] == 300
    assert by_key[("pillar", "narrow")]["obstacle_number"] == 75
    assert by_key[("pillar", "large")]["width_min"] == 0.8
    assert by_key[("pillar", "large")]["width_max"] == 1.8
    assert by_key[("wall", "large")]["wall_number"] == 100
    assert by_key[("wall", "narrow")]["wall_number"] == 25


def test_v4_7_plan_is_write_once(tmp_path):
    v4_7_maps.make_plan(tmp_path)
    path = v4_7_maps.plan_path(tmp_path)
    before = path.read_bytes()
    v4_7_maps.make_plan(tmp_path)
    assert path.read_bytes() == before
    document = json.loads(path.read_text())
    document["namespace"] = str(uuid.uuid4())
    path.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        v4_7_maps.make_plan(tmp_path)


def test_v4_7_raw_config_is_versioned_and_does_not_touch_v4_6(
        tmp_path, monkeypatch):
    map_root = tmp_path / "maps"
    profiles = v4_7_maps.load_profiles()
    for split in ("train", "valid"):
        rows = []
        for index, profile in enumerate(
                v4_7_maps.selected_profiles(split, profiles)):
            seed = v4_7_maps.SEEDS[split] + index
            identity = f"{split}:{profile['profile_name']}:{seed}:{index}"
            rows.append({
                "map_id": f"{split}_map_{index:04d}",
                "map_uuid": str(uuid.uuid5(v4_7_maps.NAMESPACE, identity)),
                "seed": seed,
                "maze_type": int(profile["maze_type"]),
                "map_type": profile["map_type"],
                "semantic_name": profile["semantic_name"],
                "profile_name": profile["profile_name"],
                "profile_hash": map_backend.profile_hash(profile),
            })
        path = map_root / f"manifests/{split}_maps.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "status": "PASS",
            "profile_version": v4_7_maps.PROFILE_VERSION,
            "room_included": False,
            "maps": rows,
        }))
    target = tmp_path / "resolved.yaml"
    historical = Path("configs/route_a_v4_6_raw_static_resolved.yaml")
    historical_hash = hashlib.sha256(historical.read_bytes()).hexdigest()
    monkeypatch.setattr(raw_config, "MAP_ROOT", map_root)
    monkeypatch.setattr(raw_config, "OUTPUT_ROOT", tmp_path / "raw")
    monkeypatch.setattr(raw_config, "TARGET", target)
    raw_config.main()
    value = load_config(target)
    assert value["dataset_version"] == "route_a_v4_7_raw_static_v1"
    assert value["random_seeds"]["sequence_base"] == 858000000
    assert set(value["scene_spatial_sampling_contract"]["map_types"]) \
        == v4_7_maps.EXPECTED_TYPES
    assert len(value["formal_splits"]["train"]["maps"]) == 48
    assert len(value["formal_splits"]["valid"]["maps"]) == 12
    spatial = value["scene_spatial_sampling_contract"]
    for name in ("pillar", "wall"):
        policy = spatial["map_types"][name]
        assert policy["maximum_frame_near_obstacle_fraction"] < 1.0
        assert policy["minimum_frame_far_fraction"] > 0.0
    assert hashlib.sha256(historical.read_bytes()).hexdigest() == historical_hash


def test_authoritative_generator_accepts_v4_7_version(tmp_path):
    value = yaml.safe_load(
        Path("configs/route_a_v4_2_raw_static_resolved.yaml").read_text()
    )
    value["dataset_version"] = "route_a_v4_7_raw_static_v1"
    spatial = value["scene_spatial_sampling_contract"]
    spatial["required_map_types"] = ["cave", "forest", "pillar", "wall"]
    spatial["map_types"].pop("room")
    spatial.pop("contract_hash")
    spatial["contract_hash"] = hashlib.sha256(_canonical(spatial)).hexdigest()
    path = tmp_path / "v4_7.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    loaded = load_config(path)
    assert loaded["dataset_version"] == "route_a_v4_7_raw_static_v1"


def test_v4_7_host_script_never_deletes_or_mentions_v4_6_data():
    script = Path("scripts/route_a_v4_7_generate_dataset_host.sh").read_text()
    assert "rm -rf" not in script
    assert "data/route_a_v4_6" not in script
    assert "data/route_a_v4_7" in script
    assert "--maps-only" in script
