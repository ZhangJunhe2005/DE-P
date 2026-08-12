from __future__ import annotations

from collections import Counter
import hashlib
import json
import uuid

import yaml

from authoritative_dataset.generate_v1 import load_config
from tools.run_dep_interactive_demo import UNGATED_STATIC_TRAINING_CONTRACTS
import tools.prepare_route_a_v4_6_raw_config as raw_config
from tools.prepare_route_a_v4_6_maps import (
    EXPECTED_TYPES, COUNTS, SEEDS, NAMESPACE,
    load_profiles, selected_profiles,
)
import tools.phase8jqv2_4m1_mixed_maps as map_backend


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()


def test_v4_6_profiles_exclude_room_and_restore_wall_density():
    profiles = load_profiles()
    assert {row["map_type"] for row in profiles} == {
        "cave", "pillar", "forest", "wall"
    }
    assert all(int(row["maze_type"]) != 6 for row in profiles)
    walls = {
        row["size_class"]: row for row in profiles
        if row["map_type"] == "wall"
    }
    assert walls["large"]["wall_number"] == 100
    assert walls["narrow"]["wall_number"] == 25
    for row in walls.values():
        assert row["wall_width_min"] == 0.5
        assert row["wall_width_max"] == 6.0
        assert row["wall_thick"] == 0.5


def test_v4_6_map_plan_is_exactly_four_type_balanced():
    profiles = load_profiles()
    for split, expected_total, expected_per_type in (
        ("train", 48, 12), ("valid", 12, 3),
    ):
        rows = selected_profiles(split, profiles)
        assert len(rows) == expected_total
        assert Counter(row["map_type"] for row in rows) == Counter({
            "cave": expected_per_type,
            "pillar": expected_per_type,
            "forest": expected_per_type,
            "wall": expected_per_type,
        })


def test_v4_6_raw_config_is_four_scene_and_hash_valid(tmp_path, monkeypatch):
    map_root = tmp_path / "maps"
    profiles = load_profiles()
    for split in ("train", "valid"):
        rows = []
        for index, profile in enumerate(selected_profiles(split, profiles)):
            seed = SEEDS[split] + index
            identity = f"{split}:{profile['profile_name']}:{seed}:{index}"
            rows.append({
                "map_id": f"{split}_map_{index:04d}",
                "map_uuid": str(uuid.uuid5(NAMESPACE, identity)),
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
            "status": "PASS", "room_included": False, "maps": rows,
        }))
    target = tmp_path / "resolved.yaml"
    monkeypatch.setattr(raw_config, "MAP_ROOT", map_root)
    monkeypatch.setattr(raw_config, "OUTPUT_ROOT", tmp_path / "raw")
    monkeypatch.setattr(raw_config, "TARGET", target)
    raw_config.main()
    value = load_config(target)
    spatial = value["scene_spatial_sampling_contract"]
    assert set(spatial["map_types"]) == EXPECTED_TYPES
    assert spatial["required_map_types"] == sorted(EXPECTED_TYPES)
    assert len(value["formal_splits"]["train"]["maps"]) == COUNTS["train"]
    assert len(value["formal_splits"]["valid"]["maps"]) == COUNTS["valid"]


def test_authoritative_generator_accepts_versioned_four_type_contract(tmp_path):
    value = yaml.safe_load(
        open("configs/route_a_v4_2_raw_static_resolved.yaml")
    )
    value["dataset_version"] = "route_a_v4_6_raw_static_v1"
    spatial = value["scene_spatial_sampling_contract"]
    spatial["required_map_types"] = ["cave", "forest", "pillar", "wall"]
    spatial["map_types"].pop("room")
    spatial.pop("contract_hash")
    spatial["contract_hash"] = hashlib.sha256(_canonical(spatial)).hexdigest()
    path = tmp_path / "v4_6.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    loaded = load_config(path)
    assert loaded["dataset_version"] == "route_a_v4_6_raw_static_v1"
    assert set(loaded["scene_spatial_sampling_contract"]["map_types"]) == {
        "cave", "pillar", "forest", "wall"
    }


def test_legacy_v4_2_five_type_contract_remains_accepted():
    loaded = load_config("configs/route_a_v4_2_raw_static_resolved.yaml")
    assert set(loaded["scene_spatial_sampling_contract"]["map_types"]) == {
        "cave", "pillar", "forest", "room", "wall"
    }


def test_v4_6_checkpoint_is_pending_closed_loop_not_legacy_gate_rejected():
    assert "route_a_static_yopo_training_v4_6_four_scene_finetune_v1" \
        in UNGATED_STATIC_TRAINING_CONTRACTS
