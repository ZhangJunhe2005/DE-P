import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from policy.checkpoint_utils import load_checkpoint_payload, unpack_checkpoint

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "run_dep_interactive_demo.py"
SPEC = importlib.util.spec_from_file_location("run_dep_interactive_demo", TOOL_PATH)
DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEMO)


@pytest.mark.parametrize("scene_name", ["cave", "forest", "pillar", "room", "wall"])
def test_all_interactive_scenes_have_existing_maps_and_valid_routes(scene_name):
    scene = DEMO.load_scene(DEMO.DEFAULT_SCENES, scene_name)
    assert scene["pointcloud"].is_file()
    start, goal, *_ = DEMO._vector_geometry(scene)
    assert len(start) == len(goal) == 3


@pytest.mark.parametrize("actor_mode,count", [
    ("none", 0),
    ("crossing", 1),
    ("head_on", 1),
    ("multi_target", 3),
])
def test_actor_scenarios_are_long_running_and_scene_relative(actor_mode, count):
    scene = DEMO.load_scene(DEMO.DEFAULT_SCENES, "forest")
    payload = DEMO.build_actor_scenario("forest", scene, actor_mode)
    actors = payload["dynamic_scenario"]["actors"]
    assert len(actors) == count
    assert payload["dynamic_scenario"]["enabled"] is bool(count)
    assert all(actor["trajectory"]["end_time"] == 3600.0 for actor in actors)


@pytest.mark.parametrize("actor_mode,count", [
    ("crossing", 2),
    ("crossing", 8),
    ("head_on", 4),
    ("multi_target", 12),
])
def test_actor_count_is_selectable_and_actors_have_3d_motion(
    actor_mode, count
):
    scene = DEMO.load_scene(DEMO.DEFAULT_SCENES, "cave")
    payload = DEMO.build_actor_scenario(
        "cave", scene, actor_mode, actor_count=count, vertical_span=3.0
    )
    actors = payload["dynamic_scenario"]["actors"]
    assert len(actors) == count
    assert len({actor["id"] for actor in actors}) == count
    assert len({
        round(actor["initial_position_world"][2], 3) for actor in actors
    }) > 1
    assert any(
        actor["trajectory"]["waypoints_world"][0][2]
        != actor["trajectory"]["waypoints_world"][1][2]
        for actor in actors
    )


def test_actor_count_contract_rejects_invalid_combinations():
    scene = DEMO.load_scene(DEMO.DEFAULT_SCENES, "forest")
    with pytest.raises(ValueError, match="requires --actor-count 0"):
        DEMO.build_actor_scenario(
            "forest", scene, "none", actor_count=2
        )
    with pytest.raises(ValueError, match=r"within \[1, 64\]"):
        DEMO.build_actor_scenario(
            "forest", scene, "crossing", actor_count=65
        )


def test_forest_demo_stays_in_trunk_layer_and_uses_authority_clearance():
    scene = DEMO.load_scene(DEMO.DEFAULT_SCENES, "forest")
    assert max(scene["start"][2], scene["suggested_goal"][2]) <= 4.0
    assert scene["actor_z_bounds"][1] <= 5.5
    payload = DEMO.build_actor_scenario(
        "forest", scene, "crossing", actor_count=4, vertical_span=2.0
    )
    scenario = payload["dynamic_scenario"]
    assert scenario["canonical_occupancy_checked"] is True
    for actor in scenario["actors"]:
        for waypoint in actor["trajectory"]["waypoints_world"]:
            assert scene["actor_z_bounds"][0] <= waypoint[2]
            assert waypoint[2] <= scene["actor_z_bounds"][1]


def test_hybrid_layout_spreads_actors_across_map_and_keeps_route_encounters():
    scene = DEMO.load_scene(DEMO.DEFAULT_SCENES, "forest")
    payload = DEMO.build_actor_scenario(
        "forest", scene, "multi_target", actor_count=16,
        vertical_span=2.0, actor_layout="hybrid", actor_seed=9917,
    )
    scenario = payload["dynamic_scenario"]
    centers = []
    for actor in scenario["actors"]:
        first, second = actor["trajectory"]["waypoints_world"]
        centers.append([(a + b) * 0.5 for a, b in zip(first, second)])
    assert scenario["corridor_actor_count"] == 5
    assert scenario["map_wide_actor_count"] == 11
    assert max(value[0] for value in centers) - min(value[0] for value in centers) > 25
    assert max(value[1] for value in centers) - min(value[1] for value in centers) > 25


def test_actor_limit_is_64_for_static_reactive_scenarios():
    scene = DEMO.load_scene(DEMO.DEFAULT_SCENES, "forest")
    payload = DEMO.build_actor_scenario(
        "forest", scene, "crossing", actor_count=32,
        actor_layout="map_wide", actor_seed=9918,
    )
    assert len(payload["dynamic_scenario"]["actors"]) == 32


def test_formal_training_checkpoint_unwraps_for_inference():
    checkpoint = (
        ROOT / "runs" / "phase8_mixed_static_yopo_v3_2_low_lr_adamw"
        / "20260730T074415Z-260761" / "checkpoints" / "best.pth"
    )
    if not checkpoint.is_file():
        pytest.skip("local formal training checkpoint is not present")
    payload = load_checkpoint_payload(checkpoint)
    state, metadata = unpack_checkpoint(payload)
    assert state
    assert all(not key.startswith("network.") for key in state)
    assert metadata["formal_training_checkpoint"] is True
    assert metadata["epoch"] == 14


def test_run_directory_epoch_selection():
    run_dir = (
        ROOT / "runs" / "phase8_mixed_static_yopo_v3_2_low_lr_adamw"
        / "20260730T074415Z-260761"
    )
    if not run_dir.is_dir():
        pytest.skip("local formal training run is not present")
    best = DEMO.resolve_checkpoint(
        Namespace(checkpoint=None, run_dir=run_dir, epoch="best")
    )
    epoch = DEMO.resolve_checkpoint(
        Namespace(checkpoint=None, run_dir=run_dir, epoch="14")
    )
    assert best.name == "best.pth"
    assert epoch.name == "epoch_014.pth"


def test_swept_sphere_detects_thin_obstacle_between_clear_samples():
    authority = DEMO.CanonicalOccupancy.__new__(DEMO.CanonicalOccupancy)
    authority.origin = np.zeros(3, dtype=np.float64)
    authority.dimensions = np.asarray([20, 20, 20], dtype=np.int64)
    authority.resolution = 0.1
    authority.grid = np.zeros((20, 20, 20), dtype=bool)
    authority.grid[10, 10, 10] = True
    authority.bounds_min = authority.origin
    authority.bounds_max = authority.origin + (
        authority.dimensions * authority.resolution
    )

    start = np.asarray([0.7, 1.05, 1.05])
    end = np.asarray([1.4, 1.05, 1.05])
    radius = 0.12
    assert not authority.sphere_collides(start, radius)
    assert not authority.sphere_collides(end, radius)
    assert authority.swept_sphere_collides(start, end, radius)


def test_swept_sphere_stays_clear_away_from_obstacle():
    authority = DEMO.CanonicalOccupancy.__new__(DEMO.CanonicalOccupancy)
    authority.origin = np.zeros(3, dtype=np.float64)
    authority.dimensions = np.asarray([20, 20, 20], dtype=np.int64)
    authority.resolution = 0.1
    authority.grid = np.zeros((20, 20, 20), dtype=bool)
    authority.grid[10, 10, 10] = True
    authority.bounds_min = authority.origin
    authority.bounds_max = authority.origin + (
        authority.dimensions * authority.resolution
    )

    assert not authority.swept_sphere_collides(
        [0.7, 0.5, 0.5], [1.4, 0.5, 0.5], 0.12
    )
