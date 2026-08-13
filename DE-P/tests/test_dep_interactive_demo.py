import importlib.util
import copy
import math
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


def _independent_route_encounter_measurements(scene, actor):
    """Numerical black-box oracle independent of the production certificate."""
    route_start = np.asarray(scene["start"], dtype=np.float64)
    route_goal = np.asarray(scene["suggested_goal"], dtype=np.float64)
    route_vector = route_goal - route_start
    route_length = float(np.linalg.norm(route_vector))
    route_direction = route_vector / route_length
    route_duration = route_length / 3.0
    first, second = (
        np.asarray(value, dtype=np.float64)
        for value in actor["trajectory"]["waypoints_world"]
    )
    actor_vector = second - first
    actor_path_length = float(np.linalg.norm(actor_vector))
    actor_direction = actor_vector / actor_path_length
    actor_speed = float(actor["trajectory"]["speed"])
    start_time = float(actor["trajectory"]["start_time"])
    first_uav = route_start + route_direction * min(
        route_length, 3.0 * start_time
    )
    relative = first - first_uav
    longitudinal = float(np.dot(relative, route_direction))
    off_axis = relative - longitudinal * route_direction
    bearing = math.degrees(math.atan2(
        float(np.linalg.norm(off_axis)), max(0.0, longitudinal)
    ))

    # Sample the actual ping-pong kinematics rather than calling the launcher's
    # analytic checker.  A 1 ms-class grid is much finer than the 2.5 m bound.
    times = np.linspace(start_time, route_duration, 16001)
    leg_duration = actor_path_length / actor_speed
    phase = (times - start_time) / leg_duration
    leg = np.floor(phase).astype(np.int64)
    amount = phase - leg
    amount = np.where(leg % 2 == 0, amount, 1.0 - amount)
    actor_positions = first + amount[:, None] * actor_vector
    uav_positions = route_start + 3.0 * times[:, None] * route_direction
    separations = np.linalg.norm(actor_positions - uav_positions, axis=1)
    closest_index = int(np.argmin(separations))
    return {
        "route_direction": route_direction,
        "actor_direction": actor_direction,
        "first": first,
        "second": second,
        "first_longitudinal": longitudinal,
        "first_bearing_deg": bearing,
        "direction_cosine": float(np.dot(actor_direction, route_direction)),
        "closest_distance_m": float(separations[closest_index]),
        "closest_time_s": float(times[closest_index]),
        "closest_actor_position": actor_positions[closest_index],
        "closest_uav_position": uav_positions[closest_index],
    }


def _uniform_stratum(value, bounds, count):
    """Return an independent half-open stratum index for emitted geometry."""
    lower, upper = map(float, bounds)
    assert lower <= float(value) <= upper
    normalized = (float(value) - lower) / (upper - lower)
    return min(count - 1, int(math.floor(normalized * count)))


def _uniform_actor_measurements(scene, actor, z_bounds=None):
    """Measure the emitted simulator trajectory without trusting metadata."""
    first, second = (
        np.asarray(value, dtype=np.float64)
        for value in actor["trajectory"]["waypoints_world"]
    )
    midpoint = 0.5 * (first + second)
    movement = second - first
    movement_norm = float(np.linalg.norm(movement))
    assert movement_norm > 1e-6
    angle = math.atan2(float(movement[1]), float(movement[0]))
    octant = int(math.floor(
        ((angle + math.pi) % (2.0 * math.pi)) * 4.0 / math.pi
    ))
    x_bounds, y_bounds = scene["actor_xy_bounds"]
    z_bounds = scene["actor_z_bounds"] if z_bounds is None else z_bounds
    route_first = np.asarray(scene["start"], dtype=np.float64)
    route_second = np.asarray(scene["suggested_goal"], dtype=np.float64)
    _, _, route_distance = DEMO._segment_segment_closest(
        first, second, route_first, route_second
    )
    return {
        "first": first,
        "second": second,
        "midpoint": midpoint,
        "movement": movement,
        "xy_stratum": (
            _uniform_stratum(midpoint[0], x_bounds, 4),
            _uniform_stratum(midpoint[1], y_bounds, 4),
        ),
        "z_stratum": _uniform_stratum(midpoint[2], z_bounds, 4),
        "direction_octant": octant,
        "route_distance_m": route_distance,
    }


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


@pytest.mark.parametrize("scene_name", ["cave", "forest", "pillar", "wall"])
def test_uniform_3d_layout_covers_full_map_height_and_canonical_free_space(
    scene_name,
):
    """The stress population must occupy XYZ strata, not one route-aligned row."""
    scenes = ROOT / "configs" / "dep_interactive_demo_scenes_v4_6.json"
    scene = DEMO.load_scene(scenes, scene_name)
    payload = DEMO.build_actor_scenario(
        scene_name, scene, "multi_target", actor_count=16,
        vertical_span=2.0, actor_layout="uniform_3d", actor_seed=9917,
    )
    scenario = payload["dynamic_scenario"]
    actors = scenario["actors"]
    z_bounds = scenario["uniform_3d_z_bounds"]
    measured = [
        _uniform_actor_measurements(scene, actor, z_bounds) for actor in actors
    ]

    assert len(actors) == 16
    assert scenario["actor_layout"] == "uniform_3d"
    assert scenario["uniform_3d_contract_version"] == "uniform_3d_layout_v1"
    assert scenario["xy_strata_shape"] == [4, 4]
    assert scenario["xy_strata_occupied"] == 16
    assert scenario["z_strata_count"] == 4
    assert scenario["z_strata_occupied"] == 4
    assert scenario["z_bounds_source"] == "canonical_flight_bounds"
    assert scenario["canonical_occupancy_checked"] is True
    assert scene["flight_bounds_min"][2] <= z_bounds[0] < z_bounds[1]
    assert z_bounds[1] <= scene["flight_bounds_max"][2]
    assert (z_bounds[1] - z_bounds[0]) >= 0.75 * (
        scene["flight_bounds_max"][2] - scene["flight_bounds_min"][2]
    )
    assert {item["xy_stratum"] for item in measured} == {
        (x_index, y_index)
        for x_index in range(4)
        for y_index in range(4)
    }
    z_counts = np.bincount(
        [item["z_stratum"] for item in measured], minlength=4
    )
    assert z_counts.tolist() == [4, 4, 4, 4]

    # Recheck final post-relocation paths against the canonical authority.  A
    # placement label cannot substitute for collision-free emitted geometry.
    authority = DEMO.CanonicalOccupancy(
        scene["authority_root"], scene["map_uuid"]
    )
    assert all(authority.actor_path_is_clear(actor) for actor in actors)
    assert all(
        "route_encounter_contract" not in actor
        and "route_encounter_evidence" not in actor
        for actor in actors
    )


def test_uniform_3d_layout_is_seeded_random_and_directionally_diverse():
    scene = DEMO.load_scene(
        ROOT / "configs" / "dep_interactive_demo_scenes_v4_6.json", "cave"
    )

    def build(seed):
        return DEMO.build_actor_scenario(
            "cave", scene, "multi_target", actor_count=16,
            vertical_span=2.0, actor_layout="uniform_3d", actor_seed=seed,
        )["dynamic_scenario"]

    first = build(9917)
    repeated = build(9917)
    alternate = build(9918)
    assert first == repeated
    assert first != alternate

    measured = [
        _uniform_actor_measurements(
            scene, actor, first["uniform_3d_z_bounds"]
        ) for actor in first["actors"]
    ]
    octants = {item["direction_octant"] for item in measured}
    vertical_signs = {
        int(np.sign(item["movement"][2])) for item in measured
        if abs(float(item["movement"][2])) > 1e-6
    }
    assert len(octants) >= 6
    assert first["motion_direction_octants_occupied"] == len(octants)
    assert first["vertical_direction_signs"] == [-1, 1]
    assert vertical_signs == {-1, 1}


def test_uniform_3d_reports_route_proximity_without_claiming_encounters():
    """Uniform map coverage is not evidence that a nominal route meets actors."""
    scene = DEMO.load_scene(
        ROOT / "configs" / "dep_interactive_demo_scenes_v4_6.json", "forest"
    )
    scenario = DEMO.build_actor_scenario(
        "forest", scene, "multi_target", actor_count=16,
        vertical_span=2.0, actor_layout="uniform_3d", actor_seed=9917,
    )["dynamic_scenario"]
    threshold = float(scenario["route_proximity_threshold_m"])
    measured = [
        _uniform_actor_measurements(
            scene, actor, scenario["uniform_3d_z_bounds"]
        )
        for actor in scenario["actors"]
    ]
    proximity_count = sum(
        item["route_distance_m"] <= threshold for item in measured
    )
    assert scenario["route_proximity_actor_count"] == proximity_count
    assert proximity_count < len(measured)
    assert scenario["route_encounter_guaranteed"] is False
    assert scenario.get("route_encounter_actor_count", 0) == 0
    assert scenario["planner_ground_truth_exposed"] is False


def test_uniform_3d_ground_truth_file_is_only_wired_to_simulator():
    """The actor fixture may diagnose perception but must not enter planning."""
    source = TOOL_PATH.read_text(encoding="utf-8")
    simulator_start = source.index('            "simulator",')
    monitor_start = source.index(
        '            "collision_monitor",', simulator_start
    )
    planner_start = source.index('            "planner",', monitor_start)
    readiness_start = source.index("        ready_topics =", planner_start)
    assert "_dynamic_scenario_file" in source[simulator_start:monitor_start]
    assert "_dynamic_scenario_file" not in source[planner_start:readiness_start]


def test_route_encounter_layout_requires_a_meaningful_multi_target_stress_set():
    scene = DEMO.load_scene(DEMO.DEFAULT_SCENES, "forest")
    with pytest.raises(ValueError, match="requires --actors multi_target"):
        DEMO.build_actor_scenario(
            "forest", scene, "crossing", actor_count=8,
            actor_layout="route_encounters",
        )
    with pytest.raises(ValueError, match="at least 8"):
        DEMO.build_actor_scenario(
            "forest", scene, "multi_target", actor_count=7,
            actor_layout="route_encounters",
        )


@pytest.mark.parametrize("scene_name", ["cave", "forest", "pillar", "wall"])
def test_route_encounters_are_balanced_delayed_and_canonically_preserved(
    scene_name,
):
    scenes = ROOT / "configs" / "dep_interactive_demo_scenes_v4_6.json"
    scene = DEMO.load_scene(scenes, scene_name)
    payload = DEMO.build_actor_scenario(
        scene_name, scene, "multi_target", actor_count=16,
        vertical_span=2.0, actor_layout="route_encounters", actor_seed=9917,
    )
    scenario = payload["dynamic_scenario"]
    assert scenario["encounter_family_counts"] == {
        "crossing": 4,
        "head_on": 4,
        "same_direction_slow": 4,
        "staggered_crossing": 4,
    }
    assert scenario["route_encounter_actor_count"] == 16
    assert scenario["route_contract_actor_count"] == 16
    assert scenario["route_contract_preserved_actor_count"] == 16
    assert scenario["canonical_occupancy_checked"] is True
    assert scenario["route_encounter_contract_version"] == (
        "route_encounter_layout_v2"
    )
    assert max(actor["route_fraction"] for actor in scenario["actors"]) - min(
        actor["route_fraction"] for actor in scenario["actors"]
    ) > 0.5
    staggered = [
        actor for actor in scenario["actors"]
        if actor["encounter_family"] == "staggered_crossing"
    ]
    assert all(actor["trajectory"]["start_time"] > 0.0 for actor in staggered)
    assert len({
        actor["trajectory"]["start_time"] for actor in staggered
    }) == len(staggered)
    for actor in scenario["actors"]:
        # This oracle reconstructs kinematics from the emitted simulator input;
        # it neither trusts evidence metadata nor calls the production helper.
        measured = _independent_route_encounter_measurements(scene, actor)
        assert measured["first_longitudinal"] >= 1.0
        assert measured["first_bearing_deg"] <= 80.0
        assert measured["closest_distance_m"] <= 2.5 + 1e-3
        assert actor["trajectory"]["start_time"] <= measured["closest_time_s"]
        assert measured["closest_time_s"] <= (
            np.linalg.norm(
                np.asarray(scene["suggested_goal"])
                - np.asarray(scene["start"])
            ) / 3.0
        )
        assert 0.0 <= float(np.dot(
            measured["closest_uav_position"] - np.asarray(scene["start"]),
            measured["route_direction"],
        )) <= np.linalg.norm(
            np.asarray(scene["suggested_goal"])
            - np.asarray(scene["start"])
        )

        family = actor["encounter_family"]
        if family in {"crossing", "staggered_crossing"}:
            assert abs(measured["direction_cosine"]) <= 0.25
            lateral_axis = np.cross(
                measured["route_direction"], [0.0, 0.0, 1.0]
            )
            lateral_axis /= np.linalg.norm(lateral_axis)
            signed_first = float(np.dot(
                measured["first"] - np.asarray(scene["start"]), lateral_axis
            ))
            signed_second = float(np.dot(
                measured["second"] - np.asarray(scene["start"]), lateral_axis
            ))
            assert signed_first * signed_second <= 1e-9
        elif family == "head_on":
            assert measured["direction_cosine"] <= -0.8
        else:
            assert family == "same_direction_slow"
            assert measured["direction_cosine"] >= 0.8
            assert actor["trajectory"]["speed"] <= 0.65

        evidence = actor["route_encounter_evidence"]
        assert evidence["passed"] is True
        assert evidence["failures"] == []
        assert evidence["first_appearance_relative_longitudinal_m"] == (
            pytest.approx(measured["first_longitudinal"], abs=1e-9)
        )
        assert evidence["outbound_velocity_route_direction_cosine"] == (
            pytest.approx(measured["direction_cosine"], abs=1e-9)
        )
        # Analytic and independently sampled closest approaches must agree.
        assert evidence["nominal_encounter_distance_m"] == pytest.approx(
            measured["closest_distance_m"], abs=0.01
        )
        assert evidence["nominal_encounter_time_s"] == pytest.approx(
            measured["closest_time_s"], abs=0.01
        )


def test_route_contract_rejects_wrong_direction_rear_birth_and_missed_timing():
    scene = DEMO.load_scene(
        ROOT / "configs" / "dep_interactive_demo_scenes_v4_6.json", "forest"
    )
    actors = DEMO.build_actor_scenario(
        "forest", scene, "multi_target", actor_count=16,
        vertical_span=2.0, actor_layout="route_encounters", actor_seed=9917,
    )["dynamic_scenario"]["actors"]

    head_on = copy.deepcopy(next(
        actor for actor in actors if actor["encounter_family"] == "head_on"
    ))
    head_on["trajectory"]["waypoints_world"].reverse()
    assert not DEMO._route_contract_is_preserved(
        head_on, head_on["trajectory"]["waypoints_world"]
    )

    rear = copy.deepcopy(next(
        actor for actor in actors
        if actor["encounter_family"] == "same_direction_slow"
    ))
    route = np.asarray(scene["suggested_goal"]) - np.asarray(scene["start"])
    route /= np.linalg.norm(route)
    rear["trajectory"]["waypoints_world"] = [
        (np.asarray(value) - 30.0 * route).tolist()
        for value in rear["trajectory"]["waypoints_world"]
    ]
    assert not DEMO._route_contract_is_preserved(
        rear, rear["trajectory"]["waypoints_world"]
    )

    delayed = copy.deepcopy(next(
        actor for actor in actors if actor["encounter_family"] == "crossing"
    ))
    delayed["trajectory"]["start_time"] = 30.0
    assert not DEMO._route_contract_is_preserved(
        delayed, delayed["trajectory"]["waypoints_world"]
    )


def test_hybrid_actors_do_not_acquire_route_encounter_contract_metadata():
    scene = DEMO.load_scene(DEMO.DEFAULT_SCENES, "forest")
    actors = DEMO.build_actor_scenario(
        "forest", scene, "multi_target", actor_count=16,
        vertical_span=2.0, actor_layout="hybrid", actor_seed=9917,
    )["dynamic_scenario"]["actors"]
    assert all("route_encounter_contract" not in actor for actor in actors)
    assert all("route_encounter_evidence" not in actor for actor in actors)


def test_actor_limit_is_64_for_static_reactive_scenarios():
    scene = DEMO.load_scene(DEMO.DEFAULT_SCENES, "forest")
    payload = DEMO.build_actor_scenario(
        "forest", scene, "crossing", actor_count=32,
        actor_layout="map_wide", actor_seed=9918,
    )
    assert len(payload["dynamic_scenario"]["actors"]) == 32


def test_uniform_3d_layout_rejects_zero_vertical_motion_contract():
    scene = DEMO.load_scene(DEMO.DEFAULT_SCENES, "forest")
    with pytest.raises(ValueError, match="requires positive actor vertical span"):
        DEMO.build_actor_scenario(
            "forest", scene, "multi_target", actor_count=16,
            vertical_span=0.0, actor_layout="uniform_3d", actor_seed=8801,
        )


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


def test_v454_checkpoint_is_pending_closed_loop_not_gate_qualified():
    checkpoint = (
        ROOT / "runs/route_a_static_yopo_v4_5_4_independent_score_shakedown"
        / "20260810T090950Z-41634" / "checkpoints" / "best.pth"
    )
    if not checkpoint.is_file():
        pytest.skip("local V4.5.4 checkpoint is not present")
    result = DEMO.checkpoint_preflight(checkpoint)
    assert result["head_variant"] == "independent"
    assert result["strict_load"] is True
    assert result["training_gate_qualified"] is None
    assert result["checkpoint_readiness"] == (
        "validation_loss_selected_pending_closed_loop"
    )


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
