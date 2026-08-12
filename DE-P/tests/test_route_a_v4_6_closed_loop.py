from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from policy.checkpoint_utils import load_checkpoint_payload, unpack_checkpoint
from tools.run_dep_interactive_demo import load_scene
from tools.validate_route_a_launch_fixture import validate_scene_config


ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "configs/dep_interactive_demo_scenes_v4_6.json"
RUN = (
    ROOT / "runs/route_a_static_yopo_v4_6_four_scene_finetune"
    / "20260811T112951Z-77428"
)


def test_v4_6_fixed_ab_matrix_has_exactly_four_new_authority_scenes():
    payload = json.loads(SCENES.read_text())
    assert set(payload["scenes"]) == {"cave", "forest", "pillar", "wall"}
    for name in sorted(payload["scenes"]):
        scene = load_scene(SCENES, name)
        assert "route_a_v4_6" in str(scene["pointcloud"])
        assert "route_a_v4_6" in str(scene["authority_root"])
        assert np.linalg.norm(
            np.asarray(scene["suggested_goal"])
            - np.asarray(scene["start"])
        ) >= 30.0
        assert 35.0 <= float(scene["reference_path_length_m"]) <= 45.0


def test_v4_6_launch_fixtures_are_inside_upstream_domain_and_locally_open():
    result = validate_scene_config(SCENES)
    assert result["status"] == "PASS"
    assert set(result["scenes"]) == {"cave", "forest", "pillar", "wall"}
    for scene in result["scenes"].values():
        assert scene["official_sample_domain"] is True
        assert scene["start_center_clearance_m"] >= 1.5
        assert scene["goal_center_clearance_m"] >= 1.5
        assert scene["start_uav_surface_gap_m"] >= 1.0
        assert scene["goal_uav_surface_gap_m"] >= 1.0
        assert scene["fov_p10_open_range_m"] >= 3.0
        assert scene["first_segment_open_range_m"] >= 3.0


def test_v4_6_best_checkpoint_is_strictly_loadable_and_pending_closed_loop():
    completion = json.loads((RUN / "training_complete.json").read_text())
    assert completion["status"] == "TRAINING_COMPLETE_PENDING_CLOSED_LOOP"
    assert completion["best_epoch"] == 22
    payload = load_checkpoint_payload(RUN / "checkpoints/best.pth")
    _, metadata = unpack_checkpoint(payload)
    assert metadata["identities"]["training_contract_version"] == (
        "route_a_static_yopo_training_v4_6_four_scene_finetune_v1"
    )
