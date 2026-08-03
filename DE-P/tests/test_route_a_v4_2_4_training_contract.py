from pathlib import Path

from ruamel.yaml import YAML

from policy.runtime_safety_v1 import RuntimeSafetyConfigV1
from policy.static_yopo_preventive_safety_v1 import PreventiveSafetyConfigV1


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/route_a_v4_2_4_preventive_safety_training.yaml"


def load_config():
    return YAML(typ="safe").load(CONFIG)


def test_preventive_training_preserves_motion_but_adds_maneuver_budget():
    config = load_config()
    assert config["objective"]["local_goal_horizon_m"] == 7.0
    gate = config["validation"]["selection_gate"]
    assert gate["selected_endpoint_distance_mean_min"] >= 3.0
    assert gate["selected_endpoint_speed_mean_min"] >= 2.0
    assert gate["hover_selection_rate_max"] <= 0.20


def test_preventive_and_kinodynamic_gates_are_mandatory():
    config = load_config()
    preventive = PreventiveSafetyConfigV1.from_mapping(
        config["preventive_safety"]
    )
    assert preventive.enabled
    assert preventive.maximum_clearance_m > 0.65
    assert config["safety_first"]["kinematic_weight"] >= 8.0
    gate = config["validation"]["selection_gate"]
    assert gate["hardware_unsafe_selection_rate_max"] <= 0.05
    assert gate["anticipatory_unsafe_selection_rate_max"] <= 0.20
    assert gate["clear_candidate_count_mean_min"] >= 3.0


def test_v4_2_4_uses_fp32_and_causal_dynamic_prediction():
    config = load_config()
    assert config["training"]["amp"] is False
    assert config["numerics"]["cnn_amp"] is False
    runtime = RuntimeSafetyConfigV1.from_mapping(
        YAML(typ="safe").load(ROOT / "config/traj_opt.yaml")["runtime_safety"]
    )
    assert runtime.dynamic_track_prediction_enabled
    assert runtime.dynamic_track_max_age_s <= 0.25
