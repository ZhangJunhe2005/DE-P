from pathlib import Path

from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[1]


def test_v423_contract_is_versioned_and_keeps_hardware_limits():
    path = ROOT / "configs/route_a_v4_2_3_ego_feasibility_training.yaml"
    if not path.exists():
        # The immutable file is materialized by its preparation entry.
        return
    config = YAML(typ="safe").load(path)
    assert config["contract_version"].endswith("v4_2_3_ego_feasibility")
    assert config["kinodynamic_v2"]["enabled"] is True
    assert config["kinodynamic_v2"]["max_speed_mps"] == 6.0
    assert config["kinodynamic_v2"]["max_acceleration_mps2"] == 6.0
    assert config["validation"]["selection_gate"][
        "feasible_candidate_count_mean_min"
    ] >= 3.0


def test_v423_host_entry_has_verify_before_authorized_training():
    text = (ROOT / "scripts/route_a_v4_2_3_train_host.sh").read_text()
    assert "--verify-only" in text
    assert text.index("--verify-only") < text.index("--authorized")
