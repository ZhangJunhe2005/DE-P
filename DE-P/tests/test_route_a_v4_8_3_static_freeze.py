from pathlib import Path

from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[1]


def test_static_freeze_selects_v483_and_rejects_v484():
    contract = YAML(typ="safe").load(
        ROOT / "configs/route_a_v4_8_3_static_freeze.yaml"
    )
    assert contract["status"] == "FROZEN_CLOSED_LOOP_SELECTED"
    assert contract["selected_version"] == "v4.8.3"
    assert contract["checkpoint"]["epoch"] == 2
    assert len(contract["checkpoint"]["sha256"]) == 64
    assert contract["closed_loop_evidence"]["goal_arrived"] is True
    assert contract["closed_loop_evidence"]["collision_events"] == 0
    assert contract["rejected_successor"]["version"] == "v4.8.4"
    assert contract["rejected_successor"]["score_selection_improved"] is False


def test_frozen_launcher_checks_exact_checkpoint_hash_and_universal_runtime():
    script = (
        ROOT / "scripts/route_a_v4_8_3_frozen_rviz_host.sh"
    ).read_text()
    assert "22e5c63c273d751c15479d70c99d9b85" in script
    assert "sha256sum" in script
    assert "route_a_v4_8_2_rviz_host.sh" in script
