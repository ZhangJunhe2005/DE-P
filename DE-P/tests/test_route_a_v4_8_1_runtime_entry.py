from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v4_8_1_rviz_entry_is_pillar_only_and_reuses_v4_8_checkpoint():
    text = (ROOT / "scripts/route_a_v4_8_1_rviz_host.sh").read_text()
    assert "route_a_static_yopo_v4_8_recovery_capacity_shakedown" in text
    assert '[[ "$SCENE" == "pillar" ]]' in text
    assert 'RUNTIME_PROFILE="v4_8_1_stable_sector_handoff"' in text
    assert 'RUNTIME_PROFILE="v4_7_balanced_dynamic"' in text
    assert "--deadlock-recovery-profile bounded_scan_v3" in text
    assert "generate_dataset" not in text
    assert "train_mixed_static_yopo_v1.py" not in text


def test_v4_8_1_ros_integration_passes_action_and_horizontal_sector():
    text = (ROOT / "test_dep_ros.py").read_text()
    assert "horizontal_sector_from_action_id(" in text
    assert '"selected_candidate_action_id"' in text
    assert '"selected_candidate_horizontal_sector_id"' in text
    assert '"recovery_handoff_confirmation_replans"' in text
    assert '"recovery_handoff_horizontal_sector_id"' in text
