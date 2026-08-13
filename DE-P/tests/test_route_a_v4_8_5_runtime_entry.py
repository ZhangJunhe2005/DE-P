from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v4_8_5_entry_strictly_reuses_frozen_v4_8_3_checkpoint():
    text = (ROOT / "scripts/route_a_v4_8_5_rviz_host.sh").read_text()
    assert "v4_8_5_motion_verified_recovery_handoff" in text
    assert "22e5c63c273d751c15479d70c99d9b85ad615b7b4c62063946a5b1683776ac60" in text
    assert "route_a_static_yopo_v4_8_3_candidate_only_shakedown" in text
    assert "--actors none" in text
    assert "--actor-count 0" in text
    assert '[[ "$SCENE" ==' not in text
    assert "train_mixed_static_yopo_v1.py" not in text
    assert "generate_dataset" not in text


def test_v4_8_5_ros_telemetry_exposes_handoff_validation_evidence():
    text = (ROOT / "test_dep_ros.py").read_text()
    assert '"recovery_handoff_validation_active"' in text
    assert '"recovery_handoff_validation_displacement_m"' in text
    assert '"recovery_handoff_validation_forward_clearance_gain_m"' in text
    assert '"recovery_handoff_validation_result"' in text
    assert '"recovery_handoff_preserved_scan_offset_deg"' in text
