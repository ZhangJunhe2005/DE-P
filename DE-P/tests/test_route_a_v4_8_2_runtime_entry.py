from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v4_8_2_rviz_entry_uses_identical_recovery_for_every_scene():
    text = (ROOT / "scripts/route_a_v4_8_2_rviz_host.sh").read_text()
    assert "v4_8_2_universal_stagnation_recovery" in text
    assert "--deadlock-recovery 1" in text
    assert "--deadlock-recovery-profile bounded_scan_v3" in text
    assert '[[ "$SCENE" ==' not in text
    assert "Pillar-only" not in text
    assert "generate_dataset" not in text
    assert "train_mixed_static_yopo_v1.py" not in text


def test_v4_8_2_ros_telemetry_exposes_measured_stagnation_evidence():
    text = (ROOT / "test_dep_ros.py").read_text()
    assert '"recovery_motion_stagnation_replans"' in text
    assert '"recovery_motion_window_displacement_m"' in text
    assert '"recovery_trigger_reason"' in text
