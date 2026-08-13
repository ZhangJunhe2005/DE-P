import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import test_dep_ros
from test_dep_ros import DepNet

from policy.runtime_profile_v4_8_5 import (
    deadlock_recovery_mapping_v4_8_5,
)
from policy.runtime_profile_v4_9 import (
    PROFILE_NAME,
    deadlock_recovery_mapping_v4_9,
)
from tools.summarize_route_a_v4_9_dynamic_runs import collect


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "route_a_v4_9_dynamic_rviz_host.sh"
PLANNER = ROOT / "test_dep_ros.py"
DEMO = ROOT / "tools" / "run_dep_interactive_demo.py"
BENCHMARK = ROOT / "tools" / "benchmark_route_a_v4_9_dynamic_safety.py"
BENCHMARK_LAUNCHER = (
    ROOT / "scripts" / "route_a_v4_9_dynamic_runtime_benchmark.sh"
)


def test_v49_launcher_freezes_weight_and_route_encounter_matrix():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "route_a_static_yopo_v4_8_3_candidate_only_shakedown" in text
    assert (
        "22e5c63c273d751c15479d70c99d9b85ad615b7b4c62063946a5b1683776ac60"
        in text
    )
    assert "--actor-count 16" in text
    assert "--actor-layout route_encounters" in text
    assert "--actors multi_target" in text
    assert "--dynamic-mode dynamic_safety" in text
    assert f"--runtime-profile {PROFILE_NAME}" in text
    assert "--deadlock-recovery-profile bounded_scan_v3" in text


def test_v49_is_wired_to_planner_and_demo_without_training_entry():
    planner = PLANNER.read_text(encoding="utf-8")
    demo = DEMO.read_text(encoding="utf-8")
    for text in (planner, demo):
        assert "V49_RUNTIME_PROFILE" in text
        assert "runtime_profile_v4_9" in text
    assert "retime_dynamic_only_candidates" in planner
    assert "dynamic_yield_suppressed_recovery" in planner
    assert "dynamic_risk_changed_selection" in planner
    assert "dynamic_command_is_fresh_v1" in planner
    assert "braking_trajectory_options" in planner
    assert "minimum_risk_uncertified_bounded_braking" in planner
    assert "route_encounters" in demo
    assert "train" not in LAUNCHER.name


def test_dynamic_freshness_is_bound_to_the_installed_action(monkeypatch):
    node = DepNet.__new__(DepNet)
    node.runtime_safety_config = SimpleNamespace(
        dynamic_command_freshness_watchdog_enabled=True,
    )
    monkeypatch.setattr(test_dep_ros.time, "monotonic", lambda: 12.5)
    trajectory = (object(), object(), object())

    node._install_trajectory(
        trajectory, 1.7, requires_dynamic_freshness=True,
    )
    assert node.active_trajectory_requires_dynamic_freshness is True
    assert node.last_dynamic_certificate_monotonic_s == 12.5

    # Installing a trajectory that did not receive a fresh dynamic check must
    # clear, not inherit or observationally refresh, the old authorization.
    node._install_trajectory(trajectory, 1.7)
    assert node.active_trajectory_requires_dynamic_freshness is False
    assert node.last_dynamic_certificate_monotonic_s is None


def test_watchdog_hold_is_published_before_releasing_planner_lock():
    source = inspect.getsource(DepNet.control_pub)
    lock_start = source.index("with self.lock:")
    stale_if = source.index("if stale_dynamic_command:")
    hold = source.index("self._publish_hold()", stale_if)
    command_use = source.index("self.ctrl_time += self.ctrl_dt")
    # Freshness check, stale hold, and the first polynomial use all reside in
    # one planner critical section.  There is no check/use deadline window and
    # no old hold can overwrite a concurrently installed fresh action.
    assert source.count("with self.lock:") == 1
    assert lock_start < stale_if < hold < command_use


def test_v49_latency_smoke_is_diagnostic_and_versioned():
    benchmark = BENCHMARK.read_text(encoding="utf-8")
    launcher = BENCHMARK_LAUNCHER.read_text(encoding="utf-8")
    assert "diagnostic_latency_smoke_not_a_gate" in benchmark
    assert "dynamic_track_count\": 16" in benchmark
    assert "route_a_v4_9_dynamic_safety" in launcher
    assert BENCHMARK_LAUNCHER.stat().st_mode & 0o111


def test_v49_does_not_change_frozen_recovery_mapping():
    base = {"enabled": True}
    assert deadlock_recovery_mapping_v4_9(base) == (
        deadlock_recovery_mapping_v4_8_5(base)
    )


def test_frozen_checkpoint_hash_is_the_declared_hash():
    checkpoint = (
        ROOT / "runs/route_a_static_yopo_v4_8_3_candidate_only_shakedown"
        / "20260813T050742Z-13178/checkpoints/best.pth"
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == (
        "22e5c63c273d751c15479d70c99d9b85ad615b7b4c62063946a5b1683776ac60"
    )


def test_empty_v49_summary_is_diagnostic_not_a_gate(tmp_path):
    result = collect(tmp_path)
    assert result["status"] == "FOUR_SCENE_EVIDENCE_INCOMPLETE"
    assert result["role"] == "diagnostic_only_no_gate_no_production_claim"
    assert result["contract"]["actor_count"] == 16
    assert result["contract"]["actor_layout"] == "route_encounters"
    encoded = str(result).lower()
    assert "qualified" not in encoded
    assert "gate_passed" not in encoded
