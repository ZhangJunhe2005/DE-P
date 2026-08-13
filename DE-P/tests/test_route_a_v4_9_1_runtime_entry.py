from pathlib import Path

from policy.runtime_profile_v4_9_1 import (
    PROFILE_NAME,
    RUNTIME_BEHAVIOR_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "route_a_v4_9_1_dynamic_rviz_host.sh"
V49_LAUNCHER = ROOT / "scripts" / "route_a_v4_9_dynamic_rviz_host.sh"
PLANNER = ROOT / "test_dep_ros.py"


def test_v491_is_an_ab_entry_over_the_same_frozen_checkpoint_and_fixture():
    current = LAUNCHER.read_text(encoding="utf-8")
    parent = V49_LAUNCHER.read_text(encoding="utf-8")
    for expected in (
        "route_a_static_yopo_v4_8_3_candidate_only_shakedown",
        "22e5c63c273d751c15479d70c99d9b85ad615b7b4c62063946a5b1683776ac60",
        "--actor-count 16",
        "--actor-layout uniform_3d",
        "--actor-seed 9098",
        "--dynamic-mode dynamic_safety",
        "--deadlock-recovery-profile bounded_scan_v3",
    ):
        assert expected in current
        assert expected in parent
    assert f"--runtime-profile {PROFILE_NAME}" in current
    assert "v4_9_dynamic_motion_preserving_safety" in parent
    assert LAUNCHER.stat().st_mode & 0o111


def test_v491_profile_and_goal_lifecycle_are_wired_to_ros():
    source = PLANNER.read_text(encoding="utf-8")
    for expected in (
        "V491_RUNTIME_PROFILE",
        "V491_RUNTIME_BEHAVIOR_VERSION",
        "recovery_conditioning_goal_v1",
        "select_recovery_subgoal_v1",
        "_activate_recovery_subgoal_locked",
        "_restore_mission_goal_locked",
        '"cancelled_by_new_mission_goal"',
        '"temporary_goal_reached"',
        '"handoff_validation_failed_to_braking"',
        '"network_stagnation_to_braking"',
        '"recovery_probe_no_handoff_brake"',
    ):
        assert expected in source


def test_v491_keeps_model_as_translation_owner_and_dynamic_veto_before_activation():
    source = PLANNER.read_text(encoding="utf-8")
    inference = source.index("endstate_pred, score_pred = self.policy")
    evaluation = source.index("evaluations = self.runtime_safety.evaluate", inference)
    proposal = source.index("recovery_subgoal_proposal = select_recovery_subgoal_v1", evaluation)
    activation = source.index("self._activate_recovery_subgoal_locked", proposal)
    install = source.index("self._install_trajectory(", activation)
    assert inference < evaluation < proposal < activation < install
    assert "mission_distance = float(np.linalg.norm(pos - self.mission_goal))" in source
    assert "self.goal = candidate_goal.copy()" in source


def test_v49_parent_launcher_remains_v49_not_v491():
    source = V49_LAUNCHER.read_text(encoding="utf-8")
    assert "v4_9_dynamic_motion_preserving_safety" in source
    assert PROFILE_NAME not in source
