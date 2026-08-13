from types import SimpleNamespace

import numpy as np
import pytest
import torch

import test_dep_ros
from test_dep_ros import DepNet

from policy.recovery_subgoal_v1 import RecoverySubgoalConfigV1
from policy.runtime_profile_v4_9 import (
    deadlock_recovery_mapping_v4_9,
    runtime_safety_mapping_v4_9,
)
from policy.runtime_profile_v4_9_1 import (
    PROFILE_NAME,
    deadlock_recovery_mapping_v4_9_1,
    runtime_safety_mapping_v4_9_1,
)


def test_profile_name_is_versioned_without_replacing_v49():
    assert PROFILE_NAME == "v4_9_1_recovery_temporary_subgoal"


def test_v491_keeps_v49_physical_and_dynamic_runtime_contract_exactly():
    base = {}
    assert runtime_safety_mapping_v4_9_1(base) == runtime_safety_mapping_v4_9(base)
    parent = deadlock_recovery_mapping_v4_9(base)
    directional = deadlock_recovery_mapping_v4_9_1(base)
    assert directional.pop("handoff_validation_window_s") == 2.50
    assert directional.pop("handoff_directional_progress_enabled") is True
    assert directional.pop("handoff_max_directional_retreat_m") == 0.10
    parent.pop("handoff_validation_window_s")
    assert directional == parent


def test_ros_entry_keeps_mission_arrival_separate_from_planning_goal():
    source = open("test_dep_ros.py", encoding="utf-8").read()
    assert "np.linalg.norm(pos - self.mission_goal)" in source
    assert "self.goal = self.mission_goal.copy()" in source
    assert '"cancelled_by_new_mission_goal"' in source
    assert "select_recovery_subgoal_v1" in source
    assert "recovery_subgoal_conditioning_goal_v1" in source
    assert "_abort_recovery_subgoal_locked" in source
    assert "_complete_recovery_subgoal_locked" in source


def test_subgoal_activation_never_overwrites_mission_and_restore_is_exact(monkeypatch):
    monkeypatch.setattr(test_dep_ros.rospy, "logwarn", lambda *args: None)
    monkeypatch.setattr(test_dep_ros.rospy, "loginfo", lambda *args: None)
    node = DepNet.__new__(DepNet)
    node.recovery_subgoal_config = RecoverySubgoalConfigV1()
    node.mission_goal = np.asarray((12.0, -4.0, 2.5))
    node.goal = node.mission_goal.copy()
    node.recovery_subgoal_world = None
    node.recovery_subgoal_origin_world = None
    node.recovery_subgoal_action_id = None
    node.recovery_subgoal_last_event = "inactive"
    proposal = SimpleNamespace(
        target_world=np.asarray((2.5, 0.0, 2.5)), action_id=7,
    )

    assert node._activate_recovery_subgoal_locked(
        proposal, np.asarray((0.0, 0.0, 2.5)),
    )
    assert node.goal.tolist() == [2.5, 0.0, 2.5]
    assert node.mission_goal.tolist() == [12.0, -4.0, 2.5]
    assert node._restore_mission_goal_locked("test")
    assert node.goal.tolist() == [12.0, -4.0, 2.5]
    assert node.mission_goal.tolist() == [12.0, -4.0, 2.5]


def test_failed_subgoal_restores_mission_and_yields_old_scan_chain(monkeypatch):
    monkeypatch.setattr(test_dep_ros.rospy, "logwarn", lambda *args: None)
    monkeypatch.setattr(test_dep_ros.rospy, "loginfo", lambda *args: None)
    node = DepNet.__new__(DepNet)
    node.mission_goal = np.asarray((12.0, -4.0, 2.5))
    node.goal = np.asarray((2.5, 0.0, 2.5))
    node.recovery_subgoal_world = node.goal.copy()
    node.recovery_subgoal_origin_world = np.zeros(3)
    node.recovery_subgoal_action_id = 7
    node.recovery_subgoal_last_event = "activated"
    node.dynamic_yield_active = True
    recovery = SimpleNamespace(
        reset_calls=[],
        reset=lambda position: recovery.reset_calls.append(
            np.asarray(position).copy()
        ),
    )
    node.deadlock_recovery = recovery

    position = np.asarray((0.1, 0.2, 2.5))
    assert node._abort_recovery_subgoal_locked(
        "handoff_validation_failed_to_braking", position,
    )
    assert node.goal.tolist() == [12.0, -4.0, 2.5]
    assert node.recovery_subgoal_world is None
    assert node.dynamic_yield_active is False
    assert len(recovery.reset_calls) == 1
    assert recovery.reset_calls[0] == pytest.approx(position)


def test_reached_subgoal_restores_mission_and_clears_handoff(monkeypatch):
    monkeypatch.setattr(test_dep_ros.rospy, "logwarn", lambda *args: None)
    monkeypatch.setattr(test_dep_ros.rospy, "loginfo", lambda *args: None)
    node = DepNet.__new__(DepNet)
    node.mission_goal = np.asarray((12.0, -4.0, 2.5))
    node.goal = np.asarray((2.5, 0.0, 2.5))
    node.recovery_subgoal_world = node.goal.copy()
    node.recovery_subgoal_origin_world = np.zeros(3)
    node.recovery_subgoal_action_id = 7
    node.recovery_subgoal_last_event = "activated"
    node.dynamic_yield_active = True
    recovery = SimpleNamespace(
        reset_calls=[],
        reset=lambda position: recovery.reset_calls.append(
            np.asarray(position).copy()
        ),
    )
    node.deadlock_recovery = recovery

    position = np.asarray((2.1, 0.0, 2.5))
    assert node._complete_recovery_subgoal_locked(position)
    assert node.goal.tolist() == [12.0, -4.0, 2.5]
    assert node.recovery_subgoal_world is None
    assert node.dynamic_yield_active is False
    assert len(recovery.reset_calls) == 1
    assert recovery.reset_calls[0] == pytest.approx(position)


def test_active_short_subgoal_conditions_network_at_ten_metre_horizon():
    node = DepNet.__new__(DepNet)
    linear = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    node.odom = SimpleNamespace(
        pose=SimpleNamespace(pose=SimpleNamespace(orientation=orientation)),
        twist=SimpleNamespace(twist=SimpleNamespace(linear=linear)),
    )
    node.Rotation_bc = np.eye(3)
    node.desire_pos = np.asarray((1.0, 2.0, 3.0))
    node.desire_vel = np.zeros(3)
    node.desire_acc = np.zeros(3)
    node.plan_from_reference = False
    node.runtime_profile = test_dep_ros.V491_RUNTIME_PROFILE
    node.deadlock_recovery_profile = "bounded_scan_v3"
    node.deadlock_recovery = SimpleNamespace(mode="network")
    node.recovery_subgoal_config = RecoverySubgoalConfigV1()
    node.recovery_subgoal_world = np.asarray((3.0, 3.0, 3.0))
    node.goal = node.recovery_subgoal_world.copy()
    node.device = torch.device("cpu")
    node.state_transform = SimpleNamespace(normalize_obs=lambda value: value)

    observation = node.process_odom()
    expected_direction = np.asarray((2.0, 1.0, 0.0))
    expected_direction /= np.linalg.norm(expected_direction)
    conditioned_delta = node.last_network_goal_world - node.desire_pos
    assert np.linalg.norm(conditioned_delta) == pytest.approx(10.0)
    assert conditioned_delta / np.linalg.norm(conditioned_delta) \
        == pytest.approx(expected_direction)
    assert node.goal.tolist() == [3.0, 3.0, 3.0]
    assert node.recovery_subgoal_conditioning_active is True
    assert observation[0, 6:9].numpy() == pytest.approx(
        expected_direction * 10.0
    )
