from types import SimpleNamespace

import numpy as np

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
    assert deadlock_recovery_mapping_v4_9_1(base) == deadlock_recovery_mapping_v4_9(base)


def test_ros_entry_keeps_mission_arrival_separate_from_planning_goal():
    source = open("test_dep_ros.py", encoding="utf-8").read()
    assert "np.linalg.norm(pos - self.mission_goal)" in source
    assert "self.goal = self.mission_goal.copy()" in source
    assert '"cancelled_by_new_mission_goal"' in source
    assert "select_recovery_subgoal_v1" in source


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
