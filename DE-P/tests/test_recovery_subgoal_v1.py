from types import SimpleNamespace

import numpy as np
import pytest

from policy.recovery_subgoal_v1 import (
    RecoverySubgoalConfigV1,
    recovery_conditioning_goal_v1,
    recovery_subgoal_restore_reason_v1,
    select_recovery_subgoal_v1,
)


class _LinearAxis:
    def __init__(self, start, end, duration=1.0):
        self.start = float(start)
        self.end = float(end)
        self.duration = float(duration)

    def get_position(self, timestamp):
        ratio = float(timestamp) / self.duration
        return self.start + ratio * (self.end - self.start)


def _candidate(endpoint):
    endpoint = np.asarray(endpoint, dtype=np.float64)
    return tuple(_LinearAxis(0.0, value) for value in endpoint)


def _evaluation(*, feasible=True, distance=3.0, clearance=0.8):
    return SimpleNamespace(
        feasible=feasible,
        endpoint_progress_m=float(distance),
        min_observed_clearance_m=clearance,
    )


def test_subgoal_lies_on_a_fully_feasible_network_candidate():
    proposal = select_recovery_subgoal_v1(
        (_candidate((4.0, 0.0, 0.0)),),
        (1.0,),
        (_evaluation(distance=4.0),),
        (0.2,),
        np.zeros(3),
    )
    assert proposal is not None
    assert proposal.action_id == 0
    assert proposal.target_world[1] == pytest.approx(0.0)
    assert proposal.target_distance_m == pytest.approx(2.5, abs=0.06)


def test_dynamic_or_static_veto_can_never_supply_a_subgoal():
    proposal = select_recovery_subgoal_v1(
        (_candidate((5.0, 0.0, 0.0)),),
        (1.0,),
        (_evaluation(feasible=False, distance=5.0),),
        (-100.0,),
        np.zeros(3),
    )
    assert proposal is None


def test_short_candidate_does_not_trigger_translation_handoff():
    config = RecoverySubgoalConfigV1(minimum_candidate_distance_m=1.0)
    proposal = select_recovery_subgoal_v1(
        (_candidate((0.7, 0.0, 0.0)),),
        (1.0,),
        (_evaluation(distance=0.7),),
        (0.0,),
        np.zeros(3),
        config,
    )
    assert proposal is None


def test_capacity_is_primary_but_network_score_breaks_sufficient_ties():
    proposals = select_recovery_subgoal_v1(
        (
            _candidate((2.6, 0.0, 0.0)),
            _candidate((0.0, 3.0, 0.0)),
            _candidate((1.5, 0.0, 0.0)),
        ),
        (1.0, 1.0, 1.0),
        (
            _evaluation(distance=2.6),
            _evaluation(distance=3.0),
            _evaluation(distance=1.5),
        ),
        (0.3, 0.1, -100.0),
        np.zeros(3),
    )
    assert proposals.action_id == 1


def test_contract_states_network_translation_and_mission_separation():
    contract = RecoverySubgoalConfigV1().contract()
    assert contract["translation_owner"] == "unchanged_learned_policy"
    assert contract["mission_goal_mutated"] is False
    assert contract["dynamic_hard_veto_preserved"] is True


def test_scan_conditioning_target_is_forward_and_does_not_need_mission_goal():
    result = recovery_conditioning_goal_v1(
        (2.0, 3.0, 1.0), (0.0, 2.0, 0.0), 10.0,
    )
    assert result == pytest.approx((2.0, 13.0, 1.0))


@pytest.mark.parametrize("transition", (
    "network_to_braking",
    "network_stagnation_to_braking",
    "handoff_validation_failed_to_braking",
))
def test_repeated_deadlock_always_restores_mission_before_new_scan(transition):
    assert recovery_subgoal_restore_reason_v1(transition) == transition


def test_dynamic_only_blockage_also_abandons_stale_temporary_goal():
    assert recovery_subgoal_restore_reason_v1(
        None, dynamic_only_blocked=True,
    ) == "dynamic_only_zero_feasible"


def test_successful_handoff_does_not_restore_until_temporary_goal_is_reached():
    assert recovery_subgoal_restore_reason_v1(
        "handoff_measured_motion_verified"
    ) is None
