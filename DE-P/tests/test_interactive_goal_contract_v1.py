import numpy as np

from policy.interactive_goal_contract_v1 import InteractiveGoalContractV1


BOUNDS = (np.array([-10.0, -8.0, 0.0]), np.array([10.0, 8.0, 6.0]))


def test_interactive_goal_can_be_replaced_midflight():
    contract = InteractiveGoalContractV1(policy="interactive")
    assert contract.assess([1.0, 2.0, 3.0], BOUNDS, 0)[0]
    assert contract.assess([-2.0, 1.0, 2.0], BOUNDS, 5)[0]
    assert contract.contract()["midflight_goal_replacement"] is True


def test_fixed_ab_accepts_exactly_one_valid_goal():
    contract = InteractiveGoalContractV1(policy="fixed_ab")
    assert contract.assess([1.0, 2.0, 3.0], BOUNDS, 0)[0]
    accepted, reason = contract.assess([2.0, 2.0, 3.0], BOUNDS, 1)
    assert not accepted
    assert "single reproducible goal" in reason


def test_goal_rejects_bounds_and_vehicle_radius_violations():
    contract = InteractiveGoalContractV1(
        policy="interactive", vehicle_radius_m=0.30
    )
    assert contract.assess([9.70, 0.0, 2.0], BOUNDS, 0)[0]
    accepted, reason = contract.assess([9.71, 0.0, 2.0], BOUNDS, 0)
    assert not accepted
    assert "vehicle-radius margin" in reason
    assert not contract.assess([float("nan"), 0.0, 2.0], BOUNDS, 0)[0]
