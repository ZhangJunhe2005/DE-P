"""Versioned Route-A local-planning supervision contract."""

from __future__ import annotations

from policy.static_yopo_contract_v1 import contract_hash


LOCAL_GOAL_OBJECTIVE_CONTRACT_V1 = {
    "contract_version": "route_a_local_goal_objective_v1",
    "route_goal_range_m": [10.0, 40.0],
    "network_goal_input": "direction_preserving_normalized",
    "local_planning_horizon_m": 10.0,
    "loss_goal_rule": "direction * min(route_goal_distance, local_horizon)",
    "score_label_uses_local_goal": True,
    "route_goal_used_by_closed_loop_replanning": True,
}

LOCAL_GOAL_OBJECTIVE_CONTRACT_V1_HASH = contract_hash(
    LOCAL_GOAL_OBJECTIVE_CONTRACT_V1
)
