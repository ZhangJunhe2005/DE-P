"""Versioned goal acceptance rules for interactive and reproducible demos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GOAL_POLICIES_V1 = ("interactive", "fixed_ab")


@dataclass(frozen=True)
class InteractiveGoalContractV1:
    """Validate a goal without changing YOPO planning semantics.

    ``interactive`` accepts every valid replacement goal, including while the
    vehicle is flying. ``fixed_ab`` accepts exactly one valid goal so two
    checkpoint runs can be compared against an identical task.
    """

    policy: str = "interactive"
    vehicle_radius_m: float = 0.30

    def __post_init__(self):
        if self.policy not in GOAL_POLICIES_V1:
            raise ValueError(f"unsupported goal policy: {self.policy}")
        if not np.isfinite(self.vehicle_radius_m) or self.vehicle_radius_m < 0:
            raise ValueError("vehicle_radius_m must be finite and non-negative")

    def assess(self, goal, flight_bounds, accepted_goal_count):
        goal = np.asarray(goal, dtype=np.float64)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            return False, "goal must be a finite XYZ vector"
        if self.policy == "fixed_ab" and int(accepted_goal_count) >= 1:
            return False, "fixed_ab already accepted its single reproducible goal"
        if flight_bounds is not None:
            lower, upper = (
                np.asarray(value, dtype=np.float64) for value in flight_bounds
            )
            if lower.shape != (3,) or upper.shape != (3,):
                raise ValueError("flight bounds must be XYZ vectors")
            margin = float(self.vehicle_radius_m)
            if np.any(goal < lower + margin) or np.any(goal > upper - margin):
                return False, (
                    "goal lies outside the canonical flight volume after "
                    f"the {margin:.2f} m vehicle-radius margin"
                )
        return True, "accepted"

    def contract(self):
        return {
            "version": "interactive_goal_contract_v1",
            "policy": self.policy,
            "midflight_goal_replacement": self.policy == "interactive",
            "maximum_accepted_goals": (
                None if self.policy == "interactive" else 1
            ),
            "vehicle_radius_m": self.vehicle_radius_m,
        }
