"""Temporary local-goal selection for bounded deadlock recovery.

The selector never invents a translational command from a depth ray.  It may
only choose a point that lies on a network-generated trajectory which already
passed the complete runtime safety evaluation (including dynamic tracks).
The ordinary mission goal therefore remains untouched while the same learned
policy briefly plans toward a certified opening.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


SUBGOAL_ABORT_RECOVERY_TRANSITIONS_V1 = frozenset({
    "network_to_braking",
    "network_stagnation_to_braking",
    "handoff_validation_failed_to_braking",
})


@dataclass(frozen=True)
class RecoverySubgoalConfigV1:
    enabled: bool = True
    minimum_candidate_distance_m: float = 1.0
    target_distance_m: float = 2.5
    arrival_radius_m: float = 0.60
    trajectory_samples: int = 81

    def validate(self):
        if self.minimum_candidate_distance_m <= 0.0:
            raise ValueError("minimum recovery candidate distance must be positive")
        if self.target_distance_m < self.minimum_candidate_distance_m:
            raise ValueError("recovery subgoal target must reach the minimum distance")
        if not 0.0 < self.arrival_radius_m < self.target_distance_m:
            raise ValueError("recovery subgoal arrival radius must be within its target distance")
        if self.trajectory_samples < 3:
            raise ValueError("recovery subgoal needs at least three trajectory samples")

    def contract(self):
        self.validate()
        return {
            "contract_version": "recovery_subgoal_v1",
            "source": "network_candidate_after_full_runtime_safety",
            "translation_owner": "unchanged_learned_policy",
            "mission_goal_mutated": False,
            "dynamic_hard_veto_preserved": True,
            **asdict(self),
        }


@dataclass(frozen=True)
class RecoverySubgoalProposalV1:
    action_id: int
    target_world: np.ndarray
    target_distance_m: float
    candidate_endpoint_distance_m: float
    network_score: float
    min_observed_clearance_m: float | None

    def as_dict(self):
        return {
            "action_id": self.action_id,
            "target_world": self.target_world.tolist(),
            "target_distance_m": self.target_distance_m,
            "candidate_endpoint_distance_m": self.candidate_endpoint_distance_m,
            "network_score": self.network_score,
            "min_observed_clearance_m": self.min_observed_clearance_m,
        }


def _sample_candidate_positions(candidate, duration_s, samples):
    times = np.linspace(0.0, float(duration_s), int(samples), dtype=np.float64)
    return np.stack([
        np.asarray([
            axis.get_position(float(timestamp)) for axis in candidate
        ], dtype=np.float64)
        for timestamp in times
    ])


def recovery_conditioning_goal_v1(
    position_world, camera_forward_world, distance_m=10.0,
):
    """Build a non-executable forward conditioning target for scan inference."""
    position = np.asarray(position_world, dtype=np.float64)
    forward = np.asarray(camera_forward_world, dtype=np.float64)
    distance = float(distance_m)
    if position.shape != (3,) or forward.shape != (3,) \
            or not np.all(np.isfinite(position)) \
            or not np.all(np.isfinite(forward)):
        raise ValueError("recovery conditioning vectors must be finite 3-vectors")
    norm = float(np.linalg.norm(forward))
    if norm <= 1.0e-9 or not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("recovery conditioning direction and distance must be positive")
    return position + forward / norm * distance


def recovery_subgoal_restore_reason_v1(
    recovery_transition, *, dynamic_only_blocked=False,
):
    """Return why an active local goal must yield back to the mission goal."""
    if dynamic_only_blocked:
        return "dynamic_only_zero_feasible"
    if recovery_transition in SUBGOAL_ABORT_RECOVERY_TRANSITIONS_V1:
        return str(recovery_transition)
    return None


def select_recovery_subgoal_v1(
    candidates, durations_s, evaluations, network_scores, origin_world,
    config=None,
):
    """Return a certified local subgoal, or ``None`` when no such path exists.

    Capacity is the primary ordering key, capped at the configured temporary
    goal distance.  Thus all sufficiently long candidates remain governed by
    the learned network score instead of an added clearance heuristic.
    """
    config = config or RecoverySubgoalConfigV1()
    config.validate()
    if not config.enabled:
        return None
    if not (
        len(candidates) == len(durations_s)
        == len(evaluations) == len(network_scores)
    ):
        raise ValueError("recovery subgoal inputs must have identical lengths")
    origin = np.asarray(origin_world, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("recovery subgoal origin must be a finite 3-vector")

    eligible = []
    for action_id, (evaluation, score) in enumerate(
        zip(evaluations, network_scores)
    ):
        endpoint_distance = float(evaluation.endpoint_progress_m)
        score = float(score)
        if (
            bool(evaluation.feasible)
            and np.isfinite(endpoint_distance)
            and endpoint_distance >= config.minimum_candidate_distance_m
            and np.isfinite(score)
        ):
            capacity = min(endpoint_distance, config.target_distance_m)
            eligible.append((-capacity, score, action_id))
    if not eligible:
        return None

    _, score, action_id = min(eligible)
    positions = _sample_candidate_positions(
        candidates[action_id], durations_s[action_id],
        config.trajectory_samples,
    )
    if positions.shape != (config.trajectory_samples, 3) \
            or not np.all(np.isfinite(positions)):
        raise ValueError("network candidate produced non-finite recovery samples")
    distances = np.linalg.norm(positions - origin[None, :], axis=1)
    target_distance = min(
        config.target_distance_m,
        float(evaluations[action_id].endpoint_progress_m),
    )
    # Keep the temporary target on the already certified trajectory prefix.
    sample_index = int(np.argmin(np.abs(distances - target_distance)))
    target = positions[sample_index].copy()
    actual_distance = float(distances[sample_index])
    if actual_distance < config.minimum_candidate_distance_m:
        return None
    clearance = evaluations[action_id].min_observed_clearance_m
    return RecoverySubgoalProposalV1(
        action_id=action_id,
        target_world=target,
        target_distance_m=actual_distance,
        candidate_endpoint_distance_m=float(
            evaluations[action_id].endpoint_progress_m
        ),
        network_score=score,
        min_observed_clearance_m=(
            None if clearance is None else float(clearance)
        ),
    )


__all__ = [
    "RecoverySubgoalConfigV1",
    "RecoverySubgoalProposalV1",
    "SUBGOAL_ABORT_RECOVERY_TRANSITIONS_V1",
    "recovery_conditioning_goal_v1",
    "recovery_subgoal_restore_reason_v1",
    "select_recovery_subgoal_v1",
]
