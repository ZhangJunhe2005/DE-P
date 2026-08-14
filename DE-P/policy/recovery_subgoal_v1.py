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
    "handoff_directional_retreat_to_braking",
})


@dataclass(frozen=True)
class RecoverySubgoalConfigV1:
    enabled: bool = True
    # A recovery goal must carry the vehicle beyond the immediate obstacle
    # shoulder.  Shorter candidates remain valid normal-flight choices, but
    # they are not sufficient evidence for a temporary escape handoff.
    minimum_candidate_distance_m: float = 3.5
    target_distance_m: float = 5.0
    arrival_radius_m: float = 0.60
    trajectory_samples: int = 81
    prefix_horizon_s: float = 0.60
    minimum_prefix_progress_m: float = 0.05
    maximum_prefix_retreat_m: float = 0.05
    policy_conditioning_distance_m: float = 10.0

    def validate(self):
        if self.minimum_candidate_distance_m <= 0.0:
            raise ValueError("minimum recovery candidate distance must be positive")
        if self.target_distance_m < self.minimum_candidate_distance_m:
            raise ValueError("recovery subgoal target must reach the minimum distance")
        if not 0.0 < self.arrival_radius_m < self.target_distance_m:
            raise ValueError("recovery subgoal arrival radius must be within its target distance")
        if self.trajectory_samples < 3:
            raise ValueError("recovery subgoal needs at least three trajectory samples")
        if self.prefix_horizon_s <= 0.0:
            raise ValueError("recovery prefix horizon must be positive")
        if self.minimum_prefix_progress_m <= 0.0:
            raise ValueError("recovery prefix progress must be positive")
        if self.maximum_prefix_retreat_m < 0.0:
            raise ValueError("recovery prefix retreat must be non-negative")
        if self.policy_conditioning_distance_m < self.target_distance_m:
            raise ValueError(
                "recovery policy conditioning distance must cover the "
                "temporary target"
            )

    def contract(self):
        self.validate()
        return {
            "contract_version": (
                "recovery_subgoal_v1_4_long_escape_rearmable_mission_attempt"
            ),
            "source": "network_candidate_after_full_runtime_safety",
            "translation_owner": "unchanged_learned_policy",
            "mission_goal_mutated": False,
            "dynamic_hard_veto_preserved": True,
            "consecutive_temporary_goals_allowed": False,
            "rearm_evidence": (
                "mission_yaw_aligned_then_fresh_mission_conditioned_replan"
            ),
            "repeat_recovery_policy": (
                "fresh_universal_stagnation_evidence_after_mission_attempt"
            ),
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
    prefix_progress_m: float
    maximum_prefix_retreat_m: float

    def as_dict(self):
        return {
            "action_id": self.action_id,
            "target_world": self.target_world.tolist(),
            "target_distance_m": self.target_distance_m,
            "candidate_endpoint_distance_m": self.candidate_endpoint_distance_m,
            "network_score": self.network_score,
            "min_observed_clearance_m": self.min_observed_clearance_m,
            "prefix_progress_m": self.prefix_progress_m,
            "maximum_prefix_retreat_m": self.maximum_prefix_retreat_m,
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


def recovery_subgoal_conditioning_goal_v1(
    position_world, temporary_target_world, distance_m=10.0,
):
    """Extend a temporary target direction to the policy's trained horizon.

    The fixed temporary target remains the lifecycle/arrival authority.  This
    extended point is input conditioning only: it prevents a bounded local
    recovery target from being interpreted as a near-goal stop command by a
    policy trained on the 10 metre local planning horizon.
    """
    position = np.asarray(position_world, dtype=np.float64)
    target = np.asarray(temporary_target_world, dtype=np.float64)
    if position.shape != (3,) or target.shape != (3,) \
            or not np.all(np.isfinite(position)) \
            or not np.all(np.isfinite(target)):
        raise ValueError("recovery subgoal vectors must be finite 3-vectors")
    return recovery_conditioning_goal_v1(
        position, target - position, distance_m,
    )


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
            positions = _sample_candidate_positions(
                candidates[action_id], durations_s[action_id],
                config.trajectory_samples,
            )
            if positions.shape != (config.trajectory_samples, 3) \
                    or not np.all(np.isfinite(positions)):
                raise ValueError(
                    "network candidate produced non-finite recovery samples"
                )
            distances = np.linalg.norm(positions - origin[None, :], axis=1)
            target_distance = min(
                config.target_distance_m,
                float(evaluation.endpoint_progress_m),
            )
            sample_index = int(np.argmin(np.abs(distances - target_distance)))
            target = positions[sample_index].copy()
            actual_distance = float(distances[sample_index])
            if actual_distance < config.minimum_candidate_distance_m:
                continue
            direction = target - origin
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= 1.0e-9:
                continue
            direction /= direction_norm
            duration = float(durations_s[action_id])
            if not np.isfinite(duration) or duration <= 0.0:
                raise ValueError("recovery candidate duration must be positive")
            prefix_index = min(
                sample_index,
                max(
                    1,
                    int(round(
                        min(config.prefix_horizon_s, duration)
                        / duration * (config.trajectory_samples - 1)
                    )),
                ),
            )
            signed_prefix = (
                positions[:prefix_index + 1] - origin[None, :]
            ) @ direction
            prefix_progress = float(signed_prefix[-1])
            maximum_retreat = max(0.0, -float(np.min(signed_prefix)))
            if (
                prefix_progress < config.minimum_prefix_progress_m
                or maximum_retreat > config.maximum_prefix_retreat_m
            ):
                continue
            capacity = min(endpoint_distance, config.target_distance_m)
            eligible.append((
                -capacity, score, action_id, target, actual_distance,
                prefix_progress, maximum_retreat,
            ))
    if not eligible:
        return None

    (
        _, score, action_id, target, actual_distance,
        prefix_progress, maximum_retreat,
    ) = min(eligible, key=lambda item: (item[0], item[1], item[2]))
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
        prefix_progress_m=prefix_progress,
        maximum_prefix_retreat_m=maximum_retreat,
    )


__all__ = [
    "RecoverySubgoalConfigV1",
    "RecoverySubgoalProposalV1",
    "SUBGOAL_ABORT_RECOVERY_TRANSITIONS_V1",
    "recovery_conditioning_goal_v1",
    "recovery_subgoal_conditioning_goal_v1",
    "recovery_subgoal_restore_reason_v1",
    "select_recovery_subgoal_v1",
]
