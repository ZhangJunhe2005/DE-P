"""Bounded deadlock recovery with observed-space escape certification.

V2 preserves V1's brake-first and breadcrumb provenance rules, but replaces an
unbounded yaw scan with a finite +/- yaw envelope.  Translation is requested
only toward the currently observed camera direction; the runtime safety shield
must independently certify the generated quintic before it is installed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DeadlockRecoveryConfigV2:
    enabled: bool = True
    zero_feasible_trigger_replans: int = 15
    collision_floor_trigger_replans: int = 5
    release_feasible_replans: int = 5
    scan_release_feasible_replans: int = 2
    stationary_speed_mps: float = 0.35
    yaw_scan_rate_deg_s: float = 45.0
    max_scan_angle_deg: float = 90.0
    max_scan_legs: int = 2
    breadcrumb_spacing_m: float = 0.12
    breadcrumb_capacity: int = 600
    retreat_enabled: bool = True
    retreat_distance_m: float = 1.20
    retreat_arrival_radius_m: float = 0.20
    observed_escape_enabled: bool = True
    observed_escape_distance_m: float = 1.20
    observed_escape_arrival_radius_m: float = 0.20
    observed_escape_min_depth_m: float = 2.00
    observed_escape_min_goal_alignment: float = -0.05
    goal_biased_scan_deadband_deg: float = 10.0

    @classmethod
    def from_mapping(cls, value):
        value = {} if value is None else dict(value)
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown deadlock_recovery v2 keys: {unknown}")
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        counters = (
            self.zero_feasible_trigger_replans,
            self.collision_floor_trigger_replans,
            self.release_feasible_replans,
            self.scan_release_feasible_replans,
            self.breadcrumb_capacity,
            self.max_scan_legs,
        )
        if min(counters) < 1:
            raise ValueError("deadlock recovery counters must be positive")
        positive = (
            self.stationary_speed_mps,
            self.yaw_scan_rate_deg_s,
            self.max_scan_angle_deg,
            self.breadcrumb_spacing_m,
            self.retreat_distance_m,
            self.retreat_arrival_radius_m,
            self.observed_escape_distance_m,
            self.observed_escape_arrival_radius_m,
            self.observed_escape_min_depth_m,
            self.goal_biased_scan_deadband_deg,
        )
        if min(positive) <= 0:
            raise ValueError("deadlock recovery distances/rates must be positive")
        if self.max_scan_angle_deg > 180:
            raise ValueError("scan envelope may not exceed 180 degrees")
        if self.retreat_arrival_radius_m >= self.retreat_distance_m:
            raise ValueError("retreat arrival radius must be smaller than distance")
        if self.observed_escape_arrival_radius_m >= self.observed_escape_distance_m:
            raise ValueError("escape arrival radius must be smaller than distance")
        if not -1.0 <= self.observed_escape_min_goal_alignment <= 1.0:
            raise ValueError("escape goal alignment must be in [-1,1]")

    def contract(self):
        return {
            "version": "deadlock_recovery_v2",
            "sensing_boundary": "forward_depth_plus_recent_flown_breadcrumbs",
            "normal_policy": "network_owns_translation",
            "deadlock_policy": "brake_bounded_scan_certified_escape",
            "uncertified_policy": "bounded_hold_no_blind_translation",
            **asdict(self),
        }


@dataclass(frozen=True)
class RecoveryDecisionV2:
    mode: str
    transition: str | None
    retreat_target_world: tuple[float, float, float] | None
    escape_target_world: tuple[float, float, float] | None
    zero_feasible_replans: int
    collision_floor_replans: int
    release_feasible_replans: int
    scan_offset_deg: float
    scan_legs_completed: int
    escape_goal_alignment: float | None


class DeadlockRecoveryV2:
    NORMAL = "network"
    BRAKING = "recovery_braking"
    RETREAT = "recovery_breadcrumb_retreat"
    YAW_SCAN = "recovery_yaw_scan"
    OBSERVED_ESCAPE = "recovery_observed_escape"
    HOLD = "recovery_bounded_hold"

    def __init__(self, config: DeadlockRecoveryConfigV2):
        config.validate()
        self.config = config
        self.breadcrumbs = deque(maxlen=config.breadcrumb_capacity)
        self.mode = self.NORMAL
        self.zero_feasible_replans = 0
        self.collision_floor_replans = 0
        self.release_feasible_replans = 0
        self.scan_direction = 1.0
        self.scan_offset_rad = 0.0
        self.scan_leg_complete = False
        self.scan_legs_completed = 0
        self.retreat_target = None
        self.escape_target = None
        self.last_position = None
        self.last_rotation = None
        self.last_goal = None
        self.escape_goal_alignment = None

    def reset(self, position_world=None):
        self.mode = self.NORMAL
        self.zero_feasible_replans = 0
        self.collision_floor_replans = 0
        self.release_feasible_replans = 0
        self.scan_direction = 1.0
        self.scan_offset_rad = 0.0
        self.scan_leg_complete = False
        self.scan_legs_completed = 0
        self.retreat_target = None
        self.escape_target = None
        self.last_position = None
        self.last_rotation = None
        self.last_goal = None
        self.escape_goal_alignment = None
        self.breadcrumbs.clear()
        if position_world is not None:
            self.record_position(position_world)

    def record_position(self, position_world):
        point = np.asarray(position_world, dtype=np.float64).reshape(3)
        if not np.isfinite(point).all():
            return
        if (
            not self.breadcrumbs
            or np.linalg.norm(point - self.breadcrumbs[-1])
            >= self.config.breadcrumb_spacing_m
        ):
            self.breadcrumbs.append(point.copy())

    @staticmethod
    def scan_direction_from_depth(depth):
        array = np.asarray(depth, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] < 4:
            return 1.0
        midpoint = array.shape[1] // 2

        def free_score(region):
            valid = region[np.isfinite(region) & (region > 0)]
            return float(np.quantile(valid, 0.65)) if len(valid) else 0.0

        return (
            1.0
            if free_score(array[:, :midpoint])
            >= free_score(array[:, midpoint:])
            else -1.0
        )

    def _breadcrumb_target(self, current_position):
        if not self.config.retreat_enabled or len(self.breadcrumbs) < 2:
            return None
        current = np.asarray(current_position, dtype=np.float64).reshape(3)
        travelled = 0.0
        previous = current
        for point in reversed(self.breadcrumbs):
            travelled += float(np.linalg.norm(previous - point))
            if travelled >= self.config.retreat_distance_m:
                return point.copy()
            previous = point
        return None

    def _observed_escape_target(self, position, rotation, depth):
        if not self.config.observed_escape_enabled or rotation is None:
            return None
        array = np.asarray(depth, dtype=np.float32)
        if array.ndim != 2 or min(array.shape) < 4:
            return None
        h, w = array.shape
        corridor = array[h // 4:3 * h // 4, 2 * w // 5:3 * w // 5]
        valid = corridor[np.isfinite(corridor) & (corridor > 0)]
        if not len(valid):
            return None
        # A low quantile prevents a few max-range pixels from certifying a
        # corridor that is mostly occupied.
        if float(np.quantile(valid, 0.10)) < self.config.observed_escape_min_depth_m:
            return None
        rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        direction = rotation[:, 0].copy()
        direction[2] = 0.0
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm < 1.0e-6:
            return None
        direction /= norm
        self.escape_goal_alignment = None
        if self.last_goal is not None:
            goal_direction = self.last_goal - np.asarray(
                position, dtype=np.float64
            ).reshape(3)
            goal_direction[2] = 0.0
            goal_norm = float(np.linalg.norm(goal_direction))
            if np.isfinite(goal_norm) and goal_norm >= 1.0e-6:
                goal_direction /= goal_norm
                self.escape_goal_alignment = float(
                    np.dot(direction, goal_direction)
                )
                if (
                    self.escape_goal_alignment
                    < self.config.observed_escape_min_goal_alignment
                ):
                    return None
        return (
            np.asarray(position, dtype=np.float64).reshape(3)
            + self.config.observed_escape_distance_m * direction
        )

    def _decision(self, transition=None):
        def target(value):
            return None if value is None else tuple(float(x) for x in value)

        return RecoveryDecisionV2(
            mode=self.mode,
            transition=transition,
            retreat_target_world=target(self.retreat_target),
            escape_target_world=target(self.escape_target),
            zero_feasible_replans=self.zero_feasible_replans,
            collision_floor_replans=self.collision_floor_replans,
            release_feasible_replans=self.release_feasible_replans,
            scan_offset_deg=math.degrees(self.scan_offset_rad),
            scan_legs_completed=self.scan_legs_completed,
            escape_goal_alignment=(
                None if self.escape_goal_alignment is None
                else float(self.escape_goal_alignment)
            ),
        )

    def _enter_scan(self, depth):
        self.mode = self.YAW_SCAN
        self.scan_direction = self.scan_direction_from_depth(depth)
        if (
            self.last_position is not None
            and self.last_rotation is not None
            and self.last_goal is not None
        ):
            goal_world = self.last_goal - self.last_position
            goal_body = self.last_rotation.T @ goal_world
            goal_yaw = math.atan2(float(goal_body[1]), float(goal_body[0]))
            if abs(math.degrees(goal_yaw)) >= self.config.goal_biased_scan_deadband_deg:
                self.scan_direction = 1.0 if goal_yaw >= 0.0 else -1.0
        self.scan_offset_rad = 0.0
        self.scan_leg_complete = False
        self.scan_legs_completed = 0

    def observe(self, feasible_candidate_count, collision_floor_present,
                speed_mps, position_world, depth,
                rotation_world_from_body=None, goal_world=None,
                goal_feasible_candidate_count=None,
                recovery_feasible_candidate_count=None):
        if not self.config.enabled:
            return self._decision()
        feasible_candidate_count = int(feasible_candidate_count)
        goal_feasible_candidate_count = (
            feasible_candidate_count
            if goal_feasible_candidate_count is None
            else int(goal_feasible_candidate_count)
        )
        recovery_feasible_candidate_count = (
            goal_feasible_candidate_count
            if recovery_feasible_candidate_count is None
            else int(recovery_feasible_candidate_count)
        )
        collision_floor_present = bool(collision_floor_present)
        speed_mps = float(speed_mps)
        position = np.asarray(position_world, dtype=np.float64).reshape(3)
        self.last_position = position.copy()
        if rotation_world_from_body is not None:
            rotation = np.asarray(
                rotation_world_from_body, dtype=np.float64
            ).reshape(3, 3)
            if np.isfinite(rotation).all():
                self.last_rotation = rotation.copy()
        if goal_world is not None:
            goal = np.asarray(goal_world, dtype=np.float64).reshape(3)
            if np.isfinite(goal).all():
                self.last_goal = goal.copy()

        release_candidate_count = (
            recovery_feasible_candidate_count
            if self.mode in (self.BRAKING, self.YAW_SCAN, self.HOLD)
            else feasible_candidate_count
        )
        if release_candidate_count > 0 and not collision_floor_present:
            self.release_feasible_replans += 1
            self.zero_feasible_replans = 0
            self.collision_floor_replans = 0
        else:
            self.release_feasible_replans = 0
            if feasible_candidate_count == 0:
                self.zero_feasible_replans += 1
            if collision_floor_present:
                self.collision_floor_replans += 1

        if self.mode == self.NORMAL:
            if self.zero_feasible_replans >= self.config.zero_feasible_trigger_replans:
                self.mode = self.BRAKING
                self.scan_direction = self.scan_direction_from_depth(depth)
                return self._decision("network_to_braking")
            return self._decision()

        if self.mode == self.BRAKING:
            if self.release_feasible_replans >= self.config.release_feasible_replans:
                self.mode = self.NORMAL
                return self._decision("braking_to_network")
            if speed_mps <= self.config.stationary_speed_mps:
                target = None
                if self.collision_floor_replans >= self.config.collision_floor_trigger_replans:
                    target = self._breadcrumb_target(position)
                if target is not None:
                    self.retreat_target = target
                    self.mode = self.RETREAT
                    return self._decision("braking_to_breadcrumb_retreat")
                self._enter_scan(depth)
                return self._decision("braking_to_yaw_scan")
            return self._decision()

        if self.mode == self.RETREAT:
            if self.retreat_target is None:
                self._enter_scan(depth)
                return self._decision("retreat_missing_target_to_yaw_scan")
            if np.linalg.norm(position - self.retreat_target) <= self.config.retreat_arrival_radius_m:
                self.retreat_target = None
                self._enter_scan(depth)
                return self._decision("breadcrumb_retreat_to_yaw_scan")
            return self._decision()

        if self.mode == self.OBSERVED_ESCAPE:
            if self.escape_target is None:
                self.mode = self.HOLD
                return self._decision("escape_missing_target_to_hold")
            if np.linalg.norm(position - self.escape_target) <= self.config.observed_escape_arrival_radius_m:
                self.escape_target = None
                self.mode = self.NORMAL
                self.zero_feasible_replans = 0
                self.release_feasible_replans = 0
                return self._decision("observed_escape_to_network")
            return self._decision()

        if self.mode == self.YAW_SCAN:
            if (
                self.release_feasible_replans
                >= self.config.scan_release_feasible_replans
            ):
                self.mode = self.NORMAL
                self.retreat_target = None
                self.escape_target = None
                return self._decision("yaw_scan_to_network")
            if self.scan_leg_complete:
                self.scan_legs_completed += 1
                target = self._observed_escape_target(
                    position, rotation_world_from_body, depth
                )
                if target is not None:
                    self.escape_target = target
                    self.mode = self.OBSERVED_ESCAPE
                    return self._decision("yaw_scan_to_observed_escape")
                if self.scan_legs_completed < self.config.max_scan_legs:
                    self.scan_direction *= -1.0
                    self.scan_leg_complete = False
                    return self._decision("yaw_scan_reverse")
                target = self._breadcrumb_target(position)
                if target is not None:
                    self.retreat_target = target
                    self.mode = self.RETREAT
                    return self._decision("yaw_scan_to_breadcrumb_retreat")
                self.mode = self.HOLD
                return self._decision("yaw_scan_exhausted_to_hold")
            return self._decision()

        if self.mode == self.HOLD:
            if self.release_feasible_replans >= self.config.release_feasible_replans:
                self.mode = self.NORMAL
                return self._decision("hold_to_network")
            return self._decision()

        raise RuntimeError(f"unknown deadlock recovery mode: {self.mode}")

    def reject_observed_escape(self):
        if self.mode != self.OBSERVED_ESCAPE:
            raise RuntimeError("observed escape rejection outside escape mode")
        self.escape_target = None
        if self.scan_legs_completed < self.config.max_scan_legs:
            self.mode = self.YAW_SCAN
            self.scan_direction *= -1.0
            self.scan_leg_complete = False
            return self._decision("observed_escape_rejected_to_reverse_scan")
        self.mode = self.HOLD
        return self._decision("observed_escape_rejected_to_hold")

    def reject_breadcrumb_retreat(self, depth):
        if self.mode != self.RETREAT:
            raise RuntimeError("breadcrumb rejection outside retreat mode")
        self.retreat_target = None
        self._enter_scan(depth)
        return self._decision("breadcrumb_retreat_rejected_to_yaw_scan")

    def yaw_command(self, last_yaw, dt):
        if self.mode != self.YAW_SCAN:
            raise RuntimeError("yaw command is only valid while scanning")
        limit = math.radians(self.config.max_scan_angle_deg)
        rate = math.radians(self.config.yaw_scan_rate_deg_s) * self.scan_direction
        old_offset = self.scan_offset_rad
        new_offset = old_offset + rate * float(dt)
        boundary = limit if self.scan_direction > 0 else -limit
        if ((self.scan_direction > 0 and new_offset >= boundary)
                or (self.scan_direction < 0 and new_offset <= boundary)):
            new_offset = boundary
            self.scan_leg_complete = True
        delta = new_offset - old_offset
        self.scan_offset_rad = new_offset
        yaw = float(last_yaw) + delta
        yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
        return yaw, (delta / float(dt) if dt > 0 else 0.0)


__all__ = [
    "DeadlockRecoveryConfigV2", "DeadlockRecoveryV2", "RecoveryDecisionV2",
]
