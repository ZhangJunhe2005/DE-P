"""Deterministic recovery for a forward-camera, forward-lattice local planner.

The neural policy is intentionally not asked to hallucinate free space behind
the vehicle.  When every forward candidate is rejected, this state machine
first stops, then changes the camera heading.  If the vehicle has already
entered the physical collision floor, it may return to a recent breadcrumb
before scanning.  Breadcrumb retreat is bounded and is never described as a
network prediction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DeadlockRecoveryConfigV1:
    enabled: bool = True
    zero_feasible_trigger_replans: int = 15
    collision_floor_trigger_replans: int = 5
    release_feasible_replans: int = 5
    stationary_speed_mps: float = 0.35
    yaw_scan_rate_deg_s: float = 45.0
    breadcrumb_spacing_m: float = 0.12
    breadcrumb_capacity: int = 600
    retreat_enabled: bool = True
    retreat_distance_m: float = 1.20
    retreat_arrival_radius_m: float = 0.20

    @classmethod
    def from_mapping(cls, value):
        value = {} if value is None else dict(value)
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown deadlock_recovery keys: {unknown}")
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        if min(
            self.zero_feasible_trigger_replans,
            self.collision_floor_trigger_replans,
            self.release_feasible_replans,
            self.breadcrumb_capacity,
        ) < 1:
            raise ValueError("deadlock recovery counters must be positive")
        if self.stationary_speed_mps <= 0 or self.yaw_scan_rate_deg_s <= 0:
            raise ValueError("deadlock recovery speed/rate must be positive")
        if min(
            self.breadcrumb_spacing_m,
            self.retreat_distance_m,
            self.retreat_arrival_radius_m,
        ) <= 0:
            raise ValueError("deadlock recovery distances must be positive")
        if self.retreat_arrival_radius_m >= self.retreat_distance_m:
            raise ValueError("retreat arrival radius must be smaller than distance")

    def contract(self):
        return {
            "version": "deadlock_recovery_v1",
            "sensing_boundary": "forward_depth_plus_recent_flown_breadcrumbs",
            "normal_policy": "network_owns_translation",
            "deadlock_policy": "brake_then_yaw_scan",
            "collision_floor_policy": "bounded_breadcrumb_retreat_then_yaw_scan",
            **asdict(self),
        }


@dataclass(frozen=True)
class RecoveryDecisionV1:
    mode: str
    transition: str | None
    retreat_target_world: tuple[float, float, float] | None
    zero_feasible_replans: int
    collision_floor_replans: int
    release_feasible_replans: int


class DeadlockRecoveryV1:
    NORMAL = "network"
    BRAKING = "recovery_braking"
    RETREAT = "recovery_breadcrumb_retreat"
    YAW_SCAN = "recovery_yaw_scan"

    def __init__(self, config: DeadlockRecoveryConfigV1):
        config.validate()
        self.config = config
        self.breadcrumbs = deque(maxlen=config.breadcrumb_capacity)
        self.mode = self.NORMAL
        self.zero_feasible_replans = 0
        self.collision_floor_replans = 0
        self.release_feasible_replans = 0
        self.scan_direction = 1.0
        self.retreat_target = None

    def reset(self, position_world=None):
        self.mode = self.NORMAL
        self.zero_feasible_replans = 0
        self.collision_floor_replans = 0
        self.release_feasible_replans = 0
        self.scan_direction = 1.0
        self.retreat_target = None
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
        left = array[:, :midpoint]
        right = array[:, midpoint:]

        def free_score(region):
            valid = region[np.isfinite(region)]
            return float(np.quantile(valid, 0.65)) if len(valid) else 0.0

        # Image-left maps to positive body yaw for the configured optical frame.
        return 1.0 if free_score(left) >= free_score(right) else -1.0

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

    def _decision(self, transition=None):
        target = None if self.retreat_target is None else tuple(
            float(value) for value in self.retreat_target
        )
        return RecoveryDecisionV1(
            mode=self.mode,
            transition=transition,
            retreat_target_world=target,
            zero_feasible_replans=self.zero_feasible_replans,
            collision_floor_replans=self.collision_floor_replans,
            release_feasible_replans=self.release_feasible_replans,
        )

    def observe(self, feasible_candidate_count, collision_floor_present,
                speed_mps, position_world, depth):
        if not self.config.enabled:
            return self._decision()
        feasible_candidate_count = int(feasible_candidate_count)
        collision_floor_present = bool(collision_floor_present)
        speed_mps = float(speed_mps)
        position = np.asarray(position_world, dtype=np.float64).reshape(3)

        if feasible_candidate_count > 0 and not collision_floor_present:
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
            if (
                self.release_feasible_replans
                >= self.config.release_feasible_replans
            ):
                self.mode = self.NORMAL
                return self._decision("braking_to_network")
            if speed_mps <= self.config.stationary_speed_mps:
                target = None
                if (
                    self.collision_floor_replans
                    >= self.config.collision_floor_trigger_replans
                ):
                    target = self._breadcrumb_target(position)
                if target is not None:
                    self.retreat_target = target
                    self.mode = self.RETREAT
                    return self._decision("braking_to_breadcrumb_retreat")
                self.mode = self.YAW_SCAN
                return self._decision("braking_to_yaw_scan")
            return self._decision()

        if self.mode == self.RETREAT:
            if self.retreat_target is None:
                self.mode = self.YAW_SCAN
                return self._decision("retreat_missing_target_to_yaw_scan")
            if (
                np.linalg.norm(position - self.retreat_target)
                <= self.config.retreat_arrival_radius_m
            ):
                self.retreat_target = None
                self.mode = self.YAW_SCAN
                self.scan_direction = self.scan_direction_from_depth(depth)
                return self._decision("breadcrumb_retreat_to_yaw_scan")
            return self._decision()

        if self.mode == self.YAW_SCAN:
            if (
                self.release_feasible_replans
                >= self.config.release_feasible_replans
            ):
                self.mode = self.NORMAL
                self.retreat_target = None
                return self._decision("yaw_scan_to_network")
            return self._decision()

        raise RuntimeError(f"unknown deadlock recovery mode: {self.mode}")

    def yaw_command(self, last_yaw, dt):
        if self.mode != self.YAW_SCAN:
            raise RuntimeError("yaw command is only valid while scanning")
        rate = math.radians(self.config.yaw_scan_rate_deg_s) * self.scan_direction
        yaw = float(last_yaw) + rate * float(dt)
        yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
        return yaw, rate
