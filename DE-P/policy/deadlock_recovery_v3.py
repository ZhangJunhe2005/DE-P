"""Minimal bounded scan-only recovery for a forward-looking local policy.

This recovery deliberately owns no translational motion.  Runtime safety keeps
commanding a finite-horizon brake while the camera makes one bounded turn.
Every new depth frame is still evaluated by the normal network and the same
runtime safety shield; control returns to the network only when its *selected*
candidate is both feasible and makes positive goal progress.

The class is versioned separately from :mod:`deadlock_recovery_v2` because V2
also contains breadcrumb/observed-space translation and a terminal HOLD state.
Those historical semantics must not be changed by the Pillar-only recovery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np


@dataclass(frozen=True)
class DeadlockRecoveryConfigV3:
    enabled: bool = True
    zero_feasible_trigger_replans: int = 15
    selected_release_replans: int = 3
    selected_min_goal_progress_m: float = 0.25
    stationary_speed_mps: float = 0.25
    yaw_scan_rate_deg_s: float = 60.0
    max_scan_angle_deg: float = 60.0
    scan_angle_step_deg: float = 30.0
    max_escalated_scan_angle_deg: float = 120.0
    scan_timeout_s: float = 2.25
    heading_commitment_s: float = 0.45
    cooldown_s: float = 2.0

    @classmethod
    def from_mapping(cls, value):
        value = {} if value is None else dict(value)
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown deadlock_recovery v3 keys: {unknown}")
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        if self.zero_feasible_trigger_replans < 1:
            raise ValueError("zero-feasible trigger must be positive")
        if self.selected_release_replans < 1:
            raise ValueError("selected-candidate release count must be positive")
        if self.selected_min_goal_progress_m < 0.0:
            raise ValueError("selected goal progress must be non-negative")
        positive = (
            self.stationary_speed_mps,
            self.yaw_scan_rate_deg_s,
            self.max_scan_angle_deg,
            self.scan_angle_step_deg,
            self.max_escalated_scan_angle_deg,
            self.scan_timeout_s,
            self.heading_commitment_s,
            self.cooldown_s,
        )
        if min(positive) <= 0.0:
            raise ValueError("scan limits, speed and cooldown must be positive")
        if self.max_scan_angle_deg > self.max_escalated_scan_angle_deg:
            raise ValueError("initial scan angle exceeds escalated scan limit")
        if self.max_escalated_scan_angle_deg > 120.0:
            raise ValueError("scan-only recovery may not exceed 120 degrees")

    def contract(self):
        return {
            "version": "deadlock_recovery_v3_bounded_scan_only",
            "trigger": "consecutive_zero_runtime_safe_candidates",
            "normal_translation_authority": "network",
            "recovery_translation_authority": "runtime_brake_only",
            "scan_direction": "depth_free_space_only_no_goal_override",
            "release": "selected_runtime_safe_goal_progress_candidate",
            "post_release_yaw": "bounded_scan_heading_commitment_translation_remains_network_owned",
            "exhaustion": "return_to_network_with_reentry_cooldown",
            "retreat_enabled": False,
            "observed_escape_enabled": False,
            "maximum_scan_legs_per_attempt": 1,
            "repeated_attempt_policy": (
                "increase_scan_angle_then_alternate_direction_up_to_120_deg"
            ),
            **asdict(self),
        }


@dataclass(frozen=True)
class RecoveryDecisionV3:
    mode: str
    transition: str | None
    retreat_target_world: None
    escape_target_world: None
    zero_feasible_replans: int
    collision_floor_replans: int
    release_feasible_replans: int
    scan_offset_deg: float
    scan_legs_completed: int
    escape_goal_alignment: None
    selected_release_replans: int
    cooldown_remaining_s: float
    scan_attempt: int
    current_scan_limit_deg: float
    scan_direction: float
    selected_candidate_eligible: bool
    selected_candidate_goal_progress_m: float | None


class DeadlockRecoveryV3:
    """Brake, look once toward the freer half-image, then yield.

    The public mode strings intentionally match V2 where they overlap, so the
    ROS command loop can remain agnostic to the recovery implementation.
    """

    NORMAL = "network"
    BRAKING = "recovery_braking"
    YAW_SCAN = "recovery_yaw_scan"

    def __init__(self, config: DeadlockRecoveryConfigV3):
        config.validate()
        self.config = config
        self.mode = self.NORMAL
        self.zero_feasible_replans = 0
        self.selected_release_count = 0
        self.scan_direction = 1.0
        self.scan_offset_rad = 0.0
        self.scan_leg_complete = False
        self.scan_legs_completed = 0
        self.scan_started_s = None
        self.unsuccessful_scan_attempts = 0
        self.current_scan_limit_deg = self.config.max_scan_angle_deg
        self.heading_commitment_until_s = 0.0
        self.cooldown_until_s = 0.0
        self.last_position = None
        self.last_selected_candidate_eligible = False
        self.last_selected_candidate_goal_progress_m = None

    @staticmethod
    def _now(now_s):
        value = time.monotonic() if now_s is None else float(now_s)
        if not math.isfinite(value):
            raise ValueError("recovery clock must be finite")
        return value

    def reset(self, position_world=None):
        self.mode = self.NORMAL
        self.zero_feasible_replans = 0
        self.selected_release_count = 0
        self.scan_direction = 1.0
        self.scan_offset_rad = 0.0
        self.scan_leg_complete = False
        self.scan_legs_completed = 0
        self.scan_started_s = None
        self.unsuccessful_scan_attempts = 0
        self.current_scan_limit_deg = self.config.max_scan_angle_deg
        self.heading_commitment_until_s = 0.0
        self.cooldown_until_s = 0.0
        self.last_position = None
        self.last_selected_candidate_eligible = False
        self.last_selected_candidate_goal_progress_m = None
        if position_world is not None:
            self.record_position(position_world)

    def record_position(self, position_world):
        point = np.asarray(position_world, dtype=np.float64).reshape(3)
        if np.isfinite(point).all():
            self.last_position = point.copy()

    @staticmethod
    def scan_direction_from_depth(depth):
        array = np.asarray(depth, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] < 4:
            return 1.0
        midpoint = array.shape[1] // 2

        def free_score(region):
            valid = region[np.isfinite(region) & (region > 0.0)]
            return float(np.quantile(valid, 0.65)) if len(valid) else 0.0

        # Image-left maps to positive body-y/positive yaw for the configured
        # optical-to-body transform; image-right therefore maps to negative yaw.
        return (
            1.0
            if free_score(array[:, :midpoint])
            >= free_score(array[:, midpoint:])
            else -1.0
        )

    def _decision(self, now_s, transition=None):
        return RecoveryDecisionV3(
            mode=self.mode,
            transition=transition,
            retreat_target_world=None,
            escape_target_world=None,
            zero_feasible_replans=self.zero_feasible_replans,
            collision_floor_replans=0,
            release_feasible_replans=self.selected_release_count,
            scan_offset_deg=math.degrees(self.scan_offset_rad),
            scan_legs_completed=self.scan_legs_completed,
            escape_goal_alignment=None,
            selected_release_replans=self.selected_release_count,
            cooldown_remaining_s=max(0.0, self.cooldown_until_s - now_s),
            scan_attempt=self.unsuccessful_scan_attempts + 1,
            current_scan_limit_deg=self.current_scan_limit_deg,
            scan_direction=self.scan_direction,
            selected_candidate_eligible=(
                self.last_selected_candidate_eligible
            ),
            selected_candidate_goal_progress_m=(
                self.last_selected_candidate_goal_progress_m
            ),
        )

    def _return_to_network(
        self, now_s, transition, preserve_scan_heading=False,
        successful_release=False,
    ):
        self.mode = self.NORMAL
        self.zero_feasible_replans = 0
        self.selected_release_count = 0
        self.scan_started_s = None
        self.heading_commitment_until_s = (
            now_s + self.config.heading_commitment_s
            if preserve_scan_heading else 0.0
        )
        self.cooldown_until_s = now_s + self.config.cooldown_s
        if successful_release:
            self.unsuccessful_scan_attempts = 0
            self.current_scan_limit_deg = self.config.max_scan_angle_deg
        return self._decision(now_s, transition)

    def heading_commitment_active(self, now_s=None):
        """Return whether yaw should briefly remain at the scan heading.

        The normal network candidate and runtime shield continue to own all
        translation during this interval; only fast goal-yaw snap-back is
        delayed.
        """
        return self._now(now_s) < self.heading_commitment_until_s

    def observe(
        self,
        feasible_candidate_count,
        collision_floor_present,
        speed_mps,
        position_world,
        depth,
        rotation_world_from_body=None,
        goal_world=None,
        goal_feasible_candidate_count=None,
        recovery_feasible_candidate_count=None,
        selected_candidate_feasible=False,
        selected_candidate_goal_progress_m=None,
        now_s=None,
    ):
        del (
            rotation_world_from_body,
            goal_world,
            goal_feasible_candidate_count,
            recovery_feasible_candidate_count,
        )
        now_s = self._now(now_s)
        self.record_position(position_world)
        if not self.config.enabled:
            return self._decision(now_s)

        feasible_candidate_count = int(feasible_candidate_count)
        collision_floor_present = bool(collision_floor_present)
        speed_mps = float(speed_mps)
        progress = selected_candidate_goal_progress_m
        selected_eligible = bool(
            selected_candidate_feasible
            and not collision_floor_present
            and progress is not None
            and math.isfinite(float(progress))
            and float(progress) >= self.config.selected_min_goal_progress_m
        )
        self.last_selected_candidate_eligible = selected_eligible
        self.last_selected_candidate_goal_progress_m = (
            None if progress is None else float(progress)
        )
        self.selected_release_count = (
            self.selected_release_count + 1 if selected_eligible else 0
        )

        if self.mode == self.NORMAL:
            # Cooldown suppresses only another recovery takeover.  It never
            # blocks a runtime-safe network candidate from being executed.
            if now_s < self.cooldown_until_s:
                self.zero_feasible_replans = 0
                return self._decision(now_s)
            if (
                selected_eligible
                and self.selected_release_count
                >= self.config.selected_release_replans
            ):
                # Sustained useful network motion closes the escalation chain;
                # the next unrelated deadlock starts with the normal 60 deg
                # observation turn.
                self.unsuccessful_scan_attempts = 0
                self.current_scan_limit_deg = self.config.max_scan_angle_deg
            self.zero_feasible_replans = (
                self.zero_feasible_replans + 1
                if feasible_candidate_count == 0 else 0
            )
            if (
                self.zero_feasible_replans
                >= self.config.zero_feasible_trigger_replans
            ):
                self.mode = self.BRAKING
                self.selected_release_count = 0
                return self._decision(now_s, "network_to_braking")
            return self._decision(now_s)

        if self.mode == self.BRAKING:
            if (
                self.selected_release_count
                >= self.config.selected_release_replans
            ):
                return self._return_to_network(
                    now_s, "braking_to_network_selected_candidate",
                    successful_release=True,
                )
            if speed_mps <= self.config.stationary_speed_mps:
                self.mode = self.YAW_SCAN
                preferred_direction = self.scan_direction_from_depth(depth)
                self.scan_direction = (
                    preferred_direction
                    if self.unsuccessful_scan_attempts % 2 == 0
                    else -preferred_direction
                )
                self.current_scan_limit_deg = min(
                    self.config.max_escalated_scan_angle_deg,
                    self.config.max_scan_angle_deg
                    + self.unsuccessful_scan_attempts
                    * self.config.scan_angle_step_deg,
                )
                self.scan_offset_rad = 0.0
                self.scan_leg_complete = False
                self.scan_legs_completed = 0
                self.scan_started_s = now_s
                return self._decision(now_s, "braking_to_bounded_scan")
            return self._decision(now_s)

        if self.mode == self.YAW_SCAN:
            if (
                self.selected_release_count
                >= self.config.selected_release_replans
            ):
                return self._return_to_network(
                    now_s,
                    "bounded_scan_to_network_selected_candidate",
                    preserve_scan_heading=True,
                    successful_release=True,
                )
            timed_out = (
                self.scan_started_s is not None
                and now_s - self.scan_started_s >= self.config.scan_timeout_s
            )
            if self.scan_leg_complete or timed_out:
                self.scan_legs_completed = 1
                if selected_eligible:
                    transition = (
                        "bounded_scan_timeout_to_network_selected_candidate"
                        if timed_out and not self.scan_leg_complete else
                        "bounded_scan_limit_to_network_selected_candidate"
                    )
                    return self._return_to_network(
                        now_s,
                        transition,
                        preserve_scan_heading=True,
                        successful_release=True,
                    )
                transition = (
                    "bounded_scan_timeout_to_network"
                    if timed_out and not self.scan_leg_complete
                    else "bounded_scan_limit_to_network"
                )
                self.unsuccessful_scan_attempts += 1
                return self._return_to_network(now_s, transition)
            return self._decision(now_s)

        raise RuntimeError(f"unknown deadlock recovery v3 mode: {self.mode}")

    def yaw_command(self, last_yaw, dt):
        if self.mode != self.YAW_SCAN:
            raise RuntimeError("yaw command is only valid while scanning")
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("yaw command dt must be finite and positive")
        # A useful network candidate has appeared at the current camera
        # heading.  Hold that view while consecutive replans confirm it,
        # instead of sweeping past the opening and resetting the evidence.
        # Translation remains the finite-horizon brake until observe() hands
        # authority back to the network.
        if self.last_selected_candidate_eligible:
            return float(last_yaw), 0.0
        limit = math.radians(self.current_scan_limit_deg)
        rate = math.radians(self.config.yaw_scan_rate_deg_s) * self.scan_direction
        old_offset = self.scan_offset_rad
        new_offset = old_offset + rate * dt
        boundary = limit if self.scan_direction > 0.0 else -limit
        if (
            (self.scan_direction > 0.0 and new_offset >= boundary)
            or (self.scan_direction < 0.0 and new_offset <= boundary)
        ):
            new_offset = boundary
            self.scan_leg_complete = True
        delta = new_offset - old_offset
        self.scan_offset_rad = new_offset
        yaw = (float(last_yaw) + delta + math.pi) % (2.0 * math.pi) - math.pi
        return yaw, delta / dt


def deadlock_recovery_mapping_v4_7_pillar(base):
    """Create the frozen, scan-only Pillar recovery configuration."""
    base = {} if base is None else dict(base)
    return {
        "enabled": bool(base.get("enabled", True)),
        "zero_feasible_trigger_replans": 15,
        "selected_release_replans": 3,
        "selected_min_goal_progress_m": 0.25,
        "stationary_speed_mps": 0.25,
        "yaw_scan_rate_deg_s": 60.0,
        "max_scan_angle_deg": 60.0,
        "scan_angle_step_deg": 30.0,
        "max_escalated_scan_angle_deg": 120.0,
        # Long enough for the largest 120 degree attempt at 60 deg/s, plus a
        # small scheduling allowance.  The angular limit remains authoritative
        # for shorter attempts.
        "scan_timeout_s": 2.25,
        "heading_commitment_s": 0.45,
        "cooldown_s": 2.0,
    }


__all__ = [
    "DeadlockRecoveryConfigV3",
    "DeadlockRecoveryV3",
    "RecoveryDecisionV3",
    "deadlock_recovery_mapping_v4_7_pillar",
]
