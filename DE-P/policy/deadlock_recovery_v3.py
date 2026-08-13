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

from collections import deque
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
    accumulate_zero_during_cooldown: bool = False
    defer_scan_reset_until_post_release_confirmation: bool = False
    post_release_confirmation_replans: int = 3
    require_consistent_selected_horizontal_sector: bool = False
    motion_stagnation_enabled: bool = False
    motion_stagnation_window_s: float = 2.0
    motion_stagnation_min_displacement_m: float = 0.20
    motion_stagnation_trigger_replans: int = 30
    provisional_handoff_validation_enabled: bool = False
    handoff_validation_window_s: float = 1.20
    handoff_min_displacement_m: float = 0.50
    handoff_min_forward_clearance_gain_m: float = 0.50
    handoff_forward_clearance_confirmation_replans: int = 5
    handoff_directional_progress_enabled: bool = False
    handoff_max_directional_retreat_m: float = 0.10

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
        if self.post_release_confirmation_replans < 1:
            raise ValueError("post-release confirmation count must be positive")
        if self.motion_stagnation_trigger_replans < 1:
            raise ValueError("motion-stagnation trigger must be positive")
        if self.motion_stagnation_window_s <= 0.0:
            raise ValueError("motion-stagnation window must be positive")
        if self.motion_stagnation_min_displacement_m <= 0.0:
            raise ValueError("motion-stagnation displacement must be positive")
        if self.handoff_validation_window_s <= 0.0:
            raise ValueError("handoff validation window must be positive")
        if self.handoff_min_displacement_m <= 0.0:
            raise ValueError("handoff displacement must be positive")
        if self.handoff_min_forward_clearance_gain_m <= 0.0:
            raise ValueError("handoff clearance gain must be positive")
        if self.handoff_forward_clearance_confirmation_replans < 1:
            raise ValueError(
                "handoff clearance confirmation count must be positive"
            )
        if self.handoff_max_directional_retreat_m < 0.0:
            raise ValueError("handoff directional retreat must be non-negative")
        if (
            self.handoff_directional_progress_enabled
            and not self.provisional_handoff_validation_enabled
        ):
            raise ValueError(
                "directional handoff requires provisional validation"
            )
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
        values = asdict(self)
        stable_sector_handoff = bool(
            self.require_consistent_selected_horizontal_sector
        )
        motion_stagnation = bool(self.motion_stagnation_enabled)
        verified_handoff = bool(
            self.provisional_handoff_validation_enabled
        )
        evidence_preserving = bool(
            self.accumulate_zero_during_cooldown
            or self.defer_scan_reset_until_post_release_confirmation
            or self.post_release_confirmation_replans
            != self.selected_release_replans
        )
        if not evidence_preserving:
            # Preserve the frozen V4.7 public contract byte-for-byte.  The new
            # fields are an opt-in V4.8 extension, not a retroactive V4.7
            # semantic change.
            values.pop("accumulate_zero_during_cooldown")
            values.pop("defer_scan_reset_until_post_release_confirmation")
            values.pop("post_release_confirmation_replans")
        if not stable_sector_handoff:
            # This opt-in field belongs to V4.8.1.  Omitting its default keeps
            # the frozen V4.7 and already validated V4.8 contracts unchanged.
            values.pop("require_consistent_selected_horizontal_sector")
        if not motion_stagnation:
            # Keep all frozen V4.7/V4.8/V4.8.1 contracts unchanged.  The
            # motion-window fields are an opt-in V4.8.2 extension.
            values.pop("motion_stagnation_enabled")
            values.pop("motion_stagnation_window_s")
            values.pop("motion_stagnation_min_displacement_m")
            values.pop("motion_stagnation_trigger_replans")
        if not verified_handoff:
            # V4.8.5 is opt-in.  Keep every frozen V4.7--V4.8.2 public
            # contract byte-for-byte compatible when motion-verified handoff
            # is disabled.
            values.pop("provisional_handoff_validation_enabled")
            values.pop("handoff_validation_window_s")
            values.pop("handoff_min_displacement_m")
            values.pop("handoff_min_forward_clearance_gain_m")
            values.pop("handoff_forward_clearance_confirmation_replans")
            values.pop("handoff_directional_progress_enabled")
            values.pop("handoff_max_directional_retreat_m")
        elif not self.handoff_directional_progress_enabled:
            # Directional progress is a V4.9.1 opt-in refinement.  Preserve
            # the frozen V4.8.5/V4.9 handoff contract exactly when disabled.
            values.pop("handoff_directional_progress_enabled")
            values.pop("handoff_max_directional_retreat_m")
        return {
            "version": (
                "deadlock_recovery_v4_4_directional_handoff"
                if self.handoff_directional_progress_enabled else
                "deadlock_recovery_v4_3_motion_verified_handoff"
                if verified_handoff else
                (
                    "deadlock_recovery_v4_2_universal_motion_stagnation"
                    if motion_stagnation else
                    (
                        "deadlock_recovery_v4_1_stable_sector_handoff"
                        if stable_sector_handoff else
                        (
                            "deadlock_recovery_v4_evidence_preserving_handoff"
                            if evidence_preserving else
                            "deadlock_recovery_v3_bounded_scan_only"
                        )
                    )
                )
            ),
            "trigger": (
                "consecutive_zero_runtime_safe_candidates_or_observed_"
                "motion_stagnation"
                if motion_stagnation else
                "consecutive_zero_runtime_safe_candidates"
            ),
            "normal_translation_authority": "network",
            "recovery_translation_authority": "runtime_brake_only",
            "scan_direction": "depth_free_space_only_no_goal_override",
            "release": (
                "provisional_same_sector_candidate_then_directional_motion_"
                "toward_temporary_goal"
                if self.handoff_directional_progress_enabled else
                "provisional_same_sector_candidate_then_measured_motion_or_"
                "stable_forward_clearance_verification"
                if verified_handoff else
                (
                    "same_horizontal_sector_selected_runtime_safe_goal_"
                    "progress_candidate"
                    if stable_sector_handoff else
                    "selected_runtime_safe_goal_progress_candidate"
                )
            ),
            "post_release_yaw": "bounded_scan_heading_commitment_translation_remains_network_owned",
            "exhaustion": "return_to_network_with_reentry_cooldown",
            "retreat_enabled": False,
            "observed_escape_enabled": False,
            "maximum_scan_legs_per_attempt": 1,
            "repeated_attempt_policy": (
                "increase_scan_angle_then_alternate_direction_up_to_120_deg"
            ),
            **values,
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
    selected_candidate_action_id: int | None
    selected_candidate_horizontal_sector_id: int | None
    selected_confirmation_horizontal_sector_id: int | None
    selected_candidate_min_observed_clearance_m: float | None
    handoff_confirmation_replans: int | None
    handoff_horizontal_sector_id: int | None
    motion_stagnation_replans: int
    motion_window_displacement_m: float | None
    motion_window_duration_s: float | None
    recovery_trigger_reason: str | None
    handoff_validation_active: bool
    handoff_validation_elapsed_s: float | None
    handoff_validation_displacement_m: float | None
    handoff_validation_directional_progress_m: float | None
    handoff_validation_max_directional_retreat_m: float | None
    handoff_validation_forward_clearance_gain_m: float | None
    handoff_forward_clearance_confirmation_replans: int
    handoff_validation_result: str | None
    handoff_preserved_scan_offset_deg: float | None
    handoff_preserved_scan_direction: float | None


def horizontal_sector_from_action_id(action_id, horizontal_sector_count):
    """Map a row-major ``[vertical, horizontal]`` action to its column.

    Network actions are flattened from the ``3 x 5`` output grid.  Actions
    separated by one horizontal row (for example 0, 5 and 10) therefore share
    the same physical horizontal sector while retaining different vertical
    choices.  Only equality of sectors is used by recovery; no left/right
    preference is introduced here.
    """
    if action_id is None:
        return None
    count = int(horizontal_sector_count)
    action = int(action_id)
    if count < 1:
        raise ValueError("horizontal sector count must be positive")
    if action < 0:
        raise ValueError("candidate action id must be non-negative")
    return action % count


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
        self.last_selected_candidate_action_id = None
        self.last_selected_candidate_horizontal_sector_id = None
        self.last_selected_candidate_min_observed_clearance_m = None
        self.selected_confirmation_horizontal_sector_id = None
        self.last_handoff_confirmation_replans = None
        self.last_handoff_horizontal_sector_id = None
        self.motion_history = deque()
        self.motion_stagnation_replans = 0
        self.motion_window_displacement_m = None
        self.motion_window_duration_s = None
        self.last_recovery_trigger_reason = None
        self.last_forward_clearance_m = None
        self.recovery_origin_forward_clearance_m = None
        self.handoff_validation_active = False
        self.handoff_validation_started_s = None
        self.handoff_validation_origin_position = None
        self.handoff_validation_origin_forward_clearance_m = None
        self.handoff_validation_displacement_m = None
        self.handoff_validation_direction_world = None
        self.handoff_validation_directional_progress_m = None
        self.handoff_validation_max_directional_retreat_m = None
        self.handoff_validation_forward_clearance_gain_m = None
        self.handoff_forward_clearance_confirmation_replans = 0
        self.handoff_validation_result = None
        self.handoff_preserved_scan_offset_rad = None
        self.handoff_preserved_scan_direction = None
        self.resume_failed_handoff_scan = False
        self.dynamic_yield_paused = False
        self.dynamic_yield_pause_started_s = None
        self.handoff_resume_rebase_pending = False
        self.handoff_pause_displacement_vector = None
        self.handoff_pause_forward_clearance_gain_m = None

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
        self.last_selected_candidate_action_id = None
        self.last_selected_candidate_horizontal_sector_id = None
        self.last_selected_candidate_min_observed_clearance_m = None
        self.selected_confirmation_horizontal_sector_id = None
        self.last_handoff_confirmation_replans = None
        self.last_handoff_horizontal_sector_id = None
        self.motion_history.clear()
        self.motion_stagnation_replans = 0
        self.motion_window_displacement_m = None
        self.motion_window_duration_s = None
        self.last_recovery_trigger_reason = None
        self.last_forward_clearance_m = None
        self.recovery_origin_forward_clearance_m = None
        self.handoff_validation_active = False
        self.handoff_validation_started_s = None
        self.handoff_validation_origin_position = None
        self.handoff_validation_origin_forward_clearance_m = None
        self.handoff_validation_displacement_m = None
        self.handoff_validation_direction_world = None
        self.handoff_validation_directional_progress_m = None
        self.handoff_validation_max_directional_retreat_m = None
        self.handoff_validation_forward_clearance_gain_m = None
        self.handoff_forward_clearance_confirmation_replans = 0
        self.handoff_validation_result = None
        self.handoff_preserved_scan_offset_rad = None
        self.handoff_preserved_scan_direction = None
        self.resume_failed_handoff_scan = False
        self.dynamic_yield_paused = False
        self.dynamic_yield_pause_started_s = None
        self.handoff_resume_rebase_pending = False
        self.handoff_pause_displacement_vector = None
        self.handoff_pause_forward_clearance_gain_m = None
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

    @staticmethod
    def forward_clearance_from_depth(depth):
        """Return a robust clearance score for the forward image centre.

        This is validation evidence, not a trajectory-safety Gate.  A low
        quantile over the centre band is deliberately used instead of one
        minimum pixel so sensor speckle cannot declare a provisional handoff
        successful.  Rotation alone must expose a consistently clearer view
        for several replans before it can substitute for measured motion.
        """
        array = np.asarray(depth, dtype=np.float32)
        if array.ndim != 2 or min(array.shape) < 4:
            return None
        height, width = array.shape
        row0, row1 = height // 4, height - height // 4
        col0, col1 = width // 3, width - width // 3
        region = array[row0:row1, col0:col1]
        valid = region[np.isfinite(region) & (region > 0.0)]
        if len(valid) == 0:
            return None
        return float(np.quantile(valid, 0.25))

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
            selected_candidate_action_id=(
                self.last_selected_candidate_action_id
            ),
            selected_candidate_horizontal_sector_id=(
                self.last_selected_candidate_horizontal_sector_id
            ),
            selected_confirmation_horizontal_sector_id=(
                self.selected_confirmation_horizontal_sector_id
            ),
            selected_candidate_min_observed_clearance_m=(
                self.last_selected_candidate_min_observed_clearance_m
            ),
            handoff_confirmation_replans=(
                self.last_handoff_confirmation_replans
            ),
            handoff_horizontal_sector_id=(
                self.last_handoff_horizontal_sector_id
            ),
            motion_stagnation_replans=self.motion_stagnation_replans,
            motion_window_displacement_m=self.motion_window_displacement_m,
            motion_window_duration_s=self.motion_window_duration_s,
            recovery_trigger_reason=self.last_recovery_trigger_reason,
            handoff_validation_active=self.handoff_validation_active,
            handoff_validation_elapsed_s=(
                None if not self.handoff_validation_active
                or self.handoff_validation_started_s is None else
                max(
                    0.0,
                    (
                        self.dynamic_yield_pause_started_s
                        if self.dynamic_yield_paused
                        and self.dynamic_yield_pause_started_s is not None
                        else now_s
                    ) - self.handoff_validation_started_s,
                )
            ),
            handoff_validation_displacement_m=(
                self.handoff_validation_displacement_m
            ),
            handoff_validation_directional_progress_m=(
                self.handoff_validation_directional_progress_m
            ),
            handoff_validation_max_directional_retreat_m=(
                self.handoff_validation_max_directional_retreat_m
            ),
            handoff_validation_forward_clearance_gain_m=(
                self.handoff_validation_forward_clearance_gain_m
            ),
            handoff_forward_clearance_confirmation_replans=(
                self.handoff_forward_clearance_confirmation_replans
            ),
            handoff_validation_result=self.handoff_validation_result,
            handoff_preserved_scan_offset_deg=(
                None if self.handoff_preserved_scan_offset_rad is None else
                math.degrees(self.handoff_preserved_scan_offset_rad)
            ),
            handoff_preserved_scan_direction=(
                self.handoff_preserved_scan_direction
            ),
        )

    def _start_handoff_validation(self, now_s):
        self.handoff_validation_active = True
        self.handoff_validation_started_s = float(now_s)
        self.handoff_resume_rebase_pending = False
        self.handoff_pause_displacement_vector = None
        self.handoff_pause_forward_clearance_gain_m = None
        self.handoff_validation_origin_position = (
            None if self.last_position is None else self.last_position.copy()
        )
        self.handoff_validation_origin_forward_clearance_m = (
            self.recovery_origin_forward_clearance_m
            if self.recovery_origin_forward_clearance_m is not None else
            self.last_forward_clearance_m
        )
        self.handoff_validation_displacement_m = 0.0
        self.handoff_validation_direction_world = None
        self.handoff_validation_directional_progress_m = 0.0
        self.handoff_validation_max_directional_retreat_m = 0.0
        self.handoff_validation_forward_clearance_gain_m = 0.0
        self.handoff_forward_clearance_confirmation_replans = 0
        self.handoff_validation_result = "pending"
        self.handoff_preserved_scan_offset_rad = float(self.scan_offset_rad)
        self.handoff_preserved_scan_direction = float(self.scan_direction)
        self.resume_failed_handoff_scan = False

    def _clear_handoff_validation(self):
        self.handoff_validation_active = False
        self.handoff_validation_started_s = None
        self.handoff_validation_origin_position = None
        self.handoff_validation_origin_forward_clearance_m = None
        self.handoff_validation_direction_world = None
        self.handoff_resume_rebase_pending = False
        self.handoff_pause_displacement_vector = None
        self.handoff_pause_forward_clearance_gain_m = None

    def _observe_handoff_validation(
        self, now_s, position_world, handoff_target_world=None,
    ):
        if not self.handoff_validation_active:
            return None
        point = np.asarray(position_world, dtype=np.float64).reshape(3)
        if self.handoff_resume_rebase_pending:
            # A dynamic yield can brake or drift the vehicle and can expose a
            # different depth view.  Neither change is evidence that the
            # provisional static opening succeeded.  Rebase the validation
            # origins on the first post-yield observation while preserving
            # exactly the displacement vector and clearance gain accumulated
            # before the pause.  Subsequent evidence then continues from that
            # frozen state instead of jumping across the paused interval.
            if self.handoff_pause_displacement_vector is not None:
                self.handoff_validation_origin_position = (
                    point - self.handoff_pause_displacement_vector
                )
            else:
                self.handoff_validation_origin_position = point.copy()
            frozen_gain = self.handoff_pause_forward_clearance_gain_m
            if frozen_gain is not None and self.last_forward_clearance_m is not None:
                self.handoff_validation_origin_forward_clearance_m = (
                    self.last_forward_clearance_m - frozen_gain
                )
            elif self.last_forward_clearance_m is not None:
                self.handoff_validation_origin_forward_clearance_m = (
                    self.last_forward_clearance_m
                )
            self.handoff_resume_rebase_pending = False
            self.handoff_pause_displacement_vector = None
            self.handoff_pause_forward_clearance_gain_m = None
            # The temporary goal stays fixed while a dynamic actor owns the
            # pause, but braking/drift can move the validation origin.  Rebuild
            # the direction below from that rebased origin instead of keeping
            # the stale pre-pause unit vector.
            self.handoff_validation_direction_world = None
        displacement = None
        if self.handoff_validation_origin_position is not None:
            displacement = point - self.handoff_validation_origin_position
            self.handoff_validation_displacement_m = float(np.linalg.norm(
                displacement
            ))
        if (
            self.config.handoff_directional_progress_enabled
            and self.handoff_validation_direction_world is None
            and handoff_target_world is not None
            and self.handoff_validation_origin_position is not None
        ):
            target = np.asarray(
                handoff_target_world, dtype=np.float64
            ).reshape(3)
            if not np.isfinite(target).all():
                raise ValueError("handoff target must be finite")
            direction = target - self.handoff_validation_origin_position
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= 1.0e-9:
                raise ValueError("handoff target must differ from its origin")
            self.handoff_validation_direction_world = (
                direction / direction_norm
            )
        if (
            self.config.handoff_directional_progress_enabled
            and displacement is not None
            and self.handoff_validation_direction_world is not None
        ):
            directional_progress = float(
                displacement @ self.handoff_validation_direction_world
            )
            self.handoff_validation_directional_progress_m = (
                directional_progress
            )
            self.handoff_validation_max_directional_retreat_m = max(
                float(self.handoff_validation_max_directional_retreat_m or 0.0),
                max(0.0, -directional_progress),
            )
        baseline = self.handoff_validation_origin_forward_clearance_m
        current = self.last_forward_clearance_m
        if baseline is not None and current is not None:
            self.handoff_validation_forward_clearance_gain_m = float(
                current - baseline
            )
            if (
                self.handoff_validation_forward_clearance_gain_m
                >= self.config.handoff_min_forward_clearance_gain_m
            ):
                self.handoff_forward_clearance_confirmation_replans += 1
            else:
                self.handoff_forward_clearance_confirmation_replans = 0
        else:
            self.handoff_validation_forward_clearance_gain_m = None
            self.handoff_forward_clearance_confirmation_replans = 0

        motion_measure = (
            self.handoff_validation_directional_progress_m
            if self.config.handoff_directional_progress_enabled else
            self.handoff_validation_displacement_m
        )
        motion_verified = bool(
            motion_measure is not None
            and motion_measure >= self.config.handoff_min_displacement_m
        )
        clearance_verified = bool(
            not self.config.handoff_directional_progress_enabled
            and
            self.handoff_forward_clearance_confirmation_replans
            >= self.config.handoff_forward_clearance_confirmation_replans
        )
        if motion_verified or clearance_verified:
            self.handoff_validation_result = (
                "measured_motion_verified"
                if motion_verified else "forward_clearance_verified"
            )
            self._clear_handoff_validation()
            self.unsuccessful_scan_attempts = 0
            self.current_scan_limit_deg = self.config.max_scan_angle_deg
            self.resume_failed_handoff_scan = False
            self.cooldown_until_s = 0.0
            self.recovery_origin_forward_clearance_m = None
            self._reset_motion_window(now_s)
            return (
                "handoff_measured_motion_verified"
                if motion_verified else
                "handoff_forward_clearance_verified"
            )

        if (
            self.config.handoff_directional_progress_enabled
            and self.handoff_validation_max_directional_retreat_m is not None
            and self.handoff_validation_max_directional_retreat_m
            > self.config.handoff_max_directional_retreat_m
        ):
            self.handoff_validation_result = "directional_retreat_exceeded"
            self._clear_handoff_validation()
            self.resume_failed_handoff_scan = True
            self.mode = self.BRAKING
            self.selected_release_count = 0
            self.selected_confirmation_horizontal_sector_id = None
            self.cooldown_until_s = 0.0
            self.last_recovery_trigger_reason = (
                "provisional_handoff_directional_retreat"
            )
            self._reset_motion_window()
            return "handoff_directional_retreat_to_braking"

        elapsed = float(now_s) - float(self.handoff_validation_started_s)
        if elapsed < self.config.handoff_validation_window_s:
            return None

        # The provisional opening did not move the vehicle out of the trap.
        # Preserve the previous direction/offset and the already escalated
        # 90/120-degree limit, brake, then continue the scan rather than
        # incorrectly resetting to a fresh 60-degree attempt.
        self.handoff_validation_result = "insufficient_escape_progress"
        self._clear_handoff_validation()
        self.resume_failed_handoff_scan = True
        self.mode = self.BRAKING
        self.selected_release_count = 0
        self.selected_confirmation_horizontal_sector_id = None
        self.cooldown_until_s = 0.0
        self.last_recovery_trigger_reason = (
            "provisional_handoff_no_escape_progress"
        )
        self._reset_motion_window()
        return "handoff_validation_failed_to_braking"

    def _reset_motion_window(self, now_s=None):
        self.motion_history.clear()
        if now_s is not None and self.last_position is not None:
            self.motion_history.append((float(now_s), self.last_position.copy()))
        self.motion_stagnation_replans = 0
        self.motion_window_displacement_m = None
        self.motion_window_duration_s = None

    def set_dynamic_yield_pause(self, paused=True, now_s=None):
        """Pause static-deadlock evidence during a causal dynamic yield.

        A crossing actor can temporarily veto every otherwise valid network
        trajectory.  Braking and high-rate replanning remain owned by the ROS
        loop, while this public boundary prevents the resulting wait from
        being mistaken for a static deadlock.  Entering *and* leaving the
        pause breaks candidate-sector confirmation and odometry-stagnation
        continuity, so evidence from opposite sides of an actor crossing can
        never be stitched into one recovery handoff.

        Scan escalation is deliberately retained: this method never changes
        ``unsuccessful_scan_attempts`` or ``current_scan_limit_deg``.  During
        provisional handoff validation, its remaining time and accumulated
        motion/clearance evidence are frozen.  Resume shifts the validation
        clock and the first ordinary observation rebases its origins, so
        braking, drift or a changed camera view during the actor yield cannot
        falsely prove or fail the static escape.

        Args:
            paused: ``True`` while the dynamic actor is the sole blocker;
                ``False`` immediately before normal observation resumes.
            now_s: Optional monotonic timestamp used by the returned decision.

        Returns:
            A public :class:`RecoveryDecisionV3` snapshot.  No private state
            serializer is required by runtime callers.
        """
        now_s = self._now(now_s)
        paused = bool(paused)
        was_paused = self.dynamic_yield_paused
        if paused and self.mode != self.NORMAL:
            raise RuntimeError(
                "dynamic-yield pause is only valid in normal network mode; "
                "provisional handoff validation is supported"
            )

        if paused and not was_paused:
            self.dynamic_yield_pause_started_s = now_s
            if self.handoff_validation_active:
                if (
                    self.last_position is not None
                    and self.handoff_validation_origin_position is not None
                ):
                    self.handoff_pause_displacement_vector = (
                        self.last_position
                        - self.handoff_validation_origin_position
                    )
                else:
                    self.handoff_pause_displacement_vector = None
                self.handoff_pause_forward_clearance_gain_m = (
                    self.handoff_validation_forward_clearance_gain_m
                )
        elif not paused and was_paused:
            pause_started = self.dynamic_yield_pause_started_s
            if pause_started is None or now_s < pause_started:
                raise ValueError("dynamic-yield pause clock moved backwards")
            if (
                self.handoff_validation_active
                and self.handoff_validation_started_s is not None
            ):
                self.handoff_validation_started_s += now_s - pause_started
                self.handoff_resume_rebase_pending = True
            self.dynamic_yield_pause_started_s = None

        # Apply on every paused frame as well as the first resumed frame.  A
        # caller can therefore remain in a dynamic yield for an arbitrary
        # number of replans without accumulating either zero-candidate or
        # measured-motion stagnation evidence.
        if paused or was_paused:
            self.zero_feasible_replans = 0
            self.selected_release_count = 0
            self.selected_confirmation_horizontal_sector_id = None
            self.last_selected_candidate_eligible = False
            self.last_selected_candidate_goal_progress_m = None
            self.last_selected_candidate_action_id = None
            self.last_selected_candidate_horizontal_sector_id = None
            self.last_selected_candidate_min_observed_clearance_m = None
            self.last_handoff_confirmation_replans = None
            self.last_handoff_horizontal_sector_id = None
            self._reset_motion_window()

        self.dynamic_yield_paused = paused
        transition = None
        if paused and not was_paused:
            transition = "network_dynamic_yield_pause_started"
        elif not paused and was_paused:
            transition = "network_dynamic_yield_pause_ended"
        return self._decision(now_s, transition)

    def _observe_motion_stagnation(self, now_s, position_world):
        """Update the scene-agnostic measured-motion stagnation evidence.

        This deliberately ignores map type, candidate count and network score.
        A recovery takeover is justified only when odometry shows that the
        vehicle has remained inside a small displacement ball for a complete
        time window and this observation persists across multiple replans.
        """
        if not self.config.motion_stagnation_enabled:
            self.motion_stagnation_replans = 0
            self.motion_window_displacement_m = None
            self.motion_window_duration_s = None
            return False

        point = np.asarray(position_world, dtype=np.float64).reshape(3)
        self.motion_history.append((float(now_s), point.copy()))
        cutoff = float(now_s) - self.config.motion_stagnation_window_s
        # Retain the newest sample at or before the cutoff so the measured
        # duration never becomes materially shorter than the stated window.
        while (
            len(self.motion_history) >= 2
            and self.motion_history[1][0] <= cutoff
        ):
            self.motion_history.popleft()
        oldest_time, oldest_position = self.motion_history[0]
        duration = float(now_s) - oldest_time
        self.motion_window_duration_s = duration
        if duration < self.config.motion_stagnation_window_s:
            self.motion_window_displacement_m = None
            self.motion_stagnation_replans = 0
            return False

        displacement = float(np.linalg.norm(point - oldest_position))
        self.motion_window_displacement_m = displacement
        if displacement < self.config.motion_stagnation_min_displacement_m:
            self.motion_stagnation_replans += 1
        else:
            self.motion_stagnation_replans = 0
        return (
            self.motion_stagnation_replans
            >= self.config.motion_stagnation_trigger_replans
        )

    def _return_to_network(
        self, now_s, transition, preserve_scan_heading=False,
        successful_release=False,
    ):
        self.last_handoff_confirmation_replans = (
            self.selected_release_count if successful_release else None
        )
        self.last_handoff_horizontal_sector_id = (
            self.selected_confirmation_horizontal_sector_id
            if successful_release else None
        )
        self.mode = self.NORMAL
        self.zero_feasible_replans = 0
        self.selected_release_count = 0
        self.selected_confirmation_horizontal_sector_id = None
        self.scan_started_s = None
        self.heading_commitment_until_s = (
            now_s + self.config.heading_commitment_s
            if preserve_scan_heading else 0.0
        )
        self.cooldown_until_s = now_s + self.config.cooldown_s
        self._reset_motion_window(now_s)
        if successful_release:
            if self.config.defer_scan_reset_until_post_release_confirmation:
                # A candidate visible for the configured scan-confirmation
                # interval is still only a provisional escape.  Count it as
                # the next escalation level
                # until normal flight sustains the same useful evidence.  This
                # prevents repeated 60-degree scans when the candidate vanishes
                # immediately after the camera starts moving again.
                self.unsuccessful_scan_attempts += 1
                self.current_scan_limit_deg = min(
                    self.config.max_escalated_scan_angle_deg,
                    self.config.max_scan_angle_deg
                    + self.unsuccessful_scan_attempts
                    * self.config.scan_angle_step_deg,
                )
            else:
                self.unsuccessful_scan_attempts = 0
                self.current_scan_limit_deg = self.config.max_scan_angle_deg
            if self.config.provisional_handoff_validation_enabled:
                self._start_handoff_validation(now_s)
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
        selected_candidate_action_id=None,
        selected_candidate_horizontal_sector_id=None,
        selected_candidate_min_observed_clearance_m=None,
        handoff_target_world=None,
        now_s=None,
    ):
        del (
            rotation_world_from_body,
            goal_world,
            goal_feasible_candidate_count,
            recovery_feasible_candidate_count,
        )
        now_s = self._now(now_s)
        if self.dynamic_yield_paused:
            raise RuntimeError(
                "resume dynamic-yield pause before calling observe()"
            )
        self.last_handoff_confirmation_replans = None
        self.last_handoff_horizontal_sector_id = None
        self.record_position(position_world)
        self.last_forward_clearance_m = self.forward_clearance_from_depth(depth)
        if not self.config.enabled:
            return self._decision(now_s)

        # A scan handoff is provisional in V4.8.5.  Normal network translation
        # remains authoritative during this bounded validation interval.  A
        # successful measured escape closes recovery; failure immediately
        # resumes the wider scan chain without waiting for a new stagnation
        # window or resetting to 60 degrees.
        handoff_validation_was_active = self.handoff_validation_active
        handoff_transition = self._observe_handoff_validation(
            now_s, position_world, handoff_target_world
        )
        if handoff_transition is not None:
            return self._decision(now_s, handoff_transition)
        if handoff_validation_was_active:
            return self._decision(now_s)

        feasible_candidate_count = int(feasible_candidate_count)
        collision_floor_present = bool(collision_floor_present)
        speed_mps = float(speed_mps)
        progress = selected_candidate_goal_progress_m
        action_id = (
            None if selected_candidate_action_id is None
            else int(selected_candidate_action_id)
        )
        sector_id = (
            None if selected_candidate_horizontal_sector_id is None
            else int(selected_candidate_horizontal_sector_id)
        )
        if action_id is not None and action_id < 0:
            raise ValueError("selected candidate action id must be non-negative")
        if sector_id is not None and sector_id < 0:
            raise ValueError(
                "selected candidate horizontal sector must be non-negative"
            )
        selected_clearance = selected_candidate_min_observed_clearance_m
        if selected_clearance is not None:
            selected_clearance = float(selected_clearance)
            if not math.isfinite(selected_clearance):
                raise ValueError("selected candidate clearance must be finite")
        selected_eligible = bool(
            selected_candidate_feasible
            and not collision_floor_present
            and progress is not None
            and math.isfinite(float(progress))
            and float(progress) >= self.config.selected_min_goal_progress_m
            and (
                not self.config.require_consistent_selected_horizontal_sector
                or self.mode == self.NORMAL
                or sector_id is not None
            )
        )
        self.last_selected_candidate_eligible = selected_eligible
        self.last_selected_candidate_goal_progress_m = (
            None if progress is None else float(progress)
        )
        self.last_selected_candidate_action_id = action_id
        self.last_selected_candidate_horizontal_sector_id = sector_id
        self.last_selected_candidate_min_observed_clearance_m = (
            selected_clearance
        )
        if not selected_eligible:
            self.selected_release_count = 0
            if self.mode != self.NORMAL:
                self.selected_confirmation_horizontal_sector_id = None
        elif (
            self.config.require_consistent_selected_horizontal_sector
            and self.mode != self.NORMAL
        ):
            if self.selected_confirmation_horizontal_sector_id == sector_id:
                self.selected_release_count += 1
            else:
                # A vertical-row change within one horizontal column is
                # allowed.  Moving to another horizontal opening starts a new
                # confirmation streak instead of stitching unrelated flickers
                # into one recovery handoff.
                self.selected_confirmation_horizontal_sector_id = sector_id
                self.selected_release_count = 1
        else:
            self.selected_release_count += 1

        if self.mode == self.NORMAL:
            motion_stagnation_triggered = self._observe_motion_stagnation(
                now_s, position_world
            )
            if (
                selected_eligible
                and self.selected_release_count
                >= self.config.post_release_confirmation_replans
            ):
                # Only sustained useful network motion closes the escalation
                # chain.  This check intentionally also runs during cooldown.
                self.unsuccessful_scan_attempts = 0
                self.current_scan_limit_deg = self.config.max_scan_angle_deg
            # Cooldown suppresses only another recovery takeover.  It never
            # blocks a runtime-safe network candidate from being executed.
            if now_s < self.cooldown_until_s:
                if self.config.accumulate_zero_during_cooldown:
                    self.zero_feasible_replans = (
                        self.zero_feasible_replans + 1
                        if feasible_candidate_count == 0 else 0
                    )
                else:
                    self.zero_feasible_replans = 0
                return self._decision(now_s)
            self.zero_feasible_replans = (
                self.zero_feasible_replans + 1
                if feasible_candidate_count == 0 else 0
            )
            if (
                self.zero_feasible_replans
                >= self.config.zero_feasible_trigger_replans
                or motion_stagnation_triggered
            ):
                self.last_recovery_trigger_reason = (
                    "zero_runtime_safe_candidates"
                    if self.zero_feasible_replans
                    >= self.config.zero_feasible_trigger_replans else
                    "observed_motion_stagnation"
                )
                self.mode = self.BRAKING
                self.recovery_origin_forward_clearance_m = (
                    self.last_forward_clearance_m
                )
                self.selected_release_count = 0
                self.selected_confirmation_horizontal_sector_id = None
                # Stop accumulating normal-flight samples, but retain the
                # triggering count/displacement in braking telemetry.
                self.motion_history.clear()
                return self._decision(
                    now_s,
                    "network_stagnation_to_braking"
                    if motion_stagnation_triggered else
                    "network_to_braking",
                )
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
                if (
                    self.resume_failed_handoff_scan
                    and self.handoff_preserved_scan_offset_rad is not None
                    and self.handoff_preserved_scan_direction is not None
                ):
                    self.scan_direction = float(
                        self.handoff_preserved_scan_direction
                    )
                    self.scan_offset_rad = float(
                        self.handoff_preserved_scan_offset_rad
                    )
                    self.resume_failed_handoff_scan = False
                else:
                    preferred_direction = self.scan_direction_from_depth(depth)
                    self.scan_direction = (
                        preferred_direction
                        if self.unsuccessful_scan_attempts % 2 == 0
                        else -preferred_direction
                    )
                    self.scan_offset_rad = 0.0
                self.current_scan_limit_deg = min(
                    self.config.max_escalated_scan_angle_deg,
                    self.config.max_scan_angle_deg
                    + self.unsuccessful_scan_attempts
                    * self.config.scan_angle_step_deg,
                )
                self.scan_leg_complete = False
                self.scan_legs_completed = 0
                self.scan_started_s = now_s
                transition = (
                    "handoff_failure_braking_to_resumed_bounded_scan"
                    if self.handoff_validation_result
                    == "insufficient_escape_progress" else
                    "braking_to_bounded_scan"
                )
                return self._decision(now_s, transition)
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
                stable_release_ready = bool(
                    self.selected_release_count
                    >= self.config.selected_release_replans
                )
                if (
                    stable_release_ready
                    if self.config.require_consistent_selected_horizontal_sector
                    else selected_eligible
                ):
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
    "horizontal_sector_from_action_id",
]
