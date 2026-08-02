"""Read-only safety bridge for a directly observed unconfirmed formal track."""

from __future__ import annotations

import numpy as np

from .provisional_dynamic_safety_state_v1 import (
    ProvisionalDynamicSafetyStateV1, ProvisionalSafetyStateTypeV1,
)


CONTRACT_VERSION = "formal_unconfirmed_safety_bridge_v1"


class FormalUnconfirmedSafetyBridgeV1:
    def __init__(
        self, maximum_age_s=.08, maximum_missed_frames=2,
        reference_uncertainty_m=.25, maximum_speed_mps=3.,
        maximum_acceleration_mps2=6.,
    ):
        self.maximum_age_s = float(maximum_age_s)
        self.maximum_missed_frames = int(maximum_missed_frames)
        self.reference_uncertainty_m = float(reference_uncertainty_m)
        self.maximum_speed_mps = float(maximum_speed_mps)
        self.maximum_acceleration_mps2 = float(
            maximum_acceleration_mps2
        )

    def build(
        self, state_id, formal_track, direct_measurement,
        timestamp, generation=0,
    ):
        if bool(formal_track["confirmed"]):
            raise ValueError("bridge accepts only unconfirmed tracks")
        if direct_measurement is None:
            raise ValueError("recent direct measurement is required")
        position = np.asarray(
            formal_track["position_world"], dtype=np.float64
        )
        velocity = np.asarray(
            formal_track["velocity_world"], dtype=np.float64
        )
        covariance = np.eye(3)*self.reference_uncertainty_m**2
        observation_id = int(direct_measurement["observation_id"])
        return ProvisionalDynamicSafetyStateV1(
            state_id=int(state_id), generation=int(generation),
            state_type=(
                ProvisionalSafetyStateTypeV1
                .FORMAL_UNCONFIRMED_SAFETY_STATE
            ),
            source_outcome_ids=(observation_id,),
            source_component_ids=(),
            source_observation_ids=(observation_id,),
            birth_timestamp=float(timestamp),
            last_evidence_timestamp=float(timestamp),
            expiry_timestamp=float(timestamp)+self.maximum_age_s,
            missed_frames=0,
            maximum_missed_frames=self.maximum_missed_frames,
            evidence_frames=1,
            position_reference="FORMAL_UNCONFIRMED_REFERENCE_WORLD",
            position_world=position,
            covariance_world=covariance,
            velocity_center_mps=velocity,
            velocity_uncertainty_mps=self.maximum_speed_mps,
            acceleration_bound_mps2=self.maximum_acceleration_mps2,
            formal_track_id=int(formal_track["track_id"]),
            formal_generation=str(formal_track["generation"]),
            confirmed=False,
            promotion_status="PROMOTION_PENDING",
            planner_semantic="ACTIVE_PROVISIONAL_RISK",
        )


__all__ = ["CONTRACT_VERSION", "FormalUnconfirmedSafetyBridgeV1"]
