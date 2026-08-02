"""Bounded development-only safety occupancy from provisional measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .provisional_measurement_chain_v1 import ProvisionalMeasurementChainV1


CONTRACT_VERSION = "provisional_dynamic_safety_hypothesis_v1"


@dataclass(frozen=True)
class ProvisionalReachableOccupancyV1:
    time_offset_s: float
    center_world: np.ndarray
    radius_m: float
    velocity_source: str

    def __post_init__(self):
        center = np.asarray(self.center_world, dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("center_world must be finite [3]")
        if self.time_offset_s < 0 or self.radius_m <= 0:
            raise ValueError("occupancy time/radius invalid")
        object.__setattr__(self, "center_world", center.copy())


@dataclass(frozen=True)
class ProvisionalDynamicSafetyHypothesisV1:
    provisional_id: int
    source_observation_ids: Tuple[int, ...]
    first_timestamp: float
    last_timestamp: float
    observation_count: int
    geometry_hypothesis: str
    support_radius_m: float
    velocity_center_mps: np.ndarray
    velocity_uncertainty_mps: float
    acceleration_bound_mps2: float
    reachable_occupancy: Tuple[ProvisionalReachableOccupancyV1, ...]
    birth_state: str
    expiry_timestamp: float
    promotion_status: str
    prediction_only: bool
    runtime_gt_used: bool = False

    def __post_init__(self):
        velocity = np.asarray(self.velocity_center_mps, dtype=np.float64)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("velocity_center_mps must be finite [3]")
        if self.runtime_gt_used:
            raise ValueError("runtime GT is forbidden")
        if self.observation_count < 1 or not self.reachable_occupancy:
            raise ValueError("hypothesis requires causal measurement occupancy")
        object.__setattr__(self, "velocity_center_mps", velocity.copy())


class ProvisionalSafetyHypothesisBuilderV1:
    def __init__(
        self, maximum_speed_mps=3.0, maximum_acceleration_mps2=6.0,
        maximum_age_s=0.35, horizon_s=1.7, samples=18,
        extent_prior_radius_m=0.45,
    ):
        self.maximum_speed_mps = float(maximum_speed_mps)
        self.maximum_acceleration_mps2 = float(maximum_acceleration_mps2)
        self.maximum_age_s = float(maximum_age_s)
        self.horizon_s = float(horizon_s)
        self.samples = int(samples)
        self.extent_prior_radius_m = float(extent_prior_radius_m)
        if min(
            self.maximum_speed_mps, self.maximum_acceleration_mps2,
            self.maximum_age_s, self.horizon_s,
            self.extent_prior_radius_m,
        ) <= 0 or self.samples < 2:
            raise ValueError("invalid provisional safety bounds")

    def build(
        self, chain: ProvisionalMeasurementChainV1, query_timestamp: float,
    ):
        if not chain.observations:
            raise ValueError("cannot birth without measurement")
        age = float(query_timestamp)-chain.last_timestamp
        if age < -1e-9 or age > self.maximum_age_s:
            raise ValueError("provisional chain is outside bounded age")
        if len(chain.observations) >= 2:
            first, last = chain.observations[-2:]
            dt = last.timestamp-first.timestamp
            velocity = (
                np.zeros(3) if dt <= 0 else
                (last.position_world-first.position_world)/dt
            )
            speed = float(np.linalg.norm(velocity))
            if speed > self.maximum_speed_mps:
                velocity *= self.maximum_speed_mps/speed
            position_std = float(np.sqrt(max(
                np.linalg.eigvalsh(first.covariance_world).max(),
                np.linalg.eigvalsh(last.covariance_world).max(), 0.,
            )))
            uncertainty = min(
                self.maximum_speed_mps,
                max(0.25, 2.*position_std/max(dt, 1e-3)),
            )
            source = "two_frame_causal_velocity_set"
        else:
            velocity = np.zeros(3)
            uncertainty = self.maximum_speed_mps
            source = "single_frame_bounded_motion_prior"
        center_now = chain.last.position_world+velocity*max(age, 0.)
        support = max(
            self.extent_prior_radius_m,
            chain.last.support_radius_m,
        )
        occupancies = []
        for tau in np.linspace(0., self.horizon_s, self.samples):
            total = age+float(tau)
            occupancies.append(ProvisionalReachableOccupancyV1(
                time_offset_s=float(tau),
                center_world=center_now+velocity*float(tau),
                radius_m=(
                    support+uncertainty*total
                    + .5*self.maximum_acceleration_mps2*total*total
                ),
                velocity_source=source,
            ))
        return ProvisionalDynamicSafetyHypothesisV1(
            provisional_id=chain.provisional_id,
            source_observation_ids=chain.source_observation_ids,
            first_timestamp=chain.first_timestamp,
            last_timestamp=chain.last_timestamp,
            observation_count=len(chain.observations),
            geometry_hypothesis="bounded_sphere_support",
            support_radius_m=support,
            velocity_center_mps=velocity,
            velocity_uncertainty_mps=uncertainty,
            acceleration_bound_mps2=self.maximum_acceleration_mps2,
            reachable_occupancy=tuple(occupancies),
            birth_state=(
                "DIRECT_SINGLE_MEASUREMENT"
                if len(chain.observations) == 1 else
                "CAUSAL_MEASUREMENT_CHAIN"
            ),
            expiry_timestamp=chain.last_timestamp+self.maximum_age_s,
            promotion_status=chain.promotion_status,
            prediction_only=age > 1e-9,
            runtime_gt_used=False,
        )


__all__ = [
    "CONTRACT_VERSION", "ProvisionalReachableOccupancyV1",
    "ProvisionalDynamicSafetyHypothesisV1",
    "ProvisionalSafetyHypothesisBuilderV1",
]
