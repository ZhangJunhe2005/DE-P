"""Versioned weak-measurement wrapper around the frozen provisional chain."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .provisional_measurement_chain_v1 import (
    ProvisionalMeasurementChainManagerV1,
    ProvisionalMeasurementChainV1, ProvisionalObservationV1,
)
from .safety_measurement_availability_v1 import (
    SafetyWeakMeasurementV1,
)


CONTRACT_VERSION = "measurement_availability_provisional_feed_v1"


class MeasurementAvailabilityProvisionalFeedV1:
    def __init__(self, contract):
        config = contract["safety_weak_measurement"]
        self.manager = ProvisionalMeasurementChainManagerV1(
            association_distance_m=config["association_distance_m"],
            maximum_age_s=config["maximum_age_s"],
            maximum_missed_frames=config["maximum_missed_frames"],
            maximum_observations=config["maximum_observations"],
            maximum_boundary_hazard_fraction=
                config["maximum_boundary_hazard_fraction"],
        )
        self.formal_tracker_feed_count = 0
        self.last_diagnostics = {}

    @staticmethod
    def _observation(row: SafetyWeakMeasurementV1):
        return ProvisionalObservationV1(
            observation_id=row.safety_measurement_id,
            timestamp=row.timestamp,
            position_world=row.position_world,
            covariance_world=row.covariance_world,
            support_radius_m=row.support_radius_m,
            pixel_bbox=row.pixel_bbox,
            point_count=row.point_count,
            boundary_hazard_fraction=
                row.boundary_hazard_fraction,
            direct_measurement=True,
            runtime_gt_used=False,
        )

    def update(self, weak_measurements, timestamp):
        rows = tuple(
            row for row in weak_measurements
            if row.provisional_eligible and not row.formal_eligible
        )
        chains = self._bounded_update(
            tuple(self._observation(row) for row in rows),
            float(timestamp),
        )
        self.last_diagnostics = {
            **self.manager.last_diagnostics,
            "weak_measurement_count": len(rows),
            "formal_tracker_feed_count":
                self.formal_tracker_feed_count,
            "strict_precedence": True,
            "runtime_gt_used": False,
        }
        return chains

    def _bounded_update(self, rows, timestamp):
        """Frozen v1 semantics with assignment restricted to finite edges.

        SciPy rejects a rectangular matrix containing an all-infinite row.
        Those rows mean "unmatched chain", not an invalid frame.  Replacing
        infinity by a finite sentinel for assignment and filtering it after
        assignment preserves the intended v1 one-to-one semantics.
        """
        manager = self.manager
        if (
            manager._last_timestamp is not None
            and timestamp <= manager._last_timestamp
        ):
            raise ValueError("timestamps must be strictly increasing")
        eligible = tuple(
            row for row in rows
            if row.boundary_hazard_fraction
            <= manager.maximum_boundary_hazard_fraction
        )
        chain_ids = sorted(manager._chains)
        cost = np.full((len(chain_ids), len(eligible)), np.inf)
        for i, chain_id in enumerate(chain_ids):
            chain = manager._chains[chain_id]
            predicted = chain.predicted_position(timestamp)
            for j, row in enumerate(eligible):
                distance = float(np.linalg.norm(
                    predicted-row.position_world
                ))
                if distance > manager.association_distance_m:
                    continue
                overlap = manager._bbox_iou(
                    chain.last.pixel_bbox, row.pixel_bbox
                )
                size_ratio = max(
                    chain.last.support_radius_m/row.support_radius_m,
                    row.support_radius_m/chain.last.support_radius_m,
                )
                if size_ratio <= 3.0:
                    cost[i, j] = distance+0.1*(1.-overlap)
        matches = []
        if cost.size and np.isfinite(cost).any():
            sentinel = (
                max(manager.association_distance_m+1., 1e6)
            )
            indices = linear_sum_assignment(
                np.where(np.isfinite(cost), cost, sentinel)
            )
            matches = [
                (chain_ids[i], int(j))
                for i, j in zip(*indices)
                if np.isfinite(cost[i, j])
            ]
        matched_chains = {chain for chain, _ in matches}
        matched_rows = {row for _, row in matches}
        for chain_id, index in matches:
            chain = manager._chains[chain_id]
            chain.observations.append(eligible[index])
            chain.observations[:] = chain.observations[
                -manager.maximum_observations:
            ]
            chain.missed_frames = 0
        for chain_id in set(chain_ids)-matched_chains:
            manager._chains[chain_id].missed_frames += 1
        for index, row in enumerate(eligible):
            if index in matched_rows:
                continue
            chain_id = manager._next_id
            manager._next_id += 1
            manager._chains[chain_id] = ProvisionalMeasurementChainV1(
                provisional_id=chain_id, observations=[row]
            )
        expired = []
        for chain_id, chain in tuple(manager._chains.items()):
            if (
                chain.missed_frames > manager.maximum_missed_frames
                or timestamp-chain.last_timestamp
                > manager.maximum_age_s
                or chain.promotion_status in {
                    "PROMOTED", "DUPLICATE_SUPPRESSED"
                }
            ):
                expired.append(chain_id)
                del manager._chains[chain_id]
        manager._last_timestamp = timestamp
        manager.last_diagnostics = {
            "contract_version": CONTRACT_VERSION,
            "measurement_count": len(rows),
            "eligible_measurement_count": len(eligible),
            "quality_rejected_measurement_count":
                len(rows)-len(eligible),
            "matches": tuple(matches),
            "new_chain_count": len(eligible)-len(matches),
            "expired_ids": tuple(sorted(expired)),
            "one_to_one": (
                len(matched_chains) == len(matches)
                and len(matched_rows) == len(matches)
            ),
            "finite_assignment_wrapper": True,
            "runtime_gt_used": False,
            "formal_track_created": False,
        }
        return manager.chains

    def reconcile(self, formal_tracks, timestamp):
        return self.manager.reconcile(formal_tracks, timestamp)

    def reset_generation(self):
        self.manager.reset()


__all__ = [
    "CONTRACT_VERSION",
    "MeasurementAvailabilityProvisionalFeedV1",
]
