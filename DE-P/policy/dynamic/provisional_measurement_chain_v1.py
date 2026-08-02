"""Development-only causal chains for measurements not yet safety-active.

This module is deliberately independent of ``TrackManager``.  It neither
creates nor mutates formal tracks and it never accepts semantic/GT identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


CONTRACT_VERSION = "provisional_measurement_chain_v1"
FORBIDDEN_FIELDS = frozenset({
    "gt_actor_id", "gt_center", "gt_velocity", "gt_shape", "gt_mask",
    "owner_map", "future_frame", "future_information",
})


def _vector(name, value, shape):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array.copy()


@dataclass(frozen=True)
class ProvisionalObservationV1:
    observation_id: int
    timestamp: float
    position_world: np.ndarray
    covariance_world: np.ndarray
    support_radius_m: float
    pixel_bbox: Tuple[int, int, int, int]
    point_count: int
    boundary_hazard_fraction: float = 0.0
    direct_measurement: bool = True
    runtime_gt_used: bool = False

    def __post_init__(self):
        if self.observation_id < 0:
            raise ValueError("observation_id must be non-negative")
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if not self.direct_measurement:
            raise ValueError("provisional observation requires a direct measurement")
        if self.runtime_gt_used:
            raise ValueError("runtime GT is forbidden")
        if self.support_radius_m <= 0 or self.point_count <= 0:
            raise ValueError("measurement support must be positive")
        if len(self.pixel_bbox) != 4:
            raise ValueError("pixel_bbox must have four entries")
        if not 0.0 <= self.boundary_hazard_fraction <= 1.0:
            raise ValueError("boundary_hazard_fraction must be in [0,1]")
        object.__setattr__(
            self, "position_world",
            _vector("position_world", self.position_world, (3,)),
        )
        covariance = _vector(
            "covariance_world", self.covariance_world, (3, 3)
        )
        if not np.allclose(covariance, covariance.T, atol=1e-9):
            raise ValueError("covariance_world must be symmetric")
        object.__setattr__(self, "covariance_world", covariance)
        object.__setattr__(
            self, "pixel_bbox", tuple(int(value) for value in self.pixel_bbox)
        )

    @classmethod
    def from_mapping(cls, value: Mapping):
        forbidden = FORBIDDEN_FIELDS.intersection(value)
        if forbidden:
            raise ValueError(f"forbidden runtime fields: {sorted(forbidden)}")
        return cls(**value)


@dataclass
class ProvisionalMeasurementChainV1:
    provisional_id: int
    observations: list[ProvisionalObservationV1] = field(default_factory=list)
    missed_frames: int = 0
    promotion_status: str = "UNBOUND"
    bound_track_identity: str | None = None

    @property
    def first_timestamp(self):
        return self.observations[0].timestamp

    @property
    def last_timestamp(self):
        return self.observations[-1].timestamp

    @property
    def last(self):
        return self.observations[-1]

    @property
    def source_observation_ids(self):
        return tuple(row.observation_id for row in self.observations)

    def predicted_position(self, timestamp):
        if len(self.observations) < 2:
            return self.last.position_world.copy()
        first, last = self.observations[-2:]
        dt = last.timestamp-first.timestamp
        velocity = (
            np.zeros(3) if dt <= 0 else
            (last.position_world-first.position_world)/dt
        )
        return last.position_world+velocity*max(0., timestamp-last.timestamp)


class ProvisionalMeasurementChainManagerV1:
    """Bounded one-to-one association for causal direct measurements."""

    def __init__(
        self, association_distance_m=1.0, maximum_age_s=0.35,
        maximum_missed_frames=1, maximum_observations=4,
        maximum_boundary_hazard_fraction=0.1,
    ):
        self.association_distance_m = float(association_distance_m)
        self.maximum_age_s = float(maximum_age_s)
        self.maximum_missed_frames = int(maximum_missed_frames)
        self.maximum_observations = int(maximum_observations)
        self.maximum_boundary_hazard_fraction = float(
            maximum_boundary_hazard_fraction
        )
        if (
            self.association_distance_m <= 0 or self.maximum_age_s <= 0
            or self.maximum_missed_frames < 0 or self.maximum_observations < 2
            or not 0. <= self.maximum_boundary_hazard_fraction <= 1.
        ):
            raise ValueError("invalid provisional chain bounds")
        self._chains: dict[int, ProvisionalMeasurementChainV1] = {}
        self._next_id = 0
        self._last_timestamp = None
        self.last_diagnostics = {}

    @property
    def chains(self):
        return tuple(self._chains[key] for key in sorted(self._chains))

    def reset(self):
        self._chains.clear()
        self._next_id = 0
        self._last_timestamp = None
        self.last_diagnostics = {}

    @staticmethod
    def _bbox_iou(left, right):
        lu0, lv0, lu1, lv1 = left
        ru0, rv0, ru1, rv1 = right
        width = max(0, min(lu1, ru1)-max(lu0, ru0)+1)
        height = max(0, min(lv1, rv1)-max(lv0, rv0)+1)
        intersection = width*height
        left_area = max(0, lu1-lu0+1)*max(0, lv1-lv0+1)
        right_area = max(0, ru1-ru0+1)*max(0, rv1-rv0+1)
        union = left_area+right_area-intersection
        return intersection/union if union else 0.

    def update(
        self, measurements: Iterable[ProvisionalObservationV1],
        timestamp: float,
    ):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("timestamps must be strictly increasing")
        supplied = tuple(measurements)
        if any(not isinstance(row, ProvisionalObservationV1) for row in supplied):
            raise TypeError("measurements must be ProvisionalObservationV1")
        if any(abs(row.timestamp-timestamp) > 1e-6 for row in supplied):
            raise ValueError("measurement timestamp mismatch")
        rows = tuple(
            row for row in supplied
            if row.boundary_hazard_fraction
            <= self.maximum_boundary_hazard_fraction
        )
        chain_ids = sorted(self._chains)
        cost = np.full((len(chain_ids), len(rows)), np.inf)
        for i, chain_id in enumerate(chain_ids):
            chain = self._chains[chain_id]
            predicted = chain.predicted_position(timestamp)
            for j, row in enumerate(rows):
                distance = float(np.linalg.norm(
                    predicted-row.position_world
                ))
                if distance <= self.association_distance_m:
                    overlap = self._bbox_iou(
                        chain.last.pixel_bbox, row.pixel_bbox
                    )
                    size_ratio = max(
                        chain.last.support_radius_m/row.support_radius_m,
                        row.support_radius_m/chain.last.support_radius_m,
                    )
                    if size_ratio <= 3.0:
                        cost[i, j] = distance+0.1*(1.-overlap)
        matches = []
        if cost.size:
            indices = linear_sum_assignment(cost)
            matches = [
                (chain_ids[i], int(j)) for i, j in zip(*indices)
                if np.isfinite(cost[i, j])
            ]
        matched_chains = {chain_id for chain_id, _ in matches}
        matched_rows = {index for _, index in matches}
        for chain_id, index in matches:
            chain = self._chains[chain_id]
            chain.observations.append(rows[index])
            chain.observations[:] = chain.observations[-self.maximum_observations:]
            chain.missed_frames = 0
        for chain_id in set(chain_ids)-matched_chains:
            self._chains[chain_id].missed_frames += 1
        for index, row in enumerate(rows):
            if index in matched_rows:
                continue
            chain_id = self._next_id
            self._next_id += 1
            self._chains[chain_id] = ProvisionalMeasurementChainV1(
                provisional_id=chain_id, observations=[row]
            )
        expired = []
        for chain_id, chain in tuple(self._chains.items()):
            if (
                chain.missed_frames > self.maximum_missed_frames
                or timestamp-chain.last_timestamp > self.maximum_age_s
                or chain.promotion_status in {
                    "PROMOTED", "DUPLICATE_SUPPRESSED"
                }
            ):
                expired.append(chain_id)
                del self._chains[chain_id]
        self._last_timestamp = timestamp
        self.last_diagnostics = {
            "contract_version": CONTRACT_VERSION,
            "measurement_count": len(supplied),
            "eligible_measurement_count": len(rows),
            "quality_rejected_measurement_count": len(supplied)-len(rows),
            "matches": tuple(matches),
            "new_chain_count": len(rows)-len(matches),
            "expired_ids": tuple(sorted(expired)),
            "one_to_one": (
                len(matched_chains) == len(matches)
                and len(matched_rows) == len(matches)
            ),
            "runtime_gt_used": False,
            "formal_track_created": False,
        }
        return self.chains

    def reconcile(self, formal_tracks: Sequence[Mapping], timestamp: float):
        """Bind/suppress matching chains without modifying formal tracks."""
        results = []
        used_tracks = set()
        for chain in self.chains:
            compatible = []
            for track in formal_tracks:
                identity = str(track["identity"])
                if identity in used_tracks:
                    continue
                provenance = set(track.get("observation_ids", ()))
                provenance_overlap = bool(
                    provenance.intersection(chain.source_observation_ids)
                )
                position = _vector(
                    "formal position", track["position_world"], (3,)
                )
                distance = float(np.linalg.norm(
                    position-chain.predicted_position(timestamp)
                ))
                if provenance_overlap or distance <= self.association_distance_m:
                    compatible.append((
                        0 if provenance_overlap else 1, distance,
                        identity, track,
                    ))
            if not compatible:
                results.append({
                    "provisional_id": chain.provisional_id,
                    "status": "INDEPENDENT", "formal_identity": None,
                })
                continue
            compatible.sort(key=lambda value: value[:3])
            _, _, identity, track = compatible[0]
            used_tracks.add(identity)
            provenance = set(track.get("observation_ids", ()))
            status = (
                "PROMOTED"
                if provenance.intersection(chain.source_observation_ids)
                else "DUPLICATE_SUPPRESSED"
            )
            chain.promotion_status = status
            chain.bound_track_identity = identity
            results.append({
                "provisional_id": chain.provisional_id,
                "status": status, "formal_identity": identity,
            })
        return tuple(results)


__all__ = [
    "CONTRACT_VERSION", "FORBIDDEN_FIELDS",
    "ProvisionalObservationV1", "ProvisionalMeasurementChainV1",
    "ProvisionalMeasurementChainManagerV1",
]
