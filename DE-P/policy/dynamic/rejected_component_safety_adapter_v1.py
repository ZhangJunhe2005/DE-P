"""Observe frozen strict rejections and derive independent safety evidence."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType

import numpy as np

from .safety_measurement_availability_v1 import (
    AvailabilityModeV1, EvidenceStateV1,
    SafetyMeasurementDecisionV1, SafetyMeasurementEvidenceV1,
    SafetyWeakMeasurementV1,
)


CONTRACT_VERSION = "rejected_component_safety_adapter_v1"


def _state(value):
    if value is None:
        return EvidenceStateV1.UNAVAILABLE
    return (
        EvidenceStateV1.AVAILABLE_TRUE if bool(value)
        else EvidenceStateV1.AVAILABLE_FALSE
    )


def strict_rejection_reasons(row, parameters):
    reasons = []
    if row["fov_fraction"] > parameters["maximum_fov_fraction"]:
        return ("component_geometry_filter:fov_boundary",)
    if row["invalid_fraction"] > parameters[
        "maximum_invalid_fraction"
    ]:
        return ("component_geometry_filter:depth_validity",)
    state = row["tracklet"]
    speed = float(np.linalg.norm(state.velocity))
    if state.support < 2:
        reasons.append("temporal_support")
    if row["pixel_count"] < parameters[
        "minimum_component_pixels_for_residual_birth"
    ]:
        reasons.append("component_size")
    if row["closer_fraction"] < parameters[
        "minimum_closer_fraction_for_birth"
    ]:
        reasons.append("closer_fraction")
    if speed > 2.5:
        reasons.append("speed_upper_bound")
    if not reasons:
        reasons.append("motion_birth_joint_gate")
    return ("measurement_confidence_filter:"+",".join(reasons),)


@dataclass
class _PreviousComponent:
    timestamp: float
    position_world: np.ndarray
    pixel_bbox: tuple
    support: int


class RejectedComponentSafetyAdapterV1:
    """Per-instance observer; the frozen `_accept` result is never changed."""

    def __init__(self, strict_filter, contract):
        self.strict_filter = strict_filter
        self.contract = contract
        self.parameters = dict(strict_filter.parameters)
        self.weak = contract["safety_weak_measurement"]
        self._original_accept = strict_filter._accept
        self._captured = {}
        self._history = []
        self._next_id = 1_000_000_000
        self._frame_index = -1
        self._timestamp = None
        self.last_decisions = ()
        adapter = self

        def observed_accept(_source, row):
            accepted = adapter._original_accept(row)
            adapter._captured[id(row)] = (row, bool(accepted))
            return accepted

        strict_filter._accept = MethodType(
            observed_accept, strict_filter
        )

    def begin_frame(self, frame_index, timestamp):
        if frame_index <= self._frame_index:
            raise ValueError("frame index must increase")
        self._captured.clear()
        self._frame_index = int(frame_index)
        self._timestamp = float(timestamp)

    @staticmethod
    def _row_fields(row):
        observation = row["observation"]
        pixels = row["pixels_vu"]
        return {
            "component_id": int(
                observation.temporary_cluster_id
            ),
            "pixel_count": int(len(pixels)),
            "centroid_world": observation.centroid_world,
            "covariance_world": observation.position_covariance,
            "extent": observation.extent,
            "pixel_bbox": observation.pixel_bbox,
            "point_count": observation.point_count,
            "depth_interval_m": (
                float(observation.bounding_box_world[:, 2].min()),
                float(observation.bounding_box_world[:, 2].max()),
            ),
            "temporal_support": int(row["tracklet"].support),
            "velocity": row["tracklet"].velocity,
            "direction_consistency":
                float(row["tracklet"].direction_consistency),
            "stable_overlap": float(row["geometric_fraction"]),
            "closer_fraction": float(row["closer_fraction"]),
            "invalid_fraction": float(row["invalid_fraction"]),
            "fov_fraction": float(row["fov_fraction"]),
            "boundary_hazard_fraction":
                float(row["boundary_hazard_fraction"]),
        }

    def _compatible_history(self, fields):
        for old in reversed(self._history[-8:]):
            if self._timestamp-old.timestamp > 0.35:
                continue
            if np.linalg.norm(
                fields["centroid_world"]-old.position_world
            ) <= self.weak["association_distance_m"]:
                return old
        return None

    def _decision(self, row):
        fields = self._row_fields(row)
        state = row["tracklet"]
        speed = float(np.linalg.norm(state.velocity))
        finite = bool(
            np.isfinite(fields["centroid_world"]).all()
            and np.isfinite(fields["covariance_world"]).all()
            and fields["point_count"]
            >= self.weak["minimum_finite_points"]
        )
        evidence = SafetyMeasurementEvidenceV1(
            finite_geometry=_state(finite),
            valid_depth_support=_state(
                fields["invalid_fraction"]
                <= self.weak["maximum_depth_invalid_fraction"]
            ),
            temporal_persistence=_state(
                fields["temporal_support"]
                >= self.weak["minimum_temporal_support"]
            ),
            stable_pixel_overlap=_state(
                fields["stable_overlap"]
                >= self.weak["minimum_stable_overlap_fraction"]
            ),
            consistent_world_motion=_state(
                self.weak["minimum_world_speed_mps"] <= speed
                <= self.weak["maximum_world_speed_mps"]
            ),
            depth_approach_trend=_state(
                fields["closer_fraction"]
                >= self.weak["minimum_approach_fraction"]
            ),
            direction_consistency=_state(
                fields["direction_consistency"]
                >= self.weak["minimum_direction_consistency"]
            ),
            boundary_safe=_state(
                fields["boundary_hazard_fraction"]
                <= self.weak["maximum_boundary_hazard_fraction"]
            ),
            fov_safe=_state(
                fields["fov_fraction"]
                <= self.weak["maximum_fov_boundary_fraction"]
            ),
            static_conflict_absent=_state(
                fields["closer_fraction"]
                >= self.weak["minimum_approach_fraction"]
                or speed >= self.weak["minimum_world_speed_mps"]
            ),
        )
        reasons = strict_rejection_reasons(row, self.parameters)
        hard = (
            not finite
            or fields["invalid_fraction"]
            > self.weak["maximum_depth_invalid_fraction"]
            or fields["fov_fraction"]
            > self.weak["maximum_fov_boundary_fraction"]
            or fields["boundary_hazard_fraction"]
            > self.weak["maximum_boundary_hazard_fraction"]
        )
        history = self._compatible_history(fields)
        temporal = (
            fields["temporal_support"]
            >= self.weak["minimum_temporal_support"]
        )
        motion = (
            self.weak["minimum_world_speed_mps"] <= speed
            <= self.weak["maximum_world_speed_mps"]
            and fields["direction_consistency"]
            >= self.weak["minimum_direction_consistency"]
        )
        approach = (
            fields["closer_fraction"]
            >= self.weak["minimum_approach_fraction"]
        )
        initializing = bool(
            fields["pixel_count"]
            >= self.weak["single_frame_minimum_pixels"]
            and fields["stable_overlap"]
            >= self.weak["minimum_stable_overlap_fraction"]
            and approach
        )
        pure_motion_birth = bool(
            fields["pixel_count"]
            >= self.weak["pure_motion_minimum_pixels"]
            and fields["temporal_support"]
            >= self.weak["pure_motion_minimum_temporal_support"]
            and motion
            and fields["direction_consistency"]
            >= self.weak[
                "pure_motion_minimum_direction_consistency"
            ]
            and fields["boundary_hazard_fraction"]
            <= self.weak[
                "pure_motion_maximum_boundary_hazard_fraction"
            ]
        )
        eligible = bool(
            not hard and (
                temporal and approach and motion
                and fields["pixel_count"]
                >= self.weak["single_frame_minimum_pixels"]
                or pure_motion_birth
                or initializing
                or history is not None and (motion or approach)
            )
        )
        if hard:
            return SafetyMeasurementDecisionV1(
                "HARD_REJECT", reasons, None, evidence
            )
        if not eligible:
            return SafetyMeasurementDecisionV1(
                "AUDIT_REQUIRED", reasons, None, evidence
            )
        if initializing and not temporal:
            mode = AvailabilityModeV1.MEASUREMENT_INITIALIZING
        elif approach:
            mode = AvailabilityModeV1.CAUSAL_APPROACH_SUPPORT
        elif history is not None:
            mode = AvailabilityModeV1.FRAGMENTED_SAFETY_SUPPORT
        else:
            mode = AvailabilityModeV1.CAUSAL_MOTION_SUPPORT
        extent = np.asarray(fields["extent"], np.float64)
        support = max(
            .05, .5*float(np.linalg.norm(extent))
            + self.weak["support_uncertainty_m"]
        )
        covariance = (
            np.asarray(fields["covariance_world"], np.float64)
            + np.eye(3)*self.weak["support_uncertainty_m"]**2
        )
        measurement = SafetyWeakMeasurementV1(
            safety_measurement_id=self._next_id,
            frame_index=self._frame_index,
            timestamp=self._timestamp,
            source_component_ids=(fields["component_id"],),
            source_component_provenance=(
                f"runtime_component:{self._frame_index}:"
                f"{fields['component_id']}"
            ),
            strict_rejection_reasons=reasons,
            availability_mode=mode,
            position_world=fields["centroid_world"],
            covariance_world=covariance,
            support_radius_m=support,
            extent_interval_m=(
                max(0., float(extent.min())/2),
                float(extent.max())/2
                + self.weak["support_uncertainty_m"],
            ),
            velocity_center_mps=np.asarray(
                state.velocity if temporal else np.zeros(3),
                np.float64,
            ),
            velocity_uncertainty_mps=(
                self.weak["maximum_speed_mps"]
                if not temporal else
                min(self.weak["maximum_speed_mps"],
                    max(.25, .5*speed))
            ),
            acceleration_bound_mps2=
                self.weak["maximum_acceleration_mps2"],
            pixel_bbox=fields["pixel_bbox"],
            point_count=fields["point_count"],
            boundary_hazard_fraction=
                fields["boundary_hazard_fraction"],
            evidence=evidence,
            expiry_timestamp=
                self._timestamp+self.weak["maximum_age_s"],
        )
        self._next_id += 1
        self._history.append(_PreviousComponent(
            self._timestamp,
            np.asarray(fields["centroid_world"]).copy(),
            fields["pixel_bbox"], fields["temporal_support"],
        ))
        return SafetyMeasurementDecisionV1(
            "CONDITIONAL_SALVAGE", reasons,
            measurement, evidence,
        )

    def finish_frame(self):
        decisions = []
        for row, accepted in self._captured.values():
            if not accepted:
                decisions.append(self._decision(row))
        self._history = [
            row for row in self._history
            if self._timestamp-row.timestamp
            <= self.weak["maximum_age_s"]
        ]
        self.last_decisions = tuple(decisions)
        return self.last_decisions


__all__ = [
    "CONTRACT_VERSION", "strict_rejection_reasons",
    "RejectedComponentSafetyAdapterV1",
]
