"""PEPCR1 Strategy B: bounded two-frame no-history bootstrap."""

from __future__ import annotations

from dataclasses import replace
import numpy as np

from .dynamic_measurement_outcome_v1 import DynamicMeasurementStatusV1
from .history_backed_support_policy_v1 import _active_support
from .pending_support_state_v1 import (
    PendingSupportStateV1, bounds_from_outcome,
)
from .provisional_evidence_authorizer_fast_v1 import (
    ProvisionalEvidenceAuthorizerFastV1,
)
from .provisional_outcome_mapper_v1 import ProvisionalOutcomeMapperV1


CONTRACT_VERSION = "two_stage_no_history_bootstrap_v1"


def _center(bounds):
    value = np.asarray(bounds, np.float64)
    return .5*(value[:3]+value[3:])


class TwoStageSupportBootstrapV1:
    strategy_id = "B_TWO_STAGE_NO_HISTORY_BOOTSTRAP"

    def __init__(self, pds_contract, evidence_contract, policy_contract):
        self.frozen_mapper = ProvisionalOutcomeMapperV1(pds_contract)
        self.authorizer = ProvisionalEvidenceAuthorizerFastV1(
            evidence_contract
        )
        self.lifecycle = pds_contract["lifecycle"]
        self.contract = policy_contract["bootstrap"]
        self._pending = {}
        self._used_pending = set()
        self._current_frame = None
        self.last_authorization = None
        self.last_pending = None
        self.last_promotion = None
        self.last_semantic = "NO_ACTIVE_DYNAMIC_RISK"

    @property
    def pending_states(self):
        return tuple(self._pending[key] for key in sorted(self._pending))

    def _expire(self, frame, timestamp):
        self._pending = {
            key: value for key, value in self._pending.items()
            if value.is_live(frame, timestamp)
            or frame == value.birth_frame
        }

    def _associate(self, bounds, frame, timestamp):
        maximum = float(self.contract["maximum_displacement_m"])
        pairs = []
        for key, pending in self._pending.items():
            if key in self._used_pending:
                continue
            if frame != pending.birth_frame+1 or not pending.is_live(
                frame, timestamp
            ):
                continue
            distance = float(np.linalg.norm(
                _center(bounds)-_center(pending.support_bounds)
            ))
            if distance <= maximum:
                pairs.append((distance, key, pending))
        if not pairs:
            return None
        _, key, pending = min(pairs, key=lambda row: (row[0], row[1]))
        self._used_pending.add(key)
        del self._pending[key]
        return pending

    def map(self, outcome, state_id, generation=0, causal_context=None):
        status = DynamicMeasurementStatusV1(outcome["status"])
        frame = int(outcome.get("frame_index", 0))
        timestamp = float(outcome["timestamp"])
        if frame != self._current_frame:
            self._current_frame = frame
            self._used_pending.clear()
            self._expire(frame, timestamp)
        self.last_pending = None
        self.last_promotion = None
        if status != DynamicMeasurementStatusV1.BOUNDED_SAFETY_SUPPORT:
            self.last_authorization = None
            state = self.frozen_mapper.map(
                outcome, state_id, generation, causal_context
            )
            self.last_semantic = (
                "NO_ACTIVE_DYNAMIC_RISK"
                if state is None else state.planner_semantic
            )
            return state
        decision = self.authorizer.authorize(outcome, causal_context)
        self.last_authorization = decision
        if (
            decision.authorized
            and decision.history_level == "RECENT_DYNAMIC_TRACK_COMPATIBLE"
        ):
            state = _active_support(
                outcome, state_id, generation, self.lifecycle
            )
            self.last_semantic = state.planner_semantic
            return state
        # The frozen PECR authorizer supplies quality, causal-motion and hard
        # negative decisions. This wrapper changes no threshold or family.
        eligible = bool(
            decision.authorized and decision.history_level == "NO_HISTORY"
            and not set(decision.negative_evidence).intersection({
                "INTERNAL_CONTRACT_ERROR", "HARD_INVALID_NO_EVIDENCE",
                "CAMERA_MOTION_ARTIFACT", "STATIC_BACKGROUND_MATCH",
                "ASSOCIATION_CONFLICT", "DUPLICATE_SOURCE",
            })
        )
        if not eligible:
            self.last_semantic = "PENDING_OR_UNKNOWN_SUPPORT"
            return None
        bounds = bounds_from_outcome(outcome)
        previous = self._associate(bounds, frame, timestamp)
        if previous is not None:
            state = _active_support(
                outcome, state_id, generation, self.lifecycle
            )
            state = replace(
                state,
                source_outcome_ids=(
                    *previous.source_outcome_ids,
                    int(outcome["outcome_id"]),
                ),
                evidence_frames=2,
            )
            self.last_promotion = {
                "pending_state_id": previous.state_id,
                "active_state_id": state.state_id,
                "one_to_one": True,
                "same_frame": False,
                "runtime_gt_used": False,
            }
            self.last_semantic = state.planner_semantic
            return state
        pending = PendingSupportStateV1(
            state_id=int(state_id), generation=int(generation),
            source_outcome_ids=(int(outcome["outcome_id"]),),
            source_component_ids=tuple(
                int(value)
                for value in outcome.get("source_component_ids", ())
            ),
            birth_frame=frame, birth_timestamp=timestamp,
            last_evidence_timestamp=timestamp,
            expiry_timestamp=timestamp+float(
                self.contract["maximum_age_s"]
            ),
            support_bounds=bounds,
            provenance_signature=("FINITE_WORLD_POSITION_SET",),
        )
        self._pending[pending.state_id] = pending
        self.last_pending = pending
        self.last_semantic = pending.planner_semantic
        return None


__all__ = ["CONTRACT_VERSION", "TwoStageSupportBootstrapV1"]
