"""Safety Evaluator V2.1 with authoritative continuous static geometry."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML

from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from geometry_authority.static_v1 import (
    AUTHORITY_VERSION,
    CONTACT_TOLERANCE_M,
    DEFAULT_UAV_RADIUS_M,
)
from loss.safety_geometry_v2_1 import (
    CertificateState,
    certify_bezier_authority,
    combine_authority_certificates,
    planning_clearance,
    quadratic_bezier_control_points,
    quintic_bezier_control_points,
)
from policy.safety_evaluator_v2 import (
    SafetyEvaluatorV2,
    SafetyEvaluatorV2Config,
)


EVALUATOR_VERSION = "safety_evaluator_v2_1"


@dataclass(frozen=True)
class SafetyEvaluatorV2_1Config(SafetyEvaluatorV2Config):
    static_geometry_authority_version: str
    static_exact_backend: str
    static_max_recursion_depth: int

    @classmethod
    def load(cls, config_path, controller_report):
        values = YAML(typ="safe").load(Path(config_path))
        base = SafetyEvaluatorV2Config.load(config_path, controller_report)
        authority = values["static_authority"]
        config = cls(
            **base.__dict__,
            static_geometry_authority_version=str(authority["version"]),
            static_exact_backend=str(authority["exact_backend"]),
            static_max_recursion_depth=int(authority["max_recursion_depth"]),
        )
        config.validate_frozen_contract()
        return config

    def validate_frozen_contract(self):
        if self.evaluator_version != EVALUATOR_VERSION:
            raise ValueError(f"expected evaluator_version={EVALUATOR_VERSION}")
        if self.static_geometry_authority_version != AUTHORITY_VERSION:
            raise ValueError("static geometry authority version mismatch")
        if abs(self.uav_radius_m - DEFAULT_UAV_RADIUS_M) > 1e-12:
            raise ValueError("V2.1 UAV radius must be 0.3 m")
        if abs(self.contact_tolerance_m - CONTACT_TOLERANCE_M) > 1e-15:
            raise ValueError("V2.1 contact tolerance must be 1e-6 m")
        if self.static_exact_backend != ExactAuthorityBVH.version:
            raise ValueError("V2.1 exact backend mismatch")
        if self.static_max_recursion_depth < 1:
            raise ValueError("static recursion depth must be positive")


class SafetyEvaluatorV2_1(SafetyEvaluatorV2):
    """V2 dynamic/timeline semantics plus exact authoritative static truth."""

    def __init__(self, config):
        config.validate_frozen_contract()
        super().__init__(config)

    @staticmethod
    def dynamic_regression_hashes():
        """Proof that V2.1 inherits, rather than copies, dynamic behavior."""
        implementation = SafetyEvaluatorV2.evaluate_dynamic
        uncertainty = SafetyEvaluatorV2.uncertainty_margin
        return {
            "evaluate_dynamic_owner": implementation.__qualname__,
            "evaluate_dynamic_sha256": hashlib.sha256(
                inspect.getsource(implementation).encode()
            ).hexdigest(),
            "uncertainty_owner": uncertainty.__qualname__,
            "uncertainty_sha256": hashlib.sha256(
                inspect.getsource(uncertainty).encode()
            ).hexdigest(),
        }

    def exact_static_timeline(self, authority_root, current_state, v1_positions):
        """Verify latency prefix and controlled quintic over full wall time."""
        timeline = self.timeline(current_state, v1_positions)
        return self.exact_static_boundary(
            authority_root,
            current_state,
            timeline["terminal_state"],
            controlled_duration_s=timeline["controlled_duration_s"],
            timeline=timeline,
        )

    def exact_static_boundary(
        self,
        authority_root,
        current_state,
        terminal_state,
        *,
        controlled_duration_s=None,
        timeline=None,
    ):
        """Verify a formal P/V/A terminal candidate over the complete horizon."""
        current_state = np.asarray(current_state, dtype=np.float64)
        terminal_state = np.asarray(terminal_state, dtype=np.float64)
        if current_state.shape != (3, 3) or terminal_state.shape != (3, 3):
            raise ValueError("current and terminal states must be [xyz,pva]")
        controlled_duration = (
            self.config.wall_clock_horizon_s - self.config.latency_s
            if controlled_duration_s is None else float(controlled_duration_s)
        )
        first_state = current_state.copy()
        latency = self.config.latency_s
        first_state[:, 0] = (
            current_state[:, 0] + latency * current_state[:, 1]
            + 0.5 * latency**2 * current_state[:, 2]
        )
        first_state[:, 1] = current_state[:, 1] + latency * current_state[:, 2]
        backend = (
            authority_root
            if isinstance(authority_root, ExactAuthorityBVH)
            else ExactAuthorityBVH(authority_root)
        )
        prefix = quadratic_bezier_control_points(
            current_state, self.config.latency_s
        )
        controlled = quintic_bezier_control_points(
            first_state, terminal_state, controlled_duration,
        )
        certificates = (
            certify_bezier_authority(
                backend, prefix,
                uav_radius_m=self.config.uav_radius_m,
                contact_tolerance_m=self.config.contact_tolerance_m,
                max_depth=self.config.static_max_recursion_depth,
            ),
            certify_bezier_authority(
                backend, controlled,
                uav_radius_m=self.config.uav_radius_m,
                contact_tolerance_m=self.config.contact_tolerance_m,
                max_depth=self.config.static_max_recursion_depth,
            ),
        )
        state = combine_authority_certificates(certificates)
        sampled_minimum = min(
            value.minimum_sampled_gap_m for value in certificates
        )
        planning_minimum = float(planning_clearance(
            sampled_minimum,
            planning_margin=self.config.static_planning_margin_m,
            tracking_control_margin=self.config.tracking_control_margin_m,
        ))
        return {
            "state": state.value,
            "physical_min_sampled_m": sampled_minimum,
            "planning_min_sampled_m": planning_minimum,
            "unknown": state is CertificateState.UNKNOWN,
            "collision": state is CertificateState.CONFIRMED_COLLISION,
            "certified_safe": state is CertificateState.CERTIFIED_SAFE,
            "segments": certificates,
            "timeline": timeline,
            "map_authority_hash": certificates[0].map_authority_hash,
        }

    @staticmethod
    def three_state_safety_first_label(
        static_states,
        dynamic_clearance,
        secondary,
        collision_severity=None,
    ):
        """Lexicographic safe < unknown < collision ordering."""
        states = np.asarray(static_states, dtype=object)
        dynamic = np.asarray(dynamic_clearance, dtype=np.float64)
        secondary = np.asarray(secondary, dtype=np.float64)
        if states.shape != dynamic.shape or states.shape != secondary.shape:
            raise ValueError("label inputs must have identical shapes")
        if not np.isfinite(dynamic).all() or not np.isfinite(secondary).all():
            raise ValueError("label inputs must be finite")
        collision_severity = (
            np.maximum(-dynamic, 0.0)
            if collision_severity is None
            else np.asarray(collision_severity, dtype=np.float64)
        )
        if collision_severity.shape != states.shape:
            raise ValueError("collision severity shape mismatch")
        spread = max(float(np.ptp(secondary)), 1e-12)
        within_class = (secondary - float(secondary.min())) / spread
        rank = np.empty(states.shape, dtype=np.float64)
        for index, value in np.ndenumerate(states):
            normalized = value.value if isinstance(value, CertificateState) else str(value)
            dynamic_collision = dynamic[index] <= CONTACT_TOLERANCE_M
            if normalized == CertificateState.CERTIFIED_SAFE.value and not dynamic_collision:
                category = 0.0
            elif normalized == CertificateState.UNKNOWN.value and not dynamic_collision:
                category = 1.0
            else:
                category = 2.0
            rank[index] = (
                category * 1_000_000.0
                + (collision_severity[index] * 1_000.0 if category == 2.0 else 0.0)
                + within_class[index]
            )
        return rank

    def canonical_spec(self):
        return {
            "evaluator_version": EVALUATOR_VERSION,
            "config_hash": self.config.canonical_hash(),
            "static_authority": self.config.static_geometry_authority_version,
            "exact_backend": self.config.static_exact_backend,
            "uav_radius_m": self.config.uav_radius_m,
            "contact_tolerance_m": self.config.contact_tolerance_m,
            "tangent_is_collision": True,
            "oob_policy": "occupied_fail_closed",
            "dynamic_regression": self.dynamic_regression_hashes(),
        }
