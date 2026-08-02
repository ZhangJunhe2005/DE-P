"""Constraint-aware checkpoint ordering for Phase 8C managed training."""

from __future__ import annotations

from dataclasses import dataclass
import math


REQUIRED_METRICS = {
    "top1_collision_fraction", "dynamic_cvar", "selected_min_clearance",
    "raw_dynamic_mean", "oracle_regret", "static_cost", "score_loss",
    "no_target_dynamic_mean", "static_smoke_pass", "static_metrics_finite",
    "estimated_gt_dynamic_cvar_gap", "estimated_gt_top1_collision_gap",
}


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


@dataclass
class CheckpointSelector:
    config: dict
    baseline: dict
    best_dynamic_key: tuple | None = None
    best_balanced_score: float | None = None

    def _validate(self, metrics):
        missing = REQUIRED_METRICS - set(metrics)
        if missing:
            raise ValueError(f"checkpoint metrics missing {sorted(missing)}")
        suites = metrics.get("suites")
        if not isinstance(suites, dict) or set(suites) != {
                "valid_estimated", "valid_gt", "valid_static"}:
            raise ValueError("checkpoint selection requires all fixed validation suites")
        if metrics.get("checkpoint_suite") != "valid_estimated":
            raise ValueError("checkpoint selection suite must be valid_estimated")
        for name in REQUIRED_METRICS - {"static_smoke_pass", "static_metrics_finite"}:
            if not _finite(metrics[name]):
                raise ValueError(f"checkpoint metric {name} is not finite")
        for name in ("static_smoke_pass", "static_metrics_finite"):
            if not isinstance(metrics[name], bool):
                raise ValueError(f"{name} must be boolean")

    @staticmethod
    def dynamic_key(metrics):
        """Lexicographic order mandated by the Phase 8C prompt."""
        return (
            float(metrics["top1_collision_fraction"]),
            float(metrics["dynamic_cvar"]),
            -float(metrics["selected_min_clearance"]),
            float(metrics["raw_dynamic_mean"]),
        )

    def balanced_constraints(self, metrics):
        hard = self.config["balanced_hard_constraints"]
        static_limit = float(self.baseline["static_cost"]) * float(
            hard["static_hard_ratio"]
        )
        checks = {
            "top1_collision": (
                metrics["top1_collision_fraction"] <= hard["top1_collision_max"]
            ),
            "no_target_dynamic_exact_zero": (
                abs(metrics["no_target_dynamic_mean"])
                <= hard["no_target_dynamic_tolerance"]
            ),
            "static_regression": metrics["static_cost"] <= static_limit,
            "static_smoke": metrics["static_smoke_pass"],
            "static_metrics_finite": metrics["static_metrics_finite"],
            "estimated_gt_dynamic_cvar_gap": (
                abs(metrics["estimated_gt_dynamic_cvar_gap"])
                <= hard["estimated_gt_dynamic_cvar_gap_max"]
            ),
            "estimated_gt_collision_gap": (
                abs(metrics["estimated_gt_top1_collision_gap"])
                <= hard["estimated_gt_top1_collision_gap_max"]
            ),
        }
        return checks

    def balanced_score(self, metrics):
        weights = self.config["balanced_weights"]
        scales = self.config["balanced_scales"]
        return (
            weights["dynamic_cvar"] * metrics["dynamic_cvar"] / scales["dynamic_cvar"]
            - weights["selected_clearance"] * metrics["selected_min_clearance"]
            / scales["selected_clearance"]
            + weights["oracle_regret"] * metrics["oracle_regret"]
            / scales["oracle_regret"]
            + weights["static_cost"] * metrics["static_cost"] / self.baseline["static_cost"]
            + weights["score_loss"] * metrics["score_loss"] / self.baseline["score_loss"]
        )

    def consider(self, metrics):
        self._validate(metrics)
        dynamic_key = self.dynamic_key(metrics)
        dynamic_improved = self.best_dynamic_key is None or dynamic_key < self.best_dynamic_key
        if dynamic_improved:
            self.best_dynamic_key = dynamic_key
        constraints = self.balanced_constraints(metrics)
        feasible = all(constraints.values())
        balanced_score = self.balanced_score(metrics) if feasible else None
        balanced_improved = bool(
            feasible and (
                self.best_balanced_score is None
                or balanced_score < self.best_balanced_score
            )
        )
        if balanced_improved:
            self.best_balanced_score = balanced_score
        warning_ratio = self.config["balanced_hard_constraints"]["static_warning_ratio"]
        return {
            "dynamic_improved": dynamic_improved,
            "dynamic_key": list(dynamic_key),
            "balanced_feasible": feasible,
            "balanced_constraints": constraints,
            "balanced_score": balanced_score,
            "balanced_improved": balanced_improved,
            "static_warning": (
                metrics["static_cost"] > self.baseline["static_cost"] * warning_ratio
            ),
        }

    def state_dict(self):
        return {
            "best_dynamic_key": (
                list(self.best_dynamic_key) if self.best_dynamic_key is not None else None
            ),
            "best_balanced_score": self.best_balanced_score,
            "rules": self.config,
            "baseline": self.baseline,
        }

    def load_state_dict(self, state):
        key = state.get("best_dynamic_key")
        self.best_dynamic_key = tuple(key) if key is not None else None
        self.best_balanced_score = state.get("best_balanced_score")
