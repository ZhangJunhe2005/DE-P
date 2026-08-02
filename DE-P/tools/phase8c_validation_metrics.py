"""Validation risk aggregation independent of scalar total loss."""

from __future__ import annotations

import math
import numpy as np
from scipy.stats import kendalltau, spearmanr
import torch


def scalar(value):
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


class ValidationMetrics:
    def __init__(self, cvar_fraction):
        self.cvar_fraction = float(cvar_fraction)
        self.dynamic_rows = []
        self.score_rows = []
        self.clearance_rows = []
        self.no_target_costs = []
        self.static_costs = []
        self.guidance_costs = []
        self.smooth_costs = []
        self.score_losses = []
        self.total_losses = []

    def add(self, kind, details):
        total = scalar(details["trajectory_loss"]) + scalar(details["score_loss"])
        self.total_losses.append(total)
        self.score_losses.append(scalar(details["score_loss"]))
        self.guidance_costs.append(scalar(details["guidance_loss"]))
        self.smooth_costs.append(scalar(details["smooth_loss"]))
        if kind == "static":
            self.static_costs.append(scalar(details["static_safety_loss"]))
            return
        raw = details["candidate_dynamic_cost_raw"].reshape(-1, 15).detach().cpu().numpy()
        score = details["predicted_score"].reshape(-1, 15).detach().cpu().numpy()
        diagnostics = details["dynamic_diagnostics"]
        clearance = diagnostics.candidate_min_clearance.detach().cpu().numpy()
        target = diagnostics.dynamic_obstacle_count.detach().cpu().numpy() > 0
        self.no_target_costs.extend(raw[~target].reshape(-1).tolist())
        if bool(target.any()):
            self.dynamic_rows.extend(raw[target])
            self.score_rows.extend(score[target])
            self.clearance_rows.extend(clearance[target])

    def finalize(self):
        if not self.dynamic_rows:
            raise ValueError("validation contains no target-bearing dynamic windows")
        raw = np.asarray(self.dynamic_rows, dtype=float)
        score = np.asarray(self.score_rows, dtype=float)
        clearance = np.asarray(self.clearance_rows, dtype=float)
        selected = score.argmin(axis=1)
        rows = np.arange(len(raw))
        selected_clearance = clearance[rows, selected]
        finite_selected = selected_clearance[np.isfinite(selected_clearance)]
        top_count = max(1, int(math.ceil(raw.shape[1] * self.cvar_fraction)))
        cvar = np.sort(raw, axis=1)[:, -top_count:].mean(axis=1)
        spearman, kendall = [], []
        for predicted, risk in zip(score, raw):
            spearman.append(float(np.nan_to_num(spearmanr(predicted, risk).statistic)))
            kendall.append(float(np.nan_to_num(kendalltau(predicted, risk).statistic)))
        return {
            "total_loss": float(np.mean(self.total_losses)),
            "raw_dynamic_mean": float(raw.mean()),
            "dynamic_cvar": float(cvar.mean()),
            "top1_collision_fraction": float((selected_clearance < 0).mean()),
            "selected_min_clearance": (
                float(finite_selected.min()) if finite_selected.size else -float("inf")
            ),
            "selected_clearance_mean": float(finite_selected.mean()),
            "safe_candidate_fraction": float((clearance >= 0).mean()),
            "oracle_regret": float((raw[rows, selected] - raw.min(axis=1)).mean()),
            "spearman": float(np.mean(spearman)),
            "kendall": float(np.mean(kendall)),
            "no_target_dynamic_mean": (
                float(np.mean(self.no_target_costs)) if self.no_target_costs else 0.0
            ),
            "no_target_dynamic_max_abs": (
                float(np.max(np.abs(self.no_target_costs))) if self.no_target_costs else 0.0
            ),
            "static_cost": (
                float(np.mean(self.static_costs)) if self.static_costs else 0.0
            ),
            "guidance": float(np.mean(self.guidance_costs)),
            "smoothness": float(np.mean(self.smooth_costs)),
            "score_loss": float(np.mean(self.score_losses)),
            "static_metrics_finite": bool(
                not self.static_costs or np.isfinite(self.static_costs).all()
            ),
            "dynamic_window_count": int(len(raw)),
            "no_target_candidate_count": int(len(self.no_target_costs)),
        }


@torch.inference_mode()
def _evaluate_dynamic(trainer, loader, kind, max_batches=None):
    trainer.policy.eval()
    accumulator = ValidationMetrics(trainer.risk_metrics_config.cvar_fraction)
    batch_count = 0
    for batch in loader:
        accumulator.add(kind, trainer.compute_batch(kind, batch))
        batch_count += 1
        if max_batches is not None and batch_count >= int(max_batches):
            break
    metrics = accumulator.finalize()
    metrics["validation_batches"] = batch_count
    return metrics


@torch.inference_mode()
def _evaluate_static(trainer, loader, max_batches=None):
    trainer.policy.eval()
    values = {name: [] for name in (
        "static_cost", "guidance", "smoothness", "score_loss",
        "selected_clearance", "score_label_mae", "selected_guidance_ratio",
    )}
    dynamic_max_abs = 0.0
    selected_safe = 0
    selected_total = 0
    for batch_index, batch in enumerate(loader):
        details = trainer.compute_batch("static", batch)
        batch_size = len(batch[0])
        score = details["predicted_score"].reshape(batch_size, 15)
        label = details["score_label"].reshape(batch_size, 15)
        selected = score.argmin(dim=1)
        rows = torch.arange(batch_size, device=score.device)
        guidance = details["candidate_guidance_cost"].reshape(batch_size, 15)
        values["static_cost"].append(scalar(details["static_safety_loss"]))
        values["guidance"].append(scalar(details["guidance_loss"]))
        values["smoothness"].append(scalar(details["smooth_loss"]))
        values["score_loss"].append(scalar(details["score_loss"]))
        values["score_label_mae"].append(scalar((score - label).abs().mean()))
        selected_guidance = guidance[rows, selected]
        median_guidance = guidance.median(dim=1).values.clamp_min(1e-8)
        values["selected_guidance_ratio"].extend(
            (selected_guidance / median_guidance).detach().cpu().tolist()
        )
        start = details["start_state_world_expanded"].permute(0, 2, 1)
        end = details["end_state_world_expanded"].permute(0, 2, 1)
        sampler = trainer.static_dep_loss.safety_loss.trajectory_sampler
        positions, _ = sampler(start, end)
        positions = positions.reshape(batch_size, 15 * positions.shape[1], 3)
        _, distance = trainer.static_dep_loss.safety_loss.get_distance_cost(
            positions, details["batch_map_id"]
        )
        clearance = distance.reshape(batch_size, 15, -1).amin(dim=2)
        selected_clearance = clearance[rows, selected]
        values["selected_clearance"].extend(
            selected_clearance.detach().cpu().tolist()
        )
        selected_safe += int((selected_clearance >= 0).sum())
        selected_total += batch_size
        dynamic_max_abs = max(
            dynamic_max_abs, abs(scalar(details["dynamic_safety_loss"]))
        )
        if max_batches is not None and batch_index + 1 >= int(max_batches):
            break
    flattened = [value for group in values.values() for value in group]
    finite = bool(flattened and np.isfinite(np.asarray(flattened)).all())
    safe_fraction = selected_safe / max(selected_total, 1)
    detour_ratio = float(np.mean(values["selected_guidance_ratio"]))
    result = {
        "static_cost": float(np.mean(values["static_cost"])),
        "guidance": float(np.mean(values["guidance"])),
        "smoothness": float(np.mean(values["smoothness"])),
        "score_loss": float(np.mean(values["score_loss"])),
        "selected_min_clearance": float(np.min(values["selected_clearance"])),
        "selected_clearance_mean": float(np.mean(values["selected_clearance"])),
        "offline_safe_selected_fraction": safe_fraction,
        "static_score_label_mae": float(np.mean(values["score_label_mae"])),
        "no_dynamic_context_max_abs": dynamic_max_abs,
        "selected_guidance_to_median_ratio": detour_ratio,
        "static_metrics_finite": finite,
        "static_smoke_pass": bool(
            finite and dynamic_max_abs == 0.0 and safe_fraction > 0.0
            and detour_ratio < 2.0
        ),
        "validation_batches": min(len(loader), int(max_batches))
        if max_batches is not None else len(loader),
    }
    return result


@torch.inference_mode()
def evaluate_validation_suites(trainer, max_batches=None):
    suites = trainer.validation_suites
    required = {"valid_gt", "valid_estimated", "valid_static"}
    missing = required - set(suites)
    if missing:
        raise ValueError(f"fixed validation suites missing: {sorted(missing)}")
    gt = _evaluate_dynamic(trainer, suites["valid_gt"], "dynamic_gt", max_batches)
    estimated = _evaluate_dynamic(
        trainer, suites["valid_estimated"], "dynamic_estimated", max_batches
    )
    static = _evaluate_static(trainer, suites["valid_static"], max_batches)
    result = dict(estimated)
    result.update({
        "static_cost": static["static_cost"],
        "static_score_loss": static["score_loss"],
        "static_selected_min_clearance": static["selected_min_clearance"],
        "static_metrics_finite": static["static_metrics_finite"],
        "static_smoke_pass": static["static_smoke_pass"],
        "estimated_gt_dynamic_cvar_gap": estimated["dynamic_cvar"] - gt["dynamic_cvar"],
        "estimated_gt_top1_collision_gap": (
            estimated["top1_collision_fraction"] - gt["top1_collision_fraction"]
        ),
        "validation_suite_version": "phase8d_fixed_v1",
        "checkpoint_suite": "valid_estimated",
        "suites": {"valid_estimated": estimated, "valid_gt": gt, "valid_static": static},
    })
    prefixes = {
        "valid_estimated": "ValidEstimated",
        "valid_gt": "ValidGT",
        "valid_static": "ValidStatic",
    }
    for suite_name, metrics in result["suites"].items():
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                trainer.tensorboard_log.add_scalar(
                    f"{prefixes[suite_name]}/{name}", value, trainer.global_step
                )
    return result


def evaluate_validation(trainer, max_batches=None):
    """Compatibility name now returning the fixed Phase 8D suite result."""
    return evaluate_validation_suites(trainer, max_batches)
