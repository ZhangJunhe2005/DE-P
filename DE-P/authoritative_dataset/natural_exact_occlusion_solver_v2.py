"""Versioned short natural full-occlusion interval and phase solver.

This module deliberately does not alter the frozen v2.1/v2.2 constructors.
Broad phase works in the camera image plane; only canonical CUDA diagnostics
may establish a proof witness.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import yaml


SOLVER_VERSION = "natural_exact_occlusion_solver_v2"
CONTRACT_VERSION = "natural_short_full_occlusion_v2"


@dataclass(frozen=True)
class FullInterval:
    enter_s: float
    exit_s: float

    @property
    def duration_s(self) -> float:
        return self.exit_s - self.enter_s


def load_contract(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text())
    if value.get("version") != CONTRACT_VERSION:
        raise ValueError("unexpected short occlusion contract version")
    radii = tuple(map(float, value["actor"]["allowed_radius_m"]))
    if radii != (0.20, 0.25, 0.30) or min(radii) < 0.20:
        raise ValueError("actor radius grid violates the frozen v2 contract")
    if tuple(value["occlusion"]["accepted_exact_gap_frames"]) != (1, 2):
        raise ValueError("only exact gap-1 and gap-2 are strict")
    return value


def classify_occlusion(projected: int, visible: int, blocked: int,
                       consecutive_full_frames: int,
                       severe_threshold: float = .95) -> str:
    projected, visible, blocked = map(
        int, (projected, visible, blocked)
    )
    if projected <= 0:
        return "not_projected"
    fraction = blocked / projected
    if visible == 0 and blocked == projected:
        if consecutive_full_frames == 1:
            return "short_full_occlusion_gap1"
        if consecutive_full_frames == 2:
            return "short_full_occlusion_gap2"
        if consecutive_full_frames >= 3:
            return "long_full_occlusion"
    if 0 < fraction < severe_threshold:
        return "partial_occlusion"
    if severe_threshold <= fraction < 1 or visible > 0:
        return "severe_partial_occlusion"
    return "directly_visible"


def contiguous_intervals(times: np.ndarray, full: np.ndarray,
                         step_s: float) -> list[FullInterval]:
    times = np.asarray(times, dtype=np.float64)
    indices = np.flatnonzero(np.asarray(full, dtype=bool))
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [
        FullInterval(float(times[g[0]]),
                     float(times[g[-1]] + step_s))
        for g in groups
    ]


def _sample_count(interval: FullInterval, period_s: float,
                  phase_s: float) -> int:
    epsilon = 1e-12
    first = math.ceil(
        (interval.enter_s - phase_s - epsilon) / period_s
    )
    last = math.floor(
        (interval.exit_s - phase_s - epsilon) / period_s
    )
    return max(0, last - first + 1)


def analytic_phase_intervals(interval: FullInterval,
                             period_s: float) -> dict[int, list[dict]]:
    """Partition [0,T) at exact enter/exit modulo-T breakpoints.

    This is an analytic breakpoint solution, not a phase grid search.
    Interior validation points avoid ambiguous samples exactly on a raster
    transition.
    """
    period_s = float(period_s)
    if interval.duration_s <= 0 or period_s <= 0:
        raise ValueError("positive interval and sample period required")
    points = {0.0, period_s}
    for value in (interval.enter_s, interval.exit_s):
        points.add(float(value % period_s))
    ordered = sorted(points)
    output: dict[int, list[dict]] = {}
    for lower, upper in zip(ordered[:-1], ordered[1:]):
        if upper - lower <= 1e-12:
            continue
        midpoint = .5 * (lower + upper)
        count = _sample_count(interval, period_s, midpoint)
        width = upper - lower
        output.setdefault(count, []).append({
            "lower_phase_s": lower,
            "upper_phase_s": upper,
            "width_s": width,
            "validation_phases_s": [
                lower + .1 * width, midpoint, upper - .1 * width,
            ],
        })
    return output


def camera_to_world(camera_position, camera_yaw, body_points):
    """YOPO camera/body convention: +x forward, +y image-right, +z down."""
    points = np.asarray(body_points, dtype=np.float64)
    c, s = math.cos(float(camera_yaw)), math.sin(float(camera_yaw))
    rotation = np.asarray(((c, -s, 0), (s, c, 0), (0, 0, 1.0)))
    return np.asarray(camera_position, dtype=np.float64) + points @ rotation.T


def pixel_line_to_world(camera_position, camera_yaw, pixels,
                        forward_depth_m, intrinsics):
    fx, fy, cx, cy = map(float, intrinsics)
    pixels = np.asarray(pixels, dtype=np.float64)
    depth = float(forward_depth_m)
    body = np.column_stack((
        np.full(len(pixels), depth),
        depth * (pixels[:, 0] - cx) / fx,
        depth * (pixels[:, 1] - cy) / fy,
    ))
    return camera_to_world(camera_position, camera_yaw, body)


def strict_full_mask(diagnostics, actor_index: int = 0) -> np.ndarray:
    projected = np.asarray(
        diagnostics["per_actor_projected_pixel_count"]
    )[:, actor_index]
    visible = np.asarray(
        diagnostics["per_actor_visible_pixel_count"]
    )[:, actor_index]
    blocked = np.asarray(
        diagnostics["per_actor_static_blocked_pixel_count"]
    )[:, actor_index]
    return (projected > 0) & (visible == 0) & (blocked == projected)


def validate_strict_window(diagnostics, gap_start: int, gap_length: int,
                           pre_frames: int = 4,
                           post_frames: int = 3) -> dict:
    full = strict_full_mask(diagnostics)
    gap_end = gap_start + gap_length
    if gap_start < pre_frames or gap_end + post_frames > len(full):
        return {"status": "FAIL", "reason": "guard_window_out_of_range"}
    expected = np.zeros_like(full)
    expected[gap_start:gap_end] = True
    local = slice(gap_start-pre_frames, gap_end+post_frames)
    illegal = any(bool(np.asarray(diagnostics[key])[local, 0].any())
                  for key in (
                      "per_actor_outside_fov",
                      "per_actor_behind_camera",
                      "per_actor_beyond_max_depth",
                  ))
    visible = np.asarray(
        diagnostics["per_actor_visible_pixel_count"]
    )[:, 0]
    direct = np.r_[
        visible[gap_start-pre_frames:gap_start],
        visible[gap_end:gap_end+post_frames],
    ]
    exact = bool(np.array_equal(full[local], expected[local]))
    return {
        "status": "PASS" if exact and not illegal and np.all(direct > 0)
        else "FAIL",
        "exact_gap": exact,
        "illegal_visibility_exit": illegal,
        "pre_post_directly_visible": bool(np.all(direct > 0)),
    }
