"""Development-only broad-phase natural foreground observability certificate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label
import yaml


OBSERVABILITY_VERSION = "natural_foreground_observability_v1"


def _canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _largest_component(mask) -> int:
    labels, count = label(np.asarray(mask, dtype=bool))
    if not count:
        return 0
    return int(np.bincount(labels.ravel())[1:].max(initial=0))


def load_observability_config(path=None):
    path = (
        Path(path) if path is not None else
        Path(__file__).resolve().parents[1]
        / "configs/natural_foreground_observability_v1.yaml"
    )
    value = yaml.safe_load(path.read_text())
    if value["version"] != OBSERVABILITY_VERSION:
        raise RuntimeError("foreground observability version mismatch")
    value["config_hash"] = _canonical_hash(value)
    return value


def certify_observability(
    composed_depth,
    static_depth,
    actor_near_depth,
    actor_visible_mask,
    actor_positions,
    camera_positions,
    camera_yaws,
    timestamps,
    gap_start,
    gap_end,
    config=None,
):
    """Score a proposal without creating a runtime detection or track."""
    config = config or load_observability_config()
    threshold = config["thresholds"]
    composed = np.asarray(composed_depth, dtype=np.float32)
    static = np.asarray(static_depth, dtype=np.float32)
    near = np.asarray(actor_near_depth, dtype=np.float32)
    visible = np.asarray(actor_visible_mask, dtype=bool)
    actor = np.asarray(actor_positions, dtype=np.float64)
    camera = np.asarray(camera_positions, dtype=np.float64)
    yaws = np.asarray(camera_yaws, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    if composed.shape != static.shape or visible.shape != composed.shape:
        raise ValueError("depth and actor masks must share [T,H,W]")
    if near.shape != composed.shape:
        raise ValueError("actor_near_depth must be [T,H,W]")
    history_frames = int(threshold["repeated_ray_history_frames"])
    rows = []
    for index in range(len(times)):
        mask = visible[index]
        projected = np.isfinite(near[index])
        contrast_values = (static[index] - near[index])[mask]
        contrast_values = contrast_values[np.isfinite(contrast_values)]
        previous_union = np.zeros_like(mask)
        for prior in range(max(0, index-history_frames), index):
            previous_union |= visible[prior]
        newly_occupied = mask & ~previous_union
        newly_freed = (
            visible[index-1] & ~mask if index else np.zeros_like(mask)
        )
        contrast_seed = (
            mask
            & np.isfinite(static[index])
            & ((static[index] - near[index])
               >= float(threshold["minimum_background_depth_contrast_m"]))
        )
        potential = newly_occupied & contrast_seed
        position = actor[index] - camera[index]
        distance = float(np.linalg.norm(position))
        direction = position / max(distance, 1e-9)
        velocity = (
            (actor[index] - actor[index-1]) / (times[index] - times[index-1])
            if index else
            (actor[1] - actor[0]) / (times[1] - times[0])
        )
        radial = float(velocity @ direction)
        tangential = float(np.linalg.norm(velocity - radial * direction))
        vv, uu = np.nonzero(mask)
        border = (
            min(
                int(uu.min()), int(vv.min()),
                composed.shape[2] - 1 - int(uu.max()),
                composed.shape[1] - 1 - int(vv.max()),
            ) if len(uu) else -1
        )
        camera_translation = (
            float(np.linalg.norm(camera[index] - camera[index-1]))
            if index else 0.0
        )
        yaw_delta = (
            float(abs(np.arctan2(
                np.sin(yaws[index] - yaws[index-1]),
                np.cos(yaws[index] - yaws[index-1]),
            ))) if index else 0.0
        )
        reasons = []
        tests = (
            (int(projected.sum()) >= threshold["minimum_projected_pixels"],
             "projected_support"),
            (int(mask.sum()) >= threshold["minimum_visible_pixels"],
             "visible_support"),
            (
                float(mask.sum() / max(projected.sum(), 1))
                >= threshold["minimum_visible_fraction"],
                "visible_fraction",
            ),
            (
                len(contrast_values) > 0
                and float(np.median(contrast_values))
                >= threshold["minimum_background_depth_contrast_m"],
                "background_depth_contrast",
            ),
            (int(potential.sum()) >= threshold["minimum_expected_seed_pixels"],
             "newly_occupied_seed_potential"),
            (_largest_component(potential)
             >= threshold["minimum_expected_component_pixels"],
             "component_support"),
            (border >= threshold["minimum_border_margin_pixels"], "fov_margin"),
            (camera_translation
             <= threshold["maximum_camera_translation_per_frame_m"],
             "camera_translation"),
            (yaw_delta <= threshold["maximum_camera_yaw_per_frame_rad"],
             "camera_yaw"),
        )
        reasons.extend(name for passed, name in tests if not passed)
        rows.append({
            "frame": index,
            "projected_pixel_count": int(projected.sum()),
            "visible_pixel_count": int(mask.sum()),
            "visible_fraction": float(mask.sum() / max(projected.sum(), 1)),
            "actor_camera_distance_m": distance,
            "actor_radial_velocity_mps": radial,
            "actor_tangential_velocity_mps": tangential,
            "median_background_depth_contrast_m": (
                float(np.median(contrast_values))
                if len(contrast_values) else None
            ),
            "minimum_background_depth_contrast_m": (
                float(np.min(contrast_values))
                if len(contrast_values) else None
            ),
            "newly_occupied_ray_count": int(newly_occupied.sum()),
            "newly_freed_ray_count": int(newly_freed.sum()),
            "history_support_projection": int((mask & previous_union).sum()),
            "free_space_seed_potential": int(newly_occupied.sum()),
            "range_seed_potential": int(contrast_seed.sum()),
            "expected_seed_count": int(potential.sum()),
            "expected_connected_component": _largest_component(potential),
            "camera_translation_m": camera_translation,
            "camera_yaw_delta_rad": yaw_delta,
            "history_warp_residual_proxy_m": camera_translation,
            "border_margin_pixels": border,
            "observable": not reasons,
            "rejection_reasons": reasons,
        })
    pre = 0
    for index in range(int(gap_start) - 1, -1, -1):
        if not rows[index]["observable"]:
            break
        pre += 1
    post = 0
    for index in range(int(gap_end) + 1, len(rows)):
        if not rows[index]["observable"]:
            break
        post += 1
    observable = (
        pre >= threshold["minimum_pre_gap_consecutive_observable_frames"]
        and post >= threshold["minimum_post_gap_consecutive_observable_frames"]
    )
    rejection = []
    if pre < threshold["minimum_pre_gap_consecutive_observable_frames"]:
        rejection.append("insufficient_pre_gap_consecutive_observability")
    if post < threshold["minimum_post_gap_consecutive_observable_frames"]:
        rejection.append("insufficient_post_gap_consecutive_observability")
    return {
        "version": OBSERVABILITY_VERSION,
        "role": "development_broad_phase_only",
        "observable": bool(observable),
        "rejection_reasons": rejection,
        "pre_gap_consecutive_observable_frames": pre,
        "post_gap_consecutive_observable_frames": post,
        "worst_frame_support": min(
            (row["visible_pixel_count"] for row in rows if row["visible_pixel_count"]),
            default=0,
        ),
        "minimum_depth_contrast_m": min(
            (
                row["minimum_background_depth_contrast_m"]
                for row in rows
                if row["minimum_background_depth_contrast_m"] is not None
            ),
            default=None,
        ),
        "minimum_expected_seed_count": min(
            (row["expected_seed_count"] for row in rows if row["visible_pixel_count"]),
            default=0,
        ),
        "minimum_component_support": min(
            (
                row["expected_connected_component"] for row in rows
                if row["visible_pixel_count"]
            ),
            default=0,
        ),
        "frames": rows,
        "config_hash": config["config_hash"],
        "implementation_hash": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "runtime_measurement_emitted": False,
        "runtime_gt_input_authorized": False,
    }


__all__ = [
    "OBSERVABILITY_VERSION", "certify_observability",
    "load_observability_config",
]
