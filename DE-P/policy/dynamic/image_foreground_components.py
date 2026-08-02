"""Linear-time seed-guided components on a raw metric range image."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as connected_components


@dataclass(frozen=True)
class ImageForegroundComponent:
    pixels_vu: np.ndarray
    seed_count: int
    free_space_seed_count: int
    range_seed_count: int
    mean_history_support: float
    mean_residual: float
    static_consistency: float
    confidence: float


def _point(depth, v, u, camera):
    z = float(depth[v, u])
    return np.asarray(((u - camera.cx) * z / camera.fx,
                       (v - camera.cy) * z / camera.fy, z), dtype=np.float32)


def grow_seeded_components(frame, seed_mask, range_seed, free_seed,
                           static_explained, support, residual, config):
    """Grow only from hard seeds and never cross depth/3-D discontinuities."""
    shape = frame.depth_m.shape
    for name, value in (("seed_mask", seed_mask), ("range_seed", range_seed),
                        ("free_seed", free_seed), ("static_explained", static_explained)):
        if np.asarray(value).shape != shape:
            raise ValueError(f"{name} shape must match raw depth")
    static_explained = np.asarray(static_explained, dtype=bool)
    support = np.asarray(support)
    residual = np.asarray(residual)
    allowed = (frame.valid_mask & ~static_explained
               & (support >= config.range_min_history_support)
               & (residual > 0.0))
    seeds = np.asarray(seed_mask, dtype=bool) & allowed
    # Pixels at a hard local range/3-D jump may remain hard seeds but cannot
    # become an unconstrained bridge into a neighbouring surface.
    z = frame.depth_m
    vv, uu = np.indices(shape, dtype=np.float32)
    camera = frame.camera_model
    xyz = np.stack(((uu - camera.cx) * z / camera.fx,
                    (vv - camera.cy) * z / camera.fy, z), axis=-1)
    edge = np.zeros(shape, dtype=bool)
    for dv, du in ((1, 0), (0, 1)):
        first = (slice(0, -1), slice(None)) if dv else (slice(None), slice(0, -1))
        second = (slice(1, None), slice(None)) if dv else (slice(None), slice(1, None))
        za, zb = z[first], z[second]
        limit = config.range_growth_abs_depth + config.range_growth_rel_depth * np.minimum(za, zb)
        jump = ((np.abs(za - zb) > limit)
                | (np.linalg.norm(xyz[first] - xyz[second], axis=-1)
                   > config.range_growth_max_3d_neighbor_distance))
        edge[first] |= jump; edge[second] |= jump
    allowed &= (~edge | seeds)
    structure = (np.ones((3, 3), dtype=np.uint8)
                 if config.range_component_connectivity == 8
                 else np.asarray(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8))
    labels, label_count = connected_components(allowed, structure=structure)
    components = []
    seed_labels = np.unique(labels[seeds])
    for component_label in seed_labels:
        if component_label == 0:
            continue
        pixels = np.argwhere(labels == component_label).astype(np.int32)
        if len(pixels) > config.range_max_component_pixels:
            continue
        component_seeds = seeds[pixels[:, 0], pixels[:, 1]]
        seed_count = int(component_seeds.sum())
        if (seed_count < config.range_min_seed_pixels
                or seed_count / len(pixels) < config.range_min_seed_fraction):
            continue
        depths = frame.depth_m[pixels[:, 0], pixels[:, 1]]
        if float(np.ptp(depths)) > config.range_max_component_depth_span:
            continue
        rv = range_seed[pixels[:, 0], pixels[:, 1]]
        fv = free_seed[pixels[:, 0], pixels[:, 1]]
        seed_fraction = seed_count / len(pixels)
        history_value = float(np.mean(support[pixels[:, 0], pixels[:, 1]]))
        residual_value = float(np.mean(np.maximum(
            residual[pixels[:, 0], pixels[:, 1]], 0.0
        )))
        confidence = float(np.clip(
            0.45 * min(1.0, seed_fraction / max(config.range_min_seed_fraction, 1e-9))
            + 0.35 * min(1.0, history_value / config.range_min_history_support)
            + 0.20 * min(1.0, residual_value / config.range_abs_residual_threshold),
            0.0, 1.0,
        ))
        components.append(ImageForegroundComponent(
            pixels, seed_count, int(fv.sum()), int(rv.sum()), history_value,
            residual_value, float(np.mean(static_explained[pixels[:, 0], pixels[:, 1]])),
            confidence,
        ))
    return tuple(components)
