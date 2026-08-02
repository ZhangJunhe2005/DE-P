"""Versioned offline candidates for temporal foreground contract review."""

from __future__ import annotations

from collections import deque
import time

import numpy as np
from scipy.ndimage import label as connected_components

from .image_foreground_components import (
    ImageForegroundComponent, grow_seeded_components,
)
from .range_image_foreground import (
    _observation, depth_edge_magnitude, reproject_history_depth,
)
from .types import DepthFrame, DynamicPerceptionConfig


CONTRACT_VERSION = "temporal_foreground_observability_contract_v2_1"


def _common(frame, history, config):
    predictions = [reproject_history_depth(old, frame) for old in history]
    shape = frame.depth_m.shape
    support = np.zeros(shape, dtype=np.int16)
    closer_count = np.zeros(shape, dtype=np.int16)
    farther_count = np.zeros(shape, dtype=np.int16)
    static_count = np.zeros(shape, dtype=np.int16)
    max_positive = np.zeros(shape, dtype=np.float32)
    max_negative = np.zeros(shape, dtype=np.float32)
    residuals = np.empty((0, *shape), dtype=np.float32)
    valid = np.empty((0, *shape), dtype=bool)
    threshold = (
        config.range_abs_residual_threshold
        + config.range_rel_residual_threshold * frame.depth_m
    )
    if predictions:
        predicted = np.stack(predictions)
        valid = np.isfinite(predicted) & frame.valid_mask[None]
        residuals = predicted - frame.depth_m[None]
        support = valid.sum(axis=0).astype(np.int16)
        closer_count = (
            valid & (residuals > threshold[None])
        ).sum(axis=0).astype(np.int16)
        farther_count = (
            valid & (residuals < -threshold[None])
        ).sum(axis=0).astype(np.int16)
        static_count = (
            valid & (
                np.abs(residuals)
                <= config.range_static_consistency_threshold
            )
        ).sum(axis=0).astype(np.int16)
        max_positive = np.max(
            np.where(valid, residuals, 0.0), axis=0
        ).astype(np.float32)
        max_negative = np.max(
            np.where(valid, -residuals, 0.0), axis=0
        ).astype(np.float32)
    return {
        "predictions": predictions, "valid": valid, "residuals": residuals,
        "support": support, "closer_count": closer_count,
        "farther_count": farther_count, "static_count": static_count,
        "max_positive": max_positive, "max_negative": max_negative,
        "threshold": threshold,
        "static_explained":
            static_count >= config.range_min_history_support,
        "edge": depth_edge_magnitude(frame.depth_m, frame.valid_mask),
    }


def _labels(frame, components):
    labels = np.full(len(frame.points_camera), -1, dtype=np.int64)
    point_lookup = np.full(frame.depth_m.shape, -1, dtype=np.int64)
    point_lookup[
        frame.pixels_uv[:, 1], frame.pixels_uv[:, 0]
    ] = np.arange(len(frame.pixels_uv), dtype=np.int64)
    component_mask = np.zeros(frame.depth_m.shape, dtype=bool)
    for index, component in enumerate(components):
        pixels = component.pixels_vu
        point_indices = point_lookup[pixels[:, 0], pixels[:, 1]]
        labels[point_indices[point_indices >= 0]] = index
        component_mask[pixels[:, 0], pixels[:, 1]] = True
    return labels, component_mask


class CandidateRangeImageForeground:
    """Causal bounded candidate; never selected without an explicit registry key."""

    def __init__(self, config: DynamicPerceptionConfig, name, parameters):
        config.validate()
        self.config = config
        self.name = str(name)
        self.parameters = dict(parameters)
        self._history = deque(maxlen=config.range_history_frames)
        self._last_timestamp = None
        self.last_diagnostics = {}
        self.last_range_seed = None
        self.last_free_seed = None
        self.last_component_mask = None

    def reset(self):
        self._history.clear()
        self._last_timestamp = None
        self.last_diagnostics = {}
        self.last_range_seed = self.last_free_seed = self.last_component_mask = None

    def _candidate_a(self, frame, free_seed, common):
        decay = float(self.parameters["decay"])
        minimum = float(self.parameters["minimum_weighted_closer_support"])
        weighted = np.zeros(frame.depth_m.shape, dtype=np.float32)
        count = len(common["residuals"])
        for index in range(count):
            age = count - 1 - index
            evidence = (
                common["valid"][index]
                & (common["residuals"][index] > common["threshold"])
            )
            weighted += evidence.astype(np.float32) * decay ** age
        strong = (
            (weighted >= minimum)
            & ~common["static_explained"]
            & (common["edge"] <= self.config.range_edge_guard_threshold)
            & frame.valid_mask
        )
        seeds = (strong | free_seed) & ~common["static_explained"]
        components = grow_seeded_components(
            frame, seeds, strong, free_seed, common["static_explained"],
            common["support"], common["max_positive"], self.config,
        )
        return components, strong, {
            "weighted_closer_support_pixels": int((weighted >= minimum).sum()),
            "weak_support_pixels": int((common["max_positive"] > 0).sum()),
        }

    def _candidate_b(self, frame, free_seed, common):
        weak_threshold = float(self.parameters["weak_residual_m"])
        minimum_strong = int(self.parameters["minimum_strong_seeds"])
        minimum_pixels = int(self.parameters["minimum_component_pixels"])
        strong = (
            (
                common["closer_count"]
                >= self.config.range_min_history_support
            ) | free_seed
        ) & ~common["static_explained"]
        weak = (
            frame.valid_mask
            & (common["support"] >= self.config.range_min_history_support)
            & (common["max_positive"] >= weak_threshold)
            & ~common["static_explained"]
            & (common["edge"] <= self.config.range_edge_guard_threshold)
        )
        structure = np.ones((3, 3), dtype=np.uint8)
        labels, count = connected_components(weak, structure=structure)
        components = []
        for component_id in range(1, count + 1):
            pixels = np.argwhere(labels == component_id).astype(np.int32)
            if len(pixels) < minimum_pixels:
                continue
            if len(pixels) > self.config.range_max_component_pixels:
                continue
            v, u = pixels.T
            strong_count = int(strong[v, u].sum())
            if strong_count < minimum_strong:
                continue
            depths = frame.depth_m[v, u]
            if float(np.ptp(depths)) > self.config.range_max_component_depth_span:
                continue
            rv = (
                common["closer_count"][v, u]
                >= self.config.range_min_history_support
            )
            fv = free_seed[v, u]
            history = float(np.mean(common["support"][v, u]))
            residual = float(np.mean(common["max_positive"][v, u]))
            confidence = float(np.clip(
                .35 * min(1.0, strong_count / max(minimum_strong, 1))
                + .35 * min(
                    1.0, history / self.config.range_min_history_support
                )
                + .30 * min(
                    1.0, residual / self.config.range_abs_residual_threshold
                ), 0, 1,
            ))
            components.append(ImageForegroundComponent(
                pixels_vu=pixels, seed_count=strong_count,
                free_space_seed_count=int(fv.sum()),
                range_seed_count=int(rv.sum()),
                mean_history_support=history, mean_residual=residual,
                static_consistency=float(np.mean(
                    common["static_explained"][v, u]
                )),
                confidence=confidence,
            ))
        return tuple(components), strong, {
            "positive_residual_component_pixels": int(weak.sum()),
            "weak_support_pixels": int(weak.sum()),
        }

    def _candidate_c(self, frame, free_seed, common):
        minimum = int(self.parameters["minimum_channel_support"])
        closer = common["closer_count"] >= minimum
        farther = common["farther_count"] >= minimum
        strong = (
            (closer | free_seed)
            & ~common["static_explained"]
            & (common["edge"] <= self.config.range_edge_guard_threshold)
            & frame.valid_mask
        )
        # Farther/disocclusion evidence is diagnostic context and may grow only
        # a component anchored by a closer/free hard seed.
        residual = np.maximum(common["max_positive"], common["max_negative"])
        components = grow_seeded_components(
            frame, strong, closer, free_seed, common["static_explained"],
            common["support"], residual, self.config,
        )
        return components, strong, {
            "closer_channel_pixels": int(closer.sum()),
            "freed_channel_pixels": int(farther.sum()),
            "channel_overlap_pixels": int((closer & farther).sum()),
            "weak_support_pixels": int((residual > 0).sum()),
        }

    def extract(self, frame: DepthFrame, free_space_seed=None):
        if not isinstance(frame, DepthFrame):
            raise TypeError("candidate foreground requires DepthFrame")
        started = time.perf_counter()
        if self._last_timestamp is not None:
            gap = frame.timestamp - self._last_timestamp
            if gap <= 0:
                raise ValueError("timestamps must be strictly increasing")
            if gap > self.config.range_reset_gap:
                self.reset()
        while self._history and (
            frame.timestamp - self._history[0].timestamp
            > self.config.range_history_max_age
        ):
            self._history.popleft()
        free_seed = (
            np.zeros(frame.depth_m.shape, dtype=bool)
            if free_space_seed is None
            else np.asarray(free_space_seed, dtype=bool)
        )
        common = _common(frame, self._history, self.config)
        if self.name == "candidate_a":
            components, strong, detail = self._candidate_a(
                frame, free_seed, common
            )
        elif self.name == "candidate_b":
            components, strong, detail = self._candidate_b(
                frame, free_seed, common
            )
        elif self.name == "candidate_c":
            components, strong, detail = self._candidate_c(
                frame, free_seed, common
            )
        else:
            raise ValueError(f"unsupported candidate extractor: {self.name}")
        observations = tuple(
            _observation(component, frame, index, self.config)
            for index, component in enumerate(components)
        )
        labels, component_mask = _labels(frame, components)
        self.last_range_seed = strong.copy()
        self.last_free_seed = free_seed.copy()
        self.last_component_mask = component_mask
        self._history.append(frame)
        self._last_timestamp = frame.timestamp
        self.last_diagnostics = {
            "mode": self.name, "contract_version": CONTRACT_VERSION,
            "history_size": len(self._history),
            "history_capacity": self.config.range_history_frames,
            "future_frames_used": 0, "gt_inputs_used": 0,
            "range_seed_count": int(strong.sum()),
            "free_space_seed_count": int(free_seed.sum()),
            "static_explained_count": int(
                common["static_explained"].sum()
            ),
            "component_count": len(components),
            "component_pixel_count": int(component_mask.sum()),
            "component_evidence": [
                {
                    "seed_count": item.seed_count,
                    "range_seed_count": item.range_seed_count,
                    "free_space_seed_count": item.free_space_seed_count,
                    "pixel_count": len(item.pixels_vu),
                    "confidence": item.confidence,
                }
                for item in components
            ],
            "runtime_ms": (time.perf_counter() - started) * 1000,
            **detail,
        }
        return observations, labels

    @property
    def history_size(self):
        return len(self._history)


class RestrictedTrackReacquisition:
    """Candidate D: current-depth evidence inside a live predicted-track ROI."""

    def __init__(self, roi_sigma=2.0, minimum_residual_pixels=4):
        self.roi_sigma = float(roi_sigma)
        self.minimum_residual_pixels = int(minimum_residual_pixels)
        if self.roi_sigma <= 0:
            raise ValueError("roi_sigma must be positive")
        if self.minimum_residual_pixels <= 0:
            raise ValueError("minimum_residual_pixels must be positive")

    def evaluate(self, residual_mask, predicted_roi, track_alive):
        residual = np.asarray(residual_mask, dtype=bool)
        roi = np.asarray(predicted_roi, dtype=bool)
        if residual.shape != roi.shape:
            raise ValueError("residual and ROI shapes differ")
        accepted = residual & roi if bool(track_alive) else np.zeros_like(roi)
        return {
            "accepted": bool(
                track_alive
                and accepted.sum() >= self.minimum_residual_pixels
            ),
            "pixel_count": int(accepted.sum()),
            "new_track_birth_allowed": False,
            "current_depth_evidence_required": True,
            "future_or_gt_used": False,
            "roi_sigma": self.roi_sigma,
        }
