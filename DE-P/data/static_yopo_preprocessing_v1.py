"""The single frozen depth preprocessing implementation for static YOPO."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StaticYOPODepthPreprocessorV1:
    height: int = 96
    width: int = 160
    depth_scale: float = 1.0
    min_depth_m: float = 0.04
    max_depth_m: float = 20.0
    inpaint_radius: int = 1

    version = "static_yopo_depth_preprocessor_v1"

    def contract(self):
        return {
            "version": self.version,
            "source": {"dtype": "float32", "unit": "m", "shape": [96, 160]},
            "resize": "INTER_NEAREST",
            "invalid": "non_finite_or_scaled_depth_below_0.04m",
            "clip": "[0,20]m",
            "normalization": "depth_m/20",
            "inpaint": "uint8_OpenCV_INPAINT_NS_radius_1",
            "output": {"dtype": "float32", "shape": [1, 96, 160]},
            **asdict(self),
        }

    @property
    def contract_hash(self):
        raw = json.dumps(self.contract(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def __call__(self, source_depth_m: np.ndarray) -> np.ndarray:
        depth = np.asarray(source_depth_m, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"source depth must be HxW, got {depth.shape}")
        if depth.shape != (self.height, self.width):
            depth = cv2.resize(
                depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST
            )
        scaled_m = depth * np.float32(self.depth_scale)
        invalid = ~np.isfinite(scaled_m) | (scaled_m < self.min_depth_m)
        normalized = np.minimum(scaled_m, self.max_depth_m) / self.max_depth_m
        normalized[invalid] = 0.0
        image_u8 = np.uint8(np.clip(normalized, 0.0, 1.0) * 255.0)
        filled = (
            cv2.inpaint(
                image_u8, np.uint8(invalid), self.inpaint_radius, cv2.INPAINT_NS
            )
            if bool(invalid.any()) else image_u8
        )
        result = filled.astype(np.float32)[None, :, :] / np.float32(255.0)
        if not np.isfinite(result).all():
            raise FloatingPointError("network-ready depth contains NaN/Inf")
        return np.ascontiguousarray(result)


DEFAULT_STATIC_YOPO_PREPROCESSOR_V1 = StaticYOPODepthPreprocessorV1()
