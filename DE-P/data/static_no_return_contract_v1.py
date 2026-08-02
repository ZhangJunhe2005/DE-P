"""SNRE-CTR1 canonical no-return and leakage semantics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from data.static_yopo_preprocessing_v1 import StaticYOPODepthPreprocessorV1


class NoReturnClass:
    CANONICAL = "CANONICAL_NO_RETURN"
    INVALID_FILL = "CONSTANT_INVALID_FILL"
    CORRUPT = "CORRUPT_CONSTANT_FRAME"
    ORDINARY = "ORDINARY_DEPTH_FRAME"


@dataclass(frozen=True)
class CanonicalNoReturnDepthV1:
    height: int = 96
    width: int = 160
    dtype: str = "float32"
    max_depth_m: float = 20.0
    invalid_sentinels: tuple = (0.0,)
    version: str = "canonical_no_return_depth_v1"

    def classify(self, depth, source_hash=None):
        value = np.asarray(depth)
        if value.shape != (self.height, self.width):
            return NoReturnClass.CORRUPT
        if value.dtype != np.float32:
            return NoReturnClass.CORRUPT
        if not np.isfinite(value).all():
            return NoReturnClass.CORRUPT
        if not bool(np.all(value == value.flat[0])):
            return NoReturnClass.ORDINARY
        constant = float(value.flat[0])
        if constant in self.invalid_sentinels:
            return NoReturnClass.INVALID_FILL
        if constant != self.max_depth_m:
            return NoReturnClass.CORRUPT
        if source_hash:
            actual = hashlib.sha256(value.tobytes()).hexdigest()
            if actual != source_hash:
                return NoReturnClass.CORRUPT
        preprocessor = StaticYOPODepthPreprocessorV1(max_depth_m=self.max_depth_m)
        network = preprocessor(value)
        if not bool(np.all(network == np.float32(1.0))):
            return NoReturnClass.CORRUPT
        return NoReturnClass.CANONICAL

    def contract(self):
        return {
            "version": self.version, "shape": [self.height, self.width],
            "dtype": self.dtype, "all_finite": True,
            "all_pixels_exact_m": self.max_depth_m,
            "invalid_sentinels_forbidden": list(self.invalid_sentinels),
            "network_value": 1.0,
        }


@dataclass(frozen=True)
class StaticDatasetLeakagePolicyV2:
    version: str = "static_dataset_leakage_policy_v2"
    canonical_no_return_depth_equivalence_allowed: bool = True
    ordinary_depth_only_equivalence_is_identity_leakage: bool = False
    source_group_overlap_hard_fail: bool = True
    sample_identity_overlap_hard_fail: bool = True
    complete_input_overlap_hard_fail: bool = True
    invalid_fill_hard_fail: bool = True
    corrupt_constant_hard_fail: bool = True
    normalized_observation_quantization: float = 1.0e-5

    def contract(self):
        return dict(self.__dict__)


def normalize_observation_v1(observation):
    value = np.asarray(observation, dtype=np.float32).copy()
    if value.shape[-1] != 9:
        raise ValueError("observation must end with nine values")
    value[..., :3] /= np.float32(6.0)
    value[..., 3:6] /= np.float32(6.0)
    norm = np.linalg.norm(value[..., 6:9], axis=-1, keepdims=True)
    value[..., 6:9] /= np.maximum(norm, np.float32(10.0))
    return value


def canonical_hash(parts):
    digest = hashlib.sha256()
    for part in parts:
        raw = part if isinstance(part, bytes) else str(part).encode()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def exact_complete_input_hash(preprocessed_depth_hash, normalized_observation,
                              validity=True, preprocessing_version="static_yopo_depth_preprocessor_v1"):
    obs = np.asarray(normalized_observation, dtype="<f4")
    return canonical_hash((
        preprocessed_depth_hash, obs.tobytes(), b"VALID" if validity else b"INVALID",
        preprocessing_version,
    ))


def tolerance_observation_hash(normalized_observation, tolerance=1.0e-5):
    value = np.asarray(normalized_observation, dtype=np.float64)
    quantized = np.rint(value / float(tolerance)).astype("<i8")
    return hashlib.sha256(quantized.tobytes()).hexdigest()


def contract_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()

