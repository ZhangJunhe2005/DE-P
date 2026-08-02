"""Frozen model, observation, normalization, and loss contract."""

from __future__ import annotations

import hashlib
import json

from config.config import cfg

MODEL_CONTRACT = {
    "name": "MixedSceneStaticYOPOV1",
    "network_class": "policy.dep_network.DepNetwork",
    "backbone_variant": "legacy",
    "head_variant": "unified",
    "depth_shape": [1, 96, 160],
    "observation_shape": [9],
    "observation_order": [
        "velocity_body_x", "velocity_body_y", "velocity_body_z",
        "acceleration_body_x", "acceleration_body_y", "acceleration_body_z",
        "goal_body_x", "goal_body_y", "goal_body_z",
    ],
    "output": {"endstate": [9, 3, 5], "score": [3, 5], "primitive_count": 15},
    "dynamic_forward": False,
}
NORMALIZATION_CONTRACT = {
    "mode": "FIXED_PHYSICAL_TRANSFORM",
    "velocity_divisor": float(cfg["vel_max_train"]),
    "acceleration_divisor": float(cfg["acc_max_train"]),
    "goal_rule": "goal/max(norm(goal),goal_length)",
    "goal_length": float(cfg["goal_length"]),
    "empirical_statistics": False,
}
LOSS_CONTRACT = {
    "name": "MixedSceneStaticYOPOLossV1",
    "implementation": "loss.loss_function.DEPLoss",
    "components": ["smoothness", "static_safety", "guidance", "detached_score_smooth_l1"],
    "weights": {"smoothness": float(cfg["ws"]), "static_safety": float(cfg["wc"]),
                "guidance": float(cfg["wg"]), "trajectory": 1.0, "score": 1.0},
    "dynamic_loss_enabled": False,
    "trajectory_horizon_s": float(cfg["sgm_time"]),
    "uav_radius_m": 0.3,
    "safety_d0_m": float(cfg["d0"]),
    "safety_decay_m": float(cfg["r"]),
}


def contract_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


MODEL_CONTRACT_HASH = contract_hash(MODEL_CONTRACT)
NORMALIZATION_CONTRACT_HASH = contract_hash(NORMALIZATION_CONTRACT)
LOSS_CONTRACT_HASH = contract_hash(LOSS_CONTRACT)

