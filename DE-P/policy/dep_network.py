"""
DE-Planner Network
forward, prediction, pre-processing, post-processing
"""

import torch
from torch import nn
import numpy as np
import json
from dataclasses import asdict
from policy.models.backbone import DepBackbone
from policy.backbone_variant import resolve_backbone_variant
from policy.models.head import DepHead
from policy.state_transform import *
from policy.dynamic.context import ATTENTION_BACKBONE_OUTPUT, DynamicContext
from policy.dynamic.types import DynamicPerceptionConfig


class DepNetwork(nn.Module):

    def __init__(
            self,
            observation_dim=9,  # 9: v_xyz, a_xyz, goal_xyz
            output_dim=10,  # 10: x_pva, y_pva, z_pva, score
            hidden_state=64,
            backbone_variant=None,
            dynamic_config=None,
            head_variant="unified",
    ):
        super(DepNetwork, self).__init__()
        self.state_transform = StateTransform()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone_variant = resolve_backbone_variant(backbone_variant)
        self.dynamic_config = dynamic_config or DynamicPerceptionConfig.from_global_config()
        self.dynamic_config.validate()
        if head_variant not in DepHead.VARIANTS:
            raise ValueError(f"head_variant must be one of {DepHead.VARIANTS}")
        self.head_variant = head_variant
        self.last_dynamic_fallback_reason = None
        print("Dynamic perception config:", json.dumps(asdict(self.dynamic_config), sort_keys=True))

        print(f"Backbone variant: {self.backbone_variant}")
        self.image_backbone = DepBackbone(hidden_state, backbone_variant=self.backbone_variant)
        self.state_backbone = nn.Sequential()
        self.dep_head = DepHead(
            hidden_state + observation_dim, output_dim, variant=self.head_variant
        )

    def _forward_impl(self, depth, obs, attention=None):
        depth_feature = self.image_backbone(
            depth, attention=attention, attention_alpha=self.dynamic_config.attention_alpha
        )
        obs_feature = self.state_backbone(obs)
        input_tensor = torch.cat((obs_feature, depth_feature), 1)
        output = self.dep_head(input_tensor)
        endstate = torch.tanh(output[:, :9])
        score = torch.nn.functional.softplus(output[:, 9])
        return endstate, score

    def forward(self, depth: torch.Tensor, obs: torch.Tensor,
                dynamic_context: DynamicContext = None) -> torch.Tensor:
        """
            forward propagation of neural network
        """
        self.last_dynamic_fallback_reason = None
        use_dynamic = self.dynamic_config.enabled and self.dynamic_config.use_attention
        if not use_dynamic:
            return self._forward_impl(depth, obs)
        if dynamic_context is None or not dynamic_context.valid:
            reason = "missing_dynamic_context" if dynamic_context is None else str(
                dynamic_context.diagnostics.get("fallback_reason", "invalid_dynamic_context")
            )
            if not self.dynamic_config.fallback_to_static:
                raise RuntimeError(f"dynamic input unavailable and static fallback is disabled: {reason}")
            self.last_dynamic_fallback_reason = reason
            return self._forward_impl(depth, obs)
        attention = dynamic_context.attention(ATTENTION_BACKBONE_OUTPUT)
        if attention is None:
            if not self.dynamic_config.fallback_to_static:
                raise RuntimeError("valid DynamicContext has no backbone attention map")
            self.last_dynamic_fallback_reason = "missing_attention_map"
            return self._forward_impl(depth, obs)
        if dynamic_context.batch_size != depth.shape[0]:
            raise ValueError(
                "DynamicContext batch size must exactly match network input; "
                "each batch sample requires an independent context"
            )
        try:
            result = self._forward_impl(depth, obs, attention)
            if not all(bool(torch.isfinite(value).all()) for value in result):
                raise FloatingPointError("dynamic network output contains NaN/Inf")
            return result
        except (FloatingPointError, ValueError) as exc:
            if not self.dynamic_config.fallback_to_static:
                raise
            self.last_dynamic_fallback_reason = f"dynamic_forward_failed: {exc}"
            return self._forward_impl(depth, obs)

    def inference(self, depth: torch.Tensor, obs: torch.Tensor,
                  dynamic_context: DynamicContext = None) -> torch.Tensor:
        """
            For network training:
            (1) normalize the input state and transform to primitive frame
            (2) forward propagation
            (3) convert the prediction to endstate in body frame.
            obs: current state in the body frame.
            return: end state in the body frame
        """
        obs = self.state_transform.normalize_obs(obs)
        obs = self.state_transform.prepare_input(obs)
        endstate_pred, score_pred = self.forward(depth, obs, dynamic_context=dynamic_context)
        endstate = self.state_transform.pred_to_endstate(endstate_pred)
        return endstate, score_pred

    def print_grad(self, grad):
        print("grad of hook: ", grad)
