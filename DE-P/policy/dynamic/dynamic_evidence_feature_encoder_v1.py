"""Small MLP+GRU M1 prototype; forward-only in DEM-DCR1."""

from __future__ import annotations

import torch
from torch import nn


CONTRACT_VERSION = "causal_feature_temporal_dynamic_evidence_v1"


class CausalFeatureTemporalDynamicEvidenceV1(nn.Module):
    def __init__(self, feature_dim=32, hidden_dim=48):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.frame_encoder = nn.Sequential(
            nn.Linear(self.feature_dim*2, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.temporal = nn.GRU(
            self.hidden_dim, self.hidden_dim, batch_first=True
        )
        self.classifier = nn.Linear(self.hidden_dim, 4)
        self.actionability = nn.Linear(self.hidden_dim, 1)

    def forward(self, features, validity_mask, time_mask):
        if features.ndim != 3 or validity_mask.shape != features.shape:
            raise ValueError("features/mask must be BxKxF")
        if time_mask.shape != features.shape[:2]:
            raise ValueError("time mask must be BxK")
        masked = features*validity_mask.to(features.dtype)
        encoded = self.frame_encoder(torch.cat((
            masked, validity_mask.to(features.dtype)
        ), dim=-1))
        encoded = encoded*time_mask.unsqueeze(-1).to(encoded.dtype)
        values, _ = self.temporal(encoded)
        index = time_mask.to(torch.int64).sum(dim=1).clamp(min=1)-1
        pooled = values[
            torch.arange(values.shape[0], device=values.device), index
        ]
        return {
            "logits": self.classifier(pooled),
            "actionability_logit": self.actionability(pooled).squeeze(-1),
        }


__all__ = ["CONTRACT_VERSION", "CausalFeatureTemporalDynamicEvidenceV1"]
