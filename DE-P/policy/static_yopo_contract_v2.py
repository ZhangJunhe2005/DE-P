"""Route-A V4 model/loss contracts with immutable physical observations."""

from __future__ import annotations

from policy.static_yopo_contract_v1 import (
    LOSS_CONTRACT,
    MODEL_CONTRACT,
    NORMALIZATION_CONTRACT,
    contract_hash,
)


MODEL_CONTRACT_V2 = {
    **MODEL_CONTRACT,
    "contract_version": "route_a_static_yopo_model_v2",
}
NORMALIZATION_CONTRACT_V2 = {
    **NORMALIZATION_CONTRACT,
    "contract_version": "physical_observation_non_mutating_v2",
    "network_transform_out_of_place": True,
    "loss_consumes_original_physical_units": True,
}
LOSS_CONTRACT_V2 = {
    **LOSS_CONTRACT,
    "contract_version": "route_a_static_yopo_loss_v2",
    "physical_start_state": True,
    "physical_goal_distance_preserved": True,
}

MODEL_CONTRACT_V2_HASH = contract_hash(MODEL_CONTRACT_V2)
NORMALIZATION_CONTRACT_V2_HASH = contract_hash(NORMALIZATION_CONTRACT_V2)
LOSS_CONTRACT_V2_HASH = contract_hash(LOSS_CONTRACT_V2)
