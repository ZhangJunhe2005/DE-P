"""Static-only YOPO training graph; dynamic modules are structurally excluded."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch.nn import functional as F

from config.config import cfg
from loss.dynamic_types import DynamicLossConfig
from loss.loss_function import DEPLoss
from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_network import DepNetwork
from policy.dynamic.types import DynamicPerceptionConfig
from policy.state_transform import state_body2world
from policy.static_yopo_safety_first_v1 import (
    StaticSafetyFirstConfigV1,
    sample_quintic_kinematic_maxima_v1,
    safety_first_score_objective_v1,
)
from policy.static_yopo_kinodynamic_v2 import (
    KinodynamicFeasibilityConfigV2,
    dense_quintic_kinodynamic_objective_v2,
)


def local_planning_goal(goal_body, horizon_m):
    """Preserve direction while bounding supervision to the local horizon."""
    if horizon_m is None:
        return goal_body
    horizon = float(horizon_m)
    if horizon <= 0.0:
        raise ValueError("local goal horizon must be positive")
    distance = goal_body.norm(dim=1, keepdim=True)
    scale = (horizon / distance.clamp(min=1e-8)).clamp(max=1.0)
    return goal_body * scale


class MixedSceneStaticYOPOV1(torch.nn.Module):
    def __init__(self, initial_checkpoint=None):
        super().__init__()
        dynamic = replace(
            DynamicPerceptionConfig.from_global_config(),
            enabled=False, use_attention=False, fallback_to_static=True,
        )
        self.network = DepNetwork(
            backbone_variant="legacy", head_variant="unified", dynamic_config=dynamic
        )
        self.initial_checkpoint_result = None
        if initial_checkpoint:
            self.initial_checkpoint_result = load_dep_checkpoint(
                self.network, initial_checkpoint, expected_variant="legacy"
            )

    def forward(self, depth, observation):
        return self.network.inference(depth, observation)


class MixedSceneStaticYOPOObjectiveV1(torch.nn.Module):
    def __init__(self, map_catalog, local_goal_horizon_m=None,
                 safety_first_config=None, kinodynamic_config=None):
        super().__init__()
        self.local_goal_horizon_m = (
            None if local_goal_horizon_m is None
            else float(local_goal_horizon_m)
        )
        config = replace(DynamicLossConfig.from_global_config(), enabled=False)
        self.dep_loss = DEPLoss(dynamic_loss_config=config, map_catalog=map_catalog)
        self.safety_first_config = StaticSafetyFirstConfigV1.from_mapping(
            safety_first_config
        )
        self.kinodynamic_config = KinodynamicFeasibilityConfigV2.from_mapping(
            kinodynamic_config
        )
        if self.dep_loss.dynamic_loss_config.enabled:
            raise RuntimeError("static objective must disable dynamic loss")

    def forward(self, model, batch):
        depth = batch["depth"]
        observation = batch["observation"]
        # Loss construction must always use physical body-frame units.  Keep
        # an explicit copy even though StateTransform.normalize_obs is now
        # out-of-place; this is a contract boundary against future regressions.
        physical_observation = observation.float().clone()
        position = batch["position_world"]
        rotation = batch["rotation_world_from_body"]
        map_id = batch["map_id"]
        batch_size = depth.shape[0]
        # The CNN may run under AMP, but polynomial construction and all
        # trajectory losses are numerically unsafe in float16 (the snap
        # quadratic form can overflow on otherwise valid samples).
        endstate, score = model(depth, observation)
        if endstate.shape != (batch_size, 9, 3, 5) or score.shape != (batch_size, 3, 5):
            raise RuntimeError("static YOPO output contract violation")
        with torch.autocast(device_type=depth.device.type, enabled=False):
            position = position.float()
            rotation = rotation.float()
            endstate = endstate.float()
            score = score.float()
            objective_goal_body = local_planning_goal(
                physical_observation[:, 6:9], self.local_goal_horizon_m
            )
            goal_world, velocity_world, acceleration_world = state_body2world(
                position, rotation, objective_goal_body,
                physical_observation[:, 0:3],
                physical_observation[:, 3:6],
            )
            start = torch.stack([position, velocity_world, acceleration_world], dim=1)
            count = cfg["traj_num"]
            flat = endstate.permute(0, 2, 3, 1).reshape(batch_size * count, 9)
            pos = position.repeat_interleave(count, 0)
            rot = rotation.repeat_interleave(count, 0)
            end_pos, end_vel, end_acc = state_body2world(
                pos, rot, flat[:, :3], flat[:, 3:6], flat[:, 6:9]
            )
            end = torch.stack([end_pos, end_vel, end_acc], dim=1)
            details = self.dep_loss(
                start.repeat_interleave(count, 0), end,
                goal_world.repeat_interleave(count, 0), map_id,
                dynamic_obstacles=None, return_details=True,
            )
            candidate_count = int(count)
            predicted_scores = score.reshape(batch_size, candidate_count)
            base_label_scores = details.detached_score_label().reshape(
                batch_size, candidate_count
            )
            candidate_static_cost = details.static_safety_cost.reshape(
                batch_size, candidate_count
            )
            kinodynamic = None
            if self.kinodynamic_config.enabled:
                kinodynamic = dense_quintic_kinodynamic_objective_v2(
                    self.dep_loss.safety_loss.trajectory_sampler,
                    start.repeat_interleave(count, 0).permute(0, 2, 1),
                    end.permute(0, 2, 1), batch_size, candidate_count,
                    self.kinodynamic_config,
                )
                trajectory_max_speed = kinodynamic["maximum_speed"]
                trajectory_max_acceleration = kinodynamic[
                    "maximum_acceleration"
                ]
            else:
                trajectory_max_speed, trajectory_max_acceleration = (
                    sample_quintic_kinematic_maxima_v1(
                        self.dep_loss.safety_loss.trajectory_sampler,
                        start.repeat_interleave(count, 0).permute(0, 2, 1),
                        end.permute(0, 2, 1), batch_size, candidate_count,
                    )
                )
            if self.safety_first_config.enabled:
                safety_first = safety_first_score_objective_v1(
                    predicted_scores, base_label_scores,
                    details.static_min_distance.reshape(batch_size, candidate_count),
                    candidate_static_cost, self.safety_first_config,
                    trajectory_max_speed, trajectory_max_acceleration,
                    candidate_kinematic_cost=(
                        None if kinodynamic is None
                        else kinodynamic["candidate_label_cost"]
                    ),
                )
                label_scores = safety_first["labels"]
                per_sample_score_loss = safety_first["regression_per_sample"]
                per_sample_ranking_loss = safety_first["ranking_per_sample"]
                per_sample_safety_cvar = safety_first["safety_cvar_per_sample"]
                per_sample_kinematic_loss = safety_first["kinematic_per_sample"]
                if kinodynamic is not None:
                    per_sample_kinematic_loss = kinodynamic["per_sample_loss"]
                selected_unsafe = safety_first["selected_unsafe"]
                selected_hardware_unsafe = safety_first[
                    "selected_hardware_unsafe"
                ]
            else:
                label_scores = base_label_scores
                per_sample_score_loss = F.smooth_l1_loss(
                    predicted_scores, label_scores, reduction="none"
                ).mean(dim=1)
                per_sample_ranking_loss = predicted_scores[:, 0] * 0.0
                per_sample_safety_cvar = candidate_static_cost[:, 0] * 0.0
                per_sample_kinematic_loss = candidate_static_cost[:, 0] * 0.0
                selected_unsafe = torch.zeros(
                    batch_size, device=endstate.device, dtype=torch.bool
                )
                selected_hardware_unsafe = selected_unsafe
            labels = label_scores.reshape(-1)
            candidate_endstate = endstate.permute(0, 2, 3, 1).reshape(
                batch_size, candidate_count, 9
            )
            selected_index = predicted_scores.argmin(dim=1)
            selected_endstate = candidate_endstate[
                torch.arange(batch_size, device=endstate.device),
                selected_index,
            ]
            selected_endpoint_distance = selected_endstate[:, :3].norm(dim=1)
            selected_endpoint_speed = selected_endstate[:, 3:6].norm(dim=1)
            selected_hover = selected_endpoint_distance.lt(0.75)
            selected_trajectory_max_speed = trajectory_max_speed[
                torch.arange(batch_size, device=endstate.device), selected_index
            ]
            selected_trajectory_max_acceleration = trajectory_max_acceleration[
                torch.arange(batch_size, device=endstate.device), selected_index
            ]
            if kinodynamic is None:
                feasible_candidate_count = (
                    (trajectory_max_speed <= self.safety_first_config.max_speed_mps)
                    & (trajectory_max_acceleration
                       <= self.safety_first_config.max_acceleration_mps2)
                ).sum(dim=1)
                candidate_time_dilation = torch.maximum(
                    trajectory_max_speed / self.safety_first_config.max_speed_mps,
                    torch.sqrt((
                        trajectory_max_acceleration
                        / self.safety_first_config.max_acceleration_mps2
                    ).clamp_min(0.0)),
                ).clamp_min(1.0)
                candidate_normal_acceleration = trajectory_max_acceleration * 0.0
            else:
                feasible_candidate_count = kinodynamic[
                    "feasible_candidate_count"
                ]
                candidate_time_dilation = kinodynamic["time_dilation_ratio"]
                candidate_normal_acceleration = kinodynamic[
                    "maximum_normal_acceleration"
                ]
            selected_time_dilation = candidate_time_dilation[
                torch.arange(batch_size, device=endstate.device), selected_index
            ]
            selected_normal_acceleration = candidate_normal_acceleration[
                torch.arange(batch_size, device=endstate.device), selected_index
            ]
            score_loss = per_sample_score_loss.mean()
            ranking_loss = per_sample_ranking_loss.mean()
            safety_cvar_loss = per_sample_safety_cvar.mean()
            kinematic_loss = per_sample_kinematic_loss.mean()
            total = (
                details.trajectory_training_loss + score_loss
                + self.safety_first_config.ranking_weight * ranking_loss
                + self.safety_first_config.safety_cvar_weight * safety_cvar_loss
                + self.safety_first_config.kinematic_weight * kinematic_loss
            )
            per_sample_smoothness = details.smooth_cost.reshape(
                batch_size, candidate_count
            ).mean(dim=1)
            per_sample_static_safety = details.static_safety_cost.reshape(
                batch_size, candidate_count
            ).mean(dim=1)
            per_sample_guidance = details.guidance_cost.reshape(
                batch_size, candidate_count
            ).mean(dim=1)
            per_sample_trajectory = (
                per_sample_smoothness
                + per_sample_static_safety
                + per_sample_guidance
            )
        return {
            "total_loss": total,
            "trajectory_loss": details.trajectory_training_loss,
            "score_loss": score_loss,
            "ranking_loss": ranking_loss,
            "safety_cvar_loss": safety_cvar_loss,
            "kinematic_loss": kinematic_loss,
            "smoothness_loss": details.smooth_cost.mean(),
            "static_safety_loss": details.static_safety_cost.mean(),
            "guidance_loss": details.guidance_cost.mean(),
            "dynamic_safety_loss": details.dynamic_safety_cost.mean(),
            "score_label": labels,
            "candidate_smooth_cost": details.smooth_cost,
            "candidate_static_cost": details.static_safety_cost,
            "candidate_guidance_cost": details.guidance_cost,
            "candidate_kinodynamic_cost": (
                candidate_static_cost * 0.0 if kinodynamic is None
                else kinodynamic["candidate_loss"]
            ),
            "per_sample_total_loss": (
                per_sample_trajectory + per_sample_score_loss
                + self.safety_first_config.ranking_weight * per_sample_ranking_loss
                + self.safety_first_config.safety_cvar_weight * per_sample_safety_cvar
                + self.safety_first_config.kinematic_weight * per_sample_kinematic_loss
            ),
            "per_sample_trajectory_loss": per_sample_trajectory,
            "per_sample_score_loss": per_sample_score_loss,
            "per_sample_ranking_loss": per_sample_ranking_loss,
            "per_sample_safety_cvar_loss": per_sample_safety_cvar,
            "per_sample_kinematic_loss": per_sample_kinematic_loss,
            "per_sample_unsafe_selection": selected_unsafe.float(),
            "per_sample_hardware_unsafe_selection":
                selected_hardware_unsafe.float(),
            "per_sample_selected_trajectory_max_speed":
                selected_trajectory_max_speed,
            "per_sample_selected_trajectory_max_acceleration":
                selected_trajectory_max_acceleration,
            "per_sample_feasible_candidate_count":
                feasible_candidate_count.to(endstate.dtype),
            "per_sample_candidate_time_dilation_mean":
                candidate_time_dilation.mean(dim=1),
            "per_sample_selected_time_dilation": selected_time_dilation,
            "per_sample_selected_normal_acceleration":
                selected_normal_acceleration,
            "per_sample_smoothness_loss": per_sample_smoothness,
            "per_sample_static_safety_loss": per_sample_static_safety,
            "per_sample_guidance_loss": per_sample_guidance,
            "per_sample_selected_endpoint_distance":
                selected_endpoint_distance,
            "per_sample_selected_endpoint_speed": selected_endpoint_speed,
            "per_sample_hover_selection": selected_hover.float(),
            "per_sample_route_goal_distance":
                physical_observation[:, 6:9].norm(dim=1),
            "per_sample_objective_goal_distance":
                objective_goal_body.norm(dim=1),
            "per_sample_score_top1_match": predicted_scores.argmin(dim=1).eq(
                label_scores.argmin(dim=1)
            ),
        }
