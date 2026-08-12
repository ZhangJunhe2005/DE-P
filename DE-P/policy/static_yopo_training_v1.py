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
from policy.static_yopo_preventive_safety_v1 import (
    PreventiveSafetyConfigV1,
    preventive_safety_objective_v1,
)
from policy.static_yopo_progress_safety_v1 import (
    ProgressSafetyConfigV1,
    progress_safety_objective_v1,
)
from policy.static_yopo_feasibility_score_v1 import (
    FeasibilityScoreConfigV1,
    feasibility_score_objective_v1,
)
from policy.static_yopo_goal_progress_v2 import (
    GoalProgressConfigV2,
    goal_directed_progress_objective_v2,
)
from policy.static_yopo_projected_score_v3 import (
    ProjectedScoreConfigV3,
    differentiable_stopping_distance_loss_v3,
    project_terminal_states_v3,
    projected_score_objective_v3,
)
from policy.static_yopo_simple_v4_3 import (
    SimpleYOPOConfigV43, simple_yopo_objective_v4_3,
)
from policy.static_yopo_parity_v4_4 import (
    StaticYOPOParityConfigV44, static_yopo_parity_objective_v4_4,
)
from policy.static_yopo_parity_v4_5 import (
    StaticYOPOParityConfigV45, static_yopo_parity_objective_v4_5,
)
from policy.static_yopo_parity_v4_5_1 import (
    StaticYOPOParityConfigV451, static_yopo_parity_objective_v4_5_1,
)
from policy.static_yopo_parity_v4_5_2 import (
    StaticYOPOParityConfigV452, static_yopo_parity_objective_v4_5_2,
)
from policy.static_yopo_parity_v4_5_3 import (
    StaticYOPOParityConfigV453, static_yopo_parity_objective_v4_5_3,
)
from policy.static_yopo_parity_v4_5_4 import (
    StaticYOPOParityConfigV454, static_yopo_parity_objective_v4_5_4,
)
from policy.static_yopo_parity_v4_5_5 import (
    StaticYOPOParityConfigV455, static_yopo_parity_objective_v4_5_5,
)
from policy.static_yopo_parity_v4_5_6 import (
    StaticYOPOParityConfigV456, static_yopo_parity_objective_v4_5_6,
)
from policy.static_yopo_parity_v4_5_8 import (
    StaticYOPOParityConfigV458, static_yopo_parity_objective_v4_5_8,
)
from policy.static_yopo_parity_v4_5_9 import (
    StaticYOPOParityConfigV459, static_yopo_parity_objective_v4_5_9,
)
from policy.static_yopo_parity_v4_5_10 import (
    StaticYOPOParityConfigV4510, static_yopo_parity_objective_v4_5_10,
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
    def __init__(self, initial_checkpoint=None, head_variant="unified",
                 allow_unified_to_split=False):
        super().__init__()
        dynamic = replace(
            DynamicPerceptionConfig.from_global_config(),
            enabled=False, use_attention=False, fallback_to_static=True,
        )
        self.network = DepNetwork(
            backbone_variant="legacy", head_variant=head_variant,
            dynamic_config=dynamic
        )
        self.initial_checkpoint_result = None
        if initial_checkpoint:
            self.initial_checkpoint_result = load_dep_checkpoint(
                self.network, initial_checkpoint, expected_variant="legacy",
                allow_unified_to_split=allow_unified_to_split,
            )

    def forward(self, depth, observation):
        return self.network.inference(depth, observation)


class MixedSceneStaticYOPOObjectiveV1(torch.nn.Module):
    def __init__(self, map_catalog, local_goal_horizon_m=None,
                 safety_first_config=None, kinodynamic_config=None,
                 preventive_safety_config=None, progress_safety_config=None,
                 feasibility_score_config=None, goal_progress_config=None,
                 projected_score_config=None, simple_yopo_v4_3_config=None,
                 static_yopo_v4_4_config=None,
                 static_yopo_v4_5_config=None,
                 static_yopo_v4_5_1_config=None,
                 static_yopo_v4_5_2_config=None,
                 static_yopo_v4_5_3_config=None,
                 static_yopo_v4_5_4_config=None,
                 static_yopo_v4_5_5_config=None,
                 static_yopo_v4_5_6_config=None,
                 static_yopo_v4_5_8_config=None,
                 static_yopo_v4_5_9_config=None,
                 static_yopo_v4_5_10_config=None):
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
        self.preventive_safety_config = PreventiveSafetyConfigV1.from_mapping(
            preventive_safety_config
        )
        self.progress_safety_config = ProgressSafetyConfigV1.from_mapping(
            progress_safety_config
        )
        self.feasibility_score_config = FeasibilityScoreConfigV1.from_mapping(
            feasibility_score_config
        )
        self.goal_progress_config = GoalProgressConfigV2.from_mapping(
            goal_progress_config
        )
        self.projected_score_config = ProjectedScoreConfigV3.from_mapping(
            projected_score_config
        )
        self.simple_yopo_v4_3_config = SimpleYOPOConfigV43.from_mapping(
            simple_yopo_v4_3_config
        )
        self.static_yopo_v4_4_config = StaticYOPOParityConfigV44.from_mapping(
            static_yopo_v4_4_config
        )
        self.static_yopo_v4_5_config = StaticYOPOParityConfigV45.from_mapping(
            static_yopo_v4_5_config
        )
        self.static_yopo_v4_5_1_config = (
            StaticYOPOParityConfigV451.from_mapping(
                static_yopo_v4_5_1_config
            )
        )
        self.static_yopo_v4_5_2_config = (
            StaticYOPOParityConfigV452.from_mapping(
                static_yopo_v4_5_2_config
            )
        )
        self.static_yopo_v4_5_3_config = (
            StaticYOPOParityConfigV453.from_mapping(
                static_yopo_v4_5_3_config
            )
        )
        self.static_yopo_v4_5_4_config = (
            StaticYOPOParityConfigV454.from_mapping(
                static_yopo_v4_5_4_config
            )
        )
        self.static_yopo_v4_5_5_config = (
            StaticYOPOParityConfigV455.from_mapping(
                static_yopo_v4_5_5_config
            )
        )
        self.static_yopo_v4_5_6_config = (
            StaticYOPOParityConfigV456.from_mapping(
                static_yopo_v4_5_6_config
            )
        )
        self.static_yopo_v4_5_8_config = (
            StaticYOPOParityConfigV458.from_mapping(
                static_yopo_v4_5_8_config
            )
        )
        self.static_yopo_v4_5_9_config = (
            StaticYOPOParityConfigV459.from_mapping(
                static_yopo_v4_5_9_config
            )
        )
        self.static_yopo_v4_5_10_config = (
            StaticYOPOParityConfigV4510.from_mapping(
                static_yopo_v4_5_10_config
            )
        )
        enabled_static_objectives = sum((
            self.simple_yopo_v4_3_config.enabled,
            self.static_yopo_v4_4_config.enabled,
            self.static_yopo_v4_5_config.enabled,
            self.static_yopo_v4_5_1_config.enabled,
            self.static_yopo_v4_5_2_config.enabled,
            self.static_yopo_v4_5_3_config.enabled,
            self.static_yopo_v4_5_4_config.enabled,
            self.static_yopo_v4_5_5_config.enabled,
            self.static_yopo_v4_5_6_config.enabled,
            self.static_yopo_v4_5_8_config.enabled,
            self.static_yopo_v4_5_9_config.enabled,
            self.static_yopo_v4_5_10_config.enabled,
        ))
        if enabled_static_objectives > 1:
            raise ValueError(
                "V4.3 through V4.5.10 objectives are mutually exclusive"
            )
        if self.static_yopo_v4_5_10_config.enabled:
            active = self.static_yopo_v4_5_10_config
            print("------ Active V4.5.10 Objective ------")
            print("| legacy DEPLoss weights above are bypassed |")
            print(f"| {'smooth unit':<16} = {active.jerk_unit_weight:6.4f} |")
            print(f"| {'safety':<16} = {active.safety_weight:6.4f} |")
            print(f"| {'guidance':<16} = {active.guidance_weight:6.4f} |")
            print(f"| {'time mean':<16} = {active.time_mean_weight:6.3f} |")
            print(f"| {'worst five':<16} = {active.worst_sample_weight:6.3f} |")
            print(f"| {'collision radius':<16} = {active.vehicle_radius_m:6.3f} m |")
            print("--------------------------------------")
        elif self.static_yopo_v4_5_9_config.enabled:
            active = self.static_yopo_v4_5_9_config
            print("------ Active V4.5.9 Objective ------")
            print("| legacy DEPLoss weights above are bypassed |")
            print(f"| {'smooth unit':<16} = {active.jerk_unit_weight:6.4f} |")
            print(f"| {'safety':<16} = {active.safety_weight:6.4f} |")
            print(f"| {'guidance':<16} = {active.guidance_weight:6.4f} |")
            print(f"| {'clear above':<16} = {active.clear_distance_m:6.3f} m |")
            print(f"| {'collision radius':<16} = {active.vehicle_radius_m:6.3f} m |")
            print(f"| {'aggregation':<16} = time mean ({active.static_safety_samples}) |")
            print("-------------------------------------")
        elif self.static_yopo_v4_5_8_config.enabled:
            active = self.static_yopo_v4_5_8_config
            print("------ Active V4.5.8 Objective ------")
            print("| legacy DEPLoss weights above are bypassed |")
            print(f"| {'smooth unit':<16} = {active.jerk_unit_weight:6.4f} |")
            print(f"| {'safety':<16} = {active.safety_weight:6.4f} |")
            print(f"| {'guidance':<16} = {active.guidance_weight:6.4f} |")
            print(f"| {'clear above':<16} = {active.clear_distance_m:6.3f} m |")
            print(f"| {'collision radius':<16} = {active.vehicle_radius_m:6.3f} m |")
            print("-------------------------------------")
        if (self.progress_safety_config.enabled
                and self.goal_progress_config.enabled):
            raise ValueError(
                "legacy distance progress and goal-directed progress cannot "
                "be enabled together"
            )
        if self.dep_loss.dynamic_loss_config.enabled:
            raise RuntimeError("static objective must disable dynamic loss")

    def _ensure_optional_config_defaults(self):
        """Keep old tests/checkpoints that construct this module manually usable.

        Historical smoke tests instantiate the objective through ``__new__`` to
        isolate the float32 trajectory-loss boundary.  Versioned objectives are
        optional, so an object created that way must behave like the original
        objective instead of failing merely because a newer config attribute is
        absent.
        """
        defaults = (
            ("feasibility_score_config", FeasibilityScoreConfigV1),
            ("goal_progress_config", GoalProgressConfigV2),
            ("projected_score_config", ProjectedScoreConfigV3),
            ("simple_yopo_v4_3_config", SimpleYOPOConfigV43),
            ("static_yopo_v4_4_config", StaticYOPOParityConfigV44),
            ("static_yopo_v4_5_config", StaticYOPOParityConfigV45),
            ("static_yopo_v4_5_1_config", StaticYOPOParityConfigV451),
            ("static_yopo_v4_5_2_config", StaticYOPOParityConfigV452),
            ("static_yopo_v4_5_3_config", StaticYOPOParityConfigV453),
            ("static_yopo_v4_5_4_config", StaticYOPOParityConfigV454),
            ("static_yopo_v4_5_5_config", StaticYOPOParityConfigV455),
            ("static_yopo_v4_5_6_config", StaticYOPOParityConfigV456),
            ("static_yopo_v4_5_8_config", StaticYOPOParityConfigV458),
            ("static_yopo_v4_5_9_config", StaticYOPOParityConfigV459),
            ("static_yopo_v4_5_10_config", StaticYOPOParityConfigV4510),
        )
        for name, config_type in defaults:
            if not hasattr(self, name):
                setattr(self, name, config_type())

    def forward(self, model, batch):
        self._ensure_optional_config_defaults()
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
            if self.static_yopo_v4_5_10_config.enabled:
                return static_yopo_parity_objective_v4_5_10(
                    sampler=self.dep_loss.safety_loss.trajectory_sampler,
                    safety_loss=self.dep_loss.safety_loss,
                    fixed=start.repeat_interleave(count, 0).permute(0, 2, 1),
                    predicted=end.permute(0, 2, 1),
                    predicted_scores=score.reshape(batch_size, count),
                    goal_world=goal_world.repeat_interleave(count, 0),
                    map_id=map_id,
                    batch_size=batch_size,
                    candidate_count=count,
                    route_goal_distance=physical_observation[:, 6:9].norm(dim=1),
                    config=self.static_yopo_v4_5_10_config,
                )
            if self.static_yopo_v4_5_9_config.enabled:
                return static_yopo_parity_objective_v4_5_9(
                    sampler=self.dep_loss.safety_loss.trajectory_sampler,
                    safety_loss=self.dep_loss.safety_loss,
                    fixed=start.repeat_interleave(count, 0).permute(0, 2, 1),
                    predicted=end.permute(0, 2, 1),
                    predicted_scores=score.reshape(batch_size, count),
                    goal_world=goal_world.repeat_interleave(count, 0),
                    map_id=map_id,
                    batch_size=batch_size,
                    candidate_count=count,
                    route_goal_distance=physical_observation[:, 6:9].norm(dim=1),
                    config=self.static_yopo_v4_5_9_config,
                )
            if self.static_yopo_v4_5_8_config.enabled:
                return static_yopo_parity_objective_v4_5_8(
                    sampler=self.dep_loss.safety_loss.trajectory_sampler,
                    safety_loss=self.dep_loss.safety_loss,
                    fixed=start.repeat_interleave(count, 0).permute(0, 2, 1),
                    predicted=end.permute(0, 2, 1),
                    predicted_scores=score.reshape(batch_size, count),
                    goal_world=goal_world.repeat_interleave(count, 0),
                    map_id=map_id,
                    batch_size=batch_size,
                    candidate_count=count,
                    route_goal_distance=physical_observation[:, 6:9].norm(dim=1),
                    config=self.static_yopo_v4_5_8_config,
                )
            if self.static_yopo_v4_5_6_config.enabled:
                return static_yopo_parity_objective_v4_5_6(
                    sampler=self.dep_loss.safety_loss.trajectory_sampler,
                    safety_loss=self.dep_loss.safety_loss,
                    fixed=start.repeat_interleave(count, 0).permute(0, 2, 1),
                    predicted=end.permute(0, 2, 1),
                    predicted_scores=score.reshape(batch_size, count),
                    goal_world=goal_world.repeat_interleave(count, 0),
                    map_id=map_id,
                    batch_size=batch_size,
                    candidate_count=count,
                    route_goal_distance=physical_observation[:, 6:9].norm(dim=1),
                    config=self.static_yopo_v4_5_6_config,
                )
            if self.static_yopo_v4_5_5_config.enabled:
                return static_yopo_parity_objective_v4_5_5(
                    sampler=self.dep_loss.safety_loss.trajectory_sampler,
                    safety_loss=self.dep_loss.safety_loss,
                    fixed=start.repeat_interleave(count, 0).permute(0, 2, 1),
                    predicted=end.permute(0, 2, 1),
                    predicted_scores=score.reshape(batch_size, count),
                    goal_world=goal_world.repeat_interleave(count, 0),
                    map_id=map_id,
                    batch_size=batch_size,
                    candidate_count=count,
                    route_goal_distance=physical_observation[:, 6:9].norm(dim=1),
                    config=self.static_yopo_v4_5_5_config,
                )
            if self.static_yopo_v4_5_4_config.enabled:
                return static_yopo_parity_objective_v4_5_4(
                    sampler=self.dep_loss.safety_loss.trajectory_sampler,
                    safety_loss=self.dep_loss.safety_loss,
                    fixed=start.repeat_interleave(count, 0).permute(0, 2, 1),
                    predicted=end.permute(0, 2, 1),
                    predicted_scores=score.reshape(batch_size, count),
                    goal_world=goal_world.repeat_interleave(count, 0),
                    map_id=map_id,
                    batch_size=batch_size,
                    candidate_count=count,
                    route_goal_distance=physical_observation[:, 6:9].norm(dim=1),
                    config=self.static_yopo_v4_5_4_config,
                )
            if self.static_yopo_v4_5_3_config.enabled:
                return static_yopo_parity_objective_v4_5_3(
                    sampler=self.dep_loss.safety_loss.trajectory_sampler,
                    safety_loss=self.dep_loss.safety_loss,
                    fixed=start.repeat_interleave(count, 0).permute(0, 2, 1),
                    predicted=end.permute(0, 2, 1),
                    predicted_scores=score.reshape(batch_size, count),
                    goal_world=goal_world.repeat_interleave(count, 0),
                    map_id=map_id,
                    batch_size=batch_size,
                    candidate_count=count,
                    route_goal_distance=physical_observation[:, 6:9].norm(dim=1),
                    config=self.static_yopo_v4_5_3_config,
                )
            if self.static_yopo_v4_5_2_config.enabled:
                return static_yopo_parity_objective_v4_5_2(
                    sampler=self.dep_loss.safety_loss.trajectory_sampler,
                    safety_loss=self.dep_loss.safety_loss,
                    fixed=start.repeat_interleave(count, 0).permute(0, 2, 1),
                    predicted=end.permute(0, 2, 1),
                    predicted_scores=score.reshape(batch_size, count),
                    goal_world=goal_world.repeat_interleave(count, 0),
                    map_id=map_id,
                    batch_size=batch_size,
                    candidate_count=count,
                    route_goal_distance=physical_observation[:, 6:9].norm(dim=1),
                    config=self.static_yopo_v4_5_2_config,
                )
            if self.static_yopo_v4_5_1_config.enabled:
                return static_yopo_parity_objective_v4_5_1(
                    sampler=self.dep_loss.safety_loss.trajectory_sampler,
                    safety_loss=self.dep_loss.safety_loss,
                    fixed=start.repeat_interleave(count, 0).permute(0, 2, 1),
                    predicted=end.permute(0, 2, 1),
                    predicted_scores=score.reshape(batch_size, count),
                    goal_world=goal_world.repeat_interleave(count, 0),
                    map_id=map_id,
                    batch_size=batch_size,
                    candidate_count=count,
                    route_goal_distance=physical_observation[:, 6:9].norm(dim=1),
                    config=self.static_yopo_v4_5_1_config,
                )
            if self.static_yopo_v4_5_config.enabled:
                return static_yopo_parity_objective_v4_5(
                    sampler=self.dep_loss.safety_loss.trajectory_sampler,
                    safety_loss=self.dep_loss.safety_loss,
                    fixed=start.repeat_interleave(count, 0).permute(0, 2, 1),
                    predicted=end.permute(0, 2, 1),
                    predicted_scores=score.reshape(batch_size, count),
                    goal_world=goal_world.repeat_interleave(count, 0),
                    map_id=map_id,
                    batch_size=batch_size,
                    candidate_count=count,
                    route_goal_distance=physical_observation[:, 6:9].norm(dim=1),
                    config=self.static_yopo_v4_5_config,
                )
            if self.static_yopo_v4_4_config.enabled:
                return static_yopo_parity_objective_v4_4(
                    sampler=self.dep_loss.safety_loss.trajectory_sampler,
                    safety_loss=self.dep_loss.safety_loss,
                    fixed=start.repeat_interleave(count, 0).permute(0, 2, 1),
                    predicted=end.permute(0, 2, 1),
                    predicted_scores=score.reshape(batch_size, count),
                    goal_world=goal_world.repeat_interleave(count, 0),
                    map_id=map_id,
                    batch_size=batch_size,
                    candidate_count=count,
                    route_goal_distance=physical_observation[:, 6:9].norm(dim=1),
                    config=self.static_yopo_v4_4_config,
                )
            if self.simple_yopo_v4_3_config.enabled:
                return simple_yopo_objective_v4_3(
                    sampler=self.dep_loss.safety_loss.trajectory_sampler,
                    safety_loss=self.dep_loss.safety_loss,
                    fixed=start.repeat_interleave(count, 0).permute(0, 2, 1),
                    predicted=end.permute(0, 2, 1),
                    predicted_scores=score.reshape(batch_size, count),
                    goal_world=goal_world.repeat_interleave(count, 0),
                    map_id=map_id,
                    batch_size=batch_size,
                    candidate_count=count,
                    route_goal_distance=physical_observation[:, 6:9].norm(dim=1),
                    config=self.simple_yopo_v4_3_config,
                )
            details = self.dep_loss(
                start.repeat_interleave(count, 0), end,
                goal_world.repeat_interleave(count, 0), map_id,
                dynamic_obstacles=None, return_details=True,
            )
            candidate_count = int(count)
            predicted_scores = score.reshape(batch_size, candidate_count)
            candidate_endstate = endstate.permute(0, 2, 3, 1).reshape(
                batch_size, candidate_count, 9
            )
            candidate_endpoint_distance = candidate_endstate[:, :, :3].norm(
                dim=2
            )
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
            elif self.safety_first_config.enabled:
                trajectory_max_speed, trajectory_max_acceleration = (
                    sample_quintic_kinematic_maxima_v1(
                        self.dep_loss.safety_loss.trajectory_sampler,
                        start.repeat_interleave(count, 0).permute(0, 2, 1),
                        end.permute(0, 2, 1), batch_size, candidate_count,
                    )
                )
            else:
                trajectory_max_speed = torch.zeros_like(
                    candidate_static_cost
                )
                trajectory_max_acceleration = torch.zeros_like(
                    candidate_static_cost
                )
            preventive = None
            if self.preventive_safety_config.enabled:
                preventive = preventive_safety_objective_v1(
                    predicted_scores,
                    details.static_min_distance.reshape(
                        batch_size, candidate_count
                    ),
                    physical_observation[:, 0:3].norm(dim=1),
                    depth,
                    self.preventive_safety_config,
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
                    candidate_preventive_cost=(
                        None if preventive is None
                        else preventive["candidate_label_cost"]
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
                safety_qualified = ~safety_first["unsafe_mask"]
            else:
                label_scores = (
                    base_label_scores if preventive is None
                    else base_label_scores + preventive["candidate_label_cost"]
                )
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
                safety_qualified = torch.ones_like(
                    candidate_static_cost, dtype=torch.bool
                )
            if preventive is None:
                zero_per_sample = candidate_static_cost[:, 0] * 0.0
                preventive = {
                    "per_sample_loss": zero_per_sample,
                    "ranking_per_sample": zero_per_sample,
                    "candidate_loss": candidate_static_cost * 0.0,
                    "required_clearance": zero_per_sample,
                    "sample_weight": zero_per_sample + 1.0,
                    "clear_candidate_count": zero_per_sample,
                    "selected_clearance": zero_per_sample,
                    "selected_below_margin": torch.zeros_like(
                        zero_per_sample, dtype=torch.bool
                    ),
                    "clear_mask": torch.ones_like(
                        candidate_static_cost, dtype=torch.bool
                    ),
                }
            # Progress is subordinate to every safety contract.  A long but
            # unsafe trajectory never receives a progress preference.
            safety_qualified = safety_qualified & preventive["clear_mask"]
            if self.goal_progress_config.enabled:
                static_clear_for_progress = (
                    details.static_min_distance.reshape(
                        batch_size, candidate_count
                    ).ge(self.safety_first_config.required_clearance_m)
                    & preventive["clear_mask"]
                ).detach()
                progress = goal_directed_progress_objective_v2(
                    predicted_scores,
                    candidate_endstate[:, :, :3],
                    objective_goal_body,
                    static_clear_for_progress,
                    self.goal_progress_config,
                )
                label_scores = (
                    label_scores + progress["candidate_label_cost"]
                ).detach()
                per_sample_score_loss = F.smooth_l1_loss(
                    predicted_scores, label_scores, reduction="none"
                ).mean(dim=1)
            elif self.progress_safety_config.enabled:
                progress = progress_safety_objective_v1(
                    predicted_scores,
                    candidate_endpoint_distance,
                    safety_qualified,
                    self.progress_safety_config,
                )
                label_scores = (
                    label_scores + progress["candidate_label_cost"]
                ).detach()
                per_sample_score_loss = F.smooth_l1_loss(
                    predicted_scores, label_scores, reduction="none"
                ).mean(dim=1)
            else:
                zero_per_sample = candidate_static_cost[:, 0] * 0.0
                progress = {
                    "per_sample_loss": zero_per_sample,
                    "ranking_per_sample": zero_per_sample,
                    "candidate_loss": candidate_static_cost * 0.0,
                    "safe_progress_candidate_count": zero_per_sample,
                    "preferred_progress_candidate_count": zero_per_sample,
                    "selected_progress": zero_per_sample,
                    "selected_insufficient_progress": torch.zeros_like(
                        zero_per_sample, dtype=torch.bool
                    ),
                    "selected_goal_progress": zero_per_sample,
                    "selected_goal_alignment": zero_per_sample,
                    "selected_reverse": torch.zeros_like(
                        zero_per_sample, dtype=torch.bool
                    ),
                }
            if "selected_goal_progress" not in progress:
                progress["selected_goal_progress"] = progress[
                    "selected_progress"
                ]
                progress["selected_goal_alignment"] = progress[
                    "selected_progress"
                ] * 0.0
                progress["selected_reverse"] = progress[
                    "selected_progress"
                ].lt(0.0)
            if self.feasibility_score_config.enabled:
                # This is the only score-selection contract in V4.2.6.
                # Preventive clearance and progress remain differentiable
                # proposal-generation losses, but cannot change the hard
                # feasible/infeasible class boundary used by the score head.
                feasibility_score = feasibility_score_objective_v1(
                    predicted_scores,
                    base_label_scores
                    + preventive["candidate_loss"].detach()
                    + progress["candidate_loss"].detach(),
                    details.static_min_distance.reshape(
                        batch_size, candidate_count
                    ),
                    trajectory_max_speed,
                    trajectory_max_acceleration,
                    self.feasibility_score_config,
                )
                label_scores = feasibility_score["labels"]
                per_sample_score_loss = feasibility_score["per_sample_loss"]
                # Ranking is included once in per_sample_score_loss.  Keep the
                # metric observable without adding it a second time below.
                per_sample_ranking_loss = feasibility_score[
                    "ranking_per_sample"
                ]
                selected_unsafe = feasibility_score[
                    "selected_hard_infeasible"
                ]
                selected_hardware_unsafe = (
                    (trajectory_max_speed > self.feasibility_score_config.max_speed_mps)
                    | (trajectory_max_acceleration
                       > self.feasibility_score_config.max_acceleration_mps2)
                )[
                    torch.arange(batch_size, device=endstate.device),
                    predicted_scores.argmin(dim=1),
                ]
            projected_score = None
            stopping_distance = None
            if self.projected_score_config.enabled:
                # Score labels are intentionally generated from a detached
                # replay of the exact candidates used by ROS. Candidate-head
                # gradients continue through the original trajectory,
                # kinodynamic and goal-progress objectives above.
                fixed_axes = start.repeat_interleave(count, 0).permute(
                    0, 2, 1
                )
                predicted_axes = end.permute(0, 2, 1)
                stopping_distance = differentiable_stopping_distance_loss_v3(
                    self.dep_loss.safety_loss.trajectory_sampler,
                    fixed_axes, predicted_axes, batch_size, candidate_count,
                    self.dep_loss.safety_loss, map_id,
                    self.projected_score_config,
                    static_distance_samples=details.static_distance_samples,
                )
                with torch.no_grad():
                    projection = project_terminal_states_v3(
                        self.dep_loss.safety_loss.trajectory_sampler,
                        fixed_axes, predicted_axes, batch_size,
                        candidate_count, self.projected_score_config,
                    )
                    positions = projection["position_world"]
                    _, static_distance = (
                        self.dep_loss.safety_loss.get_distance_cost(
                            positions.reshape(batch_size, -1, 3), map_id
                        )
                    )
                    static_distance = static_distance.reshape(
                        batch_size, candidate_count, positions.shape[2]
                    )
                    projected_derivatives = projection[
                        "projected_derivatives"
                    ]
                    projected_smoothness = (
                        self.dep_loss.smoothness_weight
                        * self.dep_loss.smoothness_loss(
                            fixed_axes, projected_derivatives
                        )
                    ).reshape(batch_size, candidate_count)
                projected_score = projected_score_objective_v3(
                    predicted_scores, projection, static_distance,
                    rotation, objective_goal_body, projected_smoothness,
                    self.projected_score_config,
                )
                label_scores = projected_score["labels"]
                per_sample_score_loss = projected_score["per_sample_loss"]
                per_sample_ranking_loss = projected_score[
                    "ranking_per_sample"
                ]
                selected_unsafe = ~projected_score["selected_safe"]
                selected_hardware_unsafe = projected_score[
                    "selected_hardware_unsafe"
                ]
                # All validation quantities below describe the projected
                # candidates, never the pre-projection training proposal.
                trajectory_max_speed = projection["maximum_speed"]
                trajectory_max_acceleration = projection[
                    "maximum_acceleration"
                ]
                feasible_candidate_count = projected_score[
                    "safe_candidate_count"
                ]
                candidate_time_dilation = projection[
                    "projection_scale"
                ].clamp_min(1.0e-6).reciprocal()
            labels = label_scores.reshape(-1)
            selected_index = predicted_scores.argmin(dim=1)
            selected_endstate = candidate_endstate[
                torch.arange(batch_size, device=endstate.device),
                selected_index,
            ]
            selected_endpoint_distance = selected_endstate[:, :3].norm(dim=1)
            selected_endpoint_speed = selected_endstate[:, 3:6].norm(dim=1)
            if projected_score is not None:
                rows = torch.arange(batch_size, device=endstate.device)
                selected_endpoint_distance = projection["position_world"][
                    rows, selected_index, -1
                ].sub(position).norm(dim=1)
                selected_endpoint_speed = projection["velocity_world"][
                    rows, selected_index, -1
                ].norm(dim=1)
                progress["selected_goal_progress"] = projected_score[
                    "selected_goal_progress"
                ]
                progress["selected_goal_alignment"] = projected_score[
                    "selected_goal_alignment"
                ]
                progress["selected_reverse"] = projected_score[
                    "selected_goal_progress"
                ].lt(-self.goal_progress_config.maximum_reverse_progress_m)
                progress["selected_insufficient_progress"] = (
                    projected_score["selected_goal_progress"]
                    < self.projected_score_config.minimum_goal_progress_m
                )
            selected_hover = selected_endpoint_distance.lt(0.75)
            selected_trajectory_max_speed = trajectory_max_speed[
                torch.arange(batch_size, device=endstate.device), selected_index
            ]
            selected_trajectory_max_acceleration = trajectory_max_acceleration[
                torch.arange(batch_size, device=endstate.device), selected_index
            ]
            if projected_score is not None:
                candidate_normal_acceleration = trajectory_max_acceleration * 0.0
            elif kinodynamic is None:
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
            preventive_safety_loss = preventive["per_sample_loss"].mean()
            preventive_ranking_loss = preventive[
                "ranking_per_sample"
            ].mean()
            progress_safety_loss = progress["per_sample_loss"].mean()
            progress_ranking_loss = progress["ranking_per_sample"].mean()
            stopping_distance_loss = (
                torch.zeros((), device=endstate.device, dtype=endstate.dtype)
                if stopping_distance is None
                else stopping_distance["per_sample_loss"].mean()
            )
            progress_loss_weight = (
                self.goal_progress_config.loss_weight
                if self.goal_progress_config.enabled
                else self.progress_safety_config.loss_weight
                if self.progress_safety_config.enabled else 0.0
            )
            progress_ranking_weight = (
                self.goal_progress_config.ranking_weight
                if self.goal_progress_config.enabled
                else self.progress_safety_config.ranking_weight
                if self.progress_safety_config.enabled else 0.0
            )
            total = (
                details.trajectory_training_loss + score_loss
                + self.safety_first_config.ranking_weight * ranking_loss
                + self.safety_first_config.safety_cvar_weight * safety_cvar_loss
                + self.safety_first_config.kinematic_weight * kinematic_loss
                + self.preventive_safety_config.loss_weight
                * preventive_safety_loss
                + self.preventive_safety_config.ranking_weight
                * preventive_ranking_loss
                + progress_loss_weight
                * progress_safety_loss
                + progress_ranking_weight
                * progress_ranking_loss
                + self.projected_score_config.candidate_stopping_loss_weight
                * stopping_distance_loss
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
            "preventive_safety_loss": preventive_safety_loss,
            "preventive_ranking_loss": preventive_ranking_loss,
            "progress_safety_loss": progress_safety_loss,
            "progress_ranking_loss": progress_ranking_loss,
            "stopping_distance_loss": stopping_distance_loss,
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
            "candidate_preventive_cost": preventive["candidate_loss"],
            "candidate_progress_cost": progress["candidate_loss"],
            "candidate_projected_safe_mask": (
                torch.zeros_like(candidate_static_cost, dtype=torch.bool)
                if projected_score is None else projected_score["safe_mask"]
            ),
            "candidate_stopping_distance_cost": (
                torch.zeros_like(candidate_static_cost)
                if stopping_distance is None
                else stopping_distance["candidate_loss"].reshape(-1)
            ),
            "per_sample_total_loss": (
                per_sample_trajectory + per_sample_score_loss
                + self.safety_first_config.ranking_weight * per_sample_ranking_loss
                + self.safety_first_config.safety_cvar_weight * per_sample_safety_cvar
                + self.safety_first_config.kinematic_weight * per_sample_kinematic_loss
                + self.preventive_safety_config.loss_weight
                * preventive["per_sample_loss"]
                + self.preventive_safety_config.ranking_weight
                * preventive["ranking_per_sample"]
                + progress_loss_weight
                * progress["per_sample_loss"]
                + progress_ranking_weight
                * progress["ranking_per_sample"]
                + self.projected_score_config.candidate_stopping_loss_weight
                * (
                    torch.zeros_like(per_sample_trajectory)
                    if stopping_distance is None
                    else stopping_distance["per_sample_loss"]
                )
            ),
            "per_sample_trajectory_loss": per_sample_trajectory,
            "per_sample_score_loss": per_sample_score_loss,
            "per_sample_ranking_loss": per_sample_ranking_loss,
            "per_sample_safety_cvar_loss": per_sample_safety_cvar,
            "per_sample_kinematic_loss": per_sample_kinematic_loss,
            "per_sample_preventive_safety_loss":
                preventive["per_sample_loss"],
            "per_sample_preventive_ranking_loss":
                preventive["ranking_per_sample"],
            "per_sample_progress_safety_loss": progress["per_sample_loss"],
            "per_sample_progress_ranking_loss": progress["ranking_per_sample"],
            "per_sample_stopping_distance_loss": (
                torch.zeros_like(per_sample_trajectory)
                if stopping_distance is None
                else stopping_distance["per_sample_loss"]
            ),
            "per_sample_stoppable_candidate_count": (
                torch.zeros_like(per_sample_trajectory)
                if stopping_distance is None
                else stopping_distance["stoppable_candidate_count"].to(
                    endstate.dtype
                )
            ),
            "per_sample_preventive_required_clearance":
                preventive["required_clearance"],
            "per_sample_preventive_sample_weight":
                preventive["sample_weight"],
            "per_sample_clear_candidate_count":
                preventive["clear_candidate_count"].to(endstate.dtype),
            "per_sample_selected_clearance": preventive["selected_clearance"],
            "per_sample_anticipatory_unsafe_selection":
                preventive["selected_below_margin"].float(),
            "per_sample_safe_progress_candidate_count":
                progress["safe_progress_candidate_count"].to(endstate.dtype),
            "per_sample_preferred_progress_candidate_count":
                progress["preferred_progress_candidate_count"].to(
                    endstate.dtype
                ),
            "per_sample_insufficient_progress_selection":
                progress["selected_insufficient_progress"].float(),
            "per_sample_selected_goal_progress":
                progress["selected_goal_progress"],
            "per_sample_selected_goal_alignment":
                progress["selected_goal_alignment"],
            "per_sample_reverse_selection":
                progress["selected_reverse"].float(),
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
            "per_sample_projected_candidate_availability": (
                torch.zeros(batch_size, device=endstate.device,
                            dtype=endstate.dtype)
                if projected_score is None
                else projected_score["candidate_available"].to(endstate.dtype)
            ),
            "per_sample_projected_safe_candidate_count": (
                torch.zeros(batch_size, device=endstate.device,
                            dtype=endstate.dtype)
                if projected_score is None
                else projected_score["safe_candidate_count"].to(endstate.dtype)
            ),
            "per_sample_projected_conditional_selection_error_numerator": (
                torch.zeros(batch_size, device=endstate.device,
                            dtype=endstate.dtype)
                if projected_score is None
                else projected_score["conditional_selection_error"].to(
                    endstate.dtype
                )
            ),
            "per_sample_projected_selected_stopping_reserve": (
                torch.zeros(batch_size, device=endstate.device,
                            dtype=endstate.dtype)
                if projected_score is None
                else projected_score["selected_stopping_reserve"]
            ),
            "per_sample_projected_selected_visible": (
                torch.zeros(batch_size, device=endstate.device,
                            dtype=endstate.dtype)
                if projected_score is None
                else projected_score["selected_visible"].to(endstate.dtype)
            ),
            "per_sample_projected_selection_regret": (
                torch.zeros(batch_size, device=endstate.device,
                            dtype=endstate.dtype)
                if projected_score is None
                else projected_score["selection_regret"]
            ),
            "per_sample_route_goal_distance":
                physical_observation[:, 6:9].norm(dim=1),
            "per_sample_objective_goal_distance":
                objective_goal_body.norm(dim=1),
            "per_sample_score_top1_match": predicted_scores.argmin(dim=1).eq(
                label_scores.argmin(dim=1)
            ),
        }
