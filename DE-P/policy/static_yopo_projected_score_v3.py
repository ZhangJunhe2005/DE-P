"""Runtime-equivalent projected score supervision for Route-A V4.2.9.

The candidate head is still trained with the differentiable, unprojected
quintic objectives.  Score supervision, however, must describe the candidates
that are actually presented to the runtime selector.  This module therefore
replays the runtime's discrete terminal-state projection under ``no_grad`` and
builds one lexicographic label from the projected trajectories:

1. hardware-feasible, camera-observable and stoppable candidates outrank all
   other candidates;
2. among safe candidates, useful goal progress outranks detours;
3. clearance beyond the physically required stopping reserve is saturated and
   never rewarded as an excuse for a longer route.

Only two offline readiness quantities are exposed: projected candidate
availability and conditional selection error.  Closed-loop arrival/collision
is deliberately left to the independent RViz evaluation layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F



def _sample_pva(sampler, fixed_derivatives, predicted_derivatives, sample_count):
    """Sample the exact inclusive time grid used by the ROS safety shield."""
    coefficients = sampler.coefficients(fixed_derivatives, predicted_derivatives)
    times = torch.linspace(
        0.0, sampler.duration, int(sample_count),
        device=coefficients.device, dtype=coefficients.dtype,
    ).view(1, -1)
    positions, velocities, accelerations = [], [], []
    for axis in range(3):
        c = coefficients[:, 6 * axis:6 * (axis + 1)]
        positions.append(
            c[:, 0:1] + c[:, 1:2] * times + c[:, 2:3] * times ** 2
            + c[:, 3:4] * times ** 3 + c[:, 4:5] * times ** 4
            + c[:, 5:6] * times ** 5
        )
        velocities.append(
            c[:, 1:2] + 2.0 * c[:, 2:3] * times
            + 3.0 * c[:, 3:4] * times ** 2
            + 4.0 * c[:, 4:5] * times ** 3
            + 5.0 * c[:, 5:6] * times ** 4
        )
        accelerations.append(
            2.0 * c[:, 2:3] + 6.0 * c[:, 3:4] * times
            + 12.0 * c[:, 4:5] * times ** 2
            + 20.0 * c[:, 5:6] * times ** 3
        )
    return (
        torch.stack(positions, dim=2),
        torch.stack(velocities, dim=2),
        torch.stack(accelerations, dim=2),
        times.squeeze(0),
    )


@dataclass(frozen=True)
class ProjectedScoreConfigV3:
    enabled: bool = False
    max_speed_mps: float = 6.0
    max_acceleration_mps2: float = 6.0
    projection_min_scale: float = 0.15
    projection_steps: int = 18
    trajectory_samples: int = 81
    limit_tolerance: float = 1.0e-3
    vehicle_radius_m: float = 0.30
    tracking_margin_m: float = 0.35
    reaction_time_s: float = 0.12
    braking_acceleration_mps2: float = 6.0
    clearance_sensor_tolerance_m: float = 0.08
    horizontal_fov_deg: float = 90.0
    vertical_fov_deg: float = 60.0
    visibility_margin_deg: float = 5.0
    minimum_path_length_m: float = 1.0
    minimum_goal_progress_m: float = 0.25
    preferred_goal_progress_m: float = 3.0
    clearance_saturation_m: float = 1.0
    unsafe_label_floor: float = 4.0
    progress_weight: float = 1.0
    detour_weight: float = 0.25
    smoothness_weight: float = 0.05
    regression_weight: float = 0.25
    listwise_weight: float = 1.0
    safety_ranking_weight: float = 2.0
    quality_ranking_weight: float = 1.0
    ranking_margin: float = 1.0
    quality_pair_gap: float = 0.10
    target_temperature: float = 0.35
    prediction_temperature: float = 1.0
    candidate_trajectory_samples: int = 41
    candidate_stopping_loss_weight: float = 1.0
    candidate_stopping_temperature_m: float = 0.25
    candidate_stopping_mean_weight: float = 0.25
    candidate_stopping_cvar_weight: float = 0.50
    candidate_stopping_coverage_weight: float = 1.0
    candidate_stopping_cvar_fraction: float = 1.0 / 3.0
    candidate_stopping_minimum_candidates: int = 3

    @classmethod
    def from_mapping(cls, value):
        value = {} if value is None else dict(value)
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown projected_score_v3 keys: {unknown}")
        result = cls(**value)
        result.validate()
        return result

    @property
    def base_clearance_m(self):
        return self.vehicle_radius_m + self.tracking_margin_m

    def validate(self):
        positive = (
            self.max_speed_mps, self.max_acceleration_mps2,
            self.projection_min_scale, self.trajectory_samples,
            self.vehicle_radius_m, self.braking_acceleration_mps2,
            self.horizontal_fov_deg, self.vertical_fov_deg,
            self.minimum_path_length_m, self.preferred_goal_progress_m,
            self.clearance_saturation_m, self.unsafe_label_floor,
            self.ranking_margin, self.quality_pair_gap,
            self.target_temperature, self.prediction_temperature,
            self.candidate_trajectory_samples,
            self.candidate_stopping_temperature_m,
            self.candidate_stopping_cvar_fraction,
            self.candidate_stopping_minimum_candidates,
        )
        if min(positive) <= 0:
            raise ValueError("projected score positive parameters must be positive")
        if not 0 < self.projection_min_scale <= 1:
            raise ValueError("projection_min_scale must be in (0, 1]")
        if self.projection_steps < 2 or self.trajectory_samples < 3:
            raise ValueError("projected score sampling contract is too small")
        if self.tracking_margin_m < 0 or self.reaction_time_s < 0 \
                or self.clearance_sensor_tolerance_m < 0:
            raise ValueError("tracking margin/reaction time must be non-negative")
        if self.minimum_goal_progress_m < 0:
            raise ValueError("minimum goal progress must be non-negative")
        if self.preferred_goal_progress_m < self.minimum_goal_progress_m:
            raise ValueError("preferred progress must cover minimum progress")
        if 2 * self.visibility_margin_deg >= min(
            self.horizontal_fov_deg, self.vertical_fov_deg
        ):
            raise ValueError("visibility margin consumes the camera field of view")
        weights = (
            self.progress_weight, self.detour_weight, self.smoothness_weight,
            self.regression_weight, self.listwise_weight,
            self.safety_ranking_weight, self.quality_ranking_weight,
            self.candidate_stopping_loss_weight,
            self.candidate_stopping_mean_weight,
            self.candidate_stopping_cvar_weight,
            self.candidate_stopping_coverage_weight,
        )
        if min(weights) < 0:
            raise ValueError("projected score weights must be non-negative")
        if not 0 < self.candidate_stopping_cvar_fraction <= 1:
            raise ValueError("candidate stopping CVaR fraction must be in (0,1]")
        if self.candidate_stopping_minimum_candidates < 1:
            raise ValueError("candidate stopping coverage count must be positive")


def differentiable_stopping_distance_loss_v3(
    sampler, fixed_derivatives, predicted_derivatives, batch_size,
    candidate_count, safety_loss, map_id, config: ProjectedScoreConfigV3,
    static_distance_samples=None,
):
    """Teach proposals the same approach-speed stopping geometry as runtime.

    Score labels use the exact detached 81-sample runtime projection. This
    companion objective is intentionally differentiable and acts only on the
    candidate generator. It cannot change the hard 6/6 limits or create an
    additional readiness Gate.
    """
    if static_distance_samples is None:
        position, _, _, times = _sample_pva(
            sampler, fixed_derivatives, predicted_derivatives,
            config.candidate_trajectory_samples,
        )
        distance = safety_loss.get_distance_cost(
            position.reshape(batch_size, -1, 3), map_id
        )[1].reshape(batch_size, candidate_count, position.shape[1])
    else:
        if static_distance_samples.ndim != 2 \
                or static_distance_samples.shape[0] != batch_size * candidate_count:
            raise ValueError("static distance samples must be [B*N,T]")
        distance = static_distance_samples.reshape(
            batch_size, candidate_count, -1
        )
        times = torch.linspace(
            sampler.duration / distance.shape[2], sampler.duration,
            distance.shape[2], device=distance.device, dtype=distance.dtype,
        )
    dt = sampler.duration / max(1, distance.shape[2])
    closing = F.relu((distance[:, :, :-1] - distance[:, :, 1:]) / dt)
    closing = torch.cat((closing, closing[:, :, -1:]), dim=2)
    hard_floor = max(
        0.0, config.vehicle_radius_m - config.clearance_sensor_tolerance_m
    )
    escaping = distance[:, :, :1] < config.base_clearance_m
    baseline = torch.where(
        escaping & (distance < config.base_clearance_m),
        torch.full_like(distance, hard_floor),
        torch.full_like(distance, config.base_clearance_m),
    )
    required = (
        baseline + closing * config.reaction_time_s
        + closing.square() / (2.0 * config.braking_acceleration_mps2)
    )
    reserve = distance - required
    temperature = config.candidate_stopping_temperature_m
    normalized_deficit = (
        F.softplus(-reserve / temperature) * temperature
        / max(config.base_clearance_m, 1.0e-6)
    ).clamp(max=4.0)
    point_loss = normalized_deficit.square()
    candidate_loss = point_loss.mean(dim=2)
    top_count = max(
        1, int(candidate_count * config.candidate_stopping_cvar_fraction + 0.999999)
    )
    coverage_count = min(
        candidate_count, config.candidate_stopping_minimum_candidates
    )
    per_sample = (
        config.candidate_stopping_mean_weight * candidate_loss.mean(dim=1)
        + config.candidate_stopping_cvar_weight
        * candidate_loss.topk(top_count, dim=1, largest=True).values.mean(dim=1)
        + config.candidate_stopping_coverage_weight
        * candidate_loss.topk(
            coverage_count, dim=1, largest=False
        ).values.mean(dim=1)
    )
    minimum_reserve = reserve.amin(dim=2)
    return {
        "per_sample_loss": per_sample,
        "candidate_loss": candidate_loss,
        "minimum_stopping_reserve": minimum_reserve,
        "stoppable_candidate_count": minimum_reserve.ge(0.0).sum(dim=1),
        "sample_times": times,
    }


def _gather_scale(tensor, indices):
    """Gather one scale per flattened candidate from ``[M,S,...]``."""
    view = [tensor.shape[0], 1] + list(tensor.shape[2:])
    expand = [tensor.shape[0], 1] + list(tensor.shape[2:])
    gather_index = indices.view(tensor.shape[0], 1, *([1] * (tensor.ndim - 2)))
    gather_index = gather_index.expand(*expand)
    return tensor.gather(1, gather_index).squeeze(1)


def project_terminal_states_v3(
    sampler, fixed_derivatives, predicted_derivatives,
    batch_size, candidate_count, config: ProjectedScoreConfigV3,
):
    """Replay ``RuntimeTrajectorySafetyV1.project_endstate_candidates``.

    Inputs use the sampler's axis-major layout ``[B*N, xyz, pva]``.  The first
    hardware-feasible scale in the same descending linspace used by ROS is
    selected.  If none succeeds, the minimum scale is retained and explicitly
    marked unsuccessful, matching the runtime fallback object.
    """
    config.validate()
    expected = int(batch_size) * int(candidate_count)
    if fixed_derivatives.shape != (expected, 3, 3) \
            or predicted_derivatives.shape != (expected, 3, 3):
        raise ValueError("projected terminal derivatives must be [B*N,3,3]")
    device, dtype = predicted_derivatives.device, predicted_derivatives.dtype
    scales = torch.linspace(
        1.0, config.projection_min_scale, config.projection_steps,
        device=device, dtype=dtype,
    )
    scale = scales.view(1, -1, 1)
    start_position = fixed_derivatives[:, :, 0]
    terminal = predicted_derivatives
    projected = terminal[:, None, :, :].expand(
        -1, config.projection_steps, -1, -1
    ).clone()
    projected[:, :, :, 0] = (
        start_position[:, None, :]
        + scale * (terminal[:, None, :, 0] - start_position[:, None, :])
    )
    projected[:, :, :, 1] = scale * terminal[:, None, :, 1]
    projected[:, :, :, 2] = scale * terminal[:, None, :, 2]
    fixed = fixed_derivatives[:, None, :, :].expand_as(projected)
    _, velocity, acceleration, sample_times = _sample_pva(
        sampler, fixed.reshape(-1, 3, 3), projected.reshape(-1, 3, 3),
        config.trajectory_samples,
    )
    sample_total = velocity.shape[1]
    velocity = velocity.reshape(expected, config.projection_steps, sample_total, 3)
    acceleration = acceleration.reshape(
        expected, config.projection_steps, sample_total, 3
    )
    maximum_speed = velocity.norm(dim=3).amax(dim=2)
    maximum_acceleration = acceleration.norm(dim=3).amax(dim=2)
    feasible = (
        maximum_speed <= config.max_speed_mps + config.limit_tolerance
    ) & (
        maximum_acceleration
        <= config.max_acceleration_mps2 + config.limit_tolerance
    )
    succeeded = feasible.any(dim=1)
    first = feasible.to(torch.int64).argmax(dim=1)
    first = torch.where(
        succeeded, first, torch.full_like(first, config.projection_steps - 1)
    )
    selected_derivatives = _gather_scale(projected, first)
    selected_scale = scales[first]
    selected_maximum_speed = maximum_speed.gather(1, first[:, None]).squeeze(1)
    selected_maximum_acceleration = maximum_acceleration.gather(
        1, first[:, None]
    ).squeeze(1)
    position, selected_velocity, selected_acceleration, sample_times = (
        _sample_pva(
            sampler, fixed_derivatives, selected_derivatives,
            config.trajectory_samples,
        )
    )
    return {
        "projected_derivatives": selected_derivatives,
        "projection_scale": selected_scale.reshape(batch_size, candidate_count),
        "projection_succeeded": succeeded.reshape(batch_size, candidate_count),
        "maximum_speed": selected_maximum_speed.reshape(
            batch_size, candidate_count
        ),
        "maximum_acceleration": selected_maximum_acceleration.reshape(
            batch_size, candidate_count
        ),
        "position_world": position.reshape(
            batch_size, candidate_count, position.shape[1], 3
        ),
        "velocity_world": selected_velocity.reshape(
            batch_size, candidate_count, selected_velocity.shape[1], 3
        ),
        "acceleration_world": selected_acceleration.reshape(
            batch_size, candidate_count, selected_acceleration.shape[1], 3
        ),
        "sample_times": sample_times,
    }


def _mean_masked(values, mask):
    count = mask.sum((1, 2))
    result = (values * mask).sum((1, 2)) / count.clamp(min=1).to(values.dtype)
    return torch.where(count > 0, result, result * 0.0)


def projected_score_objective_v3(
    predicted_scores, projection, static_distance_samples,
    rotation_world_from_body, goal_body, projected_smoothness,
    config: ProjectedScoreConfigV3,
):
    """Build projected labels, losses and the two offline readiness metrics."""
    config.validate()
    batch_size, candidate_count = predicted_scores.shape
    position_world = projection["position_world"]
    velocity_world = projection["velocity_world"]
    expected = position_world.shape[:3]
    if static_distance_samples.shape != expected:
        raise ValueError("projected static distances must be [B,N,T]")
    if rotation_world_from_body.shape != (batch_size, 3, 3):
        raise ValueError("rotation_world_from_body must be [B,3,3]")
    if goal_body.shape != (batch_size, 3):
        raise ValueError("goal_body must be [B,3]")

    dt = float(projection["sample_times"][-1]) / max(
        1, position_world.shape[2] - 1
    )
    closing = F.relu(
        (static_distance_samples[:, :, :-1]
         - static_distance_samples[:, :, 1:]) / dt
    )
    closing = torch.cat((closing, closing[:, :, -1:]), dim=2)
    hard_floor = max(
        0.0, config.vehicle_radius_m - config.clearance_sensor_tolerance_m
    )
    initial_clearance = static_distance_samples[:, :, :1]
    escaping_from_margin = initial_clearance < config.base_clearance_m
    baseline_clearance = torch.where(
        escaping_from_margin
        & (static_distance_samples < config.base_clearance_m),
        torch.full_like(static_distance_samples, hard_floor),
        torch.full_like(static_distance_samples, config.base_clearance_m),
    )
    required_clearance = (
        baseline_clearance
        + closing * config.reaction_time_s
        + closing.square() / (2.0 * config.braking_acceleration_mps2)
    )
    stopping_reserve = static_distance_samples - required_clearance
    minimum_stopping_reserve = stopping_reserve.amin(dim=2)
    minimum_static_clearance = static_distance_samples.amin(dim=2)
    stoppable = minimum_stopping_reserve.ge(0.0)

    origin = position_world[:, :, :1, :]
    relative_world = position_world - origin
    relative_body = torch.einsum(
        "bntc,bcd->bntd", relative_world, rotation_world_from_body
    )
    visible_points = relative_body[:, :, 1:, :]
    forward = visible_points[..., 0]
    horizontal = torch.atan2(visible_points[..., 1], forward)
    vertical = torch.atan2(
        visible_points[..., 2],
        torch.linalg.vector_norm(visible_points[..., :2], dim=3),
    )
    horizontal_limit = torch.deg2rad(torch.tensor(
        0.5 * config.horizontal_fov_deg - config.visibility_margin_deg,
        device=predicted_scores.device, dtype=predicted_scores.dtype,
    ))
    vertical_limit = torch.deg2rad(torch.tensor(
        0.5 * config.vertical_fov_deg - config.visibility_margin_deg,
        device=predicted_scores.device, dtype=predicted_scores.dtype,
    ))
    visible = (
        (forward > 0.0)
        & (horizontal.abs() <= horizontal_limit)
        & (vertical.abs() <= vertical_limit)
    ).all(dim=2)

    delta_world = position_world[:, :, -1, :] - position_world[:, :, 0, :]
    endpoint_body = torch.einsum(
        "bnc,bcd->bnd", delta_world, rotation_world_from_body
    )
    segments = position_world[:, :, 1:, :] - position_world[:, :, :-1, :]
    path_length = segments.norm(dim=3).sum(dim=2)
    goal_direction = goal_body / goal_body.norm(
        dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    goal_progress = (
        endpoint_body * goal_direction[:, None, :]
    ).sum(dim=2)
    goal_alignment = goal_progress / endpoint_body.norm(dim=2).clamp_min(1.0e-6)
    actionable = (
        path_length >= config.minimum_path_length_m
    ) & (
        goal_progress >= config.minimum_goal_progress_m
    )
    safe = projection["projection_succeeded"] & stoppable & visible & actionable
    available = safe.any(dim=1)

    progress_deficit = F.relu(
        config.preferred_goal_progress_m - goal_progress
    ) / config.preferred_goal_progress_m
    detour = (
        path_length - goal_progress.clamp_min(0.0)
    ) / path_length.clamp_min(1.0e-6)
    smoothness = projected_smoothness.reshape(batch_size, candidate_count)
    smoothness = smoothness / smoothness.detach().mean(
        dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    safe_quality = (
        config.progress_weight * progress_deficit
        + config.detour_weight * detour
        + config.smoothness_weight * smoothness
    )
    # Extra clearance has bounded diagnostic value only.  Once the candidate
    # owns the physical stopping reserve, no larger detour is rewarded.
    saturated_reserve = minimum_stopping_reserve.clamp(
        min=0.0, max=config.clearance_saturation_m
    ) / config.clearance_saturation_m
    safe_quality = safe_quality + 0.05 * (1.0 - saturated_reserve)

    hardware_risk = (~projection["projection_succeeded"]).to(predicted_scores.dtype)
    visibility_risk = (~visible).to(predicted_scores.dtype)
    stopping_risk = F.relu(-minimum_stopping_reserve) / max(
        config.base_clearance_m, 1.0e-6
    )
    action_risk = (
        F.relu(config.minimum_path_length_m - path_length)
        / config.minimum_path_length_m
        + F.relu(config.minimum_goal_progress_m - goal_progress)
        / max(config.preferred_goal_progress_m, 1.0e-6)
    )
    unsafe_quality = config.unsafe_label_floor + (
        2.0 * hardware_risk + visibility_risk + stopping_risk + action_risk
    )
    labels = torch.where(safe, safe_quality, unsafe_quality).detach()

    regression = F.smooth_l1_loss(
        predicted_scores, labels, reduction="none"
    ).mean(dim=1)
    target_logits = -labels / config.target_temperature
    target_logits = torch.where(
        available[:, None] & (~safe),
        torch.full_like(target_logits, -1.0e4), target_logits,
    )
    target_probability = target_logits.softmax(dim=1).detach()
    prediction_log_probability = F.log_softmax(
        -predicted_scores / config.prediction_temperature, dim=1
    )
    listwise = -(target_probability * prediction_log_probability).sum(dim=1)

    pair_loss = F.relu(
        predicted_scores[:, :, None] - predicted_scores[:, None, :]
        + config.ranking_margin
    )
    safety_pair_mask = safe[:, :, None] & (~safe)[:, None, :]
    safety_ranking = _mean_masked(pair_loss, safety_pair_mask)
    quality_pair_mask = (
        safe[:, :, None] & safe[:, None, :]
        & (safe_quality[:, :, None] + config.quality_pair_gap
           < safe_quality[:, None, :])
    )
    quality_ranking = _mean_masked(pair_loss, quality_pair_mask)
    per_sample_loss = (
        config.regression_weight * regression
        + config.listwise_weight * listwise
        + config.safety_ranking_weight * safety_ranking
        + config.quality_ranking_weight * quality_ranking
    )

    selected = predicted_scores.argmin(dim=1)
    rows = torch.arange(batch_size, device=predicted_scores.device)
    selected_safe = safe[rows, selected]
    conditional_error = available & (~selected_safe)
    safe_labels = torch.where(safe, safe_quality, torch.full_like(safe_quality, 1.0e6))
    best_safe_quality = safe_labels.amin(dim=1)
    selected_quality = safe_quality[rows, selected]
    selection_regret = torch.where(
        available & selected_safe,
        F.relu(selected_quality - best_safe_quality),
        torch.zeros_like(selected_quality),
    )
    return {
        "labels": labels,
        "per_sample_loss": per_sample_loss,
        "regression_per_sample": regression,
        "listwise_per_sample": listwise,
        "safety_ranking_per_sample": safety_ranking,
        "quality_ranking_per_sample": quality_ranking,
        "ranking_per_sample": safety_ranking + quality_ranking,
        "safe_mask": safe,
        "safe_candidate_count": safe.sum(dim=1),
        "candidate_available": available,
        "conditional_selection_error": conditional_error,
        "selected_safe": selected_safe,
        "selected_hardware_unsafe": (
            ~projection["projection_succeeded"][rows, selected]
        ),
        "selected_goal_progress": goal_progress[rows, selected],
        "selected_goal_alignment": goal_alignment[rows, selected],
        "selected_stopping_reserve": minimum_stopping_reserve[rows, selected],
        "selected_static_clearance": minimum_static_clearance[rows, selected],
        "selected_visible": visible[rows, selected],
        "selected_projection_scale": projection["projection_scale"][rows, selected],
        "selection_regret": selection_regret,
        "minimum_stopping_reserve": minimum_stopping_reserve,
        "minimum_static_clearance": minimum_static_clearance,
        "required_clearance": required_clearance,
        "visible_mask": visible,
        "goal_progress": goal_progress,
        "goal_alignment": goal_alignment,
        "path_length": path_length,
    }


__all__ = [
    "ProjectedScoreConfigV3", "project_terminal_states_v3",
    "projected_score_objective_v3", "differentiable_stopping_distance_loss_v3",
]
