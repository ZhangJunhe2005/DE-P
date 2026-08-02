import math
import torch as th
import torch.nn as nn
from dataclasses import dataclass
from config.config import cfg
from loss.safety_loss import SafetyLoss
from loss.smoothness_loss import SmoothnessLoss
from loss.guidance_loss import GuidanceLoss
from loss.dynamic_safety_loss import DynamicCollisionLoss
from loss.dynamic_types import DynamicLossConfig, DynamicObjectiveConfig, RiskMetricsConfig


@dataclass(frozen=True)
class DEPLossOutput:
    smooth_cost: th.Tensor
    static_safety_cost: th.Tensor
    guidance_cost: th.Tensor
    dynamic_safety_cost: th.Tensor
    raw_dynamic_safety_cost: th.Tensor
    dynamic_training_objective: th.Tensor
    dynamic_diagnostics: object
    static_min_distance: th.Tensor | None = None

    @property
    def trajectory_cost(self):
        return (self.smooth_cost + self.static_safety_cost
                + self.guidance_cost + self.dynamic_safety_cost)

    def detached_score_label(self):
        return self.trajectory_cost.detach()

    @property
    def trajectory_training_loss(self):
        non_dynamic = (
            self.smooth_cost + self.static_safety_cost + self.guidance_cost
        ).mean()
        return non_dynamic + self.dynamic_training_objective


class DEPLoss(nn.Module):
    def __init__(self, dynamic_loss_config=None, map_catalog=None,
                 dynamic_objective_config=None, risk_metrics_config=None):
        super(DEPLoss, self).__init__()
        self.sgm_time = cfg["sgm_time"]
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        # 生成 jerk 相关矩阵 R 和 snap 相关矩阵 R_snap
        self._C, self._B, self._L, self._R, self._R_snap = self.qp_generation()  # 新增 R_snap
        self._R = self._R.to(self.device)
        self._R_snap = self._R_snap.to(self.device)  # 新增：将 R_snap 移至设备
        self._L = self._L.to(self.device)
        vel_scale = cfg["vel_max_train"] / 1.0
        self.smoothness_weight = cfg["ws"]
        self.safety_weight = cfg["wc"]
        self.goal_weight = cfg["wg"]
        self.denormalize_weight(vel_scale)
        # 传递 R_snap 给 SmoothnessLoss，并可配置 snap 权重（从配置文件读取或固定）
        self.smoothness_loss = SmoothnessLoss(
        self._R, 
        self._R_snap, 
        snap_weight=getattr(cfg, "snap_weight", 0.1)  # 使用 getattr 兼容类属性访问
                )         # 恢复 R_snap 参数
        self.safety_loss = SafetyLoss(self._L, map_catalog=map_catalog)
        self.goal_loss = GuidanceLoss()
        self.dynamic_loss_config = dynamic_loss_config or DynamicLossConfig.from_global_config()
        self.dynamic_objective_config = (
            dynamic_objective_config or DynamicObjectiveConfig.from_global_config()
        )
        self.risk_metrics_config = risk_metrics_config or RiskMetricsConfig.from_global_config()
        self.dynamic_safety_loss = DynamicCollisionLoss(
            self.safety_loss.trajectory_sampler, self.dynamic_loss_config,
            self.risk_metrics_config,
        )
        print("------ Actual Loss ------")
        print(f"| {'smooth':<12} = {self.smoothness_weight:6.4f} |")
        print(f"| {'safety':<12} = {self.safety_weight:6.4f} |")
        print(f"| {'goal':<12} = {self.goal_weight:6.4f} |")
        print(f"| {'full ratio':<12} = {self.safety_loss.safety_full_ratio:6.4f} |")
        print(f"| {'OOB cost':<12} = {self.safety_loss.out_of_bounds_cost:6.4f} |")
        print(f"| {'guide perp':<12} = {self.goal_loss.perp_weight_max:6.4f} |")
        print(f"| {'dynamic':<12} = {self.dynamic_loss_config.weight:6.4f} |")
        print(f"| {'dynamic on':<12} = {str(self.dynamic_loss_config.enabled):>6} |")
        print("-------------------------")

    def qp_generation(self):
        A = th.zeros((6, 6))
        for i in range(3):
            A[2 * i, i] = math.factorial(i)
            for j in range(i, 6):
                A[2 * i + 1, j] = math.factorial(j) / math.factorial(j - i) * (self.sgm_time ** (j - i))

        # Jerk 相关的二次型矩阵 H（3-5 阶导数，对应加加速度）
        H_jerk = th.zeros((6, 6))
        for i in range(3, 6):
            for j in range(3, 6):
                H_jerk[i, j] = i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2) / (i + j - 5) * (self.sgm_time ** (i + j - 5))

        # Snap 相关的二次型矩阵 H_snap（4-5 阶导数，对应急动度）
        H_snap = th.zeros((6, 6))
        for i in range(4, 6):  # snap 是 4 阶导数，因此从 4 开始
            for j in range(4, 6):
                H_snap[i, j] = i * (i - 1) * (i - 2) * (i - 3) * j * (j - 1) * (j - 2) * (j - 3) / (i + j - 7) * (self.sgm_time ** (i + j - 7))

        # 生成 C, B, L, R（jerk）和 R_snap（snap）
        _C, _B, _L, _R = self.stack_opt_dep(A, H_jerk)
        _, _, _, _R_snap = self.stack_opt_dep(A, H_snap)  # 用 H_snap 生成 R_snap
        return _C, _B, _L, _R, _R_snap  # 返回 R_snap

    def stack_opt_dep(self, A, Q):
        Ct = th.zeros((6, 6))
        Ct[[0, 2, 4, 1, 3, 5], [0, 1, 2, 3, 4, 5]] = 1

        _C = th.transpose(Ct, 0, 1)
        B = th.inverse(A)
        B_T = th.transpose(B, 0, 1)
        _L = B @ Ct
        _R = _C @ (B_T) @ Q @ B @ Ct  # 基于输入的 Q（H_jerk 或 H_snap）生成对应的 R
        return _C, B, _L, _R

    def denormalize_weight(self, vel_scale):
        self.smoothness_weight = self.smoothness_weight / vel_scale ** 5
        self.safety_weight = self.safety_weight * vel_scale
        self.goal_weight = self.goal_weight

    def forward(self, state, prediction, goal, map_id, dynamic_obstacles=None,
                return_details=False):
        Df = state.permute(0, 2, 1)
        Dp = prediction.permute(0, 2, 1)

        input_device, input_dtype = prediction.device, prediction.dtype
        smoothness_cost = th.tensor(0.0, device=input_device, dtype=input_dtype, requires_grad=True)
        safety_cost = th.tensor(0.0, device=input_device, dtype=input_dtype, requires_grad=True)
        goal_cost = th.tensor(0.0, device=input_device, dtype=input_dtype, requires_grad=True)

        if self.smoothness_weight > 0:
            smoothness_cost = self.smoothness_loss(Df, Dp)
        if self.safety_weight > 0:
            safety_cost, static_min_distance = self.safety_loss(
                Df, Dp, map_id, return_min_distance=True
            )
        else:
            static_min_distance = th.full(
                (prediction.shape[0],), float("inf"),
                device=input_device, dtype=input_dtype,
            )
        if self.goal_weight > 0:
            goal_cost = self.goal_loss(Df, Dp, goal)
        weighted_smooth = self.smoothness_weight * smoothness_cost
        weighted_static = self.safety_weight * safety_cost
        weighted_goal = self.goal_weight * goal_cost

        dynamic_cost = weighted_smooth * 0.0
        raw_dynamic_cost = weighted_smooth * 0.0
        dynamic_training_objective = weighted_smooth.mean() * 0.0
        dynamic_diagnostics = None
        if self.dynamic_loss_config.enabled and dynamic_obstacles is not None:
            dynamic_obstacles = dynamic_obstacles.to(Df.device)
            dynamic_grouped, dynamic_diagnostics = self.dynamic_safety_loss(
                Df, Dp, dynamic_obstacles
            )
            raw_dynamic_cost = dynamic_grouped.reshape(-1)
            dynamic_cost = self.dynamic_loss_config.weight * raw_dynamic_cost
            fraction = self.dynamic_objective_config.cvar_fraction
            top_count = max(1, math.ceil(dynamic_grouped.shape[1] * fraction))
            cvar = dynamic_grouped.topk(top_count, dim=1, largest=True).values.mean()
            objective = (
                self.dynamic_objective_config.mean_coefficient * dynamic_grouped.mean()
                + self.dynamic_objective_config.cvar_coefficient * cvar
            ) / self.dynamic_objective_config.reference_scale
            dynamic_training_objective = self.dynamic_loss_config.weight * objective

        if return_details:
            return DEPLossOutput(
                smooth_cost=weighted_smooth,
                static_safety_cost=weighted_static,
                guidance_cost=weighted_goal,
                dynamic_safety_cost=dynamic_cost,
                raw_dynamic_safety_cost=raw_dynamic_cost,
                dynamic_training_objective=dynamic_training_objective,
                dynamic_diagnostics=dynamic_diagnostics,
                static_min_distance=static_min_distance,
            )
        # Preserve the phase-2A positional tuple for every existing static caller.
        return weighted_smooth, weighted_static, weighted_goal
