# smoothness_loss.py（恢复 snap 相关逻辑）
import torch.nn as nn
import torch as th


class SmoothnessLoss(nn.Module):
    def __init__(self, R, R_snap, snap_weight=0.1):  # 恢复 R_snap 和 snap_weight 参数
        super().__init__()
        self._R = R  # jerk 矩阵
        self._R_snap = R_snap  # snap 矩阵
        self.snap_weight = snap_weight  # snap 损失权重

    def forward(self, Df, Dp):
        batch_size = Dp.shape[0]
        # 扩展矩阵以匹配批次大小
        R = self._R.to(device=Dp.device, dtype=Dp.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        R_snap = self._R_snap.to(device=Dp.device, dtype=Dp.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 合并固定参数和决策参数
        D_all = th.cat([Df, Dp], dim=2)
        
        # 分别计算 jerk 和 snap 损失
        jerk_costs = []
        snap_costs = []
        
        for i in range(3):  # 对 x, y, z 三个轴分别计算
            d_axis = D_all[:, i].unsqueeze(2)  # 提取当前轴的参数
            jerk_cost = d_axis.transpose(1, 2) @ R @ d_axis  # jerk 损失
            snap_cost = d_axis.transpose(1, 2) @ R_snap @ d_axis  # snap 损失
            
            jerk_costs.append(jerk_cost)
            snap_costs.append(snap_cost)
        
        # 求和并加权合并
        total_jerk = sum(jerk_costs).squeeze()
        total_snap = sum(snap_costs).squeeze()
        return total_jerk + self.snap_weight * total_snap  # 合并损失
