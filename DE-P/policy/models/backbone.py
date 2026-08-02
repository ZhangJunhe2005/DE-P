import time
import torch
import numpy as np
import torch.nn as nn
import warnings
from typing import List, Callable, Optional
from policy.models.MobileNetV3 import MobileNetV3, InvertedResidualConfig, InvertedResidual, SqueezeExcitation, ConvBNActivation
from policy.backbone_variant import resolve_backbone_variant

# 辅助函数：确保通道数为8的倍数
def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 确保向下调整不超过10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# MobileNet V3 Small配置参数（适配输入尺寸[1, 96, 160]）
def mobilenet_v3_small_config(width_multi: float = 1.0) -> List[InvertedResidualConfig]:
    return [
        InvertedResidualConfig(16, 3, 16, 16, True, "RE", 2, width_multi),
        InvertedResidualConfig(16, 3, 72, 24, False, "RE", 2, width_multi),
        InvertedResidualConfig(24, 3, 88, 24, False, "RE", 1, width_multi),
        InvertedResidualConfig(24, 5, 96, 40, True, "HS", 2, width_multi),
        InvertedResidualConfig(40, 5, 240, 40, True, "HS", 1, width_multi),
        InvertedResidualConfig(40, 5, 240, 40, True, "HS", 1, width_multi),
        InvertedResidualConfig(40, 3, 120, 48, True, "HS", 1, width_multi),
        InvertedResidualConfig(48, 3, 144, 48, True, "HS", 1, width_multi),
        InvertedResidualConfig(48, 5, 288, 96, True, "HS", 2, width_multi),
        InvertedResidualConfig(96, 5, 576, 96, True, "HS", 1, width_multi),
        InvertedResidualConfig(96, 5, 576, 96, True, "HS", 1, width_multi),
    ]

class DepBackbone(nn.Module):
    def __init__(self, output_dim: int, input_size: tuple = (96, 160), backbone_variant=None):
        super(DepBackbone, self).__init__()
        self.output_dim = output_dim
        self.input_size = input_size
        self.backbone_variant = resolve_backbone_variant(backbone_variant)
        
        # 根据输入尺寸选择不同的MobileNet版本
        if input_size == (32, 64):
            self.backbone = self._build_small_backbone()
        else:
            self.backbone = self._build_standard_backbone()

    def _build_standard_backbone(self) -> nn.Module:
        """构建标准尺寸输入的MobileNet V3 Backbone"""
        width_multi = 1.0
        inverted_residual_setting = mobilenet_v3_small_config(width_multi)
        last_channel = _make_divisible(1280 * width_multi, 8)
        
        # 初始化MobileNet V3
        cnn = MobileNetV3(
            inverted_residual_setting=inverted_residual_setting,
            last_channel=last_channel,
            num_classes=1000,
            block=InvertedResidual
        )
        
        self._replace_stem(cnn)
        
        # 移除原分类器
        cnn.classifier = nn.Identity()
        
        # 计算特征通道数
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *self.input_size)
            feat = cnn.features(dummy_input)
            feat_channels = feat.shape[1]
        
        # 输出卷积层（补全括号）
        output_layer = nn.Conv2d(
            feat_channels,
            self.output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # 构建完整的backbone序列
        return nn.Sequential(
            cnn.features,
            output_layer
        )

    def _build_small_backbone(self) -> nn.Module:
        """构建小尺寸输入的MobileNet V3 Backbone（补充完整实现）"""
        width_multi = 0.75
        inverted_residual_setting = mobilenet_v3_small_config(width_multi)[:-3]  # 减少blocks数量
        last_channel = _make_divisible(1280 * width_multi, 8)
        
        cnn = MobileNetV3(
            inverted_residual_setting=inverted_residual_setting,
            last_channel=last_channel,
            num_classes=1000,
            block=InvertedResidual
        )
        
        self._replace_stem(cnn)
        
        # 移除原分类器
        cnn.classifier = nn.Identity()
        
        # 计算特征通道数
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *self.input_size)
            feat = cnn.features(dummy_input)
            feat_channels = feat.shape[1]
        
        # 输出卷积层
        output_layer = nn.Conv2d(
            feat_channels,
            self.output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        return nn.Sequential(
            cnn.features,
            output_layer
        )

    def _replace_stem(self, cnn: MobileNetV3) -> None:
        """Build either the historical nested stem or the corrected standard stem."""
        original_stem = cnn.features[0]
        original_conv = original_stem[0]
        stem_kwargs = dict(
            in_planes=1,
            out_planes=original_conv.out_channels,
            kernel_size=original_conv.kernel_size[0],
            stride=original_conv.stride[0],
            activation_layer=nn.Hardswish,
        )
        if self.backbone_variant == "legacy":
            # Preserve the historical nested module and every checkpoint key exactly.
            original_stem[0] = ConvBNActivation(
                **stem_kwargs,
                norm_layer=nn.BatchNorm2d,
            )
        else:
            # Match the MobileNetV3 stem BN settings used by the remaining network.
            norm_layer = lambda channels: nn.BatchNorm2d(
                channels, eps=0.001, momentum=0.01
            )
            cnn.features[0] = ConvBNActivation(
                **stem_kwargs,
                norm_layer=norm_layer,
            )

    def forward(self, x: torch.Tensor, attention: Optional[torch.Tensor] = None,
                attention_alpha: float = 1.0) -> torch.Tensor:
        """Extract ``[B,64,3,5]`` features and optionally modulate them.

        Attention is generated upstream.  Fusion is parameter-free, bounded,
        and occurs after the MobileNet/output convolution and before state
        concatenation: ``feature * (1 + alpha * attention)``.
        """
        feature = self.backbone(x)
        if attention is None:
            return feature
        if attention.ndim != 4 or attention.shape[1] != 1:
            raise ValueError("attention must have shape [B,1,H,W]")
        if attention.shape[0] == 1 and feature.shape[0] != 1:
            attention = attention.expand(feature.shape[0], -1, -1, -1)
        elif attention.shape[0] != feature.shape[0]:
            raise ValueError("attention batch size must equal depth batch size (or be 1)")
        attention = attention.to(device=feature.device, dtype=feature.dtype)
        if attention.shape[-2:] != feature.shape[-2:]:
            attention = torch.nn.functional.interpolate(
                attention, size=feature.shape[-2:], mode="bilinear", align_corners=False
            )
        if not bool(torch.isfinite(attention).all()):
            raise FloatingPointError("attention contains NaN/Inf")
        attention = attention.clamp(0.0, 1.0)
        return feature * (1.0 + float(attention_alpha) * attention)

class AttentionDepBackbone(nn.Module):
    def __init__(self, output_dim: int, input_size: tuple = (96, 160)):
        super(AttentionDepBackbone, self).__init__()
        warnings.warn(
            "AttentionDepBackbone is an unintegrated experimental compatibility class; "
            "stage-4 dynamic perception runs independently from DepNetwork",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            from policy.models.EkfDynPercept import DynamicObstacleAttention
            from policy.models.MonteCarloCutting import PointCloudProcessor
        except ImportError as exc:
            raise ImportError(
                "AttentionDepBackbone requires optional dynamic dependencies; "
                "install the packages documented in requirements-dynamic.txt"
            ) from exc
        self.output_dim = output_dim
        self.input_size = input_size
        
        # 构建特征提取层（拆分结构以便插入注意力）
        self._build_feature_layers()
        
        # 根据输入尺寸确定注意力模块参数
        attention_feature_size = self._get_attention_feature_size()
        from config.config import cfg
        dynamic_cfg = cfg["dynamic_perception"]
        self.obstacle_attention = DynamicObstacleAttention(
            input_feature_size=attention_feature_size,
            cluster_eps=dynamic_cfg["cluster_eps"],
            cluster_min_samples=dynamic_cfg["cluster_min_samples"],
            cluster_iterations=dynamic_cfg["cluster_max_iterations"],
        )
        
        # 输出卷积层
        self.output_layer = self._build_output_layer()
        
        # 初始化点云处理器
        self.point_cloud_processor = PointCloudProcessor(fov_angle=120, min_distance=0.5, max_distance=10.0)

    def _build_feature_layers(self):
        """构建可拆分的特征提取层，用于插入注意力模块"""
        if self.input_size == (32, 64):
            width_multi = 0.75
            inverted_residual_setting = mobilenet_v3_small_config(width_multi)[:-3]  # 精简配置
        else:
            width_multi = 1.0
            inverted_residual_setting = mobilenet_v3_small_config(width_multi)
            
        # 初始化特征提取层列表
        features = []
        # 第一层：输入卷积（适配单通道深度图）
        first_conv_out = inverted_residual_setting[0].input_c  # 从配置获取第一层输出通道
        first_conv = ConvBNActivation(
            in_planes=1,  # 单通道深度图
            out_planes=first_conv_out,
            kernel_size=3,
            stride=2,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.Hardswish
        )
        features.append(first_conv)
        
        # 添加反向残差块（注意：InvertedResidual不接受activation_layer参数）
        for cnf in inverted_residual_setting:
            block = InvertedResidual(
                cnf=cnf,
                norm_layer=nn.BatchNorm2d  # 只传递必要参数
            )
            features.append(block)
        
        # 拆分特征层：在第7个模块后插入注意力（确保索引有效）
        split_idx = min(7, len(features) - 1)  # 避免索引越界
        self.features_before_attention = nn.Sequential(*features[:split_idx])
        self.features_after_attention = nn.Sequential(*features[split_idx:])

    def _get_attention_feature_size(self) -> tuple:
        """计算注意力模块输入特征图的尺寸"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *self.input_size, device=self.device)
            feat = self.features_before_attention(dummy_input)
            return (feat.shape[2], feat.shape[3])  # (H, W)

    def _build_output_layer(self) -> nn.Module:
        """构建输出卷积层"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *self.input_size, device=self.device)
            feat = self.features_before_attention(dummy_input)
            feat = self.features_after_attention(feat)
            feat_channels = feat.shape[1]
        
        return nn.Conv2d(
            feat_channels,
            self.output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    @property
    def device(self):
        """获取模型所在设备"""
        return next(self.parameters()).device

    def forward(self, depth: torch.Tensor, pcl: np.ndarray) -> torch.Tensor:
        """
        前向传播
        Args:
            depth: 深度图像张量，形状为[B, 1, H, W]
            pcl: 点云数据，形状为[N, 3]的numpy数组
        Returns:
            融合注意力的特征图
        """
        # 1. 预处理点云：裁剪和聚类
        clipped_pcl = self.point_cloud_processor.clip_region(pcl)
        
        # 2. 提取基础特征（注意力前）
        feat = self.features_before_attention(depth)
        
        # 3. 点云转换为张量并与特征图设备对齐
        pcl_tensor = torch.from_numpy(clipped_pcl).float().to(self.device)
        
        # 4. 应用动态障碍物注意力
        attended_feat = self.obstacle_attention(pcl_tensor, feat)
        
        # 5. 继续特征提取（注意力后）
        feat = self.features_after_attention(attended_feat)
        
        # 6. 输出特征转换
        output = self.output_layer(feat)
        return output


# 测试代码
if __name__ == '__main__':
    # 记录开始时间
    total_start_time = time.time()
    
    # 测试标准输入尺寸[1, 96, 160]
    print("测试标准尺寸输入 (96, 160):")
    start_time = time.time()
    net = AttentionDepBackbone(output_dim=64, input_size=(96, 160))
    model_init_time = time.time() - start_time
    
    input_depth = torch.zeros((1, 1, 96, 160))
    
    # 预热（运行几次以确保所有层都已初始化）
    with torch.no_grad():
        for _ in range(3):
            _ = net(input_depth, np.zeros((100, 3)))
    
    # 正式测试推理时间
    start_time = time.time()
    with torch.no_grad():
        output = net(input_depth, np.zeros((100, 3)))
    inference_time = time.time() - start_time
    
    print(f"输出形状: {output.shape}")
    print(f"模型初始化时间: {model_init_time:.4f}s")
    print(f"推理时间: {inference_time:.4f}s")
    print()
    
    # 测试小尺寸输入[1, 32, 64]
    print("测试小尺寸输入 (32, 64):")
    start_time = time.time()
    net_small = AttentionDepBackbone(output_dim=64, input_size=(32, 64))
    model_small_init_time = time.time() - start_time
    
    input_small = torch.zeros((1, 1, 32, 64))
    
    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = net_small(input_small, np.zeros((100, 3)))
    
    # 正式测试推理时间
    start_time = time.time()
    with torch.no_grad():
        output_small = net_small(input_small, np.zeros((100, 3)))
    inference_small_time = time.time() - start_time
    
    print(f"输出形状: {output_small.shape}")
    print(f"模型初始化时间: {model_small_init_time:.4f}s")
    print(f"推理时间: {inference_small_time:.4f}s")
    print()
    
    
