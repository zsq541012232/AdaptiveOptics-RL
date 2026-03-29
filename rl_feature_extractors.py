from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models


class ResNetFeatureExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor using a configurable ResNet backbone."""

    _BACKBONES: Dict[str, callable] = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
    }
    _WEIGHT_ENUMS = {
        "resnet18": models.ResNet18_Weights,
        "resnet34": models.ResNet34_Weights,
        "resnet50": models.ResNet50_Weights,
        "resnet101": models.ResNet101_Weights,
    }

    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int = 256,
            backbone: str = "resnet18",
            pretrained: bool = False,
            freeze_stem: bool = False,
            input_size: Optional[Tuple[int, int]] = None,
    ):
        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 3:
            raise ValueError("ResNetFeatureExtractor requires a 3D Box observation space (C, H, W).")

        self._check_backbone(backbone)
        in_channels = observation_space.shape[0]

        super().__init__(observation_space, features_dim)

        weights = self._WEIGHT_ENUMS[backbone].DEFAULT if pretrained else None
        resnet = self._BACKBONES[backbone](weights=weights)

        if in_channels != 3:
            # 【优化 1】：调整 Stem 以适配小尺寸 PSF 图像
            resnet.conv1 = nn.Conv2d(
                in_channels,
                resnet.conv1.out_channels,
                kernel_size=3,  # 从 7 改为 3
                stride=1,  # 从 2 改为 1，避免过早丢失空间信息
                padding=1,
                bias=False,
            )
            # 移除降采样池化层，替换为恒等映射
            resnet.maxpool = nn.Identity()

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.input_size = input_size

        # 【核心修改 3】：增加自适应池化层，防止 Linear 层参数随输入尺寸爆炸
        # 无论输入多大，这里都将其压缩到 (1, 1) 的特征图
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        if freeze_stem:
            for layer in (resnet.conv1, resnet.bn1):
                for parameter in layer.parameters():
                    parameter.requires_grad = False

        # 计算展平维度
        with torch.no_grad():
            dummy_input_shape = input_size if input_size else observation_space.shape[1:]
            dummy_tensor = torch.zeros(1, in_channels, *dummy_input_shape)
            # 模拟前向传播，经过 backbone 和自适应池化
            out_tensor = self.adaptive_pool(self.backbone(dummy_tensor))
            flattened_dim = out_tensor.numel() // out_tensor.shape[0]

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(inplace=True),
        )

    @classmethod
    def _check_backbone(cls, backbone: str) -> None:
        if backbone not in cls._BACKBONES:
            supported = ", ".join(sorted(cls._BACKBONES))
            raise ValueError(f"Unsupported backbone '{backbone}'. Supported backbones: {supported}")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 插值缩放逻辑保持不变
        if self.input_size is not None and tuple(observations.shape[-2:]) != tuple(self.input_size):
            observations = F.interpolate(
                observations,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
        # 前向传播加入 adaptive_pool
        features = self.backbone(observations)
        features = self.adaptive_pool(features)
        return self.projection(features)



class SimpleCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    两层简单卷积神经网络特征提取器。
    适用于输入尺寸较小且不需要 ResNet 这种深层网络的情况。
    """

    def __init__(
            self,
            observation_space: gym.spaces.Box,
            features_dim: int = 256,
            n_filters: int = 32,
    ):
        super().__init__(observation_space, features_dim)

        in_channels = observation_space.shape[0]

        # 定义简单的两层卷积
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 使用自适应池化，确保输入尺寸变化时代码不崩
            nn.Flatten(),
        )

        # 计算卷积后的输出维度并定义投影层
        # 对于 (n_filters * 2) 的输出（经过 AdaptiveAvgPool2d((1, 1)) 后）
        self.linear = nn.Sequential(
            nn.Linear(n_filters * 2, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
