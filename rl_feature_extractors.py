from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel_size: int = 7):
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.channel_sigmoid = nn.Sigmoid()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel_size, padding=spatial_kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)
        channel_attention = self.channel_sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool))
        x = x * channel_attention

        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial = torch.amax(x, dim=1, keepdim=True)
        spatial_attention = self.spatial(torch.cat([avg_spatial, max_spatial], dim=1))
        return x * spatial_attention


class ResNetFeatureExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor using a configurable ResNet backbone and optional CBAM."""

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
        use_cbam: bool = False,
        cbam_reduction: int = 16,
        cbam_depth: int = 1,
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
            resnet.conv1 = nn.Conv2d(
                in_channels,
                resnet.conv1.out_channels,
                kernel_size=resnet.conv1.kernel_size,
                stride=resnet.conv1.stride,
                padding=resnet.conv1.padding,
                bias=False,
            )

        if use_cbam:
            resnet = self._inject_cbam(resnet, cbam_reduction, cbam_depth)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        backbone_out_dim = resnet.fc.in_features
        self.input_size = input_size

        if freeze_stem:
            for layer in (resnet.conv1, resnet.bn1):
                for parameter in layer.parameters():
                    parameter.requires_grad = False

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out_dim, features_dim),
            nn.ReLU(inplace=True),
        )

    @classmethod
    def _check_backbone(cls, backbone: str) -> None:
        if backbone not in cls._BACKBONES:
            supported = ", ".join(sorted(cls._BACKBONES))
            raise ValueError(f"Unsupported backbone '{backbone}'. Supported backbones: {supported}")

    @staticmethod
    def _inject_cbam(resnet: nn.Module, cbam_reduction: int, cbam_depth: int) -> nn.Module:
        if cbam_depth < 1:
            raise ValueError("cbam_depth must be >= 1.")
        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(resnet, layer_name)
            blocks: Iterable[nn.Module] = []
            for block in layer:
                out_channels = block.conv2.out_channels if hasattr(block, "conv2") else block.conv3.out_channels
                cbam_stack = [CBAMBlock(out_channels, reduction=cbam_reduction) for _ in range(cbam_depth)]
                blocks += [block, *cbam_stack]
            setattr(resnet, layer_name, nn.Sequential(*blocks))
        return resnet

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.input_size is not None and tuple(observations.shape[-2:]) != tuple(self.input_size):
            observations = F.interpolate(
                observations,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
        return self.projection(self.backbone(observations))
