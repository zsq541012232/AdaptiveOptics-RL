from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models


class GeMPool2d(nn.Module):
    """Generalized mean pooling, often more expressive than avg pooling."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(self.p, min=1.0, max=8.0)
        x = torch.clamp(x, min=self.eps).pow(p)
        return F.adaptive_avg_pool2d(x, output_size=(1, 1)).pow(1.0 / p)


class ChannelSE(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            ChannelSE(channels),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ResNetFeatureExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor using a ResNet backbone + stronger projection head."""

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
        features_dim: int = 128,
        backbone: str = "resnet18",
        pretrained: bool = False,
        freeze_stem: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
        head_dropout: float = 0.1,
    ):
        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 3:
            raise ValueError("ResNetFeatureExtractor requires a 3D Box observation space (C, H, W).")

        self._check_backbone(backbone)
        in_channels = observation_space.shape[0]

        super().__init__(observation_space, features_dim)

        weights = self._WEIGHT_ENUMS[backbone].DEFAULT if pretrained else None
        resnet = self._BACKBONES[backbone](weights=weights)

        # Better for AO PSF-like inputs: smaller kernel + no aggressive early downsampling.
        resnet.conv1 = nn.Conv2d(
            in_channels,
            resnet.conv1.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        resnet.maxpool = nn.Identity()

        if freeze_stem:
            for layer in (resnet.conv1, resnet.bn1):
                for parameter in layer.parameters():
                    parameter.requires_grad = False

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.input_size = input_size
        self.gem_pool = GeMPool2d(p=3.0)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        with torch.no_grad():
            dummy_input_shape = input_size if input_size else observation_space.shape[1:]
            dummy_tensor = torch.zeros(1, in_channels, *dummy_input_shape)
            feat = self.backbone(dummy_tensor)
            pooled = torch.cat(
                [
                    self.gem_pool(feat),
                    self.avg_pool(feat),
                    self.max_pool(feat),
                ],
                dim=1,
            )
            flattened_dim = pooled.numel() // pooled.shape[0]

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, features_dim * 2),
            nn.LayerNorm(features_dim * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Linear(features_dim * 2, features_dim),
            nn.LayerNorm(features_dim),
            nn.SiLU(inplace=True),
        )

    @classmethod
    def _check_backbone(cls, backbone: str) -> None:
        if backbone not in cls._BACKBONES:
            supported = ", ".join(sorted(cls._BACKBONES))
            raise ValueError(f"Unsupported backbone '{backbone}'. Supported backbones: {supported}")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.input_size is not None and tuple(observations.shape[-2:]) != tuple(self.input_size):
            observations = F.interpolate(
                observations,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
        features = self.backbone(observations)
        pooled = torch.cat(
            [
                self.gem_pool(features),
                self.avg_pool(features),
                self.max_pool(features),
            ],
            dim=1,
        )
        return self.projection(pooled)


class SimpleCNNFeatureExtractor(BaseFeaturesExtractor):
    """A stronger lightweight CNN extractor for SAC/A2C on AO images."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        width: int = 48,
        head_dropout: float = 0.05,
    ):
        super().__init__(observation_space, features_dim)

        in_channels = observation_space.shape[0]
        w1, w2, w3 = width, width * 2, width * 4

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, w1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, w1),
            nn.SiLU(inplace=True),
        )
        self.stage1 = ConvResidualBlock(w1, dropout=0.03)
        self.down1 = nn.Conv2d(w1, w2, kernel_size=3, stride=2, padding=1, bias=False)
        self.stage2 = nn.Sequential(
            nn.GroupNorm(8, w2),
            nn.SiLU(inplace=True),
            ConvResidualBlock(w2, dropout=0.05),
        )
        self.down2 = nn.Conv2d(w2, w3, kernel_size=3, stride=2, padding=1, bias=False)
        self.stage3 = nn.Sequential(
            nn.GroupNorm(8, w3),
            nn.SiLU(inplace=True),
            ConvResidualBlock(w3, dropout=0.08),
        )

        self.gem_pool = GeMPool2d(p=3.0)
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w3 * 2, features_dim * 2),
            nn.LayerNorm(features_dim * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(features_dim * 2, features_dim),
            nn.LayerNorm(features_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.stem(observations)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)

        pooled = torch.cat([self.gem_pool(x), self.max_pool(x)], dim=1)
        return self.head(pooled)
