from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class MultiResolutionEncoder(nn.Module):
    """Encoder that accepts variable HxW images and maps them to fixed-size vectors."""

    def __init__(self, in_channels: int, latent_dim: int = 512, max_side: int = 512):
        super().__init__()
        self.max_side = max_side
        widths = [64, 128, 256]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, widths[0]),
            nn.SiLU(inplace=True),
        )

        self.stage1 = nn.Sequential(ResidualBlock(widths[0]), ResidualBlock(widths[0]))
        self.down1 = nn.Conv2d(widths[0], widths[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.stage2 = nn.Sequential(
            nn.GroupNorm(8, widths[1]),
            nn.SiLU(inplace=True),
            ResidualBlock(widths[1]),
            ResidualBlock(widths[1]),
        )
        self.down2 = nn.Conv2d(widths[1], widths[2], kernel_size=3, stride=2, padding=1, bias=False)
        self.stage3 = nn.Sequential(
            nn.GroupNorm(8, widths[2]),
            nn.SiLU(inplace=True),
            ResidualBlock(widths[2]),
            ResidualBlock(widths[2]),
        )

        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(widths[2], latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(inplace=True),
        )

    def _resize_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        max_hw = max(h, w)
        if max_hw <= self.max_side:
            return x

        scale = self.max_side / max_hw
        new_h = max(16, int(round(h * scale)))
        new_w = max(16, int(round(w * scale)))
        return F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resize_if_needed(x)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        return self.proj(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_channels: int, action_dim: int, latent_dim: int = 512, max_side: int = 512):
        super().__init__()
        self.encoder = MultiResolutionEncoder(obs_channels, latent_dim=latent_dim, max_side=max_side)
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(latent_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(latent_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def _dist_and_value(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        z = self.encoder(obs)
        mean = self.actor(z)
        std = self.log_std.exp().expand_as(mean)
        value = self.critic(z).squeeze(-1)
        return Normal(mean, std), value

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self._dist_and_value(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self._dist_and_value(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


@dataclass
class PPOConfig:
    total_timesteps: int = 120_000
    rollout_steps: int = 512
    batch_size: int = 128
    epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    lr: float = 1e-4
    max_grad_norm: float = 0.5
    action_clip: float = 0.3


class AdaptivePPOTrainer:
    def __init__(self, env, model: ActorCritic, config: PPOConfig, device: str = "cuda"):
        self.env = env
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr, fused=(device == "cuda"))
        self.scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    @staticmethod
    def _to_tensor_obs(obs: np.ndarray, device: str) -> torch.Tensor:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 2:
            obs = obs[None, ...]
        if obs.ndim == 3:
            obs = obs[None, ...]
        return torch.from_numpy(obs).to(device)

    def _gae(self, rewards, values, dones, next_value):
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * next_value * mask - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    def train(self) -> Dict[str, float]:
        obs, _ = self.env.reset()
        global_step = 0
        episode_rewards = []
        running_reward = 0.0

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        rollout_id = 0
        while global_step < self.config.total_timesteps:
            rollout_obs, rollout_actions = [], []
            rollout_logp, rollout_rewards, rollout_dones, rollout_values = [], [], [], []

            for _ in range(self.config.rollout_steps):
                obs_t = self._to_tensor_obs(obs, self.device)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                    action_t, logp_t, value_t = self.model.act(obs_t)

                action_np = action_t.squeeze(0).cpu().numpy()
                action_np = np.clip(action_np, -self.config.action_clip, self.config.action_clip)
                next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
                done = terminated or truncated

                rollout_obs.append(obs_t.squeeze(0))
                rollout_actions.append(action_t.squeeze(0))
                rollout_logp.append(logp_t.squeeze(0))
                rollout_rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
                rollout_dones.append(torch.tensor(float(done), dtype=torch.float32, device=self.device))
                rollout_values.append(value_t.squeeze(0))

                running_reward += reward
                obs = next_obs
                global_step += 1

                if done:
                    episode_rewards.append(running_reward)
                    running_reward = 0.0
                    obs, _ = self.env.reset()

                if global_step >= self.config.total_timesteps:
                    break

            with torch.no_grad():
                next_obs_t = self._to_tensor_obs(obs, self.device)
                _, next_value = self.model._dist_and_value(next_obs_t)
                next_value = next_value.squeeze(0)

            obs_b = torch.stack(rollout_obs)
            actions_b = torch.stack(rollout_actions)
            old_logp_b = torch.stack(rollout_logp)
            rewards_b = torch.stack(rollout_rewards)
            dones_b = torch.stack(rollout_dones)
            values_b = torch.stack(rollout_values)

            advantages_b, returns_b = self._gae(rewards_b, values_b, dones_b, next_value)
            advantages_b = (advantages_b - advantages_b.mean()) / (advantages_b.std() + 1e-8)

            n_samples = obs_b.shape[0]
            effective_batch_size = min(self.config.batch_size, n_samples)
            for _ in range(self.config.epochs):
                indices = torch.randperm(n_samples, device=self.device)
                for start in range(0, n_samples, effective_batch_size):
                    end = min(start + effective_batch_size, n_samples)
                    idx = indices[start:end]

                    mb_obs = obs_b[idx]
                    mb_actions = actions_b[idx]
                    mb_old_logp = old_logp_b[idx]
                    mb_adv = advantages_b[idx]
                    mb_ret = returns_b[idx]

                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                        new_logp, entropy, value = self.model.evaluate(mb_obs, mb_actions)
                        ratio = (new_logp - mb_old_logp).exp()
                        unclipped = ratio * mb_adv
                        clipped = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * mb_adv
                        policy_loss = -torch.min(unclipped, clipped).mean()
                        value_loss = F.mse_loss(value, mb_ret)
                        entropy_loss = -entropy.mean()
                        loss = policy_loss + self.config.value_coef * value_loss + self.config.entropy_coef * entropy_loss

                    if not torch.isfinite(loss):
                        raise RuntimeError("Detected non-finite PPO loss. Please lower learning rate or action range.")

                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            rollout_id += 1
            if rollout_id % 10 == 0:
                recent = episode_rewards[-10:] if episode_rewards else [running_reward]
                print(
                    f"[PPO] step={global_step}/{self.config.total_timesteps}, "
                    f"episodes={len(episode_rewards)}, mean_recent_reward={float(np.mean(recent)):.5f}"
                )

        return {
            "episodes": float(len(episode_rewards)),
            "mean_reward": float(np.mean(episode_rewards[-20:]) if episode_rewards else 0.0),
            "best_reward": float(np.max(episode_rewards) if episode_rewards else 0.0),
        }
