from __future__ import annotations

from typing import Dict

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit

from gym_sharpening import Sharpening_AO_system


_ENV_REGISTRY: Dict[str, type[gym.Env]] = {
    "Sharpening_AO_system": Sharpening_AO_system,
    # Backward-compatible aliases used by existing scripts/configs.
    "Sharpening_AO_system_easy": Sharpening_AO_system,
}


class CustomEnvWrapper(gym.Env):
    def __init__(self, name: str, use_image_observation: bool = False, render_mode: str | None = None, max_episode_steps: int = 100):
        self.use_image_observation = use_image_observation
        self.render_mode = render_mode

        if name not in _ENV_REGISTRY:
            raise ValueError(f"Invalid environment name: {name}. Available: {sorted(_ENV_REGISTRY.keys())}")

        base_env = _ENV_REGISTRY[name]()
        self.env = TimeLimit(base_env, max_episode_steps=max_episode_steps)

        self.action_space = gym.spaces.Box(
            low=-0.3,
            high=0.3,
            shape=(self.env.unwrapped.num_modes,),
            dtype=np.float32,
        )

        self._base_observation_shape = self.env.observation_space.shape
        if self.use_image_observation and len(self._base_observation_shape) == 2:
            obs_shape = (1, *self._base_observation_shape)
        else:
            obs_shape = self._base_observation_shape

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)

    def _format_observation(self, observation) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)

        if self.use_image_observation and observation.ndim == 1 and len(self._base_observation_shape) == 2:
            expected_size = int(np.prod(self._base_observation_shape))
            if observation.size == expected_size:
                observation = observation.reshape(self._base_observation_shape)

        if self.use_image_observation and observation.ndim == 2:
            return observation[None, ...]
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._format_observation(observation), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return self._format_observation(observation), info

    def render(self, *args, **kwargs):
        if hasattr(self.env, "render"):
            return self.env.render(*args, **kwargs)
        return None

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
