import matplotlib
# 使用 Qt5Agg 或 TkAgg 后端，这会强制弹出独立窗口
matplotlib.use('TkAgg')
import gymnasium as gym
import numpy as np
from gym_ao.gym_sharpening import Sharpening_AO_system
from gym_ao.gym_centering import Centering_AO_system
from gym_ao.gym_sharpening_easy import Sharpening_AO_system_easy
from gym_ao.gym_darkhole import Darkhole_AO_system
import hcipy as hp
import matplotlib.pyplot as plt
import os
from gymnasium.wrappers import TimeLimit


class CustomEnvWrapper(gym.Env):
    def __init__(self, name, use_image_observation=False, render_mode=None):
        self.use_image_observation = use_image_observation
        self.render_mode = render_mode

        if name == "Sharpening_AO_system":
            self.env = Sharpening_AO_system()
        elif name == "Sharpening_AO_system_easy":
            self.env = Sharpening_AO_system_easy()
        elif name == "Centering_AO_system":
            self.env = Centering_AO_system()
        elif name == "Darkhole_AO_system":  # needs fixing for observation space
            self.env = Darkhole_AO_system()
        else:
            raise ValueError(f"Invalid environment name: {name}")

        self.env = TimeLimit(self.env, max_episode_steps=100)

        self.action_space = gym.spaces.Box(
            low=-0.3,
            high=0.3,
            shape=(self.env.unwrapped.num_modes,),
            dtype=np.float32,
        )

        self._base_observation_shape = self.env.observation_space.shape
        if self.use_image_observation and len(self._base_observation_shape) == 2:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1.0,
                shape=(1, *self._base_observation_shape),
                dtype=np.float32,
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1.0,
                shape=self._base_observation_shape,
                dtype=np.float32,
            )

    def _format_observation(self, observation):
        if isinstance(observation, hp.Field):
            observation = np.asarray(observation, dtype=np.float32)
        else:
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
        # 关键点：解包返回的 (observation, info)
        # self.env 现在是被 TimeLimit 包装过的，它会返回两个值
        observation, info = self.env.reset(seed=seed, options=options)

        # 只将真正的 observation 传给格式化函数
        formatted_obs = self._format_observation(observation)

        # 返回格式化后的 obs 和原有的 info 字典
        return formatted_obs, info

    def render(self, mode='animation', episode=None, iteration=None, tot_rewards=None, loc='test'):
        # 核心修复：获取被 TimeLimit 包装的原始环境实例
        actual_env = self.env.unwrapped

        if mode == 'animation':
            if not plt.isinteractive():
                plt.ion()
            if not hasattr(self, 'fig'):
                self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 10))

            for ax in self.axes.ravel():
                ax.cla()

            # ---------------------------------------------------------
            # 1. 左上：Focal Plane Image
            # ---------------------------------------------------------
            plt.sca(self.axes[0, 0])
            plt.axis('off')
            plt.title('Intensity Image')
            # 修复：使用实际环境中的 image 属性
            im1 = hp.imshow_field(actual_env.image, cmap='viridis', vmin=0)
            if hasattr(self, 'cbar1'):
                self.cbar1.update_normal(im1)
            else:
                self.cbar1 = plt.colorbar(im1)

            # ---------------------------------------------------------
            # 2. 右上：Log10 Image
            # ---------------------------------------------------------
            plt.sca(self.axes[0, 1])
            # 修复：使用实际环境中的 image 属性
            im2 = hp.imshow_field(np.log10(actual_env.image), vmax=0, vmin=-4, cmap='inferno')
            plt.axis('off')
            plt.title('log10 Image')
            if hasattr(self, 'cbar2'):
                self.cbar2.update_normal(im2)
            else:
                self.cbar2 = plt.colorbar(im2)

            # ---------------------------------------------------------
            # 3. 左下：Mirror Shape
            # ---------------------------------------------------------
            plt.sca(self.axes[1, 0])
            # 修复：从实际环境中获取 DM 相位和孔径
            dm_phase = actual_env.deformable_mirror.phase_for(actual_env.wavelength) * actual_env.aperture
            vmax = np.max(np.abs(dm_phase)) if np.max(np.abs(dm_phase)) > 0 else 0.1
            im3 = hp.imshow_field(dm_phase, cmap='bwr', vmin=-vmax, vmax=vmax)
            plt.axis('off')
            plt.title('DM Phase')
            if hasattr(self, 'cbar3'):
                self.cbar3.update_normal(im3)
            else:
                self.cbar3 = plt.colorbar(im3)

            # ---------------------------------------------------------
            # 4. 右下：Average Reward Plot
            # ---------------------------------------------------------
            plt.sca(self.axes[1, 1])
            # 修复：检查实际环境中的历史奖励记录
            if hasattr(actual_env, 'avg_rewards') and len(actual_env.avg_rewards) > 0:
                plt.plot(actual_env.avg_rewards, marker='.', color='blue', linestyle='-')
                plt.title('History: Avg Reward per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Mean Reward')
                plt.grid(True, alpha=0.3)

            # ---------------------------------------------------------
            # 5. 顶部总标题：显示 Strehl, Episode, Steps
            # ---------------------------------------------------------
            # 修复：获取实际环境的状态变量
            strehl_val = actual_env.strehl * 100
            curr_ep = actual_env.episode
            curr_step = actual_env.iteration

            self.fig.suptitle(
                f"Episode: {curr_ep} | Step: {curr_step} | Strehl: {strehl_val:.2f}%",
                fontsize=16, fontweight='bold', y=0.95
            )

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
