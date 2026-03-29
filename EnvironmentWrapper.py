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

    def render(self, mode='animation', **kwargs):
        actual_env = self.env.unwrapped
        if mode != 'animation':
            return

        # 1. 初始化：只创建一次图形对象
        if not hasattr(self, 'fig') or self.fig is None:
            plt.ion()
            # 增加 figsize 比例，减少重绘压力
            self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
            self.im1 = None
            self.im2 = None
            self.im3 = None
            self.line = None

        # 2. 图像更新逻辑 (避免使用 ax.cla() 和 hp.imshow_field)
        # 获取纯数据 (NumPy 数组)
        img_data = np.asarray(actual_env.image.shaped)
        dm_data = actual_env.deformable_mirror.phase_for(actual_env.wavelength) * actual_env.aperture
        dm_data = np.asarray(dm_data.shaped)

        # 图 1: Intensity
        if self.im1 is None:
            self.im1 = self.axes[0, 0].imshow(img_data, cmap='viridis', vmin=0)
            self.axes[0, 0].set_title('Intensity Image')
            self.fig.colorbar(self.im1, ax=self.axes[0, 0])
        else:
            self.im1.set_data(img_data)

        # 图 2: Log10
        log_img = np.log10(img_data + 1e-12)
        if self.im2 is None:
            self.im2 = self.axes[0, 1].imshow(log_img, vmax=0, vmin=-4, cmap='inferno')
            self.axes[0, 1].set_title('log10 Image')
            self.fig.colorbar(self.im2, ax=self.axes[0, 1])
        else:
            self.im2.set_data(log_img)

        # 图 3: DM Phase
        vmax = np.max(np.abs(dm_data)) + 1e-9
        if self.im3 is None:
            self.im3 = self.axes[1, 0].imshow(dm_data, cmap='bwr', vmin=-vmax, vmax=vmax)
            self.axes[1, 0].set_title('DM Phase')
            self.fig.colorbar(self.im3, ax=self.axes[1, 0])
        else:
            self.im3.set_data(dm_data)
            self.im3.set_clim(vmin=-vmax, vmax=vmax)

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

        # 更新标题
        self.fig.suptitle(
            f"Ep: {actual_env.episode} | Step: {actual_env.iteration} | Strehl: {actual_env.strehl * 100:.2f}%")

        # 3. 关键刷新指令
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
