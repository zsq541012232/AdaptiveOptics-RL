import os
import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from gym_ao.gym_ao.gym_sharpening import Sharpening_AO_system
import wandb
from callbacks import WandbCustomCallback
from EnvironmentWrapper import CustomEnvWrapper

# 强制进入离线模式
os.environ["WANDB_MODE"] = "offline"
# Set up Weights and Biases

config = {
    "policy_type": "MlpPolicy",
    "env_name": "Sharpening_AO_system"
}


api = wandb.Api()

runs = api.runs("zsq541012232-wuhan-university/sharpening-ao-system")

group_name = "SAC-test"

run_num = 0
for run in runs:
    if group_name in run.name:
        run_num += 1

run = wandb.init(
    group=group_name,
    name=f"SAC-test-run-{run_num}",
    project="sharpening-ao-system",
    # entity="zsq541012232-wuhan-university",
    config=config,
    sync_tensorboard=True,
    # settings=wandb.Settings(start_method="thread")
)


# class CustomEnvWrapper(gym.Env):
#     def __init__(self):
#         # Initialize your Sharpening_AO_system environment here
#         self.env = Sharpening_AO_system()
#         self.action_space = gym.spaces.Box(low=-0.3, high=0.3, shape=(400,), dtype=np.float32)
#         self.observation_space = gym.spaces.Box(low=0, high=1., shape=self.env.observation_space.shape, dtype=np.float32)
#
#     def step(self, action):
#         observation, reward, done, trunc, info = self.env.step(action)
#         if done:
#             observation = self.reset()
#         if trunc:
#             observation = self.reset()
#         return observation, reward, done, info
#
#     def reset(self, seed=None, options=None):
#         # 1. Handle the seed if the underlying env doesn't support it
#         if seed is not None:
#             self.env.seed(seed)  # Older Gym envs use .seed() instead of reset(seed=)
#
#         # 2. Call the old reset (which likely returns only 'observation')
#         observation = self.env.reset()
#
#         # 3. Gymnasium/SB3 expects (observation, info)
#         # We return an empty dict for 'info' to satisfy the new API
#         return observation, {}
#
#     def render(self, mode='human'):
#         self.env.render()

# Create the Gym wrapper
# env = CustomEnvWrapper()
# Create the Gym wrapper
env = CustomEnvWrapper(name=config["env_name"])

# Create and train the SAC model and sync with wandb
model = SAC("MlpPolicy", env, verbose=1, buffer_size=10)
model.learn(total_timesteps=110, callback=WandbCustomCallback(), progress_bar=True)

# Close the environment
env.close()

