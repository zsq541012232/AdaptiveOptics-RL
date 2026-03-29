from __future__ import annotations

import os
from typing import Any, Dict

import torch as th
from stable_baselines3 import A2C, SAC

from callbacks import TensorboardCustomCallback
from EnvironmentWrapper import CustomEnvWrapper
from rl_feature_extractors import ResNetFeatureExtractor, SimpleCNNFeatureExtractor

ALGORITHMS = {
    "A2C": A2C,
    "SAC": SAC,
}


def get_run_num(group_name: str, models_dir: str = "models") -> int:
    if not os.path.isdir(models_dir):
        return 0
    return sum(1 for run_name in os.listdir(models_dir) if run_name.startswith(f"{group_name}-"))


def default_config(algorithm: str, env_name: str) -> Dict[str, Any]:
    return {
        "algorithm": algorithm,
        "env_name": env_name,
        "policy_type": "CnnPolicy",
        "use_image_observation": True,
        "resnet_backbone": "resnet18",
        "use_pretrained_resnet": False,
        "resnet_input_size": None,
        "features_dim": 256 // 4,
        "tensorboard_log_dir": "runs",
        "render_during_training": True,
        "render_every_n_steps": 200,
        "total_timesteps": 2_000,
        "buffer_size": 10_000,
        "device": "cuda" if th.cuda.is_available() else "cpu",
    }


def build_model(config: Dict[str, Any], env: CustomEnvWrapper):
    algorithm_name = config["algorithm"]
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm '{algorithm_name}'.")

    common_kwargs = dict(
        policy=config["policy_type"],
        env=env,
        verbose=1,
        device=config["device"],
        tensorboard_log=config.get("tensorboard_log_dir", "runs"),
    )

    if config["policy_type"] == "CnnPolicy":
        # 根据配置选择类
        if config.get("use_simple_cnn", False):
            extractor_class = SimpleCNNFeatureExtractor
            extractor_kwargs = dict(features_dim=config["features_dim"])
        else:
            extractor_class = ResNetFeatureExtractor
            extractor_kwargs = dict(
                backbone=config["resnet_backbone"],
                pretrained=config["use_pretrained_resnet"],
                input_size=config.get("resnet_input_size"),
                features_dim=config["features_dim"],
            )

        common_kwargs["policy_kwargs"] = dict(
            features_extractor_class=extractor_class,
            features_extractor_kwargs=extractor_kwargs,
            share_features_extractor=False,
        )

    if algorithm_name == "SAC":
        return SAC(buffer_size=config["buffer_size"], **common_kwargs)
    return A2C(**common_kwargs)


def run_training(config: Dict[str, Any], group_name: str):
    env = CustomEnvWrapper(name=config["env_name"],
                           use_image_observation=config["use_image_observation"],
                           render_mode="animation" if config.get("render_during_training") else None)
    run_num = get_run_num(group_name=group_name)
    model = build_model(config=config, env=env)
    callback = TensorboardCustomCallback(
        render_during_training=config.get("render_during_training", False),
        render_every_n_steps=config.get("render_every_n_steps", 1),
    )
    run_name = f"{group_name}-{run_num}"
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback,
        progress_bar=True,
        tb_log_name=run_name,
    )
    model.save(f"models/{group_name}-{run_num}")
    env.close()
    return run_name
