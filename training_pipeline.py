from __future__ import annotations

import os
from typing import Any, Dict

import torch as th
import wandb
from stable_baselines3 import A2C, SAC

from callbacks import WandbCustomCallback
from EnvironmentWrapper import CustomEnvWrapper
from rl_feature_extractors import ResNetFeatureExtractor

ALGORITHMS = {
    "A2C": A2C,
    "SAC": SAC,
}


def get_run_num(runs, group_name: str) -> int:
    return sum(1 for run in runs if group_name in run.name)


def default_config(algorithm: str, env_name: str) -> Dict[str, Any]:
    return {
        "algorithm": algorithm,
        "env_name": env_name,
        "policy_type": "CnnPolicy",
        "use_image_observation": True,
        "resnet_backbone": "resnet18",
        "use_cbam": False,
        "features_dim": 256,
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
    )

    if config["policy_type"] == "CnnPolicy":
        common_kwargs["policy_kwargs"] = dict(
            features_extractor_class=ResNetFeatureExtractor,
            features_extractor_kwargs=dict(
                backbone=config["resnet_backbone"],
                use_cbam=config["use_cbam"],
                features_dim=config["features_dim"],
            ),
        )

    if algorithm_name == "SAC":
        return SAC(buffer_size=config["buffer_size"], **common_kwargs)
    return A2C(**common_kwargs)


def setup_wandb(config: Dict[str, Any], group_name: str, project: str):
    # os.environ.setdefault("WANDB_MODE", "offline")
    api = wandb.Api(api_key="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")
    runs = api.runs(project)
    run_num = get_run_num(runs, group_name)
    run = wandb.init(
        group=group_name,
        name=f"{group_name}-{run_num}",
        project=project.split("/")[-1],
        entity=project.split("/")[0],
        config=config,
        sync_tensorboard=True,
    )
    return run, run_num


def run_training(config: Dict[str, Any], wandb_project: str, group_name: str):
    env = CustomEnvWrapper(name=config["env_name"], use_image_observation=config["use_image_observation"])
    run, run_num = setup_wandb(config=config, group_name=group_name, project=wandb_project)
    model = build_model(config=config, env=env)
    model.learn(total_timesteps=config["total_timesteps"], callback=WandbCustomCallback(), progress_bar=True)
    model.save(f"models/{group_name}-{run_num}")
    wandb.finish()
    env.close()
    return run
