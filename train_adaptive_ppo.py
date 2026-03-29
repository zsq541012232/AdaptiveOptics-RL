from __future__ import annotations

import argparse
import json
import os

import torch

from EnvironmentWrapper import CustomEnvWrapper
from adaptive_ppo import ActorCritic, AdaptivePPOTrainer, PPOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train Adaptive PPO on gym_sharpening with variable image sizes.")
    parser.add_argument("--env-name", default="Sharpening_AO_system", type=str)
    parser.add_argument("--total-timesteps", default=400_000, type=int)
    parser.add_argument("--rollout-steps", default=2048, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--latent-dim", default=512, type=int)
    parser.add_argument("--encoder-max-side", default=512, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", default="models/adaptive_ppo.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    env = CustomEnvWrapper(name=args.env_name, use_image_observation=True, render_mode=None)
    obs_channels = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = ActorCritic(
        obs_channels=obs_channels,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        max_side=args.encoder_max_side,
    )
    config = PPOConfig(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    trainer = AdaptivePPOTrainer(env=env, model=model, config=config, device=args.device)
    metrics = trainer.train()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_channels": obs_channels,
            "action_dim": action_dim,
            "latent_dim": args.latent_dim,
            "max_side": args.encoder_max_side,
            "metrics": metrics,
        },
        args.save_path,
    )

    print(json.dumps(metrics, indent=2))
    env.close()


if __name__ == "__main__":
    main()
