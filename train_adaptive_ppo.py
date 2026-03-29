from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from env_adapter import CustomEnvWrapper
from adaptive_ppo import ActorCritic, AdaptivePPOTrainer, PPOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train Adaptive PPO on gym_sharpening with variable image sizes.")
    parser.add_argument("--env-name", default="Sharpening_AO_system", type=str)
    parser.add_argument("--total-timesteps", default=120_000, type=int)
    parser.add_argument("--rollout-steps", default=512, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=4, type=int)
    parser.add_argument("--latent-dim", default=256, type=int)
    parser.add_argument("--encoder-max-side", default=256, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", default="models/adaptive_ppo.pt")
    parser.add_argument("--tensorboard-log-dir", default="runs/adaptive_ppo", type=str)
    parser.add_argument("--run-name", default=None, type=str)
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

    run_name = args.run_name or f"ppo-{args.env_name}-{args.total_timesteps}steps"
    writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_log_dir, run_name))

    trainer = AdaptivePPOTrainer(env=env, model=model, config=config, device=args.device, writer=writer)
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
    print(f"TensorBoard logs saved to: {os.path.join(args.tensorboard_log_dir, run_name)}")

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
