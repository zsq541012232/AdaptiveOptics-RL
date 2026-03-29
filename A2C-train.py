from training_pipeline import default_config, run_training

config = default_config(algorithm="A2C", env_name="Sharpening_AO_system")
config.update(
    {
        "resnet_backbone": "resnet18",
        "use_cbam": False,
        "total_timesteps": 10_000,
    }
)

run_training(
    config=config,
    wandb_project="adapt_opt/sharpening-ao-system",
    group_name="A2C-resnet18",
)
