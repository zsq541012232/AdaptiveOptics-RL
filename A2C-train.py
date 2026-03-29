from training_pipeline import default_config, run_training

config = default_config(algorithm="A2C", env_name="Sharpening_AO_system")
config.update(
    {
        "resnet_backbone": "resnet18",
        "total_timesteps": 10_000,
    }
)

run_training(
    config=config,
    group_name="A2C-resnet18",
)
