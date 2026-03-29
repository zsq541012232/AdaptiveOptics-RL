from training_pipeline import default_config, run_training

config = default_config(algorithm="SAC", env_name="Sharpening_AO_system")
config.update(
    {
        "resnet_backbone": "resnet34",
        "use_cbam": True,
        "total_timesteps": 10_000,
        "buffer_size": 2_000,
    }
)

run_training(
    config=config,
    wandb_project="zsq541012232-wuhan-university/sharpening-ao-system",
    group_name="SAC-resnet34-cbam",
)
