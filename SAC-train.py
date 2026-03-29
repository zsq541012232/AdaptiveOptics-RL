from training_pipeline import default_config, run_training

config = default_config(algorithm="SAC", env_name="Sharpening_AO_system_easy")
config.update(
    {
        "resnet_backbone": "resnet34",
        "use_cbam": True,
        "cbam_depth": 2,
        "use_pretrained_resnet": True,
        "resnet_input_size": (16, 16),
        "render_during_training": True,  # set True to watch training live
        "render_every_n_steps": 1,
        "total_timesteps": 100_000,
        "buffer_size": 20_000,
    }
)

run_training(
    config=config,
    group_name="SAC-resnet34-cbamx2-pretrained",
)
