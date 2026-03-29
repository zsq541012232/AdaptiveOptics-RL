from training_pipeline import default_config, run_training

config = default_config(algorithm="SAC", env_name="Sharpening_AO_system_easy")
config.update(
    {
        "use_simple_cnn": False,
        "resnet_backbone": "resnet34",
        "use_pretrained_resnet": False,
        "resnet_input_size": (16, 16),
        "features_dim": 256 // 4,
        "render_during_training": False,  # set True to watch training live
        "render_every_n_steps": 100,
        "total_timesteps": 100_000,
        "buffer_size": 30_000,
    }
)

run_training(
    config=config,
    group_name="SAC-resnet34",
)
