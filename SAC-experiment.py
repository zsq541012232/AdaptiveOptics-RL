from training_pipeline import default_config, run_training

config = default_config(algorithm="SAC", env_name="Sharpening_AO_system")
config.update(
    {
        "resnet_backbone": "resnet34",
        "use_pretrained_resnet": True,
        "resnet_input_size": (224, 224),
        "total_timesteps": 100_000,
        "buffer_size": 10_000,
    }
)

n_runs = 3
for run_idx in range(n_runs):
    run_training(
        config=config,
        group_name="SAC-resnet34-pretrained-experiment",
    )
    print(f"Completed run {run_idx + 1}/{n_runs}")
