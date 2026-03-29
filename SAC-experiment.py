from training_pipeline import default_config, run_training

project_name = "zsq541012232-wuhan-university/sharpening-ao-system"

config = default_config(algorithm="SAC", env_name="Sharpening_AO_system")
config.update(
    {
        "resnet_backbone": "resnet34",
        "use_cbam": True,
        "total_timesteps": 100_000,
        "buffer_size": 10_000,
    }
)

n_runs = 3
for run_idx in range(n_runs):
    run_training(
        config=config,
        wandb_project=project_name,
        group_name="SAC-resnet34-cbam-experiment",
    )
    print(f"Completed run {run_idx + 1}/{n_runs}")
