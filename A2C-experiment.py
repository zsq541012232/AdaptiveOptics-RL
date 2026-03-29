from training_pipeline import default_config, run_training

config = default_config(algorithm="A2C", env_name="Centering_AO_system")
config.update(
    {
        "use_image_observation": False,
        "policy_type": "MlpPolicy",
        "total_timesteps": 100_000,
    }
)

n_runs = 3
for run_idx in range(n_runs):
    run_training(
        config=config,
        group_name="A2C-centering-mlp-experiment",
    )
    print(f"Completed run {run_idx + 1}/{n_runs}")
