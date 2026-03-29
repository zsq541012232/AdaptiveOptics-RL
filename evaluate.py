from stable_baselines3 import SAC, A2C
from env_adapter import CustomEnvWrapper
import matplotlib.pyplot as plt
import numpy as np
import tqdm

experiment_name = "Sharpening_AO_system_easy"
model_names = ['A2C-1.7rms-21act-6', 'SAC-1.7rms-21act-20000buf-2', 'SAC-1.7rms-21act-50000buf-0', 'SAC-1.7rms-21act-100000buf-2']

eval_episodes = 10000
eval_steps = 100

# Create the Gym wrapper
env = CustomEnvWrapper(name=experiment_name)

# Evaluate the agent

plt.figure(figsize=(10,5))
bins = np.linspace(0, 1, 100)

for model_name in model_names:
    # load the model
    if "SAC" in model_name:
        model = SAC.load(f"models/{model_name}")
    elif "A2C" in model_name:
        model = A2C.load(f"models/{model_name}")
    print(f"Evaluating agent {model_name} on {experiment_name}...") 
    average_reward = []
    for episode in tqdm.tqdm(range(eval_episodes)):
        rewards = []
        obs, _ = env.reset()
        for step in range(eval_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # Combine terminated and truncated for your 'done' logic
            done = terminated or truncated
            # env.render()
            rewards.append(reward)

            if done:
                break

        # keep track of rewards
        average_reward.append(sum(rewards)/len(rewards))

    # histogram of rewards
    plt.hist(average_reward, bins=bins, label=f"{model_name}", alpha=0.5)

# no agent
print(f"Evaluating no agent on {experiment_name}...")
average_reward = []
for episode in tqdm.tqdm(range(eval_episodes)):
    rewards = []
    obs, _ = env.reset()
    for step in range(eval_steps):
        # Create a zero-action array matching the N_MODES
        action = np.zeros(env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break

    # keep track of rewards
    average_reward.append(sum(rewards)/len(rewards))

# histogram of rewards
plt.hist(average_reward, bins=bins, label="No Agent", alpha=0.5)

plt.xlabel("Average Reward")
plt.ylabel("Frequency")
plt.xlim(bins[0], bins[-1]) if bins[0] < bins[-1] else plt.xlim(bins[-1], bins[0])
plt.legend()
plt.savefig(f"figures/evaluation_{experiment_name}.png")
plt.close()





    
    
