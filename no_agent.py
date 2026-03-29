import numpy as np
from EnvironmentWrapper import CustomEnvWrapper
import tqdm
from torch.utils.tensorboard import SummaryWriter

experiment_name = "sharpening-ao-system-easy"  # options: sharpening-ao-system, sharpening-ao-system-easy, centering-ao-system, darkhole-ao-system

config = {
    "env_name": "Sharpening_AO_system_easy", # needs to change for each experiment corresponding to experiment_name
    # options: Sharpening_AO_system, Sharpening_AO_system_easy, Centering_AO_system, Darkhole_AO_system
}

def get_run_num(group_name):
    import os
    if not os.path.isdir("runs"):
        return 0
    return sum(1 for name in os.listdir("runs") if name.startswith(group_name))

print("Testing the environment with no agent")
# run the environment with no actions

env = CustomEnvWrapper(name=config["env_name"])
group_name = f"no_agent-{env.env.wf_rms}rms-{env.action_space.shape[0]}act"
run_num = get_run_num(group_name)


n_runs = 3
n_steps = 200000

for run in range(n_runs):
    print(f"Run {run+1}/{n_runs}")
    run_name = f"{group_name}-{run_num}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    env.reset()
    rewards = []
    for step in tqdm.tqdm(range(n_steps)):
        action = np.zeros(env.action_space.shape)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        writer.add_scalar("train/reward", reward, step)
        if terminated or truncated:
            break
    env.close()
    writer.flush()
    writer.close()
    run_num += 1

# get the average reward and the standard deviation
rewards = np.array(rewards)
print("Average reward: ", np.mean(rewards))
print("Standard deviation: ", np.std(rewards))
print("Min/Max reward: ", np.min(rewards), "/", np.max(rewards))
