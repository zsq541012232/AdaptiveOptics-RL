import os
import numpy as np
from helper import LearningCurvePlot, smooth
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# check if the reward_data folder exists
os.makedirs('reward_data', exist_ok=True)
os.makedirs('figures/learning_curves', exist_ok=True)


def _load_reward_from_tensorboard(run_dir: str):
    event_acc = EventAccumulator(run_dir)
    event_acc.Reload()
    if "train/reward" not in event_acc.Tags().get("scalars", []):
        return []
    return [event.value for event in event_acc.Scalars("train/reward")]


def plot_learning_curves(groups, length, ylim=(0, 1), name='test', types='png', dpi=400, runs_dir='runs'):
    plot = LearningCurvePlot(y_lim=ylim, length=length)

    run_names = os.listdir(runs_dir) if os.path.isdir(runs_dir) else []

    for group in groups:
        print(f'Working on {group}')
        rewards = []
        matching_runs = [run_name for run_name in run_names if run_name.startswith(group)]

        for run_name in matching_runs:
            reward_file = f'reward_data/{run_name}.npy'
            try:
                r = np.load(reward_file)
                if len(r) < length - 1:
                    print(f'WARNING: {reward_file} has only {len(r)} entries, but {length} are required')
                else:
                    rewards.append(smooth(r[:length - 1], length // 20))
            except Exception:
                print(f'Fetching {run_name}')
                reward_data = _load_reward_from_tensorboard(os.path.join(runs_dir, run_name))
                np.save(reward_file, reward_data)
                rewards.append(smooth(reward_data[:length - 1], length // 20))

        label = f'{group.split("-")[0]} {group.split("-")[-1]}' if group.split("-")[0] == 'SAC' else f'{group.split("-")[0].replace("_", " ")}'
        plot.add_curve(rewards, label)

    plot.save(f'figures/learning_curves/{name}.{types}', dpi=dpi)


if __name__ == '__main__':
    groups = ['SAC-1.7rms-3act-1000buf', 'SAC-1.7rms-3act-10000buf', 'SAC-1.7rms-3act-20000buf', 'SAC-1.7rms-3act-100buf', 'A2C-1.7rms-3act', 'no_agent-1.7rms-3act']
    length = 100000
    plot_learning_curves(groups, length, name='easy_2zer', types='pdf', dpi=900)
