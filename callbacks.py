from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCustomCallback(BaseCallback):
    def __init__(self, render_during_training=False, render_every_n_steps=1, verbose=0):
        super(TensorboardCustomCallback, self).__init__(verbose=verbose)
        self.render_during_training = render_during_training
        self.render_every_n_steps = max(1, int(render_every_n_steps))

    # get the reward every step
    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self.logger.record("train/reward", float(rewards.mean()))

        if self.render_during_training and self.num_timesteps % self.render_every_n_steps == 0:
            self.training_env.render()
        return True


class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self):
        # Append the reward to the list after each step
        self.rewards.append(self.locals["rewards"])
        return True
    def reset(self):
        self.rewards = []
