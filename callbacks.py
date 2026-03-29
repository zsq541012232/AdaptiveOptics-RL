from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCustomCallback(BaseCallback):
    def __init__(
        self,
        render_during_training=False,
        render_every_n_steps=1,
        print_every_n_steps=0,
        verbose=0,
    ):
        super(TensorboardCustomCallback, self).__init__(verbose=verbose)
        self.render_during_training = render_during_training
        self.render_every_n_steps = max(1, int(render_every_n_steps))
        self.print_every_n_steps = max(0, int(print_every_n_steps))

    # get the reward every step
    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            reward_value = float(rewards.mean())
            self.logger.record("train/reward", reward_value)
            if self.print_every_n_steps > 0 and self.num_timesteps % self.print_every_n_steps == 0:
                print(f"[step={self.num_timesteps}] reward={reward_value:.6f}")

        if self.render_during_training and self.num_timesteps % self.render_every_n_steps == 0:
            try:
                # VecEnv path
                self.training_env.env_method("render")
            except Exception:
                # Fallback path for non-VecEnv
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
