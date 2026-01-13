import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from drone_env import DroneEnv


class PrintProgressCallback(BaseCallback):
    """
    Prints training progress % + last env info (match/dist/area/streak/lost).
    """
    def __init__(self, total_timesteps: int, print_every_steps: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.print_every_steps = int(print_every_steps)

    def _on_step(self) -> bool:
        # Progress %
        if self.n_calls % self.print_every_steps == 0:
            pct = 100.0 * (self.num_timesteps / max(1, self.total_timesteps))

            # Try to fetch info dict from vec env (DummyVecEnv => list of infos)
            infos = self.locals.get("infos", None)
            info0 = infos[0] if isinstance(infos, list) and len(infos) > 0 else {}

            # Compose a nice line
            ep = info0.get("episode_id", "?")
            st = info0.get("step_in_episode", "?")
            is_match = info0.get("is_match", None)
            dist = info0.get("dist", None)
            area = info0.get("area", None)
            streak = info0.get("focus_streak_s", None)
            lost = info0.get("lost_time_s", None)
            term = info0.get("termination_reason", "")

            parts = [f"[TRAIN] {self.num_timesteps}/{self.total_timesteps} ({pct:.1f}%)",
                     f"EP={ep} STEP={st}"]

            if is_match is not None:
                parts.append(f"MATCH={int(is_match)}")
            if dist is not None:
                parts.append(f"dist={dist:.3f}")
            if area is not None:
                parts.append(f"area={area:.4f}")
            if streak is not None:
                parts.append(f"streak={float(streak):.1f}s")
            if lost is not None:
                parts.append(f"lost={float(lost):.1f}s")
            if term:
                parts.append(f"term={term}")

            print(" | ".join(parts))

        return True


def main():
    models_dir = "models/PPO_Tracker"
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = DummyVecEnv([lambda: DroneEnv()])

    total_timesteps = 200000

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log=log_dir
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=models_dir,
        name_prefix="tracker_rl_model"
    )

    progress_callback = PrintProgressCallback(
        total_timesteps=total_timesteps,
        print_every_steps=2000
    )

    print("\n--- Training Started (with progress + detection prints) ---\n")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, progress_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[TRAIN] Interrupted. Saving progress...")

    final_path = os.path.join(models_dir, "ppo_tracker_final")
    model.save(final_path)
    print(f"[TRAIN] Model saved: {final_path}")


if __name__ == "__main__":
    main()
