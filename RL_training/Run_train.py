import torch
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from droneEnv import AirSimDroneEnv
import train_land


class TrainingStatusCallback(BaseCallback):
    def __init__(self, total_steps, save_path, verbose=0):
        super(TrainingStatusCallback, self).__init__(verbose)
        self.total_steps = total_steps
        self.save_path = save_path
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # פעם ב-4096 צעדים (סוף rollout) נדפיס סטטוס ונשמור
        if self.n_calls % 4096 == 0:
            # שמירה אוטומטית
            self.model.save(self.save_path)

            elapsed_time = time.time() - self.start_time
            progress = self.num_timesteps / self.total_steps
            if progress > 0:
                total_est_time = elapsed_time / progress
                remaining_time = total_est_time - elapsed_time

                print(f"\n" + "=" * 40)
                print(f"💾 AUTOSAVE: Model saved to {self.save_path}")
                print(f"📊 PROGRESS: {progress * 100:.1f}%")
                print(f"⏱️ ELAPSED: {elapsed_time / 60:.1f} min")
                print(f"⏳ REMAINING: {remaining_time / 60:.1f} min")
                print(f"🚀 STEPS: {self.num_timesteps}/{self.total_steps}")
                print("=" * 40 + "\n")
        return True


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = train_land.get_config()
    if not os.path.exists('output'): os.makedirs('output')

    env = AirSimDroneEnv(config=config, reward_fn=train_land.compute_reward)
    env = DummyVecEnv([lambda: env])

    total_steps = config['total_timesteps']
    save_path = f"output/{config['model_name']}"

    # בדיקה אם קיים מודל קודם וטעינה שלו
    if os.path.exists(save_path + ".zip"):
        print(f"📂 Found existing model, loading: {save_path}")
        model = PPO.load(save_path, env=env, device=device)
    else:
        print("🆕 No existing model found. Starting training from scratch.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=0.0003,
            n_steps=4096,
            batch_size=128,
            ent_coef=0.02,
            device=device,
            tensorboard_log="./logs/"
        )

    status_callback = TrainingStatusCallback(total_steps, save_path)

    print(f"🚀 Training started! Estimated total steps: {total_steps}")
    try:
        model.learn(
            total_timesteps=total_steps,
            callback=status_callback,
            tb_log_name="PPO_With_Terminal_Stats"
        )
        model.save(save_path)
        print("✅ Training Complete and Model Saved!")
    except KeyboardInterrupt:
        model.save(save_path)
        print(f"\n🛑 Training Interrupted. Progress saved to {save_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()