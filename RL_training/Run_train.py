import os
import time
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from droneEnv import DroneEnv
import train_land

class TrainingStatusCallback(BaseCallback):
    def __init__(self, total_steps, save_path):
        super().__init__()
        self.total_steps = int(total_steps)
        self.save_path = save_path
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % 4096 == 0:
            self.model.save(self.save_path)
            elapsed = (time.time() - self.start_time) / 60
            progress = self.num_timesteps / self.total_steps
            # הדפסת סטטוס אימון לקונסול
            print(f"\n--- 📈 Training Progress: {progress:.1%} ---")
            print(f"⏱️ Elapsed Time: {elapsed:.1f} mins")
            print(f"💾 Model Saved at: {self.save_path}\n")
        return True

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Environment ready. Hardware: {device}")

    config = train_land.get_config()
    os.makedirs("output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = DroneEnv()
    save_path = f"output/{config['model_name']}"

    if os.path.exists(save_path + ".zip"):
        print(f"📂 Loading existing checkpoint...")
        model = PPO.load(save_path, env=env, device=device)
    else:
        print("🆕 Initializing new PPO policy (MlpPolicy)...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1, # זה ידפיס את טבלת ה-Loss וה-Explained Variance
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            device=device,
            tensorboard_log="./logs/"
        )

    try:
        print("🎬 STARTING TRAINING. Watch both console and CV window.")
        model.learn(
            total_timesteps=int(config["total_timesteps"]),
            callback=TrainingStatusCallback(config["total_timesteps"], save_path)
        )
        model.save(save_path)
    except KeyboardInterrupt:
        print("\n🛑 Manual Stop. Saving progress...")
        model.save(save_path)
    finally:
        env.close()

if __name__ == "__main__":
    main()