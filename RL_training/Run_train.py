import os
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from droneEnv import DroneEnv

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
            print(f"\n💾 Model Saved: {self.save_path} | Steps: {self.num_timesteps}")
        return True

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on: {device}")

    # הגדרות אימון בסיסיות (אם אין לך קובץ config חיצוני)
    total_timesteps = 500000
    save_path = "output/drone_spatial_model"
    os.makedirs("output", exist_ok=True)

    env = DroneEnv()

    if os.path.exists(save_path + ".zip"):
        print(f"📂 Loading existing checkpoint...")
        model = PPO.load(save_path, env=env, device=device)
    else:
        print("🆕 Initializing new PPO policy...")
        model = PPO("MlpPolicy", env, verbose=1, device=device, tensorboard_log="./logs/")

    try:
        model.learn(total_timesteps=total_timesteps, callback=TrainingStatusCallback(total_timesteps, save_path))
    except KeyboardInterrupt:
        print("\n🛑 Saving and exiting...")
        model.save(save_path)
    finally:
        env.close()

if __name__ == "__main__":
    main()