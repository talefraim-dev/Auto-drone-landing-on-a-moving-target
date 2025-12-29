import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from droneEnv import droneEnv


# --- Callback מותאם אישית להדפסות אינפורמטיביות ---
class InfoCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(InfoCallback, self).__init__(verbose)
        self.last_time = time.time()
        self.print_freq = 60  # שניות (הדפסה כל דקה)

    def _on_step(self) -> bool:
        # בדיקה אם עבר הזמן להדפסה
        if time.time() - self.last_time > self.print_freq:
            # שליפת נתונים מהלוגר של PPO
            fps = self.model.logger.name_to_value.get("train/fps", 0)
            iterations = self.model.logger.name_to_value.get("train/iterations", 0)
            entropy_loss = self.model.logger.name_to_value.get("train/entropy_loss", 0)
            value_loss = self.model.logger.name_to_value.get("train/value_loss", 0)

            print("\n" + "=" * 40)
            print(f"🕒 TIME-BASED UPDATE")
            print(f"🚀 Total Timesteps: {self.num_timesteps}")
            print(f"📊 Training FPS: {fps:.2f}")
            print(f"📉 Entropy Loss: {entropy_loss:.4f} (Exploration measure)")
            print(f"📉 Value Loss: {value_loss:.4f} (Prediction error)")

            # אם יש נתוני Reward מה-Monitor
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([info['r'] for info in self.model.ep_info_buffer])
                mean_len = np.mean([info['l'] for info in self.model.ep_info_buffer])
                print(f"🏆 Mean Reward (last 100 episodes): {mean_reward:.2f}")
                print(f"📏 Mean Episode Length: {mean_len:.1f} steps")

            print("=" * 40 + "\n")
            self.last_time = time.time()
        return True


def train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, "SiamMask", "experiments", "siammask_sharp")

    config_path = os.path.join(exp_dir, "config_davis.json")
    model_path = os.path.join(exp_dir, "SiamMask_VOT.pth")
    rl_model_save_path = "drone_land_v3.zip"

    # אתחול הסביבה
    env = droneEnv(model_path=model_path, config_path=config_path)
    env = Monitor(env)  # חייב Monitor כדי לקבל נתוני Reward ב-Callback
    env = DummyVecEnv([lambda: env])

    # טעינה או יצירה של המודל
    if os.path.exists(rl_model_save_path):
        print(f"--- Loading existing model: {rl_model_save_path} ---")
        model = PPO.load(rl_model_save_path, env=env, device="cuda")
    else:
        print("--- No existing model found. Starting from scratch ---")
        model = PPO("MlpPolicy", env, verbose=0, device="cuda", tensorboard_log="./ppo_drone_tensorboard/")

    # יצירת ה-Callback
    info_callback = InfoCallback()

    print("\n🚀 TRAINING STARTING. Press Ctrl+C to stop and save.")

    try:
        # שימוש ב-Callback בתוך model.learn
        model.learn(
            total_timesteps=50000,
            callback=info_callback,
            reset_num_timesteps=False,
            progress_bar=True  # מוסיף פס התקדמות ויזואלי בטרמינל
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted. Saving progress...")

    model.save(rl_model_save_path)
    print(f"✅ Model saved to {rl_model_save_path}")
    env.close()


if __name__ == '__main__':
    import numpy as np  # הוספתי כאן ליתר ביטחון עבור ה-Callback

    train()