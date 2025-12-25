import torch
import os
from stable_baselines3 import PPO
from droneEnv import AirSimDroneEnv
import train_land
import time


def main():
    # 1. הגדרת הנתיב המדויק לקובץ שמצאת
    # שים לב לשימוש ב-r לפני המחרוזת כדי למנוע בעיות עם לוכסנים ב-Windows
    model_path = r"C:\Users\Tal Efraim\PycharmProjects\Auto-drone-landing-on-a-moving-target\output\output\ppo_drone_final_run.zip"

    # 2. טעינת הקונפיגורציה המקורית
    config = train_land.get_config()

    print(f"📂 Loading final model from: {model_path}")

    if not os.path.exists(model_path):
        print("❌ Error: The file path does not exist. Please check the path again.")
        return

    # 3. יצירת הסביבה
    # אנחנו משתמשים באותה סביבה ובאותה פונקציית reward כדי שהתצפיות יהיו זהות
    env = AirSimDroneEnv(config=config, reward_fn=train_land.compute_reward)

    try:
        # 4. טעינת המודל המאומן
        model = PPO.load(model_path)
        print("✅ Model loaded successfully! Starting demonstration...")

        for episode in range(1, 6):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            step_count = 0

            print(f"\n🚀 Episode {episode} Start")

            while not done:
                # predict נותן את הפעולה הכי טובה שהרשת העצבית מציעה
                action, _states = model.predict(obs, deterministic=True)

                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1

                # חישוב מרחק מהתצפית (dx, dy, dz הם שלושת האיברים הראשונים)
                dist = (obs[0] ** 2 + obs[1] ** 2 + obs[2] ** 2) ** 0.5

                # הדפסה בשורה אחת שמתעדכנת
                print(f"Step: {step_count} | Distance: {dist:.2f}m | Alt: {abs(obs[2]):.1f}m", end='\r')

                # השהיה קלה כדי שנוכל לעקוב בעין ב-AirSim
                time.sleep(0.01)

            print(f"\n🏁 Episode {episode} Finished. Total Steps: {step_count} | Total Reward: {total_reward:.2f}")
            time.sleep(2)  # הפסקה קצרה לראות את התוצאה לפני הריסט הבא

    except Exception as e:
        print(f"❌ An error occurred during testing: {e}")
    finally:
        print("Closing environment...")
        env.close()


if __name__ == "__main__":
    main()