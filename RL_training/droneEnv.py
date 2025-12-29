import cosysairsim as airsim
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
import random
import threading
import time


class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # [rel_x, rel_y, target_area, is_locked]
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        self.last_frame = None
        self.running = True
        self.window_name = "Drone High-FPS Vision"

        # רזולוציית תצוגה מוגדלת (1080p-like)
        self.display_width = 1600
        self.display_height = 900

        self.total_reward = 0
        self.steps_in_episode = 0
        self.episode_count = 0
        self.target_world_pos = None

        self.lower_green = np.array([40, 40, 40])
        self.upper_green = np.array([80, 255, 255])

        self.display_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.display_thread.start()

    def _apply_random_environment(self):
        try:
            hour = random.randint(0, 23)
            self.client.simSetTimeOfDay(True, f"{hour:02d}:00:00", True)
            self.client.simEnableWeather(True)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, random.uniform(0, 0.3))
        except:
            pass

    def _get_cv_obs(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])

        u, v, a = 0.5, 0.5, 0.0
        is_visible = 0.0

        if responses and len(responses) >= 2:
            try:
                img_seg = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height,
                                                                                               responses[0].width, 3)
                img_raw = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8).reshape(responses[1].height,
                                                                                               responses[1].width, 3)

                # הגדלת רזולוציה ב-OpenCV
                img_scene = cv2.resize(img_raw, (self.display_width, self.display_height),
                                       interpolation=cv2.INTER_LINEAR)
                h_img, w_img = img_scene.shape[:2]
                center_img = (int(w_img / 2), int(h_img / 2))

                # צלב מרכז אדום
                cv2.drawMarker(img_scene, center_img, (0, 0, 255), cv2.MARKER_CROSS, 25, 2)

                hsv = cv2.cvtColor(img_seg, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) > 50:
                        x, y, w, h = cv2.boundingRect(c)
                        u = (x + w / 2) / responses[0].width
                        v = (y + h / 2) / responses[0].height
                        a = (w * h) / (responses[0].width * responses[0].height)
                        is_visible = 1.0

                        # סקייל לריבוע ולטקסט
                        scale_x, scale_y = self.display_width / responses[0].width, self.display_height / responses[
                            0].height
                        nx, ny, nw, nh = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)

                        cv2.rectangle(img_scene, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)

                        # טקסט LOCKED בגודל מאוזן (0.45)
                        info_text = f"LOCKED | u:{u:.2f} v:{v:.2f} a:{a:.3f}"
                        cv2.putText(img_scene, info_text, (nx, ny - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
                                    cv2.LINE_AA)

                elif self.target_world_pos is not None:
                    is_visible = -1.0
                    cv2.putText(img_scene, "MEMORY MODE", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                self.last_frame = img_scene
            except Exception:
                pass

        return np.array([(u - 0.5) * 2, (v - 0.5) * 2, a, is_visible], dtype=np.float32)

    def _render_loop(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        while self.running:
            if self.last_frame is not None:
                cv2.imshow(self.window_name, self.last_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()

    def step(self, action):
        self.client.moveByVelocityBodyFrameAsync(float(action[0]) * 5, float(action[1]) * 5, float(action[2]) * 5, 0.05)

        obs = self._get_cv_obs()
        collision_info = self.client.simGetCollisionInfo()
        reward = -np.linalg.norm(obs[:2]) + (obs[2] * 20)
        if obs[3] > 0: reward += 1.0

        print(f"Step {self.steps_in_episode:03d} | u={obs[0]:.2f}, v={obs[1]:.2f}, a={obs[2]:.3f} | Rew: {reward:.2f}",
              end='\r')

        terminated = False
        reason = ""
        if collision_info.has_collided:
            reward, terminated, reason = -100, True, f"CRASH"
        elif obs[2] > 0.45:
            reward, terminated, reason = 200, True, "LANDED"

        self.total_reward += reward
        self.steps_in_episode += 1
        truncated = self.steps_in_episode > 800
        if truncated: reason = "TIMEOUT"
        if terminated or truncated: self._print_episode_summary(reason)

        return obs, reward, terminated, truncated, {}

    def _print_episode_summary(self, reason):
        self.episode_count += 1
        print(f"\n🏁 EP {self.episode_count} | {reason} | Rew: {self.total_reward:.2f} | Steps: {self.steps_in_episode}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.client.reset()
        self._apply_random_environment()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.total_reward, self.steps_in_episode = 0, 0
        return self._get_cv_obs(), {}

    def close(self):
        self.running = False