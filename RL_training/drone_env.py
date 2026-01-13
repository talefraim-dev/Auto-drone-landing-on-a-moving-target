import time
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
import cosysairsim as airsim
from object_tracker import tracker


class DroneEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.tracker = tracker()

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # Persist across whole program (ONE TIME target selection)
        self.target_fingerprint = None
        self.target_class_id = None
        self._target_initialized = False

        # Debug FPS smoothing
        self._fps_ema = 0.0

        # ----------------------------
        # Focus / Reward configuration
        # ----------------------------
        self.FOCUS_FAIL_SEC = 8.0
        self.CENTER_OK_DIST = 0.25
        self.MAX_STEP_SEC = 0.20

        # Mild penalties to prevent "free roaming" (blind / pred)
        self.PRED_PENALTY_PER_SEC = 0.40
        self.PRED_ACTION_PENALTY = 0.10

        # NEW: energy penalty always (small)
        self.ENERGY_PENALTY_K = 0.03   # small, always-on

        # NEW: stable-motion bonus when seeing target (MATCH)
        self.STABLE_BONUS_K = 0.20     # bonus scale when stable AND MATCH
        self.STABLE_DELTA_REF = 0.35   # how sensitive to action changes (0..~1)

        # Episode runtime state
        self._lost_time = 0.0
        self._focus_streak = 0.0
        self._prev_dist = None
        self._prev_area = None
        self._prev_action = np.zeros(4, dtype=np.float32)

        # Episode stats + printing (per episode)
        self.episode_id = 0
        self.step_in_episode = 0
        self.total_steps = 0

        self._ep_return = 0.0
        self._ep_match_frames = 0
        self._ep_pred_frames = 0
        self._ep_none_frames = 0
        self._ep_max_focus_streak = 0.0
        self._ep_start_time = time.time()

    def _get_frame(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])
        if not responses or not responses[0].image_data_uint8:
            return np.zeros((360, 640, 3), dtype=np.uint8)

        img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        frame = img.reshape(responses[0].height, responses[0].width, 3)
        return cv2.resize(frame, (640, 360))

    def _click_lock_once(self):
        selected = False
        win = "INITIAL SETUP: Click on your target (ONE TIME ONLY)"
        cv2.namedWindow(win)

        print("\n[ENV] === ONE-TIME TARGET SELECTION ===")
        print("[ENV] Click your target once. It will be reused across ALL resets.\n")

        def on_click(event, x, y, flags, param):
            nonlocal selected
            if event == cv2.EVENT_LBUTTONDOWN:
                cid = self.tracker.select_target_and_get_class(param["frame"], x, y)
                if cid is not None:
                    self.target_class_id = cid
                    self.target_fingerprint = self.tracker.get_target_fingerprint()
                    selected = True

        while not selected:
            frame = self._get_frame()
            cv2.setMouseCallback(win, on_click, param={"frame": frame})
            cv2.imshow(win, frame)
            cv2.waitKey(1)

        cv2.destroyWindow(win)

        self._target_initialized = True
        fp_dim = 0 if self.target_fingerprint is None else int(self.target_fingerprint.shape[0])
        print(f"[ENV] Target saved. class_id={self.target_class_id}, fingerprint_dim={fp_dim}\n")

    def _reset_episode_stats(self):
        self.step_in_episode = 0
        self._lost_time = 0.0
        self._focus_streak = 0.0
        self._prev_dist = None
        self._prev_area = None
        self._prev_action = np.zeros(4, dtype=np.float32)

        self._ep_return = 0.0
        self._ep_match_frames = 0
        self._ep_pred_frames = 0
        self._ep_none_frames = 0
        self._ep_max_focus_streak = 0.0
        self._ep_start_time = time.time()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_id += 1
        self._reset_episode_stats()

        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        frame = self._get_frame()

        # Choose target only once
        if not self._target_initialized or self.target_fingerprint is None:
            self._click_lock_once()

        # Restore target identity to tracker every reset
        self.tracker.set_target_fingerprint(self.target_fingerprint)
        self.tracker.set_target_class(self.target_class_id)

        # Try relock (never ask to click again)
        locked = self.tracker.auto_lock_on_fingerprint(frame, use_class_gate=True)
        if not locked:
            locked = self.tracker.auto_lock_on_fingerprint(frame, use_class_gate=False)

        print(f"[ENV][RESET] Episode #{self.episode_id} | relock={'OK' if locked else 'FAILED -> starting SEARCH/PRED'} | cid={self.target_class_id}")

        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        t0 = time.time()

        # Execute action
        self.client.moveByVelocityBodyFrameAsync(
            vx=float(action[0]) * 5.0,
            vy=float(action[1]) * 5.0,
            vz=float(action[2]) * 3.0,
            duration=0.1,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(action[3]) * 100.0)
        ).join()

        frame = self._get_frame()
        bbox = self.tracker.update(frame)

        dt_real = time.time() - t0
        dt = float(min(dt_real, self.MAX_STEP_SEC))
        fps = 1.0 / (dt + 1e-6)
        self._fps_ema = fps if self._fps_ema == 0 else (0.9 * self._fps_ema + 0.1 * fps)
        fps_show = int(self._fps_ema)

        is_match = (self.tracker.last_mode == "MATCH")
        is_pred = (self.tracker.last_mode == "PRED")

        self.total_steps += 1
        self.step_in_episode += 1

        # Track per-episode mode counts
        if bbox is None:
            self._ep_none_frames += 1
        elif is_match:
            self._ep_match_frames += 1
        else:
            self._ep_pred_frames += 1

        # ----------------------------
        # Base always-on energy penalty (small)
        # ----------------------------
        # Encourage not wasting energy / not "roaming for fun"
        a = np.array(action, dtype=np.float32)
        energy = float(abs(a[0]) + abs(a[1]) + abs(a[2]) + 0.5 * abs(a[3]))
        always_energy_penalty = -self.ENERGY_PENALTY_K * energy

        term_reason = ""
        done = False

        if bbox is not None:
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            rel_x, rel_y = (cx - 320) / 320, (cy - 180) / 180

            area = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (640 * 360)
            area = float(max(0.0, min(1.0, area)))

            dist = float(np.sqrt(rel_x ** 2 + rel_y ** 2))

            conf = 1.0 if is_match else 0.0
            obs = np.array([rel_x, rel_y, area, conf], dtype=np.float32)

            # Focus = MATCH + centered
            in_focus = bool(is_match and dist < self.CENTER_OK_DIST)

            if in_focus:
                self._lost_time = 0.0
                self._focus_streak += dt
            else:
                self._lost_time += dt
                self._focus_streak = 0.0

            self._ep_max_focus_streak = max(self._ep_max_focus_streak, self._focus_streak)

            # Terminate if failed to keep focus too long
            if self._lost_time >= self.FOCUS_FAIL_SEC:
                reward = -25.0
                done = True
                term_reason = "focus_timeout"
            else:
                # Rewards
                r_center = float(np.exp(-dist * 4.0))      # 0..1
                r_close = float(np.sqrt(area))             # 0..1
                r_streak = float(min(2.0, self._focus_streak / 4.0))  # 0..2

                penalty = 0.0

                # Penalize "getting worse"
                if self._prev_dist is not None and dist > self._prev_dist + 1e-4:
                    penalty -= 0.3
                if self._prev_area is not None and area < self._prev_area - 1e-5:
                    penalty -= 0.3

                # Mild penalty while in PRED (Kalman-only)
                if is_pred:
                    penalty -= self.PRED_PENALTY_PER_SEC * dt
                    # discourage aggressive roll/yaw while blind
                    penalty -= self.PRED_ACTION_PENALTY * (abs(float(action[1])) + abs(float(action[3])))

                # Soft penalty if not match (keeps preference to real tracking)
                if not is_match:
                    penalty -= 0.5

                # NEW: stable-motion bonus when MATCH
                # - reward small smooth actions (no crazy changes) while seeing target
                if is_match:
                    delta = float(np.linalg.norm(a - self._prev_action))
                    stable_factor = 1.0 - min(1.0, delta / max(1e-6, self.STABLE_DELTA_REF))  # 1 good, 0 bad
                    # also prefer moderate magnitude actions when locked (stable speed)
                    mag = float(np.linalg.norm(a))
                    mag_factor = 1.0 - min(1.0, mag)  # smaller actions => more stable
                    stable_bonus = self.STABLE_BONUS_K * (0.6 * stable_factor + 0.4 * mag_factor)
                else:
                    stable_bonus = 0.0

                reward = (2.0 * r_center) + (1.5 * r_close) + r_streak + penalty + stable_bonus + always_energy_penalty - 0.2
                done = False

            self._prev_dist = dist
            self._prev_area = area
            self._prev_action = a

        else:
            # Hard fail: no bbox at all
            obs = np.zeros(4, dtype=np.float32)
            reward = -25.0 + always_energy_penalty
            done = True
            term_reason = "no_bbox"
            self._prev_action = a

        # ----------------------------
        # HUD + overlays (including FPS)
        # ----------------------------
        self.tracker.draw(frame, fps_show)

        # Ensure FPS is shown even if tracker HUD changes
        cv2.putText(frame, f"FPS: {fps_show}", (20, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(frame,
                    f"FOCUS_STREAK: {self._focus_streak:4.1f}s | LOST_TIME: {self._lost_time:4.1f}s (fail@{self.FOCUS_FAIL_SEC:.0f}s)",
                    (20, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Tracker Debug", frame)
        cv2.waitKey(1)

        # ----------------------------
        # Episode-level prints (only when episode ends)
        # ----------------------------
        self._ep_return += float(reward)

        if done:
            elapsed = max(1e-6, time.time() - self._ep_start_time)
            eps = self.step_in_episode / elapsed

            total = max(1, self.step_in_episode)
            p_match = 100.0 * (self._ep_match_frames / total)
            p_pred = 100.0 * (self._ep_pred_frames / total)
            p_none = 100.0 * (self._ep_none_frames / total)

            print(
                f"\n[EPISODE END] #{self.episode_id} | steps={self.step_in_episode} | return={self._ep_return:+.2f} | "
                f"max_focus={self._ep_max_focus_streak:.1f}s | "
                f"MATCH={p_match:.1f}% PRED={p_pred:.1f}% NONE={p_none:.1f}% | "
                f"eps~{eps:.1f} | reason={term_reason}\n"
            )

        return obs, float(reward), bool(done), False, {}
