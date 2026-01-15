# drone_env.py
import time
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
import cosysairsim as airsim

from object_tracker import tracker
from weights_config import EnvConfig


class DroneEnv(gym.Env):
    """
    ObsDim=34 (fixed, prepared for real drone: LiDAR/GPS/IMU + wind + ground slope)

    Actions (4): [vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd]

    OBS layout (34):
      A Target (6): 0..5
        0 t_rel_x
        1 t_rel_y
        2 t_area_m11
        3 t_quality_m11 (MATCH=+1, PRED~=+0.2, NONE=-1)
        4 t_vx_img
        5 t_vy_img

      B Self state (10): 6..15
        6  vbx
        7  vby
        8  vbz
        9  ax   (reserved)
        10 ay   (reserved)
        11 yaw_rate
        12 roll
        13 pitch
        14 alt_agl
        15 alt_rate

      C LiDAR sectors (6): 16..21
        16 d_front
        17 d_front_left
        18 d_left
        19 d_right
        20 d_front_right
        21 d_down

      D Range context (2): 22..23
        22 range_to_target (real or proxy)
        23 range_rate (real or proxy)

      E Intent (4): 24..27
        24 desired_range
        25 desired_alt
        26 mode
        27 phase_progress

      F Disturbance/Wind estimate (3): 28..30
        28 disturb_bx
        29 disturb_by
        30 disturb_bz

      G Ground plane normal (3): 31..33
        31 ground_nx
        32 ground_ny
        33 ground_nz

    Notes:
    - Many blocks can be "reserved" (filled with safe defaults) until you enable them.
    - You manually harden config values in weights_config.py (or swap configs).
    """

    def __init__(self, cfg: EnvConfig | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else EnvConfig()

        # AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # Tracker
        self.tracker = tracker()

        # Spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.OBS_DIM = int(self.cfg.obs_dim)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.OBS_DIM,), dtype=np.float32)

        # One-time target persistence
        self.target_fingerprint = None
        self.target_class_id = None
        self._target_initialized = False

        # Timing / FPS
        self._fps_ema = 0.0

        # Episode state
        self.episode_id = 0
        self.step_in_episode = 0

        self._lost_time = 0.0
        self._focus_streak = 0.0
        self._pred_focus_time = 0.0

        self._ep_return = 0.0
        self._ep_match = 0
        self._ep_pred = 0
        self._ep_none = 0
        self._ep_max_focus = 0.0
        self._global_max_focus = 0.0
        self._ep_start = time.time()

        # For target img velocity features
        self._prev_rel_x = None
        self._prev_rel_y = None
        self._prev_area01 = None

        # For obstacle debug
        self._last_min_obst = None
        self._last_sectors = np.zeros(6, dtype=np.float32)

        # For disturbance estimate (need last commanded v)
        self._last_cmd_vx = 0.0
        self._last_cmd_vy = 0.0
        self._last_cmd_vz = 0.0

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    @staticmethod
    def _clip11(x: float) -> float:
        return float(np.clip(x, -1.0, 1.0))

    @staticmethod
    def _map01_to_11(x01: float) -> float:
        return float(2.0 * np.clip(x01, 0.0, 1.0) - 1.0)

    def _norm_by_max_to_11(self, x: float, max_abs: float) -> float:
        if max_abs <= 1e-6:
            return 0.0
        return self._clip11(x / max_abs)

    @staticmethod
    def _deg(rad: float) -> float:
        return float(rad * 180.0 / np.pi)

    # -----------------------------
    # Frame
    # -----------------------------
    def _get_frame(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])
        if not responses or not responses[0].image_data_uint8:
            return np.zeros((360, 640, 3), dtype=np.uint8)

        img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        frame = img.reshape(responses[0].height, responses[0].width, 3)
        return cv2.resize(frame, (640, 360))

    # -----------------------------
    # One-time click lock
    # -----------------------------
    def _click_lock_once(self):
        selected = False
        win = "INITIAL SETUP: Click target (ONE TIME)"
        cv2.namedWindow(win)

        if self.cfg.print_reset:
            print("\n[ENV] ONE-TIME TARGET SELECTION (click object)\n")

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

        if self.cfg.print_reset:
            print("[ENV] Target locked and persisted\n")

    # -----------------------------
    # Self state block (10)
    # -----------------------------
    def _get_self_state_features(self):
        if not self.cfg.use_self_state_obs:
            return (0.0,) * 10

        try:
            ms = self.client.getMultirotorState()
            k = ms.kinematics_estimated

            v = k.linear_velocity
            vbx = self._norm_by_max_to_11(float(v.x_val), self.cfg.vb_max_mps)
            vby = self._norm_by_max_to_11(float(v.y_val), self.cfg.vb_max_mps)
            vbz = self._norm_by_max_to_11(float(v.z_val), self.cfg.vb_max_mps)

            axn = 0.0
            ayn = 0.0
            if self.cfg.use_accel_slots:
                a = k.linear_acceleration
                axn = self._norm_by_max_to_11(float(a.x_val), 10.0)
                ayn = self._norm_by_max_to_11(float(a.y_val), 10.0)

            av = k.angular_velocity  # rad/s
            yaw_rate_dps = self._deg(float(av.z_val))
            yawrn = self._norm_by_max_to_11(yaw_rate_dps, self.cfg.yaw_rate_max_dps)

            q = k.orientation
            pitch_r, roll_r, _yaw_r = airsim.to_eularian_angles(q)
            pitch_deg = self._deg(float(pitch_r))
            roll_deg = self._deg(float(roll_r))

            pitchn = self._norm_by_max_to_11(pitch_deg, self.cfg.att_max_deg)
            rolln = self._norm_by_max_to_11(roll_deg, self.cfg.att_max_deg)

            pos = k.position
            alt_m = float(-pos.z_val)  # approx AGL in sim (NED)
            alt_agl_n = self._norm_by_max_to_11(alt_m, self.cfg.alt_max_m)

            alt_rate_mps = float(-v.z_val)
            alt_rate_n = self._norm_by_max_to_11(alt_rate_mps, self.cfg.alt_rate_max_mps)

            return (vbx, vby, vbz, axn, ayn, yawrn, rolln, pitchn, alt_agl_n, alt_rate_n)

        except Exception:
            return (0.0,) * 10

    # -----------------------------
    # LiDAR sectors block (6)
    # -----------------------------
    def _get_lidar_sectors(self):
        if not self.cfg.use_lidar_sectors_obs:
            self._last_min_obst = None
            self._last_sectors[:] = 0.0
            return (0.0,) * 6

        try:
            data = self.client.getDistanceSensorData(self.cfg.distance_sensor_name, vehicle_name=self.cfg.vehicle_name)
            d = float(getattr(data, "distance", 0.0))
            if not np.isfinite(d) or d <= 0.0:
                d = 0.0

            self._last_min_obst = d

            dn01 = float(np.clip(d / self.cfg.lidar_max_dist_m, 0.0, 1.0))
            v11 = self._map01_to_11(dn01)

            sectors = (v11, v11, v11, v11, v11, v11)  # replicate (POC)
            self._last_sectors[:] = np.array(sectors, dtype=np.float32)
            return sectors

        except Exception:
            self._last_min_obst = None
            self._last_sectors[:] = 0.0
            return (0.0,) * 6

    def _obstacle_penalty(self, dt: float) -> float:
        if (not self.cfg.use_obstacle_penalty) or (self._last_min_obst is None):
            return 0.0
        d = float(self._last_min_obst)
        if d <= 0.0 or d >= self.cfg.obstacle_safe_dist_m:
            return 0.0
        return -self.cfg.obstacle_penalty_k * (self.cfg.obstacle_safe_dist_m - d) * dt

    # -----------------------------
    # Range proxies (block D)
    # -----------------------------
    def _range_proxy_from_area(self, area01: float) -> float:
        far01 = 1.0 - float(np.clip(area01, 0.0, 1.0))
        return self._map01_to_11(far01)

    def _range_rate_proxy(self, area01: float, dt: float) -> float:
        if dt <= 1e-6 or self._prev_area01 is None:
            return 0.0
        da = (area01 - float(self._prev_area01)) / dt
        return self._norm_by_max_to_11(da, max_abs=2.0)

    # -----------------------------
    # Disturbance / wind estimate (block F)
    # -----------------------------
    def _disturbance_estimate(self):
        """
        Returns (disturb_bx, disturb_by, disturb_bz) normalized to [-1,1].

        Practical definition (works for sim and real):
          disturb ≈ v_measured - v_commanded

        If not enabled -> zeros.
        """
        if not self.cfg.use_disturbance_estimate:
            return (0.0, 0.0, 0.0)

        try:
            ms = self.client.getMultirotorState()
            v = ms.kinematics_estimated.linear_velocity

            dx = float(v.x_val) - float(self._last_cmd_vx)
            dy = float(v.y_val) - float(self._last_cmd_vy)
            dz = float(v.z_val) - float(self._last_cmd_vz)

            return (
                self._norm_by_max_to_11(dx, self.cfg.disturb_max_mps),
                self._norm_by_max_to_11(dy, self.cfg.disturb_max_mps),
                self._norm_by_max_to_11(dz, self.cfg.disturb_max_mps),
            )
        except Exception:
            return (0.0, 0.0, 0.0)

    # -----------------------------
    # Ground plane normal estimate (block G)
    # -----------------------------
    def _ground_normal_estimate(self):
        """
        Returns (nx, ny, nz) normalized to [-1,1].

        Default safe value for "flat ground":
          normal = (0, 0, 1)

        In real drone:
          - fit a plane under the drone using LiDAR/depth points
          - compute its normal in body/world frame
        """
        if not self.cfg.use_ground_normal_estimate:
            return (0.0, 0.0, 1.0)

        # Placeholder: you can replace this with real plane-fitting later.
        # For now keep "flat".
        return (0.0, 0.0, 1.0)

    # -----------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_id += 1
        self.step_in_episode = 0

        self._lost_time = 0.0
        self._focus_streak = 0.0
        self._pred_focus_time = 0.0

        self._ep_return = 0.0
        self._ep_match = 0
        self._ep_pred = 0
        self._ep_none = 0
        self._ep_max_focus = 0.0
        self._ep_start = time.time()

        self._prev_rel_x = None
        self._prev_rel_y = None
        self._prev_area01 = None

        self._last_min_obst = None
        self._last_sectors[:] = 0.0

        self._last_cmd_vx = 0.0
        self._last_cmd_vy = 0.0
        self._last_cmd_vz = 0.0

        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        frame = self._get_frame()

        if not self._target_initialized or self.target_fingerprint is None:
            self._click_lock_once()

        self.tracker.set_target_fingerprint(self.target_fingerprint)
        self.tracker.set_target_class(self.target_class_id)
        self.tracker.auto_lock_on_fingerprint(frame, use_class_gate=True)

        if self.cfg.print_reset:
            print(f"[RESET] Episode {self.episode_id}")

        obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        return obs, {}

    # -----------------------------
    def step(self, action):
        t0 = time.time()

        # Scale actions
        vx_cmd = float(action[0]) * self.cfg.vx_scale
        vy_cmd = float(action[1]) * self.cfg.vy_scale

        if self.cfg.freeze_vz:
            vz_cmd = 0.0
        else:
            vz_cmd = float(action[2]) * self.cfg.vz_scale

        yaw_rate_cmd = float(action[3]) * self.cfg.yaw_rate_scale_dps

        # -----------------------------
        # YAW COMPENSATION FEED (for tracker PRED correction)
        # -----------------------------
        # The tracker has no access to drone controls/ego-motion by itself.
        # We feed the commanded yaw-rate (deg/sec) each step so it can compensate
        # predicted BBox drift during PRED mode.
        self.tracker.last_yaw_rate_cmd_dps = float(yaw_rate_cmd)

        # Save command for disturbance estimate
        self._last_cmd_vx = vx_cmd
        self._last_cmd_vy = vy_cmd
        self._last_cmd_vz = vz_cmd

        # Execute
        self.client.moveByVelocityBodyFrameAsync(
            vx=vx_cmd,
            vy=vy_cmd,
            vz=vz_cmd,
            duration=float(self.cfg.cmd_duration_s),
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate_cmd)
        ).join()

        frame = self._get_frame()
        bbox = self.tracker.update(frame)

        dt = min(time.time() - t0, float(self.cfg.max_step_sec))
        fps = 1.0 / (dt + 1e-6)
        self._fps_ema = fps if self._fps_ema == 0 else 0.9 * self._fps_ema + 0.1 * fps
        fps_show = int(self._fps_ema)

        is_match = (self.tracker.last_mode == "MATCH")
        is_pred = (self.tracker.last_mode == "PRED")

        # energy penalty
        a = np.array(action, dtype=np.float32)
        energy_penalty = -float(self.cfg.energy_penalty_k) * float(np.abs(a).sum())

        # Optional collision termination
        done = False
        term_reason = ""

        if self.cfg.use_collision_termination:
            try:
                col = self.client.simGetCollisionInfo()
                if getattr(col, "has_collided", False):
                    done = True
                    term_reason = "collision"
            except Exception:
                pass

        # Build obs (always 34)
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        # Fill intent block (E)
        obs[24] = self._clip11(float(self.cfg.desired_range_norm))
        obs[25] = self._clip11(float(self.cfg.desired_alt_norm))
        obs[26] = self._clip11(float(self.cfg.mode_norm))
        obs[27] = self._clip11(float(self.cfg.phase_progress_norm))

        # Fill self state block (B)
        vbx, vby, vbz, axn, ayn, yawrn, rolln, pitchn, alt_agl_n, alt_rate_n = self._get_self_state_features()
        obs[6:16] = np.array([vbx, vby, vbz, axn, ayn, yawrn, rolln, pitchn, alt_agl_n, alt_rate_n],
                             dtype=np.float32)

        # Fill lidar sectors block (C)
        d_front, d_fl, d_left, d_right, d_fr, d_down = self._get_lidar_sectors()
        obs[16:22] = np.array([d_front, d_fl, d_left, d_right, d_fr, d_down], dtype=np.float32)

        # Disturbance / wind block (F)
        dbx, dby, dbz = self._disturbance_estimate()
        obs[28:31] = np.array([dbx, dby, dbz], dtype=np.float32)

        # Ground normal block (G)
        nx, ny, nz = self._ground_normal_estimate()
        obs[31:34] = np.array([self._clip11(nx), self._clip11(ny), self._clip11(nz)], dtype=np.float32)

        self.step_in_episode += 1

        # Reward
        reward = 0.0

        if bbox is None and not done:
            self._ep_none += 1
            obs[3] = -1.0  # quality NONE

            reward = -float(self.cfg.penalty_no_bbox) + energy_penalty
            reward += self._obstacle_penalty(dt)

            self._lost_time += dt
            self._focus_streak = 0.0
            self._pred_focus_time = 0.0

            if self._lost_time >= float(self.cfg.focus_fail_sec):
                reward -= float(self.cfg.penalty_focus_timeout)
                done = True
                term_reason = "focus_timeout"

        elif not done:
            h, w = frame.shape[:2]
            cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

            rel_x = self._clip11(float((cx - w / 2.0) / (w / 2.0)))
            rel_y = self._clip11(float((cy - h / 2.0) / (h / 2.0)))

            area01 = float(((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (w * h))
            area01 = self._clip01(area01)
            area_m11 = self._map01_to_11(area01)

            # quality
            if is_match:
                q = 1.0
                self._ep_match += 1
                reward += float(self.cfg.match_warmup_reward)
            elif is_pred:
                q = 0.2
                self._ep_pred += 1
                reward += float(self.cfg.pred_warmup_reward)
            else:
                q = -1.0

            # img velocity
            if self._prev_rel_x is None or dt <= 1e-6:
                t_vx_img = 0.0
                t_vy_img = 0.0
            else:
                vx_rel_per_s = (rel_x - float(self._prev_rel_x)) / dt
                vy_rel_per_s = (rel_y - float(self._prev_rel_y)) / dt
                t_vx_img = self._norm_by_max_to_11(vx_rel_per_s, float(self.cfg.img_v_rel_per_sec_max))
                t_vy_img = self._norm_by_max_to_11(vy_rel_per_s, float(self.cfg.img_v_rel_per_sec_max))

            self._prev_rel_x = rel_x
            self._prev_rel_y = rel_y

            # Target block (A)
            obs[0:6] = np.array([rel_x, rel_y, area_m11, self._clip11(q), t_vx_img, t_vy_img], dtype=np.float32)

            # Focus logic
            dist = float(np.sqrt(rel_x * rel_x + rel_y * rel_y))
            in_focus_match = (is_match and dist < float(self.cfg.center_ok_dist))
            in_focus_pred = (is_pred and dist < float(self.cfg.pred_center_ok_dist))

            if in_focus_match:
                self._lost_time = 0.0
                self._focus_streak += dt
                self._pred_focus_time = 0.0
            elif in_focus_pred:
                self._lost_time = 0.0
                self._focus_streak += 0.5 * dt
                self._pred_focus_time += dt
            else:
                self._lost_time += dt
                self._focus_streak = 0.0
                self._pred_focus_time = 0.0

            self._ep_max_focus = max(self._ep_max_focus, self._focus_streak)
            self._global_max_focus = max(self._global_max_focus, self._focus_streak)

            if self._pred_focus_time >= float(self.cfg.pred_focus_max_sec):
                reward -= float(self.cfg.penalty_pred_focus_timeout)
                done = True
                term_reason = "pred_focus_timeout"

            if (not done) and (self._lost_time >= float(self.cfg.focus_fail_sec)):
                reward -= float(self.cfg.penalty_focus_timeout)
                done = True
                term_reason = "focus_timeout"

            # Shaping
            reward += float(self.cfg.w_center) * float(np.exp(-dist * float(self.cfg.center_decay)))
            reward += float(self.cfg.w_area) * float(np.sqrt(area01))
            reward += float(self.cfg.w_focus) * float(min(2.0, self._focus_streak / 4.0))

            # PRED penalty
            if is_pred:
                if dist < float(self.cfg.pred_center_ok_dist):
                    reward -= 0.05 * float(self.cfg.pred_penalty_per_sec) * dt
                else:
                    reward -= float(self.cfg.pred_penalty_per_sec) * dt

            # Obstacle penalty
            reward += self._obstacle_penalty(dt)

            reward += energy_penalty

            # Range block (D)
            r_to = 0.0
            r_rate = 0.0
            if self.cfg.use_real_range_to_target:
                # reserved: plug your real sensor fusion later
                r_to = 0.0
                r_rate = 0.0
            else:
                if self.cfg.use_range_proxy_from_area:
                    r_to = self._range_proxy_from_area(area01)
                if self.cfg.use_range_rate_proxy:
                    r_rate = self._range_rate_proxy(area01, dt)

            obs[22] = self._clip11(r_to)
            obs[23] = self._clip11(r_rate)

            self._prev_area01 = area01

        else:
            # Collision termination
            obs[:] = 0.0
            obs[3] = -1.0
            reward = -float(self.cfg.penalty_collision)

        self._ep_return += float(reward)

        # HUD
        if self.cfg.show_cv_window:
            self.tracker.draw(frame, fps_show)

            mode_txt = getattr(self.tracker, "last_mode", "?")
            cv2.putText(frame, f"MODE: {mode_txt}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps_show}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, f"EP MAX FOCUS: {self._ep_max_focus:.1f}s", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(frame, f"GLOBAL MAX FOCUS: {self._global_max_focus:.1f}s", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            if self.cfg.use_lidar_sectors_obs and (self._last_min_obst is not None):
                cv2.putText(frame, f"MIN OBST: {self._last_min_obst:.2f}m", (20, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            cv2.imshow("Tracker Debug", frame)
            cv2.waitKey(1)

        # Episode summary
        if done and self.cfg.print_ep_summary:
            dur = time.time() - self._ep_start
            total = max(1, self.step_in_episode)
            print(
                f"[EP {self.episode_id}] steps={self.step_in_episode} "
                f"return={self._ep_return:+.2f} "
                f"max_focus={self._ep_max_focus:.1f}s "
                f"GLOBAL={self._global_max_focus:.1f}s "
                f"MATCH={100 * self._ep_match / total:.1f}% "
                f"PRED={100 * self._ep_pred / total:.1f}% "
                f"NONE={100 * self._ep_none / total:.1f}% "
                f"dur={dur:.1f}s reason={term_reason}"
            )

        info = {
            "episode_id": self.episode_id,
            "step_in_episode": self.step_in_episode,
            "termination_reason": term_reason,
            "focus_streak_s": float(self._focus_streak),
            "lost_time_s": float(self._lost_time),
            "global_max_focus_s": float(self._global_max_focus),
            "min_obstacle_dist_m": None if self._last_min_obst is None else float(self._last_min_obst),
        }

        return obs, float(reward), bool(done), False, info
