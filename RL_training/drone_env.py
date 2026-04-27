import time
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
import cosysairsim as airsim

from object_tracker import tracker
from rewards import RewardManager
from weights_config import EnvConfig


class DroneEnv(gym.Env):
    """
    ObsDim=48

    A Target / vision (0..7)
      0 rel_x
      1 rel_y
      2 area_m11
      3 quality_m11 (MATCH=+1, PRED=+0.2, NONE=-1)
      4 img_vel_x
      5 img_vel_y
      6 focus_streak_norm
      7 match_state (MATCH=+1, PRED=0, NONE=-1)

    B Self state (8..18)
      8 vbx
      9 vby
      10 vbz
      11 ax
      12 ay
      13 az
      14 yaw_rate
      15 roll
      16 pitch
      17 yaw_abs
      18 alt_agl

    C Obstacle awareness (19..28)
      19 front
      20 front_left
      21 left
      22 right
      23 front_right
      24 down
      25 up
      26 min_obstacle
      27 obstacle_bearing_x
      28 obstacle_bearing_y

    D Range / mission geometry (29..35)
      29 range_to_target
      30 range_rate
      31 desired_range
      32 range_error
      33 desired_alt
      34 alt_error
      35 bearing_error

    E Control history (36..43)
      36 prev_vx_cmd
      37 prev_vy_cmd
      38 prev_vz_cmd
      39 prev_yaw_cmd
      40 d_vx_cmd
      41 d_vy_cmd
      42 d_vz_cmd
      43 d_yaw_cmd

    F Mission / reserve (44..47)
      44 mode
      45 phase_progress
      46 disturb_bx
      47 disturb_by
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else EnvConfig()

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.tracker = tracker()
        self.reward_manager = RewardManager(self.cfg)

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.OBS_DIM = int(self.cfg.obs_dim)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.OBS_DIM,), dtype=np.float32)

        self.target_fingerprint = None
        self.target_class_id = None
        self._target_initialized = False

        self._fps_ema = 0.0
        self.episode_id = 0
        self.step_in_episode = 0

        self._lost_time = 0.0
        self._focus_streak = 0.0
        self._pred_focus_time = 0.0
        self._low_alt_time = 0.0
        self._stuck_time = 0.0

        self._ep_return = 0.0
        self._ep_match = 0
        self._ep_pred = 0
        self._ep_none = 0
        self._ep_max_focus = 0.0
        self._global_max_focus = 0.0
        self._ep_start = time.time()

        self._prev_rel_x = None
        self._prev_rel_y = None
        self._prev_area01 = None
        self._prev_range_proxy = None

        self._last_min_obst = None
        self._last_sectors = np.zeros(7, dtype=np.float32)

        self._last_cmd_vx = 0.0
        self._last_cmd_vy = 0.0
        self._last_cmd_vz = 0.0
        self._last_action = np.zeros(4, dtype=np.float32)
        self._prev_action = np.zeros(4, dtype=np.float32)

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

    def _get_self_state_features(self):
        if not self.cfg.use_self_state_obs:
            return (0.0,) * 11

        try:
            ms = self.client.getMultirotorState()
            k = ms.kinematics_estimated

            v = k.linear_velocity
            vbx = self._norm_by_max_to_11(float(v.x_val), self.cfg.vb_max_mps)
            vby = self._norm_by_max_to_11(float(v.y_val), self.cfg.vb_max_mps)
            vbz = self._norm_by_max_to_11(float(v.z_val), self.cfg.vb_max_mps)

            axn = ayn = azn = 0.0
            if self.cfg.use_accel_slots:
                a = k.linear_acceleration
                axn = self._norm_by_max_to_11(float(a.x_val), self.cfg.accel_max_mps2)
                ayn = self._norm_by_max_to_11(float(a.y_val), self.cfg.accel_max_mps2)
                azn = self._norm_by_max_to_11(float(a.z_val), self.cfg.accel_max_mps2)

            av = k.angular_velocity
            yaw_rate_dps = self._deg(float(av.z_val))
            yawrn = self._norm_by_max_to_11(yaw_rate_dps, self.cfg.yaw_rate_max_dps)

            q = k.orientation
            pitch_r, roll_r, yaw_r = airsim.to_eularian_angles(q)
            pitch_deg = self._deg(float(pitch_r))
            roll_deg = self._deg(float(roll_r))
            yaw_deg = self._deg(float(yaw_r))

            rolln = self._norm_by_max_to_11(roll_deg, self.cfg.att_max_deg)
            pitchn = self._norm_by_max_to_11(pitch_deg, self.cfg.att_max_deg)
            yawn = self._norm_by_max_to_11(yaw_deg, 180.0) if self.cfg.use_abs_yaw_obs else 0.0

            pos = k.position
            alt_m = float(-pos.z_val)
            alt_agl_n = self._norm_by_max_to_11(alt_m, self.cfg.alt_max_m)

            return (vbx, vby, vbz, axn, ayn, azn, yawrn, rolln, pitchn, yawn, alt_agl_n)
        except Exception:
            return (0.0,) * 11


    def _get_alt_agl_m(self) -> float | None:
        try:
            ms = self.client.getMultirotorState()
            return float(-ms.kinematics_estimated.position.z_val)
        except Exception:
            return None

    def _get_lidar_features(self):
        if not self.cfg.use_lidar_sectors_obs:
            self._last_min_obst = None
            self._last_sectors[:] = 0.0
            return (0.0,) * 10

        try:
            data = self.client.getDistanceSensorData(self.cfg.distance_sensor_name, vehicle_name=self.cfg.vehicle_name)
            d = float(getattr(data, "distance", 0.0))
            if not np.isfinite(d) or d <= 0.0:
                d = 0.0

            self._last_min_obst = d if d > 0.0 else None
            dn01 = float(np.clip((d if d > 0.0 else self.cfg.lidar_max_dist_m) / self.cfg.lidar_max_dist_m, 0.0, 1.0))
            v11 = self._map01_to_11(dn01)

            features = (v11, v11, v11, v11, v11, v11, 1.0, v11, 0.0, 0.0)
            self._last_sectors[:] = np.array([v11, v11, v11, v11, v11, v11, 1.0], dtype=np.float32)
            return features
        except Exception:
            self._last_min_obst = None
            self._last_sectors[:] = 0.0
            return (0.0,) * 10

    def _range_proxy_from_area(self, area01: float) -> float:
        far01 = 1.0 - float(np.clip(area01, 0.0, 1.0))
        return self._map01_to_11(far01)

    def _range_rate_proxy(self, range_proxy: float, dt: float) -> float:
        if dt <= 1e-6 or self._prev_range_proxy is None:
            return 0.0
        dr = (float(range_proxy) - float(self._prev_range_proxy)) / dt
        return self._norm_by_max_to_11(dr, max_abs=2.0)

    def _disturbance_estimate_2d(self):
        if not self.cfg.use_disturbance_estimate:
            return (0.0, 0.0)

        try:
            ms = self.client.getMultirotorState()
            v = ms.kinematics_estimated.linear_velocity

            dx = float(v.x_val) - float(self._last_cmd_vx)
            dy = float(v.y_val) - float(self._last_cmd_vy)

            return (
                self._norm_by_max_to_11(dx, self.cfg.disturb_max_mps),
                self._norm_by_max_to_11(dy, self.cfg.disturb_max_mps),
            )
        except Exception:
            return (0.0, 0.0)

    @staticmethod
    def _match_state_feature(is_match: bool, is_pred: bool) -> float:
        if is_match:
            return 1.0
        if is_pred:
            return 0.0
        return -1.0

    def _action_history_features(self, action: np.ndarray):
        prev = self._prev_action.copy()
        delta = action - prev
        return (
            float(prev[0]), float(prev[1]), float(prev[2]), float(prev[3]),
            float(delta[0]), float(delta[1]), float(delta[2]), float(delta[3]),
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_id += 1
        self.step_in_episode = 0

        self._lost_time = 0.0
        self._focus_streak = 0.0
        self._pred_focus_time = 0.0
        self._low_alt_time = 0.0
        self._stuck_time = 0.0

        self._ep_return = 0.0
        self._ep_match = 0
        self._ep_pred = 0
        self._ep_none = 0
        self._ep_max_focus = 0.0
        self._ep_start = time.time()

        self._prev_rel_x = None
        self._prev_rel_y = None
        self._prev_area01 = None
        self._prev_range_proxy = None

        self._last_min_obst = None
        self._last_sectors[:] = 0.0

        self._last_cmd_vx = 0.0
        self._last_cmd_vy = 0.0
        self._last_cmd_vz = 0.0
        self._last_action[:] = 0.0
        self._prev_action[:] = 0.0

        self.reward_manager.reset_episode()

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

    def step(self, action):
        t0 = time.time()
        action = np.asarray(action, dtype=np.float32)

        vx_cmd = float(action[0]) * self.cfg.vx_scale
        vy_cmd = float(action[1]) * self.cfg.vy_scale
        vz_cmd = 0.0 if self.cfg.freeze_vz else float(action[2]) * self.cfg.vz_scale
        yaw_rate_cmd = float(action[3]) * self.cfg.yaw_rate_scale_dps

        self.tracker.last_yaw_rate_cmd_dps = float(yaw_rate_cmd)

        self._last_cmd_vx = vx_cmd
        self._last_cmd_vy = vy_cmd
        self._last_cmd_vz = vz_cmd
        self._last_action[:] = action

        self.client.moveByVelocityBodyFrameAsync(
            vx=vx_cmd,
            vy=vy_cmd,
            vz=vz_cmd,
            duration=float(self.cfg.cmd_duration_s),
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate_cmd),
        ).join()

        frame = self._get_frame()
        bbox = self.tracker.update(frame)

        dt = min(time.time() - t0, float(self.cfg.max_step_sec))
        fps = 1.0 / (dt + 1e-6)
        self._fps_ema = fps if self._fps_ema == 0 else 0.9 * self._fps_ema + 0.1 * fps
        fps_show = int(self._fps_ema)

        is_match = (self.tracker.last_mode == "MATCH")
        is_pred = (self.tracker.last_mode == "PRED")

        done = False
        term_reason = ""
        collision_now = False
        focus_timeout = False
        pred_timeout = False

        if self.cfg.use_collision_termination:
            try:
                col = self.client.simGetCollisionInfo()
                if getattr(col, "has_collided", False):
                    done = True
                    collision_now = True
                    term_reason = "collision"
            except Exception:
                pass

        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        vbx, vby, vbz, axn, ayn, azn, yawrn, rolln, pitchn, yawn, alt_agl_n = self._get_self_state_features()
        obs[8:19] = np.array([vbx, vby, vbz, axn, ayn, azn, yawrn, rolln, pitchn, yawn, alt_agl_n], dtype=np.float32)
        alt_agl_m = self._get_alt_agl_m()
        speed_xy_mps = float(np.sqrt((float(vbx) * self.cfg.vb_max_mps) ** 2 + (float(vby) * self.cfg.vb_max_mps) ** 2))

        lidar_feats = self._get_lidar_features()
        obs[19:29] = np.array(lidar_feats, dtype=np.float32)

        prev_vx, prev_vy, prev_vz, prev_yaw, dvx, dvy, dvz, dyaw = self._action_history_features(action)
        obs[36:44] = np.array([prev_vx, prev_vy, prev_vz, prev_yaw, dvx, dvy, dvz, dyaw], dtype=np.float32)

        disturb_bx, disturb_by = self._disturbance_estimate_2d()
        obs[44:48] = np.array([
            self._clip11(float(self.cfg.mode_norm)),
            self._clip11(float(self.cfg.phase_progress_norm)),
            disturb_bx,
            disturb_by,
        ], dtype=np.float32)

        self.step_in_episode += 1

        alt_error = float(alt_agl_n - float(self.cfg.desired_alt_norm))

        if bbox is None and not done:
            self._ep_none += 1
            obs[3] = -1.0
            obs[7] = -1.0
            obs[31] = self._clip11(float(self.cfg.desired_range_norm))
            obs[33] = self._clip11(float(self.cfg.desired_alt_norm))
            obs[34] = self._clip11(alt_error)

            self._lost_time += dt
            self._focus_streak = 0.0
            self._pred_focus_time = 0.0

            ground_contact = False
            self._low_alt_time = 0.0

            if (not done) and self._lost_time >= float(self.cfg.focus_fail_sec):
                done = True
                focus_timeout = True
                term_reason = "focus_timeout"

            reward = self.reward_manager.compute_no_bbox_reward(
                action=action,
                dt=dt,
                vz_cmd=vz_cmd,
                min_obst_m=self._last_min_obst,
                focus_timeout=focus_timeout,
                pred_timeout=False,
                collision=False,
                alt_agl_m=alt_agl_m,
                ground_contact=ground_contact,
            )

        elif not done:
            h, w = frame.shape[:2]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            rel_x = self._clip11(float((cx - w / 2.0) / (w / 2.0)))
            rel_y = self._clip11(float((cy - h / 2.0) / (h / 2.0)))

            area01 = float(((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (w * h))
            area01 = self._clip01(area01)
            area_m11 = self._map01_to_11(area01)

            if is_match:
                q = 1.0
                self._ep_match += 1
            elif is_pred:
                q = 0.2
                self._ep_pred += 1
            else:
                q = -1.0

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

            center_error = float(np.sqrt(rel_x * rel_x + rel_y * rel_y))
            in_focus_match = (is_match and center_error < float(self.cfg.center_ok_dist))
            in_focus_pred = (is_pred and center_error < float(self.cfg.pred_center_ok_dist))

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
            focus_streak_norm = self._clip11(min(1.0, self._focus_streak / max(1e-6, float(self.cfg.focus_fail_sec))))
            match_state = self._match_state_feature(is_match, is_pred)

            range_to_target = 0.0
            range_rate = 0.0
            if self.cfg.use_real_range_to_target:
                range_to_target = 0.0
                range_rate = 0.0
            else:
                if self.cfg.use_range_proxy_from_area:
                    range_to_target = self._range_proxy_from_area(area01)
                if self.cfg.use_range_rate_proxy:
                    range_rate = self._range_rate_proxy(range_to_target, dt)
            range_error = float(range_to_target - float(self.cfg.desired_range_norm))
            bearing_error = float(rel_x)

            self._prev_range_proxy = range_to_target
            self._prev_area01 = area01

            obs[0:8] = np.array([
                rel_x, rel_y, area_m11, self._clip11(q), t_vx_img, t_vy_img,
                focus_streak_norm, match_state,
            ], dtype=np.float32)
            obs[29:36] = np.array([
                self._clip11(range_to_target),
                self._clip11(range_rate),
                self._clip11(float(self.cfg.desired_range_norm)),
                self._clip11(range_error),
                self._clip11(float(self.cfg.desired_alt_norm)),
                self._clip11(alt_error),
                self._clip11(bearing_error),
            ], dtype=np.float32)

            ground_contact = False
            self._low_alt_time = 0.0

            action_delta_norm = float(np.max(np.abs(action - self._prev_action)))
            stuck_triggered = False
            focused_now = bool(in_focus_match or in_focus_pred)
            if self.cfg.use_stuck_penalty and focused_now and speed_xy_mps < float(self.cfg.stuck_speed_thresh_mps) and action_delta_norm < float(self.cfg.stuck_action_thresh):
                self._stuck_time += dt
                stuck_triggered = self._stuck_time >= float(self.cfg.stuck_time_s)
            else:
                self._stuck_time = 0.0

            if not done:
                if self._pred_focus_time >= float(self.cfg.pred_focus_max_sec):
                    done = True
                    pred_timeout = True
                    term_reason = "pred_focus_timeout"
                elif self._lost_time >= float(self.cfg.focus_fail_sec):
                    done = True
                    focus_timeout = True
                    term_reason = "focus_timeout"

            reward = self.reward_manager.compute_tracking_reward(
                action=action,
                dt=dt,
                vz_cmd=vz_cmd,
                area01=area01,
                center_error=center_error,
                focus_streak=self._focus_streak,
                is_match=is_match,
                is_pred=is_pred,
                range_error=range_error,
                range_to_target=range_to_target,
                desired_range=float(self.cfg.desired_range_norm),
                alt_error=alt_error,
                alt_agl_m=alt_agl_m,
                bearing_error=bearing_error,
                min_obst_m=self._last_min_obst,
                focus_timeout=focus_timeout,
                pred_timeout=pred_timeout,
                collision=False,
                stuck_triggered=stuck_triggered,
                ground_contact=ground_contact,
            )

        else:
            obs[:] = 0.0
            obs[3] = -1.0
            obs[7] = -1.0
            reward = self.reward_manager.compute_no_bbox_reward(
                action=action,
                dt=dt,
                vz_cmd=vz_cmd,
                min_obst_m=self._last_min_obst,
                focus_timeout=False,
                pred_timeout=False,
                collision=True,
                alt_agl_m=alt_agl_m,
                ground_contact=False,
            )

        self._ep_return += float(reward)
        self._prev_action[:] = action

        if self.cfg.show_cv_window:
            self.tracker.draw(frame, fps_show)
            cv2.putText(frame, f"MODE: {getattr(self.tracker, 'last_mode', '?')}", (20, 90),
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

        total = max(1, self.step_in_episode)
        match_pct = 100.0 * self._ep_match / total
        pred_pct = 100.0 * self._ep_pred / total
        none_pct = 100.0 * self._ep_none / total

        if done and self.cfg.print_ep_summary:
            dur = time.time() - self._ep_start
            print(
                f"[EP {self.episode_id}] steps={self.step_in_episode} "
                f"return={self._ep_return:+.2f} "
                f"max_focus={self._ep_max_focus:.1f}s "
                f"GLOBAL={self._global_max_focus:.1f}s "
                f"MATCH={match_pct:.1f}% "
                f"PRED={pred_pct:.1f}% "
                f"NONE={none_pct:.1f}% "
                f"R_center={self.reward_manager.stats.center:+.2f} "
                f"R_obst={self.reward_manager.stats.obstacle:+.2f} "
                f"R_range={self.reward_manager.stats.range_term:+.2f} "
                f"R_smooth={self.reward_manager.stats.smooth:+.2f} "
                f"R_low_alt={self.reward_manager.stats.low_alt:+.2f} "
                f"R_too_close={self.reward_manager.stats.too_close:+.2f} "
                f"R_stuck={self.reward_manager.stats.stuck:+.2f} "
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
            "alt_agl_m": None if alt_agl_m is None else float(alt_agl_m),
            "stuck_time_s": float(self._stuck_time),
            "match_pct": float(match_pct),
            "pred_pct": float(pred_pct),
            "none_pct": float(none_pct),
            "episode_done": bool(done),
        }

        return obs, float(reward), bool(done), False, info
