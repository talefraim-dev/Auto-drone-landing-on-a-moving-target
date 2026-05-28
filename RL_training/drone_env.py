import time
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
import cosysairsim as airsim

from object_tracker import tracker
from weights_config import EnvConfig

from observation_builder import (
    ObservationBuilder,
    ObservationBuilderConfig,
    BBox,
    DroneState,
    ObstacleState,
)
from safety_filter import SafetyConfig, safety_filter
from follow_reward_v37 import FollowRewardConfig, compute_follow_reward


class DroneEnv(gym.Env):
    """
    DroneEnv v37.

    OBS_DIM = 37

    0-27:
        Vision core, image-space kinematics, drone self-state, mission state.
    28-36:
        Obstacle awareness.

    Pipeline:
        raw_action -> safety_filter -> safe_action -> AirSim
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else EnvConfig()

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.tracker = tracker()

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.OBS_DIM = int(self.cfg.obs_dim)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.OBS_DIM,), dtype=np.float32)

        if self.OBS_DIM != 37:
            raise ValueError(f"DroneEnv v37 requires cfg.obs_dim=37, got {self.OBS_DIM}")

        self.obs_builder = ObservationBuilder(
            ObservationBuilderConfig(
                image_width=int(self.cfg.image_width),
                image_height=int(self.cfg.image_height),
                max_img_velocity=float(self.cfg.img_v_rel_per_sec_max),
                max_img_acceleration=float(self.cfg.img_acc_rel_per_sec2_max),
                max_altitude_m=float(self.cfg.alt_max_m),
                max_drone_speed_mps=float(self.cfg.vb_max_mps),
                max_vertical_speed_mps=float(self.cfg.vz_max_mps),
                max_roll_rad=np.deg2rad(float(self.cfg.att_max_deg)),
                max_pitch_rad=np.deg2rad(float(self.cfg.att_max_deg)),
                max_yaw_rate_radps=np.deg2rad(float(self.cfg.yaw_rate_max_dps)),
                max_obstacle_range_m=float(self.cfg.lidar_max_dist_m),
                safe_obstacle_distance_m=float(self.cfg.obstacle_safe_dist_m),
                max_lost_target_time_s=float(self.cfg.focus_fail_sec),
            )
        )

        self.safety_config = SafetyConfig(
            safe_distance_m=float(self.cfg.obstacle_safe_dist_m),
            warning_distance_m=float(self.cfg.obstacle_warning_dist_m),
            emergency_distance_m=float(self.cfg.obstacle_emergency_dist_m),
            min_speed_scale_near_obstacle=float(self.cfg.min_speed_scale_near_obstacle),
            steer_strength=float(self.cfg.obstacle_steer_strength),
            emergency_up_cmd=float(self.cfg.emergency_up_cmd),
            enabled=bool(self.cfg.safety_enabled),
        )

        self.reward_config = FollowRewardConfig(
            w_center=float(self.cfg.w_center),
            w_distance=float(self.cfg.w_distance),
            w_visibility=float(self.cfg.w_visibility),
            w_lost_target=float(self.cfg.w_lost_target),
            w_altitude_safe=float(self.cfg.w_altitude_safe),
            w_altitude_low_penalty=float(self.cfg.w_altitude_low_penalty),
            w_altitude_high_penalty=float(self.cfg.w_altitude_high_penalty),
            w_smooth_follow=float(self.cfg.w_smooth_follow),
            w_control=float(self.cfg.w_control),
            w_action_delta=float(self.cfg.w_action_delta),
            w_obstacle=float(self.cfg.w_obstacle),
            w_safety_intervention=float(self.cfg.w_safety_intervention),
            desired_distance_proxy=float(self.cfg.desired_distance_proxy),
            distance_tolerance=float(self.cfg.distance_tolerance),
            min_safe_altitude_m=float(self.cfg.min_safe_altitude_m),
            max_safe_altitude_m=float(self.cfg.max_safe_altitude_m),
            max_img_motion=float(self.cfg.max_img_motion),
            collision_penalty=float(self.cfg.penalty_collision),
        )

        self.target_fingerprint = None
        self.target_class_id = None
        self._target_initialized = False

        self._fps_ema = 0.0
        self.episode_id = 0
        self.step_in_episode = 0

        self._focus_streak = 0.0
        self._global_max_focus = 0.0
        self._ep_max_focus = 0.0
        self._ep_return = 0.0
        self._ep_match = 0
        self._ep_pred = 0
        self._ep_none = 0
        self._ep_start = time.time()

        self._last_min_obst = None
        self._last_obstacle_state = None

        self._prev_action = np.zeros(4, dtype=np.float32)
        self._last_raw_action = np.zeros(4, dtype=np.float32)
        self._last_safe_action = np.zeros(4, dtype=np.float32)

        self._safety_interventions = 0

    @staticmethod
    def _deg(rad: float) -> float:
        return float(rad * 180.0 / np.pi)

    @staticmethod
    def _clip(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, float(value)))

    def _get_frame(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ], vehicle_name=self.cfg.vehicle_name)

        if not responses or not responses[0].image_data_uint8:
            print("[DEBUG] No image received, returning black frame")
            return np.zeros((int(self.cfg.image_height), int(self.cfg.image_width), 3), dtype=np.uint8)

        response = responses[0]
        img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        frame = img.reshape(response.height, response.width, 3)

        # AirSim usually gives RGB, OpenCV displays BGR.
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame

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

    def _bbox_from_tracker(self, bbox) -> BBox | None:
        if bbox is None:
            return None

        try:
            x1, y1, x2, y2 = map(float, bbox[:4])
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            cx = x1 + 0.5 * w
            cy = y1 + 0.5 * h

            mode = getattr(self.tracker, "last_mode", "NONE")
            if mode == "MATCH":
                conf = 1.0
            elif mode == "PRED":
                conf = 0.2
            else:
                conf = 0.0

            return BBox(cx=cx, cy=cy, w=w, h=h, conf=conf)
        except Exception:
            return None

    def _get_drone_state(self) -> DroneState:
        """
        Read drone state from AirSim.

        Note:
            In CitySample / UE large worlds, position.z may not always be a reliable
            altitude-above-ground value. Therefore, we later override altitude_m
            with DistanceDown when it is valid.
        """
        if not self.cfg.use_self_state_obs:
            return DroneState(
                altitude_m=0.0,
                vx_mps=0.0,
                vy_mps=0.0,
                vz_mps=0.0,
                roll_rad=0.0,
                pitch_rad=0.0,
                yaw_rate_radps=0.0,
            )

        try:
            ms = self.client.getMultirotorState(vehicle_name=self.cfg.vehicle_name)
            k = ms.kinematics_estimated

            v = k.linear_velocity
            av = k.angular_velocity

            q = k.orientation
            pitch_r, roll_r, yaw_r = airsim.to_eularian_angles(q)

            pos = k.position
            alt_m = float(-pos.z_val)

            return DroneState(
                altitude_m=max(0.0, alt_m),
                vx_mps=float(v.x_val),
                vy_mps=float(v.y_val),
                vz_mps=float(v.z_val),
                roll_rad=float(roll_r),
                pitch_rad=float(pitch_r),
                yaw_rate_radps=float(av.z_val),
            )
        except Exception:
            return DroneState(
                altitude_m=0.0,
                vx_mps=0.0,
                vy_mps=0.0,
                vz_mps=0.0,
                roll_rad=0.0,
                pitch_rad=0.0,
                yaw_rate_radps=0.0,
            )

    def _get_alt_agl_m(self) -> float | None:
        try:
            ms = self.client.getMultirotorState(vehicle_name=self.cfg.vehicle_name)
            return max(0.0, float(-ms.kinematics_estimated.position.z_val))
        except Exception:
            return None

    def _apply_down_distance_as_altitude_if_valid(self, drone_state: DroneState, obstacle_dict: dict) -> DroneState:
        """
        Prefer DistanceDown as altitude-above-ground estimate when available.

        Why:
            In CitySample / UE large maps, AirSim position.z can stay near 0 or be
            relative to an unexpected origin. DistanceDown is usually a better
            estimate for altitude above the actual road/ground mesh.

        Rule:
            If 0.2m < down_dist < lidar_max_dist_m, use it as altitude_m.
            Otherwise keep the AirSim kinematics altitude.
        """
        down_dist = float(obstacle_dict.get("down_dist_m", 0.0))
        max_d = float(self.cfg.lidar_max_dist_m)

        if np.isfinite(down_dist) and 0.2 < down_dist < max_d:
            drone_state.altitude_m = down_dist

        return drone_state

    def _read_distance_sensor_m(self, sensor_name: str) -> tuple[float, bool]:
        """
        Read one AirSim distance sensor.

        Returns:
            (distance_m, valid)
        """
        max_d = float(self.cfg.lidar_max_dist_m)

        try:
            data = self.client.getDistanceSensorData(
                sensor_name,
                vehicle_name=self.cfg.vehicle_name,
            )

            d = float(getattr(data, "distance", 0.0))

            if not np.isfinite(d) or d <= 0.0:
                return max_d, False

            return float(np.clip(d, 0.0, max_d)), True

        except Exception as e:
            if getattr(self.cfg, "print_sensor_errors", False):
                print(f"[WARN] Failed reading distance sensor {sensor_name}: {e}")

            return max_d, False

    def _get_obstacle_state_m(self, altitude_m: float | None = None) -> dict:
        """
        Read all directional obstacle sensors in meters.

        Important:
        min_obstacle_dist_m is horizontal only.
        down_dist_m is separate and does not participate in horizontal emergency termination.
        """
        max_d = float(self.cfg.lidar_max_dist_m)

        if not self.cfg.use_lidar_sectors_obs:
            return {
                "front_dist_m": max_d,
                "front_left_dist_m": max_d,
                "front_right_dist_m": max_d,
                "left_dist_m": max_d,
                "right_dist_m": max_d,
                "back_dist_m": max_d,
                "down_dist_m": max_d if altitude_m is None else float(altitude_m),
                "min_obstacle_dist_m": max_d,
            }

        front, _ = self._read_distance_sensor_m(self.cfg.distance_sensor_front)
        front_left, _ = self._read_distance_sensor_m(self.cfg.distance_sensor_front_left)
        front_right, _ = self._read_distance_sensor_m(self.cfg.distance_sensor_front_right)
        left, _ = self._read_distance_sensor_m(self.cfg.distance_sensor_left)
        right, _ = self._read_distance_sensor_m(self.cfg.distance_sensor_right)
        back, _ = self._read_distance_sensor_m(self.cfg.distance_sensor_back)
        down, down_valid = self._read_distance_sensor_m(self.cfg.distance_sensor_down)

        # If down sensor is invalid or returns max range, use altitude as a fallback.
        if altitude_m is not None and np.isfinite(altitude_m):
            if (not down_valid) or down >= 0.95 * max_d:
                down = float(np.clip(altitude_m, 0.0, max_d))

        horizontal_min = min(
            front,
            front_left,
            front_right,
            left,
            right,
            back,
        )

        self._last_min_obst = horizontal_min
        self._last_obstacle_state = {
            "front_dist_m": front,
            "front_left_dist_m": front_left,
            "front_right_dist_m": front_right,
            "left_dist_m": left,
            "right_dist_m": right,
            "back_dist_m": back,
            "down_dist_m": down,
            "min_obstacle_dist_m": horizontal_min,
        }

        return self._last_obstacle_state

    def _obstacle_dataclass_from_dict(self, d: dict) -> ObstacleState:
        return ObstacleState(
            front_dist_m=float(d["front_dist_m"]),
            front_left_dist_m=float(d["front_left_dist_m"]),
            front_right_dist_m=float(d["front_right_dist_m"]),
            left_dist_m=float(d["left_dist_m"]),
            right_dist_m=float(d["right_dist_m"]),
            back_dist_m=float(d["back_dist_m"]),
            down_dist_m=float(d["down_dist_m"]),
            min_obstacle_dist_m=float(d["min_obstacle_dist_m"]),
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_id += 1
        self.step_in_episode = 0

        self._focus_streak = 0.0
        self._ep_max_focus = 0.0
        self._ep_return = 0.0
        self._ep_match = 0
        self._ep_pred = 0
        self._ep_none = 0
        self._ep_start = time.time()

        self._last_min_obst = None
        self._last_obstacle_state = None

        self._prev_action[:] = 0.0
        self._last_raw_action[:] = 0.0
        self._last_safe_action[:] = 0.0
        self._safety_interventions = 0

        self.obs_builder.reset()

        self.client.reset()
        time.sleep(float(self.cfg.reset_settle_sec))

        self.client.enableApiControl(True, vehicle_name=self.cfg.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.cfg.vehicle_name)

        self.client.takeoffAsync(vehicle_name=self.cfg.vehicle_name).join()
        time.sleep(float(self.cfg.reset_settle_sec))

        # AirSim uses NED coordinates: negative Z means up.
        self.client.moveToZAsync(
            z=-float(self.cfg.reset_takeoff_altitude_m),
            velocity=float(self.cfg.reset_move_to_z_velocity),
            vehicle_name=self.cfg.vehicle_name,
        ).join()

        time.sleep(float(self.cfg.reset_settle_sec))

        frame = self._get_frame()

        if not self._target_initialized or self.target_fingerprint is None:
            self._click_lock_once()

        self.tracker.set_target_fingerprint(self.target_fingerprint)
        self.tracker.set_target_class(self.target_class_id)
        self.tracker.auto_lock_on_fingerprint(frame, use_class_gate=True)

        # Build a real initial observation instead of returning zeros.
        bbox_raw = self.tracker.update(frame)
        bbox = self._bbox_from_tracker(bbox_raw)

        drone_state = self._get_drone_state()
        obstacle_dict = self._get_obstacle_state_m(altitude_m=drone_state.altitude_m)

        # Altitude fix:
        # Use DistanceDown as the effective altitude when available.
        drone_state = self._apply_down_distance_as_altitude_if_valid(
            drone_state=drone_state,
            obstacle_dict=obstacle_dict,
        )

        obstacle_state = self._obstacle_dataclass_from_dict(obstacle_dict)

        obs, obs_dict = self.obs_builder.build(
            bbox=bbox,
            drone_state=drone_state,
            obstacle_state=obstacle_state,
            dt=float(self.cfg.cmd_duration_s),
        )

        if self.cfg.print_reset:
            print(f"[RESET] Episode {self.episode_id}")

        info = {"obs_dict": obs_dict}
        return obs, info

    def step(self, action):
        t0 = time.time()

        raw_action = np.asarray(action, dtype=np.float32)
        raw_action = np.clip(raw_action, -1.0, 1.0)

        # Read current state before applying the command.
        pre_drone_state = self._get_drone_state()
        obstacle_dict_pre = self._get_obstacle_state_m(altitude_m=pre_drone_state.altitude_m)

        # Altitude fix before safety:
        # This helps the safety layer know the actual AGL when DistanceDown is valid.
        pre_drone_state = self._apply_down_distance_as_altitude_if_valid(
            drone_state=pre_drone_state,
            obstacle_dict=obstacle_dict_pre,
        )

        drone_state_dict = {
            "altitude_m": pre_drone_state.altitude_m,
            "vz_mps": pre_drone_state.vz_mps,
            "speed_xy_mps": float(np.sqrt(pre_drone_state.vx_mps ** 2 + pre_drone_state.vy_mps ** 2)),
            "collision_detected": False,
        }

        safe_action, safety_info = safety_filter(
            raw_action=raw_action,
            obstacle_state=obstacle_dict_pre,
            drone_state=drone_state_dict,
            config=self.safety_config,
        )

        if bool(safety_info.get("safety_intervention", False)):
            self._safety_interventions += 1

        vx_cmd = float(safe_action[0]) * self.cfg.vx_scale
        vy_cmd = float(safe_action[1]) * self.cfg.vy_scale
        vz_cmd = 0.0 if self.cfg.freeze_vz else float(safe_action[2]) * self.cfg.vz_scale
        yaw_rate_cmd = float(safe_action[3]) * self.cfg.yaw_rate_scale_dps

        self.tracker.last_yaw_rate_cmd_dps = float(yaw_rate_cmd)

        self._last_raw_action[:] = raw_action
        self._last_safe_action[:] = safe_action

        self.client.moveByVelocityBodyFrameAsync(
            vx=vx_cmd,
            vy=vy_cmd,
            vz=vz_cmd,
            duration=float(self.cfg.cmd_duration_s),
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate_cmd),
            vehicle_name=self.cfg.vehicle_name,
        ).join()

        frame = self._get_frame()
        bbox_raw = self.tracker.update(frame)

        dt = min(time.time() - t0, float(self.cfg.max_step_sec))
        fps = 1.0 / (dt + 1e-6)
        self._fps_ema = fps if self._fps_ema == 0 else 0.9 * self._fps_ema + 0.1 * fps
        fps_show = int(self._fps_ema)

        is_match = (getattr(self.tracker, "last_mode", "NONE") == "MATCH")
        is_pred = (getattr(self.tracker, "last_mode", "NONE") == "PRED")

        if is_match:
            self._ep_match += 1
        elif is_pred:
            self._ep_pred += 1
        else:
            self._ep_none += 1

        if is_match or is_pred:
            self._focus_streak += dt
        else:
            self._focus_streak = 0.0

        self._ep_max_focus = max(self._ep_max_focus, self._focus_streak)
        self._global_max_focus = max(self._global_max_focus, self._focus_streak)

        bbox = self._bbox_from_tracker(bbox_raw)
        drone_state = self._get_drone_state()
        obstacle_dict = self._get_obstacle_state_m(altitude_m=drone_state.altitude_m)

        # Altitude fix after movement:
        # Use DistanceDown for observation and reward when available.
        drone_state = self._apply_down_distance_as_altitude_if_valid(
            drone_state=drone_state,
            obstacle_dict=obstacle_dict,
        )

        obstacle_state = self._obstacle_dataclass_from_dict(obstacle_dict)

        obs, obs_dict = self.obs_builder.build(
            bbox=bbox,
            drone_state=drone_state,
            obstacle_state=obstacle_state,
            dt=dt,
        )

        collision_now = False
        if self.cfg.use_collision_termination:
            try:
                col = self.client.simGetCollisionInfo(vehicle_name=self.cfg.vehicle_name)
                collision_now = bool(getattr(col, "has_collided", False))
            except Exception:
                collision_now = False

        env_reward_info = {
            "altitude_m": float(drone_state.altitude_m),
            "collision_detected": bool(collision_now),
            "safety_intervention": bool(safety_info.get("safety_intervention", False)),
        }

        reward, reward_parts = compute_follow_reward(
            obs=obs_dict,
            action=safe_action,
            prev_action=self._prev_action,
            env_info=env_reward_info,
            config=self.reward_config,
        )

        done = False
        term_reason = ""

        if collision_now:
            done = True
            term_reason = "collision"

        horizontal_min_obst = float(obstacle_dict["min_obstacle_dist_m"])
        if (
            not done
            and self.step_in_episode > int(self.cfg.ignore_obstacle_termination_first_steps)
            and horizontal_min_obst < float(self.cfg.obstacle_emergency_termination_dist_m)
        ):
            done = True
            term_reason = "emergency_horizontal_obstacle_distance"

        if not done and float(obs_dict["lost_target_time_norm"]) >= 1.0:
            done = True
            term_reason = "target_lost_too_long"

        if (
            not done
            and drone_state.altitude_m < float(self.cfg.min_termination_altitude_m)
            and self.step_in_episode > int(self.cfg.ignore_obstacle_termination_first_steps)
        ):
            done = True
            term_reason = "altitude_too_low"

        if not done and drone_state.altitude_m > float(self.cfg.max_termination_altitude_m):
            done = True
            term_reason = "altitude_too_high"

        if not done and self.step_in_episode >= int(self.cfg.max_episode_steps):
            done = True
            term_reason = "episode_timeout"

        self._ep_return += float(reward)
        self._prev_action[:] = safe_action
        self.step_in_episode += 1

        total = max(1, self.step_in_episode)
        match_pct = 100.0 * self._ep_match / total
        pred_pct = 100.0 * self._ep_pred / total
        none_pct = 100.0 * self._ep_none / total
        safety_intervention_rate = 100.0 * self._safety_interventions / total

        if self.cfg.print_obstacle_debug and self.step_in_episode % int(self.cfg.obstacle_debug_every_n_steps) == 0:
            print(
                "[OBST] "
                f"F={obstacle_dict['front_dist_m']:.2f} "
                f"FL={obstacle_dict['front_left_dist_m']:.2f} "
                f"FR={obstacle_dict['front_right_dist_m']:.2f} "
                f"L={obstacle_dict['left_dist_m']:.2f} "
                f"R={obstacle_dict['right_dist_m']:.2f} "
                f"B={obstacle_dict['back_dist_m']:.2f} "
                f"D={obstacle_dict['down_dist_m']:.2f} "
                f"HMIN={obstacle_dict['min_obstacle_dist_m']:.2f} "
                f"ALT={drone_state.altitude_m:.2f}"
            )

        if self.cfg.show_cv_window:
            self.tracker.draw(frame, fps_show)

            safety_active = bool(safety_info.get("safety_intervention", False))
            safety_reasons = safety_info.get("safety_reasons", [])
            safety_reason_text = ",".join(safety_reasons[:2]) if safety_reasons else "none"

            # Safety state label:
            # CLEAR     = no safety correction was applied
            # ACTIVE    = safety layer modified the action
            # EMERGENCY = very close horizontal obstacle or very close down distance
            if horizontal_min_obst < float(self.cfg.obstacle_emergency_dist_m) or obstacle_dict["down_dist_m"] < float(self.cfg.obstacle_emergency_dist_m):
                safety_state = "EMERGENCY"
                safety_color = (0, 0, 255)
            elif safety_active:
                safety_state = "ACTIVE"
                safety_color = (0, 200, 255)
            else:
                safety_state = "CLEAR"
                safety_color = (0, 255, 0)

            cv2.putText(frame, f"MODE: {getattr(self.tracker, 'last_mode', '?')}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps_show}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"ALT: {drone_state.altitude_m:.2f}m", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"HMIN: {horizontal_min_obst:.2f}m D:{obstacle_dict['down_dist_m']:.2f}m", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"SAFETY STATE: {safety_state}", (20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, safety_color, 2)
            cv2.putText(frame, f"SAFETY RATE: {safety_intervention_rate:.1f}% REASON: {safety_reason_text}", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, safety_color, 2)

            cv2.imshow("Tracker Debug", frame)
            cv2.waitKey(1)

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
                f"SAFETY={safety_intervention_rate:.1f}% "
                f"R_center={reward_parts.get('center_reward', 0.0):+.2f} "
                f"R_dist={reward_parts.get('distance_reward', 0.0):+.2f} "
                f"R_obst={reward_parts.get('obstacle_penalty', 0.0):+.2f} "
                f"R_smooth={reward_parts.get('smooth_follow_reward', 0.0):+.2f} "
                f"dur={dur:.1f}s reason={term_reason}"
            )

        info = {
            "episode_id": self.episode_id,
            "step_in_episode": self.step_in_episode,
            "termination_reason": term_reason,
            "focus_streak_s": float(self._focus_streak),
            "global_max_focus_s": float(self._global_max_focus),
            "min_obstacle_dist_m": float(horizontal_min_obst),
            "down_dist_m": float(obstacle_dict["down_dist_m"]),
            "alt_agl_m": float(drone_state.altitude_m),
            "match_pct": float(match_pct),
            "pred_pct": float(pred_pct),
            "none_pct": float(none_pct),
            "safety_intervention_rate_pct": float(safety_intervention_rate),
            "safety_intervention": bool(safety_info.get("safety_intervention", False)),
            "safety_reasons": safety_info.get("safety_reasons", []),
            "raw_action": raw_action.copy(),
            "safe_action": safe_action.copy(),
            "reward_parts": reward_parts,
            "obs_dict": obs_dict,
            "episode_done": bool(done),
        }

        return obs, float(reward), bool(done), False, info
