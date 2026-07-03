import time
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
import cosysairsim as airsim

from object_tracker import tracker
from tracking.target_tracker_manager import TargetTrackerManager
from weights_config import EnvConfig

from observation_builder import (
    ObservationBuilder,
    ObservationBuilderConfig,
    BBox,
    DroneState,
    ObstacleState,
)
from safety_filter import SafetyConfig, safety_filter
from lidar_processor import LidarProcessor, LidarProcessorConfig, point_cloud_to_array
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

        # ------------------------------------------------------------------
        # Moving target car actor.
        #
        # IMPORTANT:
        # This must match the Actor instance name returned by:
        # client.simListSceneObjects(".*")
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Moving target car actor and fixed start marker.
        #
        # IMPORTANT:
        # train_target_car must match the moving car Actor name returned by:
        # client.simListSceneObjects(".*")
        #
        # car_start_marker must match the empty Actor marker in Unreal.
        # The marker is used as the fixed reset pose, so the car can return
        # to the correct start position even if Play was running for a long time
        # before the Python training process started.
        # ------------------------------------------------------------------
        self.train_target_car = "BP_X6M_C_1"
        self.car_start_marker = "Actor_1"

        self.car_start_pose = self.client.simGetObjectPose(self.car_start_marker)

        if (
            self.car_start_pose is None
            or not np.isfinite(float(self.car_start_pose.position.x_val))
            or not np.isfinite(float(self.car_start_pose.position.y_val))
            or not np.isfinite(float(self.car_start_pose.position.z_val))
        ):
            raise RuntimeError(
                f"[ENV INIT] Failed to read start pose marker: {self.car_start_marker}. "
                "Make sure the marker Actor exists in Unreal and is visible to simListSceneObjects."
            )

        print(f"[ENV INIT] target car actor: {self.train_target_car}")
        print(f"[ENV INIT] target car start marker: {self.car_start_marker}")
        print("[ENV INIT] target car marker start pose:", self.car_start_pose)
        # Full dual-camera visual tracking.
        # self.tracker remains the front tracker for backward compatibility.
        self.tracker = tracker()
        self.front_tracker = self.tracker
        self.tracker_manager = self._create_tracker_manager()

        # Independent bottom-camera tracker. This uses the same target
        # fingerprint/class, but it owns its own runtime state and Kalman bridge.
        self.bottom_tracker = tracker()
        self.bottom_tracker_manager = self._create_tracker_manager()

        self._tracking_result = None
        self._stable_bbox_xyxy = None
        self._bottom_tracking_result = None
        self._bottom_stable_bbox_xyxy = None
        self._bottom_last_trusted_raw_bbox_xyxy = None
        self._bottom_kalman_fallback_active = False

        self.lidar_processor = LidarProcessor(
            LidarProcessorConfig(
                max_range_m=float(self.cfg.lidar_max_dist_m),
                min_valid_range_m=float(getattr(self.cfg, "lidar_min_valid_range_m", 0.05)),
                horizontal_abs_z_max_m=float(getattr(self.cfg, "lidar_horizontal_abs_z_max_m", 1.75)),
                down_xy_radius_m=float(getattr(self.cfg, "lidar_down_xy_radius_m", 2.0)),
                down_min_z_m=float(getattr(self.cfg, "lidar_down_min_z_m", 0.05)),
                self_ignore_radius_m=float(getattr(self.cfg, "lidar_self_ignore_radius_m", 0.75)),
            )
        )

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
            takeoff_zero_horizontal_motion=bool(getattr(self.cfg, "safety_takeoff_zero_horizontal_motion", True)),
            takeoff_force_climb=bool(getattr(self.cfg, "safety_takeoff_force_climb", False)),
            landing_allow_down_proximity=bool(getattr(self.cfg, "safety_landing_allow_down_proximity", True)),
            landing_max_descent_action_near_ground=float(getattr(self.cfg, "safety_landing_max_descent_action_near_ground", 0.18)),
            enabled=bool(self.cfg.safety_enabled),
        )

        self.reward_config = FollowRewardConfig(
            w_center=float(self.cfg.w_center),
            center_reward_alpha=float(self.cfg.center_reward_alpha),
            w_distance=float(self.cfg.w_distance),
            desired_distance_proxy=float(self.cfg.desired_distance_proxy),
            distance_tolerance=float(self.cfg.distance_tolerance),
            w_visibility=float(self.cfg.w_visibility),
            w_lost_target=float(self.cfg.w_lost_target),
            w_altitude_safe=float(self.cfg.w_altitude_safe),
            w_altitude_low_penalty=float(self.cfg.w_altitude_low_penalty),
            w_altitude_high_penalty=float(self.cfg.w_altitude_high_penalty),
            min_safe_altitude_m=float(self.cfg.min_safe_altitude_m),
            max_safe_altitude_m=float(self.cfg.max_safe_altitude_m),
            w_smooth_follow=float(self.cfg.w_smooth_follow),
            max_img_motion=float(self.cfg.max_img_motion),
            w_control=float(self.cfg.w_control),
            w_action_delta=float(self.cfg.w_action_delta),
            w_slow=float(self.cfg.w_slow),
            w_time=float(self.cfg.w_time),
            w_obstacle=float(self.cfg.w_obstacle),
            w_safety_intervention=float(self.cfg.w_safety_intervention),
            collision_penalty=float(self.cfg.penalty_collision),
            timeout_penalty=float(self.cfg.penalty_timeout),
            altitude_termination_penalty=float(self.cfg.penalty_altitude_termination),
            handoff_success_bonus=float(getattr(self.cfg, "handoff_success_bonus", 35.0)),
            w_recovery_yaw=float(getattr(self.cfg, "w_recovery_yaw", 2.5)),
            w_wrong_recovery_yaw=float(getattr(self.cfg, "w_wrong_recovery_yaw", 3.5)),
            w_yaw_when_focused_centered=float(getattr(self.cfg, "w_yaw_when_focused_centered", 150.0)),
            focused_centered_yaw_threshold=float(getattr(self.cfg, "focused_centered_yaw_threshold", 0.035)),
            focused_center_threshold=float(getattr(self.cfg, "focused_center_threshold", 0.16)),
            focused_center_x_threshold=float(getattr(self.cfg, "focused_center_x_threshold", 0.11)),
            focused_center_y_threshold=float(getattr(self.cfg, "focused_center_y_threshold", 0.16)),
            focused_bbox_conf_threshold=float(getattr(self.cfg, "focused_bbox_conf_threshold", 0.55)),
            focused_centered_yaw_power=float(getattr(self.cfg, "focused_centered_yaw_power", 1.25)),
            w_exp_focused_centered_yaw=float(getattr(self.cfg, "w_exp_focused_centered_yaw", 260.0)),
            exp_focused_yaw_gain=float(getattr(self.cfg, "exp_focused_yaw_gain", 5.0)),
            exp_focused_yaw_clip=float(getattr(self.cfg, "exp_focused_yaw_clip", 0.55)),
            w_yaw_decenter_exp_penalty=float(getattr(self.cfg, "w_yaw_decenter_exp_penalty", 340.0)),
            yaw_decenter_exp_gain=float(getattr(self.cfg, "yaw_decenter_exp_gain", 9.0)),
            yaw_decenter_error_clip=float(getattr(self.cfg, "yaw_decenter_error_clip", 0.18)),
            yaw_decenter_prev_center_threshold=float(getattr(self.cfg, "yaw_decenter_prev_center_threshold", 0.18)),
            yaw_decenter_current_max_threshold=float(getattr(self.cfg, "yaw_decenter_current_max_threshold", 0.45)),
            w_yaw_command_budget_penalty=float(getattr(self.cfg, "w_yaw_command_budget_penalty", 0.025)),
            yaw_command_budget_free_deg=float(getattr(self.cfg, "yaw_command_budget_free_deg", 25.0)),
            yaw_command_budget_clip_deg=float(getattr(self.cfg, "yaw_command_budget_clip_deg", 120.0)),
            chase_pressure_enabled=bool(getattr(self.cfg, "chase_pressure_enabled", True)),
            chase_pressure_bottom_disable=bool(getattr(self.cfg, "chase_pressure_bottom_disable", True)),
            chase_pressure_goal_distance_m=float(getattr(self.cfg, "chase_pressure_goal_distance_m", 7.5)),
            chase_pressure_far_clip_m=float(getattr(self.cfg, "chase_pressure_far_clip_m", 8.0)),
            chase_pressure_center_threshold=float(getattr(self.cfg, "chase_pressure_center_threshold", 0.28)),
            chase_pressure_no_improvement_deadband_m=float(getattr(self.cfg, "chase_pressure_no_improvement_deadband_m", 0.03)),
            chase_pressure_min_forward_action=float(getattr(self.cfg, "chase_pressure_min_forward_action", 0.10)),
            w_visible_far_no_approach_penalty=float(getattr(self.cfg, "w_visible_far_no_approach_penalty", 12.0)),
            w_centered_forward_chase_bonus=float(getattr(self.cfg, "w_centered_forward_chase_bonus", 22.0)),
            w_centered_idle_far_penalty=float(getattr(self.cfg, "w_centered_idle_far_penalty", 18.0)),
            w_retreat_while_far_penalty=float(getattr(self.cfg, "w_retreat_while_far_penalty", 16.0)),
            w_fast_chase_throttle_bonus=float(getattr(self.cfg, "w_fast_chase_throttle_bonus", 42.0)),
            w_fast_chase_progress_bonus=float(getattr(self.cfg, "w_fast_chase_progress_bonus", 70.0)),
            w_fast_chase_slow_penalty=float(getattr(self.cfg, "w_fast_chase_slow_penalty", 36.0)),
            fast_chase_far_distance_m=float(getattr(self.cfg, "fast_chase_far_distance_m", 2.50)),
            fast_chase_min_vx_action=float(getattr(self.cfg, "fast_chase_min_vx_action", 0.70)),
            w_bottom_velocity_error_penalty=float(getattr(self.cfg, "w_bottom_velocity_error_penalty", 18.0)),
            w_bottom_velocity_progress_bonus=float(getattr(self.cfg, "w_bottom_velocity_progress_bonus", 16.0)),
            w_bottom_velocity_ready_bonus=float(getattr(self.cfg, "w_bottom_velocity_ready_bonus", 28.0)),
            fine_position_reward_enabled=bool(getattr(self.cfg, "fine_position_reward_enabled", True)),
            fine_position_error_ready=float(getattr(self.cfg, "fine_position_error_ready", 0.08)),
            fine_position_error_good=float(getattr(self.cfg, "fine_position_error_good", 0.14)),
            fine_position_bonus_weight=float(getattr(self.cfg, "fine_position_bonus_weight", 95.0)),
            fine_position_bonus_gain=float(getattr(self.cfg, "fine_position_bonus_gain", 42.0)),
            fine_position_penalty_weight=float(getattr(self.cfg, "fine_position_penalty_weight", 65.0)),
            fine_position_penalty_power=float(getattr(self.cfg, "fine_position_penalty_power", 1.75)),
            fine_position_ready_bonus=float(getattr(self.cfg, "fine_position_ready_bonus", 85.0)),
            fine_position_landing_ready_bonus=float(getattr(self.cfg, "fine_position_landing_ready_bonus", 140.0)),
            fine_position_use_bbox_relative=bool(getattr(self.cfg, "fine_position_use_bbox_relative", True)),
            fine_position_bbox_safe_rel=float(getattr(self.cfg, "fine_position_bbox_safe_rel", 0.50)),
            fine_position_bbox_edge_rel=float(getattr(self.cfg, "fine_position_bbox_edge_rel", 1.00)),
            fine_position_bbox_ready_rel=float(getattr(self.cfg, "fine_position_bbox_ready_rel", 0.25)),
            fine_position_bbox_bonus_weight=float(getattr(self.cfg, "fine_position_bbox_bonus_weight", 75.0)),
            fine_position_bbox_bonus_gain=float(getattr(self.cfg, "fine_position_bbox_bonus_gain", 4.0)),
            fine_position_bbox_inside_penalty_weight=float(getattr(self.cfg, "fine_position_bbox_inside_penalty_weight", 35.0)),
            fine_position_bbox_edge_penalty_weight=float(getattr(self.cfg, "fine_position_bbox_edge_penalty_weight", 95.0)),
            fine_position_bbox_outside_penalty_weight=float(getattr(self.cfg, "fine_position_bbox_outside_penalty_weight", 260.0)),
            fine_position_bbox_ready_bonus=float(getattr(self.cfg, "fine_position_bbox_ready_bonus", 90.0)),
            fine_position_bbox_landing_ready_bonus=float(getattr(self.cfg, "fine_position_bbox_landing_ready_bonus", 160.0)),
            bottom_alignment_reward_enabled=bool(getattr(self.cfg, "bottom_alignment_reward_enabled", True)),
            bottom_alignment_y_target_abs=float(getattr(self.cfg, "bottom_alignment_y_target_abs", 0.12)),
            bottom_alignment_center_target=float(getattr(self.cfg, "bottom_alignment_center_target", 0.18)),
            w_bottom_y_error_penalty=float(getattr(self.cfg, "w_bottom_y_error_penalty", 38.0)),
            w_bottom_y_progress_reward=float(getattr(self.cfg, "w_bottom_y_progress_reward", 24.0)),
            w_bottom_y_regress_penalty=float(getattr(self.cfg, "w_bottom_y_regress_penalty", 30.0)),
            w_bottom_center_ready_bonus=float(getattr(self.cfg, "w_bottom_center_ready_bonus", 18.0)),
            w_bottom_wrong_vx_penalty=float(getattr(self.cfg, "w_bottom_wrong_vx_penalty", 22.0)),
            w_bottom_correct_vx_bonus=float(getattr(self.cfg, "w_bottom_correct_vx_bonus", 14.0)),
            bottom_alignment_progress_deadband=float(getattr(self.cfg, "bottom_alignment_progress_deadband", 0.015)),
            w_fast_reacquire=float(getattr(self.cfg, "w_fast_reacquire", 18.0)),
            w_overshoot_penalty=float(getattr(self.cfg, "w_overshoot_penalty", 10.0)),
            w_lost_time_accel_penalty=float(getattr(self.cfg, "w_lost_time_accel_penalty", 4.0)),
        )

        self.target_fingerprint = None
        self.target_class_id = None
        self._target_initialized = False

        # Runtime diagnostics for full dual-tracker verification.
        self._front_full_tracker_mode = "LOST"
        self._front_full_tracker_raw_mode = "NONE"
        self._front_full_tracker_updates = 0
        self._front_full_tracker_match_frames = 0
        self._front_full_tracker_pred_frames = 0
        self._front_full_tracker_lost_frames = 0

        self._fps_ema = 0.0
        self.episode_id = 0
        self.step_in_episode = 0

        self._focus_streak = 0.0
        self._global_max_focus = 0.0
        self._ep_max_focus = 0.0
        self._ep_return = 0.0
        self._ep_reward_part_sums = {}
        self._ep_reward_part_last = {}
        self._ep_reward_part_counts = {}
        self._ep_match = 0
        self._ep_pred = 0
        self._ep_none = 0
        self._ep_start = time.time()

        self._last_min_obst = None
        self._last_obstacle_state = None

        self._prev_action = np.zeros(4, dtype=np.float32)
        self._last_raw_action = np.zeros(4, dtype=np.float32)
        self._last_safe_action = np.zeros(4, dtype=np.float32)
        self._prev_pitch_rad_for_smooth = 0.0
        self._last_pitch_deg = 0.0
        self._last_pitch_rate_dps = 0.0
        self._last_pitch_smoothness_penalty = 0.0
        self._prev_vx_cmd_for_slew = 0.0
        self._prev_vy_cmd_for_slew = 0.0
        self._last_vx_slew_delta = 0.0
        self._last_vx_slew_limited = False
        self._last_vy_slew_delta = 0.0
        self._last_vy_slew_limited = False

        self._safety_interventions = 0
        self._focused_yaw_hard_events = 0
        self._yaw_decenter_hard_events = 0
        self._last_focused_yaw_hard_penalty = 0.0
        self._last_distance_proxy_norm = 1.0

        # Previous-step tracking state used for recovery reward shaping.
        self._prev_tracking_mode = "LOST"
        self._prev_had_target = False
        self._prev_lost_target_time_norm = 1.0
        self._prev_distance_proxy_norm = 1.0
        self._prev_err_x = 0.0
        self._prev_err_y = 0.0
        self._prev_bbox_conf = 0.0
        self._prev_bottom_abs_err_y = 1.5

        self._front_full_tracker_mode = "LOST"
        self._front_full_tracker_raw_mode = "NONE"
        self._front_full_tracker_updates = 0
        self._front_full_tracker_match_frames = 0
        self._front_full_tracker_pred_frames = 0
        self._front_full_tracker_lost_frames = 0

        self._yaw_shield_applied = False
        self._yaw_shield_events = 0
        self._last_yaw_shield_pre = 0.0
        self._last_yaw_shield_post = 0.0
        self._yaw_to_strafe_events = 0
        self._last_yaw_to_strafe_delta_vy = 0.0
        self._agent1_yaw_lock_events = 0
        self._agent1_yaw_lock_applied = False
        self._agent1_yaw_limit_events = 0
        self._agent1_yaw_limit_applied = False
        self._last_blocked_yaw_action = 0.0
        self._last_yaw_lock_penalty = 0.0
        self._last_yaw_allowed_abs = 1.0

        # Hard-fail timers for chase/center fine-tuning.
        # These are intentionally strict: the policy must stay in MATCH,
        # keep approaching when far, and avoid large yaw drift.
        self._non_match_time_s = 0.0
        self._not_approaching_time_s = 0.0
        self._chase_rule_elapsed_s = 0.0
        self._best_distance_proxy_norm = 1.0
        self._best_chase_distance_m = float("inf")
        self._current_chase_distance_m = float("inf")
        self._prev_chase_distance_m = float("inf")
        self._initial_chase_distance_m = float("inf")
        self._distance_damage_time_s = 0.0
        self._distance_damage_ratio = 0.0

        # Multi-camera soft handoff state.
        self._bottom_match = False
        self._bottom_match_streak = 0
        self._bottom_bbox_xyxy = None
        self._bottom_similarity = 0.0
        self._bottom_full_tracker_mode = "LOST"
        self._bottom_full_tracker_raw_mode = "NONE"
        self._bottom_full_tracker_confidence = 0.0
        self._bottom_full_tracker_similarity = 0.0
        self._bottom_full_tracker_updates = 0
        self._bottom_full_tracker_match_frames = 0
        self._bottom_full_tracker_pred_frames = 0
        self._bottom_full_tracker_lost_frames = 0
        self._bottom_candidate_count = 0
        self._bottom_candidate_scan_score = 0.0
        self._bottom_candidate_scan_used = False
        self._bottom_relock_attempts = 0
        self._bottom_err_x = 0.0
        self._bottom_err_y = 0.0
        self._prev_bottom_abs_err_y = 1.5
        self._bottom_weight = 0.0
        self._front_weight = 1.0
        self._bottom_confirmed = False
        self._bottom_match_fresh = False
        self._bottom_bbox_area_norm = 0.0
        self._bottom_bbox_rel_half_w = 0.0
        self._bottom_bbox_rel_half_h = 0.0
        self._bottom_bbox_rel_err_x = 0.0
        self._bottom_bbox_rel_err_y = 0.0
        self._bottom_bbox_rel_err = 999.0
        self._last_bottom_match_step = -999999
        self._handoff_candidate_gate = False
        self._handoff_ready = False
        self._handoff_phase = "CHASE_FRONT"
        self._handoff_visual_score = 0.0
        self._handoff_visual_scan_trigger = False
        self._handoff_visual_lidar_trigger = False
        self._handoff_lidar_dist_m = float("inf")
        self._handoff_lidar_direction = "none"
        self._last_bottom_scan_step = -999999

        # Dual-camera fusion state.
        self._camera_authority = "FRONT_PRIMARY"
        self._active_camera = "front"
        self._fusion_has_target = False
        self._fusion_tracking_mode = "LOST"
        self._fusion_err_x = 0.0
        self._fusion_err_y = 0.0
        self._fusion_confidence = 0.0
        self._bottom_authority_streak = 0
        self._front_authority_streak = 0
        self._last_bottom_valid_step = -999999
        self._last_front_valid_step = -999999
        self._last_bottom_alignment_assist_vx = 0.0
        self._last_bottom_alignment_assist_vy = 0.0
        self._bottom_alignment_assist_events = 0

        self._speed_stage = "CHASE_FAST"
        self._speed_stage_changes = 0
        self._last_fast_chase_assist_vx = 0.0
        self._last_stage_vx_scale = float(getattr(self.cfg, "vx_scale", 2.8))
        self._last_stage_vy_scale = float(getattr(self.cfg, "vy_scale", 2.0))
        self._last_bottom_pd_assist_vx = 0.0
        self._last_bottom_pd_assist_vy = 0.0
        self._bottom_prev_err_x_for_velocity = 0.0
        self._bottom_prev_err_y_for_velocity = 0.0
        self._bottom_img_vel_x = 0.0
        self._bottom_img_vel_y = 0.0
        self._prev_bottom_img_speed = 0.0
        self._bottom_velocity_ready = False
        self._bottom_velocity_ready_streak = 0

        self._last_approach_improvement_m = 0.0
        self._step_chase_distance_improvement_m = 0.0
        self._initial_yaw_rad = None
        self._last_yaw_delta_deg = 0.0
        self._commanded_yaw_delta_deg = 0.0
        self._cumulative_abs_yaw_cmd_deg = 0.0
        self._last_yaw_rate_cmd_dps = 0.0
        self._effective_yaw_delta_deg = 0.0

    @staticmethod
    def _deg(rad: float) -> float:
        return float(rad * 180.0 / np.pi)

    @staticmethod
    def _clip(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, float(value)))

    @staticmethod
    def _wrap_angle_rad(angle_rad: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return float((float(angle_rad) + np.pi) % (2.0 * np.pi) - np.pi)


    def _compute_safety_phase(self, drone_state: DroneState, obstacle_dict: dict, tracking_mode: str | None = None) -> str:
        """Return the current phase for state-aware LiDAR safety.

        Safety is never disabled. The phase only changes how the same LiDAR
        facts are interpreted:
            TAKEOFF -> ground below is expected; climb and avoid horizontal motion.
            CHASE   -> normal flight: all horizontal/down safety rules active.
            LANDING -> horizontal rules active; down proximity is allowed/limited.
        """
        alt_m = float(getattr(drone_state, "altitude_m", 0.0) or 0.0)
        down_m = float(obstacle_dict.get("down_dist_m", alt_m))
        clear_alt = float(getattr(self.cfg, "safety_takeoff_clear_altitude_m", 1.20))
        clear_steps = int(getattr(self.cfg, "safety_takeoff_clear_steps", 45))

        # In the current chase/follow training we do not train takeoff. If reset
        # already placed the drone in the intended airborne initial condition,
        # do not let noisy LiDAR-down/self returns keep the episode in TAKEOFF
        # forever. Safety still runs in CHASE; it just will not command an
        # unsolicited climb because of down proximity.
        if bool(getattr(self.cfg, "force_chase_after_airborne_reset", True)) and bool(getattr(self, "_airborne_reset_completed", False)):
            landing_distance_m = float(getattr(self.cfg, "safety_landing_distance_m", 2.50))
            current_chase_distance_m = self._get_drone_to_target_distance_m()
            if current_chase_distance_m is not None and np.isfinite(float(current_chase_distance_m)):
                if float(current_chase_distance_m) <= landing_distance_m:
                    return "LANDING"
            return "CHASE"

        # TAKEOFF is only active while the drone is actually near the ground.
        # Previously this phase was forced for the first N steps, which blocked
        # horizontal control even after reset/moveToZ had already placed the
        # drone at training altitude.
        #
        # Use altitude/down-range geometry, not a blind step counter.
        # In normal chase training the episode starts airborne, so the phase
        # should become CHASE immediately.
        if max(alt_m, down_m) < clear_alt:
            return "TAKEOFF"

        # Future landing phase hook. For now it activates only when the real
        # drone<->target distance is already small enough. Horizontal safety
        # remains active inside LANDING; only down proximity is interpreted as
        # expected touchdown geometry instead of an automatic emergency climb.
        landing_distance_m = float(getattr(self.cfg, "safety_landing_distance_m", 2.50))
        current_chase_distance_m = self._get_drone_to_target_distance_m()
        if current_chase_distance_m is not None and np.isfinite(float(current_chase_distance_m)):
            if float(current_chase_distance_m) <= landing_distance_m:
                return "LANDING"

        return "CHASE"

    def _force_airborne_start_pose(self) -> None:
        """Place the drone directly at the training altitude after reset.

        Chase/follow training starts from an airborne initial condition.  The
        diagnostic script confirmed that moveToZAsync(-5) works in this setup,
        so this helper now verifies the pose altitude and retries if needed.
        """
        target_alt_m = float(self.cfg.reset_takeoff_altitude_m)
        target_z_ned = -abs(target_alt_m)

        try:
            self.client.enableApiControl(True, vehicle_name=self.cfg.vehicle_name)
            self.client.armDisarm(True, vehicle_name=self.cfg.vehicle_name)

            # Wake SimpleFlight control.
            try:
                self.client.takeoffAsync(timeout_sec=4.0, vehicle_name=self.cfg.vehicle_name).join()
            except TypeError:
                self.client.takeoffAsync(vehicle_name=self.cfg.vehicle_name).join()
            except Exception:
                pass

            # Direct pose placement.
            pose = self.client.simGetVehiclePose(vehicle_name=self.cfg.vehicle_name)
            pose.position.z_val = target_z_ned
            self.client.simSetVehiclePose(pose, True, vehicle_name=self.cfg.vehicle_name)
            time.sleep(float(self.cfg.reset_settle_sec))

            # Controller stabilization at the same altitude.
            try:
                self.client.moveToZAsync(
                    target_z_ned,
                    float(self.cfg.reset_move_to_z_velocity),
                    timeout_sec=8.0,
                    vehicle_name=self.cfg.vehicle_name,
                ).join()
            except TypeError:
                self.client.moveToZAsync(
                    target_z_ned,
                    float(self.cfg.reset_move_to_z_velocity),
                    vehicle_name=self.cfg.vehicle_name,
                ).join()

            time.sleep(float(self.cfg.reset_settle_sec))

            # Stop residual motion without changing altitude.
            self.client.moveByVelocityBodyFrameAsync(
                vx=0.0,
                vy=0.0,
                vz=0.0,
                duration=0.20,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                vehicle_name=self.cfg.vehicle_name,
            ).join()

            time.sleep(float(self.cfg.reset_settle_sec))

            alt_now = self._get_alt_agl_m()
            self._airborne_reset_completed = bool(
                alt_now is not None
                and np.isfinite(float(alt_now))
                and float(alt_now) >= float(getattr(self.cfg, "reset_verify_altitude_min_m", 2.0))
            )

            if self.cfg.print_reset:
                pose_now = self.client.simGetVehiclePose(vehicle_name=self.cfg.vehicle_name)
                print(
                    "[RESET ALT] "
                    f"target_alt={target_alt_m:.2f}m "
                    f"pose_z={float(pose_now.position.z_val):+.3f} "
                    f"measured_alt={alt_now} "
                    f"airborne_ok={self._airborne_reset_completed}"
                )

        except Exception as e:
            self._airborne_reset_completed = False
            print(f"[RESET ALT WARNING] airborne reset failed: {e}")

    def _get_current_yaw_rad(self) -> float | None:
        """Read current drone yaw angle from AirSim, or None on failure."""
        try:
            ms = self.client.getMultirotorState(vehicle_name=self.cfg.vehicle_name)
            q = ms.kinematics_estimated.orientation
            _pitch_r, _roll_r, yaw_r = airsim.to_eularian_angles(q)
            return float(yaw_r)
        except Exception:
            return None

    def _get_drone_to_target_distance_m(self) -> float | None:
        """Return horizontal XY distance between the drone and the target car in meters.

        This is the strict chase metric. It is intentionally based on real
        simulator poses, not bbox size, because the policy must physically close
        the distance to the moving car.
        """
        try:
            drone_pose = self.client.simGetVehiclePose(vehicle_name=self.cfg.vehicle_name)
            car_pose = self.client.simGetObjectPose(self.train_target_car)

            if drone_pose is None or car_pose is None:
                return None

            dx = float(drone_pose.position.x_val) - float(car_pose.position.x_val)
            dy = float(drone_pose.position.y_val) - float(car_pose.position.y_val)

            dist = float(np.sqrt(dx * dx + dy * dy))
            if not np.isfinite(dist):
                return None
            return dist
        except Exception:
            return None

    @staticmethod
    def _format_reward_top_items(items, max_items: int = 8) -> str:
        """Format reward diagnostic items for compact one-line episode logs."""
        out = []
        for key, value in items[:max_items]:
            if isinstance(value, (int, float, np.floating)) and np.isfinite(float(value)):
                out.append(f"{key}={float(value):+.1f}")
        return ";".join(out) if out else "none"

    def _accumulate_reward_parts(self, reward_parts: dict) -> None:
        """Accumulate numeric reward parts over the whole episode."""
        if not isinstance(reward_parts, dict):
            return

        for key, value in reward_parts.items():
            if not isinstance(value, (int, float, np.floating)):
                continue
            value_f = float(value)
            if not np.isfinite(value_f):
                continue

            self._ep_reward_part_sums[key] = float(self._ep_reward_part_sums.get(key, 0.0) + value_f)
            self._ep_reward_part_last[key] = value_f
            self._ep_reward_part_counts[key] = int(self._ep_reward_part_counts.get(key, 0) + 1)

    def _reward_diagnostics_snapshot(self, final_reward: float) -> dict:
        """Build a compact reward accounting snapshot for episode summaries."""
        sums = dict(getattr(self, "_ep_reward_part_sums", {}) or {})

        # Exclude meta/debug fields that are not additive reward components.
        meta_prefixes = (
            "is_",
            "fine_position_stage",
            "fast_chase_active",
            "bottom_alignment_active",
            "strict_hard_fail_reason",
        )
        meta_exact = {
            "bottom_img_speed",
            "bottom_img_speed_improvement",
            "fine_position_error_inf",
            "fine_position_bbox_rel_err",
            "fine_position_bbox_rel_err_x",
            "fine_position_bbox_rel_err_y",
            "agent1_blocked_yaw_action",
            "takeoff_phase_neutral_reward",
        }

        numeric_items = []
        for key, value in sums.items():
            if key in meta_exact or any(key.startswith(prefix) for prefix in meta_prefixes):
                continue
            if not isinstance(value, (int, float, np.floating)):
                continue
            value_f = float(value)
            if not np.isfinite(value_f):
                continue
            # Avoid double-counting cumulative post-processed totals as normal reward parts.
            if key.startswith("total_reward_after_"):
                continue
            numeric_items.append((key, value_f))

        positives = sorted([(k, v) for k, v in numeric_items if v > 1e-6], key=lambda kv: kv[1], reverse=True)
        negatives = sorted([(k, v) for k, v in numeric_items if v < -1e-6], key=lambda kv: kv[1])

        # This sum is diagnostic only. Some reward_parts are informational and
        # some env-level post-processing modifies the final step reward.
        part_sum = float(sum(v for _, v in numeric_items))
        reward_diff = float(float(final_reward) - part_sum)

        terminal_keys = [
            "terminal_reason_penalty",
            "strict_hard_fail_penalty",
            "strict_hard_fail_cancelled_positive_return",
            "focused_yaw_hard_penalty",
            "focused_yaw_cancelled_positive_return",
            "takeoff_failed_penalty",
            "agent1_yaw_lock_penalty",
        ]
        terminal_items = [(k, float(sums.get(k, 0.0))) for k in terminal_keys if abs(float(sums.get(k, 0.0))) > 1e-6]

        return {
            "part_sum": part_sum,
            "reward_diff": reward_diff,
            "positive_top": positives,
            "negative_top": negatives,
            "terminal_items": terminal_items,
        }

    def _hard_fail_reward(self, reward: float, penalty: float) -> tuple[float, float]:
        """Cancel any positive episode return and apply a large failure penalty."""
        projected_return = float(self._ep_return + reward)
        if projected_return > 0.0:
            # Make final episode return exactly -penalty.
            return float(-self._ep_return - penalty), projected_return
        return float(reward - penalty), 0.0

    def _reset_target_car_debug(self, settle_sec: float = 0.10) -> None:
        """
        Reset the moving target car to the fixed TargetCarStart marker pose.
        """
        before_pose = self.client.simGetObjectPose(self.train_target_car)

        ok = self.client.simSetObjectPose(
            self.train_target_car,
            self.car_start_pose,
            teleport=True,
        )

        time.sleep(float(settle_sec))

        after_pose = self.client.simGetObjectPose(self.train_target_car)

        print(
            "[CAR RESET DEBUG] "
            f"ok={ok} "
            f"name={self.train_target_car} "
            f"marker={self.car_start_marker} "
            f"before=({before_pose.position.x_val:.2f}, "
            f"{before_pose.position.y_val:.2f}, "
            f"{before_pose.position.z_val:.2f}) "
            f"target=({self.car_start_pose.position.x_val:.2f}, "
            f"{self.car_start_pose.position.y_val:.2f}, "
            f"{self.car_start_pose.position.z_val:.2f}) "
            f"after=({after_pose.position.x_val:.2f}, "
            f"{after_pose.position.y_val:.2f}, "
            f"{after_pose.position.z_val:.2f})"
        )

        if not ok:
            raise RuntimeError(
                f"[CAR RESET DEBUG] simSetObjectPose returned False for actor '{self.train_target_car}'. "
                "Check the actor name and whether the target actor can be moved by AirSim."
            )
    def _get_frame(self, camera_name: str | None = None):
        """Read one RGB frame from an AirSim/Cosys-AirSim camera.

        Front camera is used by the current CHASE tracker.  The downward camera
        is now configurable and prepared for the next landing stage.
        """
        cam = str(camera_name if camera_name is not None else getattr(self.cfg, "front_camera_name", "0"))

        responses = self.client.simGetImages([
            airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False)
        ], vehicle_name=self.cfg.vehicle_name)

        if not responses or not responses[0].image_data_uint8:
            print(f"[DEBUG] No image received from camera '{cam}', returning black frame")
            return np.zeros((int(self.cfg.image_height), int(self.cfg.image_width), 3), dtype=np.uint8)

        response = responses[0]
        img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        frame = img.reshape(response.height, response.width, 3)

        # AirSim usually gives RGB, OpenCV displays BGR.
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame

    def _get_downward_frame(self):
        """Read the prepared downward camera for future landing/alignment mode."""
        return self._get_frame(camera_name=str(getattr(self.cfg, "downward_camera_name", "bottom_center")))

    def _bbox_xyxy_to_error(self, bbox_xyxy, frame_width: int, frame_height: int) -> tuple[float, float]:
        """Convert an XYXY bbox center into normalized image-space error."""
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        err_x = (cx - 0.5 * float(frame_width)) / max(1.0, 0.5 * float(frame_width))
        err_y = (cy - 0.5 * float(frame_height)) / max(1.0, 0.5 * float(frame_height))
        return float(np.clip(err_x, -1.5, 1.5)), float(np.clip(err_y, -1.5, 1.5))

    def _detect_bottom_target_by_fingerprint(self, downward_frame):
        """
        Detect the selected target in the downward camera.

        In the final dual-camera architecture this is a full bottom tracker
        update, not a lightweight scanner. The legacy scanner code below is kept
        as a fallback if dual_full_trackers_enabled is disabled.
        """
        if bool(getattr(self.cfg, "dual_full_trackers_enabled", True)):
            return self._update_bottom_full_tracker(
                downward_frame=downward_frame,
                dt=float(getattr(self.cfg, "cmd_duration_s", 0.10)),
            )

        result = {
            "match": False,
            "bbox_xyxy": None,
            "similarity": 0.0,
            "err_x": 0.0,
            "err_y": 0.0,
            "bbox_area_norm": 0.0,
            "candidate_count": 0,
        }

        try:
            core = getattr(self.tracker, "core", None)
            target_embedding = getattr(core, "target_embedding", None)
            if core is None or target_embedding is None:
                return result

            candidates = core._detect_candidates(downward_frame)
            result["candidate_count"] = int(len(candidates))

            target_class_id = getattr(self, "target_class_id", None)
            if target_class_id is not None:
                same_class = [c for c in candidates if int(c.cls_id) == int(target_class_id)]
                if same_class:
                    candidates = same_class

            best = None
            best_score = -1.0

            for cand in candidates:
                emb = core._embedding_from_bbox(downward_frame, cand.bbox)
                if emb is None:
                    continue

                # Both embeddings are normalized in resnet_yolo_tracker.
                score = float((target_embedding * emb).sum().detach().cpu().item())
                if score > best_score:
                    best_score = score
                    best = cand

            min_sim = float(getattr(self.cfg, "bottom_match_min_similarity", 0.45))
            if best is None or best_score < min_sim:
                result["similarity"] = max(0.0, float(best_score))
                return result

            x, y, w, h = [int(round(float(v))) for v in best.bbox[:4]]
            bbox_xyxy = [x, y, x + max(1, w), y + max(1, h)]
            h_img, w_img = downward_frame.shape[:2]
            err_x, err_y = self._bbox_xyxy_to_error(bbox_xyxy, w_img, h_img)
            bbox_area_norm = float((max(1, w) * max(1, h)) / max(1.0, float(w_img * h_img)))

            result.update(
                {
                    "match": True,
                    "bbox_xyxy": bbox_xyxy,
                    "similarity": float(best_score),
                    "err_x": float(err_x),
                    "err_y": float(err_y),
                    "bbox_area_norm": float(bbox_area_norm),
                }
            )
            return result

        except Exception as e:
            if bool(getattr(self.cfg, "print_sensor_errors", False)):
                print(f"[BOTTOM HANDOFF WARNING] bottom detection failed: {e}")
            return result

    def _compute_visual_lidar_handoff_trigger(self, obs_dict: dict, obstacle_dict: dict):
        """
        Combine front-camera geometry with LiDAR direction.

        The visual part asks:
            "Does the front bbox look like the target is getting too close /
             leaving the front view?"

        The LiDAR part asks:
            "Is there a close body in the same horizontal direction as the bbox?"

        This avoids switching to the bottom camera only from simulator ground-truth
        distance, and it also avoids switching only from noisy bbox aspect ratio.
        """
        result = {
            "trigger": False,
            "visual_score": 0.0,
            "lidar_dist_m": float("inf"),
            "lidar_direction": "none",
            "aspect_ratio": 0.0,
            "bbox_bottom": 0.0,
        }

        if not bool(getattr(self.cfg, "handoff_use_visual_lidar_trigger", True)):
            return result

        has_target = bool(float(obs_dict.get("has_target", 0.0)) > 0.5)
        if not has_target:
            return result

        bbox_w = float(obs_dict.get("bbox_w", 0.0))
        bbox_h = float(obs_dict.get("bbox_h", 0.0))
        bbox_area = float(obs_dict.get("bbox_area", 0.0))
        bbox_cy = float(obs_dict.get("bbox_cy", 0.0))
        err_x = float(obs_dict.get("err_x", 0.0))
        err_y = float(obs_dict.get("err_y", 0.0))
        area_delta = float(obs_dict.get("area_delta_norm", 0.0))

        aspect = bbox_w / max(1e-6, bbox_h)
        bbox_bottom = float(np.clip(bbox_cy + 0.5 * bbox_h, 0.0, 1.5))

        visual_score = 0.0

        if bbox_area >= float(getattr(self.cfg, "handoff_bbox_area_threshold", 0.070)):
            visual_score += 1.0

        if bbox_h >= float(getattr(self.cfg, "handoff_bbox_height_threshold", 0.32)):
            visual_score += 1.0

        if bbox_bottom >= float(getattr(self.cfg, "handoff_bbox_bottom_threshold", 0.84)):
            visual_score += 1.0

        if err_y >= float(getattr(self.cfg, "handoff_bbox_err_y_threshold", 0.45)):
            visual_score += 1.0

        if area_delta >= float(getattr(self.cfg, "handoff_bbox_area_delta_threshold", 0.12)):
            visual_score += 0.5

        aspect_min = float(getattr(self.cfg, "handoff_aspect_ratio_min", 0.65))
        aspect_max = float(getattr(self.cfg, "handoff_aspect_ratio_max", 3.20))
        if aspect <= aspect_min or aspect >= aspect_max:
            visual_score += 0.5

        # Pick the LiDAR sector that matches the horizontal direction of the bbox.
        x_deadband = float(getattr(self.cfg, "handoff_lidar_x_deadband", 0.22))
        if err_x < -x_deadband:
            lidar_direction = "front_left"
            lidar_dist = min(
                float(obstacle_dict.get("front_left_dist_m", float("inf"))),
                float(obstacle_dict.get("left_dist_m", float("inf"))),
            )
        elif err_x > x_deadband:
            lidar_direction = "front_right"
            lidar_dist = min(
                float(obstacle_dict.get("front_right_dist_m", float("inf"))),
                float(obstacle_dict.get("right_dist_m", float("inf"))),
            )
        else:
            lidar_direction = "front"
            lidar_dist = min(
                float(obstacle_dict.get("front_dist_m", float("inf"))),
                float(obstacle_dict.get("front_left_dist_m", float("inf"))),
                float(obstacle_dict.get("front_right_dist_m", float("inf"))),
            )

        lidar_close = bool(lidar_dist <= float(getattr(self.cfg, "handoff_lidar_close_dist_m", 7.0)))
        enough_visual = bool(visual_score >= float(getattr(self.cfg, "handoff_visual_score_threshold", 2.5)))
        trigger = bool(lidar_close and enough_visual)

        result.update(
            {
                "trigger": trigger,
                "visual_score": float(visual_score),
                "lidar_dist_m": float(lidar_dist),
                "lidar_direction": lidar_direction,
                "aspect_ratio": float(aspect),
                "bbox_bottom": float(bbox_bottom),
            }
        )
        return result

    def _update_soft_handoff_state(self, current_distance_m, front_tracking_mode: str, obs_dict: dict, obstacle_dict: dict | None = None):
        """
        Update front/bottom perception weights for soft handoff.

        The front camera is never switched off. The bottom camera starts
        scanning near the target and gains weight only when it confirms the
        same target fingerprint for several frames.
        """
        if not bool(getattr(self.cfg, "bottom_handoff_enabled", True)):
            self._bottom_match = False
            self._bottom_confirmed = False
            self._bottom_match_fresh = False
            self._bottom_bbox_area_norm = 0.0
            self._last_bottom_match_step = -999999
            self._handoff_candidate_gate = False
            self._bottom_weight = 0.0
            self._front_weight = 1.0
            self._handoff_ready = False
            self._handoff_phase = "CHASE_FRONT"
            return

        dist = float(current_distance_m) if current_distance_m is not None and np.isfinite(float(current_distance_m)) else float("inf")
        scan_distance = float(getattr(self.cfg, "bottom_scan_distance_m", 6.0))
        update_every = max(1, int(getattr(self.cfg, "bottom_scan_every_n_steps", 3)))

        visual_lidar = self._compute_visual_lidar_handoff_trigger(
            obs_dict=obs_dict,
            obstacle_dict=obstacle_dict or {},
        )
        self._handoff_visual_score = float(visual_lidar.get("visual_score", 0.0))
        self._handoff_visual_lidar_trigger = bool(visual_lidar.get("trigger", False))
        self._handoff_lidar_dist_m = float(visual_lidar.get("lidar_dist_m", float("inf")))
        self._handoff_lidar_direction = str(visual_lidar.get("lidar_direction", "none"))
        self._handoff_visual_scan_trigger = bool(
            self._handoff_visual_score >= float(getattr(self.cfg, "handoff_visual_scan_score_threshold", 2.5))
        )

        self._handoff_candidate_gate = bool(
            dist <= scan_distance
            or bool(self._handoff_visual_lidar_trigger)
            or bool(self._handoff_visual_scan_trigger)
        )

        # Important separation:
        #   - candidate_gate opens bottom scanning in the old handoff path.
        #   - dual_camera_fusion_enabled makes the bottom tracker run in parallel
        #     even before a hard handoff gate is open.
        fusion_enabled = bool(getattr(self.cfg, "dual_camera_fusion_enabled", True))
        full_bottom_enabled = bool(getattr(self.cfg, "dual_full_trackers_enabled", True))
        fusion_update_every = max(1, int(getattr(self.cfg, "dual_camera_fusion_scan_every_n_steps", update_every)))
        bottom_full_update_every = max(1, int(getattr(self.cfg, "bottom_full_tracker_update_every_n_steps", fusion_update_every)))

        should_scan = bool(
            (
                self._handoff_candidate_gate
                and self.step_in_episode - int(getattr(self, "_last_bottom_scan_step", -999999)) >= update_every
            )
            or (
                fusion_enabled
                and self.step_in_episode - int(getattr(self, "_last_bottom_scan_step", -999999)) >= fusion_update_every
            )
            or (
                full_bottom_enabled
                and self.step_in_episode - int(getattr(self, "_last_bottom_scan_step", -999999)) >= bottom_full_update_every
            )
        )

        if (not self._handoff_candidate_gate) and (not bool(getattr(self.cfg, "dual_camera_fusion_enabled", True))):
            # Do not let a stale or accidental bottom detection promote a handoff
            # while the front visual/LiDAR/distance gate says the target is still ahead.
            self._bottom_match = False
            self._bottom_confirmed = False
            self._bottom_match_streak = 0
            self._bottom_weight = 0.0
            self._front_weight = 1.0

        if should_scan:
            self._last_bottom_scan_step = int(self.step_in_episode)
            try:
                down_frame = self._get_downward_frame()
                self._cached_downward_frame = down_frame
                self._cached_downward_frame_step = int(self.step_in_episode)
                bottom = self._detect_bottom_target_by_fingerprint(down_frame)
            except Exception as e:
                bottom = {
                    "match": False,
                    "bbox_xyxy": None,
                    "similarity": 0.0,
                    "err_x": 0.0,
                    "err_y": 0.0,
                    "bbox_area_norm": 0.0,
                    "mode": "LOST",
                    "raw_mode": "ERROR",
                    "tracker_confidence": 0.0,
                }
                if bool(getattr(self.cfg, "print_sensor_errors", False)):
                    print(f"[BOTTOM HANDOFF WARNING] bottom frame read failed: {e}")

            self._bottom_match = bool(bottom.get("match", False))
            self._bottom_bbox_xyxy = bottom.get("bbox_xyxy", None)
            self._bottom_similarity = float(bottom.get("similarity", 0.0) or 0.0)
            self._bottom_err_x = float(bottom.get("err_x", 0.0) or 0.0)
            self._bottom_err_y = float(bottom.get("err_y", 0.0) or 0.0)
            self._bottom_bbox_area_norm = float(bottom.get("bbox_area_norm", 0.0) or 0.0)

            # BBox-relative center error:
            # Measures where the bottom-camera frame center falls relative to
            # the target bbox. This is scale-adaptive and object-shape-aware.
            self._bottom_bbox_rel_half_w = 0.0
            self._bottom_bbox_rel_half_h = 0.0
            self._bottom_bbox_rel_err_x = 0.0
            self._bottom_bbox_rel_err_y = 0.0
            self._bottom_bbox_rel_err = 999.0
            try:
                bbox_for_rel = self._bottom_bbox_xyxy
                if bbox_for_rel is not None:
                    bx1, by1, bx2, by2 = [float(v) for v in bbox_for_rel[:4]]
                    if "down_frame" in locals() and down_frame is not None:
                        img_h_rel, img_w_rel = down_frame.shape[:2]
                    else:
                        img_w_rel = 960
                        img_h_rel = 540

                    bbox_w_rel = max(1.0, bx2 - bx1)
                    bbox_h_rel = max(1.0, by2 - by1)

                    # err_x/err_y are normalized by half image size. Therefore
                    # bbox half-size must be normalized in the same units.
                    self._bottom_bbox_rel_half_w = float(bbox_w_rel / max(1.0, float(img_w_rel)))
                    self._bottom_bbox_rel_half_h = float(bbox_h_rel / max(1.0, float(img_h_rel)))

                    self._bottom_bbox_rel_err_x = float(abs(self._bottom_err_x) / max(1e-6, self._bottom_bbox_rel_half_w))
                    self._bottom_bbox_rel_err_y = float(abs(self._bottom_err_y) / max(1e-6, self._bottom_bbox_rel_half_h))
                    self._bottom_bbox_rel_err = float(max(self._bottom_bbox_rel_err_x, self._bottom_bbox_rel_err_y))
            except Exception:
                self._bottom_bbox_rel_err = 999.0

            self._bottom_full_tracker_mode = str(bottom.get("mode", getattr(self, "_bottom_full_tracker_mode", "LOST")) or "LOST").upper()
            self._bottom_full_tracker_raw_mode = str(bottom.get("raw_mode", getattr(self, "_bottom_full_tracker_raw_mode", "NONE")) or "NONE")
            self._bottom_full_tracker_confidence = float(bottom.get("tracker_confidence", getattr(self, "_bottom_full_tracker_confidence", 0.0)) or 0.0)
            self._bottom_full_tracker_similarity = float(bottom.get("similarity", getattr(self, "_bottom_full_tracker_similarity", 0.0)) or 0.0)
            self._bottom_candidate_count = int(bottom.get("candidate_count", getattr(self, "_bottom_candidate_count", 0)) or 0)
            self._bottom_candidate_scan_score = float(bottom.get("similarity", getattr(self, "_bottom_candidate_scan_score", 0.0)) or 0.0)

            if self._bottom_match:
                self._bottom_match_streak += 1
                self._last_bottom_match_step = int(self.step_in_episode)
            else:
                self._bottom_match_streak = 0

        confirm_similarity = float(getattr(self.cfg, "bottom_handoff_confirm_similarity", 0.56))
        confirm_streak = max(1, int(getattr(self.cfg, "bottom_confirmed_streak_required", 2)))
        fresh_max_steps = max(0, int(getattr(self.cfg, "bottom_success_fresh_max_steps", 2)))
        min_bbox_area = float(getattr(self.cfg, "bottom_success_min_bbox_area", 0.004))

        self._bottom_match_fresh = bool(
            int(self.step_in_episode) - int(getattr(self, "_last_bottom_match_step", -999999))
            <= fresh_max_steps
        )

        # Handoff success must describe a usable initial state for the landing
        # agent. Detecting the target anywhere in the downward camera is not
        # enough: if the car is at the image edge, Agent 1 probably arrived from
        # a side/orbit geometry and Agent 2 receives a bad state.
        bottom_abs_err_x = abs(float(getattr(self, "_bottom_err_x", 0.0)))
        bottom_abs_err_y = abs(float(getattr(self, "_bottom_err_y", 0.0)))
        bottom_center_error = float(np.sqrt(bottom_abs_err_x * bottom_abs_err_x + bottom_abs_err_y * bottom_abs_err_y))

        bottom_center_ok = True
        if bool(getattr(self.cfg, "bottom_success_requires_centered", True)):
            bottom_center_ok = bool(
                bottom_abs_err_x <= float(getattr(self.cfg, "bottom_success_max_abs_err_x", 0.22))
                and bottom_abs_err_y <= float(getattr(self.cfg, "bottom_success_max_abs_err_y", 0.28))
                and bottom_center_error <= float(getattr(self.cfg, "bottom_success_max_center_error", 0.34))
            )

        bottom_yaw_ok = True
        if bool(getattr(self.cfg, "bottom_success_respects_command_yaw", True)):
            bottom_yaw_ok = bool(
                abs(float(getattr(self, "_commanded_yaw_delta_deg", 0.0)))
                <= float(getattr(self.cfg, "bottom_success_max_abs_commanded_yaw_deg", 25.0))
                and float(getattr(self, "_cumulative_abs_yaw_cmd_deg", 0.0))
                <= float(getattr(self.cfg, "bottom_success_max_cumulative_abs_yaw_cmd_deg", 95.0))
            )

        self._bottom_center_error = float(bottom_center_error)
        self._bottom_center_ok = bool(bottom_center_ok)
        self._bottom_yaw_ok = bool(bottom_yaw_ok)

        # Handoff success is based on SECONDARY CAMERA confirmation.
        # Real simulator distance can help open bottom scanning, but it must not
        # decide whether the handoff succeeded. A valid secondary detection must
        # be fresh, stable, similar to the selected target fingerprint, have a
        # non-trivial bbox, be centered enough, and not validate a sideways
        # yaw/orbit approach.
        self._bottom_confirmed = bool(
            bool(getattr(self, "_bottom_match", False))
            and bool(getattr(self, "_bottom_match_fresh", False))
            and float(getattr(self, "_bottom_similarity", 0.0)) >= confirm_similarity
            and int(getattr(self, "_bottom_match_streak", 0)) >= confirm_streak
            and float(getattr(self, "_bottom_bbox_area_norm", 0.0)) >= min_bbox_area
            and bool(bottom_center_ok)
            and bool(bottom_yaw_ok)
        )

        # Soft weights:
        #   - far: front dominates
        #   - close + bottom match: bottom gains authority
        #   - bottom stable near handoff distance: ready for landing policy
        handoff_distance = float(getattr(self.cfg, "bottom_handoff_distance_m", 4.0))
        streak_required = max(1, int(getattr(self.cfg, "bottom_handoff_streak_required", 3)))
        proximity = 0.0
        if np.isfinite(dist):
            proximity = float(np.clip(1.0 - (dist / max(1e-6, scan_distance)), 0.0, 1.0))

        bottom_conf = float(np.clip((self._bottom_similarity - 0.40) / 0.35, 0.0, 1.0)) if self._bottom_match else 0.0
        streak_conf = float(np.clip(self._bottom_match_streak / float(streak_required), 0.0, 1.0))

        self._bottom_weight = float(np.clip(0.15 * proximity + 0.55 * bottom_conf + 0.30 * streak_conf, 0.0, 1.0))
        if not self._bottom_match:
            self._bottom_weight = min(self._bottom_weight, 0.20 * proximity)
        self._front_weight = float(np.clip(1.0 - self._bottom_weight, 0.0, 1.0))

        # ------------------------------------------------------------------
        # Dual-camera authority fusion
        # ------------------------------------------------------------------
        front_mode = str(front_tracking_mode or "LOST").upper()
        front_valid = bool(
            front_mode in {"MATCH", "PRED"}
            and float(obs_dict.get("has_target", 0.0)) > 0.5
        )

        if front_valid:
            self._last_front_valid_step = int(self.step_in_episode)

        bottom_authority_similarity = float(getattr(self.cfg, "dual_bottom_authority_similarity", 0.52))
        bottom_authority_area = float(getattr(self.cfg, "dual_bottom_authority_min_bbox_area", 0.003))
        bottom_authority_streak = max(1, int(getattr(self.cfg, "dual_bottom_authority_min_streak", 1)))

        bottom_valid = bool(
            bool(getattr(self, "_bottom_match", False))
            and bool(getattr(self, "_bottom_match_fresh", False))
            and float(getattr(self, "_bottom_similarity", 0.0)) >= bottom_authority_similarity
            and float(getattr(self, "_bottom_bbox_area_norm", 0.0)) >= bottom_authority_area
            and int(getattr(self, "_bottom_match_streak", 0)) >= bottom_authority_streak
        )

        if bottom_valid:
            self._last_bottom_valid_step = int(self.step_in_episode)

        bottom_keep_grace = int(getattr(self.cfg, "dual_bottom_keep_authority_grace_steps", 6))
        front_assist_grace = int(getattr(self.cfg, "dual_front_assist_after_bottom_lost_steps", 10))
        recently_bottom_valid = bool(
            int(self.step_in_episode) - int(getattr(self, "_last_bottom_valid_step", -999999))
            <= bottom_keep_grace
        )
        recently_front_valid = bool(
            int(self.step_in_episode) - int(getattr(self, "_last_front_valid_step", -999999))
            <= front_assist_grace
        )

        bottom_primary_streak = max(1, int(getattr(self.cfg, "dual_bottom_primary_min_streak", 2)))

        if bottom_valid:
            self._bottom_authority_streak += 1
        else:
            self._bottom_authority_streak = max(0, self._bottom_authority_streak - 1)

        if front_valid:
            self._front_authority_streak += 1
        else:
            self._front_authority_streak = max(0, self._front_authority_streak - 1)

        if bool(getattr(self.cfg, "dual_camera_fusion_enabled", True)):
            if bottom_valid and self._bottom_authority_streak >= bottom_primary_streak:
                self._camera_authority = "BOTTOM_PRIMARY"
                self._active_camera = "bottom"
            elif recently_bottom_valid and front_valid:
                self._camera_authority = "FRONT_ASSIST_BOTTOM"
                self._active_camera = "front"
            elif recently_bottom_valid and not front_valid:
                self._camera_authority = "BOTTOM_RECOVERY"
                self._active_camera = "bottom"
            elif front_valid:
                self._camera_authority = "FRONT_PRIMARY"
                self._active_camera = "front"
            else:
                self._camera_authority = "FUSION_LOST"
                self._active_camera = "none"
        else:
            self._camera_authority = "FRONT_PRIMARY"
            self._active_camera = "front"

        self._fusion_has_target = bool(
            self._camera_authority in {"FRONT_PRIMARY", "FRONT_ASSIST_BOTTOM", "BOTTOM_PRIMARY", "BOTTOM_RECOVERY"}
        )

        if self._active_camera == "bottom":
            self._fusion_tracking_mode = "MATCH" if bottom_valid else "PRED"
            self._fusion_err_x = float(getattr(self, "_bottom_err_x", 0.0))
            self._fusion_err_y = float(getattr(self, "_bottom_err_y", 0.0))
            self._fusion_confidence = float(getattr(self, "_bottom_similarity", 0.0))
        elif self._active_camera == "front":
            self._fusion_tracking_mode = front_mode if front_valid else "LOST"
            self._fusion_err_x = float(obs_dict.get("err_x", 0.0))
            self._fusion_err_y = float(obs_dict.get("err_y", 0.0))
            self._fusion_confidence = float(obs_dict.get("bbox_conf", 0.0))
        else:
            self._fusion_tracking_mode = "LOST"
            self._fusion_err_x = 0.0
            self._fusion_err_y = 0.0
            self._fusion_confidence = 0.0

        self._handoff_ready = bool(
            bool(getattr(self, "_bottom_confirmed", False))
            and self._bottom_match_streak >= streak_required
        )

        recent_bottom_scan = bool(
            self.step_in_episode - int(getattr(self, "_last_bottom_scan_step", -999999))
            <= int(getattr(self.cfg, "bottom_scan_hold_steps", 20))
        )

        if self._handoff_ready:
            self._handoff_phase = "HANDOFF_READY"
        elif bool(getattr(self, "_bottom_confirmed", False)):
            self._handoff_phase = "HANDOFF_OVERLAP"
        elif bool(getattr(self, "_handoff_candidate_gate", False)) or (
            recent_bottom_scan and bool(getattr(self, "_bottom_match", False))
        ):
            # Bottom scan may run, but this is NOT active handoff yet.
            self._handoff_phase = "BOTTOM_SCAN"
        else:
            self._handoff_phase = "CHASE_FRONT"

    def _draw_bottom_handoff_overlay(self, frame):
        """Draw bottom-camera target/handoff diagnostics on a debug frame."""
        self._draw_center_crosshair(frame, (0, 255, 0))

        if getattr(self, "_bottom_bbox_xyxy", None) is not None:
            x1, y1, x2, y2 = [int(v) for v in self._bottom_bbox_xyxy]
            color = (0, 255, 0) if bool(getattr(self, "_bottom_match", False)) else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.drawMarker(frame, ((x1 + x2) // 2, (y1 + y2) // 2), color, cv2.MARKER_CROSS, 20, 2)

        cv2.putText(
            frame,
            f"BOTTOM_MATCH={int(bool(getattr(self, '_bottom_match', False)))} "
            f"BC={int(bool(getattr(self, '_bottom_confirmed', False)))} "
            f"sim={float(getattr(self, '_bottom_similarity', 0.0)):.2f} "
            f"streak={int(getattr(self, '_bottom_match_streak', 0))}",
            (12, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"phase={str(getattr(self, '_handoff_phase', 'CHASE_FRONT'))} "
            f"front_w={float(getattr(self, '_front_weight', 1.0)):.2f} "
            f"bottom_w={float(getattr(self, '_bottom_weight', 0.0)):.2f}",
            (12, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Gate={int(bool(getattr(self, '_handoff_candidate_gate', False)))} "
            f"Fresh={int(bool(getattr(self, '_bottom_match_fresh', False)))} "
            f"Bsafe={int(float(getattr(self, '_bottom_bbox_rel_err', 999.0)) <= float(getattr(self.cfg, 'handoff_success_max_bbox_rel_err', 0.50)))} "
            f"Barea={float(getattr(self, '_bottom_bbox_area_norm', 0.0)):.3f} "
            f"VL={int(bool(getattr(self, '_handoff_visual_lidar_trigger', False)))} "
            f"VS={int(bool(getattr(self, '_handoff_visual_scan_trigger', False)))} "
            f"Vscore={float(getattr(self, '_handoff_visual_score', 0.0)):.1f} "
            f"L={float(getattr(self, '_handoff_lidar_dist_m', 999.0)):.1f}m "
            f"dir={str(getattr(self, '_handoff_lidar_direction', 'none'))}",
            (12, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 255),
            2,
        )
        return frame

    def _click_lock_once(self):
        selected = False
        win = "INITIAL SETUP: Click target (ONE TIME)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        if self.cfg.print_reset:
            print("\n[ENV] ONE-TIME TARGET SELECTION (click object)\n")

        def on_click(event, x, y, flags, param):
            nonlocal selected
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"[ENV CLICK] received click x={x} y={y}")
                cid = self.tracker.select_target_and_get_class(param["frame"], x, y)

                # Do not require class_id to be non-None.
                # If YOLO misses, object_tracker can still lock a manual fallback bbox
                # and return cid=None. The real success condition is that a bbox and
                # fingerprint exist.
                fp = self.tracker.get_target_fingerprint()
                bbox = getattr(self.tracker, "last_bbox", None)

                if bbox is not None and fp is not None:
                    self.target_class_id = cid
                    self.target_fingerprint = fp
                    selected = True
                    print(f"[ENV CLICK] target selected cid={cid} bbox={bbox}")
                else:
                    print("[ENV CLICK] click did not produce a valid target. Try clicking the car body again.")

        # Keep one named window alive and refresh the callback frame.
        # waitKey(20) gives OpenCV enough time to process mouse events reliably.
        cv2.resizeWindow(win, int(self.cfg.image_width), int(self.cfg.image_height))

        while not selected:
            frame = self._get_frame()
            cv2.setMouseCallback(win, on_click, param={"frame": frame})
            cv2.imshow(win, frame)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                raise RuntimeError("Target selection cancelled by ESC")

        cv2.destroyWindow(win)
        self._target_initialized = True

        if self.cfg.print_reset:
            print("[ENV] Target locked and persisted\n")

    def _create_tracker_manager(self) -> TargetTrackerManager:
        """
        Create the external bbox stabilizer.

        This layer sits after object_tracker.py and before the observation/controller.
        It does not replace the visual tracker. It only rejects suspicious bbox jumps,
        predicts short gaps with Kalman, and exposes MATCH/PRED/LOST for the env.
        """
        return TargetTrackerManager(
            min_tracker_confidence=0.45,
            max_center_jump_pixels=120.0,
            min_iou_with_prediction=0.05,
            max_pred_frames=25,
        )

    @staticmethod
    def _mode_to_bbox_conf(mode: str) -> float:
        if mode == "MATCH":
            return 1.0
        if mode == "PRED":
            return 0.2
        return 0.0

    def _bbox_to_observation(self, bbox, mode: str) -> BBox | None:
        """
        Convert an xyxy bbox to the ObservationBuilder BBox format.

        Important:
            In LOST mode we intentionally return None, even if the stabilizer still
            has a last stable bbox. This lets lost_target_time increase correctly
            and prevents the policy from chasing an old guess forever.
        """
        if bbox is None or mode == "LOST":
            return None

        try:
            x1, y1, x2, y2 = map(float, bbox[:4])
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            cx = x1 + 0.5 * w
            cy = y1 + 0.5 * h
            conf = self._mode_to_bbox_conf(mode)
            return BBox(cx=cx, cy=cy, w=w, h=h, conf=conf)
        except Exception:
            return None

    def _raw_tracker_confidence(self) -> float:
        """Return a simple confidence proxy from the current front visual tracker mode."""
        return self._raw_tracker_confidence_from(self.tracker)

    def _raw_tracker_confidence_from(self, tracker_obj) -> float:
        """Return a simple confidence proxy from a specific visual tracker object."""
        raw_mode = getattr(tracker_obj, "last_mode", "NONE")
        return self._mode_to_bbox_conf(raw_mode)

    def _tracker_similarity_fallback(self, tracker_obj, mode: str) -> float:
        """
        Best-effort similarity extraction from a tracker instance.

        object_tracker.py / resnet_yolo_tracker.py versions may expose the
        ResNet score under different attribute names. This helper keeps the env
        compatible and falls back to mode-based scores when no explicit score is
        available.
        """
        candidates = [
            "last_similarity",
            "last_score",
            "last_match_score",
            "last_resnet_score",
            "_last_similarity",
            "_last_score",
        ]

        for name in candidates:
            try:
                value = getattr(tracker_obj, name, None)
                if value is not None:
                    v = float(value)
                    if np.isfinite(v):
                        return float(np.clip(v, 0.0, 1.0))
            except Exception:
                pass

        core = getattr(tracker_obj, "core", None)
        if core is not None:
            for name in candidates:
                try:
                    value = getattr(core, name, None)
                    if value is not None:
                        v = float(value)
                        if np.isfinite(v):
                            return float(np.clip(v, 0.0, 1.0))
                except Exception:
                    pass

        m = str(mode or "LOST").upper()
        if m == "MATCH":
            return float(getattr(self.cfg, "bottom_full_tracker_match_similarity_fallback", 0.78))
        if m == "PRED":
            return float(getattr(self.cfg, "bottom_full_tracker_pred_similarity_fallback", 0.42))
        return float(getattr(self.cfg, "bottom_full_tracker_lost_similarity_fallback", 0.0))

    def _update_bottom_stable_tracking(self, bbox_raw, frame, dt: float) -> dict:
        """
        Bottom-camera equivalent of _update_stable_tracking.

        It uses an independent tracker instance and an independent
        TargetTrackerManager/Kalman state. This makes the bottom camera a real
        tracker, not just a one-shot handoff scanner.
        """
        frame_height, frame_width = frame.shape[:2]
        tracker_confidence = self._raw_tracker_confidence_from(self.bottom_tracker)
        raw_tracker_mode = str(getattr(self.bottom_tracker, "last_raw_mode", "") or "")
        raw_simple_mode = str(getattr(self.bottom_tracker, "last_mode", "NONE") or "NONE")

        is_raw_match = bool(
            raw_simple_mode == "MATCH"
            or raw_tracker_mode == "MATCH"
            or raw_tracker_mode.startswith("MATCH_")
            or raw_tracker_mode.startswith("CLICK_SELECT_YOLO_RESNET")
            or raw_tracker_mode.startswith("INIT_YOLO_RESNET")
        )

        if bbox_raw is not None and is_raw_match:
            bbox_raw = np.asarray(bbox_raw, dtype=np.float32)
            x1, y1, x2, y2 = bbox_raw[:4]
            x1 = float(np.clip(x1, 0, frame_width - 1))
            y1 = float(np.clip(y1, 0, frame_height - 1))
            x2 = float(np.clip(x2, x1 + 1, frame_width))
            y2 = float(np.clip(y2, y1 + 1, frame_height))
            trusted_bbox = np.asarray([x1, y1, x2, y2], dtype=np.float32)

            self._bottom_last_trusted_raw_bbox_xyxy = trusted_bbox.copy()
            self._bottom_kalman_fallback_active = False

            result = {
                "mode": "MATCH",
                "stable_bbox": trusted_bbox.copy(),
                "kalman_pred_bbox": None,
                "accepted_tracker": True,
                "tracker_confidence": float(max(tracker_confidence, 1.0)),
                "center_error": 0.0,
                "iou_with_prediction": 1.0,
                "pred_frames": 0,
                "raw_direct_yolo_resnet": True,
                "kalman_used": False,
                "raw_tracker_mode": raw_tracker_mode or raw_simple_mode,
            }
            self._bottom_tracking_result = result
            self._bottom_stable_bbox_xyxy = result.get("stable_bbox")
            return result

        last_trusted = getattr(self, "_bottom_last_trusted_raw_bbox_xyxy", None)

        if not self.bottom_tracker_manager.initialized:
            if last_trusted is not None:
                self.bottom_tracker_manager.initialize(np.asarray(last_trusted, dtype=np.float32))
                self._bottom_kalman_fallback_active = True
            else:
                result = {
                    "mode": "LOST",
                    "stable_bbox": None,
                    "kalman_pred_bbox": None,
                    "accepted_tracker": False,
                    "tracker_confidence": 0.0,
                    "center_error": None,
                    "iou_with_prediction": None,
                    "pred_frames": 0,
                    "raw_direct_yolo_resnet": False,
                    "kalman_used": False,
                    "raw_tracker_mode": raw_tracker_mode or raw_simple_mode,
                }
                self._bottom_tracking_result = result
                self._bottom_stable_bbox_xyxy = None
                return result

        result = self.bottom_tracker_manager.update(
            tracker_bbox_xyxy=None,
            tracker_confidence=0.0,
            frame_width=frame_width,
            frame_height=frame_height,
            dt=dt,
        )
        result["raw_direct_yolo_resnet"] = False
        result["kalman_used"] = True
        result["raw_tracker_mode_at_fallback"] = raw_tracker_mode or raw_simple_mode
        result["raw_tracker_mode"] = raw_tracker_mode or raw_simple_mode

        self._bottom_tracking_result = result
        self._bottom_stable_bbox_xyxy = result.get("stable_bbox")
        return result

    def _scan_bottom_target_candidate_by_fingerprint(self, downward_frame):
        """
        Non-recursive bottom YOLO+ResNet candidate scan.

        This is intentionally separate from _detect_bottom_target_by_fingerprint
        so the full bottom tracker can use it as a reacquire/measurement fallback
        without calling itself recursively.
        """
        result = {
            "match": False,
            "bbox_xyxy": None,
            "similarity": 0.0,
            "err_x": 0.0,
            "err_y": 0.0,
            "bbox_area_norm": 0.0,
            "candidate_count": 0,
            "mode": "LOST",
            "raw_mode": "SCAN_NONE",
            "tracker_confidence": 0.0,
        }

        try:
            # Prefer the bottom tracker core if available. Fall back to the
            # front tracker core because both adapters expose the same detector
            # and embedding helpers.
            core = getattr(self.bottom_tracker, "core", None)
            if core is None:
                core = getattr(self.tracker, "core", None)

            target_embedding = getattr(core, "target_embedding", None) if core is not None else None
            if core is None or target_embedding is None:
                return result

            candidates = core._detect_candidates(downward_frame)
            result["candidate_count"] = int(len(candidates))
            self._bottom_candidate_count = int(len(candidates))

            target_class_id = getattr(self, "target_class_id", None)
            if (
                bool(getattr(self.cfg, "bottom_candidate_scan_class_gate", True))
                and target_class_id is not None
            ):
                same_class = [c for c in candidates if int(c.cls_id) == int(target_class_id)]
                if same_class:
                    candidates = same_class

            best = None
            best_score = -1.0

            for cand in candidates:
                emb = core._embedding_from_bbox(downward_frame, cand.bbox)
                if emb is None:
                    continue

                # Embeddings are expected to be normalized by the adapter.
                score = float((target_embedding * emb).sum())
                if score > best_score:
                    best_score = score
                    best = cand

            if best is None:
                return result

            h, w = downward_frame.shape[:2]
            x1, y1, x2, y2 = [float(v) for v in best.bbox[:4]]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_norm = float(area / max(1.0, float(w * h)))

            err_x, err_y = self._bbox_xyxy_to_error(best.bbox, int(w), int(h))

            min_score = float(getattr(self.cfg, "bottom_candidate_scan_min_similarity", 0.46))
            min_area = float(getattr(self.cfg, "bottom_candidate_scan_min_bbox_area", 0.0015))

            result.update(
                {
                    "match": bool(best_score >= min_score and area_norm >= min_area),
                    "bbox_xyxy": np.asarray(best.bbox, dtype=np.float32),
                    "similarity": float(np.clip(best_score, 0.0, 1.0)),
                    "err_x": float(err_x),
                    "err_y": float(err_y),
                    "bbox_area_norm": float(area_norm),
                    "candidate_count": int(result["candidate_count"]),
                    "mode": "MATCH" if best_score >= min_score and area_norm >= min_area else "PRED",
                    "raw_mode": f"SCAN_YOLO_RESNET cid={int(best.cls_id)}",
                    "tracker_confidence": float(np.clip(best_score, 0.0, 1.0)),
                }
            )

            self._bottom_candidate_scan_score = float(result["similarity"])
            return result

        except Exception as e:
            if bool(getattr(self.cfg, "print_sensor_errors", False)):
                print(f"[BOTTOM SCAN WARNING] candidate scan failed: {e}")
            return result


    def _update_bottom_full_tracker(self, downward_frame, dt: float) -> dict:
        """
        Run the complete bottom-camera tracker path.

        Logic fix:
        The previous version tried auto_lock only on the first bottom update.
        If that first update happened before the car was visible in the bottom
        camera, the bottom tracker could stay empty forever even when the car
        later became obvious in the frame.

        This version:
            1. Syncs the selected target identity every update.
            2. Re-attempts bottom auto-lock while the bottom tracker is not MATCH.
            3. Runs a direct YOLO+ResNet candidate scan fallback when the full
               tracker returns no reliable bbox.
            4. Uses the scan candidate as a real measurement for fusion/handoff.
        """
        result = {
            "match": False,
            "bbox_xyxy": None,
            "similarity": 0.0,
            "err_x": 0.0,
            "err_y": 0.0,
            "bbox_area_norm": 0.0,
            "candidate_count": 0,
            "mode": "LOST",
            "tracker_confidence": 0.0,
            "raw_mode": "NONE",
        }

        self._bottom_candidate_scan_used = False

        if not bool(getattr(self.cfg, "bottom_full_tracker_enabled", True)):
            return result

        if getattr(self, "target_fingerprint", None) is None:
            return result

        try:
            self.bottom_tracker.set_target_fingerprint(self.target_fingerprint)
            self.bottom_tracker.set_target_class(self.target_class_id)

            # Important: do not auto-lock only once. During early steps the
            # downward camera may not see the car yet, so we keep trying until
            # bottom MATCH is achieved.
            should_relock = bool(
                bool(getattr(self.cfg, "bottom_relock_every_step_when_lost", True))
                and str(getattr(self, "_bottom_full_tracker_mode", "LOST")).upper() != "MATCH"
            )

            if should_relock:
                self._bottom_relock_attempts += 1
                try:
                    self.bottom_tracker.auto_lock_on_fingerprint(downward_frame, use_class_gate=True)
                except Exception:
                    pass

            bbox_raw = self.bottom_tracker.update(downward_frame)
            tracking_result = self._update_bottom_stable_tracking(
                bbox_raw=bbox_raw,
                frame=downward_frame,
                dt=float(dt),
            )

            mode = str(tracking_result.get("mode", "LOST") or "LOST").upper()
            bbox = tracking_result.get("stable_bbox", None)
            raw_mode = str(
                tracking_result.get("raw_tracker_mode", "")
                or tracking_result.get("raw_tracker_mode_at_fallback", "")
                or getattr(self.bottom_tracker, "last_mode", "NONE")
            )
            tracker_conf = float(tracking_result.get("tracker_confidence", 0.0) or 0.0)
            similarity = float(self._tracker_similarity_fallback(self.bottom_tracker, mode))

            # Fallback measurement path:
            # If the full bottom tracker has no useful bbox, perform the direct
            # scanner that worked in the previous handoff-success versions.
            scan = None
            if (
                bool(getattr(self.cfg, "bottom_use_candidate_scan_fallback", True))
                and (bbox is None or mode == "LOST")
            ):
                scan = self._scan_bottom_target_candidate_by_fingerprint(downward_frame)
                if scan.get("bbox_xyxy", None) is not None:
                    self._bottom_candidate_scan_used = True
                    result = dict(scan)

                    # Seed bottom Kalman bridge with the scan bbox so following
                    # frames can produce PRED/MATCH continuity.
                    trusted_bbox = np.asarray(scan["bbox_xyxy"], dtype=np.float32)
                    self._bottom_last_trusted_raw_bbox_xyxy = trusted_bbox.copy()
                    try:
                        if not self.bottom_tracker_manager.initialized:
                            self.bottom_tracker_manager.initialize(trusted_bbox)
                    except Exception:
                        pass

                    mode = str(scan.get("mode", "LOST") or "LOST").upper()
                    raw_mode = str(scan.get("raw_mode", "SCAN"))
                    tracker_conf = float(scan.get("tracker_confidence", 0.0) or 0.0)
                    similarity = float(scan.get("similarity", 0.0) or 0.0)
                    bbox = scan.get("bbox_xyxy", None)

                    self._bottom_full_tracker_mode = mode
                    self._bottom_full_tracker_raw_mode = raw_mode
                    self._bottom_full_tracker_confidence = tracker_conf
                    self._bottom_full_tracker_similarity = similarity
                    self._bottom_full_tracker_updates += 1

                    if mode == "MATCH":
                        self._bottom_full_tracker_match_frames += 1
                    elif mode == "PRED":
                        self._bottom_full_tracker_pred_frames += 1
                    else:
                        self._bottom_full_tracker_lost_frames += 1

                    return result

            self._bottom_full_tracker_mode = mode
            self._bottom_full_tracker_raw_mode = raw_mode
            self._bottom_full_tracker_confidence = tracker_conf
            self._bottom_full_tracker_similarity = similarity
            self._bottom_full_tracker_updates += 1

            if mode == "MATCH":
                self._bottom_full_tracker_match_frames += 1
            elif mode == "PRED":
                self._bottom_full_tracker_pred_frames += 1
            else:
                self._bottom_full_tracker_lost_frames += 1

            result["mode"] = mode
            result["raw_mode"] = raw_mode
            result["tracker_confidence"] = tracker_conf
            result["similarity"] = similarity

            if bbox is None:
                return result

            bbox = np.asarray(bbox, dtype=np.float32)
            result["bbox_xyxy"] = bbox.copy()
            result["match"] = bool(mode == "MATCH")

            h, w = downward_frame.shape[:2]
            err_x, err_y = self._bbox_xyxy_to_error(bbox, int(w), int(h))
            result["err_x"] = float(err_x)
            result["err_y"] = float(err_y)

            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            result["bbox_area_norm"] = float(area / max(1.0, float(w * h)))
            return result

        except Exception as e:
            if bool(getattr(self.cfg, "print_sensor_errors", False)):
                print(f"[BOTTOM FULL TRACKER WARNING] update failed: {e}")
            return result


    def _update_stable_tracking(self, bbox_raw, frame, dt: float) -> dict:
        """
        RAW-first tracking bridge for YOLO+ResNet.

        Policy decision:
            - When YOLO+ResNet reports RAW MATCH, the RL observation/reward uses
              the raw visual bbox directly.
            - Kalman / TargetTrackerManager is NOT allowed to smooth, lag, or
              replace a valid YOLO+ResNet MATCH.
            - Kalman is used only as a fallback when the visual tracker is not in
              MATCH, for example temporary occlusion, low confidence, or no bbox.

        This keeps the PPO model listening to the strongest sensor path:
            YOLO detections + ResNet identity matching.
        """
        frame_height, frame_width = frame.shape[:2]
        tracker_confidence = self._raw_tracker_confidence()
        raw_tracker_mode = str(getattr(self.tracker, "last_raw_mode", "") or "")
        raw_simple_mode = str(getattr(self.tracker, "last_mode", "NONE") or "NONE")

        # ResNet adapter uses raw modes like:
        #   MATCH_YOLO_RESNET score=...
        #   PRED_LOW_SCORE_YOLO_RESNET score=...
        # Legacy trackers may use exactly MATCH/PRED/NONE.
        is_raw_match = bool(
            raw_simple_mode == "MATCH"
            or raw_tracker_mode == "MATCH"
            or raw_tracker_mode.startswith("MATCH_")
            or raw_tracker_mode.startswith("CLICK_SELECT_YOLO_RESNET")
            or raw_tracker_mode.startswith("INIT_YOLO_RESNET")
        )

        # ------------------------------------------------------------------
        # Strong path: YOLO+ResNet MATCH is trusted directly.
        # ------------------------------------------------------------------
        if bbox_raw is not None and is_raw_match:
            bbox_raw = np.asarray(bbox_raw, dtype=np.float32)

            # Clamp only for safety. Do not smooth, do not Kalman-update.
            x1, y1, x2, y2 = bbox_raw[:4]
            x1 = float(np.clip(x1, 0, frame_width - 1))
            y1 = float(np.clip(y1, 0, frame_height - 1))
            x2 = float(np.clip(x2, x1 + 1, frame_width))
            y2 = float(np.clip(y2, y1 + 1, frame_height))
            trusted_bbox = np.asarray([x1, y1, x2, y2], dtype=np.float32)

            # Store last trusted visual measurement for lazy Kalman fallback init.
            self._last_trusted_raw_bbox_xyxy = trusted_bbox.copy()
            self._kalman_fallback_active = False

            result = {
                "mode": "MATCH",
                "stable_bbox": trusted_bbox.copy(),       # What RL sees.
                "kalman_pred_bbox": None,                 # No Kalman during RAW MATCH.
                "accepted_tracker": True,
                "tracker_confidence": float(max(tracker_confidence, 1.0)),
                "center_error": 0.0,
                "iou_with_prediction": 1.0,
                "pred_frames": 0,
                "raw_direct_yolo_resnet": True,
                "kalman_used": False,
            }

            self._tracking_result = result
            self._stable_bbox_xyxy = result.get("stable_bbox")
            return result

        # ------------------------------------------------------------------
        # Fallback path: visual tracker is not trusted now.
        # Use Kalman only here.
        # ------------------------------------------------------------------
        last_trusted = getattr(self, "_last_trusted_raw_bbox_xyxy", None)

        if not self.tracker_manager.initialized:
            if last_trusted is not None:
                self.tracker_manager.initialize(np.asarray(last_trusted, dtype=np.float32))
                self._kalman_fallback_active = True
            else:
                result = {
                    "mode": "LOST",
                    "stable_bbox": None,
                    "kalman_pred_bbox": None,
                    "accepted_tracker": False,
                    "tracker_confidence": 0.0,
                    "center_error": None,
                    "iou_with_prediction": None,
                    "pred_frames": 0,
                    "raw_direct_yolo_resnet": False,
                    "kalman_used": False,
                }
                self._tracking_result = result
                self._stable_bbox_xyxy = None
                return result

        # During fallback, do not feed weak RAW/PRED bbox as a measurement.
        # Let Kalman predict for a short gap. If YOLO+ResNet returns MATCH again,
        # the top branch will immediately take over and expose raw bbox directly.
        result = self.tracker_manager.update(
            tracker_bbox_xyxy=None,
            tracker_confidence=0.0,
            frame_width=frame_width,
            frame_height=frame_height,
            dt=dt,
        )
        result["raw_direct_yolo_resnet"] = False
        result["kalman_used"] = True
        result["raw_tracker_mode_at_fallback"] = raw_tracker_mode or raw_simple_mode

        self._tracking_result = result
        self._stable_bbox_xyxy = result.get("stable_bbox")
        return result


    @staticmethod
    def _draw_xyxy_box(frame, bbox, color, label: str):
        if bbox is None:
            return

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    def _draw_tracking_overlay(self, frame, bbox_raw, tracking_result: dict | None):
        """Draw raw tracker bbox, Kalman prediction, and stable bbox."""
        if tracking_result is None:
            return

        mode = tracking_result.get("mode", "LOST")
        accepted = bool(tracking_result.get("accepted_tracker", False))
        pred_frames = int(tracking_result.get("pred_frames", 0))
        raw_mode = getattr(self.tracker, "last_mode", "NONE")

        self._draw_xyxy_box(frame, bbox_raw, (0, 0, 255), f"RAW {raw_mode}")
        self._draw_xyxy_box(frame, tracking_result.get("kalman_pred_bbox"), (255, 0, 0), "KALMAN")
        self._draw_xyxy_box(frame, tracking_result.get("stable_bbox"), (0, 255, 0), f"STABLE {mode}")

        cv2.putText(
            frame,
            f"STABLE_MODE: {mode} | RAW_MODE: {raw_mode} | accepted={accepted} | pred_frames={pred_frames}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
        )

    @staticmethod
    def _draw_camera_label(frame, text: str, color=(0, 255, 255)):
        """Draw a clear label on a camera debug frame."""
        cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 520), 38), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (12, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )

    @staticmethod
    def _draw_center_crosshair(frame, color=(0, 255, 0)):
        """Draw a simple center crosshair for camera alignment debugging."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 18, cy), (cx + 18, cy), color, 2)
        cv2.line(frame, (cx, cy - 18), (cx, cy + 18), color, 2)
        cv2.circle(frame, (cx, cy), 24, color, 1)

    def _resize_debug_frame(self, frame):
        """Resize one debug frame according to cv_debug_scale."""
        scale = float(getattr(self.cfg, "cv_debug_scale", 1.0))
        scale = float(np.clip(scale, 0.25, 1.25))

        if abs(scale - 1.0) < 1e-3:
            return frame

        new_w = max(1, int(frame.shape[1] * scale))
        new_h = max(1, int(frame.shape[0] * scale))
        return cv2.resize(frame, (new_w, new_h))

    def _make_single_camera_debug_frame(self, frame, label: str, *, draw_crosshair: bool = False):
        """Return one large debug image for the currently relevant camera."""
        vis = frame.copy()
        self._draw_camera_label(vis, label, (0, 255, 255))
        if draw_crosshair:
            self._draw_center_crosshair(vis, (0, 255, 0))
        return self._resize_debug_frame(vis)

    def _make_dual_camera_debug_frame(self, front_frame, downward_frame):
        """Return one OpenCV debug image containing front and downward cameras.

        This mode is kept for manual debugging only. It is intentionally not the
        default during training because reading and rendering both cameras hurts FPS.
        """
        front_vis = front_frame.copy()
        down_vis = downward_frame.copy()

        self._draw_camera_label(front_vis, "FRONT CAMERA - CHASE/TRACKING", (0, 255, 255))
        self._draw_camera_label(down_vis, "DOWNWARD CAMERA - LANDING VIEW", (0, 255, 255))
        self._draw_bottom_handoff_overlay(down_vis)

        if front_vis.shape[:2] != down_vis.shape[:2]:
            down_vis = cv2.resize(down_vis, (front_vis.shape[1], front_vis.shape[0]))

        front_vis = self._resize_debug_frame(front_vis)
        down_vis = self._resize_debug_frame(down_vis)

        return cv2.hconcat([front_vis, down_vis])

    def _make_front_with_bottom_inset_debug_frame(self, front_frame):
        """Show front as main view and a small cached bottom scan preview.

        BOTTOM_SCAN means the downward camera is scanning in the background.
        It should not fully replace the front view until bottom confirmation is
        strong. This inset lets us debug bottom acquisition without losing the
        front-camera context.
        """
        vis = front_frame.copy()
        down = getattr(self, "_cached_downward_frame", None)

        if down is not None:
            try:
                down_vis = self._draw_bottom_handoff_overlay(down.copy())

                h, w = vis.shape[:2]
                inset_w = max(220, int(w * 0.28))
                inset_h = int(inset_w * down_vis.shape[0] / max(1, down_vis.shape[1]))
                inset_h = min(inset_h, max(120, int(h * 0.34)))

                down_vis = cv2.resize(down_vis, (inset_w, inset_h))
                x1 = max(0, w - inset_w - 12)
                y1 = max(0, h - inset_h - 12)
                x2 = x1 + inset_w
                y2 = y1 + inset_h

                cv2.rectangle(vis, (x1 - 2, y1 - 24), (x2 + 2, y2 + 2), (0, 0, 0), -1)
                cv2.putText(
                    vis,
                    "BOTTOM SCAN PREVIEW",
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )
                vis[y1:y2, x1:x2] = down_vis

            except Exception as e:
                cv2.putText(
                    vis,
                    f"BOTTOM PREVIEW ERROR: {e}",
                    (20, max(40, vis.shape[0] - 40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                )

        return self._make_single_camera_debug_frame(
            vis,
            "FRONT CAMERA - ACTIVE CHASE/TRACKING + BOTTOM SCAN PREVIEW",
            draw_crosshair=False,
        )

    def _select_cv_debug_frame(self, front_frame, safety_phase_now: str):
        """Select the camera shown in the CV debug window.

        Default behavior is "active":
            CHASE / TAKEOFF / RECOVERY -> front camera
            LANDING / ALIGN / BOTTOM   -> downward camera

        This does not change the policy or the observation yet. It only avoids
        wasting FPS on a second camera unless the active phase needs it.
        """
        mode = str(getattr(self.cfg, "cv_display_mode", "active") or "active").lower()
        phase = str(safety_phase_now or "CHASE").upper()
        handoff_phase = str(getattr(self, "_handoff_phase", "CHASE_FRONT")).upper()
        bottom_weight = float(getattr(self, "_bottom_weight", 0.0) or 0.0)

        use_bottom = False
        use_dual = False

        if mode == "dual":
            use_dual = True
        elif mode in {"bottom", "down", "downward"}:
            use_bottom = True
        elif mode == "front":
            use_bottom = False
        else:
            # Active mode. IMPORTANT:
            # BOTTOM_SCAN is only a background perception scan, not a camera
            # authority switch. Keep showing/front-controlling until bottom
            # camera actually confirms the selected target.
            use_bottom = (
                phase in {"LANDING", "LANDING_BOTTOM", "ALIGN", "ALIGN_ABOVE_TARGET", "BOTTOM"}
                or handoff_phase in {"HANDOFF_READY", "LANDING_BOTTOM"}
                or (
                    # True handoff/authority view requires confirmed bottom match.
                    handoff_phase == "HANDOFF_OVERLAP"
                    and bool(getattr(self, "_bottom_confirmed", False))
                    and bottom_weight >= float(getattr(self.cfg, "bottom_active_min_weight", 0.45))
                )
            )

        if use_dual:
            try:
                downward_frame = self._get_downward_frame()
            except Exception as e:
                downward_frame = np.zeros_like(front_frame)
                cv2.putText(
                    downward_frame,
                    f"DOWNWARD CAMERA ERROR: {e}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
            return self._make_dual_camera_debug_frame(front_frame, downward_frame)

        if use_bottom:
            try:
                downward_frame = getattr(self, "_cached_downward_frame", None)
                if downward_frame is None:
                    downward_frame = self._get_downward_frame()
                downward_frame = self._draw_bottom_handoff_overlay(downward_frame.copy())
                bottom_label = "DOWNWARD CAMERA - BOTTOM SCAN VIEW"
                if handoff_phase in {"HANDOFF_OVERLAP", "HANDOFF_READY", "LANDING_BOTTOM"}:
                    bottom_label = "DOWNWARD CAMERA - ACTIVE HANDOFF/LANDING VIEW"
                return self._make_single_camera_debug_frame(
                    downward_frame,
                    bottom_label,
                    draw_crosshair=False,
                )
            except Exception as e:
                error_frame = np.zeros_like(front_frame)
                cv2.putText(
                    error_frame,
                    f"DOWNWARD CAMERA ERROR: {e}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                return self._make_single_camera_debug_frame(
                    error_frame,
                    "DOWNWARD CAMERA ERROR",
                    draw_crosshair=False,
                )

        if (
            handoff_phase == "BOTTOM_SCAN"
            and bool(getattr(self, "_handoff_candidate_gate", False))
            and bool(getattr(self, "_handoff_visual_scan_trigger", False))
        ):
            return self._make_front_with_bottom_inset_debug_frame(front_frame)

        return self._make_single_camera_debug_frame(
            front_frame,
            "FRONT CAMERA - ACTIVE CHASE/TRACKING",
            draw_crosshair=False,
        )

    def _get_drone_state(self) -> DroneState:
        """
        Read drone state from AirSim.

        Robustness note:
            The previous implementation returned a fully zero DroneState if any
            part of state extraction failed. That made the environment report
            alt=0.00m even when AirSim pose/state altitude was ~5m.

            The altitude is mission-critical, so we compute it independently and
            preserve it even if attitude/euler extraction fails.
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

        # Safe defaults. Never let a secondary extraction failure erase altitude.
        altitude_m = 0.0
        vx_mps = 0.0
        vy_mps = 0.0
        vz_mps = 0.0
        roll_rad = 0.0
        pitch_rad = 0.0
        yaw_rate_radps = 0.0

        try:
            ms = self.client.getMultirotorState(vehicle_name=self.cfg.vehicle_name)
            k = ms.kinematics_estimated

            pos = k.position
            altitude_m = max(0.0, float(-pos.z_val))

            v = k.linear_velocity
            vx_mps = float(v.x_val)
            vy_mps = float(v.y_val)
            vz_mps = float(v.z_val)

            av = k.angular_velocity
            yaw_rate_radps = float(av.z_val)

            # Euler extraction can differ between airsim/cosysairsim versions.
            # If it fails, keep pitch/roll at 0 but preserve altitude/velocity.
            try:
                q = k.orientation
                pitch_r, roll_r, _yaw_r = airsim.to_eularian_angles(q)
                roll_rad = float(roll_r)
                pitch_rad = float(pitch_r)
            except Exception as e:
                if bool(getattr(self.cfg, "print_sensor_errors", False)):
                    print(f"[STATE WARNING] Euler extraction failed: {e}")

        except Exception as e:
            # Fallback: _get_alt_agl_m performs a narrower state/pose query.
            alt = self._get_alt_agl_m()
            if alt is not None and np.isfinite(float(alt)):
                altitude_m = max(0.0, float(alt))

            if bool(getattr(self.cfg, "print_sensor_errors", False)):
                print(f"[STATE WARNING] getMultirotorState failed in _get_drone_state: {e}")

        return DroneState(
            altitude_m=float(altitude_m),
            vx_mps=float(vx_mps),
            vy_mps=float(vy_mps),
            vz_mps=float(vz_mps),
            roll_rad=float(roll_rad),
            pitch_rad=float(pitch_rad),
            yaw_rate_radps=float(yaw_rate_radps),
        )

    def _get_alt_agl_m(self) -> float | None:
        try:
            ms = self.client.getMultirotorState(vehicle_name=self.cfg.vehicle_name)
            return max(0.0, float(-ms.kinematics_estimated.position.z_val))
        except Exception:
            return None

    def _apply_down_distance_as_altitude_if_valid(self, drone_state: DroneState, obstacle_dict: dict) -> DroneState:
        """
        Decide whether a down-range sensor may override pose/kinematics altitude.

        Important architecture decision:
            - During CHASE, altitude must come from AirSim pose/kinematics.
            - LiDAR down is still used for safety/landing awareness, but it must
              not override altitude because raw LiDAR contains self/near-body
              returns around 0.10-0.40m even when the vehicle is airborne.
            - Legacy DistanceDown may still override altitude when explicitly used.

        This keeps safety honest: we do not disable LiDAR; we calibrate its
        interpretation and avoid treating self-returns as AGL altitude.
        """
        source = str(obstacle_dict.get("obstacle_source", "")).lower()
        if "lidar" in source and not bool(getattr(self.cfg, "use_lidar_down_as_altitude", False)):
            return drone_state

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

    def _read_lidar_obstacle_state_m(self, altitude_m: float | None = None) -> dict | None:
        """Read 3D LiDAR and convert it to old sector-distance fields.

        The returned keys are compatible with ObstacleState and safety_filter:
            front/front_left/front_right/left/right/back/down/min_obstacle.
        """
        try:
            data = self.client.getLidarData(
                lidar_name=str(getattr(self.cfg, "lidar_sensor_name", "LidarSensor1")),
                vehicle_name=self.cfg.vehicle_name,
            )
            points = point_cloud_to_array(getattr(data, "point_cloud", []))
            sectors = self.lidar_processor.compute_sector_distances(
                points_xyz=points,
                altitude_fallback_m=altitude_m,
            )

            if not bool(sectors.get("lidar_valid", False)):
                return None

            # Make sure all numeric fields are clipped and finite.
            max_d = float(self.cfg.lidar_max_dist_m)
            out = {}
            for key in (
                "front_dist_m",
                "front_left_dist_m",
                "front_right_dist_m",
                "left_dist_m",
                "right_dist_m",
                "back_dist_m",
                "down_dist_m",
                "min_obstacle_dist_m",
            ):
                v = float(sectors.get(key, max_d))
                if not np.isfinite(v) or v <= 0.0:
                    v = max_d
                out[key] = float(np.clip(v, 0.0, max_d))

            out["lidar_valid"] = True
            out["lidar_point_count"] = int(sectors.get("lidar_point_count", 0))
            out["obstacle_source"] = "lidar"
            return out

        except Exception as e:
            if getattr(self.cfg, "print_sensor_errors", False):
                print(f"[WARN] Failed reading LiDAR {getattr(self.cfg, 'lidar_sensor_name', 'LidarSensor1')}: {e}")
            return None

    def _get_obstacle_state_m(self, altitude_m: float | None = None) -> dict:
        """Read obstacle state in meters.

        New primary source: 3D LiDAR sectors.
        Legacy fallback: old DistanceFront/DistanceLeft/... sensors.

        Important:
            min_obstacle_dist_m is horizontal only.
            down_dist_m is separate and does not participate in horizontal
            emergency termination.
        """
        max_d = float(self.cfg.lidar_max_dist_m)
        source = str(getattr(self.cfg, "obstacle_sensor_source", "lidar")).lower().strip()

        def empty_state(reason: str) -> dict:
            return {
                "front_dist_m": max_d,
                "front_left_dist_m": max_d,
                "front_right_dist_m": max_d,
                "left_dist_m": max_d,
                "right_dist_m": max_d,
                "back_dist_m": max_d,
                "down_dist_m": max_d if altitude_m is None else float(np.clip(float(altitude_m), 0.0, max_d)),
                "min_obstacle_dist_m": max_d,
                "lidar_valid": False,
                "lidar_point_count": 0,
                "obstacle_source": reason,
            }

        if not self.cfg.use_lidar_sectors_obs or source == "none":
            self._last_min_obst = max_d
            self._last_obstacle_state = empty_state("disabled")
            return self._last_obstacle_state

        # Preferred path: 3D LiDAR.
        if source in {"lidar", "auto"}:
            lidar_state = self._read_lidar_obstacle_state_m(altitude_m=altitude_m)
            if lidar_state is not None:
                self._last_min_obst = float(lidar_state["min_obstacle_dist_m"])
                self._last_obstacle_state = lidar_state
                return self._last_obstacle_state

            if not bool(getattr(self.cfg, "use_distance_sensor_fallback", True)):
                self._last_min_obst = max_d
                self._last_obstacle_state = empty_state("lidar_unavailable")
                return self._last_obstacle_state

        # Legacy fallback path: old directional distance sensors.
        if source in {"distance", "lidar", "auto"}:
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

            horizontal_min = min(front, front_left, front_right, left, right, back)
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
                "lidar_valid": False,
                "lidar_point_count": 0,
                "obstacle_source": "distance_fallback" if source in {"lidar", "auto"} else "distance",
            }
            return self._last_obstacle_state

        self._last_min_obst = max_d
        self._last_obstacle_state = empty_state("unknown_source")
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
        self._ep_reward_part_sums = {}
        self._ep_reward_part_last = {}
        self._ep_reward_part_counts = {}
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
        self._focused_yaw_hard_events = 0
        self._yaw_decenter_hard_events = 0
        self._last_focused_yaw_hard_penalty = 0.0
        self._last_distance_proxy_norm = 1.0
        self._prev_tracking_mode = "LOST"
        self._prev_had_target = False
        self._prev_lost_target_time_norm = 1.0
        self._prev_distance_proxy_norm = 1.0
        self._prev_err_x = 0.0
        self._prev_err_y = 0.0
        self._prev_bbox_conf = 0.0
        self._yaw_shield_applied = False
        self._yaw_shield_events = 0
        self._last_yaw_shield_pre = 0.0
        self._last_yaw_shield_post = 0.0
        self._yaw_to_strafe_events = 0
        self._last_yaw_to_strafe_delta_vy = 0.0
        self._agent1_yaw_lock_events = 0
        self._agent1_yaw_lock_applied = False
        self._agent1_yaw_limit_events = 0
        self._agent1_yaw_limit_applied = False
        self._last_blocked_yaw_action = 0.0
        self._last_yaw_lock_penalty = 0.0
        self._last_yaw_allowed_abs = 1.0
        self._non_match_time_s = 0.0
        self._not_approaching_time_s = 0.0
        self._chase_rule_elapsed_s = 0.0
        self._best_distance_proxy_norm = 1.0
        self._best_chase_distance_m = float("inf")
        self._current_chase_distance_m = float("inf")
        self._prev_chase_distance_m = float("inf")
        self._initial_chase_distance_m = float("inf")
        self._distance_damage_time_s = 0.0
        self._distance_damage_ratio = 0.0

        # Multi-camera soft handoff state.
        self._bottom_match = False
        self._bottom_match_streak = 0
        self._bottom_bbox_xyxy = None
        self._bottom_similarity = 0.0
        self._bottom_full_tracker_mode = "LOST"
        self._bottom_full_tracker_raw_mode = "NONE"
        self._bottom_full_tracker_confidence = 0.0
        self._bottom_full_tracker_similarity = 0.0
        self._bottom_full_tracker_updates = 0
        self._bottom_full_tracker_match_frames = 0
        self._bottom_full_tracker_pred_frames = 0
        self._bottom_full_tracker_lost_frames = 0
        self._bottom_tracking_result = None
        self._bottom_stable_bbox_xyxy = None
        self._bottom_last_trusted_raw_bbox_xyxy = None
        self._bottom_kalman_fallback_active = False
        self._bottom_err_x = 0.0
        self._bottom_err_y = 0.0
        self._prev_bottom_abs_err_y = 1.5
        self._bottom_weight = 0.0
        self._front_weight = 1.0
        self._bottom_confirmed = False
        self._bottom_match_fresh = False
        self._bottom_bbox_area_norm = 0.0
        self._bottom_bbox_rel_half_w = 0.0
        self._bottom_bbox_rel_half_h = 0.0
        self._bottom_bbox_rel_err_x = 0.0
        self._bottom_bbox_rel_err_y = 0.0
        self._bottom_bbox_rel_err = 999.0
        self._last_bottom_match_step = -999999
        self._handoff_candidate_gate = False
        self._handoff_ready = False
        self._handoff_phase = "CHASE_FRONT"
        self._handoff_visual_score = 0.0
        self._handoff_visual_scan_trigger = False
        self._handoff_visual_lidar_trigger = False
        self._handoff_lidar_dist_m = float("inf")
        self._handoff_lidar_direction = "none"
        self._last_bottom_scan_step = -999999

        self._last_approach_improvement_m = 0.0
        self._step_chase_distance_improvement_m = 0.0
        self._initial_yaw_rad = None
        self._last_yaw_delta_deg = 0.0
        self._commanded_yaw_delta_deg = 0.0
        self._cumulative_abs_yaw_cmd_deg = 0.0
        self._last_yaw_rate_cmd_dps = 0.0
        self._effective_yaw_delta_deg = 0.0
        self._cached_downward_frame = None
        self._cached_downward_frame_step = -1
        self._airborne_reset_completed = False

        self.tracker_manager = self._create_tracker_manager()
        self._tracking_result = None
        self._stable_bbox_xyxy = None

        self.obs_builder.reset()

        # ------------------------------------------------------------------
        # 1. Reset drone / AirSim vehicle.
        # ------------------------------------------------------------------
        self.client.reset()
        time.sleep(float(self.cfg.reset_settle_sec))

        self.client.enableApiControl(True, vehicle_name=self.cfg.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.cfg.vehicle_name)

        if bool(getattr(self.cfg, "reset_start_airborne_with_pose", True)):
            self._force_airborne_start_pose()
        else:
            self.client.takeoffAsync(vehicle_name=self.cfg.vehicle_name).join()
            time.sleep(float(self.cfg.reset_settle_sec))

            # AirSim uses NED coordinates: negative Z means up.
            self.client.moveToZAsync(
                z=-float(self.cfg.reset_takeoff_altitude_m),
                velocity=float(self.cfg.reset_move_to_z_velocity),
                vehicle_name=self.cfg.vehicle_name,
            ).join()

            time.sleep(float(self.cfg.reset_settle_sec))
            self._airborne_reset_completed = True  # regular takeoff path

        # Store initial yaw after takeoff/altitude stabilization.
        # Strict fine-tuning can terminate the episode if the drone drifts too
        # far from this initial heading instead of chasing/centering the car.
        self._initial_yaw_rad = self._get_current_yaw_rad()
        self._last_yaw_delta_deg = 0.0
        self._commanded_yaw_delta_deg = 0.0
        self._cumulative_abs_yaw_cmd_deg = 0.0
        self._last_yaw_rate_cmd_dps = 0.0
        self._effective_yaw_delta_deg = 0.0

        # ------------------------------------------------------------------
        # 2. Reset target car AFTER the drone is ready and BEFORE the first
        # tracker frame. This prints before/target/after so we can verify that
        # the reset really affected the visible actor.
        # ------------------------------------------------------------------
        self._reset_target_car_debug(settle_sec=0.10)

        # ------------------------------------------------------------------
        # 3. Get a fresh frame after the car reset.
        # ------------------------------------------------------------------
        frame = self._get_frame()

        # ------------------------------------------------------------------
        # 4. One-time manual target selection.
        #
        # The click window internally displays fresh frames until selection.
        # After it closes, get a new frame again so tracker startup uses the
        # current scene state.
        # ------------------------------------------------------------------
        if not self._target_initialized or self.target_fingerprint is None:
            self._click_lock_once()
            frame = self._get_frame()

        # ------------------------------------------------------------------
        # 5. Restore target identity and auto-lock on current frame.
        # ------------------------------------------------------------------
        self.tracker.set_target_fingerprint(self.target_fingerprint)
        self.tracker.set_target_class(self.target_class_id)
        self.tracker.auto_lock_on_fingerprint(frame, use_class_gate=True)

        # Bottom camera owns a full tracker too. It receives the same identity
        # immediately; it will lock when the target becomes visible from below.
        try:
            self.bottom_tracker.set_target_fingerprint(self.target_fingerprint)
            self.bottom_tracker.set_target_class(self.target_class_id)
        except Exception as e:
            if bool(getattr(self.cfg, "print_sensor_errors", False)):
                print(f"[BOTTOM FULL TRACKER WARNING] reset target sync failed: {e}")

        # Recreate bottom stabilizer on every episode so no stale bottom Kalman
        # state leaks between resets.
        self.bottom_tracker_manager = self._create_tracker_manager()
        self._bottom_tracking_result = None
        self._bottom_stable_bbox_xyxy = None
        self._bottom_last_trusted_raw_bbox_xyxy = None
        self._bottom_kalman_fallback_active = False

        # ------------------------------------------------------------------
        # 6. Build a real initial observation instead of returning zeros.
        # ------------------------------------------------------------------
        bbox_raw = self.tracker.update(frame)

        tracking_result = self._update_stable_tracking(
            bbox_raw=bbox_raw,
            frame=frame,
            dt=float(self.cfg.cmd_duration_s),
        )

        bbox = self._bbox_to_observation(
            tracking_result.get("stable_bbox"),
            tracking_result.get("mode", "LOST"),
        )

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

        self._last_distance_proxy_norm = float(obs_dict.get("distance_proxy_norm", 1.0))
        self._best_distance_proxy_norm = float(self._last_distance_proxy_norm)

        initial_chase_distance_m = self._get_drone_to_target_distance_m()
        if initial_chase_distance_m is not None:
            self._initial_chase_distance_m = float(initial_chase_distance_m)
            self._current_chase_distance_m = float(initial_chase_distance_m)
            self._prev_chase_distance_m = float(initial_chase_distance_m)
            self._best_chase_distance_m = float(initial_chase_distance_m)
            self._step_chase_distance_improvement_m = 0.0
        else:
            self._initial_chase_distance_m = float("inf")
            self._current_chase_distance_m = float("inf")
            self._prev_chase_distance_m = float("inf")
            self._best_chase_distance_m = float("inf")
            self._step_chase_distance_improvement_m = 0.0
        self._distance_damage_time_s = 0.0
        self._distance_damage_ratio = 0.0
        self._bottom_match = False
        self._bottom_match_streak = 0
        self._bottom_bbox_xyxy = None
        self._bottom_similarity = 0.0
        self._bottom_full_tracker_mode = "LOST"
        self._bottom_full_tracker_raw_mode = "NONE"
        self._bottom_full_tracker_confidence = 0.0
        self._bottom_full_tracker_similarity = 0.0
        self._bottom_full_tracker_updates = 0
        self._bottom_full_tracker_match_frames = 0
        self._bottom_full_tracker_pred_frames = 0
        self._bottom_full_tracker_lost_frames = 0
        self._bottom_tracking_result = None
        self._bottom_stable_bbox_xyxy = None
        self._bottom_last_trusted_raw_bbox_xyxy = None
        self._bottom_kalman_fallback_active = False
        self._bottom_err_x = 0.0
        self._bottom_err_y = 0.0
        self._prev_bottom_abs_err_y = 1.5
        self._bottom_weight = 0.0
        self._front_weight = 1.0
        self._handoff_ready = False
        self._handoff_phase = "CHASE_FRONT"
        self._handoff_visual_score = 0.0
        self._handoff_visual_lidar_trigger = False
        self._handoff_lidar_dist_m = float("inf")
        self._handoff_lidar_direction = "none"
        self._last_bottom_scan_step = -999999

        self._camera_authority = "FRONT_PRIMARY"
        self._active_camera = "front"
        self._fusion_has_target = False
        self._fusion_tracking_mode = "LOST"
        self._fusion_err_x = 0.0
        self._fusion_err_y = 0.0
        self._fusion_confidence = 0.0
        self._bottom_authority_streak = 0
        self._front_authority_streak = 0
        self._last_bottom_valid_step = -999999
        self._last_front_valid_step = -999999
        self._last_bottom_alignment_assist_vx = 0.0
        self._last_bottom_alignment_assist_vy = 0.0
        self._bottom_alignment_assist_events = 0

        self._speed_stage = "CHASE_FAST"
        self._speed_stage_changes = 0
        self._last_fast_chase_assist_vx = 0.0
        self._last_stage_vx_scale = float(getattr(self.cfg, "vx_scale", 2.8))
        self._last_stage_vy_scale = float(getattr(self.cfg, "vy_scale", 2.0))
        self._last_bottom_pd_assist_vx = 0.0
        self._last_bottom_pd_assist_vy = 0.0
        self._bottom_prev_err_x_for_velocity = 0.0
        self._bottom_prev_err_y_for_velocity = 0.0
        self._bottom_img_vel_x = 0.0
        self._bottom_img_vel_y = 0.0
        self._prev_bottom_img_speed = 0.0
        self._bottom_velocity_ready = False
        self._bottom_velocity_ready_streak = 0

        self._last_approach_improvement_m = 0.0
        self._prev_tracking_mode = str(tracking_result.get("mode", "LOST") or "LOST")
        self._prev_had_target = bool(float(obs_dict.get("has_target", 0.0)) > 0.5)
        self._prev_lost_target_time_norm = float(obs_dict.get("lost_target_time_norm", 0.0))
        self._prev_distance_proxy_norm = float(obs_dict.get("distance_proxy_norm", 1.0))
        self._prev_err_x = float(obs_dict.get("err_x", 0.0))
        self._prev_err_y = float(obs_dict.get("err_y", 0.0))
        self._prev_bbox_conf = float(obs_dict.get("bbox_conf", 0.0))

        if self.cfg.print_reset:
            print(f"[RESET] Episode {self.episode_id}")

        info = {
            "obs_dict": obs_dict,
            "front_camera_name": str(getattr(self.cfg, "front_camera_name", "front_center")),
            "downward_camera_name": str(getattr(self.cfg, "downward_camera_name", "bottom_center")),
            "front_full_tracker_updates": int(getattr(self, "_front_full_tracker_updates", 0)),
            "bottom_full_tracker_updates": int(getattr(self, "_bottom_full_tracker_updates", 0)),
            "bottom_full_tracker_mode": str(getattr(self, "_bottom_full_tracker_mode", "LOST")),
        }
        return obs, info

    def step(self, action):
        t0 = time.time()

        raw_action = np.asarray(action, dtype=np.float32)
        raw_action = np.clip(raw_action, -1.0, 1.0)

        # ------------------------------------------------------------------
        # Gentle startup yaw guard
        # ------------------------------------------------------------------
        # Do NOT reset/freeze the visual tracker here.
        # We only prevent a large random PPO yaw sample during the very first
        # startup steps of a new episode.
        #
        # Important:
        #   - forward/side/vz are NOT clamped here.
        #   - the tracker keeps its bbox/fingerprint/runtime continuity.
        #   - the policy can still move and reacquire.
        startup_yaw_guard_steps = int(getattr(self.cfg, "startup_yaw_guard_steps", 12))
        startup_max_yaw_action = float(getattr(self.cfg, "startup_max_yaw_action", 0.12))

        if self.step_in_episode < startup_yaw_guard_steps:
            raw_action[3] = np.clip(raw_action[3], -startup_max_yaw_action, startup_max_yaw_action)

            if getattr(self.cfg, "print_reset", False) and self.step_in_episode == 0:
                print(
                    "[STARTUP YAW GUARD] "
                    f"steps={startup_yaw_guard_steps} "
                    f"max_yaw_action={startup_max_yaw_action}"
                )

        # ------------------------------------------------------------------
        # Anti-yaw action shield + yaw-to-strafe conversion
        # ------------------------------------------------------------------
        # Strategy:
        # When the previous visual state says the target is visible in MATCH,
        # do not let the policy solve horizontal image error by rotating the
        # drone. Keep the front heading stable and convert horizontal image error
        # into lateral body-frame movement.
        #
        # This directly targets the observed bad habit:
        #   starts centered -> drifts right -> gets close -> yaws 90 degrees left
        #
        # Desired behavior:
        #   starts centered -> if target drifts right/left, strafe right/left
        #   while keeping yaw approximately fixed.
        self._yaw_shield_applied = False
        self._last_yaw_shield_pre = float(raw_action[3])
        self._last_yaw_shield_post = float(raw_action[3])
        self._last_yaw_to_strafe_delta_vy = 0.0

        prev_mode = str(getattr(self, "_prev_tracking_mode", "LOST") or "LOST").upper()
        prev_has_target = bool(getattr(self, "_prev_had_target", False))
        prev_err_x = float(getattr(self, "_prev_err_x", 0.0))
        prev_abs_err_x = abs(prev_err_x)
        prev_bbox_conf = float(getattr(self, "_prev_bbox_conf", 0.0))
        handoff_phase_now = str(getattr(self, "_handoff_phase", "CHASE_FRONT")).upper()
        current_distance_for_lock = float(getattr(self, "_current_chase_distance_m", float("inf")))

        visual_match_ok = (
            prev_has_target
            and (
                (not bool(getattr(self.cfg, "yaw_to_strafe_match_only", True)))
                or prev_mode == "MATCH"
            )
            and prev_bbox_conf >= float(getattr(self.cfg, "yaw_shield_min_conf", 0.45))
        )

        if bool(getattr(self.cfg, "yaw_to_strafe_enabled", True)) and visual_match_ok:
            # Convert x image error into lateral motion.
            # Convention used here:
            #   err_x > 0 means target is to the right in the image,
            #   so vy should be positive to move the drone right.
            if prev_abs_err_x > float(getattr(self.cfg, "yaw_to_strafe_x_deadband", 0.035)):
                gain = float(getattr(self.cfg, "yaw_to_strafe_gain", 0.85))
                blend = float(np.clip(getattr(self.cfg, "yaw_to_strafe_blend", 0.80), 0.0, 1.0))
                max_vy = float(getattr(self.cfg, "yaw_to_strafe_max_abs_vy", 0.75))
                desired_vy_action = float(np.clip(gain * prev_err_x, -max_vy, max_vy))
                old_vy = float(raw_action[1])
                raw_action[1] = float(np.clip((1.0 - blend) * old_vy + blend * desired_vy_action, -1.0, 1.0))
                self._last_yaw_to_strafe_delta_vy = float(raw_action[1] - old_vy)
                self._yaw_to_strafe_events += 1

            # Keep yaw tiny/zero while visible. This is an action-level rule,
            # not a terminal punishment, so PPO still receives long episodes.
            max_visible_yaw = float(getattr(self.cfg, "yaw_visible_max_abs", 0.025))
            if prev_abs_err_x < float(getattr(self.cfg, "yaw_visible_x_threshold", 0.45)):
                clipped_yaw = float(np.clip(raw_action[3], -max_visible_yaw, max_visible_yaw))
                if abs(clipped_yaw - float(raw_action[3])) > 1e-6:
                    self._yaw_shield_applied = True
                    self._yaw_shield_events += 1
                    raw_action[3] = clipped_yaw

            # Close-range heading lock: once near handoff geometry, do not rotate
            # the front camera toward the car. Let lateral motion and bottom scan
            # handle the transition.
            if bool(getattr(self.cfg, "yaw_close_heading_lock_enabled", True)):
                close_lock = (
                    handoff_phase_now == "BOTTOM_SCAN"
                    or (
                        np.isfinite(current_distance_for_lock)
                        and current_distance_for_lock <= float(getattr(self.cfg, "yaw_close_heading_lock_distance_m", 11.5))
                    )
                )
                if close_lock:
                    max_close_yaw = float(getattr(self.cfg, "yaw_close_heading_lock_max_abs_yaw", 0.0))
                    clipped_yaw = float(np.clip(raw_action[3], -max_close_yaw, max_close_yaw))
                    if abs(clipped_yaw - float(raw_action[3])) > 1e-6:
                        self._yaw_shield_applied = True
                        self._yaw_shield_events += 1
                        raw_action[3] = clipped_yaw

        elif bool(getattr(self.cfg, "yaw_action_shield_enabled", True)):
            # Conservative fallback from the previous patch:
            # only block yaw when the target is clearly horizontally centered.
            in_allowed_phase = (
                handoff_phase_now != "BOTTOM_SCAN"
                or bool(getattr(self.cfg, "yaw_shield_allow_in_handoff_scan", True))
            )
            shield_match_ok = (
                (not bool(getattr(self.cfg, "yaw_shield_match_only", True)))
                or prev_mode == "MATCH"
            )

            if (
                prev_has_target
                and shield_match_ok
                and in_allowed_phase
                and prev_bbox_conf >= float(getattr(self.cfg, "yaw_shield_min_conf", 0.45))
                and prev_abs_err_x < float(getattr(self.cfg, "yaw_shield_x_center_threshold", 0.20))
            ):
                if prev_abs_err_x < float(getattr(self.cfg, "yaw_shield_x_zero_threshold", 0.07)):
                    max_yaw_abs = 0.0
                else:
                    max_yaw_abs = float(getattr(self.cfg, "yaw_shield_max_abs_when_centered", 0.035))

                clipped_yaw = float(np.clip(raw_action[3], -max_yaw_abs, max_yaw_abs))
                if abs(clipped_yaw - float(raw_action[3])) > 1e-6:
                    self._yaw_shield_applied = True
                    self._yaw_shield_events += 1
                    raw_action[3] = clipped_yaw

        self._last_yaw_shield_post = float(raw_action[3])

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

        safety_phase_pre = self._compute_safety_phase(
            drone_state=pre_drone_state,
            obstacle_dict=obstacle_dict_pre,
            tracking_mode=getattr(self, "_prev_tracking_mode", "LOST"),
        )

        safe_action, safety_info = safety_filter(
            raw_action=raw_action,
            obstacle_state=obstacle_dict_pre,
            drone_state=drone_state_dict,
            config=self.safety_config,
            safety_phase=safety_phase_pre,
        )

        if bool(safety_info.get("safety_intervention", False)):
            self._safety_interventions += 1

        # ------------------------------------------------------------------
        # Agent-1 adaptive yaw authority
        # ------------------------------------------------------------------
        # Yaw must remain available. A target can move nonlinearly, turn, or
        # leave the camera, and then yaw is a legitimate control action.
        #
        # However, yaw should not be the default way to correct small horizontal
        # image error. For small err_x, the policy should prefer vy/strafe.
        #
        # Rule:
        #   - visible + centered      -> tiny yaw authority
        #   - visible + moderate xerr -> limited yaw authority
        #   - visible + large xerr    -> larger yaw authority
        #   - lost/recovery           -> full yaw authority
        #
        # This keeps nonlinear-turn capability while removing the free
        # close-range 90-degree orbit solution.
        self._agent1_yaw_lock_applied = False
        self._agent1_yaw_limit_applied = False
        self._last_blocked_yaw_action = 0.0
        self._last_yaw_lock_penalty = 0.0
        self._last_yaw_allowed_abs = 1.0

        if bool(getattr(self.cfg, "agent1_adaptive_yaw_enabled", True)):
            prev_mode_for_yaw = str(getattr(self, "_prev_tracking_mode", "LOST") or "LOST").upper()
            prev_has_target_for_yaw = bool(getattr(self, "_prev_had_target", False))
            prev_err_x_for_yaw = float(getattr(self, "_prev_err_x", 0.0))
            prev_abs_err_x_for_yaw = abs(prev_err_x_for_yaw)
            prev_lost_norm_for_yaw = float(getattr(self, "_prev_lost_target_time_norm", 0.0))
            phase_for_yaw = str(getattr(self, "_handoff_phase", "CHASE_FRONT")).upper()
            distance_for_yaw = float(getattr(self, "_current_chase_distance_m", float("inf")))

            yaw_allowed_abs = 1.0

            if (
                not prev_has_target_for_yaw
                or prev_mode_for_yaw not in {"MATCH", "PRED"}
            ):
                if (
                    bool(getattr(self.cfg, "agent1_yaw_full_authority_when_lost", True))
                    and prev_lost_norm_for_yaw >= float(getattr(self.cfg, "agent1_recovery_yaw_after_lost_norm", 0.28))
                ):
                    yaw_allowed_abs = 1.0
                else:
                    yaw_allowed_abs = float(getattr(self.cfg, "agent1_yaw_max_high_error", 0.42))
            else:
                if prev_abs_err_x_for_yaw <= float(getattr(self.cfg, "agent1_yaw_deadband_x", 0.04)):
                    yaw_allowed_abs = float(getattr(self.cfg, "agent1_yaw_max_centered", 0.025))
                elif prev_abs_err_x_for_yaw <= float(getattr(self.cfg, "agent1_yaw_low_error_x", 0.12)):
                    yaw_allowed_abs = float(getattr(self.cfg, "agent1_yaw_max_low_error", 0.080))
                elif prev_abs_err_x_for_yaw <= float(getattr(self.cfg, "agent1_yaw_mid_error_x", 0.28)):
                    yaw_allowed_abs = float(getattr(self.cfg, "agent1_yaw_max_mid_error", 0.180))
                else:
                    yaw_allowed_abs = float(getattr(self.cfg, "agent1_yaw_max_high_error", 0.420))

                if np.isfinite(distance_for_yaw) and distance_for_yaw <= float(getattr(self.cfg, "agent1_close_yaw_distance_m", 9.0)):
                    yaw_allowed_abs *= float(getattr(self.cfg, "agent1_close_yaw_scale", 0.45))

                if phase_for_yaw == "BOTTOM_SCAN":
                    yaw_allowed_abs *= float(getattr(self.cfg, "agent1_bottom_scan_yaw_scale", 0.35))
                    yaw_allowed_abs = max(
                        yaw_allowed_abs,
                        float(getattr(self.cfg, "agent1_min_bottom_scan_yaw_max", 0.060)),
                    )

            yaw_allowed_abs = float(np.clip(yaw_allowed_abs, 0.0, 1.0))
            self._last_yaw_allowed_abs = yaw_allowed_abs

            original_yaw_action = float(safe_action[3])
            limited_yaw_action = float(np.clip(original_yaw_action, -yaw_allowed_abs, yaw_allowed_abs))

            if abs(limited_yaw_action - original_yaw_action) > float(getattr(self.cfg, "agent1_yaw_limit_deadband", 0.010)):
                self._agent1_yaw_limit_applied = True
                self._agent1_yaw_lock_applied = True
                self._agent1_yaw_limit_events += 1
                self._agent1_yaw_lock_events += 1
                self._last_blocked_yaw_action = original_yaw_action - limited_yaw_action
                safe_action[3] = limited_yaw_action
            else:
                safe_action[3] = limited_yaw_action

        # ------------------------------------------------------------------
        # Staged speed + bottom-camera velocity matching controller
        # ------------------------------------------------------------------
        # CHASE_FAST:
        #   Bias vx toward full throttle until the target is sufficiently visible
        #   in the bottom camera.
        #
        # BOTTOM_VELOCITY_MATCH:
        #   Use image-space relative velocity d(Bcx,Bcy)/dt together with
        #   position error. This is a PD visual-servo controller:
        #       desired_vx = -Kp_y * Bcy - Kd_y * dBcy_dt
        #       desired_vy =  Kp_x * Bcx + Kd_x * dBcx_dt
        #
        # BOTTOM_LANDING_READY:
        #   Position and relative image velocity are both stable. This is the
        #   correct pre-landing state for Agent 2.
        self._last_bottom_alignment_assist_vx = 0.0
        self._last_bottom_alignment_assist_vy = 0.0
        self._last_bottom_pd_assist_vx = 0.0
        self._last_bottom_pd_assist_vy = 0.0
        self._last_fast_chase_assist_vx = 0.0

        if bool(getattr(self.cfg, "staged_speed_enabled", True)):
            barea = float(getattr(self, "_bottom_bbox_area_norm", 0.0))
            center_error = float(getattr(self, "_bottom_center_error", 0.0))
            bottom_seen = bool(getattr(self, "_bottom_match", False))
            real_dist_for_speed = float(getattr(self, "_current_chase_distance_m", float("inf")))
            stage = str(getattr(self, "_speed_stage", "CHASE_FAST"))

            enter_area = float(getattr(self.cfg, "bottom_velocity_match_enter_area", 0.036))
            release_area = float(getattr(self.cfg, "bottom_velocity_match_release_area", 0.022))
            release_center_error = float(getattr(self.cfg, "bottom_velocity_match_release_center_error", 0.68))
            ready_streak_required = int(getattr(self.cfg, "bottom_velocity_ready_streak_required", 12))

            if stage == "CHASE_FAST":
                if bottom_seen and barea >= enter_area:
                    self._speed_stage = "BOTTOM_VELOCITY_MATCH"
                    self._speed_stage_changes += 1
            elif stage in {"BOTTOM_VELOCITY_MATCH", "BOTTOM_LANDING_READY"}:
                if (not bottom_seen) or barea < release_area or center_error > release_center_error:
                    self._speed_stage = "CHASE_FAST"
                    self._speed_stage_changes += 1
                elif int(getattr(self, "_bottom_velocity_ready_streak", 0)) >= ready_streak_required:
                    self._speed_stage = "BOTTOM_LANDING_READY"
                else:
                    self._speed_stage = "BOTTOM_VELOCITY_MATCH"

            if self._speed_stage == "CHASE_FAST":
                if real_dist_for_speed > float(getattr(self.cfg, "fast_chase_min_real_distance_m", 2.20)):
                    old_vx_for_fast = float(safe_action[0])
                    floor_vx = float(getattr(self.cfg, "fast_chase_forward_action_floor", 0.92))
                    blend_fast = float(np.clip(getattr(self.cfg, "fast_chase_forward_blend", 0.70), 0.0, 1.0))
                    max_fast = float(np.clip(getattr(self.cfg, "fast_chase_max_action", 1.0), 0.0, 1.0))
                    target_vx_for_fast = float(np.clip(max(floor_vx, old_vx_for_fast), -max_fast, max_fast))
                    safe_action[0] = float(np.clip((1.0 - blend_fast) * old_vx_for_fast + blend_fast * target_vx_for_fast, -1.0, 1.0))
                    self._last_fast_chase_assist_vx = float(safe_action[0] - old_vx_for_fast)

        if (
            bool(getattr(self.cfg, "bottom_velocity_matching_enabled", True))
            and str(getattr(self, "_speed_stage", "CHASE_FAST")) in {"BOTTOM_VELOCITY_MATCH", "BOTTOM_LANDING_READY"}
            and bool(getattr(self, "_bottom_match", False))
        ):
            max_action = float(np.clip(getattr(self.cfg, "bottom_pd_max_action", 1.0), 0.05, 1.0))
            if str(getattr(self, "_speed_stage", "")) == "BOTTOM_LANDING_READY":
                max_action = float(np.clip(getattr(self.cfg, "bottom_pd_ready_max_action", 0.42), 0.05, 1.0))

            bcx = float(getattr(self, "_bottom_err_x", 0.0))
            bcy = float(getattr(self, "_bottom_err_y", 0.0))
            dbx_dt = float(getattr(self, "_bottom_img_vel_x", 0.0))
            dby_dt = float(getattr(self, "_bottom_img_vel_y", 0.0))

            desired_vx = (
                -float(getattr(self.cfg, "bottom_pd_kp_y_to_vx", 1.25)) * bcy
                -float(getattr(self.cfg, "bottom_pd_kd_y_to_vx", 0.34)) * dby_dt
            )
            desired_vy = (
                float(getattr(self.cfg, "bottom_pd_kp_x_to_vy", 0.90)) * bcx
                +float(getattr(self.cfg, "bottom_pd_kd_x_to_vy", 0.26)) * dbx_dt
            )

            desired_vx = float(np.clip(desired_vx, -max_action, max_action))
            desired_vy = float(np.clip(desired_vy, -max_action, max_action))

            blend = float(np.clip(getattr(self.cfg, "bottom_pd_blend", 0.72), 0.0, 1.0))
            old_vx_action = float(safe_action[0])
            old_vy_action = float(safe_action[1])
            safe_action[0] = float(np.clip((1.0 - blend) * old_vx_action + blend * desired_vx, -1.0, 1.0))
            safe_action[1] = float(np.clip((1.0 - blend) * old_vy_action + blend * desired_vy, -1.0, 1.0))

            self._last_bottom_alignment_assist_vx = float(safe_action[0] - old_vx_action)
            self._last_bottom_alignment_assist_vy = float(safe_action[1] - old_vy_action)
            self._last_bottom_pd_assist_vx = float(desired_vx)
            self._last_bottom_pd_assist_vy = float(desired_vy)
            self._bottom_alignment_assist_events += 1

        # Dynamic stage-specific command scaling.
        # This lets CHASE_FAST use a much higher physical speed while keeping
        # bottom-camera alignment and landing-ready motion controlled.
        speed_vx_scale = float(self.cfg.vx_scale)
        speed_vy_scale = float(self.cfg.vy_scale)

        if bool(getattr(self.cfg, "dynamic_stage_speed_enabled", True)):
            stage_for_speed = str(getattr(self, "_speed_stage", "CHASE_FAST"))
            if stage_for_speed == "CHASE_FAST":
                dist_m = float(getattr(self, "_current_chase_distance_m", float("inf")))
                near_m = float(getattr(self.cfg, "chase_fast_distance_near_m", 3.0))
                far_m = float(getattr(self.cfg, "chase_fast_distance_far_m", 14.0))
                if not np.isfinite(dist_m):
                    dist_m = far_m
                ratio = float(np.clip((dist_m - near_m) / max(1e-6, far_m - near_m), 0.0, 1.0))
                speed_vx_scale = (
                    float(getattr(self.cfg, "chase_fast_vx_scale_min", 3.5))
                    + ratio * (
                        float(getattr(self.cfg, "chase_fast_vx_scale_max", 5.8))
                        - float(getattr(self.cfg, "chase_fast_vx_scale_min", 3.5))
                    )
                )
                speed_vy_scale = float(getattr(self.cfg, "chase_fast_vy_scale", 2.4))

            elif stage_for_speed == "BOTTOM_VELOCITY_MATCH":
                speed_vx_scale = float(getattr(self.cfg, "bottom_velocity_match_vx_scale", 2.8))
                speed_vy_scale = float(getattr(self.cfg, "bottom_velocity_match_vy_scale", 2.2))

            elif stage_for_speed == "BOTTOM_LANDING_READY":
                speed_vx_scale = float(getattr(self.cfg, "bottom_landing_ready_vx_scale", 1.2))
                speed_vy_scale = float(getattr(self.cfg, "bottom_landing_ready_vy_scale", 1.0))

        self._last_stage_vx_scale = float(speed_vx_scale)
        self._last_stage_vy_scale = float(speed_vy_scale)

        vx_cmd = float(safe_action[0]) * speed_vx_scale
        vy_cmd = float(safe_action[1]) * speed_vy_scale

        # Smooth forward/lateral command changes to reduce visible attitude rocking.
        # Pitch rocking is mainly driven by abrupt vx acceleration/deceleration.
        stage_for_slew = str(getattr(self, "_speed_stage", "CHASE_FAST"))
        self._last_vx_slew_delta = 0.0
        self._last_vx_slew_limited = False
        self._last_vy_slew_delta = 0.0
        self._last_vy_slew_limited = False

        if bool(getattr(self.cfg, "vx_slew_limit_enabled", True)):
            if stage_for_slew == "BOTTOM_LANDING_READY":
                max_dvx = float(getattr(self.cfg, "landing_ready_max_vx_delta_mps_per_step", 0.25))
            elif stage_for_slew == "BOTTOM_VELOCITY_MATCH":
                max_dvx = float(getattr(self.cfg, "bottom_match_max_vx_delta_mps_per_step", 0.45))
            else:
                max_dvx = float(getattr(self.cfg, "chase_max_vx_delta_mps_per_step", 0.90))

            prev_vx_cmd = float(getattr(self, "_prev_vx_cmd_for_slew", 0.0))
            raw_vx_cmd = float(vx_cmd)
            vx_cmd = float(np.clip(raw_vx_cmd, prev_vx_cmd - max_dvx, prev_vx_cmd + max_dvx))
            self._last_vx_slew_delta = float(vx_cmd - raw_vx_cmd)
            self._last_vx_slew_limited = bool(abs(self._last_vx_slew_delta) > 1e-6)

        if bool(getattr(self.cfg, "vy_slew_limit_enabled", False)):
            if stage_for_slew == "BOTTOM_LANDING_READY":
                max_dvy = float(getattr(self.cfg, "landing_ready_max_vy_delta_mps_per_step", 0.30))
            elif stage_for_slew == "BOTTOM_VELOCITY_MATCH":
                max_dvy = float(getattr(self.cfg, "bottom_match_max_vy_delta_mps_per_step", 0.50))
            else:
                max_dvy = float(getattr(self.cfg, "chase_max_vy_delta_mps_per_step", 0.90))

            prev_vy_cmd = float(getattr(self, "_prev_vy_cmd_for_slew", 0.0))
            raw_vy_cmd = float(vy_cmd)
            vy_cmd = float(np.clip(raw_vy_cmd, prev_vy_cmd - max_dvy, prev_vy_cmd + max_dvy))
            self._last_vy_slew_delta = float(vy_cmd - raw_vy_cmd)
            self._last_vy_slew_limited = bool(abs(self._last_vy_slew_delta) > 1e-6)

        self._prev_vx_cmd_for_slew = float(vx_cmd)
        self._prev_vy_cmd_for_slew = float(vy_cmd)

        # AirSim NED:
        #   vz > 0 means down
        #   vz < 0 means up
        #
        # Important:
        # freeze_vz no longer means "send vz=0", because SimpleFlight may still
        # drift down over time. Instead, freeze_vz enables a small altitude-hold
        # controller around a configurable target altitude.
        if bool(self.cfg.freeze_vz):
            if bool(self.cfg.altitude_hold_enabled):
                target_alt_m = float(self.cfg.altitude_hold_target_m)
                alt_error_m = float(pre_drone_state.altitude_m) - target_alt_m
                vz_cmd = float(np.clip(
                    float(self.cfg.altitude_hold_kp) * alt_error_m,
                    -float(self.cfg.altitude_hold_max_vz_mps),
                    float(self.cfg.altitude_hold_max_vz_mps),
                ))
            else:
                vz_cmd = 0.0
        else:
            vz_cmd = float(safe_action[2]) * self.cfg.vz_scale

        # Dynamic minimum-distance-to-target guard.
        # distance_proxy_norm is lower when the target is closer/larger.
        # If too close, do not allow more forward movement.
        if (
            bool(self.cfg.block_forward_when_too_close)
            and float(self._last_distance_proxy_norm) < float(self.cfg.min_target_distance_proxy)
            and vx_cmd > 0.0
        ):
            vx_cmd = 0.0
            safety_info.setdefault("safety_reasons", []).append("target_too_close_block_forward")
            safety_info["safety_intervention"] = True

        # Dynamic altitude lower-bound safety clamp.
        # State-aware rule:
        #   TAKEOFF may force climb.
        #   CHASE should not climb forever from a noisy/near-body down reading;
        #   it should only block descent. This keeps the target in view and still
        #   prevents unsafe descent into the ground.
        if bool(self.cfg.altitude_safety_enabled):
            alt_m = float(pre_drone_state.altitude_m)
            phase_for_altitude = str(safety_phase_pre).upper()
            force_up_allowed = (
                phase_for_altitude == "TAKEOFF"
                or bool(getattr(self.cfg, "altitude_force_up_in_chase", False))
            )

            if alt_m <= float(self.cfg.min_termination_altitude_m):
                if force_up_allowed:
                    vz_cmd = min(vz_cmd, -abs(float(self.cfg.emergency_climb_vz_mps)))
                    safety_info.setdefault("safety_reasons", []).append("altitude_emergency_force_up")
                elif vz_cmd > 0.0:
                    vz_cmd = 0.0
                    safety_info.setdefault("safety_reasons", []).append("altitude_emergency_block_descent")
                safety_info["safety_intervention"] = True

            elif alt_m < float(self.cfg.min_safe_altitude_m):
                if force_up_allowed:
                    vz_cmd = min(vz_cmd, -abs(float(self.cfg.low_altitude_climb_vz_mps)))
                    safety_info.setdefault("safety_reasons", []).append("altitude_low_force_up")
                elif vz_cmd > 0.0:
                    vz_cmd = 0.0
                    safety_info.setdefault("safety_reasons", []).append("altitude_low_block_descent")
                safety_info["safety_intervention"] = True

        # Recovery authority:
        # While the previous observation was not a clean MATCH, give yaw more
        # authority and reduce forward motion. This gives the policy a real chance
        # to spin/search quickly before focus_fail_sec ends.
        recovery_mode_active = (
            str(getattr(self, "_prev_tracking_mode", "LOST")) != "MATCH"
            or not bool(getattr(self, "_prev_had_target", False))
        )

        yaw_scale_dps = float(self.cfg.yaw_rate_scale_dps)

        if recovery_mode_active:
            yaw_scale_dps = float(getattr(self.cfg, "recovery_yaw_rate_scale_dps", yaw_scale_dps))

            forward_scale = float(getattr(self.cfg, "recovery_forward_scale", 1.0))
            if vx_cmd > 0.0:
                vx_cmd *= float(np.clip(forward_scale, 0.0, 1.0))
                safety_info.setdefault("safety_reasons", []).append("recovery_forward_slowdown")

        yaw_rate_cmd = float(safe_action[3]) * yaw_scale_dps

        self.tracker.last_yaw_rate_cmd_dps = float(yaw_rate_cmd)

        # Robust commanded-yaw integration.
        # AirSim orientation readings can sometimes be unavailable or remain at
        # 0.0 even when the debug view clearly shows a yaw rotation. The reward
        # and termination logic must therefore also use the yaw command that was
        # actually sent to the simulator.
        cmd_dt = float(getattr(self.cfg, "cmd_duration_s", 0.10))
        self._last_yaw_rate_cmd_dps = float(yaw_rate_cmd)
        self._commanded_yaw_delta_deg += float(yaw_rate_cmd) * cmd_dt
        self._cumulative_abs_yaw_cmd_deg += abs(float(yaw_rate_cmd)) * cmd_dt

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

        tracking_result = self._update_stable_tracking(
            bbox_raw=bbox_raw,
            frame=frame,
            dt=dt,
        )
        tracking_mode = tracking_result.get("mode", "LOST")

        self._front_full_tracker_updates += 1
        self._front_full_tracker_mode = str(tracking_mode or "LOST").upper()
        self._front_full_tracker_raw_mode = str(
            getattr(self.tracker, "last_raw_mode", "") or getattr(self.tracker, "last_mode", "NONE") or "NONE"
        )
        if self._front_full_tracker_mode == "MATCH":
            self._front_full_tracker_match_frames += 1
        elif self._front_full_tracker_mode == "PRED":
            self._front_full_tracker_pred_frames += 1
        else:
            self._front_full_tracker_lost_frames += 1

        # Episode MATCH/PRED/NONE statistics are updated later, after
        # dual-camera fusion chooses the active camera. Counting here would
        # incorrectly mark successful bottom-primary tracking as NONE when
        # the front camera naturally loses the target after handoff.

        bbox = self._bbox_to_observation(
            tracking_result.get("stable_bbox"),
            tracking_mode,
        )
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

        # Current safety phase for both termination gating and reward shaping.
        # TAKEOFF is not part of the learning objective. During TAKEOFF the
        # safety layer may own the vehicle until it clears the ground, so chase
        # rules must not terminate the episode yet.
        safety_phase_now = str(safety_info.get("safety_phase", "CHASE")).upper()
        takeoff_phase_now = safety_phase_now == "TAKEOFF"
        chase_rules_enabled = bool(
            safety_phase_now == "CHASE"
            and self.step_in_episode >= int(getattr(self.cfg, "chase_rules_min_steps_after_takeoff", 0))
        )

        # ------------------------------------------------------------------
        # Strict chase/center fine-tuning monitors.
        # ------------------------------------------------------------------
        current_tracking_mode_for_rules = str(tracking_mode or "LOST").upper()
        current_has_target_for_rules = bool(float(obs_dict.get("has_target", 0.0)) > 0.5)
        current_distance_proxy = float(obs_dict.get("distance_proxy_norm", self._last_distance_proxy_norm))

        current_handoff_phase_for_rules = str(getattr(self, "_handoff_phase", "CHASE_FRONT")).upper()
        handoff_search_active = current_handoff_phase_for_rules in {"BOTTOM_SCAN", "HANDOFF_OVERLAP", "HANDOFF_READY"}

        if not chase_rules_enabled:
            self._non_match_time_s = 0.0
        elif (
            current_tracking_mode_for_rules == "MATCH"
            or bool(getattr(self, "_bottom_match", False))
            or handoff_search_active
        ):
            self._non_match_time_s = 0.0
        else:
            self._non_match_time_s += float(dt)

        # Real-distance chase shaping rule.
        # Keep the best real horizontal drone<->car distance seen so far in this
        # episode, but do not hard-fail only because the policy did not improve
        # fast enough. No-improvement is a soft learning signal.
        #
        # Hard failure is reserved for actual damage: the drone moves away from
        # the original target distance by a large ratio, e.g. +20%.
        current_chase_distance_m = self._get_drone_to_target_distance_m()

        self._update_soft_handoff_state(
            current_distance_m=current_chase_distance_m,
            front_tracking_mode=current_tracking_mode_for_rules,
            obs_dict=obs_dict,
            obstacle_dict=obstacle_dict,
        )

        # ------------------------------------------------------------------
        # Active-camera observation rebuild
        # ------------------------------------------------------------------
        # From this point onward, reward, returned observation, target-lost
        # logic, and previous-step state should follow the fused authority, not
        # blindly the front camera. If bottom is primary, bottom_err_x/y become
        # the control space.
        active_camera = str(getattr(self, "_active_camera", "front")).lower()
        active_tracking_mode = str(getattr(self, "_fusion_tracking_mode", tracking_mode) or tracking_mode)

        if (
            bool(getattr(self.cfg, "dual_camera_fusion_enabled", True))
            and active_camera == "bottom"
            and getattr(self, "_bottom_bbox_xyxy", None) is not None
        ):
            bottom_mode_for_obs = "MATCH" if bool(getattr(self, "_bottom_match", False)) else "PRED"
            bottom_bbox_for_obs = self._bbox_to_observation(
                getattr(self, "_bottom_bbox_xyxy", None),
                bottom_mode_for_obs,
            )
            if bottom_bbox_for_obs is not None:
                obs, obs_dict = self.obs_builder.build(
                    bbox=bottom_bbox_for_obs,
                    drone_state=drone_state,
                    obstacle_state=obstacle_state,
                    dt=dt,
                )
                tracking_mode = bottom_mode_for_obs
                current_tracking_mode_for_rules = bottom_mode_for_obs
                current_has_target_for_rules = True
                current_distance_proxy = float(obs_dict.get("distance_proxy_norm", current_distance_proxy))
            else:
                tracking_mode = active_tracking_mode
        else:
            tracking_mode = active_tracking_mode
            current_tracking_mode_for_rules = str(tracking_mode or "LOST").upper()
            current_has_target_for_rules = bool(float(obs_dict.get("has_target", 0.0)) > 0.5)
            current_distance_proxy = float(obs_dict.get("distance_proxy_norm", current_distance_proxy))

        obs_dict["active_camera"] = str(getattr(self, "_active_camera", "front"))
        obs_dict["camera_authority"] = str(getattr(self, "_camera_authority", "FRONT_PRIMARY"))
        obs_dict["fusion_has_target"] = float(1.0 if bool(getattr(self, "_fusion_has_target", False)) else 0.0)

        # ------------------------------------------------------------------
        # Bottom-camera relative image velocity estimation
        # ------------------------------------------------------------------
        # Estimate target motion in the downward image:
        #   dBcx/dt, dBcy/dt
        # This is the key signal for matching the moving platform velocity.
        # The EMA keeps it stable enough for RL/reward and PD assist.
        bottom_match_for_vel = bool(getattr(self, "_bottom_match", False))
        if bottom_match_for_vel:
            prev_vx_img_speed = float(np.sqrt(
                float(getattr(self, "_bottom_img_vel_x", 0.0)) ** 2
                + float(getattr(self, "_bottom_img_vel_y", 0.0)) ** 2
            ))

            curr_bcx = float(getattr(self, "_bottom_err_x", 0.0))
            curr_bcy = float(getattr(self, "_bottom_err_y", 0.0))
            prev_bcx = float(getattr(self, "_bottom_prev_err_x_for_velocity", curr_bcx))
            prev_bcy = float(getattr(self, "_bottom_prev_err_y_for_velocity", curr_bcy))
            dt_safe = max(1e-3, float(dt))

            raw_img_vel_x = float(np.clip((curr_bcx - prev_bcx) / dt_safe, -float(getattr(self.cfg, "bottom_velocity_clip_per_s", 4.0)), float(getattr(self.cfg, "bottom_velocity_clip_per_s", 4.0))))
            raw_img_vel_y = float(np.clip((curr_bcy - prev_bcy) / dt_safe, -float(getattr(self.cfg, "bottom_velocity_clip_per_s", 4.0)), float(getattr(self.cfg, "bottom_velocity_clip_per_s", 4.0))))

            alpha = float(np.clip(getattr(self.cfg, "bottom_velocity_ema_alpha", 0.35), 0.0, 1.0))
            self._bottom_img_vel_x = float(alpha * raw_img_vel_x + (1.0 - alpha) * float(getattr(self, "_bottom_img_vel_x", 0.0)))
            self._bottom_img_vel_y = float(alpha * raw_img_vel_y + (1.0 - alpha) * float(getattr(self, "_bottom_img_vel_y", 0.0)))
            self._prev_bottom_img_speed = float(prev_vx_img_speed)

            self._bottom_prev_err_x_for_velocity = curr_bcx
            self._bottom_prev_err_y_for_velocity = curr_bcy

            bottom_img_speed = float(np.sqrt(self._bottom_img_vel_x ** 2 + self._bottom_img_vel_y ** 2))
            center_error_for_ready = float(getattr(self, "_bottom_center_error", 0.0))

            self._bottom_velocity_ready = bool(
                abs(self._bottom_img_vel_x) <= float(getattr(self.cfg, "bottom_velocity_ready_abs_vx", 0.18))
                and abs(self._bottom_img_vel_y) <= float(getattr(self.cfg, "bottom_velocity_ready_abs_vy", 0.18))
                and center_error_for_ready <= float(getattr(self.cfg, "bottom_velocity_ready_center_error", 0.22))
            )
            if self._bottom_velocity_ready:
                self._bottom_velocity_ready_streak += 1
            else:
                self._bottom_velocity_ready_streak = 0
        else:
            self._bottom_img_vel_x = 0.0
            self._bottom_img_vel_y = 0.0
            self._prev_bottom_img_speed = 0.0
            self._bottom_velocity_ready = False
            self._bottom_velocity_ready_streak = 0

        # Fusion-aware episode statistics and focus streak.
        # In BOTTOM_PRIMARY, bottom MATCH is the valid target lock. The front
        # camera is expected to lose the target after the handoff, so it must not
        # drive NONE% or max_focus.
        active_mode_for_stats = str(tracking_mode or getattr(self, "_fusion_tracking_mode", "LOST") or "LOST").upper()
        if bool(getattr(self, "_bottom_match", False)) and str(getattr(self, "_active_camera", "front")).lower() == "bottom":
            active_mode_for_stats = "MATCH"

        if active_mode_for_stats == "MATCH":
            self._ep_match += 1
        elif active_mode_for_stats == "PRED":
            self._ep_pred += 1
        else:
            self._ep_none += 1

        if active_mode_for_stats in {"MATCH", "PRED"}:
            self._focus_streak += dt
        else:
            self._focus_streak = 0.0

        self._ep_max_focus = max(self._ep_max_focus, self._focus_streak)
        self._global_max_focus = max(self._global_max_focus, self._focus_streak)

        approach_min_improvement_m = float(getattr(self.cfg, "approach_min_improvement_m", 0.20))
        approach_goal_distance_m = float(getattr(self.cfg, "approach_goal_distance_m", 2.50))
        approach_warmup_sec = float(getattr(self.cfg, "approach_warmup_sec", 6.0))

        self._last_approach_improvement_m = 0.0
        self._step_chase_distance_improvement_m = 0.0
        if chase_rules_enabled:
            self._chase_rule_elapsed_s = float(getattr(self, "_chase_rule_elapsed_s", 0.0)) + float(dt)
        else:
            self._chase_rule_elapsed_s = 0.0
        chase_rule_active = bool(chase_rules_enabled and self._chase_rule_elapsed_s >= approach_warmup_sec)

        if current_has_target_for_rules and current_chase_distance_m is not None:
            previous_real_distance_m = float(getattr(self, "_current_chase_distance_m", float("inf")))
            self._prev_chase_distance_m = previous_real_distance_m
            self._current_chase_distance_m = float(current_chase_distance_m)

            if np.isfinite(previous_real_distance_m):
                self._step_chase_distance_improvement_m = float(previous_real_distance_m) - float(current_chase_distance_m)

            if not np.isfinite(float(getattr(self, "_initial_chase_distance_m", float("inf")))):
                self._initial_chase_distance_m = float(current_chase_distance_m)

            if not np.isfinite(float(self._best_chase_distance_m)):
                self._best_chase_distance_m = float(current_chase_distance_m)
                self._not_approaching_time_s = 0.0

            # During the warmup window, let the drone stabilize, auto-lock the
            # target, and begin moving. We still remember the best distance,
            # but we do NOT accumulate no-improvement pressure yet.
            elif not chase_rule_active:
                if float(current_chase_distance_m) < float(self._best_chase_distance_m):
                    self._last_approach_improvement_m = float(self._best_chase_distance_m) - float(current_chase_distance_m)
                    self._best_chase_distance_m = float(current_chase_distance_m)
                self._not_approaching_time_s = 0.0

            # Once the drone is already close enough, do not keep forcing further
            # chase improvement. At that point landing/centering should take over.
            elif float(current_chase_distance_m) <= approach_goal_distance_m:
                self._best_chase_distance_m = min(float(self._best_chase_distance_m), float(current_chase_distance_m))
                self._not_approaching_time_s = 0.0

            # Improvement means beating the best physical distance by at least a
            # small meter-level margin. This makes the rule robust to simulator
            # jitter and tiny car/drone pose noise.
            elif float(current_chase_distance_m) < float(self._best_chase_distance_m) - approach_min_improvement_m:
                self._last_approach_improvement_m = float(self._best_chase_distance_m) - float(current_chase_distance_m)
                self._best_chase_distance_m = float(current_chase_distance_m)
                self._not_approaching_time_s = 0.0
            else:
                # No-improvement is now a soft pressure signal only.
                # It should not reset the episode by itself.
                self._not_approaching_time_s += float(dt)

            # Damage-scale termination monitor:
            # terminate only if the drone moved away by a large percentage of
            # the original episode distance and stayed there for a short window.
            initial_dist_m = float(getattr(self, "_initial_chase_distance_m", float("inf")))
            damage_ratio_threshold = float(getattr(self.cfg, "distance_damage_termination_ratio", 0.20))
            if chase_rule_active and np.isfinite(initial_dist_m) and initial_dist_m > 1e-6:
                self._distance_damage_ratio = max(
                    0.0,
                    (float(current_chase_distance_m) - initial_dist_m) / initial_dist_m,
                )
                if self._distance_damage_ratio >= damage_ratio_threshold:
                    self._distance_damage_time_s += float(dt)
                else:
                    # Decay the damage timer instead of instantly clearing it.
                    # This prevents one lucky frame from hiding repeated bad drift.
                    self._distance_damage_time_s = max(0.0, float(self._distance_damage_time_s) - float(dt))
            else:
                self._distance_damage_ratio = 0.0
                self._distance_damage_time_s = 0.0
        else:
            # Do not use fake distance progress while the target is missing.
            # During warmup we do not punish this with the approach rule yet;
            # target_lost / non_match rules still handle true tracking failure.
            self._current_chase_distance_m = float("inf")
            self._distance_damage_ratio = 0.0
            if chase_rule_active:
                self._not_approaching_time_s += float(dt)
            else:
                self._not_approaching_time_s = 0.0
                self._distance_damage_time_s = 0.0

        current_yaw_rad = self._get_current_yaw_rad()
        if current_yaw_rad is not None and self._initial_yaw_rad is not None:
            self._last_yaw_delta_deg = abs(
                self._deg(self._wrap_angle_rad(current_yaw_rad - float(self._initial_yaw_rad)))
            )
        else:
            self._last_yaw_delta_deg = 0.0

        # Effective yaw deviation uses the larger of:
        #   1. actual pose yaw delta reported by AirSim
        #   2. integrated commanded yaw delta
        # This prevents a 90-degree visible turn from being logged as YawD=0.0.
        self._effective_yaw_delta_deg = max(
            float(getattr(self, "_last_yaw_delta_deg", 0.0)),
            abs(float(getattr(self, "_commanded_yaw_delta_deg", 0.0))),
        )

        self._last_distance_proxy_norm = current_distance_proxy

        collision_now = False
        collision_raw = False
        collision_object_name = ""
        collision_penetration_depth = 0.0

        if self.cfg.use_collision_termination:
            try:
                col = self.client.simGetCollisionInfo(vehicle_name=self.cfg.vehicle_name)

                collision_raw = bool(getattr(col, "has_collided", False))
                collision_object_name = str(getattr(col, "object_name", "") or "")
                collision_penetration_depth = float(getattr(col, "penetration_depth", 0.0) or 0.0)

                # In small custom training levels AirSim can briefly report a startup/floor
                # collision-like state around spawn/reset even when the drone is visibly
                # airborne. This should not end the episode at step 0.
                #
                # Use an existing config field if present, otherwise default to 20 steps.
                ignore_collision_steps = int(
                    getattr(
                        self.cfg,
                        "ignore_collision_termination_first_steps",
                        getattr(self.cfg, "ignore_obstacle_termination_first_steps", 20),
                    )
                )

                object_lower = collision_object_name.lower()
                is_floor_startup_noise = (
                    self.step_in_episode <= ignore_collision_steps
                    and ("floor" in object_lower or "ground" in object_lower or "plane" in object_lower)
                    and collision_penetration_depth <= 0.10
                )

                collision_now = bool(collision_raw and not is_floor_startup_noise)

                if collision_raw and is_floor_startup_noise and getattr(self.cfg, "print_obstacle_debug", False):
                    print(
                        "[COLLISION IGNORED] startup/floor noise "
                        f"step={self.step_in_episode} "
                        f"object={collision_object_name} "
                        f"depth={collision_penetration_depth:.3f}"
                    )

            except Exception:
                collision_now = False
                collision_raw = False

        done = False
        term_reason = ""

        if collision_now:
            done = True
            term_reason = "collision"

        horizontal_min_obst = float(obstacle_dict["min_obstacle_dist_m"])
        safety_phase_now = str(safety_info.get("safety_phase", "CHASE")).upper()
        takeoff_phase_now = bool(safety_phase_now == "TAKEOFF")
        horizontal_emergency_active = bool(not takeoff_phase_now)

        # TAKEOFF is controlled by the safety layer, not by the chase policy.
        # Therefore, chase-specific failures are gated out during TAKEOFF.
        # If TAKEOFF never clears, end the episode with a dedicated reason
        # instead of blaming tracking/non-match behavior.
        if (
            not done
            and takeoff_phase_now
            and self.step_in_episode > int(getattr(self.cfg, "safety_takeoff_max_steps", 120))
        ):
            done = True
            term_reason = "takeoff_failed_not_airborne"

        if (
            not done
            and horizontal_emergency_active
            and self.step_in_episode > int(self.cfg.ignore_obstacle_termination_first_steps)
            and horizontal_min_obst < float(self.cfg.obstacle_emergency_termination_dist_m)
        ):
            done = True
            term_reason = "emergency_horizontal_obstacle_distance"

        # Yaw failure has priority over handoff_success.
        # Otherwise the agent can rotate badly, still trigger a bottom-camera
        # match, and receive a large positive handoff_success return.
        yaw_limit_deg = float(getattr(self.cfg, "max_initial_yaw_delta_deg", 15.0))
        yaw_limit_enabled = bool(getattr(self.cfg, "handoff_success_respects_yaw_limit", True))

        commanded_yaw_limit_deg = float(getattr(self.cfg, "max_commanded_yaw_delta_deg", yaw_limit_deg))
        cumulative_yaw_limit_deg = float(getattr(self.cfg, "max_cumulative_abs_yaw_cmd_deg", 45.0))
        use_command_yaw = bool(getattr(self.cfg, "yaw_deviation_use_command_integral", True))

        effective_yaw_delta_deg = float(getattr(self, "_last_yaw_delta_deg", 0.0))
        if use_command_yaw:
            effective_yaw_delta_deg = max(
                effective_yaw_delta_deg,
                abs(float(getattr(self, "_commanded_yaw_delta_deg", 0.0))),
            )
        self._effective_yaw_delta_deg = float(effective_yaw_delta_deg)

        yaw_deviation_failed = bool(
            effective_yaw_delta_deg > yaw_limit_deg
            or (
                use_command_yaw
                and abs(float(getattr(self, "_commanded_yaw_delta_deg", 0.0))) > commanded_yaw_limit_deg
            )
            or (
                use_command_yaw
                and float(getattr(self, "_cumulative_abs_yaw_cmd_deg", 0.0)) > cumulative_yaw_limit_deg
            )
        )

        if (
            not done
            and chase_rules_enabled
            and yaw_limit_enabled
            and yaw_deviation_failed
        ):
            done = True
            term_reason = "yaw_deviation_too_large"

        if (
            not done
            and chase_rules_enabled
            and bool(getattr(self, "_handoff_ready", False))
        ):
            done = True
            term_reason = "handoff_success"

        # Target lost is a fusion-level event, not a single-camera event.
        # If bottom has authority or a recent valid match, front-camera loss is
        # expected and must not trigger panic recovery.
        front_loss_allowed_by_bottom = bool(
            bool(getattr(self, "_fusion_has_target", False))
            or bool(getattr(self, "_bottom_match", False))
            or bool(getattr(self, "_handoff_visual_scan_trigger", False))
            or str(getattr(self, "_handoff_phase", "CHASE_FRONT")).upper() in {"BOTTOM_SCAN", "HANDOFF_OVERLAP", "HANDOFF_READY"}
        )

        if (
            not done
            and (not takeoff_phase_now)
            and (not front_loss_allowed_by_bottom)
            and float(obs_dict["lost_target_time_norm"]) >= 1.0
        ):
            done = True
            term_reason = "target_lost_too_long"

        if (
            not done
            and chase_rules_enabled
            and self._non_match_time_s >= float(getattr(self.cfg, "non_match_timeout_sec", 3.0))
        ):
            done = True
            term_reason = "non_match_too_long"

        if (
            not done
            and chase_rules_enabled
            and bool(getattr(self.cfg, "terminate_on_no_best_distance_improvement", False))
            and self._not_approaching_time_s >= float(getattr(self.cfg, "not_approaching_timeout_sec", 3.0))
        ):
            done = True
            term_reason = "best_distance_not_improved_too_long"

        if (
            not done
            and chase_rules_enabled
            and self._distance_damage_time_s >= float(getattr(self.cfg, "distance_damage_timeout_sec", 2.0))
        ):
            done = True
            term_reason = "distance_damage_too_large"

        # Yaw deviation is checked before handoff_success above, so it cannot
        # be masked by a successful secondary-camera detection.

        # ------------------------------------------------------------------
        # Agent-1 success terminal: stable bottom alignment.
        # ------------------------------------------------------------------
        # A bottom-ready handoff is the objective of this training task. Once
        # the downward camera has confirmed the selected target and the target
        # is centered for enough frames, end the episode as success instead of
        # letting it run until timeout and collect unrelated penalties.
        if (
            not done
            and bool(getattr(self.cfg, "alignment_ready_terminal_enabled", True))
            and (not takeoff_phase_now)
            and bool(getattr(self, "_handoff_ready", False))
            and bool(getattr(self, "_bottom_confirmed", False))
            and bool(getattr(self, "_bottom_center_ok", False))
            and bool(getattr(self, "_bottom_yaw_ok", True))
            and int(getattr(self, "_bottom_match_streak", 0)) >= int(getattr(self.cfg, "alignment_ready_min_bottom_streak", 45))
            and (
                (not bool(getattr(self.cfg, "alignment_ready_requires_bottom_velocity", True)))
                or int(getattr(self, "_bottom_velocity_ready_streak", 0)) >= int(getattr(self.cfg, "bottom_velocity_ready_streak_required", 12))
            )
            and (
                (not bool(getattr(self.cfg, "handoff_success_requires_bbox_safe", True)))
                or float(getattr(self, "_bottom_bbox_rel_err", 999.0)) <= float(getattr(self.cfg, "handoff_success_max_bbox_rel_err", 0.50))
            )
        ):
            done = True
            term_reason = "handoff_success"

        if (
            not done
            and (not takeoff_phase_now)
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

        current_has_target = bool(float(obs_dict.get("has_target", 0.0)) > 0.5)
        current_tracking_mode = str(tracking_mode or "LOST")
        previous_tracking_mode = str(getattr(self, "_prev_tracking_mode", "LOST") or "LOST")
        previous_had_target = bool(getattr(self, "_prev_had_target", False))
        previous_lost_time_norm = float(getattr(self, "_prev_lost_target_time_norm", 0.0))
        previous_distance_proxy = float(getattr(self, "_prev_distance_proxy_norm", 1.0))

        likely_overshoot = bool(
            previous_had_target
            and not current_has_target
            and previous_distance_proxy < float(getattr(self.cfg, "overshoot_distance_proxy_threshold", 0.90))
        )

        env_reward_info = {
            "altitude_m": float(drone_state.altitude_m),
            "collision_detected": bool(collision_now),
            "safety_intervention": bool(safety_info.get("safety_intervention", False)),
            "termination_reason": term_reason,
            "non_match_time_s": float(getattr(self, "_non_match_time_s", 0.0)),
            "not_approaching_time_s": float(getattr(self, "_not_approaching_time_s", 0.0)),
            "current_chase_distance_m": float(getattr(self, "_current_chase_distance_m", float("inf"))),
            "previous_chase_distance_m": float(getattr(self, "_prev_chase_distance_m", float("inf"))),
            "initial_chase_distance_m": float(getattr(self, "_initial_chase_distance_m", float("inf"))),
            "best_chase_distance_m": float(getattr(self, "_best_chase_distance_m", float("inf"))),
            "distance_damage_time_s": float(getattr(self, "_distance_damage_time_s", 0.0)),
            "distance_damage_ratio": float(getattr(self, "_distance_damage_ratio", 0.0)),
            "front_full_tracker_mode": str(getattr(self, "_front_full_tracker_mode", "LOST")),
            "front_full_tracker_raw_mode": str(getattr(self, "_front_full_tracker_raw_mode", "NONE")),
            "front_full_tracker_updates": int(getattr(self, "_front_full_tracker_updates", 0)),
            "front_full_tracker_match_frames": int(getattr(self, "_front_full_tracker_match_frames", 0)),
            "front_full_tracker_pred_frames": int(getattr(self, "_front_full_tracker_pred_frames", 0)),
            "front_full_tracker_lost_frames": int(getattr(self, "_front_full_tracker_lost_frames", 0)),
            "bottom_match": bool(getattr(self, "_bottom_match", False)),
            "bottom_confirmed": bool(getattr(self, "_bottom_confirmed", False)),
            "bottom_full_tracker_mode": str(getattr(self, "_bottom_full_tracker_mode", "LOST")),
            "bottom_full_tracker_raw_mode": str(getattr(self, "_bottom_full_tracker_raw_mode", "NONE")),
            "bottom_full_tracker_updates": int(getattr(self, "_bottom_full_tracker_updates", 0)),
            "bottom_full_tracker_match_frames": int(getattr(self, "_bottom_full_tracker_match_frames", 0)),
            "bottom_full_tracker_pred_frames": int(getattr(self, "_bottom_full_tracker_pred_frames", 0)),
            "bottom_full_tracker_lost_frames": int(getattr(self, "_bottom_full_tracker_lost_frames", 0)),
            "bottom_full_tracker_confidence": float(getattr(self, "_bottom_full_tracker_confidence", 0.0)),
            "bottom_full_tracker_similarity": float(getattr(self, "_bottom_full_tracker_similarity", 0.0)),
            "bottom_match_fresh": bool(getattr(self, "_bottom_match_fresh", False)),
            "bottom_bbox_area_norm": float(getattr(self, "_bottom_bbox_area_norm", 0.0)),
            "bottom_err_x": float(getattr(self, "_bottom_err_x", 0.0)),
            "bottom_err_y": float(getattr(self, "_bottom_err_y", 0.0)),
            "prev_bottom_abs_err_y": float(getattr(self, "_prev_bottom_abs_err_y", abs(float(getattr(self, "_bottom_err_y", 0.0))))),
            "bottom_center_error": float(getattr(self, "_bottom_center_error", 0.0)),
            "bottom_center_ok": bool(getattr(self, "_bottom_center_ok", False)),
            "bottom_yaw_ok": bool(getattr(self, "_bottom_yaw_ok", False)),
            "handoff_candidate_gate": bool(getattr(self, "_handoff_candidate_gate", False)),
            "bottom_similarity": float(getattr(self, "_bottom_similarity", 0.0)),
            "bottom_match_streak": int(getattr(self, "_bottom_match_streak", 0)),
            "front_weight": float(getattr(self, "_front_weight", 1.0)),
            "bottom_weight": float(getattr(self, "_bottom_weight", 0.0)),
            "speed_stage": str(getattr(self, "_speed_stage", "CHASE_FAST")),
            "bottom_img_vel_x": float(getattr(self, "_bottom_img_vel_x", 0.0)),
            "bottom_img_vel_y": float(getattr(self, "_bottom_img_vel_y", 0.0)),
            "prev_bottom_img_speed": float(getattr(self, "_prev_bottom_img_speed", 0.0)),
            "bottom_velocity_ready": bool(getattr(self, "_bottom_velocity_ready", False)),
            "bottom_velocity_ready_streak": int(getattr(self, "_bottom_velocity_ready_streak", 0)),
            "bottom_bbox_rel_half_w": float(getattr(self, "_bottom_bbox_rel_half_w", 0.0)),
            "bottom_bbox_rel_half_h": float(getattr(self, "_bottom_bbox_rel_half_h", 0.0)),
            "bottom_bbox_rel_err_x": float(getattr(self, "_bottom_bbox_rel_err_x", 0.0)),
            "bottom_bbox_rel_err_y": float(getattr(self, "_bottom_bbox_rel_err_y", 0.0)),
            "bottom_bbox_rel_err": float(getattr(self, "_bottom_bbox_rel_err", 999.0)),
            "handoff_ready": bool(getattr(self, "_handoff_ready", False)),
            "handoff_phase": str(getattr(self, "_handoff_phase", "CHASE_FRONT")),
            "camera_authority": str(getattr(self, "_camera_authority", "FRONT_PRIMARY")),
            "active_camera": str(getattr(self, "_active_camera", "front")),
            "speed_stage": str(getattr(self, "_speed_stage", "CHASE_FAST")),
            "bottom_img_vel_x": float(getattr(self, "_bottom_img_vel_x", 0.0)),
            "bottom_img_vel_y": float(getattr(self, "_bottom_img_vel_y", 0.0)),
            "bottom_velocity_ready": bool(getattr(self, "_bottom_velocity_ready", False)),
            "bottom_velocity_ready_streak": int(getattr(self, "_bottom_velocity_ready_streak", 0)),
            "last_bottom_pd_assist_vx": float(getattr(self, "_last_bottom_pd_assist_vx", 0.0)),
            "last_bottom_pd_assist_vy": float(getattr(self, "_last_bottom_pd_assist_vy", 0.0)),
            "last_fast_chase_assist_vx": float(getattr(self, "_last_fast_chase_assist_vx", 0.0)),
            "last_stage_vx_scale": float(getattr(self, "_last_stage_vx_scale", float(getattr(self.cfg, "vx_scale", 0.0)))),
            "last_stage_vy_scale": float(getattr(self, "_last_stage_vy_scale", float(getattr(self.cfg, "vy_scale", 0.0)))),
            "front_camera_name": str(getattr(self.cfg, "front_camera_name", "front_center")),
            "downward_camera_name": str(getattr(self.cfg, "downward_camera_name", "bottom_center")),
            "front_full_tracker_mode": str(getattr(self, "_front_full_tracker_mode", "LOST")),
            "front_full_tracker_raw_mode": str(getattr(self, "_front_full_tracker_raw_mode", "NONE")),
            "front_full_tracker_updates": int(getattr(self, "_front_full_tracker_updates", 0)),
            "bottom_full_tracker_mode": str(getattr(self, "_bottom_full_tracker_mode", "LOST")),
            "bottom_full_tracker_raw_mode": str(getattr(self, "_bottom_full_tracker_raw_mode", "NONE")),
            "bottom_full_tracker_updates": int(getattr(self, "_bottom_full_tracker_updates", 0)),
            "bottom_full_tracker_match_frames": int(getattr(self, "_bottom_full_tracker_match_frames", 0)),
            "bottom_full_tracker_pred_frames": int(getattr(self, "_bottom_full_tracker_pred_frames", 0)),
            "bottom_full_tracker_lost_frames": int(getattr(self, "_bottom_full_tracker_lost_frames", 0)),
            "fusion_has_target": bool(getattr(self, "_fusion_has_target", False)),
            "fusion_err_x": float(getattr(self, "_fusion_err_x", 0.0)),
            "fusion_err_y": float(getattr(self, "_fusion_err_y", 0.0)),
            "fusion_confidence": float(getattr(self, "_fusion_confidence", 0.0)),
            "bottom_alignment_assist_events": int(getattr(self, "_bottom_alignment_assist_events", 0)),
            "last_bottom_alignment_assist_vx": float(getattr(self, "_last_bottom_alignment_assist_vx", 0.0)),
            "last_bottom_alignment_assist_vy": float(getattr(self, "_last_bottom_alignment_assist_vy", 0.0)),
            "handoff_visual_lidar_trigger": bool(getattr(self, "_handoff_visual_lidar_trigger", False)),
            "handoff_visual_scan_trigger": bool(getattr(self, "_handoff_visual_scan_trigger", False)),
            "handoff_visual_score": float(getattr(self, "_handoff_visual_score", 0.0)),
            "handoff_lidar_dist_m": float(getattr(self, "_handoff_lidar_dist_m", float("inf"))),
            "handoff_lidar_direction": str(getattr(self, "_handoff_lidar_direction", "none")),
            "last_approach_improvement_m": float(getattr(self, "_last_approach_improvement_m", 0.0)),
            "step_chase_distance_improvement_m": float(getattr(self, "_step_chase_distance_improvement_m", 0.0)),
            "chase_rule_elapsed_s": float(getattr(self, "_chase_rule_elapsed_s", 0.0)),
            "yaw_delta_from_initial_deg": float(getattr(self, "_last_yaw_delta_deg", 0.0)),
            "commanded_yaw_delta_deg": float(getattr(self, "_commanded_yaw_delta_deg", 0.0)),
            "cumulative_abs_yaw_cmd_deg": float(getattr(self, "_cumulative_abs_yaw_cmd_deg", 0.0)),
            "effective_yaw_delta_deg": float(getattr(self, "_effective_yaw_delta_deg", 0.0)),
            "last_yaw_rate_cmd_dps": float(getattr(self, "_last_yaw_rate_cmd_dps", 0.0)),
            "tracking_mode": current_tracking_mode,
            "previous_tracking_mode": previous_tracking_mode,
            "previous_had_target": previous_had_target,
            "previous_lost_target_time_norm": previous_lost_time_norm,
            "lost_to_match_transition": bool(previous_tracking_mode != "MATCH" and current_tracking_mode == "MATCH"),
            "likely_overshoot": likely_overshoot,
        }

        reward, reward_parts = compute_follow_reward(
            obs=obs_dict,
            action=safe_action,
            prev_action=self._prev_action,
            env_info=env_reward_info,
            config=self.reward_config,
        )

        # TAKEOFF is not part of the learned chase objective. During TAKEOFF,
        # the safety layer owns the vehicle and may clamp horizontal movement
        # and force upward velocity. Do not punish the policy with obstacle or
        # non-centering rewards while it is not allowed to control the task yet.
        if takeoff_phase_now and term_reason != "takeoff_failed_not_airborne":
            reward = 0.0
            reward_parts["takeoff_phase_neutral_reward"] = 1.0
            reward_parts["total_reward_after_takeoff_neutral"] = 0.0
        elif term_reason == "takeoff_failed_not_airborne":
            reward = -250.0
            reward_parts["takeoff_failed_penalty"] = -250.0
            reward_parts["total_reward_after_takeoff_failure"] = float(reward)

        if bool(getattr(self, "_agent1_yaw_lock_applied", False)) and not takeoff_phase_now:
            yaw_lock_penalty = (
                -float(getattr(self.cfg, "agent1_locked_yaw_request_penalty", 18.0))
                * abs(float(getattr(self, "_last_blocked_yaw_action", 0.0)))
            )
            reward += float(yaw_lock_penalty)
            self._last_yaw_lock_penalty = float(yaw_lock_penalty)
            reward_parts["agent1_yaw_lock_penalty"] = float(yaw_lock_penalty)
            reward_parts["agent1_blocked_yaw_action"] = float(getattr(self, "_last_blocked_yaw_action", 0.0))
            reward_parts["total_reward_after_agent1_yaw_lock_penalty"] = float(reward)

        # ------------------------------------------------------------------
        # Env-level hard credit reset for unnecessary yaw
        # ------------------------------------------------------------------
        # Important:
        # The reward function can compute very large negative yaw penalties, but
        # total reward is clipped for PPO stability. That means the policy may
        # still finish an episode with a high positive return even after a bad
        # yaw habit. This block makes the behavior unprofitable by cancelling
        # accumulated positive return and applying an additional hard penalty.
        #
        # This is not a generic yaw ban. It only activates when reward_parts
        # says the target was focused/centered in MATCH and yaw was unnecessary,
        # or when yaw caused/was associated with de-centering.
        # ------------------------------------------------------------------
        self._last_focused_yaw_hard_penalty = 0.0
        focused_yaw_event = bool(float(reward_parts.get("is_focused_centered_for_yaw", 0.0)) > 0.5)
        yaw_decenter_event = bool(float(reward_parts.get("is_yaw_decentering", 0.0)) > 0.5)

        bottom_authority_or_ready = bool(
            bool(getattr(self.cfg, "suppress_yaw_hard_after_bottom_authority", True))
            and (
                str(getattr(self, "_active_camera", "front")).lower() == "bottom"
            or str(getattr(self, "_camera_authority", "FRONT_PRIMARY")).upper() in {"BOTTOM_PRIMARY", "BOTTOM_RECOVERY"}
            or bool(getattr(self, "_handoff_ready", False))
            or bool(getattr(self, "_bottom_confirmed", False))
                or str(term_reason or "") == "handoff_success"
            )
        )

        if (
            bool(getattr(self.cfg, "hard_penalize_focused_centered_yaw", True))
            and not takeoff_phase_now
            and not bottom_authority_or_ready
            and term_reason not in {"collision", "takeoff_failed_not_airborne", "handoff_success"}
            and (focused_yaw_event or yaw_decenter_event)
        ):
            yaw_hard_penalty = 0.0
            if focused_yaw_event:
                yaw_hard_penalty += float(getattr(self.cfg, "focused_centered_yaw_hard_penalty", 2500.0))
                self._focused_yaw_hard_events += 1
            if yaw_decenter_event:
                yaw_hard_penalty += float(getattr(self.cfg, "yaw_decenter_hard_penalty", 3500.0))
                self._yaw_decenter_hard_events += 1

            reward, yaw_cancelled_positive_return = self._hard_fail_reward(reward, yaw_hard_penalty)

            reward_parts["focused_yaw_hard_penalty"] = float(-yaw_hard_penalty)
            reward_parts["focused_yaw_cancelled_positive_return"] = float(-yaw_cancelled_positive_return)
            reward_parts["focused_yaw_hard_event"] = float(1.0)
            reward_parts["total_reward_after_focused_yaw_hard_penalty"] = float(reward)
            self._last_focused_yaw_hard_penalty = float(-yaw_hard_penalty)

            if bool(getattr(self.cfg, "terminate_on_focused_centered_yaw_hard_fail", False)):
                done = True
                if term_reason in {None, "", "none"}:
                    term_reason = "focused_centered_yaw_hard_fail"

        if bottom_authority_or_ready and (focused_yaw_event or yaw_decenter_event):
            # Yaw-hard penalties are a front-chase anti-orbit mechanism.
            # Once bottom authority is active or the episode already reached
            # handoff_success, do not erase a successful alignment because the
            # front-centered yaw detector is no longer the relevant objective.
            reward_parts["focused_yaw_hard_suppressed_bottom"] = float(1.0)

        # ------------------------------------------------------------------
        # Pitch / attitude smoothness shaping
        # ------------------------------------------------------------------
        # Visual professionalism objective:
        # Penalize large forward/backward pitch and, more importantly, rapid
        # pitch changes. CHASE_FAST is allowed more pitch than bottom/landing
        # stages; BOTTOM_LANDING_READY is the strictest.
        self._last_pitch_smoothness_penalty = 0.0
        if bool(getattr(self.cfg, "pitch_smoothness_enabled", True)) and not takeoff_phase_now:
            pitch_rad_now = float(getattr(drone_state, "pitch_rad", 0.0))
            prev_pitch_rad = float(getattr(self, "_prev_pitch_rad_for_smooth", pitch_rad_now))
            dt_pitch = max(1e-3, float(dt))

            pitch_deg = float(np.degrees(pitch_rad_now))
            pitch_rate_dps = float(np.degrees((pitch_rad_now - prev_pitch_rad) / dt_pitch))

            stage_pitch = str(getattr(self, "_speed_stage", "CHASE_FAST"))
            if stage_pitch == "BOTTOM_LANDING_READY":
                pitch_ref_deg = float(getattr(self.cfg, "pitch_ref_landing_deg", 4.5))
                pitch_rate_ref_dps = float(getattr(self.cfg, "pitch_rate_ref_landing_dps", 16.0))
                w_pitch_abs = float(getattr(self.cfg, "w_pitch_abs_landing", 4.8))
                w_pitch_rate = float(getattr(self.cfg, "w_pitch_rate_landing", 5.0))
            elif stage_pitch == "BOTTOM_VELOCITY_MATCH":
                pitch_ref_deg = float(getattr(self.cfg, "pitch_ref_bottom_deg", 8.0))
                pitch_rate_ref_dps = float(getattr(self.cfg, "pitch_rate_ref_bottom_dps", 28.0))
                w_pitch_abs = float(getattr(self.cfg, "w_pitch_abs_bottom", 2.2))
                w_pitch_rate = float(getattr(self.cfg, "w_pitch_rate_bottom", 2.5))
            else:
                pitch_ref_deg = float(getattr(self.cfg, "pitch_ref_chase_deg", 14.0))
                pitch_rate_ref_dps = float(getattr(self.cfg, "pitch_rate_ref_chase_dps", 45.0))
                w_pitch_abs = float(getattr(self.cfg, "w_pitch_abs_chase", 0.8))
                w_pitch_rate = float(getattr(self.cfg, "w_pitch_rate_chase", 0.5))

            pitch_abs_norm = abs(pitch_deg) / max(1e-6, pitch_ref_deg)
            pitch_rate_norm = abs(pitch_rate_dps) / max(1e-6, pitch_rate_ref_dps)

            pitch_abs_penalty = -w_pitch_abs * min(4.0, pitch_abs_norm ** 1.35)
            pitch_rate_penalty = -w_pitch_rate * min(4.0, pitch_rate_norm ** 1.25)
            pitch_smoothness_penalty = float(pitch_abs_penalty + pitch_rate_penalty)

            reward += pitch_smoothness_penalty
            self._last_pitch_smoothness_penalty = pitch_smoothness_penalty
            self._last_pitch_deg = float(pitch_deg)
            self._last_pitch_rate_dps = float(pitch_rate_dps)
            self._prev_pitch_rad_for_smooth = float(pitch_rad_now)

            reward_parts["pitch_abs_penalty"] = float(pitch_abs_penalty)
            reward_parts["pitch_rate_penalty"] = float(pitch_rate_penalty)
            reward_parts["pitch_smoothness_penalty"] = float(pitch_smoothness_penalty)
            reward_parts["pitch_deg"] = float(pitch_deg)
            reward_parts["pitch_rate_dps"] = float(pitch_rate_dps)
            reward_parts["pitch_stage_ref_deg"] = float(pitch_ref_deg)
            reward_parts["pitch_stage_rate_ref_dps"] = float(pitch_rate_ref_dps)

        # ------------------------------------------------------------------
        # Hard failure rule for target_lost_too_long
        # ------------------------------------------------------------------
        # Tracking-agent rule:
        #   Losing the target for too long must NEVER be profitable.
        #
        # Problem observed:
        #   The agent can accumulate positive reward from good tracking, then
        #   lose the target and still finish the episode with a positive return.
        #   That teaches the wrong behavior: "tracking well for a while and then
        #   losing target is acceptable."
        #
        # Desired behavior:
        #   If target_lost_too_long happens:
        #       - if accumulated episode return is positive, cancel it to zero
        #       - then apply a large hard-fail penalty
        #       - if already negative, still apply the large penalty
        #
        # This is implemented in the Env because only the Env knows the running
        # episode return. The reward function only knows the current step.
        # ------------------------------------------------------------------
        hard_fail_reasons = {
            "target_lost_too_long",
            "non_match_too_long",
            "not_approaching_too_long",
            "best_distance_not_improved_too_long",
            "yaw_deviation_too_large",
        }

        if term_reason in hard_fail_reasons:
            hard_fail_penalty = float(getattr(self.cfg, "target_lost_hard_fail_penalty", 12000.0))
            if term_reason == "best_distance_not_improved_too_long":
                hard_fail_penalty += (
                    float(getattr(self.cfg, "not_approaching_penalty_growth_per_sec", 1000.0))
                    * float(getattr(self, "_not_approaching_time_s", 0.0))
                )
            elif term_reason == "yaw_deviation_too_large":
                # Rotating far away from the chase heading is a mission failure,
                # even if a weak/late bottom-camera match happened.
                hard_fail_penalty = max(
                    hard_fail_penalty,
                    float(getattr(self.cfg, "yaw_deviation_hard_fail_penalty", 18000.0)),
                )
            reward, hard_fail_cancelled_positive_return = self._hard_fail_reward(reward, hard_fail_penalty)

            reward_parts["strict_hard_fail_penalty"] = float(-hard_fail_penalty)
            reward_parts["strict_hard_fail_cancelled_positive_return"] = float(-hard_fail_cancelled_positive_return)
            reward_parts["strict_hard_fail_reason"] = str(term_reason)
            reward_parts["total_reward_after_hard_fail"] = float(reward)

        self._accumulate_reward_parts(reward_parts)

        self._ep_return += float(reward)
        self._prev_action[:] = safe_action

        # Save current tracking state for the next step. The next action can then
        # receive recovery yaw authority, and the next reward can detect
        # LOST/PRED -> MATCH reacquisition.
        self._prev_tracking_mode = current_tracking_mode
        self._prev_had_target = bool(current_has_target or bool(getattr(self, "_fusion_has_target", False)))
        self._prev_lost_target_time_norm = 0.0 if bool(getattr(self, "_fusion_has_target", False)) else float(obs_dict.get("lost_target_time_norm", 0.0))
        self._prev_distance_proxy_norm = float(obs_dict.get("distance_proxy_norm", 1.0))
        self._prev_err_x = float(obs_dict.get("err_x", getattr(self, "_fusion_err_x", 0.0)))
        self._prev_err_y = float(obs_dict.get("err_y", getattr(self, "_fusion_err_y", 0.0)))
        self._prev_bbox_conf = float(obs_dict.get("bbox_conf", getattr(self, "_fusion_confidence", 0.0)))

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
                f"PHASE={str(safety_info.get('safety_phase', 'CHASE'))} "
                f"SRC={obstacle_dict.get('obstacle_source', 'unknown')} "
                f"PTS={int(obstacle_dict.get('lidar_point_count', 0))} "
                f"ALT={drone_state.altitude_m:.2f}"
            )

        cv_render_every = max(1, int(getattr(self.cfg, "cv_debug_render_every_n_steps", 1)))
        if self.cfg.show_cv_window and (self.step_in_episode % cv_render_every == 0):
            self._draw_tracking_overlay(frame, bbox_raw, tracking_result)

            safety_active = bool(safety_info.get("safety_intervention", False))
            safety_reasons = safety_info.get("safety_reasons", [])
            safety_reason_text = ",".join(safety_reasons[:2]) if safety_reasons else "none"

            # Safety state label:
            # CLEAR       = no safety correction was applied
            # ACTIVE      = safety layer modified the action
            # TAKEOFF     = near-ground phase; down proximity is expected and forces climb
            # LANDING     = touchdown phase; horizontal safety active, down proximity allowed
            # EMERGENCY   = very close horizontal obstacle in a non-takeoff phase
            safety_phase_text = str(safety_info.get("safety_phase", "CHASE")).upper()
            if safety_phase_text == "TAKEOFF":
                safety_state = "TAKEOFF"
                safety_color = (0, 200, 255)
            elif safety_phase_text in {"LANDING", "DESCENT", "LANDING_DESCENT", "ALIGN_ABOVE_TARGET"}:
                if horizontal_min_obst < float(self.cfg.obstacle_emergency_dist_m):
                    safety_state = "EMERGENCY"
                    safety_color = (0, 0, 255)
                else:
                    safety_state = "LANDING" if safety_active else "LANDING CLEAR"
                    safety_color = (0, 200, 255) if safety_active else (0, 255, 0)
            elif horizontal_min_obst < float(self.cfg.obstacle_emergency_dist_m):
                safety_state = "EMERGENCY"
                safety_color = (0, 0, 255)
            elif safety_active:
                safety_state = "ACTIVE"
                safety_color = (0, 200, 255)
            else:
                safety_state = "CLEAR"
                safety_color = (0, 255, 0)

            cv2.putText(frame, f"MODE: {tracking_mode} RECOVERY:{int(recovery_mode_active)}", (20, 90),
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
            cv2.putText(frame, f"PHASE: {safety_phase_now} | chase_rules={int(chase_rules_enabled)}", (20, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, safety_color, 2)
            cv2.putText(
                frame,
                f"HANDOFF: {str(getattr(self, '_handoff_phase', 'CHASE_FRONT'))} "
                f"Gate={int(bool(getattr(self, '_handoff_candidate_gate', False)))} "
                f"Fresh={int(bool(getattr(self, '_bottom_match_fresh', False)))} "
                f"Bsafe={int(float(getattr(self, '_bottom_bbox_rel_err', 999.0)) <= float(getattr(self.cfg, 'handoff_success_max_bbox_rel_err', 0.50)))} "
                f"Barea={float(getattr(self, '_bottom_bbox_area_norm', 0.0)):.3f} "
                f"Bcx={float(getattr(self, '_bottom_err_x', 0.0)):+.2f} "
                f"Bcy={float(getattr(self, '_bottom_err_y', 0.0)):+.2f} "
                f"BCok={int(bool(getattr(self, '_bottom_center_ok', False)))} "
                f"BYok={int(bool(getattr(self, '_bottom_yaw_ok', False)))} "
                f"BM={int(bool(getattr(self, '_bottom_match', False)))} "
                f"BC={int(bool(getattr(self, '_bottom_confirmed', False)))} "
                f"Bsim={float(getattr(self, '_bottom_similarity', 0.0)):.2f} "
                f"VS={int(bool(getattr(self, '_handoff_visual_scan_trigger', False)))} "
                f"VL={int(bool(getattr(self, '_handoff_visual_lidar_trigger', False)))}",
                (20, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (0, 255, 255),
                2,
            )

            display_frame = self._select_cv_debug_frame(frame, safety_phase_now)
            cv2.imshow("Tracker Debug", display_frame)
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
                f"R_chaseP={reward_parts.get('visible_far_no_approach_penalty', 0.0):+.2f} "
                f"R_fwd={reward_parts.get('centered_forward_chase_bonus', 0.0):+.2f} "
                f"R_FAST={reward_parts.get('fast_chase_throttle_bonus', 0.0) + reward_parts.get('fast_chase_progress_bonus', 0.0) + reward_parts.get('fast_chase_slow_penalty', 0.0):+.2f} "
                f"R_Bvel={reward_parts.get('bottom_velocity_error_penalty', 0.0) + reward_parts.get('bottom_velocity_progress_bonus', 0.0) + reward_parts.get('bottom_velocity_ready_bonus', 0.0):+.2f} "
                f"R_Fpos={reward_parts.get('fine_position_bonus', 0.0) + reward_parts.get('fine_position_penalty', 0.0) + reward_parts.get('fine_position_ready_bonus', 0.0) + reward_parts.get('fine_position_landing_ready_bonus', 0.0):+.2f} "
                f"FposErr={reward_parts.get('fine_position_error_inf', 0.0):.3f} "
                f"FboxErr={reward_parts.get('fine_position_bbox_rel_err', 999.0):.2f} "
                f"FboxX={reward_parts.get('fine_position_bbox_rel_err_x', 0.0):.2f} "
                f"FboxY={reward_parts.get('fine_position_bbox_rel_err_y', 0.0):.2f} "
                f"R_idleFar={reward_parts.get('centered_idle_far_penalty', 0.0):+.2f} "
                f"R_Bcy={reward_parts.get('bottom_y_error_penalty', 0.0):+.2f} "
                f"R_Bprog={reward_parts.get('bottom_y_progress_reward', 0.0):+.2f} "
                f"R_Bvx={reward_parts.get('bottom_correct_vx_bonus', 0.0) + reward_parts.get('bottom_wrong_vx_penalty', 0.0):+.2f} "
                f"R_Bready={reward_parts.get('bottom_center_ready_bonus', 0.0):+.2f} "
                f"Pitch={float(getattr(self, '_last_pitch_deg', 0.0)):+.1f} "
                f"PitchRate={float(getattr(self, '_last_pitch_rate_dps', 0.0)):+.1f} "
                f"R_pitch={float(getattr(self, '_last_pitch_smoothness_penalty', 0.0)):+.2f} "
                f"VxSlew={int(bool(getattr(self, '_last_vx_slew_limited', False)))} "
                f"dVxSlew={float(getattr(self, '_last_vx_slew_delta', 0.0)):+.2f} "
                f"R_yawFocus={reward_parts.get('focused_centered_yaw_penalty', 0.0):+.2f} "
                f"R_yawExp={reward_parts.get('exp_focused_centered_yaw_penalty', 0.0):+.2f} "
                f"R_decenter={reward_parts.get('yaw_decenter_exp_penalty', 0.0):+.2f} "
                f"R_yawHard={reward_parts.get('focused_yaw_hard_penalty', 0.0):+.2f} "
                f"YawD={float(getattr(self, '_last_yaw_delta_deg', 0.0)):.1f} "
                f"CmdYawD={float(getattr(self, '_commanded_yaw_delta_deg', 0.0)):.1f} "
                f"AbsYawCmd={float(getattr(self, '_cumulative_abs_yaw_cmd_deg', 0.0)):.1f} "
                f"EffYawD={float(getattr(self, '_effective_yaw_delta_deg', 0.0)):.1f} "
                f"YShield={int(getattr(self, '_yaw_shield_events', 0))} "
                f"YStrafe={int(getattr(self, '_yaw_to_strafe_events', 0))} "
                f"YawLimit={int(getattr(self, '_agent1_yaw_limit_events', 0))} "
                f"YawMax={float(getattr(self, '_last_yaw_allowed_abs', 1.0)):.2f} "
                f"BlockYaw={float(getattr(self, '_last_blocked_yaw_action', 0.0)):+.2f} "
                f"R_yawLimit={float(getattr(self, '_last_yaw_lock_penalty', 0.0)):+.2f} "
                f"dVY={float(getattr(self, '_last_yaw_to_strafe_delta_vy', 0.0)):+.2f} "
                f"R_yawBudget={reward_parts.get('yaw_command_budget_penalty', 0.0):+.2f} "
                f"FYaw={int(float(reward_parts.get('is_focused_centered_for_yaw', 0.0)) > 0.5)} "
                f"DYaw={int(float(reward_parts.get('is_yaw_decentering', 0.0)) > 0.5)} "
                f"YHard={int(getattr(self, '_focused_yaw_hard_events', 0))} "
                f"DHard={int(getattr(self, '_yaw_decenter_hard_events', 0))} "
                f"auth={str(getattr(self, '_camera_authority', 'FRONT_PRIMARY'))} "
                f"stage={str(getattr(self, '_speed_stage', 'CHASE_FAST'))} "
                f"VXS={float(getattr(self, '_last_stage_vx_scale', float(getattr(self.cfg, 'vx_scale', 0.0)))):.2f} "
                f"VYS={float(getattr(self, '_last_stage_vy_scale', float(getattr(self.cfg, 'vy_scale', 0.0)))):.2f} "
                f"Fassist={float(getattr(self, '_last_fast_chase_assist_vx', 0.0)):+.2f} "
                f"PDvx={float(getattr(self, '_last_bottom_pd_assist_vx', 0.0)):+.2f} "
                f"PDvy={float(getattr(self, '_last_bottom_pd_assist_vy', 0.0)):+.2f} "
                f"BvxImg={float(getattr(self, '_bottom_img_vel_x', 0.0)):+.2f} "
                f"BvyImg={float(getattr(self, '_bottom_img_vel_y', 0.0)):+.2f} "
                f"BVok={int(bool(getattr(self, '_bottom_velocity_ready', False)))} "
                f"BVstreak={int(getattr(self, '_bottom_velocity_ready_streak', 0))} "
                f"cam={str(getattr(self, '_active_camera', 'front'))} "
                f"Fcam={str(getattr(self.cfg, 'front_camera_name', 'front_center'))} "
                f"Dcam={str(getattr(self.cfg, 'downward_camera_name', 'bottom_center'))} "
                f"FF={str(getattr(self, '_front_full_tracker_mode', 'LOST'))} "
                f"Fupd={int(getattr(self, '_front_full_tracker_updates', 0))} "
                f"BF={str(getattr(self, '_bottom_full_tracker_mode', 'LOST'))} "
                f"Braw={str(getattr(self, '_bottom_full_tracker_raw_mode', 'NONE'))} "
                f"Bupd={int(getattr(self, '_bottom_full_tracker_updates', 0))} "
                f"Bcand={int(getattr(self, '_bottom_candidate_count', 0))} "
                f"Bscan={float(getattr(self, '_bottom_candidate_scan_score', 0.0)):.2f} "
                f"BscanUsed={int(bool(getattr(self, '_bottom_candidate_scan_used', False)))} "
                f"Breloc={int(getattr(self, '_bottom_relock_attempts', 0))} "
                f"BmF={int(getattr(self, '_bottom_full_tracker_match_frames', 0))} "
                f"BpF={int(getattr(self, '_bottom_full_tracker_pred_frames', 0))} "
                f"BlF={int(getattr(self, '_bottom_full_tracker_lost_frames', 0))} "
                f"BAssist={int(getattr(self, '_bottom_alignment_assist_events', 0))} "
                f"dBvx={float(getattr(self, '_last_bottom_alignment_assist_vx', 0.0)):+.2f} "
                f"dBvy={float(getattr(self, '_last_bottom_alignment_assist_vy', 0.0)):+.2f} "
                f"handoff={str(getattr(self, '_handoff_phase', 'CHASE_FRONT'))} "
                f"Gate={int(bool(getattr(self, '_handoff_candidate_gate', False)))} "
                f"Fresh={int(bool(getattr(self, '_bottom_match_fresh', False)))} "
                f"Barea={float(getattr(self, '_bottom_bbox_area_norm', 0.0)):.3f} "
                f"Bcx={float(getattr(self, '_bottom_err_x', 0.0)):+.2f} "
                f"Bcy={float(getattr(self, '_bottom_err_y', 0.0)):+.2f} "
                f"BCok={int(bool(getattr(self, '_bottom_center_ok', False)))} "
                f"BYok={int(bool(getattr(self, '_bottom_yaw_ok', False)))} "
                f"BM={int(bool(getattr(self, '_bottom_match', False)))} "
                f"BC={int(bool(getattr(self, '_bottom_confirmed', False)))} "
                f"Bsim={float(getattr(self, '_bottom_similarity', 0.0)):.2f} "
                f"BF={str(getattr(self, '_bottom_full_tracker_mode', 'LOST'))} "
                f"Braw={str(getattr(self, '_bottom_full_tracker_raw_mode', 'NONE'))} "
                f"Bupd={int(getattr(self, '_bottom_full_tracker_updates', 0))} "
                f"Bcand={int(getattr(self, '_bottom_candidate_count', 0))} "
                f"Bscan={float(getattr(self, '_bottom_candidate_scan_score', 0.0)):.2f} "
                f"Bstreak={int(getattr(self, '_bottom_match_streak', 0))} "
                f"realD={float(getattr(self, '_current_chase_distance_m', 999.0)):.2f} "
                f"initD={float(getattr(self, '_initial_chase_distance_m', 999.0)):.2f} "
                f"bestD={float(getattr(self, '_best_chase_distance_m', 999.0)):.2f} "
                f"dmg={float(getattr(self, '_distance_damage_ratio', 0.0)):.2f} "
                f"VL={int(bool(getattr(self, '_handoff_visual_lidar_trigger', False)))} "
                f"VS={int(bool(getattr(self, '_handoff_visual_scan_trigger', False)))} "
                f"Vscore={float(getattr(self, '_handoff_visual_score', 0.0)):.1f} "
                f"Ldir={str(getattr(self, '_handoff_lidar_direction', 'none'))} "
                f"Ldist={float(getattr(self, '_handoff_lidar_dist_m', 999.0)):.1f} "
                f"R_obst={reward_parts.get('obstacle_penalty', 0.0):+.2f} "
                f"R_smooth={reward_parts.get('smooth_follow_reward', 0.0):+.2f} "
                f"dur={dur:.1f}s reason={term_reason}"
            )
            reward_diag = self._reward_diagnostics_snapshot(final_reward=float(self._ep_return))
            print(
                f"[EP {self.episode_id} REWARD_DIAG] "
                f"Rsum={reward_diag['part_sum']:+.2f} "
                f"Rdiff={reward_diag['reward_diff']:+.2f} "
                f"TERM={self._format_reward_top_items(reward_diag['terminal_items'], 6)}"
            )
            print(
                f"[EP {self.episode_id} NEG_TOP] "
                f"{self._format_reward_top_items(reward_diag['negative_top'], 10)}"
            )
            print(
                f"[EP {self.episode_id} POS_TOP] "
                f"{self._format_reward_top_items(reward_diag['positive_top'], 10)}"
            )


        self._prev_bottom_abs_err_y = abs(float(getattr(self, "_bottom_err_y", 0.0)))

        info = {
            "episode_id": self.episode_id,
            "step_in_episode": self.step_in_episode,
            "termination_reason": term_reason,
            "yaw_delta_from_initial_deg": float(getattr(self, "_last_yaw_delta_deg", 0.0)),
            "commanded_yaw_delta_deg": float(getattr(self, "_commanded_yaw_delta_deg", 0.0)),
            "cumulative_abs_yaw_cmd_deg": float(getattr(self, "_cumulative_abs_yaw_cmd_deg", 0.0)),
            "effective_yaw_delta_deg": float(getattr(self, "_effective_yaw_delta_deg", 0.0)),
            "last_yaw_rate_cmd_dps": float(getattr(self, "_last_yaw_rate_cmd_dps", 0.0)),
            "yaw_shield_applied": bool(getattr(self, "_yaw_shield_applied", False)),
            "yaw_shield_events": int(getattr(self, "_yaw_shield_events", 0)),
            "last_yaw_shield_pre": float(getattr(self, "_last_yaw_shield_pre", 0.0)),
            "last_yaw_shield_post": float(getattr(self, "_last_yaw_shield_post", 0.0)),
            "yaw_to_strafe_events": int(getattr(self, "_yaw_to_strafe_events", 0)),
            "last_yaw_to_strafe_delta_vy": float(getattr(self, "_last_yaw_to_strafe_delta_vy", 0.0)),
            "agent1_yaw_lock_events": int(getattr(self, "_agent1_yaw_lock_events", 0)),
            "agent1_yaw_limit_events": int(getattr(self, "_agent1_yaw_limit_events", 0)),
            "agent1_yaw_lock_applied": bool(getattr(self, "_agent1_yaw_lock_applied", False)),
            "agent1_yaw_limit_applied": bool(getattr(self, "_agent1_yaw_limit_applied", False)),
            "last_yaw_allowed_abs": float(getattr(self, "_last_yaw_allowed_abs", 1.0)),
            "last_blocked_yaw_action": float(getattr(self, "_last_blocked_yaw_action", 0.0)),
            "last_yaw_lock_penalty": float(getattr(self, "_last_yaw_lock_penalty", 0.0)),
            "focused_yaw_hard_events": int(getattr(self, "_focused_yaw_hard_events", 0)),
            "yaw_decenter_hard_events": int(getattr(self, "_yaw_decenter_hard_events", 0)),
            "last_focused_yaw_hard_penalty": float(getattr(self, "_last_focused_yaw_hard_penalty", 0.0)),
            "focus_streak_s": float(self._focus_streak),
            "global_max_focus_s": float(self._global_max_focus),
            "min_obstacle_dist_m": float(horizontal_min_obst),
            "down_dist_m": float(obstacle_dict["down_dist_m"]),
            "bottom_match": bool(getattr(self, "_bottom_match", False)),
            "bottom_confirmed": bool(getattr(self, "_bottom_confirmed", False)),
            "bottom_match_fresh": bool(getattr(self, "_bottom_match_fresh", False)),
            "bottom_bbox_area_norm": float(getattr(self, "_bottom_bbox_area_norm", 0.0)),
            "bottom_err_x": float(getattr(self, "_bottom_err_x", 0.0)),
            "bottom_err_y": float(getattr(self, "_bottom_err_y", 0.0)),
            "bottom_center_error": float(getattr(self, "_bottom_center_error", 0.0)),
            "bottom_center_ok": bool(getattr(self, "_bottom_center_ok", False)),
            "bottom_yaw_ok": bool(getattr(self, "_bottom_yaw_ok", False)),
            "handoff_candidate_gate": bool(getattr(self, "_handoff_candidate_gate", False)),
            "bottom_similarity": float(getattr(self, "_bottom_similarity", 0.0)),
            "bottom_match_streak": int(getattr(self, "_bottom_match_streak", 0)),
            "front_weight": float(getattr(self, "_front_weight", 1.0)),
            "bottom_weight": float(getattr(self, "_bottom_weight", 0.0)),
            "handoff_ready": bool(getattr(self, "_handoff_ready", False)),
            "handoff_phase": str(getattr(self, "_handoff_phase", "CHASE_FRONT")),
            "camera_authority": str(getattr(self, "_camera_authority", "FRONT_PRIMARY")),
            "active_camera": str(getattr(self, "_active_camera", "front")),
            "fusion_has_target": bool(getattr(self, "_fusion_has_target", False)),
            "fusion_err_x": float(getattr(self, "_fusion_err_x", 0.0)),
            "fusion_err_y": float(getattr(self, "_fusion_err_y", 0.0)),
            "fusion_confidence": float(getattr(self, "_fusion_confidence", 0.0)),
            "bottom_alignment_assist_events": int(getattr(self, "_bottom_alignment_assist_events", 0)),
            "last_bottom_alignment_assist_vx": float(getattr(self, "_last_bottom_alignment_assist_vx", 0.0)),
            "last_bottom_alignment_assist_vy": float(getattr(self, "_last_bottom_alignment_assist_vy", 0.0)),
            "handoff_visual_lidar_trigger": bool(getattr(self, "_handoff_visual_lidar_trigger", False)),
            "handoff_visual_scan_trigger": bool(getattr(self, "_handoff_visual_scan_trigger", False)),
            "handoff_visual_score": float(getattr(self, "_handoff_visual_score", 0.0)),
            "handoff_lidar_dist_m": float(getattr(self, "_handoff_lidar_dist_m", float("inf"))),
            "handoff_lidar_direction": str(getattr(self, "_handoff_lidar_direction", "none")),
            "obstacle_source": str(obstacle_dict.get("obstacle_source", "unknown")),
            "lidar_valid": bool(obstacle_dict.get("lidar_valid", False)),
            "lidar_point_count": int(obstacle_dict.get("lidar_point_count", 0)),
            "alt_agl_m": float(drone_state.altitude_m),
            "match_pct": float(match_pct),
            "pred_pct": float(pred_pct),
            "none_pct": float(none_pct),
            "safety_intervention_rate_pct": float(safety_intervention_rate),
            "safety_intervention": bool(safety_info.get("safety_intervention", False)),
            "safety_phase": str(safety_info.get("safety_phase", "CHASE")),
            "safety_reasons": safety_info.get("safety_reasons", []),
            "raw_action": raw_action.copy(),
            "safe_action": safe_action.copy(),
            "reward_parts": reward_parts,
            "pitch_deg": float(getattr(self, "_last_pitch_deg", 0.0)),
            "pitch_rate_dps": float(getattr(self, "_last_pitch_rate_dps", 0.0)),
            "pitch_smoothness_penalty": float(getattr(self, "_last_pitch_smoothness_penalty", 0.0)),
            "vx_slew_limited": bool(getattr(self, "_last_vx_slew_limited", False)),
            "vx_slew_delta": float(getattr(self, "_last_vx_slew_delta", 0.0)),
            "fine_position_error_inf": float(reward_parts.get("fine_position_error_inf", 0.0)),
            "fine_position_reward_total": float(
                reward_parts.get("fine_position_bonus", 0.0)
                + reward_parts.get("fine_position_penalty", 0.0)
                + reward_parts.get("fine_position_ready_bonus", 0.0)
                + reward_parts.get("fine_position_landing_ready_bonus", 0.0)
            ),
            "fine_position_bbox_rel_err": float(reward_parts.get("fine_position_bbox_rel_err", 999.0)),
            "fine_position_bbox_rel_err_x": float(reward_parts.get("fine_position_bbox_rel_err_x", 0.0)),
            "fine_position_bbox_rel_err_y": float(reward_parts.get("fine_position_bbox_rel_err_y", 0.0)),
            "bottom_y_error_penalty_info": float(reward_parts.get("bottom_y_error_penalty", 0.0)),
            "bottom_y_progress_reward_info": float(reward_parts.get("bottom_y_progress_reward", 0.0)),
            "bottom_wrong_vx_penalty_info": float(reward_parts.get("bottom_wrong_vx_penalty", 0.0)),
            "bottom_correct_vx_bonus_info": float(reward_parts.get("bottom_correct_vx_bonus", 0.0)),
            "obs_dict": obs_dict,
            "tracking_mode": tracking_mode,
            "raw_tracker_mode": getattr(self.tracker, "last_mode", "NONE"),
            "tracker_reject_reason": getattr(self.tracker, "last_reject_reason", ""),
            "tracker_reject_frames": int(getattr(self.tracker, "_reject_frames", 0)),
            "tracker_identity_metrics": dict(getattr(self.tracker, "last_identity_metrics", {}) or {}),
            "tracking_accepted": bool(tracking_result.get("accepted_tracker", False)),
            "tracking_pred_frames": int(tracking_result.get("pred_frames", 0)),
            "stable_bbox_xyxy": None if tracking_result.get("stable_bbox") is None else np.asarray(tracking_result.get("stable_bbox"), dtype=np.float32).copy(),
            "raw_bbox_xyxy": None if bbox_raw is None else np.asarray(bbox_raw, dtype=np.float32).copy(),
            "collision_raw": bool(collision_raw),
            "collision_object_name": collision_object_name,
            "collision_penetration_depth": float(collision_penetration_depth),
            "episode_done": bool(done),
        }

        return obs, float(reward), bool(done), False, info
