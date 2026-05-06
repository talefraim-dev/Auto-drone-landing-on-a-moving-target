from dataclasses import dataclass


"""
Manual:
1. To freeze vertical axis in early stage: freeze_vz = True
2. To enable LiDAR + obstacle penalties:
     - use_lidar_sectors_obs = True
     - use_obstacle_penalty = True
3. To enable disturbance channels: use_disturbance_estimate = True
4. To enable ground-normal estimation later: use_ground_normal_estimate = True
"""


@dataclass
class EnvConfig:
    # -----------------------------
    # General / UI / logs
    # -----------------------------
    show_cv_window: bool = True
    print_reset: bool = False
    print_ep_summary: bool = True

    # -----------------------------
    # Timing
    # -----------------------------
    max_step_sec: float = 0.20
    cmd_duration_s: float = 0.10

    # -----------------------------
    # Action scaling (vx, vy, vz, yaw_rate)
    # -----------------------------
    vx_scale: float = 2.0
    vy_scale: float = 2.0
    vz_scale: float = 0.5
    yaw_rate_scale_dps: float = 70.0
    freeze_vz: bool = False

    # -----------------------------
    # Observation signature
    # -----------------------------
    obs_dim: int = 48

    use_self_state_obs: bool = True
    use_accel_slots: bool = True
    use_abs_yaw_obs: bool = True

    use_lidar_sectors_obs: bool = False
    use_obstacle_penalty: bool = False
    use_collision_termination: bool = False

    use_range_proxy_from_area: bool = True
    use_range_rate_proxy: bool = True
    use_real_range_to_target: bool = False

    use_disturbance_estimate: bool = False
    use_ground_normal_estimate: bool = False

    # -----------------------------
    # Focus / termination
    # -----------------------------
    focus_fail_sec: float = 16.0
    center_ok_dist: float = 0.30
    pred_center_ok_dist: float = 0.18
    pred_focus_max_sec: float = 3.0

    # -----------------------------
    # Reward weights
    # -----------------------------
    match_warmup_reward: float = 0.9
    pred_warmup_reward: float = 0.3
    pred_penalty_per_sec: float = 0.25
    energy_penalty_k: float = 0.06

    w_center: float = 15.0
    center_decay: float = 2.0
    w_area: float = 1.2
    w_focus: float = 4.5
    w_vz: float = 0.03
    w_calm_no_bbox: float = 0.8
    w_smooth_delta: float = 1.8
    w_time_penalty: float = 0.03
    w_range_error: float = 4.0
    w_progress: float = 2.0
    w_altitude_error: float = 1.2
    w_bearing_error: float = 1.5

    penalty_focus_timeout: float = 20.0
    penalty_pred_focus_timeout: float = 20.0
    penalty_collision: float = 50.0
    penalty_no_bbox: float = 2.5

    # -----------------------------
    # Normalization scales
    # -----------------------------
    img_v_rel_per_sec_max: float = 1.5
    vb_max_mps: float = 8.0
    accel_max_mps2: float = 10.0
    yaw_rate_max_dps: float = 180.0
    att_max_deg: float = 35.0
    alt_max_m: float = 30.0
    alt_rate_max_mps: float = 6.0

    lidar_max_dist_m: float = 30.0
    obstacle_safe_dist_m: float = 2.0
    obstacle_danger_dist_m: float = 1.0
    obstacle_penalty_k: float = 1.0
    obstacle_danger_penalty_k: float = 2.0

    disturb_max_mps: float = 6.0


    # -----------------------------
    # Follow anti-cheat / safety
    # -----------------------------
    use_follow_ground_termination: bool = False
    min_follow_alt_m: float = 0.8
    ground_contact_grace_s: float = 0.5
    penalty_ground_contact: float = 25.0

    use_low_alt_penalty: bool = False
    w_low_alt: float = 4.0

    use_too_close_penalty: bool = True
    close_margin_norm: float = 0.15
    w_too_close: float = 8.0

    use_stuck_penalty: bool = True
    stuck_speed_thresh_mps: float = 0.15
    stuck_action_thresh: float = 0.08
    stuck_time_s: float = 1.5
    penalty_stuck_follow: float = 8.0

    # -----------------------------
    # AirSim sensor names
    # -----------------------------
    vehicle_name: str = "Drone1"
    distance_sensor_name: str = "Distance"

    # -----------------------------
    # Intent defaults
    # -----------------------------
    desired_range_norm: float = 0.0
    desired_alt_norm: float = 0.0
    mode_norm: float = -1.0
    phase_progress_norm: float = -1.0
