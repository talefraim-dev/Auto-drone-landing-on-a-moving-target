from dataclasses import dataclass


"""
EnvConfig v37 — UAV RL architecture.

Architecture:
1. Observation vector is 37 features.
2. Target state is estimated from vision / tracker.
3. Drone self-state is read from AirSim.
4. Obstacle awareness is based on directional AirSim Distance Sensors.
5. raw_action from the RL policy is filtered by Safety Layer before being sent to AirSim.

Action convention:
    action = [vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd]
    each value is normalized to [-1, 1]

Recommended sign convention:
    vx_cmd > 0  => move forward
    vx_cmd < 0  => move backward
    vy_cmd > 0  => move right
    vy_cmd < 0  => move left
    vz_cmd > 0  => move up
    vz_cmd < 0  => move down
"""


@dataclass
class EnvConfig:
    # -----------------------------
    # General / UI / logs
    # -----------------------------
    show_cv_window: bool = True
    print_reset: bool = False
    print_ep_summary: bool = True
    print_sensor_errors: bool = False
    print_obstacle_debug: bool = True
    obstacle_debug_every_n_steps: int = 30

    # -----------------------------
    # Timing
    # -----------------------------
    max_step_sec: float = 0.20
    cmd_duration_s: float = 0.10
    max_episode_steps: int = 2000

    # -----------------------------
    # Reset / initial flight state
    # -----------------------------
    reset_takeoff_altitude_m: float = 5.0
    reset_move_to_z_velocity: float = 2.0
    reset_settle_sec: float = 0.5

    # Do not terminate because of obstacle readings during the first few steps.
    # This protects against sensor warmup spikes after reset/takeoff.
    ignore_obstacle_termination_first_steps: int = 30

    # -----------------------------
    # Action scaling
    # raw normalized action -> real AirSim command
    # -----------------------------
    vx_scale: float = 2.0
    vy_scale: float = 2.0
    vz_scale: float = 0.5
    yaw_rate_scale_dps: float = 70.0
    freeze_vz: bool = False

    # -----------------------------
    # Observation signature — v37
    # -----------------------------
    # 0-27   : Vision + drone + mission state
    # 28-36 : Obstacle awareness
    obs_dim: int = 37

    use_self_state_obs: bool = True
    use_accel_slots: bool = False
    use_abs_yaw_obs: bool = False

    use_lidar_sectors_obs: bool = True
    use_obstacle_penalty: bool = True
    use_collision_termination: bool = True

    use_range_proxy_from_area: bool = True
    use_range_rate_proxy: bool = True
    use_real_range_to_target: bool = False

    use_disturbance_estimate: bool = False
    use_ground_normal_estimate: bool = False

    # -----------------------------
    # Vision / image normalization
    # -----------------------------
    image_width: int = 960
    image_height: int = 540

    img_v_rel_per_sec_max: float = 2.0
    img_acc_rel_per_sec2_max: float = 5.0

    # -----------------------------
    # Focus / target-loss termination
    # -----------------------------
    focus_fail_sec: float = 2.0

    center_ok_score: float = 0.75

    # Kept for tracker/backward compatibility.
    center_ok_dist: float = 0.30
    pred_center_ok_dist: float = 0.18
    pred_focus_max_sec: float = 3.0

    # -----------------------------
    # Follow Agent reward weights — v37
    # -----------------------------
    w_center: float = 3.0
    w_distance: float = 2.0
    w_visibility: float = 0.5
    w_lost_target: float = 3.0
    w_altitude_safe: float = 0.3
    w_altitude_low_penalty: float = 2.0
    w_altitude_high_penalty: float = 1.0
    w_smooth_follow: float = 1.0
    w_control: float = 0.05
    w_action_delta: float = 0.10
    w_obstacle: float = 4.0
    w_safety_intervention: float = 0.3

    penalty_collision: float = 50.0

    desired_distance_proxy: float = 0.45
    distance_tolerance: float = 0.35

    min_safe_altitude_m: float = 3.0
    max_safe_altitude_m: float = 20.0
    min_termination_altitude_m: float = 1.0
    max_termination_altitude_m: float = 30.0

    max_img_motion: float = 1.5

    # -----------------------------
    # Drone-state normalization scales
    # -----------------------------
    vb_max_mps: float = 8.0
    vz_max_mps: float = 5.0
    yaw_rate_max_dps: float = 180.0
    att_max_deg: float = 35.0
    alt_max_m: float = 30.0

    # Deprecated compatibility.
    accel_max_mps2: float = 10.0
    alt_rate_max_mps: float = 6.0
    disturb_max_mps: float = 6.0

    # -----------------------------
    # Obstacle sensors / Safety Layer
    # -----------------------------
    vehicle_name: str = "Drone1"

    distance_sensor_front: str = "DistanceFront"
    distance_sensor_front_left: str = "DistanceFrontLeft"
    distance_sensor_front_right: str = "DistanceFrontRight"
    distance_sensor_left: str = "DistanceLeft"
    distance_sensor_right: str = "DistanceRight"
    distance_sensor_back: str = "DistanceBack"
    distance_sensor_down: str = "DistanceDown"

    # Backward compatibility only. Do not use in v37 code.
    distance_sensor_name: str = "DistanceFront"

    lidar_max_dist_m: float = 20.0

    safety_enabled: bool = True
    obstacle_safe_dist_m: float = 5.0
    obstacle_warning_dist_m: float = 3.0
    obstacle_emergency_dist_m: float = 1.0
    obstacle_emergency_termination_dist_m: float = 0.5

    min_speed_scale_near_obstacle: float = 0.2
    obstacle_steer_strength: float = 0.35
    emergency_up_cmd: float = 0.2

    # Deprecated obstacle penalty names.
    obstacle_danger_dist_m: float = 1.0
    obstacle_penalty_k: float = 1.0
    obstacle_danger_penalty_k: float = 2.0

    # -----------------------------
    # Episode / success criteria
    # -----------------------------
    stable_follow_success_sec: float = 5.0
    success_centered_score_min: float = 0.75
    success_distance_proxy_min: float = 0.30
    success_distance_proxy_max: float = 0.65
    success_collision_risk_max: float = 0.3

    # -----------------------------
    # Follow anti-cheat / safety — disabled in v37 initially
    # -----------------------------
    use_follow_ground_termination: bool = False
    min_follow_alt_m: float = 0.8
    ground_contact_grace_s: float = 0.5
    penalty_ground_contact: float = 25.0

    use_low_alt_penalty: bool = False
    w_low_alt: float = 4.0

    use_too_close_penalty: bool = False
    close_margin_norm: float = 0.15
    w_too_close: float = 8.0

    use_stuck_penalty: bool = False
    stuck_speed_thresh_mps: float = 0.15
    stuck_action_thresh: float = 0.08
    stuck_time_s: float = 1.5
    penalty_stuck_follow: float = 8.0

    # -----------------------------
    # Deprecated old reward weights
    # -----------------------------
    match_warmup_reward: float = 0.9
    pred_warmup_reward: float = 0.3
    pred_penalty_per_sec: float = 0.25
    energy_penalty_k: float = 0.06

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
    penalty_no_bbox: float = 2.5

    # -----------------------------
    # Deprecated old intent defaults
    # -----------------------------
    desired_range_norm: float = 0.0
    desired_alt_norm: float = 0.0
    mode_norm: float = -1.0
    phase_progress_norm: float = -1.0
