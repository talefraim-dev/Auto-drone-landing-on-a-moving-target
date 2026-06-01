from dataclasses import dataclass


@dataclass
class EnvConfig:
    """
    Final 37-observation UAV follow/landing environment config.

    No environment variables are used in the final architecture.
    Change values here only.

    AirSim NED command convention used by moveByVelocityBodyFrameAsync:
        vx > 0  : forward
        vy > 0  : right
        vz > 0  : down
        vz < 0  : up
    """

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
    # Reset / takeoff
    # -----------------------------
    reset_takeoff_altitude_m: float = 5.0
    reset_move_to_z_velocity: float = 2.0
    reset_settle_sec: float = 0.5

    ignore_obstacle_termination_first_steps: int = 30
    ignore_collision_termination_first_steps: int = 20

    # -----------------------------
    # Action scaling
    # -----------------------------
    vx_scale: float = 2.0
    vy_scale: float = 2.0
    vz_scale: float = 0.50
    yaw_rate_scale_dps: float = 70.0

    # Stage 1 training:
    # Keep altitude stable while the policy learns yaw/forward/side control.
    # This is NOT vz=0. drone_env.py converts this to altitude hold.
    freeze_vz: bool = True

    # -----------------------------
    # Dynamic altitude safety / hold
    # -----------------------------
    altitude_hold_enabled: bool = True
    altitude_hold_target_m: float = 5.0
    altitude_hold_kp: float = 0.35
    altitude_hold_max_vz_mps: float = 0.80

    altitude_safety_enabled: bool = True
    min_safe_altitude_m: float = 2.0
    min_termination_altitude_m: float = 0.75
    max_safe_altitude_m: float = 20.0
    max_termination_altitude_m: float = 30.0
    low_altitude_climb_vz_mps: float = 0.35
    emergency_climb_vz_mps: float = 0.90

    # -----------------------------
    # Dynamic target distance safety
    # distance_proxy_norm = 1 - sqrt(bbox_area)
    # lower value means closer/larger target.
    # -----------------------------
    desired_distance_proxy: float = 0.94
    distance_tolerance: float = 0.08
    min_target_distance_proxy: float = 0.88
    block_forward_when_too_close: bool = True

    # -----------------------------
    # Observation signature — v37
    # -----------------------------
    obs_dim: int = 37

    use_self_state_obs: bool = True
    use_lidar_sectors_obs: bool = True
    use_obstacle_penalty: bool = True
    use_collision_termination: bool = True

    image_width: int = 960
    image_height: int = 540

    img_v_rel_per_sec_max: float = 2.0
    img_acc_rel_per_sec2_max: float = 5.0
    focus_fail_sec: float = 2.0
    center_ok_score: float = 0.75

    # -----------------------------
    # Drone-state normalization scales
    # -----------------------------
    vb_max_mps: float = 8.0
    vz_max_mps: float = 5.0
    yaw_rate_max_dps: float = 180.0
    att_max_deg: float = 35.0
    alt_max_m: float = 30.0
    max_img_motion: float = 1.5

    # -----------------------------
    # Reward weights — balanced final v37
    # -----------------------------
    w_center: float = 8.0
    center_reward_alpha: float = 5.0

    w_distance: float = 2.5
    w_visibility: float = 1.0
    w_lost_target: float = 6.0

    w_altitude_safe: float = 0.4
    w_altitude_low_penalty: float = 4.0
    w_altitude_high_penalty: float = 2.0

    w_smooth_follow: float = 1.8
    w_control: float = 0.05
    w_action_delta: float = 0.35
    w_slow: float = 0.20
    w_time: float = 0.015

    w_obstacle: float = 5.0
    w_safety_intervention: float = 0.5

    penalty_collision: float = 60.0
    penalty_timeout: float = 10.0
    penalty_altitude_termination: float = 20.0

    # -----------------------------
    # AirSim vehicle and distance sensors
    # -----------------------------
    vehicle_name: str = "Drone1"

    distance_sensor_front: str = "DistanceFront"
    distance_sensor_front_left: str = "DistanceFrontLeft"
    distance_sensor_front_right: str = "DistanceFrontRight"
    distance_sensor_left: str = "DistanceLeft"
    distance_sensor_right: str = "DistanceRight"
    distance_sensor_back: str = "DistanceBack"
    distance_sensor_down: str = "DistanceDown"

    lidar_max_dist_m: float = 20.0

    safety_enabled: bool = True
    obstacle_safe_dist_m: float = 5.0
    obstacle_warning_dist_m: float = 3.0
    obstacle_emergency_dist_m: float = 1.0
    obstacle_emergency_termination_dist_m: float = 0.50

    min_speed_scale_near_obstacle: float = 0.2
    obstacle_steer_strength: float = 0.35
    emergency_up_cmd: float = 0.8

    # -----------------------------
    # Success criteria
    # -----------------------------
    stable_follow_success_sec: float = 5.0
    success_centered_score_min: float = 0.75
    success_distance_proxy_min: float = 0.88
    success_distance_proxy_max: float = 0.98
    success_collision_risk_max: float = 0.3
