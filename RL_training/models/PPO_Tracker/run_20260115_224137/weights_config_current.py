# weights_config.py
from dataclasses import dataclass


"""
Manual:
1.To open vz gradient: freeze_vz = false.
2.To open LiDAR and obstacle penalty:
     -  use_lidar_sector_obs =  true
     -  use_obstacle_penalty =  true
3.To open wind: use_disturbance_estimate = True
4.To open sloppy land area: use_ground_normal_estimate = True

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
    # Action scaling (vx,vy,vz,yaw_rate)
    # -----------------------------
    vx_scale: float = 2.0
    vy_scale: float = 2.0
    vz_scale: float = 3.0
    yaw_rate_scale_dps: float = 70.0  # deg/sec

    # Stage-1 stability
    freeze_vz: bool = True

    # -----------------------------
    # Observation signature
    # -----------------------------
    obs_dim: int = 34

    # Which blocks to actually fill (others become safe defaults)
    use_self_state_obs: bool = True
    use_accel_slots: bool = False

    use_lidar_sectors_obs: bool = False
    use_obstacle_penalty: bool = False
    use_collision_termination: bool = False

    use_range_proxy_from_area: bool = False
    use_range_rate_proxy: bool = False

    # If you later have real range-to-target from LiDAR/GPS/vision fusion:
    use_real_range_to_target: bool = False

    # Wind / disturbance
    use_disturbance_estimate: bool = False  # enable when you want wind-awareness
    # Ground plane normal / slope
    use_ground_normal_estimate: bool = False  # enable when you have depth/lidar plane-fit

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

    # shaping
    w_center: float = 2.5
    center_decay: float = 3.0
    w_area: float = 1.2
    w_focus: float = 1.0  # multiplied by min(2, focus/4)

    # termination penalties
    penalty_focus_timeout: float = 20.0
    penalty_pred_focus_timeout: float = 20.0
    penalty_collision: float = 50.0

    # no bbox penalty (keep it mild to not kill early learning)
    penalty_no_bbox: float = 2.5

    # -----------------------------
    # Normalization scales (tune for sim/real)
    # -----------------------------
    img_v_rel_per_sec_max: float = 1.5

    vb_max_mps: float = 8.0
    yaw_rate_max_dps: float = 180.0
    att_max_deg: float = 35.0

    alt_max_m: float = 30.0
    alt_rate_max_mps: float = 6.0

    # LiDAR / distance sectors
    lidar_max_dist_m: float = 30.0
    obstacle_safe_dist_m: float = 2.0
    obstacle_penalty_k: float = 1.0

    # Disturbance estimate normalization (v_meas - v_cmd)
    disturb_max_mps: float = 6.0

    # -----------------------------
    # AirSim sensor names
    # -----------------------------
    vehicle_name: str = "Drone1"
    distance_sensor_name: str = "Distance"

    # -----------------------------
    # Intent defaults (stay constant unless you change manually)
    # -----------------------------
    # These are normalized targets:
    # desired_range_norm: [-1,1]  (you decide the mapping; by default 0 means "mid")
    # desired_alt_norm:   [-1,1]
    desired_range_norm: float = 0.0
    desired_alt_norm: float = 0.0

    # mode_norm: [-1,1] e.g. -1 tracking-only, 0 keep-range, +1 landing
    mode_norm: float = -1.0

    # phase_progress: [-1,1] (map 0..1 to -1..1)
    phase_progress_norm: float = -1.0
