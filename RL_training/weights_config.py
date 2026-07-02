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

    # CV debug display mode:
    #   "active" -> show only the camera used by the current mission phase.
    #               CHASE shows front camera. LANDING shows downward camera.
    #   "front"  -> always show front camera.
    #   "bottom" -> always show downward camera.
    #   "dual"   -> show both cameras side-by-side. Useful for debugging, but
    #               expensive and not recommended during training.
    cv_display_mode: str = "active"
    show_dual_camera_cv: bool = False

    # Single active camera is cheap enough to show larger.
    cv_debug_scale: float = 1.0
    dual_camera_update_every_n_steps: int = 15
    cv_debug_render_every_n_steps: int = 2
    print_reset: bool = True
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
    # For RL chase/follow training, episodes should start already airborne.
    # Takeoff itself is not part of the learned chase policy.
    reset_start_airborne_with_pose: bool = True
    reset_verify_altitude_min_m: float = 2.0

    ignore_obstacle_termination_first_steps: int = 30
    ignore_collision_termination_first_steps: int = 20

    # -----------------------------
    # Action scaling
    # -----------------------------
    vx_scale: float = 2.8
    vy_scale: float = 2.0
    vz_scale: float = 0.50
    yaw_rate_scale_dps: float = 70.0

    # -------------------------------------------------------------
    # Full-throttle chase + bottom-camera velocity matching
    # -------------------------------------------------------------
    # CHASE_FAST closes distance aggressively. Once the target is visible and
    # large enough in the bottom camera, the controller switches to
    # BOTTOM_VELOCITY_MATCH. In that state the drone estimates target relative
    # motion from d(Bcx,Bcy)/dt and matches it with a PD visual-servo command.
    staged_speed_enabled: bool = True
    fast_chase_forward_action_floor: float = 1.00
    fast_chase_forward_blend: float = 0.85
    fast_chase_max_action: float = 1.00
    fast_chase_min_real_distance_m: float = 2.20

    # Dynamic stage-specific speed scaling:
    # CHASE_FAST can use much higher forward speed to catch a faster target.
    # Bottom stages remain controlled so landing alignment is not destroyed.
    dynamic_stage_speed_enabled: bool = True
    chase_fast_vx_scale_min: float = 3.5
    chase_fast_vx_scale_max: float = 5.8
    chase_fast_vy_scale: float = 2.4
    chase_fast_distance_near_m: float = 3.0
    chase_fast_distance_far_m: float = 14.0
    bottom_velocity_match_vx_scale: float = 2.8
    bottom_velocity_match_vy_scale: float = 2.2
    bottom_landing_ready_vx_scale: float = 1.2
    bottom_landing_ready_vy_scale: float = 1.0

    bottom_velocity_match_enter_area: float = 0.036
    bottom_velocity_match_release_area: float = 0.022
    bottom_velocity_match_release_center_error: float = 0.68

    bottom_velocity_matching_enabled: bool = True
    bottom_velocity_ema_alpha: float = 0.35
    bottom_velocity_clip_per_s: float = 4.0
    bottom_velocity_ready_abs_vx: float = 0.18
    bottom_velocity_ready_abs_vy: float = 0.18
    bottom_velocity_ready_center_error: float = 0.22
    bottom_velocity_ready_streak_required: int = 12
    alignment_ready_requires_bottom_velocity: bool = True

    # PD visual-servo gains in normalized image coordinates.
    bottom_pd_kp_y_to_vx: float = 1.25
    bottom_pd_kd_y_to_vx: float = 0.34
    bottom_pd_kp_x_to_vy: float = 0.90
    bottom_pd_kd_x_to_vy: float = 0.26
    bottom_pd_blend: float = 0.72
    bottom_pd_max_action: float = 1.00
    bottom_pd_ready_max_action: float = 0.42

    # Stage reward shaping.
    w_fast_chase_throttle_bonus: float = 55.0
    w_fast_chase_progress_bonus: float = 95.0
    w_fast_chase_slow_penalty: float = 55.0
    fast_chase_far_distance_m: float = 2.50
    fast_chase_min_vx_action: float = 0.70

    w_bottom_velocity_error_penalty: float = 18.0
    w_bottom_velocity_progress_bonus: float = 16.0
    w_bottom_velocity_ready_bonus: float = 28.0

    # Fine position accuracy reward:
    # Reward/penalty based on relative position between the bottom-camera frame
    # center and the target bbox center. Applied only during bottom fine stages.
    fine_position_reward_enabled: bool = True
    fine_position_error_ready: float = 0.08
    fine_position_error_good: float = 0.14
    fine_position_bonus_weight: float = 95.0
    fine_position_bonus_gain: float = 42.0
    fine_position_penalty_weight: float = 65.0
    fine_position_penalty_power: float = 1.75
    fine_position_ready_bonus: float = 85.0
    fine_position_landing_ready_bonus: float = 140.0

    # BBox-relative fine position reward:
    # rel = |frame_center - bbox_center| / (bbox_half_size)
    # rel < 0.5 means the frame center is in the central half of the bbox.
    # rel near 1.0 means near bbox edges. rel > 1.0 means outside bbox.
    fine_position_use_bbox_relative: bool = True
    fine_position_bbox_safe_rel: float = 0.50
    fine_position_bbox_edge_rel: float = 1.00
    fine_position_bbox_ready_rel: float = 0.25
    fine_position_bbox_bonus_weight: float = 75.0
    fine_position_bbox_bonus_gain: float = 4.0
    fine_position_bbox_inside_penalty_weight: float = 35.0
    fine_position_bbox_edge_penalty_weight: float = 240.0
    fine_position_bbox_outside_penalty_weight: float = 900.0
    fine_position_bbox_ready_bonus: float = 90.0
    fine_position_bbox_landing_ready_bonus: float = 160.0

    # Handoff success must also be geometrically safe inside the target bbox.
    # This prevents ending the episode in a visually unsafe landing position.
    handoff_success_requires_bbox_safe: bool = True
    handoff_success_max_bbox_rel_err: float = 0.50

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
    desired_distance_proxy: float = 0.65
    distance_tolerance: float = 0.10
    min_target_distance_proxy: float = 0.45
    block_forward_when_too_close: bool = False

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
    # During training we give the agent a little more time to learn recovery.
    # Evaluation can be made stricter later by lowering this back to 2.0.
    focus_fail_sec: float = 3.0
    center_ok_score: float = 0.75

    # Strict chase/center fine-tuning termination rules.
    # These rules make bad exploration unprofitable by ending the episode and
    # cancelling positive return when the policy stops chasing the car.
    non_match_timeout_sec: float = 3.0
    approach_warmup_sec: float = 6.0
    not_approaching_timeout_sec: float = 5.0
    approach_target_distance_proxy: float = 0.88  # legacy bbox-proxy fallback, no longer primary
    approach_min_improvement: float = 0.010      # legacy bbox-proxy fallback, no longer primary
    approach_goal_distance_m: float = 2.50
    approach_min_improvement_m: float = 0.10
    not_approaching_penalty_growth_per_sec: float = 1000.0
    max_initial_yaw_delta_deg: float = 15.0
    # Distance-damage termination:
    # Do NOT terminate only because the drone failed to improve the best
    # distance quickly enough. Terminate only if it actually damages the
    # mission by moving away from the target by a large ratio.
    terminate_on_no_best_distance_improvement: bool = False
    distance_damage_termination_ratio: float = 0.35
    distance_damage_timeout_sec: float = 4.0
    target_lost_hard_fail_penalty: float = 12000.0

    # -----------------------------
    # Target recovery / overshoot training
    # -----------------------------
    # When the target is temporarily lost, the policy needs enough yaw authority
    # to search aggressively before the lost-target timeout ends.
    recovery_yaw_rate_scale_dps: float = 130.0
    recovery_forward_scale: float = 0.65

    # Heuristic overshoot detector:
    # If the target was very close/large and then disappears, this is usually the
    # drone passing the target. Lower distance_proxy_norm means closer target.
    overshoot_distance_proxy_threshold: float = 0.90

    # Reward shaping for fast reacquisition.
    w_recovery_yaw: float = 2.5
    w_wrong_recovery_yaw: float = 3.5
    w_fast_reacquire: float = 18.0
    w_overshoot_penalty: float = 10.0
    w_lost_time_accel_penalty: float = 4.0

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

    # Strong anti-yaw shaping:
    # When the front tracker is in MATCH and the target is already focused and
    # centered, yaw is unnecessary and should be heavily penalized.
    w_yaw_when_focused_centered: float = 150.0
    focused_centered_yaw_threshold: float = 0.035
    focused_center_threshold: float = 0.16
    focused_center_x_threshold: float = 0.11
    focused_center_y_threshold: float = 0.16
    focused_bbox_conf_threshold: float = 0.55
    focused_centered_yaw_power: float = 1.25

    # Exponential anti-yaw / anti-decentering shaping.
    w_exp_focused_centered_yaw: float = 260.0
    exp_focused_yaw_gain: float = 5.0
    exp_focused_yaw_clip: float = 0.55
    w_yaw_decenter_exp_penalty: float = 340.0
    yaw_decenter_exp_gain: float = 9.0
    yaw_decenter_error_clip: float = 0.18
    yaw_decenter_prev_center_threshold: float = 0.18
    yaw_decenter_current_max_threshold: float = 0.45

    # Hard credit reset for bad yaw habits:
    # Reward clipping can hide large per-step penalties. This Env-level rule
    # cancels accumulated positive return and applies an additional penalty when
    # the policy yawed unnecessarily in a focused/centered MATCH state.
    hard_penalize_focused_centered_yaw: bool = True
    suppress_yaw_hard_after_bottom_authority: bool = True
    focused_centered_yaw_hard_penalty: float = 2500.0
    yaw_decenter_hard_penalty: float = 3500.0
    terminate_on_focused_centered_yaw_hard_fail: bool = False

    # Handoff success must not hide a bad yaw strategy.
    # If the drone rotates too far from the initial chase heading, yaw failure
    # has priority over handoff_success.
    handoff_success_bonus: float = 260.0
    handoff_success_respects_yaw_limit: bool = False

    # Robust yaw failure detection:
    # Actual AirSim yaw readings can sometimes stay at 0.0 or be unavailable.
    # Therefore the environment also integrates the commanded yaw rate.
    yaw_deviation_use_command_integral: bool = True
    max_commanded_yaw_delta_deg: float = 45.0
    max_cumulative_abs_yaw_cmd_deg: float = 180.0
    yaw_deviation_hard_fail_penalty: float = 0.0

    # Anti-yaw action shield:
    # Keep episodes alive and prevent bad yaw commands before they reach AirSim.
    yaw_action_shield_enabled: bool = True
    yaw_shield_match_only: bool = True
    yaw_shield_x_zero_threshold: float = 0.07
    yaw_shield_x_center_threshold: float = 0.20
    yaw_shield_max_abs_when_centered: float = 0.035
    yaw_shield_min_conf: float = 0.45
    yaw_shield_allow_in_handoff_scan: bool = True

    # Yaw-to-strafe strategy:
    # In visual chase, horizontal target error should usually be corrected with
    # lateral body-frame motion (vy), not by rotating the drone. This prevents
    # the learned bad habit: drift sideways, get close, then yaw 90 degrees
    # toward the car.
    yaw_to_strafe_enabled: bool = True
    yaw_to_strafe_match_only: bool = True
    yaw_to_strafe_x_deadband: float = 0.035
    yaw_to_strafe_gain: float = 0.85
    yaw_to_strafe_blend: float = 0.80
    yaw_to_strafe_max_abs_vy: float = 0.75

    # When close to the target or already in bottom scan, keep front heading
    # stable and force yaw to zero while the target is still visible.
    yaw_close_heading_lock_enabled: bool = True
    yaw_close_heading_lock_distance_m: float = 11.5
    yaw_close_heading_lock_max_abs_yaw: float = 0.0

    # If the target is visible but no longer perfectly centered, allow only a
    # tiny yaw leakage. The strafe assist should handle x correction.
    yaw_visible_max_abs: float = 0.025
    yaw_visible_x_threshold: float = 0.45

    # Agent-1 adaptive yaw authority:
    # Yaw is allowed, but it must be earned by the visual situation.
    # Small/centered error -> tiny yaw.
    # Larger horizontal error / recovery -> more yaw.
    # This preserves the ability to follow nonlinear target turns without
    # allowing the recurring close-range 90-degree orbit behavior.
    agent1_adaptive_yaw_enabled: bool = True
    agent1_yaw_full_authority_when_lost: bool = True
    agent1_recovery_yaw_after_lost_norm: float = 0.28

    agent1_yaw_deadband_x: float = 0.04
    agent1_yaw_low_error_x: float = 0.12
    agent1_yaw_mid_error_x: float = 0.28

    agent1_yaw_max_centered: float = 0.025
    agent1_yaw_max_low_error: float = 0.080
    agent1_yaw_max_mid_error: float = 0.180
    agent1_yaw_max_high_error: float = 0.420

    # Close to handoff, yaw should still exist, but be conservative unless
    # horizontal error is very large.
    agent1_close_yaw_distance_m: float = 9.0
    agent1_close_yaw_scale: float = 0.45
    agent1_bottom_scan_yaw_scale: float = 0.35
    agent1_min_bottom_scan_yaw_max: float = 0.060

    agent1_yaw_limit_deadband: float = 0.010
    agent1_yaw_limit_penalty: float = 8.0

    # Backward-compatible names kept for older code paths/logs.
    agent1_heading_lock_enabled: bool = False
    agent1_zero_rl_yaw_while_visible: bool = False
    agent1_zero_rl_yaw_in_bottom_scan: bool = False
    agent1_allow_recovery_yaw_when_lost: bool = True
    agent1_locked_yaw_request_penalty: float = 8.0
    agent1_yaw_lock_deadband: float = 0.015

    # Soft yaw-command budget. This is not terminal.
    w_yaw_command_budget_penalty: float = 0.025
    yaw_command_budget_free_deg: float = 25.0
    yaw_command_budget_clip_deg: float = 120.0

    # Chase-pressure anti-camping reward:
    # If the target is visible and still far, focus alone is not enough. The
    # drone must reduce real distance or use forward motion when centered.
    chase_pressure_enabled: bool = True
    chase_pressure_bottom_disable: bool = True
    chase_pressure_goal_distance_m: float = 7.5
    chase_pressure_far_clip_m: float = 8.0
    chase_pressure_center_threshold: float = 0.28
    chase_pressure_no_improvement_deadband_m: float = 0.03
    chase_pressure_min_forward_action: float = 0.10
    w_visible_far_no_approach_penalty: float = 12.0
    w_centered_forward_chase_bonus: float = 22.0
    w_centered_idle_far_penalty: float = 18.0
    w_retreat_while_far_penalty: float = 16.0

    penalty_collision: float = 60.0
    penalty_timeout: float = 10.0
    penalty_altitude_termination: float = 20.0

    # -----------------------------
    # AirSim vehicle, cameras, LiDAR and fallback distance sensors
    # -----------------------------
    vehicle_name: str = "Drone1"

    # Camera names in AirSim/Cosys-AirSim settings.json.
    # Front camera is used for CHASE. Downward camera is prepared for
    # ALIGN_ABOVE_TARGET / LANDING. The landing policy is not switched on yet.
    front_camera_name: str = "front_center"
    downward_camera_name: str = "bottom_center"
    use_downward_camera: bool = True

    # Multi-camera soft handoff scaffold.
    # The front camera remains active while the bottom camera starts scanning
    # when the drone is close enough. The chase policy can finish successfully
    # with handoff_success once the bottom camera confirms the same target.
    bottom_handoff_enabled: bool = True
    bottom_scan_every_n_steps: int = 3
    bottom_scan_distance_m: float = 10.0
    bottom_handoff_distance_m: float = 4.0
    bottom_match_min_similarity: float = 0.52
    bottom_handoff_streak_required: int = 3
    bottom_confirmed_streak_required: int = 2
    bottom_handoff_confirm_similarity: float = 0.56
    bottom_active_min_weight: float = 0.45

    # End Agent-1 successfully once bottom alignment is stable.
    alignment_ready_terminal_enabled: bool = True
    alignment_ready_min_bottom_streak: int = 45
    # Handoff success is defined by secondary-camera confirmation, not by
    # real simulator distance. Distance can open scanning, but it should not
    # decide success.
    bottom_success_requires_fresh_match: bool = True
    bottom_success_fresh_max_steps: int = 2
    bottom_success_min_bbox_area: float = 0.004

    # Handoff success must mean the landing agent receives a usable state.
    # Detection anywhere in the downward camera is not enough.
    bottom_success_requires_centered: bool = True
    bottom_success_max_abs_err_x: float = 0.22
    bottom_success_max_abs_err_y: float = 0.28
    bottom_success_max_center_error: float = 0.34

    # Do not allow a sideways/orbit approach to be validated as handoff_success.
    # This uses commanded-yaw diagnostics because pose yaw may stay at 0.0.
    bottom_success_respects_command_yaw: bool = True
    bottom_success_max_abs_commanded_yaw_deg: float = 25.0
    bottom_success_max_cumulative_abs_yaw_cmd_deg: float = 95.0

    # Dual-camera full tracker architecture:
    # Both cameras own an independent visual tracker and an independent
    # TargetTrackerManager/Kalman stabilizer. The bottom camera is no longer a
    # lightweight handoff scanner.
    dual_full_trackers_enabled: bool = True
    bottom_full_tracker_enabled: bool = True
    bottom_full_tracker_update_every_n_steps: int = 1
    bottom_full_tracker_match_similarity_fallback: float = 0.78
    bottom_full_tracker_pred_similarity_fallback: float = 0.42
    bottom_full_tracker_lost_similarity_fallback: float = 0.0

    # Bottom tracker logic fix:
    # The bottom tracker must not depend only on one early auto-lock attempt.
    # If the bottom full tracker is LOST/no-bbox, run a YOLO+ResNet candidate
    # scan every step and use a confirmed candidate as a real bottom measurement.
    bottom_relock_every_step_when_lost: bool = True
    bottom_use_candidate_scan_fallback: bool = True
    bottom_candidate_scan_min_similarity: float = 0.46
    bottom_candidate_scan_min_bbox_area: float = 0.0015
    bottom_candidate_scan_class_gate: bool = True

    # Dual-camera tracker fusion:
    # Both front and bottom target perception run in parallel. The environment
    # decides which camera currently owns the control/reward authority.
    dual_camera_fusion_enabled: bool = True
    dual_camera_fusion_scan_every_n_steps: int = 1
    dual_bottom_authority_similarity: float = 0.52
    dual_bottom_authority_min_bbox_area: float = 0.003
    dual_bottom_authority_min_streak: int = 1
    dual_bottom_primary_min_streak: int = 2
    dual_bottom_keep_authority_grace_steps: int = 6
    dual_front_assist_after_bottom_lost_steps: int = 10

    # During bottom-primary alignment, the policy still controls the drone, but
    # a small action assist maps bottom-camera error into body-frame vx/vy. This
    # helps the policy learn that bottom_err_x/y are the real control space.
    dual_bottom_alignment_action_assist_enabled: bool = True
    dual_bottom_alignment_vy_gain: float = 0.90
    dual_bottom_alignment_vx_gain: float = 1.25
    dual_bottom_alignment_blend: float = 0.60
    dual_bottom_alignment_max_action: float = 0.95
    dual_bottom_err_x_to_vy_sign: float = 1.0

    # Important geometry fix:
    # In the downward camera, negative bottom_err_y means the target is high in
    # the image. The drone must move forward to get above it, so the Bcy->vx
    # sign is intentionally negative.
    dual_bottom_err_y_to_vx_sign: float = -1.0

    # Reward shaping for bottom-primary alignment.
    bottom_alignment_reward_enabled: bool = True
    bottom_alignment_y_target_abs: float = 0.12
    bottom_alignment_center_target: float = 0.18
    w_bottom_y_error_penalty: float = 38.0
    w_bottom_y_progress_reward: float = 24.0
    w_bottom_y_regress_penalty: float = 30.0
    w_bottom_center_ready_bonus: float = 18.0
    w_bottom_wrong_vx_penalty: float = 22.0
    w_bottom_correct_vx_bonus: float = 14.0
    bottom_alignment_progress_deadband: float = 0.015

    bottom_front_loss_grace_distance_m: float = 8.0

    # Visual + LiDAR handoff trigger:
    # Start bottom-camera scan if the front bbox visually indicates proximity
    # AND LiDAR sees a nearby body in the same horizontal direction as the bbox.
    handoff_use_visual_lidar_trigger: bool = True
    handoff_lidar_close_dist_m: float = 14.0
    handoff_lidar_x_deadband: float = 0.22
    handoff_visual_score_threshold: float = 2.5
    handoff_visual_scan_score_threshold: float = 2.5
    bottom_scan_hold_steps: int = 20
    handoff_bbox_area_threshold: float = 0.045
    handoff_bbox_height_threshold: float = 0.25
    handoff_bbox_bottom_threshold: float = 0.78
    handoff_bbox_err_y_threshold: float = 0.35
    handoff_bbox_area_delta_threshold: float = 0.08
    handoff_aspect_ratio_min: float = 0.65
    handoff_aspect_ratio_max: float = 3.20

    # Obstacle source:
    #   "lidar"    -> use 3D LiDAR first, optionally fallback to old sensors
    #   "distance" -> use old directional Distance* sensors only
    #   "none"     -> no obstacle readings, all distances=max range
    obstacle_sensor_source: str = "lidar"
    use_distance_sensor_fallback: bool = True

    lidar_sensor_name: str = "LidarSensor1"
    lidar_min_valid_range_m: float = 0.05
    lidar_horizontal_abs_z_max_m: float = 1.75
    lidar_down_xy_radius_m: float = 2.0
    lidar_down_min_z_m: float = 0.05

    # LiDAR calibration:
    # Cossy/AirSim LiDAR may return very short points from the drone body,
    # prop guards, sensor origin, or other self-geometry. These are not
    # external obstacles and must not drive horizontal obstacle safety.
    # Safety is still active for all points outside this self radius.
    lidar_self_ignore_radius_m: float = 0.75

    # In CHASE, altitude should come from AirSim pose/kinematics.
    # LiDAR-down remains available as a safety/landing signal, but should not
    # override altitude because raw LiDAR includes self/near-body returns.
    use_lidar_down_as_altitude: bool = False

    # Legacy fallback sensors. They are not the primary source anymore, but
    # keeping them lets you run old AirSim settings.json files safely.
    distance_sensor_front: str = "DistanceFront"
    distance_sensor_front_left: str = "DistanceFrontLeft"
    distance_sensor_front_right: str = "DistanceFrontRight"
    distance_sensor_left: str = "DistanceLeft"
    distance_sensor_right: str = "DistanceRight"
    distance_sensor_back: str = "DistanceBack"
    distance_sensor_down: str = "DistanceDown"

    lidar_max_dist_m: float = 20.0

    safety_enabled: bool = True
    obstacle_safe_dist_m: float = 3.0
    obstacle_warning_dist_m: float = 2.0
    obstacle_emergency_dist_m: float = 0.75
    obstacle_emergency_termination_dist_m: float = 0.50

    min_speed_scale_near_obstacle: float = 0.2
    obstacle_steer_strength: float = 0.35
    emergency_up_cmd: float = 0.8

    # -----------------------------
    # State-aware safety architecture
    # -----------------------------
    # Safety is never globally disabled.  The same LiDAR points are interpreted
    # according to the current mission phase:
    #   TAKEOFF: ground below is expected; hold horizontal motion and climb.
    #   CHASE: all horizontal/down safety rules are active.
    #   LANDING: horizontal safety stays active; down proximity is expected and
    #            limits descent instead of forcing emergency climb.
    safety_takeoff_clear_altitude_m: float = 1.20
    safety_takeoff_clear_steps: int = 0
    safety_takeoff_max_steps: int = 120
    chase_rules_min_steps_after_takeoff: int = 8
    safety_takeoff_zero_horizontal_motion: bool = True
    safety_takeoff_force_climb: bool = False
    force_chase_after_airborne_reset: bool = True
    altitude_force_up_in_chase: bool = False
    safety_landing_allow_down_proximity: bool = True
    safety_landing_max_descent_action_near_ground: float = 0.18
    safety_landing_distance_m: float = 2.50

    # -----------------------------
    # Success criteria
    # -----------------------------
    stable_follow_success_sec: float = 5.0
    success_centered_score_min: float = 0.75
    success_distance_proxy_min: float = 0.88
    success_distance_proxy_max: float = 0.98
    success_collision_risk_max: float = 0.3
