"""Independent Agent-2 landing environment.

This module intentionally does not subclass or mutate ``DroneEnv``. Agent 1
keeps its original configuration, initialized reward objects, safety objects,
tracker memory and control pipeline. Agent 2 owns a separate control loop that
uses:

* bottom camera only;
* immutable user-selected visual identity;
* horizontal LiDAR sectors only;
* AirSim API Z during normal visual flight;
* calibrated five-beam range height during final approach and a controlled terminal exploration window;
* AirSim collision API as the touchdown signal;
* a collision-gated landing reward bank.
"""

from __future__ import annotations

import builtins
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
import threading
from typing import Any, Callable, Optional

import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn.functional as F
import cosysairsim as airsim

from lidar_processor import LidarProcessor, LidarProcessorConfig, point_cloud_to_array
from observation_builder import (
    BBox,
    DroneState,
    ObstacleState,
    ObservationBuilder,
    ObservationBuilderConfig,
)
from resnet_yolo_tracker import YoloResNetTracker
from range_finder_array import RangeFinderArray, RangeFinderArrayConfig, SENSOR_NAMES


def _landing_console_print(text: str) -> None:
    """Print only the final landing-decision block, bypassing quiet runtime mode."""
    printer = getattr(builtins, "_rl_original_print", builtins.print)
    printer(text, flush=True)



@dataclass
class Agent2Config:
    vehicle_name: str = "Drone1"
    bottom_camera_name: str = "bottom_center"
    lidar_sensor_name: str = "LidarSensor1"
    range_sensor_names: tuple[str, ...] = SENSOR_NAMES
    range_min_distance_m: float = 0.05
    range_max_distance_m: float = 20.0
    range_min_valid_count: int = 4
    range_safe_spread_m: float = 0.25
    range_sensor_final_enabled: bool = True
    range_sensor_final_entry_m: float = 1.50
    range_sensor_final_stop_m: float = 0.35
    range_terminal_contact_vz_mps: float = 0.18
    range_terminal_contact_max_duration_s: float = 7.00  # Legacy compatibility only.
    range_terminal_contact_max_steps: int = 50
    # Low-altitude Z is range/controller governed. Agent 2 may request a climb
    # only when current geometry provides a physical justification.
    range_terminal_climb_max_vz_mps: float = 0.35
    range_terminal_climb_center_error_threshold: float = 0.25
    range_terminal_climb_spread_threshold_m: float = 0.25
    # Pre-arm above the 0.35 m handoff floor. AirSim control/perception steps
    # can cover more than the old 0.25 m arming band at 0.65 m/s, so a safe
    # snapshot is latched up to 1.00 m and refreshed while descending. The
    # actual one-way terminal handoff still occurs only at/below 0.35 m or
    # when the already-armed range geometry becomes invalid.
    range_terminal_contact_arm_max_height_m: float = 1.00
    range_terminal_contact_latch_max_age_s: float = 3.00
    range_sensor_final_recent_vision_s: float = 2.00
    range_sensor_final_max_center_error_m: float = 0.25
    range_sensor_final_min_similarity: float = 0.65
    range_require_calibration: bool = True
    image_width: int = 960
    image_height: int = 720

    # Optional scalar-only capture for offline Kalman replay. Disabled by
    # default so normal training does not recreate historical diagnostic files.
    kalman_diagnostic_logging_enabled: bool = False
    kalman_diagnostic_output_dir: str = "outputs/kalman_live_diagnostics"

    cmd_duration_s: float = 0.25
    # Agent-2-only first-contact monitor. The monitor polls AirSim while the
    # fused Agent-1/Agent-2 command is executing and cancels that command on
    # the first new collision, before the drone can bounce or slide.
    collision_poll_interval_s: float = 0.010
    collision_stop_duration_s: float = 0.050
    vx_scale_mps: float = 1.20
    vy_scale_mps: float = 1.20
    vz_scale_mps: float = 0.65
    # Forced-contact diagnostic mode: whenever the bottom camera has a LIVE
    # MATCH, Agent 2 commands a deterministic positive NED-Z descent until
    # first collision. This intentionally bypasses alignment/lock/PPO Z gates
    # so the contact-monitor path can be tested under guaranteed impact.
    force_descent_while_bottom_match: bool = False
    forced_bottom_match_descent_vz_mps: float = 0.65
    yaw_scale_dps: float = 25.0
    max_episode_steps: int = 900

    # YOLO class is used only during the first click so a bbox can be obtained.
    # After that, the selected instance is internally named ``user_target`` and
    # every YOLO candidate is judged only by ResNet visual similarity.
    strict_class_gate: bool = False
    yolo_proposal_conf: float = 0.05
    # ResNet remains the identity judge. These gates only decide whether a
    # visual match is reliable enough to be treated as a LIVE control input.
    min_match_similarity: float = 0.60
    min_match_margin: float = 0.03
    high_conf_reacquire_similarity: float = 0.78
    max_reacquire_center_jump_norm: float = 0.35
    match_confirmation_steps: int = 3
    max_prediction_steps: int = 20
    show_camera: bool = True

    # Vertical state machine. AirSim NED uses positive vz for DOWN.
    # Agent 2 is a landing-only controller: policy commands may request hover
    # or descent, but never climb. Descent is additionally blocked unless the
    # target is detected in the current frame, confirmed over consecutive
    # frames and sufficiently aligned.
    descent_min_live_match_streak: int = 3
    descent_min_similarity: float = 0.65

    # Horizontal visual-servo controller. PPO does not directly command the
    # full XY velocity anymore; it learns only a small residual around a
    # deterministic bottom-camera PD controller.
    horizontal_pd_kp_y_to_vx: float = 1.10
    horizontal_pd_kd_y_to_vx: float = 0.16
    horizontal_pd_kp_x_to_vy: float = 0.82
    horizontal_pd_kd_x_to_vy: float = 0.12
    horizontal_pd_max_action: float = 0.90
    horizontal_ppo_residual_max_action: float = 0.12
    horizontal_deadband_error: float = 0.025
    horizontal_deadband_velocity: float = 0.08
    horizontal_velocity_ema_alpha: float = 0.55
    horizontal_velocity_clip_per_s: float = 4.0

    # Moving-target velocity feed-forward is estimated from the bottom camera,
    # not from the target actor API. The estimator fuses bbox-center motion with
    # sparse optical flow inside the verified target ROI, converts the relative
    # image motion to metric body-frame motion using camera geometry and height,
    # then adds the drone's own measured ego velocity. This keeps the controller
    # compatible with a real vehicle and does not change PPO observation/action
    # spaces.
    target_velocity_feedforward_enabled: bool = True
    target_velocity_feedforward_gain: float = 1.0
    target_velocity_ema_alpha: float = 0.45
    target_velocity_max_valid_mps: float = 8.0
    target_velocity_stale_after_s: float = 0.65
    target_velocity_max_accel_mps2: float = 7.0
    bottom_camera_hfov_deg: float = 90.0
    visual_motion_min_dt_s: float = 0.03
    # A full dual-model observation can legitimately take 1-3 seconds. Accept
    # that real measurement interval for velocity division, but keep the KF and
    # acceleration update bounded separately.
    visual_motion_max_dt_s: float = 0.80
    visual_motion_measurement_max_dt_s: float = 3.00
    visual_motion_kalman_dt_max_s: float = 0.80
    visual_motion_attitude_gate_deg: float = 12.0
    visual_motion_max_yaw_rate_dps: float = 45.0
    visual_motion_yaw_compensation: bool = True
    visual_motion_prediction_max_age_s: float = 0.30
    visual_relative_velocity_ema_alpha: float = 0.55
    # Constant-velocity Kalman filter over metric body-frame state
    # [relative_x, relative_y, relative_vx, relative_vy]. It filters the noisy
    # bbox/optical-flow measurement without querying target actor XY state.
    visual_kalman_enabled: bool = True
    visual_kalman_process_accel_std_mps2: float = 2.0
    visual_kalman_position_std_m: float = 0.08
    visual_kalman_velocity_std_bbox_mps: float = 0.70
    visual_kalman_velocity_std_flow_mps: float = 0.35
    optical_flow_enabled: bool = True
    optical_flow_max_corners: int = 90
    optical_flow_quality_level: float = 0.01
    optical_flow_min_distance_px: float = 7.0
    optical_flow_window_px: int = 21
    optical_flow_max_level: int = 3
    optical_flow_min_points: int = 6
    optical_flow_max_forward_backward_error_px: float = 1.75
    optical_flow_bbox_expand: float = 1.25
    optical_flow_max_norm_velocity_per_s: float = 4.0
    optical_flow_max_bbox_disagreement_per_s: float = 1.25
    optical_flow_weight: float = 0.65
    horizontal_total_speed_max_mps: float = 6.0
    horizontal_velocity_hold_max_s: float = 0.65

    # Speed shrinks near touchdown. The final limit also considers bbox area,
    # so a wrong actor-Z estimate cannot make close-range commands aggressive.
    horizontal_speed_far_mps: float = 0.90
    horizontal_speed_mid_mps: float = 0.60
    horizontal_speed_near_mps: float = 0.35
    # Preserve enough center-correction authority during the final metre. The
    # target-velocity feed-forward still carries platform speed; this value is
    # only the bounded visual correction budget on top of it.
    horizontal_speed_touchdown_mps: float = 0.30

    # Predictive bottom-camera guidance. The primary controller operates in
    # metric body-frame coordinates: relative target position and relative
    # target velocity are propagated to t+1, and the controller corrects toward
    # that future point. Normalized-image control remains only as a fallback.
    predictive_bottom_enabled: bool = True
    predictive_bottom_extra_latency_s: float = 0.08
    predictive_bottom_horizon_min_s: float = 0.10
    predictive_bottom_horizon_max_s: float = 0.60
    predictive_metric_kp_position_per_s: float = 1.65
    predictive_metric_kd_relative_velocity: float = 0.85
    predictive_metric_catchup_enter_position_m: float = 0.55
    predictive_metric_catchup_exit_position_m: float = 0.20
    predictive_metric_catchup_enter_relative_speed_mps: float = 0.22
    predictive_metric_catchup_exit_relative_speed_mps: float = 0.09
    predictive_bottom_kp_y_to_vx_mps: float = 2.20
    predictive_bottom_kd_y_to_vx_mps: float = 0.40
    predictive_bottom_kp_x_to_vy_mps: float = 1.80
    predictive_bottom_kd_x_to_vy_mps: float = 0.35
    predictive_bottom_normal_correction_max_mps: float = 1.50
    predictive_bottom_catchup_correction_max_mps: float = 3.20
    predictive_bottom_touchdown_catchup_max_mps: float = 2.40
    predictive_bottom_catchup_enter_center_error: float = 0.24
    predictive_bottom_catchup_exit_center_error: float = 0.13
    predictive_bottom_catchup_enter_outward_speed_per_s: float = 0.10
    predictive_bottom_catchup_exit_outward_speed_per_s: float = 0.03
    predictive_bottom_catchup_exit_image_speed_per_s: float = 0.16
    predictive_bottom_catchup_release_streak: int = 2

    # A short PRED gap can continue using the visual motion estimator, but Z is
    # always held. Once the bottom view is genuinely lost, Agent 1 regains XY
    # and a deterministic, bounded climb restores field of view. The PPO policy
    # still cannot request arbitrary climb; negative NED-Z is reserved for this
    # explicit reacquisition state only.
    bottom_pred_guidance_max_age_s: float = 0.25
    reacquire_climb_enabled: bool = True
    reacquire_climb_after_s: float = 0.45
    reacquire_climb_speed_mps: float = 0.45
    reacquire_climb_target_height_m: float = 3.20
    reacquire_climb_max_duration_s: float = 4.0
    reacquire_climb_min_last_bbox_area_norm: float = 0.0

    # Landing lock. Initial acquisition is intentionally faster than the old
    # 3+5-frame handoff gate: two aligned LIVE frames are enough. Once acquired,
    # a short detector gap pauses Z but does not erase the lock, so reacquisition
    # resumes descent immediately instead of rebuilding the full streak.
    alignment_enter_center_error: float = 0.30
    alignment_enter_bbox_rel_error: float = 0.55
    alignment_exit_center_error: float = 0.42
    alignment_exit_bbox_rel_error: float = 0.72
    alignment_streak_required: int = 2
    landing_lock_max_visual_gap_steps: int = 3
    landing_lock_bad_live_release_steps: int = 3

    # BBOX-relative alignment becomes increasingly meaningful near contact,
    # when the vehicle fills a larger fraction of the image. At high altitude,
    # dividing image-center error by a narrow bbox creates a circular deadlock:
    # the drone must descend to enlarge the bbox, but descent is blocked because
    # that same relative error is too large. Use image-center geometry above
    # this height and re-enable the footprint-relative gate near touchdown.
    landing_bbox_rel_gate_below_height_m: float = 1.50
    catchup_descent_enter_center_error: float = 0.24
    catchup_descent_exit_center_error: float = 0.34

    # Control-space bbox stabilization. Identity selection still uses every raw
    # YOLO/ResNet proposal, but landing geometry and motion control consume a
    # robust median + EMA bbox. This prevents a single detector shape/center
    # jump from flipping XY commands or resetting the landing lock.
    control_bbox_filter_enabled: bool = True
    control_bbox_history_size: int = 5
    control_bbox_center_alpha: float = 0.78
    control_bbox_edge_center_alpha: float = 0.95
    control_bbox_size_alpha: float = 0.35
    control_bbox_outlier_alpha: float = 0.10
    control_bbox_outlier_center_jump_norm: float = 0.14
    control_bbox_outlier_size_ratio: float = 0.40
    control_bbox_edge_margin_px: float = 6.0

    # Stable semantic landing point. The detector bbox continues to serve
    # identity and optical-flow ROI selection, while XY control, landing lock
    # and touchdown validation use a virtual full-target anchor. When the car is
    # clipped by an image edge, its hidden extent is reconstructed from the last
    # reliable full-target aspect/size instead of treating the visible crop
    # center as the physical roof center.
    landing_anchor_enabled: bool = True
    landing_anchor_u: float = 0.50
    landing_anchor_v: float = 0.50
    landing_anchor_size_ema_alpha: float = 0.45
    landing_anchor_aspect_ema_alpha: float = 0.20
    landing_anchor_center_alpha: float = 0.90
    landing_anchor_edge_center_alpha: float = 0.98
    landing_anchor_max_outside_frame_ratio: float = 0.75

    # Terminal landing anchor. A reliable roof point is acquired while the full
    # target is still visible. Near touchdown, this point is propagated by local
    # forward/backward optical flow and is no longer replaced by a new detector
    # BBox center. This prevents a seat/window/edge crop from becoming the XY
    # landing reference when the car fills the bottom camera.
    terminal_anchor_enabled: bool = True
    terminal_anchor_lock_min_height_m: float = 0.20
    terminal_anchor_lock_max_height_m: float = 2.20
    terminal_anchor_lock_min_similarity: float = 0.70
    terminal_anchor_lock_max_center_error: float = 0.34
    terminal_anchor_lock_min_area_norm: float = 0.010
    terminal_anchor_lock_max_area_norm: float = 0.55
    terminal_anchor_min_live_streak: int = 3
    terminal_anchor_acquire_streak_required: int = 2
    terminal_anchor_edge_margin_px: float = 12.0
    terminal_anchor_candidate_max_jump_norm: float = 0.060
    terminal_anchor_flow_patch_radius_ratio: float = 0.30
    terminal_anchor_flow_patch_min_px: int = 40
    terminal_anchor_flow_patch_max_px: int = 150
    terminal_anchor_flow_max_corners: int = 80
    terminal_anchor_flow_quality_level: float = 0.010
    terminal_anchor_flow_min_distance_px: float = 5.0
    terminal_anchor_flow_window_px: int = 21
    terminal_anchor_flow_max_level: int = 3
    terminal_anchor_flow_min_points: int = 6
    terminal_anchor_flow_max_forward_backward_error_px: float = 1.50
    terminal_anchor_flow_max_step_norm: float = 0.12
    terminal_anchor_live_correction_alpha: float = 0.12
    terminal_anchor_live_correction_max_jump_norm: float = 0.080
    terminal_anchor_max_hold_steps: int = 3
    terminal_anchor_identity_grace_steps: int = 10
    terminal_anchor_freeze_adaptive_bank: bool = True

    # Controlled descent during horizontal catch-up. Catch-up is not itself a
    # vertical hazard: when the target is a confirmed LIVE match, remains well
    # inside the bottom image, and its predicted image motion is not diverging,
    # Agent 2 may acquire the landing lock and descend slowly. A genuinely
    # escaping target, PRED-only guidance, low identity confidence or excessive
    # image/metric outward motion still blocks Z immediately.
    catchup_descent_enabled: bool = True
    catchup_descent_max_vz_mps: float = 0.40
    # Near contact the old 0.18 m/s cap made the drone hover long enough for a
    # moving roof to escape. Keep a bounded but decisive final descent while
    # LIVE identity and landing-lock geometry are still valid.
    catchup_descent_touchdown_max_vz_mps: float = 0.32
    catchup_descent_touchdown_height_m: float = 1.00
    catchup_descent_max_predicted_center_error: float = 0.34
    catchup_descent_max_image_speed_per_s: float = 0.22
    catchup_descent_max_outward_speed_per_s: float = 0.06
    catchup_descent_max_metric_outward_speed_mps: float = 1.50

    # Adaptive multi-scale landing appearance memory. The two immutable identity
    # anchors are never replaced. The adaptive half-bank is refreshed in-place
    # with recent verified LIVE views; guarded immutable agreement and spatial
    # continuity prevent PRED/self-reinforcement drift.
    adaptive_embedding_enabled: bool = True
    adaptive_embedding_max_entries: int = 8
    adaptive_embedding_update_interval_steps: int = 4
    adaptive_embedding_identity_floor: float = 0.52
    adaptive_embedding_add_min_identity_similarity: float = 0.62
    adaptive_embedding_add_min_combined_similarity: float = 0.68
    adaptive_embedding_novelty_max_similarity: float = 0.997
    adaptive_embedding_bbox_scales: tuple[float, ...] = (1.00,)
    adaptive_embedding_max_additions_per_update: int = 1
    # Add a new appearance only after a cumulative target-scale change of at
    # least 25% relative to the last snapshot accepted into the bank.
    adaptive_embedding_min_scale_change_ratio: float = 0.15
    adaptive_embedding_min_live_streak: int = 3
    adaptive_embedding_max_center_jump_norm: float = 0.18
    adaptive_embedding_max_area_ratio_change: float = 2.20
    adaptive_embedding_force_refresh_age_steps: int = 16
    adaptive_embedding_chain_min_combined_similarity: float = 0.82
    adaptive_embedding_chain_min_margin: float = 0.04
    adaptive_embedding_chain_max_spatial_jump_norm: float = 0.10
    adaptive_embedding_freeze_below_height_m: float = 1.50

    lidar_max_range_m: float = 20.0
    obstacle_emergency_m: float = 0.55
    obstacle_warning_m: float = 1.20

    # Touchdown reward classification uses only evidence from the collision
    # frame. A target collision is successful when either a centered LIVE
    # bottom-camera match exists, or the current bottom image itself looks like
    # the selected target directly under the camera. Historical latches remain
    # diagnostic only and cannot grant terminal reward.
    good_collision_center_error: float = 0.32
    good_collision_bbox_rel_error: float = 0.58
    good_collision_recent_match_steps: int = 4

    # Touchdown alignment is projected from the calibrated raw LIVE BBox into
    # physical metres using target-relative height, camera HFOV and frame aspect.
    collision_xy_threshold_m: float = 0.45
    touchdown_legacy_window_steps: int = 12
    collision_geometric_max_measurement_age_steps: int = 120
    collision_geometric_min_height_m: float = 0.05
    collision_geometric_max_height_m: float = 8.0
    collision_latch_max_age_steps: int = 2
    # Touchdown success has exactly three gates:
    # 1) a physical contact event, 2) semantic ``user_target`` identity from a
    # fresh neural MATCH, valid terminal continuity, or contact-frame neural
    # appearance, and 3) calibrated metric center error within the threshold.
    # Simulator actor names and YOLO class labels are diagnostic only.
    collision_latch_center_error: float = 0.15  # normalized controller diagnostic only
    collision_latch_center_error_m: float = 0.25
    collision_latch_bbox_rel_error: float = 0.15  # quality/log compatibility only
    collision_latch_min_similarity: float = 0.65  # quality/log compatibility only

    # Terminal semantic-drift fallback. This does not alter the controller or
    # the primary BEST touchdown path. It is evaluated only when physical
    # contact and semantic identity pass but the final metric center sample
    # is missing or above threshold.
    terminal_fallback_enabled: bool = True
    terminal_fallback_recent_samples: int = 5
    terminal_fallback_min_samples: int = 5
    terminal_fallback_center_median_m: float = 0.20
    terminal_fallback_mean_relative_speed_mps: float = 0.40
    terminal_fallback_max_relative_speed_mps: float = 0.80
    terminal_fallback_min_velocity_samples: int = 3
    terminal_fallback_max_bbox_rel_error: float = 0.35
    terminal_fallback_min_similarity: float = 0.65

    # Experimental stale-identity bridge. A confirmed semantic sample may remain
    # eligible for up to eight steps only when the recent terminal trajectory is
    # substantially stronger than the normal fallback requirements.
    terminal_fallback_identity_bridge_enabled: bool = True
    terminal_fallback_identity_bridge_max_age_steps: int = 8
    terminal_fallback_identity_bridge_min_similarity: float = 0.70
    terminal_fallback_identity_bridge_max_bbox_rel_error: float = 0.35
    terminal_fallback_identity_bridge_center_median_m: float = 0.18
    terminal_fallback_identity_bridge_mean_relative_speed_mps: float = 0.35
    terminal_fallback_identity_bridge_max_relative_speed_mps: float = 0.70

    # Last-valid center latch. This terminal semantic-drift fallback is
    # refreshed only by a confirmed LIVE BBox whose
    # appearance and geometry are reliable. When the close-range image becomes
    # distorted, the last trustworthy per-frame metric center error is held.
    terminal_last_valid_center_enabled: bool = True
    terminal_last_valid_center_min_similarity: float = 0.70
    terminal_last_valid_center_max_bbox_rel_error: float = 0.35
    terminal_last_valid_center_max_age_steps: int = 8  # Compatibility diagnostic.
    terminal_last_valid_center_max_age_s: float = 3.50
    terminal_last_valid_center_error_m: float = 0.25

    # Kalman terminal bridge. Once deterministic landing descent has
    # started from a trustworthy LIVE match, a close-range visual rejection
    # switches to a bounded predict-only Kalman state. Z continues and XY uses
    # the predicted relative state instead of a distorted BBox.
    terminal_blind_descent_enabled: bool = True
    terminal_blind_descent_max_height_m: float = 1.50
    terminal_blind_descent_max_duration_s: float = 3.00
    terminal_kalman_enabled: bool = True
    terminal_kalman_max_position_std_m: float = 0.60
    terminal_kalman_rpc_center_error_m: float = 0.25

    # Terminal landing-quality reward. These terms change reward magnitude only
    # and are deliberately excluded from the binary touchdown success gate.
    landing_quality_center_bonus: float = 500.0
    landing_quality_bbox_rel_bonus: float = 300.0
    landing_quality_similarity_bonus: float = 200.0
    landing_quality_bbox_rel_reference: float = 0.60
    landing_quality_similarity_reference: float = 0.65

    # Direct contact-frame appearance test. Several centered crops are compared
    # with the immutable bottom-view target anchor. Corner crops act as a
    # current-frame floor/background control. A high absolute similarity or a
    # clear center-over-corners margin is required.
    contact_appearance_center_crop_scales: tuple[float, ...] = (0.35, 0.50, 0.70, 0.90, 1.00)
    contact_appearance_corner_crop_scale: float = 0.35
    contact_appearance_min_similarity: float = 0.60
    contact_appearance_strong_similarity: float = 0.72
    contact_appearance_min_center_margin: float = 0.03

    # Authorized-descent state is retained only for diagnostics. It no longer
    # participates in touchdown success classification.
    authorized_descent_min_vz_mps: float = 1.0e-4
    authorized_descent_extreme_live_center_error: float = 0.55

    # Legacy simulator collision-name settings are retained only for diagnostics
    # and RPC plumbing. They never grant or veto touchdown success. A real system
    # replaces the generic contact event with a physical contact sensor while
    # semantic identity remains entirely vision-derived.
    collision_object_auto_lock: bool = True
    collision_ground_tokens: tuple[str, ...] = ("floor", "ground", "landscape", "terrain")

    success_base_reward: float = 2500.0
    success_min_reward: float = 800.0
    success_max_reward: float = 6000.0
    wrong_collision_penalty: float = 1500.0

    # Successful touchdown visualization. The RPC is called only after a new
    # target collision has already passed the strict XY/contact success gate.
    # PPO receives the terminal success reward, the drone is snapped to the
    # roof anchor, the result is held briefly, and the wrapper then resets the
    # next episode normally.
    latch_on_success: bool = True
    # The verified standalone RPC test uses AirSim's default vehicle key.
    latch_vehicle_name: str = ""
    latch_target_actor_name: str = "BP_X6M_C_1"
    latch_anchor_component_name: str = "DroneLandingAnchor"
    latch_success_hold_seconds: float = 5.0

    # Dense Agent-2 reward. Agent 2 owns only Z in AGENT_1P2, therefore it is
    # rewarded only for real, aligned height progress and is not paid merely for
    # horizontal centering that belongs to the deterministic/Agent-1 controller.
    # This removes the old incentive to hover until timeout while preserving a
    # dominant terminal touchdown objective.
    dense_reward_enabled: bool = True
    dense_reward_alignment_center_error: float = 0.34
    dense_reward_descent_progress_per_m: float = 30.0
    dense_reward_near_touch_height_m: float = 1.50
    dense_reward_near_touch_multiplier: float = 1.50
    dense_reward_max_progress_m_per_step: float = 1.00
    dense_reward_landing_lock_time_penalty: float = 0.35
    dense_reward_hesitation_penalty: float = 1.50
    dense_reward_min_descent_action_when_aligned: float = 0.25
    dense_reward_unsafe_descent_penalty: float = 0.75
    timeout_penalty: float = 600.0
    target_lost_penalty: float = 300.0

    target_lost_limit_steps: int = 120

    # In AGENT_1P2, a sustained absence of a LIVE bottom-camera match requests
    # a same-episode return to Agent 1. A short grace window handles detector
    # gaps; near contact, direct bottom-image appearance can suppress a false
    # recovery request when the target fills the frame.
    recovery_no_live_match_timeout_s: float = 1.0
    recovery_no_live_match_min_steps: int = 2
    recovery_contact_guard_max_height_m: float = 0.75
    # Contact appearance may postpone recovery only briefly. It must never
    # create a low-altitude hover deadlock after the target has really escaped.
    recovery_contact_guard_max_no_live_s: float = 1.50
    # In AGENT_1P2 v12, Agent 2 never owns XY/Yaw and never requests a
    # recovery handoff. It continuously owns only Z while Agent 1 remains
    # active on both cameras.
    parallel_dual_agent_mode: bool = False

    static_target_actor_name: str = ""
    static_target_surface_altitude_m: float = 0.0


class Agent2LandingEnv(gym.Env):
    """Bottom-camera landing controller used by AGENT_2 and AGENT_1P2."""

    metadata = {"render_modes": []}

    # Process-wide cache: the calibration JSON is read at most once for each
    # absolute file path, even if Stable-Baselines creates more than one env.
    _bottom_center_calibration_cache: dict[str, tuple[float, float, str]] = {}

    @classmethod
    def _load_bottom_center_calibration_once(
        cls,
        calibration_path: Path,
    ) -> tuple[float, float, str]:
        absolute_path = str(calibration_path.resolve())
        cached = cls._bottom_center_calibration_cache.get(absolute_path)
        if cached is not None:
            return cached

        target_x = 0.5
        target_y = 0.5
        source = "DEFAULT_IMAGE_CENTER"
        try:
            payload = json.loads(calibration_path.read_text(encoding="utf-8"))
            target = payload.get("target", {})
            target_x = float(target["bbox_center_x_norm"])
            target_y = float(target["bbox_center_y_norm"])
            if not (0.0 <= target_x <= 1.0 and 0.0 <= target_y <= 1.0):
                raise ValueError(
                    f"calibrated target must be normalized to [0, 1], got "
                    f"({target_x}, {target_y})"
                )
            source = absolute_path
        except FileNotFoundError:
            print(
                "[BOTTOM CENTER CALIBRATION] JSON not found; using image center "
                f"(0.500000, 0.500000) | path={absolute_path}"
            )
        except Exception as exc:
            print(
                "[BOTTOM CENTER CALIBRATION] Invalid JSON; using image center "
                f"(0.500000, 0.500000) | path={absolute_path} error={exc}"
            )

        result = (float(target_x), float(target_y), str(source))
        cls._bottom_center_calibration_cache[absolute_path] = result
        return result

    def __init__(self, cfg: Agent2Config | None = None):
        super().__init__()
        self.cfg = cfg or Agent2Config()

        self._bottom_center_calibration_path = (
            Path(__file__).resolve().parent
            / "config"
            / "bottom_bbox_center_calibration.json"
        )
        (
            self._bottom_center_target_x_norm,
            self._bottom_center_target_y_norm,
            self._bottom_center_calibration_source,
        ) = self._load_bottom_center_calibration_once(
            self._bottom_center_calibration_path
        )
        print(
            "[BOTTOM CENTER CALIBRATION] loaded once | "
            f"target=({self._bottom_center_target_x_norm:.6f},"
            f"{self._bottom_center_target_y_norm:.6f}) "
            f"source={self._bottom_center_calibration_source}"
        )

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(46,), dtype=np.float32)

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.observation_builder = ObservationBuilder(
            ObservationBuilderConfig(
                image_width=self.cfg.image_width,
                image_height=self.cfg.image_height,
                max_altitude_m=30.0,
                max_drone_speed_mps=8.0,
                max_vertical_speed_mps=5.0,
                max_obstacle_range_m=self.cfg.lidar_max_range_m,
                safe_obstacle_distance_m=5.0,
                max_lost_target_time_s=3.0,
            )
        )
        self.lidar_processor = LidarProcessor(
            LidarProcessorConfig(max_range_m=self.cfg.lidar_max_range_m)
        )
        self.range_finder_array = RangeFinderArray(
            RangeFinderArrayConfig(
                min_distance_m=float(self.cfg.range_min_distance_m),
                max_distance_m=float(self.cfg.range_max_distance_m),
                min_valid_count=int(self.cfg.range_min_valid_count),
                safe_spread_m=float(self.cfg.range_safe_spread_m),
                sensor_final_entry_m=float(self.cfg.range_sensor_final_entry_m),
                sensor_final_stop_m=float(self.cfg.range_sensor_final_stop_m),
            )
        )
        if bool(self.cfg.range_require_calibration) and not self.range_finder_array.calibration_loaded:
            raise RuntimeError(
                "A PASS range-finder calibration is required for Agent 2. "
                "Run calibrate_and_test_range_finders.py and verify "
                "config/range_finder_calibration.json before training."
            )
        print(
            "[RANGE CALIBRATION] loaded="
            f"{int(self.range_finder_array.calibration_loaded)} "
            f"path={self.range_finder_array.config.calibration_path} "
            f"bias={self.range_finder_array.sensor_bias_m}"
        )
        self.tracker = YoloResNetTracker(
            yolo_model_path="yolo11s.pt",
            device="cuda" if torch.cuda.is_available() else "cpu",
            target_classes=None,
            yolo_conf=float(self.cfg.yolo_proposal_conf),
            click_pad=20,
            min_match_score=0.45,
            appearance_weight=1.0,
            motion_weight=0.0,
            search_window_scale=4.0,
            use_search_window=False,
            ema_alpha=1.0,
            verbose=False,
        )

        self._original_embedding: Optional[torch.Tensor] = None
        self._bottom_anchor_embedding: Optional[torch.Tensor] = None
        # Immutable identity anchors are kept in _reference_embeddings.
        # _adaptive_embeddings contains only recent, verified landing views.
        self._reference_embeddings: list[torch.Tensor] = []
        self._adaptive_embeddings: list[torch.Tensor] = []
        self._adaptive_embedding_steps: list[int] = []
        self._adaptive_embedding_scales: list[float] = []
        self._last_adaptive_embedding_update_step = -999999
        self._last_adaptive_bbox_xywh = None
        self._adaptive_embedding_updates = 0
        self._adaptive_bank_replace_index = 0
        self._adaptive_bank_cycle = 0
        self._adaptive_bank_last_action = "EMPTY"
        self._target_id = "user_target"
        self._target_class_id: Optional[int] = None
        self._last_bbox_xyxy: Optional[np.ndarray] = None
        self._control_bbox_xyxy: Optional[np.ndarray] = None
        self._control_bbox_history: list[np.ndarray] = []
        self._control_bbox_outlier_suppressed = False
        self._control_bbox_filter_mode = "INIT"
        self._landing_anchor_px: Optional[np.ndarray] = None
        self._landing_anchor_virtual_bbox_xyxy: Optional[np.ndarray] = None
        self._landing_anchor_full_size_px: Optional[np.ndarray] = None
        self._landing_anchor_aspect_ratio: Optional[float] = None
        self._landing_anchor_reference_size_px: Optional[np.ndarray] = None
        self._landing_anchor_reference_height_m: Optional[float] = None
        self._landing_anchor_mode = "INIT"
        self._landing_anchor_edge_flags = "NONE"

        # Terminal anchor state is independent of the detector BBox. Once locked,
        # it follows the same physical roof point with optical flow until reset.
        self._terminal_anchor_locked = False
        self._terminal_anchor_px: Optional[np.ndarray] = None
        self._terminal_anchor_previous_gray: Optional[np.ndarray] = None
        self._terminal_anchor_support_bbox_xyxy: Optional[np.ndarray] = None
        self._terminal_anchor_candidate_px: Optional[np.ndarray] = None
        self._terminal_anchor_acquire_streak = 0
        self._terminal_anchor_lock_step = -999999
        self._terminal_anchor_last_identity_step = -999999
        self._terminal_anchor_age_steps = 999999
        self._terminal_anchor_source = "INACTIVE"
        self._terminal_anchor_flow_points = 0
        self._terminal_anchor_flow_confidence = 0.0
        self._last_bottom_observation_monotonic = 0.0
        self._last_similarity = 0.0
        self._last_candidate_class_id: Optional[int] = None
        self._last_candidate_confidence = 0.0
        self._last_candidate_count = 0
        self._last_candidate_scores: list[dict[str, float | int]] = []
        self._prediction_steps = 0
        self._last_match_step = -999999
        self._live_match_streak = 0
        self._last_match_margin = 0.0
        self._last_spatial_jump_norm = 0.0
        self._last_match_reject_reason = ""

        self._standalone_initial_pose = None
        self._attached_from_agent1 = False
        self._target_actor_name = str(self.cfg.static_target_actor_name or "")
        # Keep target and drone in the same raw AirSim NED-Z coordinate
        # system. Height above target is target_z_ned - drone_z_ned.
        self._target_surface_z_ned = -float(self.cfg.static_target_surface_altitude_m)
        self._target_surface_altitude_m = float(self.cfg.static_target_surface_altitude_m)
        self._target_surface_source = "configured_static"

        # Scalar-only live diagnostic session. No frames or arrays are retained.
        self._kalman_diag_session_started = time.strftime("%Y%m%d_%H%M%S")
        self._kalman_diag_start_monotonic = float(time.monotonic())
        self._kalman_diag_last_monotonic: Optional[float] = None
        self._kalman_diag_csv_path: Optional[Path] = None
        self._kalman_diag_header_written = False
        self._diag_raw_metric_x_m = float("nan")
        self._diag_raw_metric_y_m = float("nan")
        self._diag_external_vx_mps = float("nan")
        self._diag_external_vy_mps = float("nan")
        self._diag_velocity_measurement_valid = False
        self._init_kalman_diagnostic_logger()

        self._step = 0
        self._reward_bank = 0.0
        self._episode_return = 0.0
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._prev_relative_height_m: Optional[float] = None
        self._lost_steps = 0
        self._collision_timestamp_at_reset = 0

        # Strict visual-alignment latch used only for touchdown classification.
        # It is reset every episode; the expected collision object may persist
        # after being safely auto-locked from a verified target touchdown.
        self._last_verified_alignment_step = -999999
        self._last_verified_alignment_center_error = 999.0
        self._last_verified_alignment_bbox_rel_error = 999.0
        self._last_verified_alignment_similarity = 0.0
        self._last_verified_alignment_err_x = 999.0
        self._last_verified_alignment_err_y = 999.0
        self._last_verified_alignment_height_m = float("inf")

        # Physical descent authorization evidence. Unlike the visual latch,
        # this is refreshed only when a non-zero descent command is actually
        # sent after passing the full vertical safety gate.
        self._authorized_descent_latched = False
        self._last_authorized_descent_step = -999999
        self._last_authorized_descent_center_error = 999.0
        self._last_authorized_descent_bbox_rel_error = 999.0
        self._last_authorized_descent_similarity = 0.0
        self._last_authorized_descent_err_x = 999.0
        self._last_authorized_descent_err_y = 999.0
        self._last_authorized_descent_height_m = float("inf")
        self._last_authorized_descent_vz_mps = 0.0
        self._authorized_descent_invalidated_reason = "never_authorized"

        self._expected_collision_object_name = str(
            self._target_actor_name or self.cfg.latch_target_actor_name or ""
        )
        self._expected_collision_object_source = "configured_target_actor"
        self._touchdown_legacy_window: list[dict[str, Any]] = []
        self._touchdown_ready_snapshot: dict[str, Any] | None = None
        self._range_terminal_contact_snapshot: dict[str, Any] | None = None
        self._range_terminal_handoff_timed_out = False
        self._last_valid_center_error_m = float("inf")
        self._last_valid_center_step = -999999
        self._last_valid_center_monotonic = float("-inf")
        self._last_valid_center_similarity = 0.0
        self._last_valid_center_bbox_rel_error = float("inf")
        self._last_valid_center_source = "NONE"
        self._terminal_kalman_state: Optional[np.ndarray] = None
        self._terminal_kalman_covariance = np.eye(4, dtype=np.float64)
        self._terminal_kalman_monotonic = float("-inf")
        self._terminal_descent_committed = False
        self._terminal_blind_descent_started_monotonic = float("-inf")

        # Most recent raw bottom-camera frame. It is used only when AirSim
        # reports a new collision, so contact appearance adds no per-step
        # inference cost.
        self._last_bottom_frame: Optional[np.ndarray] = None

        self._last_info: dict[str, Any] = {}
        self._last_vertical_control_state = "HOLD_INIT"
        self._last_descent_block_reason = "waiting_for_first_control_step"
        self._last_raw_vz_action = 0.0
        self._last_requested_vz_mps = 0.0
        self._last_applied_vz_mps = 0.0
        self._last_vertical_speed_limit_mps = 0.0
        self._last_soft_catchup_descent_active = False
        self._last_climb_command_blocked = False

        self._last_observation_monotonic: Optional[float] = None
        self._last_control_had_live_match = False
        self._last_control_err_x = 0.0
        self._last_control_err_y = 0.0
        self._control_img_vel_x = 0.0
        self._control_img_vel_y = 0.0
        self._last_control_dt_s = float(self.cfg.cmd_duration_s)
        self._predictive_catchup_active = False
        self._predictive_catchup_release_streak = 0
        self._last_predictive_guidance: dict[str, Any] = {
            "live_match": False,
            "catchup_active": False,
            "predicted_err_x": 0.0,
            "predicted_err_y": 0.0,
            "predicted_center_error": 999.0,
            "image_velocity_x_per_s": 0.0,
            "image_velocity_y_per_s": 0.0,
            "image_speed_per_s": 0.0,
            "outward_speed_per_s": 0.0,
            "prediction_horizon_s": float(self.cfg.cmd_duration_s),
            "correction_vx_mps": 0.0,
            "correction_vy_mps": 0.0,
            "correction_limit_mps": 0.0,
        }
        # Visual motion estimator. No target XY position or velocity is read
        # from the simulator API. The only target-motion measurements are the
        # verified bottom-camera ROI and optical flow inside that ROI.
        self._bottom_camera_hfov_deg = float(self.cfg.bottom_camera_hfov_deg)
        self._previous_motion_gray: Optional[np.ndarray] = None
        self._previous_motion_bbox_xyxy: Optional[np.ndarray] = None
        self._previous_motion_time: Optional[float] = None
        self._previous_relative_position_body: Optional[tuple[float, float]] = None
        self._previous_relative_height_m: Optional[float] = None
        self._previous_motion_yaw_rad: Optional[float] = None
        self._visual_motion_attitude_valid = True
        self._visual_motion_yaw_rate_dps = 0.0
        self._visual_relative_position_body_x_m = 0.0
        self._visual_relative_position_body_y_m = 0.0
        self._visual_relative_velocity_body_x_mps = 0.0
        self._visual_relative_velocity_body_y_mps = 0.0
        self._visual_relative_velocity_valid = False
        self._visual_kalman_state: Optional[np.ndarray] = None
        self._visual_kalman_covariance = np.eye(4, dtype=np.float64)
        self._visual_kalman_velocity_initialized = False
        self._visual_motion_age_s = float("inf")
        self._visual_motion_source = "WAIT"
        self._visual_bbox_velocity_x_per_s = 0.0
        self._visual_bbox_velocity_y_per_s = 0.0
        self._visual_flow_velocity_x_per_s = 0.0
        self._visual_flow_velocity_y_per_s = 0.0
        self._visual_flow_point_count = 0
        self._visual_flow_confidence = 0.0
        self._drone_body_velocity_x_mps = 0.0
        self._drone_body_velocity_y_mps = 0.0
        self._target_velocity_world_x_mps = 0.0
        self._target_velocity_world_y_mps = 0.0
        self._target_velocity_body_vx_mps = 0.0
        self._target_velocity_body_vy_mps = 0.0
        self._target_velocity_speed_mps = 0.0
        self._target_velocity_valid = False
        self._target_velocity_age_s = float("inf")
        self._reacquire_climb_started_monotonic: Optional[float] = None
        self._reacquire_climb_active = False
        self._non_live_started_monotonic: Optional[float] = None
        self._non_live_duration_s = 0.0
        self._alignment_ready_streak = 0
        self._descent_alignment_latched = False
        self._landing_lock_visual_gap_steps = 0
        self._landing_lock_bad_live_steps = 0
        self._landing_lock_acquired_step = -999999
        self._last_horizontal_control_state = "HOLD_INIT"
        self._last_pd_action_vx = 0.0
        self._last_pd_action_vy = 0.0
        self._last_residual_action_vx = 0.0
        self._last_residual_action_vy = 0.0
        self._last_horizontal_action_vx = 0.0
        self._last_horizontal_action_vy = 0.0
        self._last_horizontal_speed_limit_mps = 0.0
        self._last_velocity_ff_vx_mps = 0.0
        self._last_velocity_ff_vy_mps = 0.0
        self._last_horizontal_correction_vx_mps = 0.0
        self._last_horizontal_correction_vy_mps = 0.0
        self._last_horizontal_command_vx_mps = 0.0
        self._last_horizontal_command_vy_mps = 0.0
        self._last_horizontal_total_speed_limit_mps = 0.0
        self._episode_recenter_steps = 0
        self._episode_xy_hold_steps = 0

        self._episode_live_match_steps = 0
        self._episode_predicted_steps = 0
        self._episode_no_target_steps = 0
        self._episode_descent_requested_steps = 0
        self._episode_descent_allowed_steps = 0
        self._episode_descent_blocked_steps = 0
        self._episode_climb_command_blocked_steps = 0
        self._episode_best_center_error = float("inf")
        self._episode_best_similarity = 0.0
        self._last_dense_reward = 0.0
        self._last_dense_reward_parts: dict[str, float] = {
            "aligned_descent_progress": 0.0,
            "landing_lock_time": 0.0,
            "hesitation": 0.0,
            "unsafe_descent": 0.0,
        }

        # Optional single-command executor supplied by Agent1P2Env. When set,
        # Agent 2 computes only the gated Z command; the executor runs frozen
        # Agent 1 and transmits the fused XY/Yaw/Z command exactly once.
        self._external_command_executor: Optional[
            Callable[[float], dict[str, Any]]
        ] = None
        # Agent 2 remains the only owner of collision polling. This callback
        # performs motion cancellation only, on the command-owning Agent-1
        # environment/client, after Agent 2 detects first contact.
        self._external_motion_stop_callback: Optional[Callable[[str], None]] = None
        self._last_external_command_info: dict[str, Any] = {}

        # Collision ownership is deliberately restricted to Agent 2 during the
        # parallel landing phase. A dedicated AirSim client polls independently
        # while the blocking fused command is in flight.
        self._collision_monitor_client: Optional[Any] = None
        self._collision_monitor_lock = threading.Lock()
        self._collision_monitor_latched: Optional[tuple[str, int]] = None

    def set_external_command_executor(
        self,
        executor: Optional[Callable[[float], dict[str, Any]]],
    ) -> None:
        self._external_command_executor = executor

    def set_external_motion_stop_callback(
        self,
        callback: Optional[Callable[[str], None]],
    ) -> None:
        """Register a cancellation callback; it must never poll collision."""
        self._external_motion_stop_callback = callback

    def get_target_velocity_feedforward_body(self) -> tuple[float, float, bool]:
        """Return visually estimated target velocity in drone body axes.

        The estimate is built from bottom-camera relative motion plus the
        drone's own measured ego velocity. Target actor XY pose/velocity is not
        used anywhere in this path.
        """
        valid = bool(
            self.cfg.target_velocity_feedforward_enabled
            and self._target_velocity_valid
            and self._target_velocity_age_s
            <= float(self.cfg.target_velocity_stale_after_s)
        )
        if not valid:
            return 0.0, 0.0, False
        gain = float(self.cfg.target_velocity_feedforward_gain)
        return (
            float(getattr(self, "_target_velocity_body_vx_mps", 0.0)) * gain,
            float(getattr(self, "_target_velocity_body_vy_mps", 0.0)) * gain,
            True,
        )

    @staticmethod
    def _clip_vector(vx: float, vy: float, limit: float) -> tuple[float, float]:
        """Clip a 2-D vector without changing its direction."""
        magnitude = float(math.hypot(vx, vy))
        safe_limit = max(0.0, float(limit))
        if magnitude <= max(1.0e-9, safe_limit):
            return float(vx), float(vy)
        scale = float(safe_limit / magnitude)
        return float(vx * scale), float(vy * scale)

    def _compute_predictive_bottom_guidance(
        self,
        info: dict[str, Any],
        *,
        update_hysteresis: bool,
    ) -> dict[str, Any]:
        """Predict target relative position at t+1 from bottom-camera motion.

        The primary state is metric body-frame relative position/velocity from
        the visual estimator. A short PRED gap may continue this state without
        updating it. Normalized bbox motion is retained only as a fallback.
        """
        live_match = bool(info.get("bottom_match_live", False))
        tracker_mode = str(info.get("tracker_mode", ""))
        visual_age = float(info.get("visual_motion_age_s", float("inf")))
        terminal_kalman_prediction = bool(
            info.get("terminal_kalman_prediction_active", False)
        )
        prediction_only = bool(
            not live_match
            and (tracker_mode.startswith("PRED") or terminal_kalman_prediction)
            and visual_age <= (
                float(self.cfg.terminal_blind_descent_max_duration_s)
                if terminal_kalman_prediction
                else float(self.cfg.bottom_pred_guidance_max_age_s)
            )
            and bool(info.get("visual_relative_velocity_valid", False))
        )
        guidance_active = bool(
            bool(self.cfg.predictive_bottom_enabled)
            and (live_match or prediction_only)
        )
        center_error = float(info.get("bottom_center_error", 999.0))
        area = float(info.get("bottom_bbox_area_norm", 0.0))
        height = float(info.get("relative_height_to_target_m", float("inf")))
        landing_lock = bool(self._descent_alignment_latched)

        if not guidance_active:
            if update_hysteresis:
                self._predictive_catchup_active = False
                self._predictive_catchup_release_streak = 0
            result = {
                "live_match": False,
                "measurement_live": False,
                "guidance_active": False,
                "prediction_only": False,
                "landing_lock": landing_lock,
                "catchup_active": False,
                "bottom_correction_vx_mps": 0.0,
                "bottom_correction_vy_mps": 0.0,
                "bottom_blend_strength": 0.0,
                "center_error": center_error,
                "bbox_area_norm": area,
                "relative_height_m": height,
                "predicted_err_x": 0.0,
                "predicted_err_y": 0.0,
                "predicted_center_error": 999.0,
                "relative_position_x_m": 0.0,
                "relative_position_y_m": 0.0,
                "relative_velocity_x_mps": 0.0,
                "relative_velocity_y_mps": 0.0,
                "predicted_relative_x_m": 0.0,
                "predicted_relative_y_m": 0.0,
                "predicted_relative_distance_m": 999.0,
                "relative_speed_mps": 0.0,
                "image_velocity_x_per_s": 0.0,
                "image_velocity_y_per_s": 0.0,
                "image_speed_per_s": 0.0,
                "outward_speed_per_s": 0.0,
                "metric_outward_speed_mps": 0.0,
                "prediction_horizon_s": float(self.cfg.cmd_duration_s),
                "correction_limit_mps": 0.0,
                "controller_source": "NONE",
            }
            if update_hysteresis:
                self._last_predictive_guidance = dict(result)
            return result

        err_x = float(info.get("bottom_err_x", 0.0) or 0.0)
        err_y = float(info.get("bottom_err_y", 0.0) or 0.0)
        vel_x = float(
            info.get(
                "bottom_img_vel_x_control",
                getattr(self, "_control_img_vel_x", 0.0),
            )
            or 0.0
        )
        vel_y = float(
            info.get(
                "bottom_img_vel_y_control",
                getattr(self, "_control_img_vel_y", 0.0),
            )
            or 0.0
        )
        measured_dt = float(
            info.get(
                "bottom_control_dt_s",
                getattr(self, "_last_control_dt_s", self.cfg.cmd_duration_s),
            )
            or self.cfg.cmd_duration_s
        )
        horizon = float(
            np.clip(
                max(float(self.cfg.cmd_duration_s), measured_dt)
                + float(self.cfg.predictive_bottom_extra_latency_s),
                float(self.cfg.predictive_bottom_horizon_min_s),
                float(self.cfg.predictive_bottom_horizon_max_s),
            )
        )

        # During a short PRED interval, advance the last measured state to now
        # before predicting the next command horizon.
        prediction_age = visual_age if prediction_only else 0.0
        predicted_err_x = float(
            np.clip(err_x + vel_x * (prediction_age + horizon), -1.50, 1.50)
        )
        predicted_err_y = float(
            np.clip(err_y + vel_y * (prediction_age + horizon), -1.50, 1.50)
        )
        predicted_center = float(math.hypot(predicted_err_x, predicted_err_y))
        image_speed = float(math.hypot(vel_x, vel_y))
        current_center = float(math.hypot(err_x, err_y))
        if current_center > 1.0e-6:
            outward_speed = float((err_x * vel_x + err_y * vel_y) / current_center)
        else:
            outward_speed = 0.0

        metric_valid = bool(
            info.get("visual_relative_velocity_valid", False)
            and np.isfinite(float(info.get("visual_relative_position_body_x_m", 0.0)))
            and np.isfinite(float(info.get("visual_relative_position_body_y_m", 0.0)))
        )
        relative_x = float(info.get("visual_relative_position_body_x_m", 0.0) or 0.0)
        relative_y = float(info.get("visual_relative_position_body_y_m", 0.0) or 0.0)
        relative_vx = float(info.get("visual_relative_velocity_body_x_mps", 0.0) or 0.0)
        relative_vy = float(info.get("visual_relative_velocity_body_y_mps", 0.0) or 0.0)
        if prediction_only and metric_valid:
            relative_x += relative_vx * prediction_age
            relative_y += relative_vy * prediction_age
        predicted_relative_x = float(relative_x + relative_vx * horizon)
        predicted_relative_y = float(relative_y + relative_vy * horizon)
        predicted_relative_distance = float(
            math.hypot(predicted_relative_x, predicted_relative_y)
        )
        relative_speed = float(math.hypot(relative_vx, relative_vy))
        current_metric_distance = float(math.hypot(relative_x, relative_y))
        if current_metric_distance > 1.0e-6:
            metric_outward_speed = float(
                (relative_x * relative_vx + relative_y * relative_vy)
                / current_metric_distance
            )
        else:
            metric_outward_speed = 0.0

        metric_enter = bool(
            metric_valid
            and (
                predicted_relative_distance
                >= float(self.cfg.predictive_metric_catchup_enter_position_m)
                or metric_outward_speed
                >= float(self.cfg.predictive_metric_catchup_enter_relative_speed_mps)
            )
        )
        metric_exit = bool(
            metric_valid
            and predicted_relative_distance
            <= float(self.cfg.predictive_metric_catchup_exit_position_m)
            and abs(metric_outward_speed)
            <= float(self.cfg.predictive_metric_catchup_exit_relative_speed_mps)
            and relative_speed
            <= float(self.cfg.predictive_metric_catchup_enter_relative_speed_mps)
        )
        image_enter = bool(
            predicted_center
            >= float(self.cfg.predictive_bottom_catchup_enter_center_error)
            or outward_speed
            >= float(self.cfg.predictive_bottom_catchup_enter_outward_speed_per_s)
        )
        image_exit = bool(
            predicted_center
            <= float(self.cfg.predictive_bottom_catchup_exit_center_error)
            and outward_speed
            <= float(self.cfg.predictive_bottom_catchup_exit_outward_speed_per_s)
            and image_speed
            <= float(self.cfg.predictive_bottom_catchup_exit_image_speed_per_s)
        )
        enter_catchup = bool(metric_enter or image_enter or prediction_only)
        exit_catchup = bool((metric_exit if metric_valid else True) and image_exit)

        catchup_active = bool(getattr(self, "_predictive_catchup_active", False))
        release_streak = int(getattr(self, "_predictive_catchup_release_streak", 0))
        if update_hysteresis:
            if catchup_active:
                if exit_catchup and live_match:
                    release_streak += 1
                    if release_streak >= max(
                        1, int(self.cfg.predictive_bottom_catchup_release_streak)
                    ):
                        catchup_active = False
                        release_streak = 0
                else:
                    release_streak = 0
            elif enter_catchup:
                catchup_active = True
                release_streak = 0
            self._predictive_catchup_active = bool(catchup_active)
            self._predictive_catchup_release_streak = int(release_streak)
        else:
            catchup_active = bool(getattr(self, "_predictive_catchup_active", False) or enter_catchup)

        if metric_valid:
            correction_vx = float(
                float(self.cfg.predictive_metric_kp_position_per_s)
                * predicted_relative_x
                + float(self.cfg.predictive_metric_kd_relative_velocity)
                * relative_vx
            )
            correction_vy = float(
                float(self.cfg.predictive_metric_kp_position_per_s)
                * predicted_relative_y
                + float(self.cfg.predictive_metric_kd_relative_velocity)
                * relative_vy
            )
            controller_source = "METRIC_BOTTOM_VISION"
        else:
            correction_vx = float(
                -float(self.cfg.predictive_bottom_kp_y_to_vx_mps) * predicted_err_y
                -float(self.cfg.predictive_bottom_kd_y_to_vx_mps) * vel_y
            )
            correction_vy = float(
                float(self.cfg.predictive_bottom_kp_x_to_vy_mps) * predicted_err_x
                +float(self.cfg.predictive_bottom_kd_x_to_vy_mps) * vel_x
            )
            controller_source = "NORMALIZED_IMAGE_FALLBACK"

        if catchup_active:
            correction_limit = float(
                self.cfg.predictive_bottom_catchup_correction_max_mps
            )
            if np.isfinite(height) and height <= 0.80:
                correction_limit = min(
                    correction_limit,
                    float(self.cfg.predictive_bottom_touchdown_catchup_max_mps),
                )
        else:
            correction_limit = float(
                self.cfg.predictive_bottom_normal_correction_max_mps
            )

        correction_vx, correction_vy = self._clip_vector(
            correction_vx,
            correction_vy,
            correction_limit,
        )

        result = {
            "live_match": bool(live_match),
            "measurement_live": bool(live_match),
            "guidance_active": True,
            "prediction_only": bool(prediction_only),
            "landing_lock": landing_lock,
            "catchup_active": bool(catchup_active),
            "bottom_correction_vx_mps": float(correction_vx),
            "bottom_correction_vy_mps": float(correction_vy),
            "bottom_blend_strength": 1.0 if live_match else 0.65,
            "center_error": center_error,
            "bbox_area_norm": area,
            "relative_height_m": height,
            "predicted_err_x": float(predicted_err_x),
            "predicted_err_y": float(predicted_err_y),
            "predicted_center_error": float(predicted_center),
            "relative_position_x_m": float(relative_x),
            "relative_position_y_m": float(relative_y),
            "relative_velocity_x_mps": float(relative_vx),
            "relative_velocity_y_mps": float(relative_vy),
            "predicted_relative_x_m": float(predicted_relative_x),
            "predicted_relative_y_m": float(predicted_relative_y),
            "predicted_relative_distance_m": float(predicted_relative_distance),
            "relative_speed_mps": float(relative_speed),
            "image_velocity_x_per_s": float(vel_x),
            "image_velocity_y_per_s": float(vel_y),
            "image_speed_per_s": float(image_speed),
            "outward_speed_per_s": float(outward_speed),
            "metric_outward_speed_mps": float(metric_outward_speed),
            "prediction_horizon_s": float(horizon),
            "correction_limit_mps": float(correction_limit),
            "controller_source": controller_source,
        }
        if update_hysteresis:
            self._last_predictive_guidance = dict(result)
        return result

    def get_parallel_bottom_perception_snapshot(self) -> dict[str, Any]:
        """Export the current bottom result for Agent 1 without new inference."""
        info = dict(self._last_info or {})
        bbox = getattr(self, "_control_bbox_xyxy", None)
        if bbox is None:
            bbox = getattr(self, "_last_bbox_xyxy", None)
        if bbox is not None:
            try:
                bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4].copy()
                if bbox.size < 4 or not np.all(np.isfinite(bbox)):
                    bbox = None
            except Exception:
                bbox = None

        live = bool(info.get("bottom_match_live", False))
        recent = bool(info.get("bottom_match_recent", False))
        tracker_mode = str(info.get("tracker_mode", "") or "").upper()
        if live:
            mode = "MATCH"
        elif recent or tracker_mode.startswith("PRED"):
            mode = "PRED"
        else:
            mode = "LOST"

        anchor_px = info.get("bottom_landing_anchor_px", None)
        if anchor_px is not None:
            try:
                anchor_px = np.asarray(anchor_px, dtype=np.float32).reshape(-1)[:2].copy()
                if anchor_px.size < 2 or not np.all(np.isfinite(anchor_px)):
                    anchor_px = None
            except Exception:
                anchor_px = None

        # Agent 1's debug window displays this exact shared frame. Draw the
        # semantic point here so no Agent-1 runtime file needs to change: green
        # remains the visible bbox diagnostic, magenta is the actual XY/landing
        # control anchor.
        shared_frame = (
            None if self._last_bottom_frame is None else self._last_bottom_frame.copy()
        )
        if shared_frame is not None and anchor_px is not None:
            fh, fw = shared_frame.shape[:2]
            ax = int(round(float(anchor_px[0])))
            ay = int(round(float(anchor_px[1])))
            marker = cv2.MARKER_CROSS
            if not (0 <= ax < fw and 0 <= ay < fh):
                ax = int(np.clip(ax, 0, max(0, fw - 1)))
                ay = int(np.clip(ay, 0, max(0, fh - 1)))
                marker = cv2.MARKER_TILTED_CROSS
            cv2.drawMarker(
                shared_frame,
                (ax, ay),
                (255, 0, 255),
                marker,
                30,
                3,
            )

        return {
            "source": "AGENT2_SHARED_BOTTOM",
            "source_monotonic": float(
                getattr(self, "_last_bottom_observation_monotonic", 0.0)
                or time.monotonic()
            ),
            "frame_bgr": shared_frame,
            "match": live,
            "confirmed": bool(info.get("bottom_match_confirmed", False)),
            "recent": recent,
            "mode": mode,
            "raw_mode": tracker_mode or "AGENT2",
            "bbox_xyxy": bbox,
            "landing_anchor_px": (
                None if anchor_px is None else [float(anchor_px[0]), float(anchor_px[1])]
            ),
            "landing_anchor_mode": str(
                info.get("bottom_landing_anchor_mode", "NONE") or "NONE"
            ),
            "similarity": float(info.get("bottom_similarity", 0.0) or 0.0),
            "err_x": float(info.get("bottom_err_x", 0.0) or 0.0),
            "err_y": float(info.get("bottom_err_y", 0.0) or 0.0),
            "bbox_area_norm": float(info.get("bottom_bbox_area_norm", 0.0) or 0.0),
            "bbox_rel_err": float(info.get("bottom_bbox_rel_err", 999.0) or 999.0),
            "bbox_rel_err_x": float(info.get("bottom_bbox_rel_err_x", info.get("bottom_bbox_rel_err", 999.0)) or 999.0),
            "bbox_rel_err_y": float(info.get("bottom_bbox_rel_err_y", info.get("bottom_bbox_rel_err", 999.0)) or 999.0),
            "live_match_streak": int(info.get("bottom_live_match_streak", 0) or 0),
            "tracker_confidence": float(info.get("bottom_similarity", 0.0) or 0.0),
            "candidate_count": int(info.get("yolo_candidate_count", 0) or 0),
        }

    def get_parallel_bottom_guidance(self) -> dict[str, Any]:
        """Return the predictive bottom-camera correction cached this step."""
        if self._last_predictive_guidance:
            result = dict(self._last_predictive_guidance)
            result["landing_lock"] = bool(self._descent_alignment_latched)
            return result
        return self._compute_predictive_bottom_guidance(
            dict(self._last_info or {}),
            update_hysteresis=False,
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    def _set_identity(self, fingerprint: Any, class_id: Any = None) -> None:
        arr = np.asarray(fingerprint, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            raise ValueError("Agent 2 received an empty target fingerprint.")

        emb = torch.from_numpy(arr).float().to(self.tracker.device)
        emb = F.normalize(emb, dim=0)
        self._original_embedding = emb.detach().clone()
        self._bottom_anchor_embedding = None
        self._reference_embeddings = [self._original_embedding.detach().clone()]
        self._adaptive_embeddings = []
        self._adaptive_embedding_steps = []
        self._adaptive_embedding_scales = []
        self._last_adaptive_embedding_update_step = -999999
        self._last_adaptive_bbox_xywh = None
        self._adaptive_embedding_updates = 0
        self._adaptive_bank_replace_index = 0
        self._adaptive_bank_cycle = 0
        self._adaptive_bank_last_action = "EMPTY"
        self._target_class_id = None if class_id is None else int(class_id)

        # The internal tracker is used only as a detector/embedding backend.
        # target_class_id is kept only for diagnostics; it is never a gate.
        self.tracker.target_embedding = self._original_embedding.detach().clone()
        self.tracker.target_class_id = self._target_class_id
        self.tracker.last_bbox = None
        self.tracker.last_good_bbox = None
        self.tracker.last_score = 0.0
        self.tracker.last_mode = "IDLE"

    def _set_bottom_anchor_from_bbox(self, frame: np.ndarray, bbox_xyxy: Any) -> bool:
        """Create an immutable bottom-view reference for the same user target."""
        arr = self._validated_xyxy(bbox_xyxy, frame.shape)
        if arr is None:
            return False

        # ResNet crop coordinates must be integer pixel indices. Agent 1 stores
        # handoff boxes as float32, so normalize them at this ownership boundary.
        bbox_xywh = self._xyxy_to_xywh(arr)
        emb = self.tracker._embedding_from_bbox(frame, bbox_xywh)
        if emb is None:
            return False

        self._bottom_anchor_embedding = emb.detach().clone()
        self._reference_embeddings = [self._original_embedding.detach().clone()]
        self._reference_embeddings.append(self._bottom_anchor_embedding.detach().clone())
        return True

    def _all_reference_embeddings(self) -> list[torch.Tensor]:
        """Return immutable identity anchors followed by verified adaptive views."""
        references = [embedding for embedding in self._reference_embeddings]
        references.extend(getattr(self, "_adaptive_embeddings", []))
        return references

    @staticmethod
    def _scaled_bbox_xywh(
        bbox_xywh: Any,
        scale: float,
        frame_shape: tuple[int, ...],
    ) -> Optional[list[int]]:
        """Scale an XYWH box around its center and clamp it to the image."""
        try:
            x, y, w, h = [float(value) for value in bbox_xywh[:4]]
        except Exception:
            return None
        image_h, image_w = frame_shape[:2]
        scale = max(0.25, float(scale))
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        new_w = max(2.0, w * scale)
        new_h = max(2.0, h * scale)
        x1 = max(0.0, cx - 0.5 * new_w)
        y1 = max(0.0, cy - 0.5 * new_h)
        x2 = min(float(image_w), cx + 0.5 * new_w)
        y2 = min(float(image_h), cy + 0.5 * new_h)
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            return None
        return [
            int(round(x1)),
            int(round(y1)),
            max(1, int(round(x2 - x1))),
            max(1, int(round(y2 - y1))),
        ]

    def _append_adaptive_embedding(
        self,
        embedding: torch.Tensor,
        bbox_scale: float,
    ) -> bool:
        """Append until full, then refresh the left half and right half in order.

        Immutable anchors live in ``_reference_embeddings`` and are never touched.
        The adaptive bank is chronological: after filling N entries, slots
        0..N/2-1 are replaced by newer verified views, then slots N/2..N-1.
        This keeps landing appearances current instead of preserving stale scales.
        """
        if embedding is None or not np.isfinite(float(bbox_scale)):
            return False
        candidate = F.normalize(embedding.detach().clone(), dim=0)
        if not hasattr(self, "_adaptive_embeddings"):
            self._adaptive_embeddings = []
        if not hasattr(self, "_adaptive_embedding_steps"):
            self._adaptive_embedding_steps = []
        if not hasattr(self, "_adaptive_embedding_scales"):
            self._adaptive_embedding_scales = []

        max_entries = max(1, int(self.cfg.adaptive_embedding_max_entries))
        step = int(getattr(self, "_step", 0))
        if len(self._adaptive_embeddings) < max_entries:
            self._adaptive_embeddings.append(candidate)
            self._adaptive_embedding_steps.append(step)
            self._adaptive_embedding_scales.append(float(bbox_scale))
            self._adaptive_bank_last_action = "APPEND"
            return True

        replace_index = int(getattr(self, "_adaptive_bank_replace_index", 0)) % max_entries
        half = max(1, max_entries // 2)
        self._adaptive_embeddings[replace_index] = candidate
        self._adaptive_embedding_steps[replace_index] = step
        self._adaptive_embedding_scales[replace_index] = float(bbox_scale)
        self._adaptive_bank_last_action = (
            "REPLACE_LEFT" if replace_index < half else "REPLACE_RIGHT"
        )
        next_index = (replace_index + 1) % max_entries
        self._adaptive_bank_replace_index = next_index
        if next_index == 0:
            self._adaptive_bank_cycle = int(
                getattr(self, "_adaptive_bank_cycle", 0)
            ) + 1
        return True

    def _maybe_update_adaptive_embedding_bank(
        self,
        frame: np.ndarray,
        bbox_xywh: Any,
        immutable_similarity: float,
        combined_similarity: float,
        matched_adaptive_anchor: bool,
    ) -> int:
        """Refresh the adaptive bank with recent, strongly verified LIVE views."""
        if not bool(self.cfg.adaptive_embedding_enabled):
            return 0
        if bool(getattr(self.cfg, "terminal_anchor_freeze_adaptive_bank", True)) and bool(
            getattr(self, "_terminal_anchor_locked", False)
        ):
            return 0
        last_height = float(
            dict(getattr(self, "_last_info", {}) or {}).get(
                "relative_height_to_target_m", float("inf")
            )
        )
        if (
            np.isfinite(last_height)
            and last_height
            <= float(getattr(self.cfg, "adaptive_embedding_freeze_below_height_m", 1.50))
        ):
            return 0
        step = int(getattr(self, "_step", 0))
        interval = max(1, int(self.cfg.adaptive_embedding_update_interval_steps))
        if step - int(getattr(self, "_last_adaptive_embedding_update_step", -999999)) < interval:
            return 0
        if int(getattr(self, "_live_match_streak", 0)) < max(2, int(self.cfg.adaptive_embedding_min_live_streak)):
            return 0
        if combined_similarity < float(self.cfg.adaptive_embedding_add_min_combined_similarity):
            return 0

        immutable_trusted = bool(
            immutable_similarity >= float(self.cfg.adaptive_embedding_add_min_identity_similarity)
        )
        guarded_adaptive_chain = bool(
            matched_adaptive_anchor
            and immutable_similarity >= float(self.cfg.adaptive_embedding_identity_floor)
            and combined_similarity >= float(self.cfg.adaptive_embedding_chain_min_combined_similarity)
            and float(getattr(self, "_last_match_margin", 0.0))
            >= float(self.cfg.adaptive_embedding_chain_min_margin)
            and float(getattr(self, "_last_spatial_jump_norm", 999.0))
            <= float(self.cfg.adaptive_embedding_chain_max_spatial_jump_norm)
        )
        if not (immutable_trusted or guarded_adaptive_chain):
            return 0

        current_bbox = np.asarray(bbox_xywh, dtype=np.float32).reshape(-1)[:4]
        if current_bbox.size < 4 or not np.all(np.isfinite(current_bbox)):
            return 0
        if current_bbox[2] < 2.0 or current_bbox[3] < 2.0:
            return 0
        current_area = max(1.0, float(current_bbox[2] * current_bbox[3]))

        previous_bbox = getattr(self, "_last_adaptive_bbox_xywh", None)
        if previous_bbox is not None:
            previous_bbox = np.asarray(previous_bbox, dtype=np.float32).reshape(-1)[:4]
            previous_area = max(1.0, float(previous_bbox[2] * previous_bbox[3]))
            area_ratio = max(current_area / previous_area, previous_area / current_area)
            if area_ratio > float(self.cfg.adaptive_embedding_max_area_ratio_change):
                return 0
            image_h, image_w = frame.shape[:2]
            current_center = np.asarray(
                [current_bbox[0] + 0.5 * current_bbox[2], current_bbox[1] + 0.5 * current_bbox[3]],
                dtype=np.float32,
            )
            previous_center = np.asarray(
                [previous_bbox[0] + 0.5 * previous_bbox[2], previous_bbox[1] + 0.5 * previous_bbox[3]],
                dtype=np.float32,
            )
            center_jump = float(
                np.linalg.norm(current_center - previous_center)
                / max(1.0, float(np.hypot(image_w, image_h)))
            )
            if center_jump > float(self.cfg.adaptive_embedding_max_center_jump_norm):
                return 0

        embedding = self.tracker._embedding_from_bbox(frame, current_bbox)
        if embedding is None:
            return 0
        immutable_scores = [
            float(torch.dot(anchor, embedding).detach().cpu().item())
            for anchor in self._reference_embeddings
        ]
        crop_immutable_similarity = max(immutable_scores, default=-1.0)
        crop_trusted = bool(
            crop_immutable_similarity >= float(self.cfg.adaptive_embedding_add_min_identity_similarity)
            or (guarded_adaptive_chain and crop_immutable_similarity >= float(self.cfg.adaptive_embedding_identity_floor))
        )
        if not crop_trusted:
            return 0

        bank = list(getattr(self, "_adaptive_embeddings", []))
        candidate = F.normalize(embedding.detach().clone(), dim=0)
        max_bank_similarity = max(
            (float(torch.dot(existing, candidate).detach().cpu().item()) for existing in bank),
            default=-1.0,
        )
        oldest_age = max(
            (step - int(saved_step) for saved_step in getattr(self, "_adaptive_embedding_steps", [])),
            default=999999,
        )
        too_similar = bool(
            max_bank_similarity > float(self.cfg.adaptive_embedding_novelty_max_similarity)
            and oldest_age < int(self.cfg.adaptive_embedding_force_refresh_age_steps)
        )
        if too_similar:
            return 0

        image_h, image_w = frame.shape[:2]
        bbox_scale = float(math.sqrt(current_area / max(1.0, float(image_w * image_h))))
        if not self._append_adaptive_embedding(candidate, bbox_scale):
            return 0

        self._last_adaptive_embedding_update_step = step
        self._last_adaptive_bbox_xywh = current_bbox.copy()
        self._adaptive_embedding_updates = int(getattr(self, "_adaptive_embedding_updates", 0)) + 1
        return 1

    @staticmethod
    def _validated_xyxy(bbox_xyxy: Any, frame_shape: tuple[int, ...]) -> Optional[np.ndarray]:
        if bbox_xyxy is None:
            return None
        try:
            arr = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
            return None

        h, w = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in arr[:4]]
        x1 = float(np.clip(x1, 0.0, max(0.0, float(w - 1))))
        y1 = float(np.clip(y1, 0.0, max(0.0, float(h - 1))))
        x2 = float(np.clip(x2, x1 + 1.0, max(x1 + 1.0, float(w))))
        y2 = float(np.clip(y2, y1 + 1.0, max(y1 + 1.0, float(h))))
        if x2 <= x1 or y2 <= y1:
            return None
        return np.asarray([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def _extract_agent1_handoff_bbox(agent1_env: Any) -> Optional[np.ndarray]:
        """Read the verified bottom bbox left by Agent 1 without mutating Agent 1."""
        for name in (
            "_bottom_stable_bbox_xyxy",
            "_bottom_bbox_xyxy",
            "_bottom_last_trusted_raw_bbox_xyxy",
        ):
            value = getattr(agent1_env, name, None)
            if value is None:
                continue
            try:
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if arr.size >= 4 and np.all(np.isfinite(arr[:4])):
                return arr[:4].copy()
        return None

    def _sync_image_geometry(self, frame: np.ndarray) -> None:
        """Keep observation normalization aligned with the real AirSim frame."""
        h, w = frame.shape[:2]
        if int(self.cfg.image_width) == int(w) and int(self.cfg.image_height) == int(h):
            return
        self.cfg.image_width = int(w)
        self.cfg.image_height = int(h)
        self.observation_builder.config.image_width = int(w)
        self.observation_builder.config.image_height = int(h)

    def _select_target_from_bottom_click(self, frame: np.ndarray) -> None:
        window = "Agent 2 - click static target in BOTTOM camera"
        selected: dict[str, Any] = {"done": False, "x": 0, "y": 0}

        def on_mouse(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
                selected.update(done=True, x=int(x), y=int(y))

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, on_mouse)
        shown = frame.copy()
        cv2.putText(
            shown,
            "Click the landing target (bottom camera)",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        while not selected["done"]:
            cv2.imshow(window, shown)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                cv2.destroyWindow(window)
                raise RuntimeError("Agent-2 target selection was cancelled.")

        bbox = self.tracker.select_target(frame, selected["x"], selected["y"])
        cv2.destroyWindow(window)
        if bbox is None or self.tracker.target_embedding is None:
            raise RuntimeError("Bottom click did not produce a target bbox and fingerprint.")

        fp = self.tracker.target_embedding.detach().cpu().numpy().copy()
        self._set_identity(fp, self.tracker.target_class_id)
        self._last_bbox_xyxy = self._xywh_to_xyxy(bbox)
        self._set_bottom_anchor_from_bbox(frame, self._last_bbox_xyxy)
        print(
            f"[AGENT_2] Static target selected: target_id={self._target_id} "
            f"initial_yolo_class={self._target_class_id} "
            f"bbox={self._last_bbox_xyxy.astype(int).tolist()}"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def attach_from_agent1(self, agent1_env: Any, handoff_info: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Attach Agent 2 to the exact physical state left by Agent 1."""
        self.client = agent1_env.client
        self.cfg.vehicle_name = str(agent1_env.cfg.vehicle_name)
        self.cfg.bottom_camera_name = str(getattr(agent1_env.cfg, "downward_camera_name", "bottom_center"))
        self.cfg.lidar_sensor_name = str(getattr(agent1_env.cfg, "lidar_sensor_name", "LidarSensor1"))
        self._refresh_bottom_camera_intrinsics()

        fingerprint = getattr(agent1_env, "target_fingerprint", None)
        class_id = getattr(agent1_env, "target_class_id", None)
        self._set_identity(fingerprint, class_id)

        handoff_bbox = self._extract_agent1_handoff_bbox(agent1_env)

        self._target_actor_name = str(getattr(agent1_env, "train_target_car", "") or "")
        self._expected_collision_object_name = str(
            self._target_actor_name or self.cfg.latch_target_actor_name or ""
        )
        self._expected_collision_object_source = "agent1_target_actor"
        self._read_target_surface_altitude()
        self._attached_from_agent1 = True
        self._reset_runtime_state()

        try:
            self.client.moveByVelocityBodyFrameAsync(
                vx=0.0,
                vy=0.0,
                vz=0.0,
                duration=0.10,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                vehicle_name=self.cfg.vehicle_name,
            ).join()
        except Exception:
            pass

        frame = self._get_bottom_frame()
        self._sync_image_geometry(frame)
        validated_handoff_bbox = self._validated_xyxy(handoff_bbox, frame.shape)
        if validated_handoff_bbox is not None:
            self._last_bbox_xyxy = validated_handoff_bbox.copy()
            self._control_bbox_xyxy = validated_handoff_bbox.copy()
            self._control_bbox_history = [
                self._xyxy_to_cxcywh(validated_handoff_bbox)
            ]
            self._control_bbox_outlier_suppressed = False
            self._control_bbox_filter_mode = "HANDOFF_INIT"
            self._last_match_step = int(self._step)
            self._last_similarity = 1.0
            self.tracker.last_bbox = self._xyxy_to_xywh(validated_handoff_bbox)
            self.tracker.last_good_bbox = self.tracker.last_bbox.copy()
            self.tracker.last_mode = "HANDOFF_INIT"
            self._set_bottom_anchor_from_bbox(frame, validated_handoff_bbox)

        obs, info = self._observe(frame=frame)
        info.update(
            {
                "agent": "AGENT_2",
                "attached_from_agent1": True,
                "agent1_handoff_reason": str(handoff_info.get("termination_reason", "handoff_success")),
                "handoff_bbox_transferred": validated_handoff_bbox is not None,
                "bottom_anchor_created": self._bottom_anchor_embedding is not None,
            }
        )
        print(
            "[AGENT_1P2] Landing Z authority enabled; Agent 1 remains active | "
            f"target_id={self._target_id} initial_yolo_class={self._target_class_id} "
            f"bbox_transferred={int(validated_handoff_bbox is not None)} "
            f"bottom_anchor={int(self._bottom_anchor_embedding is not None)} "
            f"targetZ_NED={self._target_surface_z_ned:+.3f}m "
            f"source={self._target_surface_source}"
        )
        return obs, info

    def reattach_after_recovery(
        self,
        agent1_env: Any,
        handoff_info: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return control to Agent 2 after Agent 1 reacquires the same target.

        The physical world, target identity, PPO episode counters, reward bank,
        and expected collision object are preserved. Only short-term landing
        perception/control state is cleared so stale PRED/Kalman data cannot
        leak across the recovery boundary.
        """
        saved_progress = {
            "step": int(self._step),
            "reward_bank": float(self._reward_bank),
            "episode_return": float(self._episode_return),
            "collision_timestamp": int(self._collision_timestamp_at_reset),
            "live_steps": int(self._episode_live_match_steps),
            "pred_steps": int(self._episode_predicted_steps),
            "none_steps": int(self._episode_no_target_steps),
            "descent_requested": int(self._episode_descent_requested_steps),
            "descent_allowed": int(self._episode_descent_allowed_steps),
            "descent_blocked": int(self._episode_descent_blocked_steps),
            "climb_blocked": int(self._episode_climb_command_blocked_steps),
            "best_center": float(self._episode_best_center_error),
            "best_similarity": float(self._episode_best_similarity),
            "recenter_steps": int(self._episode_recenter_steps),
            "xy_hold_steps": int(self._episode_xy_hold_steps),
        }

        self.client = agent1_env.client
        self.cfg.vehicle_name = str(agent1_env.cfg.vehicle_name)
        self.cfg.bottom_camera_name = str(
            getattr(agent1_env.cfg, "downward_camera_name", "bottom_center")
        )
        self.cfg.lidar_sensor_name = str(
            getattr(agent1_env.cfg, "lidar_sensor_name", "LidarSensor1")
        )

        if self._original_embedding is None:
            self._set_identity(
                getattr(agent1_env, "target_fingerprint", None),
                getattr(agent1_env, "target_class_id", None),
            )

        handoff_bbox = self._extract_agent1_handoff_bbox(agent1_env)
        self._target_actor_name = str(getattr(agent1_env, "train_target_car", "") or "")
        self._expected_collision_object_name = str(
            self._target_actor_name or self.cfg.latch_target_actor_name or ""
        )
        self._expected_collision_object_source = "agent1_target_actor"
        self._attached_from_agent1 = True
        self._reset_runtime_state()

        self._step = saved_progress["step"]
        self._reward_bank = saved_progress["reward_bank"]
        self._episode_return = saved_progress["episode_return"]
        self._collision_timestamp_at_reset = saved_progress["collision_timestamp"]
        self._episode_live_match_steps = saved_progress["live_steps"]
        self._episode_predicted_steps = saved_progress["pred_steps"]
        self._episode_no_target_steps = saved_progress["none_steps"]
        self._episode_descent_requested_steps = saved_progress["descent_requested"]
        self._episode_descent_allowed_steps = saved_progress["descent_allowed"]
        self._episode_descent_blocked_steps = saved_progress["descent_blocked"]
        self._episode_climb_command_blocked_steps = saved_progress["climb_blocked"]
        self._episode_best_center_error = saved_progress["best_center"]
        self._episode_best_similarity = saved_progress["best_similarity"]
        self._episode_recenter_steps = saved_progress["recenter_steps"]
        self._episode_xy_hold_steps = saved_progress["xy_hold_steps"]

        try:
            self.client.moveByVelocityBodyFrameAsync(
                vx=0.0,
                vy=0.0,
                vz=0.0,
                duration=0.10,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                vehicle_name=self.cfg.vehicle_name,
            ).join()
        except Exception:
            pass

        frame = self._get_bottom_frame()
        self._sync_image_geometry(frame)
        validated_handoff_bbox = self._validated_xyxy(handoff_bbox, frame.shape)
        if validated_handoff_bbox is not None:
            self._last_bbox_xyxy = validated_handoff_bbox.copy()
            self._control_bbox_xyxy = validated_handoff_bbox.copy()
            self._control_bbox_history = [
                self._xyxy_to_cxcywh(validated_handoff_bbox)
            ]
            self._control_bbox_outlier_suppressed = False
            self._control_bbox_filter_mode = "HANDOFF_INIT"
            self._last_match_step = int(self._step)
            self._last_similarity = 1.0
            self.tracker.last_bbox = self._xyxy_to_xywh(validated_handoff_bbox)
            self.tracker.last_good_bbox = self.tracker.last_bbox.copy()
            self.tracker.last_mode = "RECOVERY_HANDOFF_INIT"
            if self._bottom_anchor_embedding is None:
                self._set_bottom_anchor_from_bbox(frame, validated_handoff_bbox)

        self._read_target_surface_altitude()
        obs, info = self._observe(frame=frame)
        info.update(
            {
                "agent": "AGENT_2",
                "reattached_after_recovery": True,
                "agent1_recovery_handoff_reason": str(
                    handoff_info.get("termination_reason", "handoff_success")
                ),
                "handoff_bbox_transferred": validated_handoff_bbox is not None,
                "bottom_anchor_preserved": self._bottom_anchor_embedding is not None,
            }
        )
        print(
            "[AGENT_1P2] Agent 1 RECOVERY OUT -> Agent 2 BACK IN | "
            f"agent2_step={self._step} bbox_transferred="
            f"{int(validated_handoff_bbox is not None)} "
            f"targetSpeed={self._target_velocity_speed_mps:.2f}m/s"
        )
        return obs, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._attached_from_agent1 = False
        self.client.enableApiControl(True, vehicle_name=self.cfg.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.cfg.vehicle_name)
        self._refresh_bottom_camera_intrinsics()

        if self._standalone_initial_pose is None:
            self._standalone_initial_pose = self.client.simGetVehiclePose(vehicle_name=self.cfg.vehicle_name)
        else:
            self.client.simSetVehiclePose(
                self._standalone_initial_pose,
                True,
                vehicle_name=self.cfg.vehicle_name,
            )
            self.client.moveByVelocityBodyFrameAsync(
                0.0,
                0.0,
                0.0,
                0.15,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                vehicle_name=self.cfg.vehicle_name,
            ).join()
            time.sleep(0.20)

        self._reset_runtime_state()
        self._read_target_surface_altitude()
        frame = self._get_bottom_frame()
        self._sync_image_geometry(frame)
        if self._original_embedding is None:
            self._select_target_from_bottom_click(frame)

        obs, info = self._observe(frame=frame)
        info.update({"agent": "AGENT_2", "standalone": True, "target_actor_moved": False})
        print(
            "[AGENT_2 RESET] target actor untouched | "
            f"targetZ_NED={self._target_surface_z_ned:+.3f}m source={self._target_surface_source}"
        )
        return obs, info

    def _reset_runtime_state(self) -> None:
        self.observation_builder.reset()
        self.range_finder_array.reset()
        self._range_terminal_contact_armed = False
        self._range_terminal_contact_armed_monotonic = float("-inf")
        self._range_terminal_contact_started_monotonic = float("-inf")
        self._range_terminal_contact_started_step = -1
        self._range_terminal_contact_last_safe_height_m = float("inf")
        self._range_terminal_handoff_active = False
        self._range_terminal_handoff_timed_out = False
        self._range_terminal_contact_snapshot = None
        self._step = 0
        self._reward_bank = 0.0
        self._episode_return = 0.0
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._prev_relative_height_m = None
        self._lost_steps = 0
        self._prediction_steps = 0
        self._last_match_step = -999999
        self._last_verified_alignment_step = -999999
        self._last_verified_alignment_center_error = 999.0
        self._last_verified_alignment_bbox_rel_error = 999.0
        self._last_verified_alignment_similarity = 0.0
        self._last_verified_alignment_err_x = 999.0
        self._last_verified_alignment_err_y = 999.0
        self._last_verified_alignment_height_m = float("inf")
        self._authorized_descent_latched = False
        self._last_authorized_descent_step = -999999
        self._last_authorized_descent_center_error = 999.0
        self._last_authorized_descent_bbox_rel_error = 999.0
        self._last_authorized_descent_similarity = 0.0
        self._last_authorized_descent_err_x = 999.0
        self._last_authorized_descent_err_y = 999.0
        self._last_authorized_descent_height_m = float("inf")
        self._last_authorized_descent_vz_mps = 0.0
        self._authorized_descent_invalidated_reason = "never_authorized"
        self._touchdown_legacy_window = []
        self._touchdown_ready_snapshot = None
        self._last_valid_center_error_m = float("inf")
        self._last_valid_center_step = -999999
        self._last_valid_center_monotonic = float("-inf")
        self._last_valid_center_similarity = 0.0
        self._last_valid_center_bbox_rel_error = float("inf")
        self._last_valid_center_source = "NONE"
        self._terminal_kalman_state = None
        self._terminal_kalman_covariance = np.eye(4, dtype=np.float64)
        self._terminal_kalman_monotonic = float("-inf")
        self._terminal_descent_committed = False
        self._terminal_blind_descent_started_monotonic = float("-inf")
        self._live_match_streak = 0
        self._last_match_margin = 0.0
        self._last_spatial_jump_norm = 0.0
        self._last_match_reject_reason = ""
        self._last_similarity = 0.0
        self._last_candidate_class_id = None
        self._last_candidate_confidence = 0.0
        self._last_candidate_count = 0
        self._last_candidate_scores = []
        self._adaptive_embeddings = []
        self._adaptive_embedding_steps = []
        self._adaptive_embedding_scales = []
        self._last_adaptive_embedding_update_step = -999999
        self._last_adaptive_bbox_xywh = None
        self._adaptive_embedding_updates = 0
        self._last_bbox_xyxy = None
        self._control_bbox_xyxy = None
        self._control_bbox_history = []
        self._control_bbox_outlier_suppressed = False
        self._control_bbox_filter_mode = "INIT"
        self._landing_anchor_px = None
        self._landing_anchor_virtual_bbox_xyxy = None
        self._landing_anchor_full_size_px = None
        self._landing_anchor_aspect_ratio = None
        self._landing_anchor_reference_size_px = None
        self._landing_anchor_reference_height_m = None
        self._landing_anchor_mode = "INIT"
        self._landing_anchor_edge_flags = "NONE"
        self._terminal_anchor_locked = False
        self._terminal_anchor_px = None
        self._terminal_anchor_previous_gray = None
        self._terminal_anchor_support_bbox_xyxy = None
        self._terminal_anchor_candidate_px = None
        self._terminal_anchor_acquire_streak = 0
        self._terminal_anchor_lock_step = -999999
        self._terminal_anchor_last_identity_step = -999999
        self._terminal_anchor_age_steps = 999999
        self._terminal_anchor_source = "INACTIVE"
        self._terminal_anchor_flow_points = 0
        self._terminal_anchor_flow_confidence = 0.0
        self._last_bottom_observation_monotonic = 0.0
        self._last_bottom_frame = None
        self._last_info = {}
        self._last_vertical_control_state = "HOLD_INIT"
        self._last_descent_block_reason = "waiting_for_first_control_step"
        self._last_raw_vz_action = 0.0
        self._last_requested_vz_mps = 0.0
        self._last_applied_vz_mps = 0.0
        self._last_vertical_speed_limit_mps = 0.0
        self._last_soft_catchup_descent_active = False
        self._last_climb_command_blocked = False
        self._last_observation_monotonic = None
        self._last_control_had_live_match = False
        self._last_control_err_x = 0.0
        self._last_control_err_y = 0.0
        self._control_img_vel_x = 0.0
        self._control_img_vel_y = 0.0
        self._last_control_dt_s = float(self.cfg.cmd_duration_s)
        self._predictive_catchup_active = False
        self._predictive_catchup_release_streak = 0
        self._last_predictive_guidance = {
            "live_match": False,
            "catchup_active": False,
            "predicted_err_x": 0.0,
            "predicted_err_y": 0.0,
            "predicted_center_error": 999.0,
            "image_velocity_x_per_s": 0.0,
            "image_velocity_y_per_s": 0.0,
            "image_speed_per_s": 0.0,
            "outward_speed_per_s": 0.0,
            "prediction_horizon_s": float(self.cfg.cmd_duration_s),
            "correction_vx_mps": 0.0,
            "correction_vy_mps": 0.0,
            "correction_limit_mps": 0.0,
        }
        self._reset_target_motion_estimator()
        self._non_live_started_monotonic = None
        self._non_live_duration_s = 0.0
        self._alignment_ready_streak = 0
        self._descent_alignment_latched = False
        self._landing_lock_visual_gap_steps = 0
        self._landing_lock_bad_live_steps = 0
        self._landing_lock_acquired_step = -999999
        self._last_horizontal_control_state = "HOLD_INIT"
        self._last_pd_action_vx = 0.0
        self._last_pd_action_vy = 0.0
        self._last_residual_action_vx = 0.0
        self._last_residual_action_vy = 0.0
        self._last_horizontal_action_vx = 0.0
        self._last_horizontal_action_vy = 0.0
        self._last_horizontal_speed_limit_mps = 0.0
        self._last_velocity_ff_vx_mps = 0.0
        self._last_velocity_ff_vy_mps = 0.0
        self._last_horizontal_correction_vx_mps = 0.0
        self._last_horizontal_correction_vy_mps = 0.0
        self._last_horizontal_command_vx_mps = 0.0
        self._last_horizontal_command_vy_mps = 0.0
        self._last_horizontal_total_speed_limit_mps = 0.0
        self._episode_recenter_steps = 0
        self._episode_xy_hold_steps = 0
        self._episode_live_match_steps = 0
        self._episode_predicted_steps = 0
        self._episode_no_target_steps = 0
        self._episode_descent_requested_steps = 0
        self._episode_descent_allowed_steps = 0
        self._episode_descent_blocked_steps = 0
        self._episode_climb_command_blocked_steps = 0
        self._episode_best_center_error = float("inf")
        self._episode_best_similarity = 0.0
        self._last_dense_reward = 0.0
        self._last_dense_reward_parts = {
            "aligned_descent_progress": 0.0,
            "landing_lock_time": 0.0,
            "hesitation": 0.0,
            "unsafe_descent": 0.0,
        }
        try:
            collision = self.client.simGetCollisionInfo(vehicle_name=self.cfg.vehicle_name)
            self._collision_timestamp_at_reset = int(getattr(collision, "time_stamp", 0) or 0)
        except Exception:
            self._collision_timestamp_at_reset = 0

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------
    @staticmethod
    def _xywh_to_xyxy(bbox: Any) -> np.ndarray:
        x, y, w, h = [float(v) for v in bbox[:4]]
        return np.asarray([x, y, x + max(1.0, w), y + max(1.0, h)], dtype=np.float32)

    @staticmethod
    def _xyxy_to_xywh(bbox: Any) -> list[int]:
        """Convert XYXY to top-left XYWH for tracker APIs."""
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        x = int(round(x1))
        y = int(round(y1))
        w = max(1, int(round(x2 - x1)))
        h = max(1, int(round(y2 - y1)))
        return [x, y, w, h]

    @staticmethod
    def _xyxy_to_cxcywh(bbox: Any) -> np.ndarray:
        """Convert XYXY to floating-point center CXCYWH for control geometry."""
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        return np.asarray(
            [
                0.5 * (x1 + x2),
                0.5 * (y1 + y2),
                max(1.0, x2 - x1),
                max(1.0, y2 - y1),
            ],
            dtype=np.float64,
        )

    def _get_bottom_frame(self) -> np.ndarray:
        responses = self.client.simGetImages(
            [airsim.ImageRequest(self.cfg.bottom_camera_name, airsim.ImageType.Scene, False, False)],
            vehicle_name=self.cfg.vehicle_name,
        )
        if not responses or not responses[0].image_data_uint8:
            return np.zeros((self.cfg.image_height, self.cfg.image_width, 3), dtype=np.uint8)
        response = responses[0]
        img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        frame = img.reshape(response.height, response.width, 3)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def _strict_track(self, frame: np.ndarray) -> tuple[Optional[np.ndarray], float, str]:
        """Track ``user_target`` with immutable identity plus verified landing views.

        YOLO remains a proposal generator. Adaptive embeddings may improve
        close-range scale coverage, but a candidate must still agree with at
        least one immutable identity anchor before it can become LIVE.
        """
        immutable_references = list(self._reference_embeddings)
        if not immutable_references and self._original_embedding is not None:
            immutable_references = [self._original_embedding]
        references = self._all_reference_embeddings()
        if not references:
            references = immutable_references
        if not immutable_references or not references:
            return None, 0.0, "NO_IDENTITY"

        candidates = self.tracker._detect_candidates(frame)
        self._last_candidate_count = int(len(candidates))
        self._last_candidate_scores = []

        if not candidates:
            self._last_candidate_class_id = None
            self._last_candidate_confidence = 0.0
            self._prediction_steps += 1
            self._live_match_streak = 0
            self._last_match_margin = 0.0
            self._last_spatial_jump_norm = 0.0
            self._last_match_reject_reason = "no_detection"
            if self._last_bbox_xyxy is not None and self._prediction_steps <= self.cfg.max_prediction_steps:
                return self._last_bbox_xyxy.copy(), float(self._last_similarity), "PRED_NO_DETECTION"
            return None, 0.0, "NO_DETECTION"

        best = None
        best_embedding: Optional[torch.Tensor] = None
        best_similarity = -1.0
        best_immutable_similarity = -1.0
        best_anchor_index = -1
        scored_candidates: list[tuple[float, Any, int, torch.Tensor, float]] = []

        for candidate_index, candidate in enumerate(candidates):
            embedding = self.tracker._embedding_from_bbox(frame, candidate.bbox)
            if embedding is None:
                self._last_candidate_scores.append(
                    {
                        "candidate_index": int(candidate_index),
                        "yolo_class_id": int(candidate.cls_id),
                        "yolo_confidence": float(candidate.conf),
                        "resnet_similarity": -1.0,
                        "immutable_similarity": -1.0,
                        "best_anchor_index": -1,
                    }
                )
                continue

            anchor_scores = [
                float(torch.dot(reference, embedding).detach().cpu().item())
                for reference in references
            ]
            immutable_scores = [
                float(torch.dot(reference, embedding).detach().cpu().item())
                for reference in immutable_references
            ]
            candidate_similarity = max(anchor_scores)
            candidate_immutable_similarity = max(immutable_scores)
            candidate_anchor_index = int(np.argmax(anchor_scores))
            self._last_candidate_scores.append(
                {
                    "candidate_index": int(candidate_index),
                    "yolo_class_id": int(candidate.cls_id),
                    "yolo_confidence": float(candidate.conf),
                    "resnet_similarity": float(candidate_similarity),
                    "immutable_similarity": float(candidate_immutable_similarity),
                    "best_anchor_index": int(candidate_anchor_index),
                }
            )

            if candidate_similarity > best_similarity:
                best_similarity = float(candidate_similarity)
                best_immutable_similarity = float(candidate_immutable_similarity)
                best = candidate
                best_embedding = embedding
                best_anchor_index = candidate_anchor_index

            scored_candidates.append(
                (
                    float(candidate_similarity),
                    candidate,
                    int(candidate_anchor_index),
                    embedding,
                    float(candidate_immutable_similarity),
                )
            )

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        second_best_similarity = (
            float(scored_candidates[1][0]) if len(scored_candidates) > 1 else -1.0
        )
        match_margin = (
            float(best_similarity - second_best_similarity)
            if second_best_similarity >= -0.5
            else 1.0
        )

        spatial_jump_norm = 0.0
        spatial_ok = True
        if best is not None and self._last_bbox_xyxy is not None:
            current_xyxy = self._xywh_to_xyxy(best.bbox)
            h, w = frame.shape[:2]
            prev_cx = 0.5 * float(self._last_bbox_xyxy[0] + self._last_bbox_xyxy[2])
            prev_cy = 0.5 * float(self._last_bbox_xyxy[1] + self._last_bbox_xyxy[3])
            curr_cx = 0.5 * float(current_xyxy[0] + current_xyxy[2])
            curr_cy = 0.5 * float(current_xyxy[1] + current_xyxy[3])
            spatial_jump_norm = float(
                math.hypot(curr_cx - prev_cx, curr_cy - prev_cy)
                / max(1.0, math.hypot(float(w), float(h)))
            )
            spatial_ok = bool(
                spatial_jump_norm <= float(self.cfg.max_reacquire_center_jump_norm)
                or best_similarity >= float(self.cfg.high_conf_reacquire_similarity)
            )

        similarity_ok = bool(
            best is not None and best_similarity >= self.cfg.min_match_similarity
        )
        adaptive_anchor_selected = bool(
            best is not None
            and best_anchor_index >= len(immutable_references)
        )
        recent_live_identity_chain = bool(
            adaptive_anchor_selected
            and int(getattr(self, "_step", 0))
            - int(getattr(self, "_last_match_step", -999999))
            <= max(2, int(self.cfg.max_prediction_steps))
        )
        immutable_identity_ok = bool(
            best is not None
            and (
                best_immutable_similarity
                >= float(self.cfg.adaptive_embedding_identity_floor)
                or recent_live_identity_chain
            )
        )
        margin_ok = bool(
            best is not None
            and (
                len(scored_candidates) <= 1
                or match_margin >= float(self.cfg.min_match_margin)
                or best_similarity >= float(self.cfg.high_conf_reacquire_similarity)
            )
        )

        self._last_match_margin = float(match_margin)
        self._last_spatial_jump_norm = float(spatial_jump_norm)

        if (
            best is None
            or not similarity_ok
            or not immutable_identity_ok
            or not margin_ok
            or not spatial_ok
        ):
            self._last_candidate_class_id = None if best is None else int(best.cls_id)
            self._last_candidate_confidence = 0.0 if best is None else float(best.conf)
            self._last_similarity = max(0.0, float(best_similarity))
            self._prediction_steps += 1
            self._live_match_streak = 0
            if best is None:
                self._last_match_reject_reason = "no_embedded_candidate"
            elif not similarity_ok:
                self._last_match_reject_reason = "low_similarity"
            elif not immutable_identity_ok:
                self._last_match_reject_reason = "identity_chain_and_immutable_floor_failed"
            elif not margin_ok:
                self._last_match_reject_reason = "ambiguous_similarity_margin"
            else:
                self._last_match_reject_reason = "implausible_spatial_jump"
            if self._last_bbox_xyxy is not None and self._prediction_steps <= self.cfg.max_prediction_steps:
                return self._last_bbox_xyxy.copy(), max(0.0, best_similarity), "PRED_REJECTED_MATCH"
            return None, max(0.0, best_similarity), "REJECTED_MATCH"

        self._prediction_steps = 0
        self._live_match_streak += 1
        self._last_match_reject_reason = ""
        self._last_bbox_xyxy = self._xywh_to_xyxy(best.bbox)
        self._last_similarity = float(best_similarity)
        self._last_match_step = int(self._step)
        self._last_candidate_class_id = int(best.cls_id)
        self._last_candidate_confidence = float(best.conf)

        additions = self._maybe_update_adaptive_embedding_bank(
            frame=frame,
            bbox_xywh=best.bbox,
            immutable_similarity=float(best_immutable_similarity),
            combined_similarity=float(best_similarity),
            matched_adaptive_anchor=bool(adaptive_anchor_selected),
        )

        self.tracker.last_bbox = list(best.bbox)
        self.tracker.last_good_bbox = list(best.bbox)
        self.tracker.last_score = float(best_similarity)
        anchor_kind = (
            "IMMUTABLE"
            if best_anchor_index < len(immutable_references)
            else "ADAPTIVE"
        )
        self.tracker.last_mode = (
            f"MATCH_USER_TARGET_{anchor_kind}_{best_anchor_index}"
            f"_BANKADD_{additions}"
        )
        return self._last_bbox_xyxy.copy(), float(best_similarity), "MATCH"

    def _get_api_state(self) -> tuple[DroneState, float, Any]:
        state = self.client.getMultirotorState(vehicle_name=self.cfg.vehicle_name)
        k = state.kinematics_estimated
        drone_z_ned = float(k.position.z_val)
        altitude = max(0.0, -drone_z_ned)
        velocity = k.linear_velocity
        angular = k.angular_velocity
        roll = 0.0
        pitch = 0.0
        try:
            pitch, roll, _yaw = airsim.to_eularian_angles(k.orientation)
        except Exception:
            pass
        drone_state = DroneState(
            altitude_m=altitude,
            vx_mps=float(velocity.x_val),
            vy_mps=float(velocity.y_val),
            vz_mps=float(velocity.z_val),
            roll_rad=float(roll),
            pitch_rad=float(pitch),
            yaw_rate_radps=float(angular.z_val),
        )
        # Both values are raw AirSim NED-Z coordinates. A drone above the
        # target has a smaller/more-negative Z, so this separation is positive.
        relative_height = float(self._target_surface_z_ned - drone_z_ned)
        return drone_state, relative_height, state

    def _select_effective_landing_height(
        self,
        *,
        live_match: bool,
        api_relative_height_m: float,
        range_state: dict[str, Any],
        now_monotonic: float | None = None,
    ) -> tuple[float, str, bool]:
        """Select the height used by final-landing observation and safety logic.

        Vision/API geometry remains authoritative while a LIVE target match is
        available. When vision is degraded, the calibrated range array may
        become authoritative only inside the final-landing envelope and only
        after a recent, safe visual alignment. This prevents a road or an
        unrelated surface from being treated as the selected moving target.
        """
        api_height = float(api_relative_height_m)
        if live_match:
            return api_height, "VISION_API_TARGET_SURFACE", False

        if bool(self.cfg.range_require_calibration) and not bool(
            range_state.get("range_calibration_loaded", False)
        ):
            return api_height, "API_FALLBACK_RANGE_NOT_CALIBRATED", False

        if not bool(range_state.get("range_height_reliable", False)):
            return api_height, "API_FALLBACK_RANGE_GEOMETRY_UNSAFE", False

        range_height = float(range_state.get("range_mean_m", float("inf")))
        if (
            not np.isfinite(range_height)
            or range_height <= float(self.cfg.range_sensor_final_stop_m)
            or range_height > float(self.cfg.range_sensor_final_entry_m)
        ):
            return api_height, "API_FALLBACK_RANGE_OUTSIDE_FINAL_ENVELOPE", False

        now_value = float(time.monotonic() if now_monotonic is None else now_monotonic)
        last_visual = float(getattr(self, "_last_valid_center_monotonic", float("-inf")))
        visual_age = now_value - last_visual
        center = float(getattr(self, "_last_valid_center_error_m", float("inf")))
        similarity = float(getattr(self, "_last_valid_center_similarity", 0.0))
        recent_visual_context_safe = bool(
            np.isfinite(visual_age)
            and 0.0 <= visual_age <= float(self.cfg.range_sensor_final_recent_vision_s)
            and np.isfinite(center)
            and center <= float(self.cfg.range_sensor_final_max_center_error_m)
            and similarity >= float(self.cfg.range_sensor_final_min_similarity)
        )

        # The normal descent gate already records the last command that was
        # physically authorized from valid target geometry. Use that evidence
        # as a second arming source, because a slow AirSim control step can make
        # the separate last-valid-center wall-clock age expire before the range
        # crosses the close-range pre-arm envelope. This does not relax XY
        # safety: the
        # recorded authorization must satisfy the same center/similarity gates.
        authorized_age_steps = int(
            self._step - int(getattr(self, "_last_authorized_descent_step", -999999))
        )
        authorized_context_safe = bool(
            getattr(self, "_authorized_descent_latched", False)
            and 0 <= authorized_age_steps <= 3
            and np.isfinite(
                float(getattr(self, "_last_authorized_descent_center_error", float("inf")))
            )
            and float(getattr(self, "_last_authorized_descent_center_error", float("inf")))
            <= float(self.cfg.range_sensor_final_max_center_error_m)
            and float(getattr(self, "_last_authorized_descent_similarity", 0.0))
            >= float(self.cfg.range_sensor_final_min_similarity)
        )
        visual_context_safe = bool(
            recent_visual_context_safe or authorized_context_safe
        )
        if not visual_context_safe:
            return api_height, "API_FALLBACK_RECENT_VISION_UNSAFE", False

        return range_height, "CALIBRATED_RANGE_ARRAY", True

    def _read_target_surface_altitude(self) -> None:
        actor = str(
            getattr(self, "_target_actor_name", "")
            or self.cfg.static_target_actor_name
            or ""
        )
        previous_z = float(
            getattr(
                self,
                "_target_surface_z_ned",
                -float(self.cfg.static_target_surface_altitude_m),
            )
        )
        previous_source = str(
            getattr(self, "_target_surface_source", "configured_static")
        )
        if actor:
            try:
                pose = self.client.simGetObjectPose(actor)
                z_ned = float(pose.position.z_val)
                if np.isfinite(z_ned):
                    self._target_surface_z_ned = z_ned
                    # Diagnostic conversion only. Relative height never uses it.
                    self._target_surface_altitude_m = max(0.0, -z_ned)
                    self._target_surface_source = "api_object_pose_z_ned_only"
                    return
            except Exception:
                pass

            # A single read failure must not replace a previously valid moving
            # platform Z with the configured ground fallback. Keep the last
            # valid API/collision value until the actor query succeeds again.
            if (
                np.isfinite(previous_z)
                and (
                    previous_source.startswith("api_object_pose_z_ned")
                    or previous_source.startswith("verified_collision_api_z_ned")
                )
            ):
                self._target_surface_z_ned = previous_z
                self._target_surface_altitude_m = max(0.0, -previous_z)
                self._target_surface_source = f"{previous_source.split('_stale')[0]}_stale"
                return

        self._target_surface_altitude_m = float(self.cfg.static_target_surface_altitude_m)
        self._target_surface_z_ned = -self._target_surface_altitude_m
        self._target_surface_source = "configured_static"

    def _reset_target_motion_estimator(self) -> None:
        self._previous_motion_gray = None
        self._previous_motion_bbox_xyxy = None
        self._previous_motion_time = None
        self._previous_relative_position_body = None
        self._previous_relative_height_m = None
        self._previous_motion_yaw_rad = None
        self._visual_motion_attitude_valid = True
        self._visual_motion_yaw_rate_dps = 0.0
        self._visual_relative_position_body_x_m = 0.0
        self._visual_relative_position_body_y_m = 0.0
        self._visual_relative_velocity_body_x_mps = 0.0
        self._visual_relative_velocity_body_y_mps = 0.0
        self._visual_relative_velocity_valid = False
        self._visual_kalman_state = None
        self._visual_kalman_covariance = np.eye(4, dtype=np.float64)
        self._visual_kalman_velocity_initialized = False
        self._visual_motion_age_s = float("inf")
        self._visual_motion_source = "WAIT"
        self._visual_bbox_velocity_x_per_s = 0.0
        self._visual_bbox_velocity_y_per_s = 0.0
        self._visual_flow_velocity_x_per_s = 0.0
        self._visual_flow_velocity_y_per_s = 0.0
        self._visual_flow_point_count = 0
        self._visual_flow_confidence = 0.0
        self._drone_body_velocity_x_mps = 0.0
        self._drone_body_velocity_y_mps = 0.0
        self._target_velocity_world_x_mps = 0.0
        self._target_velocity_world_y_mps = 0.0
        self._target_velocity_body_vx_mps = 0.0
        self._target_velocity_body_vy_mps = 0.0
        self._target_velocity_speed_mps = 0.0
        self._target_velocity_valid = False
        self._target_velocity_age_s = float("inf")
        self._reacquire_climb_started_monotonic = None
        self._reacquire_climb_active = False

    def _refresh_bottom_camera_intrinsics(self) -> None:
        """Read only the camera FOV; target state is never queried here."""
        hfov = float(self.cfg.bottom_camera_hfov_deg)
        try:
            camera_info = self.client.simGetCameraInfo(
                self.cfg.bottom_camera_name,
                vehicle_name=self.cfg.vehicle_name,
            )
            candidate = float(getattr(camera_info, "fov", hfov))
            if np.isfinite(candidate) and 20.0 <= candidate <= 170.0:
                hfov = candidate
        except Exception:
            pass
        self._bottom_camera_hfov_deg = float(hfov)

    def _camera_projection_tangents(self, frame_shape: tuple[int, ...]) -> tuple[float, float]:
        """Return tan(horizontal/vertical half FOV) for square image pixels."""
        h, w = frame_shape[:2]
        tan_h = math.tan(math.radians(float(self._bottom_camera_hfov_deg)) * 0.5)
        tan_v = tan_h * float(h) / max(1.0, float(w))
        return float(max(1.0e-4, tan_h)), float(max(1.0e-4, tan_v))

    @staticmethod
    def _body_velocity_from_api_state(api_state: Any) -> tuple[float, float, float]:
        """Convert the drone's own world-NED velocity to body XY and return yaw."""
        yaw = 0.0
        try:
            _pitch, _roll, yaw = airsim.to_eularian_angles(
                api_state.kinematics_estimated.orientation
            )
        except Exception:
            yaw = 0.0
        velocity = api_state.kinematics_estimated.linear_velocity
        world_vx = float(velocity.x_val)
        world_vy = float(velocity.y_val)
        cos_yaw = math.cos(float(yaw))
        sin_yaw = math.sin(float(yaw))
        body_vx = cos_yaw * world_vx + sin_yaw * world_vy
        body_vy = -sin_yaw * world_vx + cos_yaw * world_vy
        return float(body_vx), float(body_vy), float(yaw)

    @staticmethod
    def _attitude_from_api_state(api_state: Any) -> tuple[float, float, float]:
        """Return pitch, roll and yaw radians from the drone state."""
        try:
            pitch, roll, yaw = airsim.to_eularian_angles(
                api_state.kinematics_estimated.orientation
            )
            return float(pitch), float(roll), float(yaw)
        except Exception:
            return 0.0, 0.0, 0.0

    def _init_kalman_diagnostic_logger(self) -> None:
        """Create a scalar-only CSV session for offline A/B/C replay."""
        if not bool(getattr(self.cfg, "kalman_diagnostic_logging_enabled", False)):
            return
        try:
            base = Path(__file__).resolve().parent / str(
                getattr(self.cfg, "kalman_diagnostic_output_dir", "outputs/kalman_live_diagnostics")
            )
            session = base / f"session_{self._kalman_diag_session_started}"
            session.mkdir(parents=True, exist_ok=True)
            self._kalman_diag_csv_path = session / "kalman_replay.csv"
            (base / "latest_session.txt").write_text(str(session.resolve()), encoding="utf-8")
        except Exception as exc:
            self._kalman_diag_csv_path = None
            _landing_console_print(f"[KALMAN DIAG] logger disabled after init error: {exc}")

    @staticmethod
    def _diag_finite_or_blank(value: Any) -> Any:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        return number if np.isfinite(number) else ""

    def _append_kalman_diagnostic_row(self, info: dict[str, Any]) -> None:
        """Append one real bottom-camera observation without affecting control."""
        path = getattr(self, "_kalman_diag_csv_path", None)
        if path is None:
            return
        now = float(info.get("bottom_observation_monotonic", time.monotonic()))
        previous = self._kalman_diag_last_monotonic
        dt_s = (
            float(now - previous)
            if previous is not None and np.isfinite(now - previous) and now > previous
            else float(info.get("bottom_control_dt_s", self.cfg.cmd_duration_s))
        )
        self._kalman_diag_last_monotonic = now
        measurement_valid = bool(
            info.get("bottom_match_live", False)
            and np.isfinite(float(info.get("bottom_err_x", float("nan"))))
            and np.isfinite(float(info.get("bottom_err_y", float("nan"))))
            and np.isfinite(float(info.get("relative_height_to_target_m", float("nan"))))
        )
        row = {
            "frame": int(self._step),
            "time_s": float(now - self._kalman_diag_start_monotonic),
            "dt_s": float(max(1.0e-6, dt_s)),
            "measurement_valid": int(measurement_valid),
            "err_x_norm": self._diag_finite_or_blank(info.get("bottom_err_x")) if measurement_valid else "",
            "err_y_norm": self._diag_finite_or_blank(info.get("bottom_err_y")) if measurement_valid else "",
            "height_m": self._diag_finite_or_blank(info.get("relative_height_to_target_m")),
            "external_vx_mps": self._diag_finite_or_blank(getattr(self, "_diag_external_vx_mps", float("nan"))),
            "external_vy_mps": self._diag_finite_or_blank(getattr(self, "_diag_external_vy_mps", float("nan"))),
            "truth_x_norm": "",
            "truth_y_norm": "",
            "tracker_mode": str(info.get("tracker_mode", "")),
            "similarity": self._diag_finite_or_blank(info.get("bottom_similarity")),
            "bbox_rel_error": self._diag_finite_or_blank(info.get("bottom_bbox_rel_err")),
            "raw_metric_x_m": self._diag_finite_or_blank(getattr(self, "_diag_raw_metric_x_m", float("nan"))),
            "raw_metric_y_m": self._diag_finite_or_blank(getattr(self, "_diag_raw_metric_y_m", float("nan"))),
            "project_kalman_x_m": self._diag_finite_or_blank(
                self._visual_kalman_state[0] if self._visual_kalman_state is not None else float("nan")
            ),
            "project_kalman_y_m": self._diag_finite_or_blank(
                self._visual_kalman_state[1] if self._visual_kalman_state is not None else float("nan")
            ),
            "project_kalman_vx_mps": self._diag_finite_or_blank(
                self._visual_kalman_state[2] if self._visual_kalman_state is not None else float("nan")
            ),
            "project_kalman_vy_mps": self._diag_finite_or_blank(
                self._visual_kalman_state[3] if self._visual_kalman_state is not None else float("nan")
            ),
            "velocity_measurement_valid": int(getattr(self, "_diag_velocity_measurement_valid", False)),
            "visual_motion_source": str(info.get("visual_motion_source", "")),
            "hfov_deg": self._diag_finite_or_blank(info.get("bottom_camera_hfov_deg")),
            "frame_width_px": int(info.get("bottom_frame_width_px", self.cfg.image_width)),
            "frame_height_px": int(info.get("bottom_frame_height_px", self.cfg.image_height)),
        }
        try:
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                if not self._kalman_diag_header_written and path.stat().st_size == 0:
                    writer.writeheader()
                    self._kalman_diag_header_written = True
                writer.writerow(row)
        except Exception as exc:
            _landing_console_print(f"[KALMAN DIAG] write error; disabling logger: {exc}")
            self._kalman_diag_csv_path = None

    def _update_visual_motion_kalman(
        self,
        measured_x_m: float,
        measured_y_m: float,
        measured_vx_mps: Optional[float],
        measured_vy_mps: Optional[float],
        dt_s: float,
        flow_confidence: float,
        source: str,
    ) -> tuple[float, float, float, float, bool]:
        """Filter metric relative target state with a constant-velocity KF.

        Measurements come only from bottom-camera geometry, bbox motion and
        optical flow. The filter never reads target actor XY pose or velocity.
        """
        velocity_measurement_valid = bool(
            measured_vx_mps is not None
            and measured_vy_mps is not None
            and np.isfinite(float(measured_vx_mps))
            and np.isfinite(float(measured_vy_mps))
        )
        measured_x = float(measured_x_m)
        measured_y = float(measured_y_m)
        measured_vx = float(measured_vx_mps or 0.0)
        measured_vy = float(measured_vy_mps or 0.0)

        if not bool(self.cfg.visual_kalman_enabled):
            return (
                measured_x,
                measured_y,
                measured_vx,
                measured_vy,
                velocity_measurement_valid,
            )

        dt = float(
            np.clip(
                dt_s,
                float(self.cfg.visual_motion_min_dt_s),
                float(self.cfg.visual_motion_max_dt_s),
            )
        )
        position_var = max(1.0e-6, float(self.cfg.visual_kalman_position_std_m) ** 2)
        if str(source).startswith("BBOX+FLOW"):
            flow_weight = float(np.clip(flow_confidence, 0.0, 1.0))
            velocity_std = (
                flow_weight * float(self.cfg.visual_kalman_velocity_std_flow_mps)
                + (1.0 - flow_weight)
                * float(self.cfg.visual_kalman_velocity_std_bbox_mps)
            )
        else:
            velocity_std = float(self.cfg.visual_kalman_velocity_std_bbox_mps)
        velocity_var = max(1.0e-5, velocity_std ** 2)

        if self._visual_kalman_state is None:
            self._visual_kalman_state = np.asarray(
                [measured_x, measured_y, measured_vx, measured_vy],
                dtype=np.float64,
            )
            self._visual_kalman_covariance = np.diag(
                [position_var, position_var, velocity_var, velocity_var]
            ).astype(np.float64)
            self._visual_kalman_velocity_initialized = velocity_measurement_valid
            return (
                measured_x,
                measured_y,
                measured_vx,
                measured_vy,
                bool(self._visual_kalman_velocity_initialized),
            )

        transition = np.asarray(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        acceleration_std = max(
            1.0e-3, float(self.cfg.visual_kalman_process_accel_std_mps2)
        )
        process_map = np.asarray(
            [
                [0.5 * dt * dt, 0.0],
                [0.0, 0.5 * dt * dt],
                [dt, 0.0],
                [0.0, dt],
            ],
            dtype=np.float64,
        )
        process_noise = (acceleration_std ** 2) * (process_map @ process_map.T)
        process_noise += np.eye(4, dtype=np.float64) * 1.0e-8

        state_pred = transition @ self._visual_kalman_state
        covariance_pred = (
            transition @ self._visual_kalman_covariance @ transition.T
            + process_noise
        )

        if velocity_measurement_valid:
            measurement = np.asarray(
                [measured_x, measured_y, measured_vx, measured_vy],
                dtype=np.float64,
            )
            observation = np.eye(4, dtype=np.float64)
            measurement_noise = np.diag(
                [position_var, position_var, velocity_var, velocity_var]
            ).astype(np.float64)
        else:
            measurement = np.asarray([measured_x, measured_y], dtype=np.float64)
            observation = np.asarray(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
            measurement_noise = np.diag([position_var, position_var]).astype(
                np.float64
            )

        innovation = measurement - observation @ state_pred
        innovation_covariance = (
            observation @ covariance_pred @ observation.T + measurement_noise
        )
        try:
            kalman_gain = np.linalg.solve(
                innovation_covariance.T,
                (covariance_pred @ observation.T).T,
            ).T
        except np.linalg.LinAlgError:
            kalman_gain = covariance_pred @ observation.T @ np.linalg.pinv(
                innovation_covariance
            )
        state = state_pred + kalman_gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        # Joseph form keeps covariance positive semi-definite under rounding.
        correction = identity - kalman_gain @ observation
        covariance = (
            correction @ covariance_pred @ correction.T
            + kalman_gain @ measurement_noise @ kalman_gain.T
        )
        self._visual_kalman_state = state
        self._visual_kalman_covariance = covariance
        if velocity_measurement_valid:
            self._visual_kalman_velocity_initialized = True
        return (
            float(state[0]),
            float(state[1]),
            float(state[2]),
            float(state[3]),
            bool(self._visual_kalman_velocity_initialized),
        )

    def _terminal_kalman_prediction(self, now_monotonic: Optional[float] = None) -> dict[str, Any]:
        """Predict the latched trustworthy relative target state without vision updates."""
        if not bool(getattr(self.cfg, "terminal_kalman_enabled", True)):
            return {"valid": False, "reason": "DISABLED"}
        state = getattr(self, "_terminal_kalman_state", None)
        if state is None:
            return {"valid": False, "reason": "NOT_INITIALIZED"}
        now = float(time.monotonic() if now_monotonic is None else now_monotonic)
        source_time = float(getattr(self, "_terminal_kalman_monotonic", float("-inf")))
        age_s = float(now - source_time)
        if not np.isfinite(age_s) or age_s < 0.0:
            return {"valid": False, "reason": "INVALID_AGE", "age_s": age_s}
        max_age = float(self.cfg.terminal_blind_descent_max_duration_s)
        if age_s > max_age:
            return {"valid": False, "reason": "TIMEOUT", "age_s": age_s}

        dt = float(age_s)
        transition = np.asarray(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        acceleration_std = max(1.0e-3, float(self.cfg.visual_kalman_process_accel_std_mps2))
        process_map = np.asarray(
            [[0.5 * dt * dt, 0.0], [0.0, 0.5 * dt * dt],
             [dt, 0.0], [0.0, dt]],
            dtype=np.float64,
        )
        process_noise = (acceleration_std ** 2) * (process_map @ process_map.T)
        covariance = transition @ self._terminal_kalman_covariance @ transition.T + process_noise
        predicted = transition @ np.asarray(state, dtype=np.float64)
        x_m, y_m, vx_mps, vy_mps = [float(v) for v in predicted]
        radial_m = float(math.hypot(x_m, y_m))
        position_std_m = float(math.sqrt(max(0.0, float(np.max(np.diag(covariance)[:2])))))
        valid = bool(
            np.all(np.isfinite(predicted))
            and np.isfinite(position_std_m)
            and position_std_m <= float(self.cfg.terminal_kalman_max_position_std_m)
        )
        return {
            "valid": valid,
            "reason": "PASS" if valid else "UNCERTAINTY_TOO_HIGH",
            "age_s": age_s,
            "x_m": x_m,
            "y_m": y_m,
            "vx_mps": vx_mps,
            "vy_mps": vy_mps,
            "radial_m": radial_m,
            "position_std_m": position_std_m,
        }

    @staticmethod
    def _expanded_bbox_xyxy(
        bbox_xyxy: np.ndarray,
        frame_shape: tuple[int, ...],
        scale: float,
    ) -> np.ndarray:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = max(2.0, (x2 - x1) * float(scale))
        bh = max(2.0, (y2 - y1) * float(scale))
        return np.asarray(
            [
                np.clip(cx - 0.5 * bw, 0.0, max(0.0, w - 1.0)),
                np.clip(cy - 0.5 * bh, 0.0, max(0.0, h - 1.0)),
                np.clip(cx + 0.5 * bw, 1.0, float(w)),
                np.clip(cy + 0.5 * bh, 1.0, float(h)),
            ],
            dtype=np.float32,
        )

    def _optical_flow_velocity_norm(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        previous_bbox_xyxy: np.ndarray,
        current_bbox_xyxy: np.ndarray,
        dt: float,
    ) -> tuple[float, float, int, float]:
        """Estimate normalized target ROI motion with forward/backward LK flow."""
        if not bool(self.cfg.optical_flow_enabled):
            return 0.0, 0.0, 0, 0.0
        h, w = previous_gray.shape[:2]
        mask = np.zeros_like(previous_gray, dtype=np.uint8)
        roi = self._expanded_bbox_xyxy(
            previous_bbox_xyxy,
            previous_gray.shape,
            0.88,
        )
        x1, y1, x2, y2 = [int(round(v)) for v in roi]
        if x2 - x1 < 6 or y2 - y1 < 6:
            return 0.0, 0.0, 0, 0.0
        mask[y1:y2, x1:x2] = 255
        points0 = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=max(8, int(self.cfg.optical_flow_max_corners)),
            qualityLevel=max(1.0e-5, float(self.cfg.optical_flow_quality_level)),
            minDistance=max(2.0, float(self.cfg.optical_flow_min_distance_px)),
            mask=mask,
            blockSize=7,
        )
        if points0 is None or len(points0) < int(self.cfg.optical_flow_min_points):
            return 0.0, 0.0, 0, 0.0

        win = max(9, int(self.cfg.optical_flow_window_px))
        if win % 2 == 0:
            win += 1
        lk = dict(
            winSize=(win, win),
            maxLevel=max(0, int(self.cfg.optical_flow_max_level)),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )
        points1, status1, _error1 = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, points0, None, **lk
        )
        if points1 is None or status1 is None:
            return 0.0, 0.0, 0, 0.0
        points0_back, status_back, _error_back = cv2.calcOpticalFlowPyrLK(
            current_gray, previous_gray, points1, None, **lk
        )
        if points0_back is None or status_back is None:
            return 0.0, 0.0, 0, 0.0

        p0 = points0.reshape(-1, 2)
        p1 = points1.reshape(-1, 2)
        p0b = points0_back.reshape(-1, 2)
        valid = (status1.reshape(-1) > 0) & (status_back.reshape(-1) > 0)
        valid &= np.all(np.isfinite(p1), axis=1) & np.all(np.isfinite(p0b), axis=1)
        fb_error = np.linalg.norm(p0 - p0b, axis=1)
        valid &= fb_error <= float(self.cfg.optical_flow_max_forward_backward_error_px)

        current_roi = self._expanded_bbox_xyxy(
            current_bbox_xyxy,
            current_gray.shape,
            float(self.cfg.optical_flow_bbox_expand),
        )
        cx1, cy1, cx2, cy2 = [float(v) for v in current_roi]
        valid &= (
            (p1[:, 0] >= cx1)
            & (p1[:, 0] <= cx2)
            & (p1[:, 1] >= cy1)
            & (p1[:, 1] <= cy2)
        )
        displacement = p1[valid] - p0[valid]
        if displacement.shape[0] < int(self.cfg.optical_flow_min_points):
            return 0.0, 0.0, int(displacement.shape[0]), 0.0

        median = np.median(displacement, axis=0)
        residual = np.linalg.norm(displacement - median.reshape(1, 2), axis=1)
        mad = float(np.median(np.abs(residual - np.median(residual))))
        threshold = max(1.5, 3.5 * 1.4826 * mad)
        inliers = residual <= threshold
        displacement = displacement[inliers]
        if displacement.shape[0] < int(self.cfg.optical_flow_min_points):
            return 0.0, 0.0, int(displacement.shape[0]), 0.0

        median = np.median(displacement, axis=0)
        clip_v = float(self.cfg.optical_flow_max_norm_velocity_per_s)
        vel_x = float(np.clip(median[0] / max(1.0, 0.5 * w) / dt, -clip_v, clip_v))
        vel_y = float(np.clip(median[1] / max(1.0, 0.5 * h) / dt, -clip_v, clip_v))
        count = int(displacement.shape[0])
        confidence = float(np.clip(count / max(1.0, 0.35 * self.cfg.optical_flow_max_corners), 0.0, 1.0))
        return vel_x, vel_y, count, confidence

    def _update_visual_target_motion(
        self,
        frame: np.ndarray,
        bbox_xyxy: Optional[np.ndarray],
        metrics: dict[str, float],
        live_match: bool,
        relative_height_m: float,
        api_state: Any,
    ) -> None:
        """Fuse bbox motion and optical flow into a metric target velocity.

        The target's absolute body-frame velocity is:

            drone ego velocity + target relative velocity from bottom vision.

        This is the only target-velocity estimator used by the controller.
        """
        self._diag_raw_metric_x_m = float("nan")
        self._diag_raw_metric_y_m = float("nan")
        self._diag_external_vx_mps = float("nan")
        self._diag_external_vy_mps = float("nan")
        self._diag_velocity_measurement_valid = False
        now = float(time.monotonic())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ego_vx, ego_vy, yaw = self._body_velocity_from_api_state(api_state)
        pitch, roll, attitude_yaw = self._attitude_from_api_state(api_state)
        if np.isfinite(attitude_yaw):
            yaw = float(attitude_yaw)
        self._drone_body_velocity_x_mps = ego_vx
        self._drone_body_velocity_y_mps = ego_vy

        if self._previous_motion_time is None:
            dt = float(self.cfg.cmd_duration_s)
        else:
            dt = float(now - self._previous_motion_time)
        measurement_max_dt = float(
            getattr(
                self.cfg,
                "visual_motion_measurement_max_dt_s",
                self.cfg.visual_motion_max_dt_s,
            )
        )
        dt_valid = bool(
            np.isfinite(dt)
            and float(self.cfg.visual_motion_min_dt_s)
            <= dt
            <= measurement_max_dt
        )
        if dt_valid:
            self._last_control_dt_s = dt

        attitude_limit = float(
            getattr(self.cfg, "visual_motion_attitude_gate_deg", 12.0)
        )
        max_attitude_deg = max(
            abs(math.degrees(float(pitch))),
            abs(math.degrees(float(roll))),
        )
        attitude_valid = bool(max_attitude_deg <= attitude_limit)
        yaw_rate_dps = 0.0
        if (
            dt_valid
            and self._previous_motion_yaw_rad is not None
            and np.isfinite(float(self._previous_motion_yaw_rad))
        ):
            yaw_delta = math.atan2(
                math.sin(float(yaw) - float(self._previous_motion_yaw_rad)),
                math.cos(float(yaw) - float(self._previous_motion_yaw_rad)),
            )
            yaw_rate_dps = abs(math.degrees(yaw_delta) / max(dt, 1.0e-6))
        yaw_motion_valid = bool(
            yaw_rate_dps
            <= float(getattr(self.cfg, "visual_motion_max_yaw_rate_dps", 45.0))
        )
        self._visual_motion_attitude_valid = bool(attitude_valid and yaw_motion_valid)
        self._visual_motion_yaw_rate_dps = float(yaw_rate_dps)

        if (
            live_match
            and bbox_xyxy is not None
            and np.isfinite(relative_height_m)
            and float(relative_height_m) > 0.05
        ):
            err_x = float(metrics.get("err_x", 0.0) or 0.0)
            err_y = float(metrics.get("err_y", 0.0) or 0.0)
            height = max(0.05, float(relative_height_m))
            tan_h, tan_v = self._camera_projection_tangents(frame.shape)
            relative_x = float(-err_y * height * tan_v)
            relative_y = float(err_x * height * tan_h)
            self._visual_relative_position_body_x_m = relative_x
            self._visual_relative_position_body_y_m = relative_y

            bbox_vel_x_norm = 0.0
            bbox_vel_y_norm = 0.0
            flow_vel_x_norm = 0.0
            flow_vel_y_norm = 0.0
            flow_count = 0
            flow_confidence = 0.0
            relative_vx = 0.0
            relative_vy = 0.0
            measurement_valid = False
            source = "POSITION_ONLY"

            if (
                dt_valid
                and attitude_valid
                and yaw_motion_valid
                and self._previous_motion_bbox_xyxy is not None
                and self._previous_relative_position_body is not None
                and self._previous_motion_gray is not None
            ):
                # Differentiate the semantic landing anchor, not the visible
                # bbox center. The visible bbox is still used below as the LK
                # optical-flow ROI. This prevents edge clipping from appearing
                # as false target motion.
                bbox_vel_x_norm = float(
                    (err_x - float(self._last_control_err_x)) / dt
                )
                bbox_vel_y_norm = float(
                    (err_y - float(self._last_control_err_y)) / dt
                )
                flow_vel_x_norm, flow_vel_y_norm, flow_count, flow_confidence = (
                    self._optical_flow_velocity_norm(
                        self._previous_motion_gray,
                        gray,
                        self._previous_motion_bbox_xyxy,
                        bbox_xyxy,
                        dt,
                    )
                )

                flow_available = bool(
                    flow_count >= int(self.cfg.optical_flow_min_points)
                    and flow_confidence > 0.0
                )
                if flow_available:
                    disagreement = float(
                        math.hypot(
                            flow_vel_x_norm - bbox_vel_x_norm,
                            flow_vel_y_norm - bbox_vel_y_norm,
                        )
                    )
                    if disagreement <= float(
                        self.cfg.optical_flow_max_bbox_disagreement_per_s
                    ):
                        flow_weight = float(
                            np.clip(
                                float(self.cfg.optical_flow_weight)
                                * flow_confidence,
                                0.15,
                                0.80,
                            )
                        )
                    else:
                        flow_weight = 0.15
                    fused_vel_x_norm = float(
                        flow_weight * flow_vel_x_norm
                        + (1.0 - flow_weight) * bbox_vel_x_norm
                    )
                    fused_vel_y_norm = float(
                        flow_weight * flow_vel_y_norm
                        + (1.0 - flow_weight) * bbox_vel_y_norm
                    )
                    source = "BBOX+FLOW"
                else:
                    fused_vel_x_norm = bbox_vel_x_norm
                    fused_vel_y_norm = bbox_vel_y_norm
                    source = "BBOX"

                alpha_img = float(
                    np.clip(self.cfg.horizontal_velocity_ema_alpha, 0.0, 1.0)
                )
                self._control_img_vel_x = float(
                    alpha_img * fused_vel_x_norm
                    + (1.0 - alpha_img) * self._control_img_vel_x
                )
                self._control_img_vel_y = float(
                    alpha_img * fused_vel_y_norm
                    + (1.0 - alpha_img) * self._control_img_vel_y
                )

                previous_x, previous_y = self._previous_relative_position_body
                if (
                    bool(getattr(self.cfg, "visual_motion_yaw_compensation", True))
                    and self._previous_motion_yaw_rad is not None
                ):
                    # Express the previous body-frame target vector in the
                    # current body frame before differentiating it.
                    frame_delta = float(self._previous_motion_yaw_rad) - float(yaw)
                    c_delta = math.cos(frame_delta)
                    s_delta = math.sin(frame_delta)
                    previous_x, previous_y = (
                        c_delta * previous_x - s_delta * previous_y,
                        s_delta * previous_x + c_delta * previous_y,
                    )
                bbox_metric_vx = float((relative_x - previous_x) / dt)
                bbox_metric_vy = float((relative_y - previous_y) / dt)
                previous_height = (
                    height
                    if self._previous_relative_height_m is None
                    else float(self._previous_relative_height_m)
                )
                height_rate = float((height - previous_height) / dt)
                flow_metric_vx = float(
                    -tan_v * (height * fused_vel_y_norm + err_y * height_rate)
                )
                flow_metric_vy = float(
                    tan_h * (height * fused_vel_x_norm + err_x * height_rate)
                )
                metric_weight = 0.55 if source == "BBOX+FLOW" else 0.25
                raw_relative_vx = float(
                    metric_weight * flow_metric_vx
                    + (1.0 - metric_weight) * bbox_metric_vx
                )
                raw_relative_vy = float(
                    metric_weight * flow_metric_vy
                    + (1.0 - metric_weight) * bbox_metric_vy
                )

                max_relative = float(self.cfg.target_velocity_max_valid_mps) + 4.0
                raw_speed = float(math.hypot(raw_relative_vx, raw_relative_vy))
                if np.isfinite(raw_speed) and raw_speed <= max_relative:
                    alpha_rel = float(
                        np.clip(
                            self.cfg.visual_relative_velocity_ema_alpha,
                            0.0,
                            1.0,
                        )
                    )
                    if self._visual_relative_velocity_valid:
                        relative_vx = float(
                            alpha_rel * raw_relative_vx
                            + (1.0 - alpha_rel)
                            * self._visual_relative_velocity_body_x_mps
                        )
                        relative_vy = float(
                            alpha_rel * raw_relative_vy
                            + (1.0 - alpha_rel)
                            * self._visual_relative_velocity_body_y_mps
                        )
                    else:
                        relative_vx = raw_relative_vx
                        relative_vy = raw_relative_vy
                    measurement_valid = True

            self._diag_raw_metric_x_m = float(relative_x)
            self._diag_raw_metric_y_m = float(relative_y)
            self._diag_external_vx_mps = float(relative_vx) if measurement_valid else float("nan")
            self._diag_external_vy_mps = float(relative_vy) if measurement_valid else float("nan")
            self._diag_velocity_measurement_valid = bool(measurement_valid)

            (
                filtered_relative_x,
                filtered_relative_y,
                filtered_relative_vx,
                filtered_relative_vy,
                filtered_velocity_valid,
            ) = self._update_visual_motion_kalman(
                measured_x_m=relative_x,
                measured_y_m=relative_y,
                measured_vx_mps=relative_vx if measurement_valid else None,
                measured_vy_mps=relative_vy if measurement_valid else None,
                dt_s=(
                    min(
                        dt,
                        float(getattr(self.cfg, "visual_motion_kalman_dt_max_s", 0.80)),
                    )
                    if dt_valid
                    else float(self.cfg.cmd_duration_s)
                ),
                flow_confidence=flow_confidence,
                source=source,
            )
            self._visual_relative_position_body_x_m = float(filtered_relative_x)
            self._visual_relative_position_body_y_m = float(filtered_relative_y)
            relative_vx = float(filtered_relative_vx)
            relative_vy = float(filtered_relative_vy)
            velocity_estimate_valid = bool(filtered_velocity_valid)
            if velocity_estimate_valid and source != "POSITION_ONLY":
                source = f"{source}+KALMAN"
            self._visual_motion_source = source

            self._visual_bbox_velocity_x_per_s = float(bbox_vel_x_norm)
            self._visual_bbox_velocity_y_per_s = float(bbox_vel_y_norm)
            self._visual_flow_velocity_x_per_s = float(flow_vel_x_norm)
            self._visual_flow_velocity_y_per_s = float(flow_vel_y_norm)
            self._visual_flow_point_count = int(flow_count)
            self._visual_flow_confidence = float(flow_confidence)

            if velocity_estimate_valid:
                self._visual_relative_velocity_body_x_mps = float(relative_vx)
                self._visual_relative_velocity_body_y_mps = float(relative_vy)
                self._visual_relative_velocity_valid = True
                self._visual_motion_age_s = 0.0

                raw_target_vx = float(ego_vx + relative_vx)
                raw_target_vy = float(ego_vy + relative_vy)
                raw_target_speed = float(math.hypot(raw_target_vx, raw_target_vy))
                if raw_target_speed <= float(self.cfg.target_velocity_max_valid_mps):
                    if self._target_velocity_valid:
                        max_delta = float(
                            max(
                                0.05,
                                self.cfg.target_velocity_max_accel_mps2
                                * min(
                                    dt,
                                    float(getattr(self.cfg, "visual_motion_kalman_dt_max_s", 0.80)),
                                ),
                            )
                        )
                        delta_x = raw_target_vx - self._target_velocity_body_vx_mps
                        delta_y = raw_target_vy - self._target_velocity_body_vy_mps
                        delta_speed = float(math.hypot(delta_x, delta_y))
                        if delta_speed > max_delta:
                            scale = max_delta / max(delta_speed, 1.0e-6)
                            raw_target_vx = self._target_velocity_body_vx_mps + delta_x * scale
                            raw_target_vy = self._target_velocity_body_vy_mps + delta_y * scale
                    alpha_target = float(
                        np.clip(self.cfg.target_velocity_ema_alpha, 0.0, 1.0)
                    )
                    if self._target_velocity_valid:
                        filtered_vx = float(
                            alpha_target * raw_target_vx
                            + (1.0 - alpha_target)
                            * self._target_velocity_body_vx_mps
                        )
                        filtered_vy = float(
                            alpha_target * raw_target_vy
                            + (1.0 - alpha_target)
                            * self._target_velocity_body_vy_mps
                        )
                    else:
                        filtered_vx = raw_target_vx
                        filtered_vy = raw_target_vy
                    self._target_velocity_body_vx_mps = filtered_vx
                    self._target_velocity_body_vy_mps = filtered_vy
                    self._target_velocity_speed_mps = float(
                        math.hypot(filtered_vx, filtered_vy)
                    )
                    # Diagnostic world components are reconstructed from the
                    # visual body estimate; they are not actor API measurements.
                    cos_yaw = math.cos(yaw)
                    sin_yaw = math.sin(yaw)
                    self._target_velocity_world_x_mps = float(
                        cos_yaw * filtered_vx - sin_yaw * filtered_vy
                    )
                    self._target_velocity_world_y_mps = float(
                        sin_yaw * filtered_vx + cos_yaw * filtered_vy
                    )
                    self._target_velocity_valid = True
                    self._target_velocity_age_s = 0.0

            self._previous_motion_gray = gray
            self._previous_motion_bbox_xyxy = bbox_xyxy.copy()
            self._previous_motion_time = now
            self._previous_relative_position_body = (relative_x, relative_y)
            self._previous_relative_height_m = height
            self._previous_motion_yaw_rad = float(yaw)
            self._last_control_err_x = err_x
            self._last_control_err_y = err_y
            self._last_control_had_live_match = True
            self._last_observation_monotonic = now
            return

        # No current LIVE measurement. Preserve the estimate only for a short,
        # bounded prediction interval; never update it from PRED itself.
        if self._previous_motion_time is not None:
            age = max(0.0, float(now - self._previous_motion_time))
        else:
            age = float("inf")
        self._visual_motion_age_s = age
        self._target_velocity_age_s = age
        if age > float(self.cfg.target_velocity_stale_after_s):
            self._target_velocity_valid = False
            self._visual_relative_velocity_valid = False
            self._visual_motion_source = "STALE"
            self._control_img_vel_x = 0.0
            self._control_img_vel_y = 0.0
        self._last_control_had_live_match = False
        self._last_observation_monotonic = now

    def _update_target_body_velocity(self, api_state: Any) -> None:
        """Age the visual estimate; retained for call-site compatibility."""
        if self._previous_motion_time is None:
            self._target_velocity_age_s = float("inf")
            self._target_velocity_valid = False
            return
        age = max(0.0, float(time.monotonic() - self._previous_motion_time))
        self._target_velocity_age_s = age
        self._visual_motion_age_s = age
        if age > float(self.cfg.target_velocity_stale_after_s):
            self._target_velocity_valid = False

    def _get_obstacles(self, api_altitude_m: float, relative_height_m: float) -> dict[str, Any]:
        max_d = float(self.cfg.lidar_max_range_m)
        try:
            data = self.client.getLidarData(
                lidar_name=self.cfg.lidar_sensor_name,
                vehicle_name=self.cfg.vehicle_name,
            )
            points = point_cloud_to_array(getattr(data, "point_cloud", []))
            result = self.lidar_processor.compute_sector_distances(points, altitude_fallback_m=api_altitude_m)
        except Exception:
            result = {
                "front_dist_m": max_d,
                "front_left_dist_m": max_d,
                "front_right_dist_m": max_d,
                "left_dist_m": max_d,
                "right_dist_m": max_d,
                "back_dist_m": max_d,
                "down_dist_m": max(0.0, float(relative_height_m)),
                "min_obstacle_dist_m": max_d,
                "lidar_valid": False,
                "lidar_point_count": 0,
                "obstacle_source": "api_z_only_fallback",
            }

        # Vertical observation uses only the API NED separation to the target.
        result["down_dist_m"] = max(0.0, float(relative_height_m))
        result["obstacle_source"] = "lidar_horizontal+api_z_vertical"
        return result

    def _stabilize_control_bbox(
        self,
        bbox_xyxy: Optional[np.ndarray],
        *,
        live_match: bool,
        frame_shape: tuple[int, ...],
    ) -> Optional[np.ndarray]:
        """Return a stable control bbox without shifting it away from the target.

        ``_xyxy_to_xywh`` is a tracker helper and returns top-left ``x, y``.
        The previous implementation accidentally interpreted those values as
        center ``cx, cy``. That moved the control box by roughly half its width
        and height and became especially visible when the vehicle approached an
        image edge.

        Center and size are filtered independently here. The current verified
        LIVE center remains responsive, while median/EMA filtering is retained
        for detector width/height changes. Edge-clipped sizes never suppress a
        valid center update.
        """
        if not bool(self.cfg.control_bbox_filter_enabled):
            self._control_bbox_filter_mode = "DISABLED"
            return (
                None
                if bbox_xyxy is None
                else np.asarray(bbox_xyxy, dtype=np.float32).copy()
            )

        if bbox_xyxy is None or not bool(live_match):
            self._control_bbox_outlier_suppressed = False
            self._control_bbox_filter_mode = "HOLD_PRED"
            return (
                None
                if self._control_bbox_xyxy is None
                else self._control_bbox_xyxy.copy()
            )

        raw = self._validated_xyxy(bbox_xyxy, frame_shape)
        if raw is None:
            self._control_bbox_filter_mode = "HOLD_INVALID"
            return (
                None
                if self._control_bbox_xyxy is None
                else self._control_bbox_xyxy.copy()
            )

        raw_cxcywh = self._xyxy_to_cxcywh(raw)
        h, w = frame_shape[:2]
        edge_margin = max(1.0, float(self.cfg.control_bbox_edge_margin_px))
        touches_left = bool(float(raw[0]) <= edge_margin)
        touches_top = bool(float(raw[1]) <= edge_margin)
        touches_right = bool(float(raw[2]) >= float(w) - edge_margin)
        touches_bottom = bool(float(raw[3]) >= float(h) - edge_margin)
        edge_clipped = bool(
            touches_left or touches_top or touches_right or touches_bottom
        )

        # Keep robust history for size only. A bbox clipped by the image border
        # has a truncated visible size and must not teach the filter that the
        # physical target suddenly became smaller.
        history = list(getattr(self, "_control_bbox_history", []))
        if not edge_clipped or not history:
            history.append(raw_cxcywh.copy())
            keep = max(1, int(self.cfg.control_bbox_history_size))
            history = history[-keep:]
            self._control_bbox_history = history
        robust_size = (
            np.median(np.stack(history, axis=0), axis=0)[2:4]
            if history
            else raw_cxcywh[2:4].copy()
        )

        if self._control_bbox_xyxy is None:
            filtered_center = raw_cxcywh[0:2].copy()
            filtered_size = raw_cxcywh[2:4].copy()
            center_outlier = False
            size_outlier = False
            self._control_bbox_filter_mode = (
                "EDGE_INIT" if edge_clipped else "TRACK_INIT"
            )
        else:
            previous = self._xyxy_to_cxcywh(self._control_bbox_xyxy)
            frame_diag = max(1.0, math.hypot(float(w), float(h)))
            center_jump_norm = float(
                np.linalg.norm(raw_cxcywh[0:2] - previous[0:2]) / frame_diag
            )
            width_ratio = abs(float(robust_size[0] - previous[2])) / max(
                1.0, float(previous[2])
            )
            height_ratio = abs(float(robust_size[1] - previous[3])) / max(
                1.0, float(previous[3])
            )

            # Reaching an image edge is expected motion, not a bad center. A
            # large non-edge jump remains strongly suppressed as before.
            center_outlier = bool(
                not edge_clipped
                and center_jump_norm
                > float(self.cfg.control_bbox_outlier_center_jump_norm)
            )
            size_outlier = bool(
                max(width_ratio, height_ratio)
                > float(self.cfg.control_bbox_outlier_size_ratio)
            )
            if edge_clipped:
                center_alpha = float(self.cfg.control_bbox_edge_center_alpha)
            elif center_outlier:
                center_alpha = float(self.cfg.control_bbox_outlier_alpha)
            else:
                center_alpha = float(self.cfg.control_bbox_center_alpha)
            size_alpha = float(
                self.cfg.control_bbox_outlier_alpha
                if size_outlier
                else self.cfg.control_bbox_size_alpha
            )
            center_alpha = float(np.clip(center_alpha, 0.0, 1.0))
            size_alpha = float(np.clip(size_alpha, 0.0, 1.0))
            filtered_center = (
                center_alpha * raw_cxcywh[0:2]
                + (1.0 - center_alpha) * previous[0:2]
            )
            filtered_size = (
                size_alpha * robust_size
                + (1.0 - size_alpha) * previous[2:4]
            )
            if edge_clipped:
                self._control_bbox_filter_mode = "EDGE_FOLLOW"
            elif center_outlier:
                self._control_bbox_filter_mode = "CENTER_GUARD"
            elif size_outlier:
                self._control_bbox_filter_mode = "SIZE_GUARD"
            else:
                self._control_bbox_filter_mode = "TRACK"

        self._control_bbox_outlier_suppressed = bool(
            center_outlier or size_outlier
        )

        cx, cy = [float(v) for v in filtered_center]
        bw, bh = [float(v) for v in filtered_size]
        cx = float(np.clip(cx, 1.0, max(1.0, float(w) - 1.0)))
        cy = float(np.clip(cy, 1.0, max(1.0, float(h) - 1.0)))
        bw = float(np.clip(bw, 2.0, max(2.0, float(w))))
        bh = float(np.clip(bh, 2.0, max(2.0, float(h))))

        # Preserve the filtered center at an image edge. The old code shifted
        # the center inward until the full historical size fit in the frame,
        # leaving the rectangle on the road. Shrink only the visible extent.
        visible_bw = min(bw, max(2.0, 2.0 * min(cx, float(w) - cx)))
        visible_bh = min(bh, max(2.0, 2.0 * min(cy, float(h) - cy)))
        filtered_xyxy = np.asarray(
            [
                cx - 0.5 * visible_bw,
                cy - 0.5 * visible_bh,
                cx + 0.5 * visible_bw,
                cy + 0.5 * visible_bh,
            ],
            dtype=np.float32,
        )
        filtered_xyxy = self._validated_xyxy(filtered_xyxy, frame_shape)
        if filtered_xyxy is None:
            filtered_xyxy = raw.copy()
        self._control_bbox_xyxy = filtered_xyxy.copy()
        return filtered_xyxy

    def _update_landing_anchor_geometry(
        self,
        *,
        raw_bbox_xyxy: Optional[np.ndarray],
        control_bbox_xyxy: Optional[np.ndarray],
        frame_shape: tuple[int, ...],
        live_match: bool,
        relative_height_m: float,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return a stable roof anchor and virtual full-target bbox.

        YOLO boxes describe only the visible part of an object. Near touchdown,
        a moving car commonly crosses an image edge, so the visible bbox center
        drifts toward the remaining crop even though the physical car center did
        not move by the same amount. This method preserves a full-target size and
        aspect estimate from unclipped LIVE frames, reconstructs the hidden
        extent at an edge, and lets the semantic landing point move outside the
        image when that is where the car center actually lies.

        The returned virtual bbox is used only for geometry. Identity crops and
        optical flow continue to consume the validated visible control bbox.
        """
        control = self._validated_xyxy(control_bbox_xyxy, frame_shape)
        raw = self._validated_xyxy(raw_bbox_xyxy, frame_shape)
        if control is None:
            self._landing_anchor_mode = "HOLD_NO_BBOX"
            return (
                None if self._landing_anchor_px is None else self._landing_anchor_px.copy(),
                None
                if self._landing_anchor_virtual_bbox_xyxy is None
                else self._landing_anchor_virtual_bbox_xyxy.copy(),
            )

        if not bool(self.cfg.landing_anchor_enabled):
            cxcywh = self._xyxy_to_cxcywh(control)
            self._landing_anchor_px = cxcywh[0:2].astype(np.float32)
            self._landing_anchor_full_size_px = cxcywh[2:4].astype(np.float32)
            self._landing_anchor_virtual_bbox_xyxy = control.copy()
            self._landing_anchor_mode = "DISABLED_BBOX_CENTER"
            self._landing_anchor_edge_flags = "NONE"
            return self._landing_anchor_px.copy(), control.copy()

        if not bool(live_match):
            self._landing_anchor_mode = "HOLD_PRED"
            return (
                None if self._landing_anchor_px is None else self._landing_anchor_px.copy(),
                None
                if self._landing_anchor_virtual_bbox_xyxy is None
                else self._landing_anchor_virtual_bbox_xyxy.copy(),
            )

        visible = control if raw is None else raw
        h, w = frame_shape[:2]
        edge_margin = max(1.0, float(self.cfg.control_bbox_edge_margin_px))
        left = bool(float(visible[0]) <= edge_margin)
        top = bool(float(visible[1]) <= edge_margin)
        right = bool(float(visible[2]) >= float(w) - edge_margin)
        bottom = bool(float(visible[3]) >= float(h) - edge_margin)
        edge_clipped = bool(left or top or right or bottom)
        edge_flags = "".join(
            name for name, active in (("L", left), ("T", top), ("R", right), ("B", bottom)) if active
        ) or "NONE"

        control_cxcywh = self._xyxy_to_cxcywh(control)
        visible_cxcywh = self._xyxy_to_cxcywh(visible)
        control_center = control_cxcywh[0:2].astype(np.float64)
        control_size = np.maximum(control_cxcywh[2:4].astype(np.float64), 2.0)
        visible_size = np.maximum(visible_cxcywh[2:4].astype(np.float64), 2.0)

        if self._landing_anchor_full_size_px is None:
            self._landing_anchor_full_size_px = control_size.astype(np.float32)
        if self._landing_anchor_aspect_ratio is None:
            self._landing_anchor_aspect_ratio = float(
                np.clip(control_size[0] / max(2.0, control_size[1]), 0.05, 20.0)
            )

        aspect = float(self._landing_anchor_aspect_ratio)
        full_size_prev = np.maximum(
            np.asarray(self._landing_anchor_full_size_px, dtype=np.float64), 2.0
        )

        if not edge_clipped:
            size_measurement = control_size
            size_alpha = float(np.clip(self.cfg.landing_anchor_size_ema_alpha, 0.0, 1.0))
            full_size = size_alpha * size_measurement + (1.0 - size_alpha) * full_size_prev
            measured_aspect = float(
                np.clip(full_size[0] / max(2.0, full_size[1]), 0.05, 20.0)
            )
            aspect_alpha = float(
                np.clip(self.cfg.landing_anchor_aspect_ema_alpha, 0.0, 1.0)
            )
            aspect = float(
                aspect_alpha * measured_aspect
                + (1.0 - aspect_alpha) * aspect
            )
            full_center = control_center
            self._landing_anchor_reference_size_px = full_size.astype(np.float32)
            if np.isfinite(relative_height_m) and float(relative_height_m) > 0.05:
                self._landing_anchor_reference_height_m = float(relative_height_m)
            mode = "TRACK_FULL"
        else:
            horizontal_unclipped = bool(not left and not right)
            vertical_unclipped = bool(not top and not bottom)
            candidate_size = full_size_prev.copy()

            if horizontal_unclipped:
                candidate_size[0] = visible_size[0]
                candidate_size[1] = max(visible_size[1], candidate_size[0] / max(aspect, 1.0e-6))
            elif vertical_unclipped:
                candidate_size[1] = visible_size[1]
                candidate_size[0] = max(visible_size[0], candidate_size[1] * aspect)
            elif (
                self._landing_anchor_reference_size_px is not None
                and self._landing_anchor_reference_height_m is not None
                and np.isfinite(relative_height_m)
                and float(relative_height_m) > 0.05
            ):
                scale = float(
                    np.clip(
                        float(self._landing_anchor_reference_height_m)
                        / float(relative_height_m),
                        0.70,
                        3.00,
                    )
                )
                candidate_size = (
                    np.asarray(self._landing_anchor_reference_size_px, dtype=np.float64)
                    * scale
                )

            candidate_size = np.maximum(candidate_size, visible_size)
            # At an edge, shrinking is usually clipping rather than a real target
            # scale change. Follow growth quickly and shrink only weakly.
            growth = candidate_size >= full_size_prev
            alpha_grow = float(
                np.clip(max(0.65, self.cfg.landing_anchor_size_ema_alpha), 0.0, 1.0)
            )
            alpha_shrink = float(
                np.clip(min(0.15, self.cfg.landing_anchor_size_ema_alpha), 0.0, 1.0)
            )
            size_alpha = np.where(growth, alpha_grow, alpha_shrink)
            full_size = size_alpha * candidate_size + (1.0 - size_alpha) * full_size_prev
            full_size = np.maximum(full_size, visible_size)

            u = float(np.clip(self.cfg.landing_anchor_u, 0.0, 1.0))
            v = float(np.clip(self.cfg.landing_anchor_v, 0.0, 1.0))
            previous_anchor = (
                None
                if self._landing_anchor_px is None
                else np.asarray(self._landing_anchor_px, dtype=np.float64)
            )
            previous_full_center = (
                control_center
                if previous_anchor is None
                else previous_anchor
                - np.asarray([(u - 0.5) * full_size[0], (v - 0.5) * full_size[1]])
            )

            if left and not right:
                full_cx = float(visible[2]) - 0.5 * float(full_size[0])
            elif right and not left:
                full_cx = float(visible[0]) + 0.5 * float(full_size[0])
            elif not left and not right:
                full_cx = float(control_center[0])
            else:
                full_cx = float(previous_full_center[0])

            if top and not bottom:
                full_cy = float(visible[3]) - 0.5 * float(full_size[1])
            elif bottom and not top:
                full_cy = float(visible[1]) + 0.5 * float(full_size[1])
            elif not top and not bottom:
                full_cy = float(control_center[1])
            else:
                full_cy = float(previous_full_center[1])

            full_center = np.asarray([full_cx, full_cy], dtype=np.float64)
            mode = f"EDGE_RECON_{edge_flags}"

        self._landing_anchor_aspect_ratio = float(aspect)
        self._landing_anchor_full_size_px = np.asarray(full_size, dtype=np.float32)
        self._landing_anchor_edge_flags = edge_flags

        u = float(np.clip(self.cfg.landing_anchor_u, 0.0, 1.0))
        v = float(np.clip(self.cfg.landing_anchor_v, 0.0, 1.0))
        measured_anchor = np.asarray(
            [
                float(full_center[0]) + (u - 0.5) * float(full_size[0]),
                float(full_center[1]) + (v - 0.5) * float(full_size[1]),
            ],
            dtype=np.float64,
        )
        alpha = float(
            np.clip(
                self.cfg.landing_anchor_edge_center_alpha
                if edge_clipped
                else self.cfg.landing_anchor_center_alpha,
                0.0,
                1.0,
            )
        )
        if self._landing_anchor_px is None:
            anchor = measured_anchor
        else:
            anchor = (
                alpha * measured_anchor
                + (1.0 - alpha)
                * np.asarray(self._landing_anchor_px, dtype=np.float64)
            )

        outside = max(0.0, float(self.cfg.landing_anchor_max_outside_frame_ratio))
        anchor[0] = float(np.clip(anchor[0], -outside * w, (1.0 + outside) * w))
        anchor[1] = float(np.clip(anchor[1], -outside * h, (1.0 + outside) * h))
        virtual_center = anchor - np.asarray(
            [(u - 0.5) * full_size[0], (v - 0.5) * full_size[1]],
            dtype=np.float64,
        )
        virtual_bbox = np.asarray(
            [
                virtual_center[0] - 0.5 * full_size[0],
                virtual_center[1] - 0.5 * full_size[1],
                virtual_center[0] + 0.5 * full_size[0],
                virtual_center[1] + 0.5 * full_size[1],
            ],
            dtype=np.float32,
        )

        self._landing_anchor_px = anchor.astype(np.float32)
        self._landing_anchor_virtual_bbox_xyxy = virtual_bbox.copy()
        self._landing_anchor_mode = mode
        return self._landing_anchor_px.copy(), virtual_bbox.copy()

    @staticmethod
    def _bbox_is_edge_clipped(
        bbox_xyxy: Optional[np.ndarray],
        frame_shape: tuple[int, ...],
        margin_px: float,
    ) -> bool:
        """Return True when any visible BBox edge touches the image boundary."""
        if bbox_xyxy is None:
            return True
        arr = np.asarray(bbox_xyxy, dtype=np.float64).reshape(-1)
        if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
            return True
        h, w = frame_shape[:2]
        margin = max(0.0, float(margin_px))
        return bool(
            arr[0] <= margin
            or arr[1] <= margin
            or arr[2] >= float(w) - margin
            or arr[3] >= float(h) - margin
        )

    def _terminal_anchor_flow_step(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        anchor_px: np.ndarray,
        support_bbox_xyxy: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], int, float]:
        """Propagate a locked roof point with robust local LK optical flow."""
        if previous_gray is None or current_gray is None or anchor_px is None:
            return None, 0, 0.0
        if previous_gray.shape[:2] != current_gray.shape[:2]:
            return None, 0, 0.0

        h, w = previous_gray.shape[:2]
        anchor = np.asarray(anchor_px, dtype=np.float32).reshape(-1)
        if anchor.size < 2 or not np.all(np.isfinite(anchor[:2])):
            return None, 0, 0.0

        if support_bbox_xyxy is not None:
            support = np.asarray(support_bbox_xyxy, dtype=np.float64).reshape(-1)
            support_w = max(2.0, float(support[2] - support[0]))
            support_h = max(2.0, float(support[3] - support[1]))
            radius = float(self.cfg.terminal_anchor_flow_patch_radius_ratio) * min(
                support_w, support_h
            )
        else:
            radius = float(self.cfg.terminal_anchor_flow_patch_min_px)
        radius = int(
            round(
                np.clip(
                    radius,
                    int(self.cfg.terminal_anchor_flow_patch_min_px),
                    int(self.cfg.terminal_anchor_flow_patch_max_px),
                )
            )
        )

        center_x = int(round(float(np.clip(anchor[0], 0.0, max(0.0, w - 1.0)))))
        center_y = int(round(float(np.clip(anchor[1], 0.0, max(0.0, h - 1.0)))))
        mask = np.zeros_like(previous_gray, dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), max(8, radius), 255, -1)

        points0 = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=max(8, int(self.cfg.terminal_anchor_flow_max_corners)),
            qualityLevel=max(1.0e-5, float(self.cfg.terminal_anchor_flow_quality_level)),
            minDistance=max(2.0, float(self.cfg.terminal_anchor_flow_min_distance_px)),
            mask=mask,
            blockSize=7,
        )
        min_points = max(3, int(self.cfg.terminal_anchor_flow_min_points))
        if points0 is None or len(points0) < min_points:
            return None, 0, 0.0

        win = max(9, int(self.cfg.terminal_anchor_flow_window_px))
        if win % 2 == 0:
            win += 1
        lk = dict(
            winSize=(win, win),
            maxLevel=max(0, int(self.cfg.terminal_anchor_flow_max_level)),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01),
        )
        points1, status1, _error1 = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, points0, None, **lk
        )
        if points1 is None or status1 is None:
            return None, 0, 0.0
        points0_back, status_back, _error_back = cv2.calcOpticalFlowPyrLK(
            current_gray, previous_gray, points1, None, **lk
        )
        if points0_back is None or status_back is None:
            return None, 0, 0.0

        p0 = points0.reshape(-1, 2)
        p1 = points1.reshape(-1, 2)
        p0_back = points0_back.reshape(-1, 2)
        valid = (status1.reshape(-1) > 0) & (status_back.reshape(-1) > 0)
        valid &= np.all(np.isfinite(p1), axis=1) & np.all(np.isfinite(p0_back), axis=1)
        forward_backward_error = np.linalg.norm(p0 - p0_back, axis=1)
        valid &= forward_backward_error <= float(
            self.cfg.terminal_anchor_flow_max_forward_backward_error_px
        )

        displacement = p1[valid] - p0[valid]
        if displacement.shape[0] < min_points:
            return None, int(displacement.shape[0]), 0.0

        median_displacement = np.median(displacement, axis=0)
        residual = np.linalg.norm(
            displacement - median_displacement.reshape(1, 2), axis=1
        )
        median_residual = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median_residual)))
        inlier_threshold = max(1.5, median_residual + 3.5 * 1.4826 * mad)
        inliers = residual <= inlier_threshold
        displacement = displacement[inliers]
        if displacement.shape[0] < min_points:
            return None, int(displacement.shape[0]), 0.0

        median_displacement = np.median(displacement, axis=0)
        step_norm = float(
            np.linalg.norm(median_displacement)
            / max(1.0, float(np.hypot(w, h)))
        )
        if step_norm > float(self.cfg.terminal_anchor_flow_max_step_norm):
            return None, int(displacement.shape[0]), 0.0

        next_anchor = anchor[:2].astype(np.float64) + median_displacement.astype(np.float64)
        outside = max(0.0, float(self.cfg.landing_anchor_max_outside_frame_ratio))
        next_anchor[0] = float(np.clip(next_anchor[0], -outside * w, (1.0 + outside) * w))
        next_anchor[1] = float(np.clip(next_anchor[1], -outside * h, (1.0 + outside) * h))
        confidence = float(
            np.clip(
                displacement.shape[0]
                / max(float(min_points), float(self.cfg.terminal_anchor_flow_max_corners)),
                0.0,
                1.0,
            )
        )
        return next_anchor.astype(np.float32), int(displacement.shape[0]), confidence

    def _update_terminal_landing_anchor(
        self,
        *,
        frame: np.ndarray,
        measured_anchor_px: Optional[np.ndarray],
        raw_bbox_xyxy: Optional[np.ndarray],
        control_bbox_xyxy: Optional[np.ndarray],
        live_match: bool,
        similarity: float,
        relative_height_m: float,
    ) -> tuple[Optional[np.ndarray], bool, str]:
        """Acquire once, then follow the physical roof point independently."""
        if not bool(self.cfg.terminal_anchor_enabled):
            return measured_anchor_px, False, "DISABLED"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        measured = None
        if measured_anchor_px is not None:
            candidate = np.asarray(measured_anchor_px, dtype=np.float32).reshape(-1)
            if candidate.size >= 2 and np.all(np.isfinite(candidate[:2])):
                measured = candidate[:2].copy()

        raw = self._validated_xyxy(raw_bbox_xyxy, frame.shape)
        control = self._validated_xyxy(control_bbox_xyxy, frame.shape)
        visible = raw if raw is not None else control
        visible_metrics = self._bbox_metrics(visible, frame.shape)
        edge_clipped = self._bbox_is_edge_clipped(
            visible,
            frame.shape,
            float(self.cfg.terminal_anchor_edge_margin_px),
        )
        live_confirmed = bool(
            live_match
            and int(getattr(self, "_live_match_streak", 0))
            >= max(
                int(self.cfg.match_confirmation_steps),
                int(self.cfg.terminal_anchor_min_live_streak),
            )
            and float(similarity) >= float(self.cfg.terminal_anchor_lock_min_similarity)
        )

        if not bool(getattr(self, "_terminal_anchor_locked", False)):
            eligible = bool(
                live_confirmed
                and measured is not None
                and visible is not None
                and not edge_clipped
                and np.isfinite(relative_height_m)
                and float(self.cfg.terminal_anchor_lock_min_height_m)
                <= float(relative_height_m)
                <= float(self.cfg.terminal_anchor_lock_max_height_m)
                and float(visible_metrics["center_error"])
                <= float(self.cfg.terminal_anchor_lock_max_center_error)
                and float(self.cfg.terminal_anchor_lock_min_area_norm)
                <= float(visible_metrics["area_norm"])
                <= float(self.cfg.terminal_anchor_lock_max_area_norm)
            )
            if eligible:
                previous_candidate = getattr(self, "_terminal_anchor_candidate_px", None)
                if previous_candidate is None:
                    stable = True
                else:
                    stable = bool(
                        float(np.linalg.norm(measured - previous_candidate))
                        / max(1.0, float(np.hypot(w, h)))
                        <= float(self.cfg.terminal_anchor_candidate_max_jump_norm)
                    )
                if stable:
                    self._terminal_anchor_acquire_streak = int(
                        getattr(self, "_terminal_anchor_acquire_streak", 0)
                    ) + 1
                else:
                    self._terminal_anchor_acquire_streak = 1
                self._terminal_anchor_candidate_px = measured.copy()
            else:
                self._terminal_anchor_acquire_streak = 0
                self._terminal_anchor_candidate_px = None

            if (
                eligible
                and int(self._terminal_anchor_acquire_streak)
                >= max(1, int(self.cfg.terminal_anchor_acquire_streak_required))
            ):
                self._terminal_anchor_locked = True
                self._terminal_anchor_px = measured.copy()
                self._terminal_anchor_previous_gray = gray.copy()
                self._terminal_anchor_support_bbox_xyxy = (
                    None if visible is None else visible.copy()
                )
                self._terminal_anchor_lock_step = int(self._step)
                self._terminal_anchor_last_identity_step = int(self._step)
                self._terminal_anchor_age_steps = 0
                self._terminal_anchor_source = "TERMINAL_ANCHOR_LOCK"
                self._terminal_anchor_flow_points = 0
                self._terminal_anchor_flow_confidence = 1.0
                return self._terminal_anchor_px.copy(), True, self._terminal_anchor_source

            self._terminal_anchor_previous_gray = gray.copy()
            return measured_anchor_px, False, "WAIT_LOCK"

        # Locked mode: propagate the selected roof point before considering any
        # new detector geometry. A detector BBox may correct only a small, trusted
        # flow drift; it can never replace the terminal point with a distant crop.
        previous_gray = getattr(self, "_terminal_anchor_previous_gray", None)
        previous_anchor = getattr(self, "_terminal_anchor_px", None)
        support_bbox = getattr(self, "_terminal_anchor_support_bbox_xyxy", None)
        flow_anchor, flow_points, flow_confidence = self._terminal_anchor_flow_step(
            previous_gray,
            gray,
            previous_anchor,
            support_bbox,
        )
        self._terminal_anchor_flow_points = int(flow_points)
        self._terminal_anchor_flow_confidence = float(flow_confidence)

        flow_valid = flow_anchor is not None
        if flow_valid:
            next_anchor = np.asarray(flow_anchor, dtype=np.float32)
            self._terminal_anchor_age_steps = 0
            source = "TERMINAL_ANCHOR_FLOW"
        else:
            next_anchor = np.asarray(previous_anchor, dtype=np.float32).copy()
            self._terminal_anchor_age_steps = int(
                getattr(self, "_terminal_anchor_age_steps", 0)
            ) + 1
            source = "TERMINAL_ANCHOR_HOLD"

        trusted_live_correction = False
        if live_confirmed and measured is not None and not edge_clipped:
            correction_jump = float(
                np.linalg.norm(measured - next_anchor)
                / max(1.0, float(np.hypot(w, h)))
            )
            if correction_jump <= float(
                self.cfg.terminal_anchor_live_correction_max_jump_norm
            ):
                alpha = float(
                    np.clip(self.cfg.terminal_anchor_live_correction_alpha, 0.0, 1.0)
                )
                next_anchor = (
                    (1.0 - alpha) * next_anchor.astype(np.float64)
                    + alpha * measured.astype(np.float64)
                ).astype(np.float32)
                self._terminal_anchor_last_identity_step = int(self._step)
                self._terminal_anchor_support_bbox_xyxy = (
                    None if visible is None else visible.copy()
                )
                trusted_live_correction = True
                source = (
                    "TERMINAL_ANCHOR_FLOW_LIVE"
                    if flow_valid
                    else "TERMINAL_ANCHOR_LIVE_RECOVERY"
                )
                self._terminal_anchor_age_steps = 0
        elif live_match and flow_valid:
            # The identity network still sees the selected object, while the BBox
            # geometry is clipped or unstable. Refresh identity age only; never
            # pull the anchor toward that BBox.
            self._terminal_anchor_last_identity_step = int(self._step)

        self._terminal_anchor_px = next_anchor.copy()
        self._terminal_anchor_previous_gray = gray.copy()
        self._terminal_anchor_source = source
        identity_age = int(self._step) - int(
            getattr(self, "_terminal_anchor_last_identity_step", -999999)
        )
        valid = bool(
            int(self._terminal_anchor_age_steps)
            <= int(self.cfg.terminal_anchor_max_hold_steps)
            and identity_age <= int(self.cfg.terminal_anchor_identity_grace_steps)
        )
        if not valid:
            self._terminal_anchor_source = f"{source}_STALE"
        return self._terminal_anchor_px.copy(), valid, self._terminal_anchor_source

    @staticmethod
    def _draw_double_v_marker(
        image: np.ndarray,
        point_xy: tuple[int, int],
        color: tuple[int, int, int] = (255, 0, 255),
    ) -> None:
        """Draw two stacked downward chevrons; the lower tip is the exact point."""
        x, y = [int(value) for value in point_xy]
        for vertical_offset in (0, -16):
            tip_y = y + vertical_offset
            cv2.line(image, (x - 12, tip_y - 13), (x, tip_y), color, 3, cv2.LINE_AA)
            cv2.line(image, (x + 12, tip_y - 13), (x, tip_y), color, 3, cv2.LINE_AA)

    @staticmethod
    def _draw_frame_center_marker(
        image: np.ndarray,
        point_xy: tuple[int, int],
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> None:
        """Draw a plus sign surrounded by a circle at the calibrated frame center."""
        x, y = [int(value) for value in point_xy]
        cv2.circle(image, (x, y), 18, color, 2, cv2.LINE_AA)
        cv2.line(image, (x - 12, y), (x + 12, y), color, 3, cv2.LINE_AA)
        cv2.line(image, (x, y - 12), (x, y + 12), color, 3, cv2.LINE_AA)

    def _landing_anchor_metrics(
        self,
        *,
        anchor_px: Optional[np.ndarray],
        virtual_bbox_xyxy: Optional[np.ndarray],
        visible_bbox_xyxy: Optional[np.ndarray],
        frame_shape: tuple[int, ...],
    ) -> dict[str, float]:
        """Compute controller geometry from the semantic anchor."""
        if anchor_px is None or virtual_bbox_xyxy is None:
            return self._bbox_metrics(visible_bbox_xyxy, frame_shape)
        h, w = frame_shape[:2]
        anchor = np.asarray(anchor_px, dtype=np.float64).reshape(-1)
        virtual = np.asarray(virtual_bbox_xyxy, dtype=np.float64).reshape(-1)
        if anchor.size < 2 or virtual.size < 4 or not np.all(np.isfinite(anchor[:2])):
            return self._bbox_metrics(visible_bbox_xyxy, frame_shape)
        full_w = max(2.0, float(virtual[2] - virtual[0]))
        full_h = max(2.0, float(virtual[3] - virtual[1]))
        target_x_px = float(getattr(self, "_bottom_center_target_x_norm", 0.5)) * float(w)
        target_y_px = float(getattr(self, "_bottom_center_target_y_norm", 0.5)) * float(h)
        err_x = (float(anchor[0]) - target_x_px) / max(1.0, 0.5 * w)
        err_y = (float(anchor[1]) - target_y_px) / max(1.0, 0.5 * h)
        rel_x = abs(err_x) / max(1.0e-6, full_w / max(1.0, w))
        rel_y = abs(err_y) / max(1.0e-6, full_h / max(1.0, h))
        visible_metrics = self._bbox_metrics(visible_bbox_xyxy, frame_shape)
        return {
            "err_x": float(err_x),
            "err_y": float(err_y),
            "center_error": float(math.hypot(err_x, err_y)),
            "bbox_rel_error": float(max(rel_x, rel_y)),
            # Preserve the existing visual-scale speed schedule. Only the center
            # reference changes; target area still comes from the visible bbox.
            "area_norm": float(visible_metrics["area_norm"]),
        }

    def _bbox_metrics(
        self,
        bbox_xyxy: Optional[np.ndarray],
        frame_shape: tuple[int, ...],
    ) -> dict[str, float]:
        if bbox_xyxy is None:
            return {
                "err_x": 0.0,
                "err_y": 0.0,
                "center_error": 999.0,
                "bbox_rel_error": 999.0,
                "area_norm": 0.0,
            }
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        target_x_px = float(getattr(self, "_bottom_center_target_x_norm", 0.5)) * float(w)
        target_y_px = float(getattr(self, "_bottom_center_target_y_norm", 0.5)) * float(h)
        err_x = (cx - target_x_px) / max(1.0, 0.5 * w)
        err_y = (cy - target_y_px) / max(1.0, 0.5 * h)
        rel_x = abs(err_x) / max(1e-6, bw / max(1.0, w))
        rel_y = abs(err_y) / max(1e-6, bh / max(1.0, h))
        return {
            "err_x": float(err_x),
            "err_y": float(err_y),
            "center_error": float(math.hypot(err_x, err_y)),
            "bbox_rel_error": float(max(rel_x, rel_y)),
            "area_norm": float((bw * bh) / max(1.0, w * h)),
        }

    def _observe(self, frame: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict[str, Any]]:
        frame = self._get_bottom_frame() if frame is None else frame
        frame_capture_monotonic = float(time.monotonic())
        self._sync_image_geometry(frame)
        self._last_bottom_frame = frame.copy()
        raw_bbox_xyxy, similarity, tracker_mode = self._strict_track(frame)
        live_match = bool(tracker_mode == "MATCH")
        bbox_xyxy = self._stabilize_control_bbox(
            raw_bbox_xyxy,
            live_match=live_match,
            frame_shape=frame.shape,
        )
        self._last_bottom_observation_monotonic = frame_capture_monotonic

        # The moving platform may change world Z on uneven roads. Refresh its
        # API pose before every vertical-state calculation. No actor movement is
        # performed here; this is a read-only query.
        if self._target_actor_name:
            self._read_target_surface_altitude()

        drone_state, api_relative_height, api_state = self._get_api_state()
        range_state = self.range_finder_array.read(
            self.client, self.cfg.vehicle_name
        )
        relative_height, landing_height_source, range_height_used = (
            self._select_effective_landing_height(
                live_match=live_match,
                api_relative_height_m=api_relative_height,
                range_state=range_state,
                now_monotonic=frame_capture_monotonic,
            )
        )
        # Agent 2's altitude feature represents height over the active landing
        # surface. In vision-degraded SENSOR_FINAL this is the calibrated
        # range-array median; global World-Z remains available separately.
        drone_state.altitude_m = max(0.0, float(relative_height))
        measured_landing_anchor_px, virtual_target_bbox_xyxy = (
            self._update_landing_anchor_geometry(
                raw_bbox_xyxy=raw_bbox_xyxy,
                control_bbox_xyxy=bbox_xyxy,
                frame_shape=frame.shape,
                live_match=live_match,
                relative_height_m=relative_height,
            )
        )
        (
            terminal_anchor_px,
            terminal_anchor_valid,
            terminal_anchor_source,
        ) = self._update_terminal_landing_anchor(
            frame=frame,
            measured_anchor_px=measured_landing_anchor_px,
            raw_bbox_xyxy=raw_bbox_xyxy,
            control_bbox_xyxy=bbox_xyxy,
            live_match=live_match,
            similarity=float(similarity),
            relative_height_m=float(relative_height),
        )
        landing_anchor_px = (
            terminal_anchor_px
            if bool(getattr(self, "_terminal_anchor_locked", False))
            and terminal_anchor_px is not None
            else measured_landing_anchor_px
        )
        metrics = self._landing_anchor_metrics(
            anchor_px=landing_anchor_px,
            virtual_bbox_xyxy=virtual_target_bbox_xyxy,
            visible_bbox_xyxy=bbox_xyxy,
            frame_shape=frame.shape,
        )

        # Once the terminal roof point has been locked, touchdown geometry follows
        # that physical point rather than a fresh detector rectangle. Before lock,
        # preserve BEST's raw-LIVE/control-BBox fallback.
        if bool(getattr(self, "_terminal_anchor_locked", False)) and landing_anchor_px is not None:
            touchdown_bbox_metrics = dict(metrics)
            touchdown_bbox_source = str(terminal_anchor_source)
        else:
            touchdown_bbox_xyxy = (
                raw_bbox_xyxy if live_match and raw_bbox_xyxy is not None else bbox_xyxy
            )
            touchdown_bbox_source = (
                "RAW_LIVE_BBOX"
                if live_match and raw_bbox_xyxy is not None
                else "CONTROL_BBOX_FALLBACK"
            )
            touchdown_bbox_metrics = self._bbox_metrics(touchdown_bbox_xyxy, frame.shape)
        self._update_visual_target_motion(
            frame=frame,
            bbox_xyxy=bbox_xyxy,
            metrics=metrics,
            live_match=live_match,
            relative_height_m=relative_height,
            api_state=api_state,
        )
        drone_z_ned = float(api_state.kinematics_estimated.position.z_val)
        obstacle = self._get_obstacles(drone_state.altitude_m, relative_height)
        obstacle_state = ObstacleState(
            front_dist_m=float(obstacle["front_dist_m"]),
            front_left_dist_m=float(obstacle["front_left_dist_m"]),
            front_right_dist_m=float(obstacle["front_right_dist_m"]),
            left_dist_m=float(obstacle["left_dist_m"]),
            right_dist_m=float(obstacle["right_dist_m"]),
            back_dist_m=float(obstacle["back_dist_m"]),
            down_dist_m=max(0.0, relative_height),
            min_obstacle_dist_m=float(obstacle["min_obstacle_dist_m"]),
        )

        bbox = None
        if bbox_xyxy is not None:
            x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
            conf = float(similarity if tracker_mode == "MATCH" else 0.25)
            obs_cx = 0.5 * (x1 + x2)
            obs_cy = 0.5 * (y1 + y2)
            if landing_anchor_px is not None:
                obs_cx = float(landing_anchor_px[0])
                obs_cy = float(landing_anchor_px[1])
            bbox = BBox(
                cx=obs_cx,
                cy=obs_cy,
                w=max(1.0, x2 - x1),
                h=max(1.0, y2 - y1),
                conf=conf,
            )

        obs, obs_dict = self.observation_builder.build(
            bbox=bbox,
            drone_state=drone_state,
            obstacle_state=obstacle_state,
            dt=float(self.cfg.cmd_duration_s),
        )
        obs = np.concatenate(
            [obs, np.asarray(range_state["features"], dtype=np.float32)]
        ).astype(np.float32, copy=False)
        for feature_name, feature_value in zip(
            self.range_finder_array.FEATURE_NAMES, range_state["features"]
        ):
            obs_dict[feature_name] = float(feature_value)

        match_recent = bool(
            self._step - self._last_match_step <= self.cfg.good_collision_recent_match_steps
        )
        match_confirmed = bool(
            live_match
            and self._live_match_streak >= int(self.cfg.match_confirmation_steps)
        )
        perception_now = time.monotonic()
        if live_match:
            self._lost_steps = 0
            self._non_live_started_monotonic = None
            self._non_live_duration_s = 0.0
        else:
            self._lost_steps += 1
            if self._non_live_started_monotonic is None:
                self._non_live_started_monotonic = perception_now
            self._non_live_duration_s = max(
                0.0,
                float(perception_now - self._non_live_started_monotonic),
            )

        if live_match:
            self._episode_live_match_steps += 1
            self._episode_best_similarity = max(
                self._episode_best_similarity,
                float(similarity),
            )
            if np.isfinite(metrics["center_error"]):
                self._episode_best_center_error = min(
                    self._episode_best_center_error,
                    float(metrics["center_error"]),
                )
        elif tracker_mode.startswith("PRED"):
            self._episode_predicted_steps += 1
        else:
            self._episode_no_target_steps += 1

        info: dict[str, Any] = {
            "obs_dict": obs_dict,
            "tracker_mode": tracker_mode,
            "target_id": self._target_id,
            "target_class_id": int(self._target_class_id) if self._target_class_id is not None else -1,
            "initial_yolo_class_id": int(self._target_class_id) if self._target_class_id is not None else -1,
            "strict_class_gate": False,
            "identity_judge": "resnet_only",
            "yolo_proposal_conf_threshold": float(self.cfg.yolo_proposal_conf),
            "reference_anchor_count": int(len(self._reference_embeddings)),
            "adaptive_anchor_count": int(len(self._adaptive_embeddings)),
            "adaptive_embedding_updates": int(self._adaptive_embedding_updates),
            "adaptive_bank_last_action": str(getattr(self, "_adaptive_bank_last_action", "EMPTY")),
            "adaptive_bank_replace_index": int(getattr(self, "_adaptive_bank_replace_index", 0)),
            "adaptive_bank_cycle": int(getattr(self, "_adaptive_bank_cycle", 0)),
            "has_front_anchor": self._original_embedding is not None,
            "has_bottom_anchor": self._bottom_anchor_embedding is not None,
            "yolo_candidate_count": int(self._last_candidate_count),
            "selected_candidate_yolo_class_id": (
                int(self._last_candidate_class_id)
                if self._last_candidate_class_id is not None
                else -1
            ),
            "selected_candidate_yolo_confidence": float(self._last_candidate_confidence),
            "candidate_resnet_scores": list(self._last_candidate_scores),
            "bottom_match": live_match,
            # Fresh now means a LIVE ResNet acceptance from the current frame.
            # Recent history is exposed separately for collision evaluation.
            "bottom_match_fresh": live_match,
            "bottom_match_live": live_match,
            "bottom_match_confirmed": match_confirmed,
            "bottom_match_recent": match_recent,
            "bottom_live_match_streak": int(self._live_match_streak),
            "bottom_match_margin": float(self._last_match_margin),
            "bottom_spatial_jump_norm": float(self._last_spatial_jump_norm),
            "bottom_match_reject_reason": self._last_match_reject_reason,
            "bottom_similarity": float(similarity),
            "bottom_bbox_filter_enabled": bool(self.cfg.control_bbox_filter_enabled),
            "bottom_bbox_outlier_suppressed": bool(
                self._control_bbox_outlier_suppressed
            ),
            "bottom_bbox_filter_mode": str(self._control_bbox_filter_mode),
            "bottom_bbox_raw_xyxy": (
                None
                if raw_bbox_xyxy is None
                else [
                    float(v)
                    for v in np.asarray(raw_bbox_xyxy).reshape(-1)[:4]
                ]
            ),
            "bottom_bbox_control_xyxy": (
                None
                if bbox_xyxy is None
                else [float(v) for v in np.asarray(bbox_xyxy).reshape(-1)[:4]]
            ),
            "bottom_landing_anchor_enabled": bool(self.cfg.landing_anchor_enabled),
            "bottom_landing_anchor_mode": str(self._landing_anchor_mode),
            "bottom_landing_anchor_edge_flags": str(self._landing_anchor_edge_flags),
            "bottom_landing_anchor_px": (
                None
                if landing_anchor_px is None
                else [float(v) for v in np.asarray(landing_anchor_px).reshape(-1)[:2]]
            ),
            "bottom_terminal_anchor_locked": bool(
                getattr(self, "_terminal_anchor_locked", False)
            ),
            "bottom_terminal_anchor_valid": bool(terminal_anchor_valid),
            "bottom_terminal_anchor_source": str(terminal_anchor_source),
            "bottom_terminal_anchor_age_steps": int(
                getattr(self, "_terminal_anchor_age_steps", 999999)
            ),
            "bottom_terminal_anchor_identity_age_steps": int(
                int(self._step)
                - int(getattr(self, "_terminal_anchor_last_identity_step", -999999))
            ),
            "bottom_terminal_anchor_flow_points": int(
                getattr(self, "_terminal_anchor_flow_points", 0)
            ),
            "bottom_terminal_anchor_flow_confidence": float(
                getattr(self, "_terminal_anchor_flow_confidence", 0.0)
            ),
            "bottom_terminal_anchor_px": (
                None
                if getattr(self, "_terminal_anchor_px", None) is None
                else [
                    float(v)
                    for v in np.asarray(self._terminal_anchor_px).reshape(-1)[:2]
                ]
            ),
            "bottom_virtual_target_bbox_xyxy": (
                None
                if virtual_target_bbox_xyxy is None
                else [
                    float(v)
                    for v in np.asarray(virtual_target_bbox_xyxy).reshape(-1)[:4]
                ]
            ),
            # Controller/observation geometry (semantic anchor aware).
            "bottom_err_x": metrics["err_x"],
            "bottom_err_y": metrics["err_y"],
            "bottom_center_error": metrics["center_error"],
            "bottom_bbox_rel_err": metrics["bbox_rel_error"],
            # Touchdown geometry follows the locked terminal roof point when
            # available; before lock it falls back to BEST's raw/control BBox.
            "bottom_touchdown_bbox_err_x": touchdown_bbox_metrics["err_x"],
            "bottom_touchdown_bbox_err_y": touchdown_bbox_metrics["err_y"],
            "bottom_touchdown_bbox_center_error": touchdown_bbox_metrics["center_error"],
            "bottom_touchdown_bbox_rel_err": touchdown_bbox_metrics["bbox_rel_error"],
            "bottom_touchdown_bbox_source": str(touchdown_bbox_source),
            "bottom_observation_monotonic": float(self._last_bottom_observation_monotonic),
            "bottom_frame_width_px": int(frame.shape[1]),
            "bottom_frame_height_px": int(frame.shape[0]),
            "bottom_calibrated_target_x_norm": float(
                self._bottom_center_target_x_norm
            ),
            "bottom_calibrated_target_y_norm": float(
                self._bottom_center_target_y_norm
            ),
            "bottom_center_calibration_source": str(
                self._bottom_center_calibration_source
            ),
            "bottom_bbox_rel_gate_required": bool(
                self._bbox_relative_alignment_required(
                    {
                        "relative_height_to_target_m": relative_height,
                    }
                )
            ),
            "bottom_bbox_area_norm": metrics["area_norm"],
            "bottom_img_vel_x_control": float(self._control_img_vel_x),
            "bottom_img_vel_y_control": float(self._control_img_vel_y),
            "bottom_control_dt_s": float(self._last_control_dt_s),
            "predicted_bottom_err_x": float(self._last_predictive_guidance.get("predicted_err_x", 0.0)),
            "predicted_bottom_err_y": float(self._last_predictive_guidance.get("predicted_err_y", 0.0)),
            "predicted_bottom_center_error": float(self._last_predictive_guidance.get("predicted_center_error", 999.0)),
            "predictive_bottom_catchup_active": bool(self._predictive_catchup_active),
            "alignment_ready_streak": int(self._alignment_ready_streak),
            "descent_alignment_latched": bool(self._descent_alignment_latched),
            "landing_lock_active": bool(self._descent_alignment_latched),
            "landing_lock_visual_gap_steps": int(self._landing_lock_visual_gap_steps),
            "landing_lock_bad_live_steps": int(self._landing_lock_bad_live_steps),
            "landing_lock_age_steps": int(
                self._step - self._landing_lock_acquired_step
                if self._descent_alignment_latched
                else -1
            ),
            "horizontal_control_state": self._last_horizontal_control_state,
            "horizontal_pd_action_vx": float(self._last_pd_action_vx),
            "horizontal_pd_action_vy": float(self._last_pd_action_vy),
            "horizontal_residual_action_vx": float(self._last_residual_action_vx),
            "horizontal_residual_action_vy": float(self._last_residual_action_vy),
            "horizontal_final_action_vx": float(self._last_horizontal_action_vx),
            "horizontal_final_action_vy": float(self._last_horizontal_action_vy),
            "horizontal_speed_limit_mps": float(self._last_horizontal_speed_limit_mps),
            "target_velocity_valid": bool(getattr(self, "_target_velocity_valid", False)),
            "target_velocity_age_s": float(getattr(self, "_target_velocity_age_s", float("inf"))),
            "target_velocity_world_x_mps": float(getattr(self, "_target_velocity_world_x_mps", 0.0)),
            "target_velocity_world_y_mps": float(getattr(self, "_target_velocity_world_y_mps", 0.0)),
            "target_velocity_body_vx_mps": float(getattr(self, "_target_velocity_body_vx_mps", 0.0)),
            "target_velocity_body_vy_mps": float(getattr(self, "_target_velocity_body_vy_mps", 0.0)),
            "target_velocity_speed_mps": float(getattr(self, "_target_velocity_speed_mps", 0.0)),
            "target_velocity_source": "bottom_vision_plus_drone_ego_velocity",
            "target_actor_xy_velocity_api_used": False,
            "visual_motion_source": str(self._visual_motion_source),
            "visual_motion_age_s": float(self._visual_motion_age_s),
            "visual_motion_attitude_valid": bool(self._visual_motion_attitude_valid),
            "visual_motion_yaw_rate_dps": float(self._visual_motion_yaw_rate_dps),
            "visual_relative_position_body_x_m": float(
                self._visual_relative_position_body_x_m
            ),
            "visual_relative_position_body_y_m": float(
                self._visual_relative_position_body_y_m
            ),
            "visual_relative_velocity_body_x_mps": float(
                self._visual_relative_velocity_body_x_mps
            ),
            "visual_relative_velocity_body_y_mps": float(
                self._visual_relative_velocity_body_y_mps
            ),
            "visual_relative_velocity_valid": bool(
                self._visual_relative_velocity_valid
            ),
            "visual_kalman_enabled": bool(self.cfg.visual_kalman_enabled),
            "visual_kalman_velocity_initialized": bool(
                self._visual_kalman_velocity_initialized
            ),
            "visual_bbox_velocity_x_per_s": float(
                self._visual_bbox_velocity_x_per_s
            ),
            "visual_bbox_velocity_y_per_s": float(
                self._visual_bbox_velocity_y_per_s
            ),
            "visual_flow_velocity_x_per_s": float(
                self._visual_flow_velocity_x_per_s
            ),
            "visual_flow_velocity_y_per_s": float(
                self._visual_flow_velocity_y_per_s
            ),
            "visual_flow_point_count": int(self._visual_flow_point_count),
            "visual_flow_confidence": float(self._visual_flow_confidence),
            "drone_body_velocity_x_mps": float(self._drone_body_velocity_x_mps),
            "drone_body_velocity_y_mps": float(self._drone_body_velocity_y_mps),
            "bottom_camera_hfov_deg": float(self._bottom_camera_hfov_deg),
            "horizontal_velocity_ff_vx_mps": float(self._last_velocity_ff_vx_mps),
            "horizontal_velocity_ff_vy_mps": float(self._last_velocity_ff_vy_mps),
            "horizontal_correction_vx_mps": float(self._last_horizontal_correction_vx_mps),
            "horizontal_correction_vy_mps": float(self._last_horizontal_correction_vy_mps),
            "horizontal_command_vx_mps": float(self._last_horizontal_command_vx_mps),
            "horizontal_command_vy_mps": float(self._last_horizontal_command_vy_mps),
            "horizontal_total_speed_limit_mps": float(self._last_horizontal_total_speed_limit_mps),
            "alt_agl_m": float(drone_state.altitude_m),
            "drone_z_ned": float(drone_z_ned),
            "target_surface_z_ned": float(self._target_surface_z_ned),
            "target_surface_altitude_m": float(self._target_surface_altitude_m),
            "target_surface_source": self._target_surface_source,
            "relative_height_to_target_m": float(relative_height),
            "api_relative_height_to_target_m": float(api_relative_height),
            "landing_height_m": float(relative_height),
            "landing_height_source": str(landing_height_source),
            "range_height_used": bool(range_height_used),
            "range_calibration_loaded": bool(range_state["range_calibration_loaded"]),
            "range_surface_world_z_median": float(range_state["range_surface_world_z_median"]),
            "range_surface_world_z_spread": float(range_state["range_surface_world_z_spread"]),
            "lidar_vertical_used": False,
            "obstacle_source": obstacle["obstacle_source"],
            "lidar_valid": bool(obstacle.get("lidar_valid", False)),
            "lidar_point_count": int(obstacle.get("lidar_point_count", 0)),
            "range_top_left_m": float(range_state["range_distances_m"]["TOP_LEFT"]),
            "range_top_right_m": float(range_state["range_distances_m"]["TOP_RIGHT"]),
            "range_bottom_left_m": float(range_state["range_distances_m"]["BOTTOM_LEFT"]),
            "range_bottom_right_m": float(range_state["range_distances_m"]["BOTTOM_RIGHT"]),
            "range_center_m": float(range_state["range_distances_m"]["CENTER"]),
            "range_mean_m": float(range_state["range_mean_m"]),
            "range_spread_m": float(range_state["range_spread_m"]),
            "range_valid_count": int(range_state["range_valid_count"]),
            "range_valid_ratio": float(range_state["range_valid_ratio"]),
            "range_closing_rate_mps": float(range_state["range_closing_rate_mps"]),
            "range_height_reliable": bool(range_state["range_height_reliable"]),
            "range_sensor_final_ready": bool(range_state["range_sensor_final_ready"]),
            "front_dist_m": float(obstacle["front_dist_m"]),
            "left_dist_m": float(obstacle["left_dist_m"]),
            "right_dist_m": float(obstacle["right_dist_m"]),
            "back_dist_m": float(obstacle["back_dist_m"]),
            "lost_steps": int(self._lost_steps),
            "bottom_no_live_duration_s": float(self._non_live_duration_s),
            "reward_bank": float(self._reward_bank),
            "vertical_control_state": self._last_vertical_control_state,
            "descent_block_reason": self._last_descent_block_reason,
            "raw_vz_action": float(self._last_raw_vz_action),
            "requested_vz_mps": float(self._last_requested_vz_mps),
            "applied_vz_mps": float(self._last_applied_vz_mps),
            "vertical_speed_limit_mps": float(self._last_vertical_speed_limit_mps),
            "soft_catchup_descent_active": bool(self._last_soft_catchup_descent_active),
            "climb_command_blocked": bool(self._last_climb_command_blocked),
        }

        if self.cfg.show_camera:
            vis = frame.copy()
            h, w = vis.shape[:2]
            calibrated_center = (
                int(round(float(getattr(self, "_bottom_center_target_x_norm", 0.5)) * float(w))),
                int(round(float(getattr(self, "_bottom_center_target_y_norm", 0.5)) * float(h))),
            )
            self._draw_frame_center_marker(vis, calibrated_center)
            display_mode = tracker_mode
            if live_match and not bool(self._descent_alignment_latched):
                display_mode = (
                    f"LOCK_PENDING({self._alignment_ready_streak}/"
                    f"{int(self.cfg.alignment_streak_required)})"
                )
            elif live_match and bool(self._descent_alignment_latched):
                display_mode = "MATCH_LANDING_LOCK"
            if raw_bbox_xyxy is not None and live_match:
                rx1, ry1, rx2, ry2 = [int(v) for v in raw_bbox_xyxy]
                cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)
            if bbox_xyxy is not None:
                x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
                color = (0, 255, 0) if match_confirmed else (0, 255, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            if landing_anchor_px is not None:
                anchor_x = int(round(float(landing_anchor_px[0])))
                anchor_y = int(round(float(landing_anchor_px[1])))
                marker_x = int(np.clip(anchor_x, 14, max(14, w - 15)))
                marker_y = int(np.clip(anchor_y, 30, max(30, h - 2)))
                self._draw_double_v_marker(vis, (marker_x, marker_y))
            cv2.putText(
                vis,
                f"AGENT 2 | {self._target_id} | {display_mode} sim={similarity:.3f} "
                f"bboxFilter={self._control_bbox_filter_mode} "
                f"anchor={self._landing_anchor_mode} "
                f"terminal={terminal_anchor_source}",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                f"API height above target={relative_height:.2f}m "
                f"BRelGate={'FINAL' if self._bbox_relative_alignment_required(info) else 'CENTER_ONLY'} "
                f"YOLOcls(diag)={self._last_candidate_class_id}",
                (15, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                f"Z_CTRL={self._last_vertical_control_state} "
                f"raw={self._last_raw_vz_action:+.2f} "
                f"down={self._last_requested_vz_mps:+.2f} applied={self._last_applied_vz_mps:+.2f} "
                f"cap={self._last_vertical_speed_limit_mps:+.2f}",
                (15, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                f"XY_CTRL={self._last_horizontal_control_state} "
                f"PD=({self._last_pd_action_vx:+.2f},{self._last_pd_action_vy:+.2f}) "
                f"RES=({self._last_residual_action_vx:+.2f},{self._last_residual_action_vy:+.2f}) "
                f"CMD=({self._last_horizontal_action_vx:+.2f},{self._last_horizontal_action_vy:+.2f})",
                (15, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 255),
                2,
            )
            external = dict(getattr(self, "_last_external_command_info", {}) or {})
            if bool(self.cfg.parallel_dual_agent_mode):
                cv2.putText(
                    vis,
                    f"PHYSICAL_XY=({float(external.get('agent1_commanded_vx_mps', 0.0)):+.2f},"
                    f"{float(external.get('agent1_commanded_vy_mps', 0.0)):+.2f})m/s "
                    f"A1W={float(external.get('agent1_xy_weight', 1.0)):.2f} "
                    f"BPD=({float(external.get('bottom_guidance_vx_mps', 0.0)):+.2f},"
                    f"{float(external.get('bottom_guidance_vy_mps', 0.0)):+.2f})",
                    (15, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    vis,
                    f"FF=({float(external.get('target_velocity_ff_vx_mps', 0.0)):+.2f},"
                    f"{float(external.get('target_velocity_ff_vy_mps', 0.0)):+.2f}) "
                    f"LOCK={int(bool(self._descent_alignment_latched))} "
                    f"CATCH={int(bool(external.get('bottom_predictive_catchup', False)))} "
                    f"BANK={len(self._adaptive_embeddings)}",
                    (15, 168),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    vis,
                    f"PRED=({float(external.get('bottom_predicted_err_x', 0.0)):+.2f},"
                    f"{float(external.get('bottom_predicted_err_y', 0.0)):+.2f}) "
                    f"IVEL=({float(external.get('bottom_image_velocity_x_per_s', 0.0)):+.2f},"
                    f"{float(external.get('bottom_image_velocity_y_per_s', 0.0)):+.2f}) "
                    f"H={float(external.get('bottom_prediction_horizon_s', 0.0)):.2f}s",
                    (15, 196),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (0, 255, 255),
                    2,
                )
                target_vel_y = 224
            else:
                target_vel_y = 140
            cv2.putText(
                vis,
                f"VISUAL_TARGET_VEL={'VALID' if self._target_velocity_valid else 'WAIT'} "
                f"body=({self._target_velocity_body_vx_mps:+.2f},"
                f"{self._target_velocity_body_vy_mps:+.2f})m/s "
                f"src={self._visual_motion_source} flowN={self._visual_flow_point_count} "
                f"lost={self._lost_steps}/{self._non_live_duration_s:.1f}s",
                (15, target_vel_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Agent 2 - Bottom Landing", vis)
            cv2.waitKey(1)

        self._append_kalman_diagnostic_row(info)
        self._last_info = info
        return np.asarray(obs, dtype=np.float32), info

    # ------------------------------------------------------------------
    # Control/reward
    # ------------------------------------------------------------------
    def _update_control_image_velocity(self, metrics: dict[str, float], live_match: bool) -> None:
        """Estimate bottom-frame relative velocity using real frame spacing."""
        now = time.monotonic()
        self._last_control_dt_s = float(self.cfg.cmd_duration_s)
        if live_match:
            err_x = float(metrics.get("err_x", 0.0) or 0.0)
            err_y = float(metrics.get("err_y", 0.0) or 0.0)
            if self._last_control_had_live_match and self._last_observation_monotonic is not None:
                dt = float(np.clip(now - self._last_observation_monotonic, 0.03, 1.0))
                self._last_control_dt_s = dt
                clip_v = float(self.cfg.horizontal_velocity_clip_per_s)
                raw_vx = float(
                    np.clip(
                        (err_x - self._last_control_err_x) / dt,
                        -clip_v,
                        clip_v,
                    )
                )
                raw_vy = float(
                    np.clip(
                        (err_y - self._last_control_err_y) / dt,
                        -clip_v,
                        clip_v,
                    )
                )
                alpha = float(
                    np.clip(self.cfg.horizontal_velocity_ema_alpha, 0.0, 1.0)
                )
                self._control_img_vel_x = float(
                    alpha * raw_vx
                    + (1.0 - alpha) * self._control_img_vel_x
                )
                self._control_img_vel_y = float(
                    alpha * raw_vy
                    + (1.0 - alpha) * self._control_img_vel_y
                )
            else:
                self._control_img_vel_x = 0.0
                self._control_img_vel_y = 0.0
            self._last_control_err_x = err_x
            self._last_control_err_y = err_y
            self._last_control_had_live_match = True
        else:
            self._control_img_vel_x = 0.0
            self._control_img_vel_y = 0.0
            self._last_control_had_live_match = False
        self._last_observation_monotonic = now

    def _horizontal_speed_limit(self, info: dict[str, Any]) -> float:
        """Return a conservative XY speed limit from API height and visual scale."""
        try:
            height = float(info.get("relative_height_to_target_m", float("inf")))
        except (TypeError, ValueError):
            height = float("inf")
        try:
            area = float(info.get("bottom_bbox_area_norm", 0.0))
        except (TypeError, ValueError):
            area = 0.0

        if not np.isfinite(height):
            height_limit = float(self.cfg.horizontal_speed_mid_mps)
        elif height <= 0.8:
            height_limit = float(self.cfg.horizontal_speed_touchdown_mps)
        elif height <= 1.6:
            height_limit = float(self.cfg.horizontal_speed_near_mps)
        elif height <= 3.5:
            height_limit = float(self.cfg.horizontal_speed_mid_mps)
        else:
            height_limit = float(self.cfg.horizontal_speed_far_mps)

        if area >= 0.20:
            area_limit = float(self.cfg.horizontal_speed_touchdown_mps)
        elif area >= 0.10:
            area_limit = float(self.cfg.horizontal_speed_near_mps)
        elif area >= 0.045:
            area_limit = float(self.cfg.horizontal_speed_mid_mps)
        else:
            area_limit = float(self.cfg.horizontal_speed_far_mps)

        return float(max(0.05, min(height_limit, area_limit)))

    def _horizontal_visual_servo(
        self,
        raw: np.ndarray,
        info: dict[str, Any],
    ) -> tuple[float, float, dict[str, Any]]:
        """Combine target-velocity feed-forward and predictive bottom guidance."""
        live_match = bool(info.get("bottom_match_live", False))
        confirmed = bool(info.get("bottom_match_confirmed", False))
        correction_speed_limit = self._horizontal_speed_limit(info)

        velocity_valid = bool(info.get("target_velocity_valid", False))
        ff_gain = float(self.cfg.target_velocity_feedforward_gain)
        ff_vx = (
            ff_gain * float(info.get("target_velocity_body_vx_mps", 0.0) or 0.0)
            if velocity_valid
            else 0.0
        )
        ff_vy = (
            ff_gain * float(info.get("target_velocity_body_vy_mps", 0.0) or 0.0)
            if velocity_valid
            else 0.0
        )
        ff_speed = float(math.hypot(ff_vx, ff_vy))
        max_total = max(0.10, float(self.cfg.horizontal_total_speed_max_mps))

        guidance = self._compute_predictive_bottom_guidance(
            info,
            update_hysteresis=True,
        )
        guidance_active = bool(guidance.get("guidance_active", live_match))
        prediction_only = bool(guidance.get("prediction_only", False))

        if not guidance_active:
            self._episode_xy_hold_steps = int(
                getattr(self, "_episode_xy_hold_steps", 0)
            ) + 1
            no_live_duration_s = float(
                info.get("bottom_no_live_duration_s", 0.0) or 0.0
            )
            velocity_hold = bool(
                velocity_valid
                and no_live_duration_s
                <= float(self.cfg.horizontal_velocity_hold_max_s)
            )
            if velocity_hold:
                vx, vy = self._clip_vector(ff_vx, ff_vy, max_total)
                state = "VELOCITY_HOLD_NO_LIVE_MATCH"
            else:
                vx, vy = 0.0, 0.0
                state = "HOLD_NO_LIVE_MATCH"
            details = {
                "state": state,
                "pd_ax": 0.0,
                "pd_ay": 0.0,
                "residual_ax": 0.0,
                "residual_ay": 0.0,
                "final_ax": 0.0,
                "final_ay": 0.0,
                "speed_limit_mps": correction_speed_limit,
                "ff_vx_mps": float(ff_vx if velocity_hold else 0.0),
                "ff_vy_mps": float(ff_vy if velocity_hold else 0.0),
                "correction_vx_mps": 0.0,
                "correction_vy_mps": 0.0,
                "command_vx_mps": float(vx),
                "command_vy_mps": float(vy),
                "total_speed_limit_mps": max_total,
                "predicted_err_x": 0.0,
                "predicted_err_y": 0.0,
                "predicted_center_error": 999.0,
                "image_velocity_x_per_s": 0.0,
                "image_velocity_y_per_s": 0.0,
                "prediction_horizon_s": float(self.cfg.cmd_duration_s),
                "catchup_active": False,
                "guidance_active": False,
                "prediction_only": False,
            }
            return float(vx), float(vy), details

        predictive_vx = float(guidance["bottom_correction_vx_mps"])
        predictive_vy = float(guidance["bottom_correction_vy_mps"])
        correction_limit = max(
            0.05,
            float(guidance.get("correction_limit_mps", correction_speed_limit)),
        )

        residual_max = float(
            np.clip(self.cfg.horizontal_ppo_residual_max_action, 0.0, 0.30)
        )
        if confirmed and live_match:
            residual_ax = float(
                np.clip(float(raw[0]) * residual_max, -residual_max, residual_max)
            )
            residual_ay = float(
                np.clip(float(raw[1]) * residual_max, -residual_max, residual_max)
            )
        else:
            residual_ax = 0.0
            residual_ay = 0.0

        # PPO XY remains a small residual in standalone Agent-2 mode. In the
        # parallel wrapper this residual is masked and only deterministic
        # predictive guidance is exported.
        residual_vx = float(residual_ax * correction_speed_limit)
        residual_vy = float(residual_ay * correction_speed_limit)
        correction_vx = float(predictive_vx + residual_vx)
        correction_vy = float(predictive_vy + residual_vy)
        correction_vx, correction_vy = self._clip_vector(
            correction_vx,
            correction_vy,
            correction_limit,
        )

        total_speed_limit = float(
            min(
                max_total,
                max(
                    correction_limit,
                    ff_speed + correction_limit,
                ),
            )
        )
        vx, vy = self._clip_vector(
            ff_vx + correction_vx,
            ff_vy + correction_vy,
            total_speed_limit,
        )

        pd_ax = float(np.clip(predictive_vx / correction_limit, -1.0, 1.0))
        pd_ay = float(np.clip(predictive_vy / correction_limit, -1.0, 1.0))
        final_ax = float(np.clip(correction_vx / correction_limit, -1.0, 1.0))
        final_ay = float(np.clip(correction_vy / correction_limit, -1.0, 1.0))
        state = (
            "PREDICTIVE_BOTTOM_SHORT_PREDICTION"
            if prediction_only
            else "PREDICTIVE_BOTTOM_CATCHUP"
            if bool(guidance.get("catchup_active", False))
            else (
                "PREDICTIVE_BOTTOM_PLUS_RESIDUAL"
                if confirmed
                else "PREDICTIVE_BOTTOM_MATCH_PENDING"
            )
        )

        details = {
            "state": state,
            "pd_ax": pd_ax,
            "pd_ay": pd_ay,
            "residual_ax": residual_ax,
            "residual_ay": residual_ay,
            "final_ax": final_ax,
            "final_ay": final_ay,
            "speed_limit_mps": correction_limit,
            "ff_vx_mps": ff_vx,
            "ff_vy_mps": ff_vy,
            "correction_vx_mps": correction_vx,
            "correction_vy_mps": correction_vy,
            "command_vx_mps": vx,
            "command_vy_mps": vy,
            "total_speed_limit_mps": total_speed_limit,
            "predicted_err_x": float(guidance["predicted_err_x"]),
            "predicted_err_y": float(guidance["predicted_err_y"]),
            "predicted_center_error": float(guidance["predicted_center_error"]),
            "image_velocity_x_per_s": float(guidance["image_velocity_x_per_s"]),
            "image_velocity_y_per_s": float(guidance["image_velocity_y_per_s"]),
            "prediction_horizon_s": float(guidance["prediction_horizon_s"]),
            "catchup_active": bool(guidance["catchup_active"]),
            "guidance_active": bool(guidance_active),
            "prediction_only": bool(prediction_only),
        }
        return float(vx), float(vy), details

    def _horizontal_lidar_guard(self, vx: float, vy: float, info: dict[str, Any]) -> tuple[float, float, list[str]]:
        reasons: list[str] = []
        emergency = float(self.cfg.obstacle_emergency_m)
        warning = float(self.cfg.obstacle_warning_m)

        front = float(info["front_dist_m"])
        back = float(info["back_dist_m"])
        left = float(info["left_dist_m"])
        right = float(info["right_dist_m"])

        if vx > 0 and front < emergency:
            vx = 0.0
            reasons.append("front_emergency")
        elif vx > 0 and front < warning:
            vx *= max(0.0, (front - emergency) / max(1e-6, warning - emergency))
            reasons.append("front_slow")

        if vx < 0 and back < emergency:
            vx = 0.0
            reasons.append("back_emergency")
        if vy > 0 and right < emergency:
            vy = 0.0
            reasons.append("right_emergency")
        if vy < 0 and left < emergency:
            vy = 0.0
            reasons.append("left_emergency")
        return float(vx), float(vy), reasons

    @staticmethod
    def _scaled_frame_crop(
        frame: np.ndarray,
        scale: float,
        anchor_x: float,
        anchor_y: float,
    ) -> Optional[np.ndarray]:
        """Return an image crop centered at a normalized anchor position."""
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        scale = float(np.clip(scale, 0.05, 1.0))
        crop_w = max(3, int(round(w * scale)))
        crop_h = max(3, int(round(h * scale)))
        cx = float(np.clip(anchor_x, 0.0, 1.0)) * float(w)
        cy = float(np.clip(anchor_y, 0.0, 1.0)) * float(h)
        x1 = int(round(cx - 0.5 * crop_w))
        y1 = int(round(cy - 0.5 * crop_h))
        x1 = max(0, min(x1, w - crop_w))
        y1 = max(0, min(y1, h - crop_h))
        crop = frame[y1:y1 + crop_h, x1:x1 + crop_w]
        if crop.size == 0:
            return None
        return crop

    def _appearance_similarity_to_bottom_anchor(self, crop: np.ndarray) -> float:
        """Return the best similarity against immutable and verified views."""
        references = self._all_reference_embeddings()
        if not references:
            return 0.0
        embedding = self.tracker._embedding_from_crop(crop)
        if embedding is None:
            return 0.0
        emb = F.normalize(embedding.reshape(-1), dim=0)
        scores = [
            float(torch.dot(F.normalize(reference.reshape(-1), dim=0), emb).detach().cpu().item())
            for reference in references
            if reference is not None
        ]
        return max(scores, default=0.0)

    def _contact_appearance_evidence(self) -> dict[str, Any]:
        """Measure whether the selected target is under the bottom camera now.

        This uses only the collision-frame image. Center crops are target
        candidates; same-frame corner crops are background controls. This
        distinguishes a top touchdown from a side impact where the camera sees
        mostly floor beneath the drone.
        """
        frame = self._last_bottom_frame
        if frame is None or frame.size == 0:
            return {
                "valid": False,
                "center_similarity": 0.0,
                "corner_similarity": 0.0,
                "center_margin": 0.0,
                "best_center_scale": 0.0,
                "reason": "missing_collision_frame",
            }
        if self._bottom_anchor_embedding is None and self._original_embedding is None:
            return {
                "valid": False,
                "center_similarity": 0.0,
                "corner_similarity": 0.0,
                "center_margin": 0.0,
                "best_center_scale": 0.0,
                "reason": "missing_target_anchor",
            }

        center_scores: list[tuple[float, float]] = []
        for scale in self.cfg.contact_appearance_center_crop_scales:
            crop = self._scaled_frame_crop(frame, float(scale), 0.5, 0.5)
            if crop is None:
                continue
            center_scores.append((float(scale), self._appearance_similarity_to_bottom_anchor(crop)))

        corner_scale = float(self.cfg.contact_appearance_corner_crop_scale)
        corner_scores: list[float] = []
        # Anchors place each control crop inside a corner without going outside
        # the frame. They deliberately avoid the image center.
        offset = 0.5 * corner_scale
        for ax, ay in (
            (offset, offset),
            (1.0 - offset, offset),
            (offset, 1.0 - offset),
            (1.0 - offset, 1.0 - offset),
        ):
            crop = self._scaled_frame_crop(frame, corner_scale, ax, ay)
            if crop is None:
                continue
            corner_scores.append(self._appearance_similarity_to_bottom_anchor(crop))

        if not center_scores:
            return {
                "valid": False,
                "center_similarity": 0.0,
                "corner_similarity": max(corner_scores, default=0.0),
                "center_margin": 0.0,
                "best_center_scale": 0.0,
                "reason": "no_valid_center_crop",
            }

        best_scale, best_center = max(center_scores, key=lambda item: item[1])
        best_corner = max(corner_scores, default=0.0)
        margin = float(best_center - best_corner)
        absolute_ok = best_center >= float(self.cfg.contact_appearance_min_similarity)
        strong_ok = best_center >= float(self.cfg.contact_appearance_strong_similarity)
        contrast_ok = margin >= float(self.cfg.contact_appearance_min_center_margin)
        valid = bool(absolute_ok and (strong_ok or contrast_ok))

        if not absolute_ok:
            reason = "center_similarity_below_threshold"
        elif not strong_ok and not contrast_ok:
            reason = "center_not_stronger_than_background"
        else:
            reason = ""

        return {
            "valid": valid,
            "center_similarity": float(best_center),
            "corner_similarity": float(best_corner),
            "center_margin": margin,
            "best_center_scale": float(best_scale),
            "reason": reason,
        }

    @staticmethod
    def _normalized_collision_object_name(value: str) -> str:
        return str(value or "").strip().casefold()

    def _is_ground_collision_object(self, object_name: str) -> bool:
        normalized = self._normalized_collision_object_name(object_name)
        if not normalized:
            return True
        return any(
            str(token).strip().casefold() in normalized
            for token in self.cfg.collision_ground_tokens
            if str(token).strip()
        )

    def _legacy_touchdown_sample(self, info: dict[str, Any]) -> dict[str, Any] | None:
        """Build one confirmed LIVE bottom-camera sample for touchdown review.

        Admission is intentionally independent of the center threshold. The
        history must preserve both good and bad late alignment so an older
        centered frame can never hide a newer side impact. Binary touchdown
        centering uses the locked terminal roof point once available. Before
        terminal lock, the current raw LIVE BBox remains the fallback source.
        """
        live_match = bool(info.get("bottom_match_live", False))
        confirmed = bool(info.get("bottom_match_confirmed", False))
        terminal_valid = bool(info.get("bottom_terminal_anchor_valid", False))
        terminal_locked = bool(info.get("bottom_terminal_anchor_locked", False))
        similarity = float(info.get("bottom_similarity", 0.0))
        center_error = float(
            info.get(
                "bottom_touchdown_bbox_center_error",
                info.get("bottom_center_error", float("inf")),
            )
        )
        bbox_rel_error = float(
            info.get(
                "bottom_touchdown_bbox_rel_err",
                info.get("bottom_bbox_rel_err", float("inf")),
            )
        )
        err_x = float(
            info.get(
                "bottom_touchdown_bbox_err_x",
                info.get("bottom_err_x", float("inf")),
            )
        )
        err_y = float(
            info.get(
                "bottom_touchdown_bbox_err_y",
                info.get("bottom_err_y", float("inf")),
            )
        )

        identity_confirmed = bool((live_match and confirmed) or (terminal_locked and terminal_valid))
        if not (
            identity_confirmed
            and np.isfinite(center_error)
            and center_error >= 0.0
            and np.isfinite(err_x)
            and np.isfinite(err_y)
        ):
            return None

        return {
            "step": int(self._step),
            "error_m": float(center_error),  # compatibility: normalized, not metres
            "err_x": float(err_x),
            "err_y": float(err_y),
            "center_error": float(center_error),
            "bbox_rel_error": float(bbox_rel_error),
            "similarity": float(similarity),
            "confirmed": True,
            "identity_source": (
                "TERMINAL_ANCHOR"
                if terminal_locked and terminal_valid
                else "LIVE_MATCH"
            ),
            "relative_height_m": float(info.get("relative_height_to_target_m", float("inf"))),
            "frame_width_px": int(info.get("bottom_frame_width_px", self.cfg.image_width)),
            "frame_height_px": int(info.get("bottom_frame_height_px", self.cfg.image_height)),
            "source_monotonic": float(info.get("bottom_observation_monotonic", time.monotonic())),
            "bbox_source": str(info.get("bottom_touchdown_bbox_source", "CONTROL_BBOX_FALLBACK")),
            "decision_context": str(
                info.get("bottom_touchdown_decision_context", "PRE_CONTACT_LIVE")
            ),
            "relative_velocity_valid": bool(
                info.get("visual_relative_velocity_valid", False)
            ),
            "relative_velocity_x_mps": float(
                info.get("visual_relative_velocity_body_x_mps", float("nan"))
            ),
            "relative_velocity_y_mps": float(
                info.get("visual_relative_velocity_body_y_mps", float("nan"))
            ),
        }

    def _record_touchdown_legacy_sample(self, info: dict[str, Any]) -> None:
        """Record every confirmed LIVE sample, including off-center samples."""
        sample = self._legacy_touchdown_sample(info)
        if sample is None:
            return

        # Refresh the center latch only from a genuine confirmed LIVE BBox.
        # Terminal-anchor/PRED continuity is deliberately not allowed to update
        # the value because close-range semantic distortion is exactly the case
        # in which the previous trustworthy measurement must be preserved.
        if (
            bool(self.cfg.terminal_last_valid_center_enabled)
            and not bool(getattr(self, "_range_terminal_handoff_active", False))
        ):
            similarity = float(sample.get("similarity", 0.0))
            bbox_rel = float(sample.get("bbox_rel_error", float("inf")))
            is_live_bbox = str(sample.get("identity_source", "")) == "LIVE_MATCH"
            center_error_m, center_valid = self._geometric_xy_error_m(
                float(sample.get("err_x", float("inf"))),
                float(sample.get("err_y", float("inf"))),
                float(sample.get("relative_height_m", float("inf"))),
                float(sample.get("frame_width_px", self.cfg.image_width)),
                float(sample.get("frame_height_px", self.cfg.image_height)),
            )
            if (
                is_live_bbox
                and similarity >= float(self.cfg.terminal_last_valid_center_min_similarity)
                and np.isfinite(bbox_rel)
                and bbox_rel <= float(self.cfg.terminal_last_valid_center_max_bbox_rel_error)
                and center_valid
            ):
                self._last_valid_center_error_m = float(center_error_m)
                self._last_valid_center_step = int(self._step)
                self._last_valid_center_monotonic = float(time.monotonic())
                self._last_valid_center_similarity = float(similarity)
                self._last_valid_center_bbox_rel_error = float(bbox_rel)
                self._last_valid_center_source = str(sample.get("bbox_source", "LIVE_BBOX"))
                if (
                    self._visual_kalman_state is not None
                    and np.all(np.isfinite(self._visual_kalman_state))
                    and np.all(np.isfinite(self._visual_kalman_covariance))
                ):
                    self._terminal_kalman_state = np.asarray(
                        self._visual_kalman_state, dtype=np.float64
                    ).copy()
                    self._terminal_kalman_covariance = np.asarray(
                        self._visual_kalman_covariance, dtype=np.float64
                    ).copy()
                    self._terminal_kalman_monotonic = float(time.monotonic())

        self._touchdown_legacy_window.append(sample)
        # Compatibility field only. It now mirrors the newest confirmed sample
        # and is never used to prefer an older center-qualified frame.
        self._touchdown_ready_snapshot = dict(sample)
        max_steps = max(1, int(self.cfg.touchdown_legacy_window_steps))
        min_step = int(self._step) - max_steps
        self._touchdown_legacy_window = [
            row for row in self._touchdown_legacy_window
            if int(row.get("step", -999999)) >= min_step
        ]

    def _best_recent_legacy_touchdown_sample(
        self, pre_contact_info: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Return the newest confirmed sample at or before the contact command.

        The pre-command observation has absolute priority. History is used only
        when that observation temporarily lacks a confirmed LIVE MATCH, and the
        caller applies the short collision-age limit. No best-error selection
        and no center-qualified snapshot fallback are allowed.
        """
        current = self._legacy_touchdown_sample(pre_contact_info)
        if current is not None:
            current = dict(current)
            context = str(current.get("decision_context", "PRE_CONTACT_LIVE"))
            current["decision_source"] = (
                f"{context}:{current.get('bbox_source', 'UNKNOWN_BBOX')}"
            )
            return current

        max_steps = max(1, int(self.cfg.touchdown_legacy_window_steps))
        min_step = int(self._step) - max_steps
        candidates = [
            row for row in list(getattr(self, "_touchdown_legacy_window", []))
            if int(row.get("step", -999999)) >= min_step
        ]
        selected = max(candidates, key=lambda row: int(row["step"]), default=None)
        if selected is None:
            return None
        selected = dict(selected)
        selected["decision_source"] = f"PRE_CONTACT_FALLBACK:{selected.get('bbox_source', 'UNKNOWN_BBOX')}"
        return selected

    def _terminal_semantic_drift_fallback(self) -> dict[str, Any]:
        """Evaluate a bounded recent-history fallback for terminal BBox drift.

        Only the newest visually confirmed, geometrically projectable samples
        are considered. The median radial center error prevents one distorted
        terminal frame from dominating the decision, while the relative-speed
        gate rejects obvious lateral slip before contact.
        """
        result: dict[str, Any] = {
            "enabled": bool(self.cfg.terminal_fallback_enabled),
            "valid": False,
            "passed": False,
            "sample_count": 0,
            "velocity_sample_count": 0,
            "median_center_error_m": float("inf"),
            "mean_relative_speed_mps": float("inf"),
            "max_relative_speed_mps": float("inf"),
            "reason": "DISABLED",
        }
        if not bool(self.cfg.terminal_fallback_enabled):
            return result

        rows: list[dict[str, Any]] = []
        for raw in list(getattr(self, "_touchdown_legacy_window", [])):
            if not bool(raw.get("confirmed", False)):
                continue
            similarity = float(raw.get("similarity", 0.0))
            bbox_rel = float(raw.get("bbox_rel_error", float("inf")))
            if similarity < float(self.cfg.terminal_fallback_min_similarity):
                continue
            if not np.isfinite(bbox_rel) or bbox_rel > float(
                self.cfg.terminal_fallback_max_bbox_rel_error
            ):
                continue
            error_m, valid = self._geometric_xy_error_m(
                float(raw.get("err_x", float("inf"))),
                float(raw.get("err_y", float("inf"))),
                float(raw.get("relative_height_m", float("inf"))),
                float(raw.get("frame_width_px", self.cfg.image_width)),
                float(raw.get("frame_height_px", self.cfg.image_height)),
            )
            if not valid:
                continue
            row = dict(raw)
            row["center_error_m"] = float(error_m)
            rows.append(row)

        rows.sort(key=lambda row: int(row.get("step", -999999)))
        keep = max(1, int(self.cfg.terminal_fallback_recent_samples))
        rows = rows[-keep:]
        result["sample_count"] = len(rows)
        if len(rows) < int(self.cfg.terminal_fallback_min_samples):
            result["reason"] = "INSUFFICIENT_RECENT_RELIABLE_SAMPLES"
            return result

        center_errors = np.asarray(
            [float(row["center_error_m"]) for row in rows], dtype=np.float64
        )
        median_center = float(np.median(center_errors))

        speeds: list[float] = []
        for row in rows:
            if not bool(row.get("relative_velocity_valid", False)):
                continue
            vx = float(row.get("relative_velocity_x_mps", float("nan")))
            vy = float(row.get("relative_velocity_y_mps", float("nan")))
            if np.isfinite(vx) and np.isfinite(vy):
                speeds.append(float(math.hypot(vx, vy)))

        result["median_center_error_m"] = median_center
        result["velocity_sample_count"] = len(speeds)
        if len(speeds) < int(self.cfg.terminal_fallback_min_velocity_samples):
            result["reason"] = "INSUFFICIENT_RELATIVE_VELOCITY_SAMPLES"
            return result

        mean_speed = float(np.mean(np.asarray(speeds, dtype=np.float64)))
        max_speed = float(np.max(np.asarray(speeds, dtype=np.float64)))
        result["mean_relative_speed_mps"] = mean_speed
        result["max_relative_speed_mps"] = max_speed
        result["valid"] = True

        center_pass = median_center <= float(
            self.cfg.terminal_fallback_center_median_m
        )
        mean_speed_pass = mean_speed <= float(
            self.cfg.terminal_fallback_mean_relative_speed_mps
        )
        max_speed_pass = max_speed <= float(
            self.cfg.terminal_fallback_max_relative_speed_mps
        )
        result["passed"] = bool(center_pass and mean_speed_pass and max_speed_pass)
        result["reason"] = (
            "PASS"
            if result["passed"]
            else "CENTER_OR_RELATIVE_SPEED_GATE_FAILED"
        )
        return result

    def _collision_object_matches_target(self, collision_object: str) -> tuple[bool, str]:
        """Match AirSim's collision actor name against the configured target actor."""
        collision_name = self._normalized_collision_object_name(collision_object)
        candidate_names = [
            str(getattr(self, "_target_actor_name", "") or ""),
            str(getattr(self, "_expected_collision_object_name", "") or ""),
            str(getattr(self.cfg, "latch_target_actor_name", "") or ""),
        ]
        normalized_candidates = [
            self._normalized_collision_object_name(name)
            for name in candidate_names
            if self._normalized_collision_object_name(name)
        ]
        if not collision_name:
            return False, ""
        for expected in normalized_candidates:
            if collision_name == expected:
                return True, expected
            # Unreal PIE may prefix an otherwise identical actor instance name.
            if collision_name.endswith(expected) or expected.endswith(collision_name):
                return True, expected
        return False, normalized_candidates[0] if normalized_candidates else ""

    def _update_verified_alignment_latch(self, info: dict[str, Any]) -> None:
        """Store strict alignment diagnostics and the recent legacy XY window."""
        self._record_touchdown_legacy_sample(info)
        live_match = bool(info.get("bottom_match_live", False))
        confirmed = bool(info.get("bottom_match_confirmed", False))
        center_error = float(info.get("bottom_center_error", 999.0))
        bbox_rel = float(info.get("bottom_bbox_rel_err", 999.0))
        similarity = float(info.get("bottom_similarity", 0.0))
        err_x = float(info.get("bottom_err_x", 999.0))
        err_y = float(info.get("bottom_err_y", 999.0))
        height_m = float(info.get("relative_height_to_target_m", float("inf")))
        strict_alignment = bool(
            live_match
            and confirmed
            and np.isfinite(center_error)
            and center_error <= float(self.cfg.collision_latch_center_error)
        )
        if not strict_alignment:
            return
        self._last_verified_alignment_step = int(self._step)
        self._last_verified_alignment_center_error = float(center_error)
        self._last_verified_alignment_bbox_rel_error = float(bbox_rel)
        self._last_verified_alignment_similarity = float(similarity)
        self._last_verified_alignment_err_x = float(err_x)
        self._last_verified_alignment_err_y = float(err_y)
        self._last_verified_alignment_height_m = float(height_m)

    def _record_authorized_descent(
        self,
        pre_info: dict[str, Any],
        descent_allowed: bool,
        applied_vz_mps: float,
    ) -> None:
        """Latch only an actual descent command that passed the safety gate."""
        vz = float(applied_vz_mps)
        if not bool(descent_allowed) or vz <= float(self.cfg.authorized_descent_min_vz_mps):
            return

        center_error = float(pre_info.get("bottom_center_error", 999.0))
        bbox_rel = float(pre_info.get("bottom_bbox_rel_err", 999.0))
        similarity = float(pre_info.get("bottom_similarity", 0.0))
        err_x = float(pre_info.get("bottom_err_x", 999.0))
        err_y = float(pre_info.get("bottom_err_y", 999.0))
        height_m = float(pre_info.get("relative_height_to_target_m", float("inf")))
        live_match = bool(pre_info.get("bottom_match_live", False))
        confirmed = bool(pre_info.get("bottom_match_confirmed", False))

        # This is intentionally redundant with _vertical_control_state(). It
        # prevents a future caller from creating physical authorization without
        # current, confirmed visual evidence.
        geometry_ok, _ = self._landing_alignment_geometry(
            pre_info, acquire=False, soft_catchup=False
        )
        valid_evidence = bool(
            live_match
            and confirmed
            and np.isfinite(center_error)
            and np.isfinite(similarity)
            and geometry_ok
            and similarity >= float(self.cfg.descent_min_similarity)
        )
        if not valid_evidence:
            return

        self._authorized_descent_latched = True
        self._last_authorized_descent_step = int(self._step)
        self._last_authorized_descent_center_error = float(center_error)
        self._last_authorized_descent_bbox_rel_error = float(bbox_rel)
        self._last_authorized_descent_similarity = float(similarity)
        self._last_authorized_descent_err_x = float(err_x)
        self._last_authorized_descent_err_y = float(err_y)
        self._last_authorized_descent_height_m = float(height_m)
        self._last_authorized_descent_vz_mps = float(vz)
        self._authorized_descent_invalidated_reason = ""

    def _update_authorized_descent_latch_after_observation(
        self,
        info: dict[str, Any],
    ) -> None:
        """Keep episode-level descent authorization until reset.

        Close to contact, visual detections can disappear or become geometrically
        unreliable. Therefore age and intermediate visual drift no longer erase
        evidence that a real, alignment-authorized descent command occurred.
        A clearly off-center LIVE frame is evaluated only at collision time.
        """
        _ = info
        return

    def _recovery_request_decision(
        self,
        info: dict[str, Any],
        collision_now: bool,
    ) -> dict[str, Any]:
        """Decide whether AGENT_1P2 must return control to Agent 1."""
        result = {
            "requested": False,
            "reason": "",
            "contact_guard_active": False,
            "contact_guard_similarity": 0.0,
            "contact_guard_margin": 0.0,
        }
        if collision_now or not bool(getattr(self, "_attached_from_agent1", False)):
            return result
        if bool(info.get("bottom_match_live", False)):
            return result

        lost_steps = int(info.get("lost_steps", 0) or 0)
        no_live_duration_s = float(
            info.get("bottom_no_live_duration_s", 0.0) or 0.0
        )
        if lost_steps < int(self.cfg.recovery_no_live_match_min_steps):
            return result
        if no_live_duration_s < float(self.cfg.recovery_no_live_match_timeout_s):
            return result

        relative_height = float(info.get("relative_height_to_target_m", float("inf")))
        if (
            relative_height <= float(self.cfg.recovery_contact_guard_max_height_m)
            and no_live_duration_s
            <= float(self.cfg.recovery_contact_guard_max_no_live_s)
        ):
            contact = self._contact_appearance_evidence()
            result["contact_guard_similarity"] = float(contact["center_similarity"])
            result["contact_guard_margin"] = float(contact["center_margin"])
            if bool(contact["valid"]):
                # This is only a short collision/contact grace period. Descent
                # remains blocked without LIVE MATCH, and after the grace limit
                # Agent 1 recovery is requested even if appearance stays high.
                result["contact_guard_active"] = True
                return result

        result["requested"] = True
        result["reason"] = (
            f"bottom_live_match_missing_{no_live_duration_s:.2f}s_"
            f"{lost_steps}_steps"
        )
        return result

    def _dense_landing_reward(
        self,
        info: dict[str, Any],
        *,
        raw_vz_action: float,
        descent_allowed: bool,
        applied_vz_mps: float,
    ) -> tuple[float, dict[str, float]]:
        """Reward real aligned Z progress and discourage safe-to-descend hover.

        Agent 2 cannot physically control XY in the parallel architecture, so
        horizontal centering is used only as a safety condition/quality weight.
        The reward never pays for center quality alone. This prevents a centered
        hover from becoming more profitable than completing touchdown.
        """
        parts = {
            "aligned_descent_progress": 0.0,
            "landing_lock_time": 0.0,
            "hesitation": 0.0,
            "unsafe_descent": 0.0,
        }
        if not bool(self.cfg.dense_reward_enabled):
            return 0.0, parts

        live = bool(info.get("bottom_match_live", False))
        confirmed = bool(info.get("bottom_match_confirmed", False))
        lock_active = bool(getattr(self, "_descent_alignment_latched", False))
        center_error = float(info.get("bottom_center_error", 999.0))
        height = float(info.get("relative_height_to_target_m", float("inf")))
        max_center = max(
            1.0e-6, float(self.cfg.dense_reward_alignment_center_error)
        )
        aligned = bool(
            live
            and confirmed
            and lock_active
            and np.isfinite(center_error)
            and center_error <= max_center
        )

        if aligned:
            alignment_quality = float(
                np.clip(1.0 - center_error / max_center, 0.10, 1.0)
            )
            if self._prev_relative_height_m is not None and np.isfinite(height):
                raw_progress = float(self._prev_relative_height_m - height)
                progress_m = float(
                    np.clip(
                        raw_progress,
                        0.0,
                        max(
                            0.0,
                            float(self.cfg.dense_reward_max_progress_m_per_step),
                        ),
                    )
                )
                near_multiplier = 1.0
                if height <= float(self.cfg.dense_reward_near_touch_height_m):
                    near_multiplier = max(
                        1.0,
                        float(self.cfg.dense_reward_near_touch_multiplier),
                    )
                parts["aligned_descent_progress"] = float(
                    float(self.cfg.dense_reward_descent_progress_per_m)
                    * progress_m
                    * alignment_quality
                    * near_multiplier
                )

            # A small per-step cost starts only after a valid landing lock. It
            # makes a long hover worse than completing the already-safe descent.
            parts["landing_lock_time"] = -abs(
                float(self.cfg.dense_reward_landing_lock_time_penalty)
            )

            # When the deterministic gate says descent is safe, negative/near-
            # zero PPO Z is hesitation (negative NED-Z requests are blocked and
            # become hover). Penalize the missing positive action, not a safety-
            # blocked descent.
            if bool(descent_allowed):
                minimum = max(
                    1.0e-6,
                    float(self.cfg.dense_reward_min_descent_action_when_aligned),
                )
                positive_action = max(0.0, float(raw_vz_action))
                deficit = float(np.clip((minimum - positive_action) / minimum, 0.0, 1.0))
                parts["hesitation"] = -abs(
                    float(self.cfg.dense_reward_hesitation_penalty)
                ) * deficit
        else:
            # The gate still prevents physical descent. This small penalty tells
            # PPO not to demand a dive while LIVE geometry is visibly unsafe.
            positive_action = max(0.0, float(raw_vz_action))
            if positive_action > 0.0 and not bool(descent_allowed):
                parts["unsafe_descent"] = -abs(
                    float(self.cfg.dense_reward_unsafe_descent_penalty)
                ) * positive_action

        # Do not reward a reported height drop if no positive NED-Z command was
        # actually applied; this guards against pose/collision measurement jumps.
        if float(applied_vz_mps) <= 1.0e-4:
            parts["aligned_descent_progress"] = 0.0

        total = float(sum(parts.values()))
        return total, parts

    def _geometric_xy_error_m(
        self,
        err_x: float,
        err_y: float,
        height_m: float,
        frame_width_px: float | None = None,
        frame_height_px: float | None = None,
    ) -> tuple[float, bool]:
        """Project calibrated raw-BBox image error to radial XY metres.

        ``err_x`` and ``err_y`` are normalized by half the frame width/height.
        With square pixels, tan(VFOV/2) equals tan(HFOV/2) multiplied by H/W.
        """
        values = (float(err_x), float(err_y), float(height_m))
        if not all(np.isfinite(v) for v in values):
            return float("inf"), False
        min_h = float(self.cfg.collision_geometric_min_height_m)
        max_h = float(self.cfg.collision_geometric_max_height_m)
        if height_m < min_h or height_m > max_h:
            return float("inf"), False

        width = float(frame_width_px or self.cfg.image_width)
        height = float(frame_height_px or self.cfg.image_height)
        if width <= 1.0 or height <= 1.0:
            return float("inf"), False
        tan_h, tan_v = self._camera_projection_tangents(
            (int(round(height)), int(round(width)), 3)
        )
        offset_x_m = float(err_x) * float(height_m) * tan_h
        offset_y_m = float(err_y) * float(height_m) * tan_v
        error_m = float(math.hypot(offset_x_m, offset_y_m))
        return error_m, bool(np.isfinite(error_m))

    def _collision_xy_fusion(
        self,
        info: dict[str, Any],
        *,
        allow_historical_measurements: bool = True,
    ) -> dict[str, Any]:
        """Fuse legacy and geometric XY estimates from one observation.

        For touchdown classification, ``allow_historical_measurements`` is
        disabled so a stale alignment snapshot cannot approve a later side
        impact after the drone has already slipped.
        """
        candidates: list[tuple[str, float]] = []

        legacy_x = float(info.get("visual_relative_position_body_x_m", float("inf")))
        legacy_y = float(info.get("visual_relative_position_body_y_m", float("inf")))
        legacy_error = float(math.hypot(legacy_x, legacy_y))
        legacy_valid = bool(np.isfinite(legacy_error))
        if legacy_valid:
            candidates.append(("LEGACY_METRIC", legacy_error))

        max_age = int(self.cfg.collision_geometric_max_measurement_age_steps)
        geometric_candidates: list[tuple[str, float]] = []

        current_error, current_valid = self._geometric_xy_error_m(
            float(info.get("bottom_err_x", 999.0)),
            float(info.get("bottom_err_y", 999.0)),
            float(info.get("relative_height_to_target_m", float("inf"))),
        )
        if current_valid and bool(info.get("bottom_match_live", False)):
            geometric_candidates.append(("GEOMETRIC_CURRENT", current_error))

        if allow_historical_measurements:
            verified_age = int(
                self._step - int(getattr(self, "_last_verified_alignment_step", -999999))
            )
            verified_error, verified_valid = self._geometric_xy_error_m(
                float(getattr(self, "_last_verified_alignment_err_x", 999.0)),
                float(getattr(self, "_last_verified_alignment_err_y", 999.0)),
                float(getattr(self, "_last_verified_alignment_height_m", float("inf"))),
            )
            if verified_valid and verified_age <= max_age:
                geometric_candidates.append(("GEOMETRIC_VERIFIED", verified_error))

            authorized_age = int(
                self._step - int(getattr(self, "_last_authorized_descent_step", -999999))
            )
            authorized_error, authorized_valid = self._geometric_xy_error_m(
                float(getattr(self, "_last_authorized_descent_err_x", 999.0)),
                float(getattr(self, "_last_authorized_descent_err_y", 999.0)),
                float(getattr(self, "_last_authorized_descent_height_m", float("inf"))),
            )
            if authorized_valid and authorized_age <= max_age:
                geometric_candidates.append(("GEOMETRIC_AUTHORIZED", authorized_error))

        if geometric_candidates:
            geometric_source, geometric_error = min(
                geometric_candidates, key=lambda item: item[1]
            )
            candidates.append((geometric_source, geometric_error))
        else:
            geometric_source, geometric_error = "NONE", float("inf")

        if candidates:
            selected_source, combined_error = min(candidates, key=lambda item: item[1])
            combined_valid = True
        else:
            selected_source, combined_error, combined_valid = "NONE", float("inf"), False

        return {
            "legacy_error_m": float(legacy_error),
            "legacy_valid": bool(legacy_valid),
            "geometric_error_m": float(geometric_error),
            "geometric_valid": bool(np.isfinite(geometric_error)),
            "geometric_source": str(geometric_source),
            "combined_error_m": float(combined_error),
            "combined_valid": bool(combined_valid),
            "selected_source": str(selected_source),
            "good_xy": bool(
                combined_valid
                and combined_error <= float(self.cfg.collision_xy_threshold_m)
            ),
        }

    def _collision_reward_decision(
        self,
        info: dict[str, Any],
        collision_object: str,
        collision_now: bool = False,
        pre_contact_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Classify touchdown from physical contact and semantic target identity.

        The simulator actor name is diagnostic only and never participates in
        success. ``user_target`` identity is established at click time from
        neural embeddings and may remain valid through either a fresh visual
        MATCH, uninterrupted terminal-anchor continuity, or collision-frame
        appearance evidence. Calibrated metric center error remains an
        independent geometric gate.
        """
        if pre_contact_info is None:
            pre_contact_info = info

        live_match = bool(pre_contact_info.get("bottom_match_live", False))
        center_error = float(
            pre_contact_info.get(
                "bottom_touchdown_bbox_center_error",
                pre_contact_info.get("bottom_center_error", 999.0),
            )
        )
        bbox_rel = float(
            pre_contact_info.get(
                "bottom_touchdown_bbox_rel_err",
                pre_contact_info.get("bottom_bbox_rel_err", 999.0),
            )
        )
        similarity = float(pre_contact_info.get("bottom_similarity", 0.0))

        if collision_now:
            contact = self._contact_appearance_evidence()
        else:
            contact = {
                "valid": False,
                "center_similarity": 0.0,
                "corner_similarity": 0.0,
                "center_margin": 0.0,
                "best_center_scale": 0.0,
                "reason": "not_a_collision_step",
            }

        best_recent = self._best_recent_legacy_touchdown_sample(pre_contact_info)
        recent_error_norm = float(best_recent["center_error"]) if best_recent is not None else float("inf")
        recent_bbox_rel = float(best_recent["bbox_rel_error"]) if best_recent is not None else float("inf")
        recent_similarity = float(best_recent["similarity"]) if best_recent is not None else 0.0
        recent_age = int(self._step - int(best_recent["step"])) if best_recent is not None else 999999
        recent_source = str(best_recent.get("decision_source", "NONE")) if best_recent is not None else "NONE"
        frame_delay_ms = (
            max(0.0, (time.monotonic() - float(best_recent.get("source_monotonic", time.monotonic()))) * 1000.0)
            if best_recent is not None
            else float("inf")
        )
        recent_error_m, recent_error_m_valid = self._geometric_xy_error_m(
            float(best_recent.get("err_x", float("inf"))) if best_recent is not None else float("inf"),
            float(best_recent.get("err_y", float("inf"))) if best_recent is not None else float("inf"),
            float(best_recent.get("relative_height_m", float("inf"))) if best_recent is not None else float("inf"),
            float(best_recent.get("frame_width_px", self.cfg.image_width)) if best_recent is not None else None,
            float(best_recent.get("frame_height_px", self.cfg.image_height)) if best_recent is not None else None,
        )

        legacy_fresh_match = bool(
            best_recent is not None
            and bool(best_recent.get("confirmed", False))
            and np.isfinite(recent_error_norm)
            and recent_age <= int(self.cfg.collision_latch_max_age_steps)
        )

        # Actor names exist only in simulation and are never positive identity
        # evidence. They are used only as a negative safety veto when AirSim
        # explicitly reports contact with a known different actor.
        actor_name_match_diag, expected_collision_name_diag = (
            self._collision_object_matches_target(collision_object)
        )
        collision_object_name_available = bool(
            self._normalized_collision_object_name(collision_object)
        )
        known_wrong_object_contact = bool(
            collision_now
            and collision_object_name_available
            and bool(expected_collision_name_diag)
            and not actor_name_match_diag
        )

        # A simulator actor name is never positive identity evidence, but a
        # known mismatch is a hard veto: a safe pre-handoff snapshot cannot
        # approve physical contact with a different object.

        # Once acquired, the terminal anchor belongs to the semantic identity
        # ``user_target``. Valid uninterrupted optical-flow continuity is useful
        # close to touchdown, where a detector may see only roof/interior details
        # and therefore cannot produce a conventional full-object MATCH.
        terminal_locked = bool(
            info.get("bottom_terminal_anchor_locked", False)
            or pre_contact_info.get("bottom_terminal_anchor_locked", False)
        )
        terminal_valid = bool(
            info.get("bottom_terminal_anchor_valid", False)
            or pre_contact_info.get("bottom_terminal_anchor_valid", False)
        )
        terminal_identity_age = min(
            int(info.get("bottom_terminal_anchor_identity_age_steps", 999999)),
            int(pre_contact_info.get("bottom_terminal_anchor_identity_age_steps", 999999)),
        )
        terminal_flow_age = min(
            int(info.get("bottom_terminal_anchor_age_steps", 999999)),
            int(pre_contact_info.get("bottom_terminal_anchor_age_steps", 999999)),
        )
        terminal_identity_continuity = bool(
            collision_now
            and terminal_locked
            and terminal_valid
            and terminal_identity_age
            <= int(self.cfg.terminal_anchor_identity_grace_steps)
            and terminal_flow_age <= int(self.cfg.terminal_anchor_max_hold_steps)
        )

        # Independent semantic fallback from the current contact frame. The
        # network compares current crops against the selected target's embedding
        # bank. No actor name, YOLO class name, or simulator template is used.
        contact_identity_fallback = bool(
            collision_now
            and bool(contact.get("valid", False))
        )

        terminal_snapshot = getattr(self, "_range_terminal_contact_snapshot", None)
        terminal_snapshot_valid = bool(
            isinstance(terminal_snapshot, dict)
            and bool(terminal_snapshot.get("valid", False))
        )
        terminal_snapshot_target_id = str(
            terminal_snapshot.get("target_id", "")
            if isinstance(terminal_snapshot, dict)
            else ""
        )
        terminal_snapshot_center_error_m = float(
            terminal_snapshot.get("center_error_m", float("inf"))
            if isinstance(terminal_snapshot, dict)
            else float("inf")
        )
        terminal_snapshot_center_error_norm = float(
            terminal_snapshot.get("center_error_norm", float("inf"))
            if isinstance(terminal_snapshot, dict)
            else float("inf")
        )
        terminal_snapshot_similarity = float(
            terminal_snapshot.get("similarity", 0.0)
            if isinstance(terminal_snapshot, dict)
            else 0.0
        )
        terminal_snapshot_bbox_rel_error = float(
            terminal_snapshot.get("bbox_rel_error", float("inf"))
            if isinstance(terminal_snapshot, dict)
            else float("inf")
        )
        terminal_snapshot_captured_step = int(
            terminal_snapshot.get("captured_step", -999999)
            if isinstance(terminal_snapshot, dict)
            else -999999
        )
        terminal_snapshot_captured_monotonic = float(
            terminal_snapshot.get("captured_monotonic", float("-inf"))
            if isinstance(terminal_snapshot, dict)
            else float("-inf")
        )
        terminal_snapshot_age_steps = int(self._step - terminal_snapshot_captured_step)
        terminal_snapshot_age_s = float(
            time.monotonic() - terminal_snapshot_captured_monotonic
        )
        terminal_snapshot_identity_ok = bool(
            collision_now
            and not known_wrong_object_contact
            and bool(getattr(self, "_range_terminal_handoff_active", False))
            and terminal_snapshot_valid
            and terminal_snapshot_target_id
            == str(getattr(self, "_target_id", "user_target") or "user_target")
        )
        terminal_snapshot_center_pass = bool(
            terminal_snapshot_identity_ok
            and np.isfinite(terminal_snapshot_center_error_m)
            and terminal_snapshot_center_error_m
            <= float(self.cfg.collision_latch_center_error_m)
        )

        # Last-valid center latch. No trajectory median and no relative-
        # speed gate are used. A strong confirmed LIVE BBox refreshes this value
        # per frame; unreliable/PRED terminal frames simply leave it unchanged.
        last_valid_center_age = int(
            self._step - int(getattr(self, "_last_valid_center_step", -999999))
        )
        last_valid_center_age_s = float(
            time.monotonic()
            - float(getattr(self, "_last_valid_center_monotonic", float("-inf")))
        )
        last_valid_center_error_m = float(
            getattr(self, "_last_valid_center_error_m", float("inf"))
        )
        last_valid_center_similarity = float(
            getattr(self, "_last_valid_center_similarity", 0.0)
        )
        last_valid_center_bbox_rel = float(
            getattr(self, "_last_valid_center_bbox_rel_error", float("inf"))
        )
        last_valid_center_valid = bool(
            bool(self.cfg.terminal_last_valid_center_enabled)
            and np.isfinite(last_valid_center_error_m)
            and np.isfinite(last_valid_center_age_s)
            and last_valid_center_age_s >= 0.0
            and last_valid_center_age_s
            <= float(self.cfg.terminal_last_valid_center_max_age_s)
            and last_valid_center_similarity
            >= float(self.cfg.terminal_last_valid_center_min_similarity)
            and np.isfinite(last_valid_center_bbox_rel)
            and last_valid_center_bbox_rel
            <= float(self.cfg.terminal_last_valid_center_max_bbox_rel_error)
        )
        last_valid_center_pass = bool(
            last_valid_center_valid
            and last_valid_center_error_m
            <= float(self.cfg.terminal_last_valid_center_error_m)
        )
        terminal_kalman = self._terminal_kalman_prediction()
        terminal_kalman_valid = bool(terminal_kalman.get("valid", False))
        terminal_kalman_error_m = float(terminal_kalman.get("radial_m", float("inf")))
        terminal_kalman_pass = bool(
            terminal_kalman_valid
            and terminal_kalman_error_m
            <= float(self.cfg.terminal_kalman_rpc_center_error_m)
        )
        stale_identity_bridge = bool(
            collision_now and (last_valid_center_valid or terminal_kalman_valid)
        )

        semantic_identity_ok = bool(
            not known_wrong_object_contact
            and (
                legacy_fresh_match
                or terminal_identity_continuity
                or contact_identity_fallback
                or stale_identity_bridge
                or terminal_snapshot_identity_ok
            )
        )
        identity_age_limit = int(self.cfg.collision_latch_max_age_steps)
        if terminal_snapshot_identity_ok:
            # Once terminal handoff begins, this pre-handoff snapshot is the
            # authoritative semantic and XY evidence. Its lifetime is bounded by
            # the terminal hard timeout, not by the normal live-frame age gate.
            recent_source = "TERMINAL_HANDOFF_SAFE_SNAPSHOT"
            recent_age = int(terminal_snapshot_age_steps)
            identity_age_limit = int(terminal_snapshot_age_steps)
            recent_error_norm = float(terminal_snapshot_center_error_norm)
            recent_error_m = float(terminal_snapshot_center_error_m)
            recent_error_m_valid = bool(np.isfinite(recent_error_m))
            recent_bbox_rel = float(terminal_snapshot_bbox_rel_error)
            recent_similarity = float(terminal_snapshot_similarity)
            frame_delay_ms = (
                max(0.0, terminal_snapshot_age_s * 1000.0)
                if np.isfinite(terminal_snapshot_age_s)
                else float("inf")
            )
        elif stale_identity_bridge and not legacy_fresh_match:
            recent_source = "STALE_SEMANTIC_KINEMATIC_BRIDGE"
            identity_age_limit = int(
                self.cfg.terminal_last_valid_center_max_age_steps
            )
        elif terminal_identity_continuity and not legacy_fresh_match:
            recent_source = "TERMINAL_IDENTITY_CONTINUITY"
            recent_age = int(terminal_identity_age)
            identity_age_limit = int(self.cfg.terminal_anchor_identity_grace_steps)
            current_capture = float(
                info.get("bottom_observation_monotonic", time.monotonic())
            )
            frame_delay_ms = max(
                0.0,
                (time.monotonic() - current_capture) * 1000.0,
            )
        elif contact_identity_fallback and not legacy_fresh_match:
            recent_source = "CONTACT_FRAME_SEMANTIC_APPEARANCE"
            recent_age = 0
            identity_age_limit = 0
            current_capture = float(
                info.get("bottom_observation_monotonic", time.monotonic())
            )
            frame_delay_ms = max(
                0.0,
                (time.monotonic() - current_capture) * 1000.0,
            )
            recent_similarity = max(
                float(recent_similarity),
                float(contact.get("center_similarity", 0.0)),
            )

        # Center geometry is an independent condition. It must not be displayed
        # as FAIL merely because identity freshness failed. Final success still
        # requires both identity and center checks below.
        center_ok = bool(
            terminal_snapshot_center_pass
            or (
                recent_error_m_valid
                and recent_error_m <= float(self.cfg.collision_latch_center_error_m)
            )
        )
        fallback_used = bool(
            collision_now
            and semantic_identity_ok
            and not center_ok
            and terminal_kalman_pass
        )
        success = bool(
            collision_now and semantic_identity_ok and (center_ok or fallback_used)
        )

        if collision_now:
            center_value_text = f"{recent_error_m:.3f}m" if recent_error_m_valid else "INVALID"
            delay_text = f"{frame_delay_ms:.1f}ms" if np.isfinite(frame_delay_ms) else "INVALID"
            lines = [
                "=" * 88,
                "[A2 LANDING DECISION]",
                (
                    f"Target contact : {'PASS' if collision_now else 'FAIL'} | "
                    f"semantic_id={getattr(self, '_target_id', 'user_target')} sensor=CONTACT_EVENT"
                ),
                (
                    f"Bottom MATCH   : {'PASS' if semantic_identity_ok else 'FAIL'} | "
                    f"source={recent_source} age={recent_age} "
                    f"limit<={identity_age_limit}"
                ),
                (
                    f"Last valid XY : "
                    f"{'PASS' if last_valid_center_pass else 'FAIL'} | "
                    f"value={last_valid_center_error_m:.3f}m "
                    f"age={last_valid_center_age}steps/{last_valid_center_age_s:.2f}s "
                    f"limit<={self.cfg.terminal_last_valid_center_error_m:.3f}m/"
                    f"{self.cfg.terminal_last_valid_center_max_age_s:.2f}s "
                    f"sim={last_valid_center_similarity:.3f}"
                ),
                (
                    f"Terminal snap : "
                    f"{'PASS' if terminal_snapshot_center_pass else 'FAIL'} | "
                    f"valid={int(terminal_snapshot_valid)} "
                    f"target={terminal_snapshot_target_id or 'NONE'} "
                    f"xy={terminal_snapshot_center_error_m:.3f}m "
                    f"sim={terminal_snapshot_similarity:.3f} "
                    f"age={terminal_snapshot_age_s:.2f}s"
                ),
                (
                    f"Center error   : {'PASS' if center_ok else 'FAIL'} | "
                    f"value={center_value_text} "
                    f"limit<={self.cfg.collision_latch_center_error_m:.3f}m"
                ),
                (
                    f"Kalman XY     : "
                    f"{'PASS' if terminal_kalman_pass else 'FAIL'} | "
                    f"value={terminal_kalman_error_m:.3f}m "
                    f"age={float(terminal_kalman.get('age_s', float('inf'))):.2f}s "
                    f"std={float(terminal_kalman.get('position_std_m', float('inf'))):.3f}m "
                    f"limit<={self.cfg.terminal_kalman_rpc_center_error_m:.3f}m"
                ),
                (
                    f"Fallback      : "
                    f"{'PASS' if fallback_used else 'FAIL'} | "
                    f"method=KALMAN_PREDICT_ONLY "
                    f"source={getattr(self, '_last_valid_center_source', 'NONE')}"
                ),
                (
                    f"Quality only   : bboxRel={recent_bbox_rel:.3f} "
                    f"similarity={recent_similarity:.3f} "
                    f"contactSim={float(contact['center_similarity']):.3f} "
                    f"frameDelay={delay_text}"
                ),
                f"Decision       : {'SUCCESS -> RPC' if success else 'FAIL -> NO RPC'}",
                "=" * 88,
            ]
            _landing_console_print("\n".join(lines))

        if not collision_now:
            reject_reason = ""
        elif known_wrong_object_contact:
            reject_reason = "known_wrong_object_contact"
        elif not semantic_identity_ok:
            reject_reason = "semantic_target_identity_not_confirmed"
        elif not center_ok and not fallback_used:
            reject_reason = "recent_center_and_last_valid_center_failed"
        else:
            reject_reason = ""

        last_verified_step = int(getattr(self, "_last_verified_alignment_step", -999999))
        last_authorized_step = int(getattr(self, "_last_authorized_descent_step", -999999))
        if not success:
            success_path = "NONE"
        elif terminal_snapshot_center_pass:
            success_path = "CONTACT_TERMINAL_HANDOFF_SAFE_SNAPSHOT"
        elif fallback_used:
            success_path = "CONTACT_SEMANTIC_IDENTITY_RECENT_HISTORY_FALLBACK"
        elif legacy_fresh_match:
            success_path = "CONTACT_FRESH_SEMANTIC_MATCH_CENTER"
        elif terminal_identity_continuity:
            success_path = "CONTACT_TERMINAL_IDENTITY_CONTINUITY_CENTER"
        else:
            success_path = "CONTACT_FRAME_SEMANTIC_APPEARANCE_CENTER"

        return {
            "success": success,
            "success_path": success_path,
            "live_match_at_collision": live_match,
            "live_center_error": center_error,
            "live_bbox_rel_error": bbox_rel,
            "live_similarity": similarity,
            "contact_appearance_valid": bool(contact["valid"]),
            "contact_center_similarity": float(contact["center_similarity"]),
            "contact_corner_similarity": float(contact["corner_similarity"]),
            "contact_center_margin": float(contact["center_margin"]),
            "contact_best_center_scale": float(contact["best_center_scale"]),
            "contact_appearance_reason": str(contact["reason"]),
            "collision_xy_threshold_m": float(self.cfg.collision_xy_threshold_m),
            "collision_xy_legacy_error_m": float(recent_error_norm),
            "collision_xy_legacy_valid": bool(legacy_fresh_match),
            "collision_bottom_identity_ok": bool(semantic_identity_ok),
            "collision_legacy_fresh_match": bool(legacy_fresh_match),
            "collision_contact_identity_fallback": bool(contact_identity_fallback),
            "collision_terminal_identity_continuity": bool(terminal_identity_continuity),
            "collision_terminal_snapshot_valid": bool(terminal_snapshot_valid),
            "collision_terminal_snapshot_identity_ok": bool(terminal_snapshot_identity_ok),
            "collision_terminal_snapshot_center_pass": bool(terminal_snapshot_center_pass),
            "collision_terminal_snapshot_center_error_m": float(
                terminal_snapshot_center_error_m
            ),
            "collision_terminal_snapshot_similarity": float(
                terminal_snapshot_similarity
            ),
            "collision_terminal_snapshot_bbox_rel_error": float(
                terminal_snapshot_bbox_rel_error
            ),
            "collision_terminal_snapshot_age_steps": int(
                terminal_snapshot_age_steps
            ),
            "collision_terminal_snapshot_age_s": float(terminal_snapshot_age_s),
            "collision_semantic_identity_ok": bool(semantic_identity_ok),
            "collision_semantic_identity_source": str(recent_source),
            "collision_semantic_identity_age": int(recent_age),
            "collision_semantic_identity_age_limit": int(identity_age_limit),
            "collision_stale_identity_bridge_used": bool(stale_identity_bridge),
            "collision_stale_identity_bridge_semantics_ok": bool(last_valid_center_valid),
            "collision_stale_identity_bridge_kinematics_ok": True,
            "collision_center_ok": bool(center_ok),
            "collision_terminal_fallback_used": bool(fallback_used),
            "collision_terminal_fallback_valid": bool(terminal_kalman_valid),
            "collision_terminal_fallback_passed": bool(terminal_kalman_pass),
            "collision_terminal_fallback_reason": (
                "PASS" if terminal_kalman_pass else "KALMAN_PREDICTION_FAILED"
            ),
            "collision_terminal_fallback_sample_count": int(
                1 if last_valid_center_valid else 0
            ),
            "collision_terminal_fallback_velocity_sample_count": 0,
            "collision_terminal_fallback_median_center_error_m": float(
                last_valid_center_error_m
            ),
            "collision_terminal_fallback_mean_relative_speed_mps": float("nan"),
            "collision_terminal_fallback_max_relative_speed_mps": float("nan"),
            "collision_last_valid_center_age_steps": int(last_valid_center_age),
            "collision_last_valid_center_age_s": float(last_valid_center_age_s),
            "collision_last_valid_center_similarity": float(
                last_valid_center_similarity
            ),
            "collision_xy_geometric_error_m": float(recent_error_m),
            "collision_xy_geometric_valid": bool(recent_error_m_valid),
            "collision_xy_geometric_source": str(recent_source),
            "collision_xy_combined_error_m": float(recent_error_m),
            "collision_xy_combined_valid": bool(recent_error_m_valid),
            "collision_xy_selected_source": str(recent_source),
            "collision_recent_legacy_age": int(recent_age),
            "collision_recent_center_error": float(recent_error_norm),
            "collision_recent_center_error_m": float(recent_error_m),
            "collision_recent_frame_delay_ms": float(frame_delay_ms),
            "collision_recent_source": str(recent_source),
            "collision_recent_bbox_rel_error": float(recent_bbox_rel),
            "collision_recent_similarity": float(recent_similarity),
            "collision_center_threshold_norm": float(self.cfg.collision_latch_center_error),
            "collision_center_threshold_m": float(self.cfg.collision_latch_center_error_m),
            "collision_bbox_rel_threshold_norm": float(self.cfg.collision_latch_bbox_rel_error),
            "alignment_latch_age": int(self._step - last_verified_step),
            "last_verified_center_error": float(getattr(self, "_last_verified_alignment_center_error", 999.0)),
            "last_verified_bbox_rel_error": float(getattr(self, "_last_verified_alignment_bbox_rel_error", 999.0)),
            "last_verified_similarity": float(getattr(self, "_last_verified_alignment_similarity", 0.0)),
            "authorized_descent_latched": bool(getattr(self, "_authorized_descent_latched", False)),
            "recent_authorized_descent": False,
            "authorized_descent_latch_age": int(self._step - last_authorized_step),
            "last_authorized_descent_center_error": float(getattr(self, "_last_authorized_descent_center_error", 999.0)),
            "last_authorized_descent_bbox_rel_error": float(getattr(self, "_last_authorized_descent_bbox_rel_error", 999.0)),
            "last_authorized_descent_similarity": float(getattr(self, "_last_authorized_descent_similarity", 0.0)),
            "last_authorized_descent_vz_mps": float(getattr(self, "_last_authorized_descent_vz_mps", 0.0)),
            "authorized_descent_invalidated_reason": "diagnostic_only",
            # Actor-name fields are primarily diagnostics. A known mismatch
            # is used only as a negative safety veto; a match never proves
            # semantic target identity.
            "collision_object_matches_target": bool(actor_name_match_diag),
            "collision_object_name_available": bool(collision_object_name_available),
            "collision_known_wrong_object_contact": bool(known_wrong_object_contact),
            "collision_object_lock_created": False,
            "collision_object_lock_eligible": False,
            "expected_collision_object_name": str(expected_collision_name_diag),
            "expected_collision_object_source": "diagnostic_only",
            "collision_actor_name_used_for_decision": False,
            "collision_ground_contact": False,
            "reject_reason": reject_reason,
        }

    def _terminal_landing_quality_reward(
        self, collision_decision: dict[str, Any]
    ) -> tuple[float, dict[str, float]]:
        """Return reward-only landing quality terms; never gates success."""
        center = float(collision_decision.get("collision_recent_center_error_m", float("inf")))
        bbox_rel = float(collision_decision.get("collision_recent_bbox_rel_error", float("inf")))
        similarity = float(collision_decision.get("collision_recent_similarity", 0.0))

        center_limit = max(1.0e-6, float(self.cfg.collision_latch_center_error_m))
        center_quality = float(np.clip(1.0 - center / center_limit, 0.0, 1.0)) if np.isfinite(center) else 0.0

        bbox_ref = max(1.0e-6, float(self.cfg.landing_quality_bbox_rel_reference))
        bbox_quality = float(np.clip(1.0 - bbox_rel / bbox_ref, -1.0, 1.0)) if np.isfinite(bbox_rel) else -1.0

        sim_ref = float(self.cfg.landing_quality_similarity_reference)
        sim_den = max(1.0e-6, 1.0 - sim_ref)
        similarity_quality = float(np.clip((similarity - sim_ref) / sim_den, -1.0, 1.0)) if np.isfinite(similarity) else -1.0

        parts = {
            "center": float(self.cfg.landing_quality_center_bonus) * center_quality,
            "bbox_rel": float(self.cfg.landing_quality_bbox_rel_bonus) * bbox_quality,
            "similarity": float(self.cfg.landing_quality_similarity_bonus) * similarity_quality,
        }
        return float(sum(parts.values())), parts

    def begin_first_contact_window(self) -> None:
        """Clear the Agent-2-owned contact latch before one physical command."""
        with self._collision_monitor_lock:
            self._collision_monitor_latched = None

    def poll_first_contact(self) -> bool:
        """Poll AirSim synchronously from Agent 2 and latch first new contact.

        This method is invoked while the fused command is physically active.
        The collision RPC is owned and executed by Agent 2; Agent 1 only calls
        this callback and reacts to its boolean result by stopping motion.
        """
        with self._collision_monitor_lock:
            if self._collision_monitor_latched is not None:
                return True
        try:
            col = self.client.simGetCollisionInfo(vehicle_name=self.cfg.vehicle_name)
            collided = bool(getattr(col, "has_collided", False))
            timestamp = int(getattr(col, "time_stamp", 0) or 0)
            object_name = str(getattr(col, "object_name", "") or "")
            baseline = int(getattr(self, "_collision_timestamp_at_reset", 0) or 0)
            is_new = bool(collided and (timestamp == 0 or timestamp != baseline))
            if not is_new:
                return False
            with self._collision_monitor_lock:
                if self._collision_monitor_latched is None:
                    self._collision_monitor_latched = (object_name, timestamp)
            print(
                "[A2 FIRST CONTACT] synchronous detection | "
                f"object={object_name or 'UNKNOWN'} timestamp={timestamp} "
                f"poll={float(self.cfg.collision_poll_interval_s) * 1000.0:.1f}ms"
            )
            return True
        except Exception as exc:
            # Surface RPC failures instead of silently swallowing them. A
            # throttled message avoids flooding the training log.
            now = time.monotonic()
            last = float(getattr(self, "_last_collision_probe_error_log_s", 0.0))
            if now - last >= 1.0:
                self._last_collision_probe_error_log_s = now
                print(f"[A2 COLLISION PROBE ERROR] {type(exc).__name__}: {exc}")
            return False

    def _execute_with_agent2_collision_monitor(
        self,
        executor: Callable[[float], dict[str, Any]],
        vz_mps: float,
    ) -> tuple[dict[str, Any], bool, str, int]:
        """Execute one command with synchronous Agent-2 contact polling.

        DroneEnv invokes ``poll_first_contact`` every polling interval while
        the 0.25-second fused command is active. No msgpack RPC client is used
        from a background thread, avoiding coroutine/thread failures. The long
        asynchronous bridge is disabled in parallel landing.
        """
        self.begin_first_contact_window()
        external_info = dict(executor(vz_mps) or {})
        with self._collision_monitor_lock:
            latched = self._collision_monitor_latched
            self._collision_monitor_latched = None
        if latched is None:
            return external_info, False, "", 0
        object_name, timestamp = latched
        print(
            "[A2 FIRST CONTACT] command stopped | "
            f"object={object_name or 'UNKNOWN'} timestamp={timestamp} "
            "motion_cancelled=1"
        )
        return external_info, True, object_name, int(timestamp)

    def _new_collision(self) -> tuple[bool, str, int]:
        """Return a collision only when it is new relative to the episode reset."""
        try:
            collision = self.client.simGetCollisionInfo(
                vehicle_name=self.cfg.vehicle_name
            )
            collided = bool(getattr(collision, "has_collided", False))
            timestamp = int(getattr(collision, "time_stamp", 0) or 0)
            object_name = str(getattr(collision, "object_name", "") or "")
            is_new = bool(
                collided
                and (
                    timestamp == 0
                    or timestamp != self._collision_timestamp_at_reset
                )
            )
            return is_new, object_name, timestamp
        except Exception:
            return False, "", 0

    def _bbox_relative_alignment_required(self, info: dict[str, Any]) -> bool:
        """Require footprint-relative centering only in the final approach.

        ``bottom_bbox_rel_err`` scales the image-center offset by bbox size. At
        high altitude the same safe image-space offset therefore looks worse
        merely because the target bbox is smaller. Keeping it as an unconditional
        descent gate prevents the vehicle from ever getting close enough for the
        metric to become useful.
        """
        height = float(info.get("relative_height_to_target_m", float("inf")))
        if not np.isfinite(height):
            return True
        return bool(
            height <= float(self.cfg.landing_bbox_rel_gate_below_height_m)
        )

    def _landing_alignment_geometry(
        self,
        info: dict[str, Any],
        *,
        acquire: bool,
        soft_catchup: bool,
    ) -> tuple[bool, str]:
        """Evaluate filtered landing geometry without a high-altitude deadlock.

        Image-center error is always required. BBOX-relative error is added only
        in the final approach, where it correctly asks whether the camera center
        lies inside the target footprint.
        """
        center_error = float(info.get("bottom_center_error", 999.0))
        bbox_rel = float(info.get("bottom_bbox_rel_err", 999.0))

        if soft_catchup:
            center_limit = float(
                self.cfg.catchup_descent_enter_center_error
                if acquire
                else self.cfg.catchup_descent_exit_center_error
            )
        else:
            center_limit = float(
                self.cfg.alignment_enter_center_error
                if acquire
                else self.cfg.alignment_exit_center_error
            )

        if not np.isfinite(center_error) or center_error > center_limit:
            return False, "target_center_error_too_large"

        if self._bbox_relative_alignment_required(info):
            bbox_limit = float(
                self.cfg.alignment_enter_bbox_rel_error
                if acquire
                else self.cfg.alignment_exit_bbox_rel_error
            )
            if not np.isfinite(bbox_rel) or bbox_rel > bbox_limit:
                return False, "target_bbox_relative_error_too_large_near_contact"

        return True, "safe_filtered_landing_geometry"

    def _catchup_descent_gate(self, info: dict[str, Any]) -> tuple[bool, str]:
        """Use stable current-frame geometry; catch-up itself is not a Z veto.

        Instantaneous image velocity, metric outward velocity and attitude are
        derived from the same visual signal used by XY. Using them as absolute
        Z vetoes created a circular deadlock: a noisy XY correction increased
        pitch/velocity estimates, which then prevented landing forever. A
        confirmed LIVE identity and the filtered current-frame landing geometry
        are the vertical safety contract.
        """
        if not bool(self.cfg.catchup_descent_enabled):
            return False, "catchup_descent_disabled"
        if not bool(info.get("bottom_match_live", False)):
            return False, "catchup_descent_requires_live_match"
        if not bool(info.get("bottom_match_confirmed", False)):
            return False, "catchup_descent_requires_confirmed_match"
        if float(info.get("bottom_similarity", 0.0)) < float(
            self.cfg.descent_min_similarity
        ):
            return False, "catchup_descent_similarity_too_low"
        geometry_ok, geometry_reason = self._landing_alignment_geometry(
            info,
            acquire=not bool(self._descent_alignment_latched),
            soft_catchup=True,
        )
        if not geometry_ok:
            return False, f"catchup_descent_{geometry_reason}"

        # Keep one forward-looking guard, but base it on the filtered geometry
        # and a deliberately wider hysteresis window. Derivative image speed,
        # metric outward speed and instantaneous attitude are not vertical vetoes
        # because they are the noisy quantities that caused the original cycle.
        guidance = dict(getattr(self, "_last_predictive_guidance", {}) or {})
        current_center = float(info.get("bottom_center_error", 999.0))
        predicted_center = float(
            guidance.get("predicted_center_error", current_center)
        )
        if (
            np.isfinite(predicted_center)
            and predicted_center
            > float(self.cfg.catchup_descent_max_predicted_center_error)
        ):
            return False, "catchup_descent_predicted_center_escaping"
        return True, "safe_filtered_live_geometry"

    def _catchup_descent_speed_limit(self, info: dict[str, Any]) -> float:
        """Return a bounded positive NED-Z speed for safe catch-up descent."""
        limit = max(0.0, float(self.cfg.catchup_descent_max_vz_mps))
        height = float(info.get("relative_height_to_target_m", float("inf")))
        if np.isfinite(height) and height <= float(
            self.cfg.catchup_descent_touchdown_height_m
        ):
            limit = min(
                limit,
                max(0.0, float(self.cfg.catchup_descent_touchdown_max_vz_mps)),
            )
        return float(limit)

    def _bounded_descent_command(
        self,
        requested_vz_mps: float,
        vertical_state: str,
        descent_allowed: bool,
        info: dict[str, Any],
    ) -> tuple[float, float, bool]:
        """Apply the state-specific positive NED-Z limit in one testable place."""
        requested = max(0.0, float(requested_vz_mps))
        if not bool(descent_allowed):
            return 0.0, 0.0, False
        soft_catchup = str(vertical_state).startswith("DESCEND_SOFT_CATCHUP")
        if soft_catchup:
            limit = self._catchup_descent_speed_limit(info)
            return min(requested, limit), float(limit), True
        return requested, requested, False

    def _vertical_control_state(self, info: dict[str, Any]) -> tuple[str, bool, str]:
        """Fast-acquire landing lock with safe catch-up descent and gap tolerance.

        LIVE alignment acquires the lock. Predictive catch-up blocks descent only
        when the target is actually escaping or the visual evidence is unsafe.
        A confirmed, centered LIVE target may build the same landing-lock streak
        during catch-up and then descend under a strict speed cap. A brief
        PRED/no-detection gap still blocks Z immediately while preserving a
        previously acquired lock.
        """
        live_match = bool(info.get("bottom_match_live", False))
        similarity = float(info.get("bottom_similarity", 0.0))
        center_error = float(info.get("bottom_center_error", 999.0))
        bbox_rel = float(info.get("bottom_bbox_rel_err", 999.0))

        if not live_match:
            self._alignment_ready_streak = 0
            self._landing_lock_bad_live_steps = 0
            if bool(self._descent_alignment_latched):
                self._landing_lock_visual_gap_steps += 1
                if self._landing_lock_visual_gap_steps <= max(
                    0, int(self.cfg.landing_lock_max_visual_gap_steps)
                ):
                    return (
                        "HOLD_LANDING_LOCK_VISUAL_GAP",
                        False,
                        "descent_paused_during_short_visual_gap",
                    )
                self._descent_alignment_latched = False
                self._landing_lock_acquired_step = -999999
                return (
                    "HOLD_LANDING_LOCK_RELEASED",
                    False,
                    "landing_lock_released_after_sustained_visual_gap",
                )
            return "HOLD_NO_LIVE_MATCH", False, "target_not_detected_current_frame"

        self._landing_lock_visual_gap_steps = 0
        similarity_ok = bool(
            similarity >= float(self.cfg.descent_min_similarity)
        )
        if not similarity_ok:
            self._alignment_ready_streak = 0
            self._landing_lock_bad_live_steps = 0
            return (
                "HOLD_LOW_SIMILARITY",
                False,
                "resnet_similarity_below_descent_threshold",
            )

        catchup_active = bool(getattr(self, "_predictive_catchup_active", False))
        soft_catchup_ok = False
        soft_catchup_reason = "catchup_inactive"
        if catchup_active:
            soft_catchup_ok, soft_catchup_reason = self._catchup_descent_gate(info)
            if not soft_catchup_ok:
                self._landing_lock_bad_live_steps = 0
                if not bool(self._descent_alignment_latched):
                    self._alignment_ready_streak = 0
                return (
                    "HOLD_PREDICTIVE_CATCHUP",
                    False,
                    soft_catchup_reason,
                )

        if bool(self._descent_alignment_latched):
            if catchup_active:
                keep_ok = bool(soft_catchup_ok)
            else:
                keep_ok, _ = self._landing_alignment_geometry(
                    info, acquire=False, soft_catchup=False
                )
                keep_ok = bool(similarity_ok and keep_ok)
            if keep_ok:
                self._landing_lock_bad_live_steps = 0
                if catchup_active:
                    return (
                        "DESCEND_SOFT_CATCHUP_LANDING_LOCK",
                        True,
                        "",
                    )
                return "DESCEND_LANDING_LOCK", True, ""

            self._landing_lock_bad_live_steps += 1
            if self._landing_lock_bad_live_steps >= max(
                1, int(self.cfg.landing_lock_bad_live_release_steps)
            ):
                self._descent_alignment_latched = False
                self._landing_lock_acquired_step = -999999
                self._alignment_ready_streak = 0
                self._episode_recenter_steps = int(
                    getattr(self, "_episode_recenter_steps", 0)
                ) + 1
                return (
                    "RECENTER_LANDING_LOCK_RELEASED",
                    False,
                    "landing_lock_released_after_sustained_live_misalignment",
                )

            return (
                "RECENTER_LANDING_LOCK_HELD",
                False,
                "descent_paused_but_landing_lock_preserved",
            )

        self._landing_lock_bad_live_steps = 0
        if catchup_active:
            enter_ok = bool(soft_catchup_ok)
            enter_reason = str(soft_catchup_reason)
        else:
            enter_ok, enter_reason = self._landing_alignment_geometry(
                info, acquire=True, soft_catchup=False
            )
        if enter_ok:
            self._alignment_ready_streak += 1
        else:
            self._alignment_ready_streak = max(0, self._alignment_ready_streak - 1)

        required = max(1, int(self.cfg.alignment_streak_required))
        if self._alignment_ready_streak >= required:
            self._descent_alignment_latched = True
            self._landing_lock_acquired_step = int(getattr(self, "_step", 0))
            self._landing_lock_visual_gap_steps = 0
            self._landing_lock_bad_live_steps = 0
            if catchup_active:
                return (
                    "DESCEND_SOFT_CATCHUP_LOCK_ACQUIRED",
                    True,
                    "",
                )
            return "DESCEND_LANDING_LOCK_ACQUIRED", True, ""
        if not enter_ok:
            state = (
                "ALIGN_BBOX"
                if "bbox_relative" in str(enter_reason)
                else "ALIGN_CENTER"
            )
            return state, False, str(enter_reason)
        if catchup_active:
            return (
                "ALIGN_LOCK_PENDING_SOFT_CATCHUP",
                False,
                "safe_catchup_waiting_for_landing_lock_streak",
            )
        return "ALIGN_LOCK_PENDING", False, "landing_lock_streak_too_short"

    def _reacquire_climb_command(
        self,
        info: dict[str, Any],
    ) -> tuple[float, bool, str]:
        """Return a bounded negative NED-Z command only for visual reacquisition."""
        if not (
            bool(self.cfg.parallel_dual_agent_mode)
            and bool(self.cfg.reacquire_climb_enabled)
        ):
            self._reacquire_climb_started_monotonic = None
            self._reacquire_climb_active = False
            return 0.0, False, "disabled"

        if bool(info.get("bottom_match_live", False)):
            self._reacquire_climb_started_monotonic = None
            self._reacquire_climb_active = False
            return 0.0, False, "bottom_live"

        no_live_s = float(info.get("bottom_no_live_duration_s", 0.0) or 0.0)
        if no_live_s < float(self.cfg.reacquire_climb_after_s):
            self._reacquire_climb_started_monotonic = None
            self._reacquire_climb_active = False
            return 0.0, False, "short_visual_gap"

        height = float(info.get("relative_height_to_target_m", float("inf")))
        if not np.isfinite(height):
            self._reacquire_climb_active = False
            return 0.0, False, "invalid_height"
        if height >= float(self.cfg.reacquire_climb_target_height_m):
            self._reacquire_climb_active = False
            return 0.0, False, "reacquire_height_reached"

        now = float(time.monotonic())
        if self._reacquire_climb_started_monotonic is None:
            self._reacquire_climb_started_monotonic = now
        elapsed = float(now - self._reacquire_climb_started_monotonic)
        if elapsed > float(self.cfg.reacquire_climb_max_duration_s):
            self._reacquire_climb_active = False
            return 0.0, False, "reacquire_climb_timeout"

        self._reacquire_climb_active = True
        return (
            -abs(float(self.cfg.reacquire_climb_speed_mps)),
            True,
            "bottom_lost_expand_field_of_view",
        )

    def _latch_vehicle_after_success(self) -> tuple[bool, str]:
        """Snap the drone to the target roof after verified touchdown only."""
        if not bool(self.cfg.latch_on_success):
            return False, "disabled"

        target_actor = str(
            getattr(self, "_target_actor_name", "")
            or self.cfg.latch_target_actor_name
        )
        vehicle_name = str(self.cfg.latch_vehicle_name)
        anchor_name = str(self.cfg.latch_anchor_component_name)
        print(
            "[A2 LATCH] attempting RPC | "
            f"vehicle={vehicle_name or '<default>'} "
            f"target={target_actor} anchor={anchor_name}"
        )

        try:
            result = bool(
                self.client.client.call(
                    "simLatchVehicleToActor",
                    vehicle_name,
                    target_actor,
                    anchor_name,
                )
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"[A2 LATCH] RPC error | {message}")
            return False, message

        if not result:
            message = "RPC returned False"
            print(f"[A2 LATCH] failed | {message}")
            return False, message

        hold_seconds = max(0.0, float(self.cfg.latch_success_hold_seconds))
        print(
            "[A2 LATCH] success | "
            f"vehicle={self.cfg.latch_vehicle_name or '<default>'} "
            f"target={getattr(self, '_target_actor_name', '') or self.cfg.latch_target_actor_name} "
            f"anchor={self.cfg.latch_anchor_component_name} "
            f"hold={hold_seconds:.1f}s"
        )
        if hold_seconds > 0.0:
            time.sleep(hold_seconds)
        return True, ""

    def _capture_range_terminal_contact_snapshot(
        self,
        info: dict[str, Any],
        *,
        captured_monotonic: float,
        safe_range_m: float,
    ) -> dict[str, Any]:
        """Freeze the last trustworthy landing evidence before terminal descent.

        The snapshot is captured only while calibrated range geometry and recent
        LIVE visual evidence are both safe. It is immutable after terminal
        handoff and therefore remains authoritative when close-range imagery is
        distorted, temporarily missing, or semantically ambiguous.
        """
        center_error_m = float(
            getattr(self, "_last_valid_center_error_m", float("inf"))
        )
        similarity = float(getattr(self, "_last_valid_center_similarity", 0.0))
        bbox_rel_error = float(
            getattr(self, "_last_valid_center_bbox_rel_error", float("inf"))
        )
        target_id = str(getattr(self, "_target_id", "user_target") or "user_target")

        bbox = getattr(self, "_control_bbox_xyxy", None)
        if bbox is None:
            bbox = getattr(self, "_last_bbox_xyxy", None)
        bbox_list: list[float] | None = None
        if bbox is not None:
            bbox_arr = np.asarray(bbox, dtype=np.float64).reshape(-1)
            if bbox_arr.size == 4 and np.all(np.isfinite(bbox_arr)):
                bbox_list = [float(value) for value in bbox_arr]

        center_error_norm = float(
            info.get(
                "bottom_touchdown_bbox_center_error",
                info.get("bottom_center_error", float("inf")),
            )
        )
        snapshot_valid = bool(
            target_id == "user_target"
            and np.isfinite(center_error_m)
            and center_error_m
            <= float(self.cfg.range_sensor_final_max_center_error_m)
            and np.isfinite(similarity)
            and similarity
            >= float(self.cfg.terminal_last_valid_center_min_similarity)
            and np.isfinite(bbox_rel_error)
            and bbox_rel_error
            <= float(self.cfg.terminal_last_valid_center_max_bbox_rel_error)
            and np.isfinite(safe_range_m)
            and float(self.cfg.range_sensor_final_stop_m) < safe_range_m
            <= float(self.cfg.range_terminal_contact_arm_max_height_m)
        )

        snapshot = {
            "valid": snapshot_valid,
            "captured_step": int(self._step),
            "captured_monotonic": float(captured_monotonic),
            "target_id": target_id,
            "target_actor_name": str(getattr(self, "_target_actor_name", "") or ""),
            "center_error_m": center_error_m,
            "center_error_norm": center_error_norm,
            "similarity": similarity,
            "bbox_rel_error": bbox_rel_error,
            "bbox_xyxy": bbox_list,
            "last_safe_range_m": float(safe_range_m),
            "source": str(getattr(self, "_last_valid_center_source", "NONE")),
            "handoff_step": None,
            "handoff_monotonic": None,
        }
        self._range_terminal_contact_snapshot = snapshot
        return snapshot

    def _terminal_low_altitude_z_command(
        self,
        raw_vz_action: float,
        info: dict[str, Any],
    ) -> tuple[float, str, bool, str]:
        """Return range-governed terminal Z with conditional climb permission.

        Positive NED-Z descends and negative NED-Z climbs. Once terminal handoff
        is active, calibrated range geometry owns the normal descent command.
        Agent 2 may contribute an upward command only when current evidence
        indicates a large horizontal deviation or unsafe/asymmetric roof
        geometry. When every beam becomes invalid after the safely armed
        low-altitude handoff, the array is treated as being below its measured
        minimum and a gentle descent continues until contact or timeout.
        """
        action = float(np.clip(raw_vz_action, -1.0, 1.0))
        reliable = bool(info.get("range_height_reliable", False))
        valid_count = int(info.get("range_valid_count", 0) or 0)
        mean_m = float(info.get("range_mean_m", float("inf")))
        spread_m = float(info.get("range_spread_m", float("inf")))

        live_match = bool(info.get("bottom_match_live", False))
        similarity = float(info.get("bottom_similarity", 0.0))
        center_error = float(
            info.get(
                "bottom_touchdown_bbox_center_error",
                info.get("bottom_center_error", float("inf")),
            )
        )
        visual_deviation = bool(
            live_match
            and similarity >= float(self.cfg.range_sensor_final_min_similarity)
            and np.isfinite(center_error)
            and center_error
            > float(self.cfg.range_terminal_climb_center_error_threshold)
        )

        asymmetric_range = bool(
            valid_count >= 2
            and np.isfinite(spread_m)
            and spread_m
            > float(self.cfg.range_terminal_climb_spread_threshold_m)
        )
        partial_surface = bool(0 < valid_count < int(self.cfg.range_min_valid_count))
        range_geometry_unsafe = bool(asymmetric_range or partial_surface)
        climb_allowed = bool(visual_deviation or range_geometry_unsafe)

        if valid_count <= 0 or not np.isfinite(mean_m):
            return (
                float(self.cfg.range_terminal_contact_vz_mps),
                "range_terminal_inf_below_min_descent",
                False,
                "all_ranges_inf_after_low_altitude_handoff",
            )

        if action < 0.0 and climb_allowed:
            climb_vz = -min(
                abs(action) * float(self.cfg.vz_scale_mps),
                float(self.cfg.range_terminal_climb_max_vz_mps),
            )
            reason = (
                "large_xy_deviation"
                if visual_deviation
                else "unsafe_asymmetric_range_geometry"
            )
            return climb_vz, "range_terminal_authorized_climb", True, reason

        if range_geometry_unsafe:
            return (
                0.0,
                "range_terminal_unsafe_geometry_hold",
                False,
                "climb_not_requested",
            )

        if reliable:
            if mean_m <= float(self.cfg.range_sensor_final_stop_m):
                descent_vz = float(self.cfg.range_terminal_contact_vz_mps)
            else:
                descent_vz = float(
                    self.range_finder_array.sensor_final_descent_speed(mean_m)
                )
                if descent_vz <= 0.0:
                    descent_vz = float(self.cfg.range_terminal_contact_vz_mps)
            climb_block_reason = (
                "aligned_safe_range_geometry" if action < 0.0 else "not_requested"
            )
            return (
                descent_vz,
                "range_terminal_governed_descent",
                False,
                climb_block_reason,
            )

        # A finite but unreliable low-altitude sample that is not asymmetric
        # is treated conservatively as the close-range dead zone. Continue a
        # gentle descent rather than exposing unrestricted policy Z.
        return (
            float(self.cfg.range_terminal_contact_vz_mps),
            "range_terminal_unreliable_close_descent",
            False,
            "range_not_reliable_no_climb_evidence",
        )

    def _sensor_final_command(self, info: dict[str, Any]) -> tuple[bool, float, str]:
        """Run the calibrated final-landing handoff.

        Above the experimentally verified 0.35 m floor, the existing vision
        controller remains active and the range array is monitored. A recent,
        safe range/vision state arms the one-way handoff. At or below 0.35 m,
        the existing controlled Agent-1 X/Y/Yaw path remains active while a
        range-governed Z controller runs for at most 50 physical policy commands.
        Agent 2 may request a bounded climb only when geometry justifies it. Existing safety,
        smoothing, range, vision and collision checks remain in force.

        Touchdown success remains governed by contact with the selected target
        and the last safe XY gate. The handoff itself stays irreversible and
        replaces unrestricted terminal Z with the range-governed controller.
        """
        now = float(time.monotonic())

        # Check the one-way terminal latch before every configuration, Vision,
        # calibration, or range branch. Once handoff has started, no later
        # sample may refresh the snapshot, reset the timer, or return authority
        # to the normal state machine.
        if bool(getattr(self, "_range_terminal_handoff_active", False)):
            started_step = int(
                getattr(self, "_range_terminal_contact_started_step", self._step)
            )
            if started_step < 0:
                started_step = int(self._step)
                self._range_terminal_contact_started_step = started_step
            elapsed_steps = max(0, int(self._step) - started_step)
            if elapsed_steps < int(self.cfg.range_terminal_contact_max_steps):
                return True, 0.0, "range_floor_terminal_range_control"

            # The latch remains irreversible, but timeout is counted by actual
            # physical policy steps rather than wall-clock time.
            self._range_terminal_handoff_timed_out = True
            return True, 0.0, "range_terminal_contact_hard_timeout"

        if not bool(self.cfg.range_sensor_final_enabled):
            return False, 0.0, "disabled"

        live_vision = bool(info.get("bottom_match_live", False))
        calibrated = bool(info.get("range_calibration_loaded", False))
        reliable = bool(info.get("range_height_reliable", False))
        ready = bool(info.get("range_sensor_final_ready", False))
        range_used = bool(info.get("range_height_used", False))
        mean_m = float(info.get("range_mean_m", float("inf")))

        visual_age = float(
            now - float(getattr(self, "_last_valid_center_monotonic", float("-inf")))
        )
        center = float(getattr(self, "_last_valid_center_error_m", float("inf")))
        similarity = float(getattr(self, "_last_valid_center_similarity", 0.0))
        visual_context_safe = bool(
            np.isfinite(visual_age)
            and 0.0 <= visual_age <= float(self.cfg.range_sensor_final_recent_vision_s)
            and np.isfinite(center)
            and center <= float(self.cfg.range_sensor_final_max_center_error_m)
            and similarity >= float(self.cfg.range_sensor_final_min_similarity)
        )

        if bool(self.cfg.range_require_calibration) and not calibrated:
            return False, 0.0, "range_calibration_not_loaded"

        # Pre-arm before the rays enter their measured close-surface dead zone.
        # The upper edge is intentionally wider than the old 0.60 m limit:
        # at the forced 0.65 m/s descent rate, one slow AirSim step can skip
        # the entire 0.35-0.60 m band. The snapshot is still accepted only
        # with calibrated/reliable range plus safe LIVE visual geometry, and
        # the actual terminal handoff remains fixed at the 0.35 m floor.
        if (
            calibrated
            and reliable
            and visual_context_safe
            and float(self.cfg.range_sensor_final_stop_m) < mean_m
            <= float(self.cfg.range_terminal_contact_arm_max_height_m)
        ):
            snapshot = self._capture_range_terminal_contact_snapshot(
                info,
                captured_monotonic=now,
                safe_range_m=mean_m,
            )
            if bool(snapshot.get("valid", False)):
                self._range_terminal_contact_armed = True
                self._range_terminal_contact_armed_monotonic = now
                self._range_terminal_contact_started_monotonic = float("-inf")
                self._range_terminal_contact_last_safe_height_m = mean_m

        armed_age = float(
            now - float(getattr(self, "_range_terminal_contact_armed_monotonic", float("-inf")))
        )
        terminal_snapshot = getattr(self, "_range_terminal_contact_snapshot", None)
        terminal_armed = bool(
            getattr(self, "_range_terminal_contact_armed", False)
            and isinstance(terminal_snapshot, dict)
            and bool(terminal_snapshot.get("valid", False))
            and np.isfinite(armed_age)
            and 0.0 <= armed_age <= float(self.cfg.range_terminal_contact_latch_max_age_s)
            and float(getattr(self, "_range_terminal_contact_last_safe_height_m", float("inf")))
            <= float(self.cfg.range_terminal_contact_arm_max_height_m)
        )
        below_floor = bool(
            np.isfinite(mean_m)
            and mean_m <= float(self.cfg.range_sensor_final_stop_m)
        )
        ray_became_invalid = bool(not reliable or not np.isfinite(mean_m))

        # Deterministic handoff at 0.35 m. Live vision is deliberately no longer
        # a blocker here: the handoff occurs before close-range visual geometry
        # can corrupt XY/yaw. Invalid rays may trigger the same path only from a
        # very recent safely armed sample.
        if terminal_armed and (below_floor or ray_became_invalid):
            self._range_terminal_handoff_active = True
            self._range_terminal_handoff_timed_out = False
            self._range_terminal_contact_started_monotonic = now
            self._range_terminal_contact_started_step = int(self._step)
            terminal_snapshot["handoff_step"] = int(self._step)
            terminal_snapshot["handoff_monotonic"] = float(now)
            return True, 0.0, "range_floor_terminal_range_control"

        # Keep Agent 1 in full XY/yaw control while using the calibrated range
        # array only as a Z-speed governor inside the final 1.50 m envelope.
        # This prevents a 0.65 m/s command from skipping the complete pre-arm
        # band between perception/control steps. It is not terminal authority:
        # XY/yaw are frozen only after _range_terminal_handoff_active is set.
        if live_vision:
            if calibrated and reliable and ready:
                governed_vz = float(
                    self.range_finder_array.sensor_final_descent_speed(mean_m)
                )
                if governed_vz > 0.0:
                    return True, governed_vz, "vision_live_range_z_governor"
            return False, 0.0, "vision_live_ranges_monitoring_only"

        if not visual_context_safe:
            return False, 0.0, "recent_visual_context_unsafe"

        # Natural vision loss above the floor: calibrated range may guide Z
        # while still inside the reliable final envelope.
        if reliable and ready and range_used:
            speed = float(self.range_finder_array.sensor_final_descent_speed(mean_m))
            if speed > 0.0:
                return True, speed, "recent_vision_plus_safe_range_array"

        if not reliable:
            return False, 0.0, "range_geometry_not_reliable"
        if not ready:
            return False, 0.0, "range_outside_reliable_final_envelope"
        if not range_used:
            return False, 0.0, "range_height_not_authorized"
        return False, 0.0, "range_terminal_not_armed"

    def step(self, action):
        self._step += 1
        raw = np.asarray(action, dtype=np.float32).reshape(4)
        raw = np.clip(raw, -1.0, 1.0)

        # Use the observation already returned by reset()/the previous step for
        # pre-action safety. Perception is executed exactly once per physical
        # control step, after the command, so tracker/lost counters and visual
        # dynamics advance once rather than twice.
        if not self._last_info:
            raise RuntimeError("Agent2LandingEnv.step() called before reset()/handoff observation.")
        pre_info = dict(self._last_info)
        now_monotonic = float(time.monotonic())
        terminal_kalman = self._terminal_kalman_prediction(now_monotonic)
        try:
            pre_height = float(pre_info.get("relative_height_to_target_m", float("inf")))
        except (TypeError, ValueError):
            pre_height = float("inf")
        terminal_kalman_active = bool(
            not bool(pre_info.get("bottom_match_live", False))
            and bool(getattr(self, "_terminal_descent_committed", False))
            and bool(terminal_kalman.get("valid", False))
            and np.isfinite(pre_height)
            and pre_height <= float(self.cfg.terminal_blind_descent_max_height_m)
        )
        if terminal_kalman_active:
            pre_info["tracker_mode"] = "PRED_KALMAN_TERMINAL"
            pre_info["terminal_kalman_prediction_active"] = True
            pre_info["visual_motion_age_s"] = float(terminal_kalman["age_s"])
            # Guidance expects the state at the last real measurement and
            # advances it by visual_motion_age_s internally.
            latched = np.asarray(self._terminal_kalman_state, dtype=np.float64)
            pre_info["visual_relative_position_body_x_m"] = float(latched[0])
            pre_info["visual_relative_position_body_y_m"] = float(latched[1])
            pre_info["visual_relative_velocity_body_x_mps"] = float(latched[2])
            pre_info["visual_relative_velocity_body_y_mps"] = float(latched[3])
            pre_info["visual_relative_velocity_valid"] = True
            pre_info["target_velocity_valid"] = False

        vx, vy, horizontal = self._horizontal_visual_servo(raw, pre_info)
        self._last_horizontal_control_state = str(horizontal["state"])
        self._last_pd_action_vx = float(horizontal["pd_ax"])
        self._last_pd_action_vy = float(horizontal["pd_ay"])
        self._last_residual_action_vx = float(horizontal["residual_ax"])
        self._last_residual_action_vy = float(horizontal["residual_ay"])
        self._last_horizontal_action_vx = float(horizontal["final_ax"])
        self._last_horizontal_action_vy = float(horizontal["final_ay"])
        self._last_horizontal_speed_limit_mps = float(horizontal["speed_limit_mps"])
        self._last_velocity_ff_vx_mps = float(horizontal["ff_vx_mps"])
        self._last_velocity_ff_vy_mps = float(horizontal["ff_vy_mps"])
        self._last_horizontal_correction_vx_mps = float(horizontal["correction_vx_mps"])
        self._last_horizontal_correction_vy_mps = float(horizontal["correction_vy_mps"])
        self._last_horizontal_command_vx_mps = float(horizontal["command_vx_mps"])
        self._last_horizontal_command_vy_mps = float(horizontal["command_vy_mps"])
        self._last_horizontal_total_speed_limit_mps = float(
            horizontal["total_speed_limit_mps"]
        )
        self._last_predictive_guidance.update(
            {
                "predicted_err_x": float(horizontal.get("predicted_err_x", 0.0)),
                "predicted_err_y": float(horizontal.get("predicted_err_y", 0.0)),
                "predicted_center_error": float(horizontal.get("predicted_center_error", 999.0)),
                "image_velocity_x_per_s": float(horizontal.get("image_velocity_x_per_s", 0.0)),
                "image_velocity_y_per_s": float(horizontal.get("image_velocity_y_per_s", 0.0)),
                "prediction_horizon_s": float(horizontal.get("prediction_horizon_s", self.cfg.cmd_duration_s)),
                "catchup_active": bool(horizontal.get("catchup_active", False)),
                "guidance_active": bool(horizontal.get("guidance_active", False)),
                "prediction_only": bool(horizontal.get("prediction_only", False)),
            }
        )

        # Normal landing keeps the original descent-only contract. Inside the
        # final 50-step low-altitude window, range geometry governs Z and Agent 2
        # may request a bounded climb only when the controller authorizes it.
        raw_vz_action = float(raw[2])
        bottom_live_forced_descent = bool(
            self.cfg.force_descent_while_bottom_match
            and pre_info.get("bottom_match_live", False)
        )
        climb_command_blocked = bool(raw_vz_action < 0.0)

        last_valid_elapsed_s = float(
            now_monotonic
            - float(getattr(self, "_last_valid_center_monotonic", float("-inf")))
        )
        try:
            current_relative_height_m = float(
                pre_info.get("relative_height_to_target_m", float("inf"))
            )
        except (TypeError, ValueError):
            current_relative_height_m = float("inf")
        last_valid_center_ready = bool(
            np.isfinite(float(getattr(self, "_last_valid_center_error_m", float("inf"))))
            and float(getattr(self, "_last_valid_center_error_m", float("inf")))
            <= float(self.cfg.terminal_last_valid_center_error_m)
            and float(getattr(self, "_last_valid_center_similarity", 0.0))
            >= float(self.cfg.terminal_last_valid_center_min_similarity)
            and np.isfinite(float(getattr(self, "_last_valid_center_bbox_rel_error", float("inf"))))
            and float(getattr(self, "_last_valid_center_bbox_rel_error", float("inf")))
            <= float(self.cfg.terminal_last_valid_center_max_bbox_rel_error)
        )
        terminal_blind_descent = bool(
            bool(self.cfg.terminal_blind_descent_enabled)
            and not bottom_live_forced_descent
            and bool(getattr(self, "_terminal_descent_committed", False))
            and terminal_kalman_active
            and np.isfinite(current_relative_height_m)
            and current_relative_height_m
            <= float(self.cfg.terminal_blind_descent_max_height_m)
        )

        # Forced-contact diagnostic mode: while the downward camera has a LIVE
        # MATCH, Agent 2 continuously commands positive NED-Z until first
        # collision. PPO Z, landing-lock geometry, similarity, bbox-relative
        # gates and reacquisition logic cannot interrupt this descent. This is
        # deliberately confined to Agent 2 and is intended to validate the
        # first-contact/collision path under guaranteed physical contact.
        if bottom_live_forced_descent:
            self._terminal_descent_committed = True
            self._terminal_blind_descent_started_monotonic = float("-inf")
            requested_vz = float(self.cfg.vz_scale_mps)
            vertical_state = "FORCED_DESCENT_BOTTOM_LIVE_UNTIL_COLLISION"
            descent_allowed = True
            descent_block_reason = "none_forced_bottom_live_until_collision"
            vz = float(requested_vz)
            vertical_speed_limit = float(requested_vz)
            soft_catchup_descent = False
            reacquire_vz = 0.0
            reacquire_climb_active = False
            reacquire_climb_reason = "disabled_during_forced_bottom_live_descent"
        elif terminal_blind_descent:
            if not np.isfinite(
                float(getattr(self, "_terminal_blind_descent_started_monotonic", float("-inf")))
            ):
                self._terminal_blind_descent_started_monotonic = now_monotonic
            requested_vz = float(self.cfg.forced_bottom_match_descent_vz_mps)
            vertical_state = "TERMINAL_BLIND_DESCENT_KALMAN_PREDICT"
            descent_allowed = True
            descent_block_reason = "none_terminal_blind_descent"
            vz = max(0.0, float(requested_vz))
            vertical_speed_limit = float(vz)
            soft_catchup_descent = False
            reacquire_vz = 0.0
            reacquire_climb_active = False
            reacquire_climb_reason = "disabled_during_terminal_blind_descent"
        else:
            requested_vz = max(0.0, raw_vz_action) * float(self.cfg.vz_scale_mps)
            vertical_state, descent_allowed, descent_block_reason = self._vertical_control_state(pre_info)
            vz, vertical_speed_limit, soft_catchup_descent = (
                self._bounded_descent_command(
                    requested_vz,
                    vertical_state,
                    descent_allowed,
                    pre_info,
                )
            )
            reacquire_vz, reacquire_climb_active, reacquire_climb_reason = (
                self._reacquire_climb_command(pre_info)
            )

        # Diagnostic forced-impact mode requested for collision validation.
        # A LIVE bottom-camera MATCH has absolute Z authority: keep descending
        # at a deterministic speed until Agent 2 detects first contact. This
        # bypasses PPO hesitation, landing-lock geometry and catch-up Z vetoes,
        # but only while the current bottom frame is a LIVE MATCH.
        forced_match_descent = bool(
            self.cfg.force_descent_while_bottom_match
            and pre_info.get("bottom_match_live", False)
        )
        if forced_match_descent:
            vz = max(
                0.0,
                float(self.cfg.forced_bottom_match_descent_vz_mps),
            )
            vertical_state = "FORCED_DESCEND_BOTTOM_MATCH_UNTIL_COLLISION"
            descent_allowed = True
            soft_catchup_descent = False
            vertical_speed_limit = float(vz)
            descent_block_reason = ""
            reacquire_climb_active = False
            reacquire_climb_reason = "bottom_live_forced_descent"

        sensor_final_active, sensor_final_vz, sensor_final_reason = (
            self._sensor_final_command(pre_info)
        )
        terminal_handoff_active = bool(
            getattr(self, "_range_terminal_handoff_active", False)
        )
        terminal_hard_timeout = bool(
            terminal_handoff_active
            and sensor_final_reason == "range_terminal_contact_hard_timeout"
        )
        terminal_climb_allowed = False
        terminal_climb_reason = "not_in_terminal_handoff"
        terminal_inf_descent_active = False
        # Before terminal handoff the original descent-only contract remains.
        # After handoff, the range/controller path governs Z and policy climb
        # requests are admitted only when current geometry justifies them.
        climb_command_blocked = bool(
            raw_vz_action < 0.0 and not terminal_handoff_active
        )
        if sensor_final_active:
            if terminal_handoff_active and not terminal_hard_timeout:
                (
                    terminal_vz,
                    terminal_z_reason,
                    terminal_climb_allowed,
                    terminal_climb_reason,
                ) = self._terminal_low_altitude_z_command(
                    raw_vz_action,
                    pre_info,
                )
                requested_vz = float(terminal_vz)
                vz = float(terminal_vz)
                vertical_state = "SENSOR_FINAL_TERMINAL_RANGE_CONTROL"
                descent_allowed = bool(vz >= 0.0)
                soft_catchup_descent = False
                vertical_speed_limit = abs(float(vz))
                descent_block_reason = str(terminal_z_reason)
                reacquire_climb_active = False
                reacquire_climb_reason = str(terminal_z_reason)
                terminal_inf_descent_active = bool(
                    terminal_z_reason == "range_terminal_inf_below_min_descent"
                )
                climb_command_blocked = bool(
                    raw_vz_action < 0.0 and not terminal_climb_allowed
                )
            else:
                requested_vz = float(sensor_final_vz)
                vz = float(sensor_final_vz)
                vertical_state = (
                    "SENSOR_FINAL_TERMINAL_TIMEOUT"
                    if terminal_hard_timeout
                    else "SENSOR_FINAL_RANGE_ARRAY"
                )
                descent_allowed = not terminal_hard_timeout
                soft_catchup_descent = False
                vertical_speed_limit = abs(float(sensor_final_vz))
                descent_block_reason = (
                    "range_terminal_contact_hard_timeout"
                    if terminal_hard_timeout
                    else ""
                )
                reacquire_climb_active = False
                reacquire_climb_reason = "disabled_during_sensor_final"

        descent_requested = bool(
            requested_vz > 0.0 or forced_match_descent or terminal_blind_descent or sensor_final_active
        )
        descent_blocked = bool(descent_requested and not descent_allowed)
        if reacquire_climb_active:
            vz = float(reacquire_vz)
            vertical_state = "CLIMB_REACQUIRE_BOTTOM_VIEW"
            descent_allowed = False
            soft_catchup_descent = False
            vertical_speed_limit = 0.0
            descent_blocked = bool(descent_requested)
            descent_block_reason = reacquire_climb_reason

        self._last_vertical_control_state = vertical_state
        self._last_descent_block_reason = descent_block_reason
        self._last_raw_vz_action = float(raw_vz_action)
        self._last_requested_vz_mps = float(requested_vz)
        self._last_applied_vz_mps = float(vz)
        self._last_vertical_speed_limit_mps = float(
            vertical_speed_limit if np.isfinite(vertical_speed_limit) else requested_vz
        )
        self._last_soft_catchup_descent_active = bool(soft_catchup_descent and vz > 1.0e-4)
        self._last_climb_command_blocked = bool(climb_command_blocked)

        if climb_command_blocked:
            self._episode_climb_command_blocked_steps += 1
        if descent_requested:
            self._episode_descent_requested_steps += 1
            if descent_blocked:
                self._episode_descent_blocked_steps += 1
            else:
                self._episode_descent_allowed_steps += 1
        yaw_rate = float(np.clip(raw[3], -0.20, 0.20)) * self.cfg.yaw_scale_dps
        vx, vy, guard_reasons = self._horizontal_lidar_guard(vx, vy, pre_info)

        if terminal_handoff_active:
            # Agent 1 remains on the existing controlled X/Y/Yaw path. Terminal
            # Z is already bounded by the range governor above; do not restore
            # unrestricted signed policy authority at the command boundary.
            reacquire_climb_active = False
            vz = float(
                np.clip(
                    vz,
                    -float(self.cfg.range_terminal_climb_max_vz_mps),
                    float(self.cfg.vz_scale_mps),
                )
            )
        elif reacquire_climb_active:
            vz = min(0.0, float(vz))
        else:
            vz = max(0.0, float(vz))
        self._last_applied_vz_mps = float(vz)
        self._record_authorized_descent(pre_info, descent_allowed, vz)

        external_info: dict[str, Any] = {}
        collision_now = False
        collision_object = ""
        collision_timestamp = 0
        external_executor = getattr(self, "_external_command_executor", None)
        if external_executor is not None:
            # Agent 2 exclusively owns collision detection in parallel landing.
            # The fused Agent-1/Agent-2 command is watched while it executes, so
            # first contact cancels motion before bounce/slide can corrupt XY.
            (
                external_info,
                collision_now,
                collision_object,
                collision_timestamp,
            ) = self._execute_with_agent2_collision_monitor(external_executor, vz)
            self._last_external_command_info = external_info
            vx = float(external_info.get("agent1_commanded_vx_mps", 0.0))
            vy = float(external_info.get("agent1_commanded_vy_mps", 0.0))
            vz = float(external_info.get("agent1_commanded_vz_mps", vz))
            yaw_rate = float(
                external_info.get("agent1_commanded_yaw_rate_dps", 0.0)
            )
        else:
            self.client.moveByVelocityBodyFrameAsync(
                vx=vx,
                vy=vy,
                vz=vz,
                duration=float(self.cfg.cmd_duration_s),
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
                vehicle_name=self.cfg.vehicle_name,
            ).join()
            self._last_external_command_info = {}
            collision_now, collision_object, collision_timestamp = self._new_collision()

        self._last_applied_vz_mps = float(vz)

        # Observation may occur after the contact, but touchdown classification
        # continues to use pre_info captured before the command that touched.
        obs, info = self._observe()
        info["range_sensor_final_active"] = bool(sensor_final_active)
        info["range_sensor_final_reason"] = str(sensor_final_reason)
        info["range_sensor_final_vz_mps"] = float(sensor_final_vz if sensor_final_active else 0.0)
        info["range_terminal_contact_armed"] = bool(getattr(self, "_range_terminal_contact_armed", False))
        info["range_terminal_handoff_active"] = bool(
            getattr(self, "_range_terminal_handoff_active", False)
        )
        terminal_started_step = int(
            getattr(self, "_range_terminal_contact_started_step", -1)
        )
        info["range_terminal_exploration_steps"] = int(
            max(0, int(self._step) - terminal_started_step)
            if terminal_started_step >= 0
            else 0
        )
        info["range_terminal_exploration_max_steps"] = int(
            self.cfg.range_terminal_contact_max_steps
        )
        info["terminal_policy_control_active"] = bool(
            getattr(self, "_range_terminal_handoff_active", False)
            and not getattr(self, "_range_terminal_handoff_timed_out", False)
        )
        info["terminal_range_governed_z"] = bool(
            terminal_handoff_active and not terminal_hard_timeout
        )
        info["terminal_conditional_climb_authority"] = bool(
            terminal_handoff_active
            and not terminal_hard_timeout
            and terminal_climb_allowed
        )
        info["terminal_climb_permission_reason"] = str(terminal_climb_reason)
        info["terminal_inf_descent_active"] = bool(terminal_inf_descent_active)
        info["policy_actions_suppressed_after_terminal_handoff"] = False
        info["range_terminal_contact_last_safe_height_m"] = float(
            getattr(self, "_range_terminal_contact_last_safe_height_m", float("inf"))
        )
        terminal_snapshot = getattr(self, "_range_terminal_contact_snapshot", None)
        info["range_terminal_handoff_timed_out"] = bool(
            getattr(self, "_range_terminal_handoff_timed_out", False)
        )
        info["range_terminal_snapshot_valid"] = bool(
            isinstance(terminal_snapshot, dict)
            and terminal_snapshot.get("valid", False)
        )
        info["range_terminal_snapshot_center_error_m"] = float(
            terminal_snapshot.get("center_error_m", float("inf"))
            if isinstance(terminal_snapshot, dict)
            else float("inf")
        )
        info["range_terminal_snapshot_similarity"] = float(
            terminal_snapshot.get("similarity", 0.0)
            if isinstance(terminal_snapshot, dict)
            else 0.0
        )
        info["range_terminal_snapshot_bbox_rel_error"] = float(
            terminal_snapshot.get("bbox_rel_error", float("inf"))
            if isinstance(terminal_snapshot, dict)
            else float("inf")
        )
        info["range_terminal_snapshot_target_id"] = str(
            terminal_snapshot.get("target_id", "")
            if isinstance(terminal_snapshot, dict)
            else ""
        )
        info["range_reliable_floor_m"] = float(self.cfg.range_sensor_final_stop_m)

        center_error = float(info["bottom_center_error"])
        bbox_rel = float(info["bottom_bbox_rel_err"])
        relative_height = float(info["relative_height_to_target_m"])
        visible = bool(info["bottom_match_live"])

        center_score = 0.0 if not np.isfinite(center_error) or center_error > 2.0 else math.exp(-4.0 * center_error)
        center_bank = 8.0 * center_score if visible else 0.0
        progress_bank = 0.0
        if self._prev_relative_height_m is not None and visible:
            descent_progress = float(self._prev_relative_height_m - relative_height)
            if descent_progress > 0.0 and center_error <= 0.45:
                progress_bank = min(12.0, 80.0 * descent_progress)
        if bool(self.cfg.parallel_dual_agent_mode):
            # XY/Yaw actions remain in the 4-D space only for checkpoint
            # compatibility, but they are masked physically. Do not shape the
            # landing policy with action dimensions it does not own.
            smooth_bank = -0.15 * abs(float(raw[2] - self._prev_action[2]))
        else:
            smooth_bank = -0.15 * float(np.linalg.norm(raw - self._prev_action))
        shaping = float(center_bank + progress_bank + smooth_bank)
        self._reward_bank += max(0.0, shaping)

        # The previous collision path classified contact from ``pre_info`` captured
        # before the 250ms command. That value can be visibly stale at contact.
        # When the identity-anchored terminal point survived into the first frame
        # captured after motion cancellation, use that stopped-contact frame. A
        # detector-only post-contact BBox is never allowed to replace the older
        # BEST sample, which still prevents a seat/logo close-up from approving
        # a bad touchdown.
        collision_measurement_info = pre_info
        if collision_now and bool(info.get("bottom_terminal_anchor_valid", False)):
            collision_measurement_info = dict(info)
            collision_measurement_info["bottom_touchdown_decision_context"] = (
                "FIRST_CONTACT_STOPPED_FRAME"
            )
        collision_decision = self._collision_reward_decision(
            info=info,
            pre_contact_info=collision_measurement_info,
            collision_object=collision_object,
            collision_now=collision_now,
        )

        # Update historical diagnostics only after the touchdown decision, so
        # the post-contact frame can never approve the collision retroactively.
        self._update_verified_alignment_latch(info)
        self._update_authorized_descent_latch_after_observation(info)
        if bool(self.cfg.parallel_dual_agent_mode):
            recovery_decision = {
                "requested": False,
                "reason": "parallel_agent1_continuous",
                "contact_guard_active": False,
                "contact_guard_similarity": 0.0,
                "contact_guard_margin": 0.0,
            }
        else:
            recovery_decision = self._recovery_request_decision(
                info, collision_now
            )
        if bool(recovery_decision["requested"]):
            print(
                "[A2 RECOVERY REQUEST] "
                f"reason={recovery_decision['reason']} "
                f"height={relative_height:.2f}m "
                f"noLive={self._non_live_duration_s:.2f}s "
                f"targetSpeed={self._target_velocity_speed_mps:.2f}m/s"
            )

        dense_reward, dense_reward_parts = self._dense_landing_reward(
            info,
            raw_vz_action=raw_vz_action,
            descent_allowed=bool(descent_allowed),
            applied_vz_mps=float(vz),
        )
        self._last_dense_reward = float(dense_reward)
        self._last_dense_reward_parts = dict(dense_reward_parts)

        done = False
        reason = ""
        reward = float(dense_reward)
        good_xy = bool(collision_decision["success"])
        latch_attempted = False
        latch_succeeded = False
        latch_error = ""
        touchdown_accepted_monotonic: float | None = None

        if collision_now:
            done = True
            if good_xy:
                reason = "landing_collision_success"
                print(
                    "[A2 SUCCESS GATE] accepted | "
                    f"path={collision_decision['success_path']} "
                    f"center={collision_decision['collision_recent_center_error']:.3f} "
                    f"bboxRelQuality={collision_decision['collision_recent_bbox_rel_error']:.3f} "
                    f"simQuality={collision_decision['collision_recent_similarity']:.3f} "
                    f"age={collision_decision['collision_recent_legacy_age']}"
                )
                terminal_quality_reward, terminal_quality_parts = (
                    self._terminal_landing_quality_reward(collision_decision)
                )
                terminal_success_reward = float(np.clip(
                    self.cfg.success_base_reward
                    + self._reward_bank
                    + terminal_quality_reward,
                    self.cfg.success_min_reward,
                    self.cfg.success_max_reward,
                ))
                reward += terminal_success_reward
                print(
                    "[A2 LANDING QUALITY] "
                    f"center={terminal_quality_parts['center']:+.1f} "
                    f"bboxRel={terminal_quality_parts['bbox_rel']:+.1f} "
                    f"similarity={terminal_quality_parts['similarity']:+.1f} "
                    f"bank={self._reward_bank:+.1f} "
                    f"terminal={terminal_success_reward:+.1f}"
                )
                # Calibrate in raw AirSim NED-Z, the same coordinate used by
                # the drone state. No abs()/sign conversion is involved.
                self._target_surface_z_ned = float(info["drone_z_ned"])
                self._target_surface_altitude_m = max(0.0, -self._target_surface_z_ned)
                self._target_surface_source = "verified_collision_api_z_ned"

                # Capture time-to-touchdown before the blocking latch RPC and
                # success hold. The timing curriculum measures flight only.
                touchdown_accepted_monotonic = float(time.monotonic())

                # The latch is deliberately downstream of the verified
                # collision + XY gate. It can never turn a failed contact into
                # a success and is never called for timeout, ground contact or
                # bad alignment. The blocking hold happens only after the
                # terminal reward has been computed for this transition.
                latch_attempted = bool(self.cfg.latch_on_success)
                if latch_attempted:
                    latch_succeeded, latch_error = (
                        self._latch_vehicle_after_success()
                    )
            elif collision_decision.get(
                "collision_known_wrong_object_contact", False
            ):
                reason = "landing_collision_wrong_object"
                reward -= float(self.cfg.wrong_collision_penalty)
            else:
                reason = "landing_collision_bad_xy"
                reward -= float(self.cfg.wrong_collision_penalty)
        elif terminal_hard_timeout:
            done = True
            reason = "landing_terminal_contact_timeout"
            reward -= float(self.cfg.timeout_penalty)
        elif self._step >= int(self.cfg.max_episode_steps):
            done = True
            reason = "landing_timeout_no_collision"
            reward -= float(self.cfg.timeout_penalty)
        elif (
            not bool(self.cfg.parallel_dual_agent_mode)
            and self._lost_steps >= int(self.cfg.target_lost_limit_steps)
        ):
            done = True
            reason = "landing_target_lost"
            reward -= float(self.cfg.target_lost_penalty)

        self._episode_return += float(reward)
        self._prev_action = raw.copy()
        self._prev_relative_height_m = relative_height
        info.update(
            {
                "agent": "AGENT_2",
                "raw_action": raw.copy(),
                "parallel_dual_agent": bool(self.cfg.parallel_dual_agent_mode),
                "xy_yaw_owner": "AGENT_1" if bool(self.cfg.parallel_dual_agent_mode) else "AGENT_2",
                "z_owner": "AGENT_2",
                "agent2_xy_yaw_masked": bool(self.cfg.parallel_dual_agent_mode),
                "parallel_agent1_tracking_mode": str(external_info.get("agent1_tracking_mode", "")),
                "parallel_agent1_active_camera": str(external_info.get("agent1_active_camera", "")),
                "parallel_agent1_camera_authority": str(external_info.get("agent1_camera_authority", "")),
                "parallel_agent1_fusion_has_target": bool(external_info.get("agent1_fusion_has_target", False)),
                "parallel_agent1_bottom_match": bool(external_info.get("agent1_bottom_match", False)),
                "parallel_target_velocity_ff_valid": bool(external_info.get("target_velocity_ff_valid", False)),
                "parallel_target_velocity_ff_vx_mps": float(external_info.get("target_velocity_ff_vx_mps", 0.0)),
                "parallel_target_velocity_ff_vy_mps": float(external_info.get("target_velocity_ff_vy_mps", 0.0)),
                "parallel_target_velocity_ff_blocked_by_safety": bool(external_info.get("target_velocity_ff_blocked_by_safety", False)),
                "parallel_bottom_guidance_live": bool(external_info.get("bottom_guidance_live", False)),
                "parallel_bottom_guidance_active": bool(external_info.get("bottom_guidance_active", False)),
                "parallel_bottom_guidance_prediction_only": bool(external_info.get("bottom_guidance_prediction_only", False)),
                "parallel_bottom_guidance_landing_lock": bool(external_info.get("bottom_guidance_landing_lock", False)),
                "parallel_bottom_guidance_blend_strength": float(external_info.get("bottom_guidance_blend_strength", 0.0)),
                "parallel_bottom_guidance_vx_mps": float(external_info.get("bottom_guidance_vx_mps", 0.0)),
                "parallel_bottom_guidance_vy_mps": float(external_info.get("bottom_guidance_vy_mps", 0.0)),
                "parallel_agent1_xy_weight": float(external_info.get("agent1_xy_weight", 1.0)),
                "parallel_bottom_relative_position_x_m": float(external_info.get("bottom_relative_position_x_m", 0.0)),
                "parallel_bottom_relative_position_y_m": float(external_info.get("bottom_relative_position_y_m", 0.0)),
                "parallel_bottom_relative_velocity_x_mps": float(external_info.get("bottom_relative_velocity_x_mps", 0.0)),
                "parallel_bottom_relative_velocity_y_mps": float(external_info.get("bottom_relative_velocity_y_mps", 0.0)),
                "parallel_bottom_predicted_relative_x_m": float(external_info.get("bottom_predicted_relative_x_m", 0.0)),
                "parallel_bottom_predicted_relative_y_m": float(external_info.get("bottom_predicted_relative_y_m", 0.0)),
                "parallel_bottom_controller_source": str(external_info.get("bottom_controller_source", "NONE")),
                "parallel_physical_steps": int(external_info.get("parallel_physical_steps", 0)),
                "commanded_vx_mps": vx,
                "commanded_vy_mps": vy,
                "commanded_vz_mps": vz,
                "raw_vz_action": raw_vz_action,
                "requested_vz_mps": requested_vz,
                "applied_vz_mps": vz,
                "vertical_speed_limit_mps": float(self._last_vertical_speed_limit_mps),
                "soft_catchup_descent_active": bool(self._last_soft_catchup_descent_active),
                "climb_command_blocked": bool(climb_command_blocked),
                "reacquire_climb_active": bool(reacquire_climb_active),
                "reacquire_climb_reason": str(reacquire_climb_reason),
                "vertical_control_state": vertical_state,
                "descent_allowed": bool(descent_allowed),
                "descent_requested": bool(descent_requested),
                "descent_blocked": bool(descent_blocked),
                "descent_block_reason": descent_block_reason,
                "commanded_yaw_rate_dps": yaw_rate,
                "horizontal_guard_reasons": guard_reasons,
                "horizontal_control_state": self._last_horizontal_control_state,
                "horizontal_pd_action_vx": float(self._last_pd_action_vx),
                "horizontal_pd_action_vy": float(self._last_pd_action_vy),
                "horizontal_residual_action_vx": float(self._last_residual_action_vx),
                "horizontal_residual_action_vy": float(self._last_residual_action_vy),
                "horizontal_final_action_vx": float(self._last_horizontal_action_vx),
                "horizontal_final_action_vy": float(self._last_horizontal_action_vy),
                "horizontal_speed_limit_mps": float(self._last_horizontal_speed_limit_mps),
                "target_velocity_valid": bool(getattr(self, "_target_velocity_valid", False)),
                "target_velocity_age_s": float(getattr(self, "_target_velocity_age_s", float("inf"))),
                "target_velocity_world_x_mps": float(getattr(self, "_target_velocity_world_x_mps", 0.0)),
                "target_velocity_world_y_mps": float(getattr(self, "_target_velocity_world_y_mps", 0.0)),
                "target_velocity_body_vx_mps": float(getattr(self, "_target_velocity_body_vx_mps", 0.0)),
                "target_velocity_body_vy_mps": float(getattr(self, "_target_velocity_body_vy_mps", 0.0)),
                "target_velocity_speed_mps": float(getattr(self, "_target_velocity_speed_mps", 0.0)),
                "horizontal_velocity_ff_vx_mps": float(self._last_velocity_ff_vx_mps),
                "horizontal_velocity_ff_vy_mps": float(self._last_velocity_ff_vy_mps),
                "horizontal_correction_vx_mps": float(self._last_horizontal_correction_vx_mps),
                "horizontal_correction_vy_mps": float(self._last_horizontal_correction_vy_mps),
                "horizontal_total_speed_limit_mps": float(self._last_horizontal_total_speed_limit_mps),
                "recovery_requested": bool(recovery_decision["requested"]),
                "recovery_request_reason": str(recovery_decision["reason"]),
                "recovery_contact_guard_active": bool(recovery_decision["contact_guard_active"]),
                "recovery_contact_guard_similarity": float(recovery_decision["contact_guard_similarity"]),
                "recovery_contact_guard_margin": float(recovery_decision["contact_guard_margin"]),
                "alignment_ready_streak": int(getattr(self, "_alignment_ready_streak", 0)),
                "descent_alignment_latched": bool(getattr(self, "_descent_alignment_latched", False)),
                "reward_bank": float(self._reward_bank),
                "landing_shaping_bank_delta": shaping,
                "dense_landing_reward": float(dense_reward),
                "dense_reward_aligned_descent_progress": float(
                    dense_reward_parts["aligned_descent_progress"]
                ),
                "dense_reward_landing_lock_time": float(
                    dense_reward_parts["landing_lock_time"]
                ),
                "dense_reward_hesitation": float(
                    dense_reward_parts["hesitation"]
                ),
                "dense_reward_unsafe_descent": float(
                    dense_reward_parts["unsafe_descent"]
                ),
                "collision_new": collision_now,
                "collision_object_name": collision_object,
                "collision_timestamp": collision_timestamp,
                "collision_good_xy": good_xy,
                "latch_on_success_enabled": bool(self.cfg.latch_on_success),
                "latch_attempted": bool(latch_attempted),
                "latch_succeeded": bool(latch_succeeded),
                "latch_error": str(latch_error),
                "latch_vehicle_name": str(self.cfg.latch_vehicle_name),
                "latch_target_actor_name": str(self.cfg.latch_target_actor_name),
                "latch_anchor_component_name": str(self.cfg.latch_anchor_component_name),
                "latch_success_hold_seconds": float(self.cfg.latch_success_hold_seconds),
                "touchdown_accepted_monotonic": touchdown_accepted_monotonic,
                "terminal_signed_z_authority": False,
                "terminal_range_governed_z": bool(
                    terminal_handoff_active and not terminal_hard_timeout
                ),
                "terminal_conditional_climb_authority": bool(
                    terminal_handoff_active
                    and not terminal_hard_timeout
                    and terminal_climb_allowed
                ),
                "terminal_climb_permission_reason": str(terminal_climb_reason),
                "terminal_inf_descent_active": bool(terminal_inf_descent_active),
                "collision_alignment_success_path": collision_decision["success_path"],
                "collision_live_match_at_contact": collision_decision["live_match_at_collision"],
                "collision_contact_appearance_valid": collision_decision["contact_appearance_valid"],
                "collision_contact_center_similarity": collision_decision["contact_center_similarity"],
                "collision_contact_corner_similarity": collision_decision["contact_corner_similarity"],
                "collision_contact_center_margin": collision_decision["contact_center_margin"],
                "collision_contact_best_center_scale": collision_decision["contact_best_center_scale"],
                "collision_contact_appearance_reason": collision_decision["contact_appearance_reason"],
                "collision_xy_threshold_m": collision_decision["collision_xy_threshold_m"],
                "collision_xy_legacy_error_m": collision_decision["collision_xy_legacy_error_m"],
                "collision_xy_legacy_valid": collision_decision["collision_xy_legacy_valid"],
                "collision_xy_geometric_error_m": collision_decision["collision_xy_geometric_error_m"],
                "collision_xy_geometric_valid": collision_decision["collision_xy_geometric_valid"],
                "collision_xy_geometric_source": collision_decision["collision_xy_geometric_source"],
                "collision_xy_combined_error_m": collision_decision["collision_xy_combined_error_m"],
                "collision_xy_combined_valid": collision_decision["collision_xy_combined_valid"],
                "collision_xy_selected_source": collision_decision["collision_xy_selected_source"],
                "collision_terminal_snapshot_valid": collision_decision["collision_terminal_snapshot_valid"],
                "collision_terminal_snapshot_identity_ok": collision_decision["collision_terminal_snapshot_identity_ok"],
                "collision_terminal_snapshot_center_pass": collision_decision["collision_terminal_snapshot_center_pass"],
                "collision_terminal_snapshot_center_error_m": collision_decision["collision_terminal_snapshot_center_error_m"],
                "collision_terminal_snapshot_similarity": collision_decision["collision_terminal_snapshot_similarity"],
                "collision_terminal_snapshot_bbox_rel_error": collision_decision["collision_terminal_snapshot_bbox_rel_error"],
                "collision_terminal_snapshot_age_steps": collision_decision["collision_terminal_snapshot_age_steps"],
                "collision_terminal_snapshot_age_s": collision_decision["collision_terminal_snapshot_age_s"],
                "collision_alignment_latch_age": collision_decision["alignment_latch_age"],
                "collision_last_verified_center_error": collision_decision["last_verified_center_error"],
                "collision_last_verified_bbox_rel_error": collision_decision["last_verified_bbox_rel_error"],
                "collision_last_verified_similarity": collision_decision["last_verified_similarity"],
                "collision_authorized_descent_latched": collision_decision["authorized_descent_latched"],
                "collision_authorized_descent_latch_age": collision_decision["authorized_descent_latch_age"],
                "collision_last_authorized_descent_center_error": collision_decision["last_authorized_descent_center_error"],
                "collision_last_authorized_descent_bbox_rel_error": collision_decision["last_authorized_descent_bbox_rel_error"],
                "collision_last_authorized_descent_similarity": collision_decision["last_authorized_descent_similarity"],
                "collision_last_authorized_descent_vz_mps": collision_decision["last_authorized_descent_vz_mps"],
                "collision_authorized_descent_invalidated_reason": collision_decision["authorized_descent_invalidated_reason"],
                "collision_object_matches_target": collision_decision["collision_object_matches_target"],
                "collision_object_name_available": collision_decision["collision_object_name_available"],
                "collision_known_wrong_object_contact": collision_decision["collision_known_wrong_object_contact"],
                "collision_object_lock_created": collision_decision["collision_object_lock_created"],
                "expected_collision_object_name": collision_decision["expected_collision_object_name"],
                "expected_collision_object_source": collision_decision["expected_collision_object_source"],
                "collision_reject_reason": collision_decision["reject_reason"],
                "termination_reason": reason,
                "episode_return": float(self._episode_return),
            }
        )

        if done:
            total_perception = max(
                1,
                self._episode_live_match_steps
                + self._episode_predicted_steps
                + self._episode_no_target_steps,
            )
            live_pct = 100.0 * self._episode_live_match_steps / total_perception
            pred_pct = 100.0 * self._episode_predicted_steps / total_perception
            lost_pct = 100.0 * self._episode_no_target_steps / total_perception
            best_center = (
                self._episode_best_center_error
                if np.isfinite(self._episode_best_center_error)
                else 999.0
            )
            print(
                f"[A2 EP] steps={self._step} result={reason or 'running'} "
                f"reward={reward:+.1f} dense={self._last_dense_reward:+.2f} "
                f"bank={self._reward_bank:+.1f} "
                f"live={live_pct:.1f}% pred={pred_pct:.1f}% lost={lost_pct:.1f}% "
                f"xy={self._last_horizontal_control_state} recenter={int(getattr(self, '_episode_recenter_steps', 0))} "
                f"xyHold={int(getattr(self, '_episode_xy_hold_steps', 0))} "
                f"descent={self._episode_descent_allowed_steps}/"
                f"{self._episode_descent_requested_steps} "
                f"blocked={self._episode_descent_blocked_steps} "
                f"climbBlocked={self._episode_climb_command_blocked_steps} "
                f"centerFinal={center_error:.3f} centerBest={best_center:.3f} "
                f"simBest={self._episode_best_similarity:.3f} "
                f"A1cam={external_info.get('agent1_active_camera', 'n/a')} "
                f"A1mode={external_info.get('agent1_tracking_mode', 'n/a')} "
                f"A1fusion={int(bool(external_info.get('agent1_fusion_has_target', False)))} "
                f"FF=({float(external_info.get('target_velocity_ff_vx_mps', 0.0)):+.2f},"
                f"{float(external_info.get('target_velocity_ff_vy_mps', 0.0)):+.2f}) "
                f"Vsrc={self._visual_motion_source} flowN={self._visual_flow_point_count} "
                f"A1W={float(external_info.get('agent1_xy_weight', 1.0)):.2f} "
                f"BPD=({float(external_info.get('bottom_guidance_vx_mps', 0.0)):+.2f},"
                f"{float(external_info.get('bottom_guidance_vy_mps', 0.0)):+.2f}) "
                f"pred=({float(external_info.get('bottom_predicted_err_x', 0.0)):+.2f},"
                f"{float(external_info.get('bottom_predicted_err_y', 0.0)):+.2f}) "
                f"ivel=({float(external_info.get('bottom_image_velocity_x_per_s', 0.0)):+.2f},"
                f"{float(external_info.get('bottom_image_velocity_y_per_s', 0.0)):+.2f}) "
                f"rel=({float(external_info.get('bottom_relative_position_x_m', 0.0)):+.2f},"
                f"{float(external_info.get('bottom_relative_position_y_m', 0.0)):+.2f})m "
                f"relV=({float(external_info.get('bottom_relative_velocity_x_mps', 0.0)):+.2f},"
                f"{float(external_info.get('bottom_relative_velocity_y_mps', 0.0)):+.2f})m/s "
                f"catch={int(bool(external_info.get('bottom_predictive_catchup', False)))} "
                f"climb={int(bool(reacquire_climb_active))} "
                f"lock={int(bool(getattr(self, '_descent_alignment_latched', False)))} "
                f"bankN={len(self._adaptive_embeddings)} bankUpd={self._adaptive_embedding_updates} "
                f"collision={collision_object or 'none'} "
                f"liveAtTouch={int(bool(collision_decision['live_match_at_collision']))} "
                f"contact={int(bool(collision_decision['contact_appearance_valid']))} "
                f"contactSim={float(collision_decision['contact_center_similarity']):.3f} "
                f"cornerSim={float(collision_decision['contact_corner_similarity']):.3f} "
                f"contactMargin={float(collision_decision['contact_center_margin']):+.3f} "
                f"contactScale={float(collision_decision['contact_best_center_scale']):.2f} "
                f"xyRecent={float(collision_decision['collision_xy_legacy_error_m']):.3f} "
                f"xyAge={int(collision_decision.get('collision_recent_legacy_age', 999999))} "
                f"xySrc={collision_decision['collision_xy_selected_source']} "
                f"latchAge={int(collision_decision['alignment_latch_age'])} "
                f"lastCenter={float(collision_decision['last_verified_center_error']):.3f} "
                f"lastBBoxRel={float(collision_decision['last_verified_bbox_rel_error']):.3f} "
                f"lastSim={float(collision_decision['last_verified_similarity']):.3f} "
                f"authAge={int(collision_decision['authorized_descent_latch_age'])} "
                f"authCenter={float(collision_decision['last_authorized_descent_center_error']):.3f} "
                f"authBBoxRel={float(collision_decision['last_authorized_descent_bbox_rel_error']):.3f} "
                f"authSim={float(collision_decision['last_authorized_descent_similarity']):.3f} "
                f"authVz={float(collision_decision['last_authorized_descent_vz_mps']):.3f} "
                f"authActive={int(bool(collision_decision['authorized_descent_latched']))} "
                f"successPath={collision_decision['success_path']} "
                f"targetCollision={int(bool(collision_decision['collision_object_matches_target']))} "
                f"reject={collision_decision['reject_reason'] or 'none'} "
                f"authReject={collision_decision['authorized_descent_invalidated_reason'] or 'none'}"
            )

        return obs, float(reward), bool(done), False, info

    def register_recovery_penalty(self, penalty: float) -> None:
        """Keep Agent-2 episode diagnostics consistent with wrapper rewards."""
        value = max(0.0, float(penalty))
        self._episode_return -= value

    def close(self):
        try:
            cv2.destroyWindow("Agent 2 - Bottom Landing")
        except Exception:
            pass
