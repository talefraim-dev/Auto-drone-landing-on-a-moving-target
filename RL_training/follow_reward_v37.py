"""
follow_reward_v37.py

Fine-tuning reward for the Chase & Center tracking policy.

Goal of this version:
    The tracker is assumed to be stable. This reward trains the PPO policy to
    actively chase the selected target, keep it strongly centered in the image,
    and avoid the bad habit of simply keeping the target somewhere in frame.

Important behavior:
    1. MATCH/visibility alone is not enough for a high reward.
    2. Large center error receives a strong nonlinear penalty.
    3. Reducing center error from step to step receives progress reward.
    4. Standing still while the target is off-center is penalized.
    5. Forward chase is rewarded mainly after the target is reasonably centered.
    6. Useless yaw/action without center improvement is penalized.

Expected call signature remains unchanged:
    reward, reward_parts = compute_follow_reward(
        obs=obs_dict,
        action=safe_action,
        prev_action=self._prev_action,
        env_info=env_reward_info,
        config=self.reward_config,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass
class FollowRewardConfig:
    # Existing fields used by DroneEnv.
    w_center: float = 28.0
    center_reward_alpha: float = 2.0

    w_distance: float = 6.0
    desired_distance_proxy: float = 0.86
    distance_tolerance: float = 0.05

    w_visibility: float = 2.0
    w_lost_target: float = 16.0

    w_altitude_safe: float = 0.3
    w_altitude_low_penalty: float = 2.0
    w_altitude_high_penalty: float = 2.0
    min_safe_altitude_m: float = 2.0
    max_safe_altitude_m: float = 30.0

    w_smooth_follow: float = 1.0
    max_img_motion: float = 0.35

    w_control: float = 0.04
    w_action_delta: float = 0.20
    w_slow: float = 0.02
    w_time: float = 0.03

    w_obstacle: float = 4.0
    w_safety_intervention: float = 2.0

    collision_penalty: float = 80.0
    timeout_penalty: float = 8.0
    altitude_termination_penalty: float = 12.0
    distance_damage_terminal_penalty: float = 120.0
    handoff_success_bonus: float = 260.0

    # ------------------------------------------------------------------
    # Chase & center shaping.
    # ------------------------------------------------------------------
    # Nonlinear penalty for letting the target stay far from the center.
    w_center_error_penalty: float = 35.0
    center_error_penalty_power: float = 2.4

    # Extra penalty when target approaches frame borders.
    w_frame_edge_penalty: float = 35.0
    edge_soft_threshold: float = 0.50

    # Reward/penalty for center-error progress.
    w_center_error_exp_progress: float = 14.0
    w_center_error_exp_regress: float = 18.0
    center_error_exp_gain: float = 10.0
    center_error_exp_clip: float = 0.18
    center_progress_deadband: float = 0.002

    # Direct action alignment for centering.
    # Empirical axis diagnostic showed that when the target is low in the image
    # (positive err_y), negative vx tends to improve center error. For err_x,
    # lateral vy is used as an auxiliary correction signal.
    w_center_action_alignment: float = 6.0
    center_action_deadband: float = 0.05

    # Penalize doing almost nothing while the target is visibly off-center.
    w_offcenter_idle_penalty: float = 12.0
    offcenter_idle_threshold: float = 0.35
    idle_action_norm_threshold: float = 0.12

    # Chase/approach shaping. distance_proxy_norm is lower when the target is
    # closer/larger. If it is larger than desired, the target is too far.
    w_chase_forward_alignment: float = 4.0
    chase_center_gate: float = 0.35
    # Real physical chase shaping from DroneEnv.
    # These rewards use the actual horizontal drone-to-car distance in meters,
    # not only the bbox-size proxy. This gives PPO a positive reason to chase,
    # while hard termination remains only the failure boundary.
    # Exponential percentage-based progress reward.
    # Progress is normalized by the initial episode distance so the reward scale
    # is meaningful across different start distances.
    w_real_distance_step_progress: float = 450.0
    w_real_distance_step_regress: float = 120.0
    real_distance_step_clip_m: float = 0.45
    real_distance_progress_deadband_m: float = 0.015
    real_distance_progress_exp_gain: float = 10.0
    real_distance_progress_ratio_clip: float = 0.035

    # Bonus for beating the best physical distance achieved in the episode.
    # This is also percentage-based and exponential.
    w_real_best_distance_improvement: float = 650.0
    real_best_improvement_clip_m: float = 0.60
    real_best_distance_exp_gain: float = 8.0
    real_best_improvement_ratio_clip: float = 0.12

    # Continuous pressure while still far from the best/goal distance.
    w_real_distance_gap_penalty: float = 4.0
    real_distance_gap_clip_m: float = 8.0

    # Punish wasting the no-improvement window before hard termination fires.
    w_no_real_approach_time_penalty: float = 0.35

    # Reward forward command when it produces real distance improvement.
    w_real_chase_action_bonus: float = 6.0

    # Chase-pressure anti-camping reward:
    # The drone must not park in place while the target is visible and far.
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

    # Full-throttle chase shaping before bottom velocity matching.
    w_fast_chase_throttle_bonus: float = 55.0
    w_fast_chase_progress_bonus: float = 95.0
    w_fast_chase_slow_penalty: float = 55.0
    fast_chase_far_distance_m: float = 2.50
    fast_chase_min_vx_action: float = 0.70

    # Bottom-camera relative velocity matching.
    w_bottom_velocity_error_penalty: float = 18.0
    w_bottom_velocity_progress_bonus: float = 16.0
    w_bottom_velocity_ready_bonus: float = 28.0

    # Fine position accuracy shaping.
    fine_position_reward_enabled: bool = True
    fine_position_error_ready: float = 0.08
    fine_position_error_good: float = 0.14
    fine_position_bonus_weight: float = 95.0
    fine_position_bonus_gain: float = 42.0
    fine_position_penalty_weight: float = 65.0
    fine_position_penalty_power: float = 1.75
    fine_position_ready_bonus: float = 85.0
    fine_position_landing_ready_bonus: float = 140.0

    # BBox-relative fine position shaping.
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

    # Bottom-primary alignment shaping.
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

    # Distance keeping and convergence.
    w_distance_linear_deviation: float = 3.0
    w_distance_error_exp_progress: float = 5.0
    w_distance_error_exp_regress: float = 7.0
    distance_error_exp_gain: float = 14.0
    distance_error_exp_clip: float = 0.12
    distance_progress_deadband: float = 0.0015

    # Too close protection. For chase fine-tuning this is intentionally milder
    # than before, because the goal is to pursue until a near pre-landing range.
    w_too_close_exp_penalty: float = 4.0
    too_close_margin: float = 0.04
    too_close_exp_gain: float = 12.0

    # Yaw stability and anti-jitter.
    w_yaw_abs: float = 0.80
    w_yaw_delta: float = 1.20
    w_yaw_when_centered: float = 6.0
    center_calm_threshold: float = 0.20

    # Very strong anti-yaw rule:
    # When the tracker is in MATCH, the target is focused, and the target is
    # already centered, yaw is not useful. The correct behavior is to keep yaw
    # calm and use forward/side motion for chase and handoff geometry.
    w_yaw_when_focused_centered: float = 150.0
    focused_centered_yaw_threshold: float = 0.035
    focused_center_threshold: float = 0.16
    focused_center_x_threshold: float = 0.11
    focused_center_y_threshold: float = 0.16
    focused_bbox_conf_threshold: float = 0.55
    focused_centered_yaw_power: float = 1.25

    # Exponential yaw/de-centering punishments.
    # These make unnecessary yaw increasingly expensive as the action becomes
    # stronger or as the target drifts away from a previously centered state.
    w_exp_focused_centered_yaw: float = 260.0
    exp_focused_yaw_gain: float = 5.0
    exp_focused_yaw_clip: float = 0.55
    w_yaw_decenter_exp_penalty: float = 340.0
    yaw_decenter_exp_gain: float = 9.0
    yaw_decenter_error_clip: float = 0.18
    yaw_decenter_prev_center_threshold: float = 0.18
    yaw_decenter_current_max_threshold: float = 0.45

    # Soft command-yaw budget. This is intentionally not terminal.
    w_yaw_command_budget_penalty: float = 0.025
    yaw_command_budget_free_deg: float = 25.0
    yaw_command_budget_clip_deg: float = 120.0

    # Penalize yaw/action that does not improve centering.
    w_useless_yaw_penalty: float = 10.0
    useless_yaw_threshold: float = 0.08

    # Extra yaw discipline in MATCH.
    # If the target is already horizontally centered, yaw usually does not help.
    # In the current geometry the remaining error is mostly vertical, so yaw
    # should stay calm while vx/vy handle chase/positioning.
    w_yaw_when_x_centered: float = 8.0
    x_center_yaw_threshold: float = 0.12
    w_yaw_when_vertical_only_error: float = 5.0
    vertical_only_yaw_x_threshold: float = 0.12
    vertical_only_yaw_y_threshold: float = 0.45

    # Strong stable-chase bonus when all main goals are satisfied together.
    w_stable_follow_bonus: float = 18.0
    stable_center_threshold: float = 0.12
    stable_yaw_threshold: float = 0.08

    # Recovery shaping after the target leaves the image / tracker loses MATCH.
    w_recovery_yaw: float = 3.0
    w_wrong_recovery_yaw: float = 4.0
    w_fast_reacquire: float = 20.0
    w_overshoot_penalty: float = 14.0
    w_lost_time_accel_penalty: float = 6.0
    recovery_yaw_deadband: float = 0.08

    # Reward clipping for training stability.
    clip_total_reward: bool = True
    min_total_reward: float = -50.0
    max_total_reward: float = 50.0


def _f(x: Any, default: float = 0.0) -> float:
    """Safe float conversion."""
    try:
        v = float(x)
        if not math.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def _exp_shaping(value: float, gain: float, clip_abs: float) -> float:
    """Exponential shaping with clipping to avoid exploding rewards."""
    v = _clip(float(value), -float(clip_abs), float(clip_abs))
    return math.exp(float(gain) * abs(v)) - 1.0


def _distance_error(obs: dict[str, Any], cfg: FollowRewardConfig) -> tuple[float, float, float]:
    distance_proxy = _f(obs.get("distance_proxy_norm", 1.0), 1.0)
    distance_delta = _f(obs.get("distance_proxy_delta", 0.0), 0.0)

    desired = float(cfg.desired_distance_proxy)
    current_error = abs(distance_proxy - desired)

    prev_distance_proxy = distance_proxy - distance_delta
    prev_error = abs(prev_distance_proxy - desired)

    return distance_proxy, current_error, prev_error


def _center_error(obs: dict[str, Any]) -> tuple[float, float, float, float]:
    """
    Return:
        abs_err_x, abs_err_y, center_error, prev_center_error_estimate

    This version uses the full image-space center error, not mostly-horizontal
    weighting. For landing preparation, vertical image error matters a lot.
    """
    raw_err_x = _f(obs.get("err_x", 1.0), 1.0)
    raw_err_y = _f(obs.get("err_y", 1.0), 1.0)
    err_x = abs(raw_err_x)
    err_y = abs(raw_err_y)

    center_error = _clip(math.sqrt(err_x * err_x + err_y * err_y), 0.0, 1.5)

    img_vx = _f(obs.get("img_vx", 0.0), 0.0)
    img_vy = _f(obs.get("img_vy", 0.0), 0.0)

    prev_raw_err_x = raw_err_x - img_vx
    prev_raw_err_y = raw_err_y - img_vy
    prev_center_error = _clip(
        math.sqrt(prev_raw_err_x * prev_raw_err_x + prev_raw_err_y * prev_raw_err_y),
        0.0,
        1.5,
    )

    return err_x, err_y, center_error, prev_center_error


def compute_follow_reward(
    obs: dict[str, Any],
    action: np.ndarray,
    prev_action: np.ndarray,
    env_info: dict[str, Any] | None,
    config: FollowRewardConfig,
) -> tuple[float, dict[str, float]]:
    """Compute reward for chase-and-center target following."""
    env_info = env_info or {}

    action = np.asarray(action, dtype=np.float32).reshape(-1)
    prev_action = np.asarray(prev_action, dtype=np.float32).reshape(-1)

    if action.size < 4:
        action = np.pad(action, (0, 4 - action.size), mode="constant")
    if prev_action.size < 4:
        prev_action = np.pad(prev_action, (0, 4 - prev_action.size), mode="constant")

    has_target = _f(obs.get("has_target", 0.0), 0.0) > 0.5
    bbox_conf = _clip(_f(obs.get("bbox_conf", 0.0), 0.0), 0.0, 1.0)
    visible = 1.0 if has_target else 0.0
    lost_target_time_norm = _clip(_f(obs.get("lost_target_time_norm", 0.0), 0.0), 0.0, 1.5)

    tracking_mode = str(env_info.get("tracking_mode", "") or "").upper()
    is_match_mode = bool(has_target and tracking_mode == "MATCH")

    signed_err_x = _clip(_f(obs.get("err_x", 0.0), 0.0), -1.5, 1.5)
    signed_err_y = _clip(_f(obs.get("err_y", 0.0), 0.0), -1.5, 1.5)

    vx = float(action[0])
    vy = float(action[1])
    yaw = float(action[3])

    yaw_action = abs(yaw)
    yaw_delta = abs(float(action[3] - prev_action[3]))
    action_delta_norm = float(np.linalg.norm(action - prev_action))
    action_norm = float(np.linalg.norm(action))
    planar_action_norm = float(np.linalg.norm(action[[0, 1, 3]]))

    # ------------------------------------------------------------------
    # Center alignment.
    # ------------------------------------------------------------------
    abs_err_x, abs_err_y, center_error, prev_center_error = _center_error(obs)
    centered_score = _clip(_f(obs.get("centered_score", 1.0 - center_error), 1.0 - center_error), 0.0, 1.0)

    # Base center reward exists, but large off-center errors are punished hard.
    center_reward = float(config.w_center) * (centered_score ** float(config.center_reward_alpha)) * visible
    center_error_penalty = (
        -float(config.w_center_error_penalty)
        * (center_error ** float(config.center_error_penalty_power))
        * visible
    )

    # Frame-edge penalty: makes it uncomfortable to leave the target low/high/sideways.
    edge_x = max(0.0, abs_err_x - float(config.edge_soft_threshold))
    edge_y = max(0.0, abs_err_y - float(config.edge_soft_threshold))
    frame_edge_penalty = -float(config.w_frame_edge_penalty) * (edge_x * edge_x + edge_y * edge_y) * visible

    center_improvement = prev_center_error - center_error
    if abs(center_improvement) < float(config.center_progress_deadband):
        center_progress_reward = 0.0
        center_regress_penalty = 0.0
    elif center_improvement > 0.0:
        center_progress_reward = (
            float(config.w_center_error_exp_progress)
            * _exp_shaping(center_improvement, config.center_error_exp_gain, config.center_error_exp_clip)
            * visible
        )
        center_regress_penalty = 0.0
    else:
        center_progress_reward = 0.0
        center_regress_penalty = (
            -float(config.w_center_error_exp_regress)
            * _exp_shaping(-center_improvement, config.center_error_exp_gain, config.center_error_exp_clip)
            * visible
        )

    # Direct action alignment for centering.
    # Positive err_y means target is low; empirical test showed negative vx often helps.
    # For err_x, vy is used as lateral centering support.
    vertical_alignment = 0.0
    if abs(signed_err_y) > float(config.center_action_deadband):
        vertical_alignment = _clip((-signed_err_y) * vx, -1.0, 1.0)

    lateral_alignment = 0.0
    if abs(signed_err_x) > float(config.center_action_deadband):
        lateral_alignment = _clip(signed_err_x * vy, -1.0, 1.0)

    center_action_alignment_reward = (
        float(config.w_center_action_alignment)
        * (vertical_alignment + 0.5 * lateral_alignment)
        * visible
    )

    if visible and center_error > float(config.offcenter_idle_threshold) and planar_action_norm < float(config.idle_action_norm_threshold):
        offcenter_idle_penalty = -float(config.w_offcenter_idle_penalty) * (center_error - float(config.offcenter_idle_threshold) + 0.25)
    else:
        offcenter_idle_penalty = 0.0

    # ------------------------------------------------------------------
    # Distance keeping and chase behavior.
    # ------------------------------------------------------------------
    distance_proxy, distance_error, prev_distance_error = _distance_error(obs, config)

    tol = max(1e-6, float(config.distance_tolerance))
    distance_score = math.exp(-((distance_error / tol) ** 2))
    distance_reward = float(config.w_distance) * distance_score * visible

    distance_linear_deviation_penalty = -float(config.w_distance_linear_deviation) * distance_error * visible

    distance_error_improvement = prev_distance_error - distance_error
    if abs(distance_error_improvement) < float(config.distance_progress_deadband):
        distance_exp_progress_reward = 0.0
        distance_exp_regress_penalty = 0.0
    elif distance_error_improvement > 0.0:
        distance_exp_progress_reward = (
            float(config.w_distance_error_exp_progress)
            * _exp_shaping(distance_error_improvement, config.distance_error_exp_gain, config.distance_error_exp_clip)
            * visible
        )
        distance_exp_regress_penalty = 0.0
    else:
        distance_exp_progress_reward = 0.0
        distance_exp_regress_penalty = (
            -float(config.w_distance_error_exp_regress)
            * _exp_shaping(-distance_error_improvement, config.distance_error_exp_gain, config.distance_error_exp_clip)
            * visible
        )

    # Reward forward chase only when the target is already reasonably centered.
    # If the target is off-center, centering should dominate instead of blind forward motion.
    too_far = max(0.0, distance_proxy - float(config.desired_distance_proxy))
    if visible and center_error < float(config.chase_center_gate) and too_far > 0.0:
        chase_forward_reward = float(config.w_chase_forward_alignment) * _clip(vx, -1.0, 1.0) * _clip(too_far / 0.15, 0.0, 1.0)
    else:
        chase_forward_reward = 0.0

    too_close_threshold = float(config.desired_distance_proxy) - float(config.too_close_margin)
    if distance_proxy < too_close_threshold and visible > 0.0:
        too_close_error = too_close_threshold - distance_proxy
        too_close_penalty = -float(config.w_too_close_exp_penalty) * _exp_shaping(
            too_close_error,
            config.too_close_exp_gain,
            config.distance_error_exp_clip,
        )
    else:
        too_close_penalty = 0.0

    # ------------------------------------------------------------------
    # Real physical distance chase reward.
    # ------------------------------------------------------------------
    current_real_distance_m = _f(env_info.get("current_chase_distance_m", float("inf")), float("inf"))
    previous_real_distance_m = _f(env_info.get("previous_chase_distance_m", float("inf")), float("inf"))
    initial_real_distance_m = _f(env_info.get("initial_chase_distance_m", float("inf")), float("inf"))
    best_real_distance_m = _f(env_info.get("best_chase_distance_m", float("inf")), float("inf"))
    step_real_improvement_m = _f(env_info.get("step_chase_distance_improvement_m", 0.0), 0.0)
    best_real_improvement_m = _f(env_info.get("last_approach_improvement_m", 0.0), 0.0)
    not_approaching_time_s = max(0.0, _f(env_info.get("not_approaching_time_s", 0.0), 0.0))
    distance_damage_ratio = max(0.0, _f(env_info.get("distance_damage_ratio", 0.0), 0.0))

    real_distance_scale_m = initial_real_distance_m
    if not math.isfinite(real_distance_scale_m) or real_distance_scale_m <= 1e-6:
        real_distance_scale_m = current_real_distance_m
    if not math.isfinite(real_distance_scale_m) or real_distance_scale_m <= 1e-6:
        real_distance_scale_m = 1.0

    real_distance_step_progress_reward = 0.0
    real_distance_step_regress_penalty = 0.0
    real_best_distance_improvement_reward = 0.0
    real_distance_gap_penalty = 0.0
    no_real_approach_time_penalty = 0.0
    real_chase_action_bonus = 0.0

    if has_target and math.isfinite(current_real_distance_m):
        # Step-to-step physical approach reward.
        # This is percentage-based and exponential: a small improvement matters,
        # but larger meaningful progress receives disproportionately higher reward.
        if step_real_improvement_m > float(config.real_distance_progress_deadband_m):
            imp_m = min(step_real_improvement_m, float(config.real_distance_step_clip_m))
            imp_ratio = imp_m / max(1e-6, float(real_distance_scale_m))
            real_distance_step_progress_reward = (
                float(config.w_real_distance_step_progress)
                * _exp_shaping(
                    imp_ratio,
                    config.real_distance_progress_exp_gain,
                    config.real_distance_progress_ratio_clip,
                )
            )
        elif step_real_improvement_m < -float(config.real_distance_progress_deadband_m):
            reg_m = min(-step_real_improvement_m, float(config.real_distance_step_clip_m))
            reg_ratio = reg_m / max(1e-6, float(real_distance_scale_m))
            real_distance_step_regress_penalty = -float(config.w_real_distance_step_regress) * reg_ratio

        # Strong pulse when the drone beats the best real distance seen in this
        # episode. Also percentage-based and exponential.
        if best_real_improvement_m > 0.0:
            best_imp_m = min(best_real_improvement_m, float(config.real_best_improvement_clip_m))
            best_imp_ratio = best_imp_m / max(1e-6, float(real_distance_scale_m))
            real_best_distance_improvement_reward = (
                float(config.w_real_best_distance_improvement)
                * _exp_shaping(
                    best_imp_ratio,
                    config.real_best_distance_exp_gain,
                    config.real_best_improvement_ratio_clip,
                )
            )

        # Mild continuous pressure not to drift away from the current best.
        if math.isfinite(best_real_distance_m):
            gap_from_best = max(0.0, current_real_distance_m - best_real_distance_m)
            gap_from_best = min(gap_from_best, float(config.real_distance_gap_clip_m))
            real_distance_gap_penalty = -float(config.w_real_distance_gap_penalty) * (gap_from_best / max(1e-6, float(config.real_distance_gap_clip_m)))

        # No-improvement is a soft pressure signal only.
        # It is intentionally mild because the episode is not terminated unless
        # the drone actually moves away by a large percentage of the start distance.
        if not_approaching_time_s > 0.0:
            no_real_approach_time_penalty = -float(config.w_no_real_approach_time_penalty) * (not_approaching_time_s ** 1.15)

        # Small bonus for forward command only when the physical distance actually improved.
        if step_real_improvement_m > float(config.real_distance_progress_deadband_m):
            real_chase_action_bonus = float(config.w_real_chase_action_bonus) * max(0.0, vx) * min(1.0, step_real_improvement_m / 0.20)

    # ------------------------------------------------------------------
    # Chase-pressure anti-camping reward.
    # ------------------------------------------------------------------
    active_camera = str(env_info.get("active_camera", "front") or "front").lower()
    camera_authority = str(env_info.get("camera_authority", "FRONT_PRIMARY") or "FRONT_PRIMARY").upper()

    chase_pressure_active = bool(
        bool(config.chase_pressure_enabled)
        and has_target
        and math.isfinite(current_real_distance_m)
        and current_real_distance_m > float(config.chase_pressure_goal_distance_m)
        and not (
            bool(config.chase_pressure_bottom_disable)
            and active_camera == "bottom"
            and camera_authority in {"BOTTOM_PRIMARY", "BOTTOM_RECOVERY"}
        )
    )

    visible_far_no_approach_penalty = 0.0
    centered_forward_chase_bonus = 0.0
    centered_idle_far_penalty = 0.0
    retreat_while_far_penalty = 0.0
    chase_pressure_far_ratio = 0.0

    if chase_pressure_active:
        chase_pressure_far_ratio = _clip(
            (current_real_distance_m - float(config.chase_pressure_goal_distance_m))
            / max(1e-6, float(config.chase_pressure_far_clip_m)),
            0.0,
            1.0,
        )

        centered_factor = 0.0
        if center_error < float(config.chase_pressure_center_threshold):
            centered_factor = 1.0 - _clip(
                center_error / max(1e-6, float(config.chase_pressure_center_threshold)),
                0.0,
                1.0,
            )

        no_approach_now = bool(
            step_real_improvement_m <= float(config.chase_pressure_no_improvement_deadband_m)
        )

        if no_approach_now:
            time_factor = 1.0 + _clip(not_approaching_time_s / 4.0, 0.0, 2.0)
            visible_far_no_approach_penalty = (
                -float(config.w_visible_far_no_approach_penalty)
                * chase_pressure_far_ratio
                * time_factor
            )

        # When centered and far, forward motion is explicitly desirable.
        if centered_factor > 0.0:
            forward_action = max(0.0, vx)
            centered_forward_chase_bonus = (
                float(config.w_centered_forward_chase_bonus)
                * forward_action
                * chase_pressure_far_ratio
                * centered_factor
            )

            if vx < float(config.chase_pressure_min_forward_action):
                missing_forward = (
                    float(config.chase_pressure_min_forward_action) - max(0.0, vx)
                ) / max(1e-6, float(config.chase_pressure_min_forward_action))
                centered_idle_far_penalty = (
                    -float(config.w_centered_idle_far_penalty)
                    * missing_forward
                    * chase_pressure_far_ratio
                    * centered_factor
                )

        # Moving backward while far and visible is almost always wrong.
        if vx < -0.05:
            retreat_while_far_penalty = (
                -float(config.w_retreat_while_far_penalty)
                * abs(vx)
                * chase_pressure_far_ratio
            )

    # ------------------------------------------------------------------
    # Full-throttle chase stage.
    # ------------------------------------------------------------------
    fast_chase_throttle_bonus = 0.0
    fast_chase_progress_bonus = 0.0
    fast_chase_slow_penalty = 0.0
    fast_chase_active = bool(env_info.get("speed_stage", "") == "CHASE_FAST")

    if fast_chase_active and has_target:
        current_real_dist = _f(env_info.get("current_chase_distance_m", float("inf")), float("inf"))
        prev_real_dist = _f(env_info.get("previous_chase_distance_m", current_real_dist), current_real_dist)
        real_progress = max(0.0, prev_real_dist - current_real_dist) if math.isfinite(current_real_dist) and math.isfinite(prev_real_dist) else 0.0
        far_ratio = _clip((current_real_dist - float(config.fast_chase_far_distance_m)) / 8.0, 0.0, 1.0)

        if current_real_dist > float(config.fast_chase_far_distance_m):
            vx_action = _clip(float(vx), -1.0, 1.0)
            if vx_action > float(config.fast_chase_min_vx_action):
                fast_chase_throttle_bonus = (
                    float(config.w_fast_chase_throttle_bonus)
                    * _clip((vx_action - float(config.fast_chase_min_vx_action)) / max(1e-6, 1.0 - float(config.fast_chase_min_vx_action)), 0.0, 1.0)
                    * max(0.25, far_ratio)
                )
            else:
                fast_chase_slow_penalty = (
                    -float(config.w_fast_chase_slow_penalty)
                    * _clip((float(config.fast_chase_min_vx_action) - vx_action) / 1.2, 0.0, 1.0)
                    * max(0.25, far_ratio)
                )

            fast_chase_progress_bonus = float(config.w_fast_chase_progress_bonus) * _clip(real_progress / 0.45, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Bottom-camera relative velocity matching.
    # ------------------------------------------------------------------
    bottom_velocity_error_penalty = 0.0
    bottom_velocity_progress_bonus = 0.0
    bottom_velocity_ready_bonus = 0.0

    speed_stage = str(env_info.get("speed_stage", "") or "")
    bottom_velocity_stage = bool(speed_stage in {"BOTTOM_VELOCITY_MATCH", "BOTTOM_LANDING_READY"})
    bottom_img_vx = _clip(_f(env_info.get("bottom_img_vel_x", 0.0), 0.0), -4.0, 4.0)
    bottom_img_vy = _clip(_f(env_info.get("bottom_img_vel_y", 0.0), 0.0), -4.0, 4.0)
    bottom_img_speed = math.sqrt(bottom_img_vx * bottom_img_vx + bottom_img_vy * bottom_img_vy)
    prev_bottom_img_speed = _clip(_f(env_info.get("prev_bottom_img_speed", bottom_img_speed), bottom_img_speed), 0.0, 6.0)
    bottom_img_speed_improvement = prev_bottom_img_speed - bottom_img_speed

    if bottom_velocity_stage and bool(env_info.get("bottom_match", False)):
        bottom_velocity_error_penalty = -float(config.w_bottom_velocity_error_penalty) * _clip(bottom_img_speed / 1.2, 0.0, 1.0)
        if bottom_img_speed_improvement > 0.02:
            bottom_velocity_progress_bonus = float(config.w_bottom_velocity_progress_bonus) * _clip(bottom_img_speed_improvement / 0.35, 0.0, 1.0)
        if bool(env_info.get("bottom_velocity_ready", False)):
            bottom_velocity_ready_bonus = float(config.w_bottom_velocity_ready_bonus)

    # ------------------------------------------------------------------
    # Bottom-primary vertical alignment.
    # ------------------------------------------------------------------
    # Once the downward camera is the authority, the critical remaining error is
    # usually Bcy: target high/low in the downward image. The policy must learn
    # to drive Bcy toward zero, not merely keep the front camera centered.
    bottom_y_error_penalty = 0.0
    bottom_y_progress_reward = 0.0
    bottom_y_regress_penalty = 0.0
    bottom_center_ready_bonus = 0.0
    bottom_wrong_vx_penalty = 0.0
    bottom_correct_vx_bonus = 0.0
    bottom_alignment_active = False

    bottom_match = bool(env_info.get("bottom_match", False))
    bottom_confirmed = bool(env_info.get("bottom_confirmed", False))
    bottom_err_x = _clip(_f(env_info.get("bottom_err_x", 0.0), 0.0), -1.5, 1.5)
    bottom_err_y = _clip(_f(env_info.get("bottom_err_y", 0.0), 0.0), -1.5, 1.5)
    bottom_center_error = _clip(_f(env_info.get("bottom_center_error", math.sqrt(bottom_err_x * bottom_err_x + bottom_err_y * bottom_err_y)), 0.0), 0.0, 2.0)
    active_camera = str(env_info.get("active_camera", "front") or "front").lower()
    camera_authority = str(env_info.get("camera_authority", "FRONT_PRIMARY") or "FRONT_PRIMARY").upper()

    prev_bottom_abs_y = _clip(_f(env_info.get("prev_bottom_abs_err_y", abs(bottom_err_y)), abs(bottom_err_y)), 0.0, 1.5)
    bottom_abs_y = abs(bottom_err_y)
    bottom_y_improvement = prev_bottom_abs_y - bottom_abs_y

    # ------------------------------------------------------------------
    # Fine position accuracy reward.
    # ------------------------------------------------------------------
    # BBox-relative version:
    # Compare the bottom-camera frame center to the target bbox center, but
    # normalize by the bbox half-size instead of fixed global thresholds.
    #
    #   rel_x = |frame_center_x - bbox_center_x| / (bbox_width / 2)
    #   rel_y = |frame_center_y - bbox_center_y| / (bbox_height / 2)
    #   bbox_rel_err = max(rel_x, rel_y)
    #
    # Interpretation:
    #   bbox_rel_err < 0.5  -> frame center is inside the central half of bbox.
    #   bbox_rel_err ~= 1.0 -> frame center is near bbox boundary.
    #   bbox_rel_err > 1.0  -> frame center is outside bbox, unsafe for landing.
    fine_position_bonus = 0.0
    fine_position_penalty = 0.0
    fine_position_ready_bonus = 0.0
    fine_position_landing_ready_bonus = 0.0

    fine_position_stage = bool(speed_stage in {"BOTTOM_VELOCITY_MATCH", "BOTTOM_LANDING_READY"})
    fine_pos_err = max(abs(bottom_err_x), abs(bottom_err_y))

    bbox_rel_err_x = _clip(_f(env_info.get("bottom_bbox_rel_err_x", 999.0), 999.0), 0.0, 999.0)
    bbox_rel_err_y = _clip(_f(env_info.get("bottom_bbox_rel_err_y", 999.0), 999.0), 0.0, 999.0)
    bbox_rel_err = _clip(_f(env_info.get("bottom_bbox_rel_err", max(bbox_rel_err_x, bbox_rel_err_y)), max(bbox_rel_err_x, bbox_rel_err_y)), 0.0, 999.0)

    if bool(config.fine_position_reward_enabled) and fine_position_stage and bottom_match:
        if bool(getattr(config, "fine_position_use_bbox_relative", True)):
            safe_rel = float(config.fine_position_bbox_safe_rel)
            edge_rel = float(config.fine_position_bbox_edge_rel)
            ready_rel = float(config.fine_position_bbox_ready_rel)

            if bbox_rel_err <= safe_rel:
                # Strong bounded bonus for being inside the central part of the bbox.
                fine_position_bonus = (
                    float(config.fine_position_bbox_bonus_weight)
                    * math.exp(-float(config.fine_position_bbox_bonus_gain) * (bbox_rel_err ** 2.0))
                )
            elif bbox_rel_err <= edge_rel:
                # Inside bbox but outside the safe central half:
                # This is already unsafe for landing. Penalize immediately and
                # strongly; do not wait until the bbox boundary.
                edge_ratio = (bbox_rel_err - safe_rel) / max(1e-6, edge_rel - safe_rel)
                fine_position_penalty = -(
                    0.35 * float(config.fine_position_bbox_edge_penalty_weight)
                    + 0.65 * float(config.fine_position_bbox_edge_penalty_weight) * (edge_ratio ** 1.35)
                )
            else:
                # Outside bbox: very heavy unsafe penalty.
                outside_excess = bbox_rel_err - edge_rel
                fine_position_penalty = (
                    -float(config.fine_position_bbox_outside_penalty_weight)
                    * (1.0 + 2.0 * outside_excess ** 1.35)
                )

            if bbox_rel_err <= ready_rel:
                fine_position_ready_bonus = float(config.fine_position_bbox_ready_bonus)

            if speed_stage == "BOTTOM_LANDING_READY" and bbox_rel_err <= ready_rel:
                fine_position_landing_ready_bonus = float(config.fine_position_bbox_landing_ready_bonus)
        else:
            # Legacy absolute normalized fallback.
            fine_position_bonus = (
                float(config.fine_position_bonus_weight)
                * math.exp(-float(config.fine_position_bonus_gain) * (fine_pos_err ** 2.0))
            )
            fine_position_penalty = (
                -float(config.fine_position_penalty_weight)
                * (fine_pos_err ** float(config.fine_position_penalty_power))
            )
            if fine_pos_err <= float(config.fine_position_error_ready):
                fine_position_ready_bonus = float(config.fine_position_ready_bonus)
            if speed_stage == "BOTTOM_LANDING_READY" and fine_pos_err <= float(config.fine_position_error_good):
                fine_position_landing_ready_bonus = float(config.fine_position_landing_ready_bonus)

    bottom_alignment_active = bool(
        bool(config.bottom_alignment_reward_enabled)
        and active_camera == "bottom"
        and camera_authority in {"BOTTOM_PRIMARY", "BOTTOM_RECOVERY"}
        and bottom_match
    )

    if bottom_alignment_active:
        target_y = float(config.bottom_alignment_y_target_abs)
        y_excess = max(0.0, bottom_abs_y - target_y)
        bottom_y_error_penalty = -float(config.w_bottom_y_error_penalty) * (y_excess ** 1.35)

        if abs(bottom_y_improvement) > float(config.bottom_alignment_progress_deadband):
            if bottom_y_improvement > 0.0:
                bottom_y_progress_reward = float(config.w_bottom_y_progress_reward) * _clip(bottom_y_improvement / 0.25, 0.0, 1.0)
            else:
                bottom_y_regress_penalty = -float(config.w_bottom_y_regress_penalty) * _clip((-bottom_y_improvement) / 0.25, 0.0, 1.0)

        if bottom_center_error < float(config.bottom_alignment_center_target):
            bottom_center_ready_bonus = float(config.w_bottom_center_ready_bonus)

        # Geometry convention:
        #   bottom_err_y < 0 means the target is high in the downward image.
        #   Moving forward should reduce this error, so desired vx is -bottom_err_y.
        desired_vx_sign = float(np.sign(-bottom_err_y))
        if bottom_abs_y > target_y and abs(desired_vx_sign) > 0.0:
            commanded_vx = float(vx)
            signed_vx_alignment = desired_vx_sign * commanded_vx
            if signed_vx_alignment > 0.0:
                bottom_correct_vx_bonus = (
                    float(config.w_bottom_correct_vx_bonus)
                    * _clip(signed_vx_alignment, 0.0, 1.0)
                    * _clip(y_excess / 0.60, 0.0, 1.0)
                )
            else:
                bottom_wrong_vx_penalty = (
                    -float(config.w_bottom_wrong_vx_penalty)
                    * _clip(abs(commanded_vx), 0.0, 1.0)
                    * _clip(y_excess / 0.60, 0.0, 1.0)
                )

    # ------------------------------------------------------------------
    # Visibility / target loss.
    # ------------------------------------------------------------------
    # Visibility is intentionally modest. It should not dominate centering.
    visibility_reward = float(config.w_visibility) * bbox_conf * visible

    if has_target:
        lost_target_penalty = -float(config.w_lost_target) * (lost_target_time_norm ** 2)
    else:
        lost_target_penalty = -float(config.w_lost_target) * (1.0 + lost_target_time_norm)

    # ------------------------------------------------------------------
    # Recovery / fast reacquisition shaping.
    # ------------------------------------------------------------------
    previous_had_target = bool(env_info.get("previous_had_target", False))
    previous_lost_time_norm = _clip(_f(env_info.get("previous_lost_target_time_norm", 0.0), 0.0), 0.0, 1.5)
    lost_to_match_transition = bool(env_info.get("lost_to_match_transition", False))
    likely_overshoot = bool(env_info.get("likely_overshoot", False))

    last_known_err_x = _clip(_f(obs.get("err_x", 0.0), 0.0), -1.0, 1.0)
    yaw_search_alignment = 0.0
    recovery_yaw_reward = 0.0
    wrong_recovery_yaw_penalty = 0.0

    if not has_target and previous_had_target and abs(last_known_err_x) >= float(config.recovery_yaw_deadband):
        yaw_search_alignment = _clip(yaw * math.copysign(1.0, last_known_err_x), -1.0, 1.0)

        if yaw_search_alignment > 0.0:
            recovery_yaw_reward = (
                float(config.w_recovery_yaw)
                * yaw_search_alignment
                * (0.5 + 0.5 * _clip(abs(last_known_err_x), 0.0, 1.0))
            )
        else:
            wrong_recovery_yaw_penalty = (
                -float(config.w_wrong_recovery_yaw)
                * abs(yaw_search_alignment)
                * (0.5 + previous_lost_time_norm)
            )

    if lost_to_match_transition and has_target:
        fast_reacquire_bonus = float(config.w_fast_reacquire) * (1.0 - _clip(previous_lost_time_norm, 0.0, 1.0))
    else:
        fast_reacquire_bonus = 0.0

    overshoot_penalty = -float(config.w_overshoot_penalty) if likely_overshoot else 0.0
    lost_time_accel_penalty = -float(config.w_lost_time_accel_penalty) * (lost_target_time_norm ** 2) if not has_target else 0.0

    # ------------------------------------------------------------------
    # Smooth target motion / non-jitter behavior.
    # ------------------------------------------------------------------
    target_stability_score = _clip(_f(obs.get("target_stability_score", 0.0), 0.0), 0.0, 1.0)
    smooth_follow_score = target_stability_score

    img_motion = math.sqrt(
        _f(obs.get("img_vx", 0.0), 0.0) ** 2
        + _f(obs.get("img_vy", 0.0), 0.0) ** 2
    )
    motion_penalty_factor = _clip(1.0 - img_motion / max(1e-6, float(config.max_img_motion)), 0.0, 1.0)
    smooth_follow_reward = float(config.w_smooth_follow) * smooth_follow_score * motion_penalty_factor * visible

    # Stable chase bonus: only when centered, near desired distance, visible, and calm.
    is_stably_centered = center_error < float(config.stable_center_threshold)
    is_distance_ok = distance_error < float(config.distance_tolerance)
    is_yaw_calm = yaw_action < float(config.stable_yaw_threshold)

    if has_target and is_stably_centered and is_distance_ok and is_yaw_calm:
        stable_follow_bonus = float(config.w_stable_follow_bonus)
    else:
        stable_follow_bonus = 0.0

    # ------------------------------------------------------------------
    # Altitude / obstacle / safety.
    # ------------------------------------------------------------------
    altitude_m = _f(env_info.get("altitude_m", 0.0), 0.0)

    if float(config.min_safe_altitude_m) <= altitude_m <= float(config.max_safe_altitude_m):
        altitude_reward = float(config.w_altitude_safe)
        altitude_penalty = 0.0
    elif altitude_m < float(config.min_safe_altitude_m):
        altitude_reward = 0.0
        altitude_penalty = -float(config.w_altitude_low_penalty) * (float(config.min_safe_altitude_m) - altitude_m)
    else:
        altitude_reward = 0.0
        altitude_penalty = -float(config.w_altitude_high_penalty) * (altitude_m - float(config.max_safe_altitude_m))

    collision_risk_score = _clip(_f(obs.get("collision_risk_score", 0.0), 0.0), 0.0, 1.0)
    obstacle_penalty = -float(config.w_obstacle) * collision_risk_score

    safety_intervention_penalty = -float(config.w_safety_intervention) if bool(env_info.get("safety_intervention", False)) else 0.0

    # ------------------------------------------------------------------
    # Control / time.
    # ------------------------------------------------------------------
    control_penalty = -float(config.w_control) * action_norm
    action_delta_penalty = -float(config.w_action_delta) * action_delta_norm

    yaw_abs_penalty = -float(config.w_yaw_abs) * yaw_action
    yaw_delta_penalty = -float(config.w_yaw_delta) * yaw_delta

    if center_error < float(config.center_calm_threshold):
        yaw_centered_penalty = -float(config.w_yaw_when_centered) * yaw_action
    else:
        yaw_centered_penalty = 0.0

    # Hard shaping against the bad habit observed in training:
    # If the target is already focused and centered, yaw is harmful noise.
    # This is stronger than the generic yaw penalties and is active only in
    # clean MATCH conditions, so recovery/search yaw is still allowed.
    focused_center_threshold = float(config.focused_center_threshold)
    focused_yaw_threshold = float(config.focused_centered_yaw_threshold)

    is_focused_centered = bool(
        is_match_mode
        and bbox_conf >= float(config.focused_bbox_conf_threshold)
        and center_error < focused_center_threshold
        and abs_err_x < float(config.focused_center_x_threshold)
        and abs_err_y < float(config.focused_center_y_threshold)
    )

    if is_focused_centered and yaw_action > focused_yaw_threshold:
        excess_yaw = max(0.0, yaw_action - focused_yaw_threshold)
        normalized_excess_yaw = _clip(
            excess_yaw / max(1e-6, 1.0 - focused_yaw_threshold),
            0.0,
            1.0,
        )
        centered_strength = _clip(
            1.0 - center_error / max(1e-6, focused_center_threshold),
            0.0,
            1.0,
        )
        focus_strength = _clip(
            (bbox_conf - float(config.focused_bbox_conf_threshold))
            / max(1e-6, 1.0 - float(config.focused_bbox_conf_threshold)),
            0.0,
            1.0,
        )
        focused_centered_yaw_penalty = (
            -float(config.w_yaw_when_focused_centered)
            * (normalized_excess_yaw ** float(config.focused_centered_yaw_power))
            * (0.75 + 0.25 * focus_strength)
            * (1.0 + centered_strength)
        )

        # Exponential version of the same rule. This becomes much harsher for
        # medium/large yaw, while keeping tiny corrective yaw almost unaffected.
        exp_yaw_value = _clip(
            normalized_excess_yaw,
            0.0,
            float(config.exp_focused_yaw_clip),
        )
        exp_focused_centered_yaw_penalty = (
            -float(config.w_exp_focused_centered_yaw)
            * _exp_shaping(
                exp_yaw_value,
                float(config.exp_focused_yaw_gain),
                float(config.exp_focused_yaw_clip),
            )
            * (0.75 + 0.25 * focus_strength)
            * (1.0 + centered_strength)
        )
    else:
        focused_centered_yaw_penalty = 0.0
        exp_focused_centered_yaw_penalty = 0.0

    # If the previous frame was centered/focused and the current frame became
    # worse while yaw was used, punish it exponentially. This directly targets
    # "unnecessary yaw that pulls the target out of center".
    is_yaw_decentering = bool(
        is_match_mode
        and yaw_action > focused_yaw_threshold
        and bbox_conf >= float(config.focused_bbox_conf_threshold)
        and prev_center_error < float(config.yaw_decenter_prev_center_threshold)
        and center_error < float(config.yaw_decenter_current_max_threshold)
        and center_improvement < -float(config.center_progress_deadband)
    )

    if is_yaw_decentering:
        decenter_amount = _clip(
            -center_improvement,
            0.0,
            float(config.yaw_decenter_error_clip),
        )
        yaw_factor = _clip(
            (yaw_action - focused_yaw_threshold) / max(1e-6, 1.0 - focused_yaw_threshold),
            0.0,
            1.0,
        )
        yaw_decenter_exp_penalty = (
            -float(config.w_yaw_decenter_exp_penalty)
            * _exp_shaping(
                decenter_amount,
                float(config.yaw_decenter_exp_gain),
                float(config.yaw_decenter_error_clip),
            )
            * (0.35 + 0.65 * yaw_factor)
        )
    else:
        yaw_decenter_exp_penalty = 0.0

    if yaw_action > float(config.useless_yaw_threshold) and center_improvement <= 0.0 and visible:
        useless_yaw_penalty = -float(config.w_useless_yaw_penalty) * yaw_action * (0.5 + center_error)
    else:
        useless_yaw_penalty = 0.0

    # MATCH-only yaw discipline:
    # Do not waste yaw when horizontal error is already small. This directly
    # targets the observed bad habit: unnecessary right/left yaw while the
    # target is mainly low in the image.
    if is_match_mode and abs_err_x < float(config.x_center_yaw_threshold):
        yaw_x_centered_penalty = (
            -float(config.w_yaw_when_x_centered)
            * yaw_action
            * (1.0 + _clip(center_error, 0.0, 1.0))
        )
    else:
        yaw_x_centered_penalty = 0.0

    # If the error is mostly vertical, yaw is not the useful correction.
    # Keep yaw calm and let chase/geometry/camera tilt solve the vertical issue.
    if (
        is_match_mode
        and abs_err_x < float(config.vertical_only_yaw_x_threshold)
        and abs_err_y > float(config.vertical_only_yaw_y_threshold)
    ):
        yaw_vertical_only_penalty = (
            -float(config.w_yaw_when_vertical_only_error)
            * yaw_action
            * _clip(abs_err_y, 0.0, 1.0)
        )
    else:
        yaw_vertical_only_penalty = 0.0

    cumulative_abs_yaw_cmd_deg = _f(env_info.get("cumulative_abs_yaw_cmd_deg", 0.0), 0.0)
    yaw_budget_excess_deg = max(
        0.0,
        cumulative_abs_yaw_cmd_deg - float(config.yaw_command_budget_free_deg),
    )
    yaw_budget_excess_deg = _clip(
        yaw_budget_excess_deg,
        0.0,
        float(config.yaw_command_budget_clip_deg),
    )
    yaw_command_budget_penalty = (
        -float(config.w_yaw_command_budget_penalty)
        * (yaw_budget_excess_deg ** 2)
    )

    if is_distance_ok:
        slow_penalty = -float(config.w_slow) * abs(vx)
    else:
        slow_penalty = 0.0

    time_penalty = -float(config.w_time)

    # ------------------------------------------------------------------
    # Terminal penalties.
    # ------------------------------------------------------------------
    terminal_collision_penalty = -float(config.collision_penalty) if bool(env_info.get("collision_detected", False)) else 0.0

    term_reason = str(env_info.get("termination_reason", "") or "")
    if term_reason == "target_lost_too_long":
        terminal_reason_penalty = -float(config.w_lost_target) * 2.0
    elif term_reason in {"altitude_too_low", "altitude_too_high"}:
        terminal_reason_penalty = -float(config.altitude_termination_penalty)
    elif term_reason == "episode_timeout":
        terminal_reason_penalty = -float(config.timeout_penalty)
    elif term_reason == "emergency_horizontal_obstacle_distance":
        terminal_reason_penalty = -float(config.w_obstacle) * 2.0
    elif term_reason == "collision":
        terminal_reason_penalty = -float(config.collision_penalty)
    elif term_reason == "distance_damage_too_large":
        terminal_reason_penalty = -float(config.distance_damage_terminal_penalty) * (1.0 + distance_damage_ratio)
    elif term_reason == "handoff_success":
        terminal_reason_penalty = float(config.handoff_success_bonus)
    else:
        terminal_reason_penalty = 0.0

    total_reward = (
        center_reward
        + center_error_penalty
        + frame_edge_penalty
        + center_progress_reward
        + center_regress_penalty
        + center_action_alignment_reward
        + offcenter_idle_penalty
        + distance_reward
        + distance_linear_deviation_penalty
        + distance_exp_progress_reward
        + distance_exp_regress_penalty
        + chase_forward_reward
        + too_close_penalty
        + real_distance_step_progress_reward
        + real_distance_step_regress_penalty
        + real_best_distance_improvement_reward
        + real_distance_gap_penalty
        + no_real_approach_time_penalty
        + real_chase_action_bonus
        + visible_far_no_approach_penalty
        + centered_forward_chase_bonus
        + centered_idle_far_penalty
        + retreat_while_far_penalty
        + fast_chase_throttle_bonus
        + fast_chase_progress_bonus
        + fast_chase_slow_penalty
        + bottom_velocity_error_penalty
        + bottom_velocity_progress_bonus
        + bottom_velocity_ready_bonus
        + fine_position_bonus
        + fine_position_penalty
        + fine_position_ready_bonus
        + fine_position_landing_ready_bonus
        + bottom_y_error_penalty
        + bottom_y_progress_reward
        + bottom_y_regress_penalty
        + bottom_center_ready_bonus
        + bottom_wrong_vx_penalty
        + bottom_correct_vx_bonus
        + visibility_reward
        + lost_target_penalty
        + recovery_yaw_reward
        + wrong_recovery_yaw_penalty
        + fast_reacquire_bonus
        + overshoot_penalty
        + lost_time_accel_penalty
        + smooth_follow_reward
        + stable_follow_bonus
        + altitude_reward
        + altitude_penalty
        + yaw_abs_penalty
        + yaw_delta_penalty
        + yaw_centered_penalty
        + focused_centered_yaw_penalty
        + exp_focused_centered_yaw_penalty
        + yaw_decenter_exp_penalty
        + yaw_command_budget_penalty
        + useless_yaw_penalty
        + yaw_x_centered_penalty
        + yaw_vertical_only_penalty
        + control_penalty
        + action_delta_penalty
        + slow_penalty
        + time_penalty
        + obstacle_penalty
        + safety_intervention_penalty
        + terminal_collision_penalty
        + terminal_reason_penalty
    )

    unclipped_total_reward = float(total_reward)
    if bool(config.clip_total_reward):
        total_reward = _clip(total_reward, config.min_total_reward, config.max_total_reward)

    reward_parts = {
        # Existing keys expected by logs.
        "center_reward": float(center_reward),
        "distance_reward": float(distance_reward),
        "distance_score": float(distance_score),
        "visibility_reward": float(visibility_reward),
        "altitude_reward": float(altitude_reward + altitude_penalty),
        "smooth_follow_reward": float(smooth_follow_reward),
        "smooth_follow_score": float(smooth_follow_score),
        "control_penalty": float(control_penalty),
        "action_delta_penalty": float(action_delta_penalty),
        "obstacle_penalty": float(obstacle_penalty),
        "safety_intervention_penalty": float(safety_intervention_penalty),
        "collision_penalty": float(terminal_collision_penalty + terminal_reason_penalty),

        # Detailed keys.
        "err_x_abs": float(abs_err_x),
        "err_y_abs": float(abs_err_y),
        "center_error": float(center_error),
        "prev_center_error": float(prev_center_error),
        "centered_score": float(centered_score),
        "center_error_improvement": float(center_improvement),
        "center_error_penalty": float(center_error_penalty),
        "frame_edge_penalty": float(frame_edge_penalty),
        "center_progress_reward": float(center_progress_reward),
        "center_regress_penalty": float(center_regress_penalty),
        "center_action_alignment_reward": float(center_action_alignment_reward),
        "vertical_alignment": float(vertical_alignment),
        "lateral_alignment": float(lateral_alignment),
        "offcenter_idle_penalty": float(offcenter_idle_penalty),

        "distance_proxy_norm": float(distance_proxy),
        "distance_error": float(distance_error),
        "prev_distance_error": float(prev_distance_error),
        "distance_error_improvement": float(distance_error_improvement),
        "distance_linear_deviation_penalty": float(distance_linear_deviation_penalty),
        "distance_exp_progress_reward": float(distance_exp_progress_reward),
        "distance_exp_regress_penalty": float(distance_exp_regress_penalty),
        "chase_forward_reward": float(chase_forward_reward),
        "too_close_penalty": float(too_close_penalty),

        "current_chase_distance_m": float(current_real_distance_m) if math.isfinite(current_real_distance_m) else 9999.0,
        "previous_chase_distance_m": float(previous_real_distance_m) if math.isfinite(previous_real_distance_m) else 9999.0,
        "initial_chase_distance_m": float(initial_real_distance_m) if math.isfinite(initial_real_distance_m) else 9999.0,
        "distance_damage_ratio": float(distance_damage_ratio),
        "bottom_match": float(1.0 if bool(env_info.get("bottom_match", False)) else 0.0),
        "handoff_ready": float(1.0 if bool(env_info.get("handoff_ready", False)) else 0.0),
        "bottom_weight": float(_f(env_info.get("bottom_weight", 0.0), 0.0)),
        "best_chase_distance_m": float(best_real_distance_m) if math.isfinite(best_real_distance_m) else 9999.0,
        "step_chase_distance_improvement_m": float(step_real_improvement_m),
        "last_approach_improvement_m": float(best_real_improvement_m),
        "real_distance_step_progress_reward": float(real_distance_step_progress_reward),
        "real_distance_step_regress_penalty": float(real_distance_step_regress_penalty),
        "real_best_distance_improvement_reward": float(real_best_distance_improvement_reward),
        "real_distance_gap_penalty": float(real_distance_gap_penalty),
        "no_real_approach_time_penalty": float(no_real_approach_time_penalty),
        "real_chase_action_bonus": float(real_chase_action_bonus),
        "chase_pressure_active": float(1.0 if chase_pressure_active else 0.0),
        "chase_pressure_far_ratio": float(chase_pressure_far_ratio),
        "visible_far_no_approach_penalty": float(visible_far_no_approach_penalty),
        "centered_forward_chase_bonus": float(centered_forward_chase_bonus),
        "centered_idle_far_penalty": float(centered_idle_far_penalty),
        "retreat_while_far_penalty": float(retreat_while_far_penalty),
        "fast_chase_active": float(1.0 if fast_chase_active else 0.0),
        "fast_chase_throttle_bonus": float(fast_chase_throttle_bonus),
        "fast_chase_progress_bonus": float(fast_chase_progress_bonus),
        "fast_chase_slow_penalty": float(fast_chase_slow_penalty),
        "bottom_velocity_error_penalty": float(bottom_velocity_error_penalty),
        "bottom_velocity_progress_bonus": float(bottom_velocity_progress_bonus),
        "bottom_velocity_ready_bonus": float(bottom_velocity_ready_bonus),
        "bottom_img_speed": float(bottom_img_speed),
        "bottom_img_speed_improvement": float(bottom_img_speed_improvement),
        "fine_position_bonus": float(fine_position_bonus),
        "fine_position_penalty": float(fine_position_penalty),
        "fine_position_ready_bonus": float(fine_position_ready_bonus),
        "fine_position_landing_ready_bonus": float(fine_position_landing_ready_bonus),
        "fine_position_error_inf": float(fine_pos_err),
        "fine_position_bbox_rel_err": float(bbox_rel_err),
        "fine_position_bbox_rel_err_x": float(bbox_rel_err_x),
        "fine_position_bbox_rel_err_y": float(bbox_rel_err_y),
        "fine_position_stage": float(1.0 if fine_position_stage else 0.0),
        "bottom_alignment_active": float(1.0 if bottom_alignment_active else 0.0),
        "bottom_y_error_penalty": float(bottom_y_error_penalty),
        "bottom_y_progress_reward": float(bottom_y_progress_reward),
        "bottom_y_regress_penalty": float(bottom_y_regress_penalty),
        "bottom_center_ready_bonus": float(bottom_center_ready_bonus),
        "bottom_wrong_vx_penalty": float(bottom_wrong_vx_penalty),
        "bottom_correct_vx_bonus": float(bottom_correct_vx_bonus),
        "bottom_abs_err_y": float(bottom_abs_y),
        "bottom_y_improvement": float(bottom_y_improvement),

        "yaw_abs_penalty": float(yaw_abs_penalty),
        "yaw_delta_penalty": float(yaw_delta_penalty),
        "yaw_centered_penalty": float(yaw_centered_penalty),
        "focused_centered_yaw_penalty": float(focused_centered_yaw_penalty),
        "exp_focused_centered_yaw_penalty": float(exp_focused_centered_yaw_penalty),
        "yaw_decenter_exp_penalty": float(yaw_decenter_exp_penalty),
        "yaw_command_budget_penalty": float(yaw_command_budget_penalty),
        "yaw_budget_excess_deg": float(yaw_budget_excess_deg),
        "is_focused_centered_for_yaw": float(1.0 if is_focused_centered else 0.0),
        "is_yaw_decentering": float(1.0 if is_yaw_decentering else 0.0),
        "useless_yaw_penalty": float(useless_yaw_penalty),
        "yaw_x_centered_penalty": float(yaw_x_centered_penalty),
        "yaw_vertical_only_penalty": float(yaw_vertical_only_penalty),
        "stable_follow_bonus": float(stable_follow_bonus),
        "lost_target_penalty": float(lost_target_penalty),
        "recovery_yaw_reward": float(recovery_yaw_reward),
        "wrong_recovery_yaw_penalty": float(wrong_recovery_yaw_penalty),
        "fast_reacquire_bonus": float(fast_reacquire_bonus),
        "overshoot_penalty": float(overshoot_penalty),
        "lost_time_accel_penalty": float(lost_time_accel_penalty),
        "yaw_search_alignment": float(yaw_search_alignment),
        "previous_lost_target_time_norm": float(previous_lost_time_norm),
        "time_penalty": float(time_penalty),
        "unclipped_total_reward": float(unclipped_total_reward),
        "total_reward": float(total_reward),
    }

    return float(total_reward), reward_parts
