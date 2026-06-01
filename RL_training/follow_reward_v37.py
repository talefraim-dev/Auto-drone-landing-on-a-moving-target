"""
follow_reward_v37.py

Drop-in reward function for the current tracking/following task.

Main training goals:
    1. Keep target bbox horizontally centered with calm yaw.
    2. Reach the desired target distance quickly.
    3. Keep the desired safe distance after reaching it.
    4. Strongly reward stable follow behavior.
    5. Strongly penalize target loss, dangerous proximity, unstable yaw, collisions,
       and distance error that grows over time.

Expected call signature:
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
    w_center: float = 15.0
    center_reward_alpha: float = 2.0

    w_distance: float = 8.0
    desired_distance_proxy: float = 0.94
    distance_tolerance: float = 0.04

    w_visibility: float = 1.0
    w_lost_target: float = 8.0

    w_altitude_safe: float = 0.3
    w_altitude_low_penalty: float = 2.0
    w_altitude_high_penalty: float = 2.0
    min_safe_altitude_m: float = 2.0
    max_safe_altitude_m: float = 30.0

    w_smooth_follow: float = 2.0
    max_img_motion: float = 0.35

    w_control: float = 0.05
    w_action_delta: float = 0.15
    w_slow: float = 0.02
    w_time: float = 0.02

    w_obstacle: float = 4.0
    w_safety_intervention: float = 1.5

    collision_penalty: float = 25.0
    timeout_penalty: float = 3.0
    altitude_termination_penalty: float = 12.0

    # ------------------------------------------------------------------
    # New tracking-specific shaping.
    # ------------------------------------------------------------------

    # Medium linear penalty for not maintaining the desired distance.
    # This is intentionally separate from the Gaussian distance reward.
    w_distance_linear_deviation: float = 2.0

    # Exponential reward/penalty based on distance-error improvement.
    # If abs(distance_proxy - desired) shrinks quickly -> big positive reward.
    # If it grows quickly -> big penalty.
    w_distance_error_exp_progress: float = 7.0
    w_distance_error_exp_regress: float = 9.0
    distance_error_exp_gain: float = 18.0
    distance_error_exp_clip: float = 0.12
    distance_progress_deadband: float = 0.0015

    # Extra penalty when the drone is too close to the target.
    # distance_proxy_norm is lower when the target is closer/larger.
    w_too_close_exp_penalty: float = 8.0
    too_close_margin: float = 0.06
    too_close_exp_gain: float = 16.0

    # Yaw stability and anti-jitter.
    w_yaw_abs: float = 0.35
    w_yaw_delta: float = 1.25
    w_yaw_when_centered: float = 2.5
    center_calm_threshold: float = 0.16

    # Strong stable-follow bonus when target is centered, distance is correct,
    # target is visible, and yaw is calm.
    w_stable_follow_bonus: float = 6.0
    stable_center_threshold: float = 0.18
    stable_yaw_threshold: float = 0.18

    # Optional center-error progress shaping.
    w_center_error_exp_progress: float = 2.0
    w_center_error_exp_regress: float = 2.5
    center_error_exp_gain: float = 6.0
    center_error_exp_clip: float = 0.15
    center_progress_deadband: float = 0.002

    # Reward clipping for training stability.
    clip_total_reward: bool = True
    min_total_reward: float = -35.0
    max_total_reward: float = 35.0


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
    """
    Exponential shaping with clipping to avoid exploding rewards.
    value should already be signed.
    """
    v = _clip(float(value), -float(clip_abs), float(clip_abs))
    return math.exp(float(gain) * abs(v)) - 1.0


def _distance_error(obs: dict[str, Any], cfg: FollowRewardConfig) -> tuple[float, float, float]:
    """
    Return:
        distance_proxy_norm, current_abs_error, previous_abs_error

    Uses distance_proxy_delta when available:
        prev_distance_proxy = current_distance_proxy - distance_proxy_delta
    """
    distance_proxy = _f(obs.get("distance_proxy_norm", 1.0), 1.0)
    distance_delta = _f(obs.get("distance_proxy_delta", 0.0), 0.0)

    desired = float(cfg.desired_distance_proxy)

    current_error = abs(distance_proxy - desired)

    prev_distance_proxy = distance_proxy - distance_delta
    prev_error = abs(prev_distance_proxy - desired)

    return distance_proxy, current_error, prev_error


def _center_error(obs: dict[str, Any]) -> tuple[float, float]:
    """
    Return:
        center_error, prev_center_error_estimate

    err_x and err_y are expected to be normalized image errors.
    For yaw, err_x is the most important component.
    """
    err_x = abs(_f(obs.get("err_x", 1.0), 1.0))
    err_y = abs(_f(obs.get("err_y", 1.0), 1.0))

    # Follow task is mostly yaw-horizontal. Vertical matters less because altitude
    # is stabilized separately, so combine with stronger X weight.
    center_error = _clip(0.75 * err_x + 0.25 * err_y, 0.0, 1.5)

    img_vx = _f(obs.get("img_vx", 0.0), 0.0)
    img_vy = _f(obs.get("img_vy", 0.0), 0.0)

    # Approximate previous error using image-space motion.
    # If current bbox moved away from center, current error tends to be larger.
    prev_err_x = abs(err_x - abs(img_vx))
    prev_err_y = abs(err_y - abs(img_vy))
    prev_center_error = _clip(0.75 * prev_err_x + 0.25 * prev_err_y, 0.0, 1.5)

    return center_error, prev_center_error


def compute_follow_reward(
    obs: dict[str, Any],
    action: np.ndarray,
    prev_action: np.ndarray,
    env_info: dict[str, Any] | None,
    config: FollowRewardConfig,
) -> tuple[float, dict[str, float]]:
    """
    Compute reward for target-car following.

    Notes:
        - action layout is expected to be [vx, vy, vz, yaw_rate].
        - distance_proxy_norm is lower when target is closer/larger.
        - desired_distance_proxy defines the safe distance to maintain.
    """
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

    yaw_action = abs(float(action[3]))
    yaw_delta = abs(float(action[3] - prev_action[3]))
    action_delta_norm = float(np.linalg.norm(action - prev_action))
    action_norm = float(np.linalg.norm(action))

    # ------------------------------------------------------------------
    # Center / yaw alignment.
    # ------------------------------------------------------------------
    center_error, prev_center_error = _center_error(obs)
    centered_score = _clip(_f(obs.get("centered_score", 1.0 - center_error), 1.0 - center_error), 0.0, 1.0)

    center_reward = float(config.w_center) * (centered_score ** float(config.center_reward_alpha)) * visible

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

    yaw_abs_penalty = -float(config.w_yaw_abs) * yaw_action
    yaw_delta_penalty = -float(config.w_yaw_delta) * yaw_delta

    if center_error < float(config.center_calm_threshold):
        yaw_centered_penalty = -float(config.w_yaw_when_centered) * yaw_action
    else:
        yaw_centered_penalty = 0.0

    # ------------------------------------------------------------------
    # Distance keeping and fast convergence.
    # ------------------------------------------------------------------
    distance_proxy, distance_error, prev_distance_error = _distance_error(obs, config)

    tol = max(1e-6, float(config.distance_tolerance))
    distance_score = math.exp(-((distance_error / tol) ** 2))
    distance_reward = float(config.w_distance) * distance_score * visible

    # Medium linear safety penalty for average/continuous distance deviation.
    distance_linear_deviation_penalty = (
        -float(config.w_distance_linear_deviation) * distance_error * visible
    )

    distance_error_improvement = prev_distance_error - distance_error

    if abs(distance_error_improvement) < float(config.distance_progress_deadband):
        distance_exp_progress_reward = 0.0
        distance_exp_regress_penalty = 0.0
    elif distance_error_improvement > 0.0:
        distance_exp_progress_reward = (
            float(config.w_distance_error_exp_progress)
            * _exp_shaping(
                distance_error_improvement,
                config.distance_error_exp_gain,
                config.distance_error_exp_clip,
            )
            * visible
        )
        distance_exp_regress_penalty = 0.0
    else:
        distance_exp_progress_reward = 0.0
        distance_exp_regress_penalty = (
            -float(config.w_distance_error_exp_regress)
            * _exp_shaping(
                -distance_error_improvement,
                config.distance_error_exp_gain,
                config.distance_error_exp_clip,
            )
            * visible
        )

    too_close_threshold = float(config.desired_distance_proxy) - float(config.too_close_margin)
    if distance_proxy < too_close_threshold and visible > 0.0:
        too_close_error = too_close_threshold - distance_proxy
        too_close_penalty = (
            -float(config.w_too_close_exp_penalty)
            * _exp_shaping(too_close_error, config.too_close_exp_gain, config.distance_error_exp_clip)
        )
    else:
        too_close_penalty = 0.0

    # ------------------------------------------------------------------
    # Visibility / target loss.
    # ------------------------------------------------------------------
    visibility_reward = float(config.w_visibility) * bbox_conf * visible

    if has_target:
        lost_target_penalty = -float(config.w_lost_target) * (lost_target_time_norm ** 2)
    else:
        lost_target_penalty = -float(config.w_lost_target) * (1.0 + lost_target_time_norm)

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

    # ------------------------------------------------------------------
    # Stable follow bonus: only when all main goals are satisfied together.
    # ------------------------------------------------------------------
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
        altitude_penalty = -float(config.w_altitude_low_penalty) * (
            float(config.min_safe_altitude_m) - altitude_m
        )
    else:
        altitude_reward = 0.0
        altitude_penalty = -float(config.w_altitude_high_penalty) * (
            altitude_m - float(config.max_safe_altitude_m)
        )

    collision_risk_score = _clip(_f(obs.get("collision_risk_score", 0.0), 0.0), 0.0, 1.0)
    obstacle_penalty = -float(config.w_obstacle) * collision_risk_score

    safety_intervention_penalty = (
        -float(config.w_safety_intervention)
        if bool(env_info.get("safety_intervention", False))
        else 0.0
    )

    # ------------------------------------------------------------------
    # Control / time.
    # ------------------------------------------------------------------
    control_penalty = -float(config.w_control) * action_norm
    action_delta_penalty = -float(config.w_action_delta) * action_delta_norm

    # Small preference for not using unnecessary forward speed once distance is good.
    if is_distance_ok:
        slow_penalty = -float(config.w_slow) * abs(float(action[0]))
    else:
        slow_penalty = 0.0

    time_penalty = -float(config.w_time)

    # ------------------------------------------------------------------
    # Terminal penalties.
    # ------------------------------------------------------------------
    terminal_collision_penalty = (
        -float(config.collision_penalty)
        if bool(env_info.get("collision_detected", False))
        else 0.0
    )

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
    else:
        terminal_reason_penalty = 0.0

    total_reward = (
        center_reward
        + center_progress_reward
        + center_regress_penalty
        + distance_reward
        + distance_linear_deviation_penalty
        + distance_exp_progress_reward
        + distance_exp_regress_penalty
        + too_close_penalty
        + visibility_reward
        + lost_target_penalty
        + smooth_follow_reward
        + stable_follow_bonus
        + altitude_reward
        + altitude_penalty
        + yaw_abs_penalty
        + yaw_delta_penalty
        + yaw_centered_penalty
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

        # New detailed keys.
        "center_error": float(center_error),
        "center_error_improvement": float(center_improvement),
        "center_progress_reward": float(center_progress_reward),
        "center_regress_penalty": float(center_regress_penalty),

        "distance_proxy_norm": float(distance_proxy),
        "distance_error": float(distance_error),
        "prev_distance_error": float(prev_distance_error),
        "distance_error_improvement": float(distance_error_improvement),
        "distance_linear_deviation_penalty": float(distance_linear_deviation_penalty),
        "distance_exp_progress_reward": float(distance_exp_progress_reward),
        "distance_exp_regress_penalty": float(distance_exp_regress_penalty),
        "too_close_penalty": float(too_close_penalty),

        "yaw_abs_penalty": float(yaw_abs_penalty),
        "yaw_delta_penalty": float(yaw_delta_penalty),
        "yaw_centered_penalty": float(yaw_centered_penalty),
        "stable_follow_bonus": float(stable_follow_bonus),
        "lost_target_penalty": float(lost_target_penalty),
        "time_penalty": float(time_penalty),
        "unclipped_total_reward": float(unclipped_total_reward),
        "total_reward": float(total_reward),
    }

    return float(total_reward), reward_parts
