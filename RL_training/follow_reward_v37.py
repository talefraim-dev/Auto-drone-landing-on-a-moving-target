"""
Follow Agent reward with safety awareness — v37.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class FollowRewardConfig:
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

    desired_distance_proxy: float = 0.45
    distance_tolerance: float = 0.35

    min_safe_altitude_m: float = 3.0
    max_safe_altitude_m: float = 20.0
    max_img_motion: float = 1.5

    collision_penalty: float = 50.0


def compute_follow_reward(
    obs: Dict[str, float],
    action: np.ndarray,
    prev_action: Optional[np.ndarray],
    env_info: Dict[str, object],
    config: Optional[FollowRewardConfig] = None,
) -> Tuple[float, Dict[str, float]]:
    cfg = config or FollowRewardConfig()

    action = np.asarray(action, dtype=np.float32)
    if action.shape != (4,):
        raise ValueError("action must have shape (4,): [vx, vy, vz, yaw_rate].")

    has_target = float(obs.get("has_target", 0.0))
    centered_score = float(obs.get("centered_score", 0.0))
    distance_proxy = float(obs.get("distance_proxy_norm", 1.0))
    img_vx = float(obs.get("img_vx", 0.0))
    img_vy = float(obs.get("img_vy", 0.0))
    collision_risk_score = float(obs.get("collision_risk_score", 0.0))

    altitude_m = float(env_info.get("altitude_m", 0.0))
    collision_detected = bool(env_info.get("collision_detected", False))
    safety_intervention = bool(env_info.get("safety_intervention", False))

    parts: Dict[str, float] = {}

    center_reward = cfg.w_center * centered_score
    parts["center_reward"] = center_reward

    distance_error = abs(distance_proxy - cfg.desired_distance_proxy)
    distance_score = 1.0 - min(distance_error / cfg.distance_tolerance, 1.0)
    distance_reward = cfg.w_distance * distance_score
    parts["distance_reward"] = distance_reward
    parts["distance_score"] = distance_score

    visibility_reward = cfg.w_visibility if has_target > 0.5 else -cfg.w_lost_target
    parts["visibility_reward"] = visibility_reward

    if altitude_m < cfg.min_safe_altitude_m:
        altitude_reward = -cfg.w_altitude_low_penalty
    elif altitude_m > cfg.max_safe_altitude_m:
        altitude_reward = -cfg.w_altitude_high_penalty
    else:
        altitude_reward = cfg.w_altitude_safe
    parts["altitude_reward"] = altitude_reward

    image_motion = float((img_vx ** 2 + img_vy ** 2) ** 0.5)
    smooth_follow_score = 1.0 - min(image_motion / cfg.max_img_motion, 1.0)
    smooth_follow_reward = cfg.w_smooth_follow * smooth_follow_score
    parts["smooth_follow_reward"] = smooth_follow_reward
    parts["smooth_follow_score"] = smooth_follow_score

    control_magnitude = float(np.linalg.norm(action))
    control_penalty = cfg.w_control * control_magnitude
    parts["control_penalty"] = -control_penalty

    if prev_action is not None:
        prev_action = np.asarray(prev_action, dtype=np.float32)
        action_delta = float(np.linalg.norm(action - prev_action))
    else:
        action_delta = 0.0

    action_delta_penalty = cfg.w_action_delta * action_delta
    parts["action_delta_penalty"] = -action_delta_penalty

    obstacle_penalty = cfg.w_obstacle * collision_risk_score
    parts["obstacle_penalty"] = -obstacle_penalty

    safety_intervention_penalty = cfg.w_safety_intervention if safety_intervention else 0.0
    parts["safety_intervention_penalty"] = -safety_intervention_penalty

    collision_reward = -cfg.collision_penalty if collision_detected else 0.0
    parts["collision_penalty"] = collision_reward

    reward = (
        center_reward
        + distance_reward
        + visibility_reward
        + altitude_reward
        + smooth_follow_reward
        - control_penalty
        - action_delta_penalty
        - obstacle_penalty
        - safety_intervention_penalty
        + collision_reward
    )

    parts["total_reward"] = float(reward)
    return float(reward), parts
