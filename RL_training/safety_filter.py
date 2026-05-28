"""
Safety filter for UAV RL actions — v37.

Pipeline:
    obs -> RL policy -> raw_action -> safety_filter -> safe_action -> AirSim

Important design decision:
- min_obstacle_dist_m is horizontal only.
- down_dist_m is used only for vertical descent safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class SafetyConfig:
    safe_distance_m: float = 5.0
    warning_distance_m: float = 3.0
    emergency_distance_m: float = 1.0

    min_speed_scale_near_obstacle: float = 0.2
    steer_strength: float = 0.35
    emergency_up_cmd: float = 0.2
    enabled: bool = True


def safety_filter(
    raw_action: np.ndarray,
    obstacle_state: Dict[str, float],
    drone_state: Dict[str, float],
    config: SafetyConfig | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    cfg = config or SafetyConfig()

    raw_action = np.asarray(raw_action, dtype=np.float32)
    if raw_action.shape != (4,):
        raise ValueError("raw_action must have shape (4,): [vx, vy, vz, yaw_rate].")

    if not cfg.enabled:
        safe_action = np.clip(raw_action, -1.0, 1.0)
        return safe_action, {
            "safety_enabled": False,
            "safety_intervention": False,
            "safety_reasons": [],
            "raw_action": raw_action.copy(),
            "safe_action": safe_action.copy(),
        }

    vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd = raw_action.astype(float).copy()

    front = float(obstacle_state.get("front_dist_m", cfg.safe_distance_m))
    front_left = float(obstacle_state.get("front_left_dist_m", cfg.safe_distance_m))
    front_right = float(obstacle_state.get("front_right_dist_m", cfg.safe_distance_m))
    left = float(obstacle_state.get("left_dist_m", cfg.safe_distance_m))
    right = float(obstacle_state.get("right_dist_m", cfg.safe_distance_m))
    back = float(obstacle_state.get("back_dist_m", cfg.safe_distance_m))
    down = float(obstacle_state.get("down_dist_m", cfg.safe_distance_m))

    # Horizontal min only. Do NOT include down here.
    min_dist = float(obstacle_state.get(
        "min_obstacle_dist_m",
        min(front, front_left, front_right, left, right, back),
    ))

    reasons = []

    speed_scale = _clip(
        min_dist / cfg.safe_distance_m,
        cfg.min_speed_scale_near_obstacle,
        1.0,
    )

    old_vx, old_vy = vx_cmd, vy_cmd
    vx_cmd *= speed_scale
    vy_cmd *= speed_scale

    if abs(vx_cmd - old_vx) > 1e-6 or abs(vy_cmd - old_vy) > 1e-6:
        reasons.append("speed_scaled_near_horizontal_obstacle")

    if front < cfg.warning_distance_m and vx_cmd > 0:
        vx_cmd *= _distance_block_scale(front, cfg)
        reasons.append("front_obstacle_warning")

    if front < cfg.emergency_distance_m and vx_cmd > 0:
        vx_cmd = 0.0
        reasons.append("front_obstacle_emergency")

    if back < cfg.warning_distance_m and vx_cmd < 0:
        vx_cmd *= _distance_block_scale(back, cfg)
        reasons.append("back_obstacle_warning")

    if back < cfg.emergency_distance_m and vx_cmd < 0:
        vx_cmd = 0.0
        reasons.append("back_obstacle_emergency")

    if left < cfg.warning_distance_m and vy_cmd < 0:
        vy_cmd *= _distance_block_scale(left, cfg)
        reasons.append("left_obstacle_warning")

    if left < cfg.emergency_distance_m and vy_cmd < 0:
        vy_cmd = 0.0
        reasons.append("left_obstacle_emergency")

    if right < cfg.warning_distance_m and vy_cmd > 0:
        vy_cmd *= _distance_block_scale(right, cfg)
        reasons.append("right_obstacle_warning")

    if right < cfg.emergency_distance_m and vy_cmd > 0:
        vy_cmd = 0.0
        reasons.append("right_obstacle_emergency")

    if front < cfg.warning_distance_m:
        if left > right and left > cfg.warning_distance_m:
            vy_cmd -= cfg.steer_strength
            reasons.append("steer_left_around_front_obstacle")
        elif right > left and right > cfg.warning_distance_m:
            vy_cmd += cfg.steer_strength
            reasons.append("steer_right_around_front_obstacle")

    # Vertical / ground safety.
    # vz_cmd < 0 means descend.
    if down < cfg.warning_distance_m and vz_cmd < 0:
        vz_cmd *= _distance_block_scale(down, cfg)
        reasons.append("down_obstacle_warning")

    if down < cfg.emergency_distance_m:
        vz_cmd = max(vz_cmd, cfg.emergency_up_cmd)
        reasons.append("down_obstacle_emergency_force_up")

    safe_action = np.asarray([vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd], dtype=np.float32)
    safe_action = np.clip(safe_action, -1.0, 1.0)

    safety_intervention = bool(np.linalg.norm(safe_action - raw_action) > 1e-5)

    info = {
        "safety_enabled": True,
        "safety_intervention": safety_intervention,
        "safety_reasons": sorted(set(reasons)),
        "raw_action": raw_action.copy(),
        "safe_action": safe_action.copy(),
        "min_obstacle_dist_m": min_dist,
        "front_dist_m": front,
        "down_dist_m": down,
    }

    return safe_action, info


def compute_collision_risk(min_obstacle_dist_m: float, safe_distance_m: float) -> float:
    if safe_distance_m <= 0:
        raise ValueError("safe_distance_m must be positive.")

    risk = 1.0 - min(max(min_obstacle_dist_m, 0.0) / safe_distance_m, 1.0)
    return _clip(risk, 0.0, 1.0)


def _distance_block_scale(distance_m: float, cfg: SafetyConfig) -> float:
    denominator = cfg.warning_distance_m - cfg.emergency_distance_m
    if denominator <= 0:
        raise ValueError("warning_distance_m must be greater than emergency_distance_m.")

    return _clip(
        (distance_m - cfg.emergency_distance_m) / denominator,
        0.0,
        1.0,
    )


def _clip(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, float(value)))
