"""
Safety filter for UAV RL actions — state-aware LiDAR architecture.

Action convention inside the environment:
    raw_action = [vx, vy, vz, yaw_rate] normalized to [-1, 1]

AirSim NED convention after scaling:
    vx > 0  => forward
    vy > 0  => right
    vz > 0  => down
    vz < 0  => up

Safety is NOT disabled during landing.  Instead, the interpretation of the
same LiDAR measurements is phase-aware:

    TAKEOFF:
        Down/ground proximity is expected and forces climb.
        Horizontal motion is optionally held until clear altitude.
        Horizontal LiDAR termination is not used while the drone is still
        leaving the ground.

    CHASE / FLIGHT:
        All LiDAR sectors are active.  Horizontal obstacles slow/block motion;
        down proximity blocks descent and can force climb.

    LANDING / DESCENT:
        Horizontal sectors remain active.
        Down proximity is expected near touchdown, so it limits descent instead
        of forcing emergency climb.
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

    # Positive magnitude. The filter applies it as negative vz action to climb.
    emergency_up_cmd: float = 0.8

    # Phase-aware behavior.
    # In TAKEOFF, the ground is expected below the drone. We still force climb,
    # but we do not let ground/near-body LiDAR points create a fake horizontal
    # emergency before the drone is airborne.
    takeoff_zero_horizontal_motion: bool = True
    # For chase training we usually start already airborne. If the simulator/LiDAR
    # reports near-ground points, do not climb forever unless explicitly enabled.
    takeoff_force_climb: bool = False
    landing_allow_down_proximity: bool = True
    landing_max_descent_action_near_ground: float = 0.18

    enabled: bool = True


def safety_filter(
    raw_action: np.ndarray,
    obstacle_state: Dict[str, float],
    drone_state: Dict[str, float],
    config: SafetyConfig | None = None,
    safety_phase: str = "CHASE",
) -> Tuple[np.ndarray, Dict[str, object]]:
    cfg = config or SafetyConfig()

    raw_action = np.asarray(raw_action, dtype=np.float32)
    if raw_action.shape != (4,):
        raise ValueError("raw_action must have shape (4,): [vx, vy, vz, yaw_rate].")

    phase = str(safety_phase or "CHASE").upper().strip()

    if not cfg.enabled:
        safe_action = np.clip(raw_action, -1.0, 1.0)
        return safe_action, {
            "safety_enabled": False,
            "safety_intervention": False,
            "safety_reasons": [],
            "safety_phase": phase,
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

    min_dist = float(
        obstacle_state.get(
            "min_obstacle_dist_m",
            min(front, front_left, front_right, left, right, back),
        )
    )

    reasons = []

    takeoff_phase = phase in {"TAKEOFF", "GROUND_TAKEOFF"}
    landing_phase = phase in {"LANDING", "DESCENT", "LANDING_DESCENT", "ALIGN_ABOVE_TARGET"}
    chase_phase = not takeoff_phase and not landing_phase

    # ------------------------------------------------------------------
    # TAKEOFF phase
    # ------------------------------------------------------------------
    if takeoff_phase:
        if bool(cfg.takeoff_zero_horizontal_motion):
            if abs(vx_cmd) > 1e-6 or abs(vy_cmd) > 1e-6:
                reasons.append("takeoff_hold_horizontal_until_clear")
            vx_cmd = 0.0
            vy_cmd = 0.0

        # Ground below is expected during takeoff.
        # For chase training, do NOT climb forever just because LiDAR sees the
        # ground/body below. We only block unsafe descent by default. If a real
        # takeoff curriculum is desired, enable takeoff_force_climb in config.
        # Positive vz means descent; negative vz means climb.
        if down < cfg.warning_distance_m:
            if bool(cfg.takeoff_force_climb):
                vz_cmd = min(vz_cmd, -abs(cfg.emergency_up_cmd))
                reasons.append("takeoff_ground_below_force_up")
            elif vz_cmd > 0.0:
                vz_cmd = 0.0
                reasons.append("takeoff_ground_below_block_descent")

        safe_action = np.asarray([vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd], dtype=np.float32)
        safe_action = np.clip(safe_action, -1.0, 1.0)
        safety_intervention = bool(np.linalg.norm(safe_action - raw_action) > 1e-5)
        return safe_action, _make_info(
            enabled=True,
            intervention=safety_intervention,
            reasons=reasons,
            raw_action=raw_action,
            safe_action=safe_action,
            min_dist=min_dist,
            front=front,
            down=down,
            phase=phase,
        )

    # ------------------------------------------------------------------
    # CHASE / FLIGHT / LANDING horizontal safety
    # ------------------------------------------------------------------
    # Horizontal safety always stays active outside TAKEOFF, including landing.
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

    # ------------------------------------------------------------------
    # Down / ground safety
    # ------------------------------------------------------------------
    if landing_phase and bool(cfg.landing_allow_down_proximity):
        # In landing, down proximity is expected. We keep it safe by limiting
        # descent speed near the surface, but we do not force an upward escape
        # unless the user leaves landing mode or a real collision happens.
        if down < cfg.warning_distance_m and vz_cmd > 0:
            max_desc = float(np.clip(cfg.landing_max_descent_action_near_ground, 0.0, 1.0))
            vz_cmd = min(vz_cmd, max_desc)
            reasons.append("landing_down_proximity_limit_descent")
    else:
        if down < cfg.warning_distance_m and vz_cmd > 0:
            vz_cmd *= _distance_block_scale(down, cfg)
            reasons.append("down_obstacle_warning_block_descent")

        if down < cfg.emergency_distance_m and vz_cmd > 0:
            # In CHASE, down proximity should prevent descending into the ground,
            # but it must not create an endless climb that loses the target.
            vz_cmd = 0.0
            reasons.append("down_obstacle_emergency_block_descent")

    safe_action = np.asarray([vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd], dtype=np.float32)
    safe_action = np.clip(safe_action, -1.0, 1.0)

    safety_intervention = bool(np.linalg.norm(safe_action - raw_action) > 1e-5)

    return safe_action, _make_info(
        enabled=True,
        intervention=safety_intervention,
        reasons=reasons,
        raw_action=raw_action,
        safe_action=safe_action,
        min_dist=min_dist,
        front=front,
        down=down,
        phase=phase,
    )


def _make_info(
    enabled: bool,
    intervention: bool,
    reasons: list[str],
    raw_action: np.ndarray,
    safe_action: np.ndarray,
    min_dist: float,
    front: float,
    down: float,
    phase: str,
) -> Dict[str, object]:
    return {
        "safety_enabled": bool(enabled),
        "safety_intervention": bool(intervention),
        "safety_reasons": sorted(set(reasons)),
        "safety_phase": str(phase),
        "min_obstacle_dist_m": float(min_dist),
        "front_dist_m": float(front),
        "down_dist_m": float(down),
        "raw_action": raw_action.copy(),
        "safe_action": safe_action.copy(),
    }


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
