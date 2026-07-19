"""
validate_action_axis_response.py

Deterministic diagnostic for the UAV tracking setup.

Purpose:
    This file does NOT train and does NOT use the PPO policy.
    It tests the real control/action mapping by sending fixed, deterministic
    action pulses and measuring how the target image error changes.

Why this is useful:
    If the tracker is stable but the trained policy does not center the target,
    we need to know whether:
        1. the action axes/signs are correct, and the policy simply learned badly, or
        2. the environment/action mapping is not giving the policy a clean way to center.

Run from inside RL_training/:
    python validate_action_axis_response.py

Expected workflow:
    1. Start Unreal/AirSim.
    2. Run this script.
    3. Click the target once when the Tracker Debug window appears.
    4. Watch the printed table.

Interpretation:
    For each pulse, compare err_before -> err_after.

    err_x:
        positive means target is RIGHT of image center.
        negative means target is LEFT of image center.

    err_y:
        positive means target is LOW in the image.
        negative means target is HIGH in the image.

    A useful action should reduce center_error_norm.
"""

from __future__ import annotations

import importlib
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from drone_env import DroneEnv
from weights_config import EnvConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TASK_MODULE = "config.tracking_config"

# Number of env.step calls for each deterministic pulse.
PULSE_STEPS = 18

# Number of neutral stabilization steps between pulses.
NEUTRAL_STEPS = 8

# Action magnitude in normalized action-space [-1, +1].
# Keep this moderate so the diagnostic is safe and readable.
PULSE_MAG = 0.35

# Pause between printed pulse blocks.
PAUSE_BETWEEN_PULSES_SEC = 0.25

# If True, prints every internal step. Usually False is cleaner.
PRINT_EVERY_STEP = False


@dataclass
class ObsMetrics:
    err_x: float
    err_y: float
    center_error: float
    centered_score: float
    distance_proxy: float
    altitude_m: float
    mode: str


def load_task_config():
    module = importlib.import_module(TASK_MODULE)
    return module.TASK_CONFIG


def apply_task_config_to_env_config(cfg: EnvConfig, task) -> EnvConfig:
    """
    Local copy of Run_train.apply_task_config_to_env_config so this diagnostic
    is independent and can run without importing training code/callbacks.
    """
    cfg.training_task = task.name
    cfg.training_task_description = task.description

    cfg.freeze_vz = task.freeze_vz
    cfg.altitude_hold_enabled = task.altitude_hold_enabled
    cfg.enable_forward_motion = task.enable_forward_motion
    cfg.enable_yaw_control = task.enable_yaw_control
    cfg.enable_z_control = task.enable_z_control

    cfg.desired_distance_proxy = task.desired_distance_proxy
    cfg.distance_tolerance = task.distance_tolerance
    cfg.min_target_distance_proxy = task.min_target_distance_proxy
    cfg.block_forward_when_too_close = task.block_forward_when_too_close

    cfg.reset_takeoff_altitude_m = task.reset_takeoff_altitude_m
    cfg.altitude_hold_target_m = task.altitude_hold_target_m
    cfg.min_safe_altitude_m = task.min_safe_altitude_m
    cfg.min_termination_altitude_m = task.min_termination_altitude_m
    cfg.max_termination_altitude_m = task.max_termination_altitude_m

    cfg.max_episode_steps = task.max_episode_steps
    cfg.focus_fail_sec = task.focus_fail_sec

    cfg.w_center = task.w_center
    cfg.w_distance = task.w_distance
    cfg.w_visibility = task.w_visibility
    cfg.w_lost_target = task.w_lost_target
    cfg.w_altitude_safe = task.w_altitude_safe
    cfg.w_altitude_low_penalty = task.w_altitude_low_penalty
    cfg.w_altitude_high_penalty = task.w_altitude_high_penalty
    cfg.w_smooth_follow = task.w_smooth_follow
    cfg.w_control = task.w_control
    cfg.w_action_delta = task.w_action_delta
    cfg.w_obstacle = task.w_obstacle
    cfg.w_safety_intervention = task.w_safety_intervention

    if hasattr(cfg, "w_slow_or_stuck"):
        cfg.w_slow_or_stuck = task.w_slow_or_stuck

    cfg.penalty_collision = task.penalty_collision

    if hasattr(cfg, "penalty_timeout"):
        cfg.penalty_timeout = task.penalty_timeout

    return cfg


def metrics_from_info(info: Dict) -> ObsMetrics:
    obs = info.get("obs_dict", {}) or {}

    err_x = float(obs.get("err_x", 0.0))
    err_y = float(obs.get("err_y", 0.0))
    center_error = float(math.sqrt(err_x * err_x + err_y * err_y))
    centered_score = float(obs.get("centered_score", 0.0))
    distance_proxy = float(obs.get("distance_proxy_norm", 1.0))
    altitude_m = float(info.get("alt_agl_m", 0.0))
    mode = str(info.get("tracking_mode", "UNKNOWN"))

    return ObsMetrics(
        err_x=err_x,
        err_y=err_y,
        center_error=center_error,
        centered_score=centered_score,
        distance_proxy=distance_proxy,
        altitude_m=altitude_m,
        mode=mode,
    )


def metrics_from_reset_info(info: Dict) -> ObsMetrics:
    obs = info.get("obs_dict", {}) or {}

    err_x = float(obs.get("err_x", 0.0))
    err_y = float(obs.get("err_y", 0.0))
    center_error = float(math.sqrt(err_x * err_x + err_y * err_y))
    centered_score = float(obs.get("centered_score", 0.0))
    distance_proxy = float(obs.get("distance_proxy_norm", 1.0))
    altitude_m = float(obs.get("altitude_norm", 0.0))  # normalized, only for reset display
    mode = "RESET"

    return ObsMetrics(
        err_x=err_x,
        err_y=err_y,
        center_error=center_error,
        centered_score=centered_score,
        distance_proxy=distance_proxy,
        altitude_m=altitude_m,
        mode=mode,
    )


def fmt_metrics(m: ObsMetrics) -> str:
    return (
        f"err=({m.err_x:+.3f},{m.err_y:+.3f}) "
        f"ce={m.center_error:.3f} "
        f"score={m.centered_score:.3f} "
        f"dist={m.distance_proxy:.3f} "
        f"alt={m.altitude_m:.2f} "
        f"mode={m.mode}"
    )


def run_steps(env: DroneEnv, action: np.ndarray, steps: int, label: str) -> Tuple[ObsMetrics, bool, Dict]:
    last_info: Dict = {}
    done = False

    for i in range(int(steps)):
        _, reward, done, truncated, info = env.step(action)
        last_info = info
        m = metrics_from_info(info)

        if PRINT_EVERY_STEP:
            print(
                f"    [{label} {i + 1:02d}/{steps:02d}] "
                f"{fmt_metrics(m)} "
                f"r={reward:+.2f} "
                f"safe={np.asarray(info.get('safe_action', action), dtype=float)}"
            )

        if done or truncated:
            print(
                f"    [DONE] label={label} "
                f"reason={info.get('termination_reason', '')} "
                f"at internal step {i + 1}"
            )
            return m, True, info

    return metrics_from_info(last_info), False, last_info


def main() -> None:
    task = load_task_config()
    cfg = apply_task_config_to_env_config(EnvConfig(), task)

    # Keep the visual window ON because we are validating perception/control.
    cfg.show_cv_window = True

    # Keep this diagnostic safe.
    cfg.max_episode_steps = 2000
    cfg.print_obstacle_debug = False
    cfg.print_ep_summary = True

    print("=" * 100)
    print("[AXIS TEST] Deterministic action-axis response diagnostic")
    print(f"[AXIS TEST] Task module: {TASK_MODULE}")
    print(f"[AXIS TEST] freeze_vz={cfg.freeze_vz} altitude_hold={cfg.altitude_hold_enabled}")
    print(f"[AXIS TEST] action scales: vx={cfg.vx_scale} vy={cfg.vy_scale} vz={cfg.vz_scale} yaw_dps={cfg.yaw_rate_scale_dps}")
    print(f"[AXIS TEST] PULSE_STEPS={PULSE_STEPS} NEUTRAL_STEPS={NEUTRAL_STEPS} PULSE_MAG={PULSE_MAG}")
    print("=" * 100)

    env = DroneEnv(cfg=cfg)

    try:
        _, reset_info = env.reset()
        current = metrics_from_reset_info(reset_info)
        print(f"[RESET] {fmt_metrics(current)}")
        print()

        zero = np.zeros(4, dtype=np.float32)

        pulses: List[Tuple[str, np.ndarray]] = [
            ("VX + forward", np.array([+PULSE_MAG, 0.0, 0.0, 0.0], dtype=np.float32)),
            ("VX - backward", np.array([-PULSE_MAG, 0.0, 0.0, 0.0], dtype=np.float32)),
            ("VY + right", np.array([0.0, +PULSE_MAG, 0.0, 0.0], dtype=np.float32)),
            ("VY - left", np.array([0.0, -PULSE_MAG, 0.0, 0.0], dtype=np.float32)),
            ("YAW + clockwise?", np.array([0.0, 0.0, 0.0, +PULSE_MAG], dtype=np.float32)),
            ("YAW - counter?", np.array([0.0, 0.0, 0.0, -PULSE_MAG], dtype=np.float32)),
        ]

        print("-" * 100)
        print(
            f"{'Pulse':<18} | {'before ce':>9} | {'after ce':>8} | {'delta ce':>8} | "
            f"{'before err':>18} | {'after err':>18} | Result"
        )
        print("-" * 100)

        for label, action in pulses:
            # Stabilize briefly before each pulse.
            before, done, _ = run_steps(env, zero, NEUTRAL_STEPS, "neutral")
            if done:
                break

            after, done, _ = run_steps(env, action, PULSE_STEPS, label)

            delta_ce = after.center_error - before.center_error
            result = "IMPROVED" if delta_ce < -0.03 else ("WORSE" if delta_ce > +0.03 else "NEUTRAL")

            print(
                f"{label:<18} | "
                f"{before.center_error:>9.3f} | {after.center_error:>8.3f} | {delta_ce:>+8.3f} | "
                f"({before.err_x:+.3f},{before.err_y:+.3f}) | "
                f"({after.err_x:+.3f},{after.err_y:+.3f}) | "
                f"{result}"
            )

            if done:
                break

            time.sleep(PAUSE_BETWEEN_PULSES_SEC)

        print("-" * 100)
        print("[AXIS TEST] Done.")
        print()
        print("How to use the result:")
        print("  - If one pulse consistently reduces ce, that axis/sign can center the target.")
        print("  - If all pulses are NEUTRAL/WORSE while MATCH is stable, action mapping or camera geometry is weak for centering.")
        print("  - For landing, we need a reliable way to reduce err_x and err_y before descent is rewarded.")

    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
