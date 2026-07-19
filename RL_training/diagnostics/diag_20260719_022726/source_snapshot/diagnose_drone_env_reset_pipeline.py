"""
diagnose_drone_env_reset_pipeline.py

Purpose:
    Diagnose the specific mismatch we are seeing now:

        validate_lidar_sensor_v2.py says:
            alt_pose ~= 4.7m
            LiDAR processed sectors are OK

        But validate_tracking_policy_v2.py / DroneEnv says:
            alt=0.00m
            altitude_too_low / non_match_too_long

This script does NOT train.
This script does NOT run PPO.
This script does NOT require clicking the target by default.

It imports DroneEnv only to test the exact reset/altitude helper methods used by
the environment, then prints:
    - AirSim pose/state altitude
    - DroneEnv._get_drone_state() altitude
    - LiDAR processed obstacle dict
    - key EnvConfig values

Run from inside RL_training:
    python diagnose_drone_env_reset_pipeline.py

If you want to also run a full env.reset(), set DO_FULL_ENV_RESET = True below.
That may require clicking the target in the initial OpenCV window.
"""

from __future__ import annotations

import importlib
import time
from typing import Any

import numpy as np

try:
    import cosysairsim as airsim
except Exception:
    import airsim

from drone_env import DroneEnv
from weights_config import EnvConfig
from Run_train import apply_task_config_to_env_config


TASK_CONFIG_MODULE = "config.tracking_config"
DO_FULL_ENV_RESET = False


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        if np.isfinite(y):
            return y
    except Exception:
        pass
    return default


def build_cfg() -> EnvConfig:
    task_module = importlib.import_module(TASK_CONFIG_MODULE)
    task = task_module.TASK_CONFIG

    cfg = apply_task_config_to_env_config(EnvConfig(), task)

    # We are diagnosing reset/altitude, not visual FPS.
    cfg.show_cv_window = False
    cfg.print_reset = True
    cfg.print_obstacle_debug = True
    cfg.max_episode_steps = 50

    # Force the expected chase-training reset architecture.
    cfg.reset_start_airborne_with_pose = True
    cfg.reset_takeoff_altitude_m = 5.0
    cfg.reset_verify_altitude_min_m = 2.0
    cfg.use_lidar_down_as_altitude = False

    return cfg


def print_cfg(cfg: EnvConfig) -> None:
    print("=" * 110)
    print("[CFG]")
    keys = [
        "vehicle_name",
        "use_self_state_obs",
        "reset_start_airborne_with_pose",
        "reset_takeoff_altitude_m",
        "reset_verify_altitude_min_m",
        "force_chase_after_airborne_reset",
        "use_lidar_down_as_altitude",
        "obstacle_sensor_source",
        "lidar_sensor_name",
        "lidar_self_ignore_radius_m",
        "lidar_horizontal_abs_z_max_m",
        "lidar_down_xy_radius_m",
        "front_camera_name",
        "downward_camera_name",
    ]
    for k in keys:
        print(f"{k:36s} = {getattr(cfg, k, '<MISSING>')}")
    print("=" * 110)


def pose_alt_from_client(client, vehicle_name: str) -> None:
    pose = client.simGetVehiclePose(vehicle_name=vehicle_name)
    ms = client.getMultirotorState(vehicle_name=vehicle_name)
    p = pose.position
    sp = ms.kinematics_estimated.position

    print(
        "[AIRSIM POSE] "
        f"pose=(x={p.x_val:+.3f}, y={p.y_val:+.3f}, z={p.z_val:+.3f}) "
        f"pose_alt_proxy={max(0.0, -float(p.z_val)):.3f}m"
    )
    print(
        "[AIRSIM STATE] "
        f"state=(x={sp.x_val:+.3f}, y={sp.y_val:+.3f}, z={sp.z_val:+.3f}) "
        f"state_alt_proxy={max(0.0, -float(sp.z_val)):.3f}m "
        f"landed_state={getattr(ms, 'landed_state', 'NA')}"
    )


def env_alt_from_methods(env: DroneEnv, label: str) -> None:
    print()
    print("-" * 110)
    print(f"[{label}]")

    client = env.client
    vehicle_name = env.cfg.vehicle_name

    pose_alt_from_client(client, vehicle_name)

    alt_direct = env._get_alt_agl_m()
    ds = env._get_drone_state()
    obstacles = env._get_obstacle_state_m(altitude_m=ds.altitude_m)

    ds_after = env._apply_down_distance_as_altitude_if_valid(ds, obstacles)

    print(f"[ENV _get_alt_agl_m] {alt_direct}")
    print(f"[ENV _get_drone_state] altitude_m={ds.altitude_m:.3f} vx={ds.vx_mps:+.3f} vy={ds.vy_mps:+.3f} vz={ds.vz_mps:+.3f}")
    print(f"[ENV after apply_down] altitude_m={ds_after.altitude_m:.3f}")

    wanted = [
        "obstacle_source",
        "lidar_valid",
        "lidar_point_count",
        "front_dist_m",
        "front_left_dist_m",
        "front_right_dist_m",
        "left_dist_m",
        "right_dist_m",
        "back_dist_m",
        "down_dist_m",
        "min_obstacle_dist_m",
    ]
    print("[ENV obstacle_dict]")
    for k in wanted:
        print(f"  {k:24s}: {obstacles.get(k, '<missing>')}")

    phase = env._compute_safety_phase(
        drone_state=ds_after,
        obstacle_dict=obstacles,
        tracking_mode="MATCH",
    )
    print(f"[ENV safety_phase if MATCH] {phase}")
    print("-" * 110)


def main() -> None:
    cfg = build_cfg()
    print_cfg(cfg)

    print("[INIT] Creating DroneEnv...")
    env = DroneEnv(cfg=cfg)

    try:
        print("[STEP A] Direct AirSim reset + arm")
        env.client.reset()
        time.sleep(1.0)
        env.client.enableApiControl(True, vehicle_name=cfg.vehicle_name)
        env.client.armDisarm(True, vehicle_name=cfg.vehicle_name)
        time.sleep(0.5)
        env_alt_from_methods(env, "AFTER direct client.reset + arm")

        print()
        print("[STEP B] Calling DroneEnv._force_airborne_start_pose() directly")
        env._force_airborne_start_pose()
        time.sleep(0.5)
        env_alt_from_methods(env, "AFTER env._force_airborne_start_pose")

        if DO_FULL_ENV_RESET:
            print()
            print("[STEP C] Calling full env.reset()")
            print("        This may require clicking the target if not initialized.")
            obs, info = env.reset()
            env_alt_from_methods(env, "AFTER full env.reset")
            print("[FULL RESET INFO]")
            print(f"  info keys: {sorted(list(info.keys()))}")
            print(f"  alt_agl_m: {info.get('alt_agl_m')}")
            print(f"  safety_state: {info.get('safety_state')}")
            print(f"  safety_phase: {info.get('safety_phase')}")

        print()
        print("=" * 110)
        print("[EXPECTED]")
        print("After env._force_airborne_start_pose:")
        print("  AIRSIM pose/state altitude should be around 4.5-5.0m")
        print("  ENV _get_drone_state altitude_m should also be around 4.5-5.0m")
        print("  ENV after apply_down altitude_m should remain around 4.5-5.0m")
        print("  obstacle HMIN should NOT be 0.12m")
        print("  safety_phase should be CHASE")
        print("=" * 110)

    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
