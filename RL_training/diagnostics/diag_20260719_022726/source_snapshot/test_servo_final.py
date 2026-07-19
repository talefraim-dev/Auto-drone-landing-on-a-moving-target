import time
import numpy as np

from config.servo_config import ServoTestConfig
from drone_env import DroneEnv


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _get_obs_dict(info):
    d = info.get("obs_dict", {})
    return d if isinstance(d, dict) else {}


def _compute_servo_action(info, cfg: ServoTestConfig) -> tuple[np.ndarray, dict]:
    obs = _get_obs_dict(info)

    has_target = float(obs.get("has_target", 0.0))
    err_x = float(obs.get("err_x", 0.0))
    err_y = float(obs.get("err_y", 0.0))
    distance_proxy = float(obs.get("distance_proxy_norm", 1.0))
    alt = float(info.get("alt_agl_m", obs.get("altitude_m", cfg.altitude_hold_target_m)))

    action = np.zeros(4, dtype=np.float32)
    debug = {}

    if has_target <= 0.0:
        debug["servo"] = "NO_TARGET"
        return action, debug

    yaw = _clip(cfg.yaw_gain * err_x, -cfg.max_yaw, cfg.max_yaw)
    action[3] = yaw

    if cfg.forward_enabled:
        dist_err = float(distance_proxy - cfg.target_distance_proxy)
        if abs(dist_err) > cfg.distance_deadband:
            # If distance_proxy is larger than target, target is too far/small => move forward.
            action[0] = _clip(cfg.forward_gain * dist_err, -cfg.max_forward, cfg.max_forward)

    if cfg.enable_z_servo:
        z_cmd = _clip(cfg.z_gain * err_y, -cfg.max_z, cfg.max_z)
        action[2] = z_cmd
    elif cfg.enable_altitude_hold:
        # Normalized action. DroneEnv converts freeze_vz/alt-hold itself during training,
        # but this servo test still sends action[2]. Here we keep it very small.
        # Positive z action means down; negative means up after scaling.
        alt_error = alt - cfg.altitude_hold_target_m
        action[2] = _clip(cfg.altitude_hold_kp * alt_error, -cfg.altitude_hold_max_vz_mps, cfg.altitude_hold_max_vz_mps)

    debug.update(
        {
            "servo": "SERVO",
            "err_x": err_x,
            "err_y": err_y,
            "distance_proxy": distance_proxy,
            "alt": alt,
        }
    )
    return action, debug


def run_test():
    cfg = ServoTestConfig()
    env = DroneEnv()

    obs, info = env.reset()
    print("Initial obs shape:", obs.shape)
    print("[TEST] Final visual-servo smoke test started.")
    print(f"[TEST] config={cfg}")

    total_reward = 0.0
    last_info = info

    for step in range(int(cfg.max_steps)):
        if step < int(cfg.warmup_steps):
            action = np.zeros(4, dtype=np.float32)
            phase = "WARMUP_HOLD"
            debug = {"servo": "RESET"}
        else:
            action, debug = _compute_servo_action(last_info, cfg)
            phase = "VISUAL_SERVO"

        obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        last_info = info

        if step % 5 == 0 or done or truncated:
            obs_dict = _get_obs_dict(info)
            print(
                f"step={step:04d} phase={phase} reward={float(reward):+.3f} total={total_reward:+.2f} "
                f"mode={info.get('tracking_mode')} raw={info.get('raw_tracker_mode')} "
                f"center_err=({obs_dict.get('err_x', 0.0):+.3f},{obs_dict.get('err_y', 0.0):+.3f}) "
                f"dist={obs_dict.get('distance_proxy_norm', 0.0):.3f} alt={info.get('alt_agl_m', 0.0):.2f} "
                f"action=({action[0]:+.3f},{action[1]:+.3f},{action[2]:+.3f},{action[3]:+.3f}) "
                f"servo={debug.get('servo')} match={info.get('match_pct', 0.0):.1f}% "
                f"pred={info.get('pred_pct', 0.0):.1f}% reason={info.get('termination_reason', '')}"
            )

        if done or truncated:
            print("Episode ended early.")
            print("reason:", info.get("termination_reason", ""))
            break

        time.sleep(0.02)

    print("[TEST] Final visual-servo smoke test finished.")


if __name__ == "__main__":
    run_test()
