import time
import numpy as np

from drone_env import DroneEnv


def print_obs_summary(obs):
    names = {
        0: "rel_x",
        1: "rel_y",
        2: "area_m11",
        3: "quality",
        6: "focus_norm",
        7: "match_state",
        8: "vbx",
        9: "vby",
        10: "vbz",
        18: "alt_norm",
        29: "range_to_target",
        32: "range_error",
        34: "alt_error",
        35: "bearing_error",
    }

    parts = []
    for idx, name in names.items():
        parts.append(f"{name}={obs[idx]:+.3f}")
    print("[OBS]", " | ".join(parts))


def main():
    print("[INFO] Creating DroneEnv...")
    env = DroneEnv()

    print("[INFO] Resetting environment...")
    print("[INFO] If this is the first run, click the target once in the OpenCV window.")
    obs, info = env.reset()

    print("[OK] Reset completed.")
    print("[INFO] obs shape:", obs.shape)
    print("[INFO] obs min/max:", float(np.min(obs)), float(np.max(obs)))
    print_obs_summary(obs)

    # Manual action pattern:
    # first hover, then small forward, then small yaw corrections
    actions = []

    for _ in range(10):
        actions.append(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    for _ in range(30):
        actions.append(np.array([0.15, 0.0, 0.0, 0.0], dtype=np.float32))

    for _ in range(20):
        actions.append(np.array([0.0, 0.0, 0.0, 0.15], dtype=np.float32))

    for _ in range(20):
        actions.append(np.array([0.0, 0.0, 0.0, -0.15], dtype=np.float32))

    for _ in range(20):
        actions.append(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    total_reward = 0.0

    for i, action in enumerate(actions, start=1):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        mode = getattr(env.tracker, "last_mode", "?")

        print(
            f"\n[STEP {i:03d}] "
            f"mode={mode} "
            f"reward={reward:+.3f} "
            f"total={total_reward:+.3f} "
            f"done={terminated or truncated}"
        )

        print(
            "[INFO] "
            f"alt_agl_m={info.get('alt_agl_m')} | "
            f"lost_time_s={info.get('lost_time_s'):.2f} | "
            f"focus_streak_s={info.get('focus_streak_s'):.2f} | "
            f"match={info.get('match_pct'):.1f}% | "
            f"pred={info.get('pred_pct'):.1f}% | "
            f"none={info.get('none_pct'):.1f}% | "
            f"reason={info.get('termination_reason')}"
        )

        print_obs_summary(obs)

        time.sleep(0.05)

        if terminated or truncated:
            print("[WARN] Episode ended early.")
            break

    print("\n[INFO] Closing env...")
    env.close()
    print("[DONE] city_env_long_debug finished.")


if __name__ == "__main__":
    main()