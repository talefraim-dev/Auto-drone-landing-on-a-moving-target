import numpy as np
import matplotlib.pyplot as plt

from weights_config import EnvConfig
from drone_env import DroneEnv


def safe_get_obs(info, key, default=np.nan):
    try:
        return float(info.get("obs_dict", {}).get(key, default))
    except Exception:
        return default


def safe_get_info(info, key, default=np.nan):
    try:
        value = info.get(key, default)
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def main():
    cfg = EnvConfig()

    # Keep visual tracker window enabled if you want to watch the drone.
    cfg.show_cv_window = True

    # Keep logs readable.
    cfg.print_ep_summary = True
    cfg.print_obstacle_debug = True
    cfg.obstacle_debug_every_n_steps = 30

    env = DroneEnv(cfg)

    obs, info = env.reset()
    print("Initial obs shape:", obs.shape)
    print("Initial altitude_norm:", safe_get_obs(info, "altitude_norm"))

    max_steps = 300

    history = {
        "step": [],
        "reward": [],
        "total_reward": [],

        "alt_agl_m": [],
        "down_dist_m": [],
        "min_obstacle_dist_m": [],
        "collision_risk_score": [],

        "centered_score": [],
        "distance_proxy_norm": [],
        "target_stability_score": [],
        "has_target": [],

        "safety_intervention": [],
        "safety_intervention_rate_pct": [],

        "center_reward": [],
        "distance_reward": [],
        "altitude_reward": [],
        "smooth_follow_reward": [],
        "obstacle_penalty": [],
        "control_penalty": [],
    }

    total_reward = 0.0
    final_info = {}

    for step in range(max_steps):
        # Random action for smoke test.
        # Later, this can be replaced by model.predict(obs).
        action = env.action_space.sample()

        obs, reward, done, truncated, info = env.step(action)
        final_info = info

        total_reward += float(reward)

        reward_parts = info.get("reward_parts", {})

        history["step"].append(step)
        history["reward"].append(float(reward))
        history["total_reward"].append(float(total_reward))

        history["alt_agl_m"].append(safe_get_info(info, "alt_agl_m"))
        history["down_dist_m"].append(safe_get_info(info, "down_dist_m"))
        history["min_obstacle_dist_m"].append(safe_get_info(info, "min_obstacle_dist_m"))
        history["collision_risk_score"].append(safe_get_obs(info, "collision_risk_score"))

        history["centered_score"].append(safe_get_obs(info, "centered_score"))
        history["distance_proxy_norm"].append(safe_get_obs(info, "distance_proxy_norm"))
        history["target_stability_score"].append(safe_get_obs(info, "target_stability_score"))
        history["has_target"].append(safe_get_obs(info, "has_target"))

        history["safety_intervention"].append(1.0 if info.get("safety_intervention", False) else 0.0)
        history["safety_intervention_rate_pct"].append(safe_get_info(info, "safety_intervention_rate_pct"))

        history["center_reward"].append(float(reward_parts.get("center_reward", np.nan)))
        history["distance_reward"].append(float(reward_parts.get("distance_reward", np.nan)))
        history["altitude_reward"].append(float(reward_parts.get("altitude_reward", np.nan)))
        history["smooth_follow_reward"].append(float(reward_parts.get("smooth_follow_reward", np.nan)))
        history["obstacle_penalty"].append(float(reward_parts.get("obstacle_penalty", np.nan)))
        history["control_penalty"].append(float(reward_parts.get("control_penalty", np.nan)))

        if step % 20 == 0:
            print(
                f"step={step:03d} "
                f"reward={reward:+.3f} "
                f"total={total_reward:+.2f} "
                f"done={done} "
                f"alt={info.get('alt_agl_m', None):.2f} "
                f"hmin={info.get('min_obstacle_dist_m', None):.2f} "
                f"down={info.get('down_dist_m', None):.2f} "
                f"center={safe_get_obs(info, 'centered_score'):.2f} "
                f"dist_proxy={safe_get_obs(info, 'distance_proxy_norm'):.2f} "
                f"safety={info.get('safety_intervention', False)} "
                f"reason={info.get('termination_reason', '')}"
            )

        if done or truncated:
            print("Episode ended early.")
            print("reason:", info.get("termination_reason", ""))
            break

    print("\n===== SMOKE TEST SUMMARY =====")
    print("steps:", len(history["step"]))
    print("total_reward:", total_reward)
    print("final_alt_agl_m:", final_info.get("alt_agl_m", None))
    print("final_min_obstacle_dist_m:", final_info.get("min_obstacle_dist_m", None))
    print("final_down_dist_m:", final_info.get("down_dist_m", None))
    print("final_safety_rate_pct:", final_info.get("safety_intervention_rate_pct", None))
    print("final_reason:", final_info.get("termination_reason", ""))

    plot_smoke_test(history)


def plot_smoke_test(history):
    steps = np.asarray(history["step"])

    if len(steps) == 0:
        print("No data to plot.")
        return

    # Figure 1: Reward overview
    plt.figure(figsize=(12, 6))
    plt.plot(steps, history["reward"], label="Reward per step")
    plt.plot(steps, history["total_reward"], label="Cumulative reward")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Smoke Test — Reward Behavior")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Figure 2: Target tracking quality
    plt.figure(figsize=(12, 6))
    plt.plot(steps, history["centered_score"], label="Centered score")
    plt.plot(steps, history["distance_proxy_norm"], label="Distance proxy norm")
    plt.plot(steps, history["target_stability_score"], label="Target stability score")
    plt.plot(steps, history["has_target"], label="Has target")
    plt.xlabel("Step")
    plt.ylabel("Score / Normalized value")
    plt.title("Smoke Test — Target Tracking Signals")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Figure 3: Altitude and obstacle distances
    plt.figure(figsize=(12, 6))
    plt.plot(steps, history["alt_agl_m"], label="Altitude AGL [m]")
    plt.plot(steps, history["down_dist_m"], label="Down distance [m]")
    plt.plot(steps, history["min_obstacle_dist_m"], label="Horizontal min obstacle [m]")
    plt.xlabel("Step")
    plt.ylabel("Meters")
    plt.title("Smoke Test — Altitude and Obstacle Distances")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Figure 4: Safety behavior
    plt.figure(figsize=(12, 6))
    plt.plot(steps, history["collision_risk_score"], label="Collision risk score")
    plt.plot(steps, history["safety_intervention"], label="Safety intervention flag")
    plt.plot(steps, np.asarray(history["safety_intervention_rate_pct"]) / 100.0, label="Safety intervention rate / 100")
    plt.xlabel("Step")
    plt.ylabel("Score / Flag")
    plt.title("Smoke Test — Safety Layer Behavior")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Figure 5: Reward components
    plt.figure(figsize=(12, 6))
    plt.plot(steps, history["center_reward"], label="Center reward")
    plt.plot(steps, history["distance_reward"], label="Distance reward")
    plt.plot(steps, history["altitude_reward"], label="Altitude reward")
    plt.plot(steps, history["smooth_follow_reward"], label="Smooth follow reward")
    plt.plot(steps, history["obstacle_penalty"], label="Obstacle penalty")
    plt.plot(steps, history["control_penalty"], label="Control penalty")
    plt.xlabel("Step")
    plt.ylabel("Reward component")
    plt.title("Smoke Test — Reward Components")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
