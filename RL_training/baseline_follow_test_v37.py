import csv
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from weights_config import EnvConfig
from drone_env import DroneEnv


def clamp(x, lo=-1.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def get_obs(info, key, default=0.0):
    try:
        return float(info.get("obs_dict", {}).get(key, default))
    except Exception:
        return float(default)


def get_info(info, key, default=0.0):
    try:
        value = info.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


class BaselineFollowController:
    """
    Simple proportional baseline controller for validating the v37 environment.

    Purpose:
    - Not meant to replace RL.
    - Verifies that OBS signals produce reasonable drone actions.
    - Gives a stronger baseline than random actions before PPO training.

    Action convention:
        action = [vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd]
        normalized to [-1, 1]

    Controller idea:
    - If target is far: move forward.
    - If target is too close: move backward.
    - If target is right/left in image: yaw and side-correct.
    - Keep vertical axis mostly stable in this Follow test.
    """

    def __init__(
        self,
        desired_distance_proxy=0.45,
        k_forward=1.25,
        k_side=-0.35,
        k_yaw=0.85,
        k_vertical=-0.15,
        max_forward_cmd=0.65,
        max_side_cmd=0.35,
        max_yaw_cmd=0.65,
        max_vertical_cmd=0.20,
        lost_target_forward_cmd=0.0,
        lost_target_yaw_cmd=0.25,
    ):
        self.desired_distance_proxy = desired_distance_proxy

        self.k_forward = k_forward
        self.k_side = k_side
        self.k_yaw = k_yaw
        self.k_vertical = k_vertical

        self.max_forward_cmd = max_forward_cmd
        self.max_side_cmd = max_side_cmd
        self.max_yaw_cmd = max_yaw_cmd
        self.max_vertical_cmd = max_vertical_cmd

        self.lost_target_forward_cmd = lost_target_forward_cmd
        self.lost_target_yaw_cmd = lost_target_yaw_cmd

    def act(self, obs_dict):
        has_target = float(obs_dict.get("has_target", 0.0))
        err_x = float(obs_dict.get("err_x", 0.0))
        err_y = float(obs_dict.get("err_y", 0.0))
        distance_proxy = float(obs_dict.get("distance_proxy_norm", 1.0))

        if has_target < 0.5:
            # Basic search behavior: hold position and slowly yaw.
            return np.array(
                [
                    self.lost_target_forward_cmd,
                    0.0,
                    0.0,
                    self.lost_target_yaw_cmd,
                ],
                dtype=np.float32,
            )

        # distance_proxy:
        # 0 = close, 1 = far.
        # If target is far, distance_proxy > desired -> positive forward command.
        distance_error = distance_proxy - self.desired_distance_proxy
        vx_cmd = self.k_forward * distance_error

        # Image x error:
        # err_x > 0 means target is right side of image.
        # We use both lateral correction and yaw correction.
        vy_cmd = self.k_side * err_x
        yaw_cmd = self.k_yaw * err_x

        # Optional small vertical correction:
        # err_y > 0 means target appears low in the image.
        # For Follow baseline we keep this weak to avoid unstable altitude changes.
        vz_cmd = self.k_vertical * err_y

        vx_cmd = clamp(vx_cmd, -self.max_forward_cmd, self.max_forward_cmd)
        vy_cmd = clamp(vy_cmd, -self.max_side_cmd, self.max_side_cmd)
        vz_cmd = clamp(vz_cmd, -self.max_vertical_cmd, self.max_vertical_cmd)
        yaw_cmd = clamp(yaw_cmd, -self.max_yaw_cmd, self.max_yaw_cmd)

        return np.array([vx_cmd, vy_cmd, vz_cmd, yaw_cmd], dtype=np.float32)


def run_baseline_test(max_steps=300):
    cfg = EnvConfig()
    cfg.show_cv_window = True
    cfg.print_ep_summary = True
    cfg.print_obstacle_debug = True
    cfg.obstacle_debug_every_n_steps = 30

    env = DroneEnv(cfg)
    controller = BaselineFollowController(
        desired_distance_proxy=float(cfg.desired_distance_proxy)
    )

    obs, info = env.reset()
    print("Initial obs shape:", obs.shape)

    history = {
        "step": [],
        "reward": [],
        "total_reward": [],
        "centered_score": [],
        "distance_proxy_norm": [],
        "target_stability_score": [],
        "has_target": [],
        "alt_agl_m": [],
        "down_dist_m": [],
        "min_obstacle_dist_m": [],
        "collision_risk_score": [],
        "safety_intervention": [],
        "safety_intervention_rate_pct": [],
        "vx_cmd": [],
        "vy_cmd": [],
        "vz_cmd": [],
        "yaw_cmd": [],
        "match_pct": [],
        "pred_pct": [],
        "none_pct": [],
    }

    total_reward = 0.0
    final_info = info

    for step in range(max_steps):
        obs_dict = final_info.get("obs_dict", {})
        action = controller.act(obs_dict)

        obs, reward, done, truncated, info = env.step(action)
        final_info = info
        total_reward += float(reward)

        obs_dict = info.get("obs_dict", {})

        history["step"].append(step)
        history["reward"].append(float(reward))
        history["total_reward"].append(float(total_reward))
        history["centered_score"].append(get_obs(info, "centered_score"))
        history["distance_proxy_norm"].append(get_obs(info, "distance_proxy_norm"))
        history["target_stability_score"].append(get_obs(info, "target_stability_score"))
        history["has_target"].append(get_obs(info, "has_target"))
        history["alt_agl_m"].append(get_info(info, "alt_agl_m"))
        history["down_dist_m"].append(get_info(info, "down_dist_m"))
        history["min_obstacle_dist_m"].append(get_info(info, "min_obstacle_dist_m"))
        history["collision_risk_score"].append(get_obs(info, "collision_risk_score"))
        history["safety_intervention"].append(1.0 if info.get("safety_intervention", False) else 0.0)
        history["safety_intervention_rate_pct"].append(get_info(info, "safety_intervention_rate_pct"))

        history["vx_cmd"].append(float(action[0]))
        history["vy_cmd"].append(float(action[1]))
        history["vz_cmd"].append(float(action[2]))
        history["yaw_cmd"].append(float(action[3]))

        history["match_pct"].append(get_info(info, "match_pct"))
        history["pred_pct"].append(get_info(info, "pred_pct"))
        history["none_pct"].append(get_info(info, "none_pct"))

        if step % 20 == 0:
            print(
                f"step={step:03d} "
                f"reward={reward:+.3f} "
                f"total={total_reward:+.2f} "
                f"done={done} "
                f"center={get_obs(info, 'centered_score'):.2f} "
                f"dist_proxy={get_obs(info, 'distance_proxy_norm'):.2f} "
                f"alt={info.get('alt_agl_m', 0.0):.2f} "
                f"hmin={info.get('min_obstacle_dist_m', 0.0):.2f} "
                f"match={info.get('match_pct', 0.0):.1f}% "
                f"pred={info.get('pred_pct', 0.0):.1f}% "
                f"safety={info.get('safety_intervention', False)} "
                f"action=[{action[0]:+.2f},{action[1]:+.2f},{action[2]:+.2f},{action[3]:+.2f}] "
                f"reason={info.get('termination_reason', '')}"
            )

        if done or truncated:
            print("Episode ended early.")
            print("reason:", info.get("termination_reason", ""))
            break

    print_summary(history, total_reward, final_info)

    output_dir = make_output_dir()
    save_history_csv(history, os.path.join(output_dir, "baseline_follow_history.csv"))
    plot_baseline(history, output_dir)

    print(f"\nSaved baseline outputs to: {output_dir}")


def print_summary(history, total_reward, final_info):
    steps = len(history["step"])
    print("\n===== BASELINE FOLLOW TEST SUMMARY =====")
    print("steps:", steps)
    print("total_reward:", total_reward)
    print("final_alt_agl_m:", final_info.get("alt_agl_m", None))
    print("final_min_obstacle_dist_m:", final_info.get("min_obstacle_dist_m", None))
    print("final_down_dist_m:", final_info.get("down_dist_m", None))
    print("final_safety_rate_pct:", final_info.get("safety_intervention_rate_pct", None))
    print("final_match_pct:", final_info.get("match_pct", None))
    print("final_pred_pct:", final_info.get("pred_pct", None))
    print("final_none_pct:", final_info.get("none_pct", None))
    print("final_reason:", final_info.get("termination_reason", ""))


def make_output_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("runs", "baseline_follow_v37", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_history_csv(history, path):
    keys = list(history.keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)

        for i in range(len(history["step"])):
            writer.writerow([history[k][i] for k in keys])


def save_current_figure(output_dir, filename):
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150)
    print("Saved plot:", path)


def plot_baseline(history, output_dir):
    steps = np.asarray(history["step"])

    if len(steps) == 0:
        print("No data to plot.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(steps, history["reward"], label="Reward per step")
    plt.plot(steps, history["total_reward"], label="Cumulative reward")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Baseline Follow — Reward Behavior")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_current_figure(output_dir, "reward_behavior.png")

    plt.figure(figsize=(12, 6))
    plt.plot(steps, history["centered_score"], label="Centered score")
    plt.plot(steps, history["distance_proxy_norm"], label="Distance proxy norm")
    plt.plot(steps, history["target_stability_score"], label="Target stability score")
    plt.plot(steps, history["has_target"], label="Has target")
    plt.xlabel("Step")
    plt.ylabel("Score / Normalized value")
    plt.title("Baseline Follow — Tracking Signals")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_current_figure(output_dir, "tracking_signals.png")

    plt.figure(figsize=(12, 6))
    plt.plot(steps, history["match_pct"], label="MATCH %")
    plt.plot(steps, history["pred_pct"], label="PRED %")
    plt.plot(steps, history["none_pct"], label="NONE %")
    plt.xlabel("Step")
    plt.ylabel("Percent")
    plt.title("Baseline Follow — Tracker Mode Percentages")
    plt.ylim(-5, 105)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_current_figure(output_dir, "tracker_mode_percentages.png")

    plt.figure(figsize=(12, 6))
    plt.plot(steps, history["alt_agl_m"], label="Altitude AGL [m]")
    plt.plot(steps, history["down_dist_m"], label="Down distance [m]")
    plt.plot(steps, history["min_obstacle_dist_m"], label="Horizontal min obstacle [m]")
    plt.xlabel("Step")
    plt.ylabel("Meters")
    plt.title("Baseline Follow — Altitude and Obstacle Distances")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_current_figure(output_dir, "altitude_obstacles.png")

    plt.figure(figsize=(12, 6))
    plt.plot(steps, history["collision_risk_score"], label="Collision risk score")
    plt.plot(steps, history["safety_intervention"], label="Safety intervention flag")
    plt.plot(steps, np.asarray(history["safety_intervention_rate_pct"]) / 100.0, label="Safety rate / 100")
    plt.xlabel("Step")
    plt.ylabel("Score / Flag")
    plt.title("Baseline Follow — Safety Layer Behavior")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_current_figure(output_dir, "safety_behavior.png")

    plt.figure(figsize=(12, 6))
    plt.plot(steps, history["vx_cmd"], label="vx_cmd")
    plt.plot(steps, history["vy_cmd"], label="vy_cmd")
    plt.plot(steps, history["vz_cmd"], label="vz_cmd")
    plt.plot(steps, history["yaw_cmd"], label="yaw_cmd")
    plt.xlabel("Step")
    plt.ylabel("Normalized command")
    plt.title("Baseline Follow — Controller Commands")
    plt.ylim(-1.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_current_figure(output_dir, "controller_commands.png")

    plt.show()


if __name__ == "__main__":
    run_baseline_test(max_steps=300)
