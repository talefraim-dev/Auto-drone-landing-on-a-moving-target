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


class RelativeYawTrackerController:
    """
    Simple visual-servoing baseline controller.

    Principle:
        The controller is relative to the tracker/BBox only.

        BBox moves right  -> yaw right
        BBox moves left   -> yaw left
        BBox centered     -> yaw 0

    This version intentionally does NOT use forward motion.
    It only validates that yaw control can keep the target centered.

    Action:
        [vx_cmd, vy_cmd, vz_cmd, yaw_cmd]

    Important:
        If the drone turns the wrong way, flip yaw_sign.
    """

    def __init__(
        self,
        warmup_steps=10,
        yaw_sign=+1.0,
        k_yaw=0.35,
        max_yaw_cmd=0.22,
        deadzone_x=0.04,
        match_conf_threshold=0.8,

        # If True, allow weak correction in PRED.
        # For debugging tracker stability, start with False.
        use_pred_yaw=False,
        pred_yaw_scale=0.2,
    ):
        self.warmup_steps = warmup_steps
        self.yaw_sign = yaw_sign
        self.k_yaw = k_yaw
        self.max_yaw_cmd = max_yaw_cmd
        self.deadzone_x = deadzone_x
        self.match_conf_threshold = match_conf_threshold
        self.use_pred_yaw = use_pred_yaw
        self.pred_yaw_scale = pred_yaw_scale

    def act(self, obs_dict, step):
        if step < self.warmup_steps:
            return np.zeros(4, dtype=np.float32), "WARMUP_HOVER"

        has_target = float(obs_dict.get("has_target", 0.0))
        bbox_conf = float(obs_dict.get("bbox_conf", 0.0))
        err_x = float(obs_dict.get("err_x", 0.0))

        if has_target < 0.5:
            return np.zeros(4, dtype=np.float32), "LOST_HOVER"

        is_match = bbox_conf >= self.match_conf_threshold
        is_pred = (not is_match) and bbox_conf > 0.0

        if abs(err_x) < self.deadzone_x:
            yaw_cmd = 0.0
        else:
            yaw_cmd = self.yaw_sign * self.k_yaw * err_x

        if is_match:
            mode = "MATCH_YAW_SERVO"
            yaw_cmd = clamp(yaw_cmd, -self.max_yaw_cmd, self.max_yaw_cmd)

        elif is_pred and self.use_pred_yaw:
            mode = "PRED_WEAK_YAW"
            yaw_cmd = clamp(
                self.pred_yaw_scale * yaw_cmd,
                -self.max_yaw_cmd * self.pred_yaw_scale,
                self.max_yaw_cmd * self.pred_yaw_scale,
            )

        else:
            # Do not chase prediction by default.
            mode = "PRED_OR_LOWCONF_HOVER"
            yaw_cmd = 0.0

        vx_cmd = 0.0
        vy_cmd = 0.0
        vz_cmd = 0.0

        return np.array([vx_cmd, vy_cmd, vz_cmd, yaw_cmd], dtype=np.float32), mode


def run_test(max_steps=300):
    cfg = EnvConfig()
    cfg.show_cv_window = True
    cfg.print_ep_summary = True
    cfg.print_obstacle_debug = True
    cfg.obstacle_debug_every_n_steps = 30

    env = DroneEnv(cfg)

    controller = RelativeYawTrackerController(
        # First try +1.0 because user expects:
        # target right => yaw right.
        # If it turns opposite, change to -1.0.
        yaw_sign=+1.0,
        k_yaw=0.35,
        max_yaw_cmd=0.22,
        deadzone_x=0.04,
        warmup_steps=10,
        use_pred_yaw=False,
    )

    obs, info = env.reset()
    print("Initial obs shape:", obs.shape)

    history = {
        "step": [],
        "reward": [],
        "total_reward": [],
        "centered_score": [],
        "err_x": [],
        "err_y": [],
        "bbox_conf": [],
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
        "controller_mode": [],
    }

    total_reward = 0.0
    final_info = info

    for step in range(max_steps):
        obs_dict = final_info.get("obs_dict", {})
        action, controller_mode = controller.act(obs_dict, step)

        obs, reward, done, truncated, info = env.step(action)
        final_info = info
        total_reward += float(reward)

        history["step"].append(step)
        history["reward"].append(float(reward))
        history["total_reward"].append(float(total_reward))
        history["centered_score"].append(get_obs(info, "centered_score"))
        history["err_x"].append(get_obs(info, "err_x"))
        history["err_y"].append(get_obs(info, "err_y"))
        history["bbox_conf"].append(get_obs(info, "bbox_conf"))
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
        history["controller_mode"].append(controller_mode)

        if step % 10 == 0:
            print(
                f"step={step:03d} "
                f"reward={reward:+.3f} "
                f"total={total_reward:+.2f} "
                f"done={done} "
                f"mode={controller_mode} "
                f"conf={get_obs(info, 'bbox_conf'):.2f} "
                f"err_x={get_obs(info, 'err_x'):+.3f} "
                f"center={get_obs(info, 'centered_score'):.2f} "
                f"match={info.get('match_pct', 0.0):.1f}% "
                f"pred={info.get('pred_pct', 0.0):.1f}% "
                f"yaw={action[3]:+.3f} "
                f"reason={info.get('termination_reason', '')}"
            )

        if done or truncated:
            print("Episode ended early.")
            print("reason:", info.get("termination_reason", ""))
            break

    print_summary(history, total_reward, final_info)

    output_dir = make_output_dir()
    save_history_csv(history, os.path.join(output_dir, "relative_yaw_tracker_history.csv"))
    plot_results(history, output_dir)

    print(f"\nSaved outputs to: {output_dir}")


def print_summary(history, total_reward, final_info):
    print("\n===== RELATIVE YAW TRACKER TEST SUMMARY =====")
    print("steps:", len(history["step"]))
    print("total_reward:", total_reward)
    print("final_alt_agl_m:", final_info.get("alt_agl_m", None))
    print("final_min_obstacle_dist_m:", final_info.get("min_obstacle_dist_m", None))
    print("final_safety_rate_pct:", final_info.get("safety_intervention_rate_pct", None))
    print("final_match_pct:", final_info.get("match_pct", None))
    print("final_pred_pct:", final_info.get("pred_pct", None))
    print("final_none_pct:", final_info.get("none_pct", None))
    print("final_reason:", final_info.get("termination_reason", ""))

    mode_counts = {}
    for mode in history["controller_mode"]:
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    print("\nController mode counts:")
    for mode, count in sorted(mode_counts.items()):
        print(f"  {mode}: {count}")


def make_output_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("runs", "relative_yaw_tracker_v37", timestamp)
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


def plot_results(history, output_dir):
    steps = np.asarray(history["step"])

    if len(steps) == 0:
        print("No data to plot.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(steps, history["reward"], label="Reward per step")
    plt.plot(steps, history["total_reward"], label="Cumulative reward")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Relative Yaw Tracker — Reward Behavior")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_current_figure(output_dir, "reward_behavior.png")

    plt.figure(figsize=(12, 6))
    plt.plot(steps, history["err_x"], label="err_x")
    plt.plot(steps, history["yaw_cmd"], label="yaw_cmd")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Relative Yaw Tracker — err_x vs yaw_cmd")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_current_figure(output_dir, "err_x_vs_yaw.png")

    plt.figure(figsize=(12, 6))
    plt.plot(steps, history["centered_score"], label="Centered score")
    plt.plot(steps, history["bbox_conf"], label="BBox confidence")
    plt.plot(steps, history["has_target"], label="Has target")
    plt.plot(steps, history["target_stability_score"], label="Target stability score")
    plt.xlabel("Step")
    plt.ylabel("Score / Normalized value")
    plt.title("Relative Yaw Tracker — Tracking Signals")
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
    plt.title("Relative Yaw Tracker — Tracker Mode Percentages")
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
    plt.title("Relative Yaw Tracker — Altitude and Obstacle Distances")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_current_figure(output_dir, "altitude_obstacles.png")

    plt.figure(figsize=(12, 6))
    plt.plot(steps, history["vx_cmd"], label="vx_cmd")
    plt.plot(steps, history["vy_cmd"], label="vy_cmd")
    plt.plot(steps, history["vz_cmd"], label="vz_cmd")
    plt.plot(steps, history["yaw_cmd"], label="yaw_cmd")
    plt.xlabel("Step")
    plt.ylabel("Normalized command")
    plt.title("Relative Yaw Tracker — Controller Commands")
    plt.ylim(-1.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_current_figure(output_dir, "controller_commands.png")

    plt.show()


if __name__ == "__main__":
    run_test(max_steps=300)
