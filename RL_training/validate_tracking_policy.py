"""
validate_tracking_policy.py

Deterministic evaluation script for the already-trained tracking PPO agent.

Purpose:
    This script does NOT train.
    This script does NOT sample random actions.
    This script only loads an existing PPO checkpoint and runs:

        model.predict(obs, deterministic=True)

    The goal is to verify whether the tracking/following policy is good enough
    before moving on to landing training.

How to use:
    1. Copy this file into the RL_training folder, next to Run_train.py.
    2. Make sure Unreal / AirSim is already running.
    3. Run:

        python validate_tracking_policy.py

Notes:
    - On the first reset, click the target once in the OpenCV window.
    - The script automatically finds the latest tracking checkpoint under:

        models/PPO_Tracker/tracking/**/*.zip

    - You can also set MODEL_PATH manually below.
"""

from __future__ import annotations

import csv
import importlib
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from stable_baselines3 import PPO

from drone_env import DroneEnv
from weights_config import EnvConfig
from Run_train import apply_task_config_to_env_config


# =============================================================================
# USER SETTINGS
# =============================================================================

# Leave as None to automatically load the newest tracking checkpoint.
# Example manual path:
# MODEL_PATH = "models/PPO_Tracker/tracking/tracking_20260616_110425/tracking_ppo_200000_steps.zip"
MODEL_PATH: Optional[str] = None

TASK_CONFIG_MODULE = "config.tracking_config"
MODELS_ROOT = Path("models") / "PPO_Tracker" / "tracking"
RESULTS_DIR = Path("results") / "tracking_validation"

NUM_EPISODES = 3
MAX_STEPS_PER_EPISODE = 700

# True = show Tracker Debug window from DroneEnv.
SHOW_CV_WINDOW = True

# Print a compact line every N steps.
PRINT_EVERY_N_STEPS = 10

# Fixed seeds for reproducibility outside AirSim/Unreal physics.
SEED = 123


# =============================================================================
# HELPERS
# =============================================================================


def set_deterministic_runtime(seed: int) -> None:
    """Reduce avoidable randomness in Python / NumPy / Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # This evaluation uses inference only. These flags reduce avoidable torch-side
    # nondeterminism, but Unreal/AirSim physics and rendering may still vary slightly.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass



def find_latest_checkpoint(models_root: Path) -> Path:
    checkpoints = sorted(models_root.glob("**/*.zip"), key=lambda p: p.stat().st_mtime)

    if not checkpoints:
        raise FileNotFoundError(
            f"No PPO checkpoints found under: {models_root}\n"
            "Train or copy a tracking checkpoint first, or set MODEL_PATH manually."
        )

    return checkpoints[-1]



def load_tracking_config() -> EnvConfig:
    task_module = importlib.import_module(TASK_CONFIG_MODULE)
    task = task_module.TASK_CONFIG

    cfg = apply_task_config_to_env_config(EnvConfig(), task)

    # Evaluation should be strict and observable, but not noisy.
    cfg.show_cv_window = bool(SHOW_CV_WINDOW)
    cfg.print_reset = False
    cfg.print_ep_summary = True
    cfg.print_obstacle_debug = False
    cfg.max_episode_steps = int(MAX_STEPS_PER_EPISODE)

    # Important for this validation:
    # tracking phase should not learn/use descent. Altitude hold remains active.
    cfg.freeze_vz = True
    cfg.altitude_hold_enabled = True
    cfg.enable_z_control = False

    return cfg



def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default



def summarize_episode(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    n = len(rows)
    match_steps = sum(1 for r in rows if r.get("tracking_mode") == "MATCH")
    pred_steps = sum(1 for r in rows if r.get("tracking_mode") == "PRED")
    lost_steps = sum(1 for r in rows if r.get("tracking_mode") not in ("MATCH", "PRED"))

    center_errors = [safe_float(r.get("center_error_norm"), 1.0) for r in rows]
    distance_proxy = [safe_float(r.get("distance_proxy_norm"), 1.0) for r in rows]
    rewards = [safe_float(r.get("reward"), 0.0) for r in rows]
    altitudes = [safe_float(r.get("alt_agl_m"), 0.0) for r in rows]
    safety_flags = [bool(r.get("safety_intervention", False)) for r in rows]

    final = rows[-1]

    return {
        "steps": n,
        "return": float(sum(rewards)),
        "match_pct": 100.0 * match_steps / max(1, n),
        "pred_pct": 100.0 * pred_steps / max(1, n),
        "lost_pct": 100.0 * lost_steps / max(1, n),
        "avg_center_error_norm": float(np.mean(center_errors)),
        "max_center_error_norm": float(np.max(center_errors)),
        "avg_distance_proxy_norm": float(np.mean(distance_proxy)),
        "avg_alt_agl_m": float(np.mean(altitudes)),
        "safety_pct": 100.0 * sum(safety_flags) / max(1, n),
        "termination_reason": final.get("termination_reason", ""),
    }



def build_row(
    episode_idx: int,
    step_idx: int,
    reward: float,
    done: bool,
    action: np.ndarray,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    obs_dict = info.get("obs_dict", {}) or {}
    reward_parts = info.get("reward_parts", {}) or {}
    identity = info.get("tracker_identity_metrics", {}) or {}

    ex = safe_float(obs_dict.get("bbox_error_x_norm", obs_dict.get("center_error_x_norm", 0.0)))
    ey = safe_float(obs_dict.get("bbox_error_y_norm", obs_dict.get("center_error_y_norm", 0.0)))
    center_error_norm = float(np.sqrt(ex * ex + ey * ey))

    return {
        "episode": episode_idx,
        "step": step_idx,
        "reward": float(reward),
        "done": bool(done),
        "termination_reason": info.get("termination_reason", ""),
        "tracking_mode": info.get("tracking_mode", ""),
        "raw_tracker_mode": info.get("raw_tracker_mode", ""),
        "has_target": safe_float(obs_dict.get("has_target", 0.0)),
        "center_error_x_norm": ex,
        "center_error_y_norm": ey,
        "center_error_norm": center_error_norm,
        "centered_score": safe_float(obs_dict.get("centered_score", 0.0)),
        "distance_proxy_norm": safe_float(obs_dict.get("distance_proxy_norm", 1.0)),
        "lost_target_time_norm": safe_float(obs_dict.get("lost_target_time_norm", 0.0)),
        "alt_agl_m": safe_float(info.get("alt_agl_m", 0.0)),
        "down_dist_m": safe_float(info.get("down_dist_m", 0.0)),
        "min_obstacle_dist_m": safe_float(info.get("min_obstacle_dist_m", 0.0)),
        "safety_intervention": bool(info.get("safety_intervention", False)),
        "safety_reasons": ";".join(info.get("safety_reasons", []) or []),
        "action_vx": float(action[0]),
        "action_vy": float(action[1]),
        "action_vz": float(action[2]),
        "action_yaw": float(action[3]),
        "reward_center": safe_float(reward_parts.get("center_reward", 0.0)),
        "reward_distance": safe_float(reward_parts.get("distance_reward", 0.0)),
        "reward_visibility": safe_float(reward_parts.get("visibility_reward", 0.0)),
        "reward_lost": safe_float(reward_parts.get("lost_target_penalty", 0.0)),
        "identity_score": safe_float(identity.get("score", 0.0)),
        "identity_feat_sim": safe_float(identity.get("feat_sim", 0.0)),
        "identity_spatial_sim": safe_float(identity.get("spatial_sim", 0.0)),
        "tracker_reject_reason": info.get("tracker_reject_reason", ""),
        "tracker_reject_frames": int(info.get("tracker_reject_frames", 0)),
    }


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    set_deterministic_runtime(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 90)
    print("[VALIDATE] Deterministic tracking-policy evaluation")
    print(f"[VALIDATE] Device: {device}")

    model_path = Path(MODEL_PATH) if MODEL_PATH else find_latest_checkpoint(MODELS_ROOT)
    print(f"[VALIDATE] Model: {model_path}")

    cfg = load_tracking_config()
    print(f"[VALIDATE] Task config: {TASK_CONFIG_MODULE}")
    print(f"[VALIDATE] Episodes: {NUM_EPISODES}")
    print(f"[VALIDATE] Max steps/episode: {MAX_STEPS_PER_EPISODE}")
    print("[VALIDATE] Action mode: model.predict(obs, deterministic=True)")
    print("=" * 90)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"tracking_validation_{timestamp}.csv"
    summary_path = RESULTS_DIR / f"tracking_validation_summary_{timestamp}.txt"

    env = DroneEnv(cfg=cfg)
    model = PPO.load(str(model_path), env=None, device=device)

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    try:
        for ep in range(1, NUM_EPISODES + 1):
            print("\n" + "-" * 90)
            print(f"[EP {ep}] reset")
            obs, info = env.reset(seed=SEED + ep)

            ep_rows: List[Dict[str, Any]] = []
            ep_return = 0.0

            for step in range(1, MAX_STEPS_PER_EPISODE + 1):
                action, _state = model.predict(obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32)

                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                ep_return += float(reward)

                row = build_row(
                    episode_idx=ep,
                    step_idx=step,
                    reward=float(reward),
                    done=done,
                    action=action,
                    info=info,
                )
                ep_rows.append(row)
                all_rows.append(row)

                if step == 1 or step % PRINT_EVERY_N_STEPS == 0 or done:
                    print(
                        f"[EP {ep:02d} STEP {step:04d}] "
                        f"mode={row['tracking_mode']:<5} "
                        f"raw={row['raw_tracker_mode']:<5} "
                        f"center_err={row['center_error_norm']:.3f} "
                        f"dist={row['distance_proxy_norm']:.3f} "
                        f"alt={row['alt_agl_m']:.2f}m "
                        f"act=[{row['action_vx']:+.2f},{row['action_vy']:+.2f},{row['action_vz']:+.2f},{row['action_yaw']:+.2f}] "
                        f"r={float(reward):+.2f} "
                        f"ret={ep_return:+.2f}"
                    )

                if done:
                    break

            summary = summarize_episode(ep_rows)
            summary["episode"] = ep
            summaries.append(summary)

            print(
                f"[EP {ep}] SUMMARY "
                f"steps={summary['steps']} "
                f"return={summary['return']:+.2f} "
                f"MATCH={summary['match_pct']:.1f}% "
                f"PRED={summary['pred_pct']:.1f}% "
                f"LOST={summary['lost_pct']:.1f}% "
                f"avg_center_err={summary['avg_center_error_norm']:.3f} "
                f"safety={summary['safety_pct']:.1f}% "
                f"reason={summary['termination_reason']}"
            )

    finally:
        try:
            env.close()
        except Exception:
            pass

    if all_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    lines = []
    lines.append("Deterministic tracking-policy validation")
    lines.append(f"Model: {model_path}")
    lines.append(f"Episodes: {NUM_EPISODES}")
    lines.append(f"Max steps per episode: {MAX_STEPS_PER_EPISODE}")
    lines.append("")

    for s in summaries:
        lines.append(
            f"EP {s['episode']}: "
            f"steps={s['steps']}, "
            f"return={s['return']:+.2f}, "
            f"MATCH={s['match_pct']:.1f}%, "
            f"PRED={s['pred_pct']:.1f}%, "
            f"LOST={s['lost_pct']:.1f}%, "
            f"avg_center_err={s['avg_center_error_norm']:.3f}, "
            f"max_center_err={s['max_center_error_norm']:.3f}, "
            f"avg_dist={s['avg_distance_proxy_norm']:.3f}, "
            f"avg_alt={s['avg_alt_agl_m']:.2f}m, "
            f"safety={s['safety_pct']:.1f}%, "
            f"reason={s['termination_reason']}"
        )

    if summaries:
        lines.append("")
        lines.append("Overall:")
        lines.append(f"avg MATCH = {np.mean([s['match_pct'] for s in summaries]):.1f}%")
        lines.append(f"avg PRED  = {np.mean([s['pred_pct'] for s in summaries]):.1f}%")
        lines.append(f"avg LOST  = {np.mean([s['lost_pct'] for s in summaries]):.1f}%")
        lines.append(f"avg center error = {np.mean([s['avg_center_error_norm'] for s in summaries]):.3f}")
        lines.append(f"avg safety intervention = {np.mean([s['safety_pct'] for s in summaries]):.1f}%")

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 90)
    print(f"[VALIDATE] CSV saved: {csv_path}")
    print(f"[VALIDATE] Summary saved: {summary_path}")
    print("[VALIDATE] Done")
    print("=" * 90)


if __name__ == "__main__":
    main()
