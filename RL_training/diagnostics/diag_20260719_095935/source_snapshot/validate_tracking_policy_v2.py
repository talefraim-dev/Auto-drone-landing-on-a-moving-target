"""
validate_tracking_policy_v2.py

Deterministic tracking-policy validation with TRUE centering diagnostics.

Purpose
-------
This script does NOT train and does NOT sample random actions.
It loads an already-trained PPO tracking checkpoint and runs:

    model.predict(obs, deterministic=True)

The goal is to check two separate things:
    1. Target identity stability: MATCH / PRED / LOST
    2. Control quality: does the trained policy actually reduce bbox centering error?

Why v2 exists
-------------
The previous validation script searched for old/nonexistent keys such as
"bbox_error_x_norm" / "center_error_x_norm". In the current observation_builder.py,
the real observation keys are:

    err_x, err_y

where:
    err_x = 2 * (bbox_cx_norm - 0.5), clipped to [-1, 1]
    err_y = 2 * (bbox_cy_norm - 0.5), clipped to [-1, 1]

So this file logs err_x / err_y directly from obs_dict and also recomputes
pixel bbox geometry from stable_bbox_xyxy / raw_bbox_xyxy when available.

How to use
----------
1. Copy this file into RL_training/, next to Run_train.py and drone_env.py.
2. Make sure Unreal / AirSim is running.
3. Run:

    python validate_tracking_policy_v2.py

On reset, click the target once in the OpenCV window.
"""

from __future__ import annotations

import csv
import importlib
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO

from drone_env import DroneEnv
from weights_config import EnvConfig
from Run_train import apply_task_config_to_env_config

try:
    from observation_builder import ObservationBuilder
    FEATURE_NAMES = list(ObservationBuilder.FEATURE_NAMES)
except Exception:
    FEATURE_NAMES = []


# =============================================================================
# USER SETTINGS
# =============================================================================

# Leave as None to automatically load the newest tracking checkpoint.
# Example:
# MODEL_PATH = "models/PPO_Tracker/tracking/tracking_20260616_110425/tracking_ppo_200000_steps.zip"
MODEL_PATH: Optional[str] = None

TASK_CONFIG_MODULE = "config.tracking_config"
MODELS_ROOT = Path("models") / "PPO_Tracker" / "tracking"
RESULTS_DIR = Path("results") / "tracking_validation_v2"

NUM_EPISODES = 3
MAX_STEPS_PER_EPISODE = 700
SHOW_CV_WINDOW = True
PRINT_EVERY_N_STEPS = 10
SEED = 123

# Controls how strict the summary labels are.
CENTER_ERR_GOOD = 0.15
CENTER_ERR_OK = 0.25


# =============================================================================
# HELPERS
# =============================================================================


def set_deterministic_runtime(seed: int) -> None:
    """Reduce avoidable randomness in Python / NumPy / Torch inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
            "Train/copy a tracking checkpoint first, or set MODEL_PATH manually."
        )
    return checkpoints[-1]



def load_tracking_config() -> EnvConfig:
    task_module = importlib.import_module(TASK_CONFIG_MODULE)
    task = task_module.TASK_CONFIG

    cfg = apply_task_config_to_env_config(EnvConfig(), task)

    cfg.show_cv_window = bool(SHOW_CV_WINDOW)
    cfg.print_reset = False
    cfg.print_ep_summary = True
    cfg.print_obstacle_debug = False
    cfg.max_episode_steps = int(MAX_STEPS_PER_EPISODE)

    # For tracking evaluation, keep altitude stable.
    # We want to inspect yaw/lateral/forward behavior, not train descent.
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



def obs_vector_to_dict(obs: Any) -> Dict[str, float]:
    """Decode raw observation vector using ObservationBuilder.FEATURE_NAMES."""
    if not FEATURE_NAMES:
        return {}

    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    out: Dict[str, float] = {}
    for i, name in enumerate(FEATURE_NAMES):
        if i < len(arr):
            out[name] = safe_float(arr[i])
    return out



def merge_obs_dict(obs: Any, info: Dict[str, Any]) -> Dict[str, float]:
    """
    Prefer env-provided obs_dict, but fill missing values from the raw obs vector.
    This makes the script robust if info does not expose every feature.
    """
    from_info = dict(info.get("obs_dict", {}) or {})
    from_vec = obs_vector_to_dict(obs)

    merged = dict(from_vec)
    merged.update({k: safe_float(v) for k, v in from_info.items()})
    return merged



def bbox_to_pixel_metrics(
    bbox_xyxy: Any,
    image_width: int,
    image_height: int,
) -> Dict[str, float]:
    """Compute bbox center, size, and normalized error directly from pixels."""
    if bbox_xyxy is None:
        return {
            "px_bbox_valid": 0.0,
            "px_cx": 0.0,
            "px_cy": 0.0,
            "px_w": 0.0,
            "px_h": 0.0,
            "px_err_x": 0.0,
            "px_err_y": 0.0,
            "px_center_err": 0.0,
        }

    arr = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        return {
            "px_bbox_valid": 0.0,
            "px_cx": 0.0,
            "px_cy": 0.0,
            "px_w": 0.0,
            "px_h": 0.0,
            "px_err_x": 0.0,
            "px_err_y": 0.0,
            "px_center_err": 0.0,
        }

    x1, y1, x2, y2 = map(float, arr[:4])
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)

    # Same convention as observation_builder.py:
    # err_x > 0 means target center is to the RIGHT of image center.
    # err_y > 0 means target center is BELOW image center.
    err_x = 2.0 * ((cx / max(1.0, float(image_width))) - 0.5)
    err_y = 2.0 * ((cy / max(1.0, float(image_height))) - 0.5)
    err_x = float(np.clip(err_x, -1.0, 1.0))
    err_y = float(np.clip(err_y, -1.0, 1.0))
    center_err = float(np.sqrt(err_x * err_x + err_y * err_y))

    return {
        "px_bbox_valid": 1.0,
        "px_cx": cx,
        "px_cy": cy,
        "px_w": w,
        "px_h": h,
        "px_err_x": err_x,
        "px_err_y": err_y,
        "px_center_err": center_err,
    }



def direction_label(err_x: float, err_y: float, deadband: float = 0.05) -> str:
    parts: List[str] = []
    if err_x > deadband:
        parts.append("RIGHT")
    elif err_x < -deadband:
        parts.append("LEFT")

    if err_y > deadband:
        parts.append("LOW")
    elif err_y < -deadband:
        parts.append("HIGH")

    return "+".join(parts) if parts else "CENTER"



def center_quality_label(center_err: float) -> str:
    if center_err <= CENTER_ERR_GOOD:
        return "GOOD"
    if center_err <= CENTER_ERR_OK:
        return "OK"
    return "BAD"



def action_change_score(prev_err: Optional[float], curr_err: float) -> float:
    """
    Positive means center error improved compared to previous logged step.
    Negative means center error got worse.
    """
    if prev_err is None:
        return 0.0
    return float(prev_err - curr_err)



def build_row(
    episode_idx: int,
    step_idx: int,
    obs: Any,
    reward: float,
    done: bool,
    action: np.ndarray,
    info: Dict[str, Any],
    cfg: EnvConfig,
    prev_center_err: Optional[float],
) -> Dict[str, Any]:
    obs_dict = merge_obs_dict(obs, info)
    reward_parts = info.get("reward_parts", {}) or {}
    identity = info.get("tracker_identity_metrics", {}) or {}

    err_x = safe_float(obs_dict.get("err_x", 0.0))
    err_y = safe_float(obs_dict.get("err_y", 0.0))
    center_err = float(np.sqrt(err_x * err_x + err_y * err_y))

    stable_px = bbox_to_pixel_metrics(
        info.get("stable_bbox_xyxy"),
        image_width=int(getattr(cfg, "image_width", 960)),
        image_height=int(getattr(cfg, "image_height", 540)),
    )
    raw_px = bbox_to_pixel_metrics(
        info.get("raw_bbox_xyxy"),
        image_width=int(getattr(cfg, "image_width", 960)),
        image_height=int(getattr(cfg, "image_height", 540)),
    )

    # If stable bbox is available, use it as a secondary visual sanity check.
    # If not, fall back to observation error.
    pixel_center_err = (
        stable_px["px_center_err"]
        if stable_px["px_bbox_valid"] > 0.5
        else center_err
    )

    return {
        "episode": episode_idx,
        "step": step_idx,
        "reward": float(reward),
        "done": bool(done),
        "termination_reason": info.get("termination_reason", ""),
        "tracking_mode": info.get("tracking_mode", ""),
        "raw_tracker_mode": info.get("raw_tracker_mode", ""),
        "tracking_accepted": bool(info.get("tracking_accepted", False)),
        "tracking_pred_frames": int(info.get("tracking_pred_frames", 0)),

        # True observation-center diagnostics.
        "has_target": safe_float(obs_dict.get("has_target", 0.0)),
        "err_x": err_x,
        "err_y": err_y,
        "center_error_norm": center_err,
        "center_quality": center_quality_label(center_err),
        "target_direction": direction_label(err_x, err_y),
        "center_improvement": action_change_score(prev_center_err, center_err),
        "bbox_cx_norm": safe_float(obs_dict.get("bbox_cx", 0.0)),
        "bbox_cy_norm": safe_float(obs_dict.get("bbox_cy", 0.0)),
        "bbox_w_norm": safe_float(obs_dict.get("bbox_w", 0.0)),
        "bbox_h_norm": safe_float(obs_dict.get("bbox_h", 0.0)),
        "bbox_area_norm": safe_float(obs_dict.get("bbox_area", 0.0)),
        "bbox_conf": safe_float(obs_dict.get("bbox_conf", 0.0)),
        "centered_score": safe_float(obs_dict.get("centered_score", 0.0)),
        "target_stability_score": safe_float(obs_dict.get("target_stability_score", 0.0)),
        "landing_allowed": safe_float(obs_dict.get("landing_allowed", 0.0)),
        "img_vx": safe_float(obs_dict.get("img_vx", 0.0)),
        "img_vy": safe_float(obs_dict.get("img_vy", 0.0)),
        "distance_proxy_norm": safe_float(obs_dict.get("distance_proxy_norm", 1.0)),
        "distance_proxy_delta": safe_float(obs_dict.get("distance_proxy_delta", 0.0)),
        "lost_target_time_norm": safe_float(obs_dict.get("lost_target_time_norm", 0.0)),

        # Pixel bbox sanity check from env info.
        "stable_px_bbox_valid": stable_px["px_bbox_valid"],
        "stable_px_cx": stable_px["px_cx"],
        "stable_px_cy": stable_px["px_cy"],
        "stable_px_w": stable_px["px_w"],
        "stable_px_h": stable_px["px_h"],
        "stable_px_err_x": stable_px["px_err_x"],
        "stable_px_err_y": stable_px["px_err_y"],
        "stable_px_center_err": stable_px["px_center_err"],
        "raw_px_bbox_valid": raw_px["px_bbox_valid"],
        "raw_px_err_x": raw_px["px_err_x"],
        "raw_px_err_y": raw_px["px_err_y"],
        "raw_px_center_err": raw_px["px_center_err"],
        "pixel_center_err_used": pixel_center_err,

        # Drone / safety.
        "alt_agl_m": safe_float(info.get("alt_agl_m", 0.0)),
        "down_dist_m": safe_float(info.get("down_dist_m", 0.0)),
        "min_obstacle_dist_m": safe_float(info.get("min_obstacle_dist_m", 0.0)),
        "safety_intervention": bool(info.get("safety_intervention", False)),
        "safety_reasons": ";".join(info.get("safety_reasons", []) or []),

        # PPO action.
        "action_vx": float(action[0]),
        "action_vy": float(action[1]),
        "action_vz": float(action[2]),
        "action_yaw": float(action[3]),
        "raw_action_vx": safe_float(np.asarray(info.get("raw_action", [0, 0, 0, 0]))[0]),
        "raw_action_vy": safe_float(np.asarray(info.get("raw_action", [0, 0, 0, 0]))[1]),
        "raw_action_vz": safe_float(np.asarray(info.get("raw_action", [0, 0, 0, 0]))[2]),
        "raw_action_yaw": safe_float(np.asarray(info.get("raw_action", [0, 0, 0, 0]))[3]),
        "safe_action_vx": safe_float(np.asarray(info.get("safe_action", [0, 0, 0, 0]))[0]),
        "safe_action_vy": safe_float(np.asarray(info.get("safe_action", [0, 0, 0, 0]))[1]),
        "safe_action_vz": safe_float(np.asarray(info.get("safe_action", [0, 0, 0, 0]))[2]),
        "safe_action_yaw": safe_float(np.asarray(info.get("safe_action", [0, 0, 0, 0]))[3]),

        # Reward / identity.
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



def summarize_episode(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    n = len(rows)
    match_steps = sum(1 for r in rows if r.get("tracking_mode") == "MATCH")
    pred_steps = sum(1 for r in rows if r.get("tracking_mode") == "PRED")
    lost_steps = sum(1 for r in rows if r.get("tracking_mode") not in ("MATCH", "PRED"))

    center_errors = [safe_float(r.get("center_error_norm"), 1.0) for r in rows]
    pixel_center_errors = [safe_float(r.get("pixel_center_err_used"), 1.0) for r in rows]
    centered_scores = [safe_float(r.get("centered_score"), 0.0) for r in rows]
    distance_proxy = [safe_float(r.get("distance_proxy_norm"), 1.0) for r in rows]
    rewards = [safe_float(r.get("reward"), 0.0) for r in rows]
    altitudes = [safe_float(r.get("alt_agl_m"), 0.0) for r in rows]
    safety_flags = [bool(r.get("safety_intervention", False)) for r in rows]

    good_center_steps = sum(1 for e in center_errors if e <= CENTER_ERR_GOOD)
    ok_center_steps = sum(1 for e in center_errors if e <= CENTER_ERR_OK)

    first = rows[0]
    final = rows[-1]

    return {
        "steps": n,
        "return": float(sum(rewards)),
        "match_pct": 100.0 * match_steps / max(1, n),
        "pred_pct": 100.0 * pred_steps / max(1, n),
        "lost_pct": 100.0 * lost_steps / max(1, n),
        "avg_center_error_norm": float(np.mean(center_errors)),
        "max_center_error_norm": float(np.max(center_errors)),
        "final_center_error_norm": float(center_errors[-1]),
        "avg_pixel_center_error": float(np.mean(pixel_center_errors)),
        "avg_centered_score": float(np.mean(centered_scores)),
        "good_center_pct": 100.0 * good_center_steps / max(1, n),
        "ok_center_pct": 100.0 * ok_center_steps / max(1, n),
        "avg_distance_proxy_norm": float(np.mean(distance_proxy)),
        "avg_alt_agl_m": float(np.mean(altitudes)),
        "safety_pct": 100.0 * sum(safety_flags) / max(1, n),
        "termination_reason": final.get("termination_reason", ""),
        "start_err_x": safe_float(first.get("err_x"), 0.0),
        "start_err_y": safe_float(first.get("err_y"), 0.0),
        "final_err_x": safe_float(final.get("err_x"), 0.0),
        "final_err_y": safe_float(final.get("err_y"), 0.0),
    }


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    set_deterministic_runtime(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 104)
    print("[VALIDATE V2] Deterministic tracking-policy evaluation with TRUE centering diagnostics")
    print(f"[VALIDATE V2] Device: {device}")

    model_path = Path(MODEL_PATH) if MODEL_PATH else find_latest_checkpoint(MODELS_ROOT)
    print(f"[VALIDATE V2] Model: {model_path}")

    cfg = load_tracking_config()
    print(f"[VALIDATE V2] Task config: {TASK_CONFIG_MODULE}")
    print(f"[VALIDATE V2] Episodes: {NUM_EPISODES}")
    print(f"[VALIDATE V2] Max steps/episode: {MAX_STEPS_PER_EPISODE}")
    print("[VALIDATE V2] Action mode: model.predict(obs, deterministic=True)")
    print("[VALIDATE V2] Centering source: obs_dict['err_x'], obs_dict['err_y'] + stable_bbox pixel check")
    print("=" * 104)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"tracking_validation_v2_{timestamp}.csv"
    summary_path = RESULTS_DIR / f"tracking_validation_v2_summary_{timestamp}.txt"

    env = DroneEnv(cfg=cfg)
    model = PPO.load(str(model_path), env=None, device=device)

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    try:
        for ep in range(1, NUM_EPISODES + 1):
            print("\n" + "-" * 104)
            print(f"[EP {ep}] reset")
            obs, info = env.reset(seed=SEED + ep)

            ep_rows: List[Dict[str, Any]] = []
            ep_return = 0.0
            prev_center_err: Optional[float] = None

            for step in range(1, MAX_STEPS_PER_EPISODE + 1):
                action, _state = model.predict(obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32)

                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                ep_return += float(reward)

                row = build_row(
                    episode_idx=ep,
                    step_idx=step,
                    obs=obs,
                    reward=float(reward),
                    done=done,
                    action=action,
                    info=info,
                    cfg=cfg,
                    prev_center_err=prev_center_err,
                )
                prev_center_err = safe_float(row["center_error_norm"], prev_center_err or 0.0)

                ep_rows.append(row)
                all_rows.append(row)

                if step == 1 or step % PRINT_EVERY_N_STEPS == 0 or done:
                    print(
                        f"[EP {ep:02d} STEP {step:04d}] "
                        f"mode={row['tracking_mode']:<5} "
                        f"raw={row['raw_tracker_mode']:<5} "
                        f"err=({row['err_x']:+.3f},{row['err_y']:+.3f}) "
                        f"ce={row['center_error_norm']:.3f} "
                        f"q={row['center_quality']:<4} "
                        f"dir={row['target_direction']:<10} "
                        f"score={row['centered_score']:.3f} "
                        f"px_ce={row['pixel_center_err_used']:.3f} "
                        f"dist={row['distance_proxy_norm']:.3f} "
                        f"alt={row['alt_agl_m']:.2f}m "
                        f"act=[{row['action_vx']:+.2f},{row['action_vy']:+.2f},{row['action_vz']:+.2f},{row['action_yaw']:+.2f}] "
                        f"safe=[{row['safe_action_vx']:+.2f},{row['safe_action_vy']:+.2f},{row['safe_action_vz']:+.2f},{row['safe_action_yaw']:+.2f}] "
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
                f"avg_ce={summary['avg_center_error_norm']:.3f} "
                f"final_ce={summary['final_center_error_norm']:.3f} "
                f"GOOD={summary['good_center_pct']:.1f}% "
                f"OK={summary['ok_center_pct']:.1f}% "
                f"avg_score={summary['avg_centered_score']:.3f} "
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

    lines: List[str] = []
    lines.append("Deterministic tracking-policy validation V2")
    lines.append(f"Model: {model_path}")
    lines.append(f"Task config: {TASK_CONFIG_MODULE}")
    lines.append(f"Episodes: {NUM_EPISODES}")
    lines.append(f"Max steps per episode: {MAX_STEPS_PER_EPISODE}")
    lines.append("Centering source: err_x / err_y from observation_builder.py")
    lines.append("")

    for s in summaries:
        lines.append(
            f"EP {s['episode']}: "
            f"steps={s['steps']}, "
            f"return={s['return']:+.2f}, "
            f"MATCH={s['match_pct']:.1f}%, "
            f"PRED={s['pred_pct']:.1f}%, "
            f"LOST={s['lost_pct']:.1f}%, "
            f"avg_ce={s['avg_center_error_norm']:.3f}, "
            f"max_ce={s['max_center_error_norm']:.3f}, "
            f"final_ce={s['final_center_error_norm']:.3f}, "
            f"GOOD={s['good_center_pct']:.1f}%, "
            f"OK={s['ok_center_pct']:.1f}%, "
            f"avg_centered_score={s['avg_centered_score']:.3f}, "
            f"avg_px_ce={s['avg_pixel_center_error']:.3f}, "
            f"avg_dist={s['avg_distance_proxy_norm']:.3f}, "
            f"avg_alt={s['avg_alt_agl_m']:.2f}m, "
            f"safety={s['safety_pct']:.1f}%, "
            f"start_err=({s['start_err_x']:+.3f},{s['start_err_y']:+.3f}), "
            f"final_err=({s['final_err_x']:+.3f},{s['final_err_y']:+.3f}), "
            f"reason={s['termination_reason']}"
        )

    if summaries:
        lines.append("")
        lines.append("Overall:")
        lines.append(f"avg MATCH = {np.mean([s['match_pct'] for s in summaries]):.1f}%")
        lines.append(f"avg PRED  = {np.mean([s['pred_pct'] for s in summaries]):.1f}%")
        lines.append(f"avg LOST  = {np.mean([s['lost_pct'] for s in summaries]):.1f}%")
        lines.append(f"avg center error = {np.mean([s['avg_center_error_norm'] for s in summaries]):.3f}")
        lines.append(f"avg GOOD center pct = {np.mean([s['good_center_pct'] for s in summaries]):.1f}%")
        lines.append(f"avg OK center pct = {np.mean([s['ok_center_pct'] for s in summaries]):.1f}%")
        lines.append(f"avg safety intervention = {np.mean([s['safety_pct'] for s in summaries]):.1f}%")
        lines.append("")
        lines.append("Decision guide:")
        lines.append("- Identity is good if MATCH is high and LOST is near 0.")
        lines.append("- Centering is good if avg center error <= 0.15 and GOOD center pct is high.")
        lines.append("- If MATCH is high but avg center error is high, the tracker is fine but the policy/reward/action scaling is not centering enough.")

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 104)
    print(f"[VALIDATE V2] CSV saved: {csv_path}")
    print(f"[VALIDATE V2] Summary saved: {summary_path}")
    print("[VALIDATE V2] Done")
    print("=" * 104)


if __name__ == "__main__":
    main()
