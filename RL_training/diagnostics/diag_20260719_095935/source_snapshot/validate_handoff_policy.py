"""
validate_handoff_policy.py

Evaluation-only runner for the UAV tracking/handoff policy.

Purpose:
    Run an already-trained PPO agent WITHOUT training.
    Measure how reliably it reaches the camera handoff condition:
        front tracking -> bottom scan -> bottom confirmed -> handoff_success

Usage examples:
    python validate_handoff_policy.py --episodes 20

    python validate_handoff_policy.py --model models/PPO_Tracker/tracking/<run>/tracking_ppo_210000_steps.zip --episodes 30

    python validate_handoff_policy.py --episodes 10 --no-cv

    python validate_handoff_policy.py --episodes 20 --deterministic

Notes:
    - This script does not call model.learn().
    - It assumes it is placed inside the RL_training directory.
    - It uses the same DroneEnv and tracking task config used by Run_train.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO

from drone_env import DroneEnv
from weights_config import EnvConfig


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
        return default
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_bool(value: Any) -> bool:
    try:
        return bool(value)
    except Exception:
        return False


def _reset_env(env: DroneEnv) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Support both Gymnasium reset() -> (obs, info) and old Gym reset() -> obs."""
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        return obs, info or {}
    return out, {}


def _step_env(env: DroneEnv, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    """Support Gymnasium 5-tuple and old Gym 4-tuple step APIs."""
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated or truncated), info or {}
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, float(reward), bool(done), info or {}
    raise RuntimeError(f"Unexpected env.step() return format: {type(out)} / len={len(out) if isinstance(out, tuple) else 'NA'}")


def _extract_step_number(path: Path) -> int:
    """Extract the largest integer before '_steps' from a checkpoint filename."""
    m = re.search(r"(\d+)_steps", path.name)
    if not m:
        return -1
    return int(m.group(1))


def find_latest_checkpoint(models_root: Path = Path("models/PPO_Tracker/tracking")) -> Path:
    """Find the latest tracking PPO checkpoint."""
    if not models_root.exists():
        raise FileNotFoundError(f"Models root not found: {models_root}")

    candidates = []
    candidates.extend(models_root.rglob("tracking_ppo_*_steps.zip"))
    candidates.extend(models_root.rglob("*.zip"))

    # Avoid duplicates and sort by step number first, then modified time.
    unique = sorted(set(candidates), key=lambda p: (_extract_step_number(p), p.stat().st_mtime), reverse=True)
    if not unique:
        raise FileNotFoundError(f"No .zip checkpoints found under: {models_root}")

    return unique[0]


def build_tracking_env_config(args: argparse.Namespace) -> EnvConfig:
    """Build EnvConfig using the same tracking task config as Run_train.py when available."""
    cfg = EnvConfig()

    # Try to reuse Run_train.py's task config application so evaluation matches training.
    try:
        from Run_train import apply_task_config_to_env_config
        from config.tracking_config import TASK_CONFIG

        cfg = apply_task_config_to_env_config(cfg, TASK_CONFIG)
        print("[EVAL] Applied config.tracking_config.TASK_CONFIG via Run_train.apply_task_config_to_env_config")
    except Exception as exc:
        print(f"[EVAL WARNING] Could not apply tracking task config automatically: {exc}")
        print("[EVAL WARNING] Continuing with EnvConfig defaults + CLI overrides.")

    # Evaluation overrides.
    cfg.show_cv_window = bool(args.show_cv)
    cfg.print_ep_summary = True
    cfg.print_reset = bool(args.print_reset)
    cfg.print_obstacle_debug = bool(args.print_obstacles)
    cfg.obstacle_debug_every_n_steps = int(args.obstacle_debug_every)

    # Keep FPS reasonable. Active mode means:
    #   front main view in chase
    #   bottom/inset/active view according to the patched handoff logic
    if hasattr(cfg, "cv_display_mode"):
        cfg.cv_display_mode = str(args.cv_mode)
    if hasattr(cfg, "show_dual_camera_cv"):
        cfg.show_dual_camera_cv = bool(args.cv_mode == "dual")
    if hasattr(cfg, "cv_debug_render_every_n_steps"):
        cfg.cv_debug_render_every_n_steps = int(args.cv_every)

    cfg.max_episode_steps = int(args.max_steps)

    if args.cmd_duration is not None:
        cfg.cmd_duration_s = float(args.cmd_duration)

    return cfg


def _format_bool(value: bool) -> str:
    return "1" if value else "0"


def evaluate_policy(args: argparse.Namespace) -> None:
    model_path = Path(args.model) if args.model else find_latest_checkpoint(Path(args.models_root))
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"handoff_eval_{run_stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "handoff_eval_episodes.csv"
    json_path = out_dir / "handoff_eval_summary.json"

    print("=" * 90)
    print("[EVAL] Handoff validation only — NO TRAINING")
    print(f"[EVAL] Model       : {model_path}")
    print(f"[EVAL] Episodes    : {args.episodes}")
    print(f"[EVAL] Deterministic: {args.deterministic}")
    print(f"[EVAL] CV          : {args.show_cv} mode={args.cv_mode} every={args.cv_every}")
    print(f"[EVAL] Output CSV  : {csv_path}")
    print("=" * 90)

    cfg = build_tracking_env_config(args)

    env = DroneEnv(cfg)
    model = PPO.load(str(model_path), device=args.device)

    episode_rows = []
    aggregate_reasons = Counter()
    aggregate_phase_success = Counter()

    try:
        obs, _ = _reset_env(env)

        for ep_idx in range(1, int(args.episodes) + 1):
            ep_return = 0.0
            phases = Counter()
            last_phase = None
            last_info: Dict[str, Any] = {}

            gate_seen = False
            raw_bottom_seen = False
            bottom_confirmed_seen = False
            handoff_ready_seen = False
            fresh_seen = False
            visual_scan_seen = False
            visual_lidar_seen = False
            safety_seen = False

            first_gate_step: Optional[int] = None
            first_bottom_match_step: Optional[int] = None
            first_bottom_confirmed_step: Optional[int] = None
            first_handoff_ready_step: Optional[int] = None

            max_bsim = 0.0
            max_barea = 0.0
            max_bstreak = 0
            max_vscore = 0.0
            min_real_d = float("inf")
            min_best_d = float("inf")
            init_d = float("nan")

            step_count = 0
            reason = ""

            t0 = time.time()

            for step_idx in range(1, int(args.max_steps) + 1):
                action, _state = model.predict(obs, deterministic=bool(args.deterministic))
                obs, reward, done, info = _step_env(env, action)

                step_count = step_idx
                ep_return += float(reward)
                last_info = info or {}

                phase = str(last_info.get("handoff_phase", "UNKNOWN"))
                phases[phase] += 1

                gate = _safe_bool(last_info.get("handoff_candidate_gate", False))
                bm = _safe_bool(last_info.get("bottom_match", False))
                bc = _safe_bool(last_info.get("bottom_confirmed", False))
                fresh = _safe_bool(last_info.get("bottom_match_fresh", False))
                ready = _safe_bool(last_info.get("handoff_ready", False))
                vs = _safe_bool(last_info.get("handoff_visual_scan_trigger", False))
                vl = _safe_bool(last_info.get("handoff_visual_lidar_trigger", False))
                safety = _safe_bool(last_info.get("safety_intervention", False))

                bsim = _safe_float(last_info.get("bottom_similarity", 0.0), 0.0)
                barea = _safe_float(last_info.get("bottom_bbox_area_norm", 0.0), 0.0)
                bstreak = _safe_int(last_info.get("bottom_match_streak", 0), 0)
                vscore = _safe_float(last_info.get("handoff_visual_score", 0.0), 0.0)
                real_d = _safe_float(last_info.get("current_chase_distance_m", last_info.get("realD", float("nan"))), float("nan"))
                best_d = _safe_float(last_info.get("best_chase_distance_m", float("nan")), float("nan"))
                init_val = _safe_float(last_info.get("initial_chase_distance_m", float("nan")), float("nan"))

                if math.isfinite(init_val):
                    init_d = init_val
                if math.isfinite(real_d):
                    min_real_d = min(min_real_d, real_d)
                if math.isfinite(best_d):
                    min_best_d = min(min_best_d, best_d)

                max_bsim = max(max_bsim, bsim)
                max_barea = max(max_barea, barea)
                max_bstreak = max(max_bstreak, bstreak)
                max_vscore = max(max_vscore, vscore)

                if gate and not gate_seen:
                    gate_seen = True
                    first_gate_step = step_idx
                    print(f"[EVAL EP {ep_idx:03d}] step={step_idx:04d} Gate opened phase={phase} Vscore={vscore:.2f}")

                if bm and not raw_bottom_seen:
                    raw_bottom_seen = True
                    first_bottom_match_step = step_idx
                    print(f"[EVAL EP {ep_idx:03d}] step={step_idx:04d} BM=1 sim={bsim:.3f} area={barea:.4f} streak={bstreak}")

                if bc and not bottom_confirmed_seen:
                    bottom_confirmed_seen = True
                    first_bottom_confirmed_step = step_idx
                    print(f"[EVAL EP {ep_idx:03d}] step={step_idx:04d} BC=1 sim={bsim:.3f} area={barea:.4f} streak={bstreak}")

                if ready and not handoff_ready_seen:
                    handoff_ready_seen = True
                    first_handoff_ready_step = step_idx
                    print(f"[EVAL EP {ep_idx:03d}] step={step_idx:04d} HANDOFF_READY")

                fresh_seen = fresh_seen or fresh
                visual_scan_seen = visual_scan_seen or vs
                visual_lidar_seen = visual_lidar_seen or vl
                safety_seen = safety_seen or safety

                if phase != last_phase:
                    print(
                        f"[EVAL EP {ep_idx:03d}] step={step_idx:04d} "
                        f"phase={phase} Gate={_format_bool(gate)} BM={_format_bool(bm)} "
                        f"BC={_format_bool(bc)} Fresh={_format_bool(fresh)} "
                        f"Bsim={bsim:.3f} Barea={barea:.4f} realD={real_d if math.isfinite(real_d) else 'nan'}"
                    )
                    last_phase = phase

                if done:
                    reason = str(last_info.get("termination_reason", last_info.get("reason", "")) or "")
                    break

            duration_s = time.time() - t0

            if not reason:
                reason = str(last_info.get("termination_reason", "max_steps_reached") or "max_steps_reached")

            # Success is evaluated by the environment reason, plus the actual secondary-camera evidence.
            env_success = reason == "handoff_success"
            secondary_success = bool(
                bottom_confirmed_seen
                and raw_bottom_seen
                and fresh_seen
                and max_bsim >= float(args.min_success_bsim)
                and max_barea >= float(args.min_success_barea)
                and max_bstreak >= int(args.min_success_streak)
            )
            strict_success = bool(env_success and secondary_success)

            if strict_success:
                aggregate_phase_success["strict_success"] += 1
            elif env_success:
                aggregate_phase_success["env_success_but_weak_evidence"] += 1
            elif bottom_confirmed_seen:
                aggregate_phase_success["bottom_confirmed_no_success"] += 1
            elif raw_bottom_seen:
                aggregate_phase_success["raw_bottom_only"] += 1
            elif gate_seen:
                aggregate_phase_success["gate_only"] += 1
            else:
                aggregate_phase_success["no_handoff_progress"] += 1

            aggregate_reasons[reason] += 1

            row = {
                "episode": ep_idx,
                "steps": step_count,
                "return": round(ep_return, 4),
                "duration_s": round(duration_s, 3),
                "reason": reason,
                "env_success": int(env_success),
                "secondary_success": int(secondary_success),
                "strict_success": int(strict_success),
                "gate_seen": int(gate_seen),
                "raw_bottom_seen": int(raw_bottom_seen),
                "bottom_confirmed_seen": int(bottom_confirmed_seen),
                "handoff_ready_seen": int(handoff_ready_seen),
                "fresh_seen": int(fresh_seen),
                "visual_scan_seen": int(visual_scan_seen),
                "visual_lidar_seen": int(visual_lidar_seen),
                "safety_seen": int(safety_seen),
                "first_gate_step": "" if first_gate_step is None else first_gate_step,
                "first_bottom_match_step": "" if first_bottom_match_step is None else first_bottom_match_step,
                "first_bottom_confirmed_step": "" if first_bottom_confirmed_step is None else first_bottom_confirmed_step,
                "first_handoff_ready_step": "" if first_handoff_ready_step is None else first_handoff_ready_step,
                "max_bsim": round(max_bsim, 4),
                "max_barea": round(max_barea, 6),
                "max_bstreak": max_bstreak,
                "max_vscore": round(max_vscore, 4),
                "init_realD": "" if not math.isfinite(init_d) else round(init_d, 4),
                "min_realD": "" if not math.isfinite(min_real_d) else round(min_real_d, 4),
                "min_bestD": "" if not math.isfinite(min_best_d) else round(min_best_d, 4),
                "phases": json.dumps(dict(phases), sort_keys=True),
            }
            episode_rows.append(row)

            print(
                f"[EVAL EP {ep_idx:03d} DONE] "
                f"reason={reason} steps={step_count} return={ep_return:+.2f} "
                f"strict_success={int(strict_success)} env_success={int(env_success)} secondary={int(secondary_success)} "
                f"Gate={int(gate_seen)} BM={int(raw_bottom_seen)} BC={int(bottom_confirmed_seen)} "
                f"maxBsim={max_bsim:.3f} maxBarea={max_barea:.4f} maxStreak={max_bstreak} "
                f"minRealD={min_real_d if math.isfinite(min_real_d) else 'nan'}"
            )

            # Prepare next episode unless this was the last one.
            if ep_idx < int(args.episodes):
                obs, _ = _reset_env(env)

    finally:
        try:
            env.close()
        except Exception:
            pass

    if episode_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
            writer.writeheader()
            writer.writerows(episode_rows)

    n = max(1, len(episode_rows))
    strict_success_count = sum(int(r["strict_success"]) for r in episode_rows)
    env_success_count = sum(int(r["env_success"]) for r in episode_rows)
    secondary_success_count = sum(int(r["secondary_success"]) for r in episode_rows)
    gate_count = sum(int(r["gate_seen"]) for r in episode_rows)
    bm_count = sum(int(r["raw_bottom_seen"]) for r in episode_rows)
    bc_count = sum(int(r["bottom_confirmed_seen"]) for r in episode_rows)

    summary = {
        "model": str(model_path),
        "episodes": len(episode_rows),
        "strict_success_count": strict_success_count,
        "strict_success_rate": strict_success_count / n,
        "env_success_count": env_success_count,
        "env_success_rate": env_success_count / n,
        "secondary_success_count": secondary_success_count,
        "secondary_success_rate": secondary_success_count / n,
        "gate_seen_count": gate_count,
        "gate_seen_rate": gate_count / n,
        "raw_bottom_seen_count": bm_count,
        "raw_bottom_seen_rate": bm_count / n,
        "bottom_confirmed_count": bc_count,
        "bottom_confirmed_rate": bc_count / n,
        "termination_reasons": dict(aggregate_reasons),
        "phase_buckets": dict(aggregate_phase_success),
        "csv_path": str(csv_path),
    }

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 90)
    print("[EVAL SUMMARY]")
    print(f"episodes                : {len(episode_rows)}")
    print(f"strict_success_rate      : {strict_success_count}/{n} = {strict_success_count / n:.1%}")
    print(f"env_success_rate         : {env_success_count}/{n} = {env_success_count / n:.1%}")
    print(f"secondary_success_rate   : {secondary_success_count}/{n} = {secondary_success_count / n:.1%}")
    print(f"gate_seen_rate           : {gate_count}/{n} = {gate_count / n:.1%}")
    print(f"raw_bottom_seen_rate     : {bm_count}/{n} = {bm_count / n:.1%}")
    print(f"bottom_confirmed_rate    : {bc_count}/{n} = {bc_count / n:.1%}")
    print(f"termination_reasons      : {dict(aggregate_reasons)}")
    print(f"phase_buckets            : {dict(aggregate_phase_success)}")
    print(f"CSV                      : {csv_path}")
    print(f"JSON                     : {json_path}")
    print("=" * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO tracking/handoff policy without training.")

    parser.add_argument("--model", type=str, default="", help="Path to PPO .zip checkpoint. If omitted, latest checkpoint is used.")
    parser.add_argument("--models-root", type=str, default="models/PPO_Tracker/tracking", help="Root directory to search for checkpoints.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=700, help="Max steps per episode.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device for loading PPO model.")

    parser.add_argument("--show-cv", dest="show_cv", action="store_true", default=True, help="Show OpenCV debug window.")
    parser.add_argument("--no-cv", dest="show_cv", action="store_false", help="Disable OpenCV debug window for faster evaluation.")
    parser.add_argument("--cv-mode", type=str, default="active", choices=["active", "front", "bottom", "dual"], help="CV display mode.")
    parser.add_argument("--cv-every", type=int, default=3, help="Render CV every N env steps.")

    parser.add_argument("--cmd-duration", type=float, default=None, help="Override EnvConfig.cmd_duration_s if needed.")
    parser.add_argument("--print-reset", action="store_true", help="Print verbose reset logs.")
    parser.add_argument("--print-obstacles", action="store_true", help="Print obstacle debug logs.")
    parser.add_argument("--obstacle-debug-every", type=int, default=30, help="Print obstacle debug every N steps.")

    parser.add_argument("--min-success-bsim", type=float, default=0.56, help="Analysis threshold for secondary success similarity.")
    parser.add_argument("--min-success-barea", type=float, default=0.004, help="Analysis threshold for secondary success bbox area.")
    parser.add_argument("--min-success-streak", type=int, default=3, help="Analysis threshold for secondary success streak.")

    parser.add_argument("--output-dir", type=str, default="handoff_eval_runs", help="Directory for CSV/JSON evaluation output.")

    return parser.parse_args()


if __name__ == "__main__":
    evaluate_policy(parse_args())
