"""
Run_train.py

Final Python-only training entry point.

No .bat files.
No CLI arguments.
No environment variables.

Choose one specialist agent by changing SELECTED_TRAINING_TASK.

Available values:
    "static_landing"
    "tracking"
    "dynamic_landing"

Each task loads a different config file:
    config/static_landing_config.py
    config/tracking_config.py
    config/dynamic_landing_config.py
"""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from drone_env import DroneEnv
from weights_config import EnvConfig


# ============================================================================
# USER SELECTION
# ============================================================================
# Change this only:
SELECTED_TRAINING_TASK = "tracking"

# Clean training is safer during final project work.
RESUME_LATEST_CHECKPOINT = True


# ============================================================================
# PPO HYPERPARAMETERS
# ============================================================================
PPO_N_STEPS = 1024
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_LEARNING_RATE = 3e-4
PPO_CLIP_RANGE = 0.2
PPO_ENT_COEF = 0.01
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5


TASK_MODULES = {
    "static_landing": "config.static_landing_config",
    "tracking": "config.tracking_config",
    "dynamic_landing": "config.dynamic_landing_config",
}


def load_task_config(task_name: str):
    if task_name not in TASK_MODULES:
        valid = ", ".join(TASK_MODULES.keys())
        raise ValueError(f"Unknown task '{task_name}'. Valid tasks: {valid}")

    module = importlib.import_module(TASK_MODULES[task_name])
    return module.TASK_CONFIG


class TrainingMetricsCallback(BaseCallback):
    """
    Logs useful project metrics for plot_training_results.py.

    Important custom tags:
        custom/reward
        custom/bbox_center_error_x
        custom/bbox_center_error_y
        custom/bbox_center_error_norm
        custom/flight_smoothness
        custom/altitude_m
        custom/match_pct
        custom/pred_pct
        custom/none_pct
        custom/safety_intervention
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self._prev_safe_action = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", None)

        if rewards is not None and len(rewards) > 0:
            self.logger.record("custom/reward", float(rewards[0]))

        if not infos:
            return True

        info = infos[0]
        obs_dict = info.get("obs_dict", {}) or {}

        ex = float(obs_dict.get("bbox_error_x_norm", obs_dict.get("center_error_x_norm", 0.0)))
        ey = float(obs_dict.get("bbox_error_y_norm", obs_dict.get("center_error_y_norm", 0.0)))
        en = float((ex * ex + ey * ey) ** 0.5)

        ex = max(-1.0, min(1.0, ex))
        ey = max(-1.0, min(1.0, ey))
        en = max(0.0, min(1.0, en))

        self.logger.record("custom/bbox_center_error_x", ex)
        self.logger.record("custom/bbox_center_error_y", ey)
        self.logger.record("custom/bbox_center_error_norm", en)

        safe_action = info.get("safe_action", None)
        smoothness = 0.0

        if safe_action is not None:
            try:
                import numpy as np
                a = np.asarray(safe_action, dtype=float)
                if self._prev_safe_action is not None:
                    delta = float(np.linalg.norm(a - self._prev_safe_action))
                    smoothness = max(-1.0, min(1.0, delta))
                self._prev_safe_action = a
            except Exception:
                smoothness = 0.0

        self.logger.record("custom/flight_smoothness", smoothness)
        self.logger.record("custom/altitude_m", float(info.get("alt_agl_m", 0.0)))
        self.logger.record("custom/match_pct", float(info.get("match_pct", 0.0)))
        self.logger.record("custom/pred_pct", float(info.get("pred_pct", 0.0)))
        self.logger.record("custom/none_pct", float(info.get("none_pct", 0.0)))
        self.logger.record("custom/safety_intervention", float(bool(info.get("safety_intervention", False))))

        return True


def apply_task_config_to_env_config(cfg: EnvConfig, task) -> EnvConfig:
    """
    Apply specialist agent config to EnvConfig.

    This keeps task selection clean:
        Run_train.py selects a task.
        The selected config file defines the task.
        DroneEnv receives one complete EnvConfig.
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


def config_to_dict(cfg: EnvConfig) -> Dict[str, Any]:
    out = {}

    for k in dir(cfg):
        if k.startswith("_"):
            continue
        v = getattr(cfg, k)
        if callable(v):
            continue

        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = str(v)

    return out


def config_hash(cfg: EnvConfig) -> str:
    payload = json.dumps(config_to_dict(cfg), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def find_latest_checkpoint(task_dir: Path) -> Path | None:
    checkpoints = sorted(task_dir.glob("**/*.zip"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def main() -> None:
    task = load_task_config(SELECTED_TRAINING_TASK)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print(f"[TRAIN] Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("[TRAIN] CUDA not available, using CPU")

    cfg = apply_task_config_to_env_config(EnvConfig(), task)

    print("=" * 80)
    print(f"[TRAIN] Selected task : {task.name}")
    print(f"[TRAIN] Config file   : {TASK_MODULES[SELECTED_TRAINING_TASK]}")
    print(f"[TRAIN] Description   : {task.description}")
    print(f"[TRAIN] OBS_DIM       : {cfg.obs_dim}")
    print(f"[TRAIN] freeze_vz     : {cfg.freeze_vz}")
    print(f"[TRAIN] altitude_hold : {getattr(cfg, 'altitude_hold_enabled', None)}")
    print(f"[TRAIN] min_alt       : {cfg.min_safe_altitude_m}")
    print(f"[TRAIN] desired_dist  : {cfg.desired_distance_proxy}")
    print("=" * 80)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    models_root = Path("models") / "PPO_Tracker" / task.name
    logs_root = Path("logs") / "PPO_Tracker" / task.name

    run_dir = models_root / f"{task.run_name_prefix}_{timestamp}"
    tb_dir = logs_root / f"{task.run_name_prefix}_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = config_to_dict(cfg)
    cfg_dict["_config_hash"] = config_hash(cfg)
    cfg_dict["_selected_training_task"] = task.name
    cfg_dict["_task_config_file"] = TASK_MODULES[SELECTED_TRAINING_TASK]

    (run_dir / "training_config_snapshot.json").write_text(
        json.dumps(cfg_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    env = Monitor(DroneEnv(cfg=cfg))

    print("[TRAIN] Environment created")
    print("[TRAIN] Observation space:", env.observation_space)
    print("[TRAIN] Action space     :", env.action_space)

    logger = configure(str(tb_dir), ["stdout", "tensorboard"])

    latest_ckpt = find_latest_checkpoint(models_root) if RESUME_LATEST_CHECKPOINT else None

    if latest_ckpt is not None:
        print(f"[TRAIN] Resuming from checkpoint: {latest_ckpt}")
        model = PPO.load(str(latest_ckpt), env=env, device=device)
    else:
        print("[TRAIN] Creating new PPO model")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            device=device,
            verbose=1,
            tensorboard_log=str(tb_dir),
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            gamma=PPO_GAMMA,
            gae_lambda=PPO_GAE_LAMBDA,
            learning_rate=PPO_LEARNING_RATE,
            clip_range=PPO_CLIP_RANGE,
            ent_coef=PPO_ENT_COEF,
            vf_coef=PPO_VF_COEF,
            max_grad_norm=PPO_MAX_GRAD_NORM,
        )

    model.set_logger(logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=int(task.checkpoint_freq),
        save_path=str(run_dir),
        name_prefix=f"{task.name}_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    metrics_callback = TrainingMetricsCallback()

    print(f"[TRAIN] Starting training for {task.total_timesteps} timesteps")

    model.learn(
        total_timesteps=int(task.total_timesteps),
        callback=[checkpoint_callback, metrics_callback],
        reset_num_timesteps=latest_ckpt is None,
        progress_bar=False,
    )

    final_path = run_dir / f"{task.name}_ppo_final.zip"
    model.save(str(final_path))

    print(f"[TRAIN] Saved final model: {final_path}")


if __name__ == "__main__":
    main()
