"""Automatic alternating co-training for Agent 1 and Agent 2.

Run only:
    python Run_train_alternating_agents.py

No command-line arguments, environment variables or configuration edits are
required. Each cycle performs:
    1. Train Agent 1 while Agent 2 is frozen.
    2. Save Agent 1.
    3. Train Agent 2 while the newly saved Agent 1 is frozen.
    4. Save Agent 2.

For both models, the largest learning signal is whether the verified landing
RPC latch succeeded. Dense task rewards remain present at a reduced scale.
"""

from __future__ import annotations

import gc
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from agent1p2_env import Agent1P2Env, find_agent1_checkpoint
from agent2_landing_env import Agent2Config
from alternating_cotraining_env import (
    RPC_FAILURE_PENALTY,
    RPC_SUCCESS_REWARD,
    IDENTITY_EPISODE_CAP,
    AGENT2_IDENTITY_REWARD_SCALE,
    RpcDominantAgent2RewardEnv,
    TrainAgent1WithFrozenAgent2Env,
    RangeFinderLiveDiagnostics,
)
from config import flow_config as flow


# Deliberately fixed defaults: run the file and it handles the complete cycle.
NUMBER_OF_CYCLES = 10
STEPS_PER_AGENT1_PHASE = 20_480
STEPS_PER_AGENT2_PHASE = 20_480
CHECKPOINT_EVERY_STEPS = 10_240

PPO_N_STEPS = 1024
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 10
PPO_LEARNING_RATE = 3e-4


@dataclass(frozen=True)
class TargetIdentitySnapshot:
    """Serializable identity selected once at the beginning of the run."""

    fingerprint: np.ndarray
    class_id: int | None


def save_target_identity(snapshot: TargetIdentitySnapshot, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    class_id = -1 if snapshot.class_id is None else int(snapshot.class_id)
    np.savez_compressed(
        path,
        fingerprint=np.asarray(snapshot.fingerprint, dtype=np.float32),
        class_id=np.asarray([class_id], dtype=np.int64),
    )


def load_target_identity(path: Path) -> TargetIdentitySnapshot:
    with np.load(path, allow_pickle=False) as data:
        fingerprint = np.asarray(data["fingerprint"], dtype=np.float32).reshape(-1)
        raw_class_id = int(np.asarray(data["class_id"]).reshape(-1)[0])
    if fingerprint.size == 0 or not np.all(np.isfinite(fingerprint)):
        raise ValueError(f"Invalid target fingerprint in {path}")
    return TargetIdentitySnapshot(
        fingerprint=fingerprint.copy(),
        class_id=None if raw_class_id < 0 else raw_class_id,
    )


def inject_target_identity(parallel_env: Agent1P2Env, snapshot: TargetIdentitySnapshot) -> None:
    """Install the saved identity before the environment's first reset.

    DroneEnv.reset() then skips the click window and performs its normal
    fingerprint-based AUTO_LOCK after resetting the target actor.
    """
    env = parallel_env.agent1_env
    fingerprint = np.asarray(snapshot.fingerprint, dtype=np.float32).reshape(-1).copy()
    env.target_fingerprint = fingerprint
    env.target_class_id = snapshot.class_id
    env._target_initialized = True

    env.tracker.set_target_fingerprint(fingerprint)
    env.tracker.set_target_class(snapshot.class_id)
    try:
        env.bottom_tracker.set_target_fingerprint(fingerprint)
        env.bottom_tracker.set_target_class(snapshot.class_id)
    except Exception:
        # DroneEnv.reset() repeats this synchronization and reports sensor errors
        # according to the existing configuration.
        pass


def capture_target_identity(parallel_env: Agent1P2Env) -> TargetIdentitySnapshot:
    env = parallel_env.agent1_env
    fingerprint = getattr(env, "target_fingerprint", None)
    if fingerprint is None:
        fingerprint = env.tracker.get_target_fingerprint()
    if fingerprint is None:
        raise RuntimeError("Target selection finished without a target fingerprint.")
    arr = np.asarray(fingerprint, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise RuntimeError("Target selection produced an invalid fingerprint.")
    return TargetIdentitySnapshot(
        fingerprint=arr.copy(),
        class_id=getattr(env, "target_class_id", None),
    )


def select_target_once(
    agent1_checkpoint: Path,
    run_root: Path,
    device: str,
) -> TargetIdentitySnapshot:
    """Reset the scene, ask for one click, and persist identity for all phases."""
    identity_path = run_root / "selected_target_identity.npz"
    if identity_path.exists():
        snapshot = load_target_identity(identity_path)
        print(f"[CO-TRAIN] Loaded run target identity: {identity_path}")
        return snapshot

    print("=" * 92)
    print("[CO-TRAIN] ONE-TIME TARGET SELECTION")
    print("[CO-TRAIN] The target actor will be reset before the click window opens.")
    print("[CO-TRAIN] After this click, every phase and cycle uses AUTO_LOCK.")
    print("=" * 92)

    parallel = build_parallel_env(agent1_checkpoint, device)
    try:
        # This is the only reset in the entire script allowed to request a click.
        parallel.agent1_env.reset()
        snapshot = capture_target_identity(parallel)
        save_target_identity(snapshot, identity_path)
    finally:
        parallel.close()

    print(
        "[CO-TRAIN] Target identity saved | "
        f"class_id={snapshot.class_id} dims={snapshot.fingerprint.size} "
        f"path={identity_path}"
    )
    return snapshot


def latest_agent2_checkpoint() -> Path:
    explicit = Path(str(getattr(flow, "AGENT_2_MODEL_PATH", "")))
    if str(explicit) and explicit.is_file():
        return explicit
    if str(explicit):
        raise FileNotFoundError(f"Configured Agent-2 checkpoint not found: {explicit}")

    root = Path("models") / "AGENT_1P2"
    candidates = list(root.glob("**/agent2_ppo_*_steps.zip"))
    candidates += list(root.glob("**/agent2_ppo_final.zip"))
    if not candidates:
        raise FileNotFoundError(
            "No Agent-2 checkpoint configured and no legacy checkpoint found."
        )
    return max(candidates, key=lambda p: (p.stat().st_mtime, str(p)))


def build_agent2_config() -> Agent2Config:
    """Build the current supported parallel landing configuration."""
    cfg = Agent2Config(
        show_camera=bool(flow.SHOW_AGENT2_CAMERA),
        max_episode_steps=int(flow.AGENT2_MAX_EPISODE_STEPS),
        static_target_actor_name=str(flow.AGENT_2_TARGET_ACTOR_NAME),
        static_target_surface_altitude_m=float(
            flow.AGENT_2_STATIC_TARGET_SURFACE_ALTITUDE_M
        ),
    )

    assignments = {
        "target_velocity_ema_alpha": "PARALLEL_TARGET_VELOCITY_EMA_ALPHA",
        "horizontal_velocity_ema_alpha": "PARALLEL_BOTTOM_RELATIVE_VELOCITY_EMA_ALPHA",
        "predictive_bottom_extra_latency_s": "PARALLEL_BOTTOM_PREDICTION_EXTRA_LATENCY_S",
        "predictive_bottom_horizon_min_s": "PARALLEL_BOTTOM_PREDICTION_HORIZON_MIN_S",
        "predictive_bottom_horizon_max_s": "PARALLEL_BOTTOM_PREDICTION_HORIZON_MAX_S",
        "predictive_bottom_normal_correction_max_mps": "PARALLEL_BOTTOM_NORMAL_CORRECTION_MAX_MPS",
        "predictive_bottom_catchup_correction_max_mps": "PARALLEL_BOTTOM_CATCHUP_CORRECTION_MAX_MPS",
        "predictive_bottom_catchup_enter_center_error": "PARALLEL_BOTTOM_CATCHUP_ENTER_CENTER_ERROR",
        "predictive_bottom_catchup_exit_center_error": "PARALLEL_BOTTOM_CATCHUP_EXIT_CENTER_ERROR",
        "catchup_descent_enabled": "PARALLEL_CATCHUP_DESCENT_ENABLED",
        "catchup_descent_max_vz_mps": "PARALLEL_CATCHUP_DESCENT_MAX_VZ_MPS",
        "catchup_descent_touchdown_max_vz_mps": "PARALLEL_CATCHUP_DESCENT_TOUCHDOWN_MAX_VZ_MPS",
        "catchup_descent_touchdown_height_m": "PARALLEL_CATCHUP_DESCENT_TOUCHDOWN_HEIGHT_M",
        "catchup_descent_max_predicted_center_error": "PARALLEL_CATCHUP_DESCENT_MAX_PREDICTED_CENTER_ERROR",
        "catchup_descent_max_image_speed_per_s": "PARALLEL_CATCHUP_DESCENT_MAX_IMAGE_SPEED_PER_S",
        "catchup_descent_max_outward_speed_per_s": "PARALLEL_CATCHUP_DESCENT_MAX_OUTWARD_SPEED_PER_S",
        "catchup_descent_max_metric_outward_speed_mps": "PARALLEL_CATCHUP_DESCENT_MAX_METRIC_OUTWARD_SPEED_MPS",
        "dense_reward_enabled": "AGENT2_DENSE_REWARD_ENABLED",
        "dense_reward_alignment_center_error": "AGENT2_DENSE_REWARD_ALIGNMENT_CENTER_ERROR",
        "dense_reward_descent_progress_per_m": "AGENT2_DENSE_REWARD_DESCENT_PROGRESS_PER_M",
        "dense_reward_near_touch_height_m": "AGENT2_DENSE_REWARD_NEAR_TOUCH_HEIGHT_M",
        "dense_reward_near_touch_multiplier": "AGENT2_DENSE_REWARD_NEAR_TOUCH_MULTIPLIER",
        "dense_reward_max_progress_m_per_step": "AGENT2_DENSE_REWARD_MAX_PROGRESS_M_PER_STEP",
        "dense_reward_landing_lock_time_penalty": "AGENT2_DENSE_REWARD_LANDING_LOCK_TIME_PENALTY",
        "dense_reward_hesitation_penalty": "AGENT2_DENSE_REWARD_HESITATION_PENALTY",
        "dense_reward_min_descent_action_when_aligned": "AGENT2_DENSE_REWARD_MIN_DESCENT_ACTION_WHEN_ALIGNED",
        "dense_reward_unsafe_descent_penalty": "AGENT2_DENSE_REWARD_UNSAFE_DESCENT_PENALTY",
        "timeout_penalty": "AGENT2_TIMEOUT_PENALTY",
        "target_lost_penalty": "AGENT2_TARGET_LOST_PENALTY",
        "predictive_metric_kp_position_per_s": "PARALLEL_BOTTOM_METRIC_KP",
        "predictive_metric_kd_relative_velocity": "PARALLEL_BOTTOM_METRIC_KD",
        "bottom_camera_hfov_deg": "PARALLEL_BOTTOM_CAMERA_HFOV_DEG",
        "optical_flow_enabled": "PARALLEL_BOTTOM_OPTICAL_FLOW_ENABLED",
        "visual_kalman_enabled": "PARALLEL_BOTTOM_VISUAL_KALMAN_ENABLED",
        "reacquire_climb_enabled": "PARALLEL_REACQUIRE_CLIMB_ENABLED",
        "reacquire_climb_after_s": "PARALLEL_REACQUIRE_CLIMB_AFTER_S",
        "reacquire_climb_speed_mps": "PARALLEL_REACQUIRE_CLIMB_SPEED_MPS",
        "reacquire_climb_target_height_m": "PARALLEL_REACQUIRE_CLIMB_TARGET_HEIGHT_M",
    }
    for cfg_name, flow_name in assignments.items():
        setattr(cfg, cfg_name, getattr(flow, flow_name))
    return cfg


def build_parallel_env(
    agent1_checkpoint: Path,
    device: str,
    target_identity: TargetIdentitySnapshot | None = None,
    agent2_config: Agent2Config | None = None,
) -> Agent1P2Env:
    env = Agent1P2Env(
        agent1_model_path=str(agent1_checkpoint),
        deterministic=bool(flow.AGENT_1_DETERMINISTIC),
        max_attempts=int(flow.AGENT_1_PREPARE_MAX_ATTEMPTS),
        max_steps_per_attempt=int(flow.AGENT_1_PREPARE_MAX_STEPS_PER_ATTEMPT),
        agent2_config=agent2_config or build_agent2_config(),
        device=device,
        target_velocity_feedforward_gain=float(
            flow.PARALLEL_TARGET_VELOCITY_FEEDFORWARD_GAIN
        ),
        horizontal_total_speed_max_mps=float(
            flow.PARALLEL_HORIZONTAL_TOTAL_SPEED_MAX_MPS
        ),
        agent1_min_xy_weight_near_landing=float(
            flow.PARALLEL_AGENT1_MIN_XY_WEIGHT_NEAR_LANDING
        ),
        bottom_live_agent1_xy_weight=float(
            flow.PARALLEL_AGENT1_XY_WEIGHT_WHEN_BOTTOM_LIVE
        ),
        bottom_pred_agent1_xy_weight=float(
            flow.PARALLEL_AGENT1_XY_WEIGHT_WHEN_BOTTOM_PRED
        ),
        bottom_pd_correction_gain=float(
            flow.PARALLEL_BOTTOM_PD_CORRECTION_GAIN
        ),
        landing_bridge_max_speed_mps=float(
            flow.PARALLEL_LANDING_BRIDGE_MAX_SPEED_MPS
        ),
        landing_catchup_bridge_max_speed_mps=float(
            flow.PARALLEL_LANDING_CATCHUP_BRIDGE_MAX_SPEED_MPS
        ),
        near_ground_descent_max_mps=float(
            flow.PARALLEL_NEAR_GROUND_DESCENT_MAX_MPS
        ),
    )
    if target_identity is not None:
        inject_target_identity(env, target_identity)
    return env


def load_or_create(model_path: Path | None, env, device: str) -> PPO:
    model = None
    if model_path is not None and model_path.exists():
        try:
            model = PPO.load(str(model_path), env=env, device=device)
        except (ValueError, AssertionError) as exc:
            print(
                "[CO-TRAIN] Checkpoint observation/action space is incompatible "
                "with the five-range-finder architecture; Agent 2 starts fresh. "
                f"checkpoint={model_path} error={exc}"
            )
    if model is None:
        model = PPO(
            "MlpPolicy",
            env,
            device=device,
            verbose=0,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=PPO_LEARNING_RATE,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
    model.verbose = 0
    return model


def release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_agent1_phase(
    cycle: int,
    agent1_checkpoint: Path,
    agent2_checkpoint: Path,
    run_root: Path,
    device: str,
    target_identity: TargetIdentitySnapshot,
) -> Path:
    phase_dir = run_root / f"cycle_{cycle:02d}" / "agent1_phase"
    log_dir = phase_dir / "tensorboard"
    phase_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    parallel = build_parallel_env(agent1_checkpoint, device, target_identity)
    raw_env = TrainAgent1WithFrozenAgent2Env(
        parallel_env=parallel,
        frozen_agent2_checkpoint=agent2_checkpoint,
        device=device,
    )
    env = Monitor(raw_env)
    model = load_or_create(agent1_checkpoint, env, device)
    model.set_logger(configure(str(log_dir), ["stdout", "tensorboard"]))

    checkpoint = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY_STEPS,
        save_path=str(phase_dir),
        name_prefix="agent1_cotraining",
    )
    print("=" * 92)
    print(f"[CO-TRAIN] CYCLE {cycle}/{NUMBER_OF_CYCLES} | TRAIN AGENT 1")
    print(f"[CO-TRAIN] Trainable : Agent 1 (X/Y/Yaw)")
    print(f"[CO-TRAIN] Frozen    : Agent 2 from {agent2_checkpoint}")
    print(f"[CO-TRAIN] RPC +{RPC_SUCCESS_REWARD:.0f} / terminal no-RPC -{RPC_FAILURE_PENALTY:.0f}")
    model.learn(
        total_timesteps=STEPS_PER_AGENT1_PHASE,
        callback=checkpoint,
        reset_num_timesteps=False,
        progress_bar=True,
    )

    final_path = phase_dir / "agent1_ppo_final"
    model.save(str(final_path))
    final_zip = final_path.with_suffix(".zip")

    source_snapshot = agent1_checkpoint.parent / "training_config_snapshot.json"
    if source_snapshot.exists():
        shutil.copy2(source_snapshot, phase_dir / "training_config_snapshot.json")

    env.close()
    release_model(model)
    return final_zip


def train_agent2_phase(
    cycle: int,
    agent1_checkpoint: Path,
    agent2_checkpoint: Path,
    run_root: Path,
    device: str,
    target_identity: TargetIdentitySnapshot,
) -> Path:
    phase_dir = run_root / f"cycle_{cycle:02d}" / "agent2_phase"
    log_dir = phase_dir / "tensorboard"
    phase_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    parallel = build_parallel_env(agent1_checkpoint, device, target_identity)
    raw_env = RpcDominantAgent2RewardEnv(parallel)
    diagnostic_env = RangeFinderLiveDiagnostics(
        raw_env, label=f"CYCLE_{cycle:02d}_AGENT2", print_every_steps=10
    )
    env = Monitor(diagnostic_env)
    model = load_or_create(agent2_checkpoint, env, device)
    model.set_logger(configure(str(log_dir), ["stdout", "tensorboard"]))

    checkpoint = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY_STEPS,
        save_path=str(phase_dir),
        name_prefix="agent2_cotraining",
    )
    print("=" * 92)
    print(f"[CO-TRAIN] CYCLE {cycle}/{NUMBER_OF_CYCLES} | TRAIN AGENT 2")
    print(f"[CO-TRAIN] Trainable : Agent 2 (NED-Z)")
    print(f"[CO-TRAIN] Frozen    : Agent 1 from {agent1_checkpoint}")
    print(f"[CO-TRAIN] RPC +{RPC_SUCCESS_REWARD:.0f} / terminal no-RPC -{RPC_FAILURE_PENALTY:.0f}")
    model.learn(
        total_timesteps=STEPS_PER_AGENT2_PHASE,
        callback=checkpoint,
        reset_num_timesteps=False,
        progress_bar=True,
    )

    final_path = phase_dir / "agent2_ppo_final"
    model.save(str(final_path))
    final_zip = final_path.with_suffix(".zip")

    env.close()
    release_model(model)
    return final_zip


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[CO-TRAIN] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("[CO-TRAIN] CUDA unavailable; using CPU")

    agent1_checkpoint = find_agent1_checkpoint(str(flow.AGENT_1_MODEL_PATH))
    agent2_checkpoint = latest_agent2_checkpoint()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = Path("models") / "ALTERNATING_CO_TRAINING" / f"run_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    target_identity = select_target_once(agent1_checkpoint, run_root, device)

    manifest = {
        "started_at": timestamp,
        "cycles": NUMBER_OF_CYCLES,
        "steps_per_agent1_phase": STEPS_PER_AGENT1_PHASE,
        "steps_per_agent2_phase": STEPS_PER_AGENT2_PHASE,
        "initial_agent1_checkpoint": str(agent1_checkpoint),
        "initial_agent2_checkpoint": str(agent2_checkpoint),
        "rpc_success_reward": RPC_SUCCESS_REWARD,
        "terminal_without_rpc_penalty": RPC_FAILURE_PENALTY,
        "identity_reward_episode_cap": IDENTITY_EPISODE_CAP,
        "agent2_identity_reward_scale": AGENT2_IDENTITY_REWARD_SCALE,
        "target_identity_file": str(run_root / "selected_target_identity.npz"),
        "target_class_id": target_identity.class_id,
        "target_fingerprint_dimensions": int(target_identity.fingerprint.size),
    }
    (run_root / "co_training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("=" * 92)
    print("[CO-TRAIN] AUTOMATIC ALTERNATING TRAINING")
    print(f"[CO-TRAIN] Cycles             : {NUMBER_OF_CYCLES}")
    print(f"[CO-TRAIN] Agent-1 phase steps: {STEPS_PER_AGENT1_PHASE}")
    print(f"[CO-TRAIN] Agent-2 phase steps: {STEPS_PER_AGENT2_PHASE}")
    print(f"[CO-TRAIN] Initial Agent 1    : {agent1_checkpoint}")
    print(f"[CO-TRAIN] Initial Agent 2    : {agent2_checkpoint}")
    print(f"[CO-TRAIN] Output             : {run_root}")
    print("[CO-TRAIN] Range diagnostics  : passive only; no Vision/control changes")
    print(f"[CO-TRAIN] Target selection   : ONE CLICK, then automatic AUTO_LOCK")
    print(f"[CO-TRAIN] Target identity    : {run_root / 'selected_target_identity.npz'}")
    print(f"[CO-TRAIN] Identity shaping   : LIVE streak + scale-jump guard, cap +/-{IDENTITY_EPISODE_CAP:.0f}")
    print(f"[CO-TRAIN] Agent-2 ID share   : {AGENT2_IDENTITY_REWARD_SCALE:.2f}x")
    print("=" * 92)

    for cycle in range(1, NUMBER_OF_CYCLES + 1):
        # Agent 2 is trained first in each cycle because its observation space
        # changed from 37 to 46 features when the five range finders were added.
        agent2_checkpoint = train_agent2_phase(
            cycle, agent1_checkpoint, agent2_checkpoint, run_root, device,
            target_identity,
        )
        agent1_checkpoint = train_agent1_phase(
            cycle, agent1_checkpoint, agent2_checkpoint, run_root, device,
            target_identity,
        )

        cycle_state = {
            "cycle": cycle,
            "agent1_checkpoint": str(agent1_checkpoint),
            "agent2_checkpoint": str(agent2_checkpoint),
        }
        (run_root / "latest_cycle.json").write_text(
            json.dumps(cycle_state, indent=2), encoding="utf-8"
        )
        print(
            f"[CO-TRAIN] CYCLE {cycle} COMPLETE | "
            f"Agent1={agent1_checkpoint} | Agent2={agent2_checkpoint}"
        )

    print("=" * 92)
    print("[CO-TRAIN] ALL CYCLES COMPLETE")
    print(f"[CO-TRAIN] Final Agent 1: {agent1_checkpoint}")
    print(f"[CO-TRAIN] Final Agent 2: {agent2_checkpoint}")
    print("=" * 92)


if __name__ == "__main__":
    print(
        "[DEPRECATED] Use Run_train_cooperative_final.py. "
        "Redirecting to the synchronized final trainer."
    )
    from Run_train_cooperative_final import main as cooperative_main

    raise SystemExit(cooperative_main())
