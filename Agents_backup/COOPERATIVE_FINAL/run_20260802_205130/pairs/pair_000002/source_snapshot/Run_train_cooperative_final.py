"""Resumable final cooperative training for Agent 1 and Agent 2.

Usage:
    python Run_train_cooperative_final.py --stage smoke
    python Run_train_cooperative_final.py --stage pilot
    python Run_train_cooperative_final.py --stage long

Both agents learn throughout the run. Stable-Baselines3 PPO updates one policy
at a time in short synchronized phases. After every phase, the updated policy
and its current partner are stored as one atomic checkpoint pair.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from agent1p2_env import find_agent1_checkpoint
from alternating_cotraining_env import (
    AGENT2_IDENTITY_REWARD_SCALE,
    DENSE_EPISODE_CAP,
    IDENTITY_EPISODE_CAP,
    RPC_FAILURE_PENALTY,
    RPC_SUCCESS_REWARD,
    RangeFinderLiveDiagnostics,
    RpcDominantAgent2RewardEnv,
    TrainAgent1WithFrozenAgent2Env,
)
from config import flow_config as flow
from cooperative_training_config import (
    ACTIVE_POINTER,
    AGENT1_LEARNING_RATE,
    AGENT2_LEARNING_RATE,
    BEST_PAIR_MIN_EPISODES,
    EXPECTED_AGENT2_OBS_DIM,
    HEALTH_WINDOW_EPISODES,
    LAST_COMPLETED_POINTER,
    PHASE_STEPS,
    PPO_BATCH_SIZE,
    PPO_CLIP_RANGE,
    PPO_DEVICE,
    PPO_ENT_COEF,
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    PPO_MAX_GRAD_NORM,
    PPO_N_EPOCHS,
    PPO_N_STEPS,
    PPO_VF_COEF,
    RECOVERY_CHECKPOINT_EVERY_STEPS,
    RUNS_ROOT,
    SEED,
    STAGE_TARGET_STEPS_PER_AGENT,
    NATIVE_E2E_SUMMARY,
)
from paired_checkpoint_manager import (
    atomic_json,
    checkpoint_space_dim,
    create_atomic_pair,
    maybe_publish_best_pair,
    verify_sb3_zip,
)
from Run_train_alternating_agents import (
    TargetIdentitySnapshot,
    build_agent2_config,
    build_parallel_env,
    latest_agent2_checkpoint,
    load_target_identity,
    select_target_once,
)
from training_safety import FiniteTrainingGuard, PhaseHealthCallback


ROLE_AGENT1 = "AGENT1"
ROLE_AGENT2 = "AGENT2"


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_snapshot_files() -> list[Path]:
    return [
        Path(__file__),
        Path("cooperative_training_config.py"),
        Path("paired_checkpoint_manager.py"),
        Path("training_safety.py"),
        Path("alternating_cotraining_env.py"),
        Path("agent1p2_env.py"),
        Path("agent2_landing_env.py"),
        Path("range_finder_array.py"),
        Path("config") / "flow_config.py",
        Path("config") / "range_finder_calibration.json",
        Path("config") / "bottom_bbox_center_calibration.json",
        Path("settings.json"),
    ]


def _agent2_training_config():
    cfg = build_agent2_config()
    cfg.show_camera = False
    # This switch exists only for physical contact diagnostics. Leaving it on
    # would overwrite Agent-2's Z action and make PPO learn from an action that
    # was never executed.
    cfg.force_descent_while_bottom_match = False
    return cfg


def _build_agent2_env(
    agent1_checkpoint: Path,
    target_identity: TargetIdentitySnapshot,
    label: str,
):
    parallel = build_parallel_env(
        agent1_checkpoint,
        PPO_DEVICE,
        target_identity=target_identity,
        agent2_config=_agent2_training_config(),
    )
    wrapped = RpcDominantAgent2RewardEnv(parallel)
    wrapped = FiniteTrainingGuard(wrapped)
    wrapped = RangeFinderLiveDiagnostics(wrapped, label=label, print_every_steps=25)
    return Monitor(wrapped)


def _build_agent1_env(
    agent1_checkpoint: Path,
    agent2_checkpoint: Path,
    target_identity: TargetIdentitySnapshot,
    label: str,
):
    parallel = build_parallel_env(
        agent1_checkpoint,
        PPO_DEVICE,
        target_identity=target_identity,
        agent2_config=_agent2_training_config(),
    )
    wrapped = TrainAgent1WithFrozenAgent2Env(
        parallel_env=parallel,
        frozen_agent2_checkpoint=agent2_checkpoint,
        device=PPO_DEVICE,
    )
    wrapped = FiniteTrainingGuard(wrapped)
    wrapped = RangeFinderLiveDiagnostics(wrapped, label=label, print_every_steps=25)
    return Monitor(wrapped)


def _new_ppo(role: str, env) -> PPO:
    learning_rate = (
        AGENT1_LEARNING_RATE if role == ROLE_AGENT1 else AGENT2_LEARNING_RATE
    )
    return PPO(
        "MlpPolicy",
        env,
        device=PPO_DEVICE,
        verbose=0,
        seed=SEED,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        learning_rate=learning_rate,
        clip_range=PPO_CLIP_RANGE,
        ent_coef=PPO_ENT_COEF,
        vf_coef=PPO_VF_COEF,
        max_grad_norm=PPO_MAX_GRAD_NORM,
    )


def _load_model(role: str, checkpoint: Path | None, env) -> PPO:
    if checkpoint is not None and checkpoint.is_file():
        try:
            model = PPO.load(str(checkpoint), env=env, device=PPO_DEVICE)
            if role == ROLE_AGENT2:
                dim = checkpoint_space_dim(checkpoint, "observation_space")
                if dim != EXPECTED_AGENT2_OBS_DIM:
                    raise ValueError(
                        f"Agent-2 checkpoint observation dim is {dim}, expected "
                        f"{EXPECTED_AGENT2_OBS_DIM}."
                    )
            return model
        except (ValueError, AssertionError) as exc:
            if role != ROLE_AGENT2:
                raise
            print(
                "[COOP FINAL] Legacy/incompatible Agent-2 checkpoint is not used "
                f"for learning. Starting a clean 46-observation policy. error={exc}"
            )
    return _new_ppo(role, env)


def _latest_numbered_checkpoint(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    candidates = list(directory.glob("*.zip"))
    if not candidates:
        return None

    def key(path: Path) -> tuple[int, float]:
        values = [int(part) for part in path.stem.split("_") if part.isdigit()]
        return (max(values) if values else -1, path.stat().st_mtime)

    return max(candidates, key=key)


def _same_checkpoint(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).resolve(strict=False) == Path(right).resolve(strict=False)
    except Exception:
        return False


def _native_pilot_e2e_passed(state: dict[str, Any]) -> bool:
    """Accept E2E evidence only when it belongs to the current synchronized pair."""
    if not NATIVE_E2E_SUMMARY.is_file():
        return False
    try:
        summary = _read_json(NATIVE_E2E_SUMMARY)
    except Exception:
        return False

    current_agent1 = state.get("current_agent1_checkpoint")
    current_agent2 = state.get("current_agent2_checkpoint")
    if not current_agent1 or not current_agent2:
        return False

    return bool(
        summary.get("result") == "PASS_END_TO_END"
        and summary.get("ready_for_serious_training", False)
        and int(summary.get("checkpoint_observation_dim", -1)) == EXPECTED_AGENT2_OBS_DIM
        and int(summary.get("runtime_observation_dim", -2)) == EXPECTED_AGENT2_OBS_DIM
        and str(summary.get("observation_compatibility", "")).upper() == "NATIVE"
        and _same_checkpoint(summary.get("agent1_checkpoint", ""), current_agent1)
        and _same_checkpoint(summary.get("agent2_checkpoint", ""), current_agent2)
    )


def _create_new_run(requested_stage: str) -> tuple[Path, dict[str, Any]]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = RUNS_ROOT / f"run_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=False)

    agent1_checkpoint = find_agent1_checkpoint(str(flow.AGENT_1_MODEL_PATH))
    legacy_agent2 = latest_agent2_checkpoint()
    target_identity = select_target_once(agent1_checkpoint, run_root, PPO_DEVICE)

    state = {
        "version": "COOPERATIVE_FINAL_V1",
        "status": "initialized",
        "created_utc": _utc(),
        "requested_stage": requested_stage,
        "run_root": str(run_root),
        "current_agent1_checkpoint": str(agent1_checkpoint),
        "current_agent2_checkpoint": None,
        "legacy_agent2_seed_checkpoint": str(legacy_agent2),
        "target_identity_file": str(run_root / "selected_target_identity.npz"),
        "next_role": ROLE_AGENT2,
        "pair_generation": 0,
        "agent1_trained_steps": 0,
        "agent2_trained_steps": 0,
        "active_phase": None,
        "reward_contract": {
            "rpc_success_reward": RPC_SUCCESS_REWARD,
            "rpc_failure_penalty": RPC_FAILURE_PENALTY,
            "dense_episode_cap": DENSE_EPISODE_CAP,
            "agent1_identity_episode_cap": IDENTITY_EPISODE_CAP,
            "agent2_identity_episode_cap": IDENTITY_EPISODE_CAP
            * AGENT2_IDENTITY_REWARD_SCALE,
        },
        "agent2_observation_dim": EXPECTED_AGENT2_OBS_DIM,
        "range_features_seen_directly_by_agent2": True,
        "agent1_trainable": True,
        "agent2_trainable": True,
        "terminal_controller_deterministic": True,
        "forced_contact_diagnostic_mode": False,
    }
    state_path = run_root / "run_state.json"
    atomic_json(state_path, state)
    atomic_json(ACTIVE_POINTER, {"run_root": str(run_root), "state_path": str(state_path)})
    return run_root, state


def _load_or_create_run(requested_stage: str) -> tuple[Path, dict[str, Any]]:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    if not ACTIVE_POINTER.is_file():
        return _create_new_run(requested_stage)
    pointer = _read_json(ACTIVE_POINTER)
    run_root = Path(pointer["run_root"])
    state_path = Path(pointer.get("state_path", run_root / "run_state.json"))
    if not run_root.is_dir() or not state_path.is_file():
        raise FileNotFoundError(
            "models/COOPERATIVE_FINAL/active_run.json points to a missing run. "
            "Restore it or remove the pointer intentionally."
        )
    state = _read_json(state_path)
    state["requested_stage"] = requested_stage
    atomic_json(state_path, state)
    print(f"[COOP FINAL] Resuming active cooperative run: {run_root}")
    return run_root, state


def _stage_target(stage: str) -> int:
    return int(STAGE_TARGET_STEPS_PER_AGENT[stage])


def _both_reached(state: dict[str, Any], target: int) -> bool:
    return bool(
        int(state.get("agent1_trained_steps", 0)) >= target
        and int(state.get("agent2_trained_steps", 0)) >= target
    )


def _recovery_checkpoint(run_root: Path, phase_id: str) -> Path | None:
    return _latest_numbered_checkpoint(run_root / "recovery" / phase_id)


def _copy_agent1_snapshot(source_checkpoint: Path, destination_dir: Path) -> None:
    source = source_checkpoint.parent / "training_config_snapshot.json"
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing Agent-1 training_config_snapshot.json next to {source_checkpoint}"
        )
    shutil.copy2(source, destination_dir / "training_config_snapshot.json")


def _run_phase(
    run_root: Path,
    state: dict[str, Any],
    role: str,
    target_steps_per_agent: int,
) -> tuple[Path, dict[str, Any], int]:
    state_path = run_root / "run_state.json"
    role_key = "agent1_trained_steps" if role == ROLE_AGENT1 else "agent2_trained_steps"
    already_trained = int(state.get(role_key, 0))
    requested_delta = min(PHASE_STEPS, max(0, target_steps_per_agent - already_trained))
    if requested_delta <= 0:
        raise RuntimeError(f"No remaining steps for {role} at target {target_steps_per_agent}")

    generation = int(state.get("pair_generation", 0)) + 1
    phase_id = f"phase_{generation:06d}_{role.lower()}"
    phase_dir = run_root / "phases" / phase_id
    phase_dir.mkdir(parents=True, exist_ok=True)
    log_dir = phase_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)

    agent1_checkpoint = Path(state["current_agent1_checkpoint"])
    agent2_value = state.get("current_agent2_checkpoint")
    agent2_checkpoint = Path(agent2_value) if agent2_value else None
    identity = load_target_identity(Path(state["target_identity_file"]))

    if role == ROLE_AGENT2:
        env = _build_agent2_env(agent1_checkpoint, identity, phase_id)
        seed_checkpoint = agent2_checkpoint
        if seed_checkpoint is None:
            seed_checkpoint = Path(state["legacy_agent2_seed_checkpoint"])
    else:
        if agent2_checkpoint is None:
            raise RuntimeError("Agent 1 cannot train before a native Agent-2 checkpoint exists.")
        env = _build_agent1_env(agent1_checkpoint, agent2_checkpoint, identity, phase_id)
        seed_checkpoint = agent1_checkpoint

    active = state.get("active_phase") or {}
    resuming_same_phase = bool(active.get("phase_id") == phase_id and active.get("role") == role)
    recovery = _recovery_checkpoint(run_root, phase_id) if resuming_same_phase else None
    load_checkpoint = recovery or seed_checkpoint

    model: PPO | None = None
    health: PhaseHealthCallback | None = None
    try:
        model = _load_model(role, load_checkpoint, env)
        if resuming_same_phase:
            phase_start_timesteps = int(active["phase_start_model_timesteps"])
            phase_target_delta = int(active["phase_target_delta"])
        else:
            phase_start_timesteps = int(model.num_timesteps)
            phase_target_delta = int(requested_delta)
            state["active_phase"] = {
                "phase_id": phase_id,
                "role": role,
                "phase_start_model_timesteps": phase_start_timesteps,
                "phase_target_delta": phase_target_delta,
                "started_utc": _utc(),
            }
            state["status"] = "training"
            atomic_json(state_path, state)

        completed_delta = max(0, int(model.num_timesteps) - phase_start_timesteps)
        remaining = max(0, phase_target_delta - completed_delta)
        print("=" * 100)
        print(f"[COOP FINAL] {phase_id}")
        print(f"[COOP FINAL] Trainable policy : {role}")
        print(
            "[COOP FINAL] Partner policy   : "
            f"{ROLE_AGENT1 if role == ROLE_AGENT2 else ROLE_AGENT2} (current pair member)"
        )
        print(f"[COOP FINAL] Remaining steps  : {remaining:,}")
        print(f"[COOP FINAL] PPO device       : {PPO_DEVICE}")
        print("[COOP FINAL] Agent-2 obs      : 46, including nine range features")
        print("[COOP FINAL] Terminal phase   : deterministic; policy actions suppressed")
        print("=" * 100)

        if remaining > 0:
            model.set_logger(configure(str(log_dir), ["stdout", "tensorboard"]))
            recovery_dir = run_root / "recovery" / phase_id
            recovery_dir.mkdir(parents=True, exist_ok=True)
            checkpoints = CheckpointCallback(
                save_freq=RECOVERY_CHECKPOINT_EVERY_STEPS,
                save_path=str(recovery_dir),
                name_prefix=role.lower(),
            )
            health = PhaseHealthCallback(
                role=role,
                phase_dir=phase_dir,
                state_path=state_path,
                window_episodes=HEALTH_WINDOW_EPISODES,
            )
            model.learn(
                total_timesteps=remaining,
                callback=CallbackList([checkpoints, health]),
                reset_num_timesteps=False,
                progress_bar=True,
            )
        else:
            health = PhaseHealthCallback(role, phase_dir, state_path)

        actual_delta = max(0, int(model.num_timesteps) - phase_start_timesteps)
        output_dir = phase_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_base = output_dir / role.lower()
        model.save(str(output_base))
        output_checkpoint = output_base.with_suffix(".zip")
        verify_sb3_zip(output_checkpoint)
        if role == ROLE_AGENT1:
            _copy_agent1_snapshot(agent1_checkpoint, output_dir)

        metrics = health.summary() if health is not None else {}
        atomic_json(phase_dir / "phase_complete.json", {
            "phase_id": phase_id,
            "role": role,
            "actual_delta": actual_delta,
            "model_timesteps": int(model.num_timesteps),
            "checkpoint": str(output_checkpoint),
            "metrics": metrics,
            "completed_utc": _utc(),
        })
        return output_checkpoint, metrics, actual_delta

    except (KeyboardInterrupt, Exception) as exc:
        if model is not None:
            recovery_dir = run_root / "recovery" / phase_id
            recovery_dir.mkdir(parents=True, exist_ok=True)
            recovery_base = recovery_dir / f"{role.lower()}_recovery_{int(model.num_timesteps)}"
            model.save(str(recovery_base))
            state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed_fast"
            state["last_error"] = f"{type(exc).__name__}: {exc}"
            state["last_recovery_checkpoint"] = str(recovery_base.with_suffix(".zip"))
            state["updated_utc"] = _utc()
            atomic_json(state_path, state)
            print(f"[COOP FINAL] Recovery checkpoint saved: {recovery_base.with_suffix('.zip')}")
        raise
    finally:
        env.close()


def _publish_phase_pair(
    run_root: Path,
    state: dict[str, Any],
    role: str,
    updated_checkpoint: Path,
    metrics: dict[str, Any],
    actual_delta: int,
) -> Path:
    if role == ROLE_AGENT1:
        agent1_checkpoint = updated_checkpoint
        agent2_checkpoint = Path(state["current_agent2_checkpoint"])
    else:
        agent1_checkpoint = Path(state["current_agent1_checkpoint"])
        agent2_checkpoint = updated_checkpoint

    generation = int(state.get("pair_generation", 0)) + 1
    role_key = "agent1_trained_steps" if role == ROLE_AGENT1 else "agent2_trained_steps"
    state[role_key] = int(state.get(role_key, 0)) + int(actual_delta)

    pair_state = {
        "version": "COOPERATIVE_FINAL_V1",
        "updated_role": role,
        "agent1_trained_steps": int(state.get("agent1_trained_steps", 0)),
        "agent2_trained_steps": int(state.get("agent2_trained_steps", 0)),
        "phase_metrics": metrics,
        "reward_contract": state["reward_contract"],
        "agent2_range_observation_enabled": True,
        "forced_contact_diagnostic_mode": False,
        "terminal_controller_deterministic": True,
    }
    pair_dir = create_atomic_pair(
        run_root=run_root,
        generation=generation,
        agent1_checkpoint=agent1_checkpoint,
        agent2_checkpoint=agent2_checkpoint,
        pair_state=pair_state,
        snapshot_files=_source_snapshot_files(),
    )

    state.update(
        {
            "pair_generation": generation,
            "current_pair_dir": str(pair_dir),
            "current_agent1_checkpoint": str(pair_dir / "agent1.zip"),
            "current_agent2_checkpoint": str(pair_dir / "agent2.zip"),
            "next_role": ROLE_AGENT1 if role == ROLE_AGENT2 else ROLE_AGENT2,
            "active_phase": None,
            "status": "pair_saved",
            "updated_utc": _utc(),
        }
    )
    atomic_json(run_root / "run_state.json", state)

    if maybe_publish_best_pair(
        run_root,
        pair_dir,
        metrics,
        minimum_episodes=BEST_PAIR_MIN_EPISODES,
    ):
        print(f"[COOP FINAL] New best synchronized pair: {pair_dir}")
    return pair_dir


def _print_reward_contract() -> None:
    agent1_failure_max = DENSE_EPISODE_CAP + IDENTITY_EPISODE_CAP - RPC_FAILURE_PENALTY
    agent2_failure_max = (
        DENSE_EPISODE_CAP
        + IDENTITY_EPISODE_CAP * AGENT2_IDENTITY_REWARD_SCALE
        - RPC_FAILURE_PENALTY
    )
    agent1_success_min = RPC_SUCCESS_REWARD - DENSE_EPISODE_CAP - IDENTITY_EPISODE_CAP
    agent2_success_min = (
        RPC_SUCCESS_REWARD
        - DENSE_EPISODE_CAP
        - IDENTITY_EPISODE_CAP * AGENT2_IDENTITY_REWARD_SCALE
    )
    print(
        "[COOP FINAL] Reward bounds | "
        f"A1 failure<={agent1_failure_max:+.0f}, A1 success>={agent1_success_min:+.0f}, "
        f"A2 failure<={agent2_failure_max:+.0f}, A2 success>={agent2_success_min:+.0f}"
    )
    if not (agent1_failure_max < 0 < agent1_success_min and agent2_failure_max < 0 < agent2_success_min):
        raise RuntimeError("Reward dominance invariant is not satisfied.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Final synchronized cooperative PPO training")
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_TARGET_STEPS_PER_AGENT),
        default="pilot",
        help=(
            "smoke=2,048, pilot=20,480, pilot_extended=51,200, "
            "long=250,000 total additional steps per agent"
        ),
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"[COOP FINAL] Vision CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("[COOP FINAL] CUDA unavailable; vision will use CPU")
    _print_reward_contract()

    run_root, state = _load_or_create_run(args.stage)
    state_path = run_root / "run_state.json"
    target = _stage_target(args.stage)

    while not _both_reached(state, target):
        pilot_target = _stage_target("pilot")
        if args.stage == "long" and _both_reached(state, pilot_target) and not _native_pilot_e2e_passed(state):
            state.update(
                {
                    "status": "awaiting_native_46_e2e",
                    "updated_utc": _utc(),
                    "required_test": (
                        "python test_agent2_vision_range_sanity.py "
                        f"--agent1-checkpoint {state['current_agent1_checkpoint']} "
                        f"--agent2-checkpoint {state['current_agent2_checkpoint']}"
                    ),
                }
            )
            atomic_json(state_path, state)
            print("=" * 100)
            print("[COOP FINAL] PILOT COMPLETE. LONG TRAINING PAUSED BY SAFETY GATE.")
            print("[COOP FINAL] Run the current pair through a native 46-observation E2E test:")
            print(state["required_test"])
            print("[COOP FINAL] After PASS_END_TO_END, run the same --stage long command again.")
            print("=" * 100)
            return 2

        role = str(state.get("next_role", ROLE_AGENT2))
        if role == ROLE_AGENT1 and not state.get("current_agent2_checkpoint"):
            role = ROLE_AGENT2
        role_steps = int(
            state.get("agent1_trained_steps" if role == ROLE_AGENT1 else "agent2_trained_steps", 0)
        )
        if role_steps >= target:
            role = ROLE_AGENT2 if role == ROLE_AGENT1 else ROLE_AGENT1

        updated, metrics, actual_delta = _run_phase(
            run_root, state, role, target
        )
        pair_dir = _publish_phase_pair(
            run_root, state, role, updated, metrics, actual_delta
        )
        print(
            f"[COOP FINAL] Pair {pair_dir.name} saved | "
            f"A1+={state['agent1_trained_steps']:,} A2+={state['agent2_trained_steps']:,}"
        )
        state = _read_json(state_path)

    state.update(
        {
            "status": f"{args.stage}_complete",
            "completed_stage": args.stage,
            "updated_utc": _utc(),
        }
    )
    atomic_json(state_path, state)

    print("=" * 100)
    print(f"[COOP FINAL] {args.stage.upper()} COMPLETE")
    print(f"[COOP FINAL] Latest pair: {state.get('current_pair_dir')}")
    print(f"[COOP FINAL] Best pair  : {run_root / 'best_pair.json'}")
    print(
        "[COOP FINAL] Evaluate current pair: python test_agent2_vision_range_sanity.py "
        f"--agent1-checkpoint {state['current_agent1_checkpoint']} "
        f"--agent2-checkpoint {state['current_agent2_checkpoint']}"
    )
    print("=" * 100)

    if args.stage == "long":
        state["status"] = "completed"
        atomic_json(state_path, state)
        atomic_json(LAST_COMPLETED_POINTER, state)
        if ACTIVE_POINTER.is_file():
            ACTIVE_POINTER.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
