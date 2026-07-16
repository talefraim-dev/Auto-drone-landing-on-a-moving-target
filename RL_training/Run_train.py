"""Three-mode training entry point with strict Agent-1/Agent-2 isolation.

Change only ``TRAINING_MODE`` in config/flow_config.py and run:
    python Run_train.py
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from config import flow_config as flow


def _latest_step_checkpoint(root: Path) -> Path | None:
    def step(path: Path) -> int:
        match = re.search(r"_(\d+)_steps\.zip$", path.name)
        return int(match.group(1)) if match else -1

    candidates = [p for p in root.glob("**/*.zip") if step(p) >= 0]
    return max(candidates, key=lambda p: (step(p), str(p))) if candidates else None


def _train_agent2(mode: str) -> None:
    from agent2_landing_env import Agent2Config, Agent2LandingEnv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Agent2Config(
        show_camera=bool(flow.SHOW_AGENT2_CAMERA),
        max_episode_steps=int(flow.AGENT2_MAX_EPISODE_STEPS),
        static_target_actor_name=str(flow.AGENT_2_TARGET_ACTOR_NAME),
        static_target_surface_altitude_m=float(flow.AGENT_2_STATIC_TARGET_SURFACE_ALTITUDE_M),
    )

    if mode == "AGENT_1P2":
        from agent1p2_env import Agent1P2Env

        env_raw = Agent1P2Env(
            agent1_model_path=str(flow.AGENT_1_MODEL_PATH),
            deterministic=bool(flow.AGENT_1_DETERMINISTIC),
            max_attempts=int(flow.AGENT_1_PREPARE_MAX_ATTEMPTS),
            max_steps_per_attempt=int(flow.AGENT_1_PREPARE_MAX_STEPS_PER_ATTEMPT),
            agent2_config=cfg,
            device=device,
        )
    else:
        env_raw = Agent2LandingEnv(cfg=cfg)

    env = Monitor(env_raw)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_root = Path("models") / mode
    run_dir = model_root / f"agent2_{timestamp}"
    log_dir = Path("logs") / mode / f"agent2_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    latest = _latest_step_checkpoint(model_root) if bool(flow.RESUME_AGENT_2) else None
    if latest:
        print(f"[TRAIN] Resuming Agent 2 from: {latest}")
        model = PPO.load(str(latest), env=env, device=device)
    else:
        print("[TRAIN] Creating fresh Agent-2 PPO")
        model = PPO(
            "MlpPolicy",
            env,
            device=device,
            verbose=1,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=str(log_dir),
        )

    model.set_logger(configure(str(log_dir), ["stdout", "tensorboard"]))
    checkpoint = CheckpointCallback(
        save_freq=int(flow.CHECKPOINT_FREQUENCY),
        save_path=str(run_dir),
        name_prefix="agent2_ppo",
    )

    print("=" * 92)
    print(f"[FLOW] TRAINING_MODE    : {mode}")
    print("[FLOW] Agent 1 runtime  : original DroneEnv + exact checkpoint snapshot")
    print("[FLOW] Agent 2 runtime  : independent bottom-camera environment")
    print("[FLOW] Vertical source  : AirSim API Z only")
    print("[FLOW] LiDAR            : horizontal obstacle sectors only")
    print("[FLOW] Reward           : landing-only, collision-gated")
    print("=" * 92)

    try:
        model.learn(
            total_timesteps=int(flow.TOTAL_AGENT2_TIMESTEPS),
            callback=checkpoint,
            reset_num_timesteps=latest is None,
            progress_bar=False,
        )
        final_path = run_dir / "agent2_ppo_final.zip"
        model.save(str(final_path))
        print(f"[TRAIN] Saved final Agent-2 model: {final_path}")
    finally:
        env.close()


def main() -> None:
    mode = str(flow.TRAINING_MODE).upper().strip()
    if mode == "AGENT_1":
        print("[FLOW] AGENT_1 selected: executing preserved original Run_train.py logic.")
        import Run_train_agent1_original

        Run_train_agent1_original.main()
        return
    if mode not in {"AGENT_2", "AGENT_1P2"}:
        raise ValueError("TRAINING_MODE must be AGENT_1, AGENT_2, or AGENT_1P2")
    _train_agent2(mode)


if __name__ == "__main__":
    main()
