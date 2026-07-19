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

        # A faster velocity EMA reduces lag when the moving target accelerates
        # or changes pace. This modifies only the deterministic feed-forward
        # path and keeps the PPO observation/action shapes unchanged.
        cfg.target_velocity_ema_alpha = float(
            flow.PARALLEL_TARGET_VELOCITY_EMA_ALPHA
        )
        cfg.horizontal_velocity_ema_alpha = float(
            flow.PARALLEL_BOTTOM_RELATIVE_VELOCITY_EMA_ALPHA
        )
        cfg.predictive_bottom_extra_latency_s = float(
            flow.PARALLEL_BOTTOM_PREDICTION_EXTRA_LATENCY_S
        )
        cfg.predictive_bottom_horizon_min_s = float(
            flow.PARALLEL_BOTTOM_PREDICTION_HORIZON_MIN_S
        )
        cfg.predictive_bottom_horizon_max_s = float(
            flow.PARALLEL_BOTTOM_PREDICTION_HORIZON_MAX_S
        )
        cfg.predictive_bottom_normal_correction_max_mps = float(
            flow.PARALLEL_BOTTOM_NORMAL_CORRECTION_MAX_MPS
        )
        cfg.predictive_bottom_catchup_correction_max_mps = float(
            flow.PARALLEL_BOTTOM_CATCHUP_CORRECTION_MAX_MPS
        )
        cfg.predictive_bottom_catchup_enter_center_error = float(
            flow.PARALLEL_BOTTOM_CATCHUP_ENTER_CENTER_ERROR
        )
        cfg.predictive_bottom_catchup_exit_center_error = float(
            flow.PARALLEL_BOTTOM_CATCHUP_EXIT_CENTER_ERROR
        )

        env_raw = Agent1P2Env(
            agent1_model_path=str(flow.AGENT_1_MODEL_PATH),
            deterministic=bool(flow.AGENT_1_DETERMINISTIC),
            max_attempts=int(flow.AGENT_1_PREPARE_MAX_ATTEMPTS),
            max_steps_per_attempt=int(flow.AGENT_1_PREPARE_MAX_STEPS_PER_ATTEMPT),
            agent2_config=cfg,
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
            bottom_pd_correction_gain=float(
                flow.PARALLEL_BOTTOM_PD_CORRECTION_GAIN
            ),
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
    print("[FLOW] Agent 1 runtime  : always active, frozen, front+bottom fusion")
    print("[FLOW] Agent 1 owns     : XY + Yaw + target-velocity matching")
    print("[FLOW] Agent 2 runtime  : always active landing controller")
    print("[FLOW] Agent 2 owns     : Z only (AirSim API NED-Z)")
    print("[FLOW] AirSim commands  : exactly one fused command per step")
    print("[FLOW] Lost bottom view : descent blocked; Agent 1 keeps chasing/searching")
    print(
        "[FLOW] Target velocity   : "
        f"gain={float(flow.PARALLEL_TARGET_VELOCITY_FEEDFORWARD_GAIN):.2f} "
        f"ema={float(flow.PARALLEL_TARGET_VELOCITY_EMA_ALPHA):.2f}"
    )
    print(
        "[FLOW] XY mixer          : bottom LIVE predictive authority + target velocity; "
        f"A1W_bottom={float(flow.PARALLEL_AGENT1_XY_WEIGHT_WHEN_BOTTOM_LIVE):.2f} "
        f"bottomGain={float(flow.PARALLEL_BOTTOM_PD_CORRECTION_GAIN):.2f}"
    )
    print(
        "[FLOW] Bottom prediction : t+1 image position + relative image velocity | "
        f"horizon={float(flow.PARALLEL_BOTTOM_PREDICTION_HORIZON_MIN_S):.2f}-"
        f"{float(flow.PARALLEL_BOTTOM_PREDICTION_HORIZON_MAX_S):.2f}s"
    )
    print(
        "[FLOW] Catch-up mode     : predicted drift pauses Z; "
        f"normalMax={float(flow.PARALLEL_BOTTOM_NORMAL_CORRECTION_MAX_MPS):.2f}m/s "
        f"catchMax={float(flow.PARALLEL_BOTTOM_CATCHUP_CORRECTION_MAX_MPS):.2f}m/s"
    )
    print("[FLOW] Front camera      : search/reacquire fallback; never overrides Bottom LIVE XY")
    print("[FLOW] Landing lock      : 2-frame acquire; short PRED gaps pause Z without reset")
    print("[FLOW] Appearance bank   : immutable identity + bounded adaptive landing views")
    print("[FLOW] Touchdown gate   : bad LIVE centering cannot use appearance fallback")
    print("[FLOW] Recovery cycles  : removed")
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
