"""Final cooperative-training configuration.

Agent 1 and Agent 2 are both trained throughout the run. Stable-Baselines3
optimizes one PPO model at a time, so training uses short synchronized phases:
Agent 2 learns with the current Agent 1 pair member, then Agent 1 learns with
the newly updated Agent 2. Every completed phase produces an atomic checkpoint
pair. Neither agent is frozen for the complete run.
"""

from __future__ import annotations

from pathlib import Path

# Training stages. Smoke and pilot continue into the same active run. Long may
# start only after the pilot pair passes a native 46-observation E2E test.
STAGE_TARGET_STEPS_PER_AGENT = {
    "smoke": 2_048,
    "pilot": 20_480,
    "pilot_extended": 51_200,
    "long": 250_000,
}

# Short phases keep both policies adapting to each other while remaining fully
# compatible with Stable-Baselines3 PPO. The last phase is shortened as needed.
PHASE_STEPS = 10_240
RECOVERY_CHECKPOINT_EVERY_STEPS = 2_048

PPO_DEVICE = "cpu"  # MLP PPO is faster/steadier on CPU; CUDA remains for vision.
SEED = 42
PPO_N_STEPS = 1_024
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.20
PPO_ENT_COEF = 0.01
PPO_VF_COEF = 0.50
PPO_MAX_GRAD_NORM = 0.50

# Agent 1 already contains useful XY/Yaw behavior, so it learns conservatively.
AGENT1_LEARNING_RATE = 1.0e-4
# Agent 2 starts a new 46-observation policy and may adapt faster.
AGENT2_LEARNING_RATE = 3.0e-4

# Pair selection uses actual completed episodes, not PPO loss alone.
BEST_PAIR_MIN_EPISODES = 5
HEALTH_WINDOW_EPISODES = 20

FINAL_MODELS_ROOT = Path("models") / "FINAL_MODELS"
FINAL_MODELS_MANIFEST = FINAL_MODELS_ROOT / "final_models.json"
FINAL_AGENT1_CHECKPOINT = FINAL_MODELS_ROOT / "agent1_final.zip"
FINAL_AGENT2_CHECKPOINT = FINAL_MODELS_ROOT / "agent2_final.zip"

RUNS_ROOT = Path("models") / "COOPERATIVE_FINAL"
ACTIVE_POINTER = RUNS_ROOT / "active_run.json"
LAST_COMPLETED_POINTER = RUNS_ROOT / "last_completed_run.json"

EXPECTED_AGENT1_OBS_DIM = 37
EXPECTED_AGENT1_ACTION_DIM = 4
EXPECTED_AGENT2_OBS_DIM = 46
EXPECTED_AGENT2_ACTION_DIM = 4
RANGE_FEATURE_COUNT = 9

# A native 46-observation pilot E2E pass is mandatory before --stage long.
NATIVE_E2E_SUMMARY = Path("diagnostics") / "agent2_range_e2e_latest.json"
