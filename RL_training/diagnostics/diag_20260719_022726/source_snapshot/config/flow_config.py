"""Single place to select the training flow.

AGENT_1P2 uses parallel dual-agent control:
    Agent 1 stays active on every step and owns XY/Yaw.
    Agent 2 stays active on every step and owns Z only.
    Front and bottom tracking remain active continuously.
"""

TRAINING_MODE = "AGENT_1P2"

TOTAL_AGENT2_TIMESTEPS = 20_480
CHECKPOINT_FREQUENCY = 10_240

# Agent 1 checkpoint selection. Leave empty to select the checkpoint with the
# largest ``*_steps.zip`` number under models/PPO_Tracker/tracking.
AGENT_1_MODEL_PATH = ""
AGENT_1_DETERMINISTIC = True
AGENT_1_PREPARE_MAX_ATTEMPTS = 3
AGENT_1_PREPARE_MAX_STEPS_PER_ATTEMPT = 700

# Parallel command mixer. The feed-forward is the filtered vehicle velocity in
# drone body axes. Agent 1 adds it to its relative-position correction.
PARALLEL_TARGET_VELOCITY_FEEDFORWARD_GAIN = 1.20
PARALLEL_TARGET_VELOCITY_EMA_ALPHA = 0.55
PARALLEL_HORIZONTAL_TOTAL_SPEED_MAX_MPS = 6.0

# Bottom LIVE is authoritative for XY during landing. Agent 1 still performs
# inference on every step and owns yaw/search, but its front/PRED XY is muted
# while the bottom predictive controller has a current target. Full Agent-1 XY
# returns immediately when the bottom camera loses LIVE MATCH.
PARALLEL_AGENT1_MIN_XY_WEIGHT_NEAR_LANDING = 0.20  # legacy compatibility
PARALLEL_AGENT1_XY_WEIGHT_WHEN_BOTTOM_LIVE = 0.00
PARALLEL_BOTTOM_PD_CORRECTION_GAIN = 1.00

# Bottom-camera t+1 predictor and catch-up controller. The horizon uses actual
# frame spacing plus a small actuation-latency allowance, then clamps to the
# configured safe range.
PARALLEL_BOTTOM_RELATIVE_VELOCITY_EMA_ALPHA = 0.55
PARALLEL_BOTTOM_PREDICTION_EXTRA_LATENCY_S = 0.08
PARALLEL_BOTTOM_PREDICTION_HORIZON_MIN_S = 0.10
PARALLEL_BOTTOM_PREDICTION_HORIZON_MAX_S = 0.60
PARALLEL_BOTTOM_NORMAL_CORRECTION_MAX_MPS = 1.10
PARALLEL_BOTTOM_CATCHUP_CORRECTION_MAX_MPS = 2.20
PARALLEL_BOTTOM_CATCHUP_ENTER_CENTER_ERROR = 0.24
PARALLEL_BOTTOM_CATCHUP_EXIT_CENTER_ERROR = 0.13

# Agent 2 resume policy.
RESUME_AGENT_2 = True

# Standalone AGENT_2 options.
AGENT_2_TARGET_ACTOR_NAME = ""
AGENT_2_STATIC_TARGET_SURFACE_ALTITUDE_M = 0.0

SHOW_AGENT2_CAMERA = True
AGENT2_MAX_EPISODE_STEPS = 900
